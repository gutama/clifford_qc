#!/usr/bin/env python3
"""Bootstrap intervals at the stop of target-aware acquisition.

The declared target-acquisition study (``docs/restart/TARGET_ACQUISITION.md``)
found that the delta-method interval reported when ``acquire_ritz`` stops
covered too rarely on its held-out six-qubit bank: a solve that drops a
retained direction can report a small standard error with an energy that is
too high. This asks whether a grouped bootstrap at the stopping look, which
re-solves every resample and so sees that truncation, restores coverage. The
systems, schedule, targets, ``delta`` and solver policy are the declared
study's, unchanged. Two modes:

- ``screen``: development, on the declared study's own seeds. Every stopped run
  is regenerated with ``acquire_ritz``, checked against the committed record,
  and its cache at the stop bootstrapped. Several interval constructions were
  compared on these rows before the fresh run; the record keeps the quantiles
  they need.
- ``fresh``: the confirmation, on seed root 2026111901, under the criteria in
  ``docs/restart/bootstrap_stop_criteria.md``, which were fixed before it ran.
  Arm A keeps the declared stop and reports ``E +- z r`` there, with ``r`` the
  bootstrap spread ``(q84 - q16) / 2``. Arm B, exploratory, stops at the first
  look where both the delta-method standard error and ``r`` are at or below
  the target.

The fresh look loop mirrors ``acquire_ritz`` with no stopping rule, so a stopped
run is this run truncated, and both arms come from one pass. The checker
replays it against ``acquire_ritz``, including its ``bootstrap_replicates``
option, which reproduces arm A's spread exactly.

    python benchmarks/run_bootstrap_stop.py --mode fresh --jobs 4 \\
        --output reproductions/bootstrap_stop.json.gz
    python benchmarks/check_bootstrap_stop.py --report
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmarks.run_interval_coverage import build_system, derived_seed  # noqa: E402
from benchmarks.run_target_acquisition import (  # noqa: E402
    RESULTS as DECLARED_RESULTS, load_declaration, measurement_seed, schedule, target_key,
)
from clifford_qc.backends.finite_shot import FiniteShotBackend  # noqa: E402
from clifford_qc.measurement.functionals import ritz_functional  # noqa: E402
from clifford_qc.subspace.acquisition import (  # noqa: E402
    AcquisitionError, acquire_ritz, neyman_group_plan, uniform_group_plan,
)
from clifford_qc.subspace.uncertainty import bootstrap_ritz_samples  # noqa: E402

CRITERIA = ROOT / "docs" / "restart" / "bootstrap_stop_criteria.md"
RESULTS = ROOT / "docs" / "restart" / "bootstrap_stop_results.json.gz"
SCREEN = ROOT / "docs" / "restart" / "bootstrap_stop_screen.json.gz"
RESULTS_SCHEMA = "clifford_qc.bootstrap_stop_results.v1"
SCREEN_SCHEMA = "clifford_qc.bootstrap_stop_screen.v1"
FRESH_ROOT = 2026111901
REPLICAS = 200
BOOTSTRAP_REPLICATES = 500
# Bootstrap seeds come from their own stream, so no draw shares a measurement seed.
BOOTSTRAP_STREAM = 2_000_000
# Quantiles kept for each fresh bootstrap; the spread is (q84 - q16) / 2.
FRESH_LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
SOLVE_FAILURES = (ValueError, np.linalg.LinAlgError)
# The sources whose bytes decide a row.
IMPLEMENTATION = (
    "benchmarks/run_bootstrap_stop.py",
    "benchmarks/run_target_acquisition.py",
    "benchmarks/run_interval_coverage.py",
    "clifford_qc/subspace/acquisition.py",
    "clifford_qc/subspace/uncertainty.py",
    "clifford_qc/subspace/linalg.py",
    "clifford_qc/measurement/session.py",
    "clifford_qc/measurement/functionals.py",
    "clifford_qc/measurement/cache.py",
    "clifford_qc/measurement/grouping.py",
    "clifford_qc/backends/finite_shot.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule_z(declaration: dict) -> float:
    """The quantile ``acquire_ritz`` uses at a stop: two-sided ``delta`` over every look."""
    from statistics import NormalDist

    looks = len(declaration["cumulative_shots_per_group"])
    return NormalDist().inv_cdf(1.0 - declaration["delta"] / (2 * looks))


def screen_levels(declaration: dict) -> tuple[float, ...]:
    """Quantiles the development constructions need, including the schedule-wide tails."""
    looks = len(declaration["cumulative_shots_per_group"])
    tail = declaration["delta"] / (2 * looks)
    return (tail, 0.025, 0.16, 0.5, 0.84, 0.975, 1.0 - tail)


# ------------------------------------------------------------------- fresh mode

def fresh_row(declaration: dict, system: int, replica: int, allocation_index: int, *,
              root: int = FRESH_ROOT, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Both arms of one fresh run, from one pass over the schedule."""
    spec = declaration["systems"][system]
    shared, design = build_system(spec)
    pilot, budgets = schedule(declaration, len(shared.groups))
    allocation = declaration["allocations"][allocation_index]
    seed = derived_seed(root, system, replica)
    targets = list(declaration["target_std_errors"])
    backend = FiniteShotBackend(seed=seed)
    cache = shared.measure(backend, pilot)
    arm_a: dict = {}
    arm_b: dict = {}
    last = None

    def snapshot(index, point, sigma, t, tag):
        draws = bootstrap_ritz_samples(
            shared, cache, result=point, replicates=replicates,
            seed=derived_seed(root, BOOTSTRAP_STREAM + system, replica,
                              10 * allocation_index + t, index))
        q = np.quantile(draws, list(FRESH_LEVELS))
        return {"look": index, "shots": int(cache.total_shots),
                "estimate": float(point.energies[0]), "sigma": sigma,
                "rank": int(point.effective_rank), "robust": float((q[3] - q[1]) / 2),
                "quantiles": [float(v) for v in q], "stopped": tag}

    for index in range(len(budgets) + 1):
        try:
            point = shared.solve(cache, **declaration["solver_kwargs"])
        except SOLVE_FAILURES as error:
            return {"system": spec["key"], "allocation": allocation, "replica": replica,
                    "failed": f"look {index}: {type(error).__name__}: {error}",
                    "A": arm_a, "B": arm_b}
        functional = ritz_functional(shared.bank, point.indices, point.ritz_vector(0),
                                     point.energies[0])
        sigma = math.sqrt(max(0.0, functional.variance(cache)))
        last = (index, point, sigma)
        for t, goal in enumerate(targets):
            if sigma > goal:
                continue
            boot = None
            if goal not in arm_a:
                boot = snapshot(index, point, sigma, t, "target_met")
                arm_a[goal] = boot
            if goal not in arm_b:
                boot = boot or snapshot(index, point, sigma, t, "target_met")
                if boot["robust"] <= goal:
                    arm_b[goal] = boot
        if all(g in arm_a and g in arm_b for g in targets) or index == len(budgets):
            break
        if allocation == "uniform":
            plan = uniform_group_plan(shared.groups, budgets[index])
        else:
            plan = neyman_group_plan(shared.groups, cache, functional, budgets[index])
        if not plan:
            break
        shared.measure_plan(backend, plan, cache)
    index, point, sigma = last
    for t, goal in enumerate(targets):
        for arm in (arm_a, arm_b):
            if goal not in arm:
                arm[goal] = snapshot(index, point, sigma, t, "schedule_exhausted")
    return {"system": spec["key"], "allocation": allocation, "replica": replica,
            "seed": seed, "target": design["target"], "exact_rank": design["exact_rank"],
            "A": {repr(k): v for k, v in arm_a.items()},
            "B": {repr(k): v for k, v in arm_b.items()}}


def _fresh_work(task: tuple) -> list[dict]:
    declaration, system, replica, root, replicates = task
    return [fresh_row(declaration, system, replica, a, root=root, replicates=replicates)
            for a in range(len(declaration["allocations"]))]


# ------------------------------------------------------------------ screen mode

def _declared_stops() -> dict:
    record = json.load(gzip.open(DECLARED_RESULTS))
    return {(r["system"], r["allocation"], r["replica"]): r["stopping"] for r in record["rows"]}


def screen_rows(declaration: dict, system: int, replica: int, *,
                replicates: int = BOOTSTRAP_REPLICATES, stops: dict | None = None) -> list[dict]:
    """Every stopped run of one declared replica, bootstrapped at its stop."""
    spec = declaration["systems"][system]
    shared, design = build_system(spec)
    pilot, budgets = schedule(declaration, len(shared.groups))
    seed = measurement_seed(declaration, system, replica)
    stops = stops if stops is not None else _declared_stops()
    levels = screen_levels(declaration)
    out = []
    for a, allocation in enumerate(declaration["allocations"]):
        for t, goal in enumerate(declaration["target_std_errors"]):
            record = stops[(spec["key"], allocation, replica)][target_key(goal)]
            row = {"system": spec["key"], "allocation": allocation, "replica": replica,
                   "target_std_error": goal, "exact_rank": design["exact_rank"]}
            try:
                found = acquire_ritz(shared, FiniteShotBackend(seed=seed),
                                     pilot_shots_per_group=pilot, round_budgets=budgets,
                                     allocation=allocation, target_std_error=goal,
                                     solver_kwargs=dict(declaration["solver_kwargs"]),
                                     delta=declaration["delta"])
            except AcquisitionError:
                row.update(outcome="solve_failed", matches=record["outcome"] == "solve_failed")
                out.append(row)
                continue
            point = found.result
            estimate = float(point.energies[0])
            row.update(outcome="ok", stopped=found.stopped, looks=len(found.looks),
                       shots=found.shots, estimate=estimate, error=estimate - design["target"],
                       sigma=found.std_error, rank=int(point.effective_rank),
                       matches=(record["estimate"] == estimate
                                and record["looks_taken"] == len(found.looks)))
            boot_seed = derived_seed(declaration["seed_root"], BOOTSTRAP_STREAM + system,
                                     replica, 10 * a + t)
            try:
                values = bootstrap_ritz_samples(shared, found.cache, result=point,
                                                replicates=replicates, seed=boot_seed)
            except SOLVE_FAILURES as error:
                row.update(boot_failed=f"{type(error).__name__}: {error}")
            else:
                row.update(screen_summary(values, levels))
            out.append(row)
    return out


def screen_summary(values, levels) -> dict:
    """What the development constructions read from one set of bootstrap draws."""
    values = np.asarray(values, dtype=float)
    return {"quantiles": [float(v) for v in np.quantile(values, list(levels))],
            "draws": int(values.size)}


def _screen_work(task: tuple) -> list[dict]:
    declaration, system, replica, replicates = task
    return screen_rows(declaration, system, replica, replicates=replicates)


# ------------------------------------------------------------------- provenance

def _git() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {"git_head": git("rev-parse", "HEAD"),
                "dirty_tree": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "dirty_tree": None}


def provenance(started: float) -> dict:
    import scipy

    return {**_git(), "producer": "benchmarks/run_bootstrap_stop.py",
            "producer_sha256": sha256(ROOT / "benchmarks" / "run_bootstrap_stop.py"),
            "started_unix": started, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "implementation": {path: sha256(ROOT / path) for path in IMPLEMENTATION}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("fresh", "screen"), default="fresh")
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in (RESULTS.resolve(), SCREEN.resolve()) or args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    declaration = load_declaration()
    started = time.time()
    stamp = provenance(started)
    systems = range(len(declaration["systems"]))
    # Longest first: the six-qubit bank dominates.
    order = sorted(((s, r) for s in systems for r in range(args.replicas)), key=lambda t: -t[0])
    if args.mode == "fresh":
        tasks = [(declaration, s, r, FRESH_ROOT, args.bootstrap_replicates) for s, r in order]
        work = _fresh_work
    else:
        tasks = [(declaration, s, r, args.bootstrap_replicates) for s, r in order]
        work = _screen_work
    rows: list[dict] = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for done, chunk in enumerate(pool.map(work, tasks, chunksize=2), start=1):
            rows.extend(chunk)
            if done % 50 == 0:
                print(f"{done}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    record = {"schema": RESULTS_SCHEMA if args.mode == "fresh" else SCREEN_SCHEMA,
              "seed_root": FRESH_ROOT if args.mode == "fresh" else declaration["seed_root"],
              "replicas": args.replicas, "bootstrap_replicates": args.bootstrap_replicates,
              "criteria_sha256": sha256(CRITERIA), "provenance": stamp,
              "seconds": time.time() - started, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt") as handle:
        json.dump(record, handle)
    print(f"wrote {args.output} ({len(rows)} rows, {record['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
