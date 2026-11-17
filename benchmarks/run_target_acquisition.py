#!/usr/bin/env python3
"""Run the declared target-aware acquisition study and write its record.

The study, its systems, schedule, stopping targets, seeds and reporting labels
are fixed by ``docs/restart/target_acquisition_declaration.json`` before any
evaluation seed is drawn; ``docs/restart/TARGET_ACQUISITION.md`` states what is
and is not claimed. This producer only executes that declaration:

    python benchmarks/run_target_acquisition.py --jobs 4

Every (system, replica) is one independent, seeded unit of work, so the record
does not depend on ``--jobs``. An existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_interval_coverage import (  # noqa: E402
    MODELS, build_system, derived_seed, sha256, write_record,
)
from clifford_qc.backends.finite_shot import FiniteShotBackend  # noqa: E402
from clifford_qc.reproducibility import stamp_record  # noqa: E402
from clifford_qc.subspace.acquisition import (  # noqa: E402
    ALLOCATIONS, AcquisitionError, acquire_ritz,
)
from clifford_qc.subspace.coverage import delta_method_applies, wilson_interval  # noqa: E402

DECLARATION = ROOT / "docs" / "restart" / "target_acquisition_declaration.json"
RESULTS = ROOT / "docs" / "restart" / "target_acquisition_results.json.gz"
DECLARATION_SCHEMA = "clifford_qc.target_acquisition_declaration.v1"
RESULTS_SCHEMA = "clifford_qc.target_acquisition_results.v1"
# The sources whose bytes decide a replica row. A later change to any of them
# means the committed rows describe the recorded implementation, not the live one.
IMPLEMENTATION = (
    "benchmarks/run_target_acquisition.py",
    "benchmarks/run_interval_coverage.py",
    "clifford_qc/subspace/acquisition.py",
    "clifford_qc/subspace/coverage.py",
    "clifford_qc/subspace/uncertainty.py",
    "clifford_qc/subspace/linalg.py",
    "clifford_qc/measurement/session.py",
    "clifford_qc/measurement/allocation.py",
    "clifford_qc/measurement/functionals.py",
    "clifford_qc/measurement/cache.py",
    "clifford_qc/measurement/confidence.py",
    "clifford_qc/measurement/grouping.py",
    "clifford_qc/backends/finite_shot.py",
)
UNIFORM, TARGETED = ALLOCATIONS
STOP_REASONS = ("target_met", "schedule_exhausted", "deterministic")
COVERAGE_LABELS = ("under", "consistent", "over")
EFFICIENCY_LABELS = ("targeted_better", "indistinguishable", "uniform_better")
VERDICTS = ("fewer_shots_at_matched_error", "no_resolved_difference",
            "more_shots_at_matched_error")


def implementation_digests(root: Path = ROOT) -> dict[str, str]:
    return {relative: sha256(root / relative) for relative in IMPLEMENTATION}


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _fraction(value) -> bool:
    return isinstance(value, float) and 0.0 < value < 1.0


def load_declaration(path: Path = DECLARATION) -> dict:
    """Read and validate a declaration; every later step trusts its shape."""
    declaration = json.loads(Path(path).read_text(encoding="utf-8"))
    if declaration.get("schema") != DECLARATION_SCHEMA:
        raise ValueError("unsupported target-acquisition declaration schema")
    if declaration.get("target") != "exact_subspace_ground_energy":
        raise ValueError("the only supported target is exact_subspace_ground_energy")
    systems = [system["key"] for system in declaration["systems"]]
    if len(set(systems)) != len(systems) or not systems:
        raise ValueError("system keys must be nonempty and distinct")
    for system in declaration["systems"]:
        if system["model"].get("family") not in MODELS:
            raise ValueError(f"unknown model family in {system['key']}")
        if system.get("role") not in ("development", "held_out"):
            raise ValueError(f"{system['key']} must be development or held_out")
    if declaration["allocations"] != list(ALLOCATIONS):
        raise ValueError(f"allocations must be exactly {list(ALLOCATIONS)}")
    cumulative = declaration["cumulative_shots_per_group"]
    if len(cumulative) < 2 or not all(_positive_int(shots) for shots in cumulative) \
            or any(b <= a for a, b in zip(cumulative, cumulative[1:])):
        raise ValueError("cumulative_shots_per_group must be increasing positive integers "
                         "with at least one round after the pilot")
    targets = declaration["target_std_errors"]
    if not targets or len(set(targets)) != len(targets) or any(
            not isinstance(target, float) or not math.isfinite(target) or target <= 0
            for target in targets):
        raise ValueError("target_std_errors must be distinct positive finite floats")
    if not delta_method_applies(declaration["solver_kwargs"]):
        raise ValueError("the declared estimator must be one the delta method accepts")
    if not _positive_int(declaration["replicas"]) or declaration["replicas"] < 2:
        raise ValueError("replicas must be an integer of at least two")
    if not _fraction(declaration["delta"]) or not _fraction(declaration["summary_confidence"]):
        raise ValueError("delta and summary_confidence must be floats in (0, 1)")
    comparison = declaration["comparison"]
    if not _positive_int(comparison["bootstrap_replicates"]) \
            or comparison["bootstrap_replicates"] < 2:
        raise ValueError("comparison.bootstrap_replicates must be at least two")
    if not _positive_int(declaration["spot_replay_replicas"]) \
            or declaration["spot_replay_replicas"] > declaration["replicas"]:
        raise ValueError("spot_replay_replicas must be between one and replicas")
    return declaration


def target_key(target: float) -> str:
    """The canonical dictionary key of a stopping target."""
    return repr(float(target))


def schedule(declaration: dict, groups: int) -> tuple[int, list[int]]:
    """``(pilot shots per group, physical-shot budget of each round)`` for a bank."""
    cumulative = declaration["cumulative_shots_per_group"]
    return cumulative[0], [groups * (b - a) for a, b in zip(cumulative, cumulative[1:])]


def measurement_seed(declaration: dict, system: int, replica: int) -> int:
    return derived_seed(declaration["seed_root"], system, replica)


def comparison_seed(declaration: dict, system: int, slot: int) -> int:
    """Seed of one paired bootstrap; the offset keeps it off every measurement stream."""
    return derived_seed(declaration["seed_root"], 1_000_000 + system, slot)


def _bounds(interval, target: float) -> dict:
    lower, upper = float(interval.lower), float(interval.upper)
    return {"lower": lower, "upper": upper, "covered": bool(lower <= target <= upper)}


def acquisition_row(shared, *, target: float, seed: int, pilot: int, budgets: list[int],
                    allocation: str, target_std_error: float | None, solver_kwargs: dict,
                    delta: float, ledger: bool) -> dict:
    """One schedule run against a known ``target``, as a JSON-safe row.

    A look that cannot be solved is an outcome, not an exception: the row keeps
    the shots already spent and the looks that did resolve.
    """
    row: dict = {"outcome": "ok", "solve_error": None, "stopped": None, "looks_taken": 0,
                 "shots": 0, "estimate": None, "std_error": None,
                 "interval": None, "naive_interval": None}
    try:
        found = acquire_ritz(
            shared, FiniteShotBackend(seed=seed), pilot_shots_per_group=pilot,
            round_budgets=budgets, allocation=allocation,
            target_std_error=target_std_error, solver_kwargs=solver_kwargs, delta=delta)
    except AcquisitionError as error:
        looks = error.looks
        row.update({"outcome": "solve_failed", "solve_error": str(error),
                    "looks_taken": len(looks), "shots": error.shots})
    else:
        looks = found.looks
        row.update({"stopped": found.stopped, "looks_taken": len(looks),
                    "shots": found.shots, "estimate": float(found.result.energies[0]),
                    "std_error": found.std_error,
                    "interval": _bounds(found.interval, target),
                    "naive_interval": _bounds(found.naive_interval, target)})
    if ledger:
        row["ledger"] = {
            "energies": [look["energy"] for look in looks],
            "std_errors": [look["std_error"] for look in looks],
            "retained_ranks": [look["retained_rank"] for look in looks],
            "smallest_retained_overlaps": [look["smallest_retained_overlap"]
                                           for look in looks]}
    return row


def replica_rows(declaration: dict, system: int, replica: int) -> list[dict]:
    """Both allocations' rows for one replica of one system.

    The allocations share the measurement seed, so they read the same pilot and
    their difference on a replica is an allocation difference. Each allocation
    runs the whole schedule once with no stopping rule and once per declared
    stopping target; a stopped run is that same seeded run, ended early.
    """
    spec = declaration["systems"][system]
    shared, design = build_system(spec)
    pilot, budgets = schedule(declaration, len(shared.groups))
    seed = measurement_seed(declaration, system, replica)
    rows = []
    for allocation in declaration["allocations"]:
        common = {"target": design["target"], "seed": seed, "pilot": pilot,
                  "budgets": budgets, "allocation": allocation,
                  "solver_kwargs": dict(declaration["solver_kwargs"]),
                  "delta": declaration["delta"]}
        rows.append({
            "system": spec["key"], "allocation": allocation, "replica": replica,
            "measurement_seed": seed,
            "fixed": acquisition_row(shared, target_std_error=None, ledger=True, **common),
            "stopping": {target_key(goal): acquisition_row(
                shared, target_std_error=goal, ledger=False, **common)
                for goal in declaration["target_std_errors"]}})
    return rows


def _work(task: tuple) -> list[dict]:
    declaration, system, replica = task
    return replica_rows(declaration, system, replica)


# ------------------------------------------------------------------ summaries

def _proportion(successes: int, trials: int, confidence: float, nominal: float) -> dict:
    lower, upper = wilson_interval(successes, trials, confidence)
    label = "under" if upper < nominal else "over" if lower > nominal else "consistent"
    return {"covered": successes, "n": trials, "rate": successes / trials,
            "wilson": [lower, upper], "nominal": nominal, "label": label}


def _error_statistics(errors: list[float]) -> dict | None:
    if not errors:
        return None
    values = np.asarray(errors, dtype=float)
    return {"n": len(errors), "rmse": float(np.sqrt(np.mean(values ** 2))),
            "median_abs_error": float(np.median(np.abs(values))),
            "max_abs_error": float(np.max(np.abs(values))),
            "mean_error": float(np.mean(values))}


def _rms(values: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _ratio_interval(numerators: list[np.ndarray], denominators: list[np.ndarray], *,
                    seed: int, replicates: int, confidence: float) -> dict:
    """A ratio of products of means over paired replicas, with a paired bootstrap.

    The statistic is ``prod(mean(n)) / prod(mean(d))``: one array each is a plain
    ratio of means, and (shots, squared error) on both sides is the ratio of
    work-normalized mean-squared errors. Replicas are resampled whole, so the
    pairing through the shared pilot is kept, and the interval is the
    percentile interval. ``None`` means undefined, not zero.
    """
    pairs = len(numerators[0])
    out: dict = {"pairs": pairs, "ratio": None, "interval": None}

    def product(arrays: list[np.ndarray], draws=None) -> np.ndarray:
        total = np.ones(() if draws is None else len(draws))
        for values in arrays:
            total = total * (values.mean() if draws is None else values[draws].mean(axis=1))
        return total

    if pairs < 2 or not float(product(denominators)) > 0.0:
        return out
    out["ratio"] = float(product(numerators) / product(denominators))
    draws = np.random.default_rng(seed).integers(0, pairs, size=(replicates, pairs))
    below = product(denominators, draws)
    if np.any(below <= 0.0):
        return out
    tail = (1.0 - confidence) / 2.0
    out["interval"] = [float(value) for value in np.quantile(
        product(numerators, draws) / below, [tail, 1.0 - tail])]
    return out


def _direction(found: dict) -> str:
    """Whether a uniform-over-targeted ratio is resolved away from one."""
    if found["interval"] is None:
        return "indistinguishable"
    lower, upper = found["interval"]
    return ("targeted_better" if lower > 1.0 else
            "uniform_better" if upper < 1.0 else "indistinguishable")


def summarize_fixed(members: list[dict], declaration: dict, design: dict) -> dict:
    """Error at every look of the full schedule, and coverage at its last."""
    target, groups = design["target"], design["qwc_groups"]
    looks = []
    for index, per_group in enumerate(declaration["cumulative_shots_per_group"]):
        present = [row["fixed"]["ledger"] for row in members
                   if len(row["fixed"]["ledger"]["energies"]) > index]
        looks.append({
            "look": index, "shots_per_group": per_group, "shots": per_group * groups,
            "errors": _error_statistics([ledger["energies"][index] - target
                                         for ledger in present]),
            "rms_std_error": _rms([ledger["std_errors"][index] for ledger in present]),
            "mean_retained_rank": (float(np.mean([ledger["retained_ranks"][index]
                                                  for ledger in present]))
                                   if present else None)})
    ok = [row["fixed"] for row in members if row["fixed"]["outcome"] == "ok"]
    return {"replicas": len(members), "solve_failures": len(members) - len(ok),
            "looks": looks,
            "final_coverage": _proportion(
                sum(run["interval"]["covered"] for run in ok), len(members),
                declaration["summary_confidence"], 1.0 - declaration["delta"])}


def summarize_stopping(members: list[dict], declaration: dict, design: dict,
                       goal: float) -> dict:
    """Cost, realized error and coverage of the stopped run, failures included."""
    runs = [row["stopping"][target_key(goal)] for row in members]
    ok = [run for run in runs if run["outcome"] == "ok"]
    confidence, nominal = declaration["summary_confidence"], 1.0 - declaration["delta"]
    shots = [run["shots"] for run in runs]
    return {
        "replicas": len(runs), "solve_failures": len(runs) - len(ok),
        "stopped": {reason: sum(run["stopped"] == reason for run in ok)
                    for reason in STOP_REASONS},
        "mean_shots": float(np.mean(shots)), "median_shots": float(np.median(shots)),
        "mean_shots_per_group": float(np.mean(shots)) / design["qwc_groups"],
        "mean_looks": float(np.mean([run["looks_taken"] for run in runs])),
        "errors": _error_statistics([run["estimate"] - design["target"] for run in ok]),
        "rms_std_error": _rms([run["std_error"] for run in ok]),
        "coverage": {
            "schedule_valid": _proportion(sum(run["interval"]["covered"] for run in ok),
                                          len(runs), confidence, nominal),
            "naive": _proportion(sum(run["naive_interval"]["covered"] for run in ok),
                                 len(runs), confidence, nominal)}}


def compare(by_allocation: dict[str, list[dict]], summaries: dict, declaration: dict,
            design: dict, system: int) -> dict:
    """Uniform against targeted, paired by replica: at equal shots and at equal error.

    At equal shots the figure is the ratio of mean-squared errors at the last
    look of the full schedule. Under a stopping target the two allocations stop
    at different looks with slightly different realized errors, so shots alone
    would not be a comparison at matched error. The figure is the ratio of
    work-normalized mean-squared errors, ``mean shots x MSE``: the shots uniform
    would need for the error targeting realized, over the shots targeting
    spent, if error falls as one over root shots.
    """
    resample = {"replicates": declaration["comparison"]["bootstrap_replicates"],
                "confidence": declaration["summary_confidence"]}
    arms = {name: {row["replica"]: row for row in by_allocation[name]}
            for name in (UNIFORM, TARGETED)}
    replicas = sorted(set(arms[UNIFORM]) & set(arms[TARGETED]))
    target = design["target"]

    def paired(select) -> tuple[list[int], dict[str, list[dict]]]:
        """Replicas on which both allocations produced an estimate, and their runs."""
        both = [r for r in replicas
                if all(select(arms[name][r])["outcome"] == "ok" for name in arms)]
        return both, {name: [select(rows[r]) for r in both] for name, rows in arms.items()}

    def squared(runs: list[dict]) -> np.ndarray:
        return np.array([(run["estimate"] - target) ** 2 for run in runs])

    _, runs = paired(lambda row: row["fixed"])
    budget = _ratio_interval([squared(runs[UNIFORM])], [squared(runs[TARGETED])],
                             seed=comparison_seed(declaration, system, 0), **resample)
    budget["label"] = _direction(budget)
    budget["shots"] = declaration["cumulative_shots_per_group"][-1] * design["qwc_groups"]

    stopping = {}
    for slot, goal in enumerate(declaration["target_std_errors"], start=1):
        key = target_key(goal)
        every = {name: np.array([float(rows[r]["stopping"][key]["shots"]) for r in replicas])
                 for name, rows in arms.items()}
        _, runs = paired(lambda row: row["stopping"][key])
        spent = {name: np.array([float(run["shots"]) for run in found])
                 for name, found in runs.items()}
        seed = comparison_seed(declaration, system, slot)
        matched = _ratio_interval([spent[UNIFORM], squared(runs[UNIFORM])],
                                  [spent[TARGETED], squared(runs[TARGETED])],
                                  seed=seed, **resample)
        direction = _direction(matched)
        stopping[key] = {
            # Every replica counts toward cost, a failed one at the shots it spent.
            "shots": _ratio_interval([every[UNIFORM]], [every[TARGETED]],
                                     seed=seed, **resample),
            "matched_error": matched, "label": direction,
            "verdict": {"targeted_better": "fewer_shots_at_matched_error",
                        "indistinguishable": "no_resolved_difference",
                        "uniform_better": "more_shots_at_matched_error"}[direction],
            "exhausted": {name: summaries[name]["stopping"][key]["stopped"]
                          ["schedule_exhausted"] for name in (UNIFORM, TARGETED)}}
    return {"matched_budget": budget, "stopping": stopping}


def summarize(declaration: dict, rows: list[dict]) -> dict:
    """Per-system summaries, paired comparisons and the headline label counts."""
    systems = []
    headline: dict = {
        "matched_budget": {label: 0 for label in EFFICIENCY_LABELS},
        "stopping": {target_key(goal): {verdict: 0 for verdict in VERDICTS}
                     for goal in declaration["target_std_errors"]},
        "coverage": {kind: {label: 0 for label in COVERAGE_LABELS}
                     for kind in ("fixed_final", "schedule_valid", "naive")},
        "solve_failures": 0}
    for index, spec in enumerate(declaration["systems"]):
        design = build_system(spec)[1]
        by_allocation = {name: [row for row in rows if row["system"] == spec["key"]
                                and row["allocation"] == name]
                         for name in declaration["allocations"]}
        summaries = {}
        for name, members in by_allocation.items():
            fixed = summarize_fixed(members, declaration, design)
            stopping = {target_key(goal): summarize_stopping(members, declaration, design, goal)
                        for goal in declaration["target_std_errors"]}
            summaries[name] = {"fixed": fixed, "stopping": stopping}
            headline["coverage"]["fixed_final"][fixed["final_coverage"]["label"]] += 1
            headline["solve_failures"] += fixed["solve_failures"]
            for found in stopping.values():
                headline["solve_failures"] += found["solve_failures"]
                for kind in ("schedule_valid", "naive"):
                    headline["coverage"][kind][found["coverage"][kind]["label"]] += 1
        comparison = compare(by_allocation, summaries, declaration, design, index)
        headline["matched_budget"][comparison["matched_budget"]["label"]] += 1
        for key, found in comparison["stopping"].items():
            headline["stopping"][key][found["verdict"]] += 1
        systems.append({"system": spec["key"], "role": spec["role"],
                        "allocations": summaries, "comparison": comparison})
    return {"summaries": systems, "headline": headline}


# ------------------------------------------------------------------ execution

def expected_tasks(declaration: dict, *, replicas: int | None = None) -> list[tuple]:
    count = declaration["replicas"] if replicas is None else min(declaration["replicas"],
                                                                  replicas)
    return [(declaration, system, replica)
            for system in range(len(declaration["systems"])) for replica in range(count)]


def run(declaration: dict, *, jobs: int = 1) -> list[dict]:
    tasks = expected_tasks(declaration)
    if jobs == 1:
        batches = [_work(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            batches = list(pool.map(_work, tasks, chunksize=4))
    return [row for batch in batches for row in batch]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_record(declaration_path: Path, rows: list[dict], *, declaration_commit: str) -> dict:
    declaration = load_declaration(declaration_path)
    designs = {system["key"]: build_system(system)[1] for system in declaration["systems"]}
    return {
        "schema": RESULTS_SCHEMA,
        "declaration": declaration,
        "declaration_sha256": sha256(declaration_path),
        "declaration_commit": declaration_commit,
        "implementation": implementation_digests(),
        "systems": designs,
        "rows": rows,
        **summarize(declaration, rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, default=DECLARATION)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.output.exists():
        print(f"FAIL {args.output} exists; a declared record is never overwritten")
        return 1
    declaration = load_declaration(args.declaration)
    relative = args.declaration.resolve().relative_to(ROOT).as_posix()
    if _git("status", "--porcelain", "--", relative, *IMPLEMENTATION):
        print("FAIL the declaration and implementation must be committed before the run")
        return 1
    commit = _git("log", "-1", "--format=%H", "--", relative)
    rows = run(declaration, jobs=args.jobs)
    record = stamp_record(build_record(args.declaration, rows, declaration_commit=commit))
    write_record(record, args.output)
    print(f"wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
