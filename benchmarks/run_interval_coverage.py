#!/usr/bin/env python3
"""Run the declared interval-coverage study and write its record.

The study, its systems, estimator arms, seeds and reporting labels are fixed
by ``docs/restart/interval_coverage_declaration.json`` before any evaluation
seed is drawn; ``docs/restart/INTERVAL_COVERAGE.md`` states what is and is not
claimed. This producer only executes that declaration:

    python benchmarks/run_interval_coverage.py --jobs 4

Every (cell, replica) is one independent, seeded unit of work, so the record
does not depend on ``--jobs``. An existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifford_qc.algorithms.pools import local_pool, odd_y_filter  # noqa: E402
from clifford_qc.backends import ExactMVBackend  # noqa: E402
from clifford_qc.backends.finite_shot import FiniteShotBackend  # noqa: E402
from clifford_qc.measurement.session import SharedMeasurement  # noqa: E402
from clifford_qc.models.spin import tfim, xxz  # noqa: E402
from clifford_qc.reproducibility import stamp_record  # noqa: E402
from clifford_qc.subspace import (  # noqa: E402
    MatrixElementBank, identity_generator, krylov_response, pauli_orbit,
)
from clifford_qc.subspace.coverage import (  # noqa: E402
    METHODS, interval_replica, level_key, summarize_coverage,
)

DECLARATION = ROOT / "docs" / "restart" / "interval_coverage_declaration.json"
RESULTS = ROOT / "docs" / "restart" / "interval_coverage_results.json.gz"
DECLARATION_SCHEMA = "clifford_qc.interval_coverage_declaration.v1"
RESULTS_SCHEMA = "clifford_qc.interval_coverage_results.v1"
MODELS = {"tfim": tfim, "xxz": xxz}
# The sources whose bytes decide a replica row. A later change to any of them
# means the committed rows describe the recorded implementation, not the live one.
IMPLEMENTATION = (
    "benchmarks/run_interval_coverage.py",
    "clifford_qc/subspace/coverage.py",
    "clifford_qc/subspace/uncertainty.py",
    "clifford_qc/subspace/pencil.py",
    "clifford_qc/subspace/linalg.py",
    "clifford_qc/measurement/session.py",
    "clifford_qc/measurement/functionals.py",
    "clifford_qc/measurement/cache.py",
    "clifford_qc/measurement/confidence.py",
    "clifford_qc/measurement/grouping.py",
    "clifford_qc/backends/finite_shot.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_digests(root: Path = ROOT) -> dict[str, str]:
    return {relative: sha256(root / relative) for relative in IMPLEMENTATION}


def load_declaration(path: Path = DECLARATION) -> dict:
    """Read and validate a declaration; every later step trusts its shape."""
    declaration = json.loads(Path(path).read_text(encoding="utf-8"))
    if declaration.get("schema") != DECLARATION_SCHEMA:
        raise ValueError("unsupported interval-coverage declaration schema")
    if declaration.get("target") != "exact_subspace_ground_energy":
        raise ValueError("the only supported target is exact_subspace_ground_energy")
    systems = [system["key"] for system in declaration["systems"]]
    if len(set(systems)) != len(systems) or not systems:
        raise ValueError("system keys must be nonempty and distinct")
    for system in declaration["systems"]:
        if system["model"].get("family") not in MODELS:
            raise ValueError(f"unknown model family in {system['key']}")
    shots = declaration["shots_per_group"]
    if not shots or len(set(shots)) != len(shots) or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in shots):
        raise ValueError("shots_per_group must be distinct positive integers")
    if any(method not in METHODS for method in declaration["methods"]):
        raise ValueError("unknown interval method")
    arms = [arm["key"] for arm in declaration["arms"]]
    if len(set(arms)) != len(arms) or not arms:
        raise ValueError("arm keys must be nonempty and distinct")
    for arm in declaration["arms"]:
        if not set(arm["systems"]) <= set(systems):
            raise ValueError(f"arm {arm['key']} names an undeclared system")
        if arm["replicas"] < 1 or arm["bootstrap_replicates"] < 2:
            raise ValueError(f"arm {arm['key']} needs replicas and bootstrap replicates")
        budgets = arm.get("ratio_error_budget_by_shots")
        if budgets is not None and set(budgets) != {str(s) for s in shots}:
            raise ValueError(f"arm {arm['key']} must declare one budget per shot count")
    if declaration["spot_replay_replicas"] < 1:
        raise ValueError("spot_replay_replicas must be positive")
    return declaration


def cells(declaration: dict) -> list[dict]:
    """The ordered system x shots grid; a cell's index seeds its measurements."""
    out = []
    for system in declaration["systems"]:
        for shots in declaration["shots_per_group"]:
            out.append({"cell": len(out), "system": system["key"], "shots_per_group": shots})
    return out


def arm_kwargs(arm: dict, shots: int) -> dict:
    kwargs = dict(arm["solver_kwargs"])
    budgets = arm.get("ratio_error_budget_by_shots")
    if budgets is not None:
        kwargs["ratio_error_budget"] = budgets[str(shots)]
    return kwargs


def derived_seed(*entropy: int) -> int:
    """One 64-bit seed from the declared root and a row's coordinates."""
    return int(np.random.SeedSequence(list(entropy)).generate_state(1, dtype=np.uint64)[0])


def measurement_seed(declaration: dict, cell: int, replica: int) -> int:
    return derived_seed(declaration["seed_root"], cell, replica)


def bootstrap_seed(declaration: dict, cell: int, replica: int, arm_index: int) -> int:
    return derived_seed(declaration["seed_root"], cell, replica, arm_index + 1)


_SYSTEMS: dict[str, tuple] = {}


def build_system(spec: dict) -> tuple:
    """``(shared, design)`` for one declared system; exact arithmetic only."""
    key = json.dumps(spec, sort_keys=True)
    if key not in _SYSTEMS:
        parameters = dict(spec["model"])
        model = MODELS[parameters.pop("family")](**parameters)
        n = spec["model"]["n"]
        reference = ExactMVBackend().state(model.reference, ())
        words = [op.word for op in odd_y_filter(local_pool(n))]
        generators = [identity_generator(n)] + pauli_orbit(words[:spec["pool_words"]])
        if spec["krylov_powers"]:
            generators += krylov_response(model.hamiltonian, spec["krylov_powers"])
        bank = MatrixElementBank(reference, model.hamiltonian, generators)
        shared = SharedMeasurement(bank)
        exact = bank.solve()
        design = {
            "generators": len(generators),
            "exact_rank": int(exact.effective_rank),
            "exact_condition_number": float(exact.condition_number),
            "measured_words": len(shared.words),
            "qwc_groups": len(shared.groups),
            "target": float(exact.ground_energy),
        }
        _SYSTEMS[key] = (shared, design)
    shared, design = _SYSTEMS[key]
    return shared, dict(design)


def replica_rows(declaration: dict, cell: dict, replica: int) -> list[dict]:
    """Every declared arm's row for one measurement of one cell.

    All arms read the same joint histograms, so differences between arms on a
    replica are estimator differences, not sampling differences.
    """
    spec = next(s for s in declaration["systems"] if s["key"] == cell["system"])
    shared, design = build_system(spec)
    seed = measurement_seed(declaration, cell["cell"], replica)
    cache = shared.measure(FiniteShotBackend(seed=seed), cell["shots_per_group"])
    rows = []
    for index, arm in enumerate(declaration["arms"]):
        if cell["system"] not in arm["systems"] or replica >= arm["replicas"]:
            continue
        resample = bootstrap_seed(declaration, cell["cell"], replica, index)
        row = interval_replica(
            shared, cache, target=design["target"],
            solver_kwargs=arm_kwargs(arm, cell["shots_per_group"]),
            levels=declaration["levels"], methods=declaration["methods"],
            bootstrap_replicates=arm["bootstrap_replicates"], bootstrap_seed=resample)
        rows.append({"cell": cell["cell"], "arm": arm["key"], "replica": replica,
                     "measurement_seed": seed, "bootstrap_seed": resample, **row})
    return rows


def _work(task: tuple) -> list[dict]:
    declaration, cell, replica = task
    return replica_rows(declaration, cell, replica)


def classify(level: dict) -> str:
    """The declared label for one all-replica coverage rate."""
    lower, upper = level["all_replicas"]["wilson"]
    if upper < level["nominal"]:
        return "under"
    if lower > level["nominal"]:
        return "over"
    return "consistent"


def summarize(declaration: dict, rows: list[dict]) -> dict:
    """Per-cell summaries, declared labels and the headline label counts."""
    summaries = []
    headline: dict[str, dict] = {}
    for cell in cells(declaration):
        for arm in declaration["arms"]:
            members = [row for row in rows
                       if row["cell"] == cell["cell"] and row["arm"] == arm["key"]]
            if not members:
                continue
            summary = summarize_coverage(
                members, levels=declaration["levels"], methods=declaration["methods"],
                strata_edges=declaration["strata_edges"],
                confidence=declaration["summary_confidence"])
            labels: dict[str, dict] = {}
            for method, found in summary["methods"].items():
                if not found["applicable"]:
                    continue
                labels[method] = {}
                tally = headline.setdefault(arm["key"], {}).setdefault(
                    method, {"under": 0, "consistent": 0, "over": 0,
                             "interval_failures": 0, "solve_failures": 0})
                tally["interval_failures"] += found["interval_failures"]
                tally["solve_failures"] += summary["solve_failures"]
                for level in declaration["levels"]:
                    label = classify(found["levels"][level_key(level)])
                    labels[method][level_key(level)] = label
                    tally[label] += 1
            summaries.append({**cell, "arm": arm["key"], "labels": labels, "summary": summary})
    return {"cells": summaries, "headline": headline}


def expected_tasks(declaration: dict, *, replicas: int | None = None) -> list[tuple]:
    most = max(arm["replicas"] for arm in declaration["arms"])
    count = most if replicas is None else min(most, replicas)
    return [(declaration, cell, replica)
            for cell in cells(declaration) for replica in range(count)]


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


def write_record(record: dict, output: Path) -> None:
    """Deterministic gzip bytes: no timestamp, sorted keys, no NaN."""
    payload = json.dumps(record, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode()
    with open(output, "xb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as stream:
            stream.write(payload)


def read_record(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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
