"""Accuracy-only K=64/128 packet probes on the preregistered large sectors.

This is deliberately a separate producer from packet word pricing.  The v2
manifest authorizes these supports only as exact-simulation accuracy probes;
it does not authorize building their potentially enormous cross-element word
banks or grouping them.  Every row therefore carries ``W_total = null`` and an
explicit scope boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from benchmarks import run_prd_case_paper_suite as suite


DEFAULT_CONFIG = ROOT / "configs" / "prd_case_paper_suite_v2.json"
DEFAULT_RESULTS = Path("/tmp/clifford_qc_prd_case_large_sector_packets")
SCHEMA = "clifford_qc.prd_case_large_sector_packets.v1"
CHECKPOINT_SCHEMA = "clifford_qc.prd_case_large_sector_checkpoint.v1"


def build_tasks(manifest: dict, *, system_ids=()) -> list[dict]:
    selected = set(system_ids)
    extension = manifest["exact_simulation"]["accuracy_only_packet_extension"]
    packet_k = [int(value) for value in
                extension["packet_k_for_sector_dimension_at_least_400"]]
    tasks = []
    for system in manifest["systems"]:
        dimension = int(system.get("expected", {}).get("sector_dimension", 0))
        if (not system["default_run"] or system["analysis_role"] != "primary"
                or dimension < 400
                or (selected and system["id"] not in selected)):
            continue
        tasks.append({
            "task_id": system["id"],
            "system": system["id"],
            "m_budgets": list(system["m_budgets"]),
            "packet_k": packet_k,
            "builder": system["builder"],
            "expected": system["expected"],
        })
    return tasks


def _run_row(pe, counting, reference, diagonal, M, K, mu, exact_energy):
    before = counting.matvecs
    started = time.perf_counter()
    run = pe.expand(
        counting, reference, int(M), diagonal=diagonal,
        mu=float(mu), packet_K=int(K))
    wall = time.perf_counter() - started
    if "breakdown" in run:
        raise AssertionError(f"M={M}, K={K}: {run['breakdown']}")
    energy = float(run["energy"])
    return {
        "method": f"packet_davidson[K={int(K)}]",
        "evidence_category": "exact_simulation_accuracy_only",
        "M": int(run["M"]),
        "declared_total_directions": int(M),
        "realized_total_directions": int(run["M"]),
        "packet_K": int(K),
        "mu": float(mu),
        "energy": energy,
        "energy_error": energy - exact_energy,
        "absolute_error": abs(energy - exact_energy),
        "variance": float(run["variance"]),
        "true_residual": float(run["true_residual"]),
        "retained_rank": int(run["retained_rank"]),
        "kappa_S": float(run["kappa_S"]),
        "orthonormality_defect": float(run["orthonormality_defect"]),
        "matvecs": int(counting.matvecs - before),
        "wall_seconds": wall,
        "nested_energies": [float(value) for value in run["energies"]],
        "packet_support_cardinalities": [int(len(support))
                                         for support in run["packet_supports"]],
        "packet_norm_captures": [float(value)
                                 for value in run["packet_norm_captures"]],
        "W_total": None,
        "W_incremental": None,
        "W_scope": (
            "not evaluated: K=64/128 is preregistered as accuracy-only and "
            "does not authorize cross-element word pricing or grouping"),
        "grouping_contexts": {
            "status": "forbidden_for_accuracy_only_extension",
            "scope": "no packet word bank was constructed",
        },
    }


def check_invariants(record: dict, tolerance=1e-9) -> None:
    by_M = {}
    for row in record["rows"]:
        if row["M"] != row["declared_total_directions"]:
            raise AssertionError("large-sector packet arm did not realize M")
        if row["retained_rank"] != row["M"]:
            raise AssertionError("large-sector packet arm lost effective rank")
        if row["energy_error"] < -tolerance:
            raise AssertionError("large-sector packet arm violates variationality")
        if row["W_total"] is not None:
            raise AssertionError("accuracy-only extension priced a forbidden W")
        if any(not 0.0 <= value <= 1.0 + 1e-12
               for value in row["packet_norm_captures"]):
            raise AssertionError("packet norm capture is outside [0, 1]")
        by_M.setdefault(row["M"], []).append(row)
    for M, rows in by_M.items():
        rows.sort(key=lambda row: row["packet_K"])
        if any(right["absolute_error"] > left["absolute_error"] + tolerance
               for left, right in zip(rows, rows[1:])):
            raise AssertionError(
                f"M={M}: larger top-K support worsened packet accuracy")


def execute_task(task: dict, manifest: dict) -> dict:
    from benchmarks import run_preconditioned_expansion as pe
    from clifford_qc.backends import SectorStatevectorBackend
    from clifford_qc.reproducibility import stamp_record

    model, construction = suite.build_system_from_spec(
        task["system"], task["builder"])
    backend = SectorStatevectorBackend(
        model.n, int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]))
    expected = task["expected"]
    if int(model.n) != int(expected["n_qubits"]):
        raise AssertionError("large-sector construction changed n_qubits")
    if int(backend.dimension) != int(expected["sector_dimension"]):
        raise AssertionError("large-sector construction changed sector dimension")
    operator = backend.operator(model.hamiltonian)
    counting = pe._CountingOperator(operator)
    exact_energy = float(backend.ground_state(model.hamiltonian, k=1)[0][0])
    diagonal_before = counting.matvecs
    diagonal = pe._sector_diagonal(counting, backend.dimension)
    diagonal_matvecs = counting.matvecs - diagonal_before
    policies = pe.reference_policies(
        model, backend, diagonal, int(manifest["exact_simulation"]["seed"]))
    primary = policies["model_reference"]
    reference = np.zeros(backend.dimension, dtype=complex)
    reference[primary["position"]] = 1.0

    rows = []
    sweeps = []
    for M in task["m_budgets"]:
        before = counting.matvecs
        started = time.perf_counter()
        sweep = pe.select_shift(
            counting, reference, int(M), diagonal,
            manifest["exact_simulation"]["shift_grid"])
        sweep["M"] = int(M)
        sweep["matvecs"] = int(counting.matvecs - before)
        sweep["wall_seconds"] = time.perf_counter() - started
        sweeps.append(sweep)
        for K in task["packet_k"]:
            rows.append(_run_row(
                pe, counting, reference, diagonal, M, K,
                sweep["selected_mu"], exact_energy))

    record = {
        "schema": SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "execution_stage": "large_sector_packet_accuracy",
        "system": task["system"],
        "construction": construction,
        "n_qubits": int(model.n),
        "n_electrons": int(model.metadata["n_electrons"]),
        "sz": float(model.metadata["sz"]),
        "sector_dimension": int(backend.dimension),
        "spin_ordering": str(backend.spin_ordering),
        "reference_policy": "model_reference",
        "reference_identity": primary["identity"],
        "exact_energy": exact_energy,
        "diagonal_matvecs": int(diagonal_matvecs),
        "shift_sweeps": sweeps,
        "rows": rows,
        "paper_analysis_complete": False,
        "pending_stage": "finite_shot",
        "claim_boundary": (
            "K=64/128 exact accuracy only. No word bank, grouping, shot, "
            "hardware-runtime, implementability, or quantum-advantage claim."),
    }
    check_invariants(record)
    return stamp_record(record)


def _checkpoint(manifest, tasks):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "created_utc_unix": time.time(),
        "tasks": {task["task_id"]: {"status": "pending"} for task in tasks},
    }


def load_checkpoint(path, manifest, tasks):
    if not path.exists():
        return _checkpoint(manifest, tasks)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("large-sector checkpoint schema is incompatible")
    if document.get("config_sha256") != suite.config_sha256(manifest):
        raise ValueError("large-sector checkpoint belongs to another manifest")
    for task in tasks:
        document["tasks"].setdefault(task["task_id"], {"status": "pending"})
    return document


def _csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--systems", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    manifest = suite.load_manifest(args.config)
    tasks = build_tasks(manifest, system_ids=_csv(args.systems))
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "config_sha256": suite.config_sha256(manifest),
        "task_count": len(tasks),
        "implemented_stage": "large_sector_packet_accuracy",
        "tasks": tasks,
    }, indent=2))
    if not args.execute:
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.results_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, manifest, tasks)
    suite._atomic_json(checkpoint_path, checkpoint)
    failures = 0
    for task in tasks:
        state = checkpoint["tasks"][task["task_id"]]
        if state.get("status") == "completed":
            continue
        state.update({"status": "running", "started_utc_unix": time.time()})
        suite._atomic_json(checkpoint_path, checkpoint)
        output = args.results_dir / f"{task['task_id']}.json"
        try:
            suite._atomic_json(output, execute_task(task, manifest))
            state.update({"status": "completed", "output": output.name,
                          "finished_utc_unix": time.time()})
        except Exception as error:
            failures += 1
            state.update({"status": "failed", "error": repr(error),
                          "finished_utc_unix": time.time()})
            suite._atomic_json(checkpoint_path, checkpoint)
            if args.fail_fast:
                raise
        suite._atomic_json(checkpoint_path, checkpoint)
    if failures:
        raise SystemExit(f"{failures} task(s) failed; see {checkpoint_path}")


if __name__ == "__main__":
    main()
