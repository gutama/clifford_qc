"""Checkpointed matched-M A-CASE stage for the PRD-CASE paper suite.

Each task runs one system and one generator resolution through the largest
declared budget, then snapshots every preregistered prefix.  A single nested
trajectory is mathematically identical to rerunning each smaller fixed budget,
while avoiding repeated candidate scoring.  Both the retained measurement
bank and the larger selection cache are priced: rejected candidates are not
free merely because they do not enter the Ritz subspace.

Dry-run is the default.  Use ``--execute`` to write one provenance-stamped JSON
record per (system, resolution) task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from benchmarks import run_prd_case_paper_suite as suite


DEFAULT_CONFIG = ROOT / "configs" / "prd_case_paper_suite_v2.json"
DEFAULT_RESULTS = Path("/tmp/clifford_qc_prd_case_matched_acase")
SCHEMA = "clifford_qc.prd_case_matched_acase_arm.v1"
CHECKPOINT_SCHEMA = "clifford_qc.prd_case_matched_acase_checkpoint.v1"
METHODS = ("acase_determinant", "acase_word")
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


def _strict_methods(values) -> tuple[str, ...]:
    methods = tuple(values) if values else METHODS
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown matched A-CASE methods: {unknown}")
    return tuple(method for method in METHODS if method in methods)


def build_tasks(manifest: dict, *, system_ids=(), methods=(),
                include_exploratory=False) -> list[dict]:
    """One task per system and resolution, carrying every matched M prefix."""
    selected = set(system_ids)
    method_order = _strict_methods(methods)
    global_m = manifest["exact_simulation"]["m_budgets"]
    tasks = []
    for system in manifest["systems"]:
        enabled = bool(system["default_run"])
        if include_exploratory and system["analysis_role"] == "exploratory_scaling":
            enabled = True
        if not enabled or (selected and system["id"] not in selected):
            continue
        budgets = tuple(int(value) for value in system.get("m_budgets", global_m))
        for method in method_order:
            tasks.append({
                "task_id": f"{system['id']}__{method}",
                "system": system["id"],
                "method": method,
                "m_budgets": list(budgets),
                "analysis_role": system["analysis_role"],
                "builder": system["builder"],
                "expected": system.get("expected", {}),
                "validation_target": system.get("validation_target"),
                "energy_tolerance": system.get("energy_tolerance"),
            })
    return tasks


def word_candidates(model, determinants):
    """Odd-Y Pauli resolution of the declared determinant-excitation pool."""
    from clifford_qc.algorithms.pools import is_odd_y
    from clifford_qc.ir import PauliWord
    from clifford_qc.subspace import pauli_orbit

    words = {}
    for generator in determinants:
        for code in sorted(generator.mv.terms):
            word = PauliWord(model.n, code)
            if code and is_odd_y(word):
                words.setdefault(code, word)
    if not words:
        raise ValueError("determinant pool has no odd-Y Pauli resolution")
    return pauli_orbit(list(words.values()))


def _direction_fingerprint(vector: np.ndarray) -> str:
    """Phase-invariant digest of one normalized reference-acted direction."""
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-14:
        raise ValueError("cannot fingerprint a zero reference-acted direction")
    vector = vector / norm
    pivot = int(np.argmax(np.abs(vector)))
    vector = vector * np.conjugate(vector[pivot]) / abs(vector[pivot])
    payload = [
        [int(index), round(float(value.real), 12), round(float(value.imag), 12)]
        for index, value in enumerate(vector) if abs(value) > 1e-12
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _grouping_context(bank, indices, word_limit: int) -> dict:
    words = bank.words(indices)
    if len(words) > word_limit:
        return {
            "status": "skipped_above_declared_limit",
            "word_universe": len(words),
            "word_limit": int(word_limit),
            "scope": "retained A-CASE S/H bank",
        }
    from clifford_qc.measurement.grouping import qwc_groups

    started = time.perf_counter()
    groups = len(qwc_groups(words))
    return {
        "status": "computed",
        "word_universe": len(words),
        "qwc_groups": int(groups),
        "grouping_seconds": time.perf_counter() - started,
        "grouping_rule": "greedy qubit-wise commuting partition",
        "scope": "retained A-CASE S/H bank",
    }


def _reference_identity(model, backend) -> tuple[str, np.ndarray]:
    state = backend.state_from_program(model.reference)
    position = int(np.argmax(np.abs(state)))
    mask = int(backend.basis[position])
    return f"model_reference:mask={mask}", state


def _snapshot(adaptive, M: int, exact_energy: float, reference_vector,
              hamiltonian_operator, direction_vectors, fingerprints,
              baseline_W: int, grouping_word_limit: int) -> dict:
    if adaptive.basis_size < M:
        raise AssertionError(
            f"A-CASE stopped at M={adaptive.basis_size}, below declared M={M}: "
            f"{adaptive.stopped_reason}")
    records = adaptive.records[:M - 1]
    indices = adaptive.indices[:M]
    solved = adaptive.bank.solve(indices)
    resources = solved.resources
    coefficients = np.asarray(solved.coefficients[:, 0], dtype=complex)
    basis = np.column_stack(direction_vectors[:M])
    state = basis @ coefficients
    state = state / np.linalg.norm(state)
    acted = hamiltonian_operator.matvec(state)
    residual = acted - solved.ground_energy * state
    variance = max(0.0, float(np.vdot(residual, residual).real))
    overlap = adaptive.bank.matrices(indices)[0]
    normalization_defect = abs(float(
        (coefficients.conj() @ overlap @ coefficients).real) - 1.0)
    energy_error = float(solved.ground_energy - exact_energy)
    retained_W = int(resources["word_universe"])
    last = records[-1]
    return {
        "M": int(M),
        "declared_total_directions": int(M),
        "realized_total_directions": len(indices),
        "retained_rank": int(solved.effective_rank),
        "kappa_S": float(solved.condition_number),
        "orthonormality_defect": normalization_defect,
        "energy": float(solved.ground_energy),
        "energy_error": energy_error,
        "absolute_error": abs(energy_error),
        "chemical_accuracy": bool(abs(energy_error) <= CHEMICAL_ACCURACY_HARTREE),
        "variance": variance,
        "true_residual": float(np.sqrt(variance)),
        "mu": None,
        "matvecs": 0,
        "matvecs_scope": (
            "A-CASE selection uses cached matrix elements, not sector-H "
            "matvecs; the one residual-diagnostic matvec is excluded"),
        "selection_work": int(sum(record.candidates_scored for record in records)),
        "W_total": retained_W,
        "W_incremental": retained_W - int(baseline_W),
        "W_selection_cache": int(last.selection_word_universe),
        "W_scope": (
            "W_total is the complete retained identity-plus-selected A-CASE "
            "basis, including every overlap and Hamiltonian cross element; "
            "W_selection_cache additionally includes all candidate rows "
            "measured through this selection step, including rejected rows."),
        "grouping_contexts": _grouping_context(
            adaptive.bank, indices, grouping_word_limit),
        "wall_seconds": float(
            adaptive.resources["initialization_seconds"]
            + sum(record.step_seconds for record in records)),
        "wall_seconds_scope": "nested A-CASE growth through this M",
        "selected_labels": list(solved.basis_labels),
        "direction_fingerprints": list(fingerprints[:M]),
        "max_generator_support": int(resources["max_generator_support"]),
        "max_overlap_element_support": int(
            resources["max_overlap_element_support"]),
        "max_hamiltonian_element_support": int(
            resources["max_hamiltonian_element_support"]),
        "nested_energies": [float(value) for value in adaptive.energy_history[:M]],
    }


def check_arm_invariants(record: dict, manifest: dict) -> None:
    tolerance = float(
        manifest["producer_invariants"]["variational_tolerance_hartree"])
    nested_tolerance = float(
        manifest["producer_invariants"]["nested_energy_tolerance_hartree"])
    previous = None
    for row in record["rows"]:
        if row["realized_total_directions"] != row["M"]:
            raise AssertionError("A-CASE did not realize its declared M")
        if row["retained_rank"] != row["M"]:
            raise AssertionError("A-CASE retained rank does not match M")
        if row["energy_error"] < -tolerance:
            raise AssertionError("A-CASE violates the exact variational bound")
        if previous is not None and row["energy"] > previous + nested_tolerance:
            raise AssertionError("nested A-CASE energy increased with M")
        if row["W_total"] < record["hamiltonian_words"]:
            raise AssertionError("retained word ledger dropped Hamiltonian words")
        if row["W_selection_cache"] < row["W_total"]:
            raise AssertionError("selection cache is narrower than retained bank")
        previous = row["energy"]
    certificate = record.get("sector_certificate")
    if record["method"] == "acase_word":
        if not certificate or certificate["max_sector_leakage"] > record["leakage_tol"]:
            raise AssertionError("word-resolution span lacks its sector certificate")


def execute_task(task: dict, manifest: dict) -> dict:
    from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
    from clifford_qc.subspace import (
        determinant_excitations,
        occupied_spin_orbitals,
        run_acase,
    )
    from clifford_qc.reproducibility import stamp_record

    model, construction = suite.build_system_from_spec(
        task["system"], task["builder"])
    backend = SectorStatevectorBackend(
        model.n, int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]))
    exact_energy = float(backend.ground_state(model.hamiltonian, k=1)[0][0])
    hamiltonian_operator = backend.operator(model.hamiltonian)
    reference_identity, reference_vector = _reference_identity(model, backend)
    rho = ExactMVBackend().state(model.reference, ())
    determinants = determinant_excitations(
        model.n, occupied_spin_orbitals(model), max_rank=2)
    target = (int(model.metadata["n_electrons"]),
              float(model.metadata["sz"]))
    leakage_tol = 1e-9
    if task["method"] == "acase_determinant":
        candidates = determinants
        resolution = "whole symmetry-preserving determinant excitations"
        leakage_mode = "operator"
        sector_target = None
    else:
        candidates = word_candidates(model, determinants)
        resolution = "odd-Y Pauli words from the same determinant excitations"
        leakage_mode = "operator_then_reference"
        sector_target = target

    max_M = max(task["m_budgets"])
    adaptive = run_acase(
        rho, model.hamiltonian, candidates,
        max_size=max_M - 1,
        min_lowering=0.0,
        leakage_tol=leakage_tol,
        leakage_mode=leakage_mode,
        sector_target=sector_target,
        exact_ground_energy=exact_energy,
    )
    expected = task.get("expected") or {}
    observed = {"n_qubits": int(model.n),
                "sector_dimension": int(backend.dimension)}
    mismatches = {key: {"expected": int(expected[key]),
                        "observed": observed[key]}
                  for key in observed if key in expected
                  and int(expected[key]) != observed[key]}
    if mismatches:
        raise AssertionError(f"{task['task_id']}: construction mismatch {mismatches}")

    retained_generators = [adaptive.bank._generators[index]
                           for index in adaptive.indices]
    direction_vectors = [
        backend.operator(generator.mv, validate_sector=False).matvec(reference_vector)
        for generator in retained_generators
    ]
    fingerprints = [_direction_fingerprint(vector)
                    for vector in direction_vectors]
    baseline_W = adaptive.bank.resources(adaptive.indices[:1])["word_universe"]
    rows = [
        _snapshot(
            adaptive, int(M), exact_energy, reference_vector,
            hamiltonian_operator, direction_vectors, fingerprints,
            int(baseline_W),
            int(manifest["exact_simulation"]["grouping_word_limit"]),
        )
        for M in task["m_budgets"]
    ]
    record = {
        "schema": SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "execution_stage": "matched_acase",
        "evidence_category": "exact_simulation",
        "method": task["method"],
        "candidate_resolution": resolution,
        "candidate_pool_size": len(candidates),
        "determinant_pool_size": len(determinants),
        "system": task["system"],
        "analysis_role": task["analysis_role"],
        "construction": construction,
        "n_qubits": int(model.n),
        "n_electrons": target[0],
        "sz": target[1],
        "sector_dimension": int(backend.dimension),
        "spin_ordering": str(backend.spin_ordering),
        "reference_policy": "model_reference",
        "reference_identity": reference_identity,
        "exact_energy": exact_energy,
        "hamiltonian_words": len(model.hamiltonian.terms),
        "leakage_tol": leakage_tol,
        "leakage_mode": leakage_mode,
        "sector_certificate": adaptive.resources.get("sector_certificate"),
        "rows": rows,
        "arm_wall_seconds": float(adaptive.resources["growth_seconds"]),
        "stopped_reason": adaptive.stopped_reason,
        "paper_analysis_complete": False,
        "pending_stages": ["large_sector_packet_accuracy", "finite_shot"],
        "claim_boundary": (
            "Exact fixed-reference A-CASE arithmetic and exact algorithmic "
            "word counts only. QWC grouping is a constructive upper bound. "
            "No physical shot, runtime, or quantum-advantage claim is made."),
    }
    check_arm_invariants(record, manifest)
    return stamp_record(record)


def _new_checkpoint(manifest: dict, tasks: list[dict]) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "created_utc_unix": time.time(),
        "tasks": {task["task_id"]: {"status": "pending"} for task in tasks},
    }


def load_checkpoint(path: Path, manifest: dict, tasks: list[dict]) -> dict:
    if not path.exists():
        return _new_checkpoint(manifest, tasks)
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("matched A-CASE checkpoint schema is incompatible")
    if checkpoint.get("config_sha256") != suite.config_sha256(manifest):
        raise ValueError("checkpoint belongs to a different manifest; refusing resume")
    for task in tasks:
        checkpoint.setdefault("tasks", {}).setdefault(
            task["task_id"], {"status": "pending"})
    return checkpoint


def cross_check_resolution_records(tasks: list[dict], results_dir: Path,
                                   tolerance=1e-10) -> dict:
    checks = []
    systems = sorted({task["system"] for task in tasks})
    for system in systems:
        paths = {method: results_dir / f"{system}__{method}.json"
                 for method in METHODS}
        if not all(path.exists() for path in paths.values()):
            checks.append({"system": system, "status": "arm_missing"})
            continue
        documents = {method: json.loads(path.read_text(encoding="utf-8"))
                     for method, path in paths.items()}
        coarse = {row["M"]: row for row in documents["acase_determinant"]["rows"]}
        fine = {row["M"]: row for row in documents["acase_word"]["rows"]}
        for M in sorted(set(coarse) & set(fine)):
            energy_gap = abs(float(coarse[M]["energy"] - fine[M]["energy"]))
            same_span = (sorted(coarse[M]["direction_fingerprints"])
                         == sorted(fine[M]["direction_fingerprints"]))
            status = "passed" if energy_gap <= tolerance and same_span else "failed"
            check = {"system": system, "M": M, "energy_gap": energy_gap,
                     "energy_tolerance": tolerance,
                     "direction_fingerprints_match": same_span,
                     "status": status}
            checks.append(check)
            if status == "failed":
                raise AssertionError(
                    f"{system} M={M}: A-CASE resolutions disagree: {check}")
    return {"schema": "clifford_qc.prd_case_acase_resolution_checks.v1",
            "checks": checks}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--systems", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--include-exploratory", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    manifest = suite.load_manifest(args.config)
    if manifest["schema"] != suite.CONFIRMATORY_SCHEMA:
        raise SystemExit("matched A-CASE stage requires the v2 resolved manifest")
    systems = _parse_csv(args.systems)
    methods = _parse_csv(args.methods)
    known = {entry["id"] for entry in manifest["systems"]}
    unknown = sorted(set(systems) - known)
    if unknown:
        raise SystemExit(f"unknown systems: {unknown}")
    tasks = build_tasks(
        manifest, system_ids=systems, methods=methods,
        include_exploratory=args.include_exploratory)
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise SystemExit("--max-tasks must be positive")
        tasks = tasks[:args.max_tasks]

    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "config_sha256": suite.config_sha256(manifest),
        "task_count": len(tasks),
        "system_count": len({task["system"] for task in tasks}),
        "implemented_stage": "matched_acase",
        "paper_analysis_complete": False,
        "tasks": [{"task_id": task["task_id"],
                   "m_budgets": task["m_budgets"]} for task in tasks],
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
            record = execute_task(task, manifest)
            suite._atomic_json(output, record)
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
    suite._atomic_json(
        args.results_dir / "resolution_cross_checks.json",
        cross_check_resolution_records(tasks, args.results_dir))


if __name__ == "__main__":
    main()
