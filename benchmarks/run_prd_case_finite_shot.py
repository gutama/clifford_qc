"""Checkpointed finite-shot estimator stress test for frozen packet Davidson.

The v3 manifest fixes the systems, M, K, packet supports and coefficients,
QWC grouping, total physical-shot budgets, allocation rule, seed derivation,
and overlap regularizers before this producer is executed.  Dry-run is the
default.  ``--execute`` writes one seed-level record per measurement cell and
one summary over the completed seed distributions.
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


DEFAULT_CONFIG = ROOT / "configs" / "prd_case_paper_suite_v3.json"
DEFAULT_RESULTS = Path("/tmp/clifford_qc_prd_case_finite_shot")
SCHEMA = "clifford_qc.prd_case_finite_shot_seed.v1"
BASIS_SCHEMA = "clifford_qc.prd_case_finite_shot_basis.v1"
SUMMARY_SCHEMA = "clifford_qc.prd_case_finite_shot_summary.v1"
CHECKPOINT_SCHEMA = "clifford_qc.prd_case_finite_shot_checkpoint.v1"


def _canonical_bytes(document) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _digest(document) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _atomic_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _csv(value, convert=str):
    return tuple(convert(item.strip()) for item in value.split(",") if item.strip())


def load_frozen_bases(config_path: Path, manifest: dict) -> dict:
    finite = manifest["finite_shot_tier"]
    path = (config_path.parent / finite["basis_file"]).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "clifford_qc.prd_case_frozen_packet_bases.v1":
        raise ValueError("frozen packet-basis schema is incompatible")
    if document.get("M") != finite["M"] or \
            document.get("packet_K") != finite["packet_K"]:
        raise ValueError("frozen packet-basis M/K differs from the v3 manifest")
    if document.get("generated_before_finite_shot_sampling") is not True:
        raise ValueError("packet bases do not declare pre-sampling generation")
    if set(document.get("systems") or {}) != set(finite["systems"]):
        raise ValueError("frozen packet-basis systems differ from the manifest")
    for system, payload in document["systems"].items():
        claimed = payload.get("basis_sha256")
        canonical = {key: value for key, value in payload.items()
                     if key != "basis_sha256"}
        if claimed != _digest(canonical):
            raise ValueError(f"{system}: frozen packet-basis digest is invalid")
        expected = finite["frozen_bases"][system]
        if claimed != expected["expected_basis_sha256"]:
            raise ValueError(f"{system}: packet-basis digest differs from v3")
        if payload["selected_mu"] != expected["selected_mu"]:
            raise ValueError(f"{system}: frozen shift differs from v3")
        if abs(float(payload["source_exact_packet_energy"])
               - float(expected["expected_exact_packet_energy"])) > 1e-12:
            raise ValueError(f"{system}: frozen packet energy differs from v3")
        if payload.get("hamiltonian_sha256") != _digest(
                payload.get("hamiltonian_terms")):
            raise ValueError(f"{system}: frozen Hamiltonian digest is invalid")
        packets = payload.get("packets") or []
        if len(packets) != finite["M"] - 1:
            raise ValueError(f"{system}: expected M-1 frozen packets")
        for step, packet in enumerate(packets):
            masks = packet.get("masks") or []
            coefficients = packet.get("real_coefficients") or []
            if len(masks) != finite["packet_K"] or len(coefficients) != len(masks):
                raise ValueError(f"{system}: malformed frozen packet {step}")
    return document


def build_tasks(manifest: dict, *, system_ids=(), shot_budgets=(), seeds=()):
    finite = manifest["finite_shot_tier"]
    selected_systems = set(system_ids)
    selected_budgets = set(shot_budgets)
    selected_seeds = set(seeds)
    tasks = []
    for system_index, system in enumerate(finite["systems"]):
        if selected_systems and system not in selected_systems:
            continue
        for budget in finite["shot_budgets"]:
            if selected_budgets and budget not in selected_budgets:
                continue
            for seed in finite["seeds"]:
                if selected_seeds and seed not in selected_seeds:
                    continue
                tasks.append({
                    "task_id": f"{system}__shots{budget}__seed{seed:02d}",
                    "system": system,
                    "system_index": int(system_index),
                    "shot_budget": int(budget),
                    "seed": int(seed),
                })
    return tasks


def _complex_matrix(matrix: np.ndarray):
    return [[[float(value.real), float(value.imag)] for value in row]
            for row in np.asarray(matrix, dtype=complex)]


def _plan_shots(groups, plan) -> int:
    total = 0
    for group in groups:
        counts = {int(plan.get(word.code, 0)) for word in group}
        if len(counts) != 1:
            raise AssertionError("a physical QWC group has inconsistent shots")
        total += next(iter(counts))
    return total


def _static_plan(session, finite: dict, total_budget: int):
    from clifford_qc.measurement import variance_optimal_group_plan

    allocation = finite["allocation"]
    overlap_budget = int(round(total_budget * allocation["overlap_fraction"]))
    hamiltonian_budget = total_budget - overlap_budget
    cache = session.new_cache()
    overlap = variance_optimal_group_plan(
        session.groups, cache,
        session.matrix_functionals(overlap=True, hamiltonian=False),
        overlap_budget,
        aggregation=allocation["within_stratum_aggregation"].split("_")[0],
        min_shots=allocation["minimum_shots_per_touched_group_per_stratum"])
    hamiltonian = variance_optimal_group_plan(
        session.groups, cache,
        session.matrix_functionals(overlap=False, hamiltonian=True),
        hamiltonian_budget,
        aggregation=allocation["within_stratum_aggregation"].split("_")[0],
        min_shots=allocation["minimum_shots_per_touched_group_per_stratum"])
    plan = {word.code: int(overlap.get(word.code, 0)
                           + hamiltonian.get(word.code, 0))
            for group in session.groups for word in group}
    physical_shots = _plan_shots(session.groups, plan)
    if physical_shots != total_budget:
        raise AssertionError(
            f"allocation spends {physical_shots}, expected {total_budget}")
    ledger = []
    for group in session.groups:
        ledger.append({
            "basis": [[int(qubit), str(letter)] for qubit, letter in
                      sorted(__import__(
                          "clifford_qc.measurement.grouping",
                          fromlist=["shared_basis"]).shared_basis(group).items())],
            "shots": int(plan[group[0].code]),
            "word_count": len(group),
            "word_codes_sha256": _digest(sorted(int(word.code) for word in group)),
        })
    return plan, {
        "total_budget": int(total_budget),
        "overlap_budget": int(overlap_budget),
        "hamiltonian_budget": int(hamiltonian_budget),
        "qwc_groups": len(session.groups),
        "min_shots_per_group": min(row["shots"] for row in ledger),
        "max_shots_per_group": max(row["shots"] for row in ledger),
        "plan_sha256": _digest(ledger),
        "groups": ledger,
    }


def _prepare_system(system: str, manifest: dict, frozen: dict):
    from benchmarks import run_preconditioned_expansion as pe
    from clifford_qc.backends import (
        ExactMVBackend, FiniteShotBackend, SectorStatevectorBackend,
    )
    from clifford_qc.ir import PauliSum
    from clifford_qc.measurement.session import SharedMeasurement
    from clifford_qc.subspace import MatrixElementBank, identity_generator

    finite = manifest["finite_shot_tier"]
    spec = next(entry for entry in manifest["systems"] if entry["id"] == system)
    declaration = frozen["systems"][system]
    n_qubits = int(declaration["n_qubits"])
    hamiltonian_terms = declaration["hamiltonian_terms"]
    if declaration["hamiltonian_sha256"] != _digest(hamiltonian_terms):
        raise ValueError(f"{system}: frozen Hamiltonian digest is invalid")
    hamiltonian = PauliSum(n_qubits, {
        int(code): complex(float(real), float(imaginary))
        for code, real, imaginary in hamiltonian_terms})
    sector = SectorStatevectorBackend(
        n_qubits, int(declaration["n_electrons"]), float(declaration["sz"]),
        spin_ordering=declaration["spin_ordering"])
    expected = spec.get("expected") or {}
    if n_qubits != int(expected["n_qubits"]) or \
            int(sector.dimension) != int(expected["sector_dimension"]):
        raise AssertionError(f"{system}: frozen instance dimensions changed")

    reference_occupied = declaration["reference_occupied"]
    sector_masks = set(int(mask) for mask in sector.basis.tolist())
    compiled = []
    outside = []
    for step, packet in enumerate(declaration["packets"]):
        masks = [int(mask) for mask in packet["masks"]]
        outside.extend(mask for mask in masks if mask not in sector_masks)
        pairs = pe._packet_generators(
            reference_occupied, masks, n_qubits, f"finite_pkt{step}",
            packet["real_coefficients"])
        if not pairs:
            raise AssertionError(f"{system}: frozen packet {step} is empty")
        compiled.append(pe._compile_packet(pairs, f"packet[step={step}]"))
    if outside:
        raise AssertionError(f"{system}: frozen packets leave the sector")

    rho = ExactMVBackend().state(
        pe.determinant_program(n_qubits, reference_occupied), ())
    bank = MatrixElementBank(
        rho, hamiltonian, [identity_generator(n_qubits)] + compiled)
    session = SharedMeasurement(bank)
    S_exact, H_exact = session.exact_matrices()
    S_bank, H_bank = bank.matrices()
    assembly_defect = max(float(np.max(np.abs(S_exact - S_bank))),
                          float(np.max(np.abs(H_exact - H_bank))))
    if assembly_defect > 1e-10:
        raise AssertionError(f"{system}: exact measured-bank assembly changed")
    solved = bank.solve()
    packet_energy = float(solved.ground_energy)
    source_packet_energy = float(declaration["source_exact_packet_energy"])
    if abs(packet_energy - source_packet_energy) > 1e-7:
        raise AssertionError(f"{system}: frozen packet energy changed")
    exact_energy = float(sector.ground_state(hamiltonian, k=1)[0][0])

    plans, ledgers = {}, {}
    for budget in finite["shot_budgets"]:
        plans[int(budget)], ledgers[str(int(budget))] = _static_plan(
            session, finite, int(budget))
    sampler = FiniteShotBackend(seed=0)
    started = time.perf_counter()
    prepared_groups = sampler.prepare_grouped_state(rho, session.groups)
    distribution_setup_seconds = time.perf_counter() - started
    if prepared_groups != len(session.groups):
        raise AssertionError("not every QWC distribution was prepared")

    basis_record = {
        "schema": BASIS_SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "system": system,
        "construction": declaration["construction"],
        "frozen_hamiltonian_sha256": declaration["hamiltonian_sha256"],
        "n_qubits": n_qubits,
        "sector_dimension": int(sector.dimension),
        "spin_ordering": str(sector.spin_ordering),
        "reference_identity": declaration["reference_identity"],
        "reference_occupied": list(reference_occupied),
        "M": int(finite["M"]),
        "packet_K": int(finite["packet_K"]),
        "selected_mu": float(declaration["selected_mu"]),
        "basis_sha256": declaration["basis_sha256"],
        "exact_energy": exact_energy,
        "exact_packet_energy": packet_energy,
        "exact_packet_absolute_error_mHa": abs(packet_energy - exact_energy) * 1000.0,
        "exact_effective_rank": int(solved.effective_rank),
        "exact_kappa_S": float(solved.condition_number),
        "exact_overlap_eigenvalues": [float(value) for value in
                                      solved.overlap_eigenvalues],
        "exact_assembly_max_defect": assembly_defect,
        "measured_words_excluding_identity": len(session.words),
        "qwc_groups": len(session.groups),
        "group_distribution_setup_seconds": distribution_setup_seconds,
        "allocation_ledgers": ledgers,
        "sector_certificate": {
            "packet_determinants": sum(len(packet["masks"])
                                       for packet in declaration["packets"]),
            "outside_sector": [],
            "in_sector": True,
        },
        "claim_boundary": finite["claim_boundary"],
    }
    return {
        "sector": sector, "session": session,
        "sampler": sampler, "plans": plans, "basis_record": basis_record,
        "exact_energy": exact_energy, "exact_packet_energy": packet_energy,
        "basis_sha256": declaration["basis_sha256"],
    }


def _solve(session, cache, exact_energy, exact_packet_energy, finite, policy):
    solver = finite["solver"]
    if policy == "modewise_per_mode":
        strategy, threshold_policy = "modewise", "per_mode"
    elif policy == "entrywise_uniform":
        strategy, threshold_policy = "entrywise", "uniform"
    else:  # pragma: no cover - producer-controlled
        raise ValueError(f"unknown overlap policy {policy}")
    try:
        result = session.solve(
            cache, tau_s=float(solver["tau_s"]),
            rel_tau=float(solver["relative_tau_s"]),
            max_condition=float(solver["max_condition"]),
            norm_floor=float(solver["norm_floor"]), calibrate_overlap=True,
            overlap_delta=float(solver["overlap_delta"]),
            overlap_bound=solver["overlap_bound"],
            overlap_method=solver["overlap_multiple_testing"],
            overlap_strategy=strategy, overlap_policy=threshold_policy,
            overlap_safety=float(solver["overlap_safety"]))
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "policy": policy, "status": "failed_solve",
            "error_type": type(error).__name__, "error": str(error),
            "energy": None, "energy_error_mHa": None,
            "absolute_error_mHa": None, "sampling_error_mHa": None,
            "retained_rank": None, "kappa_S": None,
            "variational_violation": None, "regularized": None,
        }
    energy = float(result.ground_energy)
    resources = result.resources
    tolerance = float(
        finite["summary"]["variational_violation_tolerance_hartree"])
    return {
        "policy": policy, "status": "ok", "energy": energy,
        "energy_error_mHa": (energy - exact_energy) * 1000.0,
        "absolute_error_mHa": abs(energy - exact_energy) * 1000.0,
        "sampling_error_mHa": (energy - exact_packet_energy) * 1000.0,
        "retained_rank": int(result.effective_rank),
        "kappa_S": float(result.condition_number),
        "overlap_negative_modes": int(resources["overlap_negative_modes"]),
        "overlap_noise_floor": float(resources["overlap_noise_floor"]),
        "overlap_threshold": float(resources["overlap_threshold"]),
        "overlap_threshold_per_mode": list(
            resources["overlap_threshold_per_mode"]),
        "overlap_decision_margin_per_mode": list(
            resources["overlap_decision_margin_per_mode"]),
        "overlap_eigenvalues": [float(value) for value in
                                result.overlap_eigenvalues],
        "overlap_calibration": resources["overlap_calibration"],
        "variational_violation": bool(energy < exact_energy - tolerance),
        "regularized": bool(result.effective_rank < finite["M"]),
        "chemical_accuracy": bool(
            abs(energy - exact_energy) * 1000.0
            <= finite["summary"]["chemical_accuracy_mHa"]),
    }


def execute_task(task: dict, context: dict, manifest: dict) -> dict:
    from clifford_qc.reproducibility import stamp_record

    finite = manifest["finite_shot_tier"]
    session = context["session"]
    plan = context["plans"][task["shot_budget"]]
    seed_sequence = np.random.SeedSequence([
        int(finite["summary"]["bootstrap_seed"]), task["system_index"],
        task["shot_budget"], task["seed"],
    ])
    context["sampler"].reseed(seed_sequence)
    cache = session.new_cache()
    started = time.perf_counter()
    session.measure_plan(context["sampler"], plan, cache)
    measurement_seconds = time.perf_counter() - started
    if cache.total_shots != task["shot_budget"]:
        raise AssertionError("measured physical shots differ from the task budget")
    if cache.total_circuits != len(session.groups):
        raise AssertionError("not every frozen QWC group was measured")
    S_hat, H_hat = session.matrices(cache)
    hermitian_defect = max(float(np.max(np.abs(S_hat - S_hat.conj().T))),
                           float(np.max(np.abs(H_hat - H_hat.conj().T))))
    if hermitian_defect > 1e-12:
        raise AssertionError("measured matrix assembly lost Hermiticity")

    primary = finite["solver"]["primary_policy"]
    sensitivity = finite["solver"]["sensitivity_policy"]
    record = {
        "schema": SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "execution_stage": "finite_shot",
        "task": task,
        "method": "packet_davidson_frozen_basis",
        "evidence_category": "finite_shot_simulation",
        "M": int(finite["M"]), "packet_K": int(finite["packet_K"]),
        "basis_sha256": context["basis_sha256"],
        "shot_budget_scope": finite["shot_budget_scope"],
        "physical_shots": int(cache.total_shots),
        "qwc_circuits": int(cache.total_circuits),
        "measured_words_excluding_identity": len(session.words),
        "plan_sha256": context["basis_record"]["allocation_ledgers"][
            str(task["shot_budget"])]["plan_sha256"],
        "measurement_seconds_excluding_group_distribution_setup": measurement_seconds,
        "exact_energy": context["exact_energy"],
        "exact_packet_energy": context["exact_packet_energy"],
        "S_hat": _complex_matrix(S_hat), "H_hat": _complex_matrix(H_hat),
        "hermitian_assembly_defect": hermitian_defect,
        "solves": [
            _solve(session, cache, context["exact_energy"],
                   context["exact_packet_energy"], finite, primary),
            _solve(session, cache, context["exact_energy"],
                   context["exact_packet_energy"], finite, sensitivity),
        ],
        "primary_policy": primary,
        "claim_boundary": finite["claim_boundary"],
    }
    return stamp_record(record)


def _new_checkpoint(manifest: dict, tasks) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "created_utc_unix": time.time(),
        "tasks": {task["task_id"]: {"status": "pending"} for task in tasks},
    }


def load_checkpoint(path: Path, manifest: dict, tasks) -> dict:
    if not path.exists():
        return _new_checkpoint(manifest, tasks)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != CHECKPOINT_SCHEMA or \
            document.get("config_sha256") != suite.config_sha256(manifest):
        raise ValueError("finite-shot checkpoint belongs to another protocol")
    for task in tasks:
        document["tasks"].setdefault(task["task_id"], {"status": "pending"})
    return document


def _bootstrap_median_ci(values, *, resamples, seed, confidence):
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None
    rng = np.random.default_rng(seed)
    medians = np.median(
        data[rng.integers(0, data.size, size=(int(resamples), data.size))], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return [float(np.quantile(medians, alpha)),
            float(np.quantile(medians, 1.0 - alpha))]


def summarize(manifest: dict, results_dir: Path, tasks) -> dict:
    finite = manifest["finite_shot_tier"]
    records = []
    for task in tasks:
        path = results_dir / f"{task['task_id']}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    cells = []
    for system in finite["systems"]:
        for budget in finite["shot_budgets"]:
            selected = [record for record in records
                        if record["task"]["system"] == system
                        and record["task"]["shot_budget"] == budget]
            for policy in (finite["solver"]["primary_policy"],
                           finite["solver"]["sensitivity_policy"]):
                rows = [next(row for row in record["solves"]
                             if row["policy"] == policy) for record in selected]
                good = [row for row in rows if row["status"] == "ok"]
                absolute = [row["absolute_error_mHa"] for row in good]
                signed = [row["energy_error_mHa"] for row in good]
                bootstrap_seed = int.from_bytes(hashlib.sha256(
                    f"{system}:{budget}:{policy}".encode()).digest()[:8], "big") \
                    ^ int(finite["summary"]["bootstrap_seed"])
                interval = (None if not signed else [
                    float(np.quantile(signed, 0.025)),
                    float(np.quantile(signed, 0.975))])
                cells.append({
                    "system": system, "shot_budget": int(budget),
                    "policy": policy, "is_primary": bool(
                        policy == finite["solver"]["primary_policy"]),
                    "attempted_seeds": len(rows), "successful_solves": len(good),
                    "median_energy_error_mHa": (
                        None if not signed else float(np.median(signed))),
                    "median_absolute_error_mHa": (
                        None if not absolute else float(np.median(absolute))),
                    "seed_level_error_interval_95_mHa": interval,
                    "bootstrap_median_absolute_error_ci_95_mHa":
                        _bootstrap_median_ci(
                            absolute,
                            resamples=finite["summary"]["bootstrap_resamples"],
                            seed=bootstrap_seed,
                            confidence=finite["summary"]["confidence_level"]),
                    "variational_violation_rate": (
                        None if not good else float(np.mean(
                            [row["variational_violation"] for row in good]))),
                    "regularization_frequency": (
                        None if not good else float(np.mean(
                            [row["regularized"] for row in good]))),
                    "failed_solve_rate": (
                        None if not rows else 1.0 - len(good) / len(rows)),
                    "chemical_accuracy_rate": (
                        None if not good else float(np.mean(
                            [row["chemical_accuracy"] for row in good]))),
                    "median_retained_rank": (
                        None if not good else float(np.median(
                            [row["retained_rank"] for row in good]))),
                })
    return {
        "schema": SUMMARY_SCHEMA,
        "config_sha256": suite.config_sha256(manifest),
        "measurement_tasks_expected": len(tasks),
        "measurement_tasks_completed": len(records),
        "complete": len(records) == len(tasks),
        "cells": cells,
        "summary_protocol": finite["summary"],
        "claim_boundary": finite["claim_boundary"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--systems", default="")
    parser.add_argument("--shot-budgets", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    manifest = suite.load_manifest(args.config)
    frozen = load_frozen_bases(args.config, manifest)
    tasks = build_tasks(
        manifest, system_ids=_csv(args.systems),
        shot_budgets=_csv(args.shot_budgets, int), seeds=_csv(args.seeds, int))
    print(json.dumps({
        "mode": ("summary" if args.summary_only else
                 "prepare" if args.prepare_only else
                 "execute" if args.execute else "dry-run"),
        "config_sha256": suite.config_sha256(manifest),
        "task_count": len(tasks),
        "systems": sorted({task["system"] for task in tasks}),
        "shot_budget_scope": manifest["finite_shot_tier"]["shot_budget_scope"],
        "method": "packet_davidson", "M": 7, "packet_K": 16,
    }, indent=2))
    if not (args.execute or args.prepare_only or args.summary_only):
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        _atomic_json(args.results_dir / "summary.json",
                     summarize(manifest, args.results_dir, tasks))
        return

    checkpoint_path = args.results_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, manifest, tasks)
    contexts = {}
    failures = 0
    for task in tasks:
        system = task["system"]
        if system not in contexts:
            contexts[system] = _prepare_system(system, manifest, frozen)
            _atomic_json(args.results_dir / f"{system}__basis.json",
                         contexts[system]["basis_record"])
        if args.prepare_only:
            continue
        state = checkpoint["tasks"][task["task_id"]]
        output = args.results_dir / f"{task['task_id']}.json"
        if state.get("status") == "completed" and output.exists():
            continue
        started = time.time()
        state.update({"status": "running", "started_utc_unix": started})
        _atomic_json(checkpoint_path, checkpoint)
        try:
            record = execute_task(task, contexts[system], manifest)
            _atomic_json(output, record)
            state.update({"status": "completed", "finished_utc_unix": time.time(),
                          "output": output.name})
        except Exception as error:  # pragma: no cover - runtime checkpoint path
            failures += 1
            state.update({"status": "failed", "finished_utc_unix": time.time(),
                          "error_type": type(error).__name__, "error": str(error)})
            if args.fail_fast:
                _atomic_json(checkpoint_path, checkpoint)
                raise
        _atomic_json(checkpoint_path, checkpoint)
        print(json.dumps({"task": task["task_id"], **state}), flush=True)
    if not args.prepare_only:
        _atomic_json(args.results_dir / "summary.json",
                     summarize(manifest, args.results_dir, tasks))
    if failures:
        raise SystemExit(f"{failures} finite-shot task(s) failed")


if __name__ == "__main__":
    main()
