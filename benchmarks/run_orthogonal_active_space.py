"""Bounded orthogonal-correction pilot; default validates the frozen design only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.evidence import declare_evidence
from clifford_qc.models.fcidump import fcidump_model, read_fcidump
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.fermionic_generators import occupied_spin_orbitals
from clifford_qc.subspace.orthogonal import (
    canonical_phase, occupation_packets, solve_orthogonal_corrections,
)

CONFIG = ROOT / "benchmarks/configs/orthogonal_active_space.json"
DECLARATION = ROOT / "benchmarks/ORTHOGONAL_ACTIVE_SPACE_DECLARATION.md"
RECORD = ROOT / "benchmarks/reference_results/orthogonal_active_space.json"
SCHEMA = "clifford_qc.orthogonal_active_space.v1"
SOURCES = (
    "benchmarks/run_orthogonal_active_space.py",
    "benchmarks/check_orthogonal_active_space.py",
    "benchmarks/phase17_ceo_oracle.py",
    "clifford_qc/subspace/orthogonal.py",
    "clifford_qc/backends/sector_statevector.py",
    "clifford_qc/models/fcidump.py",
    "clifford_qc/fermion.py",
)
ARMS = ("orthogonal", "same_span_mixed", "dimension_matched_determinants",
        "support_matched_determinants")
NOISE_ARMS = ("orthogonal_known_overlap", "mixed_known_overlap", "mixed_noisy_overlap")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(array):
    values = np.asarray(array, dtype=complex)
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_design(path=CONFIG):
    config = json.loads(Path(path).read_text())
    if config["schema"] != "clifford_qc.orthogonal_active_space_design.v1":
        raise ValueError("unknown design schema")
    if config["correction_budgets"] != [0, 1, 2, 4]:
        raise ValueError("the frozen prefix grid changed")
    ids = [case["id"] for case in config["cases"]]
    if ids != ["h4", "beh2", "lih", "h2o"]:
        raise ValueError("the complete ordered four-system scope is required")
    for case in config["cases"]:
        path = ROOT / case["path"]
        if not path.is_relative_to(ROOT) or sha(path) != case["sha256"]:
            raise ValueError(f"input digest differs: {case['id']}")
        data = read_fcidump(path)
        active = case["active_spatial_orbitals"]
        if (len(active) != 2 or len(set(active)) != 2 or min(active) < 0
                or max(active) >= data.n_orbitals):
            raise ValueError("two distinct spatial orbitals are required")
        dimension = (math.comb(data.n_orbitals, (data.n_electrons + data.ms2) // 2)
                     * math.comb(data.n_orbitals, (data.n_electrons - data.ms2) // 2))
        if dimension > config["maximum_sector_dimension"]:
            raise ValueError("declared sector exceeds the experiment ceiling")
    return config


class CountedAction:
    def __init__(self, operator):
        self.operator, self.calls = operator, 0

    def __call__(self, vector):
        self.calls += 1
        return self.operator.matvec(vector)


def generalized(h, s, floor=1e-12):
    values, vectors = np.linalg.eigh(s)
    if values[0] <= floor:
        raise ValueError("overlap is not positive definite at the declared floor")
    whitening = vectors / np.sqrt(values)
    projected = whitening.conj().T @ h @ whitening
    energies, coefficients = np.linalg.eigh((projected + projected.conj().T) / 2)
    return energies, whitening @ coefficients


def readout(basis, apply, tolerance, *, orthogonal=False):
    before = apply.calls
    if orthogonal:
        result = solve_orthogonal_corrections(apply, basis, tolerance=tolerance)
        h, s = result.projected_hamiltonian, np.eye(basis.shape[1])
        energies, coefficients, actions = result.energies, result.coefficients, result.actions
    else:
        actions = np.column_stack([apply(column) for column in basis.T])
        h, s = basis.conj().T @ actions, basis.conj().T @ basis
        energies, coefficients = generalized(h, s)
    c = coefficients[:, 0]
    norm = math.sqrt(float(np.vdot(basis @ c, basis @ c).real))
    residual = (actions @ c - energies[0] * (basis @ c)) / norm
    record = {
        "energy": float(energies[0]), "energies": energies.tolist(),
        "residual_norm": float(np.linalg.norm(residual)),
        "basis_dimension": int(basis.shape[1]),
        "determinant_support": int(np.count_nonzero(np.any(np.abs(basis) > 1e-12, axis=1))),
        "overlap_condition": float(np.linalg.cond(s)),
        "overlap_identity_defect": float(np.max(np.abs(s - np.eye(s.shape[0])))),
        "projected_hamiltonian": encode(h), "projected_overlap": encode(s),
        "ground_coefficients": encode(c), "h_action_calls": apply.calls - before,
        "vector_payload_bytes": int(basis.nbytes + actions.nbytes),
    }
    return record, h, s


def lowering(energy, diagonal, coupling):
    gap = float(diagonal - energy)
    return max(0.0, float((np.hypot(gap, 2 * abs(coupling)) - gap) / 2))


def rank_scores(entries):
    """Descending lowering, with 1e-12 absolute numerical ties by integer key."""
    pending, order = dict(entries), []
    while pending:
        maximum = max(pending.values())
        key = min(key for key, score in pending.items() if maximum - score <= 1e-12)
        order.append(key)
        del pending[key]
    return order


def symmetric_noise(rng, dimension, scale):
    upper = np.triu(rng.normal(0.0, scale, (dimension, dimension)))
    return upper + upper.T - np.diag(np.diag(upper))


def summarize_noise(rows, target):
    result = {}
    for arm in NOISE_ARMS:
        errors = [row[arm]["error"] for row in rows if row[arm]["error"] is not None]
        result[arm] = {
            "replicas": len(rows), "successes": len(errors), "failures": len(rows) - len(errors),
            "conditional_rmse": (float(np.sqrt(np.mean(np.square(errors)))) if errors else None),
            "unconditional_target_fraction": sum(abs(error) <= target for error in errors) / len(rows),
            "variational_violations": sum(row[arm]["variational_violation"] is True for row in rows),
        }
    return result


def noise_probe(h, classical_energy, classical_coefficients, classical_dimension,
                exact_energy, config, case_index):
    noise = config["noise"]
    size = h.shape[0]
    transform = np.eye(size, dtype=complex)
    transform[:classical_dimension, classical_dimension:] = (
        config["same_span_mixing"] * classical_coefficients[:, None])
    overlap = transform.conj().T @ transform
    centered = h - classical_energy * np.eye(size)
    exact_retained = float(np.linalg.eigvalsh(h)[0])
    scales = []
    for scale_index, scale in enumerate(noise["scales"]):
        rows = []
        for replica in range(noise["replicas"]):
            h_rng = np.random.default_rng(np.random.SeedSequence(
                [noise["seed"], case_index, scale_index, replica, 0]))
            s_rng = np.random.default_rng(np.random.SeedSequence(
                [noise["seed"], case_index, scale_index, replica, 1]))
            delta = symmetric_noise(h_rng, size, scale["hamiltonian_hartree"])
            delta[:classical_dimension, :classical_dimension] = 0
            sampled = centered + delta
            mixed_h = transform.conj().T @ sampled @ transform
            delta_s = symmetric_noise(s_rng, size, scale["overlap_dimensionless"])
            delta_s[:classical_dimension, :classical_dimension] = 0
            noisy_overlap = overlap + delta_s
            values = {
                "orthogonal_known_overlap": float(np.linalg.eigvalsh(sampled)[0]),
                "mixed_known_overlap": float(generalized(mixed_h, overlap)[0][0]),
            }
            try:
                values["mixed_noisy_overlap"] = float(generalized(
                    mixed_h, noisy_overlap, noise["minimum_overlap_eigenvalue"])[0][0])
            except ValueError:
                values["mixed_noisy_overlap"] = None
            row = {"replica": replica,
                   "minimum_noisy_overlap_eigenvalue": float(np.linalg.eigvalsh(noisy_overlap)[0])}
            for arm, value in values.items():
                energy = None if value is None else classical_energy + value
                row[arm] = {"error": None if energy is None else energy - exact_retained,
                            "variational_violation": None if energy is None else energy < exact_energy - 1e-9}
            rows.append(row)
        scales.append({"scale_index": scale_index, "replicas": rows,
                       "summary": summarize_noise(rows, config["accuracy_target_hartree"])})
    return {"basis_dimension": size, "scales": scales}


def run_case(case, config, case_index):
    started = time.perf_counter()
    model = fcidump_model(ROOT / case["path"])
    backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], model.metadata["sz"])
    if backend.dimension > config["maximum_sector_dimension"]:
        raise ValueError("sector exceeds frozen ceiling")
    operator = backend.operator(model.hamiltonian)
    apply = CountedAction(operator)
    occupied = occupied_spin_orbitals(model)
    hf = sum(1 << (model.n - 1 - mode) for mode in occupied)
    active_mask = sum(1 << (model.n - 1 - (2 * p + spin))
                      for p in case["active_spatial_orbitals"] for spin in (0, 1))
    external_mask = ((1 << model.n) - 1) ^ active_mask
    classical_indices = np.flatnonzero((backend.basis & external_mask) == (hf & external_mask))
    if classical_indices.size != 4:
        raise ValueError("the frozen classical space must be CAS(2e,2o), dimension four")
    classical = np.eye(backend.dimension, dtype=complex)[:, classical_indices]
    reference = solve_orthogonal_corrections(apply, classical)
    coefficients = canonical_phase(reference.coefficients[:, 0])
    state = classical @ coefficients
    e_classical = reference.energy
    residual = apply(state)
    residual[classical_indices] = 0
    diagonal = np.empty(backend.dimension)
    unit = np.zeros(backend.dimension, complex)
    for index in range(backend.dimension):
        unit[index] = 1
        diagonal[index] = float(apply(unit)[index].real)
        unit[index] = 0
    shift = config["preconditioner_shift_hartree"]
    denominator = diagonal - e_classical + shift
    denominator = np.where(np.abs(denominator) < shift, shift, denominator)
    packets = occupation_packets(backend.basis, residual / denominator, classical_indices,
                                 external_mask, tolerance=config["packet_zero_tolerance"])
    scored = []
    for packet in packets:
        acted = apply(packet.vector)
        score = lowering(e_classical, np.vdot(packet.vector, acted).real,
                         np.vdot(state, acted))
        scored.append((score, packet))
    by_pattern = {packet.pattern: (score, packet) for score, packet in scored}
    scored = [by_pattern[pattern] for pattern in rank_scores(
        (pattern, entry[0]) for pattern, entry in by_pattern.items())]
    external = sorted(set(range(backend.dimension)) - set(classical_indices.tolist()))
    determinant_order = rank_scores((d, lowering(e_classical, diagonal[d], residual[d]))
                                    for d in external)
    selection_calls = apply.calls
    trajectory = []
    for budget in config["correction_budgets"]:
        selected = [packet for _, packet in scored[:budget]]
        corrections = (np.column_stack([packet.vector for packet in selected]) if selected
                       else np.empty((backend.dimension, 0), complex))
        basis = np.column_stack((classical, corrections))
        orthogonal, h, s = readout(basis, apply, config["orthogonality_tolerance"], orthogonal=True)
        transform = np.eye(basis.shape[1], dtype=complex)
        transform[:4, 4:] = config["same_span_mixing"] * coefficients[:, None]
        mixed, _, _ = readout(basis @ transform, apply, config["orthogonality_tolerance"])
        simple = classical_indices.tolist() + determinant_order[:len(selected)]
        dimension_control, _, _ = readout(np.eye(backend.dimension, dtype=complex)[:, simple],
                                        apply, config["orthogonality_tolerance"], orthogonal=True)
        support = np.flatnonzero(np.any(np.abs(basis) > config["packet_zero_tolerance"], axis=1))
        support_control, _, _ = readout(np.eye(backend.dimension, dtype=complex)[:, support],
                                      apply, config["orthogonality_tolerance"], orthogonal=True)
        trajectory.append({"requested_corrections": budget, "retained_corrections": len(selected),
                           "patterns": [packet.pattern for packet in selected],
                           "arms": dict(zip(ARMS, (orthogonal, mixed, dimension_control, support_control)))})
    # Spectrum is an accuracy readout only, after every candidate/prefix is frozen.
    reference_started = time.perf_counter()
    full_h = operator.restrict(np.arange(backend.dimension))
    exact = np.linalg.eigvalsh(full_h)
    ordering = classical_indices.tolist() + external
    closure = np.linalg.eigvalsh(full_h[np.ix_(ordering, ordering)])
    reference_seconds = time.perf_counter() - reference_started
    for prefix in trajectory:
        for arm in prefix["arms"].values():
            arm["error"] = arm["energy"] - float(exact[0])
            arm["target_reached"] = abs(arm["error"]) <= config["accuracy_target_hartree"]
    noise = noise_probe(h, e_classical, coefficients, 4, float(exact[0]), config, case_index)
    return {
        "id": case["id"], "n_qubits": model.n, "reference_word": hf,
        "n_electrons": model.metadata["n_electrons"], "sz": model.metadata["sz"],
        "sector_words": backend.basis.tolist(), "classical_indices": classical_indices.tolist(),
        "classical_energy": e_classical, "classical_coefficients": encode(coefficients),
        "packet_ranking": [{"pattern": packet.pattern, "score": score,
                            "indices": np.flatnonzero(np.abs(packet.vector) > config["packet_zero_tolerance"]).tolist(),
                            "coefficients": encode(packet.vector[np.abs(packet.vector) > config["packet_zero_tolerance"]])}
                           for score, packet in scored],
        "determinant_ranking": determinant_order,
        "trajectory": trajectory,
        "exact_reference": {"energy": float(exact[0]), "energies": exact.tolist(),
                            "wall_seconds": reference_seconds, "matrix_bytes": full_h.nbytes},
        "complete_complement_spectrum_defect": float(np.max(np.abs(closure - exact))),
        "noise": noise,
        "resources": {"selection_h_action_calls": selection_calls,
                      "total_h_action_calls": apply.calls,
                      "diagonal_entries_constructed": backend.dimension,
                      "packet_candidates_scored": len(scored),
                      "determinant_candidates_scored": len(external),
                      "operator_storage": operator.memory_estimate(),
                      "wall_seconds_including_reference_and_noise": time.perf_counter() - started,
                      "quantum_state_preparation": "unimplemented_unpriced",
                      "quantum_measurement_shots": "unimplemented_unpriced"},
    }


def execute(config):
    if config != load_design():
        raise ValueError("execution requires the frozen design without overrides")
    if git("rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("execution requires full Git history")
    declaration_commit = git("log", "-1", "--format=%H", "--", str(DECLARATION.relative_to(ROOT)))
    source_commit = git("rev-parse", "HEAD")
    if not declaration_commit or declaration_commit == source_commit:
        raise ValueError("the declaration must precede the implementation commit")
    git("merge-base", "--is-ancestor", declaration_commit, source_commit)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("execution requires a clean committed source tree")
    for path in (CONFIG, DECLARATION):
        declared = subprocess.check_output(
            ["git", "show", f"{declaration_commit}:{path.relative_to(ROOT)}"], cwd=ROOT)
        if declared != path.read_bytes():
            raise ValueError("design bytes differ from the earlier declaration commit")
    identity = {path: sha(ROOT / path) for path in SOURCES}
    for path, digest in identity.items():
        committed = subprocess.check_output(["git", "show", f"{source_commit}:{path}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"execution source is not committed: {path}")
    cases = [run_case(case, config, i) for i, case in enumerate(config["cases"])]
    record = {"schema": SCHEMA, "config": config, "config_sha256": sha(CONFIG),
              "declaration_commit": declaration_commit, "declaration_sha256": sha(DECLARATION),
              "source_commit": source_commit, "source_sha256": identity, "cases": cases,
              "quantum_advantage_claim": False, "hardware_execution": False}
    record = declare_evidence(record, {
        "energies_spans_residuals": {"tier": "exact_small_sector_simulation",
            "scope": "Explicit sector vectors, classically constructed corrections and exact Hamiltonian actions; no scalable or hardware selection claim."},
        "noise": {"tier": "illustrative_model",
            "scope": "Centered Gaussian matrix-entry perturbations; not Pauli samples, physical shots, hardware errors or confidence certificates."},
        "resources": {"tier": "classical_process_measurement",
            "scope": "Observed wall times and explicit array/operator payload bytes; no peak RSS, optimized implementation or quantum total-cost comparison."},
    })
    return stamp_record(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=RECORD)
    args = parser.parse_args(argv)
    config = load_design()
    if not args.execute:
        print("VALID design: four fixed systems; no campaign outcomes computed")
        return 0
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = execute(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
