"""Bounded orthogonal-correction pilot; default validates the frozen design only."""
from __future__ import annotations

import argparse
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

from benchmarks.archive_provenance import HistoryError, commit_binding, history_kind, safe_path
from benchmarks.metadata_io import sha256
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.evidence import declare_evidence
from clifford_qc.models.fcidump import fcidump_model, read_fcidump
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.fermionic_generators import occupied_spin_orbitals
from clifford_qc.subspace.orthogonal import (
    canonical_phase, solve_orthogonal_corrections,
)

from clifford_qc.subspace.spin_packets import singlet_packets, spin_squared_operator

CONFIG = ROOT / "benchmarks/configs/orthogonal_active_space_v2.json"
DECLARATION = ROOT / "benchmarks/ORTHOGONAL_ACTIVE_SPACE_V2_DECLARATION.md"
RECORD = ROOT / "benchmarks/reference_results/orthogonal_active_space_v2.json"
SCHEMA = "clifford_qc.orthogonal_active_space.v2"
SOURCES = (
    "benchmarks/run_orthogonal_active_space.py",
    "benchmarks/check_orthogonal_active_space.py",
    "benchmarks/phase17_ceo_oracle.py",
    "clifford_qc/subspace/orthogonal.py",
    "clifford_qc/subspace/spin_packets.py",
    "benchmarks/archive_provenance.py",
    "benchmarks/metadata_io.py",
    "clifford_qc/backends/sector_statevector.py",
    "clifford_qc/models/fcidump.py",
    "clifford_qc/fermion.py",
)
ARMS = ("orthogonal", "same_span_mixed", "dimension_matched_determinants",
        "support_matched_determinants")
NOISE_ARMS = ("orthogonal_known_overlap", "mixed_known_overlap", "mixed_noisy_overlap")


def sha(path):
    return sha256(Path(path))


def encode(array):
    values = np.asarray(array, dtype=complex)
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=10).strip()


def load_design(path=CONFIG):
    config = json.loads(Path(path).read_text())
    if config["schema"] != "clifford_qc.orthogonal_active_space_design.v2":
        raise ValueError("unknown design schema")
    if config["correction_budgets"] != [0, 1, 2, 4]:
        raise ValueError("the frozen prefix grid changed")
    ids = [case["id"] for case in config["cases"]]
    if ids != ["h4", "beh2", "lih", "h2o"]:
        raise ValueError("the complete ordered four-system scope is required")
    for case in config["cases"]:
        path = safe_path(ROOT, case["path"])
        if sha(path) != case["sha256"]:
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


def readout(basis, apply, tolerance, *, orthogonal=False, s2=None, support_tolerance=1e-9):
    before = apply.calls
    if orthogonal:
        result = solve_orthogonal_corrections(apply, basis, tolerance=tolerance)
        h, s = result.projected_hamiltonian, basis.conj().T @ basis
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
        "determinant_support": int(np.count_nonzero(np.any(np.abs(basis) > support_tolerance, axis=1))),
        "overlap_condition": float(np.linalg.cond(s)),
        "overlap_identity_defect": float(np.max(np.abs(s - np.eye(s.shape[0])))),
        "projected_hamiltonian": encode(h), "projected_overlap": encode(s),
        "ground_coefficients": encode(c), "h_action_calls": apply.calls - before,
        "vector_payload_bytes": int(basis.nbytes + actions.nbytes),
    }
    if s2 is not None:
        state = basis @ c / norm
        record["spin_squared"] = float(np.vdot(state, s2 @ state).real)
        record["spin_action_norm"] = float(np.linalg.norm(s2 @ state))
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
    s2_operator = backend.operator(spin_squared_operator(model.n))
    s2 = s2_operator.restrict(np.arange(backend.dimension))
    spin_work = {}
    packets = singlet_packets(backend.basis, residual / denominator, classical_indices,
                              external_mask, model.n, s2,
                              tolerance=config["packet_zero_tolerance"],
                              relative_tolerance=config["packet_relative_tolerance"],
                              spin_tolerance=config["spin_tolerance"], diagnostics=spin_work)
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
        orthogonal, h, s = readout(basis, apply, config["orthogonality_tolerance"], orthogonal=True, s2=s2)
        transform = np.eye(basis.shape[1], dtype=complex)
        transform[:4, 4:] = config["same_span_mixing"] * coefficients[:, None]
        mixed, _, _ = readout(basis @ transform, apply, config["orthogonality_tolerance"], s2=s2)
        simple = classical_indices.tolist() + determinant_order[:len(selected)]
        dimension_control, _, _ = readout(np.eye(backend.dimension, dtype=complex)[:, simple],
                                        apply, config["orthogonality_tolerance"], orthogonal=True, s2=s2)
        support = np.flatnonzero(np.any(np.abs(basis) > config["packet_zero_tolerance"], axis=1))
        support_control, _, _ = readout(np.eye(backend.dimension, dtype=complex)[:, support],
                                      apply, config["orthogonality_tolerance"], orthogonal=True, s2=s2)
        if max(orthogonal["spin_action_norm"], mixed["spin_action_norm"]) > config["spin_tolerance"]:
            raise ValueError("packet Ritz state is not a singlet")
        trajectory.append({"requested_corrections": budget, "retained_corrections": len(selected),
                           "patterns": [packet.pattern for packet in selected],
                           "arms": dict(zip(ARMS, (orthogonal, mixed, dimension_control, support_control)))})
    # Spectrum is an accuracy readout only, after every candidate/prefix is frozen.
    reference_started = time.perf_counter()
    full_h = operator.restrict(np.arange(backend.dimension))
    exact = np.linalg.eigvalsh(full_h)
    partition_valid = partition_is_complete(classical_indices, external, backend.dimension)
    if not partition_valid:
        raise ValueError("classical/complement partition is invalid")
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
        "classical_complement_partition_valid": partition_valid,
        "noise": noise,
        "resources": {"selection_h_action_calls": selection_calls,
                      "total_h_action_calls": apply.calls,
                      "diagonal_entries_constructed": backend.dimension,
                      "packet_candidates_scored": len(scored),
                      "determinant_candidates_scored": len(external),
                      "operator_storage": operator.memory_estimate(),
                      "spin_matrix_bytes": s2.nbytes,
                      "spin_packet_construction": spin_work,
                      "spin_operator_storage": s2_operator.memory_estimate(),
                      "spin_matrix_columns_constructed": backend.dimension,
                      "wall_seconds_including_reference_and_noise": time.perf_counter() - started,
                      "quantum_state_preparation": "unimplemented_unpriced",
                      "quantum_measurement_shots": "unimplemented_unpriced"},
    }


def partition_is_complete(classical, external, dimension):
    left, right = list(classical), list(external)
    return (len(set(left)) == len(left) and len(set(right)) == len(right)
            and set(left).isdisjoint(right) and set(left) | set(right) == set(range(dimension)))


def validate_declaration(declaration_commit, source_commit, config, *, root=ROOT):
    """Bind design bytes and require new revision code to be absent at declaration."""
    history_kind(root)
    if declaration_commit == source_commit:
        raise ValueError("the declaration must precede the implementation commit")
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", declaration_commit, source_commit],
                       cwd=root, capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HistoryError("declaration is not an ancestor of execution") from exc
    for path in (CONFIG, DECLARATION):
        relative = str(path.relative_to(ROOT))
        if commit_binding(root, declaration_commit, relative)["sha256"] != sha(root / relative):
            raise ValueError("design bytes differ from the earlier declaration commit")
    paths = config["implementation_paths_absent_at_declaration"]
    if paths != ["clifford_qc/subspace/spin_packets.py"]:
        raise ValueError("revision implementation absence manifest differs")
    for path in paths:
        if subprocess.run(["git", "cat-file", "-e", f"{declaration_commit}:{path}"],
                          cwd=root, capture_output=True, timeout=10).returncode == 0:
            raise ValueError(f"implementation already exists at declaration: {path}")
        commit_binding(root, source_commit, path)


def execution_preflight():
    """Use the same stamping/environment and dirty-tree rules before science."""
    history_kind(ROOT)
    stamp = stamp_record({})["provenance"]
    source_commit = git("rev-parse", "HEAD")
    if (git("status", "--porcelain") or stamp["git_dirty"] is not False
            or stamp["git_sha"] != source_commit):
        raise ValueError("execution requires a clean committed source tree including untracked files")
    return source_commit, stamp


def execute(config):
    if config != load_design():
        raise ValueError("execution requires the frozen design without overrides")
    source_commit, provenance = execution_preflight()
    declaration_commit = git("log", "-1", "--format=%H", "--", str(DECLARATION.relative_to(ROOT)))
    validate_declaration(declaration_commit, source_commit, config)
    identity = {path: sha(safe_path(ROOT, path)) for path in SOURCES}
    for path, digest in identity.items():
        if commit_binding(ROOT, source_commit, path)["sha256"] != digest:
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
    # Recheck before returning: a changed tree must not produce an apparently clean receipt.
    final_source, _ = execution_preflight()
    if final_source != source_commit:
        raise ValueError("execution source changed during campaign")
    return stamp_record(record, provenance=provenance)


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
