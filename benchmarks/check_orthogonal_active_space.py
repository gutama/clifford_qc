"""Replay the orthogonal-correction pilot with independent occupation matrices.

No producer, native sector operator, Pauli product, packet builder or new solver
is imported. FCIDUMP parsing and the existing independent fermionic sign oracle
are shared. Timings and compiled native storage are observations, not rebuilt
performance guarantees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.metadata_io import read_json
from benchmarks.phase17_ceo_oracle import hamiltonian
from clifford_qc.evidence import validate_evidence
from clifford_qc.models.fcidump import read_fcidump

RECORD = ROOT / "benchmarks/reference_results/orthogonal_active_space.json"
CONFIG = ROOT / "benchmarks/configs/orthogonal_active_space.json"
DECLARATION = ROOT / "benchmarks/ORTHOGONAL_ACTIVE_SPACE_DECLARATION.md"
EXECUTION_BUNDLE = "benchmarks/provenance/orthogonal_active_space.bundle"
EXECUTION_BUNDLE_SHA256 = "4c701798891ad541d98c8e882517a43b59b5f29ca8cbac00e51783f6c2e7f434"
ARMS = ("orthogonal", "same_span_mixed", "dimension_matched_determinants",
        "support_matched_determinants")
NOISE_ARMS = ("orthogonal_known_overlap", "mixed_known_overlap", "mixed_noisy_overlap")
SOURCES = (
    "benchmarks/run_orthogonal_active_space.py", "benchmarks/check_orthogonal_active_space.py",
    "benchmarks/phase17_ceo_oracle.py", "clifford_qc/subspace/orthogonal.py",
    "clifford_qc/backends/sector_statevector.py", "clifford_qc/models/fcidump.py",
    "clifford_qc/fermion.py",
)


def decode(value):
    real, imag = np.asarray(value["real"]), np.asarray(value["imag"])
    if real.shape != imag.shape or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("invalid complex array")
    return real + 1j * imag


def close(a, b, tolerance=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    return (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=tolerance, rtol=0))


def phase(vector):
    pivot = vector[np.argmax(np.abs(vector))]
    return vector * (np.conjugate(pivot) / abs(pivot))


def ordered(scores):
    keys = list(scores)
    answer = []
    while keys:
        best = max(scores[key] for key in keys)
        chosen = sorted(key for key in keys if scores[key] >= best - 1e-12)[0]
        answer.append(chosen)
        keys.remove(chosen)
    return answer


def independent_generalized(h, s):
    lower = np.linalg.cholesky(s)
    left = np.linalg.solve(lower, h)
    reduced = np.linalg.solve(lower.conj(), left.T).T
    energies, vectors = np.linalg.eigh((reduced + reduced.conj().T) / 2)
    return energies, np.linalg.solve(lower.conj().T, vectors)


def packet_agrees(row, expected, words, classical_indices, external_mask, zero_tolerance):
    """Compare reconstructed vectors, not unstable thresholded index lists.

    Independent integral summation can move a numerical zero across the 1e-12
    serialization cutoff after normalization of a small packet. The existing
    1e-9 vector tolerance still bounds every omitted or included amplitude.
    Structural sector/classical-support membership is checked exactly.
    """
    indices = row["indices"]
    if (not indices or any(type(i) is not int or i < 0 or i >= len(words) for i in indices)
            or sorted(set(indices)) != indices
            or any(i in classical_indices or words[i] & external_mask != row["pattern"]
                   for i in indices)):
        return False
    coefficients = decode(row["coefficients"])
    if coefficients.shape != (len(indices),) or np.any(abs(coefficients) <= zero_tolerance):
        return False
    reconstructed = np.zeros(len(words), complex)
    reconstructed[indices] = coefficients
    overlap = np.vdot(expected, reconstructed)
    return abs(overlap) > 0 and close(reconstructed, expected * overlap / abs(overlap))


def verify_case(entry, case, config, h, words):
    problems = []

    def require(ok, message):
        if not ok:
            problems.append(f"{case['id']}: {message}")

    n, dimension = entry["n_qubits"], len(words)
    require(entry["id"] == case["id"] and entry["sector_words"] == words, "case/sector identity")
    require(dimension <= config["maximum_sector_dimension"], "sector ceiling")
    hf = entry["reference_word"]
    mask = sum(1 << (n - 1 - 2 * p - spin)
               for p in case["active_spatial_orbitals"] for spin in (0, 1))
    external_mask = ((1 << n) - 1) ^ mask
    classical_indices = [i for i, word in enumerate(words)
                         if word & external_mask == hf & external_mask]
    require(len(classical_indices) == 4 and entry["classical_indices"] == classical_indices,
            "complete classical active space")
    classical = np.eye(dimension, dtype=complex)[:, classical_indices]
    values, vectors = np.linalg.eigh(h[np.ix_(classical_indices, classical_indices)])
    e_classical, c = float(values[0]), phase(vectors[:, 0])
    state = classical @ c
    require(close(entry["classical_energy"], e_classical)
            and close(decode(entry["classical_coefficients"]), c), "classical eigensolution")
    residual = h @ state
    residual[classical_indices] = 0
    denominator = np.diag(h).real - e_classical + config["preconditioner_shift_hartree"]
    for i in range(dimension):
        if abs(denominator[i]) < config["preconditioner_shift_hartree"]:
            denominator[i] = config["preconditioner_shift_hartree"]
    raw = residual / denominator
    external = [i for i in range(dimension) if i not in classical_indices]
    packets, scores = {}, {}

    def score(vector):
        coupling = np.vdot(state, h @ vector)
        diagonal = float(np.vdot(vector, h @ vector).real) - e_classical
        return max(0.0, -float(np.linalg.eigvalsh(
            np.array([[0, coupling], [coupling.conjugate(), diagonal]]))[0]))

    for pattern in sorted({words[i] & external_mask for i in external}):
        vector = np.zeros(dimension, complex)
        for i in external:
            if words[i] & external_mask == pattern:
                vector[i] = raw[i]
        norm = np.linalg.norm(vector)
        if norm > config["packet_zero_tolerance"]:
            packets[pattern] = phase(vector / norm)
            scores[pattern] = score(packets[pattern])
    packet_order = ordered(scores)
    require([row["pattern"] for row in entry["packet_ranking"]] == packet_order,
            "complete packet ordering")
    for row in entry["packet_ranking"]:
        expected = packets[row["pattern"]]
        agrees = packet_agrees(row, expected, words, classical_indices, external_mask,
                                config["packet_zero_tolerance"])
        require(close(row["score"], scores[row["pattern"]]) and agrees,
                "packet support/coefficients/score")
        if agrees:
            # Re-express the independently rebuilt vector in the recorded
            # global-phase convention before checking coordinate-dependent H.
            overlap = np.vdot(expected[row["indices"]], decode(row["coefficients"]))
            packets[row["pattern"]] = expected * overlap / abs(overlap)
    determinant_order = ordered({i: score(np.eye(dimension, dtype=complex)[:, i])
                                 for i in external})
    require(entry["determinant_ranking"] == determinant_order, "complete determinant ordering")
    exact = np.linalg.eigvalsh(h)
    require(close(entry["exact_reference"]["energies"], exact)
            and close(entry["exact_reference"]["energy"], exact[0]), "independent reference spectrum")
    require(entry["exact_reference"]["matrix_bytes"] == dimension * dimension * 16,
            "reference matrix payload")
    permuted = h[np.ix_(classical_indices + external, classical_indices + external)]
    defect = float(np.max(np.abs(np.linalg.eigvalsh(permuted) - exact)))
    require(close(entry["complete_complement_spectrum_defect"], defect, 1e-10)
            and defect < 1e-9, "complete complement closure")
    require([row["requested_corrections"] for row in entry["trajectory"]]
            == config["correction_budgets"], "complete prefix grid")
    previous = e_classical
    projection_calls = 0
    final_h = None
    for prefix in entry["trajectory"]:
        patterns = packet_order[:prefix["requested_corrections"]]
        require(prefix["patterns"] == patterns and prefix["retained_corrections"] == len(patterns),
                "nested packet prefix")
        extra = [packets[pattern] for pattern in patterns]
        basis = np.column_stack([*classical.T, *extra])
        transform = np.eye(basis.shape[1], dtype=complex)
        transform[:4, 4:] = config["same_span_mixing"] * c[:, None]
        simple = classical_indices + determinant_order[:len(patterns)]
        support = np.flatnonzero(np.any(np.abs(basis) > config["packet_zero_tolerance"], axis=1))
        bases = (basis, basis @ transform, np.eye(dimension, dtype=complex)[:, simple],
                 np.eye(dimension, dtype=complex)[:, support])
        require(set(prefix["arms"]) == set(ARMS), "complete control arms")
        for name, columns in zip(ARMS, bases):
            observed = prefix["arms"][name]
            projected, overlap = columns.conj().T @ h @ columns, columns.conj().T @ columns
            es, cs = independent_generalized(projected, overlap)
            psi = columns @ cs[:, 0]
            residual_norm = float(np.linalg.norm(h @ psi - es[0] * psi))
            size = columns.shape[1]
            support_count = int(np.count_nonzero(np.any(np.abs(columns) > 1e-12, axis=1)))
            require(close(decode(observed["projected_hamiltonian"]), projected)
                    and close(decode(observed["projected_overlap"]), overlap), f"{name} pencil")
            require(close(observed["energies"], es) and close(observed["energy"], es[0])
                    and close(observed["residual_norm"], residual_norm), f"{name} eigenvalues/residual")
            stored_c = decode(observed["ground_coefficients"])
            require(stored_c.shape == (size,) and close(np.vdot(stored_c, overlap @ stored_c), 1)
                    and np.linalg.norm(projected @ stored_c - es[0] * (overlap @ stored_c)) < 1e-8,
                    f"{name} recorded eigenvector")
            require(observed["basis_dimension"] == size
                    and observed["determinant_support"] == support_count
                    and observed["h_action_calls"] == size
                    and observed["vector_payload_bytes"] == 32 * size * dimension,
                    f"{name} work/payload accounting")
            require(close(observed["overlap_condition"], np.linalg.cond(overlap), 1e-7)
                    and close(observed["overlap_identity_defect"], np.max(abs(overlap - np.eye(size)))),
                    f"{name} conditioning")
            error = float(es[0] - exact[0])
            require(close(observed["error"], error)
                    and type(observed["target_reached"]) is bool
                    and observed["target_reached"] == (abs(error) <= config["accuracy_target_hartree"]),
                    f"{name} target readout")
            projection_calls += size
        energies = {name: prefix["arms"][name]["energy"] for name in ARMS}
        require(energies["orthogonal"] <= previous + 1e-9, "nested variational monotonicity")
        require(abs(energies["orthogonal"] - energies["same_span_mixed"]) < 1e-9,
                "same-span coordinate equivalence")
        require(energies["support_matched_determinants"] <= energies["orthogonal"] + 1e-9,
                "classical support containment")
        require(min(energies.values()) >= exact[0] - 1e-9, "variational bound")
        previous = energies["orthogonal"]
        final_h = basis.conj().T @ h @ basis
    resources = entry["resources"]
    selection_calls = 4 + 1 + dimension + len(packet_order)
    require(resources["selection_h_action_calls"] == selection_calls
            and resources["total_h_action_calls"] == selection_calls + projection_calls
            and resources["diagonal_entries_constructed"] == dimension
            and resources["packet_candidates_scored"] == len(packet_order)
            and resources["determinant_candidates_scored"] == len(external), "selection work")
    require(resources["quantum_state_preparation"] == "unimplemented_unpriced"
            and resources["quantum_measurement_shots"] == "unimplemented_unpriced", "quantum scope")
    require(np.isfinite(resources["wall_seconds_including_reference_and_noise"])
            and resources["wall_seconds_including_reference_and_noise"] > 0
            and entry["exact_reference"]["wall_seconds"] > 0, "observed timings")
    require(isinstance(resources["operator_storage"], dict)
            and all(type(v) is int and v >= 0 for v in resources["operator_storage"].values()),
            "observed native payload metadata")
    if final_h is not None:
        problems.extend(verify_noise(entry["noise"], final_h, e_classical, c,
                                     float(exact[0]), config, config["cases"].index(case)))
    return problems


def verify_noise(observed, h, e_classical, c, exact_energy, config, case_index):
    problems = []
    size = h.shape[0]
    transform = np.eye(size, dtype=complex)
    transform[:4, 4:] = config["same_span_mixing"] * c[:, None]
    overlap = transform.conj().T @ transform
    retained = float(np.linalg.eigvalsh(h)[0])
    noise = config["noise"]
    if observed["basis_dimension"] != size or [row["scale_index"] for row in observed["scales"]] != list(range(len(noise["scales"]))):
        return ["noise scope/grid incomplete"]
    for scale_index, scale in enumerate(noise["scales"]):
        block = observed["scales"][scale_index]
        if [row["replica"] for row in block["replicas"]] != list(range(noise["replicas"])):
            problems.append("noise replicas incomplete")
            continue
        expected_errors = {arm: [] for arm in NOISE_ARMS}
        violations = {arm: 0 for arm in NOISE_ARMS}
        for replica, row in enumerate(block["replicas"]):
            draws = []
            for namespace, sigma in enumerate((scale["hamiltonian_hartree"], scale["overlap_dimensionless"])):
                generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                    [noise["seed"], case_index, scale_index, replica, namespace])))
                values = generator.normal(0, sigma, size=(size, size))
                perturbation = np.zeros((size, size))
                for i in range(size):
                    for j in range(i, size):
                        if i >= 4 or j >= 4:
                            perturbation[i, j] = perturbation[j, i] = values[i, j]
                draws.append(perturbation)
            matrix = h - e_classical * np.eye(size) + draws[0]
            mixed = transform.conj().T @ matrix @ transform
            noisy_s = overlap + draws[1]
            smallest = float(np.linalg.eigvalsh(noisy_s)[0])
            eigenvalues = {
                "orthogonal_known_overlap": float(np.linalg.eigvalsh(matrix)[0]),
                "mixed_known_overlap": float(independent_generalized(mixed, overlap)[0][0]),
                "mixed_noisy_overlap": (None if smallest <= noise["minimum_overlap_eigenvalue"]
                    else float(independent_generalized(mixed, noisy_s)[0][0])),
            }
            if not close(row["minimum_noisy_overlap_eigenvalue"], smallest):
                problems.append("noise overlap eigenvalue differs")
            if abs(eigenvalues["orthogonal_known_overlap"] - eigenvalues["mixed_known_overlap"]) > 1e-9:
                problems.append("congruently transported noise changed the exact-span energy")
            for arm, value in eigenvalues.items():
                expected = None if value is None else e_classical + value - retained
                actual = row[arm]
                violation = None if value is None else e_classical + value < exact_energy - 1e-9
                if ((expected is None and actual["error"] is not None)
                        or (expected is not None and (actual["error"] is None or not close(actual["error"], expected)))
                        or actual["variational_violation"] is not violation):
                    problems.append(f"{arm} noise outcome differs")
                if expected is not None:
                    expected_errors[arm].append(expected)
                    violations[arm] += int(violation)
        if set(block["summary"]) != set(NOISE_ARMS):
            problems.append("noise summary arms incomplete")
            continue
        for arm, errors in expected_errors.items():
            expected = {"replicas": noise["replicas"], "successes": len(errors),
                        "failures": noise["replicas"] - len(errors),
                        "conditional_rmse": float(np.linalg.norm(errors) / np.sqrt(len(errors))) if errors else None,
                        "unconditional_target_fraction": sum(abs(e) <= config["accuracy_target_hartree"] for e in errors) / noise["replicas"],
                        "variational_violations": violations[arm]}
            actual = block["summary"][arm]
            if set(actual) != set(expected) or any(
                    actual.get(key) != value if value is None or isinstance(value, int)
                    else actual.get(key) is None or not close(actual[key], value)
                    for key, value in expected.items()):
                problems.append(f"{arm} noise failure/accuracy accounting differs")
    return problems


def ensure_execution_history(record, root=ROOT):
    """Restore inert original Git objects if publication reauthored commits.

    The checked-in, digest-bound bundle depends on the unchanged main baseline.
    Unbundling installs objects only: no branch, worktree or remote is changed.
    Scientific replay still verifies source bytes and declaration ancestry.
    """
    bundle = root / EXECUTION_BUNDLE
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != EXECUTION_BUNDLE_SHA256:
        raise ValueError("execution bundle digest differs")
    commits = [record["declaration_commit"], record["source_commit"]]
    if any(not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in commits):
        raise ValueError("execution identities must be complete commit SHAs")
    missing = any(subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=root,
                                capture_output=True).returncode for sha in commits)
    if missing:
        subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=root,
                       check=True, capture_output=True)
        subprocess.run(["git", "bundle", "unbundle", str(bundle)], cwd=root,
                       check=True, capture_output=True)
        for sha in commits:
            subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=root,
                           check=True, capture_output=True)


def verify(record, *, check_sources=True):
    config = read_json(CONFIG)
    validate_evidence(record)
    problems = []
    if (record["schema"] != "clifford_qc.orthogonal_active_space.v1"
            or record["config"] != config
            or record["config_sha256"] != hashlib.sha256(CONFIG.read_bytes()).hexdigest()
            or record["declaration_sha256"] != hashlib.sha256(DECLARATION.read_bytes()).hexdigest()
            or record["quantum_advantage_claim"] is not False or record["hardware_execution"] is not False):
        problems.append("record/design/scope identity differs")
    expected_tiers = {"energies_spans_residuals": "exact_small_sector_simulation",
                      "noise": "illustrative_model", "resources": "classical_process_measurement"}
    if {key: value["tier"] for key, value in record["evidence"].items()} != expected_tiers:
        problems.append("per-quantity evidence tiers differ")
    if [case["id"] for case in record["cases"]] != [case["id"] for case in config["cases"]]:
        return problems + ["complete ordered case grid required"]
    if check_sources:
        if subprocess.check_output(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, text=True).strip() != "false":
            problems.append("full Git history is required")
        ensure_execution_history(record)
        if set(record["source_sha256"]) != set(SOURCES):
            problems.append("execution source manifest incomplete")
        else:
            for path, digest in record["source_sha256"].items():
                content = subprocess.check_output(["git", "show", f"{record['source_commit']}:{path}"], cwd=ROOT)
                if hashlib.sha256(content).hexdigest() != digest:
                    problems.append(f"execution source digest differs: {path}")
                # Verification may improve after observations without rewriting
                # the frozen producer, experiment or historical checker bytes.
                if (path != "benchmarks/check_orthogonal_active_space.py"
                        and hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest):
                    problems.append(f"live execution source differs: {path}")
        if record["declaration_commit"] == record["source_commit"]:
            problems.append("declaration did not precede implementation")
        for relative in (CONFIG.relative_to(ROOT), DECLARATION.relative_to(ROOT)):
            old = subprocess.check_output(["git", "show", f"{record['declaration_commit']}:{relative}"], cwd=ROOT)
            if old != (ROOT / relative).read_bytes():
                problems.append(f"declaration bytes changed: {relative}")
        subprocess.check_call(["git", "merge-base", "--is-ancestor", record["declaration_commit"], record["source_commit"]], cwd=ROOT)
        if record["provenance"]["git_sha"] != record["source_commit"] or record["provenance"]["git_dirty"] is not False:
            problems.append("execution provenance does not name a clean source")
    for observed, case in zip(record["cases"], config["cases"]):
        path = ROOT / case["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            problems.append(f"{case['id']} input digest differs")
            continue
        data = read_fcidump(path)
        n = 2 * data.n_orbitals
        words = [word for word in range(1 << n) if word.bit_count() == data.n_electrons
                 and sum((1 if j % 2 == 0 else -1) * ((word >> (n - 1 - j)) & 1)
                         for j in range(n)) == data.ms2]
        if (observed["n_qubits"] != n or observed["n_electrons"] != data.n_electrons
                or observed["sz"] != data.sz
                or observed["reference_word"] != ((1 << data.n_electrons) - 1) << (n - data.n_electrons)):
            problems.append(f"{case['id']} fermionic metadata differs")
            continue
        h, _ = hamiltonian(path, words)
        problems.extend(verify_case(observed, case, config, h, words))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args(argv)
    try:
        problems = verify(read_json(args.record))
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        problems = [f"record could not be verified: {exc}"]
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print("VALID: independent occupation replay, controls, noise and source chronology")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
