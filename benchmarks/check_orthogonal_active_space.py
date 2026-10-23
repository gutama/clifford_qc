"""Replay the orthogonal-correction pilot with independent occupation matrices.

No producer, native sector operator, Pauli product, packet builder or new solver
is imported. FCIDUMP parsing and the existing independent fermionic sign oracle
are shared. Timings and compiled native storage are observations, not rebuilt
performance guarantees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.archive_provenance import HistoryError, commit_binding, history_kind, safe_path
from benchmarks.metadata_io import read_json, sha256
from benchmarks.phase17_ceo_oracle import _fermionic_action, hamiltonian
from clifford_qc.evidence import validate_evidence
from clifford_qc.models.fcidump import read_fcidump

RECORD = ROOT / "benchmarks/reference_results/orthogonal_active_space.json"
CONFIG = ROOT / "benchmarks/configs/orthogonal_active_space.json"
DECLARATION = ROOT / "benchmarks/ORTHOGONAL_ACTIVE_SPACE_DECLARATION.md"
V2_CONFIG = ROOT / "benchmarks/configs/orthogonal_active_space_v2.json"
V2_DECLARATION = ROOT / "benchmarks/ORTHOGONAL_ACTIVE_SPACE_V2_DECLARATION.md"
V2_RECORD = ROOT / "benchmarks/reference_results/orthogonal_active_space_v2.json"
V2_OBSERVATION_COMMIT = "d45ef931c91d25f76047bc6ef01e003c70eaf434"
V2_EXECUTION_BUNDLE = "benchmarks/provenance/orthogonal_active_space_v2.bundle"
V2_EXECUTION_BUNDLE_SHA256 = "c0ddff528a235a26c02a4fb7b3a510ed6c8799fb5c08d9b647e1ddc578e6594b"
OBSERVATION_COMMIT = "2e354b690e7b7f1bcb642dc9fac453a2b09e8be9"
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

V2_SOURCES = SOURCES + ("clifford_qc/subspace/spin_packets.py",
                        "benchmarks/archive_provenance.py", "benchmarks/metadata_io.py")


def independent_spin_squared(words, n):
    """Occupation-sign S², independent of the producer's JW Pauli operator."""
    lookup = {word: i for i, word in enumerate(words)}
    result = np.zeros((len(words), len(words)), complex)
    for column, word in enumerate(words):
        sz = sum((1 if j % 2 == 0 else -1) * ((word >> (n - 1 - j)) & 1)
                 for j in range(n)) / 2
        result[column, column] += sz**2
        for p in range(n // 2):
            for q in range(n // 2):
                for spin in (0, 1):
                    sequence = [(2*p+spin, True), (2*p+1-spin, False),
                                (2*q+1-spin, True), (2*q+spin, False)]
                    acted = _fermionic_action(word, sequence, n)
                    if acted is not None:
                        target, sign = acted
                        if target in lookup:
                            result[lookup[target], column] += .5 * sign
    return result


def external_spatial_pattern(word, mask, n):
    counts = []
    for p in range(n // 2):
        counts.append(sum(bool(word & mask & (1 << (n - 1 - 2*p - spin)))
                          for spin in (0, 1)))
    return sum(count * 3**p for p, count in enumerate(counts))


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


def packet_agrees(row, expected, words, classical_indices, external_mask, zero_tolerance, *,
                   n=None):
    """Compare reconstructed vectors, not unstable thresholded index lists.

    Independent integral summation can move a numerical zero across the 1e-12
    serialization cutoff after normalization of a small packet. The existing
    1e-9 vector tolerance still bounds every omitted or included amplitude.
    Structural sector/classical-support membership is checked exactly.
    """
    indices = row["indices"]
    if (not indices or any(type(i) is not int or i < 0 or i >= len(words) for i in indices)
            or sorted(set(indices)) != indices
            or any(i in classical_indices or (words[i] & external_mask if n is None
                       else external_spatial_pattern(words[i], external_mask, n)) != row["pattern"]
                   for i in indices)):
        return False
    coefficients = decode(row["coefficients"])
    if coefficients.shape != (len(indices),) or np.any(abs(coefficients) <= zero_tolerance):
        return False
    reconstructed = np.zeros(len(words), complex)
    reconstructed[indices] = coefficients
    pivot = reconstructed[int(np.argmax(abs(reconstructed)))]
    if abs(pivot.imag) > 1e-12 or pivot.real <= 0:
        return False
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
    packets, scores, recorded_packets = {}, {}, {}
    revised = config["schema"].endswith(".v2")
    s2 = independent_spin_squared(words, n)

    def score(vector):
        coupling = np.vdot(state, h @ vector)
        diagonal = float(np.vdot(vector, h @ vector).real) - e_classical
        return max(0.0, -float(np.linalg.eigvalsh(
            np.array([[0, coupling], [coupling.conjugate(), diagonal]]))[0]))

    patterns = [(external_spatial_pattern(word, external_mask, n) if revised
                 else word & external_mask) for word in words]
    cutoff = max(config["packet_zero_tolerance"], config.get("packet_relative_tolerance", 0)
                 * float(np.linalg.norm(raw[external])))
    spin_eigensolves = 0
    for pattern in sorted({patterns[i] for i in external}):
        indices = [i for i in external if patterns[i] == pattern]
        vector = np.zeros(dimension, complex)
        vector[indices] = raw[indices]
        if np.linalg.norm(vector) <= cutoff:
            continue
        if revised:
            spin_eigensolves += 1
            values, columns = np.linalg.eigh(s2[np.ix_(indices, indices)])
            singlets = columns[:, abs(values) <= config["spin_tolerance"]]
            vector[indices] = singlets @ (singlets.conj().T @ raw[indices])
        norm = np.linalg.norm(vector)
        if norm > cutoff:
            packets[pattern] = phase(vector / norm)
            scores[pattern] = score(packets[pattern])
    packet_order = ordered(scores)
    require([row["pattern"] for row in entry["packet_ranking"]] == packet_order,
            "complete packet ordering")
    for row in entry["packet_ranking"]:
        expected = packets[row["pattern"]]
        agrees = packet_agrees(row, expected, words, classical_indices, external_mask,
                                config["packet_zero_tolerance"], n=n if revised else None)
        require(close(row["score"], scores[row["pattern"]]) and agrees,
                "packet support/coefficients/score")
        recorded_vector = np.zeros(dimension, complex)
        recorded_vector[row["indices"]] = decode(row["coefficients"])
        recorded_packets[row["pattern"]] = recorded_vector
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
    require(set(classical_indices).isdisjoint(external)
            and sorted(classical_indices + external) == list(range(dimension)),
            "classical/complement partition")
    if revised:
        require(entry["classical_complement_partition_valid"] is True, "partition receipt")
    else:
        # Preserve the old scalar as historical data, not an independent
        # diagnostic: a permutation necessarily has the same spectrum.
        require(0 <= entry["complete_complement_spectrum_defect"] < 1e-9,
                "historical permutation round-off")
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
        serialized_basis = np.column_stack([*classical.T,
                            *[recorded_packets[pattern] for pattern in patterns]])
        support = np.flatnonzero(np.any(serialized_basis != 0, axis=1))
        bases = (basis, basis @ transform, np.eye(dimension, dtype=complex)[:, simple],
                 np.eye(dimension, dtype=complex)[:, support])
        require(set(prefix["arms"]) == set(ARMS), "complete control arms")
        for name, columns in zip(ARMS, bases):
            observed = prefix["arms"][name]
            projected, overlap = columns.conj().T @ h @ columns, columns.conj().T @ columns
            es, cs = independent_generalized(projected, overlap)
            psi = columns @ cs[:, 0]
            residual_norm = float(np.linalg.norm(h @ psi - es[0] * psi))
            if revised:
                spin = float(np.vdot(psi, s2 @ psi).real)
                spin_norm = float(np.linalg.norm(s2 @ psi))
                require(close(observed["spin_squared"], spin)
                        and close(observed["spin_action_norm"], spin_norm), f"{name} spin diagnostics")
                if name in ("orthogonal", "same_span_mixed"):
                    require(spin_norm <= config["spin_tolerance"], f"{name} singlet state")
            size = columns.shape[1]
            support_count = (len(support) if name in ("orthogonal", "same_span_mixed")
                             else len(simple) if name == "dimension_matched_determinants"
                             else len(support))
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
    if revised:
        require(resources["spin_packet_construction"]["block_eigensolves"] == spin_eigensolves
                and close(resources["spin_packet_construction"]["norm_cutoff"], cutoff, 1e-12)
                and resources["spin_matrix_bytes"] == s2.nbytes
                and resources["spin_matrix_columns_constructed"] == dimension, "spin construction work")
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


def git_run(root, *args, check=True):
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              check=check, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HistoryError(f"Git history operation failed ({args[0]}): {type(exc).__name__}") from exc


def ensure_execution_history(record, root=ROOT):
    """Restore digest-bound inert v1 objects only after full-history inspection."""
    history_kind(root)
    revised = record["schema"].endswith(".v2")
    relative_bundle, digest, observation = ((V2_EXECUTION_BUNDLE, V2_EXECUTION_BUNDLE_SHA256, V2_OBSERVATION_COMMIT)
        if revised else (EXECUTION_BUNDLE, EXECUTION_BUNDLE_SHA256, OBSERVATION_COMMIT))
    bundle = safe_path(root, relative_bundle)
    if sha256(bundle) != digest:
        raise ValueError("execution bundle digest differs")
    commits = [record["declaration_commit"], record["source_commit"], observation]
    if any(not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in commits):
        raise ValueError("execution identities must be complete commit SHAs")
    missing = any(git_run(root, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode
                  for sha in commits)
    if missing:
        git_run(root, "bundle", "verify", str(bundle))
        git_run(root, "bundle", "unbundle", str(bundle))
        for sha in commits:
            git_run(root, "cat-file", "-e", f"{sha}^{{commit}}")


def source_problems(record, config_path, declaration_path, record_path, *, root=ROOT):
    """Bind preserved observations to historical execution bytes and ancestry.

    Current implementations are exercised by behavioral tests, independently
    of these historical identities; reproducing a pilot uses its source commit.
    """
    history_kind(root)  # Stop before restoring objects in shallow/archive trees.
    revised = record["schema"].endswith(".v2")
    ensure_execution_history(record, root)
    observation = V2_OBSERVATION_COMMIT if revised else OBSERVATION_COMMIT
    if revised:
        changed = git_run(root, "diff-tree", "--no-commit-id", "--name-only", "-r", observation).stdout.decode().splitlines()
        if changed != [str(V2_RECORD.relative_to(ROOT))] or observation == record["source_commit"]:
            return ["revised observations must first be committed alone after implementation"]
        git_run(root, "merge-base", "--is-ancestor", record["source_commit"], observation)
    relative_record = str(record_path.relative_to(ROOT))
    expected_record = commit_binding(root, observation, relative_record)["sha256"]
    if sha256(safe_path(root, relative_record)) != expected_record:
        return ["original observation bytes differ from their observation commit"]
    if read_json(root / relative_record) != record:
        return ["inspected record differs from preserved observation bytes"]
    sources = V2_SOURCES if revised else SOURCES
    if set(record["source_sha256"]) != set(sources):
        return ["execution source manifest incomplete"]
    problems = []
    for path, digest in record["source_sha256"].items():
        if commit_binding(root, record["source_commit"], path)["sha256"] != digest:
            problems.append(f"execution source digest differs: {path}")
    if record["declaration_commit"] == record["source_commit"]:
        problems.append("declaration did not precede implementation")
    for path in (config_path, declaration_path):
        relative = str(path.relative_to(ROOT))
        if commit_binding(root, record["declaration_commit"], relative)["sha256"] != sha256(root / relative):
            problems.append(f"declaration bytes changed: {relative}")
    absent = (record["config"]["implementation_paths_absent_at_declaration"] if revised else
              ["benchmarks/run_orthogonal_active_space.py", "clifford_qc/subspace/orthogonal.py"])
    if revised and absent != ["clifford_qc/subspace/spin_packets.py"]:
        problems.append("revision implementation absence manifest differs")
    for path in absent:
        if git_run(root, "cat-file", "-e", f"{record['declaration_commit']}:{path}", check=False).returncode == 0:
            problems.append(f"implementation already exists at declaration: {path}")
    git_run(root, "merge-base", "--is-ancestor", record["declaration_commit"], record["source_commit"])
    if record["provenance"]["git_sha"] != record["source_commit"] or record["provenance"]["git_dirty"] is not False:
        problems.append("execution provenance does not name a clean source")
    return problems


def verify(record, *, check_sources=True, record_path=None):
    revised = record["schema"] == "clifford_qc.orthogonal_active_space.v2"
    config_path, declaration_path = (V2_CONFIG, V2_DECLARATION) if revised else (CONFIG, DECLARATION)
    record_path = (record_path or (V2_RECORD if revised else RECORD)).resolve()
    config = read_json(config_path)
    validate_evidence(record)
    problems = []
    if (record["schema"] not in ("clifford_qc.orthogonal_active_space.v1", "clifford_qc.orthogonal_active_space.v2")
            or record["config"] != config or record["config_sha256"] != sha256(config_path)
            or record["declaration_sha256"] != sha256(declaration_path)
            or record["quantum_advantage_claim"] is not False or record["hardware_execution"] is not False):
        problems.append("record/design/scope identity differs")
    expected_tiers = {"energies_spans_residuals": "exact_small_sector_simulation",
                      "noise": "illustrative_model", "resources": "classical_process_measurement"}
    if {key: value["tier"] for key, value in record["evidence"].items()} != expected_tiers:
        problems.append("per-quantity evidence tiers differ")
    if [case["id"] for case in record["cases"]] != [case["id"] for case in config["cases"]]:
        return problems + ["complete ordered case grid required"]
    if check_sources:
        try:
            problems.extend(source_problems(record, config_path, declaration_path, record_path))
        except (HistoryError, ValueError) as exc:
            return problems + [str(exc)]
    for observed, case in zip(record["cases"], config["cases"]):
        path = safe_path(ROOT, case["path"])
        if sha256(path) != case["sha256"]:
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
    parser.add_argument("--record", type=Path)
    parser.add_argument("--science-only", action="store_true",
                        help="explicitly replay science without certifying observation bytes or chronology")
    args = parser.parse_args(argv)
    try:
        paths = [args.record] if args.record else [RECORD, V2_RECORD]
        problems = []
        for path in paths:
            problems.extend(verify(read_json(path), record_path=path, check_sources=not args.science_only))
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        problems = [f"record could not be verified: {exc}"]
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print("VALID: independent occupation replay, controls and noise; "
              + ("observation bindings and chronology UNVERIFIED" if args.science_only
                 else "preserved observation bytes and source chronology"))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
