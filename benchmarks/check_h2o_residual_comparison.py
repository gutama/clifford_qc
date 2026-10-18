"""Independent occupation-H replay and merge-safe observation binding."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.archive_provenance import commit_binding, history_kind
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from benchmarks.metadata_io import read_json, sha256
from benchmarks.phase17_ceo_oracle import hamiltonian
from benchmarks.run_h2o_residual_comparison import ARMS, BUNDLE, CONFIG, DECLARATION, RECEIPT, RECORD, SOURCES
from clifford_qc.evidence import validate_evidence


def occupation_basis(n_orbitals=6, pairs=4):
    return sorted(sum(1 << (2*n_orbitals - 1 - 2*p - spin)
                      for spin, occupied in enumerate((alpha, beta)) for p in occupied)
                  for alpha in combinations(range(n_orbitals), pairs)
                  for beta in combinations(range(n_orbitals), pairs))


def spatial_key(word, n, orbitals):
    return tuple(((word >> (n - 2 - 2*p)) & 3).bit_count() for p in orbitals)


def orthogonalize(vector, span, cutoff):
    result = np.array(vector, complex, copy=True)
    for _ in (0, 1):
        result -= span @ (span.conj().T @ result)
    length = np.linalg.norm(result)
    if length <= cutoff:
        return None
    result /= length
    pivot = result[np.argmax(abs(result))]
    return result * pivot.conjugate() / abs(pivot)


def independent_frame(words, n, s2, tolerance):
    # Separate tuple partition and projector construction; no producer builder.
    groups = {}
    for i, word in enumerate(words):
        key = sum(count * 3**p for p, count in enumerate(spatial_key(word, n, range(n//2))))
        groups.setdefault(key, []).append(i)
    columns, labels = [], []
    for key, indices in sorted(groups.items()):
        values, eigenvectors = np.linalg.eigh(s2[np.ix_(indices, indices)])
        zero = eigenvectors[:, abs(values) <= tolerance]
        projector = zero @ zero.conj().T
        local = np.empty((len(indices), 0), complex)
        for j, index in enumerate(indices):
            direction = orthogonalize(projector[:, j], local, tolerance)
            if direction is not None:
                local = np.column_stack((local, direction))
                embedded = np.zeros(len(words), complex)
                embedded[indices] = direction
                columns.append(embedded)
                labels.append((key, index))
    return np.column_stack(columns), labels


def decode(columns, dimension):
    result = np.zeros((dimension, len(columns)), complex)
    for j, column in enumerate(columns):
        indices = column["indices"]
        if sorted(set(indices)) != indices or any(type(i) is not int or not 0 <= i < dimension for i in indices):
            raise ValueError("invalid serialized support")
        values = np.array(column["real"]) + 1j*np.array(column["imag"])
        if values.shape != (len(indices),) or not np.isfinite(values).all():
            raise ValueError("invalid serialized amplitudes")
        result[indices, j] = values
    return result


def cas_indices(frame, words, n, hf, active, tolerance):
    external = [p for p in range(n//2) if p not in active]
    forbidden = [i for i, word in enumerate(words)
                 if spatial_key(word, n, external) != spatial_key(hf, n, external)]
    return [j for j, column in enumerate(frame.T)
            if np.max(abs(column[forbidden]), initial=0) <= tolerance]


def best_key(scores, tolerance):
    maximum = max(scores.values())
    return min(key for key in scores if scores[key] >= maximum - tolerance)


def verify_worker(result, config, h, words, s2):
    """Rebuild selection decisions, exact pencils, work counts and support."""
    problems = []

    def require(condition, label):
        if not condition:
            problems.append(label)

    def close(actual, expected, label, tolerance=1e-8):
        require(np.shape(actual) == np.shape(expected)
                and np.allclose(actual, expected, atol=tolerance, rtol=0), label)

    # Only synthetic fixtures override the declared 12-qubit register.
    n, dimension = config.get("test_n_qubits", 12), len(words)
    hf = sum(1 << (n - 1 - j) for j in range(config.get("test_n_electrons", 8)))
    frame, labels = independent_frame(words, n, s2, config["spin_tolerance"])
    active = config["input"]["active_spatial_orbitals"].copy()
    selected = cas_indices(frame, words, n, hf, active, config["spin_tolerance"])
    initial = selected.copy()
    external = [p for p in range(n//2) if p not in active]
    groups = {}
    for key, column in enumerate(frame.T):
        if key in initial:
            continue
        counts = spatial_key(words[np.argmax(abs(column))], n, range(n//2))
        pattern = sum((counts[p] if p in external else 0) * 3**p for p in range(n//2))
        groups.setdefault(pattern, []).append(key)
    basis = np.empty((dimension, 0), complex)
    cached = set()
    calls, scored = len(initial), 0
    expected_columns, expected_selection, expected_score = frame[:, initial], {"initial_csf_indices": initial}, None
    previous_energy, previous_time, previous_rss = float("inf"), -1., 0
    pending_terminal = None
    for index, row in enumerate(result["trajectory"]):
        added = decode(row["added_columns"], dimension)
        close(added @ added.conj().T, expected_columns @ expected_columns.conj().T, "selected correction span")
        require(row["selection"] == expected_selection, "selection identity")
        if expected_score is not None:
            close(row["selection_score"], expected_score, "selection score")
        else:
            require(row["selection_score"] is None, "initial selection score")
        basis = np.column_stack((basis, added))
        close(basis.conj().T @ basis, np.eye(basis.shape[1]), "whole-span orthogonality", 1e-9)
        projected = basis.conj().T @ h @ basis
        energies, c = np.linalg.eigh((projected + projected.conj().T) / 2)
        energy = float(energies[0])
        state = basis @ c[:, 0]
        residual = h @ state - energy * state
        close(row["energy"], energy, "Ritz energy")
        close(row["residual_norm"], np.linalg.norm(residual), "true residual")
        close(row["spin_action_norm"], np.linalg.norm(s2 @ state), "spin diagnostic")
        require(np.linalg.norm(s2 @ state) <= config["spin_tolerance"], "singlet state")
        require(row["step"] == index and row["dimension"] == basis.shape[1], "prefix grid")
        require(row["h_action_calls"] == calls and row["candidates_scored"] == scored, "cumulative work")
        require(row["retained_payload_bytes"] == basis.size * 32, "retained V/HV payload")
        require(row["candidate_cache_payload_bytes"] == len(cached) * dimension * 32, "candidate cache payload")
        require(row["amplitude_nonzeros"] == np.count_nonzero(abs(basis) > config["serialization_tolerance"]), "amplitude support")
        require(row["determinant_support"] == np.count_nonzero(np.any(abs(basis) > config["serialization_tolerance"], axis=1)), "determinant support")
        require(energy <= previous_energy + 1e-9, "nested monotonicity")
        require(np.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= previous_time, "cumulative timing")
        require(type(row["peak_rss_bytes"]) is int and row["peak_rss_bytes"] >= max(1, previous_rss), "OS RSS high-water")
        previous_energy, previous_time, previous_rss = energy, row["wall_seconds"], row["peak_rss_bytes"]
        terminal = ("residual_tolerance" if np.linalg.norm(residual) <= config["residual_tolerance_hartree"]
                    else "column_ceiling" if basis.shape[1] >= config["maximum_columns"]
                    else "growth_step_ceiling" if index == config["maximum_growth_steps"] else None)
        if terminal:
            require(index == len(result["trajectory"])-1, "trajectory continued after stopping")
            pending_terminal = terminal
            break
        scores, candidates = {}, {}
        arm = result["arm"]
        if arm == "residual_packets":
            raw = residual / np.maximum(h.diagonal().real - energy + config["preconditioner_shift_hartree"], config["preconditioner_shift_hartree"])
            cutoff = max(config["direction_absolute_tolerance"], config["direction_relative_tolerance"] * np.linalg.norm(raw))
            for key, ids in groups.items():
                block = frame[:, ids]
                direction = orthogonalize(block @ (block.conj().T @ raw), basis, cutoff)
                if direction is not None:
                    candidates[key] = direction
        elif arm == "spin_adapted_selected_ci":
            for key in range(frame.shape[1]):
                if key not in selected:
                    candidates[key] = frame[:, key]
            calls += len(set(candidates) - cached)
            cached.update(candidates)
        elif arm == "adaptive_active_space":
            for p in range(n//2):
                if p not in active:
                    changed = [i for i, word in enumerate(words)
                               if spatial_key(word, n, [p]) != spatial_key(hf, n, [p])]
                    scores[p] = float(np.linalg.norm(residual[changed])**2)
        else:
            raise ValueError("unknown arm")
        for key, direction in candidates.items():
            gap = float(np.vdot(direction, h @ direction).real - energy)
            coupling = abs(np.vdot(direction, residual))
            scores[key] = max(0., float((np.sqrt(gap**2 + 4*coupling**2) - gap) / 2))
        if arm == "residual_packets":
            calls += len(candidates)
        scored += len(scores)
        if not scores:
            pending_terminal = "no_admissible_direction"
            require(index == len(result["trajectory"])-1, "missing candidate stop")
            break
        key = best_key(scores, config["score_tie_tolerance"])
        # Independent integral summation can perturb numerically tied scores.
        if index+1 < len(result["trajectory"]):
            observed = result["trajectory"][index+1]["selection"]
            observed_key = observed.get("candidate", observed.get("orbital"))
            require(observed_key in scores and scores[key] - scores.get(observed_key, -1) <= 2e-10,
                    "highest-scoring candidate")
            if observed_key in scores:
                key = observed_key
        expected_score = scores[key]
        if arm == "adaptive_active_space":
            active.append(key)
            enlarged = cas_indices(frame, words, n, hf, active, config["spin_tolerance"])
            extra = [j for j in enlarged if j not in selected]
            if not extra or len(enlarged) > config["maximum_columns"]:
                pending_terminal = "no_admissible_block"
                break
            expected_columns = frame[:, extra]
            expected_selection = {"orbital": key, "active_orbitals": sorted(active), "csf_indices": extra}
            selected = enlarged
            calls += len(extra)
        else:
            expected_columns = candidates[key][:, None]
            expected_selection = {"candidate": key}
            if arm == "spin_adapted_selected_ci":
                selected.append(key)
                cached.remove(key)
    require(bool(result["trajectory"]) and result["stop_reason"] == pending_terminal, "stopping rule")
    require(result["final_h_action_calls"] == calls and result["final_candidates_scored"] == scored, "terminal work")
    require(result["solver_wall_seconds"] >= previous_time and result["final_peak_rss_bytes"] >= previous_rss, "terminal cost")
    setup = result.get("setup")
    if setup:
        require(setup["sector_dimension"] == dimension and setup["singlet_dimension"] == frame.shape[1], "setup dimensions")
        require(setup["csf_labels"] == [list(label) for label in labels], "deterministic CSF labels")
        require(setup["array_payload_bytes"] == 16*(dimension**2 + frame.size + dimension), "setup payload")
    return problems


def restore_and_bind(record):
    history_kind(ROOT)
    if record != read_json(ROOT / RECORD):
        raise ValueError("inspected object differs from the preserved observations")
    receipt = read_json(ROOT / RECEIPT)
    if (receipt["schema"] != "clifford_qc.h2o_residual_publication.v1"
            or receipt["bundle_sha256"] != sha256(ROOT / BUNDLE)
            or receipt["observation_sha256"] != sha256(ROOT / RECORD)):
        raise ValueError("publication/observation byte binding differs")
    subprocess.run(["git", "bundle", "verify", str(ROOT / BUNDLE)], cwd=ROOT,
                   check=True, capture_output=True, timeout=15)
    subprocess.run(["git", "bundle", "unbundle", str(ROOT / BUNDLE)], cwd=ROOT,
                   check=True, capture_output=True, timeout=15)
    if commit_binding(ROOT, receipt["observation_commit"], RECORD)["sha256"] != sha256(ROOT / RECORD):
        raise ValueError("observation bytes differ from their observation commit")
    if set(record["source_sha256"]) != set(SOURCES):
        raise ValueError("source manifest differs")
    for path, digest in record["source_sha256"].items():
        if commit_binding(ROOT, record["source_commit"], path)["sha256"] != digest:
            raise ValueError(f"source binding differs: {path}")
    for path, key in ((CONFIG, "config_sha256"), (DECLARATION, "declaration_sha256")):
        if (commit_binding(ROOT, record["declaration_commit"], path)["sha256"] != record[key]
                or sha256(ROOT / path) != record[key]):
            raise ValueError("declared design binding differs")
    for earlier, later in ((record["declaration_commit"], record["source_commit"]),
                           (record["source_commit"], receipt["observation_commit"])):
        if earlier == later:
            raise ValueError("declaration/execution/observation must be distinct")
        subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later], cwd=ROOT,
                       check=True, capture_output=True, timeout=15)
    for path in record["config"]["implementation_paths_absent_at_declaration"]:
        if subprocess.run(["git", "cat-file", "-e", f"{record['declaration_commit']}:{path}"],
                          cwd=ROOT, capture_output=True, timeout=15).returncode == 0:
            raise ValueError("implementation predates declaration")
    provenance = record["provenance"]
    if provenance["git_sha"] != record["source_commit"] or provenance["git_dirty"] is not False:
        raise ValueError("execution provenance differs")


def verify(record, *, bindings=True):
    if bindings:
        restore_and_bind(record)
    validate_evidence(record)
    config = read_json(ROOT / CONFIG)
    if (record["config"] != config or record["schema"] != "clifford_qc.h2o_residual_comparison.v1"
            or record["hardware_execution"] is not False or record["quantum_advantage_claim"] is not False
            or sha256(ROOT / config["input"]["path"]) != config["input"]["sha256"]):
        raise ValueError("experiment identity differs")
    words = occupation_basis()
    h, _ = hamiltonian(ROOT / config["input"]["path"], words)
    s2 = independent_spin_squared(words, 12)
    reference = float(np.linalg.eigvalsh(h)[0])
    problems = []
    if abs(record["reference"]["energy"] - reference) > 1e-9:
        problems.append("reference energy")
    expected = [(repeat, arm) for repeat in range(config["repeats"])
                for arm in list(ARMS)[repeat:] + list(ARMS)[:repeat]]
    if [(w["repeat"], w["arm"]) for w in record["workers"]] != expected:
        problems.append("all repeats/arms in declared rotated order required")
    for i, entry in enumerate(record["workers"]):
        if entry["order_index"] != i or entry["status"] not in ("ok", "failed", "timeout"):
            problems.append("worker identity/status")
        if entry["status"] != "ok":
            if not entry.get("diagnostic") or "result" in entry:
                problems.append("failed worker disclosure")
            continue
        result = entry["result"]
        if result["arm"] != entry["arm"]:
            problems.append("worker arm mismatch")
        problems.extend(f"{entry['arm']}/{entry['repeat']}: {p}"
                        for p in verify_worker(result, config, h, words, s2))
        for row in result["trajectory"]:
            error = row["energy"] - reference
            if (abs(row["error_hartree"] - error) > 1e-9
                    or row["target_reached"] != (abs(error) <= config["accuracy_target_hartree"])
                    or error < -1e-8):
                problems.append("post-run target/error/variational readout")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        problems = verify(read_json(ROOT / RECORD), bindings=not args.science_only)
    except (OSError, ValueError, TypeError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        problems = [str(exc)]
    for problem in problems:
        print(f"FAIL H2O: {problem}")
    if not problems:
        print("OK independent H2O replay" + ("; chronology/observation bindings UNVERIFIED" if args.science_only else "; observation bytes and chronology verified"))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
