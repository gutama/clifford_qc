"""Declared exact H2O comparison; without --execute only validate the design."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.archive_provenance import commit_binding, history_kind, safe_path
from benchmarks.metadata_io import read_json, sha256
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.evidence import declare_evidence
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.pauli_action import parity, word_masks
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.action_space import ActionSpace, energy_lowering, orthogonal_direction
from clifford_qc.subspace.spin_packets import spatial_pattern, spin_squared_operator

CONFIG = "benchmarks/configs/h2o_residual_comparison.json"
DECLARATION = "benchmarks/H2O_RESIDUAL_COMPARISON_DECLARATION.md"
RECORD = "benchmarks/reference_results/h2o_residual_comparison.json"
RECEIPT = "benchmarks/provenance/h2o_residual_comparison.json"
BUNDLE = "benchmarks/provenance/h2o_residual_comparison.bundle"
ARMS = ("residual_packets", "spin_adapted_selected_ci", "adaptive_active_space")
SOURCES = (
    "clifford_qc/subspace/action_space.py", "benchmarks/run_h2o_residual_comparison.py",
    "benchmarks/summarize_h2o_residual_comparison.py",
    "benchmarks/check_h2o_residual_comparison.py", "benchmarks/phase17_ceo_oracle.py",
    "benchmarks/check_orthogonal_active_space.py", "clifford_qc/subspace/spin_packets.py",
    "clifford_qc/subspace/orthogonal.py", "clifford_qc/backends/sector_statevector.py",
    "clifford_qc/models/fcidump.py", "clifford_qc/pauli_action.py",
    "clifford_qc/reproducibility.py", "benchmarks/archive_provenance.py",
    "benchmarks/metadata_io.py", "clifford_qc/fermion.py",
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=15).strip()


def load_design():
    config = read_json(ROOT / CONFIG)
    if (config["schema"] != "clifford_qc.h2o_residual_comparison_design.v1"
            or tuple(config["arms"]) != ARMS or config["repeats"] != 3
            or config["maximum_sector_dimension"] != 225 or config["maximum_columns"] != 105
            or config["input"]["active_spatial_orbitals"] != [3, 4]
            or sha256(safe_path(ROOT, config["input"]["path"])) != config["input"]["sha256"]):
        raise ValueError("frozen H2O design/input differs")
    return config


def preflight(config):
    history_kind(ROOT)
    provenance = stamp_record({})["provenance"]
    source = git("rev-parse", "HEAD")
    if git("status", "--porcelain") or provenance["git_dirty"] or provenance["git_sha"] != source:
        raise ValueError("execution requires clean committed sources, including untracked files")
    if any(os.environ.get(k) != "1" for k in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("declare one numerical-library thread before execution")
    declaration = git("log", "-1", "--format=%H", "--", DECLARATION)
    if declaration == source:
        raise ValueError("declaration must precede implementation")
    subprocess.run(["git", "merge-base", "--is-ancestor", declaration, source],
                   cwd=ROOT, check=True, capture_output=True, timeout=15)
    for path in (CONFIG, DECLARATION):
        if commit_binding(ROOT, declaration, path)["sha256"] != sha256(ROOT / path):
            raise ValueError("declared design bytes changed")
    for path in config["implementation_paths_absent_at_declaration"]:
        if subprocess.run(["git", "cat-file", "-e", f"{declaration}:{path}"],
                          cwd=ROOT, capture_output=True, timeout=15).returncode == 0:
            raise ValueError("implementation exists at declaration")
    return source, declaration, provenance


def singlet_frame(words, n, s2, tolerance):
    """Canonical coordinate-axis basis of each spatial-configuration singlet block."""
    groups = np.array([spatial_pattern(int(w), (1 << n) - 1, n) for w in words])
    columns, labels = [], []
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        values, vectors = np.linalg.eigh(s2[np.ix_(indices, indices)])
        selected = vectors[:, abs(values) <= tolerance]
        projector = selected @ selected.conj().T
        local = np.empty((len(indices), 0), complex)
        for j in range(len(indices)):
            direction = orthogonal_direction(projector[:, j], local, cutoff=tolerance)
            if direction is not None:
                local = np.column_stack((local, direction))
                vector = np.zeros(len(words), complex)
                vector[indices] = direction
                columns.append(vector)
                labels.append((int(group), int(indices[j])))
    return np.column_stack(columns), labels


def active_columns(frame, words, n, hf, active, tolerance):
    mask = sum(1 << (n - 1 - 2*p - spin) for p in active for spin in (0, 1))
    outside = ((1 << n) - 1) ^ mask
    allowed = (words & outside) == (hf & outside)
    return [i for i, column in enumerate(frame.T) if np.max(abs(column[~allowed]), initial=0) <= tolerance]


def pick(scores, tolerance):
    maximum = max(scores.values())
    return min(key for key in scores if scores[key] >= maximum - tolerance)


def encode_columns(columns, tolerance):
    answer = []
    for column in np.asarray(columns).T:
        indices = np.flatnonzero(abs(column) > tolerance)
        answer.append({"indices": indices.tolist(), "real": column[indices].real.tolist(),
                       "imag": column[indices].imag.tolist()})
    return answer


def peak_rss_bytes():
    # Linux getrusage reports KiB; Darwin reports bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def trajectory(apply, diagonal, words, n, hf, frame, config, arm, s2):
    """An exact action-only solve; no reference eigenvalue is accepted here."""
    active = list(config["input"]["active_spatial_orbitals"])
    selected = active_columns(frame, words, n, hf, active, config["spin_tolerance"])
    initial = frame[:, selected]
    started = time.perf_counter()
    space = ActionSpace(apply, initial, tolerance=config["orthogonality_tolerance"])
    scored, candidate_cache, rows = 0, {}, []
    added, choice, choice_score = initial, {"initial_csf_indices": selected.copy()}, None
    external_mask = ((1 << n) - 1) ^ sum(
        1 << (n - 1 - 2*p - spin) for p in active for spin in (0, 1))
    patterns = np.array([spatial_pattern(int(w), external_mask, n) for w in words])
    frame_patterns = [int(patterns[np.argmax(abs(column))]) for column in frame.T]
    groups = {key: frame[:, [i for i, k in enumerate(frame_patterns) if k == key
                            and i not in selected]] for key in sorted(set(frame_patterns))}
    stop = None
    for step in range(config["maximum_growth_steps"] + 1):
        energy, state, residual = space.solve()
        if np.linalg.norm(s2 @ state) > config["spin_tolerance"]:
            raise ValueError("Ritz state escaped the declared singlet sector")
        rows.append({
            "step": step, "energy": energy, "residual_norm": float(np.linalg.norm(residual)),
            "spin_action_norm": float(np.linalg.norm(s2 @ state)),
            "dimension": space.basis.shape[1], "h_action_calls": space.h_action_calls,
            "candidates_scored": scored, "added_columns": encode_columns(added, config["serialization_tolerance"]),
            "selection": choice, "selection_score": choice_score,
            "retained_payload_bytes": space.payload_bytes,
            "candidate_cache_payload_bytes": sum(v.nbytes + a.nbytes for v, a in candidate_cache.values()),
            "determinant_support": int(np.count_nonzero(np.any(abs(space.basis) > config["serialization_tolerance"], axis=1))),
            "amplitude_nonzeros": int(np.count_nonzero(abs(space.basis) > config["serialization_tolerance"])),
            "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": peak_rss_bytes(),
        })
        if rows[-1]["residual_norm"] <= config["residual_tolerance_hartree"]:
            stop = "residual_tolerance"
        elif space.basis.shape[1] >= config["maximum_columns"]:
            stop = "column_ceiling"
        elif step == config["maximum_growth_steps"]:
            stop = "growth_step_ceiling"
        if stop:
            break
        candidates, scores = {}, {}
        if arm == "residual_packets":
            raw = residual / np.maximum(diagonal - energy + config["preconditioner_shift_hartree"],
                                        config["preconditioner_shift_hartree"])
            cutoff = max(config["direction_absolute_tolerance"],
                         config["direction_relative_tolerance"] * np.linalg.norm(raw))
            for key, group in groups.items():
                direction = orthogonal_direction(group @ (group.conj().T @ raw), space.basis,
                                                 cutoff=cutoff)
                if direction is not None:
                    acted = space.action(direction)
                    candidates[key] = (direction, acted)
                    scores[key] = energy_lowering(energy, np.vdot(direction, acted).real,
                                                  np.vdot(state, acted))
        elif arm == "spin_adapted_selected_ci":
            for key in range(frame.shape[1]):
                if key in selected:
                    continue
                if key not in candidate_cache:
                    direction = frame[:, key]
                    candidate_cache[key] = direction, space.action(direction)
                direction, acted = candidate_cache[key]
                candidates[key] = direction, acted
                scores[key] = energy_lowering(energy, np.vdot(direction, acted).real,
                                              np.vdot(state, acted))
        elif arm == "adaptive_active_space":
            for p in range(n // 2):
                if p in active:
                    continue
                mask = 3 << (n - 2 - 2*p)
                changed = (words & mask) != (hf & mask)
                scores[p] = float(np.sum(abs(residual[changed])**2))
        else:
            raise ValueError("unknown arm")
        scored += len(scores)
        if not scores:
            stop = "no_admissible_direction"
            break
        key = pick(scores, config["score_tie_tolerance"])
        choice_score = scores[key]
        if arm == "adaptive_active_space":
            active.append(key)
            enlarged = active_columns(frame, words, n, hf, active, config["spin_tolerance"])
            extra = [i for i in enlarged if i not in selected]
            if not extra or len(enlarged) > config["maximum_columns"]:
                stop = "no_admissible_block"
                break
            added = frame[:, extra]
            selected = enlarged
            space.extend(added)
            choice = {"orbital": key, "active_orbitals": sorted(active), "csf_indices": extra}
        else:
            direction, acted = candidates[key]
            added = direction[:, None]
            space.extend(added, actions=acted[:, None])
            choice = {"candidate": key}
            if arm == "spin_adapted_selected_ci":
                selected.append(key)
                # Selected actions now live in HV, not in the external cache.
                candidate_cache.pop(key)
    return {"arm": arm, "trajectory": rows, "stop_reason": stop,
            "final_h_action_calls": space.h_action_calls, "final_candidates_scored": scored,
            "solver_wall_seconds": time.perf_counter() - started,
            "final_peak_rss_bytes": peak_rss_bytes()}


def worker(arm, config):
    preflight(config)
    start = time.perf_counter()
    model = fcidump_model(ROOT / config["input"]["path"])
    backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], model.metadata["sz"])
    if backend.dimension > config["maximum_sector_dimension"]:
        raise ValueError("sector ceiling exceeded")
    operator = backend.operator(model.hamiltonian)
    spin = backend.operator(spin_squared_operator(model.n))
    s2 = spin.restrict(np.arange(backend.dimension))
    frame, labels = singlet_frame(backend.basis, model.n, s2, config["spin_tolerance"])
    diagonal = np.zeros(backend.dimension, complex)
    diagonal_terms = 0
    for code, coefficient in model.hamiltonian.to_mv().terms.items():
        x, z, y = word_masks(model.n, code)
        if x == 0:
            diagonal += coefficient * (1j**y) * (1 - 2*parity(backend.basis, z))
            diagonal_terms += 1
    if np.max(abs(diagonal.imag)) > config["orthogonality_tolerance"]:
        raise ValueError("nonreal Hermitian diagonal")
    hf = sum(1 << (model.n - 1 - mode) for mode in range(model.metadata["n_electrons"]))
    setup = {"wall_seconds": time.perf_counter() - start,
             "peak_rss_bytes": peak_rss_bytes(), "sector_dimension": backend.dimension,
             "singlet_dimension": frame.shape[1], "spin_matrix_columns": backend.dimension,
             "diagonal_pauli_terms": diagonal_terms,
             "array_payload_bytes": s2.nbytes + frame.nbytes + diagonal.nbytes,
             "hamiltonian_operator_storage": operator.memory_estimate(),
             "spin_operator_storage": spin.memory_estimate(), "csf_labels": labels}
    result = trajectory(operator.matvec, diagonal.real, backend.basis, model.n, hf, frame,
                        config, arm, s2)
    result["setup"] = setup
    return result


def execute(config):
    source, declaration, provenance = preflight(config)
    identity = {path: sha256(safe_path(ROOT, path)) for path in SOURCES}
    for path, digest in identity.items():
        if commit_binding(ROOT, source, path)["sha256"] != digest:
            raise ValueError(f"uncommitted source: {path}")
    workers = []
    with tempfile.TemporaryDirectory(prefix="h2o-comparison-") as tmp:
        for repeat in range(config["repeats"]):
            order = config["arms"][repeat:] + config["arms"][:repeat]
            for arm in order:
                output = Path(tmp) / f"{repeat}-{arm}.json"
                entry = {"repeat": repeat, "arm": arm, "order_index": len(workers)}
                start = time.perf_counter()
                try:
                    done = subprocess.run([sys.executable, __file__, "--worker", arm,
                                           "--output", str(output)], cwd=ROOT, text=True,
                                          capture_output=True, timeout=config["worker_timeout_seconds"])
                    if done.returncode:
                        entry.update(status="failed", diagnostic=(done.stderr or done.stdout)[-4000:])
                    else:
                        entry.update(status="ok", result=read_json(output))
                except subprocess.TimeoutExpired:
                    entry.update(status="timeout", diagnostic="declared worker timeout")
                entry["process_wall_seconds"] = time.perf_counter() - start
                workers.append(entry)
    # The independent full-sector energy is obtained only after every arm finishes.
    from benchmarks.phase17_ceo_oracle import hamiltonian
    from clifford_qc.backends.sector_statevector import sector_basis
    words = sector_basis(12, 8, 0).tolist()
    start = time.perf_counter()
    h, _ = hamiltonian(ROOT / config["input"]["path"], words)
    reference = float(np.linalg.eigvalsh(h)[0])
    for entry in workers:
        if entry["status"] == "ok":
            for row in entry["result"]["trajectory"]:
                row["error_hartree"] = row["energy"] - reference
                row["target_reached"] = abs(row["error_hartree"]) <= config["accuracy_target_hartree"]
    record = {"schema": "clifford_qc.h2o_residual_comparison.v1", "config": config,
              "config_sha256": sha256(ROOT / CONFIG), "declaration_commit": declaration,
              "declaration_sha256": sha256(ROOT / DECLARATION), "source_commit": source,
              "source_sha256": identity, "workers": workers,
              "reference": {"energy": reference, "wall_seconds": time.perf_counter() - start,
                            "role": "independent_occupation_evaluation_after_all_workers"},
              "hardware_execution": False, "quantum_advantage_claim": False}
    record = declare_evidence(record, {
        "energies_selection_residuals": {"tier": "exact_small_sector_simulation",
            "scope": "One fixed 225-determinant H2O sector, explicitly stored vectors and dense small-sector singlet setup; no scaling or optimized SCI claim."},
        "costs": {"tier": "classical_process_measurement",
            "scope": "Three fresh processes per arm: cumulative wall times, counted H actions, explicit array payloads and OS process high-water RSS including imports/setup; reference evaluation excluded."},
    })
    if preflight(config)[0] != source:
        raise ValueError("source changed during execution")
    return stamp_record(record, provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=ROOT / RECORD)
    args = parser.parse_args(argv)
    config = load_design()
    if not args.execute and not args.worker:
        print("VALID frozen design; no new molecular outcomes computed")
        return 0
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = worker(args.worker, config) if args.worker else execute(config)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
