"""Declared exact packet work comparison; without --execute only validate the design."""
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

CONFIG = "benchmarks/configs/packet_work_reuse.json"
DECLARATION = "benchmarks/PACKET_WORK_REUSE_DECLARATION.md"
RECORD = "benchmarks/reference_results/packet_work_reuse.json"
RECEIPT = "benchmarks/provenance/packet_work_reuse.json"
BUNDLE = "benchmarks/provenance/packet_work_reuse.bundle"
ARMS = ("original", "reuse", "shortlist", "combined", "selected_ci")
from benchmarks.run_h2o_residual_comparison import (
    SOURCES as HISTORICAL_SOURCES, active_columns, encode_columns, peak_rss_bytes,
    pick, singlet_frame,
)
from clifford_qc.subspace.packet_selection import (
    FrameActionCache, orthogonal_action, residual_shortlist,
)
SOURCES = tuple(sorted(set(HISTORICAL_SOURCES) | {
    "clifford_qc/subspace/packet_selection.py", "benchmarks/run_packet_work_reuse.py",
    "benchmarks/check_packet_work_reuse.py", "benchmarks/summarize_packet_work_reuse.py",
    "clifford_qc/evidence.py", "clifford_qc/record_environment.py",
}))



def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=15).strip()


def load_design():
    config = read_json(ROOT / CONFIG)
    if (config["schema"] != "clifford_qc.packet_work_reuse_design.v1"
            or tuple(config["arms"]) != ARMS or config["repeats"] != 3
            or config["maximum_sector_dimension"] != 225 or config["maximum_columns"] != 105
            or config["input"]["active_spatial_orbitals"] != [3, 4]
            or config["shortlist_fraction"] != 0.8 or config["audit_interval"] != 5
            or sha256(safe_path(ROOT, config["input"]["path"])) != config["input"]["sha256"]):
        raise ValueError("frozen packet design/input differs")
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


def trajectory(apply, diagonal, words, n, hf, frame, config, arm, s2):
    """Run one fixed-policy classical arm; no reference energy is accepted."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    started = time.perf_counter()
    selected = active_columns(frame, words, n, hf,
                              config["input"]["active_spatial_orbitals"], config["spin_tolerance"])
    initial = frame[:, selected]
    space = ActionSpace(apply, initial, tolerance=config["orthogonality_tolerance"])
    reuse, screening = arm in ("reuse", "combined"), arm in ("shortlist", "combined")
    cache = FrameActionCache(frame, space.action) if reuse else None
    if cache:
        cache.seed(selected, space.actions)
    external_mask = ((1 << n) - 1) ^ sum(
        1 << (n - 1 - 2*p - spin) for p in config["input"]["active_spatial_orbitals"]
        for spin in (0, 1))
    patterns = np.array([spatial_pattern(int(w), external_mask, n) for w in words])
    groups = {}
    for j, column in enumerate(frame.T):
        if j not in selected:
            groups.setdefault(int(patterns[np.argmax(abs(column))]), []).append(j)
    # Index lists share the original frame: no separate dense group snapshots.
    ci_cache, rows, decisions = {}, [], []
    scored, cheap_count = 0, 0
    added, choice, choice_score = initial, {"initial_csf_indices": selected.copy()}, None
    for step in range(config["maximum_growth_steps"] + 1):
        energy, state, residual = space.solve()
        spin_norm = float(np.linalg.norm(s2 @ state))
        if spin_norm > config["spin_tolerance"]:
            raise ValueError("Ritz state escaped singlet sector")
        rows.append({
            "step": step, "energy": energy, "residual_norm": float(np.linalg.norm(residual)),
            "spin_action_norm": spin_norm, "dimension": space.basis.shape[1],
            "h_action_calls": space.h_action_calls, "candidates_scored": scored,
            "cheap_candidates": cheap_count, "added_columns": encode_columns(added, config["serialization_tolerance"]),
            "selection": choice, "selection_score": choice_score,
            "retained_payload_bytes": space.payload_bytes,
            "candidate_cache_payload_bytes": sum(v.nbytes + a.nbytes for v, a in ci_cache.values()),
            "frame_cache_indices": list(cache.indices) if cache else [],
            "frame_snapshot_payload_bytes": cache.frame_payload_bytes if cache else 0,
            "frame_actions_payload_bytes": cache.action_payload_bytes if cache else 0,
            "determinant_support": int(np.count_nonzero(np.any(abs(space.basis) > config["serialization_tolerance"], axis=1))),
            "amplitude_nonzeros": int(np.count_nonzero(abs(space.basis) > config["serialization_tolerance"])),
            "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": peak_rss_bytes(),
        })
        stop = ("residual_tolerance" if rows[-1]["residual_norm"] <= config["residual_tolerance_hartree"]
                else "column_ceiling" if space.basis.shape[1] >= config["maximum_columns"]
                else "growth_step_ceiling" if step == config["maximum_growth_steps"] else None)
        if stop:
            break
        directions, coefficients, indicators = {}, {}, {}
        if arm == "selected_ci":
            directions = {j: frame[:, j] for j in range(frame.shape[1]) if j not in selected}
            indicators = {j: float(abs(np.vdot(v, residual))**2) for j, v in directions.items()}
        else:
            raw = residual / np.maximum(diagonal - energy + config["preconditioner_shift_hartree"],
                                        config["preconditioner_shift_hartree"])
            cutoff = max(config["direction_absolute_tolerance"],
                         config["direction_relative_tolerance"] * np.linalg.norm(raw))
            for key, ids in sorted(groups.items()):
                block = frame[:, ids]
                a = block.conj().T @ raw
                direction = orthogonal_direction(block @ a, space.basis, cutoff=cutoff)
                if direction is not None:
                    directions[key], coefficients[key] = direction, a
                    indicators[key] = float(np.linalg.norm(block.conj().T @ residual)**2)
        cheap_count += len(directions)
        if not directions:
            stop = "no_admissible_direction"
            break
        short = list(residual_shortlist(indicators, fraction=config["shortlist_fraction"])) if screening else sorted(directions)
        audit = screening and (step + 1) % config["audit_interval"] == 0
        evaluated = sorted(directions) if audit else short.copy()
        pairs, scores = {}, {}

        def evaluate(keys):
            for key in keys:
                if key in scores:
                    continue
                direction = directions[key]
                if cache:
                    vector, acted = cache.combine(groups[key], coefficients[key])
                    pair = orthogonal_action(vector, acted, space.basis, space.actions, cutoff=cutoff)
                    if pair is None or np.max(abs(pair[0] - direction)) > config["orthogonality_tolerance"]:
                        raise ValueError("cached and direct packet directions differ")
                    direction, acted = pair
                elif arm == "selected_ci":
                    if key not in ci_cache:
                        ci_cache[key] = direction, space.action(direction)
                    direction, acted = ci_cache[key]
                else:
                    acted = space.action(direction)
                pairs[key] = direction, acted
                scores[key] = energy_lowering(energy, np.vdot(direction, acted).real,
                                              np.vdot(state, acted))

        evaluate(evaluated)
        fallback = screening and not audit and max(scores.values()) <= config["score_tie_tolerance"] and len(scores) < len(directions)
        if fallback:
            evaluated = sorted(directions)
            evaluate(evaluated)
        scored += len(scores)
        key = pick(scores, config["score_tie_tolerance"])
        decisions.append({"step": step, "indicators": {str(k): v for k, v in indicators.items()},
                          "shortlist": short, "evaluated": evaluated, "audit": audit,
                          "fallback": fallback, "scores": {str(k): v for k, v in scores.items()},
                          "selected": key, "audit_missed_winner": bool(audit and key not in short),
                          "audit_score_loss": max(0., max(scores.values()) - max(scores[k] for k in short)) if audit else None})
        direction, acted = pairs[key]
        added = direction[:, None]
        space.extend(added, actions=acted[:, None])
        choice, choice_score = {"candidate": key}, scores[key]
        if arm == "selected_ci":
            selected.append(key)
            ci_cache.pop(key)
    return {"arm": arm, "trajectory": rows, "decisions": decisions, "stop_reason": stop,
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
    record = {"schema": "clifford_qc.packet_work_reuse.v1", "config": config,
              "config_sha256": sha256(ROOT / CONFIG), "declaration_commit": declaration,
              "declaration_sha256": sha256(ROOT / DECLARATION), "source_commit": source,
              "source_sha256": identity, "workers": workers,
              "reference": {"energy": reference, "wall_seconds": time.perf_counter() - start,
                            "role": "independent_occupation_evaluation_after_all_workers"},
              "hardware_execution": False, "quantum_advantage_claim": False}
    record = declare_evidence(record, {
        "energies_selection_residuals": {"tier": "exact_small_sector_simulation",
            "scope": "One fixed 225-determinant H2O sector with declared action caching and heuristic residual screening, explicitly stored vectors and dense small-sector singlet setup; no scaling or optimized SCI claim."},
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
