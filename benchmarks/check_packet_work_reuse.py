"""Independent dense occupation replay of packet screening and action counts."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.archive_provenance import commit_binding, history_kind
from benchmarks.check_h2o_residual_comparison import (
    occupation_basis, independent_frame, spatial_key, cas_indices,
    orthogonalize, decode, best_key,
)
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from benchmarks.metadata_io import read_json, sha256
from benchmarks.phase17_ceo_oracle import hamiltonian
from benchmarks.run_packet_work_reuse import ARMS, BUNDLE, CONFIG, DECLARATION, RECEIPT, RECORD, SOURCES
from clifford_qc.evidence import validate_evidence


def verify_worker(result, config, h, words, s2):
    problems = []

    def require(condition, label):
        if not condition:
            problems.append(label)

    def close(actual, expected, label, tolerance=2e-8):
        require(np.shape(actual) == np.shape(expected) and
                np.allclose(actual, expected, atol=tolerance, rtol=0), label)

    n, d = config.get("test_n_qubits", 12), len(words)
    hf = sum(1 << (n - 1 - j) for j in range(config.get("test_n_electrons", 8)))
    frame, labels = independent_frame(words, n, s2, config["spin_tolerance"])
    active = config["input"]["active_spatial_orbitals"]
    selected = cas_indices(frame, words, n, hf, active, config["spin_tolerance"])
    initial = selected.copy()
    groups = {}
    for j, column in enumerate(frame.T):
        if j not in initial:
            counts = spatial_key(words[np.argmax(abs(column))], n, range(n//2))
            key = sum((counts[p] if p not in active else 0) * 3**p for p in range(n//2))
            groups.setdefault(key, []).append(j)
    arm = result["arm"]
    require(arm in ARMS, "arm identity")
    reuse, screening = arm in ("reuse", "combined"), arm in ("shortlist", "combined")
    populated, ci_cached = set(initial) if reuse else set(), set()
    calls, scored, cheap = len(initial), 0, 0
    expected, choice, selected_score = frame[:, initial], {"initial_csf_indices": initial}, None
    span = np.empty((d, 0), complex)
    previous_energy, previous_time, previous_rss = float("inf"), -1., 0
    terminal, decision_count = None, 0
    for step, row in enumerate(result["trajectory"]):
        added = decode(row["added_columns"], d)
        close(added @ added.conj().T, expected @ expected.conj().T, "selected packet span")
        require(row["selection"] == choice, "selection identity")
        if selected_score is None:
            require(row["selection_score"] is None, "initial score")
        else:
            close(row["selection_score"], selected_score, "selection score")
        span = np.column_stack((span, added))
        close(span.conj().T @ span, np.eye(span.shape[1]), "whole-span orthogonality", 1e-9)
        values, vectors = np.linalg.eigh(span.conj().T @ h @ span)
        energy, state = float(values[0]), span @ vectors[:, 0]
        residual = h @ state - energy * state
        close(row["energy"], energy, "Ritz energy")
        close(row["residual_norm"], np.linalg.norm(residual), "full residual")
        close(row["spin_action_norm"], np.linalg.norm(s2 @ state), "spin diagnostic")
        require(np.linalg.norm(s2 @ state) <= config["spin_tolerance"], "singlet state")
        require(row["step"] == step and row["dimension"] == span.shape[1], "prefix identity")
        require((row["h_action_calls"], row["candidates_scored"], row["cheap_candidates"]) == (calls, scored, cheap), "work counts")
        require(row["frame_cache_indices"] == sorted(populated), "frame cache population")
        require(row["frame_snapshot_payload_bytes"] == (frame.nbytes if reuse else 0), "frame snapshot payload")
        require(row["frame_actions_payload_bytes"] == len(populated)*d*16, "frame actions payload")
        require(row["candidate_cache_payload_bytes"] == len(ci_cached)*d*32, "SCI cache payload")
        require(row["retained_payload_bytes"] == span.size*32, "V/HV payload")
        require(row["amplitude_nonzeros"] == np.count_nonzero(abs(span) > config["serialization_tolerance"]), "amplitude support")
        require(row["determinant_support"] == np.count_nonzero(np.any(abs(span) > config["serialization_tolerance"], axis=1)), "determinant support")
        require(energy <= previous_energy + 1e-9, "nested monotonicity")
        require(np.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= previous_time, "prefix timing")
        require(type(row["peak_rss_bytes"]) is int and row["peak_rss_bytes"] >= max(1, previous_rss), "prefix RSS")
        previous_energy, previous_time, previous_rss = energy, row["wall_seconds"], row["peak_rss_bytes"]
        terminal = ("residual_tolerance" if np.linalg.norm(residual) <= config["residual_tolerance_hartree"]
                    else "column_ceiling" if span.shape[1] >= config["maximum_columns"]
                    else "growth_step_ceiling" if step == config["maximum_growth_steps"] else None)
        if terminal:
            require(step == len(result["trajectory"])-1, "continued after terminal prefix")
            break
        candidates, indicators = {}, {}
        if arm == "selected_ci":
            candidates = {j: frame[:, j] for j in range(frame.shape[1]) if j not in selected}
            indicators = {j: float(abs(np.vdot(v, residual))**2) for j, v in candidates.items()}
        else:
            raw = residual / np.maximum(h.diagonal().real - energy + config["preconditioner_shift_hartree"], config["preconditioner_shift_hartree"])
            cutoff = max(config["direction_absolute_tolerance"], config["direction_relative_tolerance"] * np.linalg.norm(raw))
            for key, ids in sorted(groups.items()):
                block = frame[:, ids]
                direction = orthogonalize(block @ (block.conj().T @ raw), span, cutoff)
                if direction is not None:
                    candidates[key] = direction
                    indicators[key] = float(np.linalg.norm(block.conj().T @ residual)**2)
        cheap += len(candidates)
        if not candidates:
            terminal = "no_admissible_direction"
            require(step == len(result["trajectory"])-1, "missing candidate stop")
            break
        if step >= len(result["decisions"]):
            problems.append("missing decision")
            break
        decision = result["decisions"][step]
        decision_count += 1
        require(set(decision["indicators"]) == {str(k) for k in indicators}, "indicator universe")
        if set(decision["indicators"]) != {str(k) for k in indicators}:
            return problems
        indicator_scale = max(sum(indicators.values()), 1e-20)
        for key, expected_indicator in indicators.items():
            close(decision["indicators"][str(key)], expected_indicator,
                  "residual indicators", max(1e-24, 1e-10*indicator_scale))
        # Exact policy replay uses independently checked recorded indicators:
        # separate BLAS reductions can swap symmetric last-bit ties.
        policy_indicators = {int(k): v for k,v in decision["indicators"].items()}
        ordered = sorted(policy_indicators, key=lambda k: (-policy_indicators[k], k))
        short = sorted(candidates)
        if screening:
            total, covered, short = sum(policy_indicators.values()), 0., []
            for k in ordered:
                short.append(k)
                covered += policy_indicators[k]
                if total > 0 and covered >= config["shortlist_fraction"] * total:
                    break
        audit = screening and (step + 1) % config["audit_interval"] == 0
        evaluated = sorted(candidates) if audit else short.copy()
        all_scores = {}
        for key, v in candidates.items():
            gap = np.vdot(v, h @ v).real - energy
            coupling = abs(np.vdot(v, residual))
            all_scores[key] = max(0., float((np.hypot(gap, 2*coupling)-gap)/2))
        fallback = screening and not audit and max(all_scores[k] for k in evaluated) <= config["score_tie_tolerance"] and len(evaluated) < len(candidates)
        if fallback:
            evaluated = sorted(candidates)
        require(decision["step"] == step and decision["audit"] == audit and decision["fallback"] == fallback, "decision policy")
        require(decision["shortlist"] == short, "shortlist identity")
        require(decision["evaluated"] == evaluated, "evaluated universe")
        require(set(decision["scores"]) == {str(k) for k in evaluated}, "score universe")
        for key in evaluated:
            close(decision["scores"][str(key)], all_scores[key], "exact score")
        scores = {key: all_scores[key] for key in evaluated}
        winner = best_key(scores, config["score_tie_tolerance"])
        observed = decision["selected"]
        require(observed in scores and scores[winner]-scores.get(observed, -1) <= 2e-10, "best evaluated selection")
        # The recorded scores have been independently checked. Enforce their
        # exact declared tie-break too, rather than widening it to replay noise.
        recorded_scores = {int(k): v for k,v in decision["scores"].items()}
        require(observed == best_key(recorded_scores, config["score_tie_tolerance"]), "recorded score tie-break")
        if observed in scores:
            winner = observed
        require(decision["audit_missed_winner"] == bool(audit and winner not in short), "audit miss")
        if audit:
            close(decision["audit_score_loss"], max(0., max(scores.values()) - max(scores[k] for k in short)), "audit loss")
            close(decision["audit_score_loss"], max(0., max(recorded_scores.values()) - max(recorded_scores[k] for k in short)), "recorded audit loss", 1e-14)
        else:
            require(decision["audit_score_loss"] is None, "unaudited loss")
        if reuse:
            requested = {j for key in evaluated for j in groups[key]}
            calls += len(requested - populated)
            populated |= requested
        elif arm == "selected_ci":
            calls += len(set(evaluated) - ci_cached)
            ci_cached |= set(evaluated)
        else:
            calls += len(evaluated)
        scored += len(evaluated)
        expected, choice, selected_score = candidates[winner][:, None], {"candidate": winner}, scores[winner]
        if arm == "selected_ci":
            selected.append(winner)
            ci_cached.remove(winner)
    require(bool(result["trajectory"]) and result["stop_reason"] == terminal, "stopping rule")
    require(len(result["decisions"]) == decision_count, "decision count")
    require(result["final_h_action_calls"] == calls and result["final_candidates_scored"] == scored, "terminal work")
    require(result["solver_wall_seconds"] >= previous_time and result["final_peak_rss_bytes"] >= previous_rss, "terminal costs")
    if reuse:
        require(calls <= frame.shape[1], "fixed-frame action ceiling")
    setup = result.get("setup")
    if setup:
        require(setup["sector_dimension"] == d and setup["singlet_dimension"] == frame.shape[1], "setup dimensions")
        require(setup["csf_labels"] == [list(label) for label in labels], "setup frame labels")
        require(setup["array_payload_bytes"] == 16*(d*d + frame.size + d), "setup payload")
    return problems


def restore_and_bind(record):
    history_kind(ROOT)
    if record != read_json(ROOT / RECORD):
        raise ValueError("inspected object differs from the preserved observations")
    receipt = read_json(ROOT / RECEIPT)
    if (receipt["schema"] != "clifford_qc.packet_work_reuse_publication.v1"
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
    if (record["config"] != config or record["schema"] != "clifford_qc.packet_work_reuse.v1"
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
    parser.add_argument("--record", type=Path, default=ROOT / RECORD)
    args = parser.parse_args(argv)
    try:
        problems = verify(read_json(args.record), bindings=not args.science_only)
    except (OSError, ValueError, TypeError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        problems = [str(exc)]
    for problem in problems:
        print(f"FAIL packet work: {problem}")
    if not problems:
        print("OK independent packet work replay" + ("; chronology/observation bindings UNVERIFIED" if args.science_only else "; observation bytes and chronology verified"))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
