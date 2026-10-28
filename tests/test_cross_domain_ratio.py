"""Undeclared toy outcomes and outcome-free molecular structural controls."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import run_cross_domain_ratio as runner


def toy_config():
    c = deepcopy(runner.load_declaration())
    c.update(total_shot_budgets=[8192], replicas=2, master_seed=8877)
    c["cases"] = [{"name": "undeclared_two_qubit_test", "kind": "tfim", "role": "development_primary",
                   "n": 2, "unit": "t", "target_estimation_abs_error": .05,
                   "J": .8, "h": .9, "theta": .7, "phi": .17}]
    return c


def test_fixed_declaration_inputs_and_admission_bound_without_drawing(monkeypatch):
    c = runner.load_declaration()
    assert runner.array_admission_bound(8) == 378535936
    assert runner.array_admission_bound(8) < c["owned_array_cap_bytes"]
    monkeypatch.setattr(runner.JointShotSimulator, "draw", lambda *a: pytest.fail("structural preflight drew outcomes"))
    for case in c["cases"]:
        h, generators, state = runner.build_case(case)
        from_circuit = runner.preparation_state_from_circuit(case)
        phase = np.vdot(state, from_circuit)
        np.testing.assert_allclose(from_circuit, state*phase/abs(phase), atol=1e-12)
        gates = runner.preparation_circuit(case)
        prep = runner.preparation_resources(case)
        assert sum(len(qs) == 1 for _, qs, _ in gates) == prep["one_qubit_gates"]
        assert sum(len(qs) == 2 for _, qs, _ in gates) == prep["two_qubit_gates"]
        if case["kind"] != "molecule":
            continue
        independent = runner.assessment_data(case, h, generators, state, include_spectrum=False)
        basis = np.column_stack([runner.dense_operator(g)@state for _, g in generators])
        np.testing.assert_allclose(basis.conj().T@basis, independent["overlap"], atol=1e-12)
        np.testing.assert_allclose(basis.conj().T@runner.dense_operator(h)@basis,
                                   independent["hamiltonian"], atol=1e-10)
        assert independent["reference_spin_defect"] < 1e-12
        assert len(independent["sector_words"]) == 36
        assert not np.allclose(independent["overlap"], np.eye(3))
        assert np.linalg.eigvalsh(independent["overlap"])[0] > .05
        assert all(word.bit_count() == 4 for word in independent["sector_words"])


def test_case_targets_and_input_hashes_cannot_silently_drift(tmp_path):
    c = runner.load_declaration()
    for field, bad in (("fcidump_sha256", "0"*64), ("theta", .31), ("target_spin", 1)):
        altered = deepcopy(c); altered["cases"][1][field] = bad
        path = tmp_path/(field+".json"); path.write_text(json.dumps(altered))
        with pytest.raises(ValueError): runner.load_declaration(path)
    altered = deepcopy(c); altered["cases"][1]["target_estimation_abs_error"] = .05
    path = tmp_path/"target.json"; path.write_text(json.dumps(altered))
    with pytest.raises(ValueError): runner.load_declaration(path)


def test_toy_sampled_case_replays_coefficients_covariance_and_all_outcomes(monkeypatch, tmp_path):
    c = toy_config()
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 1000)
    checkpoint = tmp_path/"case.json"
    one = runner.run_case(c, 0, checkpoint=checkpoint)
    two = runner.run_case(c, 0)
    assert len(one["rows"]) == c["replicas"]*2
    assert json.loads(checkpoint.read_text())["status"] == "complete"
    runner.compare_records(one, two)
    rows = one["rows"]
    assert any(row["status"] == "resolved" for row in rows)
    for replica in range(c["replicas"]):
        ratio, bank = [r for r in rows if r["replica"] == replica]
        np.testing.assert_allclose(runner.decode_complex(ratio["coefficients"]),
                                   runner.decode_complex(bank["coefficients"]), atol=1e-12)
        assert ratio["pilot_identity"] == bank["pilot_identity"]
        assert ratio["evaluation_seed"] != bank["evaluation_seed"] != ratio["pilot_seed"]
    for row in rows:
        assert row["total_shots"] == row["pilot_shots"]+row["evaluation_shots"] == 8192
        assert row["classical_seconds"] >= row["setup_seconds"]
        assert row["end_to_end_seconds"] == row["device_seconds"]+row["classical_seconds"]
        assert row["unit"] == "t" and row["target_estimation_abs_error"] == .05
        assert row["worker_pid"] == os.getpid()
        if row["status"] == "resolved":
            assert row["denominator_interval"][0] > c["denominator_floor"]
            assert row["estimation_error"] == abs(row["estimate"]-row["true_direction_energy"])
            assert row["interval"][0] <= row["estimate"] <= row["interval"][1]
            assert math.isfinite(row["numerator_denominator_covariance"])
    summaries = runner.summarize(rows, c)
    assert all(s["attempts"] == c["replicas"] for s in summaries)
    assert runner.verdict(summaries, c)["status"] == "closed_on_declared_scope"


def test_assessment_occurs_after_inference_and_cannot_supply_allocation(monkeypatch):
    c = toy_config(); c["replicas"] = 1
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 0)
    events = []
    original_eval, original_assess = runner.evaluate_ratio, runner.assessment_data
    def evaluate(*a, **k):
        events.append("inference")
        return original_eval(*a, **k)
    def assess(*a, **k):
        assert events and events[-1] == "inference"
        events.append("assessment")
        return original_assess(*a, **k)
    monkeypatch.setattr(runner, "evaluate_ratio", evaluate)
    monkeypatch.setattr(runner, "assessment_data", assess)
    result = runner.run_case(c, 0)
    assert len(result["rows"]) == 2
    assert events.count("assessment") == 1
    assert events[0] == "inference"


def test_preallocation_and_pilot_failures_keep_unconditional_denominators(monkeypatch):
    c = toy_config(); c["owned_array_cap_bytes"] = 1
    result = runner.run_case(c, 0)
    assert len(result["rows"]) == 4
    assert all(r["status"] == "resource_failed" and r["total_shots"] == 0 for r in result["rows"])
    assert all(s["successes"] == 0 and s["attempts"] == 2 for s in runner.summarize(result["rows"], c))
    c = toy_config()
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 0)
    def fail(*a, **k): raise np.linalg.LinAlgError("controlled pilot failure")
    monkeypatch.setattr(runner.SampledProjectedProvider, "freeze_ratio", fail)
    result = runner.run_case(c, 0)
    assert all(r["status"] == "pilot_failed" and r["total_shots"] == 2048 for r in result["rows"])
    assert all(r["device_seconds"] > 0 for r in result["rows"])


def test_atomic_checkpoint_preserves_previous_bytes_on_nonfinite_failure(tmp_path):
    path = tmp_path/"checkpoint.json"
    runner.atomic_json(path, {"ok": 1})
    before = path.read_bytes()
    with pytest.raises(ValueError): runner.atomic_json(path, {"bad": float("nan")})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_missing_worker_outcomes_remain_failures_and_do_not_promote():
    c = toy_config()
    rows = runner._fill_failures(c["cases"][0], c, 0, [], "worker unavailable")
    assert len(rows) == 4
    assert all(r["status"] == "worker_failed" and r["total_shots"] == 0 for r in rows)
    summary = runner.summarize(rows, c)
    assert all(r["worker_failures"] == 2 and r["successes"] == 0 for r in summary)
    decision = runner.verdict(summary, c)
    assert decision["status"] == "closed_on_declared_scope"
    assert decision["comparisons"][0]["cost_reduction"] is None


def test_clean_scientific_sources_and_committed_declaration_required(monkeypatch):
    c = toy_config()
    monkeypatch.setattr(runner, "scientific_sources", lambda config: ("some_uncommitted_source.py",))
    called = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: called.append(a[0]))
    def git(*args):
        if args[:2] == ("cat-file", "-e"):
            raise runner.subprocess.CalledProcessError(1, args)
        return "test"
    monkeypatch.setattr(runner, "git", git)
    with pytest.raises(runner.subprocess.CalledProcessError): runner._committed_preflight(c)
    assert called and called[0][:3] == ["git", "diff", "--exit-code"]


def test_frozen_stack_and_single_thread_rules_are_enforced(monkeypatch):
    c = runner.load_declaration()
    monkeypatch.setattr(runner.platform, "python_version", lambda: "0.0.0")
    with pytest.raises(ValueError, match="environment mismatch"):
        runner.validate_execution_environment(c)


def test_completed_draw_is_charged_when_pilot_validation_fails(monkeypatch):
    import clifford_qc.sampled_reference as sampled
    c = toy_config(); c["replicas"] = 1
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 0)
    def fail(*a, **k): raise ValueError("controlled post-draw validation failure")
    monkeypatch.setattr(sampled, "validate_batch", fail)
    result = runner.run_case(c, 0)
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert row["status"] == "pilot_failed"
        assert row["pilot_shots"] == row["total_shots"] == 2048
        assert row["attempted_physical_shots"] == 2048
        assert row["costs_complete"] and not row["physical_shots_unknown"]
        assert row["device_seconds"] > 0
        assert row["end_to_end_seconds"] == row["device_seconds"]+row["classical_seconds"]


def test_interrupted_raw_draw_is_not_recorded_as_zero_cost(monkeypatch):
    c = toy_config(); c["replicas"] = 1
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 0)
    def fail(*a, **k): raise MemoryError("interrupted raw draw with unknown partial spending")
    monkeypatch.setattr(runner.JointShotSimulator, "draw", fail)
    result = runner.run_case(c, 0)
    for row in result["rows"]:
        assert row["status"] == "resource_failed"
        assert row["physical_shots_unknown"] and not row["costs_complete"]
        assert row["attempted_physical_shots"] == 2048
        assert row["device_seconds"] is row["end_to_end_seconds"] is None
    summary = runner.summarize(result["rows"], c)
    assert all(r["incomplete_costs"] == 1 for r in summary)
    assert runner.verdict(summary, c)["comparisons"][0]["cost_reduction"] is None


def test_group_moments_reconstruct_covariance_and_delta_variance(monkeypatch):
    c = toy_config(); c["replicas"] = 1
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 0)
    out = runner.run_case(c, 0)
    for row in out["rows"]:
        assert row["status"] == "resolved"
        moments = row["evaluation_group_moments"]
        covariance = sum((g["sum_hs"]/g["shots"]-g["sum_h"]*g["sum_s"]/g["shots"]**2)/g["shots"]
                         for g in moments)
        vh = sum((g["sum_h2"]/g["shots"]-(g["sum_h"]/g["shots"])**2)/g["shots"] for g in moments)
        vs = sum((g["sum_s2"]/g["shots"]-(g["sum_s"]/g["shots"])**2)/g["shots"] for g in moments)
        assert covariance == pytest.approx(row["numerator_denominator_covariance"], abs=1e-12)
        expected = max(0., (vh+row["raw_ratio"]**2*vs-2*row["raw_ratio"]*covariance)/row["denominator"]**2)
        assert expected == pytest.approx(row["delta_variance"], abs=1e-12)


def test_private_workers_cannot_draw_without_live_claim_or_bound_replay(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(runner, "_common_git_directory", lambda: tmp_path)
    monkeypatch.setattr(runner, "claim_path", lambda: tmp_path/"missing-claim.json")
    args = SimpleNamespace(authorization=None, worker_replay=False, replay_receipt=None)
    with pytest.raises(ValueError, match="nonce"):
        runner.worker_authorized(args, "execution", "declaration")
    args.authorization = "nonce"
    with pytest.raises(FileNotFoundError):
        runner.worker_authorized(args, "execution", "declaration")
    args.worker_replay = True
    with pytest.raises(ValueError, match="receipt"):
        runner.worker_authorized(args, "execution", "declaration")
    receipt = tmp_path/"replay.json"
    receipt.write_text(json.dumps({"mode": "replay", "record_path": str(tmp_path/"missing.json")}))
    args.replay_receipt = receipt
    with pytest.raises(ValueError, match="existing result"):
        runner.worker_authorized(args, "execution", "declaration")


def test_structural_preflight_can_omit_ground_spectra(monkeypatch):
    case = toy_config()["cases"][0]
    h, generators, state = runner.build_case(case)
    def no_spectrum(*a, **k): pytest.fail("structural check computed a ground spectrum")
    monkeypatch.setattr(np.linalg, "eigvalsh", no_spectrum)
    checked = runner.assessment_data(case, h, generators, state, include_spectrum=False)
    assert checked["ground_energy"] is None


def test_exclusive_result_write_never_overwrites_an_existing_record(tmp_path):
    path = tmp_path/"result.json"
    runner.atomic_json(path, {"first": True}, exclusive=True)
    before = path.read_bytes()
    with pytest.raises(FileExistsError): runner.atomic_json(path, {"second": True}, exclusive=True)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_compressed_result_is_deterministic_and_exclusive(tmp_path):
    from benchmarks import check_cross_domain_ratio as checker
    value = {"toy": [1, .2], "metadata": {"unconditional": True}}
    first, second = tmp_path/"one.json.gz", tmp_path/"two.json.gz"
    runner.atomic_json(first, value, exclusive=True)
    runner.atomic_json(second, value, exclusive=True)
    assert first.read_bytes() == second.read_bytes()
    assert runner.load_record(first) == checker.load_record(first) == value
    with pytest.raises(FileExistsError):
        runner.atomic_json(first, {"replacement": 1}, exclusive=True)
    assert runner.load_record(first) == value


def test_setup_resource_failure_is_checkpointed_before_worker_returns(tmp_path):
    c = toy_config(); c["owned_array_cap_bytes"] = 1
    checkpoint = tmp_path/"failed-worker.json"
    result = runner.run_case(c, 0, checkpoint=checkpoint)
    assert checkpoint.exists()
    assert json.loads(checkpoint.read_text()) == result
    assert all(r["status"] == "resource_failed" for r in result["rows"])
