"""Independent science/accounting controls; sampled fixtures are undeclared toys."""
from copy import deepcopy
import hashlib
import json
import os

import numpy as np
import pytest

from benchmarks import check_cross_domain_ratio as checker


@pytest.fixture(scope="module")
def toy_observations(tmp_path_factory):
    from benchmarks import run_cross_domain_ratio as runner

    config = deepcopy(runner.load_declaration())
    config.update(total_shot_budgets=[8192], replicas=2, master_seed=8877,
                  process_rss_cap_bytes=2**60)
    config["cases"] = [{"name": "undeclared_two_qubit_checker", "kind": "tfim",
                        "role": "development_primary", "n": 2, "unit": "t",
                        "target_estimation_abs_error": .05, "J": .8, "h": .9,
                        "theta": .7, "phi": .17}]
    directory = tmp_path_factory.mktemp("cross-domain-checker")
    checkpoint = directory/"worker.json"
    observed = runner.run_case(config, 0, checkpoint=checkpoint)
    require_rows = [row for row in observed["rows"] if row["status"] in ("resolved", "denominator_unresolved")]
    assert len(require_rows) == len(observed["rows"])
    assert any(row["status"] == "resolved" for row in observed["rows"])
    record = {"schema": "clifford_qc.cross_domain_ratio_results.v1",
              "declaration_commit": "1"*40, "execution_commit": "2"*40,
              "declaration_sha256": "3"*64,
              "source_sha256": {},
              "inputs": [observed["input"]], "rows": observed["rows"],
              "summaries": runner.summarize(observed["rows"], config),
              "verdict": runner.verdict(runner.summarize(observed["rows"], config), config),
              "worker_receipts": [{"case": config["cases"][0]["name"], "returncode": 0,
                                   "completed": True, "retained_observations": len(observed["rows"]),
                                   "observed_outcomes": len(observed["rows"]),
                                   "scheduled_outcomes": len(observed["rows"]), "imputed_failures": 0,
                                   "case_timeout_seconds": config["case_timeout_seconds"], "timed_out": False,
                                   "source_manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
                                   "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                   "worker_seconds": 1., "worker_pid": os.getpid(),
                                   "execution_commit": "2"*40, "declaration_commit": "1"*40,
                                   "declaration_sha256": "3"*64}]}
    return config, record


def install_toy_contract(monkeypatch, tmp_path, config):
    """Exercise semantic checking only; strict historical checks have own tests."""
    path = tmp_path/checker.DECLARATION
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    # The real declaration has a fixed three-case grid. Only this unit fixture
    # bypasses that policy/history to exercise all record semantics cheaply.
    monkeypatch.setattr(checker, "validate_declaration", lambda value: None)
    monkeypatch.setattr(checker, "check_history", lambda record, value: None)


def test_independent_checker_accepts_toy_joint_observations_without_sampling_or_rss(
        toy_observations, monkeypatch, tmp_path):
    import resource
    from benchmarks import run_cross_domain_ratio as runner

    config, record = toy_observations
    install_toy_contract(monkeypatch, tmp_path, config)
    monkeypatch.setattr(resource, "getrusage", lambda *args: pytest.fail("validator applied runtime RSS"))
    monkeypatch.setattr(runner, "setup", lambda *args: pytest.fail("validator executed producer setup"))
    monkeypatch.setattr(runner.JointShotSimulator, "draw", lambda *args: pytest.fail("validator redrew outcomes"))
    assert checker.check(record) == 4


@pytest.mark.parametrize("mutation", ["shots", "unit", "target", "coefficient", "truth", "interval",
                                      "covariance", "summary", "verdict", "missing", "worker", "rss",
                                      "joint_moments", "pilot_pencil", "denominator"])
def test_independent_checker_rejects_semantic_tampering(toy_observations, monkeypatch, tmp_path, mutation):
    config, original = toy_observations
    install_toy_contract(monkeypatch, tmp_path, config)
    record = deepcopy(original)
    row = next(row for row in record["rows"] if row["status"] == "resolved")
    if mutation == "shots": row["total_shots"] += 1
    elif mutation == "unit": row["unit"] = "Ha"
    elif mutation == "target": row["target_estimation_abs_error"] = 1.
    elif mutation == "coefficient": row["coefficients"][0][0] += .1
    elif mutation == "truth": row["true_direction_energy"] += .1
    elif mutation == "interval": row["numerator_interval"] = [row["numerator_interval"][0]-.1, row["numerator_interval"][1]]
    elif mutation == "covariance": row["numerator_denominator_covariance"] += .1
    elif mutation == "summary": record["summaries"][0]["successes"] += 1
    elif mutation == "verdict": record["verdict"]["status"] = "promote_on_declared_scope"
    elif mutation == "missing": record["rows"].pop()
    elif mutation == "worker": row["worker_pid"] += 1
    elif mutation == "rss": row["process_peak_rss_bytes"] = config["process_rss_cap_bytes"]+1
    elif mutation == "joint_moments": row["evaluation_group_moments"][0]["sum_h"] += 100
    elif mutation == "pilot_pencil": row["pilot_overlap"][0][0][0] += .1
    elif mutation == "denominator": row["denominator_interval"] = [0., row["denominator_interval"][1]]
    with pytest.raises(AssertionError):
        checker.check(record)


@pytest.mark.parametrize("index", [1, 2])
def test_molecular_assessment_is_in_the_declared_singlet_sector(index):
    config = json.loads((checker.ROOT/checker.DECLARATION).read_text())
    case = config["cases"][index]
    data = checker.case_data(case)
    from benchmarks.check_orthogonal_active_space import independent_spin_squared
    spin = independent_spin_squared(data["words"], case["n"])
    np.testing.assert_allclose(spin@data["basis"], 0, atol=1e-12)
    assert len(data["words"]) == 36
    assert all(word.bit_count() == 4 for word in data["words"])
    assert not np.allclose(data["overlap"], np.eye(3))
    provider = checker.structural_provider(case)
    native_h = checker._pauli_restriction(provider.bank.hamiltonian, data["words"])
    np.testing.assert_allclose(native_h, data["hamiltonian"], atol=1e-10)


@pytest.mark.parametrize("mutation", ["unit", "target", "scope", "budget", "resource"])
def test_independent_declaration_validation_rejects_scope_drift(mutation):
    config = json.loads((checker.ROOT/checker.DECLARATION).read_text())
    if mutation == "unit": config["cases"][1]["unit"] = "t"
    elif mutation == "target": config["cases"][1]["target_estimation_abs_error"] = .05
    elif mutation == "scope": config["cases"][1]["target_spin"] = 1
    elif mutation == "budget": config["total_shot_budgets"].append(1048576)
    elif mutation == "resource": config["process_rss_cap_bytes"] *= 2
    with pytest.raises(AssertionError): checker.validate_declaration(config)


def test_failed_attempts_remain_in_denominators_and_prevent_promotion(toy_observations):
    config, original = toy_observations
    rows = deepcopy(original["rows"])
    rows[0].update(status="worker_failed", estimate=None, estimation_error=None)
    summaries = checker.independent_summaries(rows, config)
    first = next(row for row in summaries if row["arm"] == rows[0]["arm"])
    assert first["attempts"] == 2 and first["worker_failures"] == 1
    assert first["success_probability"] <= .5
    assert checker.independent_verdict(summaries, config)["status"] == "closed_on_declared_scope"


def test_unknown_draw_costs_stay_failed_and_unavailable(toy_observations, monkeypatch, tmp_path):
    config, original = toy_observations
    install_toy_contract(monkeypatch, tmp_path, config)
    record = deepcopy(original)
    row = record["rows"][0]
    row.update(status="evaluation_failed", failure_reason="raw draw interrupted", estimate=None,
               physical_shots_unknown=True, costs_complete=False,
               attempted_physical_shots=row["budget"], evaluation_shots=0,
               total_shots=row["pilot_shots"], device_seconds=None, end_to_end_seconds=None)
    record["summaries"] = checker.independent_summaries(record["rows"], config)
    record["verdict"] = checker.independent_verdict(record["summaries"], config)
    assert checker.check(record) == 4
    group = next(item for item in record["summaries"] if item["arm"] == row["arm"])
    assert group["incomplete_costs"] == 1
    assert record["verdict"]["status"] == "closed_on_declared_scope"


def test_incomplete_worker_receipt_keeps_observed_rows_and_unknown_spending(
        toy_observations, monkeypatch, tmp_path):
    config, original = toy_observations
    install_toy_contract(monkeypatch, tmp_path, config)
    record = deepcopy(original)
    row = record["rows"][-1]
    row.update(status="worker_failed", estimate=None, estimation_error=None,
               pilot_shots=0, evaluation_shots=0, total_shots=0,
               physical_shots_unknown=True, costs_complete=False,
               attempted_physical_shots=None, device_seconds=None, end_to_end_seconds=None)
    for key in ("pilot_counts", "evaluation_counts", "evaluation_settings"):
        row.pop(key, None)
    receipt = record["worker_receipts"][0]
    receipt.update(completed=False, returncode=-15, timed_out=True,
                   observed_outcomes=3, retained_observations=3, imputed_failures=1)
    record["summaries"] = checker.independent_summaries(record["rows"], config)
    record["verdict"] = checker.independent_verdict(record["summaries"], config)
    assert checker.check(record) == 4
    receipt["retained_observations"] = 4
    with pytest.raises(AssertionError, match="observation denominator"):
        checker.check(record)


def test_historical_source_coverage_cannot_be_silently_reduced(monkeypatch):
    from types import SimpleNamespace
    config = json.loads((checker.ROOT/checker.DECLARATION).read_text())
    declaration = (checker.ROOT/checker.DECLARATION).read_bytes()
    record = {"declaration_commit": "1"*40, "execution_commit": "2"*40,
              "declaration_sha256": hashlib.sha256(declaration).hexdigest(), "source_sha256": {}}

    def history(*args):
        if args[0] == "rev-parse": return b"false"
        if args[0] == "show": return declaration
        if args[0] == "ls-tree": return b"clifford_qc/multivector.py\n"
        pytest.fail("coverage check failed to reject before fetching other sources")

    monkeypatch.setattr(checker, "git", history)
    monkeypatch.setattr(checker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(AssertionError, match="complete execution source coverage"):
        checker.check_history(record, config)


@pytest.mark.skipif(not (checker.ROOT/checker.RESULT).exists(), reason="declared campaign not executed")
def test_published_cross_domain_sources_and_all_outcomes():
    record = checker.load_record(checker.ROOT/checker.RESULT)
    assert checker.check(record) == 2304


@pytest.mark.skipif(not (checker.ROOT/checker.RESULT).exists(), reason="declared campaign not executed")
@pytest.mark.parametrize("field", ["source_sha256", "input_sha256"])
def test_published_cross_domain_rejects_source_and_input_tampering(field):
    record = checker.load_record(checker.ROOT/checker.RESULT)
    first = next(iter(record[field]))
    record[field][first] = "0"*64
    with pytest.raises(AssertionError): checker.check(record)
