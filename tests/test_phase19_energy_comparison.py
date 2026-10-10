"""Exercise the producer/checker on undeclared toys, never molecular outcomes."""

from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import inspect
import json
import math
import subprocess

import numpy as np
import pytest

from benchmarks import check_phase19_energy_comparison as checker
from benchmarks import phase19_structure as structure
from benchmarks import run_phase19_energy_comparison as producer
from benchmarks.check_phase19_preregistration import load_config
from clifford_qc.ir import PauliSum
from clifford_qc.measurement.cliques import compile_clique_measurement_plan, _setting_resources
from clifford_qc.measurement.cost import DeviceCard
from clifford_qc.measurement.grouping import qwc_groups, shared_basis


def toy_bank(labels=None):
    functional = PauliSum.from_labels(labels or {
        "II": -0.25, "XI": -0.6, "YI": 0.2, "ZI": -0.8,
        "IX": 0.1, "XX": 0.3, "ZZ": 0.7})
    coefficients = {word.code: float(value.real) for word, value in functional.items()}
    members = sorted((word for word, _ in functional.items() if word.code), key=lambda word: word.code)
    clique = compile_clique_measurement_plan(functional)
    clique_rows = [{"word_codes": [word.code for word in item.words],
        "max_abs_value": abs(item.weight),
        "rotations": [{"word_code": rotor.word.code, "angle": rotor.angle} for rotor in item.rotations],
        "readout": {"kind": "weighted_parity", "weight": item.weight,
                    "qubits": list(item.readout_qubits)},
        "gate_sequence_sha256": structure._gate_sequence_digest(item.ops),
        "resources": asdict(item.resources)} for item in clique.settings]
    qwc_rows = []
    for group in qwc_groups(members):
        group = sorted(group, key=lambda word: word.code)
        ops = []
        for q, letter in sorted(shared_basis(group).items()):
            if letter == "Y":
                ops.append(("sdg", q))
            if letter in {"X", "Y"}:
                ops.append(("h", q))
        qwc_rows.append({"word_codes": [word.code for word in group],
            "max_abs_value": math.fsum(abs(coefficients[word.code]) for word in group),
            "rotations": [], "readout": {"kind": "weighted_sum_of_parities",
                "word_qubits": [list(word.support()) for word in group]},
            "gate_sequence_sha256": structure._gate_sequence_digest(ops),
            "resources": asdict(_setting_resources(tuple(ops)))})
    return {"id": "undeclared_two_qubit_toy", "n_qubits": functional.n,
        "reference_occupied_spin_orbitals": [1],
        "coefficients_hex": [[code, value.hex()] for code, value in sorted(coefficients.items())],
        "arms": [{"name": "qwc", "settings": qwc_rows},
                 {"name": "clique", "settings": clique_rows}]}


def toy_config():
    config = deepcopy(load_config())
    protocol = config["protocol"]
    protocol["effective_shot_endpoints"] = [128, 8192]
    protocol["accuracy_hartree"] = 0.8
    protocol["confidence"].update(family=2, rounds=2)
    protocol["seeds"].update(headline=7, covariance_audit=11)
    protocol["covariance_audit"].update(replicas=1000, effective_shots_per_replica=512)
    return config


def toy_cards():
    ideal = DeviceCard("toy-ideal", 1, 0.05, 0.3, 1, 1, 0, 0, 0,
                       "all-to-all", False)
    noisy = DeviceCard("toy-inadmissible", 1, 0.05, 0.3, 1, 1, 0.2, 0.2, 0.2,
                       "line", True, routing_2q_multiplier=2.5,
                       routing_depth_multiplier=1.5, fidelity_floor=0.99)
    return [ideal, noisy]


@pytest.fixture(scope="module")
def toy():
    bank, config, cards = toy_bank(), toy_config(), toy_cards()
    cell, built, sampled = producer.run_bank(bank, config, cards, 0)
    record = producer.assemble_record(config, [cell], built, sampled, {})
    return bank, config, cards, record


def test_full_independent_toy_replay(toy):
    bank, config, cards, record = toy
    assert record["verdict"] == "VALID"
    assert checker.check_record(record, config, [bank], cards, check_provenance=False) == []
    assert all(len(arm["endpoints"]) == 2 for arm in record["banks"][0]["arms"])
    assert all(arm["device_reports"][1]["status"] == "inadmissible"
               for arm in record["banks"][0]["arms"])
    assert all(arm["device_reports"][1]["C_time_epsilon_us"] is None
               for arm in record["banks"][0]["arms"])
    assert record["accounting"]["headline_effective_shots"] == 2 * (128 + 8192)
    assert record["accounting"]["audit_effective_shots"] == 2 * 512 * 1000


@pytest.mark.parametrize("mutation", [
    "histogram", "mean", "variance", "radius", "schedule", "seed_digest",
    "replica", "law", "decision", "censoring", "price", "identity", "extra_key",
    "bool_count", "numerical_allowance", "audit_variance", "declaration", "accounting",
])
def test_checker_rejects_corrupted_record(toy, mutation):
    bank, config, cards, original = toy
    record = deepcopy(original)
    cell = record["banks"][0]
    arm = cell["arms"][0]
    endpoint = arm["endpoints"][0]
    if mutation == "histogram":
        endpoint["histograms"][0][0][1] += 1
    elif mutation == "bool_count":
        endpoint["histograms"][0][0][1] = True
    elif mutation == "mean":
        endpoint["setting_means"][0] += 0.1
    elif mutation == "variance":
        endpoint["setting_sample_variances"][0] += 0.1
    elif mutation == "radius":
        endpoint["total_radius"] = 0.0
    elif mutation == "numerical_allowance":
        endpoint["total_radius"] += 0.1
    elif mutation == "schedule":
        endpoint["shots"][0] += 1
    elif mutation == "seed_digest":
        arm["covariance_audit"]["histograms_sha256"] = "0" * 64
    elif mutation == "replica":
        arm["covariance_audit"]["estimates"][0] += 0.1
    elif mutation == "audit_variance":
        arm["covariance_audit"]["predicted_variance"] += 0.1
    elif mutation == "law":
        arm["laws"][0]["raw_probabilities"][0] += 0.1
    elif mutation == "decision":
        record["shot_decisions"] = ["invented"]
    elif mutation == "censoring":
        arm["right_censored"] = not arm["right_censored"]
    elif mutation == "price":
        arm["device_reports"][0]["total_raw_shots"] += 1
    elif mutation == "identity":
        cell["structural_bank"]["coefficients_hex"][0][1] = "0x1.0p+0"
    elif mutation == "extra_key":
        arm["covariance_audit"]["hidden_result"] = 1
    elif mutation == "declaration":
        record["declaration_sha256"] = "0" * 64
    elif mutation == "accounting":
        record["accounting"]["audit_effective_shots"] = True
    try:
        problems = checker.check_record(record, config, [bank], cards, check_provenance=False)
    except (ValueError, TypeError):
        return
    assert problems


def test_qwc_joint_histogram_retains_covariance():
    # Two identical +/-1 word readouts: Var(P+P)=4, not Var(P)+Var(P)=2.
    histogram, values = [[0, 50], [3, 50]], np.array([2.0, 0.0, 0.0, -2.0])
    mean, variance = producer.sample_statistics(histogram, values, 100)
    assert mean == 0 and variance == pytest.approx(400 / 99)
    assert checker.histogram_statistics(histogram, values, 100) == (mean, variance)


def test_centering_and_exact_zero_variance():
    histogram, values = [[0, 100], [1, 101]], np.array([1e10, 1e10 + 0.125])
    actual = producer.sample_statistics(histogram, values, 201)
    expected = checker.histogram_statistics(histogram, values, 201)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert producer.sample_statistics([[0, 500]], values, 500) == (1e10, 0.0)


def test_rational_reference_and_qubit_order():
    bank = toy_bank({"II": 0.25, "ZI": 0.5, "IZ": 0.125, "XX": 0.2})
    assert checker.reference_index(bank) == 1  # |01>, q0 leftmost
    rational = checker.rational_reference(bank)
    assert Fraction(rational["numerator"], rational["denominator"]) == Fraction(5, 8)
    ops = [("h", 0), ("sdg", 1), ("cx", 0, 1), ("rz", -0.72, 0), ("h", 1)]
    state = producer.statevector_circuit(2, ops, 1)
    np.testing.assert_allclose(state, checker.dense_unitary(2, ops)[:, 1], atol=1e-15)


def test_probability_policy_preserves_tiny_nonzero_mass():
    law, correction = checker.corrected_probabilities([1 - 1e-14, 1e-14, -1e-16], "frozen")
    assert law[1] > 0 and law[2] == 0
    assert correction["clipped_mass"] == 1e-16
    with pytest.raises(ValueError):
        checker.corrected_probabilities([0.5, 0.6], "frozen")
    with pytest.raises(ValueError):
        checker.corrected_probabilities([1.0 + 2e-12, -2e-12], "frozen")


def test_multinomial_large_budget_without_per_shot_arrays():
    histogram = producer.sample_histogram([0.25, 0.75], 8, (0, 0, 0, 0), 1 << 32)
    assert sum(count for _, count in histogram) == 1 << 32
    assert histogram == checker.seeded_histogram([0.25, 0.75], 8, (0, 0, 0, 0), 1 << 32)


def test_zero_prediction_campaign_and_right_censoring():
    bank, config, cards = toy_bank({"II": 0.2, "ZI": -0.5, "IZ": 0.1}), toy_config(), toy_cards()
    config["protocol"]["accuracy_hartree"] = 1e-15
    cell, built, sampled = producer.run_bank(bank, config, cards, 0)
    for arm in cell["arms"]:
        audit = arm["covariance_audit"]
        assert audit["predicted_variance"] == audit["empirical_variance"] == 0
        assert audit["ratio"] is None and audit["passed"]
        assert arm["right_censored"]
        assert arm["device_reports"][0]["status"] == "right_censored"
    record = producer.assemble_record(config, [cell], built, sampled, {})
    assert record["shot_decisions"] == ["INDETERMINATE"]
    assert checker.check_record(record, config, [bank], cards, check_provenance=False) == []


def test_failed_audit_invalidates_entire_campaign():
    bank, config, cards = toy_bank(), toy_config(), toy_cards()
    config["protocol"]["covariance_audit"]["acceptable_empirical_to_predicted_interval"] = [2.0, 3.0]
    cell, built, sampled = producer.run_bank(bank, config, cards, 0)
    record = producer.assemble_record(config, [cell], built, sampled, {})
    assert record["verdict"] == "INVALID" and record["shot_decisions"] == [None]
    assert cell["arms"][0]["covariance_audit"]["estimates"]
    assert checker.check_record(record, config, [bank], cards, check_provenance=False) == []


@pytest.mark.parametrize("failure", ["normalization", "nonfinite"])
def test_failed_probability_gate_keeps_diagnostics_without_sampling(monkeypatch, failure):
    bank, config, cards = toy_bank(), toy_config(), toy_cards()
    original = producer.statevector_circuit
    def broken(*args):
        state = original(*args)
        return state * (2 if failure == "normalization" else float("nan"))
    monkeypatch.setattr(producer, "statevector_circuit", broken)
    monkeypatch.setattr(producer, "sample_histogram", lambda *args: pytest.fail("blocked laws sampled"))
    cell, built, sampled = producer.run_bank(bank, config, cards, 0)
    record = producer.assemble_record(config, [cell], built, sampled, {})
    assert sampled == 0 and record["verdict"] == "INVALID"
    assert all(arm["sampling_status"] == "blocked" for arm in cell["arms"])
    assert all(arm["laws"][0]["failure"] for arm in cell["arms"])
    json.dumps(record, allow_nan=False)
    assert checker.check_record(record, config, [bank], cards, check_provenance=False) == []


def test_operator_failure_retains_invalid_cell_without_sampling(monkeypatch):
    bank, config, cards = toy_bank(), toy_config(), toy_cards()
    # A signed readout corruption leaves the gate topology intact, but fails
    # full operator reconstruction before a single RNG stream is touched.
    bank["arms"][1]["settings"][0]["readout"]["weight"] *= -1
    monkeypatch.setattr(producer, "sample_histogram", lambda *args: pytest.fail("invalid operator sampled"))
    cell, built, sampled = producer.run_bank(bank, config, cards, 0)
    assert cell["blocking_failures"] and cell["arms"] == [] and sampled == 0
    record = producer.assemble_record(config, [cell], built, sampled, {})
    assert checker.check_record(record, config, [bank], cards, check_provenance=False) == []


@pytest.mark.parametrize("qwc,clique,expected", [
    (8192, 128, "MATERIAL_REDUCTION"), (128, 128, "NO_MATERIAL_REDUCTION"),
    (None, 128, "MATERIAL_REDUCTION_CENSORED_BASELINE"),
    (None, 8192, "INDETERMINATE"), (128, None, "NO_MATERIAL_REDUCTION"),
    (None, None, "INDETERMINATE")])
def test_censoring_and_materiality_rules(qwc, clique, expected):
    assert producer.shot_decision(qwc, clique, toy_config()) == expected
    assert checker.decision(qwc, clique, toy_config()) == expected


def test_execution_claim_cannot_be_replaced(tmp_path):
    path = tmp_path / "claim.json"
    producer.claim_execution(path, {"git_commit": "first"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        producer.claim_execution(path, {"git_commit": "replacement"})
    assert path.read_bytes() == original


def test_cli_race_preserves_another_execution_claim(tmp_path, monkeypatch):
    claim = tmp_path / "claim.json"
    producer.claim_execution(claim, {"git_commit": "other"})
    original = claim.read_bytes()
    monkeypatch.setattr(producer, "execution_preflight", lambda: (load_config(), {}, claim))
    monkeypatch.setattr(producer, "stamp_record", lambda record, p: {"provenance": p})
    assert producer.main(["--execute"]) == 1
    assert claim.read_bytes() == original


def test_fatal_cli_failure_retains_consumed_claim(tmp_path, monkeypatch):
    claim = tmp_path / "claim.json"
    p = {"git_commit": "pinned-source"}
    monkeypatch.setattr(producer, "execution_preflight", lambda: (load_config(), p, claim))
    monkeypatch.setattr(producer, "stamp_record", lambda record, p: {"provenance": p})
    def fail(*args):
        raise ValueError("injected toy failure before molecular evaluation")
    monkeypatch.setattr(producer, "run_bank", fail)
    assert producer.main(["--execute"]) == 1
    content = json.loads(claim.read_text())
    assert content["status"] == "failed" and content["provenance"] == p
    assert content["completed_banks"] == []
    with pytest.raises(FileExistsError):
        producer.claim_execution(claim, p)


def test_default_commands_never_evaluate_molecular_outcomes(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("molecular state laws must not run during default preflight")
    monkeypatch.setattr(checker, "dense_oracle", forbidden)
    monkeypatch.setattr(producer, "run_bank", forbidden)
    # Pin the record-absent state: once the campaign record lands, the default
    # checker command replays it by design.
    monkeypatch.setattr(checker, "RECORD", tmp_path / "absent.json")
    assert producer.main([]) == 0
    assert checker.main() == 0


def test_checker_has_no_producer_import_or_calculation_call():
    source = inspect.getsource(checker)
    assert "from benchmarks.run_phase19" not in source
    assert "import run_phase19" not in source


def test_dirty_execution_refused_before_sampling():
    # This is a feature worktree under development; pin the guard via a fake git
    # runner rather than depending on the user's actual working-tree state or on
    # clone depth (CI checks out shallow, which preflight refuses first).
    original = subprocess.run
    def git_run(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "--is-shallow-repository"]:
            return subprocess.CompletedProcess(command, 0, "false\n", "")
        if command[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, " M implementation.py\n", "")
        return original(command, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "run", git_run)
        with pytest.raises(ValueError, match="clean committed tree"):
            producer.execution_preflight()


def test_environment_requires_frozen_sampling_version():
    reference = {"python": "3.12.3", "dependencies": {"numpy": "2.5.2", "scipy": "1.18.0"}}
    p = {"python": "3.12.14", "dependencies": {"numpy": "2.5.2", "scipy": "1.18.0"}}
    assert checker.environment_problems(p, reference) == []
    p["dependencies"]["numpy"] = "2.3.5"
    assert checker.environment_problems(p, reference)


def test_execution_source_and_record_commit_order(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True,
                              text=True, capture_output=True).stdout.strip()
    def commit(message):
        git("add", ".")
        git("commit", "-m", message)
        return git("rev-parse", "HEAD")
    git("init", "-b", "main")
    git("config", "user.name", "Phase 19 toy")
    git("config", "user.email", "phase19@example.invalid")
    config = load_config()
    declaration = tmp_path / structure.CONFIG_PATH
    declaration.parent.mkdir(parents=True)
    declaration.write_text(json.dumps(config))
    p = {"python": "3.12.14", "dependencies": {"numpy": "2.5.2"}}
    environment = tmp_path / config["lineage"]["environment_reference"]
    environment.parent.mkdir(parents=True)
    environment.write_text(json.dumps({"provenance": p}))
    commit("declare toy history")
    git("switch", "-c", "declaration-branch")
    (tmp_path / "merge-marker").write_text("toy")
    commit("declaration branch marker")
    git("switch", "main")
    git("merge", "--no-ff", "declaration-branch", "-m", "merge declaration")
    merge = git("rev-parse", "HEAD")
    monkeypatch.setattr(checker, "MERGE", merge)
    for relative in checker.SOURCES:
        source = tmp_path / relative
        source.write_text("# toy execution source\n")
    execution = commit("land source separately")
    p.update(git_dirty=False, git_sha=execution, git_commit=execution,
             preregistration_merge_commit=merge,
             source_files=[structure.file_binding(path, tmp_path) for path in checker.SOURCES])
    record = {"declaration_sha256": checker.EXPECTED_CONFIG_SHA256, "provenance": p}
    (tmp_path / structure.RECORD_PATH).write_text(json.dumps(record))
    commit("record alone")
    assert checker.provenance_problems(record, config, tmp_path) == []
    corrupted = deepcopy(record)
    corrupted["provenance"]["git_dirty"] = True
    assert checker.provenance_problems(corrupted, config, tmp_path)
    corrupted = deepcopy(record)
    corrupted["provenance"]["source_files"][0]["sha256"] = "0" * 64
    assert checker.provenance_problems(corrupted, config, tmp_path)
    (tmp_path / checker.SOURCES[0]).write_text("# changed execution source\n")
    assert checker.provenance_problems(record, config, tmp_path)
    (tmp_path / checker.SOURCES[0]).write_text("# toy execution source\n")
    (tmp_path / "extra-record-change").write_text("must not share the first record commit")
    git("add", ".")
    git("commit", "--amend", "--no-edit")
    assert any("committed alone" in problem
               for problem in checker.provenance_problems(record, config, tmp_path))
