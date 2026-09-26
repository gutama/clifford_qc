"""The Phase 16A producer and result checker, exercised without the experiment.

Nothing here samples a declared instance. The end-to-end tests run the whole
pipeline on the 2-site Hubbard dimer, a system outside the experiment, with
the frozen rules and a reduced grid. The decision rule is tested on synthetic
cells, and the checker must re-derive the producer's decision independently.
Each tampered record must be caught by the check written for it.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from benchmarks import check_phase16a_preregistration as gate
from benchmarks import check_phase16a_te_qsci as checker
from benchmarks import run_phase16a_te_qsci as producer
from clifford_qc.models.lattice import hubbard

pytest.importorskip("scipy")


def _build(_spec):
    return hubbard(2)


@pytest.fixture(scope="module")
def toy():
    """The frozen config, pointed at the dimer with a reduced grid."""
    config = copy.deepcopy(gate.load_config())
    spec = {"description": "hubbard(2): outside the experiment", "builder": "test",
            "role": "required_decision"}
    config["instances"] = {"toy": spec}
    config["required_decision_instances"] = ["toy"]
    config["diagnostic_instances"] = []
    config["seeds"]["instance_order"] = ["toy"]
    config["shot_grid"]["total_shots"] = [16, 64]
    config["admission_criterion"]["top_budget"] = 64
    config["replicas"] = 4
    measured = gate.structural_quantities(config, spec, model=_build(spec))
    keys = gate._INT_FIELDS + gate._LIST_INT_FIELDS + gate._FLOAT_FIELDS + ("times",)
    config["measured_before_freezing"] = {"toy": {k: measured[k] for k in keys}}
    raw = json.dumps(config).encode()
    record = producer.run_experiment(config, config_bytes=raw, build=_build)
    return config, raw, record


def _all_problems(config, raw, record, *, replay=True):
    problems = (checker.declaration_problems(config, record, raw)
                + checker.completeness_problems(config, record))
    if problems:
        return problems
    problems += checker.freeze_problems(config, record)
    problems += checker.statistic_problems(config, record)
    problems += checker.control_problems(config, record, build=_build)
    if replay:
        problems += checker.replay_problems(config, record, build=_build)
    return problems


# ------------------------------------------------------------------ rules

def test_seed_rule_is_the_declared_formula_and_separates_every_cell():
    config = gate.load_config()
    seen = set()
    for instance in config["seeds"]["instance_order"]:
        for arm in producer.SAMPLED_ARMS:
            for budget in range(3):
                for replica in range(3):
                    seed = producer.seed_for(config, instance, arm, budget, replica)
                    key = (config["seeds"]["instance_order"].index(instance),
                           config["seeds"]["arm_indices"][arm], budget, replica)
                    expected = int(np.random.SeedSequence(
                        config["seeds"]["root"], spawn_key=key).generate_state(1)[0])
                    assert seed == expected
                    seen.add(seed)
    assert len(seen) == 3 * 3 * 3 * 3


def _cells(errors_by_budget, control_errors_by_budget):
    return {str(total): {"errors": list(errors),
                         "iterated_control": {"errors": list(control)}}
            for total, errors, control in zip(
                (16, 64, 256), errors_by_budget, control_errors_by_budget)}


@pytest.mark.parametrize("errors, controls, status, reached", [
    ([[3e-3] * 3, [1e-3] * 3, [1e-4] * 3], [[0] * 3, [2e-3] * 3, [0] * 3], "PASS", 64),
    ([[3e-3] * 3, [1e-3] * 3, [1e-4] * 3], [[0] * 3, [5e-4] * 3, [0] * 3], "FAIL", 64),
    ([[1e-3] * 3, [1e-4] * 3, [0] * 3], [[1e-3] * 3, [0] * 3, [0] * 3], "FAIL", 16),
    ([[3e-3] * 3, [2e-3] * 3, [1.7e-3] * 3], [[0] * 3] * 3, "UNDETERMINED", None),
])
def test_the_rule_reads_the_first_budget_that_reaches_the_target(errors, controls,
                                                                 status, reached):
    config = gate.load_config()
    cells = _cells(errors, controls)
    decision = producer.evaluate(config, cells, {"ok": True})
    derived = checker.derive_status(config, cells, {"ok": True})
    assert (decision["status"], decision["shots_to_target"]) == (status, reached)
    assert derived[:2] == (status, reached)


def test_producer_and_checker_agree_on_random_cells():
    config = gate.load_config()
    rng = np.random.default_rng(7)
    for _ in range(200):
        errors = rng.choice([0.0, 5e-4, 1.5e-3, 1.7e-3, 4e-3], size=(3, 5))
        controls = rng.choice([0.0, 1e-4, 1.5e-3, 1e-2], size=(3, 5))
        cells = _cells(errors, controls)
        decision = producer.evaluate(config, cells, {"ok": True})
        status, reached, median = checker.derive_status(config, cells, {"ok": True})
        assert decision["status"] == status
        assert decision["shots_to_target"] == reached
        assert checker._close(decision["median_paired_difference"], median)
    assert producer.evaluate(config, cells, {"ok": False})["status"] == "INVALID"


def test_verdicts_come_from_required_statuses_only():
    assert checker.derive_verdict(["PASS", "PASS"]) == "GO"
    assert checker.derive_verdict(["FAIL", "FAIL"]) == "NO_GO"
    assert checker.derive_verdict(["PASS", "UNDETERMINED"]) == "CONDITIONAL"
    assert checker.derive_verdict(["PASS", "INVALID"]) == "INVALID"
    for statuses in (["PASS", "PASS"], ["FAIL", "PASS"], ["INVALID", "FAIL"]):
        assert checker.derive_verdict(statuses) == gate.verdict_of(statuses)


# ------------------------------------------------------------------ end to end

def test_the_toy_run_re_derives_and_replays_cleanly(toy):
    config, raw, record = toy
    assert record["schema"] == producer.SCHEMA
    assert record["quantum_advantage_claim"] is False
    entry = record["instances"]["toy"]
    assert all(entry["deterministic_checks"].values())
    assert entry["decision"]["status"] in checker.STATUSES
    assert record["decision"]["verdict"] in {"GO", "NO_GO", "CONDITIONAL", "INVALID"}
    assert _all_problems(config, raw, record) == []
    assert checker.replay_problems(config, record, replicas=None, build=_build) == []


def _tamper(record, edit):
    broken = copy.deepcopy(record)
    edit(broken)
    return broken


def _cell(record, arm=producer.CANDIDATE, total="64"):
    return record["instances"]["toy"]["arms"][arm]["budgets"][total]


TAMPERS = {
    "a stored median": (
        lambda r: _cell(r).__setitem__("median_error", 1.0), "median_error drifted"),
    "a stored verdict": (
        lambda r: r["decision"].__setitem__("verdict", "GO" if r["decision"]["verdict"]
                                            != "GO" else "NO_GO"), "recorded verdict"),
    "a stored status": (
        lambda r: r["instances"]["toy"]["decision"].__setitem__("status", "PASS")
        if r["instances"]["toy"]["decision"]["status"] != "PASS"
        else r["instances"]["toy"]["decision"].__setitem__("status", "FAIL"),
        "recorded status"),
    "a seed": (lambda r: _cell(r)["seeds"].__setitem__(0, 1), "declared streams"),
    "a dropped replica": (lambda r: _cell(r)["errors"].pop(), "entries"),
    "an inherited claim boundary": (
        lambda r: r.__setitem__("claim_boundary",
                                r["preregistration"]["config_claim_boundary_at_landing"]),
        "inherits the config's claim boundary"),
    "an advantage claim": (
        lambda r: r.__setitem__("quantum_advantage_claim", True), "quantum_advantage_claim"),
    "a control error": (
        lambda r: _cell(r)["iterated_control"]["errors"].__setitem__(0, -1.0),
        "recomputes to"),
    "a failed check left unenforced": (
        lambda r: r["instances"]["toy"]["deterministic_checks"].__setitem__(
            "trotter_steps_match_frozen", False), "did not make the instance INVALID"),
    "a diagnostic in the verdict": (
        lambda r: r["decision"]["instance_statuses"].__setitem__("h4_r09", "PASS"),
        "instance statuses"),
    "a frozen number moved": (
        lambda r: r["instances"]["toy"]["measured_at_execution"].__setitem__(
            "trotter_steps", [9, 9, 9, 9]), "not the frozen"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_each_tamper_is_caught(toy, name):
    config, raw, record = toy
    edit, expected = TAMPERS[name]
    problems = _all_problems(config, raw, _tamper(record, edit), replay=False)
    assert any(expected in problem for problem in problems), problems


def test_a_sampled_value_that_was_not_drawn_fails_the_replay(toy):
    config, raw, record = toy
    broken = _tamper(record, lambda r: _cell(r, "exact_ground_oracle", "16")[
        "unique_configurations"].__setitem__(0, 99))
    assert any("replays to" in p
               for p in checker.replay_problems(config, broken, build=_build))


def test_a_changed_config_breaks_the_digest(toy):
    config, raw, record = toy
    assert any("digest" in problem for problem in checker.declaration_problems(
        config, record, raw + b" "))


# ------------------------------------------------------------------ refusals

def test_the_producer_refuses_a_reduced_run_on_the_committed_path(capsys):
    assert producer.main(["--replicas", "2"]) == 1
    assert "reduced run" in capsys.readouterr().out
    assert producer.main(["--instances", "beh2_r13264"]) == 1


def test_the_producer_refuses_to_overwrite_a_record(tmp_path, monkeypatch, capsys):
    existing = tmp_path / "record.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(producer, "RECORD", existing)
    assert producer.main(["--out", str(existing)]) == 1
    assert "runs once" in capsys.readouterr().out


def test_the_checker_fails_without_a_record(tmp_path, capsys):
    assert checker.main(["--record", str(tmp_path / "absent.json")]) == 1
    assert "missing record" in capsys.readouterr().out
