"""Q18-S1 correctness on undeclared toys; no new declared-bank result here."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import check_phase15_measured_residual_preregistration as original
from benchmarks import check_phase15_preregistration as preflight
from benchmarks import check_phase15_residual_successor as checker
from benchmarks import check_phase15_residual_successor_declaration as declaration
from benchmarks import run_phase15_residual_successor as producer
from benchmarks.phase15_residual_successor import (
    derivative_check, integer_oracle_allocation, summarize_derivative,
)
from benchmarks.run_mapping_axis import _raw_pool
from benchmarks.run_phase15_h2_preflight import _digest
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import SecondMomentBank
from clifford_qc.subspace.generators import identity_generator


@pytest.fixture(scope="module")
def toy():
    model = hubbard(2)
    selected = [identity_generator(model.n), *_raw_pool(model)[:2]]
    config = copy.deepcopy(declaration.load_config())
    # A fresh toy execution has no post-execution metadata correction.
    config["revisions"] = config["revisions"][:1]
    bank = preflight.first_moment_bank(model, selected)
    second = SecondMomentBank(bank)
    root = second.residual(bank.solve(), 0)
    sh, combined = set(bank.word_set()), set(bank.word_set()) | set(second.word_set())
    committed = {
        "validation_energy": root.energy, "variance": root.variance,
        "cancellation_scale": root.cancellation_scale, "variance_resolved": root.resolved,
        "preflight_sh_word_universe": len(sh), "ledger_sh_word_universe": len(sh),
        "raw_combined_word_universe": len(combined), "raw_combined_words_sha256": _digest(combined),
        "mapping_protocol": "qwc_groups",
        "mapping_settings": len(original.group_words(sh, "qwc_groups", model.n)),
        "mapping_energy_variance": None,
    }
    config["banks"]["systems"] = ["toy"]
    config["grouping"]["declared_protocol_by_system"] = {"toy": "qwc_groups"}
    config["grouping"]["alternative_protocol_by_system"] = {"toy": "qwc_basis_cover"}
    config["measured_before_freezing"] = {
        "toy": original.quantities_for(model, selected, committed, "qwc_groups", "qwc_basis_cover")}
    rows = original.block_rows(bank, second, list(range(len(selected))))
    means = original.reference_means(bank.reference, combined)
    weights, variance = original.residual_functional(rows, len(selected), means)
    validation = {"banks": {"toy": {"roots": [root.as_dict()]}}}
    raw = json.dumps(config).encode()
    record = producer.run_successor(config, config_bytes=raw, inputs=lambda name: (model, selected),
                                    validation=validation)
    record["provenance"] = {"git_dirty": False}
    return {"model": model, "selected": selected, "bank": bank, "rows": rows,
            "means": means, "weights": weights, "variance": variance,
            "scale": root.cancellation_scale, "config": config, "raw": raw,
            "record": record, "validation": validation}


def test_toy_record_rederives_and_rebuilds(toy):
    assert checker.rederivation_problems(toy["config"], toy["record"], toy["raw"],
                                        toy["validation"]) == []
    assert checker.recompute_problems(toy["config"], toy["record"], toy["validation"],
                                     inputs=lambda name: (toy["model"], toy["selected"])) == []
    assert toy["record"]["successor"]["design_status"] == "post_hoc"
    assert toy["record"]["banks"]["toy"]["deterministic_checks"][
        "linearization_matches_finite_differences"]
    json.dumps(toy["record"], allow_nan=False)


def test_derivative_matches_independent_dense_density_perturbation(toy):
    scipy = pytest.importorskip("scipy.linalg")
    bank, model = toy["bank"], toy["model"]
    generators = [to_matrix(generator.mv) for generator in toy["selected"]]
    H, rho = to_matrix(bank.hamiltonian), to_matrix(bank.reference)
    direction = original.check_directions(toy["weights"], bank.word_set(), seed=20261003)[
        "sh_gaussian"]
    from clifford_qc.multivector import MV
    delta = sum((value * to_matrix(MV(model.n, {word: 1.0}))
                 for word, value in direction.items()), np.zeros_like(rho)) / (2 ** model.n)

    def dense_value(t):
        state = rho + t * delta
        matrices = [np.array([[np.trace(A.conj().T @ observable @ B @ state)
                               for B in generators] for A in generators])
                    for observable in (np.eye(H.shape[0]), H, H @ H)]
        S, projected, K = matrices
        values, vectors = scipy.eigh(projected, S)
        c = vectors[:, 0]
        return float((c.conj() @ K @ c).real) - values[0] ** 2

    h = 0.00025
    central = lambda h: (dense_value(h) - dense_value(-h)) / (2 * h)
    dense = (4 * central(h) - central(2 * h)) / 3
    checked = derivative_check(toy["rows"], len(generators), toy["means"], toy["weights"],
                               direction, rule=toy["config"]["derivative_validation"],
                               cancellation_scale=toy["scale"])
    assert checked["passes"]
    assert checked["numeric"] == pytest.approx(dense, rel=1e-7, abs=1e-10)


def test_dropped_ritz_vector_response_is_rejected(toy):
    from clifford_qc.subspace.linalg import solve_projected
    S, H, K = original.projected_matrices(toy["rows"], len(toy["selected"]), toy["means"])
    solve = solve_projected(S, H)
    c, energy = solve.coefficients[:, 0], float(solve.energies[0])
    variance = float((c.conj() @ K @ c).real) - energy ** 2
    naive = {}
    for (a, b), (s, h, q) in toy["rows"].items():
        for word in set(s) | set(h) | set(q):
            value = np.conj(c[a]) * c[b] * (q.get(word, 0) - 2 * energy * h.get(word, 0)
                                            + (energy ** 2 - variance) * s.get(word, 0))
            naive[word] = naive.get(word, 0) + (value.real if a == b else 2 * value.real)
    direction = original.check_directions(naive, toy["bank"].word_set(), seed=20261003)[
        "sh_gaussian"]
    checked = derivative_check(toy["rows"], len(toy["selected"]), toy["means"], naive,
                               direction, rule=toy["config"]["derivative_validation"],
                               cancellation_scale=toy["scale"])
    assert not checked["passes"]


def _polynomial_evaluations(rule):
    f = lambda x: 3 * x + 1000 * x ** 3
    return [{"step": h, "plus": f(h), "minus": f(-h),
             "numeric": (f(h) - f(-h)) / (2 * h), "domain_valid": True}
            for h in rule["steps"]]


def test_richardson_removes_known_cubic_truncation_without_loosening_tolerance():
    rule = declaration.load_config()["derivative_validation"]
    evaluations = _polynomial_evaluations(rule)
    assert abs(evaluations[1]["numeric"] - 3) > 1e-6 * 3
    checked = summarize_derivative(evaluations, 3, 3, 1, rule)
    assert checked["passes"] and checked["numeric"] == pytest.approx(3, abs=1e-12)
    assert checker.derivative_problems(checked, rule, 3, 1) == ([], True)
    wrong = summarize_derivative(evaluations, 3.01, 3.01, 1, rule)
    assert not wrong["passes"]


def test_a_bad_endpoint_or_unstable_pair_is_not_silently_discarded():
    rule = declaration.load_config()["derivative_validation"]
    evaluations = _polynomial_evaluations(rule)
    evaluations[0]["domain_valid"] = False
    assert not summarize_derivative(evaluations, 3, 3, 1, rule)["passes"]
    evaluations[0]["domain_valid"] = True
    evaluations[-2]["numeric"] += 0.01
    checked = summarize_derivative(evaluations, 3, 3, 1, rule)
    assert not checked["stable"] and not checked["passes"]


@pytest.mark.parametrize("variances", [[0, 1, 4], [0, 0], [0.01, 100, 0], []])
def test_integer_allocation_meets_precision_and_charges_every_setting(variances):
    allocation = integer_oracle_allocation(variances, 0.03, pilot_shots=64, minimum_shots=2)
    assert all(n >= 2 and isinstance(n, int) for n in allocation["shots_per_setting"])
    assert allocation["pilot_shots"] == 64 * len(variances)
    assert allocation["total_shots"] == allocation["production_shots"] + allocation["pilot_shots"]
    assert allocation["achieved_variance"] <= 0.03 ** 2 * (1 + 1e-12)
    lower = math.fsum(math.sqrt(v) for v in variances) ** 2 / 0.03 ** 2
    assert allocation["production_shots"] >= lower * (1 - 1e-12)


@pytest.mark.parametrize("variances,error,pilot,floor", [
    ([-1], 0.1, 64, 1), ([float("nan")], 0.1, 64, 1), ([1], 0, 64, 1),
    ([1], float("inf"), 64, 1), ([1], 0.1, True, 1), ([1], 0.1, 64, 0),
])
def test_invalid_allocation_inputs_fail(variances, error, pilot, floor):
    with pytest.raises(ValueError):
        integer_oracle_allocation(variances, error, pilot_shots=pilot, minimum_shots=floor)


def _mutate(record, path, value):
    bad = copy.deepcopy(record)
    node = bad
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return bad


@pytest.mark.parametrize("path,value", [
    (("claim_boundary",), "preregistered success"),
    (("successor", "design_status"), "preregistered"),
    (("config_digest",), "0" * 64),
    (("provenance", "git_dirty"), True),
    (("decision", "verdict"), "FULL"),
    (("banks", "toy", "finite_differences", "functional", "passes"), False),
    (("banks", "toy", "finite_differences", "functional", "numeric"), 123),
    (("banks", "toy", "finite_differences", "functional", "evaluations", 0, "numeric"), 123),
    (("banks", "toy", "protocols", "qwc_groups", "cost_ratio"), 0.1),
    (("banks", "toy", "protocols", "qwc_groups", "allocation_diagnostic", "estimator_licensed"), True),
    (("banks", "toy", "protocols", "qwc_groups", "allocation_diagnostic", "energy", "pilot_shots"), 0),
    (("banks", "toy", "protocols", "qwc_groups", "allocation_diagnostic", "residual", "shots_per_setting", 0), 0),
])
def test_each_forged_summary_is_caught(toy, path, value):
    bad = _mutate(toy["record"], path, value)
    assert checker.rederivation_problems(toy["config"], bad, toy["raw"], toy["validation"])


def test_self_consistent_endpoint_forgery_requires_and_fails_a_rebuild(toy):
    bad = copy.deepcopy(toy["record"])
    row = bad["banks"]["toy"]["finite_differences"]["functional"]
    # Equal shifts preserve every central difference and its outcome.
    for endpoint in row["evaluations"]:
        endpoint["plus"] += 1e-6
        endpoint["minus"] += 1e-6
    assert checker.rederivation_problems(toy["config"], bad, toy["raw"], toy["validation"]) == []
    assert checker.recompute_problems(toy["config"], bad, toy["validation"],
                                     inputs=lambda name: (toy["model"], toy["selected"]))


@pytest.mark.parametrize("path,value", [
    (("successor", "design_status"), "preregistered"),
    (("successor", "predecessor_sha256"), "0" * 64),
    (("statistic", "max_ratio"), 1000000),
    (("grouping", "allocation"), "neyman"),
    (("derivative_validation", "steps"), [0.01, 0.001, 0.0001, 0.00001]),
    (("derivative_validation", "relative_tolerance"), 0.1),
    (("allocation_diagnostic", "decision_role"), "licence"),
    (("allocation_diagnostic", "standard_error_fraction_of_sigma"), 1),
    (("allocation_diagnostic", "minimum_shots_per_setting"), False),
])
def test_declaration_rejects_question_drift_and_evidence_upgrades(path, value):
    config = _mutate(declaration.load_config(), path, value)
    assert declaration.static_problems(config)


def test_new_declaration_is_valid_and_predecessor_is_still_invalid():
    config = declaration.load_config()
    assert declaration.static_problems(config) == []
    assert hashlib.sha256(original.RECORD.read_bytes()).hexdigest() == config["successor"][
        "predecessor_sha256"]
    assert json.loads(original.RECORD.read_text())["decision"]["verdict"] == "INVALID"


@pytest.mark.parametrize("key", ["schema", "producer", "checker", "record"])
def test_record_contract_rejects_predecessor_artifacts(key):
    config = declaration.load_config()
    config["record_requirements"][key] = original.load_config()["record_requirements"][key]
    assert any(f"record_requirements {key}" in p for p in declaration.static_problems(config))


def test_metadata_repair_cannot_change_a_numerical_declaration():
    config = declaration.load_config()
    config["derivative_validation"]["steps"] = [0.004, 0.002, 0.001, 0.0005]
    assert declaration.metadata_correction_problems(config)


@pytest.mark.parametrize("path,value", [
    (("metadata_correction", "execution_config_digest"), "0" * 64),
    (("metadata_correction", "execution_record_sha256"), "0" * 64),
    (("elapsed_seconds",), 1.0),
    (("provenance", "git_sha"), "0" * 40),
])
def test_metadata_repair_rejects_changes_to_executed_record(path, value):
    config = declaration.load_config()
    record = _mutate(json.loads(declaration.RECORD.read_text()), path, value)
    validation = json.loads(original.VALIDATION.read_text())
    assert checker.rederivation_problems(config, record, declaration.CONFIG.read_bytes(), validation)


@pytest.mark.parametrize("first_config,first_record,last_config,last_record,bad_pair", [
    ("declaration", "execution", "correction", "corrected_record", None),
    ("same", "same", "correction", "corrected_record", None),
    ("declaration", "execution", "same", "same", None),
    ("declaration", "execution", "correction", "corrected_record",
     ("correction", "corrected_record")),
])
def test_execution_and_metadata_repair_each_require_prior_declarations(
        monkeypatch, first_config, first_record, last_config, last_record, bad_pair):
    monkeypatch.setattr(declaration, "_execution_commit", lambda: first_config)
    monkeypatch.setattr(preflight, "_first_commit", lambda path: first_record)
    monkeypatch.setattr(preflight, "_last_commit", lambda path:
                        last_config if path == declaration.CONFIG else last_record)
    monkeypatch.setattr(preflight, "_shallow_boundary", lambda: set())
    monkeypatch.setattr(declaration.subprocess, "run", lambda args, **kwargs:
                        SimpleNamespace(returncode=int(tuple(args[-2:]) == bad_pair)))
    problems = declaration.commit_order_problems([])
    assert bool(problems) == (first_config == first_record or last_config == last_record
                              or bad_pair is not None)


def test_producer_refuses_original_and_alternative_output_paths(tmp_path):
    for path in (original.RECORD, tmp_path / "alternative.json"):
        assert producer.refusals(producer.argparse.Namespace(out=path))


def test_producer_refuses_an_existing_successor(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    path.write_text("{}")
    monkeypatch.setattr(producer, "RECORD", path)
    assert "runs once" in producer.refusals(producer.argparse.Namespace(out=path))[0]


def test_declaration_loader_rejects_result_fields(tmp_path):
    config = declaration.load_config()
    config["verdict"] = "NONE"
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="result field"):
        declaration.load_config(path)


def test_committed_post_hoc_uniform_none_and_unlicensed_allocation():
    config = declaration.load_config()
    record = json.loads(declaration.RECORD.read_text())
    validation = json.loads(original.VALIDATION.read_text())
    assert checker.rederivation_problems(config, record, declaration.CONFIG.read_bytes(),
                                        validation) == []
    assert record["decision"]["verdict"] == "NONE"
    assert set(record["decision"]["statuses"].values()) == {"PROHIBITIVE"}
    for entry in record["banks"].values():
        assert all(entry["deterministic_checks"].values())
        for price in entry["protocols"].values():
            assert price["allocation_diagnostic"]["estimator_licensed"] is False
    assert json.loads(original.RECORD.read_text())["decision"]["verdict"] == "INVALID"
