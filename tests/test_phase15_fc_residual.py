"""Q18-S4 correctness on the undeclared Hubbard dimer; no declared-bank result here."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math

import numpy as np
import pytest

from benchmarks import check_phase15_fc_residual as checker
from benchmarks import check_phase15_fc_residual_declaration as declaration
from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as lineage
from benchmarks import check_phase15_residual_successor_declaration as q18s1_declaration
from benchmarks import phase15_fc_residual as fc
from benchmarks import run_phase15_fc_residual as producer
from benchmarks import run_phase15_residual_successor as q18s1_producer
from benchmarks.run_mapping_axis import _raw_pool
from benchmarks.run_phase15_h2_preflight import _digest
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.pauli_action import PauliLinearOperator
from clifford_qc.subspace import SecondMomentBank
from clifford_qc.subspace.generators import identity_generator

TOY = "toy"


def _provenance():
    from clifford_qc.reproducibility import execution_provenance

    provenance = execution_provenance()
    provenance.update(git_sha="0" * 40, git_dirty=False)
    return json.loads(json.dumps(provenance))


def _committed(bank, second, root, model):
    sh = set(bank.word_set())
    combined = sh | set(second.word_set())
    return {
        "validation_energy": root.energy, "variance": root.variance,
        "cancellation_scale": root.cancellation_scale, "variance_resolved": root.resolved,
        "preflight_sh_word_universe": len(sh), "ledger_sh_word_universe": len(sh),
        "raw_combined_word_universe": len(combined),
        "raw_combined_words_sha256": _digest(combined),
        "mapping_protocol": "qwc_groups",
        "mapping_settings": len(q18.group_words(sh, "qwc_groups", model.n)),
        "mapping_energy_variance": None,
    }


@pytest.fixture(scope="module")
def toy():
    model = hubbard(2)
    selected = [identity_generator(model.n), *_raw_pool(model)[:2]]
    bank = lineage.first_moment_bank(model, selected)
    second = SecondMomentBank(bank)
    root = second.residual(bank.solve(), 0)
    committed = _committed(bank, second, root, model)
    validation = {"banks": {TOY: {"roots": [root.as_dict()]}}}

    # Q18-S1's own producer on the toy is the anchor Q18-S4 must reproduce.
    s1 = copy.deepcopy(q18s1_declaration.load_config())
    s1["revisions"] = s1["revisions"][:1]
    s1["banks"]["systems"] = [TOY]
    s1["grouping"]["declared_protocol_by_system"] = {TOY: "qwc_groups"}
    s1["grouping"]["alternative_protocol_by_system"] = {TOY: "qwc_basis_cover"}
    s1["measured_before_freezing"] = {TOY: q18.quantities_for(
        model, selected, committed, "qwc_groups", "qwc_basis_cover")}
    predecessor = q18s1_producer.run_successor(
        s1, config_bytes=json.dumps(s1).encode(), inputs=lambda name: (model, selected),
        validation=validation)

    config = copy.deepcopy(declaration.load_config())
    config["banks"]["systems"] = [TOY]
    config["grouping"]["declared_protocol_by_system"] = {TOY: "fc_greedy"}
    config["grouping"]["alternative_protocol_by_system"] = {TOY: "fc_isotropic_cover"}
    config["measured_before_freezing"] = {
        TOY: declaration.quantities_for(model, selected, committed)}
    raw = json.dumps(config).encode()
    record = producer.run(config, config_bytes=raw, inputs=lambda name: (model, selected),
                          validation=validation, predecessor=predecessor)
    record["provenance"] = _provenance()
    record = json.loads(json.dumps(record, allow_nan=False))
    return {"model": model, "selected": selected, "bank": bank, "second": second,
            "config": config, "raw": raw, "record": record, "validation": validation,
            "predecessor": predecessor}


def _rederive(toy, record):
    return checker.rederivation_problems(toy["config"], record, toy["raw"],
                                         predecessor=toy["predecessor"],
                                         validation=toy["validation"])


def _rebuild(toy, record):
    return checker.recompute_problems(toy["config"], record,
                                      inputs=lambda name: (toy["model"], toy["selected"]),
                                      validation=toy["validation"],
                                      predecessor=toy["predecessor"])


def _mutate(value, path, new):
    value = copy.deepcopy(value)
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new
    return value


# ------------------------------------------------------------------ the toy run

def test_toy_record_rederives_and_rebuilds(toy):
    assert _rederive(toy, toy["record"]) == []
    assert _rebuild(toy, toy["record"]) == []
    entry = toy["record"]["banks"][TOY]
    assert all(entry["deterministic_checks"].values())
    assert set(entry["protocols"]) == set(fc.PROTOCOLS)


def test_the_dimer_puts_the_two_protocols_on_opposite_sides_of_ten(toy):
    entry = toy["record"]["banks"][TOY]
    greedy = entry["protocols"]["fc_greedy"]["cost_ratio"]
    cover = entry["protocols"]["fc_isotropic_cover"]["cost_ratio"]
    assert greedy == pytest.approx(0.0019095882, rel=1e-6)
    assert cover == pytest.approx(37.18317221, rel=1e-6)
    assert entry["status"] == "GROUPING_SENSITIVE"
    assert toy["record"]["decision"]["verdict"] == "NONE"


def test_the_functionals_are_q18s1s(toy):
    anchor = toy["record"]["banks"][TOY]["predecessor_anchor"]
    assert anchor and all(row["agrees"] for row in anchor.values())
    qwc = toy["record"]["banks"][TOY]["q18s1_uniform_qwc"]
    assert set(qwc) == {"qwc_groups", "qwc_basis_cover"}


def test_setting_variances_are_dense_variances(toy):
    model, bank = toy["model"], toy["bank"]
    rows = q18.block_rows(bank, toy["second"], list(range(len(toy["selected"]))))
    combined = q18.row_words(rows)
    means = q18.reference_means(bank.reference, combined)
    weights, _ = q18.residual_functional(rows, len(toy["selected"]), means)
    weights = {int(word): float(value) for word, value in weights.items() if word != 0}
    rho = to_matrix(bank.reference)
    for protocol in fc.PROTOCOLS:
        groups = fc.fc_partition(combined, protocol, model.n)
        recorded = toy["record"]["banks"][TOY]["protocols"][protocol][
            "residual_setting_variances"]
        for group, value in zip(groups, recorded):
            matrices = [to_matrix(MV(model.n, {code: 1.0})) for code in group]
            for a, b in itertools.combinations(matrices, 2):
                assert np.allclose(a @ b, b @ a)
            O = sum(weights.get(code, 0.0) * m for code, m in zip(group, matrices))
            dense = np.trace(rho @ O @ O).real - np.trace(rho @ O).real ** 2
            assert value == pytest.approx(dense, abs=1e-12 * max(1.0, abs(dense)))


# ------------------------------------------------------------------ pieces

def test_anticommuting_pairs_match_dense_commutators():
    rng = np.random.default_rng(3)
    codes = [int(code) for code in rng.choice(4 ** 3, size=24, replace=False)]
    matrices = [to_matrix(MV(3, {code: 1.0})) for code in codes]
    dense = sum(not np.allclose(a @ b, b @ a)
                for a, b in itertools.combinations(matrices, 2))
    assert fc.anticommuting_pairs(3, codes) == dense
    assert fc.anticommuting_pairs(3, codes[:1]) == 0


def test_partition_problems_catch_overlap_gap_and_anticommutation():
    x, z, xx = 0b01, 0b11, 0b0101  # X0, Z0, X0 X1 on two qubits
    universe = {0, x, z, xx}
    assert fc.partition_problems([[x, xx], [z]], universe, 2) == []
    assert "more than one" in fc.partition_problems([[x, xx], [z, x]], universe, 2)[0]
    assert "cover" in fc.partition_problems([[x, xx]], universe, 2)[0]
    assert "not fully commuting" in fc.partition_problems([[x, z], [xx]], universe, 2)[0]


def test_partitions_are_deterministic_and_their_digest_canonical(toy):
    words = set(toy["bank"].word_set())
    for protocol in fc.PROTOCOLS:
        first = fc.fc_partition(words, protocol, toy["model"].n)
        assert first == fc.fc_partition(set(sorted(words, reverse=True)), protocol,
                                        toy["model"].n)
        shuffled = [list(reversed(group)) for group in reversed(first)]
        assert fc.partition_digest(shuffled) == fc.partition_digest(first)
    with pytest.raises(ValueError):
        fc.fc_partition(words, "qwc_groups", toy["model"].n)


def test_the_checkers_pauli_action_is_the_librarys():
    rng = np.random.default_rng(5)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    for code in rng.choice(4 ** 4, size=40, replace=False):
        library = PauliLinearOperator(MV(4, {int(code): 1.0})).matvec(psi)
        assert np.allclose(checker.pauli_image(4, int(code), psi), library)


def test_the_checkers_ladder_and_combination_are_q18s():
    for declared, alternative in itertools.product((5.0, 10.0, 10.5), repeat=2):
        assert checker.ladder(declared, alternative) == q18.status_of(declared, alternative,
                                                                      10.0)
    statuses = ("AFFORDABLE", "PROHIBITIVE", "GROUPING_SENSITIVE", "INVALID")
    for combination in itertools.product(statuses, repeat=3):
        assert checker.combine(combination) == q18.verdict_of(combination)


# ------------------------------------------------------------------ forgeries

@pytest.mark.parametrize("path,value", [
    (("banks", TOY, "status"), "AFFORDABLE"),
    (("decision", "verdict"), "FULL"),
    (("decision", "statuses", TOY), "PROHIBITIVE"),
    (("decision", "consequence"), "anything"),
    (("banks", TOY, "protocols", "fc_greedy", "cost_ratio"), 11.0),
    (("banks", TOY, "protocols", "fc_isotropic_cover", "neyman_ratio"), 1.0),
    (("banks", TOY, "protocols", "fc_greedy", "residual_setting_variances", 0), 9.0),
    (("banks", TOY, "protocols", "fc_greedy", "energy_settings"), 99),
    (("banks", TOY, "protocols", "fc_greedy", "statevector_agreement"), False),
    (("banks", TOY, "protocols", "fc_greedy", "statevector_worst_relative_deviation"), 1.0),
    (("banks", TOY, "protocols", "fc_greedy", "partition_problems"), ["forged"]),
    (("banks", TOY, "protocols", "fc_greedy", "role"), "alternative"),
    (("banks", TOY, "predecessor_anchor", "variance", "q18s1"), 1.0),
    (("banks", TOY, "predecessor_anchor", "energy_functional_words", "agrees"), False),
    (("banks", TOY, "deterministic_checks", "partitions_valid"), False),
    (("banks", TOY, "finite_differences", "functional", "passes"), False),
    (("banks", TOY, "finite_differences", "sh_gaussian", "evaluations", 0, "numeric"), 1.0),
    (("banks", TOY, "estimator", "residual_norm"), 9.0),
    (("claim_boundary",), "a different boundary"),
    (("quantum_advantage_claim",), True),
    (("excluded",), {}),
    (("successor", "estimator_licensed"), True),
])
def test_each_forged_field_is_caught(toy, path, value):
    assert _rederive(toy, _mutate(toy["record"], path, value))


@pytest.mark.parametrize("protocol", fc.PROTOCOLS)
@pytest.mark.parametrize("side", ("energy", "residual"))
def test_permuted_setting_variances_fail_the_rebuild(toy, protocol, side):
    bad = copy.deepcopy(toy["record"])
    values = bad["banks"][TOY]["protocols"][protocol][f"{side}_setting_variances"]
    low = min(range(len(values)), key=values.__getitem__)
    high = max(range(len(values)), key=values.__getitem__)
    assert values[high] - values[low] > 1e-9
    values[low], values[high] = values[high], values[low]
    # Sums, Neyman sums, ratios and the partition digest all remain unchanged.
    assert _rederive(toy, bad) == []
    assert any(f"{side}_setting_variances does not rebuild" in p
               for p in _rebuild(toy, bad))


def test_a_self_consistent_forgery_fails_the_rebuild(toy):
    bad = copy.deepcopy(toy["record"])
    price = bad["banks"][TOY]["protocols"]["fc_isotropic_cover"]
    # Halving every residual setting variance keeps every restated sum, ratio and
    # status (R stays above ten), so only a rebuild can tell.
    price["residual_setting_variances"] = [v / 2.0 for v in price["residual_setting_variances"]]
    price["residual_variance_one_shot"] = math.fsum(price["residual_setting_variances"])
    price["residual_neyman_sum"] = math.fsum(
        math.sqrt(v) for v in price["residual_setting_variances"])
    variance = bad["banks"][TOY]["estimator"]["variance"]
    price["cost_ratio"] = (price["residual_settings"] * price["residual_variance_one_shot"]) / (
        4.0 * variance * price["energy_settings"] * price["energy_variance_one_shot"])
    price["neyman_ratio"] = price["residual_neyman_sum"] ** 2 / (
        4.0 * variance * price["energy_neyman_sum"] ** 2)
    assert price["cost_ratio"] > 10.0
    assert _rederive(toy, bad) == []
    assert any("does not rebuild" in p for p in _rebuild(toy, bad))


# ------------------------------------------------------------------ declaration

def test_the_committed_declaration_is_valid():
    config = declaration.load_config()
    assert declaration.static_problems(config) == []
    assert tuple(config["banks"]["systems"]) == declaration.SYSTEMS
    assert "h2o_cas8e6o" in config["banks"]["excluded"]
    for frozen in config["measured_before_freezing"].values():
        assert tuple(frozen) == declaration.FROZEN_FIELDS
        assert set(frozen["fc_energy_settings"]) == set(fc.PROTOCOLS)


@pytest.mark.parametrize("path,value", [
    (("question_id",), "Q18-S1"),
    (("statistic", "max_ratio"), 100),
    (("decision_rule", "combination_rule"), "anything AFFORDABLE gives FULL"),
    (("linearization", "finite_difference", "seed"), 1),
    (("derivative_validation", "steps"), [0.01, 0.005, 0.0025, 0.00125]),
    (("grouping", "allocation"), "neyman"),
    (("grouping", "declared_protocol_by_system", "beh2"), "fc_isotropic_cover"),
    (("grouping", "alternative_protocol_by_system", "h4"), "qwc_basis_cover"),
    (("banks", "systems"), ["h4", "h4_converged", "beh2", "hubbard_2x2", "h2o_cas8e6o"]),
    (("banks", "excluded", "h2o_cas8e6o"), ""),
    (("evidence", "design_status"), "preregistered"),
    (("evidence", "label"), "sampled"),
    (("prespecified_followup", "permitted"), "a cheaper protocol"),
    (("successor", "q18s1_outcome"), "FULL"),
    (("successor", "estimator_licensed"), True),
    (("successor", "q18s1_record", "sha256"), "0" * 64),
    (("measured_before_freezing", "beh2", "variance"), 1.0),
    (("reachable_verdicts",), ["FULL"]),
    (("record_requirements", "record"), "benchmarks/reference_results/other.json"),
    (("claim_boundary",), "a run has been made"),
])
def test_the_declaration_rejects_drift(path, value):
    assert declaration.static_problems(_mutate(declaration.load_config(), path, value))


def test_the_loader_refuses_result_shaped_keys(tmp_path):
    config = declaration.load_config()
    config["measured_before_freezing"]["beh2"]["cost_ratio"] = 1.0
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="result-shaped"):
        declaration.load_config(path)


def test_structural_drift_is_caught_without_recomputing_a_bank():
    config = declaration.load_config()
    frozen = config["measured_before_freezing"]
    assert declaration.structural_problems(config, quantities=lambda name: frozen[name]) == []
    drifted = copy.deepcopy(frozen)
    drifted["hubbard_2x2"]["fc_energy_settings"]["fc_greedy"] += 1
    drifted["h4"]["full_rank"] = False
    problems = declaration.structural_problems(config, quantities=lambda name: drifted[name])
    assert any("fc_energy_settings" in p for p in problems)
    assert any("premise full_rank" in p for p in problems)


def test_the_producer_writes_once_and_only_its_path(tmp_path, monkeypatch):
    assert producer.refusals(argparse.Namespace(out=tmp_path / "elsewhere.json"))
    path = tmp_path / "record.json"
    path.write_text("{}")
    monkeypatch.setattr(producer, "RECORD", path)
    monkeypatch.setattr(declaration, "RECORD", path)
    assert "runs once" in producer.refusals(argparse.Namespace(out=path))[0]


# ------------------------------------------------------------------ the committed record

def test_the_committed_record_is_the_executed_one_plus_v1_evidence():
    from clifford_qc.evidence import validate_evidence

    record = json.loads(producer.RECORD.read_text(encoding="utf-8"))
    validate_evidence(record)
    executed = fc.executed_record(record)
    assert hashlib.sha256(fc.record_bytes(executed)).hexdigest() == (
        checker.EXECUTED_RECORD_SHA256)
    assert checker.correction_problems(record) == []
    config = declaration.load_config()
    assert checker.rederivation_problems(config, record, declaration.CONFIG.read_bytes()) == []
    assert record["decision"]["verdict"] == "NONE"
    assert set(record["decision"]["statuses"].values()) == {"PROHIBITIVE"}


@pytest.mark.parametrize("path,value", [
    (("banks", "hubbard_2x2", "protocols", "fc_greedy", "cost_ratio"), 9.0),
    (("evidence", "fully_commuting_cost_ratios", "tier"), "exact_algebra"),
    (("metadata_correction", "scope"), "anything"),
    (("metadata_correction", "executed_evidence", "label"), "sampled"),
    (("elapsed_seconds",), 1.0),
])
def test_a_tampered_correction_is_caught(path, value):
    record = json.loads(producer.RECORD.read_text(encoding="utf-8"))
    assert checker.correction_problems(_mutate(record, path, value))


def test_only_the_executed_bytes_take_the_correction(toy):
    with pytest.raises(ValueError, match="executed record"):
        fc.correct_evidence(toy["record"], "0" * 64)
    executed = hashlib.sha256(fc.record_bytes(toy["record"])).hexdigest()
    corrected = fc.correct_evidence(toy["record"], executed)
    assert fc.executed_record(corrected) == toy["record"]
    assert checker.correction_problems(corrected, executed_sha256=executed) == []
    with pytest.raises(ValueError):
        fc.correct_evidence(corrected, executed)
    with pytest.raises(ValueError, match="already"):
        fc.correct_evidence(corrected, hashlib.sha256(fc.record_bytes(corrected)).hexdigest())
