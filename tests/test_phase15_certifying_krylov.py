"""Q18-S3 correctness on the undeclared Hubbard dimer; no declared-prefix result here."""

from __future__ import annotations

import copy
import json
import math
from decimal import Decimal

import numpy as np
import pytest

from benchmarks import check_phase15_certifying_krylov as checker
from benchmarks import check_phase15_certifying_krylov_declaration as declaration
from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as preflight
from benchmarks import check_phase15_residual_successor_declaration as q18s1_declaration
from benchmarks import phase15_certifying_krylov as krylov
from benchmarks import phase15_certifying_screen as screen
from benchmarks import run_phase15_certifying_krylov as producer
from benchmarks import run_phase15_residual_successor as q18s1
from benchmarks.run_mapping_axis import _raw_pool
from benchmarks.run_phase15_h2_preflight import _digest
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import SecondMomentBank, krylov_response
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.projection import MatrixElementBank

ORDER = 3
LINEAGE_SIZE = 2


def _q18s1_entry(model, selected):
    """Q18-S1's own producer path on a toy bank: the frozen-bank lineage target."""
    config = copy.deepcopy(q18s1_declaration.load_config())
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
        "mapping_settings": len(q18.group_words(sh, "qwc_groups", model.n)),
        "mapping_energy_variance": None,
    }
    frozen = q18.quantities_for(model, selected, committed, "qwc_groups", "qwc_basis_cover")
    return q18s1.evaluate_bank(config, "hubbard_2x2", model, selected, frozen, root.as_dict())


@pytest.fixture(scope="module")
def toy():
    model = hubbard(2)
    prefixes = list(range(1, ORDER + 2))
    structure = declaration.first_moment_structure(model, ORDER, prefixes)
    bank = structure.pop("bank")
    final = bank.solve()
    config = copy.deepcopy(declaration.load_config())
    config["basis"]["prefixes"] = prefixes
    config["practical_allocation"]["replicate_count"] = 40
    config["measured_before_freezing"] = structure
    frozen = [identity_generator(model.n), *_raw_pool(model)][:LINEAGE_SIZE]
    return {
        "model": model, "config": config, "raw": json.dumps(config).encode(),
        "frozen": frozen, "predecessor": _q18s1_entry(model, frozen),
        "screen_record": {"spectral_reference": structure["spectrum"]},
        "ladder": {"energy": float(final.ground_energy),
                   "retained_rank": int(final.effective_rank)},
    }


def _run(toy):
    record = producer.run(toy["config"], config_bytes=toy["raw"], model=toy["model"],
                          predecessor=toy["predecessor"], screen_record=toy["screen_record"],
                          ladder=toy["ladder"], frozen=toy["frozen"])
    record["provenance"] = _provenance()
    return json.loads(json.dumps(record, allow_nan=False))


@pytest.fixture(scope="module")
def toy_record(toy):
    return _run(toy)


def _provenance():
    from clifford_qc.reproducibility import execution_provenance

    provenance = execution_provenance()
    provenance.update(git_sha="0" * 40, git_dirty=False)
    return json.loads(json.dumps(provenance))


def _rederive(toy, record):
    return checker.rederivation_problems(toy["config"], record, toy["raw"], toy["predecessor"],
                                         toy["screen_record"], toy["ladder"])


# ------------------------------------------------------------------ the bank

def test_krylov_rows_are_the_base_banks_products_to_the_bit():
    model = hubbard(2)
    bank = krylov.krylov_bank(model, ORDER)
    base = MatrixElementBank(bank.reference, model.hamiltonian,
                             [identity_generator(model.n), *krylov_response(
                                 model.hamiltonian, ORDER)])
    square = model.hamiltonian.to_mv() * model.hamiltonian.to_mv()
    for i in range(ORDER + 1):
        for j in range(ORDER + 1):
            assert bank.overlap_operator(i, j).terms == base.overlap_operator(i, j).terms
            assert bank.element_operator(i, j).terms == base.element_operator(i, j).terms
            assert (bank.observable_operator(square, i, j).terms
                    == base.observable_operator(square, i, j).terms)
    S, H = bank.matrices()
    S0, H0 = base.matrices()
    assert np.array_equal(S, S0) and np.array_equal(H, H0)
    second, second0 = SecondMomentBank(bank), SecondMomentBank(base)
    assert np.array_equal(second.matrix(), second0.matrix())


def test_krylov_bank_falls_back_for_other_observables():
    model = hubbard(2)
    bank = krylov.krylov_bank(model, 2)
    base = MatrixElementBank(bank.reference, model.hamiltonian,
                             [identity_generator(model.n), *krylov_response(model.hamiltonian, 2)])
    from clifford_qc.multivector import MV

    number = MV(model.n, {3 << 2: 1.0})  # Z on qubit 1
    assert bank.observable_operator(number, 0, 2).terms == base.observable_operator(
        number, 0, 2).terms


# ------------------------------------------------------------------ eighty digits

def _dimer_rows(size):
    model = hubbard(2)
    bank = krylov.krylov_bank(model, size - 1)
    second = SecondMomentBank(bank)
    rows = q18.block_rows(bank, second, list(range(size)))
    means = q18.reference_means(bank.reference, q18.row_words(rows))
    return bank, second, rows, means


def test_decimal_pencil_reproduces_the_double_precision_pair():
    bank, second, rows, means = _dimer_rows(2)
    result = bank.solve()
    residual = second.residual(result, 0)
    precise = krylov.precise_values(rows, 2, means, shift=krylov.ground_shift(result))
    assert precise["energy"] == pytest.approx(result.ground_energy, abs=1e-13)
    assert precise["variance"] == pytest.approx(residual.variance, rel=1e-12)


def test_decimal_derivatives_match_both_analytic_functionals():
    from clifford_qc.measurement.functionals import ritz_functional

    bank, _, rows, means = _dimer_rows(2)
    result = bank.solve()
    weights, _ = q18.residual_functional(rows, 2, means)
    weights = {int(w): float(v) for w, v in weights.items()}
    energy = dict(ritz_functional(bank, result.indices, result.ritz_vector(),
                                  result.ground_energy).coefficients)
    directions = q18.check_directions(weights, set(bank.word_set()), seed=20261003)
    for direction in directions.values():
        check = krylov.linearization_check(rows, 2, means, energy, weights, direction,
                                           shift=krylov.ground_shift(result),
                                           relative_tolerance=1e-10)
        assert check["energy"]["passes"] and check["variance"]["passes"]
    # A dropped Ritz-vector term is caught.
    broken = {w: 1.03 * v for w, v in weights.items()}
    check = krylov.linearization_check(rows, 2, means, energy, broken, directions["functional"],
                                       shift=krylov.ground_shift(result),
                                       relative_tolerance=1e-6)
    assert not check["variance"]["passes"]


def test_energy_functional_from_rows_is_ritz_functional():
    from clifford_qc.measurement.functionals import ritz_functional

    bank, _, rows, means = _dimer_rows(3)
    result = bank.solve()
    mine, energy = krylov.energy_functional(rows, 3, means)
    theirs = ritz_functional(bank, result.indices, result.ritz_vector(), result.ground_energy)
    assert energy == pytest.approx(result.ground_energy, abs=1e-12)
    for word, value in theirs.coefficients.items():
        assert mine[word] == pytest.approx(value, abs=1e-12)


def test_decimal_pencil_refuses_complex_rows():
    with pytest.raises(ValueError, match="real rows"):
        krylov.DecimalPencil({(0, 0): ({1: 1j}, {}, {})}, 1, {1: 0.5})


def test_decimal_ritz_solves_a_known_pencil():
    S = [[Decimal(2), Decimal(1)], [Decimal(1), Decimal(2)]]
    H = [[Decimal(1), Decimal(0)], [Decimal(0), Decimal(3)]]
    K = [[Decimal(1), Decimal(0)], [Decimal(0), Decimal(9)]]
    values = np.linalg.eigvals(np.linalg.solve(np.array(S, float), np.array(H, float)))
    energy, variance = krylov.decimal_ritz(S, H, K, Decimal(float(min(values.real)) - 1e-3))
    assert float(energy) == pytest.approx(float(min(values.real)), abs=1e-14)
    assert float(variance) >= -1e-30


# ------------------------------------------------------------------ the rules

class _Result:
    def __init__(self, eigenvalues, rank, condition=1.0):
        self.overlap_eigenvalues = tuple(eigenvalues)
        self.effective_rank = rank
        self.condition_number = condition


@pytest.mark.parametrize("eigenvalues, rank, stable", [
    ((2.0, 1e-9), 2, True),
    ((2.0, 0.99e-9), 2, False),
    ((2.0, 1e-3), 1, False),
    ((2e3, 2.5e-8), 2, True),  # condition 8e10
    ((2e3, 1e-8), 2, False),  # condition 2e11 > 1e11
    ((2.0, 0.0), 1, False),
    ((2.0, -3e-17), 1, False),  # rounding on a singular overlap, sign by platform
])
def test_rank_margin_boundaries(eigenvalues, rank, stable):
    margin = krylov.rank_margin(_Result(eigenvalues, rank), 2)
    assert margin["rank_stable"] is stable
    problems, restated = checker.rank_problems(margin, 2, "toy")
    assert problems == [] and restated is stable


def test_a_singular_overlap_has_no_condition_number():
    margin = krylov.rank_margin(_Result((2.0, -3e-17), 1), 2)
    assert margin["overlap_condition_number"] is None
    json.dumps(margin, allow_nan=False)
    forged = dict(margin, overlap_condition_number=1e17)
    problems, _ = checker.rank_problems(forged, 2, "toy")
    assert problems == ["toy: overlap condition number does not derive"]
    assert declaration._compare({"a": {"overlap_condition_number": 1e17}},
                                {"a": {"overlap_condition_number": None}}) != []


@pytest.mark.parametrize("oracle, practical, final", [
    ("CLOSED", {}, "CLOSED"), ("UNREACHED", {}, "UNREACHED"), ("INVALID", {}, "INVALID"),
    ("GROUPING_SENSITIVE", {}, "GROUPING_SENSITIVE"),
    ("OPEN", {"7": "PRACTICAL_AFFORDABLE", "6": "VALIDATION_FAILED"}, "LICENSED"),
    ("OPEN", {"7": "PRACTICAL_PROHIBITIVE", "6": "VALIDATION_FAILED"}, "VALIDATION_FAILED"),
    ("OPEN", {"7": "PRACTICAL_PROHIBITIVE"}, "PRACTICAL_PROHIBITIVE"),
])
def test_final_verdict_truth_table(oracle, practical, final):
    assert krylov.final_verdict(oracle, practical) == final


def test_an_open_oracle_needs_a_practical_prefix():
    with pytest.raises(ValueError):
        krylov.final_verdict("OPEN", {})


def test_effective_shots_charge_a_misallocated_pilot():
    row = {"pilot_shots": 100, "production_shots": 1000, "standard_error": 2.0}
    assert krylov.effective_shots(row, 1.0) == 100 + 4000
    assert krylov.effective_shots(dict(row, standard_error=1.0), 1.0) == 1100


def _replicate(energy, sigma, se_e=1.0, se_s=1.0, shots=(10, 100, 10, 100)):
    pe, ne, ps, ns = shots
    return {"energy": {"failure": None, "estimate": energy, "standard_error": se_e,
                       "pilot_shots": pe, "production_shots": ne, "total_shots": pe + ne},
            "residual": {"failure": None, "estimate": sigma, "variance_estimate": sigma ** 2,
                         "standard_error": se_s, "pilot_shots": ps, "production_shots": ns,
                         "total_shots": ps + ns}}


def test_practical_summary_rules():
    rule = declaration.load_config()["practical_allocation"]
    budget = {"energy": {"target_standard_error": 1.0}, "residual": {"target_standard_error": 1.0}}
    rng = np.random.default_rng(1)
    rows = [_replicate(float(e), 5.0 + float(s)) for e, s in rng.standard_normal((400, 2))]
    out = krylov.summarize_practical(rows, budget, rule, exact_energy=0.0, exact_sigma=5.0,
                                     first_excited=100.0)
    assert out["status"] == "PRACTICAL_AFFORDABLE"
    assert out["summary"]["expected_effective_cost_ratio"] == pytest.approx(1.0)
    expensive = [_replicate(float(e), 5.0 + float(s), shots=(10, 100, 10, 1200))
                 for e, s in rng.standard_normal((400, 2))]
    assert krylov.summarize_practical(expensive, budget, rule, exact_energy=0.0,
                                      exact_sigma=5.0, first_excited=100.0)[
        "status"] == "PRACTICAL_PROHIBITIVE"
    overconfident = [_replicate(float(e), 5.0 + 2.0 * float(s))
                     for e, s in rng.standard_normal((400, 2))]
    out = krylov.summarize_practical(overconfident, budget, rule, exact_energy=0.0,
                                     exact_sigma=5.0, first_excited=100.0)
    assert out["status"] == "VALIDATION_FAILED" and not out["validation_checks"][
        "sigma_coverage"]
    failed = copy.deepcopy(rows)
    failed[3]["residual"] = {"failure": "pilot_pencil_rank_deficient"}
    assert krylov.summarize_practical(failed, budget, rule, exact_energy=0.0, exact_sigma=5.0,
                                      first_excited=100.0)["status"] == "VALIDATION_FAILED"


def test_setting_sampler_reproduces_exact_setting_variances():
    _, second, rows, means = _dimer_rows(2)
    bank = second.bank
    weights, _ = q18.residual_functional(rows, 2, means)
    words = q18.row_words(rows)
    groups = q18.group_words(words, "qwc_groups", bank.n)
    sampler = krylov.SettingSampler(bank.reference, groups)
    exact = q18.group_variances(bank.reference, groups, {w: v for w, v in weights.items() if w})
    shots = [200_000] * len(sampler)
    counts = sampler.draw(np.random.default_rng(3), shots)
    sampled = sampler.variances(counts, shots, weights)
    for a, b in zip(sampled, exact):
        assert a == pytest.approx(b, rel=2e-2, abs=1e-9)
    means_sampled = sampler.means(counts, shots)
    for word in words:
        assert means_sampled[word] == pytest.approx(means[word], abs=1e-2)


# ------------------------------------------------------------------ the toy run

def test_toy_record_rederives_and_rebuilds(toy, toy_record):
    assert _rederive(toy, toy_record) == []
    assert checker.recompute_problems(
        toy["config"], toy_record, model=toy["model"], predecessor=toy["predecessor"],
        screen_record=toy["screen_record"], ladder=toy["ladder"], frozen=toy["frozen"]) == []


def test_toy_domain_statuses_and_verdicts(toy_record):
    prefixes = toy_record["basis"]["prefixes"]
    assert prefixes["1"]["role"] == "diagnostic" and not prefixes["1"]["certifying"]
    assert prefixes["2"]["certifying"] and prefixes["2"]["role"] == "decision"
    assert prefixes["3"]["status"] == "UNRESOLVED"
    assert prefixes["4"]["status"] == "RANK_UNSTABLE" and "precise" not in prefixes["4"]
    for size in ("1", "2"):
        assert all(prefixes[size]["deterministic_checks"].values())
    decision = toy_record["decision"]
    assert decision["domain"] == ["2"] and decision["oracle_verdict"] == "OPEN"
    assert decision["practical_prefixes"] == ["2"]
    assert decision["final_verdict"] == krylov.final_verdict(
        "OPEN", decision["practical_statuses"])
    practical = toy_record["practical"]["2"]
    assert len(practical["replicates"]) == 40 and practical["failures"] == 0
    assert toy_record["lineage"]["passes"]
    assert toy_record["evidence_schema"] == "clifford_qc.evidence.v1"


def test_the_practical_stage_is_seeded(toy, toy_record):
    again = _run(toy)
    assert again["practical"] == toy_record["practical"]


@pytest.mark.parametrize("path, value", [
    (("decision", "final_verdict"), "CLOSED"),
    (("decision", "oracle_verdict"), "CLOSED"),
    (("basis", "prefixes", "2", "certifying"), False),
    (("basis", "prefixes", "2", "status"), "PROHIBITIVE"),
    (("basis", "prefixes", "1", "role"), "decision"),
    (("basis", "prefixes", "3", "status"), "RANK_UNSTABLE"),
    (("basis", "prefixes", "4", "status"), "UNRESOLVED"),
    (("basis", "prefixes", "2", "rank", "rank_stable"), False),
    (("basis", "prefixes", "2", "protocols", "qwc_groups", "neyman_ratio"), 20.0),
    (("basis", "prefixes", "2", "deterministic_checks", "dense_variance_agrees"), False),
    (("basis", "prefixes", "2", "precise", "variance"), 1.0),
    (("basis", "prefixes", "2", "precise_derivatives", "functional", "variance", "numeric"),
     123.0),
    (("lineage", "passes"), False),
    (("lineage", "ladder_row", "energy"), False),
    (("practical", "2", "status"), "PRACTICAL_AFFORDABLE"),
    (("practical", "2", "summary", "sigma_coverage"), 0.5),
    (("quantum_advantage_claim",), True),
    (("evidence", "oracle_cost_ratios", "tier"), "exact_algebra"),
])
def test_each_forged_field_is_caught(toy, toy_record, path, value):
    forged = copy.deepcopy(toy_record)
    target = forged
    for key in path[:-1]:
        target = target[key]
    if target.get(path[-1]) == value:
        value = "forged"
    target[path[-1]] = value
    assert _rederive(toy, forged)


def test_the_practical_stage_runs_exactly_on_the_affordable_domain(toy, toy_record):
    extra = copy.deepcopy(toy_record)
    extra["practical"]["1"] = copy.deepcopy(extra["practical"]["2"])
    missing = copy.deepcopy(toy_record)
    missing["practical"] = {}
    for forged in (extra, missing):
        assert any("practical stage ran" in p for p in _rederive(toy, forged))


def test_a_rebuilt_ratio_must_match(toy, toy_record):
    forged = copy.deepcopy(toy_record)
    price = forged["basis"]["prefixes"]["2"]["protocols"]["qwc_groups"]
    price["residual_setting_variances"] = [v * 4.0 for v in price["residual_setting_variances"]]
    price["residual_variance_one_shot"] *= 4.0
    price["residual_neyman_sum"] *= 2.0
    price["cost_ratio"] *= 4.0
    price["neyman_ratio"] *= 4.0
    assert checker.recompute_problems(
        toy["config"], forged, model=toy["model"], predecessor=toy["predecessor"],
        screen_record=toy["screen_record"], ladder=toy["ladder"], frozen=toy["frozen"])


# ------------------------------------------------------------------ the declaration

def test_declaration_is_valid_and_result_free():
    config = declaration.load_config()
    assert declaration.static_problems(config) == []
    assert config["contains_results"] is False
    assert config["evidence"]["basis_selection"] == "post_hoc"
    frozen = config["measured_before_freezing"]["basis"]
    stable = [m for m, row in frozen.items() if row["passes_rank_margin"]]
    below = [m for m, row in frozen.items() if row["below_midpoint"]]
    assert sorted(set(stable) & set(below), key=int) == ["6", "7"]


@pytest.mark.parametrize("path, value", [
    (("statistic", "max_ratio"), 20),
    (("rank_margin", "factor"), 100.0),
    (("derivative_validation", "step"), "1e-20"),
    (("derivative_validation", "relative_tolerance"), 1e-4),
    (("fixed_step_validation", "steps"), [0.001, 0.0005, 0.00025, 0.000125]),
    (("practical_allocation", "replicate_count"), 1),
    (("practical_allocation", "pilot_fraction"), 2.0),
    (("practical_allocation", "max_ratio"), 30),
    (("practical_allocation", "validation", "min_coverage"), 1.5),
    (("practical_allocation", "confidence_family", "two_sided_z"), 1.645),
    (("certification", "variance_agreement"), 1e-3),
    (("basis", "prefixes"), [6, 7]),
    (("basis", "labels"), ["I", "H^1"]),
    (("grouping", "alternative_protocol"), None),
    (("evidence", "basis_selection"), "preregistered"),
    (("decision_rule", "final_verdicts"), ["CLOSED"]),
    (("question_id",), "Q18-S2"),
    (("claim_boundary",), "This was run once and has been read."),
])
def test_declaration_rejects_drift(path, value):
    config = copy.deepcopy(declaration.load_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert declaration.static_problems(config)


def test_declaration_loader_rejects_result_fields(tmp_path):
    config = declaration.load_config()
    config["basis"]["final_verdict"] = "CLOSED"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="result-shaped"):
        declaration.load_config(path)


def test_premises_reject_an_inexact_power_on_a_priced_prefix():
    config = declaration.load_config()
    quantities = copy.deepcopy(config["measured_before_freezing"])
    assert declaration.premise_problems(quantities, config) == []
    quantities["premises"]["exact_power_limit"] = 13
    assert declaration.premise_problems(quantities, config)
    quantities = copy.deepcopy(config["measured_before_freezing"])
    quantities["ladder"]["energy_reproduces"] = False
    assert declaration.premise_problems(quantities, config)


def test_producer_refuses_another_path_and_a_second_run(tmp_path, monkeypatch):
    class Args:
        out = tmp_path / "elsewhere.json"

    assert producer.refusals(Args()) == ["Q18-S3 writes only its declared record path"]
    Args.out = declaration.RECORD
    monkeypatch.setattr(type(declaration.RECORD), "exists", lambda self: True)
    assert producer.refusals(Args()) == ["Q18-S3 runs once; use its checker to rebuild"]


def test_spectral_tolerance_is_confined_to_eigenvalue_fields():
    assert declaration._compare({"a": {"smallest_overlap_eigenvalue": 1.0}},
                                {"a": {"smallest_overlap_eigenvalue": 1.001}}) == []
    assert declaration._compare({"a": {"ground_energy": 1.0}},
                                {"a": {"ground_energy": 1.001}})
    assert declaration._compare({"a": True}, {"a": 1})


def test_condition_tolerance_grows_with_conditioning():
    assert checker.condition_tolerance({"rank": {"retained_condition_number": 1.0}}) == 1e-9
    loose = checker.condition_tolerance({"rank": {"retained_condition_number": 1e10}})
    assert 1e-4 < loose < 1e-2
    assert math.isfinite(loose)


def test_the_energy_campaign_survives_k_words_outside_its_universe(toy, toy_record):
    entry = toy_record["basis"]["prefixes"]["1"]
    universes = entry["universes"]
    assert universes["measured_combined_words"] > universes["measured_sh_words"]
    config = copy.deepcopy(toy["config"])
    config["practical_allocation"]["replicate_count"] = 3
    out = krylov.practical_stage(config, toy["model"], entry, toy_record["spectral_reference"])
    assert len(out["replicates"]) == 3
    assert all(row["energy"]["failure"] is None for row in out["replicates"])


def test_a_rounding_level_reference_mean_rebuilds_to_its_declared_bound(toy, toy_record):
    """The residual's reference mean is zero; another BLAS kernel moves its rounding."""
    def rebuild(record):
        return checker.recompute_problems(
            toy["config"], record, model=toy["model"], predecessor=toy["predecessor"],
            screen_record=toy["screen_record"], ladder=toy["ladder"], frozen=toy["frozen"])

    entry = toy_record["basis"]["prefixes"]["2"]
    bound = 1e-9 * (1.0 + entry["cancellation_scale"])
    moved = copy.deepcopy(toy_record)
    moved["basis"]["prefixes"]["2"]["estimator"]["residual_reference_mean"] += 0.5 * bound
    assert rebuild(moved) == []
    forged = copy.deepcopy(toy_record)
    forged["basis"]["prefixes"]["2"]["estimator"]["residual_reference_mean"] += 2.0 * bound
    assert any("residual_reference_mean" in p for p in rebuild(forged))
