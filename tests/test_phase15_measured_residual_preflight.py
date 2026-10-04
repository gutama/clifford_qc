"""The Q18 producer and result checker, run without the preflight.

Nothing here forms a second-moment row on a declared bank. The end-to-end
tests price two banks on the 2-site Hubbard dimer, which is outside the
declaration, under the frozen rules and protocols. The per-setting variances
are held to an independent dense computation, and the linearized residual
variance to a Monte Carlo of the full nonlinear estimator. The checker must
re-derive the producer's decision on its own, and each tampered record must be
caught by the check written for it.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from benchmarks import check_phase15_measured_residual_preflight as checker
from benchmarks import check_phase15_measured_residual_preregistration as gate
from benchmarks import check_phase15_preregistration as preflight_gate
from benchmarks import run_phase15_measured_residual_preflight as producer
from benchmarks.run_mapping_axis import _raw_pool
from benchmarks.run_phase15_h2_preflight import _digest
from clifford_qc.dense_reference import to_matrix
from clifford_qc.measurement.functionals import ritz_functional
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.subspace import SecondMomentBank
from clifford_qc.subspace.generators import identity_generator

PROTOCOLS = {"toy": ("qwc_groups", "qwc_basis_cover"), "toy_small": ("qwc_basis_cover", None)}


@pytest.fixture(scope="module")
def dimer():
    model = hubbard(2)
    raw = _raw_pool(model)
    identity = identity_generator(model.n)
    return {"toy": (model, [identity, raw[0], raw[1]]),
            "toy_small": (model, [identity, raw[0]])}


def _toy_setup(dimer):
    """A config and validation record for the toy banks, built the declared way."""
    config = copy.deepcopy(gate.load_config())
    config["banks"]["systems"] = sorted(dimer)
    config["grouping"]["declared_protocol_by_system"] = {k: v[0] for k, v in PROTOCOLS.items()}
    config["grouping"]["alternative_protocol_by_system"] = {k: v[1] for k, v in PROTOCOLS.items()}
    validation, measured = {"banks": {}}, {}
    for name, (model, selected) in dimer.items():
        bank = preflight_gate.first_moment_bank(model, selected)
        second = SecondMomentBank(bank)
        root = second.residual(bank.solve(), 0)
        validation["banks"][name] = {"roots": [root.as_dict()]}
        sh = set(bank.word_set())
        combined = sh | set(second.word_set())
        protocol, alternative = PROTOCOLS[name]
        committed = {
            "validation_energy": root.energy, "variance": root.variance,
            "cancellation_scale": root.cancellation_scale,
            "variance_resolved": root.resolved,
            "preflight_sh_word_universe": len(sh), "ledger_sh_word_universe": len(sh),
            "raw_combined_word_universe": len(combined),
            "raw_combined_words_sha256": _digest(combined),
            "mapping_protocol": protocol,
            "mapping_settings": len(gate.group_words(sh, protocol, model.n)),
            "mapping_energy_variance": None,
        }
        measured[name] = gate.quantities_for(model, selected, committed, protocol, alternative)
    config["measured_before_freezing"] = measured
    return config, validation


@pytest.fixture(scope="module")
def toy(dimer):
    config, validation = _toy_setup(dimer)
    raw = json.dumps(config).encode()
    record = producer.run_preflight(config, config_bytes=raw, inputs=dimer.__getitem__,
                                    validation=validation)
    record["provenance"] = {"git_dirty": False}
    return config, raw, record, validation


def _all_problems(config, raw, record, validation, inputs, *, recompute=True):
    problems = (checker.declaration_problems(config, record, raw)
                + checker.completeness_problems(config, record))
    if problems:
        return problems
    problems += checker.freeze_problems(config, record)
    problems += checker.rederivation_problems(config, record, validation)
    if recompute:
        problems += checker.recompute_problems(config, record, validation, inputs=inputs)
    return problems


# ------------------------------------------------------------------ end to end

def test_the_toy_run_re_derives_and_recomputes_cleanly(toy, dimer):
    config, raw, record, validation = toy
    assert record["schema"] == producer.SCHEMA == config["record_requirements"]["schema"]
    assert record["quantum_advantage_claim"] is False
    json.dumps(record)  # plain types only
    for entry in record["banks"].values():
        assert all(entry["deterministic_checks"].values())
    assert _all_problems(config, raw, record, validation, dimer.__getitem__) == []


def test_the_toy_banks_reach_two_statuses_and_a_restricted_verdict(toy):
    """The dimer's full block is far costlier than its two-generator one."""
    _, _, record, _ = toy
    assert record["decision"]["statuses"] == {"toy": "PROHIBITIVE",
                                              "toy_small": "AFFORDABLE"}
    assert record["decision"]["verdict"] == "RESTRICTED"
    toy_prices = record["banks"]["toy"]["protocols"]
    assert set(toy_prices) == {"qwc_groups", "qwc_basis_cover"}
    assert set(record["banks"]["toy_small"]["protocols"]) == {"qwc_basis_cover"}


def test_partitions_cover_the_measured_universes(toy):
    _, _, record, _ = toy
    for entry in record["banks"].values():
        universes = entry["universes"]
        assert universes["measured_sh_words"] == universes["raw_sh_word_universe"] - 1
        for price in entry["protocols"].values():
            assert price["energy_partitioned_words"] == universes["measured_sh_words"]
            assert price["residual_partitioned_words"] == universes["measured_combined_words"]


def _residual_and_energy(dimer, name):
    model, selected = dimer[name]
    bank = preflight_gate.first_moment_bank(model, selected)
    second = SecondMomentBank(bank)
    rows = gate.block_rows(bank, second, list(range(len(selected))))
    means = gate.reference_means(bank.reference, gate.row_words(rows))
    weights, variance = gate.residual_functional(rows, len(selected), means)
    result = bank.solve()
    energy = ritz_functional(bank, result.indices, result.ritz_vector(), result.ground_energy)
    return model, bank, rows, means, weights, variance, dict(energy.coefficients)


def test_group_variances_match_an_independent_dense_computation(toy, dimer):
    """Var_psi of each setting's operator from dense matrices, no MV product."""
    _, _, record, _ = toy
    model, bank, rows, means, weights, variance, energy = _residual_and_energy(dimer, "toy")
    psi = np.zeros(2 ** model.n)
    psi[np.argmax(np.abs(np.diag(to_matrix(bank.reference))))] = 1.0
    for side, functional, key, universe in (
            ("energy", energy, "energy_variance_one_shot", set(bank.word_set())),
            ("residual", {w: v for w, v in weights.items() if w}, "residual_variance_one_shot",
             gate.row_words(rows))):
        groups = gate.group_words(universe, "qwc_groups", model.n)
        total = 0.0
        for group in groups:
            operator = np.zeros((2 ** model.n,) * 2)
            for word in group:
                operator = operator + functional.get(word.code, 0.0) * to_matrix(
                    MV(model.n, {word.code: 1.0}))
            mean = np.real(psi @ operator @ psi)
            total += float(np.real(psi @ operator @ operator @ psi) - mean ** 2)
        recorded = record["banks"]["toy"]["protocols"]["qwc_groups"][key]
        assert recorded == pytest.approx(total, rel=1e-10), side


def _group_covariance(model, group, means):
    """Exact covariance of a QWC setting's word outcomes on the reference."""
    codes = [word.code for word in group]
    matrices = {c: to_matrix(MV(model.n, {c: 1.0})) for c in codes}
    covariance = np.zeros((len(codes), len(codes)))
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            product = MV(model.n, {a: 1.0}) * MV(model.n, {b: 1.0})
            joint = sum(value * means.get(code, 0.0) for code, value in product.terms.items())
            covariance[i, j] = float(np.real(joint)) - means[a] * means[b]
            assert np.allclose(matrices[a] @ matrices[b], matrices[b] @ matrices[a])
    return codes, covariance


def test_the_linearized_variance_predicts_the_nonlinear_estimator(dimer):
    """Draw word means with each setting's exact covariance at N shots, and
    re-solve the measured pencil: the spread of sigma-hat^2 must be V / N.
    This is the claim the ratio rests on, Ritz-vector term included."""
    model, bank, rows, means, weights, variance, _ = _residual_and_energy(dimer, "toy")
    groups = gate.group_words(gate.row_words(rows), "qwc_groups", model.n)
    predicted = sum(gate.group_variances(bank.reference, groups,
                                         {w: v for w, v in weights.items() if w}))
    shots, draws = 10 ** 6, 4000
    rng = np.random.default_rng(20261004)
    factors = []
    for group in groups:
        codes, covariance = _group_covariance(model, group, means)
        values, vectors = np.linalg.eigh(covariance)
        factors.append((codes, vectors * np.sqrt(np.clip(values, 0.0, None) / shots)))
    samples = []
    for _ in range(draws):
        noisy = dict(means)
        for codes, factor in factors:
            noise = factor @ rng.standard_normal(factor.shape[1])
            for code, delta in zip(codes, noise):
                noisy[code] = means[code] + float(delta)
        samples.append(gate.residual_variance(rows, 3, noisy))
    empirical = float(np.var(samples, ddof=1)) * shots
    # The sample variance of 4000 Gaussian draws has relative error ~ sqrt(2/4000).
    assert empirical == pytest.approx(predicted, rel=0.1)
    assert float(np.mean(samples)) == pytest.approx(variance, abs=5 * np.sqrt(predicted / shots))


# ------------------------------------------------------------------ refusals

def test_a_bank_subset_may_not_write_the_committed_record():
    args = producer.argparse.Namespace(out=producer.RECORD, banks=["beh2"], overwrite=False)
    assert any("subset" in p for p in producer.refusals(args, reduced=True))


def test_an_existing_record_is_not_overwritten(tmp_path, monkeypatch):
    record = tmp_path / "record.json"
    record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(producer, "RECORD", record)
    args = producer.argparse.Namespace(out=record, banks=None, overwrite=False)
    assert any("runs once" in p for p in producer.refusals(args, reduced=False))
    args.overwrite = True
    assert producer.refusals(args, reduced=False) == []


def test_another_path_is_never_refused(tmp_path):
    args = producer.argparse.Namespace(out=tmp_path / "x.json", banks=["toy"], overwrite=False)
    assert producer.refusals(args, reduced=True) == []


# ------------------------------------------------------------------ tampering

def _edit(path, value):
    def apply(record):
        node = record
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value(node[path[-1]]) if callable(value) else value
    return apply


TAMPERS = {
    "ratio": (_edit(("banks", "toy_small", "protocols", "qwc_basis_cover", "cost_ratio"),
                    lambda r: r * 0.5), "is not the derived"),
    "neyman": (_edit(("banks", "toy", "protocols", "qwc_groups", "neyman_ratio"),
                     lambda r: r * 2.0), "Neyman ratio"),
    "neyman_bound": (_edit(("banks", "toy", "protocols", "qwc_groups", "energy_neyman_sum"),
                           lambda s: s * 10.0), "Cauchy-Schwarz"),
    "status": (_edit(("banks", "toy", "status"), "AFFORDABLE"), "is not the rule's"),
    "verdict": (_edit(("decision", "verdict"), "FULL"), "recorded verdict"),
    "consequence": (_edit(("decision", "consequence"), "build everything"), "consequence"),
    "statuses": (_edit(("decision", "statuses", "toy"), "AFFORDABLE"), "summary's statuses"),
    "fd_outcome": (_edit(("banks", "toy", "finite_differences", "functional", "passes"), False),
                   "outcome contradicts"),
    "fd_tolerance": (_edit(("banks", "toy", "finite_differences", "sh_gaussian", "tolerance"),
                           lambda t: t * 100), "tolerance is not"),
    "settings": (_edit(("banks", "toy", "protocols", "qwc_groups", "energy_settings"),
                       lambda g: g + 1), "deterministic checks"),
    "digest": (_edit(("banks", "toy", "universes", "raw_combined_words_sha256"), "0" * 64),
               "deterministic checks"),
    "measured": (_edit(("banks", "toy", "universes", "measured_sh_words"), lambda m: m + 1),
                 "without the identity"),
    "scale": (_edit(("banks", "toy", "estimator", "cancellation_scale"), lambda k: k * 2),
              "cancellation scale"),
    "norm": (_edit(("banks", "toy", "estimator", "residual_norm"), lambda s: s * 1.1),
             "root of its variance"),
    "half": (_edit(("banks", "toy", "estimator", "matched_precision_half"), lambda s: s * 2),
             "sigma / 4"),
    "check_flag": (_edit(("banks", "toy", "deterministic_checks", "partitions_valid"), False),
                   "deterministic checks"),
    "reference": (_edit(("banks", "toy", "mapping_axis_energy_variance"), 0.5),
                  "energy-variance reference"),
    "frozen": (_edit(("banks", "toy", "measured_at_execution", "energy_settings"),
                     lambda g: g + 1), "at execution"),
    "advantage": (_edit(("quantum_advantage_claim",), True), "quantum_advantage_claim"),
    "boundary": (lambda r: r.__setitem__("claim_boundary", r["preregistration"][
        "config_claim_boundary_at_landing"]), "inherits"),
    "denial": (_edit(("claim_boundary",), lambda b: b + " It holds no ratio."), "denies"),
    "boundary_replaced": (_edit(("claim_boundary",), "These ratios license any reading."),
                          "word for word"),
    "digest_config": (_edit(("config_digest",), "0" * 64), "digest no longer matches"),
    "statistic": (_edit(("statistic", "max_ratio"), 100), "statistic is not"),
    "protocol_role": (_edit(("banks", "toy", "protocols", "qwc_basis_cover", "role"),
                            "declared"), "are not the declared"),
    "direction": (lambda r: r["banks"]["toy"]["finite_differences"].pop("sh_gaussian"),
                  "frozen directions"),
    "missing_bank": (lambda r: r["banks"].pop("toy_small"), "are not the declared"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_each_tamper_is_caught_by_its_check(toy, dimer, name):
    config, raw, record, validation = toy
    tamper, needle = TAMPERS[name]
    bad = copy.deepcopy(record)
    tamper(bad)
    problems = _all_problems(config, raw, bad, validation, dimer.__getitem__, recompute=False)
    assert any(needle in problem for problem in problems), problems


def test_recomputation_catches_a_consistent_forgery(toy, dimer):
    """Both variances scaled together: every ratio, Neyman sum and status still
    re-derives, so only rebuilding the bank can see the forgery."""
    config, raw, record, validation = toy
    bad = copy.deepcopy(record)
    price = bad["banks"]["toy"]["protocols"]["qwc_groups"]
    for key in ("energy_variance_one_shot", "residual_variance_one_shot"):
        price[key] *= 0.25
    for key in ("energy_neyman_sum", "residual_neyman_sum"):
        price[key] *= 0.5
    assert _all_problems(config, raw, bad, validation, dimer.__getitem__,
                         recompute=False) == []
    problems = checker.recompute_problems(config, bad, validation, inputs=dimer.__getitem__)
    assert any("residual_variance_one_shot" in p for p in problems), problems


@pytest.mark.parametrize("field", ["numeric", "analytic"])
def test_recomputation_catches_a_moved_finite_difference_value(toy, dimer, field):
    """Half a tolerance moves no outcome, so only the rebuild can see it."""
    config, raw, record, validation = toy
    bad = copy.deepcopy(record)
    row = bad["banks"]["toy"]["finite_differences"]["functional"]
    row[field] += 0.5 * row["tolerance"]
    assert _all_problems(config, raw, bad, validation, dimer.__getitem__,
                         recompute=False) == []
    problems = checker.recompute_problems(config, bad, validation, inputs=dimer.__getitem__)
    assert any("functional finite difference recomputes" in p for p in problems), problems


def test_the_checker_refuses_an_unclean_record(tmp_path, toy, capsys):
    _, _, record, _ = toy
    path = tmp_path / "record.json"
    path.write_text(json.dumps(dict(record, provenance={"git_dirty": True})),
                    encoding="utf-8")
    assert checker.main(["--record", str(path), "--no-recompute"]) == 1
    assert "does not show a clean tree" in capsys.readouterr().out


def test_the_checker_refuses_a_missing_record(tmp_path, capsys):
    assert checker.main(["--record", str(tmp_path / "absent.json")]) == 1
    assert "missing record" in capsys.readouterr().out


# ------------------------------------------------------------------ the committed record

def test_the_committed_record_re_derives_without_a_rebuild():
    """The five-bank record, re-read under the checker without rebuilding a bank."""
    config = gate.load_config()
    record = json.loads(producer.RECORD.read_text(encoding="utf-8"))
    validation = json.loads(gate.VALIDATION.read_text(encoding="utf-8"))
    assert _all_problems(config, gate.CONFIG.read_bytes(), record, validation, None,
                         recompute=False) == []
    assert record["provenance"]["git_dirty"] is False
    assert record["decision"]["verdict"] == "INVALID"
    failed = {name: [check for check, ok in entry["deterministic_checks"].items() if not ok]
              for name, entry in record["banks"].items()}
    assert failed == {"h4": [], "h4_converged": [], "beh2": [], "h2o_cas8e6o": [],
                      "hubbard_2x2": ["linearization_matches_finite_differences"]}


def test_the_producer_refuses_to_overwrite_the_committed_record(capsys):
    assert producer.main([]) == 1
    assert "runs once" in capsys.readouterr().out


def test_the_hubbard_miss_is_truncation_at_the_frozen_step():
    """Post-hoc, not preregistered: the six-step sweep PLAN.md and
    REPRODUCING.md quote. The failed check's discrepancy scales as the step
    squared, so the analytic linearization is correct and the frozen step is
    too coarse on this bank; every step at or below half of it passes."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    config = gate.load_config()
    rule = config["linearization"]["finite_difference"]
    record = json.loads(producer.RECORD.read_text(encoding="utf-8"))
    recorded = record["banks"]["hubbard_2x2"]["finite_differences"]["functional"]
    model, selected = preflight_gate.bank_inputs("hubbard_2x2")
    bank = preflight_gate.first_moment_bank(model, selected)
    rows = gate.block_rows(bank, SecondMomentBank(bank), list(range(len(selected))))
    means = gate.reference_means(bank.reference, gate.row_words(rows))
    weights, _ = gate.residual_functional(rows, len(selected), means)
    weights = {int(word): float(value) for word, value in weights.items()}
    direction = gate.check_directions(weights, set(bank.word_set()),
                                      seed=int(rule["seed"]))["functional"]
    scale = record["banks"]["hubbard_2x2"]["estimator"]["cancellation_scale"]
    # step: (documented discrepancy, passes)
    documented = {2e-3: (1.41e-4, False), 1e-3: (3.53e-5, False), 5e-4: (8.82e-6, True),
                  2.5e-4: (2.20e-6, True), 1.25e-4: (5.51e-7, True), 6.25e-5: (1.37e-7, True)}
    assert float(rule["step"]) in documented
    misses = {}
    for step, (discrepancy, passes) in documented.items():
        row = gate.finite_difference_check(
            rows, len(selected), means, weights, direction, step=step,
            relative_tolerance=float(rule["relative_tolerance"]),
            cancellation_scale=scale, resolution=RESOLUTION)
        misses[step] = abs(row["numeric"] - row["analytic"])
        assert misses[step] == pytest.approx(discrepancy, rel=5e-3), step
        assert row["passes"] is passes, step
    assert misses[float(rule["step"])] == pytest.approx(
        abs(recorded["numeric"] - recorded["analytic"]), rel=1e-3)
    steps = sorted(documented, reverse=True)
    for coarse, fine in zip(steps, steps[1:]):
        assert misses[coarse] / misses[fine] == pytest.approx(4.0, rel=0.01), (coarse, fine)
