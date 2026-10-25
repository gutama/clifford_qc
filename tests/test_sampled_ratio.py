"""Actual joint shots, packed compilation and independent ratio inference."""
from dataclasses import replace
import math

import numpy as np
import pytest

from clifford_qc.backends.protocol import MeasurementBatch
from clifford_qc.dense_reference import to_matrix
from clifford_qc.measurement.compiled import CompiledMeasurementSampler
from clifford_qc.measurement.functionals import (
    WordFunctional, coupling_functional, quadratic_functional, ritz_functional,
)
from clifford_qc.measurement.ratio import (
    evaluate_ratio, hoeffding_radius, pilot_setting_plan, uniform_setting_plan,
)
from clifford_qc.reference_states import (
    DeterminantReference, SampledReference, UnsupportedReferenceOperation, require_exact_reference,
)
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.states import bell_density, ket_density
from clifford_qc.multivector import MV
from clifford_qc.subspace.projection import MatrixElementBank
from clifford_qc.subspace.providers import MeasurementProvider
from clifford_qc.subspace.sampled import SampledProjectedProvider


def sampled(rho):
    def draw(settings, counts, seed, identity):
        batch = CompiledMeasurementSampler(seed).sample_from_state(
            rho, settings, {s.key: count for s, count in zip(settings, counts)})
        return replace(batch, state_key=identity)
    return PreparedSampledReference(rho.n, "unit-fixture", draw)


def fixture():
    reference = sampled(bell_density())
    h = MV.from_terms(2, {"ZI": 0.7, "XX": -0.4, "YY": 0.1, "IZ": -0.2})
    generators = [("I", MV.scalar(2, 1)),
                  ("dressed", MV.from_terms(2, {"ZI": 1, "XI": 0.3j}))]
    return reference, h, generators


def forbid(*args, **kwargs):
    raise AssertionError("exact evaluation or row materialization was used")


def test_sampled_only_provider_runs_and_refuses_exact_evaluation(monkeypatch):
    reference, h, generators = fixture()
    assert isinstance(reference, SampledReference)
    with pytest.raises(UnsupportedReferenceOperation):
        require_exact_reference(reference)
    provider = SampledProjectedProvider(reference, h, generators)
    assert isinstance(provider, MeasurementProvider)
    with pytest.raises(UnsupportedReferenceOperation):
        provider.bank.matrices()
    monkeypatch.setattr(provider.bank, "overlap_operator", forbid)
    monkeypatch.setattr(provider.bank, "element_operator", forbid)
    for store in provider.bank._packed.values():
        monkeypatch.setattr(store, "materialize", forbid)
    pilot = reference.sample(provider.plan.settings,
                             uniform_setting_plan(provider.plan.settings, 20000), seed=1)
    frozen = provider.freeze_ratio(pilot, 50000)
    before = frozen.identity
    observations = reference.sample(frozen.settings, frozen.counts, seed=2)
    estimate = evaluate_ratio(frozen, observations)
    assert frozen.identity == before
    assert estimate.status == "resolved"
    assert estimate.total_shots == 70000
    assert estimate.evaluation_shots == 50000
    assert estimate.interval[0] <= estimate.estimate <= estimate.interval[1]
    assert estimate.denominator_interval[0] > frozen.denominator_floor
    with pytest.raises(ValueError):
        frozen.coefficients.flags.writeable = True
    with pytest.raises(TypeError):
        frozen.numerator.coefficients[99] = 0
    with pytest.raises(ValueError, match="independent"):
        evaluate_ratio(frozen, replace(observations, seed=1))
    with pytest.raises(ValueError, match="frozen"):
        evaluate_ratio(frozen, replace(observations, counts=(1,)*len(frozen.counts)))
    failed = evaluate_ratio(replace(frozen, denominator_floor=100), observations)
    assert failed.status == "denominator_unresolved" and failed.estimate is None
    assert failed.interval is None and failed.total_shots == 70000


@pytest.mark.parametrize("storage", ["object", "packed"])
def test_streamed_complex_quadratic_and_coupling_match_independent_matrices(storage, monkeypatch):
    h = MV.from_terms(2, {"ZI": 0.6, "XY": 0.8, "IX": -0.4})
    gens = [("I", MV.scalar(2, 1)),
            ("A", MV.from_terms(2, {"XZ": 0.3j, "YY": 0.7, "ZI": -0.4j})),
            ("B", MV.from_terms(2, {"XI": -0.2j, "ZY": 1}))]
    bank = MatrixElementBank(DeterminantReference("10"), h, gens, storage=storage)
    monkeypatch.setattr(bank, "overlap_operator", forbid)
    monkeypatch.setattr(bank, "element_operator", forbid)
    for store in bank._packed.values():
        monkeypatch.setattr(store, "materialize", forbid)
    # Complex state and coefficients expose conjugation, ordering and signs.
    psi = np.array([1, 2j, -0.4+0.6j, 0.2j]); psi /= np.linalg.norm(psi)
    c = np.array([0.3j, 0.7-0.2j])
    order = (1, 0)
    matrices = [to_matrix(bank.generator(i).mv) for i in order]
    b = sum(value * matrix for value, matrix in zip(c, matrices))
    mu = {code: np.vdot(psi, to_matrix(MV(2, {code: 1})) @ psi).real
          for code in range(16)}

    def evaluate(f):
        return f.constant + sum(value*mu[code] for code, value in f.coefficients.items())

    energy = -0.8
    s = quadratic_functional(bank, order, c, "overlap")
    numerator = quadratic_functional(bank, order, c, "element", storage=storage)
    assert evaluate(s) == pytest.approx(np.vdot(b@psi, b@psi).real, abs=1e-13)
    assert evaluate(numerator) == pytest.approx(
        np.vdot(b@psi, to_matrix(h)@b@psi).real, abs=1e-13)
    linear = ritz_functional(bank, order, c, energy)
    assert evaluate(linear) == pytest.approx(evaluate(numerator)-energy*evaluate(s), abs=1e-13)
    real, imag = coupling_functional(bank, order, c, energy, 2)
    expected = np.vdot(b@psi, (to_matrix(h)-energy*np.eye(4))@to_matrix(gens[2][1])@psi)
    assert complex(evaluate(real), evaluate(imag)) == pytest.approx(expected, abs=1e-13)


def test_ratio_uses_joint_covariance_and_valid_dependent_rectangle():
    from clifford_qc.measurement.planning import compile_block_measurement_plan
    from clifford_qc.measurement.cache import GroupedWordCache
    from clifford_qc.ir import PauliWord
    code = PauliWord.from_label("ZI").code
    plan = compile_block_measurement_plan(2, [code], 2, method="graph_state")
    reference = sampled(bell_density())
    cache = GroupedWordCache(2)
    obs = reference.sample(plan.settings, (100,), seed=5)
    cache.add_batch(obs.batch)
    h, s = WordFunctional({code: 2}, 3), WordFunctional({code: 1}, 2)
    assert h.covariance(s, cache) == pytest.approx(2*s.variance(cache))
    assert hoeffding_radius(s, cache, 0.05) == pytest.approx(math.sqrt(2*math.log(40)/100))
    assert hoeffding_radius(WordFunctional({}, 1), cache, 0.05) == 0
    counts = pilot_setting_plan(plan.settings, cache, [h-s], 31, minimum=3)
    assert counts == (31,)


def test_preparation_and_schedule_mismatches_fail_before_inference():
    reference, h, gens = fixture()
    provider = SampledProjectedProvider(reference, h, gens)
    settings = provider.plan.settings
    valid = reference.sample(settings, 10, seed=9)
    for batch in (replace(valid.batch, state_key="wrong"),
                  replace(valid.batch, circuits=1000),
                  replace(valid.batch, plus_counts={c: 0 for c in valid.batch.shots})):
        broken = PreparedSampledReference(2, "unit-fixture", lambda *a: batch)
        with pytest.raises(ValueError):
            broken.sample(settings, 10, seed=10)
    with pytest.raises(ValueError):
        reference.sample(settings, [1.2]*len(settings), seed=1)
    with pytest.raises(ValueError):
        reference.sample(settings, [True]*len(settings), seed=1)
    with pytest.raises(ValueError):
        reference.sample(settings, [1]*len(settings), seed=True)
    with pytest.raises(ValueError):
        uniform_setting_plan(settings, 1, minimum=2)
    with pytest.raises(ValueError):
        provider.reconstruct(replace(valid, reference_identity="wrong"))


def test_large_sampled_metadata_compilation_never_builds_density(monkeypatch):
    import clifford_qc.states as states
    monkeypatch.setattr(states, "ket_density", forbid)
    reference = PreparedSampledReference(256, "large-device-preparation", forbid)
    bank = MatrixElementBank(reference, MV(256, {3: 0.5}),
                             [("I", MV.scalar(256, 1))], coefficient_only=True, storage="packed")
    f = quadratic_functional(bank, (0,), np.array([1]), "element")
    assert dict(f.coefficients) == {3: 0.5}
    with pytest.raises(UnsupportedReferenceOperation):
        bank.solve()


def test_deterministic_algebraic_overlap_and_empty_measurement():
    ref = PreparedSampledReference(1, "empty", lambda s, c, seed, identity:
                                   MeasurementBatch(1, {}, {}, 0, state_key=identity))
    provider = SampledProjectedProvider(ref, MV.scalar(1, -2), [("I", MV.scalar(1, 1))])
    pilot = ref.sample((), (), seed=1)
    frozen = provider.freeze_ratio(pilot, 0)
    result = evaluate_ratio(frozen, ref.sample((), (), seed=2))
    assert result.estimate == -2 and result.interval == (-2, -2)
    assert result.denominator_interval == (1, 1) and result.total_shots == 0


def test_joint_ratio_cancellation_and_small_budget_failures():
    reference, h, gens = fixture()
    provider = SampledProjectedProvider(reference, h, gens)
    pilot = reference.sample(provider.plan.settings, 200, seed=71)
    frozen = provider.freeze_ratio(pilot, 5000)
    denominator = frozen.denominator
    paired = replace(frozen, numerator=WordFunctional(
        {code: 2*value for code, value in denominator.coefficients.items()},
        2*denominator.constant))
    result = evaluate_ratio(paired, reference.sample(paired.settings, paired.counts, seed=72))
    assert result.estimate == pytest.approx(2)
    assert result.delta_variance == pytest.approx(0, abs=1e-14)
    with pytest.raises(ValueError, match="minimum"):
        provider.freeze_ratio(pilot, 1)
    # Rank/solve failure remains explicit rather than requesting an exact fallback.
    with pytest.raises(ValueError):
        provider.freeze_ratio(pilot, 5000, tau_s=100)


def test_independent_simulator_signed_readout_law_matches_tensor_paulis():
    from benchmarks.run_sampled_ratio import JointShotSimulator, dense_word
    from clifford_qc.measurement.planning import compile_block_measurement_plan
    from clifford_qc.ir import PauliWord
    psi = np.array([1, 2j, 0.3-0.2j, -0.4]); psi /= np.linalg.norm(psi)
    codes = [PauliWord.from_label(label).code for label in ("XX", "YY", "ZZ", "XI", "YZ")]
    plan = compile_block_measurement_plan(2, codes, 2, method="graph_state")
    simulator = JointShotSimulator(psi, 2)
    for setting in plan.settings:
        probabilities = simulator.probabilities(setting)
        assert probabilities.sum() == pytest.approx(1)
        for code in setting.assigned_word_codes:
            sign, positions = setting.readouts[code]
            measured = sum(probability*sign*(-1)**sum(format(i, "02b")[p] == "1" for p in positions)
                           for i, probability in enumerate(probabilities))
            assert measured == pytest.approx(np.vdot(psi, dense_word(2, code)@psi).real, abs=1e-13)


def test_binomial_bounds_and_unconditional_failure_denominator():
    from benchmarks.run_sampled_ratio import exact_binomial_lower, summarize, verdict
    assert exact_binomial_lower(0, 128, 0.05) == 0
    assert exact_binomial_lower(128, 128, 0.05) == pytest.approx(0.05**(1/128))
    config = {"cases": [{"name": "unit", "role": "development_primary"}],
              "total_shot_budgets": [100], "target_estimation_abs_error": 0.05,
              "target_success_probability": 0.9, "success_probability_alpha": 0.05}
    rows = []
    for arm in ("ratio", "bank"):
        for status, error in (("resolved", 0.01), ("denominator_unresolved", None),
                              ("pilot_failed", None), ("resource_failed", None)):
            rows.append({"case": "unit", "budget": 100, "arm": arm, "status": status,
                         "estimation_error": error, "retained_space_error": error,
                         "ground_error": error, "end_to_end_seconds": 1,
                         "device_seconds": 0.9, "classical_seconds": 0.1, "total_shots": 100})
    summaries = summarize(rows, config)
    assert all(s["attempts"] == 4 and s["successes"] == 1 and s["success_probability"] == 0.25
               and s["pilot_failures"] == s["denominator_failures"] == s["resource_failures"] == 1
               for s in summaries)
    assert verdict(summaries, config)["status"] == "closed_on_declared_scope"


def test_array_admission_precedes_allocation_and_retains_all_attempts(monkeypatch):
    from benchmarks import run_sampled_ratio as runner
    config = runner.load_declaration()
    config["owned_array_cap_bytes"] = 1
    monkeypatch.setattr(runner, "build_case", forbid)
    result = runner.run(config, "unit-execution", "unit-declaration")
    expected = len(config["cases"])*len(config["total_shot_budgets"])*config["replicas"]*2
    assert len(result["rows"]) == expected
    assert all(row["status"] == "resource_failed" and row["total_shots"] == 0
               for row in result["rows"])
    assert result["verdict"]["status"] == "closed_on_declared_scope"


@pytest.mark.parametrize("mutation", ["source", "shots", "summary", "verdict"])
def test_published_record_checker_rejects_tampering(mutation):
    import json
    from benchmarks.check_sampled_ratio import check
    from benchmarks.run_sampled_ratio import ROOT, RESULT
    record = json.loads((ROOT/RESULT).read_text())
    if mutation == "source":
        record["source_sha256"]["clifford_qc/measurement/ratio.py"] = "0"*64
    elif mutation == "shots":
        record["rows"][0]["total_shots"] += 1
    elif mutation == "summary":
        record["summaries"][0]["successes"] += 1
    else:
        record["verdict"]["status"] = "promote_on_declared_scope"
    with pytest.raises(AssertionError):
        check(record)
