"""Measured-solve response, complete shot ledgers and hard acquisition limits."""

from pathlib import Path

import numpy as np
import pytest

from clifford_qc import X, Z, ket_density
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.backends.protocol import GroupSample, MeasurementBatch, state_fingerprint
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import DeviceCard
from clifford_qc.measurement.cache import GroupedWordCache
from clifford_qc.measurement.functionals import WordFunctional
from clifford_qc.measurement.grouping import shared_basis
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (
    Generator, IncrementalCostModel, MatrixElementBank, SharedMeasurement,
    acquire_to_accuracy, expected_lowering_estimate, identity_generator,
    krylov_response, pauli_orbit, ritz_uncertainty,
)

ROOT = Path(__file__).resolve().parents[1]


class RecordedBackend(FiniteShotBackend):
    def __init__(self, seed=17):
        super().__init__(seed=seed)
        self.spent = []

    def sample_grouped_from_state(self, rho, groups, shots):
        batch = super().sample_grouped_from_state(rho, groups, shots)
        self.spent.append(batch.hardware_shots)
        return batch


@pytest.fixture
def spin_problem():
    H = tfim(3, J=1.0, h=1.0).hamiltonian
    return (ket_density(3, "000"), H,
            krylov_response(H, 3) + pauli_orbit([w for w, _ in H.items()])[:5])


def run(spin_problem, backend=None, **kwargs):
    return acquire_to_accuracy(
        *spin_problem, backend if backend is not None else RecordedBackend(),
        **{"target_accuracy": 1e-5, "pilot_shots": 60, "max_steps": 5,
           "max_total_shots": 100_000, **kwargs})


def assert_current_estimate(found):
    session = SharedMeasurement(found.bank, found.indices, pooling=found.cache.pooling)
    fresh = session.solve(found.cache)
    interval = ritz_uncertainty(session, found.cache, found.result)
    assert found.result.resources["shots"] == found.cache.total_shots == found.total_shots
    assert found.energy == pytest.approx(fresh.ground_energy, abs=1e-12)
    assert found.ci_interval == interval
    assert sum(step.shots_added for step in found.history) == found.total_shots
    assert sum(step.circuits_added for step in found.history) == found.total_circuits
    assert sum(step.time_us_added for step in found.history) == pytest.approx(found.total_time_us)


@pytest.mark.parametrize("pooling", ["shots", "assigned"])
def test_screening_measurements_are_included_before_reporting(spin_problem, pooling):
    found = run(spin_problem, max_steps=1, pooling=pooling)
    assert found.total_shots > found.history[0].cumulative_shots
    assert_current_estimate(found)


def test_a_basis_limit_cannot_erase_unassessed_bias(spin_problem):
    found = run(spin_problem, max_basis_size=2, target_accuracy=0.1)
    assert not found.converged
    assert "basis budget" in found.stopped_reason
    assert found.total_error_estimate > found.target_accuracy
    assert found.estimated_remaining_bias > 0.0
    assert_current_estimate(found)


def test_a_basis_limit_keeps_bias_when_all_candidate_words_are_measured():
    H = tfim(2, J=1.0, h=1.0).hamiltonian
    pool = pauli_orbit([PauliWord(2, code) for code in range(1, 16)])
    found = acquire_to_accuracy(
        ket_density(2, "00"), H, pool, RecordedBackend(), target_accuracy=0.1,
        max_basis_size=2, pilot_shots=5000, max_total_shots=100_000)
    assert not found.converged
    assert "basis budget" in found.stopped_reason
    assert np.isfinite(found.estimated_remaining_bias)
    assert found.estimated_remaining_bias > found.target_accuracy
    assert_current_estimate(found)


@pytest.mark.parametrize("shots", [9, 100, 540, 700, 1000])
def test_pilot_screening_and_refinement_stay_within_the_shot_cap(spin_problem, shots):
    backend = RecordedBackend()
    found = run(spin_problem, backend, max_total_shots=shots)
    assert sum(backend.spent) == found.total_shots <= shots
    assert not found.converged
    assert_current_estimate(found)


@pytest.mark.parametrize("shots", [1, 8])
def test_an_unfunded_initial_pilot_is_rejected_before_sampling(spin_problem, shots):
    backend = RecordedBackend()
    with pytest.raises(ValueError, match="initial pilot.*budget"):
        run(spin_problem, backend, max_total_shots=shots)
    assert backend.spent == []


@pytest.fixture
def timed_cost():
    card = DeviceCard.load(ROOT / "benchmarks/configs/device_cards/superconducting-like.json")
    return IncrementalCostModel(time_weight_us=1.0, device_card=card)


def test_an_unfunded_time_budget_is_rejected_before_sampling(spin_problem, timed_cost):
    backend = RecordedBackend()
    with pytest.raises(ValueError, match="initial pilot.*hardware time budget"):
        run(spin_problem, backend, cost_model=timed_cost, max_total_time_us=1.0)
    assert backend.spent == []


@pytest.mark.parametrize("time_us", [30.0, 1700.0, 2000.0, 2500.0])
def test_every_batch_respects_the_remaining_hardware_time(spin_problem, timed_cost, time_us):
    backend = RecordedBackend()
    found = run(spin_problem, backend, cost_model=timed_cost, max_total_time_us=time_us)
    assert found.total_time_us <= time_us
    assert sum(backend.spent) == found.total_shots
    assert not found.converged
    assert_current_estimate(found)


def test_near_degenerate_basis_uncertainty_matches_repeated_measurements():
    H = 0.01 * Z(2, 0) + X(2, 1) + X(2, 0) * X(2, 1)
    bank = MatrixElementBank(ket_density(2, "00"), H)
    basis = list(bank.extend([identity_generator(2), Generator("X0", X(2, 0))]))
    candidate = bank.add(Generator("X1", X(2, 1)))
    full = SharedMeasurement(bank, [*basis, candidate], pooling="shots")
    subspace = SharedMeasurement(bank, basis, groups=full.groups, pooling="shots")
    values, reported = [], []
    for seed in range(200):
        cache = full.measure(FiniteShotBackend(seed=seed), 100_000_000)
        point = subspace.solve(cache)
        lowering, std, *_ = expected_lowering_estimate(bank, basis, point, candidate, cache=cache)
        values.append(lowering)
        reported.append(std)
    empirical = np.std(values, ddof=1)
    assert np.median(reported) == pytest.approx(empirical, rel=0.2)


def test_a_degenerate_measured_root_has_unresolved_lowering_uncertainty():
    H = X(2, 1) + X(2, 0) * X(2, 1)
    rho = ket_density(2, "00")
    bank = MatrixElementBank(rho, H)
    basis = list(bank.extend([identity_generator(2), Generator("X0", X(2, 0))]))
    candidate = bank.add(Generator("X1", X(2, 1)))
    full = SharedMeasurement(bank, [*basis, candidate], pooling="shots")
    groups = []
    for group in full.groups:
        bdict = shared_basis(group)
        groups.append(GroupSample(
            support=tuple(sorted(bdict)), basis=tuple(sorted(bdict.items())),
            hist={"00": 25, "01": 25, "10": 25, "11": 25}, shots=100,
            word_codes=tuple(word.code for word in group)))
    cache = GroupedWordCache(2, pooling="shots")
    cache.add_batch(MeasurementBatch(n=2, shots={}, plus_counts={}, circuits=len(groups),
                                     groups=tuple(groups), state_key=state_fingerprint(rho)))
    point = SharedMeasurement(bank, basis, groups=full.groups, pooling="shots").solve(cache)
    assert point.energies[0] == point.energies[1]
    _, std, *_ = expected_lowering_estimate(bank, basis, point, candidate, cache=cache)
    assert np.isinf(std)


@pytest.mark.parametrize("pencil", ["real", "complex", "truncated"])
def test_uncertainty_matches_finite_differences_of_the_complete_measured_estimator(pencil):
    H = tfim(3, J=1.0, h=1.0).hamiltonian
    bank = MatrixElementBank(ket_density(3, "000"), H)
    generators = [identity_generator(3), *krylov_response(H, 1)]
    candidate_generator = krylov_response(H, 2)[1]
    if pencil != "real":
        complex_operator = X(3, 0) + 1j * X(3, 1)
        generators = [identity_generator(3), Generator("complex", complex_operator)]
        candidate_generator = Generator("candidate", X(3, 2) + Z(3, 0))
        if pencil == "truncated":
            generators.append(Generator("dependent", 2 * complex_operator + generators[0].mv))
    basis = list(bank.extend(generators))
    candidate = bank.add(candidate_generator)
    full = SharedMeasurement(bank, [*basis, candidate], pooling="shots")
    subspace = SharedMeasurement(bank, basis, groups=full.groups, pooling="shots")
    cache = full.measure(FiniteShotBackend(seed=12), 10_000)
    solve_kwargs = {"tau_s": 0.01} if pencil == "truncated" else {}
    point = subspace.solve(cache, **solve_kwargs)
    assert point.effective_rank == (2 if pencil == "truncated" else len(basis))
    _, std, _, orth_std, _ = expected_lowering_estimate(bank, basis, point, candidate, cache=cache)
    estimate = cache.candidate_estimate
    low_gradient, orth_gradient = {}, {}
    step = 1e-6
    try:
        for word in full.words:
            values = []
            for sign in (-1, 1):
                cache.candidate_estimate = lambda coefficients, s=sign, c=word.code: (
                    estimate(coefficients) + s * step * coefficients.get(c, 0.0))
                perturbed = subspace.solve(cache, **solve_kwargs)
                values.append(expected_lowering_estimate(
                    bank, basis, perturbed, candidate, cache=cache))
            low_gradient[word.code] = (values[1][0] - values[0][0]) / (2 * step)
            orth_gradient[word.code] = (values[1][2] - values[0][2]) / (2 * step)
    finally:
        cache.candidate_estimate = estimate
    assert std == pytest.approx(np.sqrt(WordFunctional(low_gradient).variance(cache)), rel=1e-4)
    assert orth_std == pytest.approx(np.sqrt(WordFunctional(orth_gradient).variance(cache)), rel=1e-4)


# ------------------------------------------------- review fixes for PR #177

def _code(mv):
    [code] = mv.terms
    return code


def test_words_registered_after_a_setting_are_visible_to_the_allocator():
    from clifford_qc.measurement.allocation import variance_optimal_group_plan
    from clifford_qc.subspace.information import _IncrementalGroupRegistry

    x0x1, x1 = _code(X(2, 0) * X(2, 1)), _code(X(2, 1))
    registry = _IncrementalGroupRegistry(2)
    [group] = registry.register_words([x0x1])
    assert registry.register_words([x1]) == []
    cache = GroupedWordCache(2, pooling="shots")
    cache.add_batch(FiniteShotBackend(seed=3).sample_grouped_from_state(
        ket_density(2, "00"), registry.groups, 200))
    target = [WordFunctional({x1: 1.0})]
    # The group was opened with X0X1 alone; X1 is read by it but not listed.
    assert variance_optimal_group_plan(registry.groups, cache, target, 50) == {}
    plan = variance_optimal_group_plan(registry.reader_groups(), cache, target, 50)
    assert plan == {x0x1: 50, x1: 50}
    # The reader view changes nothing about the registered readout.
    assert [[w.code for w in g] for g in registry.groups] == [[x0x1]]


def test_a_word_is_owned_by_one_setting_even_when_several_read_it():
    from clifford_qc.subspace.information import _IncrementalGroupRegistry

    x0, x1 = _code(X(2, 0)), _code(X(2, 1))
    x0z1 = _code(X(2, 0) * Z(2, 1))
    registry = _IncrementalGroupRegistry(2)
    registry.register_words([x0z1, x1])  # two settings; the first also reads X0
    registry.register_words([x0])
    registry.register_words([x1, x0])  # re-registering must not move or copy a word
    listed = [w.code for group in registry.reader_groups() for w in group]
    assert sorted(listed) == sorted({x0, x1, x0z1})


def test_a_session_never_registers_an_unsampled_setting():
    from clifford_qc.subspace.information import _IncrementalGroupRegistry

    x0, z0 = _code(X(2, 0)), _code(Z(2, 0))
    registry = _IncrementalGroupRegistry(2)
    registry.register_words([x0])
    assert [[w.code for w in g] for g in registry.session_groups([x0])] == [[x0]]
    with pytest.raises(ValueError, match="not registered"):
        registry.session_groups([x0, z0])
    assert len(registry.groups) == 1


def _score(index, *, score, lowering, rejected=None):
    from clifford_qc.subspace.information import InformationScore

    return InformationScore(
        index=index, label=f"c{index}", residual_coupling=0.0,
        predicted_lowering=lowering, lowering_std=0.0, risk_adjusted_lowering=lowering,
        orthogonal_fraction=1.0, orthogonal_std=0.0, new_words=0, incremental_groups=0,
        incremental_time_us=0.0, classical_products=0, classical_terms=0,
        incremental_cost=0.0, score=score, rejected=rejected)


def test_the_lowering_threshold_filters_before_the_ratio_argmax():
    from clifford_qc.subspace.information import select_information_candidate

    cheap = _score(0, score=5e-5, lowering=5e-11)  # no new words, negligible lowering
    costly = _score(1, score=3.3e-5, lowering=1e-2)  # 300 new words, 10 mHa
    assert select_information_candidate([cheap, costly]).index == 0
    assert select_information_candidate([cheap, costly], min_lowering=1e-6).index == 1
    assert select_information_candidate([cheap], min_lowering=1e-6) is None
    rejected = _score(1, score=1.0, lowering=1.0, rejected="orthogonal fraction too small")
    assert select_information_candidate([cheap, rejected], min_lowering=1e-6) is None
    with pytest.raises(ValueError):
        select_information_candidate([cheap], lowering="score")


def test_a_cheap_negligible_candidate_does_not_end_the_resource_aware_run(spin_problem,
                                                                          monkeypatch):
    from clifford_qc.subspace import information

    rho, H, candidates = spin_problem
    bank = MatrixElementBank(rho, H)
    pool = bank.extend(candidates)

    def fabricated(bank, basis, result, idx, **kwargs):
        if idx == pool[0]:
            return _score(idx, score=5e-5, lowering=5e-11)
        if idx == pool[1]:
            return _score(idx, score=3.3e-5, lowering=1e-2)
        return _score(idx, score=0.0, lowering=0.0, rejected="orthogonal fraction too small")

    monkeypatch.setattr(information, "score_information_candidate", fabricated)
    found = information.run_resource_aware_acase(
        rho, H, candidates, bank=bank, cost_model=IncrementalCostModel(), max_size=1,
        min_lowering=1e-6)
    assert found.indices[-1] == pool[1]
    assert [record.selected_label for record in found.records] == [f"c{pool[1]}"]
    assert found.stopped_reason == "basis budget reached"


def test_both_variants_certify_the_retained_sector_like_run_acase(monkeypatch):
    from clifford_qc.models.lattice import hubbard
    from clifford_qc.subspace import (
        determinant_excitations, occupied_spin_orbitals, run_dynamic_commutator_acase,
        run_resource_aware_acase,
    )
    from clifford_qc.subspace import information

    model = hubbard(2, t=1.0, U=4.0)
    occupied = occupied_spin_orbitals(model)
    reference = ket_density(model.n, "".join(
        "1" if q in occupied else "0" for q in range(model.n)))
    pool = determinant_excitations(model.n, occupied)
    H = model.hamiltonian
    kwargs = dict(leakage_tol=1e-8, leakage_mode="reference", max_size=2)
    resource = run_resource_aware_acase(
        reference, H, pool, cost_model=IncrementalCostModel(), **kwargs)
    dynamic = run_dynamic_commutator_acase(reference, H, pool, max_depth=1, **kwargs)
    for found in (resource, dynamic):
        certificate = found.resources["sector_certificate"]
        assert certificate["max_sector_leakage"] <= 1e-8
    # Operator-mode leakage needs no span certificate, as in run_acase.
    plain = run_resource_aware_acase(
        reference, H, pool, cost_model=IncrementalCostModel(), max_size=2,
        leakage_tol=1e-8)
    assert "sector_certificate" not in plain.resources
    # A span that leaks more than its candidates did is an error, not a result.
    monkeypatch.setattr(information, "subspace_sector_certificate",
                        lambda *args, **kw: {"max_sector_leakage": 1.0})
    with pytest.raises(ValueError, match="sector certificate"):
        run_resource_aware_acase(
            reference, H, pool, cost_model=IncrementalCostModel(), **kwargs)
    with pytest.raises(ValueError, match="sector certificate"):
        run_dynamic_commutator_acase(reference, H, pool, max_depth=1, **kwargs)


def test_a_word_another_setting_already_reads_costs_nothing():
    from clifford_qc.subspace import estimate_incremental_cost

    # With H = X0X1 the candidate X1 needs the words X0, X1 and X0X1 only.
    bank = MatrixElementBank(ket_density(2, "00"), X(2, 0) * X(2, 1))
    basis = bank.extend([identity_generator(2)])
    [candidate] = bank.extend([Generator("X1", X(2, 1))])
    # Settings X0 and X0X1: X1 is compatible with the first but read by the second.
    settings = (_code(X(2, 0)), _code(X(2, 0) * X(2, 1)))
    found = estimate_incremental_cost(
        bank, basis, candidate, cost_model=IncrementalCostModel(setting_weight=1.0),
        basis_words=frozenset(), qwc_bases=settings)
    new_codes = set()
    for index in (*basis, candidate):
        bank.update_pair_support(new_codes, index, candidate)
    assert all(any(information_reads(code, setting) for setting in settings)
               for code in new_codes - {0})
    assert found.widened_groups == 0
    assert found.incremental_groups == 0


def information_reads(code, setting):
    from clifford_qc.subspace.information import _qwc_reads_word

    return _qwc_reads_word(2, code, setting)


def test_a_time_budget_too_small_for_one_shot_is_named_as_the_stop(spin_problem, timed_cost):
    from clifford_qc.measurement.cost import setting_duration_us
    from clifford_qc.subspace.information import basis_dict_to_code, qwc_setting_resources

    pilot = run(spin_problem, cost_model=timed_cost, max_steps=0, min_lowering=1e9,
                max_basis_size=8)
    durations = [setting_duration_us(timed_cost.device_card, qwc_setting_resources(
        3, basis_dict_to_code(dict(group["key"]))))
        for group in pilot.cache.group_states()]
    budget = pilot.total_time_us + 0.5 * min(durations)
    found = run(spin_problem, cost_model=timed_cost, min_lowering=1e9, max_basis_size=8,
                max_total_time_us=budget)
    assert found.total_time_us == pytest.approx(pilot.total_time_us)
    assert found.stopped_reason == "hardware time budget reached"


def _measured_basis(seed=5):
    H = tfim(3, J=1.0, h=1.0).hamiltonian
    bank = MatrixElementBank(ket_density(3, "000"), H)
    basis = list(bank.extend([identity_generator(3)] + krylov_response(H, 2)))
    candidates = bank.extend(pauli_orbit([w for w, _ in H.items()])[:3])
    full = SharedMeasurement(bank, [*basis, *candidates], pooling="shots")
    cache = full.measure(FiniteShotBackend(seed=seed), 4000)
    point = SharedMeasurement(bank, basis, groups=full.groups, pooling="shots").solve(cache)
    return bank, basis, candidates, cache, point


def test_a_candidate_column_is_built_once_per_score(monkeypatch):
    from clifford_qc.subspace import information

    bank, basis, candidates, cache, point = _measured_basis()
    reference = expected_lowering_estimate(bank, basis, point, candidates[0], cache=cache)
    calls = []
    real = information.entry_functionals
    monkeypatch.setattr(information, "entry_functionals",
                        lambda *args: calls.append(args[1:]) or real(*args))
    assert expected_lowering_estimate(
        bank, basis, point, candidates[0], cache=cache) == reference
    # One (candidate, candidate) entry plus one per basis row, whatever the rank.
    assert point.coefficients.shape[1] > 1
    assert len(calls) == len(basis) + 1
    assert len(set(calls)) == len(calls)


def test_the_basis_response_is_computed_once_per_solve(monkeypatch):
    from clifford_qc.subspace import information

    bank, basis, candidates, cache, point = _measured_basis()
    information._RESPONSE_MEMO.clear()
    computed = []
    real = information._basis_response
    monkeypatch.setattr(information, "_basis_response",
                        lambda *args: computed.append(1) or real(*args))
    first = [expected_lowering_estimate(bank, basis, point, c, cache=cache) for c in candidates]
    scores = [information.score_information_candidate(
        bank, basis, point, c, cost_model=IncrementalCostModel(), cache=cache)
        for c in candidates]
    assert len(computed) == 1
    accepted = [(score, row) for score, row in zip(scores, first) if score.accepted]
    assert accepted
    assert all(score.predicted_lowering == row[0] for score, row in accepted)
    # Another solve, or the same solve against changed outcomes, is not a hit.
    other_bank, other_basis, other_candidates, other_cache, other_point = _measured_basis(seed=6)
    expected_lowering_estimate(other_bank, other_basis, other_point, other_candidates[0],
                               cache=other_cache)
    assert len(computed) == 2
    cache.add_batch(FiniteShotBackend(seed=9).sample_grouped_from_state(
        bank.reference, SharedMeasurement(bank, basis, pooling="shots").groups, 10))
    with pytest.raises(ValueError):
        expected_lowering_estimate(bank, basis, point, candidates[0], cache=cache)
    assert len(computed) == 3
