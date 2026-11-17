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
