"""Finite-shot uncertainty for the complete nonlinear response pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend, FiniteShotBackend
from clifford_qc.fermion import c_op
from clifford_qc.models import load_effective_hamiltonian, magnetization
from clifford_qc.subspace import (
    HEURISTIC,
    MatrixElementBank,
    ResponseMeasurement,
    bootstrap_response,
    determinant_excitations,
    identity_generator,
    lehmann_spectrum,
    occupied_spin_orbitals,
    static_susceptibility,
)

DATA = (Path(__file__).parents[1] / "examples" / "data"
        / "wannier_hubbard_dimer.json")


@pytest.fixture()
def dimer_response():
    model = load_effective_hamiltonian(DATA)
    rho = ExactMVBackend().state(model.reference, ())
    candidates = determinant_excitations(
        model.n, occupied_spin_orbitals(model))
    bank = MatrixElementBank(
        rho, model.hamiltonian, [identity_generator(model.n), *candidates])
    staggered = magnetization(model, 0) + (-1.0) * magnetization(model, 1)
    return bank, staggered, ResponseMeasurement(
        bank, staggered, label="staggered_spin")


def test_response_measurement_infinite_shot_limit_matches_exact_route(dimer_response):
    bank, staggered, measurement = dimer_response
    exact = bank.solve()
    expected_lines = lehmann_spectrum(exact, staggered, min_weight=1e-12)
    reconstructed = measurement.exact_spectrum(min_weight=1e-12)

    assert np.allclose(
        measurement.exact_observable_matrix(),
        bank.project_observable(staggered),
        atol=1e-12,
    )
    assert reconstructed.result.energies == pytest.approx(
        exact.energies, abs=1e-12)
    assert len(reconstructed.lines) == len(expected_lines) == 1
    assert reconstructed.lines[0].excitation_energy == pytest.approx(
        expected_lines[0].excitation_energy, abs=1e-12)
    assert reconstructed.lines[0].weight == pytest.approx(
        expected_lines[0].weight, abs=1e-12)
    assert reconstructed.susceptibility == pytest.approx(
        static_susceptibility(expected_lines), abs=1e-12)


def test_grouped_bootstrap_reruns_the_whole_response_pipeline(dimer_response):
    _, _, measurement = dimer_response
    cache = measurement.measure(FiniteShotBackend(seed=123), 4000)
    omega = np.linspace(0.0, 2.0, 41)
    uncertainty = bootstrap_response(
        measurement,
        cache,
        replicates=40,
        seed=9,
        min_weight=1e-3,
        frequencies=omega,
        broadening=0.1,
    )

    assert uncertainty.evidence == HEURISTIC
    assert not uncertainty.certified
    assert uncertainty.replicates_succeeded == 40
    assert uncertainty.rank_failures == 0
    assert uncertainty.root_collision_failures == 0
    assert uncertainty.solver_failures == 0
    assert len(uncertainty.lines) == 1
    line = uncertainty.lines[0]
    assert not line.certified
    for interval in (
            line.excitation_energy, line.weight, uncertainty.susceptibility):
        assert interval.evidence == HEURISTIC
        assert not interval.certified
        assert np.isfinite([interval.estimate, interval.lower, interval.upper]).all()
        assert interval.lower < interval.upper
    assert uncertainty.broadened_estimate.shape == omega.shape
    assert uncertainty.broadened_lower.shape == omega.shape
    assert uncertainty.broadened_upper.shape == omega.shape
    assert np.all(uncertainty.broadened_lower <= uncertainty.broadened_upper)
    assert uncertainty.spectrum.resources["response_word_universe"] == 63
    assert uncertainty.spectrum.resources["response_qwc_groups"] == 25
    assert uncertainty.spectrum.resources["shots"] == 100_000


def test_response_measurement_refuses_nonhermitian_observable(dimer_response):
    bank, _, _ = dimer_response
    with pytest.raises(ValueError, match="Hermitian"):
        ResponseMeasurement(bank, c_op(bank.n, 0))


@pytest.mark.parametrize("kwargs, message", [
    ({"replicates": 1}, "at least two"),
    ({"delta": 0.0}, "delta"),
    ({"frequencies": [0.0, 1.0]}, "supplied together"),
    ({"minimum_success_fraction": 0.0}, "success_fraction"),
])
def test_response_bootstrap_validates_its_statistical_contract(
        dimer_response, kwargs, message):
    _, _, measurement = dimer_response
    cache = measurement.measure(FiniteShotBackend(seed=3), 100)
    with pytest.raises(ValueError, match=message):
        bootstrap_response(measurement, cache, **kwargs)
