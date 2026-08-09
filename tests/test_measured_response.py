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
    BootstrapResponse,
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


def test_intervals_report_what_they_are_conditioned_on(dimer_response):
    """Rejecting replicas is not neutral, so the conditioning must be legible.

    Replicas whose rank moved or whose roots collided are dropped, which removes
    exactly the draws that would have widened the band -- the interval is
    therefore narrowest where the pipeline is least stable. The failure counts
    alone leave that to be inferred; ``acceptance_rate`` states it.
    """
    _, _, measurement = dimer_response
    cache = measurement.measure(FiniteShotBackend(seed=5), 4000)
    result = bootstrap_response(measurement, cache, replicates=40, seed=3)

    assert result.acceptance_rate == pytest.approx(
        result.replicates_succeeded / result.replicates_requested)
    assert 0.0 <= result.acceptance_rate <= 1.0
    # the accounting closes: every requested replica either succeeded or is
    # counted under exactly one failure mode
    assert (result.replicates_succeeded + result.rank_failures
            + result.root_collision_failures
            + result.solver_failures) == result.replicates_requested
    # and the conditioning is stated where a reader meets the interval
    assert "conditional" in BootstrapResponse.__doc__
    assert "acceptance_rate" in BootstrapResponse.__doc__
    assert result.certified is False


def test_replica_census_fingerprints_every_replica(dimer_response):
    """Totals cannot localize a census disagreement; the census can.

    Two environments that accept different numbers of replicas produce the
    same pair of totals whatever the cause. The per-replica record carries the
    eigenvalue whose sign decided each acceptance, so the replicas that moved
    -- and whether each was a marginal flip or a wholesale shift -- are
    recoverable by diffing two records.
    """
    _, _, measurement = dimer_response
    cache = measurement.measure(FiniteShotBackend(seed=5), 4000)
    result = bootstrap_response(measurement, cache, replicates=40, seed=3)

    census = result.replica_census
    assert len(census) == result.replicates_requested
    # draw order, no gaps: a diff against another run aligns index by index
    assert [item.index for item in census] == list(range(len(census)))

    outcomes = [item.outcome for item in census]
    assert set(outcomes) <= {"accepted", "rank", "root_collision", "solver"}
    # the census reproduces the aggregate counters rather than restating them
    assert outcomes.count("accepted") == result.replicates_succeeded
    assert outcomes.count("rank") == result.rank_failures
    assert outcomes.count("root_collision") == result.root_collision_failures
    assert outcomes.count("solver") == result.solver_failures

    for item in census:
        if item.outcome == "solver":
            # a solve that raised has no pencil to report
            assert item.overlap_eigenvalue_min is None
            assert item.effective_rank is None
            assert item.rank_decision_margin is None
            assert item.controlling_mode is None
            continue
        assert isinstance(item.overlap_eigenvalue_min, float)
        assert isinstance(item.effective_rank, int)
        # the decision itself, not a proxy for it: retention is
        # `value > cutoff` mode by mode, so the margin and the cutoff that
        # applied are what a later run must be diffed against
        assert isinstance(item.rank_decision_margin, float)
        assert isinstance(item.overlap_threshold, float)
        assert isinstance(item.controlling_mode, int)
        assert 0 <= item.controlling_mode < item.effective_rank + 4
    accepted = [item for item in census if item.outcome == "accepted"]
    # every accepted replica matched the point estimate's rank, by definition
    # of the gate -- so the census cannot silently disagree with it
    point_rank = result.spectrum.result.effective_rank
    assert all(item.effective_rank == point_rank for item in accepted)


def test_acceptance_rate_is_defined_when_nothing_was_requested():
    empty = BootstrapResponse(
        spectrum=None, lines=(), susceptibility=None, evidence=HEURISTIC,
        delta=0.05, replicates_requested=0, replicates_succeeded=0,
        rank_failures=0, root_collision_failures=0, solver_failures=0)
    assert empty.acceptance_rate == 0.0


def test_rank_decision_margin_tracks_the_cutoff_not_the_sign(dimer_response):
    """A positive smallest eigenvalue below its cutoff is still dropped.

    The census would be misleading if it recorded only ``lambda_min``: modes
    are retained against ``value > cutoff``, so raising the cutoff above a
    positive eigenvalue drops that mode. The recorded margin must follow the
    cutoff, and go negative exactly when the mode is rejected while the
    eigenvalue itself stays positive.
    """
    _, _, measurement = dimer_response
    cache = measurement.measure(FiniteShotBackend(seed=5), 4000)

    relaxed = measurement.spectrum(cache, initial_state=0, min_weight=1e-3)
    values = relaxed.result.overlap_eigenvalues  # descending
    lambda_min = relaxed.result.resources["overlap_eigenvalue_min"]
    assert lambda_min > 0.0
    assert lambda_min == pytest.approx(values[-1])

    # a cutoff between the smallest mode and the next one: the eigenvalue is
    # unchanged and still positive, but the mode is now excluded
    cutoff = 0.5 * (values[-1] + values[-2])
    strict = measurement.spectrum(cache, initial_state=0, min_weight=1e-3,
                                  tau_s=cutoff)
    assert strict.result.resources["overlap_eigenvalue_min"] == pytest.approx(
        lambda_min)
    assert strict.result.effective_rank == relaxed.result.effective_rank - 1

    margins = strict.result.resources["overlap_decision_margin_per_mode"]
    thresholds = strict.result.resources["overlap_threshold_per_mode"]
    # descending order, so the smallest mode is last, and it is below cutoff
    assert margins[-1] < 0.0
    assert thresholds[-1] == pytest.approx(cutoff)
    # the controlling mode is the one nearest its cutoff, and here that is the
    # rejected mode -- so the fingerprint reports a negative margin while
    # lambda_min stays positive, which a sign test on lambda_min would miss
    assert margins[-1] == pytest.approx(lambda_min - cutoff)
