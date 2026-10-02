"""True residuals, numerical resolution, and separate ground-state evidence."""

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from clifford_qc.pauli import I, X, Z
from clifford_qc.states import ket_density
from clifford_qc.subspace import (
    ACASEConfig, ConvergenceConfig, GroundStateReference, MatrixElementBank,
    RitzResidual, SecondMomentBank, StreamingMatrixElementBank,
    convergence_report, dense_residual_norm, identity_generator, run_acase,
    variance_extrapolation_diagnostic,
)


def _one_direction(rho=None, h=None):
    return MatrixElementBank(
        ket_density(1, "0") if rho is None else rho,
        Z(1, 0) + X(1, 0) if h is None else h, [identity_generator(1)])


def test_stalled_growth_has_a_nonzero_full_residual():
    """Z duplicates the reference; the projected solve is stationary, not exact."""
    run = run_acase(ket_density(1, "0"), Z(1, 0) + X(1, 0), [Z(1, 0)],
                    convergence=ConvergenceConfig(0.1))
    report, = run.convergence_reports
    assert run.stopped_reason.startswith("every candidate rejected")
    assert report.stopped_reason == run.stopped_reason
    assert report.energy_change is None
    assert report.residual_status == "above_tolerance"
    assert report.residual.residual_norm == pytest.approx(1.0)
    dense = dense_residual_norm(run.bank.reference, run.bank.hamiltonian,
                                [run.bank.generator(i) for i in run.indices], run.result)
    assert report.residual.residual_norm == pytest.approx(dense)
    assert report.ground_state_status == "not_assessed"


@pytest.mark.parametrize("bits, expected", [("1", "ground_dominated"),
                                            ("0", "not_established")])
def test_zero_residual_does_not_identify_the_ground_state(bits, expected):
    bank = _one_direction(ket_density(1, bits), Z(1, 0))
    result = bank.solve()
    without = convergence_report(result, ConvergenceConfig(0.1))
    assert without.residual_status == "unresolved"
    assert without.residual.variance == pytest.approx(0.0)
    assert without.ground_state_status == "not_assessed"
    with_reference = convergence_report(result, ConvergenceConfig(
        0.1, GroundStateReference(-1.0, 1.0, "exact full-space spectrum of Z")))
    assert with_reference.ground_state_status == expected
    assert with_reference.ground_weight_lower_bound == (1.0 if bits == "1" else 0.0)
    json.dumps(with_reference.as_dict(), allow_nan=False)


@pytest.mark.parametrize("ground_weight", [0.01, 0.99])
def test_mirror_states_have_the_same_small_residual_but_different_ground_weight(ground_weight):
    rho = 0.5 * (I(1) + 2 * math.sqrt(ground_weight * (1 - ground_weight)) * X(1, 0)
                 + (1 - 2 * ground_weight) * Z(1, 0))
    bank = _one_direction(rho, Z(1, 0))
    report = convergence_report(bank.solve(), ConvergenceConfig(
        0.3, GroundStateReference(-1.0, 1.0, "exact full-space spectrum of Z")))
    assert report.residual_status == "within_tolerance"
    assert report.residual.residual_norm == pytest.approx(2 * math.sqrt(.99 * .01))
    assert report.ground_weight_lower_bound == pytest.approx(ground_weight)
    assert report.ground_state_status == (
        "ground_dominated" if ground_weight > 0.5 else "not_established")


def test_a_reference_with_a_ground_energy_above_the_rayleigh_energy_is_inconsistent():
    result = _one_direction(ket_density(1, "1"), Z(1, 0)).solve()
    report = convergence_report(result, ConvergenceConfig(
        .1, GroundStateReference(0.0, 1.0, "wrong spectrum")))
    assert report.ground_state_status == "inconsistent_reference"
    assert report.ground_weight_lower_bound is None


def test_ground_weight_uses_the_rayleigh_energy_not_a_regularized_eigenvalue(monkeypatch):
    bank = _one_direction()
    result = bank.solve()
    moments = SecondMomentBank(bank)
    residual = replace(moments.residual(result), energy=-1.0)
    monkeypatch.setattr(moments, "residual", lambda *args: residual)
    report = convergence_report(result, ConvergenceConfig(
        .1, GroundStateReference(-1.0, 1.0, "full-space test reference")), moments=moments)
    assert report.residual.energy == -1.0
    assert report.residual.rayleigh_energy == 1.0
    assert report.ground_weight_lower_bound == 0.0


@pytest.mark.parametrize("variance, status", [(-1e-13, "unresolved"), (-1.0, "invalid")])
def test_negative_variances_are_not_clipped_into_success(monkeypatch, variance, status):
    bank = _one_direction()
    result = bank.solve()
    moments = SecondMomentBank(bank)
    residual = replace(moments.residual(result), variance=variance, residual_norm=0.0)
    monkeypatch.setattr(moments, "residual", lambda *args: residual)
    report = convergence_report(result, ConvergenceConfig(0.1), moments=moments)
    assert report.residual_status == status
    assert report.residual.variance == variance


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_nonphysical_tolerances_are_refused(value):
    with pytest.raises(ValueError):
        ConvergenceConfig(value)


@pytest.mark.parametrize("ground, first, source", [
    (0.0, 0.0, "degenerate"), (1.0, 0.0, "reversed"),
    (float("nan"), 1.0, "nonfinite"), (0.0, float("inf"), "nonfinite"),
    (0.0, 1.0, ""),
])
def test_invalid_spectral_references_are_refused(ground, first, source):
    with pytest.raises(ValueError):
        GroundStateReference(ground, first, source)


def test_a_foreign_bank_and_a_bankless_result_are_refused():
    first, second = _one_direction(), _one_direction()
    result = first.solve()
    with pytest.raises(ValueError, match="not solved on this bank"):
        convergence_report(result, ConvergenceConfig(.1), moments=SecondMomentBank(second))
    with pytest.raises(ValueError, match="MatrixElementBank"):
        convergence_report(replace(result, bank=None), ConvergenceConfig(.1))


def test_default_growth_builds_no_second_moment(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("default growth must not construct H squared")
    monkeypatch.setattr(SecondMomentBank, "__init__", refuse)
    run = run_acase(ket_density(1, "0"), Z(1, 0) + X(1, 0), [X(1, 0)], max_size=1)
    assert run.convergence_reports == ()
    assert "convergence_reporting" not in run.resources
    assert "H^2" not in run.resources["projected_observables"]


@pytest.mark.parametrize("storage, streaming", [("object", False), ("packed", False),
                                               ("packed", True)])
def test_reporting_preserves_growth_and_materializes_only_the_retained_block(storage, streaming):
    rho, h, pool = ket_density(2, "00"), Z(2, 0) + X(2, 0) + .2 * X(2, 1), [X(2, 0), X(2, 1)]
    baseline = run_acase(rho, h, pool, max_size=1)
    cls = StreamingMatrixElementBank if streaming else MatrixElementBank
    kwargs = {"frontier_pairs": 1} if streaming else {}
    bank = cls(rho, h, storage=storage, **kwargs)
    actual = run_acase(rho, h, pool, bank=bank, max_size=1,
                       convergence=ConvergenceConfig(.1))
    assert actual.labels == baseline.labels
    assert actual.energy_history == baseline.energy_history
    assert actual.stopped_reason == baseline.stopped_reason
    np.testing.assert_array_equal(actual.result.coefficients, baseline.result.coefficients)
    report, = actual.convergence_reports
    assert report.energy_change == actual.energy_history[-2] - actual.energy_history[-1]
    assert report.residual.residual_norm == pytest.approx(.2)
    retained = set(actual.indices)
    rows = next(record["operators"] for record in bank._observables.values()
                if record["label"] == "H^2")
    assert len(rows) == len(retained) * (len(retained) + 1) // 2
    assert all(i in retained and j in retained for i, j in rows)
    assert actual.resources["coefficient_occurrences"] == bank.resources(actual.indices)[
        "coefficient_occurrences"]


def test_configured_multiroot_reporting_shares_one_second_moment_bank(monkeypatch):
    calls = []
    constructor = SecondMomentBank.__init__
    def counted(self, bank):
        calls.append(bank)
        constructor(self, bank)
    monkeypatch.setattr(SecondMomentBank, "__init__", counted)
    run = run_acase(ket_density(1, "0"), Z(1, 0), [Z(1, 0)],
                    initial=[identity_generator(1), X(1, 0)], config=ACASEConfig(
                        roots=2, convergence=ConvergenceConfig(
                            .1, GroundStateReference(-1, 1, "full Z spectrum"))))
    assert len(calls) == 1
    assert [r.residual.root for r in run.convergence_reports] == [0, 1]
    assert [r.ground_state_status for r in run.convergence_reports] == [
        "ground_dominated", "not_established"]


def _point(energy, variance, scale=1.0, root=0):
    return RitzResidual(root, energy, energy, variance + energy ** 2, variance,
                        math.sqrt(max(variance, 0.0)), scale)


def test_q17_matches_the_existing_five_bank_record_without_recomputing_any_prefix():
    record = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/reference_results"
                         / "phase15_variance_extrapolation.json").read_text())
    for entry in record["banks"].values():
        points = [_point(r["energy"], r["variance"], r["cancellation_scale"])
                  for r in entry["trajectory"]]
        fit = variance_extrapolation_diagnostic(points)
        old = entry["window"]["fit"]
        assert fit.extrapolable == old["extrapolable"]
        assert fit.energy == pytest.approx(old["intercept"], abs=1e-10)
        assert fit.slope == pytest.approx(old["slope"], rel=1e-7)
        assert fit.residual_rms == pytest.approx(old["residual_rms"], abs=1e-11)
        row = fit.as_dict()
        assert row["variational"] is False and row["estimator"] is False
        assert row["improvement_evidence_banks"] == ["h4", "h2o_cas8e6o"]


@pytest.mark.parametrize("points, reason", [
    ([_point(1.3,.3), _point(1.2,.2), _point(1.1,0)], "not resolved"),
    ([_point(1.3,.2), _point(1.2,.2+1e-13), _point(1.1,.2)], "rounding"),
    ([_point(1.3,.1), _point(1.2,.2), _point(1.1,.3)], "not positive"),
])
def test_q17_refuses_unresolved_rounding_only_and_nonpositive_windows(points, reason):
    fit = variance_extrapolation_diagnostic(points)
    assert not fit.extrapolable and reason in fit.reason
    assert fit.energy is None


def test_q17_refuses_short_excited_and_non_nested_trajectories():
    points = [_point(1.3,.3), _point(1.2,.2), _point(1.1,.1)]
    for bad in (points[:2], [replace(p, root=1) for p in points], points[::-1]):
        with pytest.raises(ValueError):
            variance_extrapolation_diagnostic(bad)
