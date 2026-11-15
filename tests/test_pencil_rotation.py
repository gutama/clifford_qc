"""Rotation identities, independent Ritz oracles, and measured-policy binding."""

import json

import numpy as np
import pytest

from clifford_qc import X, Z, ket_density
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.subspace import Generator, MatrixElementBank, identity_generator
from clifford_qc.subspace.linalg import solve_projected
from clifford_qc.subspace.pencil import rotation_parameters, solve_rotated
from clifford_qc.subspace.uncertainty import bootstrap_ritz, ritz_uncertainty


@pytest.mark.parametrize("angle", [-.12, .18])
def test_full_rank_complex_pencil_matches_independent_orthogonal_span_oracle(angle):
    rng = np.random.default_rng(51)
    V = rng.normal(size=(7, 4)) + 1j * rng.normal(size=(7, 4))
    H_full = np.diag(np.linspace(-4, 2, 7))
    S, H = V.conj().T @ V, V.conj().T @ H_full @ V
    before = S.copy(), H.copy()
    Q, _ = np.linalg.qr(V)
    expected = np.linalg.eigvalsh(Q.conj().T @ H_full @ Q)
    result = solve_rotated(S, H, angle=angle, energy_shift=-2, energy_scale=3)
    np.testing.assert_allclose(result.energies, expected, atol=1e-12)
    C = result.coefficients
    np.testing.assert_allclose(C.conj().T @ S @ C, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(H @ C, (S @ C) * result.energies, atol=1e-11)
    np.testing.assert_array_equal(S, before[0])
    np.testing.assert_array_equal(H, before[1])
    assert np.diff(result.energies).min() > 0
    json.dumps(result.resources, allow_nan=False)


def test_zero_angle_is_a_bitwise_control_even_with_nontrivial_energy_units():
    S = np.array([[2, .3j], [-.3j, .7]])
    H = np.array([[-3, .4], [.4, 2]])
    control = solve_projected(S, H, tau_s=.01)
    rotated = solve_rotated(S, H, angle=0, energy_shift=1e6, energy_scale=1e-7, tau_s=.01)
    assert rotated.energies == control.energies
    np.testing.assert_array_equal(rotated.coefficients, control.coefficients)
    assert rotated.resources["pencil_rotation"]["zero_angle_control"]


def test_energy_units_and_generator_coordinates_are_covariant():
    S = np.array([[2, .3j], [-.3j, .7]])
    H = np.array([[-3, .4], [.4, 2]])
    policy = dict(angle=.15, energy_shift=-1, energy_scale=3)
    result = solve_rotated(S, H, **policy)
    factors = np.array([.2j, -3])
    transform = lambda M: factors.conj()[:, None] * M * factors[None, :]
    changed = solve_rotated(transform(S), transform(H), **policy)
    np.testing.assert_allclose(result.energies, changed.energies, atol=1e-12)
    # Compare physical states up to their arbitrary column phases.
    C = factors[:, None] * changed.coefficients
    np.testing.assert_allclose(np.abs(result.coefficients.conj().T @ S @ C), np.eye(2),
                               atol=1e-12)
    scaled = solve_rotated(S, 4 * H + 7 * S, angle=.15, energy_shift=3, energy_scale=12)
    np.testing.assert_allclose(scaled.energies, 4 * np.array(result.energies) + 7, atol=1e-12)
    np.testing.assert_allclose(scaled.coefficients, result.coefficients, atol=1e-12)


def test_truncated_rotation_reports_the_physical_rayleigh_quotient():
    S = np.array([[1, .95], [.95, 1]])
    H = np.array([[-.4, -.2], [-.2, .1]])
    result = solve_rotated(S, H, angle=.12, energy_shift=-2, energy_scale=2, tau_s=.2)
    assert result.effective_rank == 1
    C = result.coefficients
    np.testing.assert_allclose(C.conj().T @ S @ C, [[1]], atol=1e-12)
    np.testing.assert_allclose(C.conj().T @ H @ C, [[result.ground_energy]], atol=1e-12)
    assert result.resources["smallest_retained_overlap_eigenvalue"] > .2


def test_original_dead_rows_stay_dead_and_labels_keep_original_coordinates():
    S = np.diag([2., -.1, .5, 0.])
    H = np.diag([-2., 200., .25, -200.])
    result = solve_rotated(S, H, ["a", "dead", "b", "zero"], angle=.1, energy_scale=2)
    np.testing.assert_allclose(result.energies, [-1., .5], atol=1e-12)
    np.testing.assert_array_equal(result.coefficients[[1, 3]], np.zeros((2, 2)))
    assert result.basis_labels == ("a", "dead", "b", "zero")
    assert result.resources["basis_size"] == 4
    assert result.resources["pencil_rotation"]["original_live_indices"] == (0, 2)


@pytest.mark.parametrize("kwargs", [
    {"angle": np.nan}, {"angle": np.inf}, {"angle": np.pi / 2},
    {"angle": -np.pi / 2}, {"angle": True}, {"angle": 1j},
    {"angle": []}, {"angle": .1, "energy_shift": np.nan},
    {"angle": .1, "energy_scale": 0}, {"angle": .1, "energy_scale": -1},
    {"angle": .1, "energy_scale": np.inf}, {"angle": .1, "pole_tolerance": 0},
    {"angle": .1, "pole_tolerance": 1},
])
def test_invalid_rotation_controls_fail_before_solving(kwargs):
    with pytest.raises(ValueError):
        solve_rotated(np.eye(2), np.eye(2), **kwargs)


@pytest.mark.parametrize("H,message", [
    ([[1]], "match the overlap"), ([[1, np.nan], [np.nan, 1]], "finite"),
    ([[1, 1j], [1j, 1]], "Hermitian"),
])
def test_malformed_hamiltonians_are_rejected(H, message):
    with pytest.raises(ValueError, match=message):
        solve_rotated(np.eye(2), H, angle=.1)


def test_indefinite_rotated_metric_does_not_silently_lose_negative_directions():
    with pytest.raises(ValueError, match="rotated overlap is indefinite"):
        solve_rotated(np.eye(2), [[0, 4], [4, 0]], angle=.4)


def test_rotated_dead_diagonal_is_an_explicit_failure():
    theta = .25
    H = np.diag([-np.cos(theta) / np.sin(theta), 1.])
    with pytest.raises(ValueError, match="dead live diagonal"):
        solve_rotated(np.eye(2), H, angle=theta)


def test_inverse_poles_and_negative_physical_norm_branches_are_rejected():
    with pytest.raises(ValueError, match="pole or nonpositive physical-norm"):
        solve_rotated(np.eye(2), np.diag([1e12, 1]), angle=.5)
    S = np.array([[1, 1.2], [1.2, 1]])
    with pytest.raises(ValueError, match="pole or nonpositive physical-norm"):
        solve_rotated(S, 10 * np.eye(2), angle=.2)


@pytest.fixture
def measured():
    bank = MatrixElementBank(ket_density(1, "0"), -Z(1, 0) - .3 * X(1, 0),
                             [identity_generator(1), Generator("X", X(1, 0))])
    shared = SharedMeasurement(bank, pooling="shots")
    return shared, shared.measure(FiniteShotBackend(seed=91), 1000)


@pytest.mark.parametrize("repair", ["none", "nearest_correlation"])
def test_measured_rotation_preserves_shots_units_and_distinct_spectra(measured, repair):
    shared, cache = measured
    S, H = shared.matrices(cache)
    before = cache.total_shots, cache.total_circuits
    point = shared.solve(cache, pencil_angle=.1, energy_shift=-1, energy_scale=2,
                         overlap_repair=repair)
    np.testing.assert_allclose(point.energies, solve_projected(S, H).energies, atol=1e-12)
    assert (cache.total_shots, cache.total_circuits) == before
    assert point.resources["shots"] == before[0]
    assert point.estimator_parameters["pencil_angle"] == .1
    assert point.resources["rotated_overlap_eigenvalues"] == point.overlap_eigenvalues
    assert point.resources["measured_overlap_eigenvalues"] == pytest.approx(
        solve_projected(S, H).overlap_eigenvalues)
    assert point.resources["physical_overlap_eigenvalues"] != point.overlap_eigenvalues
    if repair != "none":
        assert point.resources["repaired_overlap_eigenvalues"] == pytest.approx(
            point.resources["physical_overlap_eigenvalues"])
    with pytest.raises(ValueError, match="delta method does not include"):
        ritz_uncertainty(shared, cache, point)


def test_joint_bootstrap_reuses_the_frozen_rotation_policy(measured, monkeypatch):
    shared, cache = measured
    point = shared.solve(cache, pencil_angle=.1, energy_shift=-1, energy_scale=2,
                         overlap_repair="nearest_correlation")
    solve, replicas = shared.solve, []

    def track(replica, **kwargs):
        assert replica.pooling == "shots"
        assert kwargs == point.estimator_parameters
        result = solve(replica, **kwargs)
        replicas.append(result)
        return result

    monkeypatch.setattr(shared, "solve", track)
    interval = bootstrap_ritz(shared, cache, result=point, replicates=12, seed=12)
    assert len(replicas) == 12
    assert all(r.resources["pencil_rotation"]["angle"] == .1 for r in replicas)
    assert interval.estimate == point.ground_energy
    assert np.isfinite([interval.lower, interval.upper, interval.std_error]).all()
    assert interval.evidence == "heuristic" and not interval.certified


@pytest.mark.parametrize("kwargs", [
    {"pencil_angle": .2}, {"energy_shift": 0}, {"energy_scale": 3},
    {"pole_tolerance": 1e-6},
])
def test_bootstrap_rejects_conflicting_rotation_controls(measured, kwargs):
    shared, cache = measured
    point = shared.solve(cache, pencil_angle=.1, energy_shift=-1, energy_scale=2)
    with pytest.raises(ValueError, match="conflict with the point estimator"):
        bootstrap_ritz(shared, cache, result=point, solver_kwargs=kwargs, replicates=2)


def test_calibrated_radii_cannot_be_reused_after_rotation(measured):
    shared, cache = measured
    with pytest.raises(ValueError, match="rotation cannot use shot-calibrated"):
        shared.solve(cache, pencil_angle=.1, calibrate_overlap=True)


def test_rotation_policy_normalizes_numpy_scalar_controls():
    policy = rotation_parameters(angle=np.float64(.1), energy_scale=np.int64(2))
    assert type(policy["angle"]) is float and type(policy["energy_scale"]) is float
