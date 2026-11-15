"""Physical constraints, independent optima, and the measured consumer path."""

import numpy as np
import pytest

from clifford_qc import X, Y, Z, ket_density
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.subspace import Generator, MatrixElementBank, identity_generator
from clifford_qc.subspace.linalg import solve_projected
from clifford_qc.subspace.pencil import repair_overlap
from clifford_qc.subspace.uncertainty import bootstrap_ritz, ritz_uncertainty


@pytest.mark.parametrize("z", [0.2 + 0.3j, 1.3 - 0.7j, -1.7j, 1.0])
def test_two_direction_projection_matches_the_analytic_complex_optimum(z):
    # For a 2x2 correlation matrix feasibility is precisely |z| <= 1.
    scale = np.array([2.0, 0.3])
    C = np.array([[1, z], [np.conjugate(z), 1]])
    S = C * scale[:, None] * scale[None, :]
    before = S.copy()
    repaired, diagnostics = repair_overlap(S)
    optimal = z / max(1.0, abs(z))
    expected = np.array([[1, optimal], [np.conjugate(optimal), 1]])
    expected = expected * scale[:, None] * scale[None, :]
    np.testing.assert_allclose(repaired, expected, atol=1e-9)
    np.testing.assert_array_equal(S, before)
    np.testing.assert_array_equal(repaired.diagonal(), S.diagonal())
    assert diagnostics["normalized_min_eigenvalue_after"] >= -1e-12


def test_three_direction_projection_matches_an_independent_boundary_solution():
    # Symmetry reduces the optimum to C01=C12=t, C02=s=2*t**2-1.
    # Stationarity of 4*(t-1)**2 + 2*s**2 gives 4*t**3-t-1=0.
    C = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    t = next(root.real for root in np.roots([4, 0, -1, -1])
             if abs(root.imag) < 1e-12)
    s = 2 * t * t - 1
    expected = [[1, t, s], [t, 1, t], [s, t, 1]]
    repaired, _ = repair_overlap(C, tolerance=1e-12)
    np.testing.assert_allclose(repaired, expected, atol=1e-9)


def test_repair_is_covariant_under_generator_rescaling_and_rephasing():
    rng = np.random.default_rng(16)
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    S = M + M.conj().T
    np.fill_diagonal(S, 1)
    factors = np.array([0.2, 3.0, 1.1, 0.7]) * np.exp(1j * rng.normal(size=4))
    transformed = factors.conj()[:, None] * S * factors[None, :]
    repaired, _ = repair_overlap(S)
    other, _ = repair_overlap(transformed)
    expected = factors.conj()[:, None] * repaired * factors[None, :]
    np.testing.assert_allclose(other, expected, atol=1e-8)
    assert np.linalg.eigvalsh(other)[0] >= -1e-12


def test_complex_psd_gram_matrix_is_a_fixed_point_and_preserves_the_pencil():
    rng = np.random.default_rng(5)
    V = rng.normal(size=(7, 4)) + 1j * rng.normal(size=(7, 4))
    H = np.diag(np.arange(7) - 4)
    S, Hm = V.conj().T @ V, V.conj().T @ H @ V
    repaired, info = repair_overlap(S)
    np.testing.assert_allclose(repaired, S, atol=1e-13)
    np.testing.assert_allclose(solve_projected(repaired, Hm).energies,
                               solve_projected(S, Hm).energies, atol=1e-12)
    assert info["iterations"] == 0


def test_zero_and_annihilated_rows_are_decoupled_without_inventing_norms():
    S = np.array([[4, 2, 5, 7], [2, 9, 3, 4], [5, 3, 0, 1], [7, 4, 1, 1e-32]])
    repaired, info = repair_overlap(S)
    expected = np.diag([4, 9, 0, 1e-32]).astype(complex)
    expected[:2, :2] = S[:2, :2]
    np.testing.assert_allclose(repaired, expected, atol=1e-15)
    assert info["decoupled_generators"] == 2
    assert np.linalg.eigvalsh(repaired)[0] >= -1e-14


@pytest.mark.parametrize("S, message", [
    ([], "nonempty square"), ([[1, 0]], "nonempty square"),
    ([[1, np.nan], [np.nan, 1]], "finite"),
    ([[1, 1j], [1j, 1]], "Hermitian"),
    ([[1, 0], [0, -0.1]], "negative diagonal"),
    (np.zeros((2, 2)), "annihilates"),
])
def test_invalid_overlap_is_rejected(S, message):
    with pytest.raises(ValueError, match=message):
        repair_overlap(S)


@pytest.mark.parametrize("kwargs", [
    {"tolerance": 0}, {"tolerance": np.nan}, {"tolerance": 1},
    {"norm_floor": -1}, {"norm_floor": np.inf},
    {"max_iterations": 0}, {"max_iterations": 1.5}, {"max_iterations": True},
])
def test_invalid_iteration_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        repair_overlap(np.eye(2), **kwargs)


def test_nonconvergence_is_a_reportable_failure():
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        repair_overlap([[1, 2], [2, 1]], max_iterations=1)


@pytest.fixture
def measured():
    bank = MatrixElementBank(
        ket_density(1, "0"), -Z(1, 0) - 0.3 * X(1, 0) + 0.2 * Y(1, 0),
        [identity_generator(1), Generator("X", X(1, 0)), Generator("Y", Y(1, 0))])
    shared = SharedMeasurement(bank)
    cache = shared.measure(FiniteShotBackend(seed=23), 250)
    return shared, cache


def test_repair_is_opt_in_and_uses_no_additional_shots(measured):
    shared, cache = measured
    S, H = shared.matrices(cache)
    # The exact null relation X|0> = -iY|0> is broken by independent X/Y shots.
    assert np.linalg.eigvalsh(S)[0] < 0
    repaired, _ = repair_overlap(S)
    shots, circuits = cache.total_shots, cache.total_circuits
    kwargs = {"tau_s": 1e-4}
    baseline = shared.solve(cache, **kwargs)
    result = shared.solve(cache, overlap_repair="nearest_correlation", **kwargs)
    np.testing.assert_allclose(baseline.energies, solve_projected(S, H, **kwargs).energies)
    np.testing.assert_allclose(result.energies, solve_projected(repaired, H, **kwargs).energies)
    assert cache.total_shots == shots == result.resources["shots"]
    assert cache.total_circuits == circuits
    assert result.resources["evidence"] == "heuristic"
    assert result.resources["overlap_repair_diagnostics"]["normalized_correction_fro"] > 0
    np.testing.assert_array_equal(shared.matrices(cache)[0], S)


def test_nonlinear_repair_cannot_silently_reuse_the_unrepaired_delta_method(measured):
    shared, cache = measured
    result = shared.solve(cache, overlap_repair="nearest_correlation")
    with pytest.raises(ValueError, match="delta method does not include"):
        ritz_uncertainty(shared, cache, result)


def test_bootstrap_reruns_the_repair_on_joint_histograms(measured, monkeypatch):
    shared, cache = measured
    original_solve = shared.solve
    results = []

    def track(replica, **kwargs):
        result = original_solve(replica, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(shared, "solve", track)
    shots = cache.total_shots
    interval = bootstrap_ritz(
        shared, cache, replicates=12, seed=17,
        solver_kwargs={"overlap_repair": "nearest_correlation", "tau_s": 1e-4})
    assert len(results) == 12
    assert all(r.resources["shots"] == shots for r in results)
    assert all(r.resources["overlap_repair"] == "nearest_correlation" for r in results)
    assert cache.total_shots == shots
    assert interval.evidence == "heuristic" and not interval.certified
    assert np.isfinite([interval.lower, interval.upper, interval.std_error]).all()


@pytest.mark.parametrize("kwargs, message", [
    ({"overlap_repair": "other"}, "overlap_repair must"),
    ({"overlap_repair": "nearest_correlation", "calibrate_overlap": True},
     "cannot use shot-calibrated radii"),
])
def test_unvalidated_repair_policies_fail_explicitly(measured, kwargs, message):
    shared, cache = measured
    with pytest.raises(ValueError, match=message):
        shared.solve(cache, **kwargs)


def test_psd_overlap_repair_does_not_restore_the_variational_bound():
    repaired, _ = repair_overlap(np.eye(2))
    # A physical reference ground energy of -1 does not constrain noisy H.
    assert solve_projected(repaired, np.diag([-2.0, 1.0])).ground_energy < -1
