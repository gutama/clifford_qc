"""Physical constraints, independent optima, and the measured consumer path."""

import numpy as np
import pytest

from clifford_qc import I, X, Y, Z, ket_density
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.subspace import Generator, MatrixElementBank, identity_generator
from clifford_qc.subspace.linalg import normalize_overlap, solve_projected
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
    (np.zeros((2, 2)), "annihilates"),
])
def test_invalid_overlap_is_rejected(S, message):
    with pytest.raises(ValueError, match=message):
        repair_overlap(S)


@pytest.mark.parametrize("kwargs", [
    {"tolerance": 0}, {"tolerance": np.nan}, {"tolerance": 1},
    {"norm_floor": -1}, {"norm_floor": np.inf},
    {"max_iterations": 0}, {"max_iterations": 1.5}, {"max_iterations": True},
    {"max_iterations": np.bool_(True)},
])
def test_invalid_iteration_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        repair_overlap(np.eye(2), **kwargs)


def test_nonconvergence_is_a_reportable_failure():
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        repair_overlap([[1, 1, 0], [1, 1, 1], [0, 1, 1]], max_iterations=1)


def test_negative_measured_norms_follow_the_solver_live_mask():
    S = np.array([[1, 0.3, 0.4], [0.3, -0.1, 0.2], [0.4, 0.2, 1]])
    H = np.diag([-1.0, 100.0, 1.0])
    repaired, info = repair_overlap(S)
    frame = normalize_overlap(S)
    assert info["live_indices"] == tuple(frame.indices) == (0, 2)
    assert info["nonpositive_diagonal_indices"] == (1,)
    assert repaired[1, 1] == S[1, 1]
    assert np.count_nonzero(repaired[1]) == 1
    np.testing.assert_allclose(solve_projected(repaired, H).energies,
                               solve_projected(S, H).energies)


@pytest.mark.parametrize("s", [1e-8, 1e-10, 1e-14, 1e-18, 1e-24])
def test_nearly_dead_live_row_has_a_certified_projection_without_a_new_cut(s):
    S = np.array([[1, .3, 1e-3], [.3, 1, 2e-3], [1e-3, 2e-3, s]])
    repaired, info = repair_overlap(S, max_iterations=np.int64(1000))
    assert info["live_indices"] == (0, 1, 2)
    assert info["rank_one_kkt_certified"]
    scale = np.sqrt(S.diagonal())
    np.testing.assert_allclose((repaired / scale[:, None]) / scale[None, :], np.ones((3, 3)),
                               atol=1e-10)
    np.testing.assert_array_equal(repaired.diagonal(), S.diagonal())
    result = solve_projected(repaired, np.diag([-1.0, 1.0, 0.0]))
    assert np.isfinite(result.ground_energy)


def test_shot_noise_on_an_annihilating_generator_does_not_abort_repair_or_bootstrap():
    theta = .31
    axis = np.sin(2 * theta) * X(1, 0) + np.cos(2 * theta) * Z(1, 0)
    rho = .5 * (I(1) + axis)
    annihilator = .5 * (I(1) - axis)
    bank = MatrixElementBank(rho, -Z(1, 0) - .3 * X(1, 0),
                             [identity_generator(1), Generator("annihilator", annihilator)])
    shared = SharedMeasurement(bank)
    negative_cache = None
    for seed in range(40):
        cache = shared.measure(FiniteShotBackend(seed=seed), 200)
        baseline = shared.solve(cache)
        point = shared.solve(cache, overlap_repair="nearest_correlation")
        if shared.matrices(cache)[0][1, 1].real < 0:
            negative_cache = cache
            assert point.resources["overlap_repair_diagnostics"]["nonpositive_diagonal_indices"] == (1,)
            assert point.ground_energy == pytest.approx(baseline.ground_energy)
    assert negative_cache is not None
    point = shared.solve(negative_cache, overlap_repair="nearest_correlation")
    interval = bootstrap_ritz(shared, negative_cache, result=point, replicates=20, seed=19)
    assert np.isfinite([interval.lower, interval.upper]).all()
    assert interval.estimate == point.ground_energy


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
    shared = SharedMeasurement(shared.bank, pooling="shots")
    cache = shared.measure(FiniteShotBackend(seed=23), 250)
    original_solve = shared.solve
    point = shared.solve(cache, overlap_repair="nearest_correlation", tau_s=1e-4)
    results = []

    def track(replica, **kwargs):
        assert replica.pooling == cache.pooling == "shots"
        result = original_solve(replica, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(shared, "solve", track)
    shots = cache.total_shots
    interval = bootstrap_ritz(
        shared, cache, replicates=12, seed=17, result=point)
    assert len(results) == 12
    assert all(r.resources["shots"] == shots for r in results)
    assert all(r.resources["overlap_repair"] == "nearest_correlation" for r in results)
    assert cache.total_shots == shots
    assert interval.evidence == "heuristic" and not interval.certified
    assert interval.estimate == point.ground_energy
    assert np.isfinite([interval.lower, interval.upper, interval.std_error]).all()


@pytest.mark.parametrize("kwargs", [
    {"overlap_repair": "none"}, {"tau_s": .01}, {"unknown": True},
])
def test_bootstrap_rejects_a_policy_that_conflicts_with_its_point_result(measured, kwargs):
    shared, cache = measured
    point = shared.solve(cache, overlap_repair="nearest_correlation", tau_s=1e-4)
    with pytest.raises(ValueError, match="conflict with the point estimator"):
        bootstrap_ritz(shared, cache, result=point, solver_kwargs=kwargs, replicates=2)


def test_uncertainty_rejects_low_level_repaired_results_without_estimator_provenance(measured):
    shared, cache = measured
    S, H = shared.matrices(cache)
    repaired, _ = repair_overlap(S)
    point = solve_projected(repaired, H).with_bank(shared.bank, shared.indices)
    with pytest.raises(ValueError, match="no measured estimator provenance"):
        ritz_uncertainty(shared, cache, point)
    with pytest.raises(ValueError, match="no measured estimator provenance"):
        bootstrap_ritz(shared, cache, result=point, replicates=2)


def test_bound_uncertainty_rejects_different_outcomes_at_the_same_shot_budget(measured):
    shared, cache = measured
    point = shared.solve(cache)
    other = shared.measure(FiniteShotBackend(seed=24), 250)
    assert other.total_shots == cache.total_shots
    with pytest.raises(ValueError, match="different joint outcomes"):
        bootstrap_ritz(shared, other, result=point, replicates=2)
    with pytest.raises(ValueError, match="different joint outcomes"):
        ritz_uncertainty(shared, other, point)


def test_repaired_results_keep_measured_and_solved_overlap_diagnostics_separate(measured):
    shared, cache = measured
    baseline = shared.solve(cache)
    point = shared.solve(cache, overlap_repair="nearest_correlation")
    assert point.resources["overlap_negative_modes"] == baseline.resources["overlap_negative_modes"] > 0
    assert point.resources["overlap_eigenvalue_min"] == pytest.approx(
        baseline.resources["overlap_eigenvalue_min"])
    assert point.resources["measured_overlap_eigenvalues"] == pytest.approx(baseline.overlap_eigenvalues)
    assert point.resources["repaired_overlap_negative_modes"] == 0
    assert point.resources["repaired_overlap_eigenvalues"] == point.overlap_eigenvalues
    assert point.resources["repaired_overlap_eigenvalue_min"] >= -1e-12


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
