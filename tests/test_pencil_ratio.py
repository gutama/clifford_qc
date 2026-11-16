"""Covariance-aware constrained ratio: covariance identities, the propagated
standard error against Monte Carlo, the program's gates, and policy binding.

The estimator minimizes ``c^* H c / c^* S c`` subject to a propagated standard
error budget (Fischer et al., arXiv:2510.25640). These tests establish what it
computes and that its covariance is the grouped measurement's own. One test
documents a property rather than a success: minimizing a noisy ratio is biased
downward, so the result is variational only within the budget.
"""

import math

import numpy as np
import pytest

from clifford_qc import X, Y, Z, ket_density
from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement.functionals import WordFunctional, covariance_matrix
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (
    Generator, MatrixElementBank, identity_generator, krylov_response, pauli_orbit,
    ritz_functional,
)
from clifford_qc.subspace.pencil import (
    _HermitianLayout, hermitian_parameter_layout, hermitian_parameters, ratio_statistics,
    solve_constrained_ratio,
)
from clifford_qc.subspace.uncertainty import bootstrap_ritz, ritz_uncertainty

pytest.importorskip("scipy")

N = 4


@pytest.fixture(scope="module")
def setting():
    model = tfim(N, J=1.0, h=1.0)
    rho = ExactMVBackend().state(model.reference, ())
    words = [op.word for op in odd_y_filter(local_pool(N))]
    return model, rho, words


@pytest.fixture(scope="module")
def well_conditioned(setting):
    model, rho, words = setting
    bank = MatrixElementBank(rho, model.hamiltonian,
                             [identity_generator(N)] + pauli_orbit(words[:3]))
    return bank, SharedMeasurement(bank)


@pytest.fixture(scope="module")
def ill_conditioned(setting):
    """kappa_S ~ 200: the thresholded solve has rare catastrophic replicas."""
    model, rho, words = setting
    bank = MatrixElementBank(rho, model.hamiltonian,
                             [identity_generator(N)] + pauli_orbit(words[:6])
                             + krylov_response(model.hamiltonian, 2))
    return bank, SharedMeasurement(bank)


@pytest.fixture
def qubit():
    bank = MatrixElementBank(
        ket_density(1, "0"), -Z(1, 0) - 0.3 * X(1, 0) + 0.2 * Y(1, 0),
        [identity_generator(1), Generator("X", X(1, 0)), Generator("Y", Y(1, 0))])
    shared = SharedMeasurement(bank)
    return shared, shared.measure(FiniteShotBackend(seed=23), 250)


# ------------------------------------------------------------ coordinates

def test_hermitian_layout_order_and_identities():
    assert hermitian_parameter_layout(2) == (
        (0, 0, "real"), (0, 1, "real"), (0, 1, "imag"), (1, 1, "real"))
    assert len(hermitian_parameter_layout(5)) == 25
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    M = A + A.conj().T
    layout = _HermitianLayout(4)
    theta = hermitian_parameters(M)
    np.testing.assert_allclose(layout.matrix(theta), M, atol=1e-14)
    for _ in range(5):
        c = rng.normal(size=4) + 1j * rng.normal(size=4)
        assert layout.features(c) @ theta == pytest.approx((c.conj() @ M @ c).real, abs=1e-12)


@pytest.mark.parametrize("size", [0, -1, 2.0, True])
def test_layout_rejects_invalid_sizes(size):
    with pytest.raises(ValueError, match="positive integer"):
        hermitian_parameter_layout(size)


@pytest.mark.parametrize("pooling", ["assigned", "shots"])
def test_covariance_matrix_equals_pairwise_functional_covariances(ill_conditioned, pooling):
    bank, _ = ill_conditioned
    shared = SharedMeasurement(bank, pooling=pooling)
    cache = shared.measure(FiniteShotBackend(seed=5), 300)
    functionals = shared.matrix_functionals()[:40]
    matrix = covariance_matrix(functionals, cache)
    pairwise = np.array([[a.covariance(b, cache) for b in functionals] for a in functionals])
    np.testing.assert_allclose(matrix, pairwise, rtol=1e-12, atol=1e-12 * np.abs(pairwise).max())
    assert np.linalg.eigvalsh(matrix)[0] > -1e-10 * np.abs(matrix).max()


def test_covariance_matrix_refuses_an_unmeasured_word(well_conditioned):
    _, shared = well_conditioned
    cache = shared.measure(FiniteShotBackend(seed=6), 50)
    measured = {word.code for word in shared.words}
    missing = next(code for code in range(1, 4 ** N) if code not in measured)
    with pytest.raises(ValueError, match="not measured"):
        covariance_matrix([WordFunctional({missing: 1.0})], cache)
    deterministic = covariance_matrix([WordFunctional({}, 2.0)], cache)
    np.testing.assert_array_equal(deterministic, np.zeros((1, 1)))


def test_entry_coordinates_reproduce_the_measured_matrices(ill_conditioned):
    _, shared = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=8), 200)
    S, H = shared.matrices(cache)
    values = np.array([f.estimate(cache) for f in shared.entry_functionals()])
    count = S.shape[0] ** 2
    np.testing.assert_array_equal(values[:count], hermitian_parameters(S))
    np.testing.assert_array_equal(values[count:], hermitian_parameters(H))
    assert shared.entry_covariance(cache).shape == (2 * count, 2 * count)


# -------------------------------------------- the propagated standard error

def test_ratio_variance_at_a_ritz_vector_is_the_ritz_functional_variance(ill_conditioned):
    """With ``c^* S c = 1`` the ratio Jacobian is ``c^*(dH - E dS)c``: the same
    functional the existing delta method uses, so the two variances agree."""
    bank, shared = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=13), 2000)
    S, H = shared.matrices(cache)
    result = shared.solve(cache, tau_s=1e-3)
    c = result.ritz_vector(0)
    stats = ratio_statistics(S, H, shared.entry_covariance(cache), c)
    assert stats["denominator"] == pytest.approx(1.0, abs=1e-10)
    expected = ritz_functional(bank, shared.indices, c, result.ground_energy).variance(cache)
    assert stats["standard_error"] ** 2 == pytest.approx(expected, rel=1e-10)


def test_the_reference_direction_reduces_to_the_direct_standard_error(ill_conditioned):
    """Fischer et al.'s check: ``c = (1, 0, ...)`` is the standard error of the mean."""
    _, shared = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=14), 500)
    S, H = shared.matrices(cache)
    start = np.zeros(S.shape[0])
    start[0] = 1.0
    stats = ratio_statistics(S, H, shared.entry_covariance(cache), start)
    direct = shared.entry_functionals()[S.shape[0] ** 2]  # H_00
    assert stats["energy"] == pytest.approx(direct.estimate(cache), abs=1e-12)
    assert stats["standard_error"] == pytest.approx(math.sqrt(direct.variance(cache)), rel=1e-10)
    assert stats["denominator_cv"] == 0.0


@pytest.mark.parametrize("fixture", ["well_conditioned", "ill_conditioned"])
def test_propagated_standard_error_matches_a_monte_carlo_ground_truth(fixture, request):
    """At fixed coefficients the full grouped covariance predicts the spread of
    the measured ratio over independent measurement seeds."""
    bank, shared = request.getfixturevalue(fixture)
    c = bank.solve().ritz_vector(0)
    energies, sigmas = [], []
    for seed in range(48):
        cache = shared.measure(FiniteShotBackend(seed=700 + seed), 2000)
        S, H = shared.matrices(cache)
        stats = ratio_statistics(S, H, shared.entry_covariance(cache), c)
        energies.append(stats["energy"])
        sigmas.append(stats["standard_error"])
    empirical = float(np.std(energies, ddof=1))
    assert abs(float(np.mean(sigmas)) - empirical) <= 0.25 * empirical


def test_ratio_statistics_requires_a_positive_denominator():
    S = np.diag([1.0, -0.5]).astype(complex)
    with pytest.raises(ValueError, match="must be positive"):
        ratio_statistics(S, np.eye(2), np.zeros((8, 8)), np.array([0.0, 1.0]))


# ------------------------------------------------------------- the program

def test_negligible_covariance_recovers_the_generalized_ground_state():
    """With no shot noise the budget never binds, and minimizing the Rayleigh
    quotient of a positive-definite pencil is its lowest generalized root."""
    rng = np.random.default_rng(31)
    V = rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4))
    H_full = np.diag(np.linspace(-3, 2, 6))
    S, H = V.conj().T @ V, V.conj().T @ H_full @ V
    result = solve_constrained_ratio(S, H, np.zeros((32, 32)), error_budget=1e-3)
    Q, _ = np.linalg.qr(V)
    expected = np.linalg.eigvalsh(Q.conj().T @ H_full @ Q)[0]
    assert result.energies == pytest.approx((expected,), abs=1e-7)
    c = result.ritz_vector(0)
    assert (c.conj() @ S @ c).real == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(H @ c, expected * (S @ c), atol=1e-5)
    assert result.resources["constrained_ratio"]["standard_error"] == 0.0


@pytest.mark.parametrize("budget", [0.005, 0.02, 0.1])
def test_measured_endpoints_respect_the_budget_and_denominator_gate(ill_conditioned, budget):
    _, shared = ill_conditioned
    for seed in range(6):
        cache = shared.measure(FiniteShotBackend(seed=500 + seed), 2000)
        S, H = shared.matrices(cache)
        cov = shared.entry_covariance(cache)
        result = solve_constrained_ratio(S, H, cov, error_budget=budget)
        ratio = result.resources["constrained_ratio"]
        stats = ratio_statistics(S, H, cov, result.ritz_vector(0))
        assert stats["energy"] == pytest.approx(result.ground_energy, abs=1e-12)
        assert stats["denominator"] == pytest.approx(1.0, abs=1e-10)
        assert stats["standard_error"] <= budget * (1 + 1e-6)
        assert stats["denominator_cv"] <= 0.1 * (1 + 1e-6)
        assert ratio["standard_error"] == pytest.approx(stats["standard_error"], rel=1e-8)
        assert len(result.energies) == 1


def test_tail_control_and_its_documented_downward_bias(ill_conditioned):
    """The constrained ratio trades the thresholded solve's rare catastrophes
    for a bias: minimizing a noisy ratio exploits noise up to about the budget.
    Asserted in the robust direction only."""
    bank, shared = ill_conditioned
    exact = bank.solve().ground_energy
    budget = 0.01
    errors = []
    for seed in range(12):
        cache = shared.measure(FiniteShotBackend(seed=500 + seed), 2000)
        S, H = shared.matrices(cache)
        result = solve_constrained_ratio(S, H, shared.entry_covariance(cache),
                                         error_budget=budget)
        errors.append(result.ground_energy - exact)
    errors = np.array(errors)
    assert np.max(np.abs(errors)) < 20 * budget
    assert np.mean(errors < 0) >= 0.75


def test_an_unattainable_budget_is_an_explicit_failure(ill_conditioned):
    _, shared = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=40), 200)
    S, H = shared.matrices(cache)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        solve_constrained_ratio(S, H, shared.entry_covariance(cache), error_budget=1e-9)


def test_iteration_exhaustion_is_an_explicit_failure(ill_conditioned):
    _, shared = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=41), 2000)
    S, H = shared.matrices(cache)
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        solve_constrained_ratio(S, H, shared.entry_covariance(cache), error_budget=0.02,
                                max_iterations=1)


def test_a_line_search_stop_is_accepted_only_with_a_kkt_certificate(ill_conditioned):
    _, shared = ill_conditioned
    stops = set()
    for seed in range(4):
        cache = shared.measure(FiniteShotBackend(seed=900 + seed), 2000)
        S, H = shared.matrices(cache)
        cov = shared.entry_covariance(cache)
        tight = solve_constrained_ratio(S, H, cov, error_budget=0.02, tolerance=1e-14)
        default = solve_constrained_ratio(S, H, cov, error_budget=0.02)
        ratio = tight.resources["constrained_ratio"]
        stops.add(ratio["stopping"])
        if ratio["stopping"] == "line_search_kkt_certified":
            assert 0.0 <= ratio["kkt_residual"] <= 1e-4
        assert tight.ground_energy == pytest.approx(default.ground_energy, abs=1e-6)
    assert "line_search_kkt_certified" in stops


def test_a_feasible_start_is_never_beaten_by_a_worse_endpoint():
    """A single live generator leaves only the start; it is returned as is."""
    S = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=complex)
    H = np.array([[-3.0, 0.0], [0.0, 1.0]], dtype=complex)
    result = solve_constrained_ratio(S, H, np.zeros((8, 8)), labels=["A", "dead"],
                                     error_budget=0.1)
    ratio = result.resources["constrained_ratio"]
    assert result.energies == pytest.approx((-1.5,), abs=1e-14)
    assert ratio["selected"] == "reference_start" and ratio["start_feasible"]
    assert result.coefficients[1, 0] == 0.0
    assert result.resources["dropped_generators"] == ("dead",)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("variance_s, variance_h, budget", [
    (0.0, 1.0, 0.1),  # unresolved energy
    (1.0, 0.0, 10.0),  # resolved energy budget, unresolved denominator
])
def test_a_single_live_generator_must_satisfy_both_feasibility_gates(
        padded, variance_s, variance_h, budget):
    size, live = (3, 1) if padded else (1, 0)
    S, H = np.zeros((size, size)), np.zeros((size, size))
    S[live, live], H[live, live] = 1.0, -1.0
    index = hermitian_parameter_layout(size).index((live, live, "real"))
    cov = np.zeros((2 * size ** 2, 2 * size ** 2))
    cov[index, index] = variance_s
    cov[size ** 2 + index, size ** 2 + index] = variance_h
    with pytest.raises(ValueError, match="error budget and denominator gate"):
        solve_constrained_ratio(S, H, cov, error_budget=budget)


def test_measured_identity_only_bank_rejects_an_unattainable_budget():
    bank = MatrixElementBank(ket_density(1, "0"), X(1, 0), [identity_generator(1)])
    shared = SharedMeasurement(bank)
    cache = shared.measure(FiniteShotBackend(seed=0), 100)
    before = cache.total_shots, cache.total_circuits
    with pytest.raises(ValueError, match="error budget and denominator gate"):
        shared.solve(cache, ratio_error_budget=.001)
    assert (cache.total_shots, cache.total_circuits) == before
    with pytest.raises(ValueError, match="error budget and denominator gate"):
        bootstrap_ritz(shared, cache, replicates=2,
                       solver_kwargs={"ratio_error_budget": .001})


@pytest.mark.parametrize("budget", [.05, .25, .75])
def test_stationary_reference_and_infeasible_ground_start_do_not_return_a_maximum(budget):
    # For unit c, p=|c_1|^2 gives E=1-2p, sigma=p, and denominator CV=0.
    # Thus the constrained minimum is analytically p=budget, E=1-2*budget.
    # Both pure directions have zero objective gradient. The reference is a
    # feasible strict maximum; the unconstrained ground root is infeasible.
    S, H, cov = np.eye(2), np.diag([1.0, -1.0]), np.zeros((8, 8))
    cov[-1, -1] = 1.0
    result = solve_constrained_ratio(S, H, cov, error_budget=budget)
    assert result.ground_energy == pytest.approx(1 - 2 * budget, abs=5e-6)
    stats = ratio_statistics(S, H, cov, result.ritz_vector())
    assert stats["standard_error"] <= budget * (1 + 1e-6)
    assert stats["denominator_cv"] == 0


def test_dead_rows_stay_exactly_zero(ill_conditioned):
    _, shared = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=42), 2000)
    S, H = shared.matrices(cache)
    cov = shared.entry_covariance(cache)
    m = S.shape[0]
    S2 = np.zeros((m + 1, m + 1), dtype=complex)
    H2 = np.zeros((m + 1, m + 1), dtype=complex)
    S2[:m, :m], H2[:m, :m], H2[m, m] = S, H, 5.0
    index = [k for k, (a, b, _) in enumerate(hermitian_parameter_layout(m + 1))
             if a < m and b < m]
    count = (m + 1) ** 2
    full = np.concatenate([index, [count + k for k in index]])
    cov2 = np.zeros((2 * count, 2 * count))
    cov2[np.ix_(full, full)] = cov
    padded = solve_constrained_ratio(S2, H2, cov2, error_budget=0.02)
    plain = solve_constrained_ratio(S, H, cov, error_budget=0.02)
    assert padded.coefficients[m, 0] == 0.0
    assert padded.ground_energy == pytest.approx(plain.ground_energy, abs=1e-9)


@pytest.mark.parametrize("kwargs, message", [
    ({"error_budget": 0.0}, "error_budget"),
    ({"error_budget": float("nan")}, "error_budget"),
    ({"error_budget": True}, "error_budget"),
    ({"error_budget": 0.1, "max_denominator_cv": -1.0}, "max_denominator_cv"),
    ({"error_budget": 0.1, "tolerance": 1.0}, "tolerance"),
    ({"error_budget": 0.1, "max_iterations": 0}, "max_iterations"),
    ({"error_budget": 0.1, "max_iterations": True}, "max_iterations"),
])
def test_invalid_controls_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        solve_constrained_ratio(np.eye(2), np.eye(2), np.zeros((8, 8)), **kwargs)


@pytest.mark.parametrize("covariance, message", [
    (np.zeros((4, 4)), "8 x 8"),
    (np.diag([1.0, -1.0, 0, 0, 0, 0, 0, 0]), "positive semidefinite"),
    (np.triu(np.ones((8, 8))), "symmetric"),
])
def test_invalid_covariances_are_rejected(covariance, message):
    with pytest.raises(ValueError, match=message):
        solve_constrained_ratio(np.eye(2), np.eye(2), covariance, error_budget=0.1)


def test_a_stationary_reference_does_not_trap_the_search():
    """The reference is an exact eigenvector here, so its gradient vanishes and
    SLSQP cannot leave it; the canonical ground-root start finds the minimum."""
    result = solve_constrained_ratio(np.eye(2), np.diag([1.0, -1.0]), np.zeros((8, 8)),
                                     error_budget=0.1, max_iterations=np.int64(50))
    assert result.energies == pytest.approx((-1.0,), abs=1e-8)
    runs = {run["start"]: run for run in result.resources["constrained_ratio"]["runs"]}
    assert runs["reference"]["energy"] == pytest.approx(1.0)
    assert result.resources["constrained_ratio"]["selected"] == "canonical_ground_root"


def test_missing_scipy_names_the_extra(monkeypatch):
    from clifford_qc import capabilities

    monkeypatch.setattr(capabilities, "missing_modules", lambda name: ("scipy",))
    with pytest.raises(ModuleNotFoundError, match="research"):
        solve_constrained_ratio(np.eye(2), np.eye(2), np.zeros((8, 8)), error_budget=0.1)


# ------------------------------------------------- the measured, bound policy

def test_measured_ratio_is_opt_in_and_spends_no_shots(qubit):
    shared, cache = qubit
    shots, circuits = cache.total_shots, cache.total_circuits
    S, H = shared.matrices(cache)
    point = shared.solve(cache, ratio_error_budget=0.05)
    direct = solve_constrained_ratio(S, H, shared.entry_covariance(cache), error_budget=0.05)
    assert point.energies == direct.energies
    assert (cache.total_shots, cache.total_circuits) == (shots, circuits)
    assert point.resources["evidence"] == "heuristic"
    assert point.resources["overlap_threshold_policy"] == "constrained_ratio"
    assert point.estimator_parameters["ratio_error_budget"] == 0.05
    assert point.bank is shared.bank and point.indices == shared.indices
    assert shared.solve(cache).estimator_parameters["ratio_error_budget"] is None


def test_bootstrap_reruns_the_constrained_search_on_joint_histograms(qubit, monkeypatch):
    shared, cache = qubit
    point = shared.solve(cache, ratio_error_budget=0.05)
    original = shared.solve
    calls = []

    def track(replica, **kwargs):
        calls.append(kwargs)
        return original(replica, **kwargs)

    monkeypatch.setattr(shared, "solve", track)
    interval = bootstrap_ritz(shared, cache, result=point, replicates=12, seed=4)
    assert len(calls) == 12
    assert all(kwargs == point.estimator_parameters for kwargs in calls)
    assert interval.estimate == point.ground_energy
    assert interval.evidence == "heuristic"
    with pytest.raises(ValueError, match="conflict"):
        bootstrap_ritz(shared, cache, result=point, replicates=2,
                       solver_kwargs={"ratio_error_budget": 0.1})


def test_the_fixed_coefficient_delta_method_rejects_the_constrained_search(qubit):
    shared, cache = qubit
    point = shared.solve(cache, ratio_error_budget=0.05)
    with pytest.raises(ValueError, match="constrained-ratio search"):
        ritz_uncertainty(shared, cache, point)


@pytest.mark.parametrize("kwargs, message", [
    ({"overlap_repair": "nearest_correlation"}, "cannot be combined"),
    ({"pencil_angle": 0.1}, "cannot be combined"),
    ({"calibrate_overlap": True}, "cannot be combined"),
    ({"tau_s": 1e-4}, "thresholds do not apply"),
    ({"rel_tau": 1e-3}, "thresholds do not apply"),
    ({"max_condition": 1e6}, "thresholds do not apply"),
    ({"overlap_regularizer": "ridge"}, "thresholds do not apply"),
])
def test_threshold_and_regularization_controls_are_rejected(qubit, kwargs, message):
    shared, cache = qubit
    with pytest.raises(ValueError, match=message):
        shared.solve(cache, ratio_error_budget=0.05, **kwargs)
