"""Relative pivot ties preserve rays, Ritz spectra, and cached-action residuals."""
import numpy as np
import pytest

from clifford_qc.subspace import canonical_phase_tolerant, deterministic_phase_pivot
from clifford_qc.subspace.action_space import ActionSpace, orthogonal_direction
from clifford_qc.subspace.orthogonal import canonical_phase


@pytest.mark.parametrize("vector, expected", [
    ([0, -1j, 1, -1], 1),
    ([1, 1j * (1 + 1e-14)], 0),
    ([1, 1j * (1 - 1e-14)], 0),
    ([0, 1e-20, -2j], 2),
    ([0, 0, 0], None),
])
def test_first_nonzero_component_in_relative_tie_band(vector, expected):
    assert deterministic_phase_pivot(vector) == expected


def test_exact_binary_band_boundary_and_strict_zero_tolerance():
    tolerance = 2.0 ** -10
    boundary = 1 - tolerance
    assert deterministic_phase_pivot([boundary, 1], tie_tolerance=tolerance) == 0
    assert deterministic_phase_pivot(
        [np.nextafter(boundary, 0), 1], tie_tolerance=tolerance,
    ) == 1
    assert deterministic_phase_pivot([boundary, 1], tie_tolerance=0) == 1
    assert deterministic_phase_pivot([1, -1j], tie_tolerance=0) == 0
    # Even a nearly unit tolerance cannot choose a zero pivot.
    assert deterministic_phase_pivot([0, 1e-300], tie_tolerance=np.nextafter(1., 0)) == 1


def test_near_tie_perturbations_keep_the_same_representative():
    low = np.array([1, 1j * (1 - 1e-14)])
    high = np.array([1, 1j * (1 + 1e-14)])
    assert np.argmax(abs(low)) == 0
    assert np.argmax(abs(high)) == 1
    np.testing.assert_allclose(canonical_phase_tolerant(low), low, atol=0)
    np.testing.assert_allclose(canonical_phase_tolerant(high), high, atol=0)
    assert np.linalg.norm(canonical_phase(low) - canonical_phase(high)) > 1
    for vector in (low, high):
        for angle in (-2.7, -.2, 1.1):
            for scale in (1e-200, 1, 1e200):
                transformed = vector * (scale * np.exp(1j * angle))
                assert deterministic_phase_pivot(transformed) == 0
                np.testing.assert_allclose(canonical_phase_tolerant(transformed) / scale,
                                           vector, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("tolerance", [
    -1e-12, 1, 2, np.inf, -np.inf, np.nan, True, np.bool_(False),
    1j, "0.1", None, [0.1], np.array(0.1), 10 ** 1000,
])
def test_invalid_tolerance_is_rejected(tolerance):
    for function in (deterministic_phase_pivot, canonical_phase_tolerant):
        with pytest.raises(ValueError, match="phase tie tolerance"):
            function([1, 1j], tie_tolerance=tolerance)


@pytest.mark.parametrize("vector", [[], 1, [[1, 2]], [np.inf], [np.nan], [1j * np.inf]])
def test_invalid_vectors_are_rejected(vector):
    for function in (deterministic_phase_pivot, canonical_phase_tolerant):
        with pytest.raises(ValueError, match="finite nonempty one-dimensional"):
            function(vector)


@pytest.mark.parametrize("scale", [1e-320, 1e-300, 1e300])
def test_safe_scaling_has_no_absolute_amplitude_floor(scale):
    vector = scale * np.array([0, -1j, 1])
    assert deterministic_phase_pivot(vector) == 1
    result = canonical_phase_tolerant(vector)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.real / scale, [0, 1, 0], atol=0)
    np.testing.assert_allclose(result.imag / scale, [0, 0, 1], atol=0)


def test_unrepresentable_rotation_is_rejected_but_pivot_remains_defined():
    maximum = np.finfo(float).max
    vector = np.array([complex(maximum, maximum)])
    assert deterministic_phase_pivot(vector) == 0
    with pytest.raises(ValueError, match="not representable"):
        canonical_phase_tolerant(vector)


@pytest.mark.parametrize("zero", [False, True])
def test_result_is_an_independent_complex_copy(zero):
    vector = np.zeros(3) if zero else np.array([0., -2., 1.])
    original = vector.copy()
    result = canonical_phase_tolerant(vector)
    assert result.dtype == np.complex128
    assert not np.shares_memory(result, vector)
    np.testing.assert_array_equal(vector, original)
    result[0] = 3j
    np.testing.assert_array_equal(vector, original)


@pytest.mark.parametrize("seed", range(6))
def test_global_phase_scale_and_physical_invariants(seed):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=7) + 1j * rng.normal(size=7)
    vector[0] = 2j * max(abs(vector))
    result = canonical_phase_tolerant(vector)
    assert deterministic_phase_pivot(vector) == 0
    assert result[0].real > 0
    assert abs(result[0].imag) < 1e-14
    np.testing.assert_allclose(canonical_phase_tolerant(result), result, atol=2e-14)
    assert np.linalg.norm(result) == pytest.approx(np.linalg.norm(vector), rel=1e-14)
    np.testing.assert_allclose(np.outer(result, result.conj()),
                               np.outer(vector, vector.conj()), atol=2e-14)
    matrix = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    matrix += matrix.conj().T
    np.testing.assert_allclose(np.vdot(result, matrix @ result),
                               np.vdot(vector, matrix @ vector), atol=1e-12)
    for scale in (1e-250, 1, 1e250):
        for angle in (-2.7, -.2, 1.1):
            transformed = vector * (scale * np.exp(1j * angle))
            assert deterministic_phase_pivot(transformed) == 0
            np.testing.assert_allclose(canonical_phase_tolerant(transformed) / scale,
                                       result, rtol=2e-14, atol=2e-14)


def test_column_rephasing_preserves_generalized_ritz_spectrum_and_gram():
    eigh = pytest.importorskip("scipy.linalg").eigh
    rng = np.random.default_rng(73)
    basis = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
    matrix = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    matrix += matrix.conj().T
    rotated = np.column_stack([canonical_phase_tolerant(column) for column in basis.T])
    phases = np.array([np.vdot(column, other) / np.vdot(column, column)
                       for column, other in zip(basis.T, rotated.T)])
    h, s = basis.conj().T @ matrix @ basis, basis.conj().T @ basis
    hr, sr = rotated.conj().T @ matrix @ rotated, rotated.conj().T @ rotated
    np.testing.assert_allclose(hr, phases.conj()[:, None] * h * phases, atol=1e-13)
    np.testing.assert_allclose(sr, phases.conj()[:, None] * s * phases, atol=1e-13)
    np.testing.assert_allclose(eigh(h, s, eigvals_only=True),
                               eigh(hr, sr, eigvals_only=True), atol=1e-13)
    orthonormal, _ = np.linalg.qr(basis)
    rotated_q = np.column_stack([canonical_phase_tolerant(v) for v in orthonormal.T])
    np.testing.assert_allclose(rotated_q.conj().T @ rotated_q, np.eye(3), atol=1e-14)


def test_action_space_opt_in_preserves_ray_residual_and_action_count():
    vector = np.array([1, 1j * (1 + 1e-8), .2])
    vector /= np.linalg.norm(vector)
    h = -np.outer(vector, vector.conj())
    basis = np.eye(3)[:, :2]
    calls = []

    def apply(v):
        calls.append(v.copy())
        return h @ v

    strict = ActionSpace(apply, basis)
    tolerant = ActionSpace(apply, basis, phase_tie_tolerance=1e-6)
    values, coefficients = np.linalg.eigh(basis.conj().T @ h @ basis)
    expected = canonical_phase(basis @ coefficients[:, 0])
    energy, state, residual = tolerant.solve()
    strict_energy, strict_state, _ = strict.solve()
    np.testing.assert_array_equal(strict_state, expected)
    assert energy == strict_energy == pytest.approx(values[0])
    assert state[0].real > 0 and abs(state[0].imag) < 1e-14
    assert strict_state[1].real > 0 and abs(strict_state[1].imag) < 1e-14
    np.testing.assert_allclose(np.outer(state, state.conj()),
                               np.outer(strict_state, strict_state.conj()), atol=1e-14)
    np.testing.assert_allclose(residual, h @ state - energy * state, atol=1e-14)
    assert np.linalg.norm(residual) > .1
    np.testing.assert_array_equal(tolerant.basis, strict.basis)
    np.testing.assert_array_equal(tolerant.actions, strict.actions)
    for _ in range(2):
        repeated = tolerant.solve()
        np.testing.assert_array_equal(repeated[1], state)
        np.testing.assert_array_equal(repeated[2], residual)
    assert strict.h_action_calls == tolerant.h_action_calls == 2
    assert len(calls) == 4


def test_orthogonal_direction_opt_in_and_dependence_cutoff():
    vector = np.array([1, 1j * (1 + 1e-8), .7j])
    basis = np.eye(3)[:, 2:]
    projected = vector.copy()
    for _ in range(2):
        projected -= basis @ (basis.conj().T @ projected)
    strict = orthogonal_direction(vector, basis)
    np.testing.assert_array_equal(strict, canonical_phase(projected / np.linalg.norm(projected)))
    tolerant = orthogonal_direction(vector, basis, phase_tie_tolerance=1e-6)
    reused = orthogonal_direction(vector, basis, adjoint=basis.conj().T,
                                  phase_tie_tolerance=1e-6)
    np.testing.assert_array_equal(tolerant, reused)
    assert tolerant[0].real > 0 and abs(tolerant[0].imag) < 1e-14
    np.testing.assert_allclose(basis.conj().T @ tolerant, 0, atol=1e-14)
    np.testing.assert_allclose(np.outer(tolerant, tolerant.conj()),
                               np.outer(strict, strict.conj()), atol=1e-14)
    for tolerance in (0, 1e-6):
        assert orthogonal_direction(basis[:, 0], basis, phase_tie_tolerance=tolerance) is None
        assert orthogonal_direction(1e-12 * vector, basis,
                                    phase_tie_tolerance=tolerance) is None


@pytest.mark.parametrize("tolerance", [-1, 1, np.nan, True, 1j, [1e-6]])
def test_integration_rejects_invalid_tolerance_before_callbacks(tolerance):
    def forbidden(_):
        pytest.fail("invalid tolerance must be rejected before an action")

    with pytest.raises(ValueError, match="phase tie tolerance"):
        ActionSpace(forbidden, np.eye(2), phase_tie_tolerance=tolerance)
    with pytest.raises(ValueError, match="phase tie tolerance"):
        orthogonal_direction([1, 0], np.eye(2)[:, :1], phase_tie_tolerance=tolerance)
