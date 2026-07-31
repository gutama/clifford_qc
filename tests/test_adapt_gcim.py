"""Independent invariants for the exact fixed-angle ADAPT-GCIM comparator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.matrix import to_matrix
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import (
    Generator,
    adapt_gcim_gradient,
    determinant_excitations,
    occupied_spin_orbitals,
    pure_statevector,
    run_adapt_gcim,
)


@pytest.fixture(scope="module")
def dimer_contract():
    model = hubbard((1, 2), t=1.0, U=4.0, periodic=False)
    rho = ExactMVBackend().state(model.reference, ())
    pool = determinant_excitations(
        model.n, occupied_spin_orbitals(model), max_rank=2)
    return model, rho, pool


def _apply_exp(generator: np.ndarray, theta: float,
               state: np.ndarray) -> np.ndarray:
    """Independent fixed-angle action used only by the tests."""
    values, vectors = np.linalg.eigh(1j * generator)
    return vectors @ (
        np.exp(-1j * theta * values) * (vectors.conj().T @ state))


def test_gradient_matches_an_independent_central_difference(dimer_contract):
    model, rho, pool = dimer_contract
    psi = pure_statevector(rho)
    H = to_matrix(model.hamiltonian.to_mv())
    A = to_matrix(pool[0].mv)
    epsilon = 1e-6

    def energy(theta):
        state = _apply_exp(A, theta, psi)
        return float(np.vdot(state, H @ state).real)

    finite_difference = (energy(epsilon) - energy(-epsilon)) / (2 * epsilon)
    assert adapt_gcim_gradient(psi, H, A) == pytest.approx(
        finite_difference, abs=2e-9)


def test_published_two_states_per_iteration_and_fixed_angle(dimer_contract):
    model, rho, pool = dimer_contract
    result = run_adapt_gcim(
        rho, model.hamiltonian, pool, max_iterations=3)

    assert result.theta == pytest.approx(math.pi / 4.0)
    assert [record.basis_size for record in result.records] == [2, 4, 6]
    assert result.basis.shape == (2 ** model.n, 6)
    assert len(set(result.selected_indices)) == 3
    assert result.basis_rotor_depths == (0, 1, 1, 2, 1, 3)
    assert np.allclose(result.surrogate_state, result.basis[:, -1])

    resources = result.result.resources
    assert resources["basis_rotor_applications"] == 8
    assert resources["selection_evaluations"] == 3 + 2 + 1
    assert resources["hamiltonian_matrix_pairs"] == 21
    assert resources["overlap_offdiagonal_pairs"] == 15


def test_projected_pencil_is_direct_and_variational(dimer_contract):
    model, rho, pool = dimer_contract
    result = run_adapt_gcim(
        rho, model.hamiltonian, pool, max_iterations=3)
    H = to_matrix(model.hamiltonian.to_mv())
    basis = result.basis

    assert np.allclose(
        result.overlap_matrix, basis.conj().T @ basis, atol=1e-12)
    assert np.allclose(
        result.hamiltonian_matrix, basis.conj().T @ H @ basis, atol=1e-12)
    assert np.allclose(
        result.overlap_matrix, result.overlap_matrix.conj().T, atol=1e-12)
    assert np.allclose(
        result.hamiltonian_matrix,
        result.hamiltonian_matrix.conj().T, atol=1e-12)
    assert np.allclose(np.diag(result.overlap_matrix), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(result.overlap_matrix).min() >= -1e-12

    trajectory = np.array([record.ground_energy for record in result.records])
    assert np.all(np.diff(trajectory) <= 1e-10)
    assert trajectory[-1] >= np.linalg.eigvalsh(H)[0] - 1e-10


def test_selection_and_energy_are_deterministic(dimer_contract):
    model, rho, pool = dimer_contract
    first = run_adapt_gcim(
        rho, model.hamiltonian, pool, max_iterations=3)
    second = run_adapt_gcim(
        rho, model.hamiltonian, pool, max_iterations=3)
    assert first.selected_labels == second.selected_labels
    assert first.energy == pytest.approx(second.energy, abs=1e-13)


def test_rejects_a_generator_that_is_not_antihermitian(dimer_contract):
    model, rho, _ = dimer_contract
    bad = Generator("Hermitian identity", rho.__class__.scalar(model.n, 1.0))
    with pytest.raises(ValueError, match="not anti-Hermitian"):
        run_adapt_gcim(
            rho, model.hamiltonian, [bad], max_iterations=1)
