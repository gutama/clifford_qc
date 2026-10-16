"""Exact fixed-angle ADAPT-GCIM for small-system validation and comparison.

This module implements the algorithm of Zheng *et al.*, npj Quantum
Information 10, 128 (2024), rather than using ``generator coordinate`` as a
loose synonym for a fixed excitation basis.

At iteration ``k`` the method

1. evaluates the zero-parameter energy gradient of every unused
   anti-Hermitian excitation on the cumulative surrogate state;
2. selects the largest absolute gradient;
3. applies that excitation with the fixed published angle ``theta = pi/4``;
4. adds ``U_k |ref>`` and ``U_k ... U_1 |ref>`` to the generating-function
   basis (the two coincide at ``k=1``); and
5. solves the nonorthogonal generalized eigenproblem.

Consequently the nominal basis size is exactly ``M = 2k``.  No variational
parameter optimization is performed.  The implementation is deliberately
dense: its purpose is an independently inspectable exact oracle on the
eight-qubit H4 benchmark, not a scaling path or a hardware cost model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..selection import TIE_ATOL, TIE_RTOL, canonical_argmax
from ..ir import PauliSum
from ..dense_reference import to_matrix
from ..multivector import MV
from .generator_core import Generator, as_generators
from .reference import pure_statevector
from .solver import SubspaceResult, solve_projected


@dataclass(frozen=True)
class AdaptGCIMIteration:
    """One exact ADAPT-GCIM selection and projected solve."""

    iteration: int
    selected_index: int
    selected_label: str
    gradient: float
    active_candidates: int
    basis_size: int
    basis_rotor_depths: tuple[int, ...]
    ground_energy: float
    effective_rank: int
    condition_number: float


@dataclass(frozen=True)
class AdaptGCIMResult:
    """Exact ADAPT-GCIM trajectory and final nonorthogonal pencil.

    ``basis`` contains the explicitly prepared generating-function states as
    columns.  Keeping it is appropriate here because this module is a dense
    small-system oracle; the A-CASE operator route intentionally never forms
    the analogous vectors.
    """

    result: SubspaceResult
    records: tuple[AdaptGCIMIteration, ...]
    selected_indices: tuple[int, ...]
    selected_labels: tuple[str, ...]
    theta: float
    basis: np.ndarray
    overlap_matrix: np.ndarray
    hamiltonian_matrix: np.ndarray
    basis_rotor_depths: tuple[int, ...]
    surrogate_state: np.ndarray

    @property
    def energy(self) -> float:
        return self.result.ground_energy


def _as_mv(operator) -> MV:
    if isinstance(operator, MV):
        return operator
    if isinstance(operator, PauliSum):
        return operator.to_mv()
    raise TypeError(f"expected MV or PauliSum, got {type(operator).__name__}")


def adapt_gcim_gradient(psi: np.ndarray, hamiltonian: np.ndarray,
                        generator: np.ndarray) -> float:
    """Derivative of ``<psi|e^-tA H e^tA|psi>`` at ``t=0``.

    For anti-Hermitian ``A`` this is the real scalar
    ``<psi|[H,A]|psi> = 2 Re <H psi|A psi>``.  The explicit public helper lets
    tests compare the selector against an independent central difference.
    """
    psi = np.asarray(psi, dtype=complex)
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    generator = np.asarray(generator, dtype=complex)
    return float(2.0 * np.vdot(hamiltonian @ psi, generator @ psi).real)


def _unitary_action(generator: np.ndarray, states: np.ndarray,
                    theta: float) -> np.ndarray:
    """Apply ``exp(theta A)`` using only NumPy and anti-Hermiticity.

    ``iA`` is Hermitian, so its eigendecomposition is unitary and substantially
    safer than a generic eigenvector inversion.  ``states`` may be one vector
    or a matrix whose columns are vectors.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(1j * generator)
    phases = np.exp(-1j * theta * eigenvalues)
    states = np.asarray(states, dtype=complex)
    was_vector = states.ndim == 1
    if was_vector:
        states = states[:, None]
    out = eigenvectors @ (
        phases[:, None] * (eigenvectors.conj().T @ states))
    return out[:, 0] if was_vector else out


def run_adapt_gcim(
        rho: MV,
        hamiltonian,
        generators: Sequence,
        *,
        max_iterations: int,
        theta: float = math.pi / 4.0,
        tau_s: float = 1e-13,
        max_condition: float = math.inf,
        antihermitian_atol: float = 1e-10,
) -> AdaptGCIMResult:
    """Run the published fixed-angle ADAPT-GCIM construction exactly.

    Parameters are intentionally narrow.  Candidate reuse and parameter
    optimization are absent because they are absent from the algorithm being
    compared.  ``tau_s=1e-13`` follows the exact-simulation threshold used by
    the reference implementation.  No separate condition-number cap is imposed
    by default, so the comparator does not silently replace that published
    overlap rule with A-CASE's regularization policy.
    """
    gens = as_generators(generators)
    H_mv = _as_mv(hamiltonian)
    if not H_mv.is_hermitian():
        raise ValueError("Hamiltonian must be Hermitian")
    if H_mv.n != rho.n or any(g.n != rho.n for g in gens):
        raise ValueError("reference, Hamiltonian, and generators disagree on n")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if max_iterations > len(gens):
        raise ValueError("max_iterations exceeds the unused-generator pool")
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")

    psi_ref = pure_statevector(rho)
    H_dense = to_matrix(H_mv)
    generator_matrices: list[np.ndarray] = []
    for generator in gens:
        matrix = to_matrix(generator.mv)
        if not np.allclose(matrix.conj().T, -matrix,
                           atol=antihermitian_atol, rtol=0.0):
            raise ValueError(
                f"ADAPT-GCIM generator {generator.label!r} is not "
                "anti-Hermitian")
        generator_matrices.append(matrix)

    selected: list[int] = []
    unused = list(range(len(gens)))
    cumulative = psi_ref.copy()
    basis_states = [psi_ref.copy()]
    basis_labels = ["HF"]
    basis_depths = [0]
    records: list[AdaptGCIMIteration] = []
    solution: SubspaceResult | None = None
    S = np.ones((1, 1), dtype=complex)
    Hm = np.array([[np.vdot(psi_ref, H_dense @ psi_ref)]], dtype=complex)

    for iteration in range(1, max_iterations + 1):
        gradients = {
            index: adapt_gcim_gradient(
                cumulative, H_dense, generator_matrices[index])
            for index in unused
        }
        chosen = canonical_argmax(
            unused, lambda index: abs(gradients[index]),
            rtol=TIE_RTOL, atol=TIE_ATOL)
        selected.append(chosen)
        unused.remove(chosen)

        # One eigendecomposition applies the selected fixed-angle unitary to
        # both states required by the published basis rule.
        inputs = np.column_stack((psi_ref, cumulative))
        outputs = _unitary_action(generator_matrices[chosen], inputs, theta)
        single = outputs[:, 0]
        cumulative = outputs[:, 1]

        if iteration == 1:
            # U1|ref> is both the single-generator and cumulative state.
            basis_states.append(cumulative.copy())
            basis_labels.append(f"U[{gens[chosen].label}] HF")
            basis_depths.append(1)
        else:
            basis_states.extend((single.copy(), cumulative.copy()))
            basis_labels.extend((
                f"U[{gens[chosen].label}] HF",
                " ".join(
                    f"U[{gens[i].label}]" for i in reversed(selected)) +
                " HF",
            ))
            basis_depths.extend((1, iteration))

        basis = np.column_stack(basis_states)
        S = basis.conj().T @ basis
        Hm = basis.conj().T @ H_dense @ basis
        # Round-off from dense products is removed only at the structural
        # Hermitian boundary expected by solve_projected.
        S = 0.5 * (S + S.conj().T)
        Hm = 0.5 * (Hm + Hm.conj().T)
        resources = {
            "method": "exact ADAPT-GCIM",
            "theta": float(theta),
            "iterations": iteration,
            "basis_size": basis.shape[1],
            "basis_rotor_depths": tuple(basis_depths),
            "basis_rotor_applications": int(sum(basis_depths)),
            "selection_evaluations": int(
                sum(len(gens) - k for k in range(iteration))),
            "hamiltonian_matrix_pairs": basis.shape[1] *
            (basis.shape[1] + 1) // 2,
            "overlap_offdiagonal_pairs": basis.shape[1] *
            (basis.shape[1] - 1) // 2,
        }
        solution = solve_projected(
            S, Hm, basis_labels, tau_s=tau_s,
            max_condition=max_condition, resources=resources)
        records.append(AdaptGCIMIteration(
            iteration=iteration,
            selected_index=chosen,
            selected_label=gens[chosen].label,
            gradient=gradients[chosen],
            active_candidates=len(gradients),
            basis_size=basis.shape[1],
            basis_rotor_depths=tuple(basis_depths),
            ground_energy=solution.ground_energy,
            effective_rank=solution.effective_rank,
            condition_number=solution.condition_number,
        ))

    assert solution is not None
    return AdaptGCIMResult(
        result=solution,
        records=tuple(records),
        selected_indices=tuple(selected),
        selected_labels=tuple(gens[i].label for i in selected),
        theta=float(theta),
        basis=np.column_stack(basis_states),
        overlap_matrix=S,
        hamiltonian_matrix=Hm,
        basis_rotor_depths=tuple(basis_depths),
        surrogate_state=cumulative,
    )
