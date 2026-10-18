"""Incremental exact Ritz solves from cached Hamiltonian actions.

This classical interface stores explicit vectors. It supplies neither quantum
state preparation nor measured matrix elements, and does not promise sparse
state storage or polynomial scaling.
"""
from __future__ import annotations

import numpy as np

from .orthogonal import canonical_phase


def orthogonal_direction(vector, basis, *, cutoff=1e-10):
    """Twice project against the entire orthonormal span; reject dependence."""
    vector = np.array(vector, dtype=complex, copy=True)
    basis = np.asarray(basis, dtype=complex)
    if (vector.ndim != 1 or basis.ndim != 2 or basis.shape[0] != vector.size
            or not np.isfinite(vector).all() or not np.isfinite(basis).all()
            or not np.isfinite(cutoff) or cutoff <= 0):
        raise ValueError("finite matching vector/basis and positive cutoff required")
    for _ in range(2):
        vector -= basis @ (basis.conj().T @ vector)
    norm = float(np.linalg.norm(vector))
    return None if norm <= cutoff else canonical_phase(vector / norm)


def energy_lowering(energy, diagonal, coupling):
    """Exact two-dimensional lowering for one orthogonal normalized direction."""
    gap = float(diagonal - energy)
    return max(0.0, float((np.hypot(gap, 2 * abs(coupling)) - gap) / 2))


class ActionSpace:
    """Keep V and HV, applying H only to newly supplied columns.

    ``extend(..., actions=...)`` accepts caller-owned, already counted actions
    (e.g. from candidate screening). Their correctness is the caller's contract;
    their shapes, finiteness and projected Hermiticity are checked here.
    """

    def __init__(self, apply, basis, *, tolerance=1e-10):
        if not callable(apply) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("an action callback and positive tolerance required")
        basis = np.asarray(basis, dtype=complex)
        if basis.ndim != 2 or not all(basis.shape):
            raise ValueError("a nonempty matrix of basis columns required")
        self.apply, self.tolerance = apply, tolerance
        self.basis = np.empty((basis.shape[0], 0), complex)
        self.actions = self.basis.copy()
        self.h_action_calls = 0
        self.extend(basis)

    def action(self, vector):
        vector = np.asarray(vector, dtype=complex)
        if vector.shape != (self.basis.shape[0],) or not np.isfinite(vector).all():
            raise ValueError("invalid action input vector")
        result = np.asarray(self.apply(vector), dtype=complex)
        self.h_action_calls += 1
        if result.shape != vector.shape or not np.isfinite(result).all():
            raise ValueError("Hamiltonian action returned an invalid vector")
        return result.copy()

    def extend(self, columns, *, actions=None):
        columns = np.asarray(columns, dtype=complex)
        if (columns.ndim != 2 or columns.shape[0] != self.basis.shape[0]
                or not columns.shape[1] or not np.isfinite(columns).all()):
            raise ValueError("finite matching nonempty columns required")
        combined = np.column_stack((self.basis, columns))
        if np.max(abs(combined.conj().T @ combined - np.eye(combined.shape[1]))) > self.tolerance:
            raise ValueError("new columns must be orthonormal against the whole span")
        if actions is None:
            actions = np.column_stack([self.action(column) for column in columns.T])
        actions = np.asarray(actions, dtype=complex)
        if actions.shape != columns.shape or not np.isfinite(actions).all():
            raise ValueError("cached actions must match the new columns")
        combined_actions = np.column_stack((self.actions, actions))
        projected = combined.conj().T @ combined_actions
        scale = max(1., float(np.max(abs(projected))))
        if np.max(abs(projected - projected.conj().T)) > self.tolerance * scale:
            raise ValueError("projected Hamiltonian action must be Hermitian")
        self.basis, self.actions = combined, combined_actions

    def solve(self):
        projected = self.basis.conj().T @ self.actions
        values, coefficients = np.linalg.eigh((projected + projected.conj().T) / 2)
        state = canonical_phase(self.basis @ coefficients[:, 0])
        c = self.basis.conj().T @ state
        residual = self.actions @ c - values[0] * state
        return float(values[0]), state, residual

    @property
    def payload_bytes(self):
        return self.basis.nbytes + self.actions.nbytes
