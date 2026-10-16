"""Dense-state reference backend.

Program evolution is an independent small-system cross-check built on
``dense_reference.to_matrix``; it is never the native representation.
Observable expectations use packed Pauli action directly, so they do not
materialize a second dense operator.
"""

from __future__ import annotations

import numpy as np

from ..dense_reference import density_from_statevector, to_matrix
from ..ir import PauliSum, Program
from ..multivector import MV
from ..pauli_action import apply_pauli_sum


class DenseStatevectorBackend:
    def _vector(self, program: Program, values, initial_state):
        n = program.n
        if initial_state is None:
            psi = np.zeros(2 ** n, dtype=complex)
            psi[0] = 1.0
        else:
            # accept a pure density MV and take its dominant eigenvector
            rho = to_matrix(initial_state)
            vals, vecs = np.linalg.eigh(rho)
            if not np.isclose(vals[-1], 1.0, atol=1e-9):
                raise ValueError("dense backend needs a pure initial state")
            psi = vecs[:, -1]
        U = to_matrix(program.unitary(values))
        return U @ psi

    def state(self, program: Program, values=None, initial_state: MV | None = None) -> MV:
        return density_from_statevector(self._vector(program, values, initial_state))

    def expectation(self, program: Program, observable: PauliSum, values=None,
        initial_state: MV | None = None) -> float:
        psi = self._vector(program, values, initial_state)
        return float(np.real(np.vdot(psi, apply_pauli_sum(observable, psi))))
