"""Dense-matrix reference backend.

An independent numerical cross-check built on ``matrix.to_matrix``; it is
never the native representation. Useful for validating the MV path and for
somewhat larger pure-state checks.
"""

from __future__ import annotations

import numpy as np

from ..matrix import to_matrix
from ..multivector import MV
from ..states import ket_density
from ..ir import PauliSum, Program


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
        from ..matrix import density_from_statevector
        return density_from_statevector(self._vector(program, values, initial_state))

    def expectation(self, program: Program, observable: PauliSum, values=None,
                    initial_state: MV | None = None) -> float:
        psi = self._vector(program, values, initial_state)
        O = to_matrix(observable.to_mv())
        return float(np.real(np.vdot(psi, O @ psi)))
