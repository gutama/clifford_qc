"""Dense cross-check for the A-CASE projected matrices (validation only).

The operator route assembles ``S`` and ``H`` from Pauli-word pairings against a
single reference density, never touching a state vector. This module does the
opposite: it materializes ``|psi>``, materializes every basis state
``A_i|psi>``, and forms the Gram and Hamiltonian matrices with dense linear
algebra. Agreement between the two is the Phase-1 correctness statement -- and
the one that would catch a conjugation or reversion slip in the pairing, since
the dense route has no word coordinates to get wrong.

Dense throughout, so it is a small-``n`` oracle, exactly as ``matrix.py`` is
for the algebra kernel. Nothing here belongs on a scaling path.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..matrix import to_matrix
from ..multivector import MV
from .generators import as_generators
from .solver import SubspaceResult, solve_projected


def pure_statevector(rho: MV, tol: float = 1e-9) -> np.ndarray:
    """The state vector of a pure density multivector, phase-fixed.

    Raises when ``rho`` is mixed: the dense route is written for the pure
    reference the method assumes, and silently returning a leading eigenvector
    of a mixed state would make the cross-check agree for the wrong reason.
    """
    values, vectors = np.linalg.eigh(to_matrix(rho))
    if abs(values[-1] - 1.0) > tol or abs(values[-2]) > tol:
        raise ValueError("reference density is not a pure state")
    psi = vectors[:, -1]
    lead = psi[int(np.argmax(np.abs(psi)))]
    return psi * (np.conjugate(lead) / abs(lead))


def dense_basis(rho: MV, generators: Sequence) -> np.ndarray:
    """Columns ``A_i|psi>`` -- the states the operator route never prepares."""
    psi = pure_statevector(rho)
    return np.column_stack([to_matrix(g.mv) @ psi for g in as_generators(generators)])


def dense_projected_matrices(rho: MV, hamiltonian, generators: Sequence
                             ) -> tuple[np.ndarray, np.ndarray]:
    """``(S, H)`` from explicit basis vectors: ``S = B' B``, ``H = B' H_dense B``."""
    basis = dense_basis(rho, generators)
    H_dense = to_matrix(hamiltonian if isinstance(hamiltonian, MV)
                        else hamiltonian.to_mv())
    return basis.conj().T @ basis, basis.conj().T @ H_dense @ basis


def dense_subspace(rho: MV, hamiltonian, generators: Sequence, **kwargs) -> SubspaceResult:
    """Same thresholded GEP, fed by the dense matrices instead of the pairings."""
    gens = as_generators(generators)
    S, Hm = dense_projected_matrices(rho, hamiltonian, gens)
    return solve_projected(S, Hm, [g.label for g in gens], **kwargs)
