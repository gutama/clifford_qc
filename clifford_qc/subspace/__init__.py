"""A-CASE: Rayleigh-Ritz in Clifford-algebra operator-response subspaces.

Phase 1 of ``ACASE_RESEARCH_PLAN.md``: exact, fixed-basis solves. The basis
states ``A_i|psi>`` stay virtual -- every projected matrix element is an
expectation on one reference state, assembled from the bilinear Pauli-word
trace pairing ``MV.trace_pairing``.
"""

from .generators import (
    Generator, as_generators, commutator_response, identity_generator,
    krylov_response, pauli_orbit, response_hierarchy,
)
from .reference import (
    dense_basis, dense_projected_matrices, dense_subspace, pure_statevector,
)
from .solver import (
    DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S, SubspaceResult,
    canonical_eigh, projected_matrices, solve_projected, solve_subspace,
)

__all__ = [
    "Generator", "as_generators", "commutator_response", "identity_generator",
    "krylov_response", "pauli_orbit", "response_hierarchy",
    "dense_basis", "dense_projected_matrices", "dense_subspace",
    "pure_statevector",
    "DEFAULT_MAX_CONDITION", "DEFAULT_NORM_FLOOR", "DEFAULT_TAU_S",
    "SubspaceResult", "canonical_eigh", "projected_matrices",
    "solve_projected", "solve_subspace",
]
