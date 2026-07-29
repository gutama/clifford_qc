"""A-CASE: Rayleigh-Ritz in Clifford-algebra operator-response subspaces.

Phases 1-3 of ``ACASE_RESEARCH_PLAN.md``: exact fixed-basis solves, a cached
matrix-element bank with projected observables, and adaptive basis growth by
certified-by-construction generator selection. The basis states
``A_i|psi>`` stay virtual throughout -- every projected matrix element is an
expectation on one reference state, assembled from the bilinear Pauli-word
trace pairing ``MV.trace_pairing``.
"""

from .adaptive import (
    AdaptiveResult, CandidateScore, GrowthRecord, adapt_warm_start, run_acase,
    score_candidate, sector_leakage, select_candidate,
)
from .elements import MatrixElementBank
from .generators import (
    Generator, as_generators, commutator_response, fermionic_excitation_generators,
    identity_generator, krylov_response, pauli_orbit, response_hierarchy,
)
from .reference import (
    dense_basis, dense_projected_matrices, dense_residual_norm, dense_subspace,
    pure_statevector,
)
from .solver import (
    DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S, SubspaceResult,
    canonical_eigh, projected_matrices, solve_projected, solve_subspace,
)

__all__ = [
    "AdaptiveResult", "CandidateScore", "GrowthRecord", "adapt_warm_start",
    "run_acase", "score_candidate", "sector_leakage", "select_candidate",
    "Generator", "MatrixElementBank", "as_generators", "commutator_response",
    "fermionic_excitation_generators", "identity_generator", "krylov_response",
    "pauli_orbit", "response_hierarchy",
    "dense_basis", "dense_projected_matrices", "dense_residual_norm",
    "dense_subspace", "pure_statevector",
    "DEFAULT_MAX_CONDITION", "DEFAULT_NORM_FLOOR", "DEFAULT_TAU_S",
    "SubspaceResult", "canonical_eigh", "projected_matrices",
    "solve_projected", "solve_subspace",
]
