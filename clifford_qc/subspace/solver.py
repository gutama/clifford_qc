"""Compatibility facade for projected subspace construction and solving."""

from .contracts import as_multivector, check_reference
from .linalg import (
    DEFAULT_MAX_CONDITION,
    DEFAULT_NORM_FLOOR,
    DEFAULT_TAU_S,
    SubspaceResult,
    canonical_block,
    canonical_eigh,
    degenerate_blocks,
    fix_phase,
    solve_projected,
)
from .projection import ProjectedProblem, projected_matrices, solve_subspace

# Historical internal names retained for downstream tests and callers.
_as_mv = as_multivector
_check_reference = check_reference
_degenerate_blocks = degenerate_blocks
_fix_phase = fix_phase
_canonical_block = canonical_block

__all__ = [
    "DEFAULT_MAX_CONDITION",
    "DEFAULT_NORM_FLOOR",
    "DEFAULT_TAU_S",
    "ProjectedProblem",
    "SubspaceResult",
    "canonical_eigh",
    "projected_matrices",
    "solve_projected",
    "solve_subspace",
]
