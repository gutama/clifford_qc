"""Particle-number and spin-sector diagnostics for subspace generators."""

from __future__ import annotations

from functools import lru_cache

from ..multivector import MV
from .contracts import as_multivector


@lru_cache(maxsize=32)
def sector_operators(n: int) -> tuple[MV, MV]:
    from ..fermion import total_number_op, total_sz_op
    return total_number_op(n), total_sz_op(n)


def sector_leakage(generator) -> dict[str, float]:
    """Relative operator-level leakage out of the (N, Sz) sectors."""
    operator = getattr(generator, "mv", generator)
    A = as_multivector(operator)
    norm = A.norm_hs()
    if norm <= 0.0:
        return {"particle_number": 0.0, "sz": 0.0}
    number, sz = sector_operators(A.n)
    return {
        "particle_number": (A * number - number * A).norm_hs() / norm,
        "sz": (A * sz - sz * A).norm_hs() / norm,
    }
