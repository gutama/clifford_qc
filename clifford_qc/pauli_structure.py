"""Binary structural diagnostics for sparse Pauli operators.

The X masks of Pauli words are packed binary rows.  Their GF(2) row rank is a
cheap invariant of the operator's support and, for a spin-conserving
Jordan--Wigner Hamiltonian on ``2N`` spin orbitals, obeys

``r_X <= 2(N - 1) = n_qubits - 2``.

The two missing directions are the independent spin-up and spin-down parity
checks.  A violation therefore means that the claimed construction path is not
spin conserving, or that the mapping/model construction is wrong.  Phase 13
applies this check to the declared construction matrix before Phase 14 starts;
the validator is also the explicit hook a future grouping boundary can call.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

from .ir import PauliSum
from .multivector import MV
from .pauli_action import word_masks


__all__ = [
    "gf2_rank",
    "spin_conserving_x_rank_ceiling",
    "validate_spin_conserving_x_rank",
    "x_mask_rank",
]


def gf2_rank(rows: Iterable[int], width: int) -> int:
    """Return the rank of packed binary ``rows`` over GF(2).

    ``width`` is explicit so an out-of-domain bit is rejected instead of being
    silently ignored by the elimination loop.
    """
    if not isinstance(width, Integral) or isinstance(width, bool):
        raise TypeError("width must be an integer")
    width = int(width)
    if width < 0:
        raise ValueError("width must be non-negative")

    work = []
    for row in rows:
        if not isinstance(row, Integral) or isinstance(row, bool):
            raise TypeError("GF(2) rows must be integers")
        row = int(row)
        if row < 0:
            raise ValueError("GF(2) rows must be non-negative")
        if row.bit_length() > width:
            raise ValueError(f"GF(2) row {row:#b} exceeds width {width}")
        if row:
            work.append(row)

    rank = 0
    for column in range(width):
        pivot = next(
            (index for index in range(rank, len(work))
             if (work[index] >> column) & 1),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for index in range(len(work)):
            if index != rank and ((work[index] >> column) & 1):
                work[index] ^= work[rank]
        rank += 1
        if rank == len(work):
            break
    return rank


def _as_mv(operator: MV | PauliSum) -> MV:
    if isinstance(operator, PauliSum):
        return operator.to_mv()
    if isinstance(operator, MV):
        return operator
    raise TypeError(f"expected MV or PauliSum, got {type(operator).__name__}")


def x_mask_rank(operator: MV | PauliSum) -> int:
    """Return the GF(2) rank of the operator's distinct Pauli X masks."""
    mv = _as_mv(operator)
    masks = {word_masks(mv.n, code)[0] for code in mv.terms}
    return gf2_rank(masks, mv.n)


def spin_conserving_x_rank_ceiling(n_spin_orbitals: int) -> int:
    """Return ``2(N - 1)`` for an even, non-empty spin-orbital register."""
    if not isinstance(n_spin_orbitals, Integral) or isinstance(n_spin_orbitals, bool):
        raise TypeError("n_spin_orbitals must be an integer")
    n_spin_orbitals = int(n_spin_orbitals)
    if n_spin_orbitals < 2 or n_spin_orbitals % 2:
        raise ValueError("spin-conserving rank requires a positive even spin-orbital count")
    return n_spin_orbitals - 2


def validate_spin_conserving_x_rank(operator: MV | PauliSum) -> int:
    """Return ``r_X`` or reject an operator the caller declares spin conserving.

    Passing this necessary structural check does not prove spin conservation;
    failing it disproves the claim.  Construction and grouping APIs remain
    generic, so a boundary that relies on the claim must call this validator.
    """
    mv = _as_mv(operator)
    rank = x_mask_rank(mv)
    ceiling = spin_conserving_x_rank_ceiling(mv.n)
    if rank > ceiling:
        raise ValueError(
            "spin-conserving Jordan-Wigner X-mask rank violated: "
            f"r_X={rank} exceeds {ceiling} for {mv.n} spin orbitals"
        )
    return rank
