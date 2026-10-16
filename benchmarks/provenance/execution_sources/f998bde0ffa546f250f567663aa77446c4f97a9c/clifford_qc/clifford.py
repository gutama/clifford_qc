from __future__ import annotations

from .multivector import MV, validate_n, validate_qubit
from .pauli import I


def gamma(n: int, i: int) -> MV:
    """Jordan-Wigner Clifford generator.

    gamma_{2j}   = Z_0 ... Z_{j-1} X_j
    gamma_{2j+1} = Z_0 ... Z_{j-1} Y_j
    """
    validate_n(n)
    if not isinstance(i, int) or not (0 <= i < 2 * n):
        raise ValueError(f"gamma index must be in [0, {2*n}), got {i!r}")
    j, kind = divmod(i, 2)
    label = "Z" * j + ("X" if kind == 0 else "Y") + "I" * (n - j - 1)
    return MV.word(n, label)


def pseudoscalar(n: int) -> MV:
    out = I(n)
    for i in range(2 * n):
        out = out * gamma(n, i)
    return out


def blade_from_mask(n: int, mask: int) -> MV:
    validate_n(n)
    if not isinstance(mask, int) or not (0 <= mask < (1 << (2 * n))):
        raise ValueError(f"mask must be in [0, 2**(2n)), got {mask!r}")
    out = I(n)
    for i in range(2 * n):
        if (mask >> i) & 1:
            out = out * gamma(n, i)
    return out


def grade(A: MV, g: int) -> MV:
    return A.grade(g)


def grades(A: MV) -> set[int]:
    return A.grades()
