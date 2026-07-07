from __future__ import annotations

from .multivector import MV, validate_qubit
from .clifford import gamma
from .pauli import I, Z


def c_op(n: int, j: int) -> MV:
    """Fermionic annihilation operator c_j = 1/2(gamma_2j + i gamma_2j+1)."""
    validate_qubit(n, j)
    return 0.5 * (gamma(n, 2 * j) + 1j * gamma(n, 2 * j + 1))


def cdag_op(n: int, j: int) -> MV:
    validate_qubit(n, j)
    return 0.5 * (gamma(n, 2 * j) - 1j * gamma(n, 2 * j + 1))


def number_op(n: int, j: int) -> MV:
    validate_qubit(n, j)
    return cdag_op(n, j) * c_op(n, j)


def number_op_pauli(n: int, j: int) -> MV:
    """Equivalent qubit expression n_j = 1/2(1 - Z_j)."""
    validate_qubit(n, j)
    return 0.5 * (I(n) - Z(n, j))
