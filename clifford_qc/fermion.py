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


def total_number_op(n: int) -> MV:
    """Total particle number N = sum_j n_j over all spin orbitals."""
    out = MV(n)
    for j in range(n):
        out = out + number_op_pauli(n, j)
    return out


def total_sz_op(n: int) -> MV:
    """Total spin projection S_z = 1/2 sum_j (-1)^j n_j.

    Interleaved Jordan-Wigner ordering, the convention the OpenFermion bridge
    and the chemistry models use: even spin-orbital indices carry spin up and
    odd ones spin down. A generator that conserves particle number need not
    conserve this -- two spin-up electrons excited into two spin-down orbitals
    keep ``N`` and change ``S_z`` by 2 -- which is why the two are separate
    diagnostics rather than one symmetry check.
    """
    out = MV(n)
    for j in range(n):
        out = out + (0.5 if j % 2 == 0 else -0.5) * number_op_pauli(n, j)
    return out
