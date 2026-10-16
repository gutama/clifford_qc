from __future__ import annotations

import math

from .multivector import MV, validate_qubit
from .pauli import I, X, Y, Z


def check_kraus(kraus: list[MV], tol: float = 1e-9) -> bool:
    if not kraus:
        raise ValueError("kraus list must be non-empty")
    n = kraus[0].n
    acc = MV(n)
    for K in kraus:
        if K.n != n:
            raise ValueError("all Kraus operators must live in the same algebra")
        acc = acc + K.dagger() * K
    return acc.is_close(I(n), tol)


def apply_channel(rho: MV, kraus: list[MV], *, check: bool = False) -> MV:
    if check and not check_kraus(kraus):
        raise ValueError("Kraus operators do not satisfy Σ K†K = 1")
    out = MV(rho.n)
    for K in kraus:
        if K.n != rho.n:
            raise ValueError("Kraus operator and rho have different n")
        out = out + K * rho * K.dagger()
    return out


def depolarizing(n: int, j: int, p: float):
    validate_qubit(n, j)
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0,1]")
    q = math.sqrt(p / 3)
    return [math.sqrt(1 - p) * I(n), q * X(n, j), q * Y(n, j), q * Z(n, j)]


def dephasing(n: int, j: int, p: float):
    validate_qubit(n, j)
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0,1]")
    return [math.sqrt(1 - p) * I(n), math.sqrt(p) * Z(n, j)]


def amplitude_damping(n: int, j: int, gamma: float):
    validate_qubit(n, j)
    if not (0 <= gamma <= 1):
        raise ValueError("gamma must be in [0,1]")
    P0 = 0.5 * (I(n) + Z(n, j))
    P1 = 0.5 * (I(n) - Z(n, j))
    sigma_minus = 0.5 * (X(n, j) + 1j * Y(n, j))  # |0><1|
    return [P0 + math.sqrt(1 - gamma) * P1, math.sqrt(gamma) * sigma_minus]
