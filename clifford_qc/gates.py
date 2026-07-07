from __future__ import annotations

import cmath
import math

from .multivector import MV, validate_qubit
from .pauli import I, X, Y, Z


def H(n: int, j: int) -> MV:
    return (1 / math.sqrt(2)) * (X(n, j) + Z(n, j))


def S(n: int, j: int) -> MV:
    return 0.5 * ((1 + 1j) * I(n) + (1 - 1j) * Z(n, j))


def T(n: int, j: int) -> MV:
    w = cmath.exp(1j * math.pi / 4)
    return 0.5 * ((1 + w) * I(n) + (1 - w) * Z(n, j))

# Backward-friendly aliases.
H_gate = H
S_gate = S
T_gate = T


def rotor(P: MV, theta: float, *, check_word: bool = False) -> MV:
    """Exact exp(-i theta P/2) for Hermitian Pauli-word generators P^2=1."""
    if check_word:
        if P.nnz() != 1 or not (P * P).is_close(I(P.n), 1e-9) or not P.is_hermitian():
            raise ValueError("rotor expects a Hermitian Pauli-word generator with P^2=1")
    return math.cos(theta / 2) * I(P.n) - 1j * math.sin(theta / 2) * P


def RX(n: int, j: int, theta: float) -> MV:
    return rotor(X(n, j), theta)


def RY(n: int, j: int, theta: float) -> MV:
    return rotor(Y(n, j), theta)


def RZ(n: int, j: int, theta: float) -> MV:
    return rotor(Z(n, j), theta)


def controlled(U: MV, ctrl: int, *, check_identity_on_control: bool = False) -> MV:
    n = U.n
    validate_qubit(n, ctrl, "ctrl")
    if check_identity_on_control:
        bad = [code for code in U.terms if ((code >> (2 * ctrl)) & 3) != 0]
        if bad:
            raise ValueError("controlled(U, ctrl) expects U to be identity on ctrl")
    return 0.5 * (I(n) + Z(n, ctrl)) + 0.5 * (I(n) - Z(n, ctrl)) * U


def CNOT(n: int, c: int, t: int) -> MV:
    return controlled(X(n, t), c, check_identity_on_control=True)


def CZ(n: int, c: int, t: int) -> MV:
    return controlled(Z(n, t), c, check_identity_on_control=True)


def SWAP(n: int, a: int, b: int) -> MV:
    validate_qubit(n, a, "a")
    validate_qubit(n, b, "b")
    if a == b:
        return I(n)
    return 0.5 * (I(n) + X(n, a) * X(n, b) + Y(n, a) * Y(n, b) + Z(n, a) * Z(n, b))


def TOFFOLI(n: int, c1: int, c2: int, t: int) -> MV:
    return controlled(controlled(X(n, t), c2, check_identity_on_control=True), c1, check_identity_on_control=True)


def expm_taylor(A: MV, order: int = 40) -> MV:
    """Scaling-and-squaring Taylor exp(A), intended for small/validation use."""
    s = max(0, int(math.ceil(math.log2(max(A.norm_hs(), 1e-30)))) + 1)
    B = (2.0 ** (-s)) * A
    term, out = I(A.n), I(A.n)
    for k in range(1, order + 1):
        term = (1.0 / k) * (term * B)
        out = out + term
    for _ in range(s):
        out = out * out
    return out


def _check_h_terms(H_terms: list[tuple[float, MV]]) -> int:
    if not H_terms:
        raise ValueError("H_terms must be non-empty")
    n = H_terms[0][1].n
    for _, P in H_terms:
        if P.n != n:
            raise ValueError("all Hamiltonian terms must live in the same algebra")
    return n


def trotter_unitary(H_terms: list[tuple[float, MV]], t: float, steps: int) -> MV:
    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = t / steps
    n = _check_h_terms(H_terms)
    step = I(n)
    for c, P in H_terms:
        step = step * rotor(P, 2 * c * dt)
    U = I(n)
    for _ in range(steps):
        U = U * step
    return U


def trotter2_unitary(H_terms: list[tuple[float, MV]], t: float, steps: int) -> MV:
    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = t / steps
    n = _check_h_terms(H_terms)
    half_step = I(n)
    for c, P in H_terms:
        half_step = half_step * rotor(P, c * dt)
    step = half_step
    for c, P in reversed(H_terms):
        step = step * rotor(P, c * dt)
    U = I(n)
    for _ in range(steps):
        U = U * step
    return U
