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


def _rotor_unchecked(P: MV, theta: float) -> MV:
    """exp(-i theta P/2) assuming the caller guarantees P is an involution.

    For internal callers that build P themselves from a typed Pauli word, so
    the precondition holds by construction and needs no runtime product.
    """
    return math.cos(theta / 2) * I(P.n) - 1j * math.sin(theta / 2) * P


def _require_involution(P: MV) -> None:
    """Reject generators for which the closed-form rotor is simply false.

    ``P^2 = 1`` is the whole reason the exponential resums to cos/sin: it
    collapses every even power of the series. It is also the *exact*
    discriminator, which the previous word-shaped test was not, in either
    direction -- ``H = (X+Z)/sqrt(2)`` is a legitimate involution with two
    terms, while ``2X`` is a single term and is not one.
    """
    if not P.is_hermitian():
        raise ValueError("rotor expects a Hermitian generator")
    if not (P * P).is_close(I(P.n), 1e-9):
        raise ValueError(
            "rotor expects a generator with P^2 = 1; got one with P^2 != 1, "
            "for which cos(theta/2) - i sin(theta/2) P is neither exp(-i theta P/2) "
            "nor unitary (X + Z is a typical mistake -- (X + Z)/sqrt(2) is the involution)")


def rotor(P: MV, theta: float, *, check_word: bool = True) -> MV:
    """Exact exp(-i theta P/2) for Hermitian generators with P^2 = 1.

    Validated by default. The closed form is exact only for a Hermitian
    involution, and a caller who passes anything else gets a multivector that
    is neither the exponential nor unitary, with no indication of it -- a
    silently invalid propagator rather than an error.
    """
    if check_word:
        _require_involution(P)
    return _rotor_unchecked(P, theta)


def RX(n: int, j: int, theta: float) -> MV:
    return _rotor_unchecked(X(n, j), theta)


def RY(n: int, j: int, theta: float) -> MV:
    return _rotor_unchecked(Y(n, j), theta)


def RZ(n: int, j: int, theta: float) -> MV:
    return _rotor_unchecked(Z(n, j), theta)


def controlled(U: MV, ctrl: int, *, check_identity_on_control: bool = True) -> MV:
    """Control U on qubit ``ctrl``, which U must not itself act on.

    Checked by default: the projector form below assumes U commutes with the
    control's Z, and if U acts on ``ctrl`` the result is not unitary
    (``controlled(X_0, ctrl=0)`` yields C^dag C = I + Z).
    """
    n = U.n
    validate_qubit(n, ctrl, "ctrl")
    if check_identity_on_control:
        bad = [code for code in U.terms if ((code >> (2 * ctrl)) & 3) != 0]
        if bad:
            raise ValueError("controlled(U, ctrl) expects U to be identity on ctrl")
    return 0.5 * (I(n) + Z(n, ctrl)) + 0.5 * (I(n) - Z(n, ctrl)) * U


def CNOT(n: int, c: int, t: int) -> MV:
    return controlled(X(n, t), c)


def CZ(n: int, c: int, t: int) -> MV:
    return controlled(Z(n, t), c)


def SWAP(n: int, a: int, b: int) -> MV:
    validate_qubit(n, a, "a")
    validate_qubit(n, b, "b")
    if a == b:
        return I(n)
    return 0.5 * (I(n) + X(n, a) * X(n, b) + Y(n, a) * Y(n, b) + Z(n, a) * Z(n, b))


def TOFFOLI(n: int, c1: int, c2: int, t: int) -> MV:
    return controlled(controlled(X(n, t), c2), c1)


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
