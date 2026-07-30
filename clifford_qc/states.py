from __future__ import annotations

from .multivector import MV, validate_n, validate_qubit
from .pauli import I, Z
from .gates import H, CNOT


def ket_density(n: int, bits: str) -> MV:
    """Computational basis density |bits><bits|."""
    validate_n(n)
    if len(bits) != n or any(b not in "01" for b in bits):
        raise ValueError(f"bits must be a length-{n} string over '0'/'1'")
    rho = I(n)
    for j, b in enumerate(bits):
        rho = rho * (0.5 * (I(n) + (1 if b == "0" else -1) * Z(n, j)))
    return rho


def computational_projector(n: int, bits: str) -> MV:
    return ket_density(n, bits)


def plus_density(n: int) -> MV:
    rho = ket_density(n, "0" * n)
    for j in range(n):
        rho = evolve(rho, H(n, j))
    return rho


def evolve(rho: MV, U: MV) -> MV:
    if rho.n != U.n:
        raise ValueError("state and unitary live in different algebras")
    return U * rho * U.dagger()


def expectation(rho: MV, O: MV) -> complex:
    if rho.n != O.n:
        raise ValueError("state and observable live in different algebras")
    return (O * rho).trace()


def probability(rho: MV, projector: MV, *, clip: bool = True) -> float:
    p = (projector * rho).trace().real
    if clip and abs(p) < 1e-12:
        p = 0.0
    if clip and abs(p - 1.0) < 1e-12:
        p = 1.0
    return p


def purity(rho: MV) -> float:
    return (rho * rho).trace().real


def computational_probabilities(rho: MV) -> dict[str, float]:
    """All ``2^n`` computational-basis probabilities of a density operator.

    A Z-basis readout only sees the *diagonal* Pauli content of ``rho`` -- the
    words spelled from ``I`` and ``Z`` alone. Writing ``|b><b|`` as
    ``prod_j (I + s_j Z_j)/2`` and expanding, its only nonzero Pauli
    coefficients are those diagonal words, each carrying
    ``(-1)^popcount(z & b) / 2^n``, so

        p(b) = sum_z rho_z (-1)^popcount(z & b),

    which is the Walsh-Hadamard transform of the diagonal coefficients indexed
    by their qubit mask. That is ``O(2^n n)`` for every outcome at once, against
    the ``O(4^n)`` word products of building each ``ket_density`` projector and
    multiplying it out -- one eight-qubit readout was 11 s, which made grouped
    finite-shot sampling unaffordable past four qubits (every QWC group needs
    exactly this distribution).

    Same values as the projector route, including its snapping of values within
    ``1e-12`` of 0 or 1.
    """
    import numpy as np

    n = rho.n
    size = 1 << n
    diagonal = np.zeros(size)
    for code, coeff in rho.terms.items():
        mask = 0
        for j in range(n):
            letter = (code >> (2 * j)) & 3
            if letter == 0:
                continue
            if letter != 3:  # an X or Y anywhere: traceless against every |b><b|
                break
            mask |= 1 << (n - 1 - j)  # bit order of format(k, "0nb"): qubit 0 first
        else:
            diagonal[mask] += coeff.real

    # In-place fast Walsh-Hadamard transform: H[b][m] = (-1)^popcount(b & m).
    step = 1
    while step < size:
        block = diagonal.reshape(-1, 2, step)
        low = block[:, 0, :].copy()
        high = block[:, 1, :]
        block[:, 0, :] = low + high
        block[:, 1, :] = low - high
        step *= 2

    out = {}
    for k in range(size):
        p = float(diagonal[k])
        if abs(p) < 1e-12:
            p = 0.0
        elif abs(p - 1.0) < 1e-12:
            p = 1.0
        out[format(k, f"0{n}b")] = p
    return out


def measure(rho: MV, projectors: list[MV], *, check_projectors: bool = False):
    out = []
    for P in projectors:
        if P.n != rho.n:
            raise ValueError("projector and density operator have different n")
        if check_projectors and not P.is_projector():
            raise ValueError("measurement element is not an orthogonal projector")
        p = probability(rho, P, clip=False)
        post = P * rho * P
        if p > 1e-12:
            post = (1.0 / p) * post
        out.append((p, post))
    return out


def z_projectors(n: int, j: int):
    validate_qubit(n, j)
    return [0.5 * (I(n) + Z(n, j)), 0.5 * (I(n) - Z(n, j))]


def partial_trace(rho: MV, traced: set[int]) -> MV:
    for j in traced:
        validate_qubit(rho.n, j)
    keep = [j for j in range(rho.n) if j not in traced]
    out: dict[int, complex] = {}
    for code, coeff in rho.terms.items():
        if any(((code >> (2 * j)) & 3) for j in traced):
            continue
        new = 0
        for pos, j in enumerate(keep):
            new |= ((code >> (2 * j)) & 3) << (2 * pos)
        out[new] = out.get(new, 0.0) + coeff * (2 ** len(traced))
    return MV(len(keep), out)


def partial_transpose(rho: MV, qubits: set[int]) -> MV:
    for j in qubits:
        validate_qubit(rho.n, j)
    out: dict[int, complex] = {}
    for code, coeff in rho.terms.items():
        sign = (-1) ** sum(1 for j in qubits if ((code >> (2 * j)) & 3) == 2)
        out[code] = sign * coeff
    return MV(rho.n, out)


def bell_density() -> MV:
    return evolve(ket_density(2, "00"), CNOT(2, 0, 1) * H(2, 0))


def ghz_density(n: int = 3) -> MV:
    if n < 2:
        raise ValueError("GHZ needs at least two qubits")
    rho = evolve(ket_density(n, "0" * n), H(n, 0))
    for j in range(n - 1):
        rho = evolve(rho, CNOT(n, j, j + 1))
    return rho
