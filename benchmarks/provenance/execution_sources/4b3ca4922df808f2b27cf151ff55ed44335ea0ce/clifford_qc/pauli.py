from __future__ import annotations

from .multivector import MV, validate_n, validate_qubit


def I(n: int) -> MV:
    validate_n(n)
    return MV.scalar(n, 1.0)


def X(n: int, j: int) -> MV:
    validate_qubit(n, j)
    return MV.word(n, "I" * j + "X" + "I" * (n - j - 1))


def Y(n: int, j: int) -> MV:
    validate_qubit(n, j)
    return MV.word(n, "I" * j + "Y" + "I" * (n - j - 1))


def Z(n: int, j: int) -> MV:
    validate_qubit(n, j)
    return MV.word(n, "I" * j + "Z" + "I" * (n - j - 1))


def P(label: str, c: complex = 1.0) -> MV:
    """Readable Pauli word constructor: P('XIZ') -> MV in n=3."""
    return MV.from_label(label, c)


def pauli_string(n: int, label: str, c: complex = 1.0) -> MV:
    return MV.word(n, label, c)


def comm(A: MV, B: MV) -> MV:
    return A * B - B * A


def anticomm(A: MV, B: MV) -> MV:
    return A * B + B * A


def pauli_basis(n: int) -> list[MV]:
    validate_n(n)
    return [MV(n, {code: 1.0}) for code in range(4 ** n)]


def tensor(A: MV, B: MV) -> MV:
    """Tensor product with B appended after A: matrix(tensor(A,B)) = kron(A,B)."""
    out: dict[int, complex] = {}
    for ca, va in A.terms.items():
        for cb, vb in B.terms.items():
            code = ca | (cb << (2 * A.n))
            out[code] = out.get(code, 0.0) + va * vb
    return MV(A.n + B.n, out)
