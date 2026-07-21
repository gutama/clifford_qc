from __future__ import annotations

import math
from functools import lru_cache
from numbers import Number
from typing import Dict, Iterable

TOL = 1e-12
PAULI_LETTERS = {0: "I", 1: "X", 2: "Y", 3: "Z"}
LETTER_CODE = {v: k for k, v in PAULI_LETTERS.items()}

# Single-qubit Pauli multiplication table: (left, right) -> (phase, result).
_PTAB: dict[tuple[int, int], tuple[complex, int]] = {}
for a in range(4):
    _PTAB[(0, a)] = (1 + 0j, a)
    _PTAB[(a, 0)] = (1 + 0j, a)
    _PTAB[(a, a)] = (1 + 0j, 0)
_PTAB[(1, 2)] = (1j, 3);   _PTAB[(2, 1)] = (-1j, 3)  # XY = iZ
_PTAB[(2, 3)] = (1j, 1);   _PTAB[(3, 2)] = (-1j, 1)  # YZ = iX
_PTAB[(3, 1)] = (1j, 2);   _PTAB[(1, 3)] = (-1j, 2)  # ZX = iY


def validate_n(n: int) -> None:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")


def validate_qubit(n: int, j: int, name: str = "j") -> None:
    validate_n(n)
    if not isinstance(j, int) or not (0 <= j < n):
        raise ValueError(f"{name} must be an int in [0, {n}), got {j!r}")


def validate_word_code(n: int, code: int) -> None:
    validate_n(n)
    if not isinstance(code, int) or not (0 <= code < 4 ** n):
        raise ValueError(f"word code must be an int in [0, 4**n), got {code!r}")


def label_to_code(label: str, n: int | None = None) -> int:
    """Encode a Pauli label such as ``'XIZ'``. Qubit 0 is the leftmost letter."""
    label = label.upper()
    if n is None:
        n = len(label)
    validate_n(n)
    if len(label) != n:
        raise ValueError(f"expected {n} Pauli letters, got {len(label)}")
    code = 0
    for j, ch in enumerate(label):
        if ch not in LETTER_CODE:
            raise ValueError(f"invalid Pauli letter {ch!r}; use I, X, Y, Z")
        code |= LETTER_CODE[ch] << (2 * j)
    return code


def code_to_label(n: int, code: int) -> str:
    validate_word_code(n, code)
    return "".join(PAULI_LETTERS[(code >> (2 * j)) & 3] for j in range(n))


def _word_mul_ref(n: int, a: int, b: int) -> tuple[complex, int]:
    """Reference Pauli-word product: per-qubit table lookup, O(n).

    Retained as the correctness oracle for the packed ``word_mul`` below and
    exercised directly by the cross-validation tests.
    """
    validate_word_code(n, a)
    validate_word_code(n, b)
    phase, out = 1 + 0j, 0
    for j in range(n):
        la = (a >> (2 * j)) & 3
        lb = (b >> (2 * j)) & 3
        ph, lc = _PTAB[(la, lb)]
        phase *= ph
        out |= lc << (2 * j)
    return phase, out


# i^e for e in {0,1,2,3}: the only phases a Pauli-word product can carry.
_PHASE4 = (1 + 0j, 1j, -1 + 0j, -1j)


@lru_cache(maxsize=1024)
def _lane_mask(n: int) -> int:
    """Bit 0 of every 2-bit lane set: 0b...010101 over ``n`` lanes."""
    return ((1 << (2 * n)) - 1) // 3 if n else 0


@lru_cache(maxsize=1_000_000)
def word_mul(n: int, a: int, b: int) -> tuple[complex, int]:
    """Multiply two encoded Pauli words in the same n-qubit algebra.

    Packed binary-symplectic form. Writing each single-qubit letter as
    ``i^{xz} X^x Z^z`` (so ``I,X,Y,Z`` stay Hermitian), the product word is
    the lane-wise XOR ``a ^ b`` and the accumulated phase is ``i^e`` with

        e = (x_a·z_a) + (x_b·z_b) - (x_c·z_c) + 2 (z_a·x_b)   (mod 4),

    each dot product a popcount of an AND over the packed x/z bit planes.
    This replaces the O(n) per-qubit loop with a handful of bitwise ops and
    ``int.bit_count()`` calls; ``_word_mul_ref`` is the equivalent reference.
    """
    validate_word_code(n, a)
    validate_word_code(n, b)
    lo = _lane_mask(n)
    za = (a >> 1) & lo
    xa = (a & lo) ^ za
    zb = (b >> 1) & lo
    xb = (b & lo) ^ zb
    c = a ^ b
    zc = (c >> 1) & lo
    xc = (c & lo) ^ zc
    e = ((xa & za).bit_count() + (xb & zb).bit_count()
         - (xc & zc).bit_count() + 2 * (za & xb).bit_count()) % 4
    return _PHASE4[e], c


class MV:
    """Sparse multivector/operator in the Pauli-word basis of Cl(2n,C).

    The object represents an element of the complexified Clifford algebra
    ``Cl(2n,C)`` through the Jordan-Wigner image of its blades. Multiplication
    is the geometric/operator product, implemented as Pauli-word multiplication.
    """

    __slots__ = ("n", "terms")

    def __init__(self, n: int, terms: dict[int, complex] | None = None):
        validate_n(n)
        self.n = n
        self.terms: dict[int, complex] = {}
        if terms:
            for k, v in terms.items():
                kk = int(k)
                validate_word_code(n, kk)
                cv = complex(v)
                if abs(cv) > TOL:
                    self.terms[kk] = self.terms.get(kk, 0.0) + cv
            self.terms = {k: v for k, v in self.terms.items() if abs(v) > TOL}

    @staticmethod
    def scalar(n: int, c: complex = 1.0) -> "MV":
        return MV(n, {0: c})

    @staticmethod
    def word(n: int, label: str, c: complex = 1.0) -> "MV":
        return MV(n, {label_to_code(label, n): c})

    @staticmethod
    def from_label(label: str, c: complex = 1.0) -> "MV":
        return MV.word(len(label), label, c)

    @staticmethod
    def from_terms(n: int, terms: dict[str, complex]) -> "MV":
        out = MV(n)
        for label, coeff in terms.items():
            out = out + MV.word(n, label, coeff)
        return out

    def copy(self) -> "MV":
        return MV(self.n, dict(self.terms))

    def _coerce(self, other) -> "MV":
        if isinstance(other, Number):
            return MV.scalar(self.n, complex(other))
        if isinstance(other, MV):
            if other.n != self.n:
                raise ValueError("mixed algebra sizes")
            return other
        raise TypeError(type(other))

    def __add__(self, other):
        other = self._coerce(other)
        terms = dict(self.terms)
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0.0) + v
        return MV(self.n, terms)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1) * self._coerce(other)

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __neg__(self):
        return (-1) * self

    def __rmul__(self, c):
        if isinstance(c, Number):
            return MV(self.n, {k: complex(c) * v for k, v in self.terms.items()})
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return MV(self.n, {k: v * complex(other) for k, v in self.terms.items()})
        other = self._coerce(other)
        out: dict[int, complex] = {}
        for a, ca in self.terms.items():
            for b, cb in other.terms.items():
                ph, m = word_mul(self.n, a, b)
                out[m] = out.get(m, 0.0) + ca * cb * ph
        return MV(self.n, out)

    def __matmul__(self, other):
        """Explicit synonym for the geometric/operator product."""
        return self * other

    def __truediv__(self, c):
        if not isinstance(c, Number):
            return NotImplemented
        z = complex(c)
        if abs(z) <= TOL:
            raise ZeroDivisionError("division by zero scalar")
        return (1.0 / z) * self

    def dagger(self) -> "MV":
        """Hermitian adjoint. Pauli words are Hermitian, so conjugate coefficients."""
        return MV(self.n, {k: v.conjugate() for k, v in self.terms.items()})

    def scalar_part(self) -> complex:
        return self.terms.get(0, 0.0)

    def trace(self) -> complex:
        return (2 ** self.n) * self.scalar_part()

    def norm_hs(self) -> float:
        """Hilbert-Schmidt norm, sqrt(Tr(A†A))."""
        return math.sqrt(2 ** self.n * sum(abs(v) ** 2 for v in self.terms.values()))

    def is_close(self, other, tol: float = 1e-9) -> bool:
        return (self - self._coerce(other)).norm_hs() < tol

    def is_zero(self, tol: float = 1e-12) -> bool:
        return self.norm_hs() < tol

    def is_hermitian(self, tol: float = 1e-9) -> bool:
        return self.is_close(self.dagger(), tol)

    def is_unitary(self, tol: float = 1e-9) -> bool:
        from .pauli import I
        return (self.dagger() * self).is_close(I(self.n), tol)

    def is_projector(self, tol: float = 1e-9) -> bool:
        return self.is_hermitian(tol) and (self * self).is_close(self, tol)

    def is_density(self, tol: float = 1e-9, check_psd: bool = True) -> bool:
        ok = self.is_hermitian(tol) and abs(self.trace() - 1) < tol
        if ok and check_psd:
            import numpy as np
            from .matrix import to_matrix
            ok = bool(np.min(np.linalg.eigvalsh(to_matrix(self))) >= -tol)
        return ok

    def nnz(self) -> int:
        return len(self.terms)

    def memory_estimate(self) -> int:
        """Rough byte estimate for sparse term storage, excluding Python dict overhead."""
        return self.nnz() * (8 + 16)

    def word_letters(self, code: int) -> str:
        return code_to_label(self.n, code)

    def support(self) -> list[str]:
        return [code_to_label(self.n, k) for k in sorted(self.terms)]

    def to_labels(self, *, tol: float = 1e-12) -> dict[str, complex]:
        return {code_to_label(self.n, k): v for k, v in sorted(self.terms.items()) if abs(v) > tol}

    def pretty(self, *, tol: float = 1e-12, precision: int = 4) -> str:
        if not self.terms:
            return "0"
        parts: list[str] = []
        for label, coeff in self.to_labels(tol=tol).items():
            parts.append(f"({coeff:.{precision}g})·{label}")
        return " + ".join(parts) if parts else "0"

    def _letters(self, code: int) -> list[int]:
        validate_word_code(self.n, code)
        return [(code >> (2 * j)) & 3 for j in range(self.n)]

    def blade_mask(self, code: int) -> int:
        """Decode one Pauli word to its Clifford generator subset via inverse JW."""
        letters = self._letters(code)
        mask = 0
        above_generator_count = 0
        for j in range(self.n - 1, -1, -1):
            letter = letters[j]
            if above_generator_count % 2 == 1:
                # Undo the Z string contributed by generators on higher qubits.
                letter = {0: 3, 3: 0, 1: 2, 2: 1}[letter]
            own = {0: 0, 1: 1, 2: 2, 3: 3}[letter]
            if own in (1, 3):
                mask |= 1 << (2 * j)
            if own in (2, 3):
                mask |= 1 << (2 * j + 1)
            above_generator_count += 1 if own in (1, 2) else (2 if own == 3 else 0)
        return mask

    def grade(self, g: int) -> "MV":
        return MV(self.n, {k: v for k, v in self.terms.items() if self.blade_mask(k).bit_count() == g})

    def grades(self) -> set[int]:
        return {self.blade_mask(k).bit_count() for k in self.terms}

    def __repr__(self) -> str:
        return self.pretty()
