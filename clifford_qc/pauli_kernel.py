"""Packed Pauli-word primitives shared by every execution layer.

The kernel owns only the representation contract: two bits per qubit with
qubit 0 in the leftmost label position. Higher layers may attach Clifford,
measurement, or state semantics, but they should not duplicate these bit
operations.
"""

from __future__ import annotations

from functools import lru_cache


PAULI_LETTERS = {0: "I", 1: "X", 2: "Y", 3: "Z"}
LETTER_CODE = {v: k for k, v in PAULI_LETTERS.items()}

_PTAB: dict[tuple[int, int], tuple[complex, int]] = {}
for _a in range(4):
    _PTAB[(0, _a)] = (1 + 0j, _a)
    _PTAB[(_a, 0)] = (1 + 0j, _a)
    _PTAB[(_a, _a)] = (1 + 0j, 0)
_PTAB[(1, 2)] = (1j, 3);   _PTAB[(2, 1)] = (-1j, 3)
_PTAB[(2, 3)] = (1j, 1);   _PTAB[(3, 2)] = (-1j, 1)
_PTAB[(3, 1)] = (1j, 2);   _PTAB[(1, 3)] = (-1j, 2)


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
    """Encode a Pauli label such as XIZ."""
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


def word_mul_reference(n: int, a: int, b: int) -> tuple[complex, int]:
    """Reference O(n) Pauli-word product used as an independent oracle."""
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


_PHASE4 = (1 + 0j, 1j, -1 + 0j, -1j)


@lru_cache(maxsize=1024)
def pauli_lane_mask(n: int) -> int:
    """Bit 0 of every two-bit lane over n lanes."""
    validate_n(n)
    return ((1 << (2 * n)) - 1) // 3 if n else 0


@lru_cache(maxsize=1_000_000)
def _word_mul_unchecked(n: int, a: int, b: int) -> tuple[complex, int]:
    """Hot-path packed product for codes already validated by their owner."""
    lo = pauli_lane_mask(n)
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


def word_mul(n: int, a: int, b: int) -> tuple[complex, int]:
    """Multiply two packed Pauli words in the same n-qubit algebra."""
    validate_word_code(n, a)
    validate_word_code(n, b)
    # Keep one shared cache for both public and internal multiplication. The
    # public boundary pays validation on every call; MV hot paths already own
    # that invariant and call the cached unchecked form directly.
    return _word_mul_unchecked(n, a, b)


_lane_mask = pauli_lane_mask
_word_mul_ref = word_mul_reference
