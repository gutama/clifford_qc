"""Cross-validation of the packed binary-symplectic ``word_mul``.

``word_mul`` multiplies two encoded Pauli words with packed x/z bit-plane
operations and popcounts. This module pins it against ``_word_mul_ref`` (the
per-qubit table-lookup reference), against the known single-qubit product
table, and against dense-operator multiplication via ``MV``.
"""

import random

import pytest

from clifford_qc.multivector import (
    MV, _PTAB, _word_mul_ref, code_to_label, label_to_code, word_mul,
)
from clifford_qc.pauli_kernel import _word_mul_unchecked


# ---------------------------------------------------------------------------
# Packed vs. reference


@pytest.mark.parametrize("n", [1, 2, 3])
def test_packed_matches_reference_exhaustive(n):
    """Every ordered pair of words agrees with the table-lookup reference."""
    for a in range(4 ** n):
        for b in range(4 ** n):
            expected = _word_mul_ref(n, a, b)
            assert word_mul(n, a, b) == expected, (
                f"{code_to_label(n, a)} * {code_to_label(n, b)}")
            # MV multiplication uses this hot path only after construction has
            # validated its packed codes, so it must remain the same algebra.
            assert _word_mul_unchecked(n, a, b) == expected


@pytest.mark.parametrize("n", [4, 5, 6, 8])
def test_packed_matches_reference_random(n):
    rng = random.Random(1000 + n)
    for _ in range(4000):
        a = rng.randrange(4 ** n)
        b = rng.randrange(4 ** n)
        assert word_mul(n, a, b) == _word_mul_ref(n, a, b)


def test_packed_matches_reference_wide_register():
    """Beyond a 64-bit register the lane mask must still be correct."""
    n = 40
    rng = random.Random(40)
    for _ in range(500):
        a = rng.randrange(4 ** n)
        b = rng.randrange(4 ** n)
        assert word_mul(n, a, b) == _word_mul_ref(n, a, b)


# ---------------------------------------------------------------------------
# Single-qubit ground truth


def test_single_qubit_table():
    for (la, lb), (phase, lc) in _PTAB.items():
        assert word_mul(1, la, lb) == (phase, lc)


def test_known_products():
    def m(x, y):
        ph, out = word_mul(1, label_to_code(x, 1), label_to_code(y, 1))
        return ph, code_to_label(1, out)

    assert m("X", "Y") == (1j, "Z")
    assert m("Y", "X") == (-1j, "Z")
    assert m("Y", "Z") == (1j, "X")
    assert m("Z", "Y") == (-1j, "X")
    assert m("Z", "X") == (1j, "Y")
    assert m("X", "Z") == (-1j, "Y")
    assert m("X", "X") == (1 + 0j, "I")
    assert m("I", "Y") == (1 + 0j, "Y")


# ---------------------------------------------------------------------------
# Structural invariants


@pytest.mark.parametrize("n", [1, 2, 3])
def test_result_word_is_lanewise_xor(n):
    for a in range(4 ** n):
        for b in range(4 ** n):
            _, out = word_mul(n, a, b)
            assert out == a ^ b


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_phase_is_fourth_root_of_unity(n):
    rng = random.Random(n)
    for _ in range(500):
        a, b = rng.randrange(4 ** n), rng.randrange(4 ** n)
        ph, _ = word_mul(n, a, b)
        assert ph in (1 + 0j, 1j, -1 + 0j, -1j)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_agrees_with_dense_mv_product(n):
    """The (phase, word) product reproduces the dense operator product."""
    rng = random.Random(50 + n)
    for _ in range(60):
        a, b = rng.randrange(4 ** n), rng.randrange(4 ** n)
        ph, out = word_mul(n, a, b)
        lhs = MV(n, {a: 1.0}) * MV(n, {b: 1.0})
        rhs = MV(n, {out: ph})
        assert lhs.is_close(rhs, tol=1e-9)


def test_conjugate_symmetry_of_phase():
    """(A B)(B A) share the result word; their phases are complex conjugates
    (words either commute -> equal phases, or anticommute -> opposite)."""
    n = 4
    rng = random.Random(3)
    for _ in range(1000):
        a, b = rng.randrange(4 ** n), rng.randrange(4 ** n)
        ph_ab, out_ab = word_mul(n, a, b)
        ph_ba, out_ba = word_mul(n, b, a)
        assert out_ab == out_ba
        assert ph_ab == pytest.approx(ph_ba.conjugate())
