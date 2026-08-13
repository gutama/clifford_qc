"""Focused correctness checks for the hierarchy's R1 statistical ledger."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("stim")

from benchmarks.run_clifford_hierarchy import (
    _pauli_quadratic_variance,
    load_device_cards,
)
from clifford_qc.ir import PauliWord
from clifford_qc.pauli_kernel import word_mul
from clifford_qc.states import ket_density, expectation


CARDS = Path(__file__).parents[1] / "benchmarks" / "configs" / "device_cards"


def test_explicit_device_card_order_is_canonicalized_by_card_name():
    cards = load_device_cards([
        CARDS / "superconducting-like.json",
        CARDS / "ion-like.json",
        CARDS / "logical-alltoall.json",
    ])
    assert [card.name for card in cards] == [
        "ion-like", "logical-alltoall", "superconducting-like"
    ]


def _variance_case(n, labels, coefficients, state):
    rho = ket_density(n, state)
    codes = np.asarray(
        [PauliWord.from_label(label).code for label in labels], dtype=np.uint64
    )
    coefficients = np.asarray(coefficients, dtype=float)
    observable = sum(
        (coefficient * PauliWord(n, int(code)).to_mv()
         for code, coefficient in zip(codes, coefficients)),
        PauliWord(n, 0).to_mv(0.0),
    )
    mean = expectation(rho, observable).real
    direct = expectation(rho, observable * observable).real - mean * mean
    assert _pauli_quadratic_variance(n, rho, codes, coefficients) == pytest.approx(
        direct, abs=1e-14
    )


def test_vectorized_pauli_variance_matches_direct_operator_square():
    _variance_case(2, ["ZI", "IZ", "ZZ"], [0.3, -0.8, 1.2], "00")


@pytest.mark.parametrize(
    "labels, state",
    [
        # Z-only words leave every phase term at zero, so the cases below carry
        # X and Y content: they are the ones that exercise the sign of the
        # packed-product exponent rather than the identity branch of it.
        (["XX", "YY", "ZZ"], "00"),
        (["XY", "YX", "ZZ"], "01"),
        (["XZ", "YY", "ZX"], "10"),
        (["XXI", "YYI", "ZZI", "IIX"], "000"),
        (["XXX", "YYX", "ZZI"], "011"),
    ],
)
def test_pauli_variance_covers_the_phase_algebra(labels, state):
    n = len(labels[0])
    coefficients = [0.3, -0.8, 1.2, 0.5][: len(labels)]
    _variance_case(n, labels, coefficients, state)


def test_vectorized_phase_agrees_with_the_pauli_kernel_product():
    """Pin the array-form exponent against the kernel that owns the algebra."""
    n = 3
    labels = ["XYZ", "YZX", "ZXY", "XXI", "IIZ", "YIY"]
    codes = [PauliWord.from_label(label).code for label in labels]
    lo = np.uint64(((1 << (2 * n)) - 1) // 3)
    array = np.asarray(codes, dtype=np.uint64)
    z = (array >> np.uint64(1)) & lo
    x = (array & lo) ^ z
    xz_count = _popcount(x & z)
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            product = array[i] ^ array[j]
            z_product = (product >> np.uint64(1)) & lo
            x_product = (product & lo) ^ z_product
            exponent = int(
                int(xz_count[i]) + int(xz_count[j])
                - int(_popcount(x_product & z_product))
                + 2 * int(_popcount(z[i] & x[j]))
            ) % 4
            phase, expected_code = word_mul(n, a, b)
            assert int(product) == expected_code
            assert (1j ** exponent) == pytest.approx(phase)


def _popcount(values):
    from benchmarks.run_clifford_hierarchy import _popcount_u64

    result = _popcount_u64(np.atleast_1d(np.asarray(values, dtype=np.uint64)))
    return result if np.ndim(values) else result[0]


def test_variance_rejects_an_anticommuting_pseudo_setting():
    rho = ket_density(1, "0")
    codes = np.asarray([
        PauliWord.from_label("X").code,
        PauliWord.from_label("Z").code,
    ], dtype=np.uint64)
    with pytest.raises(AssertionError, match="anticommuting"):
        _pauli_quadratic_variance(1, rho, codes, np.asarray([1.0, 1.0]))


def test_variance_uses_sparse_moments_but_rejects_codes_wider_than_uint64():
    rho = SimpleNamespace(terms={0: 1.0 / 2**20})
    assert _pauli_quadratic_variance(
        20, rho, np.asarray([0], dtype=np.uint64), np.asarray([1.0])
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match=r"\[0, 32\]"):
        _pauli_quadratic_variance(
            33, rho, np.asarray([], dtype=np.uint64), np.asarray([])
        )
