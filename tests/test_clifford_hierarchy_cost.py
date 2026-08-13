"""Focused correctness checks for the hierarchy's R1 statistical ledger."""

import numpy as np
import pytest

pytest.importorskip("stim")

from benchmarks.run_clifford_hierarchy import _pauli_quadratic_variance
from clifford_qc.ir import PauliWord
from clifford_qc.states import ket_density, expectation


def test_vectorized_pauli_variance_matches_direct_operator_square():
    rho = ket_density(2, "00")
    codes = np.asarray([
        PauliWord.from_label("ZI").code,
        PauliWord.from_label("IZ").code,
        PauliWord.from_label("ZZ").code,
    ], dtype=np.uint64)
    coefficients = np.asarray([0.3, -0.8, 1.2])
    observable = sum(
        (coefficient * PauliWord(2, int(code)).to_mv()
         for code, coefficient in zip(codes, coefficients)),
        PauliWord(2, 0).to_mv(0.0),
    )
    mean = expectation(rho, observable).real
    direct = expectation(rho, observable * observable).real - mean * mean
    assert _pauli_quadratic_variance(2, rho, codes, coefficients) == pytest.approx(
        direct, abs=1e-14
    )


def test_variance_rejects_an_anticommuting_pseudo_setting():
    rho = ket_density(1, "0")
    codes = np.asarray([
        PauliWord.from_label("X").code,
        PauliWord.from_label("Z").code,
    ], dtype=np.uint64)
    with pytest.raises(AssertionError, match="anticommuting"):
        _pauli_quadratic_variance(1, rho, codes, np.asarray([1.0, 1.0]))
