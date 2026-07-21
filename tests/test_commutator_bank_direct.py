"""Cross-validation of the direct commutator-bank construction.

``CommutatorBank`` builds each selection observable ``G_j = -i/2 [H, P_j]``
directly from the anticommuting terms of ``H`` (one Pauli-word product per
surviving term). This module pins that fast path against the reference
construction ``-i/2 (H P_j - P_j H)`` formed from two full operator products,
row by row, and independently validates the packed anticommutation test.

The reference path (`clifford_qc.pauli.comm` over `MV`) and the fast path
(`word_mul` + `_pauli_anticommute`) share no code, so agreement is a genuine
two-implementation check — not a tautology.
"""

import random

import pytest

from clifford_qc.algorithms import local_pool
from clifford_qc.ir import PauliSum, PauliWord
from clifford_qc.measurement import CommutatorBank
from clifford_qc.measurement.bank import _pauli_anticommute
from clifford_qc.models import random_ising, tfim
from clifford_qc.pauli import comm
from clifford_qc.multivector import code_to_label, label_to_code


def _reference_row(H_mv, word: PauliWord, tol=1e-12):
    """G_j = -i/2 [H, P_j] via two full operator products (reference path)."""
    G = -0.5j * comm(H_mv, word.to_mv())
    return {code: c.real for code, c in G.terms.items() if abs(c) > tol}


def _assert_bank_matches_reference(hamiltonian: PauliSum, words):
    bank = CommutatorBank(hamiltonian, words)
    H_mv = hamiltonian.to_mv()
    for j, w in enumerate(words):
        ref = _reference_row(H_mv, w)
        got = bank.coeffs[j]
        assert set(got) == set(ref), (
            f"word set differs for {w.label}: "
            f"only-fast={set(got) - set(ref)}, only-ref={set(ref) - set(got)}")
        for code in ref:
            assert got[code] == pytest.approx(ref[code], abs=1e-12), (
                f"coefficient differs on {code_to_label(hamiltonian.n, code)} "
                f"for {w.label}")


# ---------------------------------------------------------------------------
# Row-by-row equality across models


@pytest.mark.parametrize("n,J,h", [(2, 1.0, 0.5), (3, 1.3, 0.7),
                                   (4, 1.0, 1.0), (4, 0.4, 1.9)])
def test_direct_rows_match_reference_tfim(n, J, h):
    m = tfim(n, J=J, h=h)
    words = [op.word for op in local_pool(n, periodic_context=False)]
    _assert_bank_matches_reference(m.hamiltonian, words)


@pytest.mark.parametrize("seed", range(4))
def test_direct_rows_match_reference_random_ising(seed):
    m = random_ising(4, seed=seed)
    words = [op.word for op in local_pool(4, periodic_context=False)]
    _assert_bank_matches_reference(m.hamiltonian, words)


def test_direct_rows_match_reference_with_y_terms():
    """TFIM has only X and ZZ; a Y-bearing Hamiltonian and Y-bearing pool
    exercise the imaginary XY/YZ/ZX phases in the Pauli product that drive the
    sign of each commutator coefficient."""
    H = PauliSum.from_labels({
        "YI": 0.5, "IY": -0.3, "YZ": 0.7, "ZY": 0.7, "XY": 0.25,
        "YX": -0.25, "XX": 0.9, "ZZ": -1.1, "YY": 0.4,
    }, n=2)
    words = [PauliWord.from_label(l) for l in
             ("XI", "IX", "YI", "IY", "ZI", "IZ", "XY", "YX", "YZ", "ZY", "YY")]
    _assert_bank_matches_reference(H, words)


def test_direct_rows_match_reference_random_hamiltonians():
    rng = random.Random(20260721)
    for _ in range(30):
        n = rng.randint(1, 4)
        k = rng.randint(1, 8)
        codes = rng.sample(range(4 ** n), min(k, 4 ** n))
        H = PauliSum(n, {c: rng.uniform(-2.0, 2.0) for c in codes})
        wcodes = rng.sample(range(4 ** n), min(6, 4 ** n))
        words = [PauliWord(n, c) for c in wcodes]
        _assert_bank_matches_reference(H, words)


# ---------------------------------------------------------------------------
# The packed anticommutation predicate


def _anticommute_via_letters(n, a, b):
    """Independent oracle: parity of qubits where both letters are non-I and
    distinct."""
    la, lb = code_to_label(n, a), code_to_label(n, b)
    return sum(1 for x, y in zip(la, lb)
               if x != "I" and y != "I" and x != y) % 2


def test_pauli_anticommute_matches_letter_oracle():
    rng = random.Random(11)
    for _ in range(2000):
        n = rng.randint(1, 5)
        a = rng.randrange(4 ** n)
        b = rng.randrange(4 ** n)
        assert _pauli_anticommute(n, a, b) == _anticommute_via_letters(n, a, b)


def test_pauli_anticommute_matches_operator_bracket():
    """Anticommute iff the operator commutator [A,B] = AB - BA is nonzero
    (for Pauli words the bracket is either 0 or 2AB)."""
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 4)
        a, b = rng.randrange(4 ** n), rng.randrange(4 ** n)
        A, B = PauliWord(n, a).to_mv(), PauliWord(n, b).to_mv()
        bracket_zero = (A * B - B * A).is_zero()
        assert bool(_pauli_anticommute(n, a, b)) == (not bracket_zero)


def test_identity_commutes_with_everything():
    for n in range(1, 4):
        for b in range(4 ** n):
            assert _pauli_anticommute(n, 0, b) == 0
            assert _pauli_anticommute(n, b, 0) == 0


# ---------------------------------------------------------------------------
# Optional: agreement with OpenFermion's commutator, if installed


def test_direct_rows_match_openfermion():
    of = pytest.importorskip("openfermion")
    from clifford_qc.bridges.openfermion_bridge import pauli_sum_to_qubit_operator

    H = PauliSum.from_labels({"XX": 0.5, "YY": -0.4, "ZZ": 1.2,
                              "XZ": 0.3, "YI": 0.2}, n=2)
    words = [PauliWord.from_label(l) for l in ("XI", "IY", "ZZ", "XY")]
    bank = CommutatorBank(H, words)
    qop_H = pauli_sum_to_qubit_operator(H)
    for j, w in enumerate(words):
        qop_P = pauli_sum_to_qubit_operator(PauliSum(2, {w.code: 1.0}))
        g = -0.5j * of.commutator(qop_H, qop_P)  # OpenFermion QubitOperator
        ref = {}
        for term, coeff in g.terms.items():
            code = 0
            for q, p in term:
                code |= {"X": 1, "Y": 2, "Z": 3}[p] << (2 * q)
            if abs(coeff) > 1e-12:
                ref[code] = coeff.real
        assert set(bank.coeffs[j]) == set(ref)
        for code in ref:
            assert bank.coeffs[j][code] == pytest.approx(ref[code], abs=1e-12)
