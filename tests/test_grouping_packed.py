"""Cross-validation of the packed QWC grouping.

The packed compatibility test and bitset greedy must (a) agree with an
independent letter-by-letter oracle, and (b) produce the *same* partition as
a plain letter-based greedy, so the committed measurement-circuit counts are
unchanged. The memoized partition must also equal a freshly computed one.
"""

import random

import pytest

from clifford_qc.algorithms import local_pool
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import CommutatorBank
from clifford_qc.measurement.grouping import (
    _qwc_codes, qubit_wise_commute, qwc_groups, shared_basis,
)
from clifford_qc.models import random_ising, tfim
from clifford_qc.multivector import code_to_label


# ---------------------------------------------------------------------------
# Packed compatibility test vs. letter oracle


def _qwc_letters(n, a, b):
    la, lb = code_to_label(n, a), code_to_label(n, b)
    return all(x == "I" or y == "I" or x == y for x, y in zip(la, lb))


def test_qwc_codes_matches_letter_oracle():
    rng = random.Random(0)
    for _ in range(4000):
        n = rng.randint(1, 5)
        a, b = rng.randrange(4 ** n), rng.randrange(4 ** n)
        assert _qwc_codes(n, a, b) == _qwc_letters(n, a, b)


def test_qwc_codes_matches_letters_exhaustive_small():
    for n in (1, 2, 3):
        for a in range(4 ** n):
            for b in range(4 ** n):
                assert _qwc_codes(n, a, b) == _qwc_letters(n, a, b)


def test_qubit_wise_commute_wrapper_and_errors():
    assert qubit_wise_commute(PauliWord.from_label("XIZ"), PauliWord.from_label("XYZ"))
    assert not qubit_wise_commute(PauliWord.from_label("XX"), PauliWord.from_label("YY"))
    with pytest.raises(ValueError):
        qubit_wise_commute(PauliWord(2, 0), PauliWord(3, 0))


# ---------------------------------------------------------------------------
# Partition equals a plain letter-based greedy (output preservation)


def _reference_partition(words):
    """The pre-optimization greedy, verbatim, over PauliWord letters."""
    words = list(words)
    if not words:
        return []
    conflicts = [sum(1 for other in words if other is not w
                     and not _qwc_letters(w.n, w.code, other.code)) for w in words]
    order = sorted(range(len(words)), key=lambda i: -conflicts[i])
    groups = []
    for i in order:
        w = words[i]
        for group in groups:
            if all(_qwc_letters(w.n, w.code, m.code) for m in group):
                group.append(w)
                break
        else:
            groups.append([w])
    return [[w.code for w in g] for g in groups]


def _codes(groups):
    return [[w.code for w in g] for g in groups]


def _assert_matches_reference(words):
    assert _codes(qwc_groups(words)) == _reference_partition(words)


def test_partition_matches_reference_tfim():
    for n in (2, 3, 4):
        bank = CommutatorBank(tfim(n, 1.0, 1.0).hamiltonian,
                              [op.word for op in local_pool(n, periodic_context=False)])
        _assert_matches_reference(bank.words)


def test_partition_matches_reference_random_ising():
    for seed in range(5):
        bank = CommutatorBank(random_ising(4, seed=seed).hamiltonian,
                              [op.word for op in local_pool(4, periodic_context=False)])
        _assert_matches_reference(bank.words)


def test_partition_matches_reference_random_wordsets():
    rng = random.Random(99)
    for _ in range(40):
        n = rng.randint(1, 4)
        k = rng.randint(1, min(12, 4 ** n))
        codes = rng.sample(range(4 ** n), k)
        words = [PauliWord(n, c) for c in sorted(codes)]
        _assert_matches_reference(words)


# ---------------------------------------------------------------------------
# Partition validity and cache reuse


def test_groups_are_valid_partition():
    bank = CommutatorBank(tfim(4, 1.0, 0.8).hamiltonian,
                          [op.word for op in local_pool(4, periodic_context=False)])
    groups = qwc_groups(bank.words)
    seen = []
    for group in groups:
        for a in group:
            for b in group:
                assert qubit_wise_commute(a, b)
        shared_basis(group)  # a consistent basis exists
        seen.extend(w.code for w in group)
    assert sorted(seen) == sorted(w.code for w in bank.words)  # exact partition


def test_cache_reuse_returns_equal_partition():
    words = [PauliWord.from_label(l) for l in
             ("XIX", "ZZI", "IYY", "XXI", "IZZ", "YIY")]
    first = _codes(qwc_groups(words))
    second = _codes(qwc_groups(list(words)))  # fresh list, same codes/order
    assert first == second
    # fresh PauliWord objects are returned each call, not cached instances
    ga, gb = qwc_groups(words), qwc_groups(words)
    assert ga[0][0] is not gb[0][0]


def test_subset_after_elimination_regroups_validly():
    bank = CommutatorBank(tfim(4, 1.0, 1.0).hamiltonian,
                          [op.word for op in local_pool(4, periodic_context=False)])
    subset = bank.words_for([0, 2, 3])
    _assert_matches_reference(subset)
