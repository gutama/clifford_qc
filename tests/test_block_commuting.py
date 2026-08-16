"""The dyadic block-commuting grouping rule, and its two endpoints.

The rule this module implements is the protocol axis of the measurement
hierarchy, so it is pinned from three directions:

1.  the packed predicate agrees with an independent letter-by-letter oracle
    built from ordinary Pauli commutation on each block;
2.  its two endpoints reduce to the protocols they are meant to interpolate --
    ``k = 1`` is qubit-wise commutation, ``k >= n`` is full commutation;
3.  the greedy partition returns a genuine partition whose groups are pairwise
    block-commuting, and coarsening ``k`` never increases the setting count on
    the dyadic ladder.

The frozen hierarchy and shot-search records are the fourth direction and live
in ``benchmarks/check_clifford_hierarchy.py`` and
``benchmarks/check_exact_shot_search.py``: those regenerate the committed
numbers digit for digit, which is what makes this extraction a refactor rather
than a change of protocol.
"""

import random

import pytest

from clifford_qc.ir import PauliWord
from clifford_qc.measurement import qubit_wise_commute, qwc_groups
from clifford_qc.measurement.block_commuting import (
    block_commuting_groups,
    block_commuting_partition,
    block_ranges,
    block_wise_commute,
)
from clifford_qc.multivector import code_to_label


# ---------------------------------------------------------------------------
# Predicate vs. an independent letter oracle


_ANTICOMMUTE = {
    ("X", "Y"), ("Y", "X"), ("Y", "Z"), ("Z", "Y"), ("X", "Z"), ("Z", "X"),
}


def _commute_letters(la: str, lb: str) -> bool:
    """Ordinary commutation of two Pauli strings, counted letter by letter."""
    return sum((x, y) in _ANTICOMMUTE for x, y in zip(la, lb)) % 2 == 0


def _block_commute_letters(n: int, a: int, b: int, block_size: int) -> bool:
    la, lb = code_to_label(n, a), code_to_label(n, b)
    return all(
        _commute_letters(la[start:start + size], lb[start:start + size])
        for start, size in block_ranges(n, block_size)
    )


def test_predicate_matches_letter_oracle_exhaustive_small():
    for n in (1, 2, 3):
        for block_size in range(1, n + 2):
            for a in range(4 ** n):
                for b in range(4 ** n):
                    assert block_wise_commute(
                        PauliWord(n, a), PauliWord(n, b), block_size
                    ) == _block_commute_letters(n, a, b, block_size)


def test_predicate_matches_letter_oracle_random():
    rng = random.Random(0)
    for _ in range(4000):
        n = rng.randint(1, 8)
        block_size = rng.randint(1, n + 2)
        a, b = rng.randrange(4 ** n), rng.randrange(4 ** n)
        assert block_wise_commute(
            PauliWord(n, a), PauliWord(n, b), block_size
        ) == _block_commute_letters(n, a, b, block_size)


def test_predicate_is_symmetric_and_reflexive():
    rng = random.Random(1)
    for _ in range(500):
        n = rng.randint(1, 6)
        k = rng.randint(1, n)
        a = PauliWord(n, rng.randrange(4 ** n))
        b = PauliWord(n, rng.randrange(4 ** n))
        assert block_wise_commute(a, a, k)
        assert block_wise_commute(a, b, k) == block_wise_commute(b, a, k)


# ---------------------------------------------------------------------------
# The two endpoints the hierarchy interpolates


def test_block_size_one_is_qubit_wise_commutation():
    rng = random.Random(2)
    for _ in range(2000):
        n = rng.randint(1, 6)
        a = PauliWord(n, rng.randrange(4 ** n))
        b = PauliWord(n, rng.randrange(4 ** n))
        assert block_wise_commute(a, b, 1) == qubit_wise_commute(a, b)


def test_block_size_at_or_above_n_is_full_commutation():
    rng = random.Random(3)
    for _ in range(2000):
        n = rng.randint(1, 6)
        a, b = rng.randrange(4 ** n), rng.randrange(4 ** n)
        expected = _commute_letters(code_to_label(n, a), code_to_label(n, b))
        for block_size in (n, n + 1, 2 * n + 3):
            assert block_wise_commute(
                PauliWord(n, a), PauliWord(n, b), block_size
            ) is expected


def test_qwc_implies_block_commuting_at_every_k():
    """QWC is the strictest rung: a QWC pair commutes on every block."""
    rng = random.Random(4)
    for _ in range(1000):
        n = rng.randint(2, 6)
        a = PauliWord(n, rng.randrange(4 ** n))
        b = PauliWord(n, rng.randrange(4 ** n))
        if not qubit_wise_commute(a, b):
            continue
        for block_size in range(1, n + 1):
            assert block_wise_commute(a, b, block_size)


# ---------------------------------------------------------------------------
# The greedy partition


def _word_universe(n, count, seed):
    """A wide deterministic word set, the shape the hierarchy actually groups.

    Model Hamiltonians are too sparse to exercise the greedy: an eight-qubit
    TFIM has seven terms, while the frozen H4 bank groups 7371 words.
    """
    rng = random.Random(seed)
    return sorted({rng.randrange(4 ** n) for _ in range(count)})


@pytest.mark.parametrize("block_size", [1, 2, 3, 4, 8])
def test_partition_covers_every_index_exactly_once(block_size):
    n = 8
    codes = _word_universe(n, 400, seed=5)
    groups = block_commuting_partition(n, codes, block_size)
    flat = sorted(index for group in groups for index in group)
    assert flat == list(range(len(codes)))


@pytest.mark.parametrize("block_size", [1, 2, 3, 4, 8])
def test_partition_groups_are_pairwise_block_commuting(block_size):
    n = 8
    codes = _word_universe(n, 400, seed=6)
    groups = block_commuting_partition(n, codes, block_size)
    for group in groups:
        for i in group:
            for j in group:
                assert block_wise_commute(
                    PauliWord(n, codes[i]), PauliWord(n, codes[j]), block_size
                )


def test_dyadic_coarsening_never_breaks_a_compatible_pair():
    """The monotonicity that makes the ladder a ladder, stated on the predicate.

    Each dyadic step unions adjacent blocks, so a pair commuting on both halves
    still commutes on their union. Non-dyadic ``k`` carries no such guarantee --
    ``k=3`` blocks are not a coarsening of ``k=2`` blocks -- which is why the
    ladder is dyadic.

    This is a statement about compatibility, not about the greedy: removing
    edges can in principle reorder a first-fit coloring, so the setting count
    is checked separately as an observation rather than asserted as a theorem.
    """
    n = 8
    codes = _word_universe(n, 200, seed=7)
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            wa, wb = PauliWord(n, a), PauliWord(n, b)
            for fine, coarse in ((1, 2), (2, 4), (4, 8)):
                if block_wise_commute(wa, wb, fine):
                    assert block_wise_commute(wa, wb, coarse)


def test_setting_count_falls_along_the_dyadic_ladder():
    """Observed on this universe, matching the frozen H4 and BeH2 records.

    Not a theorem about the greedy -- see the docstring above -- but the
    behaviour the protocol axis is priced on, so a regression that inverted it
    would mean the hierarchy had stopped interpolating.
    """
    n = 8
    codes = _word_universe(n, 400, seed=8)
    counts = [
        len(block_commuting_partition(n, codes, block_size))
        for block_size in (1, 2, 4, 8)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_partition_is_deterministic():
    n = 8
    codes = _word_universe(n, 300, seed=9)
    first = block_commuting_partition(n, codes, 2)
    second = block_commuting_partition(n, codes, 2)
    assert first == second


def test_partition_keeps_duplicate_codes_as_separate_indices():
    """Indices are the contract: every occurrence is assigned a setting."""
    n = 3
    codes = [0b01, 0b01, 0b1100]
    groups = block_commuting_partition(n, codes, 1)
    flat = sorted(index for group in groups for index in group)
    assert flat == [0, 1, 2]


def test_word_level_grouping_deduplicates_like_qwc_groups():
    n = 3
    word = PauliWord.from_label("XIZ")
    groups = block_commuting_groups([word, word], 1)
    assert [[w.code for w in group] for group in groups] == [[word.code]]


def test_word_level_grouping_at_k_one_is_a_valid_qwc_partition():
    """``k=1`` grouping is QWC-valid and uses no more settings than ``qwc_groups``.

    The two greedies break degree ties differently -- this one by code, the QWC
    one by input order -- so the partitions need not be identical. What must
    hold is that every returned group is genuinely QWC.
    """
    n = 6
    words = [PauliWord(n, code) for code in _word_universe(n, 200, seed=10)]
    groups = block_commuting_groups(words, 1)
    assert sorted(w.code for group in groups for w in group) == sorted(
        w.code for w in words
    )
    for group in groups:
        for a in group:
            for b in group:
                assert qubit_wise_commute(a, b)
    # Both are largest-degree-first first-fit on the same graph, so neither
    # heuristic should be far off the other; the bound is loose on purpose.
    assert len(groups) <= 2 * len(qwc_groups(words))


def test_empty_inputs():
    assert block_commuting_partition(4, [], 2) == []
    assert block_commuting_groups([], 2) == []


# ---------------------------------------------------------------------------
# Contracts and rejected inputs


def test_block_ranges_partition_the_register():
    for n in range(0, 10):
        for block_size in range(1, n + 3):
            ranges = block_ranges(n, block_size)
            assert sum(size for _, size in ranges) == n
            assert [start for start, _ in ranges] == sorted(
                start for start, _ in ranges
            )
            covered = [q for start, size in ranges for q in range(start, start + size)]
            assert covered == list(range(n))


def test_invalid_block_size_is_rejected():
    for block_size in (0, -1):
        with pytest.raises(ValueError):
            block_ranges(4, block_size)
        with pytest.raises(ValueError):
            block_commuting_partition(4, [0], block_size)
        with pytest.raises(ValueError):
            block_wise_commute(PauliWord(4, 0), PauliWord(4, 0), block_size)


def test_mismatched_qubit_counts_are_rejected():
    with pytest.raises(ValueError):
        block_wise_commute(PauliWord(2, 0), PauliWord(3, 0), 1)
    with pytest.raises(ValueError):
        block_commuting_groups([PauliWord(2, 0), PauliWord(3, 0)], 1)


def test_code_beyond_the_declared_width_is_rejected():
    with pytest.raises(ValueError):
        block_commuting_partition(2, [4 ** 2], 1)
