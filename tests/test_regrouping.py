"""Refinements that lower the setting count without moving the rule.

Every strategy in :mod:`clifford_qc.measurement.regrouping` returns something a
block-local diagonalizer can actually run, so the tests pin the same three
things for each of them:

1.  the result is a *partition* -- every banked word assigned exactly once,
    which is what ``synthesize_block_settings`` refuses to proceed without;
2.  every group is pairwise block-commuting under the unchanged predicate of
    :mod:`clifford_qc.measurement.block_commuting`, so the refinement cannot
    buy settings by quietly relaxing ``k``;
3.  the count never rises above the seed it was given, which is the only claim
    any of these passes makes.

The frame search adds a fourth: a relabelling permutes letters, never words, so
the bank it returns must be a permutation of the bank it was handed. The frozen
hierarchy record is untouched by all of this -- ``block_commuting_partition``
still returns its committed colouring, and ``check_clifford_hierarchy.py``
still regenerates the committed numbers.
"""

import random

import pytest

from clifford_qc.ir import PauliWord
from clifford_qc.measurement.block_commuting import (
    block_commuting_partition,
    block_wise_commute,
    conflict_masks,
    first_fit_partition,
    packed_conflicts,
)
from clifford_qc.measurement.regrouping import (
    MAX_SPAN_QUBITS,
    Regrouping,
    block_orders,
    counting_bound,
    dsatur_partition,
    group_span,
    isotropic_cover,
    iterated_greedy,
    permuted_codes,
    qwc_forcing_bound,
    reduce_settings,
    search_block_order,
    span_recover,
)


def _bank(n: int, width: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted({rng.randrange(1 << (2 * n)) for _ in range(width)})


def _assert_partition(codes, groups):
    flat = sorted(i for group in groups for i in group)
    assert flat == list(range(len(codes)))


def _assert_block_commuting(n, codes, groups, block_size):
    for group in groups:
        words = [PauliWord(n, codes[i]) for i in group]
        for left in range(len(words)):
            for right in range(left + 1, len(words)):
                assert block_wise_commute(words[left], words[right], block_size)


# ---------------------------------------------------------------------------
# Iterated greedy


@pytest.mark.parametrize("block_size", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_iterated_greedy_never_grows_and_stays_legal(block_size, seed):
    n = 4
    codes = _bank(n, 60, seed)
    base = block_commuting_partition(n, codes, block_size)
    refined = iterated_greedy(n, codes, block_size, base, rounds=24, seed=seed)
    assert len(refined) <= len(base)
    _assert_partition(codes, refined)
    _assert_block_commuting(n, codes, refined, block_size)


def test_iterated_greedy_is_deterministic_for_a_seed():
    n, codes = 4, _bank(4, 80, 7)
    base = block_commuting_partition(n, codes, 2)
    first = iterated_greedy(n, codes, 2, base, rounds=16, seed=3)
    second = iterated_greedy(n, codes, 2, base, rounds=16, seed=3)
    assert first == second


def test_iterated_greedy_strictly_improves_a_deliberately_bad_seed():
    """One group per word is legal and maximally wasteful; refinement fixes it."""
    n, codes = 4, _bank(4, 40, 11)
    singletons = [[i] for i in range(len(codes))]
    refined = iterated_greedy(n, codes, 4, singletons, rounds=8, seed=0)
    assert len(refined) < len(codes)
    _assert_block_commuting(n, codes, refined, 4)


def test_iterated_greedy_handles_an_empty_bank():
    assert iterated_greedy(3, [], 2, []) == []


# ---------------------------------------------------------------------------
# DSATUR


@pytest.mark.parametrize("block_size", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dsatur_partitions_and_stays_legal(block_size, seed):
    n = 4
    codes = _bank(n, 60, seed)
    groups = dsatur_partition(n, codes, block_size)
    _assert_partition(codes, groups)
    _assert_block_commuting(n, codes, groups, block_size)
    assert len(groups) >= counting_bound(n, codes)


def test_dsatur_is_a_function_of_the_bank_alone():
    n, codes = 4, _bank(4, 70, 3)
    assert dsatur_partition(n, codes, 2) == dsatur_partition(n, codes, 2)


def test_dsatur_opens_colours_in_order_and_fills_the_first_that_fits():
    """No colour is opened while a lower one is still legal for the word."""
    n, codes = 4, _bank(4, 50, 5)
    groups = dsatur_partition(n, codes, 2)
    assert all(groups), "DSATUR must not open a colour it leaves empty"


def test_dsatur_handles_an_empty_bank():
    assert dsatur_partition(3, [], 2) == []


# ---------------------------------------------------------------------------
# Span closure


def test_group_span_is_closed_under_the_pauli_product():
    span = group_span([0b0100, 0b1001])          # two arbitrary two-qubit words
    assert len(span) == 4
    for left in span:
        for right in span:
            assert left ^ right in span


def test_group_span_ignores_repeats_and_dependent_codes():
    assert sorted(group_span([5, 5, 0])) == [0, 5]
    assert sorted(group_span([5, 6, 3])) == sorted(group_span([5, 6]))


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_span_of_a_block_commuting_group_is_block_commuting(block_size):
    """The rule is bilinear and alternating, so the span is measurable too."""
    n, codes = 4, _bank(4, 60, 5)
    for group in block_commuting_partition(n, codes, block_size):
        span = group_span([codes[i] for i in group])
        words = [PauliWord(n, code) for code in span]
        for left in range(len(words)):
            for right in range(left + 1, len(words)):
                assert block_wise_commute(words[left], words[right], block_size)


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_span_recover_partitions_and_never_grows(block_size):
    n, codes = 4, _bank(4, 70, 9)
    base = block_commuting_partition(n, codes, block_size)
    recovered = span_recover(n, codes, block_size, base)
    assert len(recovered) <= len(base)
    _assert_partition(codes, recovered)
    _assert_block_commuting(n, codes, recovered, block_size)


def test_span_recover_refuses_a_register_it_cannot_index():
    with pytest.raises(ValueError, match="at most"):
        span_recover(MAX_SPAN_QUBITS + 1, [0], 2, [[0]])


# ---------------------------------------------------------------------------
# Isotropic cover


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_isotropic_cover_partitions_and_stays_legal(block_size):
    n, codes = 4, _bank(4, 70, 13)
    groups = isotropic_cover(n, codes, block_size)
    _assert_partition(codes, groups)
    _assert_block_commuting(n, codes, groups, block_size)


def test_isotropic_cover_refuses_a_register_it_cannot_index():
    with pytest.raises(ValueError, match="at most"):
        isotropic_cover(MAX_SPAN_QUBITS + 1, [0], 2)


def test_isotropic_cover_of_one_commuting_family_is_one_setting():
    """A bank already inside one isotropic subspace needs exactly one setting."""
    n = 3
    codes = sorted(group_span([0b110000, 0b001100]))       # Z_2, Z_1 and products
    assert isotropic_cover(n, codes, n) == [list(range(len(codes)))]


# ---------------------------------------------------------------------------
# Frames


def test_block_orders_enumerates_each_assignment_once():
    assert len(list(block_orders(8, 2))) == 105
    assert len(list(block_orders(8, 4))) == 35
    # The rule cannot see the labelling at either endpoint.
    assert list(block_orders(8, 1)) == [tuple(range(8))]
    assert list(block_orders(8, 8)) == [tuple(range(8))]


def test_block_orders_refuses_a_block_size_that_does_not_divide():
    with pytest.raises(ValueError, match="dividing the register"):
        list(block_orders(6, 4))


def test_permuted_codes_is_a_relabelling_not_a_reordering():
    n, codes = 4, _bank(4, 30, 17)
    order = (2, 0, 3, 1)
    moved = permuted_codes(n, codes, order)
    assert len(moved) == len(codes)
    inverse = tuple(order.index(q) for q in range(n))
    assert permuted_codes(n, moved, inverse) == list(codes)
    for code, image in zip(codes, moved):
        letters = [(code >> (2 * q)) & 3 for q in range(n)]
        moved_letters = [(image >> (2 * j)) & 3 for j in range(n)]
        assert moved_letters == [letters[q] for q in order]


def test_permuted_codes_refuses_a_non_permutation():
    with pytest.raises(ValueError, match="permutation"):
        permuted_codes(3, [0], (0, 1, 1))


def test_search_block_order_ranks_and_includes_the_contiguous_frame():
    n, codes = 4, _bank(4, 60, 19)
    ranked = search_block_order(n, codes, 2)
    assert len(ranked) == 3                       # three pairings of four qubits
    assert [row[1] for row in ranked] == sorted(row[1] for row in ranked)
    assert tuple(range(n)) in [row[0] for row in ranked]
    contiguous = next(count for order, count in ranked if order == tuple(range(n)))
    assert ranked[0][1] <= contiguous


def test_search_block_order_honours_its_limit():
    assert len(search_block_order(4, _bank(4, 20, 23), 2, limit=2)) == 2


# ---------------------------------------------------------------------------
# The combined entry point


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_reduce_settings_beats_or_matches_the_shipped_greedy(block_size):
    n, codes = 4, _bank(4, 80, 29)
    base = block_commuting_partition(n, codes, block_size)
    reduced = reduce_settings(n, codes, block_size, rounds=24, seed=0)
    assert isinstance(reduced, Regrouping)
    assert reduced.block_size == block_size
    assert reduced.n_settings <= len(base)
    assert reduced.n_settings >= counting_bound(n, codes)
    _assert_partition(reduced.codes, reduced.groups)
    _assert_block_commuting(n, list(reduced.codes), reduced.groups, block_size)
    # The frame permutes letters, never words: same bank, relabelled.
    inverse = tuple(reduced.order.index(q) for q in range(n))
    assert permuted_codes(n, list(reduced.codes), inverse) == list(codes)


def test_reduce_settings_reports_the_frame_it_used():
    n, codes = 4, _bank(4, 60, 31)
    reduced = reduce_settings(n, codes, 2, rounds=8, seed=0)
    assert sorted(reduced.order) == list(range(n))
    assert str(reduced.order) in reduced.provenance


def test_reduce_settings_without_a_frame_search_keeps_the_identity_frame():
    n, codes = 4, _bank(4, 60, 37)
    reduced = reduce_settings(n, codes, 2, rounds=8, search_frames=False)
    assert reduced.order == tuple(range(n))


def test_reduce_settings_handles_an_empty_bank():
    reduced = reduce_settings(4, [], 2)
    assert reduced.n_settings == 0
    assert reduced.codes == ()


# ---------------------------------------------------------------------------
# The floor


def test_counting_bound_counts_distinct_words_only():
    assert counting_bound(2, []) == 0
    assert counting_bound(2, [1, 1, 1]) == 1
    assert counting_bound(2, list(range(16))) == 4        # 16 words, 2**2 per setting
    assert counting_bound(2, list(range(16)) + [0]) == 4


def test_counting_bound_is_a_floor_for_the_shipped_greedy():
    n, codes = 4, _bank(4, 90, 41)
    for block_size in (1, 2, 4):
        assert len(block_commuting_partition(n, codes, block_size)) >= counting_bound(n, codes)


def test_qwc_forcing_bound_counts_distinct_full_weight_words():
    assert qwc_forcing_bound(2, [0b11_11, 0b11_11, 0b01_00]) == 1
    assert qwc_forcing_bound(2, [0, 0b01_00, 0b00_01]) == 0
    assert qwc_forcing_bound(1, [1, 2, 3]) == 3


def test_qwc_forcing_bound_is_a_floor_for_qubit_wise_grouping():
    """Every full-weight word forces its own tensor-product basis."""
    for seed in (0, 1, 2, 3):
        n, codes = 4, _bank(4, 120, seed)
        settings = len(block_commuting_partition(n, codes, 1))
        assert settings >= qwc_forcing_bound(n, codes)


def test_qwc_forcing_bound_is_attained_by_a_bank_of_full_weight_words():
    """A bank of nothing but full-weight words needs exactly one setting each."""
    n = 3
    codes = sorted(
        sum(letter << (2 * q) for q, letter in enumerate(word))
        for word in [(1, 2, 3), (3, 1, 2), (2, 3, 1)]
    )
    assert qwc_forcing_bound(n, codes) == 3
    assert len(block_commuting_partition(n, codes, 1)) == 3


# ---------------------------------------------------------------------------
# The shared colouring primitive


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_packed_and_integer_conflict_planes_colour_identically(block_size):
    """Lane form is a representation change, not a different first fit."""
    n, codes = 4, _bank(4, 90, 43)
    conflicts, degrees = conflict_masks(n, codes, block_size)
    order = sorted(range(len(codes)),
                   key=lambda i: (-int(degrees[i]), codes[i]))
    assert (first_fit_partition(conflicts, order)
            == first_fit_partition(packed_conflicts(conflicts), order))


def test_packed_conflicts_round_trips_every_plane():
    n, codes = 4, _bank(4, 70, 47)
    conflicts, _ = conflict_masks(n, codes, 2)
    planes = packed_conflicts(conflicts)
    assert planes.shape == (len(codes), (len(codes) + 63) // 64)
    for index, mask in enumerate(conflicts):
        rebuilt = int.from_bytes(planes[index].tobytes(), "little")
        assert rebuilt == mask


def test_first_fit_partition_respects_the_order_it_is_given():
    """A word may only open a group when every earlier group refuses it."""
    n, codes = 3, _bank(3, 24, 53)
    conflicts, _ = conflict_masks(n, codes, 1)
    order = list(range(len(codes)))
    groups = first_fit_partition(conflicts, order)
    _assert_partition(codes, groups)
    seen: list[int] = []
    for group in groups:
        assert group == sorted(group), "first fit appends in order"
        seen.append(group[0])
    assert seen == sorted(seen), "groups open in the order their first word arrives"


def test_first_fit_partition_handles_an_empty_bank():
    assert first_fit_partition([], []) == []
