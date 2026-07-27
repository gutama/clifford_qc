"""Selection must not hand a tied step to floating-point noise.

The H4 pool is heavily degenerate by symmetry -- 160 candidates carry twelve
distinct gradient magnitudes, in tie groups as large as 72 -- so ``max``,
which takes whichever candidate is *strictly* largest, lets a perturbation
far below any physical scale decide which operator is appended. Two runs of
the same trajectory on four BLAS threads exchanged nine of twelve operators
that way. These tests pin the tolerance-aware replacement.
"""

import numpy as np
import pytest

from clifford_qc.algorithms.adapt import (
    TIE_ATOL, TIE_RTOL, canonical_argmax,
)


def test_reduces_to_plain_max_when_candidates_are_separated():
    """Not a change of selection rule: with a clear winner it *is* max."""
    scores = {0: 0.1, 1: 0.9, 2: 0.4}
    assert canonical_argmax(list(scores), lambda i: scores[i]) == 1


def test_exact_ties_resolve_to_the_lowest_index():
    scores = {5: 1.0, 2: 1.0, 9: 1.0, 7: 0.5}
    assert canonical_argmax(list(scores), lambda i: scores[i]) == 2
    # candidate order must not matter
    assert canonical_argmax([9, 7, 5, 2], lambda i: scores[i]) == 2


def test_noise_below_tolerance_does_not_move_the_choice():
    """The property the H4 trajectory needs, stated directly."""
    base = {0: 0.137, 1: 0.137, 2: 0.137, 3: 0.095}
    chosen = canonical_argmax(list(base), lambda i: base[i])
    rng = np.random.default_rng(0)
    moved_plain = 0
    for _ in range(500):
        p = {i: v * (1 + rng.normal(0, 1e-13)) for i, v in base.items()}
        assert canonical_argmax(list(p), lambda i: p[i]) == chosen
        if max(p, key=lambda i: p[i]) != chosen:
            moved_plain += 1
    assert moved_plain > 100      # plain max really is unstable here


def test_separations_wider_than_the_tolerance_are_still_resolved():
    """It must not merge genuinely distinct candidates."""
    lead = 0.137
    gap = 10 * (TIE_ATOL + TIE_RTOL * lead)
    scores = {0: lead - gap, 1: lead}
    assert canonical_argmax(list(scores), lambda i: scores[i]) == 1


def test_tolerance_scales_with_the_leader():
    """Relative, so near-zero leaders are not swallowed by an absolute width.

    H4 has near-zero gradients ~1e-7 separated by 4.9e-9. An absolute
    tolerance wide enough for a 0.137 leader would merge those; a relative one
    shrinks with the leader and keeps them apart.
    """
    tiny = {0: 1.0715974585115418e-07, 1: 1.1205956104709536e-07}
    assert canonical_argmax(list(tiny), lambda i: tiny[i]) == 1
    # while a pair differing below float resolution is treated as tied
    pair = {3: 1.1205956104709536e-07, 1: 1.1205956101933978e-07}
    assert canonical_argmax(list(pair), lambda i: pair[i]) == 1


def test_all_zero_candidates_are_tied():
    scores = {4: 0.0, 1: 0.0, 8: 0.0}
    assert canonical_argmax(list(scores), lambda i: scores[i]) == 1


def test_single_candidate():
    assert canonical_argmax([3], lambda i: 0.5) == 3


@pytest.mark.parametrize("seed", range(5))
def test_agrees_with_max_on_random_well_separated_scores(seed):
    rng = np.random.default_rng(seed)
    scores = {i: float(v) for i, v in enumerate(rng.normal(size=40))}
    mag = lambda i: abs(scores[i])
    assert canonical_argmax(list(scores), mag) == max(scores, key=mag)
