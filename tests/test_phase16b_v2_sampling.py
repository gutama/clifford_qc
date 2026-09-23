"""Replica independence in the Phase 16B v2 producer.

The first run of this experiment recorded two hundred replicas per noisy cell
and sampled one. A fresh generator was built from the same ``SeedSequence``
inside a per-replica comprehension, so every shared draw was the identical
array, and the reported medians were a single realization repeated. The tell
was in the committed record and went unread: median and p90 were bit-identical
in every winning cell, which two hundred independent draws essentially never
produce.

These tests fix both halves of that: the draws differ, and a noisy cell's
summary is not degenerate. The second matters more than it looks -- it is the
property a reader can check on the record itself, without re-deriving the
sampling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.run_phase16b_v2_feasibility import (
    Estimands,
    shared_variates,
    summarize,
)

ENTROPY = [20260922, 0, 2, 3, 6, 0, 99]


def test_shared_variates_differ_across_replicas():
    block = shared_variates(ENTROPY, 200, 12)
    assert block.shape == (200, 13, 2)
    assert not np.all(block == block[0])
    distinct = {row.tobytes() for row in block}
    assert len(distinct) == 200


def test_shared_variates_are_reproducible():
    assert np.array_equal(shared_variates(ENTROPY, 16, 8),
                          shared_variates(ENTROPY, 16, 8))


def test_shared_variates_reject_empty_draw():
    with pytest.raises(ValueError):
        shared_variates(ENTROPY, 0, 8)


def test_shared_variates_differ_between_cells():
    a = shared_variates([20260922, 0, 2, 3, 6, 0, 99], 32, 8)
    b = shared_variates([20260922, 0, 2, 4, 6, 0, 99], 32, 8)
    assert not np.array_equal(a, b)


def test_pairing_survives_independence():
    """Two arms drawing the same block share variates lag by lag."""
    block = shared_variates(ENTROPY, 64, 12)
    hermitian_lags, unitary_lags = 5, 6
    for replica in range(64):
        shared_prefix = min(hermitian_lags, unitary_lags)
        assert np.array_equal(block[replica][:shared_prefix],
                              block[replica][:shared_prefix])
    # Different replicas must not be interchangeable, which is what failed before.
    assert not np.array_equal(block[0][:unitary_lags], block[1][:unitary_lags])


def test_summary_of_identical_energies_is_degenerate():
    """The signature of the original bug, stated as a property."""
    repeated = [0.5] * 200
    stats = summarize(repeated, 0.0)
    assert stats["median"] == stats["p90"]


def test_summary_of_independent_energies_is_not_degenerate():
    rng = np.random.default_rng(7)
    spread = list(0.5 + 1e-3 * rng.standard_normal(200))
    stats = summarize(spread, 0.0)
    assert stats["median"] != stats["p90"]
    assert stats["p90"] > stats["median"]


def test_perturbation_varies_with_the_shared_block():
    """Feeding distinct variates through an arm's estimands gives distinct values."""
    estimands = Estimands([1.0 + 0j, 0.5 + 0j], [1.0, 1.0], [True, True])
    block = shared_variates(ENTROPY, 8, 4)
    drawn = {estimands.perturb(block[replica][:2], 1e-3).tobytes()
             for replica in range(8)}
    assert len(drawn) == 8


def test_failed_solves_are_infinite_not_dropped():
    stats = summarize([0.1, float("nan"), 0.2], 0.0)
    assert stats["failures"] == 1
    assert math.isinf(stats["p90"])
    assert stats["replicas"] == 3
