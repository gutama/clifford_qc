"""Strict certification vs confidence-guided fallback (PRA revision).

A strict run stops (abstains) when a selection is ambiguous at the budget;
a fallback run accepts the empirical leader but marks it not-certified.
Only strictly resolved (or exact) selections are labelled certified.
"""

import pytest

from clifford_qc.models import tfim
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import UniformDoubling
from clifford_qc.algorithms import (
    ConfidenceSelector, SelectionStatus, local_pool, run_adapt,
)
from clifford_qc.algorithms.adapt import _is_certified


def _run(accept, seed=3, budget=64):
    return run_adapt(
        tfim(4, 1.0, 1.0), local_pool(4, periodic_context=False),
        backend=FiniteShotBackend(seed=seed),
        selector=ConfidenceSelector(delta=0.05),  # no near_tol -> pure strict resolution
        allocator=UniformDoubling(base=256, max_factor=budget),
        max_operators=8, grouping=True, accept_ambiguous=accept)


def test_certified_only_for_resolved_or_exact():
    assert _is_certified(SelectionStatus.RESOLVED_BEST)
    assert _is_certified(SelectionStatus.EXACT)
    for s in (SelectionStatus.RESOLVED_NEAR_OPTIMAL,
              SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS,
              SelectionStatus.BELOW_THRESHOLD, SelectionStatus.RANDOM,
              SelectionStatus.FAST_PROXY):
        assert not _is_certified(s)


def test_strict_abstains_on_ambiguity():
    strict = _run(accept=False)
    assert strict.metadata["certification_mode"] == "strict"
    assert strict.abstentions >= 1
    assert "abstention" in strict.stopped_reason
    # a strict run never keeps an uncertified operator
    assert all(r.certified for r in strict.records if r.selected_label)


def test_fallback_proceeds_and_labels_uncertified():
    fallback = _run(accept=True)
    assert fallback.metadata["certification_mode"] == "fallback"
    assert fallback.abstentions == 0
    assert len(fallback.labels) > 0
    selected = [r for r in fallback.records if r.selected_label]
    # certified flag agrees with resolved_best status exactly
    for r in selected:
        assert r.certified == (r.status is SelectionStatus.RESOLVED_BEST)
    # fallback accepts some non-certified (ambiguous) operators here
    assert any(not r.certified for r in selected)


def test_exact_adapt_records_are_certified():
    res = run_adapt(tfim(4, 1.0, 1.0), local_pool(4, periodic_context=False),
                    max_operators=6)
    assert res.metadata["certification_mode"] == "n/a"
    assert all(r.certified for r in res.records if r.selected_label)


def test_bound_recorded_in_metadata():
    for bound in ("normal", "eb"):
        r = run_adapt(tfim(3, 1.0, 1.0), local_pool(3, periodic_context=False),
                      backend=FiniteShotBackend(seed=1),
                      selector=ConfidenceSelector(delta=0.05, near_tol=0.05, bound=bound),
                      allocator=UniformDoubling(base=128, max_factor=16),
                      max_operators=3, grouping=True)
        assert r.metadata["bound"] == bound
