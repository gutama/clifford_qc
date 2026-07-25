"""Strict certification vs confidence-guided fallback (PRA revision).

A strict run stops (abstains) when a selection is not strictly resolved;
a fallback run accepts the empirical leader but marks it not-certified.
Strict *resolution* is bound-independent (best-arm fired or exact), but a
genuine finite-sample *certificate* requires the empirical-Bernstein ('eb')
bound; the normal bound only resolves asymptotically. These are recorded
separately (``certification`` level and the derived ``certified`` bool).
"""

from clifford_qc.models import tfim
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import UniformDoubling, VarianceProportional
from clifford_qc.algorithms import (
    ConfidenceSelector, SelectionStatus, local_pool, run_adapt,
)
from clifford_qc.algorithms.adapt import (
    _certification_level, _is_certified, _is_resolved,
)


def test_confidence_bounds_keep_declared_family_after_elimination():
    """The alpha budget cannot grow merely because selection removed arms."""
    model = tfim(3, 1.0, 0.7)
    pool = local_pool(3, periodic_context=False)
    from clifford_qc.measurement import CommutatorBank, GroupedWordCache
    bank = CommutatorBank(model.hamiltonian, [op.word for op in pool])
    cache = GroupedWordCache(model.n)
    selector = ConfidenceSelector(bound="eb")
    # The guard pins the fixed-family contract used by select().
    try:
        selector._bounds(bank, cache, list(range(3)), rounds=2, family_size=2)
    except ValueError as exc:
        assert "family_size" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("smaller post-elimination family was accepted")


def _run(accept, seed=3, budget=64):
    return run_adapt(
        tfim(4, 1.0, 1.0), local_pool(4, periodic_context=False),
        backend=FiniteShotBackend(seed=seed),
        selector=ConfidenceSelector(delta=0.05),  # no near_tol -> pure strict resolution
        allocator=UniformDoubling(base=256, max_factor=budget),
        max_operators=8, grouping=True, accept_ambiguous=accept)


def test_resolution_covers_exact_best_and_eps_best():
    # both exact-best and eps-best are resolutions (eps-best is what lets a
    # strict run pass through symmetry-tied states)
    assert _is_resolved(SelectionStatus.RESOLVED_BEST)
    assert _is_resolved(SelectionStatus.RESOLVED_EPS_BEST)
    assert _is_resolved(SelectionStatus.EXACT)
    for s in (SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS,
              SelectionStatus.BELOW_THRESHOLD, SelectionStatus.RANDOM,
              SelectionStatus.FAST_PROXY):
        assert not _is_resolved(s)


def test_certification_level_separates_finite_sample_from_asymptotic():
    # exact gradient: certain
    assert _certification_level(SelectionStatus.EXACT, None) == "exact"
    # resolved (exact-best OR eps-best) under empirical-Bernstein: finite-sample
    assert _certification_level(SelectionStatus.RESOLVED_BEST, "eb") == "finite_sample"
    assert _certification_level(SelectionStatus.RESOLVED_EPS_BEST, "eb") == "finite_sample"
    # resolved under the normal bound: asymptotic only, NOT a certificate
    assert _certification_level(SelectionStatus.RESOLVED_BEST, "normal") == "asymptotic"
    assert _certification_level(SelectionStatus.RESOLVED_EPS_BEST, "normal") == "asymptotic"
    # unresolved outcomes carry no certification regardless of bound
    for s in (SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS,
              SelectionStatus.BELOW_THRESHOLD, SelectionStatus.RANDOM,
              SelectionStatus.FAST_PROXY):
        assert _certification_level(s, "eb") == "none"

    assert not _is_certified("exact")  # exact is distinct, not statistical
    assert _is_certified("finite_sample")
    assert not _is_certified("asymptotic")  # the key separation
    assert not _is_certified("none")


def test_strict_abstains_on_ambiguity():
    strict = _run(accept=False)  # default normal bound
    assert strict.metadata["certification_mode"] == "strict"
    assert strict.abstentions >= 1
    assert "abstention" in strict.stopped_reason
    # a strict run keeps only strictly resolved operators...
    selected = [r for r in strict.records if r.selected_label]
    assert all(_is_resolved(r.status) for r in selected)
    # ...but under the normal bound that resolution is asymptotic, not a
    # finite-sample certificate.
    assert all(r.certification == "asymptotic" for r in selected)
    assert not any(r.certified for r in selected)


def test_strict_accepts_certified_eps_best():
    # With the eps-best rule enabled (near_tol), a strict run RESOLVES and
    # certifies eps-best selections instead of abstaining, so it makes progress
    # through the symmetry-tied reference state (6-fold gradient tie for TFIM).
    res = run_adapt(
        tfim(4, 1.0, 1.0), local_pool(4, periodic_context=False),
        backend=FiniteShotBackend(seed=3),
        selector=ConfidenceSelector(delta=0.05, near_tol=0.30, bound="eb"),
        allocator=UniformDoubling(base=512, max_factor=64),
        max_operators=4, grouping=True, accept_ambiguous=False)
    selected = [r for r in res.records if r.selected_label]
    assert selected  # did not stall at operator 1
    # the first (symmetric) step resolves only via the eps-best rule
    assert any(r.resolution == "eps_best" for r in selected)
    for r in selected:
        assert r.resolution in ("best", "eps_best")
        assert r.certification == "finite_sample"  # eb bound
        assert r.certified


def test_fallback_proceeds_and_labels_uncertified():
    fallback = _run(accept=True)  # default normal bound
    assert fallback.metadata["certification_mode"] == "fallback"
    assert fallback.abstentions == 0
    assert len(fallback.labels) > 0
    selected = [r for r in fallback.records if r.selected_label]
    # a certification level is present exactly for resolved selections
    for r in selected:
        assert (r.certification != "none") == _is_resolved(r.status)
    # normal bound never yields a finite-sample certificate
    assert all(r.certification in ("asymptotic", "none") for r in selected)
    assert not any(r.certified for r in selected)
    # fallback accepts some unresolved (ambiguous) operators here
    assert any(r.certification == "none" for r in selected)


def test_eb_bound_never_labels_asymptotic():
    """Under the empirical-Bernstein bound a selection is either a genuine
    finite-sample certificate or unresolved -- never the asymptotic level the
    normal bound produces. (The RESOLVED_BEST+eb -> finite_sample mapping
    itself is pinned by test_certification_level_*.)"""
    res = run_adapt(
        tfim(4, 1.0, 1.0), local_pool(4, periodic_context=False),
        backend=FiniteShotBackend(seed=3),
        selector=ConfidenceSelector(delta=0.05, bound="eb"),
        allocator=UniformDoubling(base=256, max_factor=64),
        max_operators=8, grouping=True, accept_ambiguous=True)
    assert res.metadata["bound"] == "eb"
    for r in res.records:
        assert r.certification in ("finite_sample", "none")
        assert r.certified == (r.certification == "finite_sample")


def test_eb_rejects_outcome_adaptive_sample_counts():
    """Variance-proportional endpoints need a confidence sequence."""
    import pytest
    with pytest.raises(ValueError, match="predeclared fixed-endpoint"):
        run_adapt(
            tfim(3, 1.0, 0.7), local_pool(3, periodic_context=False),
            backend=FiniteShotBackend(seed=1),
            selector=ConfidenceSelector(delta=0.05, bound="eb"),
            allocator=VarianceProportional(256, max_rounds=3),
            max_operators=1, grouping=True, accept_ambiguous=False)


def test_exact_adapt_records_are_exact_not_statistically_certified():
    res = run_adapt(tfim(4, 1.0, 1.0), local_pool(4, periodic_context=False),
                    max_operators=6)
    assert res.metadata["certification_mode"] == "n/a"
    selected = [r for r in res.records if r.selected_label]
    assert all(r.certification == "exact" for r in selected)
    assert not any(r.certified for r in selected)


def test_bound_recorded_in_metadata():
    for bound in ("normal", "eb"):
        r = run_adapt(tfim(3, 1.0, 1.0), local_pool(3, periodic_context=False),
                      backend=FiniteShotBackend(seed=1),
                      selector=ConfidenceSelector(delta=0.05, near_tol=0.05, bound=bound),
                      allocator=UniformDoubling(base=128, max_factor=16),
                      max_operators=3, grouping=True)
        assert r.metadata["bound"] == bound
        assert r.metadata["simultaneous_method"] == "bonferroni"
        assert r.metadata["confidence_delta_per_selection"] == 0.05
        assert r.metadata["certification_scope"] == (
            "per_selection_call" if bound == "eb" else None)
        assert r.metadata["allocation_finite_schedule_valid"] is True
