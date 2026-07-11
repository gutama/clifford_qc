"""Measurement layer: word cache, commutator bank, confidence bounds, and
allocation policies (backlog items 8-10)."""

import math

import numpy as np
import pytest

from clifford_qc.backends import FiniteShotBackend
from clifford_qc.backends.protocol import MeasurementBatch
from clifford_qc.ir import PauliSum, PauliWord
from clifford_qc.measurement import (
    WordCache, CommutatorBank, UniformFixed, UniformDoubling, VarianceProportional,
    jeffreys_mean_var, simultaneous_z_radius, empirical_bernstein_radius,
)
from clifford_qc.models import tfim
from clifford_qc.algorithms import local_pool
from clifford_qc.pauli import comm
from clifford_qc.states import expectation


# ---------------------------------------------------------------------------
# WordCache


def _batch(n, entries, circuits=None):
    return MeasurementBatch(
        n=n, shots={c: N for c, (N, _) in entries.items()},
        plus_counts={c: p for c, (_, p) in entries.items()},
        circuits=circuits if circuits is not None else len(entries))


def test_cache_accumulates_across_rounds():
    cache = WordCache(2)
    cache.add_batch(_batch(2, {5: (100, 80)}))
    cache.add_batch(_batch(2, {5: (50, 10), 9: (10, 5)}))
    assert cache.shots(5) == 150
    assert cache.mean(5) == pytest.approx((2 * 90 - 150) / 150)
    assert cache.unique_words() == 2
    assert cache.total_shots == 160
    assert cache.rounds == 2


def test_cache_unmeasured_word_is_maximally_uncertain():
    cache = WordCache(2)
    mean, var = cache.mean_var(3)
    assert mean == 0.0 and var == float("inf")
    with pytest.raises(KeyError):
        cache.mean(3)


def test_cache_jeffreys_never_zero_variance():
    cache = WordCache(1)
    cache.add_batch(_batch(1, {1: (100, 100)}))  # all +1 outcomes
    _, var = cache.mean_var(1)
    assert var > 0.0


# ---------------------------------------------------------------------------
# CommutatorBank


def test_bank_scores_agree_with_direct_commutators():
    m = tfim(4, J=1.3, h=0.7)
    pool = local_pool(4, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool],
                          [op.label for op in pool])
    rho = m.reference.state()
    H = m.hamiltonian.to_mv()
    for j, op in enumerate(pool):
        direct = expectation(rho, -0.5j * comm(H, op.word.to_mv())).real
        assert bank.exact_score(j, rho) == pytest.approx(direct, abs=1e-12)


def test_bank_estimate_reconstructs_from_exact_cache():
    """Feeding the cache exact means (huge N) reproduces the exact scores."""
    m = tfim(3)
    pool = local_pool(3, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    rho = m.reference.state()

    class ExactCache:
        def mean_var(self, code):
            ev = expectation(rho, PauliWord(3, code).to_mv()).real
            return ev, 0.0

    for j in range(len(bank)):
        est, var = bank.estimate(j, ExactCache())
        assert est == pytest.approx(bank.exact_score(j, rho), abs=1e-12)
        assert var == 0.0


def test_bank_shared_words_are_deduplicated():
    m = tfim(4)
    pool = local_pool(4, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    per_candidate = sum(len(row) for row in bank.coeffs)
    assert len(bank.words) < per_candidate  # overlap exists and is shared
    subset = bank.words_for([0, 1])
    union = set(bank.coeffs[0]) | set(bank.coeffs[1])
    assert {w.code for w in subset} == union


# ---------------------------------------------------------------------------
# Confidence


def test_jeffreys_mean_var_formulas():
    mean, var = jeffreys_mean_var(80, 100)
    p = 80.5 / 101.0
    assert mean == pytest.approx(2 * p - 1)
    assert var == pytest.approx((1 - mean ** 2) / 101.0)


def test_simultaneous_radius_grows_with_m_and_shrinks_with_delta():
    r1 = simultaneous_z_radius(0.01, 0.05, 1)
    r10 = simultaneous_z_radius(0.01, 0.05, 10)
    assert r10 > r1
    assert simultaneous_z_radius(0.01, 0.20, 10) < r10
    assert simultaneous_z_radius(float("inf"), 0.05, 3) == float("inf")


def test_empirical_bernstein_radius_shrinks_as_sqrt_n():
    r100 = empirical_bernstein_radius(0.25, 100, 0.05)
    r10000 = empirical_bernstein_radius(0.25, 10_000, 0.05)
    assert r10000 < r100 / 5


def test_empirical_bernstein_rejects_bad_inputs():
    with pytest.raises(ValueError, match="delta"):
        empirical_bernstein_radius(0.25, 100, 0.0)
    with pytest.raises(ValueError, match="delta"):
        empirical_bernstein_radius(0.25, 100, 1.0)
    with pytest.raises(ValueError, match="value_range"):
        empirical_bernstein_radius(0.25, 100, 0.05, value_range=0.0)


def test_simultaneous_coverage_on_synthetic_words():
    """Simultaneous intervals cover all true candidate values at >= 1-delta."""
    rng = np.random.default_rng(0)
    delta = 0.1
    m_cands = 4
    coeffs = rng.normal(size=(m_cands, 6))
    true_means = rng.uniform(-0.8, 0.8, size=6)
    true_vals = coeffs @ true_means
    N = 400
    misses = 0
    reps = 400
    for _ in range(reps):
        plus = rng.binomial(N, (1 + true_means) / 2)
        est_means = np.array([jeffreys_mean_var(p, N)[0] for p in plus])
        est_vars = np.array([jeffreys_mean_var(p, N)[1] for p in plus])
        est = coeffs @ est_means
        radius = np.array([simultaneous_z_radius(float(v), delta, m_cands)
                           for v in (coeffs ** 2) @ est_vars])
        if np.any(np.abs(est - true_vals) > radius):
            misses += 1
    assert misses / reps <= delta + 0.03  # nominal rate plus sampling slack


# ---------------------------------------------------------------------------
# Allocation


def _setup_bank_cache():
    m = tfim(3)
    pool = local_pool(3, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    return m, pool, bank, WordCache(3), FiniteShotBackend(seed=0)


def test_uniform_doubling_only_adds_the_difference():
    m, pool, bank, cache, backend = _setup_bank_cache()
    rho = m.reference.state()
    alloc = UniformDoubling(base=64, max_factor=4)
    active = list(range(len(bank)))

    plan0 = alloc.plan(0, bank, cache, active)
    assert all(v == 64 for v in plan0.values())
    words = [w for w in bank.words_for(active) if w.code in plan0]
    cache.add_batch(backend.sample_words_from_state(rho, words, plan0))

    plan1 = alloc.plan(1, bank, cache, active)
    assert all(v == 64 for v in plan1.values())  # 128 target - 64 held
    words = [w for w in bank.words_for(active) if w.code in plan1]
    cache.add_batch(backend.sample_words_from_state(rho, words, plan1))
    assert all(cache.shots(w.code) == 128 for w in bank.words_for(active))

    assert alloc.plan(3, bank, cache, active) == {}  # beyond max_factor


def test_uniform_fixed_is_single_round():
    _, _, bank, cache, _ = _setup_bank_cache()
    alloc = UniformFixed(100)
    assert alloc.plan(0, bank, cache, [0, 1])
    assert alloc.plan(1, bank, cache, [0, 1]) == {}


def test_variance_proportional_rejects_bad_configuration():
    with pytest.raises(ValueError, match="round_budget"):
        VarianceProportional(round_budget=0)
    with pytest.raises(ValueError, match="growth"):
        VarianceProportional(round_budget=100, growth=0.5)
    with pytest.raises(ValueError, match="max_rounds"):
        VarianceProportional(round_budget=100, max_rounds=0)


def test_variance_proportional_bootstraps_unmeasured_words():
    m, pool, bank, cache, backend = _setup_bank_cache()
    alloc = VarianceProportional(round_budget=300, max_rounds=3)
    plan = alloc.plan(0, bank, cache, list(range(len(bank))))
    assert plan
    assert all(v >= 1 for v in plan.values())
    assert alloc.plan(3, bank, cache, [0]) == {}
