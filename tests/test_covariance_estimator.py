"""Covariance-aware grouped variance and finite-schedule-valid radii (PRA revision).

The old per-word (diagonal) variance ignores within-QWC-group covariance and
can be wrong in either direction; the grouped cache carries the covariance
exactly. These tests pin the correctness that the certification guarantee
depends on.
"""

import numpy as np
import pytest

from clifford_qc.ir import PauliWord
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.backends.protocol import MeasurementBatch
from clifford_qc.measurement import (
    CommutatorBank, GroupedWordCache, candidate_radius, empirical_bernstein_radius,
    qwc_groups,
)
from clifford_qc.models import tfim
from clifford_qc.states import bell_density, ghz_density


def _cache(rho, groups, shots, seed, n):
    c = GroupedWordCache(n)
    c.add_batch(FiniteShotBackend(seed=seed).sample_grouped_from_state(rho, groups, shots))
    return c


def test_grouped_estimate_matches_exact_score():
    m = tfim(4, 1.0, 1.0)
    pool = tfim  # placeholder
    from clifford_qc.algorithms import local_pool
    pool = local_pool(4, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    rho = m.reference.state()
    groups = qwc_groups(bank.words)
    cache = GroupedWordCache(4)
    backend = FiniteShotBackend(seed=1)
    for _ in range(4):
        cache.add_batch(backend.sample_grouped_from_state(rho, groups, 20000))
    for j in range(len(bank)):
        assert cache.candidate_estimate(bank.coeffs[j]) == pytest.approx(
            bank.exact_score(j, rho), abs=0.05)


def test_covariance_aware_variance_matches_empirical_bell():
    """G = Z0 + Z1 on the Bell state: Z0,Z1 perfectly correlated. True
    Var(g_hat) = 4/N; the diagonal gives 2/N (2x under-estimate)."""
    rho = bell_density()
    coeffs = {PauliWord.from_label("ZI").code: 1.0, PauliWord.from_label("IZ").code: 1.0}
    words = [PauliWord.from_label("ZI"), PauliWord.from_label("IZ")]
    groups = qwc_groups(words)
    N = 400
    emp = np.var([_cache(rho, groups, N, s, 2).candidate_estimate(coeffs)
                  for s in range(1500)], ddof=1)
    c = _cache(rho, groups, N, 999, 2)
    cov = sum(sv / n for n, sv, _ in c.candidate_group_terms(coeffs))
    diag = sum(coeffs[cd] ** 2 * c.mean_var(cd)[1] for cd in coeffs)
    assert cov == pytest.approx(emp, rel=0.25)          # covariance-aware is right
    assert diag < 0.7 * emp                             # diagonal under-estimates
    assert cov > 1.6 * diag                             # ~2x correction


def test_covariance_aware_variance_zero_on_deterministic_stabilizer():
    """On GHZ, ZZ stabilizers are +1 deterministically, so a candidate built
    from them has exactly zero variance -- the diagonal wrongly reports > 0."""
    rho = ghz_density(3)
    words = [PauliWord.from_label(l) for l in ("ZZI", "IZZ")]
    coeffs = {words[0].code: 1.0, words[1].code: 1.0}
    groups = qwc_groups(words)
    c = _cache(rho, groups, 500, 3, 3)
    cov = sum(sv / n for n, sv, _ in c.candidate_group_terms(coeffs))
    diag = sum(coeffs[cd] ** 2 * c.mean_var(cd)[1] for cd in coeffs)
    assert cov == pytest.approx(0.0, abs=1e-12)
    assert diag > 0.0


def test_candidate_radius_normal_and_eb_shrink_with_shots():
    terms_small = [(100, 0.5, 2.0)]
    terms_big = [(10_000, 0.5, 2.0)]
    for bound in ("normal", "eb"):
        r_small = candidate_radius(terms_small, 0.05, 3, bound=bound, rounds=4)
        r_big = candidate_radius(terms_big, 0.05, 3, bound=bound, rounds=4)
        assert r_big < r_small
    # EB is a strictly more conservative finite-sample bound than the normal one
    assert candidate_radius(terms_small, 0.05, 3, bound="eb", rounds=4) > \
        candidate_radius(terms_small, 0.05, 3, bound="normal", rounds=4)


def test_candidate_radius_unmeasured_is_infinite():
    assert candidate_radius([], 0.05, 2, bound="normal") == float("inf")
    assert candidate_radius([(0, 0.1, 2.0)], 0.05, 2, bound="eb", rounds=2) == float("inf")


def test_empirical_bernstein_is_valid_finite_sample_bound():
    """Empirical-Bernstein coverage on a bounded mean should exceed 1-delta."""
    rng = np.random.default_rng(0)
    delta = 0.1
    N = 300
    misses = 0
    reps = 500
    for _ in range(reps):
        x = rng.uniform(-1, 1, size=N)  # bounded in [-1,1], range 2
        m_hat = x.mean()
        v = x.var(ddof=0)
        r = empirical_bernstein_radius(v, N, delta, value_range=2.0)
        if abs(m_hat - 0.0) > r:  # true mean is 0
            misses += 1
    assert misses / reps <= delta  # EB is conservative: coverage well above 1-delta


def test_grouped_cache_requires_grouped_batch():
    cache = GroupedWordCache(2)
    plain = MeasurementBatch(n=2, shots={1: 10}, plus_counts={1: 6}, circuits=1)
    with pytest.raises(ValueError, match="grouped batch"):
        cache.add_batch(plain)


def test_fast_infinite_shot_uses_exact_populations():
    """Infinite-shot FAST proxy is deterministic (no sampling) and reports
    zero shot cost -- it is the N -> infinity population limit."""
    from clifford_qc.algorithms import FastInspiredSelector, local_pool
    from clifford_qc.measurement import CommutatorBank
    m = tfim(3, 1.0, 0.5)
    pool = local_pool(3, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    rho = m.reference.state()
    sel = FastInspiredSelector(None, infinite_shot=True)
    assert sel.shots == 0
    a = sel.pick(rho, pool, list(range(len(pool))), m.hamiltonian, bank)
    b = FastInspiredSelector(None, infinite_shot=True).pick(
        rho, pool, list(range(len(pool))), m.hamiltonian, bank)
    assert a == b  # deterministic
    with pytest.raises(ValueError, match="shots must be positive"):
        FastInspiredSelector(0)
