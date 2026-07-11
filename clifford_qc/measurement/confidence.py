"""Confidence machinery for finite-shot candidate selection.

Tier 1 (robust default): Jeffreys-pseudocount word variances propagated
through the linear estimator, with a Sidak/Bonferroni simultaneous
correction across candidates.

Tier 2 (publication-grade): per-word empirical-Bernstein radii, valid for
bounded +/-1 outcomes and safe under adaptive stopping when combined with
a union bound over rounds (the selector spends delta over a geometric
round schedule).
"""

from __future__ import annotations

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def jeffreys_mean_var(plus: int, N: int) -> tuple[float, float]:
    """Jeffreys-smoothed (mean, variance) for the +/-1 mean from N shots.

    p~ = (N+ + 1/2)/(N + 1); mean~ = 2 p~ - 1; var = (1 - mean~^2)/(N + 1).
    """
    if N <= 0:
        return 0.0, float("inf")
    p = (plus + 0.5) / (N + 1.0)
    mean = 2.0 * p - 1.0
    return mean, max(0.0, 1.0 - mean * mean) / (N + 1.0)


def sidak_per_test_delta(delta: float, m: int, method: str = "sidak") -> float:
    """Per-test error budget for m simultaneous tests."""
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if m < 1:
        raise ValueError("m must be positive")
    if method == "sidak":
        return 1.0 - (1.0 - delta) ** (1.0 / m)
    if method == "bonferroni":
        return delta / m
    raise ValueError("method must be 'sidak' or 'bonferroni'")


def simultaneous_z_radius(variance: float, delta: float, m: int,
                          method: str = "sidak") -> float:
    """Two-sided normal confidence radius at simultaneous level 1-delta
    across m candidates: z_{1 - delta'/2} * sqrt(variance)."""
    if variance == float("inf"):
        return float("inf")
    d = sidak_per_test_delta(delta, m, method)
    z = _NORMAL.inv_cdf(1.0 - d / 2.0)
    return z * math.sqrt(max(0.0, variance))


def empirical_bernstein_radius(sample_var: float, N: int, delta: float,
                               value_range: float = 2.0) -> float:
    """Empirical-Bernstein radius for a mean of N bounded observations.

    |mean_hat - mean| <= sqrt(2 v ln(3/delta) / N) + 3 R ln(3/delta) / N
    with observed sample variance v and range R (2 for +/-1 outcomes).
    Maurer & Pontil (2009).
    """
    if N <= 0:
        return float("inf")
    log_term = math.log(3.0 / delta)
    return math.sqrt(2.0 * max(0.0, sample_var) * log_term / N) \
        + 3.0 * value_range * log_term / N
