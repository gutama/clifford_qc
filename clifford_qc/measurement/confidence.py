"""Confidence machinery for finite-shot candidate selection.

The selection score of candidate ``j`` is the linear functional
``g_j = sum_w c_jw <W_w>`` of Pauli-word expectations. Words are measured
in qubit-wise-commuting (QWC) groups; within a group the words are read
from one joint computational sample, so their outcomes are correlated,
and across groups the circuits are independent. We therefore estimate the
variance of ``g_hat_j`` group by group, using the actual per-shot combined
value of a candidate within a group, which is simultaneously
covariance-aware (it carries the within-group correlations exactly) and
compatible with a finite-sample concentration bound.

Two radii are provided on ``g_hat_j``:

- ``normal``: a covariance-aware Gaussian radius, fast and tight, valid
  asymptotically; used with a simultaneous Šidák/Bonferroni correction
  across candidates.
- ``eb`` (anytime-valid): an empirical-Bernstein radius on each group's
  bounded per-shot contribution, summed over groups. Combined with a union
  bound over candidates and over the finite schedule of decision rounds it
  yields a genuine finite-sample guarantee (Proposition below), unlike the
  Gaussian approximation.

Certification guarantee
-----------------------
Fix an error budget ``delta`` and a strict selector that (i) makes
decisions only at the ``R`` rounds of a predeclared allocation schedule,
(ii) at each round spends ``delta/R`` split across the ``m`` active
candidates, forming for each candidate a two-sided radius that holds with
probability at least ``1 - delta/(R m)`` for its (bounded) estimate, and
(iii) returns ``resolved_best`` only when the empirical leader's lower
bound strictly exceeds every rival's upper bound. Then, over the whole run,

    Pr(strict selector returns a resolved operator that is not the true
       argmax of |g_j|)  <=  delta,

under the assumptions: (A1) shots are independent given the fixed ansatz
state; (A2) each candidate's per-group per-shot contribution lies in a
known bounded range (it does: it is a signed sum of +/-1 word outcomes);
(A3) the radii are valid finite-sample bounds for bounded means -- true
for the empirical-Bernstein radius, and asymptotic for the Gaussian one.
The proof is a union bound: the resolution rule fails only if some active
candidate's true |g_j| lies outside its radius at some decision round; each
such event has probability at most ``delta/(R m)``; there are at most
``R m`` of them. Ties in |g_j| are never resolved (Eq. resolve is strict),
so an exact symmetry-tie yields abstention, not a wrong selection.
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
    with observed sample variance v and range R. Maurer & Pontil (2009).
    A true finite-sample two-sided bound for observations in an interval of
    width ``value_range``.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if value_range <= 0.0:
        raise ValueError("value_range must be positive")
    if N <= 0:
        return float("inf")
    log_term = math.log(3.0 / delta)
    return math.sqrt(2.0 * max(0.0, sample_var) * log_term / N) \
        + 3.0 * value_range * log_term / N


def candidate_radius(group_terms, delta: float, m: int, *, bound: str = "normal",
                     method: str = "sidak", rounds: int = 1) -> float:
    """Simultaneous two-sided radius on g_hat_j = sum_g mean(v_{j,g}).

    ``group_terms`` is an iterable of ``(N_g, sample_var_g, range_g)`` for
    the groups in which candidate ``j`` has support: ``N_g`` shots, sample
    variance and full range of the candidate's per-shot combined value
    ``v_{j,g}`` within group ``g``. Groups are independent circuits, so the
    variances add.

    - ``bound='normal'``: covariance-aware Gaussian radius at simultaneous
      level ``1-delta`` across ``m`` candidates.
    - ``bound='eb'``: anytime-valid empirical-Bernstein radius. The budget
      is split across ``m`` candidates and ``rounds`` decision rounds by a
      union bound, so the guarantee holds uniformly over the schedule.
    """
    terms = list(group_terms)
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not terms or any(N <= 0 for N, _, _ in terms):
        return float("inf")
    if bound == "normal":
        var = sum(sv / N for N, sv, _ in terms)
        return simultaneous_z_radius(var, delta / rounds, m, method)
    if bound == "eb":
        per = delta / (m * rounds)
        return sum(empirical_bernstein_radius(sv, N, per, value_range=max(rng, 1e-12))
                   for N, sv, rng in terms)
    raise ValueError("bound must be 'normal' or 'eb'")
