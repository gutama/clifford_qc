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
  only *asymptotically*; used with a dependence-safe Bonferroni correction
  across candidates by default. It supports asymptotic resolution, not a
  finite-sample certificate. Šidák remains available only as an explicit
  approximation when independence is defensible.
- ``eb`` (finite-schedule-valid): an empirical-Bernstein radius on each
  group's bounded per-shot contribution, summed over groups. Each group
  radius is a fixed-``N`` two-sided bound; the error budget is split over
  decision rounds, the fixed candidate family, and every QWC group touched by that
  candidate. A union bound over those events makes the summed candidate
  radius valid uniformly across the predeclared schedule (Proposition
  below), yielding a genuine finite-sample guarantee unlike the Gaussian
  approximation. This is
  finite-schedule-valid, not anytime-valid: it holds over the fixed set of
  ``R`` rounds, not simultaneously over all sample sizes (which would
  require a confidence sequence via a supermartingale / Ville's inequality).

Certification guarantee
-----------------------
Fix an error budget ``delta`` and a strict selector that (i) makes
decisions only at the ``R`` rounds of a predeclared allocation schedule,
(ii) at each round spends ``delta/R`` split across the fixed family of ``M``
candidates declared before measurement and, for candidate ``j``, across its
``q_j`` measured QWC groups,
forming group radii with failure probability at most
``delta/(R M q_j)`` whose sum bounds the candidate estimate, and
(iii) returns ``resolved_best`` only when the empirical leader's lower
bound strictly exceeds every rival's upper bound. Then, over one selector
call at one fixed ansatz state (including all of its decision rounds),

    Pr(strict selector returns a resolved operator that is not the true
       argmax of |g_j|)  <=  delta,

under the assumptions: (A1) shots are independent given the fixed ansatz
state; (A2) each candidate's per-group per-shot contribution lies in a
known bounded range (it does: it is a signed sum of +/-1 word outcomes);
(A3) every cumulative sample endpoint used for a decision is fixed before
the samples are observed; and (A4) the radii are valid finite-sample bounds
for bounded means -- true for the empirical-Bernstein radius, and asymptotic
for the Gaussian one.
The proof is a union bound: the resolution rule fails only if some active
candidate's true |g_j| lies outside its radius at some decision round; that
can happen only if one of its group means misses. The candidate-group events
for candidate ``j`` at a round sum to at most ``delta/(R M)``, and the fixed
candidate-family and round unions sum to ``delta``. The family size is not
reduced after data-dependent elimination; doing so would need a separate
alpha-recycling argument.

Exact-best vs eps-best. The strict exact-best rule resolves only when the
leader's lower bound strictly exceeds every rival's upper bound, so an exact
symmetry-tie in |g_j| (several operators sharing the top gradient, generic at
symmetric ansatz states) is never resolved -- it yields abstention, not a
wrong selection, but it also stalls a strict trajectory at the first
symmetric state. The eps-best rule instead resolves when the leader's lower
bound clears every rival's upper bound up to a tolerance ``eps``,
``L_best >= max_k U_k - eps``. On the same 1-delta interval event this
certifies ``|g_best| >= max_k |g_k| - eps`` (an (eps, delta)-PAC guarantee):
the selected operator's gradient is within ``eps`` of the maximum, and an
exact tie is resolved for any ``eps > 2r``. The eps-best rule uses the same
intervals, so it consumes no extra budget; it weakens *what* is certified
(an eps-best operator instead of the exact argmax) while keeping the same
confidence level, and it is what makes complete finite-sample-certified
trajectories possible.

The budget is per selector call, not automatically per multi-step ADAPT
trajectory. To guarantee a trajectory-level error budget ``delta_total``
over at most ``K`` selections by a union bound, instantiate the selector
with ``delta=delta_total/K``.
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
                          method: str = "bonferroni") -> float:
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
    with observed sample variance v and range R. This is the two-sided
    empirical-Bernstein bound of Audibert, Munos & Szepesvári (2009,
    Thm. 1), which holds with probability >= 1 - delta for observations in
    an interval of width ``value_range`` (their [0,1] statement rescaled by
    R). It is a true finite-sample bound at fixed N. (The one-sided
    Maurer & Pontil (2009) bound, with ln(2/delta) and a 7/(3(N-1)) linear
    term, is a different constant; we use the AMS two-sided form.)
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
                     method: str = "bonferroni", rounds: int = 1) -> float:
    """Simultaneous two-sided radius on g_hat_j = sum_g mean(v_{j,g}).

    ``group_terms`` is an iterable of ``(N_g, sample_var_g, range_g)`` for
    the groups in which candidate ``j`` has support: ``N_g`` shots, sample
    variance and full range of the candidate's per-shot combined value
    ``v_{j,g}`` within group ``g``. Groups are independent circuits, so the
    variances add.

    - ``bound='normal'``: covariance-aware Gaussian radius at simultaneous
      level ``1-delta`` across ``m`` candidates.
    - ``bound='eb'``: finite-schedule-valid empirical-Bernstein radius. For
      a candidate touching ``q`` groups, each group receives
      ``delta/(m * rounds * q)``. The union bound therefore covers every
      candidate-group event at every predeclared round (not all sample sizes).
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
        # The candidate estimate is a sum of group means.  We bound each mean
        # and sum the radii, so the familywise budget must include the number
        # of group events being union-bounded.  Omitting this factor gives a
        # candidate failure budget as large as q times the advertised value.
        per = delta / (m * rounds * len(terms))
        return sum(empirical_bernstein_radius(sv, N, per, value_range=max(rng, 1e-12))
                   for N, sv, rng in terms)
    raise ValueError("bound must be 'normal' or 'eb'")
