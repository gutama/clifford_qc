"""Independent, fixed-endpoint evaluation of a pilot-frozen Rayleigh ratio."""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..reference_states import _digest
from ..sampled_reference import nonnegative_integer, settings_identity, validate_batch
from .cache import GroupedWordCache
from .functionals import WordFunctional


def uniform_setting_plan(settings, budget, *, minimum=2):
    budget = nonnegative_integer(budget, "budget")
    minimum = nonnegative_integer(minimum, "minimum")
    size = len(settings)
    if not size:
        return ()
    if budget < minimum * size:
        raise ValueError("budget cannot cover every setting at its minimum")
    quotient, remainder = divmod(budget, size)
    return tuple(quotient + (i < remainder) for i in range(size))


def pilot_setting_plan(settings, cache, functionals, budget, *, minimum=2):
    """Neyman allocation from pilot covariance; equal physical-shot costs.

    This is a *new independent batch*. Every setting gets its minimum even
    when it was already observed in the pilot or its pilot variance is zero.
    Integer remainder ties follow setting order; the budget is exact.
    A family minimizes the sum-of-variances proxy used by the complete bank.
    """
    baseline = uniform_setting_plan(settings, budget, minimum=minimum)
    if not settings:
        return baseline
    scores = {s.key: 0.0 for s in settings}
    for functional in functionals:
        if functional.is_deterministic:
            continue
        statistics = cache.candidate_group_statistics(functional.coefficients)
        if statistics is None:
            raise ValueError("pilot does not cover the allocation functional")
        for key in scores:
            scores[key] += statistics.get(key, (0, 0.0, 0.0))[1]
    weights = [math.sqrt(max(0.0, scores[s.key])) for s in settings]
    if sum(weights) == 0:
        return baseline
    remaining = budget - minimum * len(settings)
    raw = [remaining * weight / sum(weights) for weight in weights]
    extra = [math.floor(value) for value in raw]
    order = sorted(range(len(settings)), key=lambda i: (-(raw[i]-extra[i]), i))
    for i in order[:remaining-sum(extra)]:
        extra[i] += 1
    return tuple(minimum + value for value in extra)


def hoeffding_radius(functional, cache, delta):
    """Two-sided bounded-independent-shot radius, with within-shot dependence.

    Group g contributes n_g independent observations with range R_g/n_g
    each. Hoeffding gives r² = log(2/delta)/2 * sum_g R_g²/n_g.
    Group ranges use the coefficient l1 bound, never the observed range.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    if cache.pooling != "assigned":
        raise ValueError("ratio confidence requires single assignment")
    statistics = functional.group_terms(cache)
    if statistics is None:
        raise ValueError("functional has unmeasured words")
    return math.sqrt(0.5 * math.log(2/delta) * sum(
        width * width / shots for shots, _, width in statistics))


def _functional_identity(f):
    return [f.constant, sorted(f.coefficients.items())]


@dataclass(frozen=True)
class FrozenRatioPlan:
    n: int
    reference_identity: str
    hamiltonian_identity: str
    pilot_identity: str
    pilot_seed: int
    pilot_shots: int
    pilot_energy: float
    coefficients: object
    retained_frame: object
    numerator: WordFunctional
    denominator: WordFunctional
    settings: tuple
    counts: tuple[int, ...]
    delta: float
    denominator_floor: float
    circuit_identity: str

    @property
    def identity(self):
        return _digest(["frozen_ratio.v1", self.reference_identity,
                        self.hamiltonian_identity, self.pilot_identity,
                        self.pilot_energy, self.coefficients.tobytes().hex(),
                        self.retained_frame.tobytes().hex(),
                        _functional_identity(self.numerator),
                        _functional_identity(self.denominator),
                        self.circuit_identity, self.counts, self.delta, self.denominator_floor])


@dataclass(frozen=True)
class RatioEstimate:
    status: str
    estimate: float | None
    raw_ratio: float | None
    numerator: float
    denominator: float
    numerator_interval: tuple[float, float]
    denominator_interval: tuple[float, float]
    interval: tuple[float, float] | None
    numerator_denominator_covariance: float
    delta_variance: float | None
    evaluation_shots: int
    total_shots: int
    plan_identity: str
    observation_identity: str
    evidence: str = "finite_sample_conditional_on_pilot_iid_bounded_shots"


def evaluate_ratio(plan: FrozenRatioPlan, observations):
    """Evaluate H/S from the *same* fresh histograms, without refitting c.

    A Bonferroni joint rectangle for H and S has coverage >=1-delta without
    assuming they are independent. With S's lower endpoint above the floor,
    its four ratio corners enclose the true frozen-direction quotient.
    The covariance/delta variance is a separate asymptotic diagnostic.
    """
    if (observations.seed == plan.pilot_seed
            or observations.identity == plan.pilot_identity):
        raise ValueError("evaluation must use an independent pilot/evaluation seed")
    if (observations.reference_identity != plan.reference_identity
            or observations.settings_identity != plan.circuit_identity
            or settings_identity(plan.settings) != plan.circuit_identity
            or observations.counts != plan.counts):
        raise ValueError("observations do not match the frozen evaluation plan")
    validate_batch(observations.batch, plan.reference_identity, plan.n,
                   plan.settings, plan.counts)
    cache = GroupedWordCache(plan.n)
    if observations.batch.groups:
        cache.add_batch(observations.batch)
    h, s = plan.numerator.estimate(cache), plan.denominator.estimate(cache)
    rh = hoeffding_radius(plan.numerator, cache, plan.delta/2)
    rs = hoeffding_radius(plan.denominator, cache, plan.delta/2)
    hi, si = (h-rh, h+rh), (s-rs, s+rs)
    covariance = plan.numerator.covariance(plan.denominator, cache)
    raw = h/s if s != 0 else None
    variance = None if raw is None else max(0.0, (
        plan.numerator.variance(cache) + raw**2 * plan.denominator.variance(cache)
        - 2*raw*covariance) / s**2)
    valid = si[0] > plan.denominator_floor
    corners = [u/v for u in hi for v in si] if valid else []
    return RatioEstimate(
        "resolved" if valid else "denominator_unresolved", raw if valid else None,
        raw, h, s, hi, si, (min(corners), max(corners)) if valid else None,
        covariance, variance, cache.total_shots, plan.pilot_shots+cache.total_shots,
        plan.identity, observations.identity)


__all__ = ["FrozenRatioPlan", "RatioEstimate", "evaluate_ratio", "hoeffding_radius",
           "pilot_setting_plan", "uniform_setting_plan"]
