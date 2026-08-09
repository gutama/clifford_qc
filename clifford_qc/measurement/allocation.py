"""Shot-allocation policies.

A policy plans the *additional* shots for the next round, given the bank,
the cumulative cache, and the still-active candidate set. Cumulative reuse
is structural: policies only ever add shots on top of the cache. Candidate
elimination itself lives in the selector (successive elimination = any
policy + ``eliminate=True``).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .functionals import WordFunctional
from .grouping import shared_basis


class UniformFixed:
    """One round: the same fixed number of shots for every active word."""

    finite_schedule_valid = True

    def __init__(self, shots_per_word: int):
        if shots_per_word <= 0:
            raise ValueError("shots_per_word must be positive")
        self.shots_per_word = shots_per_word

    def plan(self, round_index: int, bank, cache, active: Sequence[int]) -> dict[int, int]:
        if round_index > 0:
            return {}
        return {w.code: self.shots_per_word for w in bank.words_for(active)}

    def planned_rounds(self) -> int:
        return 1


class UniformDoubling:
    """Escalate every active word to base * 2^round cumulative shots.

    The current algorithm of the standalone noisy-ADAPT study, expressed as
    cumulative targets: each round only adds the difference.
    """

    finite_schedule_valid = True

    def __init__(self, base: int, max_factor: int = 64):
        if base <= 0 or max_factor < 1:
            raise ValueError("base must be positive and max_factor >= 1")
        self.base = base
        self.max_factor = max_factor

    def plan(self, round_index: int, bank, cache, active: Sequence[int]) -> dict[int, int]:
        factor = 2 ** round_index
        if factor > self.max_factor:
            return {}
        target = self.base * factor
        plan = {}
        for w in bank.words_for(active):
            add = target - cache.shots(w.code)
            if add > 0:
                plan[w.code] = add
        return plan

    def planned_rounds(self) -> int:
        return self.max_factor.bit_length()


class VarianceProportional:
    """Split a growing round budget across active words by their current
    contribution to unresolved candidate variance, sum_j c_jw^2 var_w.

    Words never measured get one bootstrap shot minimum so the weights are
    defined next round.
    """

    # The next sample counts depend on earlier outcomes.  Fixed-N empirical-
    # Bernstein bounds over a predeclared grid do not by themselves certify
    # this policy; it needs a confidence sequence or an explicit union over
    # all attainable sample sizes.
    finite_schedule_valid = False

    def __init__(self, round_budget: int, growth: float = 2.0, max_rounds: int = 8):
        if round_budget <= 0:
            raise ValueError("round_budget must be positive")
        if growth < 1.0:
            raise ValueError("growth must be >= 1 (budgets must not shrink)")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.round_budget = round_budget
        self.growth = growth
        self.max_rounds = max_rounds

    def plan(self, round_index: int, bank, cache, active: Sequence[int]) -> dict[int, int]:
        if round_index >= self.max_rounds:
            return {}
        words = bank.words_for(active)
        if not words:
            return {}
        budget = int(self.round_budget * self.growth ** round_index)
        weights = {}
        for w in words:
            _, v = cache.mean_var(w.code)
            weight = 0.0
            for j in active:
                c = bank.coeffs[j].get(w.code)
                if c is not None:
                    weight += c * c * (1.0 if v == float("inf") else v)
            weights[w.code] = weight
        total = sum(weights.values())
        if total <= 0.0:
            return {}
        plan = {}
        for code, weight in weights.items():
            add = int(round(budget * weight / total))
            if cache.shots(code) == 0:
                add = max(add, 1)
            if add > 0:
                plan[code] = add
        return plan

    def planned_rounds(self) -> int:
        return self.max_rounds


def variance_optimal_group_plan(groups, cache, functionals: Sequence[WordFunctional],
                                budget: int, *, aggregation: str = "sum",
                                costs: Mapping[tuple, float] | None = None,
                                min_shots: int = 1) -> dict[int, int]:
    """Allocate a physical-shot budget across QWC measurement groups.

    For a scalar functional with per-shot group variance ``V_g`` and equal
    circuit costs, Neyman allocation is ``N_g proportional sqrt(V_g)``.  For a
    family of targets the default minimizes the sum of their first-order
    variances, using ``sqrt(sum_j V_jg / cost_g)``.  ``aggregation='max'`` is a
    conservative worst-target proxy rather than a closed-form minimax solver.

    The return value is word-keyed for compatibility with sampling backends,
    but every word in a group receives the same count.  Consequently the sum
    of one representative count per group -- not the sum over returned values
    -- equals ``budget``.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if min_shots < 0:
        raise ValueError("min_shots must be nonnegative")
    if aggregation not in ("sum", "max"):
        raise ValueError("aggregation must be 'sum' or 'max'")
    group_rows = []
    seen_codes: set[int] = set()
    seen_keys: set[tuple] = set()
    for group in groups:
        words = tuple(group)
        if not words:
            continue
        codes = tuple(word.code for word in words)
        if seen_codes.intersection(codes):
            raise ValueError("a word is assigned to more than one measurement group")
        seen_codes.update(codes)
        key = tuple(sorted(shared_basis(words).items()))
        if key in seen_keys:
            raise ValueError("measurement groups must have distinct shared bases")
        seen_keys.add(key)
        group_rows.append((key, words, set(codes)))
    if not group_rows:
        return {}

    targets = [functional for functional in functionals
               if not functional.is_deterministic]
    if not targets:
        return {}
    per_target: list[dict[tuple, float]] = []
    touched: set[tuple] = set()
    for functional in targets:
        rows = {}
        statistics = cache.candidate_group_statistics(functional.coefficients)
        for key, _, codes in group_rows:
            coeffs = [functional.coefficients[code]
                      for code in codes if code in functional.coefficients]
            if not coeffs:
                continue
            touched.add(key)
            if statistics is not None and key in statistics:
                rows[key] = max(0.0, float(statistics[key][1]))
            else:
                # Before the pilot, |sum c_w o_w| <= sum |c_w|.  Its square is
                # a safe per-shot variance proxy and keeps every touched group
                # eligible for its first sample.
                rows[key] = float(sum(abs(value) for value in coeffs) ** 2)
        per_target.append(rows)
    if not touched:
        return {}

    score = {}
    for key in touched:
        values = [row.get(key, 0.0) for row in per_target]
        score[key] = sum(values) if aggregation == "sum" else max(values)
    cost_map = dict(costs or {})
    for key in touched:
        if cost_map.get(key, 1.0) <= 0.0:
            raise ValueError("group costs must be positive")

    mandatory = {}
    for key, words, _ in group_rows:
        if key not in touched:
            continue
        current = max((cache.shots(word.code) for word in words), default=0)
        mandatory[key] = min_shots if current == 0 else 0
    required = sum(mandatory.values())
    if required > budget:
        raise ValueError(
            "budget is too small to bootstrap every touched measurement group")
    remaining = budget - required
    weights = {
        key: math.sqrt(score[key] / cost_map.get(key, 1.0))
        for key in touched
    }
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        weights = {key: 1.0 for key in touched}
        total_weight = float(len(touched))
    raw = {key: remaining * weights[key] / total_weight for key in touched}
    extra = {key: int(math.floor(raw[key])) for key in touched}
    leftovers = remaining - sum(extra.values())
    order = sorted(touched, key=lambda key: (-(raw[key] - extra[key]), key))
    for key in order[:leftovers]:
        extra[key] += 1
    group_shots = {key: mandatory.get(key, 0) + extra[key] for key in touched}

    plan: dict[int, int] = {}
    for key, words, _ in group_rows:
        shots = group_shots.get(key, 0)
        if shots > 0:
            plan.update({word.code: shots for word in words})
    return plan


class GroupVarianceOptimal:
    """Adaptive covariance-aware allocation over physical measurement groups.

    Unlike :class:`VarianceProportional`, this policy never pretends the words
    in one QWC circuit have independent shot counts.  It can be constructed
    before the grouping is known and bound by the workflow through
    :meth:`bind_groups` once per fixed-reference selection step.
    """

    finite_schedule_valid = False

    def __init__(self, round_budget: int, *, groups=None, growth: float = 2.0,
                 max_rounds: int = 8, aggregation: str = "sum",
                 min_shots: int = 1, costs: Mapping[tuple, float] | None = None):
        if round_budget <= 0:
            raise ValueError("round_budget must be positive")
        if growth < 1.0:
            raise ValueError("growth must be >= 1 (budgets must not shrink)")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if aggregation not in ("sum", "max"):
            raise ValueError("aggregation must be 'sum' or 'max'")
        if min_shots < 0:
            raise ValueError("min_shots must be nonnegative")
        self.round_budget = int(round_budget)
        self.growth = float(growth)
        self.max_rounds = int(max_rounds)
        self.aggregation = aggregation
        self.min_shots = int(min_shots)
        self.costs = dict(costs or {})
        self.groups = None
        if groups is not None:
            self.bind_groups(groups)

    def bind_groups(self, groups) -> None:
        self.groups = tuple(tuple(group) for group in groups)

    def plan(self, round_index: int, bank, cache, active: Sequence[int]) -> dict[int, int]:
        if round_index >= self.max_rounds:
            return {}
        if self.groups is None:
            raise ValueError("GroupVarianceOptimal must be bound to fixed groups")
        budget = int(self.round_budget * self.growth ** round_index)
        functionals = [WordFunctional({code: float(value)
                                       for code, value in bank.coeffs[index].items()})
                       for index in active]
        return variance_optimal_group_plan(
            self.groups, cache, functionals, budget,
            aggregation=self.aggregation, costs=self.costs,
            min_shots=self.min_shots)

    def planned_rounds(self) -> int:
        return self.max_rounds


__all__ = [
    "UniformFixed", "UniformDoubling", "VarianceProportional",
    "GroupVarianceOptimal", "variance_optimal_group_plan",
]
