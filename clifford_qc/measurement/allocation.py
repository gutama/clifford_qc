"""Shot-allocation policies.

A policy plans the *additional* shots for the next round, given the bank,
the cumulative cache, and the still-active candidate set. Cumulative reuse
is structural: policies only ever add shots on top of the cache. Candidate
elimination itself lives in the selector (successive elimination = any
policy + ``eliminate=True``).
"""

from __future__ import annotations

from typing import Sequence


class UniformFixed:
    """One round: the same fixed number of shots for every active word."""

    def __init__(self, shots_per_word: int):
        if shots_per_word <= 0:
            raise ValueError("shots_per_word must be positive")
        self.shots_per_word = shots_per_word

    def plan(self, round_index: int, bank, cache, active: Sequence[int]) -> dict[int, int]:
        if round_index > 0:
            return {}
        return {w.code: self.shots_per_word for w in bank.words_for(active)}


class UniformDoubling:
    """Escalate every active word to base * 2^round cumulative shots.

    The current algorithm of the standalone noisy-ADAPT study, expressed as
    cumulative targets: each round only adds the difference.
    """

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


class VarianceProportional:
    """Split a growing round budget across active words by their current
    contribution to unresolved candidate variance, sum_j c_jw^2 var_w.

    Words never measured get one bootstrap shot minimum so the weights are
    defined next round.
    """

    def __init__(self, round_budget: int, growth: float = 2.0, max_rounds: int = 8):
        if round_budget <= 0:
            raise ValueError("round_budget must be positive")
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
