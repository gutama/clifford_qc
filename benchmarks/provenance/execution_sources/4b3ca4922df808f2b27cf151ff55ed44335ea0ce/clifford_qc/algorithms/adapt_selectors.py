"""ADAPT-VQE operator-selection protocols and concrete strategies.

Selection is shared infrastructure, not a generic adaptive solver.  These
strategies decide *which operator* a caller should consider; ``run_adapt``
still owns circuit growth and parameter re-optimization, while A-CASE keeps
its separate Ritz-lowering mathematics in ``subspace.adaptive``.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from ..measurement.bank import CommutatorBank
from ..measurement.cache import WordCache
from ..measurement.confidence import candidate_radius, simultaneous_z_radius
from ..selection import SelectionStatus, canonical_argmax, eta_required

SelectionMode = Literal["confidence", "random", "population_proxy"]


@runtime_checkable
class AdaptSelectorProtocol(Protocol):
    """Minimal capability marker used by the ADAPT orchestration layer."""

    selection_mode: SelectionMode


@runtime_checkable
class ConfidenceSelectionProtocol(AdaptSelectorProtocol, Protocol):
    bound: str
    near_tol: float | None
    method: str
    delta: float

    def select(self, bank: CommutatorBank, cache: WordCache, sampler, allocator,
               candidates: Sequence[int]) -> tuple[int | None, SelectionStatus, dict]: ...


@runtime_checkable
class RandomSelectionProtocol(AdaptSelectorProtocol, Protocol):
    def pick(self, candidates: Sequence[int]) -> int: ...


@runtime_checkable
class PopulationSelectionProtocol(AdaptSelectorProtocol, Protocol):
    shots: int

    def pick(self, rho, pool, candidates: Sequence[int], hamiltonian,
             bank: CommutatorBank | None = None, tol: float = 1e-12): ...


class ConfidenceSelector:
    """Best-arm selection on ``|g_j|`` with simultaneous confidence bounds."""

    selection_mode: SelectionMode = "confidence"

    def __init__(self, delta: float = 0.05, threshold: float = 1e-6,
                 method: str = "bonferroni", eliminate: bool = True,
                 near_tol: float | None = None, bound: str = "normal"):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if bound not in ("normal", "eb"):
            raise ValueError("bound must be 'normal' or 'eb'")
        self.delta = delta
        self.threshold = threshold
        self.method = method
        self.eliminate = eliminate
        self.near_tol = near_tol
        self.bound = bound

    def _bounds(self, bank, cache, active, rounds, family_size=None):
        """Return ``{j: (estimate, lower, upper)}`` on ``|g_j|``."""
        # Keep the simultaneous-testing family fixed at the candidate set
        # declared before measurement. Recycling the smaller active set after
        # data-dependent elimination would require an alpha-recycling proof.
        m = len(active) if family_size is None else int(family_size)
        if m < len(active):
            raise ValueError("family_size cannot be smaller than active set")
        grouped = hasattr(cache, "candidate_group_terms")
        out = {}
        for j in active:
            if grouped:
                est = cache.candidate_estimate(bank.coeffs[j])
                terms = cache.candidate_group_terms(bank.coeffs[j])
                radius = (float("inf") if terms is None else
                          candidate_radius(terms, self.delta, m, bound=self.bound,
                                           method=self.method, rounds=rounds))
            else:
                est, variance = bank.estimate(j, cache)
                radius = simultaneous_z_radius(
                    variance, self.delta / rounds, m, self.method)
            out[j] = (est, max(0.0, abs(est) - radius), abs(est) + radius)
        return out

    def select(self, bank: CommutatorBank, cache: WordCache, sampler, allocator,
               candidates: Sequence[int]) -> tuple[int | None, SelectionStatus, dict]:
        """Allocate, measure, bound and resolve one finite-shot selection."""
        active = list(candidates)
        if not active:
            raise ValueError("no candidates to select from")
        if self.bound == "eb" and not getattr(allocator, "finite_schedule_valid", False):
            raise ValueError(
                "empirical-Bernstein certification requires a predeclared "
                "fixed-endpoint allocation schedule; use UniformFixed or "
                "UniformDoubling, or use bound='normal' for adaptive allocation")
        family_size = len(active)
        planned_rounds = getattr(allocator, "planned_rounds", None)
        if not callable(planned_rounds):
            raise TypeError("allocation policy must implement planned_rounds()")
        rounds = int(planned_rounds())
        if rounds < 1:
            raise ValueError("allocation policy planned_rounds() must be positive")
        best = None
        bounds: dict[int, tuple[float, float, float]] = {}
        for round_index in range(rounds):
            plan = allocator.plan(round_index, bank, cache, active)
            if not plan:
                raise ValueError("allocation schedule ended before planned_rounds()")
            words = [word for word in bank.words_for(active) if word.code in plan]
            cache.add_batch(sampler(words, plan))

            bounds = self._bounds(bank, cache, active, rounds, family_size)
            best = canonical_argmax(active, lambda j: abs(bounds[j][0]))
            best_lower = bounds[best][1]
            rival_upper = max((bounds[j][2] for j in active if j != best), default=0.0)

            if all(bounds[j][2] < self.threshold for j in active):
                return None, SelectionStatus.BELOW_THRESHOLD, self._diag(best, bounds, active)
            if best_lower >= self.threshold and best_lower > rival_upper:
                return best, SelectionStatus.RESOLVED_BEST, self._diag(best, bounds, active)
            if (self.near_tol is not None and best_lower >= self.threshold
                    and rival_upper - best_lower <= self.near_tol):
                return best, SelectionStatus.RESOLVED_EPS_BEST, self._diag(best, bounds, active)
            if self.eliminate:
                active = [j for j in active if j == best or bounds[j][2] >= best_lower]
        return (best, SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS,
                self._diag(best, bounds, active))

    @staticmethod
    def _diag(best, bounds, active) -> dict:
        out = {"active_candidates": len(active)}
        if best is not None and best in bounds:
            estimate, lower, upper = bounds[best]
            rival_upper = max((bounds[j][2] for j in active if j != best), default=0.0)
            out.update(estimate=estimate, lower_bound=lower, upper_bound=upper,
                       rival_upper=rival_upper,
                       eta_required=eta_required(lower, rival_upper))
        return out


class RandomSelector:
    """Uniform-random operator selection: the zero-measurement baseline."""

    selection_mode: SelectionMode = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def pick(self, candidates: Sequence[int]) -> int:
        return int(self.rng.choice(list(candidates)))


class FastInspiredSelector:
    """Determinant-population proxy selection (FAST-VQE-inspired baseline)."""

    selection_mode: SelectionMode = "population_proxy"

    def __init__(self, shots: int | None, seed: int = 0, *,
                 infinite_shot: bool = False):
        if not infinite_shot and (shots is None or shots <= 0):
            raise ValueError("shots must be positive unless infinite_shot=True")
        self.shots = 0 if infinite_shot else shots
        self.infinite_shot = infinite_shot
        self.rng = np.random.default_rng(seed)
        self._h_by_flips: dict[int, float] | None = None

    @staticmethod
    def _flip_mask(word) -> int:
        mask = 0
        for j in word.support():
            if word.letter(j) in ("X", "Y"):
                mask |= 1 << j
        return mask

    def _hamiltonian_connectivity(self, bank: CommutatorBank, hamiltonian
                                  ) -> dict[int, float]:
        if self._h_by_flips is None:
            by_flips: dict[int, float] = {}
            for word, coeff in hamiltonian.items():
                mask = self._flip_mask(word)
                if mask:
                    by_flips[mask] = by_flips.get(mask, 0.0) + abs(coeff)
            self._h_by_flips = by_flips
        return self._h_by_flips

    def pick(self, rho, pool, candidates: Sequence[int], hamiltonian,
             bank: CommutatorBank | None = None, tol: float = 1e-12):
        """Return ``(best index or None, proxy score)`` from one sampling round."""
        from ..states import computational_probabilities

        h_conn = self._hamiltonian_connectivity(bank, hamiltonian)
        outcomes = sorted(computational_probabilities(rho).items())
        probabilities = np.clip([probability for _, probability in outcomes], 0.0, None)
        probabilities = probabilities / probabilities.sum()
        if self.infinite_shot:
            p_hat = {bits: probability for (bits, _), probability
                     in zip(outcomes, probabilities) if probability > 0.0}
        else:
            counts = self.rng.multinomial(self.shots, probabilities)
            p_hat = {bits: count / self.shots for (bits, _), count
                     in zip(outcomes, counts) if count}

        def flipped(bits: str, mask: int) -> str:
            return "".join(("1" if char == "0" else "0") if (mask >> j) & 1 else char
                           for j, char in enumerate(bits))

        best, best_score = None, 0.0
        for j in candidates:
            mask = self._flip_mask(pool[j].word)
            if not mask:
                continue
            coupling = h_conn.get(mask, 0.0)
            score = 0.0
            for bits, probability in p_hat.items():
                partner = flipped(bits, mask)
                score += (probability * coupling
                          + math.sqrt(probability * p_hat.get(partner, 0.0)))
            if score > best_score + tol:
                best, best_score = j, score
        return best, best_score


__all__ = [
    "SelectionMode", "AdaptSelectorProtocol", "ConfidenceSelectionProtocol",
    "RandomSelectionProtocol", "PopulationSelectionProtocol", "ConfidenceSelector",
    "RandomSelector", "FastInspiredSelector",
]
