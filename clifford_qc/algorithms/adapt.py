"""ADAPT-VQE with exact or confidence-certified finite-shot selection.

The finite-shot path implements the Paper A machinery end to end:

- one ``CommutatorBank`` holds every selection observable G_j = -i/2 [H,P_j]
  over the shared word set;
- one ``WordCache`` per step accumulates all measurements while the state is
  fixed, so shared words are measured once per allocation round (H2);
- an allocation policy plans additional shots each round (cumulative reuse);
- ``ConfidenceSelector`` applies simultaneous confidence bounds on |g_j| and
  returns an explicit four-way outcome — it never silently picks the
  empirical maximum while ambiguity remains (H3).

Only the operator *selection* is noisy; parameter re-optimization stays
exact (adjoint gradients), isolating ranking noise as in the standalone
study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from ..ir import Parameter, Program, Rotor, adjoint_gradient
from ..backends.exact_mv import ExactMVBackend
from ..backends.finite_shot import FiniteShotBackend
from ..measurement.bank import CommutatorBank
from ..measurement.cache import WordCache
from ..measurement.confidence import simultaneous_z_radius
from ..measurement.grouping import qwc_groups
from .layering import build_layer
from .optimize import minimize_energy
from .pools import PoolOperator


class SelectionStatus(Enum):
    RESOLVED_BEST = "resolved_best"
    RESOLVED_NEAR_OPTIMAL = "resolved_near_optimal"
    BELOW_THRESHOLD = "below_threshold"
    BUDGET_EXHAUSTED_AMBIGUOUS = "budget_exhausted_ambiguous"
    EXACT = "exact"
    RANDOM = "random"
    FAST_PROXY = "fast_proxy"


@dataclass(frozen=True)
class SelectionRecord:
    step: int
    selected_label: str | None
    status: SelectionStatus
    estimate: float | None
    lower_bound: float | None
    upper_bound: float | None
    exact_gradient: float | None
    exact_gradient_max: float | None
    shots_added: int
    cumulative_shots: int
    unique_words_measured: int
    circuits_executed: int
    active_candidates: int
    energy: float | None = None
    layer_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdaptResult:
    labels: tuple[str, ...]
    parameters: tuple[float, ...]
    energy: float
    exact_ground_energy: float | None
    relative_error: float | None
    records: tuple[SelectionRecord, ...]
    total_shots: int
    total_circuits: int
    support_peak: int
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    optimizer_evaluations: int = 0


class ConfidenceSelector:
    """Best-arm selection on |g_j| with simultaneous confidence bounds.

    Selects candidate ĵ = argmax |ĝ_j| only once its lower bound clears
    every rival's upper bound (wrong-selection probability <= delta under
    the tier-1 normal approximation; the delta budget is split across
    candidates by Sidak/Bonferroni and across allocation rounds by a union
    bound). ``eliminate=True`` drops candidates whose upper bound falls
    below the best lower bound (successive elimination), shrinking the
    measured word set in later rounds.
    """

    def __init__(self, delta: float = 0.05, threshold: float = 1e-6,
                 method: str = "sidak", eliminate: bool = True,
                 near_tol: float | None = None):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        self.delta = delta
        self.threshold = threshold
        self.method = method
        self.eliminate = eliminate
        self.near_tol = near_tol

    @staticmethod
    def _planned_rounds(allocator) -> int:
        if hasattr(allocator, "max_rounds"):
            return max(1, int(allocator.max_rounds))
        if hasattr(allocator, "max_factor"):
            return max(1, int(math.log2(allocator.max_factor)) + 1)
        return 1

    def select(self, bank: CommutatorBank, cache: WordCache, sampler, allocator,
               candidates: Sequence[int]) -> tuple[int | None, SelectionStatus, dict]:
        """One selection: allocate, measure, bound, and decide.

        ``sampler(words, plan) -> MeasurementBatch`` measures on the current
        (fixed) state. Returns ``(index, status, diagnostics)``; index is the
        empirical best even for ambiguous outcomes so the caller can choose
        its acceptance policy.
        """
        active = list(candidates)
        if not active:
            raise ValueError("no candidates to select from")
        delta_round = self.delta / self._planned_rounds(allocator)
        best = None
        bounds: dict[int, tuple[float, float, float]] = {}
        for round_index in range(10 ** 6):
            plan = allocator.plan(round_index, bank, cache, active)
            if not plan:
                status = SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS
                break
            words = [w for w in bank.words_for(active) if w.code in plan]
            cache.add_batch(sampler(words, plan))

            bounds = {}
            for j in active:
                est, var = bank.estimate(j, cache)
                r = simultaneous_z_radius(var, delta_round, len(active), self.method)
                bounds[j] = (est, max(0.0, abs(est) - r), abs(est) + r)
            best = max(active, key=lambda j: abs(bounds[j][0]))
            best_lower = bounds[best][1]
            rival_upper = max((bounds[j][2] for j in active if j != best), default=0.0)

            if all(bounds[j][2] < self.threshold for j in active):
                return None, SelectionStatus.BELOW_THRESHOLD, self._diag(best, bounds, active)
            if best_lower >= self.threshold and best_lower > rival_upper:
                return best, SelectionStatus.RESOLVED_BEST, self._diag(best, bounds, active)
            if (self.near_tol is not None and best_lower >= self.threshold
                    and rival_upper - best_lower <= self.near_tol):
                return best, SelectionStatus.RESOLVED_NEAR_OPTIMAL, self._diag(best, bounds, active)
            if self.eliminate:
                active = [j for j in active
                          if j == best or bounds[j][2] >= best_lower]
        else:  # pragma: no cover - loop bound is effectively unreachable
            status = SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS
        return best, status, self._diag(best, bounds, active)

    @staticmethod
    def _diag(best, bounds, active) -> dict:
        out = {"active_candidates": len(active)}
        if best is not None and best in bounds:
            est, lo, up = bounds[best]
            out.update(estimate=est, lower_bound=lo, upper_bound=up)
        return out


def _cache_estimates(bank: CommutatorBank, cache: WordCache,
                     candidates: Sequence[int]) -> dict[int, float]:
    """Point estimates for candidates whose every word has been measured
    (candidates eliminated before round 0 completed have no valid estimate)."""
    out = {}
    for j in candidates:
        if all(cache.shots(code) > 0 for code in bank.coeffs[j]):
            out[j] = bank.estimate(j, cache)[0]
    return out


class RandomSelector:
    """Uniform-random operator selection: the zero-measurement baseline."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def pick(self, candidates: Sequence[int]) -> int:
        return int(self.rng.choice(list(candidates)))


class FastInspiredSelector:
    """Determinant-population proxy selection (FAST-VQE-inspired baseline).

    One computational-basis measurement circuit per step: sample N
    bitstrings of the current state, then score each candidate word P by

        score(P) = sum_b p_hat(b) * |H|(flips(P))
                 + sum_b sqrt(p_hat(b) * p_hat(b ^ flips(P)))

    where flips(P) are P's X/Y positions and |H|(f) = sum of |c_w| over
    Hamiltonian words with the same flip pattern. The first term is the
    population-weighted Hamiltonian connectivity of the determinants a
    candidate would couple (MP2-flavored; nonzero even at a bare
    determinant reference), the second rewards candidates linking two
    already-populated determinants. Everything besides the N samples is
    classical post-processing of known H coefficients, so the measurement
    cost is one circuit and N shots per step. Determinant populations —
    not the commutator gradient — are the signal, which is why this is a
    chemistry baseline: phases are invisible to the proxy.
    """

    def __init__(self, shots: int, seed: int):
        if shots <= 0:
            raise ValueError("shots must be positive")
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self._h_by_flips: dict[int, float] | None = None

    @staticmethod
    def _flip_mask(word) -> int:
        mask = 0
        for j in word.support():
            if word.letter(j) in ("X", "Y"):
                mask |= 1 << j
        return mask

    def _hamiltonian_connectivity(self, bank: CommutatorBank, hamiltonian) -> dict[int, float]:
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
        """(best index or None, its proxy score) from one sampling round."""
        from ..states import computational_probabilities

        h_conn = self._hamiltonian_connectivity(bank, hamiltonian)
        outcomes = sorted(computational_probabilities(rho).items())
        probs = np.clip([p for _, p in outcomes], 0.0, None)
        counts = self.rng.multinomial(self.shots, probs / probs.sum())
        p_hat = {bits: c / self.shots for (bits, _), c in zip(outcomes, counts) if c}

        def flipped(bits: str, mask: int) -> str:
            return "".join(("1" if ch == "0" else "0") if (mask >> j) & 1 else ch
                           for j, ch in enumerate(bits))

        best, best_score = None, 0.0
        for j in candidates:
            mask = self._flip_mask(pool[j].word)
            if not mask:
                continue  # Z-only candidates move no populations
            coupling = h_conn.get(mask, 0.0)
            score = 0.0
            for bits, p in p_hat.items():
                partner = flipped(bits, mask)
                score += p * coupling + math.sqrt(p * p_hat.get(partner, 0.0))
            if score > best_score + tol:
                best, best_score = j, score
        return best, best_score


def _ansatz_program(model, chosen: Sequence[PoolOperator]) -> Program:
    prog = Program(model.n)
    for op in model.reference.ops:
        prog.append(op)
    for k, pool_op in enumerate(chosen):
        prog.append(Rotor(pool_op.word, Parameter(f"t{k}")))
    return prog


def run_adapt(model, pool: Sequence[PoolOperator], *,
              backend: FiniteShotBackend | None = None,
              selector: "ConfidenceSelector | RandomSelector | FastInspiredSelector | None" = None,
              allocator=None,
              max_operators: int = 10, threshold: float = 1e-6,
              allow_repeats: bool = False, accept_ambiguous: bool = True,
              optimizer_method: str = "auto", maxiter: int = 350,
              compute_exact_reference: bool = True,
              track_exact_scores: bool = True,
              grouping: bool = False,
              subpool_size: int | None = None, subpool_seed: int = 0,
              layer_alpha: float | None = None) -> AdaptResult:
    """ADAPT-VQE. Selection is exact when ``selector`` is None, uniform-random
    with a ``RandomSelector`` (zero-measurement baseline), and finite-shot
    (requiring a sampling ``backend`` and an ``allocator``) with a
    ``ConfidenceSelector``.

    ``track_exact_scores`` also records the exact gradient of each selection
    for regret/near-optimality analysis (simulator-only diagnostic).
    ``grouping`` measures qubit-wise-commuting word groups one circuit each.
    ``subpool_size`` restricts every step's candidates to a seeded random
    subpool (subpool exploration); exact scores and regret are then relative
    to the subpool. ``layer_alpha`` appends, after each selection, the
    mutually commuting candidates scoring at least ``layer_alpha`` times the
    selected score (layered ADAPT).
    """
    random_mode = isinstance(selector, RandomSelector)
    fast_mode = isinstance(selector, FastInspiredSelector)
    noisy = selector is not None and not random_mode and not fast_mode
    if noisy and (backend is None or allocator is None):
        raise ValueError("finite-shot selection needs a sampling backend and an allocator")
    if subpool_size is not None and subpool_size < 1:
        raise ValueError("subpool_size must be positive")
    subpool_rng = np.random.default_rng(subpool_seed)
    exact_backend = backend.inner if isinstance(backend, FiniteShotBackend) else \
        (backend if backend is not None else ExactMVBackend())

    pool = list(pool)
    if not pool:
        raise ValueError("empty operator pool")
    H = model.hamiltonian
    bank = CommutatorBank(H, [op.word for op in pool], [op.label for op in pool])

    E0 = None
    if compute_exact_reference:
        from ..matrix import exact_ground
        E0, _ = exact_ground(H.to_mv())

    chosen: list[PoolOperator] = []
    theta: tuple[float, ...] = ()
    used: set[int] = set()
    records: list[SelectionRecord] = []
    total_shots = 0
    total_circuits = 0
    support_peak = 0
    optimizer_evaluations = 0
    energy = exact_backend.expectation(_ansatz_program(model, []), H, ())
    stopped_reason = "operator budget reached"

    for step in range(1, max_operators + 1):
        program = _ansatz_program(model, chosen)
        rho = exact_backend.state(program, theta)
        support_peak = max(support_peak, getattr(exact_backend, "support_peak", 0))
        untried = [i for i in range(len(pool)) if allow_repeats or i not in used]
        if not untried:
            stopped_reason = "pool exhausted"
            break

        cache = WordCache(model.n) if noisy else None
        # A below-threshold subpool triggers a redraw from the untried
        # remainder (subpool exploration); only a dead remainder stops the run.
        while True:
            if subpool_size is not None and len(untried) > subpool_size:
                candidates = sorted(subpool_rng.choice(untried, size=subpool_size,
                                                       replace=False).tolist())
            else:
                candidates = list(untried)

            exact_scores = None
            exact_max = None
            if track_exact_scores or not noisy or random_mode:
                exact_scores = {i: bank.exact_score(i, rho) for i in candidates}
                exact_max = max(abs(v) for v in exact_scores.values())

            shots_added = 0
            words_measured = 0
            if fast_mode:
                idx, proxy_score = selector.pick(rho, pool, candidates, H, bank)
                shots_added = selector.shots
                total_shots += selector.shots
                total_circuits += 1
                if idx is None:
                    status = SelectionStatus.BELOW_THRESHOLD
                    diag = {"estimate": 0.0, "active_candidates": len(candidates)}
                else:
                    status = SelectionStatus.FAST_PROXY
                    diag = {"estimate": proxy_score,
                            "active_candidates": len(candidates)}
            elif random_mode:
                if exact_max < threshold:
                    idx, status = None, SelectionStatus.BELOW_THRESHOLD
                    diag = {"active_candidates": len(candidates)}
                else:
                    idx = selector.pick(candidates)
                    status = SelectionStatus.RANDOM
                    diag = {"active_candidates": len(candidates)}
            elif not noisy:
                best = max(candidates, key=lambda i: abs(exact_scores[i]))
                diag = {"estimate": exact_scores[best],
                        "exact_gradient": exact_scores[best],
                        "active_candidates": len(candidates)}
                if abs(exact_scores[best]) < threshold:
                    idx, status = None, SelectionStatus.BELOW_THRESHOLD
                else:
                    idx, status = best, SelectionStatus.EXACT
            else:
                if grouping:
                    sampler = lambda words, plan: backend.sample_grouped_from_state(
                        rho, qwc_groups(words), plan)
                else:
                    sampler = lambda words, plan: backend.sample_words_from_state(
                        rho, words, plan)
                idx, status, diag = selector.select(bank, cache, sampler, allocator,
                                                    candidates)
                shots_added = cache.total_shots  # cumulative over this step's redraws
                words_measured = cache.unique_words()

            if status is SelectionStatus.BELOW_THRESHOLD and subpool_size is not None:
                remaining = [c for c in untried if c not in candidates]
                if remaining:
                    untried = remaining
                    continue
            break

        if noisy:
            total_shots += cache.total_shots
            total_circuits += cache.total_circuits
        if status is SelectionStatus.BELOW_THRESHOLD:
            if fast_mode:
                stopped_reason = "proxy score below threshold"
            elif noisy:
                stopped_reason = "no candidate above threshold"
            else:
                stopped_reason = "exact gradient below threshold"
        elif status is SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS and not accept_ambiguous:
            stopped_reason = "selection ambiguous at budget"
        if idx is None or (status is SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS
                           and not accept_ambiguous):
            records.append(SelectionRecord(
                step, None, status, diag.get("estimate"), diag.get("lower_bound"),
                diag.get("upper_bound"), diag.get("exact_gradient"), exact_max,
                shots_added, total_shots, words_measured, total_circuits,
                diag.get("active_candidates", len(candidates))))
            break

        chosen.append(pool[idx])
        used.add(idx)
        theta = theta + (0.0,)
        layer_labels: tuple[str, ...] = ()
        if layer_alpha is not None:
            if noisy and cache is not None:  # hardware-realistic: measured estimates
                layer_scores = _cache_estimates(bank, cache, candidates)
            else:
                layer_scores = dict(exact_scores or {})
            if idx in layer_scores and layer_scores[idx] != 0.0:
                layer = build_layer(pool, layer_scores, idx, candidates, layer_alpha)
                for j in layer[1:]:
                    chosen.append(pool[j])
                    used.add(j)
                    theta = theta + (0.0,)
                layer_labels = tuple(pool[j].label for j in layer[1:])
        program = _ansatz_program(model, chosen)

        def f_eg(x):
            E = exact_backend.expectation(program, H, x)
            return E, adjoint_gradient(program, H, x)

        res = minimize_energy(f_eg, theta, method=optimizer_method,
                              maxiter=maxiter, bound=E0)
        theta = res.x
        energy = res.fun
        optimizer_evaluations += res.evaluations
        support_peak = max(support_peak, getattr(exact_backend, "support_peak", 0))
        records.append(SelectionRecord(
            step, pool[idx].label, status, diag.get("estimate"),
            diag.get("lower_bound"), diag.get("upper_bound"),
            exact_scores[idx] if exact_scores else None, exact_max,
            shots_added, total_shots, words_measured, total_circuits,
            diag.get("active_candidates", len(candidates)), energy=energy,
            layer_labels=layer_labels))

    rel = None
    if E0 is not None:
        rel = abs(energy - E0) / max(abs(E0), 1e-12)
    return AdaptResult(
        labels=tuple(op.label for op in chosen), parameters=theta, energy=energy,
        exact_ground_energy=E0, relative_error=rel, records=tuple(records),
        total_shots=total_shots, total_circuits=total_circuits,
        support_peak=support_peak, optimizer_evaluations=optimizer_evaluations,
        stopped_reason=stopped_reason,
        metadata={"model": model.name, "pool_size": len(pool),
                  "noisy_selection": noisy},
    )
