"""ADAPT-VQE with exact or confidence-controlled finite-shot selection.

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
study. Consequently the reported ``total_shots``/``total_circuits`` account
for the *selection* measurements only: the energy and its gradient during
parameter optimization are evaluated exactly and cost no shots here. The
measurement-efficiency claims are therefore about selection cost under an
exact optimizer, not an end-to-end hardware shot budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..ir import Parameter, Program, Rotor, adjoint_gradient
from ..backends.exact_mv import ExactMVBackend
from ..backends.finite_shot import FiniteShotBackend
from ..measurement.bank import CommutatorBank
from ..measurement.cache import WordCache, GroupedWordCache
from ..measurement.confidence import simultaneous_z_radius, candidate_radius
from ..measurement.grouping import qwc_groups
from .layering import build_layer
from .optimize import minimize_energy
from .pools import PoolOperator
from ..selection import (
    TIE_ATOL,
    TIE_RTOL,
    SelectionStatus,
    canonical_argmax,
    certification_level,
    eta_required,
    is_certified,
    is_resolved,
    resolution_kind,
)

# Compatibility aliases for the published ADAPT module API.
_is_resolved = is_resolved
_resolution_kind = resolution_kind
_certification_level = certification_level
_is_certified = is_certified
_eta_required = eta_required


def _rank_and_gap(exact_scores, idx):
    """Post-hoc strength diagnostics for one selection.

    Returns ``(rank, top_gap)`` where ``rank`` is the 1-based position of
    candidate ``idx`` in the descending ``|g|`` ordering of the same candidate
    set (1 = the true argmax; ties share the best rank) and ``top_gap`` is
    ``|g|_max - |g|_runner-up``, the separation the selector had to resolve.
    An exact symmetry tie has ``top_gap == 0``. Both are ``None`` when the
    noiseless scores were not tracked; ``rank`` is ``None`` for an abstention.
    """
    if not exact_scores:
        return None, None
    mags = sorted((abs(v) for v in exact_scores.values()), reverse=True)
    top_gap = (mags[0] - mags[1]) if len(mags) > 1 else None
    if idx is None or idx not in exact_scores:
        return None, top_gap
    g = abs(exact_scores[idx])
    # ties share the best rank: count only strictly larger magnitudes
    rank = 1 + sum(1 for v in exact_scores.values() if abs(v) > g + 1e-12)
    return rank, top_gap


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
    certification: str = "none"
    resolution: str = "none"
    certified: bool = False
    # Post-hoc diagnostics of *how strong* a resolution was, computed from the
    # noiseless gradients of the same candidate set. ``exact_rank`` is the
    # selected candidate's 1-based position in the |g| ordering (1 = argmax);
    # ``exact_top_gap`` is |g|_max - |g|_runner-up, the gap the selector had to
    # resolve. Both are None when exact scores were not tracked.
    exact_rank: int | None = None
    exact_top_gap: float | None = None
    # Strongest *multiplicative* certificate the same intervals support:
    # |g_sel| >= (1 - eta_required) * max_k |g_k| on the 1-delta event.
    eta_required: float | None = None


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
    abstentions: int = 0


class ConfidenceSelector:
    """Best-arm selection on |g_j| with simultaneous confidence bounds.

    Two resolution rules, tried in order:

    - *exact-best*: resolve ĵ = argmax |ĝ_j| once its lower bound strictly
      exceeds every rival's upper bound, certifying it is the true argmax.
    - *eps-best* (enabled by ``near_tol=eps``): resolve ĵ once its lower bound
      clears every rival's upper bound up to ``eps``,
      ``L_ĵ >= max_k U_k - eps``. On the 1-delta interval event this certifies
      ``|g_ĵ| >= max_k |g_k| - eps`` -- an (eps, delta)-PAC ε-best guarantee.
      Unlike the exact-best rule it *resolves exact symmetry ties* (gap 0):
      when several operators share the top gradient (generic at symmetric
      ansatz states), any one of them is eps-best for any eps >= 0, so the
      selector commits instead of abstaining forever. Both rules use the same
      delta-budget intervals, so eps-best costs no extra budget; it changes
      *what* is certified (an operator within eps of the best), not the
      confidence level.

    The normal path is an asymptotic approximation; the empirical-Bernstein
    path controls the (eps-)best error at ``delta`` by splitting the budget
    across rounds, the fixed candidate family declared before measurement, and
    every QWC group summed into a candidate. Normal intervals use
    dependence-safe Bonferroni by default. ``eliminate=True`` drops candidates
    whose upper bound falls below the best lower bound (successive
    elimination); an eliminated candidate had ``U_k < L_best`` and so cannot be
    the max on the valid-interval event, keeping the eps-best certificate
    intact.
    """

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
        """Return {j: (estimate, lower, upper)} on |g_j|.

        Uses the covariance-aware group radius (finite-schedule-valid when the
        'eb' bound is selected) when the cache exposes joint-group statistics
        (grouped path), and the per-word normal propagation otherwise
        (ungrouped path)."""
        # Keep the simultaneous-testing family fixed at the candidate set
        # declared before measurement.  The active set is data-dependent;
        # recycling its smaller size after elimination would require a
        # separate alpha-recycling proof.  A fixed family size is conservative
        # and makes the finite-schedule union bound valid under elimination.
        m = len(active) if family_size is None else int(family_size)
        if m < len(active):
            raise ValueError("family_size cannot be smaller than active set")
        grouped = hasattr(cache, "candidate_group_terms")
        out = {}
        for j in active:
            if grouped:
                est = cache.candidate_estimate(bank.coeffs[j])
                terms = cache.candidate_group_terms(bank.coeffs[j])
                r = (float("inf") if terms is None else
                     candidate_radius(terms, self.delta, m, bound=self.bound,
                                      method=self.method, rounds=rounds))
            else:
                est, var = bank.estimate(j, cache)
                r = simultaneous_z_radius(var, self.delta / rounds, m, self.method)
            out[j] = (est, max(0.0, abs(est) - r), abs(est) + r)
        return out

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
        if self.bound == "eb" and not getattr(allocator, "finite_schedule_valid", False):
            raise ValueError(
                "empirical-Bernstein certification requires a predeclared "
                "fixed-endpoint allocation schedule; use UniformFixed or "
                "UniformDoubling, or use bound='normal' for adaptive allocation")
        family_size = len(active)
        rounds = self._planned_rounds(allocator)
        best = None
        bounds: dict[int, tuple[float, float, float]] = {}
        for round_index in range(10 ** 6):
            plan = allocator.plan(round_index, bank, cache, active)
            if not plan:
                status = SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS
                break
            words = [w for w in bank.words_for(active) if w.code in plan]
            cache.add_batch(sampler(words, plan))

            bounds = self._bounds(bank, cache, active, rounds, family_size)
            # Same tolerance rule as the exact path. Two arms whose selection
            # observables are related by a symmetry share a word support and
            # coefficient magnitudes, so they draw the *same* estimate from
            # the shared cache and tie bitwise; nominating by strict argmax
            # then depends on iteration order. This is close to a no-op --
            # sampling noise separates genuinely different arms far above the
            # tolerance -- but it removes the order dependence where it does
            # occur, and the certificate is checked against whichever arm is
            # nominated either way.
            best = canonical_argmax(active, lambda j: abs(bounds[j][0]))
            best_lower = bounds[best][1]
            rival_upper = max((bounds[j][2] for j in active if j != best), default=0.0)

            if all(bounds[j][2] < self.threshold for j in active):
                return None, SelectionStatus.BELOW_THRESHOLD, self._diag(best, bounds, active)
            if best_lower >= self.threshold and best_lower > rival_upper:
                return best, SelectionStatus.RESOLVED_BEST, self._diag(best, bounds, active)
            # eps-best: leader clears every rival up to the tolerance eps.
            # Certifies |g_best| >= max_k |g_k| - eps and resolves exact ties.
            if (self.near_tol is not None and best_lower >= self.threshold
                    and rival_upper - best_lower <= self.near_tol):
                return best, SelectionStatus.RESOLVED_EPS_BEST, self._diag(best, bounds, active)
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
            rival_upper = max((bounds[j][2] for j in active if j != best),
                              default=0.0)
            out.update(estimate=est, lower_bound=lo, upper_bound=up,
                       rival_upper=rival_upper,
                       eta_required=_eta_required(lo, rival_upper))
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

    def __init__(self, shots: int | None, seed: int = 0, *,
                 infinite_shot: bool = False):
        """``infinite_shot=True`` uses the exact computational-basis
        probabilities in place of sampled populations (the N -> infinity
        limit), isolating whether a proxy failure is intrinsic to the
        determinant-population signal or merely shot noise."""
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
        probs = probs / probs.sum()
        if self.infinite_shot:
            # exact populations: the N -> infinity determinant-population proxy
            p_hat = {bits: p for (bits, _), p in zip(outcomes, probs) if p > 0.0}
        else:
            counts = self.rng.multinomial(self.shots, probs)
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


def ansatz_program(model, chosen: Sequence[PoolOperator]) -> Program:
    prog = Program(model.n)
    for op in model.reference.ops:
        prog.append(op)
    for k, pool_op in enumerate(chosen):
        prog.append(Rotor(pool_op.word, Parameter(f"t{k}")))
    return prog


# Historical name kept for callers that imported the private helper.
_ansatz_program = ansatz_program


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
    selector_bound = getattr(selector, "bound", None)  # 'normal' | 'eb' | None
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
    abstentions = 0
    energy = exact_backend.expectation(ansatz_program(model, []), H, ())
    stopped_reason = "operator budget reached"

    for step in range(1, max_operators + 1):
        program = ansatz_program(model, chosen)
        rho = exact_backend.state(program, theta)
        support_peak = max(support_peak, getattr(exact_backend, "support_peak", 0))
        untried = [i for i in range(len(pool)) if allow_repeats or i not in used]
        if not untried:
            stopped_reason = "pool exhausted"
            break

        cache = (GroupedWordCache(model.n) if (noisy and grouping)
                 else WordCache(model.n) if noisy else None)
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
                # Stop on the shared ``threshold`` like every other mode, so
                # the proxy score gates termination consistently (the selector
                # only reports ``idx=None`` when nothing moves populations).
                if idx is None or proxy_score < threshold:
                    idx = None
                    status = SelectionStatus.BELOW_THRESHOLD
                    diag = {"estimate": proxy_score,
                            "active_candidates": len(candidates)}
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
                best = canonical_argmax(candidates,
                                        lambda i: abs(exact_scores[i]))
                diag = {"estimate": exact_scores[best],
                        "exact_gradient": exact_scores[best],
                        "active_candidates": len(candidates)}
                if abs(exact_scores[best]) < threshold:
                    idx, status = None, SelectionStatus.BELOW_THRESHOLD
                else:
                    idx, status = best, SelectionStatus.EXACT
            else:
                if grouping:
                    # Fix the QWC grouping over the whole candidate word set for
                    # the step, so the same measurement circuits recur every
                    # round; this makes each group's samples i.i.d. and its
                    # cumulative histogram a sufficient statistic, which the
                    # covariance-aware / finite-schedule-valid variance requires.
                    fixed_groups = qwc_groups(bank.words_for(candidates))
                    sampler = lambda words, plan: backend.sample_grouped_from_state(
                        rho, fixed_groups, plan)
                else:
                    sampler = lambda words, plan: backend.sample_words_from_state(
                        rho, words, plan)
                idx, status, diag = selector.select(bank, cache, sampler, allocator,
                                                    candidates)
                shots_added = cache.total_shots  # cumulative over this step's redraws
                # distinct Pauli words whose expectation was estimated: the shared
                # word set. The grouped cache stores joint histograms per QWC basis,
                # not per word, so count the words directly from the bank.
                words_measured = (len(bank.words_for(candidates)) if grouping
                                  else cache.unique_words())

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
        elif (status is SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS
              and not accept_ambiguous):
            stopped_reason = "selection not resolved at budget (strict abstention)"
        # Strict mode abstains only on an exhausted ambiguous budget. An
        # exact-best or eps-best resolution is a certificate, so strict accepts
        # it; the eps-best rule is what lets a strict run make progress through
        # symmetric (tied-gradient) states instead of stalling.
        strict_abstain = (status is SelectionStatus.BUDGET_EXHAUSTED_AMBIGUOUS
                          and not accept_ambiguous)
        if strict_abstain:
            abstentions += 1
        sel_rank, top_gap = _rank_and_gap(exact_scores,
                                          None if strict_abstain else idx)
        if idx is None or strict_abstain:
            records.append(SelectionRecord(
                step, None, status, diag.get("estimate"), diag.get("lower_bound"),
                diag.get("upper_bound"), diag.get("exact_gradient"), exact_max,
                shots_added, total_shots, words_measured, total_circuits,
                diag.get("active_candidates", len(candidates)),
                certification=_certification_level(status, selector_bound),
                resolution=_resolution_kind(status),
                certified=_is_certified(_certification_level(status, selector_bound)),
                exact_rank=sel_rank, exact_top_gap=top_gap,
                eta_required=diag.get("eta_required")))
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
        program = ansatz_program(model, chosen)

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
            layer_labels=layer_labels,
            certification=_certification_level(status, selector_bound),
            resolution=_resolution_kind(status),
            certified=_is_certified(_certification_level(status, selector_bound)),
            exact_rank=sel_rank, exact_top_gap=top_gap,
            eta_required=diag.get("eta_required")))

    rel = None
    if E0 is not None:
        rel = abs(energy - E0) / max(abs(E0), 1e-12)
    mode = "n/a" if not noisy else ("strict" if not accept_ambiguous else "fallback")
    return AdaptResult(
        labels=tuple(op.label for op in chosen), parameters=theta, energy=energy,
        exact_ground_energy=E0, relative_error=rel, records=tuple(records),
        total_shots=total_shots, total_circuits=total_circuits,
        support_peak=support_peak, optimizer_evaluations=optimizer_evaluations,
        stopped_reason=stopped_reason, abstentions=abstentions,
        metadata={"model": model.name, "pool_size": len(pool),
                  "noisy_selection": noisy, "certification_mode": mode,
                  "bound": selector_bound,
                  "eps_best_tol": getattr(selector, "near_tol", None),
                  "selection_rule": (
                      "eps_best" if getattr(selector, "near_tol", None) is not None
                      else "exact_best") if noisy else None,
                  "simultaneous_method": getattr(selector, "method", None),
                  "confidence_delta_per_selection": (
                      getattr(selector, "delta", None) if noisy else None),
                  "certification_scope": (
                      "per_selection_call" if selector_bound == "eb" else None),
                  "allocation_finite_schedule_valid": (
                      getattr(allocator, "finite_schedule_valid", None)
                      if noisy else None),
                  "shot_accounting": "selection_only_exact_optimizer"},
    )
