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

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..ir import Parameter, Program, Rotor, adjoint_gradient
from ..backends.exact_mv import ExactMVBackend
from ..backends.finite_shot import FiniteShotBackend
from ..measurement.bank import CommutatorBank
from ..measurement.cache import WordCache, GroupedWordCache
from ..measurement.grouping import qwc_groups
from .adapt_selectors import (
    AdaptSelectorProtocol,
    ConfidenceSelector,
    FastInspiredSelector,
    RandomSelector,
)
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


@dataclass(frozen=True)
class AdaptConfig:
    """Serializable ADAPT-VQE workflow knobs.

    Backends, selectors, and shot allocators are deliberately *not* config
    fields: they are runtime services with their own state.  Keeping them out
    makes a config safe to record in benchmark metadata and keeps the workflow
    boundary independent of any one execution backend.
    """

    max_operators: int = 10
    threshold: float = 1e-6
    allow_repeats: bool = False
    accept_ambiguous: bool = True
    optimizer_method: str = "auto"
    maxiter: int = 350
    compute_exact_reference: bool = True
    track_exact_scores: bool = True
    grouping: bool = False
    subpool_size: int | None = None
    subpool_seed: int = 0
    layer_alpha: float | None = None

    def __post_init__(self):
        if self.max_operators < 0:
            raise ValueError("max_operators must be nonnegative")
        if self.threshold < 0.0:
            raise ValueError("threshold must be nonnegative")
        if self.maxiter < 1:
            raise ValueError("maxiter must be positive")
        if self.subpool_size is not None and self.subpool_size < 1:
            raise ValueError("subpool_size must be positive")
        if self.layer_alpha is not None and not (0.0 < self.layer_alpha <= 1.0):
            raise ValueError("layer_alpha must be in (0, 1]")


@dataclass(frozen=True)
class AdaptState:
    """Immutable orchestration state between ADAPT selection rounds.

    The state stores pool indices rather than ``PoolOperator`` objects, so the
    transition is a small value object and can be replayed against the same
    declared pool.  Numerical/measurement work happens outside this object;
    :func:`adapt_step` is the pure reducer that applies one completed round.
    """

    chosen_indices: tuple[int, ...] = ()
    parameters: tuple[float, ...] = ()
    used_indices: frozenset[int] = frozenset()
    records: tuple[SelectionRecord, ...] = ()
    total_shots: int = 0
    total_circuits: int = 0
    support_peak: int = 0
    optimizer_evaluations: int = 0
    abstentions: int = 0
    energy: float = 0.0
    stopped_reason: str = "operator budget reached"


@dataclass(frozen=True)
class AdaptStepOutcome:
    """Effects computed for one round before the pure state transition."""

    selected_indices: tuple[int, ...] = ()
    parameters: tuple[float, ...] | None = None
    energy: float | None = None
    record: SelectionRecord | None = None
    total_shots: int | None = None
    total_circuits: int | None = None
    support_peak: int | None = None
    optimizer_evaluations_added: int = 0
    abstention_added: int = 0
    stopped_reason: str | None = None


def adapt_step(state: AdaptState, outcome: AdaptStepOutcome) -> AdaptState:
    """Pure ADAPT workflow transition for one already-evaluated round.

    Selection, sampling and parameter optimization are intentionally outside
    this reducer.  Given the same ``state`` and ``outcome`` it always returns
    the same next state; that is the boundary racing/successive-halving logic
    can target without becoming coupled to the VQE optimizer.
    """
    selected = tuple(int(index) for index in outcome.selected_indices)
    return AdaptState(
        chosen_indices=state.chosen_indices + selected,
        parameters=(state.parameters if outcome.parameters is None
                    else tuple(float(value) for value in outcome.parameters)),
        used_indices=state.used_indices.union(selected),
        records=(state.records if outcome.record is None
                 else state.records + (outcome.record,)),
        total_shots=(state.total_shots if outcome.total_shots is None
                     else int(outcome.total_shots)),
        total_circuits=(state.total_circuits if outcome.total_circuits is None
                        else int(outcome.total_circuits)),
        support_peak=(state.support_peak if outcome.support_peak is None
                      else max(state.support_peak, int(outcome.support_peak))),
        optimizer_evaluations=(state.optimizer_evaluations
                               + int(outcome.optimizer_evaluations_added)),
        abstentions=state.abstentions + int(outcome.abstention_added),
        energy=state.energy if outcome.energy is None else float(outcome.energy),
        stopped_reason=(state.stopped_reason if outcome.stopped_reason is None
                        else outcome.stopped_reason),
    )


def _cache_estimates(bank: CommutatorBank, cache: WordCache,
                     candidates: Sequence[int]) -> dict[int, float]:
    """Point estimates for candidates whose every word has been measured
    (candidates eliminated before round 0 completed have no valid estimate)."""
    out = {}
    for j in candidates:
        if all(cache.shots(code) > 0 for code in bank.coeffs[j]):
            out[j] = bank.estimate(j, cache)[0]
    return out


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
              selector: AdaptSelectorProtocol | None = None,
              allocator=None,
              max_operators: int = 10, threshold: float = 1e-6,
              allow_repeats: bool = False, accept_ambiguous: bool = True,
              optimizer_method: str = "auto", maxiter: int = 350,
              compute_exact_reference: bool = True,
              track_exact_scores: bool = True,
              grouping: bool = False,
              subpool_size: int | None = None, subpool_seed: int = 0,
              layer_alpha: float | None = None,
              config: AdaptConfig | None = None) -> AdaptResult:
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

    ``config`` is the structured alternative to the legacy keyword surface.
    The old keywords remain the stable facade; when a non-default legacy value
    is supplied together with ``config`` the call is rejected rather than
    silently choosing one source of truth.
    """
    legacy_config = AdaptConfig(
        max_operators=max_operators, threshold=threshold,
        allow_repeats=allow_repeats, accept_ambiguous=accept_ambiguous,
        optimizer_method=optimizer_method, maxiter=maxiter,
        compute_exact_reference=compute_exact_reference,
        track_exact_scores=track_exact_scores, grouping=grouping,
        subpool_size=subpool_size, subpool_seed=subpool_seed,
        layer_alpha=layer_alpha,
    )
    if config is None:
        config = legacy_config
    elif legacy_config != AdaptConfig():
        raise ValueError("pass either AdaptConfig or non-default legacy workflow "
                         "keywords, not both")
    max_operators = config.max_operators
    threshold = config.threshold
    allow_repeats = config.allow_repeats
    accept_ambiguous = config.accept_ambiguous
    optimizer_method = config.optimizer_method
    maxiter = config.maxiter
    compute_exact_reference = config.compute_exact_reference
    track_exact_scores = config.track_exact_scores
    grouping = config.grouping
    subpool_size = config.subpool_size
    subpool_seed = config.subpool_seed
    layer_alpha = config.layer_alpha

    # A selector advertises its scientific signal rather than coupling this
    # workflow to a concrete implementation class.  Selectors written against
    # the pre-protocol API did not carry this marker; treating those as
    # confidence selectors preserves the historical custom-selector contract.
    selection_mode = (None if selector is None else
                      getattr(selector, "selection_mode", "confidence"))
    if selection_mode not in (None, "confidence", "random", "population_proxy"):
        raise TypeError(f"unsupported selector selection_mode: {selection_mode!r}")
    random_mode = selection_mode == "random"
    fast_mode = selection_mode == "population_proxy"
    noisy = selection_mode == "confidence"
    selector_bound = getattr(selector, "bound", None)  # 'normal' | 'eb' | None
    if noisy and (backend is None or allocator is None):
        raise ValueError("finite-shot selection needs a sampling backend and an allocator")
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
        from ..dense_reference import exact_ground
        E0, _ = exact_ground(H.to_mv())

    initial_energy = exact_backend.expectation(ansatz_program(model, []), H, ())
    state = AdaptState(energy=initial_energy,
                       support_peak=getattr(exact_backend, "support_peak", 0))

    for step in range(1, max_operators + 1):
        chosen = [pool[index] for index in state.chosen_indices]
        theta = state.parameters
        used = set(state.used_indices)
        total_shots = state.total_shots
        total_circuits = state.total_circuits
        support_peak = state.support_peak
        stopped_reason = state.stopped_reason
        program = ansatz_program(model, chosen)
        rho = exact_backend.state(program, theta)
        support_peak = max(support_peak, getattr(exact_backend, "support_peak", 0))
        untried = [i for i in range(len(pool)) if allow_repeats or i not in used]
        if not untried:
            state = adapt_step(
                state, AdaptStepOutcome(stopped_reason="pool exhausted",
                                        support_peak=support_peak))
            break

        cache = None
        step_shots = 0
        step_circuits = 0
        step_word_codes: set[int] = set()
        # A below-threshold subpool triggers a redraw from the untried
        # remainder (subpool exploration); only a dead remainder stops the run.
        while True:
            if subpool_size is not None and len(untried) > subpool_size:
                candidates = sorted(subpool_rng.choice(untried, size=subpool_size,
                                                       replace=False).tolist())
            else:
                candidates = list(untried)

            # A redraw is a new, data-dependent candidate experiment.  Its
            # samples are charged, but never pooled into the next redraw's
            # confidence calculation.
            cache = (GroupedWordCache(model.n) if (noisy and grouping)
                     else WordCache(model.n) if noisy else None)

            exact_scores = None
            exact_max = None
            if track_exact_scores or (not noisy and not random_mode):
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
                step_shots += cache.total_shots
                step_circuits += cache.total_circuits
                step_word_codes.update(w.code for w in bank.words_for(candidates)
                                       if cache.shots(w.code) > 0)
                shots_added = step_shots
                # distinct Pauli words whose expectation was estimated: the shared
                # word set. The grouped cache stores joint histograms per QWC basis,
                # not per word, so count the words directly from the bank.
                words_measured = len(step_word_codes)

            if status is SelectionStatus.BELOW_THRESHOLD and subpool_size is not None:
                remaining = [c for c in untried if c not in candidates]
                if remaining:
                    untried = remaining
                    continue
            break

        if noisy:
            total_shots += step_shots
            total_circuits += step_circuits
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
        sel_rank, top_gap = _rank_and_gap(exact_scores,
                                          None if strict_abstain else idx)
        if idx is None or strict_abstain:
            record = SelectionRecord(
                step, None, status, diag.get("estimate"), diag.get("lower_bound"),
                diag.get("upper_bound"), diag.get("exact_gradient"), exact_max,
                shots_added, total_shots, words_measured, total_circuits,
                diag.get("active_candidates", len(candidates)),
                certification=_certification_level(status, selector_bound),
                resolution=_resolution_kind(status),
                certified=_is_certified(_certification_level(status, selector_bound)),
                exact_rank=sel_rank, exact_top_gap=top_gap,
                eta_required=diag.get("eta_required"))
            state = adapt_step(
                state,
                AdaptStepOutcome(
                    record=record, total_shots=total_shots,
                    total_circuits=total_circuits, support_peak=support_peak,
                    abstention_added=1 if strict_abstain else 0,
                    stopped_reason=stopped_reason,
                ),
            )
            break

        selected_indices = [idx]
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
                    selected_indices.append(j)
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
        support_peak = max(support_peak, getattr(exact_backend, "support_peak", 0))
        record = SelectionRecord(
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
            eta_required=diag.get("eta_required"))
        state = adapt_step(
            state,
            AdaptStepOutcome(
                selected_indices=tuple(selected_indices), parameters=theta,
                energy=energy, record=record, total_shots=total_shots,
                total_circuits=total_circuits, support_peak=support_peak,
                optimizer_evaluations_added=res.evaluations,
            ),
        )

    rel = None
    if E0 is not None:
        rel = abs(state.energy - E0) / max(abs(E0), 1e-12)
    mode = "n/a" if not noisy else ("strict" if not accept_ambiguous else "fallback")
    return AdaptResult(
        labels=tuple(pool[index].label for index in state.chosen_indices),
        parameters=state.parameters, energy=state.energy,
        exact_ground_energy=E0, relative_error=rel, records=state.records,
        total_shots=state.total_shots, total_circuits=state.total_circuits,
        support_peak=state.support_peak,
        optimizer_evaluations=state.optimizer_evaluations,
        stopped_reason=state.stopped_reason, abstentions=state.abstentions,
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
                  "shot_accounting": "selection_only_exact_optimizer",
                  "subpool_redraw_cache": (
                      "fresh_per_redraw_discarded_shots_charged"
                      if noisy and subpool_size is not None else None),
                  "random_termination": (
                      "operator_or_pool_budget_only" if random_mode else None)},
    )
