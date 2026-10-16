"""Shared grouped-measurement sessions for projected subspaces."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..ir import PauliWord
from ..selection import EvidenceLevel
from .cache import GroupedWordCache
from .functionals import WordFunctional, entry_functionals
from .grouping import qwc_groups

if TYPE_CHECKING:
    from ..subspace.linalg import SubspaceResult
    from ..subspace.projection import MatrixElementBank

# Kept equal to the canonical linalg defaults without importing the higher
# layer while this lower-level measurement module is initialized.
DEFAULT_TAU_S = 1e-10
DEFAULT_MAX_CONDITION = 1e12
DEFAULT_NORM_FLOOR = 1e-14

HEURISTIC = EvidenceLevel.HEURISTIC.value

def select_scored_rank(solved):
    """The scored rank a confidence sweep returns: least score, least rank.

    ``solved`` is ``[(score, result, sigma, rank), ...]``. Factored out of
    :meth:`SharedMeasurement.solve_selected_rank` because the tie rule is the
    part worth testing on its own: exact ties in the score are unreachable from
    noisy data but entirely reachable from exact matrices, where interlacing
    leaves a decoupled mode's rank at the same energy as its predecessor's.

    Ties resolve to the *smaller* rank. Among equal scores that basis is never
    worse -- same value, fewer noise-carrying directions, better conditioning --
    and it is what ``gamma -> 0+`` already converges to, since ``sigma_hat``
    grows with the rank. Iterating in ascending rank and keeping a strict
    ``<`` would give the same answer; the explicit key states the rule instead
    of leaving it to the loop order.
    """
    if not solved:
        raise ValueError("no attainable rank produced a solvable pencil")
    return min(solved, key=lambda entry: (entry[0], entry[3]))


class SharedMeasurement:
    """One QWC-grouped measurement of a bank subspace's whole word universe (4A).

    Every entry of ``(S, H)`` is reconstructed from the same shots, which is the
    measurement-sharing proposition made concrete: a word appearing in many
    element operators is paid for once. The grouping is fixed at construction so
    the same circuits recur every batch, which is what makes each group's
    cumulative histogram a sufficient statistic -- and what the
    finite-sample bounds require.

    ``pooling`` is handed to the caches this session creates.  ``'shots'`` reads
    every word from every setting whose histogram contains it rather than from
    the one setting the partition assigned it to -- the same circuits, the same
    shots, a lower-variance reconstruction (see :class:`GroupedWordCache`).  It
    is off by default because committed records were produced under the
    single-assignment estimator.
    """

    def __init__(self, bank: MatrixElementBank, indices: Sequence[int] | None = None,
                 *, groups: Sequence[Sequence[PauliWord]] | None = None,
                 pooling: str = "assigned", coefficient_storage: str = "object"):
        if pooling not in ("assigned", "shots"):
            raise ValueError("pooling must be 'assigned' or 'shots'")
        if coefficient_storage not in ("object", "packed"):
            raise ValueError("coefficient_storage must be object or packed")
        self.coefficient_storage = coefficient_storage
        self.pooling = pooling
        self.bank = bank
        self.indices = bank.resolve(indices)
        self._pairs = {(i, j): entry_functionals(bank, i, j, storage=coefficient_storage)
                       for a, i in enumerate(self.indices)
                       for j in self.indices[a:]}
        codes = sorted({code
                        for pair in self._pairs.values()
                        for part in pair for f in part
                        for code in f.coefficients})
        self.words = tuple(PauliWord(bank.n, code) for code in codes)
        # A caller may supply a wider grouping -- one covering a superset of
        # these words -- so that a sub-block and the full universe are read out
        # of the *same* circuits and the same cache.
        self.groups = qwc_groups(list(self.words)) if groups is None else [
            list(group) for group in groups]
        covered = {w.code for group in self.groups for w in group}
        if not covered.issuperset(codes):
            raise ValueError("supplied groups do not cover this subspace's words")

    def diagonal_functional(self, index: int) -> WordFunctional:
        """``S_ii`` as a functional -- the candidate norm, when it must be measured."""
        return self._pairs[(index, index)][0][0]

    def matrix_functionals(self, *, overlap: bool = True,
                           hamiltonian: bool = True) -> tuple[WordFunctional, ...]:
        """Independent real functionals defining the measured ``(S, H)``.

        The upper triangle is structurally mirrored by :meth:`matrices`, so
        returning its real and imaginary parts once gives an allocation policy
        a nonduplicated target family.  Deterministic entries are omitted:
        they neither need shots nor contribute estimator variance.
        """
        if not overlap and not hamiltonian:
            return ()
        out = []
        for pair in self._pairs.values():
            selected = pair if overlap and hamiltonian else (
                pair[0:1] if overlap else pair[1:2])
            out.extend(functional for parts in selected for functional in parts
                       if not functional.is_deterministic)
        return tuple(out)

    @property
    def n(self) -> int:
        return self.bank.n

    def new_cache(self) -> GroupedWordCache:
        return GroupedWordCache(self.n, pooling=self.pooling)

    def measure(self, backend, shots_per_group: int,
                cache: GroupedWordCache | None = None) -> GroupedWordCache:
        """Sample every group ``shots_per_group`` times into (a fresh) cache.

        A predeclared uniform allocation: the schedule's endpoints are fixed
        before any outcome is seen, which is the condition the
        empirical-Bernstein bounds are valid under.
        """
        if shots_per_group <= 0:
            raise ValueError("shots_per_group must be positive")
        if not self.groups:
            raise ValueError("nothing to measure: the universe is empty")
        cache = self.new_cache() if cache is None else cache
        batch = backend.sample_grouped_from_state(self.bank.reference, self.groups,
                                                 shots_per_group)
        cache.add_batch(batch)
        return cache

    def measure_plan(self, backend, plan: dict[int, int],
                     cache: GroupedWordCache | None = None) -> GroupedWordCache:
        """Measure a word-keyed plan whose counts are constant within groups.

        Allocation policies return word keys for backend compatibility, while
        the physical resource is one shot count per group.  Rejecting
        inconsistent counts here prevents a caller from accidentally relying
        on the backend's historical ``max(group member counts)`` fallback and
        misreporting the spent budget.
        """
        if not plan:
            raise ValueError("measurement plan is empty")
        for group in self.groups:
            counts = {int(plan.get(word.code, 0)) for word in group}
            if len(counts) > 1:
                raise ValueError("all words in a measurement group need the same shots")
            if next(iter(counts), 0) < 0:
                raise ValueError("shots must be nonnegative")
        cache = self.new_cache() if cache is None else cache
        batch = backend.sample_grouped_from_state(self.bank.reference, self.groups, plan)
        cache.add_batch(batch)
        return cache

    def _assemble(self, evaluate) -> tuple[np.ndarray, np.ndarray]:
        m = len(self.indices)
        S = np.zeros((m, m), dtype=complex)
        Hm = np.zeros((m, m), dtype=complex)
        for a, i in enumerate(self.indices):
            for b, j in enumerate(self.indices[a:], start=a):
                (s_re, s_im), (h_re, h_im) = self._pairs[(i, j)]
                S[a, b] = complex(evaluate(s_re), evaluate(s_im))
                Hm[a, b] = complex(evaluate(h_re), evaluate(h_im))
        # Same structural Hermiticity as the exact path: the lower triangle is
        # the conjugate by definition, never an independent estimate that would
        # have to be symmetrized (and would double the noise on the diagonal).
        for b in range(m):
            S[b, b] = S[b, b].real
            Hm[b, b] = Hm[b, b].real
            for a in range(b):
                S[b, a] = S[a, b].conjugate()
                Hm[b, a] = Hm[a, b].conjugate()
        return S, Hm

    def matrices(self, cache: GroupedWordCache) -> tuple[np.ndarray, np.ndarray]:
        """``(S_hat, H_hat)`` from measured word means."""
        return self._assemble(lambda f: f.estimate(cache))

    def exact_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """The infinite-shot limit of the same reconstruction (4A acceptance)."""
        rho = self.bank.reference
        return self._assemble(lambda f: f.exact(rho))

    def calibrated_overlap_floor(self, cache: GroupedWordCache, *,
                                 delta: float = 0.05, bound: str = "normal",
                                 method: str = "bonferroni", safety: float = 1.0,
                                 strategy: str = "modewise",
                                 policy: str = "per_mode",
                                 norm_floor: float = DEFAULT_NORM_FLOOR,
                                 overlap: np.ndarray | None = None) -> dict:
        """Estimate a shot-budget-dependent cutoff for normalized overlap modes.

        ``strategy='modewise'`` projects the measured overlap uncertainty onto
        each observed normalized overlap eigenvector and reports that mode's
        own simultaneous covariance-aware radius.  The more conservative
        ``'entrywise'`` strategy takes the spectral norm of a matrix of
        simultaneous entrywise radii, which is a single number by construction.

        ``policy='per_mode'`` keeps the modewise radii separate, so the solver
        can ask of each mode whether it stands above *its own* shot noise.
        ``policy='uniform'`` collapses them to their maximum and applies that
        to every mode; it is strictly more aggressive, because one badly
        resolved direction then sets the cutoff for well resolved ones, and it
        is retained only as a comparator.

        Normalization and the live-generator mask follow
        :func:`~clifford_qc.subspace.linalg.solve_projected` exactly -- same
        ``norm_floor``, same dropped rows, same ``canonical_eigh`` ordering --
        because a floor derived from a different matrix than the one being
        truncated is not a floor for it.  A generator that annihilates the
        reference is dropped rather than divided by ``norm_floor``, which
        would otherwise inflate its radius without bound.

        This is a noise-aware regularization *diagnostic*, not a variational or
        coverage certificate: the same data select the retained rank and solve
        the pencil, and adaptive shot schedules make fixed-endpoint empirical
        Bernstein semantics inapplicable unless an independent frozen batch is
        used.
        """
        from ..subspace.linalg import canonical_eigh

        if safety <= 0.0 or not math.isfinite(safety):
            raise ValueError("overlap safety factor must be positive and finite")
        if strategy not in ("modewise", "entrywise"):
            raise ValueError("overlap strategy must be 'modewise' or 'entrywise'")
        if policy not in ("per_mode", "uniform"):
            raise ValueError("overlap policy must be 'per_mode' or 'uniform'")
        overlap = self.matrices(cache)[0] if overlap is None else np.asarray(overlap)

        # Mirror solve_projected: drop annihilated directions, normalize the
        # survivors by their measured norms.
        norms = np.sqrt(np.clip(overlap.diagonal().real, 0.0, None))
        live = norms > norm_floor
        if not live.any():
            raise ValueError("every generator annihilates the reference state")
        scaling = np.where(live, norms, 1.0)
        index = np.flatnonzero(live)
        normalized = ((overlap / scaling[:, None]) / scaling[None, :])[
            np.ix_(index, index)]
        live_size = int(live.sum())

        if strategy == "entrywise":
            stochastic = sum(
                not functional.is_deterministic
                for pair in self._pairs.values() for functional in pair[0])
            family = max(1, stochastic)
            radii = np.zeros((live_size, live_size), dtype=float)
            for a, position in enumerate(index):
                i = self.indices[position]
                for b, other in enumerate(index[a:], start=a):
                    j = self.indices[other]
                    real, imag = self._pairs[(i, j)][0]
                    real_radius = real.radius(
                        cache, delta, family, bound=bound, method=method)
                    imag_radius = imag.radius(
                        cache, delta, family, bound=bound, method=method)
                    radius = math.hypot(real_radius, imag_radius)
                    radii[a, b] = radii[b, a] = radius
            live_scaling = scaling[index]
            normalized_radii = ((radii / live_scaling[:, None])
                                / live_scaling[None, :])
            mode_radii = [float(np.linalg.norm(normalized_radii, ord=2))] * live_size
        else:
            # Treat the measured normalization as fixed and propagate the full
            # grouped covariance of u^* S_bar u for every observed mode.  The
            # resulting delta-method threshold is intentionally labelled
            # heuristic below: both u and the diagonal scaling are data-derived.
            _, vectors = canonical_eigh(normalized)
            family = live_size
            mode_radii = []
            for column in range(live_size):
                # Full-length generator-coordinate amplitudes, zero on the
                # dropped rows, so the pair loop below needs no special case.
                amplitudes = np.zeros(len(self.indices), dtype=complex)
                amplitudes[index] = vectors[:, column] / scaling[index]
                coefficients: dict[int, float] = {}

                def add(functional: WordFunctional, weight: float) -> None:
                    if weight == 0.0:
                        return
                    for code, value in functional.coefficients.items():
                        coefficients[code] = coefficients.get(code, 0.0) \
                            + weight * value

                for a, i in enumerate(self.indices):
                    for b, j in enumerate(self.indices[a:], start=a):
                        weight = np.conjugate(amplitudes[a]) * amplitudes[b]
                        if weight == 0.0:
                            continue
                        real, imag = self._pairs[(i, j)][0]
                        if a == b:
                            add(real, float(weight.real))
                        else:
                            add(real, float(2.0 * weight.real))
                            add(imag, float(-2.0 * weight.imag))
                functional = WordFunctional({
                    code: value for code, value in coefficients.items()
                    if value != 0.0
                })
                mode_radii.append(float(functional.radius(
                    cache, delta, family, bound=bound, method=method)))

        raw = float(max(mode_radii, default=0.0))
        if not all(math.isfinite(radius) for radius in mode_radii):
            raise ValueError("overlap noise floor is unresolved: some words are unmeasured")
        thresholds = tuple(safety * radius for radius in mode_radii)
        if policy == "uniform":
            thresholds = tuple(safety * raw for _ in mode_radii)
        return {
            "threshold": safety * raw,
            "mode_thresholds": thresholds,
            "raw_noise_radius": raw,
            "family_size": family,
            "live_generators": live_size,
            "policy": policy,
            "delta": float(delta),
            "bound": bound,
            "method": method,
            "strategy": strategy,
            "safety": float(safety),
            "evidence": HEURISTIC,
        }

    def solve(self, cache: GroupedWordCache, *, tau_s: float = DEFAULT_TAU_S,
              rel_tau: float = 0.0, max_condition: float = DEFAULT_MAX_CONDITION,
              norm_floor: float = DEFAULT_NORM_FLOOR,
              calibrate_overlap: bool = False,
              overlap_delta: float = 0.05,
              overlap_bound: str = "normal",
              overlap_method: str = "bonferroni",
              overlap_strategy: str = "modewise",
              overlap_policy: str = "per_mode",
              overlap_safety: float = 1.0,
              overlap_regularizer: str = "truncate") -> SubspaceResult:
        """Thresholded GEP on the measured matrices.

        The result reports the same conditioning metrics as the exact path plus
        ``overlap_negative_modes``: with estimated entries ``S_hat`` is no longer
        positive semidefinite, and thresholding those modes away is a repair
        whose effect on the variational bound is not established (Q3).

        ``calibrate_overlap`` replaces the fixed numerical cutoff with the
        shot-calibrated one of :meth:`calibrated_overlap_floor`.  It defaults
        to off: the calibrated rule trades tail risk for bias, and which of
        those a caller wants is not a default the solver can pick.

        ``overlap_regularizer`` chooses what the calibrated radii are *used
        for*: ``'truncate'`` makes each radius a retention threshold (the mode
        is kept whole or dropped whole), ``'ridge'`` adds it to that mode's
        overlap eigenvalue instead, damping the mode by
        ``lambda/(lambda + r)`` rather than deciding its fate.  Truncation is
        a rank decision taken on noisy data; the ridge replaces it with a
        continuous one, at the cost of solving a perturbed metric.  It has no
        effect without ``calibrate_overlap``.
        """
        from ..subspace.linalg import solve_projected

        if overlap_regularizer not in ("truncate", "ridge"):
            raise ValueError("overlap_regularizer must be 'truncate' or 'ridge'")
        S, Hm = self.matrices(cache)
        calibration = None
        noise_floor: float | tuple[float, ...] = 0.0
        ridge: float | tuple[float, ...] = 0.0
        if calibrate_overlap:
            calibration = self.calibrated_overlap_floor(
                cache, delta=overlap_delta, bound=overlap_bound,
                method=overlap_method, strategy=overlap_strategy,
                safety=overlap_safety, policy=overlap_policy,
                norm_floor=norm_floor, overlap=S)
            if overlap_regularizer == "ridge":
                ridge = calibration["mode_thresholds"]
            else:
                noise_floor = calibration["mode_thresholds"]
        resources = dict(self.bank.resources(self.indices))
        resources.update({
            "evidence": HEURISTIC,
            "shots": cache.total_shots,
            "circuits": cache.total_circuits,
            "qwc_groups": len(self.groups),
            "word_pooling": cache.pooling,
            "measured_words": len(self.words),
            "shots_per_measured_word": (cache.total_shots / len(self.words)
                                        if self.words else 0.0),
            "overlap_threshold_policy": (
                f"shot_calibrated_{overlap_policy}" if calibrate_overlap
                else "fixed"),
            "overlap_regularizer": (overlap_regularizer if calibrate_overlap
                                    else "truncate"),
            "overlap_calibration": calibration,
        })
        return solve_projected(S, Hm,
                               [self.bank.generator(i).label for i in self.indices],
                               tau_s=tau_s, rel_tau=rel_tau,
                               max_condition=max_condition,
                               overlap_noise_floor=noise_floor,
                               overlap_ridge=ridge,
                               norm_floor=norm_floor,
                               resources=resources).with_bank(self.bank, self.indices)

    def solve_selected_rank(self, cache: GroupedWordCache, *, gamma: float = 2.0,
                            tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
                            max_condition: float = DEFAULT_MAX_CONDITION,
                            norm_floor: float = DEFAULT_NORM_FLOOR
                            ) -> SubspaceResult:
        """Solve at the rank minimizing ``E_hat(k) + gamma * sigma_hat(k)``.

        The overlap-threshold rules decide the retained rank by asking whether a
        mode stands above its own shot noise -- a question about ``S`` alone.
        The quantity actually at risk is the Ritz value, and a mode's danger is
        how much noise it lets into *that*: a barely resolved direction is
        harmless if the Hamiltonian hardly couples to it and ruinous if it does.
        This rule asks that question directly.  Each attainable rank is solved,
        its Ritz value paired with the delta-method standard error of
        :func:`ritz_functional` at that solution, and the rank minimizing the
        upper confidence bound is returned.

        Adding a mode lowers ``E_hat`` by Cauchy interlacing and raises
        ``sigma_hat`` once the mode is noise-dominated, so the score has an
        interior minimum and ``gamma`` prices one against the other: as
        ``gamma`` grows the rule retreats toward rank one, and as it falls the
        rule approaches "take the lowest energy".  ``gamma = 2`` is a two-sigma
        price and the default.

        **Ties go to the smaller rank.**  Interlacing makes ``E_hat``
        non-increasing in the rank but not strictly decreasing: a mode that
        decouples from the current ground Ritz vector adds nothing, and on a
        symmetric model with exact matrices whole runs of ranks share one energy
        (the four-qubit XXZ bank ties ranks 2, 3 and 4). Among equal scores the
        smallest rank is returned, which is both the parsimonious choice -- same
        value, fewer noise-carrying directions, better conditioning -- and the
        continuous one, since ``sigma_hat`` rises with the rank, so the
        ``gamma -> 0+`` limit already selects the smallest tied rank.  ``gamma =
        0`` therefore means "lowest energy, smallest rank achieving it", not
        "full rank"; it is the limit of its own neighbourhood rather than a
        special case grafted onto it.

        This is a selection rule, not a certificate.  The same cache supplies
        the matrices, the rank, and the error bar, so the reported ``sigma`` is
        an asymptotic quantity conditioned on a data-dependent choice, and the
        returned value is not a variational bound on ``E_0``.

        Each candidate rank costs one solve and one covariance-vector product
        over the subspace's word universe, so the sweep is ``M`` times the cost
        of a single :meth:`solve` -- cheap against the shots that produced the
        cache, but not free.
        """
        from ..subspace.linalg import canonical_eigh, solve_projected
        from .functionals import ritz_functional

        if gamma < 0.0 or not math.isfinite(gamma):
            raise ValueError("gamma must be nonnegative and finite")
        S, Hm = self.matrices(cache)
        labels = [self.bank.generator(i).label for i in self.indices]

        norms = np.sqrt(np.clip(S.diagonal().real, 0.0, None))
        live = norms > norm_floor
        if not live.any():
            raise ValueError("every generator annihilates the reference state")
        scaling = np.where(live, norms, 1.0)
        index = np.flatnonzero(live)
        normalized = ((S / scaling[:, None]) / scaling[None, :])[np.ix_(index, index)]
        # Ascending, the order solve_projected's per-mode vectors are given in.
        values, _ = canonical_eigh(normalized)

        solved = []
        scores = []
        for rank in range(1, values.size + 1):
            # Realize exactly this rank by floor-ing every mode below the cut at
            # its own value: retention is a strict ``>``, so a mode floored at
            # itself is dropped whatever the gaps or degeneracies are.
            floor = np.zeros(values.size)
            drop = values.size - rank
            floor[:drop] = np.clip(values[:drop], 0.0, None)
            try:
                candidate = solve_projected(
                    S, Hm, labels, tau_s=tau_s, rel_tau=rel_tau,
                    max_condition=max_condition, overlap_noise_floor=floor,
                    norm_floor=norm_floor).with_bank(self.bank, self.indices)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if candidate.effective_rank != rank:
                # A deterministic rule (tau_s, the condition cap) already
                # forbids this rank; scoring it would misreport what was solved.
                continue
            functional = ritz_functional(self.bank, self.indices,
                                         candidate.ritz_vector(0),
                                         candidate.ground_energy)
            variance = functional.variance(cache)
            sigma = math.sqrt(max(0.0, variance))
            score = candidate.ground_energy + gamma * sigma
            scores.append({"rank": rank, "energy": candidate.ground_energy,
                           "sigma": sigma, "score": score})
            solved.append((score, candidate, sigma, rank))
        if not solved:
            raise ValueError("no attainable rank produced a solvable pencil")
        best = select_scored_rank(solved)

        score, result, sigma, rank = best
        resources = dict(result.resources)
        resources.update({
            "evidence": HEURISTIC,
            "shots": cache.total_shots,
            "circuits": cache.total_circuits,
            "qwc_groups": len(self.groups),
            "word_pooling": cache.pooling,
            "measured_words": len(self.words),
            "shots_per_measured_word": (cache.total_shots / len(self.words)
                                        if self.words else 0.0),
            "overlap_threshold_policy": "confidence_selected_rank",
            "rank_selection_gamma": float(gamma),
            "rank_selection_score": float(score),
            "rank_selection_sigma": float(sigma),
            "rank_selection_rank": int(rank),
            "rank_selection_scores": tuple(
                (entry["rank"], entry["energy"], entry["sigma"], entry["score"])
                for entry in scores),
        })
        return replace(result, resources=resources)


# --------------------------------------------------------------- 4B: uncertainty

__all__ = ["SharedMeasurement", "select_scored_rank"]
