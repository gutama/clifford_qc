"""Finite-shot uncertainty for the nonlinear projected-response pipeline.

An exact response line is already nonlinear in the measured primitives:

1. reconstruct the shared ``(S, H)`` matrices;
2. threshold and solve the generalized eigenproblem;
3. reconstruct the projected observable ``Q_sub``;
4. form a transition amplitude, square its modulus, subtract Ritz energies,
   and optionally divide by that gap for a susceptibility.

Linear confidence bounds for fixed Pauli-word functionals therefore do not
become response-line certificates by substitution.  This module supplies the
missing honest layer: one grouped measurement covers ``S``, ``H``, and
``Q_sub``, while a grouped nonparametric bootstrap resamples the joint outcome
histograms and reruns the *entire* nonlinear pipeline.

The output evidence label is always ``heuristic``.  Percentile intervals do
not have a finite-sample coverage proof, individual root intervals require
isolated ordered roots, and replicas whose thresholded rank changes are
reported and excluded.  The API deliberately has no ``finite_sample`` option;
adding one requires a confidence set for the full correlated matrix pencil and
observable, plus a root-identification argument.

One consequence of that exclusion is worth stating rather than leaving to be
inferred from the failure counts: the intervals are **conditional on the
surviving replicas**.  Discarding the resamples on which the rank moved or two
roots collided removes exactly the draws that would have widened the band, so
the interval is narrowest in the ill-conditioned regime this diagnostic exists
to expose.  ``BootstrapResponse.acceptance_rate`` reports the conditioning, and
should be quoted next to any interval taken from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..backends.protocol import GroupSample, MeasurementBatch
from ..measurement.cache import GroupedWordCache
from ..measurement.functionals import split_complex_coefficients
from ..measurement.grouping import qwc_groups
from ..ir import PauliWord
from .elements import MatrixElementBank
from .measured import HEURISTIC, Interval, SharedMeasurement
from .response import ResponseLine, broaden_response, static_susceptibility
from .contracts import as_multivector
from .solver import (DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S,
                     SubspaceResult)


@dataclass(frozen=True)
class MeasuredResponseSpectrum:
    """One measured projected spectrum before uncertainty resampling."""

    result: SubspaceResult
    lines: tuple[ResponseLine, ...]
    susceptibility: float
    observable_matrix: np.ndarray
    overlap_matrix: np.ndarray
    resources: dict


@dataclass(frozen=True)
class ResponseLineUncertainty:
    """Heuristic intervals for one root-resolved line."""

    line: ResponseLine
    excitation_energy: Interval
    weight: Interval

    @property
    def certified(self) -> bool:
        return False


@dataclass(frozen=True)
class ReplicaOutcome:
    """What one bootstrap replica did, and the number that decided it.

    The census exists to make a disagreement between two runs diagnosable.
    Acceptance requires ``effective_rank`` to match the point estimate's, and
    the solver sets that rank by testing each overlap mode against its own
    cutoff, ``value > max(tau_s, rel_tau * lambda_max, lambda_max /
    max_condition, noise_floor)``.  That cutoff is per-mode whenever a
    calibrated floor is in play, so neither the smallest eigenvalue nor its
    sign decides the rank in general: a positive minimum eigenvalue below its
    own cutoff is still dropped.

    The decisive quantity is therefore recorded directly.
    ``rank_decision_margin`` is the signed distance from its cutoff of the mode
    sitting closest to one -- the mode that flips the retained count first --
    and ``controlling_mode`` indexes it in descending-eigenvalue order, with
    ``overlap_threshold`` the cutoff that actually applied to it.  A run that
    accepts a different number of replicas can then be diffed against a
    committed record replica by replica: the movers are visible, and each
    mover's margin says whether the decision was marginal or the pipeline
    moved wholesale.  Without this, a census disagreement is only ever a pair
    of totals.

    ``outcome`` is ``"accepted"`` or the rejection reason -- ``"rank"``,
    ``"root_collision"``, or ``"solver"``.  A replica whose solve raised has
    no pencil to report, so every numeric field is ``None``.
    """

    index: int
    outcome: str
    overlap_eigenvalue_min: float | None
    effective_rank: int | None
    overlap_threshold: float | None = None
    rank_decision_margin: float | None = None
    controlling_mode: int | None = None


def _rank_decision_fingerprint(result: SubspaceResult) -> dict:
    """Locate the overlap mode that decides this solve's retained rank.

    Retention is ``value > cutoff`` mode by mode, so the mode nearest its own
    cutoff is the one whose side would flip first, taking the retained count
    -- and therefore the rank gate -- with it.  Its signed distance is the
    margin worth recording; a scalar threshold cannot express this once the
    cutoff varies per mode.
    """
    resources = result.resources
    margins = resources.get("overlap_decision_margin_per_mode")
    thresholds = resources.get("overlap_threshold_per_mode")
    minimum = resources.get("overlap_eigenvalue_min")
    fingerprint = {
        "overlap_eigenvalue_min": None if minimum is None else float(minimum),
        "overlap_threshold": None,
        "rank_decision_margin": None,
        "controlling_mode": None,
    }
    if not margins:
        return fingerprint
    controlling = min(range(len(margins)), key=lambda i: abs(margins[i]))
    fingerprint["controlling_mode"] = int(controlling)
    fingerprint["rank_decision_margin"] = float(margins[controlling])
    if thresholds is not None:
        fingerprint["overlap_threshold"] = float(thresholds[controlling])
    return fingerprint


@dataclass(frozen=True)
class BootstrapResponse:
    """Whole-pipeline grouped-bootstrap result.

    ``broadened_*`` are populated only when a frequency grid and a broadening
    are supplied together.  All interval arrays are pointwise percentile
    intervals, not a simultaneous band.

    **Every interval here is conditional on the replicas that survived.**
    Replicas whose thresholded rank moved, whose roots collided, or whose solve
    failed are excluded, so the reported spread describes the pipeline *given
    that it behaved.*  That conditioning is not neutral: it narrows the interval
    exactly where the pipeline is least stable, because the replicas that would
    have widened it are the ones being dropped.  ``acceptance_rate`` is the
    number to read beside the interval -- a band from 200/200 replicas and a
    band from 120/200 are not the same claim, and the second is the one to
    distrust.
    """

    spectrum: MeasuredResponseSpectrum
    lines: tuple[ResponseLineUncertainty, ...]
    susceptibility: Interval
    evidence: str
    delta: float
    replicates_requested: int
    replicates_succeeded: int
    rank_failures: int
    root_collision_failures: int
    solver_failures: int
    replica_census: tuple[ReplicaOutcome, ...] = ()
    frequencies: np.ndarray | None = None
    broadened_estimate: np.ndarray | None = None
    broadened_lower: np.ndarray | None = None
    broadened_upper: np.ndarray | None = None

    @property
    def certified(self) -> bool:
        """A grouped bootstrap is a diagnostic, never a certificate."""
        return False

    @property
    def acceptance_rate(self) -> float:
        """Fraction of replicas the intervals are conditioned on.

        Below 1.0 the pipeline failed to reproduce itself on some resamples,
        and the interval is correspondingly optimistic.
        """
        if self.replicates_requested <= 0:
            return 0.0
        return self.replicates_succeeded / self.replicates_requested


def _projected_lines(result: SubspaceResult, observable_matrix: np.ndarray,
                     overlap_matrix: np.ndarray, *, initial_state: int = 0,
                     include_elastic: bool = False, min_weight: float = 0.0,
                     energy_tol: float = 1e-12) -> tuple[ResponseLine, ...]:
    roots = len(result.energies)
    if not 0 <= initial_state < roots:
        raise IndexError(f"initial_state must be in [0, {roots})")
    if min_weight < 0.0:
        raise ValueError("min_weight must be non-negative")
    ci = result.coefficients[:, initial_state]
    norm_i = float((ci.conj() @ overlap_matrix @ ci).real)
    if norm_i <= DEFAULT_NORM_FLOOR:
        raise ValueError("initial Ritz root has unresolved measured norm")
    lines: list[ResponseLine] = []
    for final_state, final_energy in enumerate(result.energies):
        gap = float(final_energy - result.energies[initial_state])
        if final_state == initial_state and not include_elastic:
            continue
        if gap < -energy_tol:
            continue
        cf = result.coefficients[:, final_state]
        norm_f = float((cf.conj() @ overlap_matrix @ cf).real)
        if norm_f <= DEFAULT_NORM_FLOOR:
            raise ValueError("final Ritz root has unresolved measured norm")
        amplitude = complex(cf.conj() @ observable_matrix @ ci) / math.sqrt(
            norm_f * norm_i)
        weight = float(abs(amplitude) ** 2)
        if weight + 1e-15 < min_weight:
            continue
        lines.append(ResponseLine(
            initial_state=initial_state,
            final_state=final_state,
            excitation_energy=max(0.0, gap),
            weight=max(0.0, weight),
            amplitude=amplitude,
        ))
    return tuple(lines)


class ResponseMeasurement(SharedMeasurement):
    """One grouped measurement plan for ``S``, ``H``, and a Hermitian ``Q_sub``."""

    def __init__(self, bank: MatrixElementBank, observable,
                 indices: Sequence[int] | None = None, *,
                 label: str = "response_observable"):
        super().__init__(bank, indices)
        self.observable = as_multivector(observable)
        if self.observable.n != bank.n:
            raise ValueError("observable lives in a different algebra")
        if not self.observable.is_hermitian():
            raise ValueError("linear response requires a Hermitian observable")
        self.observable_label = str(label)
        if not self.observable_label:
            raise ValueError("label must be non-empty")

        self._observable_pairs = {}
        observable_codes: set[int] = set()
        for a, i in enumerate(self.indices):
            for j in self.indices[a:]:
                operator = bank.observable_operator(
                    self.observable, i, j, label=self.observable_label)
                observable_codes.update(operator.terms)
                self._observable_pairs[(i, j)] = split_complex_coefficients(
                    operator.terms)
        self._observable_codes = frozenset(observable_codes)

        codes = {
            code
            for pair in self._pairs.values()
            for part in pair
            for functional in part
            for code in functional.coefficients
        }
        codes.update(
            code
            for pair in self._observable_pairs.values()
            for functional in pair
            for code in functional.coefficients
        )
        self.words = tuple(PauliWord(bank.n, code) for code in sorted(codes))
        self.groups = qwc_groups(list(self.words))

    def _assemble_observable(self, evaluate) -> np.ndarray:
        m = len(self.indices)
        matrix = np.zeros((m, m), dtype=complex)
        for a, i in enumerate(self.indices):
            for b, j in enumerate(self.indices[a:], start=a):
                real, imag = self._observable_pairs[(i, j)]
                matrix[a, b] = complex(evaluate(real), evaluate(imag))
        for b in range(m):
            matrix[b, b] = matrix[b, b].real
            for a in range(b):
                matrix[b, a] = matrix[a, b].conjugate()
        return matrix

    def observable_matrix(self, cache: GroupedWordCache) -> np.ndarray:
        """Reconstruct ``Q_sub`` from the same cache as ``(S, H)``."""
        return self._assemble_observable(lambda functional: functional.estimate(cache))

    def exact_observable_matrix(self) -> np.ndarray:
        """Infinite-shot reconstruction through the measured coefficient maps."""
        return self._assemble_observable(
            lambda functional: functional.exact(self.bank.reference))

    def spectrum(self, cache: GroupedWordCache, *, initial_state: int = 0,
                 include_elastic: bool = False, min_weight: float = 0.0,
                 gap_floor: float = 1e-12, tau_s: float = DEFAULT_TAU_S,
                 rel_tau: float = 0.0,
                 max_condition: float = DEFAULT_MAX_CONDITION,
                 norm_floor: float = DEFAULT_NORM_FLOOR,
                 calibrate_overlap: bool = False,
                 overlap_delta: float = 0.05,
                 overlap_bound: str = "normal",
                 overlap_method: str = "bonferroni",
                 overlap_strategy: str = "modewise",
                 overlap_safety: float = 1.0,
                 ) -> MeasuredResponseSpectrum:
        """Run the measured matrix-pencil and response pipeline once."""
        result = self.solve(
            cache, tau_s=tau_s, rel_tau=rel_tau,
            max_condition=max_condition, norm_floor=norm_floor,
            calibrate_overlap=calibrate_overlap,
            overlap_delta=overlap_delta, overlap_bound=overlap_bound,
            overlap_method=overlap_method, overlap_strategy=overlap_strategy,
            overlap_safety=overlap_safety)
        overlap, _ = self.matrices(cache)
        observable = self.observable_matrix(cache)
        lines = _projected_lines(
            result, observable, overlap,
            initial_state=initial_state,
            include_elastic=include_elastic,
            min_weight=min_weight,
        )
        susceptibility = static_susceptibility(lines, gap_floor=gap_floor)
        resources = dict(result.resources)
        energy_words = self.bank.word_set(self.indices)
        resources.update({
            "observable": self.observable_label,
            "response_word_universe": len(self.words),
            "response_qwc_groups": len(self.groups),
            "observable_words": len(self._observable_codes),
            "observable_additional_words": len(
                self._observable_codes - energy_words),
            "response_evidence": HEURISTIC,
        })
        return MeasuredResponseSpectrum(
            result=result,
            lines=lines,
            susceptibility=susceptibility,
            observable_matrix=observable,
            overlap_matrix=overlap,
            resources=resources,
        )

    def exact_spectrum(self, *, initial_state: int = 0,
                       include_elastic: bool = False,
                       min_weight: float = 0.0,
                       gap_floor: float = 1e-12
                       ) -> MeasuredResponseSpectrum:
        """Infinite-shot limit of the identical response reconstruction."""
        result = self.bank.solve(self.indices)
        overlap, _ = self.exact_matrices()
        observable = self.exact_observable_matrix()
        lines = _projected_lines(
            result, observable, overlap,
            initial_state=initial_state,
            include_elastic=include_elastic,
            min_weight=min_weight,
        )
        return MeasuredResponseSpectrum(
            result=result,
            lines=lines,
            susceptibility=static_susceptibility(lines, gap_floor=gap_floor),
            observable_matrix=observable,
            overlap_matrix=overlap,
            resources={
                "evidence": "exact",
                "response_word_universe": len(self.words),
                "response_qwc_groups": len(self.groups),
            },
        )


def _resampled_cache(measurement: ResponseMeasurement, cache: GroupedWordCache,
                     rng: np.random.Generator) -> GroupedWordCache:
    groups = []
    for group in cache.group_states():
        keys = list(group["hist"])
        counts = np.asarray([group["hist"][key] for key in keys], dtype=float)
        probabilities = counts / counts.sum()
        drawn = rng.multinomial(group["shots"], probabilities)
        groups.append(GroupSample(
            support=group["support"],
            basis=group["basis"],
            hist={key: int(count) for key, count in zip(keys, drawn) if count},
            shots=group["shots"],
            word_codes=group["word_codes"],
            setting_key=group.get("setting_key"),
            readouts=group.get("readouts", {}),
        ))
    replica = measurement.new_cache()
    replica.add_batch(MeasurementBatch(
        n=measurement.n,
        shots={},
        plus_counts={},
        circuits=len(groups),
        groups=tuple(groups),
        state_key=cache.state_key,
    ))
    return replica


def _minimum_root_gap(energies: Sequence[float]) -> float:
    values = np.asarray(energies, dtype=float)
    if values.size < 2:
        return float("inf")
    return float(np.min(np.diff(values)))


def _percentile_interval(values: Sequence[float], estimate: float,
                         delta: float) -> Interval:
    array = np.asarray(values, dtype=float)
    lower, upper = np.quantile(array, [delta / 2.0, 1.0 - delta / 2.0])
    return Interval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        delta=delta,
        evidence=HEURISTIC,
        std_error=float(np.std(array, ddof=1)) if array.size > 1 else None,
    )


def bootstrap_response(measurement: ResponseMeasurement,
                       cache: GroupedWordCache, *, initial_state: int = 0,
                       replicates: int = 200, seed: int = 0,
                       delta: float = 0.05, min_weight: float = 0.0,
                       gap_floor: float = 1e-12,
                       root_gap_tolerance: float = 1e-8,
                       minimum_success_fraction: float = 0.8,
                       frequencies=None, broadening: float | None = None,
                       solver_kwargs: dict | None = None) -> BootstrapResponse:
    """Grouped percentile bootstrap of the complete measured response pipeline.

    Root-resolved lines are matched by ordered Ritz-root index.  This is
    meaningful only for isolated roots, so both the point estimate and every
    accepted replica must have adjacent gaps above ``root_gap_tolerance``.
    Rank-changing, colliding, or failed replicas are counted explicitly.

    The result is a *heuristic* uncertainty diagnostic.  ``delta`` chooses
    percentile endpoints; it is not a proven miscoverage probability.
    """
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if root_gap_tolerance < 0.0 or not math.isfinite(root_gap_tolerance):
        raise ValueError("root_gap_tolerance must be non-negative and finite")
    if not 0.0 < minimum_success_fraction <= 1.0:
        raise ValueError("minimum_success_fraction must be in (0, 1]")
    if (frequencies is None) != (broadening is None):
        raise ValueError("frequencies and broadening must be supplied together")

    kwargs = dict(solver_kwargs or {})
    point = measurement.spectrum(
        cache,
        initial_state=initial_state,
        min_weight=min_weight,
        gap_floor=gap_floor,
        **kwargs,
    )
    if _minimum_root_gap(point.result.energies) <= root_gap_tolerance:
        raise ValueError(
            "root-resolved response is ill-defined for near-degenerate Ritz roots")
    target_states = tuple(line.final_state for line in point.lines)
    if not target_states:
        raise ValueError("the measured spectrum has no response lines")

    omega = None if frequencies is None else np.asarray(frequencies, dtype=float)
    if omega is not None and not np.all(np.isfinite(omega)):
        raise ValueError("frequencies must be finite")
    point_broadened = None
    if omega is not None:
        point_broadened = broaden_response(point.lines, omega, float(broadening))

    gap_samples = {state: [] for state in target_states}
    weight_samples = {state: [] for state in target_states}
    susceptibility_samples: list[float] = []
    broadened_samples: list[np.ndarray] = []
    rank_failures = root_failures = solver_failures = 0
    census: list[ReplicaOutcome] = []
    rng = np.random.default_rng(seed)

    for index in range(replicates):
        replica = _resampled_cache(measurement, cache, rng)
        try:
            spectrum = measurement.spectrum(
                replica,
                initial_state=initial_state,
                min_weight=min_weight,
                gap_floor=gap_floor,
                **kwargs,
            )
        except (ValueError, np.linalg.LinAlgError, IndexError):
            solver_failures += 1
            census.append(ReplicaOutcome(index, "solver", None, None))
            continue
        # The fingerprint of the rank gate below, recorded for every replica
        # including the accepted ones: a future run that accepts a different
        # count is diffed against this, not against a total.
        fingerprint = _rank_decision_fingerprint(spectrum.result)
        rank = int(spectrum.result.effective_rank)

        def outcome(label: str) -> ReplicaOutcome:
            return ReplicaOutcome(index, label, effective_rank=rank,
                                  **fingerprint)

        if rank != point.result.effective_rank:
            rank_failures += 1
            census.append(outcome("rank"))
            continue
        if _minimum_root_gap(spectrum.result.energies) <= root_gap_tolerance:
            root_failures += 1
            census.append(outcome("root_collision"))
            continue
        by_state = {line.final_state: line for line in spectrum.lines}
        if any(state not in by_state for state in target_states):
            root_failures += 1
            census.append(outcome("root_collision"))
            continue
        census.append(outcome("accepted"))
        for state in target_states:
            gap_samples[state].append(by_state[state].excitation_energy)
            weight_samples[state].append(by_state[state].weight)
        susceptibility_samples.append(spectrum.susceptibility)
        if omega is not None:
            broadened_samples.append(
                broaden_response(spectrum.lines, omega, float(broadening)))

    succeeded = len(susceptibility_samples)
    required = math.ceil(minimum_success_fraction * replicates)
    if succeeded < required:
        raise RuntimeError(
            f"only {succeeded}/{replicates} bootstrap replicas were usable; "
            f"{required} required (rank={rank_failures}, "
            f"root={root_failures}, solver={solver_failures})")

    point_by_state = {line.final_state: line for line in point.lines}
    line_intervals = []
    for state in target_states:
        line = point_by_state[state]
        line_intervals.append(ResponseLineUncertainty(
            line=line,
            excitation_energy=_percentile_interval(
                gap_samples[state], line.excitation_energy, delta),
            weight=_percentile_interval(
                weight_samples[state], line.weight, delta),
        ))
    susceptibility = _percentile_interval(
        susceptibility_samples, point.susceptibility, delta)

    lower = upper = None
    if omega is not None:
        samples = np.asarray(broadened_samples)
        lower, upper = np.quantile(
            samples, [delta / 2.0, 1.0 - delta / 2.0], axis=0)

    return BootstrapResponse(
        spectrum=point,
        lines=tuple(line_intervals),
        susceptibility=susceptibility,
        evidence=HEURISTIC,
        delta=delta,
        replicates_requested=replicates,
        replicates_succeeded=succeeded,
        rank_failures=rank_failures,
        root_collision_failures=root_failures,
        solver_failures=solver_failures,
        replica_census=tuple(census),
        frequencies=omega,
        broadened_estimate=point_broadened,
        broadened_lower=lower,
        broadened_upper=upper,
    )


__all__ = [
    "BootstrapResponse",
    "MeasuredResponseSpectrum",
    "ReplicaOutcome",
    "ResponseLineUncertainty",
    "ResponseMeasurement",
    "bootstrap_response",
]
