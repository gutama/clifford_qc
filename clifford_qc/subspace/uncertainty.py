"""Uncertainty diagnostics for noisy projected eigenproblems."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..backends.protocol import GroupSample, MeasurementBatch
from ..measurement.cache import GroupedWordCache
from ..measurement.functionals import ritz_functional
from ..measurement.session import SharedMeasurement, grouped_cache_fingerprint
from ..selection import EvidenceLevel
from .linalg import SubspaceResult

ASYMPTOTIC = EvidenceLevel.ASYMPTOTIC.value
HEURISTIC = EvidenceLevel.HEURISTIC.value
FINITE_SAMPLE = EvidenceLevel.FINITE_SAMPLE.value

@dataclass(frozen=True)
class Interval:
    """An estimate with a two-sided interval and an explicit evidence label."""

    estimate: float
    lower: float
    upper: float
    delta: float
    evidence: str
    std_error: float | None = None

    @property
    def certified(self) -> bool:
        """Only a finite-sample interval is a certificate."""
        return self.evidence == FINITE_SAMPLE


def _point_estimator(shared: SharedMeasurement, cache: GroupedWordCache,
                     result: SubspaceResult) -> dict:
    if result.estimator_parameters is None:
        raise ValueError("uncertainty requires a point result from SharedMeasurement.solve; "
                         "low-level results have no measured estimator provenance")
    if result.bank is not shared.bank or result.indices != shared.indices:
        raise ValueError("point result belongs to a different measurement session")
    if (result.resources.get("shots") != cache.total_shots or
            result.resources.get("circuits") != cache.total_circuits):
        raise ValueError("point result and cache have different measurement budgets")
    if result.measurement_fingerprint != grouped_cache_fingerprint(cache):
        raise ValueError("point result and cache contain different joint outcomes")
    return dict(result.estimator_parameters)


def ritz_uncertainty(shared: SharedMeasurement, cache: GroupedWordCache,
                     result: SubspaceResult, *, root: int = 0, delta: float = 0.05,
                     family: int = 1) -> Interval:
    """Delta-method interval on a measured Ritz value -- ``asymptotic``, not certified.

    Two approximations stack, and both are why the label is what it is: the
    Gaussian radius is asymptotic in the shot count, and the linearization
    ignores the second-order response of the Ritz value (including the fact
    that the retained eigenspace itself moved with the data). It is a useful
    error bar and not a guarantee.
    """
    parameters = _point_estimator(shared, cache, result)
    if parameters.get("overlap_repair", "none") != "none":
        raise ValueError("the delta method does not include nonlinear overlap repair; "
                         "use bootstrap_ritz(shared, cache, result=result)")
    if parameters.get("pencil_angle") is not None:
        raise ValueError("the delta method does not include rotated thresholding; "
                         "use bootstrap_ritz(shared, cache, result=result)")
    if parameters.get("ratio_error_budget") is not None:
        raise ValueError("the delta method at fixed coefficients omits the constrained-ratio "
                         "search; use bootstrap_ritz(shared, cache, result=result)")
    functional = ritz_functional(shared.bank, result.indices,
                                 result.coefficients[:, root], result.energies[root])
    variance = functional.variance(cache)
    radius = functional.radius(cache, delta, family, bound="normal")
    energy = result.energies[root]
    return Interval(estimate=energy, lower=energy - radius, upper=energy + radius,
                    delta=delta, evidence=ASYMPTOTIC,
                    std_error=math.sqrt(max(0.0, variance)))


def bootstrap_ritz_samples(shared: SharedMeasurement, cache: GroupedWordCache, *,
                           root: int = 0, replicates: int = 200, seed: int = 0,
                           solver_kwargs: dict | None = None,
                           result: SubspaceResult | None = None) -> np.ndarray:
    """The resampled Ritz values behind :func:`bootstrap_ritz`, in draw order.

    Each group's outcome histogram is resampled multinomially at its own shot
    count, then the whole pipeline is rerun under the policy :func:`bootstrap_ritz`
    describes. A caller that needs several intervals, or more than one interval
    construction, from one set of draws starts here. A failed replica raises,
    including ``ValueError`` if it no longer retains the requested root.
    """
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    kwargs = dict(solver_kwargs or {})
    if result is not None:
        recorded = _point_estimator(shared, cache, result)
        for key, value in kwargs.items():
            if key not in recorded or value != recorded[key]:
                raise ValueError(f"solver_kwargs conflict with the point estimator: {key}")
        kwargs = recorded
    rng = np.random.default_rng(seed)
    states = cache.group_states()
    samples: list[float] = []
    for _ in range(replicates):
        groups = []
        for group in states:
            keys = list(group["hist"])
            counts = np.array([group["hist"][k] for k in keys], dtype=float)
            probs = counts / counts.sum()
            drawn = rng.multinomial(group["shots"], probs)
            groups.append(GroupSample(
                support=group["support"], basis=group["basis"],
                hist={k: int(c) for k, c in zip(keys, drawn) if c},
                shots=group["shots"], word_codes=group["word_codes"],
                setting_key=group.get("setting_key"),
                readouts=group.get("readouts", {})))
        # The reconstruction policy is part of the estimator being resampled.
        replica = GroupedWordCache(shared.n, pooling=cache.pooling)
        replica.add_batch(MeasurementBatch(n=shared.n, shots={}, plus_counts={},
                                           circuits=len(groups), groups=tuple(groups),
                                           state_key=cache.state_key))
        energies = shared.solve(replica, **kwargs).energies
        if not -len(energies) <= root < len(energies):
            raise ValueError(f"bootstrap replica {len(samples)} retained {len(energies)} "
                             f"root(s); root {root} is unavailable")
        samples.append(energies[root])
    return np.array(samples)


def bootstrap_ritz(shared: SharedMeasurement, cache: GroupedWordCache, *,
                   root: int = 0, replicates: int = 200, seed: int = 0,
                   delta: float = 0.05, solver_kwargs: dict | None = None,
                   result: SubspaceResult | None = None) -> Interval:
    """Grouped bootstrap on the Ritz value -- ``heuristic``, and the cross-check
    the delta method needs.

    Each group's outcome histogram is resampled multinomially at its own shot
    count (groups are independent circuits, so this is the right resampling
    unit), then the whole pipeline -- reconstruct, threshold, diagonalize -- is
    rerun. It therefore captures what the linearization drops: the movement of
    the retained eigenspace itself. It captures nothing about coverage under a
    different true state, which is why it is heuristic.

    With ``result``, infer the full solver policy from that measured point
    result and reject conflicting ``solver_kwargs``. Its point energy is the
    returned estimate. Without ``result``, the legacy API bootstraps only the
    explicitly supplied/default policy; it is not bound to a separate point
    estimate. Repair and fixed-angle rotation are rerun on every resampled pencil.
    Data-selected angles are unsupported. A failed replica
    raises; it is never silently removed to narrow the interval.
    """
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    values = bootstrap_ritz_samples(shared, cache, root=root, replicates=replicates,
                                    seed=seed, solver_kwargs=solver_kwargs, result=result)
    lower, upper = np.quantile(values, [delta / 2.0, 1.0 - delta / 2.0])
    estimate = float(result.energies[root]) if result is not None else float(np.mean(values))
    return Interval(estimate=estimate, lower=float(lower),
                    upper=float(upper), delta=delta, evidence=HEURISTIC,
                    std_error=float(np.std(values, ddof=1)) if replicates > 1 else None)

__all__ = ["Interval", "ritz_uncertainty", "bootstrap_ritz", "bootstrap_ritz_samples"]
