"""Uncertainty diagnostics for noisy projected eigenproblems."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..backends.protocol import GroupSample, MeasurementBatch
from ..measurement.cache import GroupedWordCache
from ..measurement.functionals import ritz_functional
from ..measurement.session import SharedMeasurement
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
    functional = ritz_functional(shared.bank, result.indices,
                                 result.coefficients[:, root], result.energies[root])
    variance = functional.variance(cache)
    radius = functional.radius(cache, delta, family, bound="normal")
    energy = result.energies[root]
    return Interval(estimate=energy, lower=energy - radius, upper=energy + radius,
                    delta=delta, evidence=ASYMPTOTIC,
                    std_error=math.sqrt(max(0.0, variance)))


def bootstrap_ritz(shared: SharedMeasurement, cache: GroupedWordCache, *,
                   root: int = 0, replicates: int = 200, seed: int = 0,
                   delta: float = 0.05, solver_kwargs: dict | None = None) -> Interval:
    """Grouped bootstrap on the Ritz value -- ``heuristic``, and the cross-check
    the delta method needs.

    Each group's outcome histogram is resampled multinomially at its own shot
    count (groups are independent circuits, so this is the right resampling
    unit), then the whole pipeline -- reconstruct, threshold, diagonalize -- is
    rerun. It therefore captures what the linearization drops: the movement of
    the retained eigenspace itself. It captures nothing about coverage under a
    different true state, which is why it is heuristic.
    """
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")

    rng = np.random.default_rng(seed)
    states = cache.group_states()
    samples: list[float] = []
    kwargs = solver_kwargs or {}
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
        replica = shared.new_cache()
        replica.add_batch(MeasurementBatch(n=shared.n, shots={}, plus_counts={},
                                           circuits=len(groups), groups=tuple(groups),
                                           state_key=cache.state_key))
        samples.append(shared.solve(replica, **kwargs).energies[root])
    values = np.array(samples)
    lower, upper = np.quantile(values, [delta / 2.0, 1.0 - delta / 2.0])
    return Interval(estimate=float(np.mean(values)), lower=float(lower),
                    upper=float(upper), delta=delta, evidence=HEURISTIC,
                    std_error=float(np.std(values, ddof=1)) if replicates > 1 else None)

__all__ = ["Interval", "ritz_uncertainty", "bootstrap_ritz"]
