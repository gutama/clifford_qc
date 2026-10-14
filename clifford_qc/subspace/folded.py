"""Phase 15: folded-spectrum roots from the second-moment block.

For a real target ``w`` and the span of ``A_j|psi>``, the folded operator
``(H - w)^2`` projects to

    F(w) = K - 2 w H + w^2 S,

with ``K`` the :class:`~.second_moment.SecondMomentBank` block. The roots of
``F(w) c = mu S c`` are the stationary values of
``||(H - w)|Psi>||^2 / <Psi|Psi>`` on the span, ascending. The lowest root is
the state in the span that is closest, by residual, to an eigenstate with
eigenvalue ``w``: the folded-spectrum method's projected form. No new
measurement and no new row are needed, only the three blocks the bank already
pairs, so a target costs one small generalized eigensolve.

Every root's normalized state carries two inclusion statements that need no
spectral reference, because ``||(H - w)Psi||^2`` is a weighted mean of
``(E_k - w)^2`` over the eigenvalues ``E_k`` the state touches:

* **Target interval.** Some eigenvalue lies in ``[w - sqrt(mu), w + sqrt(mu)]``.
* **Weinstein interval.** With ``E = <H>`` and ``sigma^2 = <H^2> - E^2 =
  mu - (E - w)^2``, some eigenvalue lies in ``[E - sigma, E + sigma]``. Since
  ``sigma <= sqrt(mu)``, this interval is never wider.

They hold for interior roots as well as the ground root. Both bounds are
widened by their resolution floors (``RESOLUTION`` times the cancellation
scale), so a value that is rounding does not produce a false tight interval.

Two uses follow. With ``w`` set to a Ritz value ``theta_k``, the lowest root is
the *refined Ritz vector* (Jia, 1997). It minimizes ``||(H - theta_k) x||`` over
the span, so its target bound never exceeds the Ritz vector's own residual
``sigma_k``. Setting ``w`` to the lowest root's own energy and solving again
never raises ``mu``, since ``mu_next <= sigma^2 <= mu``. The iteration therefore
settles on a state whose energy equals its target. There ``mu = sigma^2``, and
the state is a stationary point of the variance on the span.

The values are exact pairings with the reference, with the same storage
licence as ``SecondMomentBank``: banks outside the five Phase 15 bases need
their own preflight. Nothing here estimates ``K`` from samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .linalg import SubspaceResult, solve_projected
from .second_moment import RESOLUTION, SecondMomentBank

FOLDED_ROOT_SCHEMA = "clifford_qc.folded_spectrum_root.v1"
VARIANCE_MINIMIZATION_SCHEMA = "clifford_qc.variance_minimization.v1"
# The variance-minimizing iteration stops once the target moves by less than
# this, relative to max(1, |target|): the fixed point E = w to near rounding.
DEFAULT_TARGET_TOLERANCE = 1e-12


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class FoldedRoot:
    """One folded-spectrum root on a span, with its inclusion intervals.

    ``folded_value`` is ``||(H - target)|Psi>||^2 / <Psi|Psi>`` and
    ``variance`` is ``<H^2> - <H>^2`` at the Rayleigh ``energy``. Both are
    differences of large terms, reported signed. The two scales are the
    magnitudes they cancel: ``|c|'(|K| + 2|w||H| + w^2|S|)|c|`` and
    ``|c|'|K||c|``, each per unit norm. ``coefficients`` are in the original
    generator coordinates, normalized ``c'Sc = 1``.
    """

    root: int
    target: float
    folded_value: float
    energy: float
    second_moment: float
    variance: float
    folded_scale: float
    variance_scale: float
    coefficients: np.ndarray

    @property
    def resolved(self) -> bool:
        """The folded value stands above the rounding of its own terms."""
        return self.folded_value > RESOLUTION * self.folded_scale

    @property
    def variance_resolved(self) -> bool:
        return self.variance > RESOLUTION * self.variance_scale

    @property
    def residual_norm(self) -> float:
        """``sigma``, clipped at zero."""
        return math.sqrt(max(self.variance, 0.0))

    @property
    def target_bound(self) -> float:
        """Half-width of the target interval, widened by its resolution floor."""
        return math.sqrt(max(self.folded_value, 0.0) + RESOLUTION * self.folded_scale)

    @property
    def weinstein_bound(self) -> float:
        """Half-width of the Weinstein interval, widened by its resolution floor."""
        return math.sqrt(max(self.variance, 0.0) + RESOLUTION * self.variance_scale)

    def target_interval(self) -> tuple[float, float]:
        """An eigenvalue of ``H`` lies here, for this root's state."""
        return self.target - self.target_bound, self.target + self.target_bound

    def weinstein_interval(self) -> tuple[float, float]:
        """An eigenvalue of ``H`` lies here too; it is never the wider of the two."""
        return self.energy - self.weinstein_bound, self.energy + self.weinstein_bound

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FOLDED_ROOT_SCHEMA, "evidence": "exact",
            "root": self.root, "target": self.target, "folded_value": self.folded_value,
            "energy": self.energy, "second_moment": self.second_moment,
            "variance": self.variance, "residual_norm": self.residual_norm,
            "folded_scale": self.folded_scale, "variance_scale": self.variance_scale,
            "resolved": self.resolved, "variance_resolved": self.variance_resolved,
            "target_interval": list(self.target_interval()),
            "weinstein_interval": list(self.weinstein_interval()),
        }


@dataclass(frozen=True)
class FoldedSpectrumResult:
    """Every folded root of one target, ascending in folded value."""

    target: float
    roots: tuple[FoldedRoot, ...]
    effective_rank: int
    condition_number: float
    indices: tuple[int, ...]
    basis_labels: tuple[str, ...]

    @property
    def lowest(self) -> FoldedRoot:
        return self.roots[0]


@dataclass(frozen=True)
class VarianceMinimization:
    """The self-consistent folded iteration from one starting target.

    ``trace`` holds ``(target, folded_value, energy, variance, folded_scale)``
    per step. The folded value is non-increasing in exact arithmetic;
    ``monotone`` says whether it was, to each step's resolution floor.
    """

    start: float
    root: FoldedRoot
    trace: tuple[tuple[float, float, float, float, float], ...]
    converged: bool
    monotone: bool

    @property
    def iterations(self) -> int:
        return len(self.trace)

    def as_dict(self) -> dict[str, Any]:
        return {"schema": VARIANCE_MINIMIZATION_SCHEMA, "evidence": "exact",
                "start": self.start, "iterations": self.iterations,
                "converged": self.converged, "monotone": self.monotone,
                "trace": [list(step) for step in self.trace], "root": self.root.as_dict()}


class FoldedSpectrum:
    """Folded-spectrum roots over one block of a second-moment bank.

    The ``S``, ``H`` and ``K`` blocks are formed once; each target then costs
    one solve of the folded pencil. ``solve_kwargs`` reach ``solve_projected``
    unchanged, so pass the thresholds of the Ritz solve being refined if they
    were not the defaults: the retained subspace is decided on ``S`` alone,
    exactly as for the Ritz roots.
    """

    def __init__(self, moments: SecondMomentBank, indices: Sequence[int] | None = None,
                 **solve_kwargs):
        if not isinstance(moments, SecondMomentBank):
            raise TypeError("FoldedSpectrum needs a SecondMomentBank")
        bank = moments.bank
        self._moments = moments
        self._indices = tuple(bank.resolve(indices))
        self._labels = tuple(bank.labels[i] for i in self._indices)
        S, H = bank.matrices(self._indices)
        self._S, self._H = S, H
        self._K = moments.matrix(self._indices)
        self._absolute = (np.abs(S), np.abs(H), np.abs(self._K))
        self._solve_kwargs = dict(solve_kwargs)

    @property
    def moments(self) -> SecondMomentBank:
        return self._moments

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    def folded_matrix(self, target: float) -> np.ndarray:
        """``F(w) = K - 2 w H + w^2 S``, Hermitian by construction."""
        w = _finite(target, "target")
        F = self._K - 2.0 * w * self._H + (w * w) * self._S
        return 0.5 * (F + F.conj().T)

    def _root(self, k: int, target: float, c: np.ndarray) -> FoldedRoot:
        S, H, K = self._S, self._H, self._K
        absS, absH, absK = self._absolute
        norm = float((c.conj() @ S @ c).real)
        if norm <= 0.0:
            raise ValueError("a folded root has no norm on this span")
        energy = float((c.conj() @ H @ c).real) / norm
        second = float((c.conj() @ K @ c).real) / norm
        folded = float((c.conj() @ self.folded_matrix(target) @ c).real) / norm
        magnitude = np.abs(c)
        w = abs(target)
        k_scale = float(magnitude @ absK @ magnitude) / norm
        folded_scale = float(magnitude @ (absK + 2.0 * w * absH + w * w * absS)
                             @ magnitude) / norm
        return FoldedRoot(
            root=k, target=float(target), folded_value=folded, energy=energy,
            second_moment=second, variance=second - energy * energy,
            folded_scale=folded_scale, variance_scale=k_scale,
            coefficients=c / math.sqrt(norm))

    def roots(self, target: float, count: int | None = None) -> FoldedSpectrumResult:
        """The folded roots of ``target``, ascending; ``count`` keeps the lowest."""
        w = _finite(target, "target")
        solved = solve_projected(self._S, self.folded_matrix(w), self._labels,
                                 **self._solve_kwargs)
        total = len(solved.energies)
        keep = total if count is None else int(count)
        if not 1 <= keep <= total:
            raise ValueError(f"count must lie in [1, {total}]")
        roots = tuple(self._root(k, w, solved.coefficients[:, k]) for k in range(keep))
        return FoldedSpectrumResult(
            target=w, roots=roots, effective_rank=int(solved.effective_rank),
            condition_number=float(solved.condition_number), indices=self._indices,
            basis_labels=self._labels)

    def refined_ritz(self, ritz_value: float) -> FoldedRoot:
        """The refined Ritz root of ``ritz_value``: the lowest folded root there."""
        return self.roots(ritz_value, count=1).lowest

    def minimize_variance(self, start: float, *, max_iterations: int = 100,
                          tolerance: float = DEFAULT_TARGET_TOLERANCE
                          ) -> VarianceMinimization:
        """Iterate ``w <- E`` of the lowest folded root, starting from ``start``.

        Stops when the target moves by at most ``tolerance * max(1, |w|)`` or
        after ``max_iterations`` solves, whichever is first.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        tolerance = _finite(tolerance, "tolerance")
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        w = _finite(start, "start")
        trace = []
        monotone, converged = True, False
        root = None
        for _ in range(max_iterations):
            root = self.roots(w, count=1).lowest
            if trace:
                floor = RESOLUTION * root.folded_scale
                monotone &= root.folded_value <= trace[-1][1] + floor
            trace.append((root.target, root.folded_value, root.energy, root.variance,
                          root.folded_scale))
            if abs(root.energy - w) <= tolerance * max(1.0, abs(w)):
                converged = True
                break
            w = root.energy
        return VarianceMinimization(start=float(start), root=root, trace=tuple(trace),
                                    converged=converged, monotone=bool(monotone))


def folded_spectrum_roots(moments: SecondMomentBank, target: float, *,
                          indices: Sequence[int] | None = None, count: int | None = None,
                          **solve_kwargs) -> FoldedSpectrumResult:
    """The folded roots of ``target`` over one block; see :class:`FoldedSpectrum`."""
    return FoldedSpectrum(moments, indices, **solve_kwargs).roots(target, count)


def refined_ritz_root(result: SubspaceResult, k: int = 0, *,
                      moments: SecondMomentBank | None = None, **solve_kwargs) -> FoldedRoot:
    """The refined Ritz vector of root ``k`` of a bank solve.

    Its folded value at ``theta_k`` never exceeds the Ritz vector's own
    variance there, since the Ritz vector lies in the same retained span.
    """
    if result.bank is None:
        raise ValueError("refinement needs a result solved on a MatrixElementBank")
    if not 0 <= k < len(result.energies):
        raise IndexError(f"root {k} out of range")
    moments = SecondMomentBank(result.bank) if moments is None else moments
    if moments.bank is not result.bank:
        raise ValueError("the second moments belong to a different bank")
    spectrum = FoldedSpectrum(moments, result.indices, **solve_kwargs)
    return spectrum.refined_ritz(float(result.energies[k]))
