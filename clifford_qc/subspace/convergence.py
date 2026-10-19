"""Exact convergence diagnostics over a retained Ritz block (Phase 15).

An adaptive stopping reason describes the candidate pool, not convergence to
an eigenstate. These reports attach the true residual from SecondMomentBank
without changing selection or stopping. A small residual identifies proximity
to *some* eigenvalue; ground-state dominance needs separate spectral input.

The variance-resolution floor is a numerical cancellation convention, not a
finite-sample confidence bound. Unresolved or significantly negative variances
never become a successful residual assessment by clipping them to zero.

Q17's three-prefix extrapolation is exposed separately as a diagnostic. The
committed improvement evidence concerns the frozen H4 and H2O bases only;
the fit is neither variational nor a supported energy estimator.

Per-root Weinstein intervals each hold *an* eigenvalue, but overlapping ones
may hold the same one. :func:`ritz_clusters` counts instead: by Kahan's
theorem (Parlett, *The Symmetric Eigenvalue Problem*, Thm 11.5.1) the Ritz
values of an orthonormal block ``Q`` pair one-to-one with distinct eigenvalues
of ``H`` within ``||HQ - Q Theta||_2``, and ``R'R = Q'H^2Q - Theta^2`` is
exact from ``(S, H, K)``. Roots are merged until the cluster hulls are
disjoint, so each hull holds at least as many eigenvalues as it has roots.
"""

from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .contracts import check_reference
from .linalg import SubspaceResult
from .second_moment import RESOLUTION, RitzResidual, SecondMomentBank
from .second_moment_storage import SecondMomentStoragePolicy

CONVERGENCE_SCHEMA = "clifford_qc.convergence_report.v1"
EXTRAPOLATION_SCHEMA = "clifford_qc.variance_extrapolation_diagnostic.v1"
RITZ_CLUSTERS_SCHEMA = "clifford_qc.ritz_clusters.v1"
# The reference's unit-trace tolerance, applied to its purity.
PURITY_TOLERANCE = 1e-9


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class GroundStateReference:
    """Caller-supplied exact ground and first-excited energies.

    Both energies must refer to this Hamiltonian in a space containing the
    reported state. A sector spectrum is admissible only after independently
    checking that state's sector. ``source`` records that scope and provenance;
    this class does not compute a spectrum or establish sector membership.
    Projected Ritz roots are not an exact spectral reference.
    """

    ground_energy: float
    first_excited_energy: float
    source: str

    def __post_init__(self):
        ground = _finite(self.ground_energy, "ground_energy")
        first = _finite(self.first_excited_energy, "first_excited_energy")
        if first <= ground:
            raise ValueError("first_excited_energy must exceed ground_energy")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must describe the spectral reference and its scope")
        object.__setattr__(self, "ground_energy", ground)
        object.__setattr__(self, "first_excited_energy", first)


@dataclass(frozen=True)
class ConvergenceConfig:
    """Opt-in reporting controls; the tolerance has the Hamiltonian's units."""

    residual_tolerance: float
    ground_reference: GroundStateReference | None = None
    storage_policy: SecondMomentStoragePolicy | None = None

    def __post_init__(self):
        tolerance = _finite(self.residual_tolerance, "residual_tolerance")
        if tolerance < 0.0:
            raise ValueError("residual_tolerance must be nonnegative")
        object.__setattr__(self, "residual_tolerance", tolerance)
        if self.ground_reference is not None and not isinstance(
                self.ground_reference, GroundStateReference):
            raise TypeError("ground_reference must be a GroundStateReference")
        if self.storage_policy is not None and not isinstance(
                self.storage_policy, SecondMomentStoragePolicy):
            raise TypeError("storage_policy must be a SecondMomentStoragePolicy")


@dataclass(frozen=True)
class ConvergenceReport:
    """One root's residual assessment and separate ground-state assessment.

    ``residual_status`` is within_tolerance, above_tolerance, unresolved, or
    invalid. ``ground_state_status`` is not_assessed, ground_dominated,
    not_established, or inconsistent_reference. Neither is a measured
    convergence certificate. The ground-weight bound uses the Rayleigh energy,
    rather than assuming every regularized solve's eigenvalue equals it.
    """

    residual: RitzResidual
    variance_resolution: float
    residual_tolerance: float
    residual_status: str
    basis_size: int
    effective_rank: int
    condition_number: float
    energy_change: float | None
    stopped_reason: str | None
    ground_state_status: str
    ground_weight_lower_bound: float | None
    ground_reference: GroundStateReference | None

    def as_dict(self) -> dict:
        row = asdict(self)
        row["residual"] = self.residual.as_dict()
        return {"schema": CONVERGENCE_SCHEMA, "evidence": "exact", **row}


def convergence_report(result: SubspaceResult, config: ConvergenceConfig, *,
                       root: int = 0, moments: SecondMomentBank | None = None,
                       energy_change: float | None = None,
                       stopped_reason: str | None = None) -> ConvergenceReport:
    """Assess one solved Ritz state, constructing only its retained H-squared block.

    ``moments`` can be shared across roots. Automatic construction runs the
    storage preflight on ``result.indices`` with ``config.storage_policy``.
    Supplied moments retain the policy and block they were constructed with.
    """
    if not isinstance(config, ConvergenceConfig):
        raise TypeError("config must be a ConvergenceConfig")
    if result.bank is None:
        raise ValueError("convergence reporting needs a result solved on a MatrixElementBank")
    if energy_change is not None:
        energy_change = _finite(energy_change, "energy_change")
    moments = (SecondMomentBank(result.bank, result.indices,
                                storage_policy=config.storage_policy)
               if moments is None else moments)
    residual = moments.residual(result, root)
    for name in ("energy", "rayleigh_energy", "second_moment", "variance",
                 "residual_norm", "cancellation_scale"):
        _finite(getattr(residual, name), name)
    if residual.cancellation_scale < 0.0:
        raise ValueError("cancellation_scale must be nonnegative")
    floor = RESOLUTION * residual.cancellation_scale
    if residual.variance < -floor:
        status = "invalid"
    elif not residual.resolved:
        status = "unresolved"
    elif residual.residual_norm <= config.residual_tolerance:
        status = "within_tolerance"
    else:
        status = "above_tolerance"

    reference = config.ground_reference
    ground_status, weight = "not_assessed", None
    if reference is not None:
        energy = residual.rayleigh_energy
        error = energy - reference.ground_energy
        rounding = 100.0 * math.ulp(1.0) * max(1.0, abs(energy),
                                              abs(reference.ground_energy))
        if error < -rounding:
            ground_status = "inconsistent_reference"
        else:
            gap = reference.first_excited_energy - reference.ground_energy
            weight = max(0.0, min(1.0, 1.0 - max(error, 0.0) / gap))
            ground_status = "ground_dominated" if weight > 0.5 else "not_established"
    return ConvergenceReport(
        residual=residual, variance_resolution=floor,
        residual_tolerance=float(config.residual_tolerance), residual_status=status,
        basis_size=len(result.indices), effective_rank=result.effective_rank,
        condition_number=result.condition_number, energy_change=energy_change,
        stopped_reason=stopped_reason, ground_state_status=ground_status,
        ground_weight_lower_bound=weight, ground_reference=reference,
    )


@dataclass(frozen=True)
class VarianceExtrapolationDiagnostic:
    """Q17's fixed three-point fit, reported without an improvement claim."""

    extrapolable: bool
    reason: str | None
    energy: float | None = None
    slope: float | None = None
    residual_rms: float | None = None

    def as_dict(self) -> dict:
        return {"schema": EXTRAPOLATION_SCHEMA, "evidence": "exact",
                "window": 3, "variational": False, "estimator": False,
                "improvement_evidence_banks": ["h4", "h2o_cas8e6o"],
                **asdict(self)}


def variance_extrapolation_diagnostic(
        trajectory: Sequence[RitzResidual]) -> VarianceExtrapolationDiagnostic:
    """Fit E = a + b variance through the last three nested ground-root prefixes.

    The caller supplies prefix residuals of the same Hamiltonian, reference,
    and growth order. Only root zero is supported by Q17. No prefix solve or
    H-squared row is computed here. Other bases carry no improvement evidence.
    """
    if len(trajectory) < 3:
        raise ValueError("Q17 needs at least three prefix residuals")
    points = tuple(trajectory[-3:])
    if any(not isinstance(p, RitzResidual) or p.root != 0 for p in points):
        raise ValueError("Q17 needs ground-root RitzResidual values")
    for point in points:
        for name in ("energy", "variance", "cancellation_scale"):
            _finite(getattr(point, name), name)
        if point.cancellation_scale < 0.0:
            raise ValueError("cancellation_scale must be nonnegative")
    if any(b.energy > a.energy + 1e-10 for a, b in zip(points, points[1:])):
        raise ValueError("nested-prefix ground energies must be non-increasing")
    if not all(p.resolved for p in points):
        return VarianceExtrapolationDiagnostic(False, "a window variance is not resolved")
    xs = [p.variance for p in points]
    ys = [p.energy for p in points]
    if max(xs) - min(xs) <= RESOLUTION * max(p.cancellation_scale for p in points):
        return VarianceExtrapolationDiagnostic(
            False, "the window variances differ only by rounding")
    xbar, ybar = math.fsum(xs) / 3, math.fsum(ys) / 3
    slope = (math.fsum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
             / math.fsum((x - xbar) ** 2 for x in xs))
    intercept = ybar - slope * xbar
    rms = math.sqrt(math.fsum((y - intercept - slope * x) ** 2
                             for x, y in zip(xs, ys)) / 3)
    return VarianceExtrapolationDiagnostic(
        slope > 0.0, None if slope > 0.0 else "the slope is not positive",
        intercept if slope > 0.0 else None, slope, rms,
    )


@dataclass(frozen=True)
class RitzCluster:
    """Ritz roots whose hull holds at least ``count`` eigenvalues of ``H``.

    ``ritz_values`` are the eigenvalues of ``H`` on the cluster's own span,
    which equal the solve's energies to rounding. ``residual_squared`` is the
    largest eigenvalue of ``R'R``, reported signed, and ``resolution`` its
    rounding floor, ``RESOLUTION * || |Q|'|K||Q| ||_2``. ``radius`` widens the
    residual by that floor, so a one-root cluster carries its root's Weinstein
    half-width, to rounding.
    """

    roots: tuple[int, ...]
    ritz_values: tuple[float, ...]
    residual_squared: float
    resolution: float

    @property
    def count(self) -> int:
        return len(self.roots)

    @property
    def resolved(self) -> bool:
        """The block residual stands above the rounding of its own terms."""
        return self.residual_squared > self.resolution

    @property
    def residual_norm(self) -> float:
        """``||R||_2``, clipped at zero."""
        return math.sqrt(max(self.residual_squared, 0.0))

    @property
    def radius(self) -> float:
        return math.sqrt(max(self.residual_squared, 0.0) + self.resolution)

    def interval(self) -> tuple[float, float]:
        """At least ``count`` eigenvalues of ``H``, with multiplicity, lie here."""
        return min(self.ritz_values) - self.radius, max(self.ritz_values) + self.radius

    def as_dict(self) -> dict[str, Any]:
        return {"roots": list(self.roots), "count": self.count,
                "ritz_values": list(self.ritz_values),
                "residual_squared": self.residual_squared,
                "residual_norm": self.residual_norm, "resolution": self.resolution,
                "resolved": self.resolved, "radius": self.radius,
                "interval": list(self.interval())}


@dataclass(frozen=True)
class RitzClusters:
    """Disjoint cluster hulls, ascending, covering the requested roots.

    The hulls are disjoint, so their counts add: together they hold at least
    ``len(roots)`` distinct eigenvalues. A hull may hold more eigenvalues
    than its count, and a cluster with roots ``0..m-1`` need not hold the
    ``m`` lowest eigenvalues; neither statement follows from the span.
    """

    roots: tuple[int, ...]
    clusters: tuple[RitzCluster, ...]
    basis_size: int
    effective_rank: int

    def as_dict(self) -> dict[str, Any]:
        return {"schema": RITZ_CLUSTERS_SCHEMA, "evidence": "exact",
                "roots": list(self.roots), "basis_size": self.basis_size,
                "effective_rank": self.effective_rank,
                "clusters": [cluster.as_dict() for cluster in self.clusters]}


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _cluster(roots: tuple[int, ...], C: np.ndarray, S: np.ndarray, H: np.ndarray,
             K: np.ndarray, absolute_K: np.ndarray) -> RitzCluster:
    """Kahan's bound for the span of ``C``'s columns, orthonormalized there."""
    gram_values, gram_vectors = np.linalg.eigh(_hermitian(C.conj().T @ S @ C))
    if gram_values[0] <= 0.0:
        raise ValueError(f"Ritz roots {roots} are not independent on this span")
    # Löwdin orthonormalization: the bound then depends on the span alone,
    # not on how the solve happened to normalize its vectors.
    Q = C @ (gram_vectors / np.sqrt(gram_values)) @ gram_vectors.conj().T
    M = _hermitian(Q.conj().T @ H @ Q)
    gram = _hermitian(Q.conj().T @ K @ Q - M @ M)
    magnitude = np.abs(Q)
    scale = float(np.linalg.eigvalsh(_hermitian(magnitude.T @ absolute_K @ magnitude))[-1])
    return RitzCluster(
        roots=roots, ritz_values=tuple(float(v) for v in np.linalg.eigvalsh(M)),
        residual_squared=float(np.linalg.eigvalsh(gram)[-1]),
        resolution=RESOLUTION * scale)


def ritz_clusters(result: SubspaceResult, roots: Sequence[int] | None = None, *,
                  moments: SecondMomentBank | None = None,
                  storage_policy: SecondMomentStoragePolicy | None = None
                  ) -> RitzClusters:
    """Partition Ritz roots into clusters with disjoint, counting hulls.

    Each root starts alone, with its Weinstein interval. Clusters whose hulls
    meet, endpoints included, are merged and their block residual recomputed;
    a block's residual is never below any of its sub-blocks', so hulls only
    grow and the loop ends after at most ``len(roots)`` passes. ``roots``
    defaults to every root of the solve. Automatic construction preflights
    only ``result.indices``; supply ``storage_policy`` for its budget, or pass
    already preflighted ``moments``. A solve with a nonzero ``overlap_ridge``
    is refused: its vectors are Ritz vectors of the ridged metric, not of
    ``(S, H)``. So is a mixed reference: its blocks are Gram matrices of
    ``A_j rho^(1/2)`` under left multiplication by ``H``, which repeats every
    eigenvalue, so two roots could pair with copies of one eigenvalue.
    """
    if result.bank is None:
        raise ValueError("Ritz clusters need a result solved on a MatrixElementBank")
    purity = check_reference(result.bank.reference)
    if abs(purity - 1.0) > PURITY_TOLERANCE:
        raise ValueError(f"Ritz clusters count eigenvalues only over a pure reference; "
                         f"this one has purity {purity:.6g}")
    if result.resources.get("overlap_ridge", 0.0) != 0.0:
        raise ValueError("overlap_ridge changes the overlap metric, so the Ritz vectors "
                         "do not solve (S, H); cluster a truncated solve instead")
    total = len(result.energies)
    order = tuple(range(total)) if roots is None else tuple(operator.index(r) for r in roots)
    if not order:
        raise ValueError("no roots to cluster")
    if len(set(order)) != len(order):
        raise ValueError("roots must be distinct")
    if any(not 0 <= r < total for r in order):
        raise IndexError(f"roots must lie in [0, {total})")
    order = tuple(sorted(order))
    if moments is not None and storage_policy is not None:
        raise ValueError("set storage_policy when constructing the supplied moments")
    if moments is not None and not isinstance(moments, SecondMomentBank):
        raise TypeError("moments must be a SecondMomentBank")
    moments = (SecondMomentBank(result.bank, result.indices, storage_policy=storage_policy)
               if moments is None else moments)
    if moments.bank is not result.bank:
        raise ValueError("the second moments belong to a different bank")
    S, H = result.bank.matrices(result.indices)
    K = moments.matrix(result.indices)
    absolute_K = np.abs(K)
    coefficients = np.asarray(result.coefficients)

    def build(members: tuple[int, ...]) -> RitzCluster:
        cluster = _cluster(members, coefficients[:, list(members)], S, H, K, absolute_K)
        if cluster.residual_squared < -cluster.resolution:
            raise ValueError(f"roots {members} have a negative block residual "
                             f"{cluster.residual_squared:.3g} below its rounding floor; "
                             "(S, H, K) are not exact pairings of one Hermitian H")
        return cluster

    clusters = [build((r,)) for r in order]
    while True:
        clusters.sort(key=lambda cluster: cluster.interval()[0])
        groups: list[tuple[int, ...]] = []
        upper = -math.inf
        for cluster in clusters:
            lo, hi = cluster.interval()
            if groups and lo <= upper:
                groups[-1] += cluster.roots
            else:
                groups.append(cluster.roots)
            upper = max(upper, hi)
        if len(groups) == len(clusters):
            break
        clusters = [build(tuple(sorted(group))) for group in groups]
    return RitzClusters(roots=order, clusters=tuple(clusters),
                        basis_size=len(result.indices),
                        effective_rank=int(result.effective_rank))
