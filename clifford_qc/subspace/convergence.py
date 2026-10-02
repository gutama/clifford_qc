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
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .linalg import SubspaceResult
from .second_moment import RESOLUTION, RitzResidual, SecondMomentBank

CONVERGENCE_SCHEMA = "clifford_qc.convergence_report.v1"
EXTRAPOLATION_SCHEMA = "clifford_qc.variance_extrapolation_diagnostic.v1"


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

    def __post_init__(self):
        tolerance = _finite(self.residual_tolerance, "residual_tolerance")
        if tolerance < 0.0:
            raise ValueError("residual_tolerance must be nonnegative")
        object.__setattr__(self, "residual_tolerance", tolerance)
        if self.ground_reference is not None and not isinstance(
                self.ground_reference, GroundStateReference):
            raise TypeError("ground_reference must be a GroundStateReference")


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

    ``moments`` can be shared across roots. Banks outside the five frozen
    Phase 15 bases require their own storage preflight, as SecondMomentBank
    documents; opting in is not an automatic licence for a larger bank.
    """
    if not isinstance(config, ConvergenceConfig):
        raise TypeError("config must be a ConvergenceConfig")
    if result.bank is None:
        raise ValueError("convergence reporting needs a result solved on a MatrixElementBank")
    if energy_change is not None:
        energy_change = _finite(energy_change, "energy_change")
    moments = SecondMomentBank(result.bank) if moments is None else moments
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
