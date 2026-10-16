"""Shared deterministic selection and evidence contracts."""

from __future__ import annotations

from enum import Enum


TIE_RTOL = 1e-9
TIE_ATOL = 1e-12


class EvidenceLevel(str, Enum):
    EXACT = "exact"
    ASYMPTOTIC = "asymptotic"
    FINITE_SAMPLE = "finite_sample"
    HEURISTIC = "heuristic"
    NONE = "none"


class SelectionStatus(Enum):
    RESOLVED_BEST = "resolved_best"
    RESOLVED_EPS_BEST = "resolved_eps_best"
    BELOW_THRESHOLD = "below_threshold"
    BUDGET_EXHAUSTED_AMBIGUOUS = "budget_exhausted_ambiguous"
    EXACT = "exact"
    RANDOM = "random"
    FAST_PROXY = "fast_proxy"


_RESOLVED_STATUSES = frozenset(
    {SelectionStatus.RESOLVED_BEST, SelectionStatus.RESOLVED_EPS_BEST,
     SelectionStatus.EXACT})


def is_resolved(status: SelectionStatus) -> bool:
    return status in _RESOLVED_STATUSES


def resolution_kind(status: SelectionStatus) -> str:
    if status is SelectionStatus.EXACT:
        return "exact"
    if status is SelectionStatus.RESOLVED_BEST:
        return "best"
    if status is SelectionStatus.RESOLVED_EPS_BEST:
        return "eps_best"
    return "none"


def certification_level(status: SelectionStatus, bound: str | None) -> str:
    if status is SelectionStatus.EXACT:
        return EvidenceLevel.EXACT.value
    if status in (SelectionStatus.RESOLVED_BEST,
                  SelectionStatus.RESOLVED_EPS_BEST):
        return (EvidenceLevel.FINITE_SAMPLE.value
                if bound == "eb" else EvidenceLevel.ASYMPTOTIC.value)
    return EvidenceLevel.NONE.value


def is_certified(level: str) -> bool:
    return level == EvidenceLevel.FINITE_SAMPLE.value


def eta_required(best_lower, rival_upper):
    if rival_upper is None or rival_upper <= 0.0:
        return 0.0 if best_lower is not None and best_lower > 0.0 else None
    return float(min(1.0, max(
        0.0, (rival_upper - best_lower) / rival_upper)))


def canonical_argmax(candidates, magnitude, rtol=TIE_RTOL, atol=TIE_ATOL):
    """Deterministic argmax with canonical resolution of numerical ties."""
    best = max(magnitude(i) for i in candidates)
    tol = atol + rtol * abs(best)
    return min(i for i in candidates if magnitude(i) >= best - tol)
