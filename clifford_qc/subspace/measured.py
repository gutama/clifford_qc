"""Compatibility facade for finite-shot projected-subspace APIs.

Implementations now live in focused measurement, uncertainty, and certification
modules. Existing clifford_qc.subspace.measured imports remain valid.
"""

from __future__ import annotations

from ..measurement.functionals import (
    IDENTITY_CODE,
    WordFunctional,
    coupling_functional,
    entry_functionals,
    ritz_functional,
    split_complex_coefficients,
)
from ..measurement.session import SharedMeasurement
from ..selection import EvidenceLevel
from .certification import (
    CertifiedGrowthRecord,
    CertifiedResult,
    CouplingBound,
    certify_couplings,
    run_certified_acase,
)
from .linalg import solve_projected
from .uncertainty import Interval, bootstrap_ritz, ritz_uncertainty

EXACT = EvidenceLevel.EXACT.value
ASYMPTOTIC = EvidenceLevel.ASYMPTOTIC.value
HEURISTIC = EvidenceLevel.HEURISTIC.value
FINITE_SAMPLE = EvidenceLevel.FINITE_SAMPLE.value

# Compatibility for code that used the old private helper.
_split_complex = split_complex_coefficients

__all__ = [
    "EXACT", "ASYMPTOTIC", "HEURISTIC", "FINITE_SAMPLE",
    "IDENTITY_CODE", "WordFunctional", "SharedMeasurement", "Interval",
    "entry_functionals", "ritz_functional", "coupling_functional",
    "split_complex_coefficients", "ritz_uncertainty", "bootstrap_ritz",
    "CouplingBound", "CertifiedGrowthRecord", "CertifiedResult",
    "certify_couplings", "run_certified_acase", "solve_projected",
]
