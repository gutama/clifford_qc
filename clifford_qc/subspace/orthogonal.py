"""Exact classical action on a known orthogonal correction space.

This experimental interface stores explicit vectors. It neither prepares them
on a quantum device nor infers their overlap from measurements. Callers must
supply an orthonormal classical span and mutually orthonormal corrections
orthogonal to that *whole span*. Orthogonality to one reference eigenvector is
insufficient when the classical span contains several vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class OccupationPacket:
    pattern: int
    vector: np.ndarray


@dataclass(frozen=True)
class OrthogonalCorrectionResult:
    energies: np.ndarray
    coefficients: np.ndarray
    basis: np.ndarray
    actions: np.ndarray
    projected_hamiltonian: np.ndarray
    orthogonality_defect: float
    residual_norm: float

    @property
    def energy(self) -> float:
        return float(self.energies[0])

    @property
    def state(self) -> np.ndarray:
        return self.basis @ self.coefficients[:, 0]


def canonical_phase(vector: np.ndarray) -> np.ndarray:
    """Fix global phase by the first largest-magnitude component."""
    out = np.array(vector, dtype=complex, copy=True)
    pivot = out[int(np.argmax(np.abs(out)))]
    if abs(pivot) > 0:
        out *= np.conjugate(pivot) / abs(pivot)
    return out


def occupation_packets(words, amplitudes, excluded, external_mask: int, *,
                       tolerance: float = 1e-12) -> tuple[OccupationPacket, ...]:
    """Partition classical amplitudes into disjoint external occupation classes.

    ``excluded`` indexes the entire classical determinant span. No amplitude
    there is silently discarded: nonzero input above tolerance is an error.
    Stored packet vectors still have full input dimension; this is a bounded
    reference implementation, not a sparse or large-sector storage claim.
    """
    raw_words = np.asarray(words)
    raw_excluded = np.asarray(excluded)
    if (raw_words.ndim != 1 or raw_words.dtype.kind not in "iu"
            or raw_excluded.ndim != 1 or (raw_excluded.size
                                         and raw_excluded.dtype.kind not in "iu")):
        raise ValueError("words and excluded must be one-dimensional integer arrays")
    words = raw_words.astype(np.int64)
    excluded = raw_excluded.astype(np.int64)
    amplitudes = np.asarray(amplitudes, dtype=complex)
    if (not words.size or amplitudes.shape != words.shape
            or not np.isfinite(amplitudes).all() or np.any(words < 0)
            or np.unique(words).size != words.size):
        raise ValueError("distinct nonnegative words and finite matching amplitudes required")
    if (not np.isfinite(tolerance) or tolerance <= 0
            or isinstance(external_mask, bool) or not isinstance(external_mask, int)
            or external_mask < 0):
        raise ValueError("positive tolerance and a nonnegative integer external_mask required")
    if (np.any(excluded < 0) or np.any(excluded >= words.size)
            or np.unique(excluded).size != excluded.size):
        raise ValueError("excluded indices must be distinct and in range")
    if excluded.size and np.max(np.abs(amplitudes[excluded])) > tolerance:
        raise ValueError("corrections have support on the classical span")
    live = np.ones(words.size, bool)
    live[excluded] = False
    patterns = words & external_mask
    out = []
    for pattern in sorted(set(patterns[live].tolist())):
        vector = np.where(live & (patterns == pattern), amplitudes, 0.0)
        norm = float(np.linalg.norm(vector))
        if norm > tolerance:
            out.append(OccupationPacket(int(pattern), canonical_phase(vector / norm)))
    return tuple(out)


def solve_orthogonal_corrections(apply: Callable, classical_basis, corrections=None, *,
                                 tolerance: float = 1e-10) -> OrthogonalCorrectionResult:
    """Solve on an explicitly known orthonormal span using one H action/column.

    A non-Hermitian projected action or a nonorthogonal supplied basis is
    rejected. No truncation or whitening changes the declared space. The
    residual uses full supplied H actions, not a residual within the projection.
    """
    classical = np.array(classical_basis, dtype=complex, copy=True)
    if (not np.isfinite(tolerance) or tolerance <= 0 or classical.ndim != 2
            or not classical.shape[0] or not classical.shape[1]
            or not np.isfinite(classical).all()):
        raise ValueError("a nonempty finite classical basis and positive tolerance required")
    extra = (np.empty((classical.shape[0], 0), complex) if corrections is None
             else np.array(corrections, dtype=complex, copy=True))
    if (extra.ndim != 2 or extra.shape[0] != classical.shape[0]
            or not np.isfinite(extra).all()):
        raise ValueError("corrections must be finite columns matching the classical dimension")
    basis = np.column_stack((classical, extra))
    defect = float(np.max(np.abs(basis.conj().T @ basis - np.eye(basis.shape[1]))))
    if defect > tolerance:
        raise ValueError("corrections and classical basis must be mutually orthonormal")
    actions = []
    for column in basis.T:
        result = np.asarray(apply(column), dtype=complex)
        if result.shape != column.shape or not np.isfinite(result).all():
            raise ValueError("Hamiltonian action returned an invalid vector")
        actions.append(result.copy())
    actions = np.column_stack(actions)
    projected = basis.conj().T @ actions
    if np.max(np.abs(projected - projected.conj().T)) > tolerance * max(
            1.0, float(np.max(np.abs(projected)))):
        raise ValueError("projected Hamiltonian action must be Hermitian")
    projected = (projected + projected.conj().T) / 2
    energies, coefficients = np.linalg.eigh(projected)
    residual = (actions - energies[0] * basis) @ coefficients[:, 0]
    return OrthogonalCorrectionResult(energies, coefficients, basis, actions, projected,
                                     defect, float(np.linalg.norm(residual)))
