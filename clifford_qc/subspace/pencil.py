"""Opt-in post-processing of measured overlap matrices.

An A-CASE generator need not be unitary, so its overlap diagonal need not be
one. We project the *normalized* live block onto the correlation matrices,
then restore the measured diagonal. This minimizes a normalized Frobenius
distance, not the unscaled distance and not a covariance-weighted distance.
It supplies neither a variational bound nor an energy confidence interval.
"""

from __future__ import annotations

import math

import numpy as np

from .linalg import DEFAULT_NORM_FLOOR


def repair_overlap(overlap: np.ndarray, *, norm_floor: float = DEFAULT_NORM_FLOOR,
                   tolerance: float = 1e-10, max_iterations: int = 1000
                   ) -> tuple[np.ndarray, dict]:
    """Nearest-correlation repair with the original nonnegative diagonal.

    Higham's alternating projections with Dykstra's correction solve the
    Hermitian nearest-correlation problem on the normalized live block.
    A final PSD projection and diagonal congruence remove the tolerance-sized
    feasibility defect; the distance optimum is therefore numerical, to the
    requested iteration tolerance. No positive eigenvalue floor is added.

    Rows whose norm is at most ``norm_floor`` are decoupled and retain their
    diagonal; these are the rows the canonical solver drops. A negative
    diagonal cannot be preserved by a PSD matrix and is rejected explicitly.
    Nonconvergence raises rather than returning an unconverged repair.

    Reference: N. J. Higham, IMA J. Numer. Anal. 22 (2002), 329-343,
    doi:10.1093/imanum/22.3.329. The complex-Hermitian extension uses the same
    orthogonal projections in the real Frobenius inner-product space.
    """
    S = np.asarray(overlap, dtype=complex)
    if S.ndim != 2 or S.shape[0] == 0 or S.shape[0] != S.shape[1]:
        raise ValueError("overlap must be a nonempty square matrix")
    if not np.all(np.isfinite(S)):
        raise ValueError("overlap must be finite")
    if not np.allclose(S, S.conj().T, atol=1e-10, rtol=0.0):
        raise ValueError("overlap must be Hermitian")
    if not math.isfinite(norm_floor) or norm_floor < 0.0:
        raise ValueError("norm_floor must be nonnegative and finite")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be finite and in (0, 1)")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
            or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    diagonal = S.diagonal().real.copy()
    if np.any(diagonal < 0.0):
        raise ValueError("nearest-correlation repair cannot preserve a negative diagonal")
    norms = np.sqrt(diagonal)
    live = np.flatnonzero(norms > norm_floor)
    if not live.size:
        raise ValueError("every generator annihilates the reference state")
    scale = norms[live]
    C = (S[np.ix_(live, live)] / scale[:, None]) / scale[None, :]
    if not np.all(np.isfinite(C)):
        raise ValueError("normalized overlap is not finite")
    C = 0.5 * (C + C.conj().T)
    np.fill_diagonal(C, 1.0)
    input_min = float(np.linalg.eigvalsh(C)[0])
    Y = C.copy()
    iterations = 0
    if input_min < 0.0:
        correction = np.zeros_like(C)
        X = C.copy()
        for iterations in range(1, max_iterations + 1):
            old_X, old_Y = X, Y
            R = Y - correction
            values, vectors = np.linalg.eigh(R)
            X = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T
            correction = X - R
            Y = X.copy()
            np.fill_diagonal(Y, 1.0)
            size = max(float(np.linalg.norm(Y, "fro")), 1.0)
            change = max(float(np.linalg.norm(X - old_X, "fro")),
                         float(np.linalg.norm(Y - old_Y, "fro")),
                         float(np.linalg.norm(Y - X, "fro"))) / size
            if change <= tolerance and np.linalg.eigvalsh(Y)[0] >= -tolerance:
                break
        else:
            raise np.linalg.LinAlgError(
                f"nearest-correlation repair did not converge in {max_iterations} iterations")
        # Make the returned matrix feasible to roundoff, with no ridge.
        values, vectors = np.linalg.eigh(Y)
        Y = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T
        unit_scale = np.sqrt(Y.diagonal().real)
        Y = (Y / unit_scale[:, None]) / unit_scale[None, :]
        Y = 0.5 * (Y + Y.conj().T)
        np.fill_diagonal(Y, 1.0)

    repaired = np.diag(diagonal).astype(complex)
    repaired[np.ix_(live, live)] = (Y * scale[:, None]) * scale[None, :]
    np.fill_diagonal(repaired, diagonal)
    return repaired, {
        "method": "normalized_higham_dykstra",
        "evidence": "heuristic",
        "iterations": iterations,
        "tolerance": float(tolerance),
        "live_generators": int(live.size),
        "decoupled_generators": int(S.shape[0] - live.size),
        "normalized_min_eigenvalue_before": input_min,
        "normalized_min_eigenvalue_after": float(np.linalg.eigvalsh(Y)[0]),
        "normalized_correction_fro": float(np.linalg.norm(Y - C, "fro")),
        "correction_fro": float(np.linalg.norm(repaired - S, "fro")),
    }
