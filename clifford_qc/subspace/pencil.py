"""Opt-in post-processing of measured overlap matrices.

An A-CASE generator need not be unitary, so its overlap diagonal need not be
one. We project the *normalized* live block onto the correlation matrices,
then restore the measured diagonal. This minimizes a normalized Frobenius
distance, not the unscaled distance and not a covariance-weighted distance.
It supplies neither a variational bound nor an energy confidence interval.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from .linalg import DEFAULT_NORM_FLOOR, normalize_overlap


def _rank_one_projection(C: np.ndarray, vector: np.ndarray, tolerance: float
                         ) -> np.ndarray | None:
    """Accept a rank-one candidate only with a primal/dual KKT certificate.

    For X=zz*, |z_i|=1, stationarity is X-C+diag(y)-Z=0,
    Z>=0 and Zz=0. Set y_i=Re(((C-X)z)_i / z_i) and test these
    conditions. This avoids slow alternating projections for large, aligned
    off-diagonal errors without clipping data or dropping positive live rows.
    """
    if np.any(np.abs(vector) <= np.finfo(float).eps):
        return None
    z = vector / np.abs(vector)
    X = np.outer(z, z.conj())
    residual = C - X
    multipliers = (residual @ z) * z.conj()
    slack = np.diag(multipliers.real) - residual
    limit = max(tolerance, 50.0 * np.finfo(float).eps * np.linalg.norm(C, ord=2))
    if (np.max(np.abs(multipliers.imag)) > limit or
            np.linalg.eigvalsh(slack)[0] < -limit or
            np.linalg.norm(slack @ z, ord=np.inf) > limit):
        return None
    return X


def repair_overlap(overlap: np.ndarray, *, norm_floor: float = DEFAULT_NORM_FLOOR,
                   tolerance: float = 1e-10, max_iterations: int | np.integer = 1000
                   ) -> tuple[np.ndarray, dict]:
    """Nearest-correlation repair of the solver's live block.

    Higham's alternating projections with Dykstra's correction solve the
    Hermitian nearest-correlation problem on the normalized live block.
    A final PSD projection and diagonal congruence remove the tolerance-sized
    feasibility defect; the distance optimum is therefore numerical, to the
    requested iteration tolerance. No positive eigenvalue floor is added.

    Rows whose norm is at most ``norm_floor`` are decoupled and retain their
    diagonal, including negative measurements; these are exactly the rows the
    canonical solver drops. Only the live block is constrained to be PSD.
    Nonconvergence raises rather than returning an unconverged repair.

    Reference: N. J. Higham, IMA J. Numer. Anal. 22 (2002), 329-343,
    doi:10.1093/imanum/22.3.329. The complex-Hermitian extension uses the same
    orthogonal projections in the real Frobenius inner-product space.
    """
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be finite and in (0, 1)")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) \
            or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)
    S = np.asarray(overlap, dtype=complex)
    frame = normalize_overlap(S, norm_floor=norm_floor)
    diagonal, live = frame.diagonal, frame.indices
    scale = frame.norms[live]
    C = frame.matrix
    C = 0.5 * (C + C.conj().T)
    np.fill_diagonal(C, 1.0)
    input_values, input_vectors = np.linalg.eigh(C)
    input_min = float(input_values[0])
    Y = C.copy()
    iterations = 0
    certificate = (_rank_one_projection(C, input_vectors[:, -1], tolerance)
                   if input_min < 0.0 else None)
    if certificate is not None:
        Y = certificate
        np.fill_diagonal(Y, 1.0)
    elif input_min < 0.0:
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
        "method": ("normalized_rank_one_kkt" if certificate is not None
                   else "normalized_higham_dykstra"),
        "rank_one_kkt_certified": certificate is not None,
        "evidence": "heuristic",
        "iterations": iterations,
        "tolerance": float(tolerance),
        "live_generators": int(live.size),
        "decoupled_generators": int(S.shape[0] - live.size),
        "nonpositive_diagonal_indices": tuple(int(i) for i in np.flatnonzero(diagonal <= 0)),
        "live_indices": tuple(int(i) for i in live),
        "normalized_min_eigenvalue_before": input_min,
        "normalized_min_eigenvalue_after": float(np.linalg.eigvalsh(Y)[0]),
        "normalized_correction_fro": float(np.linalg.norm(Y - C, "fro")),
        "correction_fro": float(np.linalg.norm(repaired - S, "fro")),
    }
