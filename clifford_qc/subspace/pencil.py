"""Opt-in post-processing of measured overlap matrices.

An A-CASE generator need not be unitary, so its overlap diagonal need not be
one. We project the *normalized* live block onto the correlation matrices,
then restore the measured diagonal. This minimizes a normalized Frobenius
distance, not the unscaled distance and not a covariance-weighted distance.
It supplies neither a variational bound nor an energy confidence interval.
"""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .linalg import (DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S,
                     SubspaceResult, fix_phase, normalize_overlap, solve_projected)


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


def rotation_parameters(*, angle: float, energy_shift: float = 0.0,
                        energy_scale: float = 1.0, pole_tolerance: float = 1e-10
                        ) -> dict[str, float]:
    """Validate a fixed rotation policy; angles are radians, scale is positive.

    This policy contains no data-dependent angle selector. Choose the angle
    independently of evaluation data, or label any angle sweep exploratory.
    """
    values = {"angle": angle, "energy_shift": energy_shift,
              "energy_scale": energy_scale, "pole_tolerance": pole_tolerance}
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a finite real scalar")
        try:
            values[key] = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{key} must be a finite real scalar") from error
        if not math.isfinite(values[key]):
            raise ValueError(f"{key} must be a finite real scalar")
    if not -math.pi / 2 < values["angle"] < math.pi / 2:
        raise ValueError("angle must be in (-pi/2, pi/2)")
    if values["energy_scale"] <= 0:
        raise ValueError("energy_scale must be positive")
    if not 0 < values["pole_tolerance"] < 1:
        raise ValueError("pole_tolerance must be in (0, 1)")
    return values


def solve_rotated(S: np.ndarray, H: np.ndarray, labels: Sequence[str] | None = None, *, angle: float,
                  energy_shift: float = 0.0, energy_scale: float = 1.0,
                  pole_tolerance: float = 1e-10, tau_s: float = DEFAULT_TAU_S,
                  rel_tau: float = 0.0, max_condition: float = DEFAULT_MAX_CONDITION,
                  norm_floor: float = DEFAULT_NORM_FLOOR,
                  resources: dict | None = None) -> SubspaceResult:
    """Fixed-angle thresholding followed by the inverse eigenvalue map.

    On the original live generators, set h=(H-shift*S)/scale, then solve
    (h*cos(angle)-S*sin(angle), S*cos(angle)+h*sin(angle)). The canonical
    solver normalizes the rotated metric's diagonal before thresholding.
    Return physical energies sorted ascending and coefficients normalized
    in the supplied, unrotated S (the repaired S when repair was requested).

    This conservative first policy requires the rotated live metric to be
    PSD to roundoff, with positive diagonal. It rejects poles and nonpositive
    physical norms instead of dropping roots or crossing eigenvalue branches.
    Shot-derived floors and ridges are deliberately unsupported. Truncation
    changes the retained span; it need not preserve the unrotated Ritz values.
    Zero angle delegates exactly to the original numerical solver.

    See De Vriendt et al., arXiv:2602.11985, for rotated thresholding. This
    implementation does not implement their data-selected angle heuristic.
    """
    policy = rotation_parameters(angle=angle, energy_shift=energy_shift,
                                 energy_scale=energy_scale, pole_tolerance=pole_tolerance)
    S, H = np.asarray(S, dtype=complex), np.asarray(H, dtype=complex)
    frame = normalize_overlap(S, norm_floor=norm_floor)
    if H.shape != S.shape or not np.all(np.isfinite(H)):
        raise ValueError("Hamiltonian must be finite and match the overlap shape")
    if not np.allclose(H, H.conj().T, atol=1e-10, rtol=0.0):
        raise ValueError("Hamiltonian must be Hermitian")
    labels = tuple(labels) if labels is not None else tuple(f"A{i}" for i in range(len(S)))
    if len(labels) != len(S):
        raise ValueError("labels and matrix size disagree")
    if policy["angle"] == 0.0:
        result = solve_projected(S, H, labels, tau_s=tau_s, rel_tau=rel_tau,
                                 max_condition=max_condition, norm_floor=norm_floor,
                                 resources=resources)
        return replace(result, resources={**result.resources, "pencil_rotation": {
            **policy, "angle_policy": "fixed", "evidence": "heuristic",
            "zero_angle_control": True}})

    live = frame.indices
    S_live = frame.matrix
    H_live = ((H / frame.scaling[:, None]) / frame.scaling[None, :])[np.ix_(live, live)]
    h = (H_live - policy["energy_shift"] * S_live) / policy["energy_scale"]
    cosine, sine = math.cos(policy["angle"]), math.sin(policy["angle"])
    rotated_S, rotated_H = cosine * S_live + sine * h, cosine * h - sine * S_live
    if not np.all(np.isfinite(rotated_S)) or not np.all(np.isfinite(rotated_H)):
        raise ValueError("rotated pencil must be finite")
    metric_values = np.linalg.eigvalsh(rotated_S)
    roundoff = 100 * np.finfo(float).eps * max(float(np.max(np.abs(metric_values))), 1.0)
    if metric_values[0] < -roundoff:
        raise ValueError("rotated overlap is indefinite on the original live generators")
    if np.any(rotated_S.diagonal().real <= norm_floor ** 2):
        raise ValueError("rotated overlap has a nonpositive or dead live diagonal")
    result = solve_projected(rotated_S, rotated_H, [labels[i] for i in live],
                             tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition,
                             norm_floor=norm_floor, resources=resources)
    mu = np.asarray(result.energies)
    denominator = cosine - sine * mu
    pole_scale = np.maximum(1.0, np.abs(cosine) + np.abs(sine * mu))
    if np.any(denominator <= policy["pole_tolerance"] * pole_scale):
        raise ValueError("inverse rotation reaches a pole or nonpositive physical-norm branch")
    energies = policy["energy_shift"] + policy["energy_scale"] * (
        cosine * mu + sine) / denominator
    if not np.all(np.isfinite(energies)):
        raise ValueError("inverse rotation produced nonfinite physical energies")
    vectors = result.coefficients.copy()
    norms = np.einsum("ik,ik->k", vectors.conj(), S_live @ vectors).real
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("rotated Ritz vector has a nonpositive physical norm")
    vectors /= np.sqrt(norms)[None, :]
    coefficients = np.zeros((len(S), len(mu)), dtype=complex)
    coefficients[live, :] = vectors / frame.scaling[live, None]
    order = np.argsort(energies, kind="stable")
    coefficients = np.column_stack([fix_phase(coefficients[:, i]) for i in order])
    out = dict(result.resources)
    out.update({"basis_size": len(S), "live_generators": int(live.size),
                "pencil_rotation": {**policy, "angle_policy": "fixed", "evidence": "heuristic",
                    "zero_angle_control": False,
                    "rotated_metric_min_eigenvalue": float(metric_values[0]),
                    "rotated_energies": tuple(float(v) for v in mu[order]),
                    "inverse_denominators": tuple(float(v) for v in denominator[order]),
                    "original_live_indices": tuple(int(i) for i in live)},
                "dropped_generators": tuple(label for label, keep in zip(labels, frame.live)
                                            if not keep)})
    return replace(result, energies=tuple(float(e) for e in energies[order]),
                   coefficients=coefficients, basis_labels=labels, resources=out)
