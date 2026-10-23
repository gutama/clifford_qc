"""Deterministic representatives of complex rays with relative pivot ties.

This convention is opt-in for historical solvers. A tolerance band stabilizes
near-equal pivot magnitudes; it cannot make the representative continuous at
the band's boundary. No normalization or component truncation is performed.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _validate_phase_tie_tolerance(tie_tolerance):
    if isinstance(tie_tolerance, (bool, np.bool_)) or not isinstance(tie_tolerance, Real):
        raise ValueError("phase tie tolerance must be a finite real scalar in [0, 1)")
    try:
        tolerance = float(tie_tolerance)
    except (OverflowError, ValueError) as exc:
        raise ValueError("phase tie tolerance must be a finite real scalar in [0, 1)") from exc
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("phase tie tolerance must be a finite real scalar in [0, 1)")
    return tolerance


def _phase_vector(vector):
    out = np.array(vector, dtype=np.complex128, copy=True)
    if out.ndim != 1 or not out.size or not np.isfinite(out).all():
        raise ValueError("a finite nonempty one-dimensional vector is required")
    return out


def _pivot(vector, tolerance):
    # Scaling real and imaginary components separately avoids overflow in abs
    # and in complex division, including for very small nonzero vectors.
    scale = max(float(np.max(np.abs(vector.real))), float(np.max(np.abs(vector.imag))))
    if scale == 0:
        return None
    with np.errstate(under="ignore"):
        magnitudes = np.hypot(vector.real / scale, vector.imag / scale)
    maximum = float(np.max(magnitudes))
    eligible = (magnitudes > 0) & (maximum - magnitudes <= tolerance * maximum)
    return int(np.flatnonzero(eligible)[0])


def deterministic_phase_pivot(vector, *, tie_tolerance=1e-12):
    """Return the first nonzero component within a relative band of the maximum.

    The band is ``max_magnitude - magnitude <= tie_tolerance * max_magnitude``
    after safe common scaling. ``0 <= tie_tolerance < 1`` is required. Return
    ``None`` for an all-zero vector. There is no absolute amplitude floor.
    """
    tolerance = _validate_phase_tie_tolerance(tie_tolerance)
    return _pivot(_phase_vector(vector), tolerance)


def canonical_phase_tolerant(vector, *, tie_tolerance=1e-12):
    """Copy a vector and rotate its tolerant pivot onto the nonnegative real axis.

    Preserve its norm and physical ray to floating-point accuracy. An all-zero
    input returns an independent zero copy. Reject nonfinite inputs and a
    rotated result that cannot be represented in complex128.
    """
    tolerance = _validate_phase_tie_tolerance(tie_tolerance)
    out = _phase_vector(vector)
    pivot = _pivot(out, tolerance)
    if pivot is None:
        return out
    value = out[pivot]
    scale = max(abs(float(value.real)), abs(float(value.imag)))
    real, imag = float(value.real) / scale, float(value.imag) / scale
    phase = complex(real, -imag) / math.hypot(real, imag)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            out *= phase
    except FloatingPointError as exc:
        raise ValueError("phase-rotated vector is not representable in complex128") from exc
    if not np.isfinite(out).all():
        raise ValueError("phase-rotated vector is not representable in complex128")
    return out
