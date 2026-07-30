"""Low-energy response functions from an exact projected A-CASE spectrum.

The projected solver already supplies energies and transition matrix elements.
This module turns those primitives into the Lehmann representation without
materializing a Ritz state.  It belongs to the exact/asymptotic analysis layer:
finite-shot confidence intervals for the nonlinear spectral weights are not
claimed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .solver import SubspaceResult


@dataclass(frozen=True)
class ResponseLine:
    """One positive-frequency Lehmann line ``|<f|O|i>|^2``."""

    initial_state: int
    final_state: int
    excitation_energy: float
    weight: float
    amplitude: complex


def lehmann_spectrum(result: SubspaceResult, observable, *, initial_state: int = 0,
                     include_elastic: bool = False, min_weight: float = 0.0,
                     energy_tol: float = 1e-12) -> tuple[ResponseLine, ...]:
    """Positive-frequency response lines within a projected Ritz subspace.

    The returned spectrum is exact for the supplied projected matrices.  It is
    the physical spectrum only to the extent that the subspace spans the states
    reached by ``observable``; callers should report the basis and conditioning
    beside it, as the showcase does.
    """
    roots = len(result.energies)
    if not 0 <= initial_state < roots:
        raise IndexError(f"initial_state must be in [0, {roots})")
    if min_weight < 0.0:
        raise ValueError("min_weight must be non-negative")
    initial_energy = result.energies[initial_state]
    lines: list[ResponseLine] = []
    for final_state, final_energy in enumerate(result.energies):
        gap = float(final_energy - initial_energy)
        if final_state == initial_state and not include_elastic:
            continue
        if gap < -energy_tol:
            continue
        amplitude = result.transition(observable, final_state, initial_state)
        weight = float(abs(amplitude) ** 2)
        if weight + 1e-15 < min_weight:
            continue
        lines.append(ResponseLine(
            initial_state=initial_state,
            final_state=final_state,
            excitation_energy=max(0.0, gap),
            weight=max(0.0, weight),
            amplitude=amplitude,
        ))
    return tuple(lines)


def static_susceptibility(lines: Sequence[ResponseLine], *,
                          gap_floor: float = 1e-12) -> float:
    """Zero-temperature susceptibility ``2 sum_f w_f / (E_f-E_0)``.

    This is the response to a perturbation ``-field * observable`` for a
    non-degenerate initial state.  A line at or below ``gap_floor`` signals a
    degeneracy for which the non-degenerate expression is undefined.
    """
    if gap_floor <= 0.0:
        raise ValueError("gap_floor must be positive")
    total = 0.0
    for line in lines:
        if line.excitation_energy <= gap_floor and line.weight > 0.0:
            raise ValueError("static susceptibility is singular in a degenerate subspace")
        if line.excitation_energy > gap_floor:
            total += 2.0 * line.weight / line.excitation_energy
    return float(total)


def broaden_response(lines: Sequence[ResponseLine], frequencies,
                     broadening: float) -> np.ndarray:
    """Normalized Lorentzian broadening of discrete response lines."""
    if broadening <= 0.0 or not math.isfinite(broadening):
        raise ValueError("broadening must be positive and finite")
    omega = np.asarray(frequencies, dtype=float)
    if not np.all(np.isfinite(omega)):
        raise ValueError("frequencies must be finite")
    out = np.zeros_like(omega)
    for line in lines:
        delta = omega - line.excitation_energy
        out += line.weight * broadening / (math.pi * (delta * delta + broadening ** 2))
    return out


__all__ = [
    "ResponseLine",
    "broaden_response",
    "lehmann_spectrum",
    "static_susceptibility",
]
