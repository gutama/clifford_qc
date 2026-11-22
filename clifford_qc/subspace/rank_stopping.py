"""Measured, normalization-aware overlap diagnostics for precision stopping."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from ..measurement.functionals import WordFunctional, quadratic_functional
from .linalg import canonical_eigh, normalize_overlap


@dataclass(frozen=True)
class RankStopPolicy:
    """Require full declared rank at consecutive statistically separated looks.

    A heuristic delta-method guard, not a confidence certificate. Require the
    full bank, not merely the same (possibly truncated) rank twice. The
    declared bank must therefore be independent. With Ritz targeting, an
    unresolved rank buys overlap-sensitive and energy-sensitive shots.
    """

    consecutive_looks: int = 2
    margin_delta: float = 0.05

    def __post_init__(self):
        if (isinstance(self.consecutive_looks, (bool, np.bool_)) or
                not isinstance(self.consecutive_looks, Integral) or self.consecutive_looks < 2):
            raise ValueError('consecutive_looks must be an integer >= 2')
        if (isinstance(self.margin_delta, (bool, np.bool_)) or
                not isinstance(self.margin_delta, Real) or not 0 < self.margin_delta < 1):
            raise ValueError('margin_delta must be in (0,1)')


def _scaled(functional, weight):
    return WordFunctional({code: weight*value for code, value in functional.coefficients.items()
                           if weight*value != 0.}, weight*functional.constant)


def normalized_mode_functional(shared, frame, eigenvalue, vector):
    """Derivative of u† D^-1/2 S D^-1/2 u, including measured D's motion."""
    c = np.zeros(len(shared.indices), complex)
    c[frame.indices] = vector/frame.scaling[frame.indices]
    functional = quadratic_functional(shared.bank, shared.indices, c, 'overlap')
    for index, amplitude in zip(frame.indices, vector, strict=True):
        weight = -float(eigenvalue)*abs(amplitude)**2/frame.scaling[index]**2
        functional = functional + _scaled(shared.diagonal_functional(shared.indices[index]), weight)
    return functional


def rank_margin(shared, cache, point, policy: RankStopPolicy, family: int):
    """Full-rank cutoff separation and its weakest margin's shot functional.

    The smallest normalized eigenvalue must clear each of the absolute,
    relative and conditioning cutoffs. Using all three avoids differentiating
    a max at a tie. Derivatives of lambda_min-k*lambda_max include covariance
    between modes. Bonferroni pays for three comparisons and every scheduled
    look; Gaussian plug-in radii are heuristic under adaptive sampling, near
    degeneracy and data-derived eigenvectors. No exact expectations enter.
    """
    params = point.estimator_parameters
    if params is None or params.get('calibrate_overlap'):
        raise ValueError('rank margin requires a measured fixed-cutoff solve')
    overlap, _ = shared.matrices(cache)
    frame = normalize_overlap(overlap, norm_floor=params['norm_floor'])
    expected = len(shared.indices)
    live = len(frame.indices)
    if live != expected:
        return {'required_rank': expected, 'live_rank': live, 'resolved': False,
                'comparisons': [], 'lower_margin': None, 'weakest': None,
                'evidence': 'heuristic'}, None
    values, vectors = canonical_eigh(frame.matrix)
    weak, strong = float(values[0]), float(values[-1])
    weak_f = normalized_mode_functional(shared, frame, weak, vectors[:, 0])
    strong_f = normalized_mode_functional(shared, frame, strong, vectors[:, -1])
    comparisons = []
    gradients = []
    floors = [('absolute', float(params['tau_s']), weak_f),
              ('relative', float(params['rel_tau'])*strong,
               weak_f-_scaled(strong_f, params['rel_tau'])),
              ('condition', strong/params['max_condition'],
               weak_f-_scaled(strong_f, 1/params['max_condition']))]
    for name, threshold, functional in floors:
        radius = float(functional.radius(cache, float(policy.margin_delta), 3*family, bound='normal'))
        if not math.isfinite(radius):
            return {'required_rank': expected, 'live_rank': live, 'resolved': False,
                    'comparisons': [], 'lower_margin': None, 'weakest': weak,
                    'evidence': 'heuristic'}, None
        margin = weak-threshold
        comparisons.append({'cutoff': name, 'threshold': threshold, 'margin': margin,
                            'radius': radius, 'lower_margin': margin-radius})
        gradients.append(functional)
    worst = min(range(3), key=lambda i: comparisons[i]['lower_margin'])
    lower = comparisons[worst]['lower_margin']
    return {'required_rank': expected, 'live_rank': live,
            'resolved': bool(point.effective_rank == expected and lower > 0.),
            'comparisons': comparisons, 'lower_margin': lower, 'weakest': weak,
            'evidence': 'heuristic'}, gradients[worst]


__all__ = ['RankStopPolicy', 'rank_margin', 'normalized_mode_functional']
