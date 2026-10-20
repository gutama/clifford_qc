"""Exact fixed-frame action reuse and heuristic residual packet shortlisting.

Explicit classical vectors only. Retaining HF can cost as much as a complete
frame action table; neither API is a quantum measurement protocol.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


class FrameActionCache:
    """Own an immutable orthonormal frame and lazily retain its H actions.

    ``apply`` must be a fixed linear Hermitian operator for the cache lifetime.
    Seed actions have the same caller correctness contract as ActionSpace's
    supplied actions. The callback, not cache hits or seeds, counts H calls.
    ``payload_bytes`` includes the owned frame snapshot and populated actions;
    it excludes Python containers and temporary combinations.
    """

    def __init__(self, frame, apply, *, tolerance=1e-10):
        frame = np.array(frame, dtype=complex, copy=True)
        if (not callable(apply) or frame.ndim != 2 or not all(frame.shape)
                or not np.isfinite(frame).all() or not np.isfinite(tolerance)
                or tolerance <= 0):
            raise ValueError("finite nonempty frame, callback and positive tolerance required")
        if np.max(abs(frame.conj().T @ frame - np.eye(frame.shape[1]))) > tolerance:
            raise ValueError("fixed frame must be orthonormal")
        frame.flags.writeable = False
        self._frame, self._apply, self._actions = frame, apply, {}

    def _indices(self, indices):
        indices = tuple(indices)
        if (len(set(indices)) != len(indices) or any(type(i) is not int or
                not 0 <= i < self._frame.shape[1] for i in indices)):
            raise ValueError("unique in-range integer frame indices required")
        return indices

    def seed(self, indices, actions):
        """Seed already priced columns; refuse overwriting any populated entry."""
        indices = self._indices(indices)
        actions = np.asarray(actions, dtype=complex)
        if (actions.shape != (self._frame.shape[0], len(indices))
                or not np.isfinite(actions).all() or set(indices) & self._actions.keys()):
            raise ValueError("finite matching unpopulated seed actions required")
        for i, column in zip(indices, actions.T):
            self._actions[i] = column.copy()

    def combine(self, indices, coefficients):
        """Return F[:, indices] @ coefficients and its cached exact action.

        Populate all requested columns, including zero-coefficient ones. Return
        independent arrays so callers cannot mutate cached coefficients/actions.
        """
        indices = self._indices(indices)
        coefficients = np.asarray(coefficients, dtype=complex)
        if (not indices or coefficients.shape != (len(indices),)
                or not np.isfinite(coefficients).all()):
            raise ValueError("nonempty matching finite coefficients required")
        for i in indices:
            if i not in self._actions:
                acted = np.asarray(self._apply(self._frame[:, i]), dtype=complex)
                if acted.shape != (self._frame.shape[0],) or not np.isfinite(acted).all():
                    raise ValueError("Hamiltonian action returned an invalid vector")
                self._actions[i] = acted.copy()
        return (self._frame[:, indices] @ coefficients,
                np.column_stack([self._actions[i] for i in indices]) @ coefficients)

    @property
    def indices(self):
        return tuple(sorted(self._actions))

    @property
    def frame_payload_bytes(self):
        return self._frame.nbytes

    @property
    def action_payload_bytes(self):
        return sum(column.nbytes for column in self._actions.values())

    @property
    def payload_bytes(self):
        return self.frame_payload_bytes + self.action_payload_bytes


def orthogonal_action(vector, acted, basis, actions, *, cutoff=1e-10):
    """Propagate a vector/action pair through both projection passes and phase.

    ``actions`` must equal H @ basis and ``acted`` must equal H @ vector.
    Linearity then supplies the orthogonalized action without another H call.
    """
    vector, acted = [np.array(x, dtype=complex, copy=True) for x in (vector, acted)]
    basis, actions = [np.asarray(x, dtype=complex) for x in (basis, actions)]
    if (vector.ndim != 1 or acted.shape != vector.shape or basis.ndim != 2
            or basis.shape[0] != vector.size or actions.shape != basis.shape
            or not all(np.isfinite(x).all() for x in (vector, acted, basis, actions))
            or not np.isfinite(cutoff) or cutoff <= 0):
        raise ValueError("finite matching vector/action pairs and positive cutoff required")
    for _ in range(2):
        coefficients = basis.conj().T @ vector
        vector -= basis @ coefficients
        acted -= actions @ coefficients
    norm = float(np.linalg.norm(vector))
    if norm <= cutoff:
        return None
    vector /= norm
    acted /= norm
    pivot = vector[np.argmax(abs(vector))]
    phase = pivot.conjugate() / abs(pivot)
    return vector * phase, acted * phase


def residual_shortlist(indicators: Mapping[int, float], *, fraction=0.8):
    """Smallest descending prefix covering ``fraction`` of indicator mass.

    Ties use integer keys. Zero total mass returns all keys. This is a
    deterministic screening heuristic, not a convergence/error certificate.
    """
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("shortlist fraction must lie in (0, 1]")
    if any(type(key) is not int or not np.isfinite(value) or value < 0
           for key, value in indicators.items()):
        raise ValueError("integer keys and finite nonnegative indicators required")
    ordered = sorted(indicators, key=lambda key: (-indicators[key], key))
    total = sum(indicators.values())
    if not np.isfinite(total):
        raise ValueError("indicator total must be finite")
    if total == 0:
        return tuple(ordered)
    covered, selected = 0., []
    for key in ordered:
        selected.append(key)
        covered += indicators[key]
        if covered >= fraction * total:
            break
    return tuple(selected)
