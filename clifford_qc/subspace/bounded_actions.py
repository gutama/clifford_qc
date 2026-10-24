"""Admitted fixed-capacity projection from cached actions.

Array/workspace envelopes exclude Python, allocator and vendor-library overhead.
RSS sampling is separate from admission and is not an OS allocation ceiling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import sys

import numpy as np

from .linalg import solve_projected


class ActionMemoryLimit(MemoryError):
    """Admission or a monitored RSS boundary was exceeded."""


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def process_rss_bytes():
    """Current Linux resident pages; never substitute a high-water statistic."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("current RSS monitoring requires Linux /proc")
    with open('/proc/self/statm') as stream:
        return int(stream.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')


@dataclass(frozen=True)
class ActionMemoryPlan:
    dimension: int
    capacity: int
    roots: int
    components: tuple[tuple[str, int], ...]
    max_owned_bytes: int
    max_rss_bytes: int | None
    rss_headroom_bytes: int

    @classmethod
    def admit(cls, dimension, capacity, *, roots=1, ncv=0, input_bytes=0,
              max_owned_bytes=256 * 2**20, max_rss_bytes=None, rss_headroom_bytes=0):
        for name, value in (("dimension", dimension), ("capacity", capacity),
                            ("roots", roots), ("max_owned_bytes", max_owned_bytes)):
            positive_integer(value, name)
        if not roots <= capacity <= dimension:
            raise ValueError("require roots <= capacity <= dimension")
        for name, value in (("ncv", ncv), ("input_bytes", input_bytes),
                            ("rss_headroom_bytes", rss_headroom_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if ncv and not roots + 1 < ncv <= dimension:
            raise ValueError("ARPACK requires roots + 1 < ncv <= dimension")
        if max_rss_bytes is not None:
            positive_integer(max_rss_bytes, "max_rss_bytes")
            if rss_headroom_bytes >= max_rss_bytes:
                raise ValueError("RSS headroom must be smaller than the RSS ceiling")
        components = {
            "basis_actions": 32 * dimension * capacity,
            "projection_and_small_solve": 16 * 24 * capacity**2,
            "candidate_ritz_restart_snapshots": 16 * dimension * (12 * roots + 4 + 4 * capacity),
            "operator_workspace_envelope": 256 * dimension,
            "input_arrays_and_coefficients": input_bytes,
            "arpack_workspace_envelope": 16 * (
                2 * dimension * ncv + 8 * dimension + 12 * ncv**2 + 40 * ncv) if ncv else 0,
        }
        if sum(components.values()) > max_owned_bytes:
            raise ActionMemoryLimit("owned-array plan exceeds max_owned_bytes before allocation")
        return cls(dimension, capacity, roots, tuple(components.items()), max_owned_bytes,
                   max_rss_bytes, rss_headroom_bytes)

    @property
    def planned_bytes(self):
        return sum(value for _, value in self.components)

    def as_dict(self):
        return {**asdict(self), "components": dict(self.components), "planned_bytes": self.planned_bytes,
                "scope": "declared array payload and conservative workspace envelopes; excludes Python, allocator and vendor-library overhead; not an RSS guarantee"}


class RSSMonitor:
    def __init__(self, plan, *, read_rss=None):
        self.plan = plan
        self.read_rss = process_rss_bytes if read_rss is None else read_rss
        self.peak_bytes = self.samples = 0
        self.unavailable_reason = None

    def check(self, *, admission=False):
        if self.unavailable_reason is not None:
            if self.plan.max_rss_bytes is not None:
                raise ActionMemoryLimit("configured RSS ceiling cannot be monitored: " + self.unavailable_reason)
            return
        try:
            value = int(self.read_rss())
        except (OSError, RuntimeError) as exc:
            self.unavailable_reason = str(exc)
            if self.plan.max_rss_bytes is not None:
                raise ActionMemoryLimit("configured RSS ceiling cannot be monitored: " + str(exc)) from exc
            return
        self.samples += 1
        self.peak_bytes = max(self.peak_bytes, value)
        limit = self.plan.max_rss_bytes
        if limit is not None and value + (self.plan.rss_headroom_bytes if admission else 0) > limit:
            raise ActionMemoryLimit("monitored process RSS ceiling exceeded")

    def as_dict(self):
        return {"peak_observed_bytes": self.peak_bytes if self.samples else None, "samples": self.samples,
                "available": self.unavailable_reason is None, "unavailable_reason": self.unavailable_reason,
                "ceiling_bytes": self.plan.max_rss_bytes, "headroom_bytes": self.plan.rss_headroom_bytes,
                "enforcement": "boundary sampling before/after actions and solves; no OS limit; transient peaks may be missed"}


class BoundedActionSpace:
    """One V/HV allocation, live prefixes and incremental S/H borders.

    Columns must be orthonormal; supplied cached actions are a caller contract.
    Numerical results own snapshots independent of later growth or restart.
    """
    def __init__(self, apply, plan, *, tolerance=1e-10, monitor=None):
        if not callable(apply) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("an action callback and positive tolerance are required")
        self.plan, self.apply, self.tolerance = plan, apply, tolerance
        self.monitor = monitor if monitor is not None else RSSMonitor(plan)
        self.monitor.check(admission=True)
        self._basis = np.empty((plan.dimension, plan.capacity), complex)
        self._actions = np.empty_like(self._basis)
        self._overlap = np.empty((plan.capacity, plan.capacity), complex)
        self._hamiltonian = np.empty_like(self._overlap)
        self.size = self.h_action_columns = self.restarts = self.resets = 0

    def reset(self, *, restart=False):
        """Clear the live prefix without reallocating, and count the operation."""
        self.size = 0
        self.resets += 1
        self.restarts += int(restart)

    @property
    def basis(self):
        view = self._basis[:, :self.size].view()
        view.flags.writeable = False
        return view

    @property
    def actions(self):
        view = self._actions[:, :self.size].view()
        view.flags.writeable = False
        return view

    @property
    def payload_bytes(self):
        return sum(x.nbytes for x in (self._basis, self._actions, self._overlap, self._hamiltonian))

    def extend(self, columns, *, actions=None):
        columns = np.asarray(columns, complex)
        if (columns.ndim != 2 or columns.shape[0] != self.plan.dimension
                or not columns.shape[1] or not np.isfinite(columns).all()):
            raise ValueError("finite nonempty matching columns required")
        if self.size + columns.shape[1] > self.plan.capacity:
            raise ActionMemoryLimit("action-space capacity exceeded before applying H")
        if actions is not None:
            actions = np.asarray(actions, complex)
            if actions.shape != columns.shape or not np.isfinite(actions).all():
                raise ValueError("cached actions must match columns")
        previous = self.size
        try:
            for j, vector in enumerate(columns.T):
                k = self.size
                for i in range(k):
                    if abs(np.vdot(self._basis[:, i], vector)) > self.tolerance:
                        raise ValueError("columns must be orthonormal against the whole span")
                if abs(np.vdot(vector, vector) - 1) > self.tolerance:
                    raise ValueError("columns must have unit norm")
                self.monitor.check()
                if actions is None:
                    self.h_action_columns += 1
                    acted = np.asarray(self.apply(vector), complex)
                else:
                    acted = actions[:, j]
                if acted.shape != vector.shape or not np.isfinite(acted).all():
                    raise ValueError("invalid Hamiltonian action")
                self._basis[:, k], self._actions[:, k] = vector, acted
                for i in range(k + 1):
                    h = np.vdot(self._basis[:, i], acted)
                    reverse = np.vdot(self._actions[:, i], vector)
                    if abs(h - reverse) > self.tolerance * max(1., abs(h), abs(reverse)):
                        raise ValueError("projected Hamiltonian action must be Hermitian")
                    s = np.vdot(self._basis[:, i], vector)
                    self._overlap[i, k], self._overlap[k, i] = s, s.conjugate()
                    self._hamiltonian[i, k], self._hamiltonian[k, i] = h, h.conjugate()
                self.size += 1
                self.monitor.check()
        except Exception:
            self.size = previous
            raise

    def matrices(self):
        return (self._overlap[:self.size, :self.size].copy(),
                self._hamiltonian[:self.size, :self.size].copy())

    def solve(self):
        if not self.size:
            raise ValueError("cannot solve an empty action space")
        self.monitor.check()
        s, h = self.matrices()
        result = solve_projected(s, h)
        self.monitor.check()
        return result

    def ritz_actions(self, result, roots=1):
        if not 1 <= roots <= result.effective_rank:
            raise ValueError("invalid retained root count")
        coefficients = result.coefficients[:, :roots]
        return self.basis @ coefficients, self.actions @ coefficients

    def restart(self, columns, actions):
        columns, actions = np.array(columns, complex, copy=True), np.array(actions, complex, copy=True)
        if columns.ndim != 2 or columns.shape[1] > self.plan.roots:
            raise ActionMemoryLimit("restart block exceeds admitted roots")
        if (columns.shape != actions.shape or columns.shape[0] != self.plan.dimension
                or not columns.shape[1] or not np.isfinite(columns).all() or not np.isfinite(actions).all()
                or np.max(abs(columns.conj().T @ columns - np.eye(columns.shape[1]))) > self.tolerance):
            raise ValueError("restart columns/actions must be finite and orthonormal")
        projected = columns.conj().T @ actions
        if not np.allclose(projected, projected.conj().T, atol=self.tolerance, rtol=self.tolerance):
            raise ValueError("restart Hamiltonian must be Hermitian")
        self.reset(restart=True)
        try:
            self.extend(columns, actions=actions)
        except Exception:
            self.size = 0
            raise
