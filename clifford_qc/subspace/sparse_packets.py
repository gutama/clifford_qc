"""Sparse spin-adapted setup and experimental early residual-packet screening.

Explicit classical sector vectors, with SciPy (``.[research]``). Dense spin
blocks are bounded before construction. The accepted basis still costs D*M;
this is not a quantum state-preparation or finite-shot matrix-element API.
"""
from __future__ import annotations

import time
import numpy as np

from ..capabilities import require
require("sparse_linalg")
from scipy.sparse import csc_matrix

from .action_space import ActionSpace, energy_lowering, orthogonal_direction
from .packet_selection import residual_shortlist
from .spin_packets import spatial_pattern, spin_squared_operator
from .orthogonal import canonical_phase


def sparse_payload(matrix):
    """Owned sparse numerical arrays; excludes Python objects and temporaries."""
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


class _StepProjection:
    """One fixed solver-owned span; borrow V and own V† until the next extension.

    ActionSpace replaces, rather than mutates, its accepted basis. This private
    cache lives only inside one growth step, so it cannot outlive a span update.
    No candidate directions or Hamiltonian actions are stored here.
    """

    def __init__(self, basis):
        basis = np.asarray(basis, dtype=complex)
        if basis.ndim != 2 or not np.isfinite(basis).all():
            raise ValueError("finite basis matrix required")
        self.basis = basis
        self.adjoint = basis.conj().T
        self.adjoint.flags.writeable = False

    def direction(self, vector, *, cutoff):
        vector = np.array(vector, dtype=complex, copy=True)
        if (vector.ndim != 1 or vector.size != self.basis.shape[0]
                or not np.isfinite(vector).all() or not np.isfinite(cutoff) or cutoff <= 0):
            raise ValueError("finite matching vector and positive cutoff required")
        for _ in range(2):
            vector -= self.basis @ (self.adjoint @ vector)
        norm = float(np.linalg.norm(vector))
        return None if norm <= cutoff else canonical_phase(vector / norm)


class SparseSingletFrame:
    """Canonical CSFs built in local spatial blocks and retained as sparse CSC.

    Only interleaved, explicitly Sz=0 occupation sectors are supported. The
    canonical coordinate-axis convention matches the dense reference builder.
    Public operations return independent outputs; input basis words are copied.
    """

    def __init__(self, backend, *, tolerance=1e-9, max_block_dimension=256):
        if (backend.n % 2 or backend.sz != 0 or backend.spin_ordering != "interleaved"
                or not np.isfinite(tolerance) or tolerance <= 0
                or type(max_block_dimension) is not int or max_block_dimension < 1):
            raise ValueError("interleaved Sz=0 sector, positive tolerance and integer block cap required")
        self.words = backend.basis.copy()
        self.words.flags.writeable = False
        self.n, self.tolerance = backend.n, tolerance
        mask = (1 << self.n) - 1
        keys = np.array([spatial_pattern(int(w), mask, self.n) for w in self.words])
        order = np.argsort(keys, kind="stable")
        splits = np.flatnonzero(np.diff(keys[order])) + 1
        blocks = np.split(order, splits)
        self.maximum_block_dimension = max(map(len, blocks))
        if self.maximum_block_dimension > max_block_dimension:
            raise ValueError("spin block exceeds max_block_dimension before dense construction")
        self.spin = backend.operator(spin_squared_operator(self.n)).to_csr()
        rr, cc, data, labels, representatives = [], [], [], [], []
        count = 0
        for indices in blocks:
            block = self.spin[indices][:, indices].toarray()
            values, vectors = np.linalg.eigh(block)
            singlets = vectors[:, abs(values) <= tolerance]
            projector = singlets @ singlets.conj().T
            local = np.empty((len(indices), 0), complex)
            for j in range(len(indices)):
                direction = orthogonal_direction(projector[:, j], local, cutoff=tolerance)
                if direction is None:
                    continue
                local = np.column_stack((local, direction))
                nonzero = np.flatnonzero(direction != 0)
                rr.extend(indices[nonzero].tolist())
                cc.extend([count] * len(nonzero))
                data.extend(direction[nonzero].tolist())
                labels.append((int(keys[indices[0]]), int(indices[j])))
                representatives.append(int(self.words[indices[0]]))
                count += 1
        self.labels = tuple(labels)
        self.representatives = np.asarray(representatives, dtype=np.int64)
        self.matrix = csc_matrix((np.asarray(data, complex), (rr, cc)),
                                 shape=(backend.dimension, count))
        self.matrix.sort_indices()
        if not count:
            raise ValueError("sector has no singlet columns")

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def payload_bytes(self):
        return (sparse_payload(self.matrix) + sparse_payload(self.spin)
                + self.words.nbytes + self.representatives.nbytes)

    def columns(self, indices):
        """Materialize only the explicitly requested sector columns."""
        return self.matrix[:, indices].toarray()

    def adjoint(self, vector):
        return np.asarray(self.matrix.conj().T @ vector)

    def combine(self, indices, coefficients):
        return np.asarray(self.matrix[:, indices] @ coefficients)

    def active_columns(self, hf, orbitals):
        if (not orbitals or len(set(orbitals)) != len(orbitals)
                or any(type(p) is not int or not 0 <= p < self.n // 2 for p in orbitals)):
            raise ValueError("distinct in-range spatial orbitals required")
        mask = sum(1 << (self.n - 1 - 2*p - spin) for p in orbitals for spin in (0, 1))
        outside = ((1 << self.n) - 1) ^ mask
        allowed = (self.words & outside) == (hf & outside)
        selected = []
        for j in range(self.shape[1]):
            start, stop = self.matrix.indptr[j:j+2]
            rows, values = self.matrix.indices[start:stop], self.matrix.data[start:stop]
            if np.max(abs(values[~allowed[rows]]), initial=0) <= self.tolerance:
                selected.append(j)
        if not selected:
            raise ValueError("active occupation constraint contains no singlet columns")
        return selected, outside


def _encode(columns, tolerance):
    result = []
    for column in columns.T:
        live = np.flatnonzero(abs(column) > tolerance)
        result.append({"indices": live.tolist(), "real": column[live].real.tolist(),
                       "imag": column[live].imag.tolist()})
    return result


def run_sparse_packets(apply, diagonal, frame, hf, active_orbitals, config, *, policy="exhaustive",
                       cache_mode="none"):
    """Exact classical packet growth; optional marks precede admissibility.

    ``config`` supplies the declared cutoffs/budgets. There is no reference
    energy argument. The early policy audits all groups every
    ``config["audit_interval"]`` decisions (five in the declared campaign)
    and falls back on empty/low-gain shortlists; it is a heuristic.
    ``cache_mode`` is opt-in: ``basis`` reuses one adjoint per fixed growth
    span, ``frame`` reuses group blocks/the frame adjoint, and ``combined``
    enables both. All cache construction is charged to solver time. Caches
    borrow the run's fixed frame/span and never change selection or H calls.
    """
    if policy not in ("exhaustive", "early"):
        raise ValueError("policy must be exhaustive or early")
    if cache_mode not in ("none", "basis", "frame", "combined"):
        raise ValueError("cache_mode must be none, basis, frame or combined")
    diagonal = np.asarray(diagonal)
    if diagonal.shape != (frame.shape[0],) or not np.isfinite(diagonal).all() or np.iscomplexobj(diagonal):
        raise ValueError("finite real sector diagonal required")
    positive = ("preconditioner_shift", "residual_tolerance", "direction_absolute_tolerance",
                "orthogonality_tolerance", "spin_tolerance", "score_tie_tolerance",
                "serialization_tolerance")
    if (any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive)
            or not np.isfinite(config["direction_relative_tolerance"])
            or config["direction_relative_tolerance"] < 0
            or not 0 < config["shortlist_fraction"] <= 1
            or any(type(config[k]) is not int or config[k] < 1 for k in
                   ("maximum_columns", "maximum_growth_steps", "audit_interval"))):
        raise ValueError("invalid packet policy cutoff or budget")
    started = time.perf_counter()
    selected, outside = frame.active_columns(hf, active_orbitals)
    if len(selected) > config["maximum_columns"]:
        raise ValueError("initial span exceeds column budget")
    initial = frame.columns(selected)
    space = ActionSpace(apply, initial, tolerance=config["orthogonality_tolerance"])
    groups = {}
    for j, word in enumerate(frame.representatives):
        if j not in selected:
            key = spatial_pattern(int(word), outside, frame.n)
            groups.setdefault(key, []).append(j)
    cache_started = time.perf_counter()
    use_basis = cache_mode in ("basis", "combined")
    use_frame = cache_mode in ("frame", "combined")
    blocks = {key: frame.matrix[:, ids] for key, ids in groups.items()} if use_frame else {}
    frame_adjoint = frame.matrix.conj().T if use_frame else None
    fixed_cache_bytes = (sum(sparse_payload(block) for block in blocks.values())
                         + (sparse_payload(frame_adjoint) if use_frame else 0))
    cache_stats = {"mode": cache_mode, "basis_adjoint_builds": 0,
                   "group_block_builds": len(blocks), "frame_adjoint_builds": int(use_frame),
                   "projected_candidates": 0, "peak_owned_bytes": fixed_cache_bytes,
                   "fixed_owned_bytes": fixed_cache_bytes, "peak_basis_adjoint_bytes": 0,
                   "frame_nnz": int(frame.matrix.nnz) if use_frame else 0,
                   "group_nnz": sum(int(block.nnz) for block in blocks.values()),
                   "group_columns": sum(block.shape[1] for block in blocks.values()),
                   "index_itemsize": frame.matrix.indices.dtype.itemsize,
                   "indptr_itemsize": frame.matrix.indptr.dtype.itemsize,
                   "construction_wall_seconds": time.perf_counter()-cache_started}
    rows, decisions = [], []
    materialized = scored = 0
    added, choice, score = initial, {"initial_csf_indices": selected}, None
    for step in range(config["maximum_growth_steps"] + 1):
        energy, state, residual = space.solve()
        spin_norm = float(np.linalg.norm(frame.spin @ state))
        if spin_norm > config["spin_tolerance"]:
            raise ValueError("Ritz state escaped singlet sector")
        rows.append({"step": step, "energy": energy, "residual_norm": float(np.linalg.norm(residual)),
            "spin_action_norm": spin_norm, "dimension": space.basis.shape[1],
            "h_action_calls": space.h_action_calls, "candidates_scored": scored,
            "materialized_candidates": materialized, "added_columns": _encode(added, config["serialization_tolerance"]),
            "selection": choice, "selection_score": score, "retained_payload_bytes": space.payload_bytes,
            "wall_seconds": time.perf_counter() - started})
        stop = ("residual_tolerance" if rows[-1]["residual_norm"] <= config["residual_tolerance"]
                else "column_ceiling" if space.basis.shape[1] >= config["maximum_columns"]
                else "growth_step_ceiling" if step == config["maximum_growth_steps"] else None)
        if stop:
            break
        raw = residual / np.maximum(diagonal - energy + config["preconditioner_shift"],
                                    config["preconditioner_shift"])
        cutoff = max(config["direction_absolute_tolerance"],
                     config["direction_relative_tolerance"] * np.linalg.norm(raw))
        if use_frame:
            coefficients = np.asarray(frame_adjoint @ raw)
            residual_coefficients = np.asarray(frame_adjoint @ residual)
        else:
            coefficients, residual_coefficients = frame.adjoint(raw), frame.adjoint(residual)
        cache_started = time.perf_counter()
        projector = _StepProjection(space.basis) if use_basis else None
        if projector is not None:
            size = projector.adjoint.nbytes
            cache_stats['basis_adjoint_builds'] += 1
            cache_stats['peak_basis_adjoint_bytes'] = max(cache_stats['peak_basis_adjoint_bytes'], size)
            cache_stats['peak_owned_bytes'] = max(cache_stats['peak_owned_bytes'], fixed_cache_bytes+size)
        cache_stats['construction_wall_seconds'] += time.perf_counter()-cache_started
        indicators = {key: float(np.linalg.norm(residual_coefficients[ids])**2)
                      for key, ids in groups.items()}
        short = (list(residual_shortlist(indicators, fraction=config["shortlist_fraction"]))
                 if policy == "early" else sorted(groups))
        audit = policy == "early" and (step + 1) % config["audit_interval"] == 0
        attempted, scores, pairs = set(), {}, {}

        def evaluate(keys):
            nonlocal materialized
            for key in keys:
                if key in attempted:
                    continue
                attempted.add(key)
                materialized += 1
                ids = groups[key]
                vector = (np.asarray(blocks[key] @ coefficients[ids]) if use_frame
                          else frame.combine(ids, coefficients[ids]))
                direction = (projector.direction(vector, cutoff=cutoff) if projector is not None
                             else orthogonal_direction(vector, space.basis, cutoff=cutoff))
                cache_stats['projected_candidates'] += int(projector is not None)
                if direction is None:
                    continue
                acted = space.action(direction)
                pairs[key] = direction, acted
                scores[key] = energy_lowering(energy, np.vdot(direction, acted).real,
                                              np.vdot(state, acted))

        evaluate(sorted(groups) if audit else short)
        fallback = (policy == "early" and not audit and len(attempted) < len(groups)
                    and (not scores or max(scores.values()) <= config["score_tie_tolerance"]))
        if fallback:
            evaluate(sorted(groups))
        scored += len(scores)
        if not scores:
            stop = "no_admissible_direction"
            break
        maximum = max(scores.values())
        key = min(k for k, value in scores.items() if value >= maximum - config["score_tie_tolerance"])
        marked_scores = [scores[k] for k in short if k in scores]
        decisions.append({"step": step, "indicators": {str(k): v for k, v in indicators.items()},
            "shortlist": short, "attempted": sorted(attempted),
            "scores": {str(k): v for k, v in scores.items()}, "selected": key,
            "audit": audit, "fallback": fallback,
            "audit_missed_winner": bool(audit and key not in short),
            "audit_score_loss": max(0., maximum - max(marked_scores, default=0.)) if audit else None})
        direction, acted = pairs[key]
        added = direction[:, None]
        space.extend(added, actions=acted[:, None])
        projector = None  # Release the old span's adjoint before constructing the next.
        choice, score = {"candidate": key}, scores[key]
    return {"policy": policy, "trajectory": rows, "decisions": decisions,
            "cache": cache_stats,
            "stop_reason": stop, "final_h_action_calls": space.h_action_calls,
            "final_candidates_scored": scored, "materialized_candidates": materialized,
            "solver_wall_seconds": time.perf_counter() - started}
