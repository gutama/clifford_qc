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


def sparse_payload(matrix):
    """Owned sparse numerical arrays; excludes Python objects and temporaries."""
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


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


def run_sparse_packets(apply, diagonal, frame, hf, active_orbitals, config, *, policy="exhaustive"):
    """Exact classical packet growth; optional marks precede admissibility.

    ``config`` supplies the declared cutoffs/budgets. There is no reference
    energy argument. The early policy audits all groups every
    ``config["audit_interval"]`` decisions (five in the declared campaign)
    and falls back on empty/low-gain shortlists; it is a heuristic.
    """
    if policy not in ("exhaustive", "early"):
        raise ValueError("policy must be exhaustive or early")
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
        coefficients, residual_coefficients = frame.adjoint(raw), frame.adjoint(residual)
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
                vector = frame.combine(ids, coefficients[ids])
                direction = orthogonal_direction(vector, space.basis, cutoff=cutoff)
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
        choice, score = {"candidate": key}, scores[key]
    return {"policy": policy, "trajectory": rows, "decisions": decisions,
            "stop_reason": stop, "final_h_action_calls": space.h_action_calls,
            "final_candidates_scored": scored, "materialized_candidates": materialized,
            "solver_wall_seconds": time.perf_counter() - started}
