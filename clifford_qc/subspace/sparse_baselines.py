"""Exact classical comparison arms on the sparse singlet frame.

``run_selected_ci`` adds one singlet-frame column per step by exact
two-dimensional energy lowering, computing each candidate's H action once.
``run_singlet_davidson`` adds the frame projection of the packets'
preconditioned residual as one direction per step: every packet group merged.
Both share the packet solver's initial span, Ritz solve, cutoffs and stopping
rules. Neither is an optimized selected CI nor a bounded-support,
measurement-priced method.
"""
from __future__ import annotations

import time

import numpy as np

from .action_space import ActionSpace, energy_lowering, orthogonal_direction

NEAR_TIE = 1e-8  # Scores this close to a step's best are recorded for exact tie-break replay.


def _start(apply, diagonal, frame, hf, active_orbitals, config):
    diagonal = np.asarray(diagonal)
    if diagonal.shape != (frame.shape[0],) or not np.isfinite(diagonal).all() or np.iscomplexobj(diagonal):
        raise ValueError("finite real sector diagonal required")
    positive = ("preconditioner_shift", "residual_tolerance", "direction_absolute_tolerance",
                "orthogonality_tolerance", "spin_tolerance", "score_tie_tolerance")
    if (any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive)
            or not np.isfinite(config["direction_relative_tolerance"])
            or config["direction_relative_tolerance"] < 0
            or any(type(config[k]) is not int or config[k] < 1 for k in
                   ("maximum_columns", "maximum_growth_steps"))):
        raise ValueError("invalid baseline cutoff or budget")
    selected, _ = frame.active_columns(hf, active_orbitals)
    if len(selected) > config["maximum_columns"]:
        raise ValueError("initial span exceeds column budget")
    return diagonal, selected, ActionSpace(apply, frame.columns(selected), tolerance=config["orthogonality_tolerance"])


def _grow(space, frame, config, selected, choose, work, observe, started):
    """Shared prefix loop; ``choose`` returns (key, direction, action, score, decision) or None."""
    rows, decisions, observations = [], [], []
    choice, score, stop = {"initial_csf_indices": selected}, None, None
    for step in range(config["maximum_growth_steps"] + 1):
        energy, state, residual = space.solve()
        spin_norm = float(np.linalg.norm(frame.spin @ state))
        if spin_norm > config["spin_tolerance"]:
            raise ValueError("Ritz state escaped singlet sector")
        rows.append({"step": step, "energy": energy, "residual_norm": float(np.linalg.norm(residual)),
            "spin_action_norm": spin_norm, "dimension": space.basis.shape[1],
            "h_action_calls": space.h_action_calls, "candidates_scored": work["scored"],
            "materialized_candidates": work["materialized"], "selection": choice,
            "selection_score": score, "retained_payload_bytes": space.payload_bytes,
            "wall_seconds": time.perf_counter() - started})
        if observe is not None:
            observations.append(observe(rows[-1], space.basis, state))
        stop = ("residual_tolerance" if rows[-1]["residual_norm"] <= config["residual_tolerance"]
                else "column_ceiling" if space.basis.shape[1] >= config["maximum_columns"]
                else "growth_step_ceiling" if step == config["maximum_growth_steps"] else None)
        if stop:
            break
        picked = choose(energy, state, residual)
        if picked is None:
            stop = "no_admissible_direction"
            break
        key, direction, acted, score, decision = picked
        decisions.append({"step": step, **decision})
        space.extend(direction[:, None], actions=acted[:, None])
        choice = {"candidate": key}
    result = {"trajectory": rows, "decisions": decisions, "stop_reason": stop,
              "final_h_action_calls": space.h_action_calls, "final_candidates_scored": work["scored"],
              "materialized_candidates": work["materialized"], "solver_wall_seconds": time.perf_counter() - started}
    if observe is not None:
        result["observations"] = observations
    return result


def run_selected_ci(apply, diagonal, frame, hf, active_orbitals, config, *, observe=None):
    """Greedy spin-adapted selected CI: one frame column per step.

    Every unselected column is scored by its exact two-dimensional lowering
    against the current Ritz state, ties within ``score_tie_tolerance`` by
    column index. Candidate H actions are computed once, on first scoring, and
    retained until selected; that work and storage are charged. Frame columns
    are orthonormal, so a candidate is already orthogonal to the span.
    """
    started = time.perf_counter()
    diagonal, selected, space = _start(apply, diagonal, frame, hf, active_orbitals, config)
    keys = np.array([j for j in range(frame.shape[1]) if j not in set(selected)], dtype=np.int64)
    tolerance, work, cache = config["score_tie_tolerance"], {"scored": 0, "materialized": 0}, {}

    def choose(energy, state, residual):
        if not cache:
            actions, diagonal_terms = np.empty((frame.shape[0], keys.size), complex), np.empty(keys.size)
            for i, j in enumerate(keys):
                column = frame.columns([int(j)])[:, 0]
                actions[:, i] = space.action(column)
                diagonal_terms[i] = np.vdot(column, actions[:, i]).real
            cache.update(actions=actions, diagonal=diagonal_terms, live=np.ones(keys.size, bool))
            work["materialized"] += int(keys.size)
        live = cache["live"]
        if not live.any():
            return None
        couplings = cache["actions"].T @ state.conj()  # vdot(state, action) for every candidate.
        gap = cache["diagonal"] - energy
        values = np.maximum(0.0, (np.hypot(gap, 2 * abs(couplings)) - gap) / 2)
        best = float(values[live].max())
        work["scored"] += int(live.sum())
        # Keys ascend, so the first live near-best position is the minimum key.
        position = int(np.flatnonzero(live & (values >= best - tolerance))[0])
        near = {str(int(keys[i])): float(values[i]) for i in np.flatnonzero(live & (values >= best - NEAR_TIE))}
        admissible = int(live.sum())
        live[position] = False
        key = int(keys[position])
        return key, frame.columns([key])[:, 0], cache["actions"][:, position], float(values[position]), {
            "selected": key, "score": float(values[position]), "admissible": admissible, "near_ties": near}

    return {"arm": "selected_ci", **_grow(space, frame, config, selected, choose, work, observe, started)}


def run_singlet_davidson(apply, diagonal, frame, hf, active_orbitals, config, *, observe=None):
    """Add the frame projection F F^T of the preconditioned residual each step.

    The preconditioner, shift floor and direction cutoffs are the packets'.
    This is the packet direction with every group merged, so it measures what
    restricting each step to one group costs; it is a reference, not a
    bounded-support generator.
    """
    started = time.perf_counter()
    diagonal, selected, space = _start(apply, diagonal, frame, hf, active_orbitals, config)
    shift, work = config["preconditioner_shift"], {"scored": 0, "materialized": 0}
    columns = np.arange(frame.shape[1])

    def choose(energy, state, residual):
        raw = residual / np.maximum(diagonal - energy + shift, shift)
        cutoff = max(config["direction_absolute_tolerance"],
                     config["direction_relative_tolerance"] * np.linalg.norm(raw))
        work["materialized"] += 1
        direction = orthogonal_direction(frame.combine(columns, frame.adjoint(raw)), space.basis, cutoff=cutoff)
        if direction is None:
            return None
        acted = space.action(direction)
        work["scored"] += 1
        score = energy_lowering(energy, np.vdot(direction, acted).real, np.vdot(state, acted))
        return "residual", direction, acted, score, {"score": score}

    return {"arm": "davidson", **_grow(space, frame, config, selected, choose, work, observe, started)}
