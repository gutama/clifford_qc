"""Empirical coverage of measured Ritz-value intervals.

An interval that is right on a typical run and wrong in the tail cannot be told
from a calibrated one by looking at a single measurement. The only check is to
repeat the measurement many times against a known target and count. This module
is that instrument: :func:`interval_replica` evaluates every interval on one
measurement, and :func:`summarize_coverage` counts over replicas.

Three interval constructions are evaluated:

- ``delta_normal``: the Gaussian delta-method interval of
  :func:`~clifford_qc.subspace.uncertainty.ritz_uncertainty`. It applies only to
  estimators that method accepts.
- ``bootstrap_percentile``: quantiles of the grouped bootstrap that reruns the
  whole estimator, as :func:`~clifford_qc.subspace.uncertainty.bootstrap_ritz`.
- ``bootstrap_basic``: the same draws reflected about the point estimate,
  ``[2 E - q_hi, 2 E - q_lo]``. A percentile interval repeats an estimator's
  bias; the basic interval subtracts it to first order.

A failed solve or a failed interval is counted, never dropped: the all-replica
coverage rate treats a failure as not covered, and the conditional rate states
its own denominator. Nothing here is a certificate; it measures calibration on
the systems it is run on.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from ..measurement.cache import GroupedWordCache
from ..measurement.session import SharedMeasurement
from .uncertainty import bootstrap_ritz_samples, ritz_uncertainty

METHODS = ("delta_normal", "bootstrap_percentile", "bootstrap_basic")
_EXPECTED_FAILURES = (ValueError, np.linalg.LinAlgError, ModuleNotFoundError)


def level_key(level: float) -> str:
    """The canonical dictionary key of a nominal coverage level."""
    return repr(float(level))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95
                    ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the normal approximation it stays inside ``[0, 1]`` and is sensible
    at zero or all successes, which coverage counts regularly reach.
    """
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials and at least one trial")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / trials
                         + z * z / (4.0 * trials * trials)) / denominator
    # The closed form returns the boundary only up to roundoff; pin it.
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == trials else min(1.0, centre + half)
    return lower, upper


def _validated_levels(levels: Sequence[float]) -> tuple[float, ...]:
    out = tuple(float(level) for level in levels)
    if not out or len(set(out)) != len(out) or any(not 0.0 < level < 1.0 for level in out):
        raise ValueError("levels must be distinct nominal coverages in (0, 1)")
    return out


def _validated_methods(methods: Sequence[str]) -> tuple[str, ...]:
    out = tuple(methods)
    if not out or len(set(out)) != len(out) or any(method not in METHODS for method in out):
        raise ValueError(f"methods must be distinct members of {METHODS}")
    return out


def _message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def delta_method_applies(solver_kwargs: dict) -> bool:
    """Whether the fixed-coefficient delta method accepts this estimator policy."""
    return (solver_kwargs.get("overlap_repair", "none") == "none"
            and solver_kwargs.get("pencil_angle") is None
            and solver_kwargs.get("ratio_error_budget") is None)


def interval_replica(shared: SharedMeasurement, cache: GroupedWordCache, *,
                     target: float, solver_kwargs: dict | None = None,
                     levels: Sequence[float] = (0.68, 0.95),
                     methods: Sequence[str] = METHODS,
                     bootstrap_replicates: int = 200, bootstrap_seed: int = 0,
                     root: int = 0) -> dict[str, Any]:
    """Every requested interval on one measurement, against a known ``target``.

    ``target`` is the value the intervals are meant to contain; the caller
    decides what that is. The solve uses ``solver_kwargs`` through
    :meth:`SharedMeasurement.solve`, and the bootstrap is bound to that point
    result, so it reruns exactly the same policy. One set of bootstrap draws
    serves both bootstrap constructions and every level.

    The returned row is JSON-safe. ``outcome`` is ``"ok"`` or
    ``"solve_failed"``. Each method reports ``"ok"``, ``"failed"`` or
    ``"not_applicable"``; an interval contains its target when
    ``lower <= target <= upper``. The stratification value is the smallest
    retained overlap eigenvalue of the solved metric, or the smallest measured
    normalized overlap eigenvalue for an estimator that retains no modes.
    """
    levels = _validated_levels(levels)
    methods = _validated_methods(methods)
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    kwargs = dict(solver_kwargs or {})
    row: dict[str, Any] = {
        "outcome": "ok", "solve_error": None, "estimate": None, "error": None,
        "shots": int(cache.total_shots), "retained_rank": None,
        "stratification_value": None, "stratification_source": None, "intervals": {},
    }
    try:
        point = shared.solve(cache, **kwargs)
        if not -len(point.energies) <= root < len(point.energies):
            raise ValueError(f"point solve retained {len(point.energies)} root(s); "
                             f"root {root} is unavailable")
        estimate = float(point.energies[root])
    except _EXPECTED_FAILURES as error:
        row.update({"outcome": "solve_failed", "solve_error": _message(error)})
        return row
    retained = point.resources.get("smallest_retained_overlap_eigenvalue")
    source = "smallest_retained_overlap_eigenvalue"
    if retained is None:
        retained = point.resources.get("overlap_eigenvalue_min")
        source = "measured_overlap_eigenvalue_min"
    row.update({
        "estimate": estimate, "error": estimate - target,
        "retained_rank": int(point.effective_rank),
        "stratification_value": None if retained is None else float(retained),
        "stratification_source": None if retained is None else source,
    })

    def entry(lower: float, upper: float) -> dict[str, Any]:
        return {"lower": float(lower), "upper": float(upper),
                "covered": bool(lower <= target <= upper)}

    if "delta_normal" in methods:
        if not delta_method_applies(kwargs):
            row["intervals"]["delta_normal"] = {
                "status": "not_applicable", "error": None, "std_error": None, "levels": {}}
        else:
            try:
                found = {level: ritz_uncertainty(shared, cache, point, root=root,
                                                 delta=1.0 - level) for level in levels}
                row["intervals"]["delta_normal"] = {
                    "status": "ok", "error": None,
                    "std_error": float(found[levels[0]].std_error or 0.0),
                    "levels": {level_key(level): entry(found[level].lower, found[level].upper)
                               for level in levels}}
            except _EXPECTED_FAILURES as error:
                row["intervals"]["delta_normal"] = {
                    "status": "failed", "error": _message(error), "std_error": None,
                    "levels": {}}
    wanted = [method for method in methods if method.startswith("bootstrap_")]
    if wanted:
        try:
            samples = bootstrap_ritz_samples(shared, cache, root=root, result=point,
                                             replicates=bootstrap_replicates,
                                             seed=bootstrap_seed)
        except _EXPECTED_FAILURES as error:
            for method in wanted:
                row["intervals"][method] = {"status": "failed", "error": _message(error),
                                            "std_error": None, "levels": {}}
        else:
            spread = float(np.std(samples, ddof=1))
            for method in wanted:
                found_levels = {}
                for level in levels:
                    low, high = np.quantile(samples, [(1.0 - level) / 2.0,
                                                      (1.0 + level) / 2.0])
                    if method == "bootstrap_basic":
                        low, high = 2.0 * estimate - high, 2.0 * estimate - low
                    found_levels[level_key(level)] = entry(low, high)
                row["intervals"][method] = {"status": "ok", "error": None,
                                            "std_error": spread, "levels": found_levels}
    return row


def _proportion(successes: int, trials: int, confidence: float) -> dict[str, Any]:
    if trials == 0:
        return {"n": 0, "covered": 0, "rate": None, "wilson": None}
    lower, upper = wilson_interval(successes, trials, confidence)
    return {"n": int(trials), "covered": int(successes), "rate": successes / trials,
            "wilson": [lower, upper]}


def _counts(messages: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for message in messages:
        out[message] = out.get(message, 0) + 1
    return dict(sorted(out.items()))


def _method_summary(rows: Sequence[dict], method: str, levels: Sequence[float],
                    confidence: float) -> dict[str, Any]:
    """Coverage of one method over ``rows``; failures count against all-replica rates."""
    states: list[dict[str, Any]] = [
        row["intervals"].get(method, {"status": "failed", "levels": {},
                                      "error": "interval missing"})
        if row["outcome"] == "ok" else {"status": "solve_failed", "levels": {}}
        for row in rows]
    solved = [state for state in states if state["status"] != "solve_failed"]
    if solved and all(state["status"] == "not_applicable" for state in solved):
        return {"applicable": False}
    good = [(row, state) for row, state in zip(rows, states) if state["status"] == "ok"]
    out: dict[str, Any] = {
        "applicable": True,
        "interval_failures": sum(state["status"] == "failed" for state in states),
        "interval_failure_messages": _counts([state["error"] for state in states
                                              if state["status"] == "failed"]),
        "levels": {},
    }
    for level in levels:
        key = level_key(level)
        entries = [(row, state["levels"][key]) for row, state in good]
        covered = sum(item["covered"] for _, item in entries)
        widths = [item["upper"] - item["lower"] for _, item in entries]
        target_above = sum(item["upper"] < row["estimate"] - row["error"]
                           for row, item in entries)
        target_below = sum(row["estimate"] - row["error"] < item["lower"]
                           for row, item in entries)
        out["levels"][key] = {
            "nominal": float(level),
            "all_replicas": _proportion(covered, len(rows), confidence),
            "successful_intervals": _proportion(covered, len(entries), confidence),
            "target_above_interval": int(target_above),
            "target_below_interval": int(target_below),
            "median_width": float(np.median(widths)) if widths else None,
        }
    return out


def summarize_coverage(rows: Sequence[dict], *, levels: Sequence[float] = (0.68, 0.95),
                       methods: Sequence[str] = METHODS,
                       strata_edges: Sequence[float] = (),
                       confidence: float = 0.95) -> dict[str, Any]:
    """Count coverage, failures and estimator error over replica rows.

    For every method and level two rates are reported with Wilson intervals at
    ``confidence``. ``all_replicas`` divides by every replica, so a failed solve
    or failed interval counts as not covered. ``successful_intervals`` divides
    by the replicas whose interval exists and says so in its ``n``.

    ``strata_edges`` are fixed, increasing cut points on each row's
    stratification value. Stratum ``k`` holds values in
    ``[edge[k-1], edge[k])``; the first is unbounded below and the last above.
    Rows whose solve failed have no value and are reported separately.
    """
    levels = _validated_levels(levels)
    methods = _validated_methods(methods)
    edges = tuple(float(edge) for edge in strata_edges)
    if any(not math.isfinite(edge) for edge in edges) or \
            any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("strata_edges must be finite and strictly increasing")
    rows = list(rows)
    if not rows:
        raise ValueError("no replica rows to summarize")
    solved = [row for row in rows if row["outcome"] == "ok"]
    errors = np.array([row["error"] for row in solved], dtype=float)
    estimate = None
    if solved:
        estimate = {
            "n": len(solved),
            "mean_error": float(np.mean(errors)),
            "median_abs_error": float(np.median(np.abs(errors))),
            "p90_abs_error": float(np.quantile(np.abs(errors), 0.9)),
            "max_abs_error": float(np.max(np.abs(errors))),
            "std": float(np.std(errors, ddof=1)) if len(solved) > 1 else None,
            "below_target": int(np.sum(errors < 0.0)),
        }
    strata = []
    bounds = (None,) + edges + (None,)
    for lower, upper in zip(bounds, bounds[1:]):
        members = [row for row in solved
                   if row["stratification_value"] is not None
                   and (lower is None or row["stratification_value"] >= lower)
                   and (upper is None or row["stratification_value"] < upper)]
        strata.append({
            "lower": lower, "upper": upper, "n": len(members),
            "methods": ({method: _method_summary(members, method, levels, confidence)
                         for method in methods} if members else {}),
        })
    return {
        "replicas": len(rows),
        "solve_failures": len(rows) - len(solved),
        "solve_failure_messages": _counts([row["solve_error"] for row in rows
                                           if row["outcome"] != "ok"]),
        "estimate": estimate,
        "methods": {method: _method_summary(rows, method, levels, confidence)
                    for method in methods},
        "strata": strata,
        "unstratified_solved": sum(row["stratification_value"] is None for row in solved),
        "confidence": float(confidence),
    }


__all__ = ["METHODS", "delta_method_applies", "interval_replica", "level_key",
           "summarize_coverage", "wilson_interval"]
