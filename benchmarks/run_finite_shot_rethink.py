#!/usr/bin/env python3
"""Where the finite-shot error actually comes from, and what removes it.

``run_finite_shot_optimization.py`` fixed one projected TFIM subspace and one
physical shot budget and crossed *acquisition* (uniform against covariance-aware
group allocation) with *overlap regularization* (fixed cutoff against
shot-calibrated cutoffs).  Its verdict was that the tail, not the median,
dominates the error, and that no arm removed the tail without paying tens of
millihartree in truncation bias.

This experiment holds acquisition fixed at uniform and crosses the two stages
that come after the shots are spent, which the earlier study never varied:

*Reconstruction.*  ``assigned`` reads each word from the single QWC group the
partition gave it.  ``pooled`` reads it from every group whose basis records
it, weighted by shot count.  Identical circuits, identical shots, identical
histograms -- only which of the recorded outcomes the estimator is allowed to
look at differs.  The partition is a *scheduling* device, and using it as an
*estimation* device discards readings already paid for.

*Rank rule.*  ``fixed`` and ``calibrated`` decide the retained rank from the
overlap spectrum alone.  ``selected`` scores each attainable rank by
``E_hat(k) + gamma*sigma_hat(k)`` and takes the minimizer, which asks about the
quantity at risk (the Ritz value) rather than about ``S``.  ``ridge`` replaces
the rank decision with a smooth per-mode damping and is included because it is
the obvious thing to try; the record shows it is also wrong.

Every arm at one budget consumes the *same* cache per replica, so the whole
table is a comparison of estimators on shared data, not of measurement budgets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import SharedMeasurement, ritz_functional

from run_finite_shot_optimization import _setting


# ``selected`` prices one mode's extra noise against its extra lowering at two
# sigma.  Declared here rather than swept, so the record carries one rule.
DEFAULT_GAMMA = 2.0


def _sessions(bank, shared):
    """The two reconstructions, over the *same* grouping.

    Passing ``shared.groups`` explicitly keeps the partition identical rather
    than relying on the grouping cache to return the same one twice.
    """
    return {
        "assigned": shared,
        "pooled": SharedMeasurement(bank, groups=shared.groups, pooling="shots"),
    }


def _rules(gamma: float):
    return {
        "fixed": lambda session, cache: session.solve(cache),
        "calibrated": lambda session, cache: session.solve(
            cache, calibrate_overlap=True, overlap_policy="per_mode"),
        "ridge": lambda session, cache: session.solve(
            cache, calibrate_overlap=True, overlap_policy="per_mode",
            overlap_regularizer="ridge"),
        "selected": lambda session, cache: session.solve_selected_rank(
            cache, gamma=gamma),
    }


def _solve(rule, session, cache, diagnostics):
    try:
        result = rule(session, cache)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"failure": type(exc).__name__, "message": str(exc)}
    return {
        "failure": None,
        "energy": result.ground_energy,
        "rank": result.effective_rank,
        "condition_number": result.condition_number,
        "overlap_negative_modes": result.resources["overlap_negative_modes"],
        **diagnostics,
    }


def _summary(rows, exact):
    successes = [row for row in rows if row["failure"] is None]
    if not successes:
        return {"replicas": len(rows), "successful_solves": 0,
                "failures": dict(sorted(Counter(
                    row["failure"] for row in rows
                    if row["failure"] is not None).items()))}
    errors = np.asarray([row["energy"] - exact.ground_energy
                         for row in successes], dtype=float)
    absolute = np.abs(errors)
    return {
        "replicas": len(rows),
        "successful_solves": len(successes),
        "failures": dict(sorted(Counter(
            row["failure"] for row in rows if row["failure"] is not None).items())),
        "bias_millihartree": 1000.0 * float(np.mean(errors)),
        "rmse_millihartree": 1000.0 * float(np.sqrt(np.mean(errors ** 2))),
        "median_absolute_error_millihartree": 1000.0 * float(np.median(absolute)),
        "p95_absolute_error_millihartree": 1000.0 * float(np.quantile(absolute, 0.95)),
        "max_absolute_error_millihartree": 1000.0 * float(np.max(absolute)),
        "below_exact_count": int(np.sum(errors < 0.0)),
        "catastrophic_error_count": int(np.sum(absolute > 0.1)),
        "exact_rank_count": sum(row["rank"] == exact.effective_rank
                                for row in successes),
        "rank_histogram": {str(key): value for key, value in sorted(
            Counter(row["rank"] for row in successes).items())},
        "median_condition_number": float(np.median(
            [row["condition_number"] for row in successes])),
        "median_exact_ritz_variance": float(np.median(
            [row["exact_ritz_variance"] for row in successes])),
        "median_projected_matrix_variance_sum": float(np.median(
            [row["projected_matrix_variance_sum"] for row in successes])),
    }


def run(replicas: int, budgets: tuple[int, ...], seed: int,
        gamma: float = DEFAULT_GAMMA):
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if not budgets:
        raise ValueError("at least one shots-per-group budget is required")
    bank, shared, exact = _setting()
    sessions = _sessions(bank, shared)
    rules = _rules(gamma)

    # Evaluated on every cache purely to describe acquisition quality; the exact
    # Ritz coefficients never enter any arm's estimate.
    exact_target = ritz_functional(bank, exact.indices, exact.ritz_vector(0),
                                   exact.ground_energy)
    matrix_targets = shared.matrix_functionals()

    arms: dict[str, dict] = {}
    reader_counts: dict[str, float] = {}
    for shots_per_group in budgets:
        raw = {f"{estimator}_{rule}": []
               for estimator in sessions for rule in rules}
        for replica in range(replicas):
            backend = FiniteShotBackend(seed + replica)
            for estimator, session in sessions.items():
                # One cache per (estimator, replica): the two estimators are fed
                # the same seed, so they see the same outcomes, and every rule
                # within an estimator shares one cache.
                backend.reseed(seed + replica)
                cache = session.measure(backend, shots_per_group)
                diagnostics = {
                    "exact_ritz_variance": exact_target.variance(cache),
                    "projected_matrix_variance_sum": sum(
                        target.variance(cache) for target in matrix_targets),
                }
                if estimator == "pooled" and replica == 0:
                    reader_counts[str(shots_per_group)] = float(np.mean(
                        [cache.reader_count(word.code) for word in session.words]))
                for name, rule in rules.items():
                    raw[f"{estimator}_{name}"].append(
                        _solve(rule, session, cache, diagnostics))
        arms[str(shots_per_group)] = {
            name: _summary(rows, exact) for name, rows in raw.items()}

    return stamp_record({
        "schema": "clifford_qc.finite_shot_rethink.v1",
        "evidence": "heuristic",
        "interpretation": (
            "Monte Carlo diagnostic on one fixed four-qubit projected subspace. "
            "Every arm at a budget reads the same shots, so differences are "
            "estimator differences, not measurement-budget differences. Not a "
            "hardware result and not a finite-sample coverage certificate: the "
            "same data supply the matrices, the retained rank, and the error "
            "bar, so no arm's value is a variational bound on E_0."
        ),
        "setting": {
            "model": "four-qubit TFIM (J=h=1)",
            "basis_size": len(exact.basis_labels),
            "exact_effective_rank": exact.effective_rank,
            "exact_condition_number": exact.condition_number,
            "exact_projected_ground_energy": exact.ground_energy,
            "measurement_words": len(shared.words),
            "qwc_groups": len(shared.groups),
            "mean_reading_groups_per_word": reader_counts,
        },
        "protocol": {
            "allocation": "uniform shots per QWC group",
            "shots_per_group": list(budgets),
            "physical_shots_per_replica": [
                shots * len(shared.groups) for shots in budgets],
            "replicas": replicas,
            "seed": seed,
            "rank_selection_gamma": gamma,
            "estimators": {
                "assigned": "word read from its assigned QWC group only",
                "pooled": ("word read from every group whose basis records it, "
                           "weighted by that group's shots"),
            },
            "rank_rules": {
                "fixed": "numerical overlap cutoff",
                "calibrated": "per-mode shot-calibrated overlap cutoff",
                "ridge": "per-mode shot-calibrated overlap ridge (no truncation)",
                "selected": "rank minimizing E_hat + gamma * sigma_hat",
            },
        },
        "arms": arms,
    })


def _table(record) -> str:
    lines = []
    for budget, arms in record["arms"].items():
        shots = int(budget) * record["setting"]["qwc_groups"]
        lines.append(f"\n{budget} shots/group ({shots} setting-shots per replica)")
        lines.append(f"{'estimator':10s} {'rank rule':11s} {'median':>8s} {'p95':>9s} "
                     f"{'max':>10s} {'RMSE':>10s} {'bias':>9s} {'>0.1Ha':>7s} "
                     f"{'rank*':>6s}")
        for name, summary in arms.items():
            estimator, rule = name.split("_", 1)
            if not summary.get("successful_solves"):
                lines.append(f"{estimator:10s} {rule:11s}   no successful solves")
                continue
            lines.append(
                f"{estimator:10s} {rule:11s} "
                f"{summary['median_absolute_error_millihartree']:8.2f} "
                f"{summary['p95_absolute_error_millihartree']:9.2f} "
                f"{summary['max_absolute_error_millihartree']:10.2f} "
                f"{summary['rmse_millihartree']:10.2f} "
                f"{summary['bias_millihartree']:9.2f} "
                f"{summary['catastrophic_error_count']:7d} "
                f"{summary['exact_rank_count']:6d}")
    lines.append("\nErrors in mHa against the exact projected energy; "
                 "rank* counts replicas recovering the exact effective rank.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=200)
    parser.add_argument("--shots-per-group", type=int, nargs="+",
                        default=[250, 2000])
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument(
        "--out", type=Path,
        default=Path("benchmarks/reference_results/finite_shot_rethink.json"))
    args = parser.parse_args()
    record = run(args.replicas, tuple(args.shots_per_group), args.seed,
                 gamma=args.gamma)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(_table(record))
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
