#!/usr/bin/env python3
"""Matched finite-shot test of group allocation and overlap regularization.

The experiment fixes one projected TFIM subspace and one *physical* shot
budget.  It then crosses two acquisition policies

* uniform shots per QWC measurement setting;
* a uniform pilot followed by covariance-aware Neyman allocation for the
  pilot Ritz functional;

with fixed and shot-calibrated overlap truncation.  Fixed/calibrated solver
arms reuse the same cache, so regularization is compared without charging a
second measurement budget.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement import variance_optimal_group_plan
from clifford_qc.models.spin import tfim
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import (
    MatrixElementBank,
    SharedMeasurement,
    identity_generator,
    krylov_response,
    pauli_orbit,
    ritz_functional,
)


def _setting():
    model = tfim(4, J=1.0, h=1.0)
    reference = ExactMVBackend().state(model.reference, ())
    words = [operator.word for operator in odd_y_filter(local_pool(4))]
    bank = MatrixElementBank(
        reference,
        model.hamiltonian,
        [identity_generator(4)]
        + pauli_orbit(words[:6])
        + krylov_response(model.hamiltonian, 2),
    )
    return bank, SharedMeasurement(bank), bank.solve()


def _uniform_cache(shared, seed: int, total_budget: int):
    groups = len(shared.groups)
    if total_budget % groups:
        raise ValueError("uniform total budget must be divisible by the group count")
    cache = shared.measure(FiniteShotBackend(seed), total_budget // groups)
    return cache, {
        "pilot_shots_per_group": 0,
        "adaptive_shots": 0,
        "allocation_target": "uniform",
    }


def _adaptive_cache(shared, seed: int, total_budget: int,
                    pilot_shots_per_group: int):
    groups = len(shared.groups)
    pilot_budget = pilot_shots_per_group * groups
    if pilot_budget >= total_budget:
        raise ValueError("pilot budget must be smaller than the total budget")
    backend = FiniteShotBackend(seed)
    cache = shared.measure(backend, pilot_shots_per_group)
    pilot = shared.solve(cache, calibrate_overlap=True)
    target = ritz_functional(
        shared.bank,
        pilot.indices,
        pilot.ritz_vector(0),
        pilot.ground_energy,
    )
    targets = [target, *shared.matrix_functionals()]
    plan = variance_optimal_group_plan(
        shared.groups,
        cache,
        targets,
        total_budget - pilot_budget,
        min_shots=0,
    )
    shared.measure_plan(backend, plan, cache)
    spent = sum(next(iter({plan.get(word.code, 0) for word in group}))
                for group in shared.groups)
    if spent != total_budget - pilot_budget or cache.total_shots != total_budget:
        raise AssertionError("adaptive allocation did not spend its physical budget")
    added = [next(iter({plan.get(word.code, 0) for word in group}))
             for group in shared.groups]
    return cache, {
        "pilot_shots_per_group": pilot_shots_per_group,
        "adaptive_shots": spent,
        "allocation_target": (
            "sum of projected S/H entry variances and pilot Ritz "
            "first-order variance"
        ),
        "additional_group_shots_min": min(added),
        "additional_group_shots_max": max(added),
        "additional_group_shots_nonzero": sum(value > 0 for value in added),
    }


def _solve(shared, cache, calibrated: bool, variance_diagnostics):
    try:
        result = shared.solve(cache, calibrate_overlap=calibrated)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"failure": type(exc).__name__, "message": str(exc)}
    return {
        "failure": None,
        "energy": result.ground_energy,
        "rank": result.effective_rank,
        "condition_number": result.condition_number,
        "overlap_threshold": result.resources["overlap_threshold"],
        "overlap_negative_modes": result.resources["overlap_negative_modes"],
        **variance_diagnostics,
    }


def _summary(rows, exact):
    successes = [row for row in rows if row["failure"] is None]
    errors = np.asarray(
        [row["energy"] - exact.ground_energy for row in successes], dtype=float)
    absolute = np.abs(errors)
    thresholds = np.asarray(
        [row["overlap_threshold"] for row in successes], dtype=float)
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
        "rank_histogram": {
            str(key): value for key, value in sorted(Counter(
                row["rank"] for row in successes).items())
        },
        "median_condition_number": float(np.median(
            [row["condition_number"] for row in successes])),
        "median_overlap_threshold": float(np.median(thresholds)),
        "median_exact_ritz_variance": float(np.median(
            [row["exact_ritz_variance"] for row in successes])),
        "median_projected_matrix_variance_sum": float(np.median(
            [row["projected_matrix_variance_sum"] for row in successes])),
    }


def run(replicas: int, shots_per_group: int, pilot_shots_per_group: int,
        seed: int):
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    bank, shared, exact = _setting()
    total_budget = shots_per_group * len(shared.groups)
    raw = {
        "uniform_fixed": [],
        "uniform_calibrated": [],
        "group_optimal_fixed": [],
        "group_optimal_calibrated": [],
    }
    exact_target = ritz_functional(
        bank, exact.indices, exact.ritz_vector(0), exact.ground_energy)
    matrix_targets = shared.matrix_functionals()

    def variance_diagnostics(cache):
        return {
            # Exact coefficients are used only to evaluate acquisition quality,
            # never to construct the adaptive plan.
            "exact_ritz_variance": exact_target.variance(cache),
            "projected_matrix_variance_sum": sum(
                target.variance(cache) for target in matrix_targets),
        }

    allocation_diagnostics = []
    for replica in range(replicas):
        sample_seed = seed + replica
        uniform, _ = _uniform_cache(shared, sample_seed, total_budget)
        adaptive, diagnostics = _adaptive_cache(
            shared, sample_seed, total_budget, pilot_shots_per_group)
        allocation_diagnostics.append(diagnostics)
        uniform_variance = variance_diagnostics(uniform)
        adaptive_variance = variance_diagnostics(adaptive)
        raw["uniform_fixed"].append(
            _solve(shared, uniform, False, uniform_variance))
        raw["uniform_calibrated"].append(
            _solve(shared, uniform, True, uniform_variance))
        raw["group_optimal_fixed"].append(
            _solve(shared, adaptive, False, adaptive_variance))
        raw["group_optimal_calibrated"].append(
            _solve(shared, adaptive, True, adaptive_variance))

    return stamp_record({
        "schema": "clifford_qc.finite_shot_optimization.v1",
        "evidence": "heuristic",
        "interpretation": (
            "Monte Carlo diagnostic on one fixed four-qubit projected subspace; "
            "not a hardware result or finite-sample coverage certificate."
        ),
        "setting": {
            "model": "four-qubit TFIM (J=h=1)",
            "basis_size": len(exact.basis_labels),
            "exact_effective_rank": exact.effective_rank,
            "exact_condition_number": exact.condition_number,
            "exact_projected_ground_energy": exact.ground_energy,
            "measurement_words": len(shared.words),
            "qwc_groups": len(shared.groups),
        },
        "budget": {
            "physical_shots_per_replica": total_budget,
            "uniform_shots_per_group": shots_per_group,
            "adaptive_pilot_shots_per_group": pilot_shots_per_group,
            "adaptive_pilot_physical_shots": (
                pilot_shots_per_group * len(shared.groups)),
        },
        "allocation_diagnostics": {
            key: float(np.median([row[key] for row in allocation_diagnostics]))
            for key in (
                "additional_group_shots_min",
                "additional_group_shots_max",
                "additional_group_shots_nonzero",
            )
        },
        "arms": {name: _summary(rows, exact) for name, rows in raw.items()},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=200)
    parser.add_argument("--shots-per-group", type=int, default=2000)
    parser.add_argument("--pilot-shots-per-group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reference_results/finite_shot_optimization.json"),
    )
    args = parser.parse_args()
    record = run(
        args.replicas,
        args.shots_per_group,
        args.pilot_shots_per_group,
        args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["arms"], indent=2, sort_keys=True))
    print(args.out)


if __name__ == "__main__":
    main()
