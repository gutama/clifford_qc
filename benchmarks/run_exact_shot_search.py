#!/usr/bin/env python3
"""R1 oracle-reference nonlinear shot-to-target search.

The hierarchy's schema-v3 ledger propagates the exact-bank Ritz Jacobian and
is intentionally asymptotic.  This benchmark closes the other R1 tier: it
samples the actual joint bitstrings from every synthesized Clifford setting,
reconstructs the measured ``(S, H)`` pencil, reruns the data-dependent
``E + 2 sigma`` rank sweep, and compares the resulting energy to the full
exact reference.

The target tier is ``exact`` because the comparator is an exact oracle.  The
Monte Carlo uncertainty on the shot search is a bootstrap diagnostic and is
therefore ``heuristic``; it is not a finite-sample energy certificate or a
hardware prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np

from clifford_qc.measurement.cache import GroupedWordCache
from clifford_qc.measurement.compiled import CompiledMeasurementSampler
from clifford_qc.measurement.cost import (
    cost_schedule,
    inflate_shots_for_fidelity,
)
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.reproducibility import stamp_record

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_clifford_hierarchy import (
        ACCURACY_TARGET_MILLIHARTREE,
        BLOCK_SIZES,
        SYSTEMS,
        _compiled_settings,
        _partition,
        _synthesize,
        load_device_cards,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_clifford_hierarchy import (
        ACCURACY_TARGET_MILLIHARTREE,
        BLOCK_SIZES,
        SYSTEMS,
        _compiled_settings,
        _partition,
        _synthesize,
        load_device_cards,
    )


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference_results" / "exact_shot_search.json"
SEARCH_ENDPOINTS = (64, 256, 1_024, 4_096, 16_384, 65_536)
EXPLORATORY_REPLICAS = 30
CONFIRMATORY_REPLICAS = 100
EXPLORATORY_SEED = 260_813_000
CONFIRMATORY_SEED = 260_913_000
BOOTSTRAP_SEED = 261_013_000
BOOTSTRAP_REPLICATES = 10_000
DELTA = 0.05
RANK_SELECTION_GAMMA = 2.0
ESTIMATORS = ("single_assignment", "pooled")


def _summary(rows: list[dict], exact_energy: float, *, bootstrap_seed: int) -> dict:
    """One endpoint's oracle-error summary and one-sided bootstrap RMSE UCB."""
    successes = [row for row in rows if row["failure"] is None]
    failures = Counter(row["failure"] for row in rows if row["failure"] is not None)
    base = {
        "replicas": len(rows),
        "successful_solves": len(successes),
        "failures": dict(sorted(failures.items())),
        "zero_failure_gate": len(successes) == len(rows),
    }
    if not successes:
        return {
            **base,
            "bias_millihartree": None,
            "rmse_millihartree": None,
            "rmse_one_sided_95pct_upper_millihartree": None,
            "median_absolute_error_millihartree": None,
            "p95_absolute_error_millihartree": None,
            "max_absolute_error_millihartree": None,
            "chemical_accuracy_fraction": 0.0,
            "below_exact_count": 0,
            "rank_histogram": {},
            "passes_target": False,
        }
    errors = 1_000.0 * np.asarray(
        [row["energy"] - exact_energy for row in successes], dtype=float
    )
    absolute = np.abs(errors)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(errors), size=(BOOTSTRAP_REPLICATES, len(errors)), endpoint=False
    )
    bootstrap_rmse = np.sqrt(np.mean(np.square(errors[indices]), axis=1))
    upper = float(np.quantile(bootstrap_rmse, 1.0 - DELTA))
    passes = bool(
        len(successes) == len(rows)
        and upper <= ACCURACY_TARGET_MILLIHARTREE
    )
    return {
        **base,
        "bias_millihartree": float(np.mean(errors)),
        "rmse_millihartree": rmse,
        "rmse_one_sided_95pct_upper_millihartree": upper,
        "median_absolute_error_millihartree": float(np.median(absolute)),
        "p95_absolute_error_millihartree": float(np.quantile(absolute, 0.95)),
        "max_absolute_error_millihartree": float(np.max(absolute)),
        "chemical_accuracy_fraction": float(np.mean(
            absolute <= ACCURACY_TARGET_MILLIHARTREE
        )),
        "below_exact_count": int(np.sum(errors < 0.0)),
        "rank_histogram": {
            str(rank): count for rank, count in sorted(Counter(
                row["rank"] for row in successes
            ).items())
        },
        "passes_target": passes,
    }


def _solve(shared: SharedMeasurement, cache: GroupedWordCache) -> dict:
    try:
        result = shared.solve_selected_rank(cache, gamma=RANK_SELECTION_GAMMA)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"failure": type(exc).__name__}
    return {
        "failure": None,
        "energy": float(result.ground_energy),
        "rank": int(result.effective_rank),
    }


def _run_endpoints(
    *,
    reference,
    bank,
    result,
    settings,
    requested: dict[str, tuple[int, ...]],
    replicas: int,
    seed: int,
    block_size: int,
) -> dict[str, dict[int, list[dict]]]:
    """Run nested endpoint samples; estimator arms share every outcome."""
    endpoints = sorted(set(endpoint for values in requested.values() for endpoint in values))
    if not endpoints:
        return {estimator: {} for estimator in ESTIMATORS}
    by_code = {word.code: word for word in bank.words(result.indices)}
    groups = [[by_code[code] for code in setting.assigned_word_codes if code != 0]
              for setting in settings]
    sessions = {
        "single_assignment": SharedMeasurement(
            bank, result.indices, groups=groups, pooling="assigned",
        ),
        "pooled": SharedMeasurement(
            bank, result.indices, groups=groups, pooling="shots",
        ),
    }

    raw = {
        estimator: {endpoint: [] for endpoint in requested[estimator]}
        for estimator in ESTIMATORS
    }
    sampler = CompiledMeasurementSampler(seed)
    sampler.prepare(reference, settings)
    for replica in range(replicas):
        sampler.reseed(seed + 100_000 * block_size + replica)
        caches = {
            "single_assignment": GroupedWordCache(reference.n, pooling="assigned"),
            "pooled": GroupedWordCache(reference.n, pooling="shots"),
        }
        previous = 0
        for endpoint in endpoints:
            batch = sampler.sample_from_state(reference, settings, endpoint - previous)
            previous = endpoint
            for cache in caches.values():
                cache.add_batch(batch)
            for estimator in ESTIMATORS:
                if endpoint in raw[estimator]:
                    raw[estimator][endpoint].append(
                        _solve(sessions[estimator], caches[estimator])
                    )
    return raw


def _summaries(raw: dict, exact_energy: float, *, block_size: int,
               seed_offset: int) -> dict[str, list[dict]]:
    output = {}
    for estimator_index, estimator in enumerate(ESTIMATORS):
        output[estimator] = []
        for endpoint, rows in sorted(raw[estimator].items()):
            summary = _summary(
                rows,
                exact_energy,
                bootstrap_seed=(
                    BOOTSTRAP_SEED + seed_offset + 1_000_000 * block_size
                    + 100_000 * estimator_index + endpoint
                ),
            )
            output[estimator].append({
                "effective_shots_per_setting": endpoint,
                **summary,
            })
    return output


def _exploratory_bracket(rows: list[dict]) -> tuple[int | None, int | None]:
    """First endpoint whose pass persists through all larger endpoints."""
    passing_index = next(
        (index for index in range(len(rows))
         if all(row["passes_target"] for row in rows[index:])),
        None,
    )
    if passing_index is None:
        return rows[-1]["effective_shots_per_setting"], None
    high = rows[passing_index]["effective_shots_per_setting"]
    low = (None if passing_index == 0
           else rows[passing_index - 1]["effective_shots_per_setting"])
    return low, high


def _confirm_result(exploration: list[dict], confirmation: list[dict]) -> dict:
    low, high = _exploratory_bracket(exploration)
    by_endpoint = {row["effective_shots_per_setting"]: row for row in confirmation}
    if high is None:
        return {
            "status": "not_bracketed_within_search_grid",
            "last_tested_effective_shots_per_setting": low,
            "confirmed_failing_effective_shots_per_setting": None,
            "confirmed_passing_effective_shots_per_setting": None,
        }
    high_passes = by_endpoint[high]["passes_target"]
    low_fails = low is None or not by_endpoint[low]["passes_target"]
    if high_passes and low_fails:
        status = "confirmed_upper_bound" if low is None else "confirmed_bracket"
    elif high_passes:
        status = "crossing_below_exploratory_bracket"
    else:
        status = "exploratory_crossing_not_confirmed"
    return {
        "status": status,
        "confirmed_failing_effective_shots_per_setting": (
            low if low is not None and low_fails else None
        ),
        "confirmed_passing_effective_shots_per_setting": high if high_passes else None,
    }


def _device_prices(cards, resources, effective_shots: int | None, n_qubits: int) -> dict:
    if effective_shots is None:
        return {}
    prices = {}
    for card in cards:
        raw = inflate_shots_for_fidelity(card, resources, effective_shots, n_qubits)
        prices[card.name] = cost_schedule(
            card,
            resources,
            raw,
            n_qubits=n_qubits,
            evidence_tier="exact",
            epsilon=ACCURACY_TARGET_MILLIHARTREE * 1e-3,
        )
    return prices


def _search_rung(arguments) -> dict:
    block_size, exploratory_replicas, confirmatory_replicas, cards = arguments
    problem = SYSTEMS["beh2"]()
    codes = problem["codes"]
    bank = problem["_matrix_bank"]
    result = problem["_result"]
    exact_energy = problem["_exact_ground_energy"]
    groups = _partition(problem["n_qubits"], codes, block_size)
    row, resources, compatibility, _ = _synthesize(
        problem["n_qubits"], codes, groups, block_size
    )
    settings = _compiled_settings(problem["n_qubits"], codes, groups, block_size)
    if any(len(setting.readouts) != int(compatibility[index].sum())
           for index, setting in enumerate(settings)):
        raise AssertionError("compiled readouts disagree with hierarchy compatibility")

    requested = {estimator: SEARCH_ENDPOINTS for estimator in ESTIMATORS}
    exploratory_raw = _run_endpoints(
        reference=bank.reference,
        bank=bank,
        result=result,
        settings=settings,
        requested=requested,
        replicas=exploratory_replicas,
        seed=EXPLORATORY_SEED,
        block_size=block_size,
    )
    exploration = _summaries(
        exploratory_raw, exact_energy, block_size=block_size, seed_offset=0
    )
    confirmation_requested = {}
    for estimator in ESTIMATORS:
        low, high = _exploratory_bracket(exploration[estimator])
        confirmation_requested[estimator] = tuple(
            endpoint for endpoint in (low, high) if endpoint is not None
        )
    confirmation_raw = _run_endpoints(
        reference=bank.reference,
        bank=bank,
        result=result,
        settings=settings,
        requested=confirmation_requested,
        replicas=confirmatory_replicas,
        seed=CONFIRMATORY_SEED,
        block_size=block_size,
    )
    confirmation = _summaries(
        confirmation_raw, exact_energy, block_size=block_size, seed_offset=10_000_000
    )

    estimators = {}
    for estimator in ESTIMATORS:
        search = _confirm_result(exploration[estimator], confirmation[estimator])
        effective = search["confirmed_passing_effective_shots_per_setting"]
        estimators[estimator] = {
            "exploration": exploration[estimator],
            "confirmation": confirmation[estimator],
            "shot_to_target": search,
            "device_costs": _device_prices(
                cards, resources, effective, problem["n_qubits"]
            ),
        }
    return {
        "block_size": block_size,
        "settings": row["settings"],
        "compatible_readouts": int(compatibility.sum()),
        "mean_reader_count": float(compatibility.sum(axis=0).mean()),
        "estimators": estimators,
    }


def _ordering(rows: list[dict], cards) -> dict:
    output = {}
    for card in cards:
        by_estimator = {}
        for estimator in ESTIMATORS:
            live = []
            for row in rows:
                cost = row["estimators"][estimator]["device_costs"].get(card.name)
                if cost is None:
                    continue
                value = cost["accuracy"]["C_time_epsilon_us"]
                if value is not None:
                    live.append({
                        "block_size": row["block_size"],
                        "C_time_epsilon_us": value,
                    })
            by_estimator[estimator] = sorted(
                live, key=lambda item: (item["C_time_epsilon_us"], item["block_size"])
            )
        assigned = [row["block_size"] for row in by_estimator["single_assignment"]]
        pooled = [row["block_size"] for row in by_estimator["pooled"]]
        common = set(assigned) & set(pooled)
        by_estimator["compared_block_sizes"] = sorted(common)
        by_estimator["pooling_reorders_protocols"] = (
            [value for value in assigned if value in common]
            != [value for value in pooled if value in common]
        ) if common else None
        output[card.name] = by_estimator
    return output


def build_record(*, exploratory_replicas: int = EXPLORATORY_REPLICAS,
                 confirmatory_replicas: int = CONFIRMATORY_REPLICAS,
                 workers: int = 1) -> dict:
    if exploratory_replicas <= 0 or confirmatory_replicas <= 0:
        raise ValueError("replica counts must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    cards = load_device_cards()
    h4 = SYSTEMS["h4"]()
    h4_bias = abs(float(h4["error_millihartree"]))
    h4_rows = [{
        "block_size": block_size,
        "estimators": {
            estimator: {
                "shot_to_target": {
                    "status": "bias_floor_exceeds_target",
                    "confirmed_failing_effective_shots_per_setting": None,
                    "confirmed_passing_effective_shots_per_setting": None,
                },
                "device_costs": {},
            }
            for estimator in ESTIMATORS
        },
    } for block_size in BLOCK_SIZES]
    arguments = [
        (block_size, exploratory_replicas, confirmatory_replicas, cards)
        for block_size in BLOCK_SIZES
    ]
    if workers == 1:
        beh2_rows = [_search_rung(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            beh2_rows = list(pool.map(_search_rung, arguments))
    beh2_rows.sort(key=lambda row: row["block_size"])
    return {
        "schema": "clifford_qc.exact_shot_search.v1",
        "evidence_tier": "exact",
        "search_uncertainty_evidence": "heuristic",
        "accuracy_target_millihartree": ACCURACY_TARGET_MILLIHARTREE,
        "primary_metric": (
            "replica RMSE of the nonlinear selected-rank Ritz energy against "
            "the full exact ground energy"
        ),
        "pass_rule": (
            "zero solver failures and the one-sided 95% nonparametric-bootstrap "
            "upper bound on RMSE <= 1.6 mHa"
        ),
        "claim_boundary": (
            "oracle-reference benchmark comparison with heuristic Monte Carlo "
            "uncertainty; not a finite-sample energy certificate, deployable "
            "stopping rule, calibrated device-noise simulation, or hardware result"
        ),
        "protocol": {
            "rank_rule": "minimize measured E(rank) + gamma*sigma(rank)",
            "rank_selection_gamma": RANK_SELECTION_GAMMA,
            "search_endpoints_effective_shots_per_setting": list(SEARCH_ENDPOINTS),
            "nested_endpoints": True,
            "exploratory_replicas": exploratory_replicas,
            "confirmatory_replicas": confirmatory_replicas,
            "exploratory_seed": EXPLORATORY_SEED,
            "confirmatory_seed": CONFIRMATORY_SEED,
            "paired_estimators": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "delta": DELTA,
            "calibration_artifact": "benchmarks/reference_results/finite_shot_rethink.json",
        },
        "device_cards": [
            {"sha256": card.sha256, **card.to_dict()} for card in cards
        ],
        "systems": {
            "h4": {
                "exact_ground_energy": h4["_exact_ground_energy"],
                "exact_subspace_bias_millihartree": h4_bias,
                "status": "bias_floor_exceeds_target",
                "rows": h4_rows,
            },
            "beh2": {
                "exact_ground_energy": SYSTEMS["beh2"]()["_exact_ground_energy"],
                "exact_subspace_bias_millihartree": abs(
                    SYSTEMS["beh2"]()["error_millihartree"]
                ),
                "status": "searched",
                "rows": beh2_rows,
                "accuracy_matched_ordering": _ordering(beh2_rows, cards),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploratory-replicas", type=int, default=EXPLORATORY_REPLICAS)
    parser.add_argument("--confirmatory-replicas", type=int, default=CONFIRMATORY_REPLICAS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = stamp_record(build_record(
        exploratory_replicas=args.exploratory_replicas,
        confirmatory_replicas=args.confirmatory_replicas,
        workers=args.workers,
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    for system, payload in record["systems"].items():
        print(f"{system}: {payload['status']}")
        for row in payload["rows"]:
            outcomes = ", ".join(
                f"{name}={arm['shot_to_target']['status']}:"
                f"{arm['shot_to_target']['confirmed_passing_effective_shots_per_setting']}"
                for name, arm in row["estimators"].items()
            )
            print(f"  k={row['block_size']}: {outcomes}")
    print(args.out)


if __name__ == "__main__":
    main()
