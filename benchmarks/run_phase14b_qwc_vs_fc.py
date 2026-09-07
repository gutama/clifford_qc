#!/usr/bin/env python3
"""Run the one preregistered Phase 14b QWC-versus-FC experiment.

The result-free contract was merged as PR #82 before this producer existed.
This script executes that contract without adapting the grouping, allocator,
shot grid, confidence family, seeds, audit, or verdict rule to the outcomes.

    python benchmarks/run_phase14b_qwc_vs_fc.py
    python benchmarks/check_phase14b_qwc_vs_fc.py
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from clifford_qc.measurement import (
    CompiledMeasurementSampler,
    GroupedWordCache,
    compile_block_measurement_plan,
)
from clifford_qc.measurement.cost import (
    cost_schedule,
    inflate_shots_for_fidelity,
)
from clifford_qc.measurement.functionals import ritz_functional
from clifford_qc.reproducibility import stamp_record
from clifford_qc.states import computational_probabilities
from clifford_qc.subspace.restriction import Restriction

try:  # package import in tests versus direct script execution
    from benchmarks.check_phase14b_preregistration import (
        CONFIG,
        load_config,
        static_problems,
    )
    from benchmarks.run_clifford_hierarchy import _beh2_bank, load_device_cards
except ImportError:  # pragma: no cover - direct script execution
    from check_phase14b_preregistration import CONFIG, load_config, static_problems
    from run_clifford_hierarchy import _beh2_bank, load_device_cards


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference_results" / "phase14b_qwc_vs_fc.json"
SCHEMA = "clifford_qc.phase14b_qwc_vs_fc.v1"
PREREGISTRATION_MERGE_COMMIT = "bb7a76a4da06112a8c1f8501e21b00b4cf25f998"
PROTOCOL_NAMES = ("qwc", "fully_commuting")
ESTIMATORS = ("single_assignment", "pooled")


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derived_seed(root: int, protocol_index: int, row_index: int) -> int:
    """Turn the preregistered SeedSequence namespace into a JSON-safe seed."""
    sequence = np.random.SeedSequence(
        int(root), spawn_key=(int(protocol_index), int(row_index))
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def coefficient_range_neyman_schedule(
    plan,
    coefficients: dict[int, float],
    total_shots: int,
    *,
    min_shots_per_setting: int = 2,
) -> tuple[int, ...]:
    """Largest-remainder allocation under the frozen assigned-coefficient score.

    Fractions retain the exact binary values of the already-fixed Ritz
    coefficients. This makes the integer apportionment independent of an
    intermediate floating-point normalization or summation order.
    """
    if isinstance(total_shots, bool) or not isinstance(total_shots, int):
        raise TypeError("total_shots must be an integer")
    if min_shots_per_setting != 2:
        raise ValueError("Phase 14b freezes a two-shot setting floor")
    settings = tuple(plan.settings)
    floor_total = min_shots_per_setting * len(settings)
    if total_shots < floor_total:
        raise ValueError("total_shots is smaller than the setting floors")

    scores = tuple(
        sum(
            (Fraction.from_float(abs(float(coefficients.get(code, 0.0))))
             for code in setting.assigned_word_codes),
            start=Fraction(0),
        )
        for setting in settings
    )
    score_total = sum(scores, start=Fraction(0))
    if score_total <= 0:
        raise ValueError("the Ritz functional touches no compiled setting")
    remaining = total_shots - floor_total
    quotas = tuple(Fraction(remaining) * score / score_total for score in scores)
    additions = [quota.numerator // quota.denominator for quota in quotas]
    leftover = remaining - sum(additions)
    order = sorted(
        range(len(settings)),
        key=lambda index: (-(quotas[index] - additions[index]), index),
    )
    for index in order[:leftover]:
        additions[index] += 1
    schedule = tuple(min_shots_per_setting + value for value in additions)
    if sum(schedule) != total_shots:
        raise AssertionError("largest-remainder allocation lost shots")
    if any(value < min_shots_per_setting for value in schedule):
        raise AssertionError("largest-remainder allocation violated its floor")
    return schedule


def _setting_resources(plan) -> list[dict[str, int]]:
    return [
        {
            "n_1q": int(row.n_1q),
            "n_2q": int(row.n_2q),
            "d_1q": int(row.d_1q),
            "d_2q": int(row.d_2q),
        }
        for row in plan.resources
    ]


def _resource_sums(plan) -> dict[str, int]:
    rows = _setting_resources(plan)
    return {
        "N_1q": sum(row["n_1q"] for row in rows),
        "D_1q": sum(row["d_1q"] for row in rows),
        "N_2q": sum(row["n_2q"] for row in rows),
        "D_2q": sum(row["d_2q"] for row in rows),
    }


def _plan_snapshot(plan, coefficients: dict[int, float]) -> dict:
    partition = [list(setting.assigned_word_codes) for setting in plan.settings]
    readouts = [
        [
            [int(code), int(sign), list(positions)]
            for code, (sign, positions) in sorted(setting.readouts.items())
        ]
        for setting in plan.settings
    ]
    touched = sum(
        any(coefficients.get(code, 0.0) != 0.0
            for code in setting.assigned_word_codes)
        for setting in plan.settings
    )
    return {
        "block_size": int(plan.block_size),
        "settings": len(plan.settings),
        "assigned_words": sum(len(row) for row in partition),
        "readable_word_setting_pairs": sum(len(row) for row in readouts),
        "ritz_touched_assigned_settings": int(touched),
        "partition_sha256": _canonical_sha256(partition),
        "signed_readouts_sha256": _canonical_sha256(readouts),
        "resource_sums": _resource_sums(plan),
        "setting_resources": _setting_resources(plan),
        "z_only_restrictions_checked": int(
            plan.synthesis.z_only_restrictions_checked
        ),
    }


def _direct_word_means(reference, codes: Sequence[int]) -> dict[int, float]:
    scale = float(2 ** reference.n)
    return {
        int(code): scale * complex(reference.terms.get(int(code), 0.0)).real
        for code in codes
    }


def _compiled_word_means(reference, plan) -> dict[int, float]:
    """Exact means obtained through each compiled setting's signed readout."""
    means: dict[int, float] = {}
    for setting in plan.settings:
        rotated = Restriction.encoding(setting.clifford).state(reference)
        probabilities = computational_probabilities(rotated)
        for code in setting.assigned_word_codes:
            sign, positions = setting.readouts[code]
            mean = 0.0
            for bits, probability in probabilities.items():
                parity = sum(bits[position] == "1" for position in positions) % 2
                mean += float(probability) * sign * (-1.0 if parity else 1.0)
            means[int(code)] = mean
    if set(means) != set(plan.codes):
        raise AssertionError("compiled exact means do not cover the word universe")
    return means


def _operator_expectation(operator, means: dict[int, float]) -> complex:
    return sum(
        complex(coefficient) * (1.0 if code == 0 else means[code])
        for code, coefficient in operator.terms.items()
    )


def exact_reconstruction(bank, indices: Sequence[int], plan) -> dict[str, float]:
    """Reconstruct exact S and H through compiled signed parity readouts."""
    means = _compiled_word_means(bank.reference, plan)
    direct = _direct_word_means(bank.reference, plan.codes)
    max_word_error = max(abs(means[code] - direct[code]) for code in plan.codes)
    expected_s, expected_h = bank.matrices(indices)
    size = len(indices)
    actual_s = np.empty((size, size), dtype=complex)
    actual_h = np.empty((size, size), dtype=complex)
    for row, i in enumerate(indices):
        for column, j in enumerate(indices):
            actual_s[row, column] = _operator_expectation(
                bank.overlap_operator(i, j), means
            )
            actual_h[row, column] = _operator_expectation(
                bank.element_operator(i, j), means
            )
    return {
        "max_signed_readout_word_mean_error": float(max_word_error),
        "max_S_entry_error": float(np.max(np.abs(actual_s - expected_s))),
        "max_H_entry_error": float(np.max(np.abs(actual_h - expected_h))),
    }


def _seed_row(root: int, protocol_index: int, row_index: int) -> dict:
    return {
        "root": int(root),
        "spawn_key": [int(protocol_index), int(row_index)],
        "derived_seed": derived_seed(root, protocol_index, row_index),
    }


def _schedule_map(plan, schedule: Sequence[int]) -> dict[tuple, int]:
    return {
        setting.key: int(shots)
        for setting, shots in zip(plan.settings, schedule)
    }


def _headline_rows(
    reference,
    plan,
    functional,
    protocol_index: int,
    endpoints: Sequence[int],
    *,
    seed_root: int,
    confidence: dict,
    stochastic_radius: float,
    min_shots_per_setting: int,
) -> list[dict]:
    sampler = CompiledMeasurementSampler(0)
    sampler.prepare(reference, plan.settings)
    exact_value = functional.exact(reference)
    rows = []
    for endpoint_index, total in enumerate(endpoints):
        schedule = coefficient_range_neyman_schedule(
            plan,
            functional.coefficients,
            int(total),
            min_shots_per_setting=min_shots_per_setting,
        )
        seed = _seed_row(seed_root, protocol_index, endpoint_index)
        sampler.reseed(seed["derived_seed"])
        batch = sampler.sample_from_state(
            reference, plan.settings, _schedule_map(plan, schedule)
        )
        cache = GroupedWordCache(reference.n, pooling="assigned")
        cache.add_batch(batch)
        estimate = float(functional.estimate(cache))
        radius = float(functional.radius(
            cache,
            float(confidence["delta"]),
            family=int(confidence["family"]),
            bound="eb",
            rounds=int(confidence["rounds"]),
            method="bonferroni",
        ))
        variance = float(functional.variance(cache))
        rows.append({
            "endpoint_index": endpoint_index,
            "total_physical_shots": int(total),
            "shot_vector": list(schedule),
            "shot_vector_sha256": _canonical_sha256(list(schedule)),
            "seed": seed,
            "ritz_estimate_hartree": estimate,
            "ritz_exact_hartree": float(exact_value),
            "estimate_error_hartree": estimate - float(exact_value),
            "covariance_aware_variance_hartree2": variance,
            "empirical_bernstein_radius_hartree": radius,
            "passes_stochastic_radius": radius <= stochastic_radius,
        })
    return rows


def _certification(rows: Sequence[dict], max_endpoint: int) -> dict:
    passing = next((row for row in rows if row["passes_stochastic_radius"]), None)
    if passing is None:
        return {
            "status": "right_censored",
            "certified_endpoint_index": None,
            "certified_total_physical_shots": None,
            "right_censored_above": int(max_endpoint),
        }
    return {
        "status": "certified",
        "certified_endpoint_index": int(passing["endpoint_index"]),
        "certified_total_physical_shots": int(passing["total_physical_shots"]),
        "right_censored_above": None,
    }


def _audit(
    reference,
    plan,
    functional,
    protocol_index: int,
    *,
    replicas: int,
    total_shots: int,
    seed_root: int,
    min_shots_per_setting: int,
) -> dict:
    schedule = coefficient_range_neyman_schedule(
        plan,
        functional.coefficients,
        total_shots,
        min_shots_per_setting=min_shots_per_setting,
    )
    sampler = CompiledMeasurementSampler(0)
    sampler.prepare(reference, plan.settings)
    estimates = {name: [] for name in ESTIMATORS}
    predictions = {name: [] for name in ESTIMATORS}
    max_weight_error = 0.0
    seeds = []
    for replica_index in range(replicas):
        seed = _seed_row(seed_root, protocol_index, replica_index)
        seeds.append(seed)
        sampler.reseed(seed["derived_seed"])
        batch = sampler.sample_from_state(
            reference, plan.settings, _schedule_map(plan, schedule)
        )
        for name, pooling in (("single_assignment", "assigned"), ("pooled", "shots")):
            cache = GroupedWordCache(reference.n, pooling=pooling)
            cache.add_batch(batch)
            if pooling == "shots":
                max_weight_error = max(
                    max_weight_error,
                    max(
                        abs(sum(cache.group_weights(code).values()) - 1.0)
                        for code in plan.codes
                    ),
                )
            estimate = float(functional.estimate(cache))
            prediction = float(functional.variance(cache))
            if not math.isfinite(prediction) or prediction < 0.0:
                raise AssertionError("covariance-aware variance is invalid")
            estimates[name].append(estimate)
            predictions[name].append(prediction)

    cells = {}
    for name in ESTIMATORS:
        empirical = float(np.var(np.asarray(estimates[name]), ddof=1))
        predicted = float(np.mean(np.asarray(predictions[name])))
        cells[name] = {
            "replicas": replicas,
            "empirical_variance_hartree2": empirical,
            "mean_predicted_variance_hartree2": predicted,
            "empirical_to_predicted_ratio": empirical / predicted,
        }
    return {
        "total_physical_shots_per_replica": total_shots,
        "shot_vector": list(schedule),
        "shot_vector_sha256": _canonical_sha256(list(schedule)),
        "replica_seeds": seeds,
        "max_pooled_weight_sum_error": float(max_weight_error),
        "estimators": cells,
    }


def _device_costs(
    cards,
    plan,
    certified_row: dict | None,
    n_qubits: int,
    stochastic_radius: float,
) -> dict:
    if certified_row is None:
        return {
            card.name: {
                "status": "not_priced_without_certified_endpoint",
                "device_card_sha256": card.sha256,
            }
            for card in cards
        }
    resources = list(plan.resources)
    effective = [int(value) for value in certified_row["shot_vector"]]
    output = {}
    for card in cards:
        raw = inflate_shots_for_fidelity(card, resources, effective, n_qubits)
        ledger = cost_schedule(
            card,
            resources,
            raw,
            n_qubits=n_qubits,
            evidence_tier="exact",
            epsilon=stochastic_radius,
        )
        output[card.name] = {
            "status": "priced" if ledger["admissible"] else "inadmissible",
            "effective_shot_vector": effective,
            "fidelity_inflated_raw_shot_vector": raw,
            "ledger": ledger,
        }
    return output


def _decision(protocols: dict, config: dict, blocking_failures: Sequence[str]) -> dict:
    if blocking_failures:
        return {
            "status": "protocol_failure",
            "shot_efficiency_go": None,
            "fully_commuting_to_qwc_certified_shot_ratio": None,
            "blocking_failures": list(blocking_failures),
            "card_specific": {},
        }
    qwc_certification = protocols["qwc"]["certification"]
    qwc = qwc_certification["certified_total_physical_shots"]
    full = protocols["fully_commuting"]["certification"][
        "certified_total_physical_shots"
    ]
    threshold = float(config["acceptance"]["material_shot_reduction_fraction"])
    ratio = None if qwc is None or full is None else full / qwc
    qwc_comparison_floor = (
        qwc if qwc is not None else qwc_certification["right_censored_above"]
    )
    go = (
        full is not None
        and qwc_comparison_floor is not None
        and full <= threshold * qwc_comparison_floor
    )
    card_specific = {}
    for card in (row["name"] for row in config["protocol"]["device_cards"]):
        qwc_cost = protocols["qwc"]["device_costs"][card]
        full_cost = protocols["fully_commuting"]["device_costs"][card]
        qwc_time = qwc_cost.get("ledger", {}).get("accuracy", {}).get(
            "C_time_epsilon_us"
        )
        full_time = full_cost.get("ledger", {}).get("accuracy", {}).get(
            "C_time_epsilon_us"
        )
        time_ratio = None if qwc_time is None or full_time is None else full_time / qwc_time
        card_specific[card] = {
            "status": (
                "compared" if time_ratio is not None else
                "unresolved_or_inadmissible"
            ),
            "fully_commuting_to_qwc_runtime_ratio": time_ratio,
            "fully_commuting_faster": None if time_ratio is None else time_ratio < 1.0,
        }
    return {
        "status": "go" if go else "no_go",
        "shot_efficiency_go": bool(go),
        "fully_commuting_to_qwc_certified_shot_ratio": ratio,
        "blocking_failures": [],
        "card_specific": card_specific,
    }


def build_record(
    *,
    config: dict | None = None,
    endpoints: Sequence[int] | None = None,
    audit_replicas: int | None = None,
) -> dict:
    """Build the deterministic record; overrides exist only for fast unit tests."""
    config = load_config(CONFIG) if config is None else config
    preregistration_problems = static_problems(config)
    if preregistration_problems:
        raise ValueError(
            "Phase 14b preregistration failed: " + preregistration_problems[0]
        )
    frozen_endpoints = tuple(config["protocol"]["total_physical_shot_endpoints"])
    endpoints = frozen_endpoints if endpoints is None else tuple(int(x) for x in endpoints)
    audit = config["protocol"]["covariance_audit"]
    replicas = int(audit["replicas"] if audit_replicas is None else audit_replicas)
    if replicas < 2:
        raise ValueError("the covariance audit needs at least two replicas")

    source = _beh2_bank()
    codes = tuple(int(code) for code in source["codes"] if int(code) != 0)
    bank = source["_matrix_bank"]
    result = source["_result"]
    functional = ritz_functional(
        bank, result.indices, result.ritz_vector(), result.ground_energy
    )
    cards = load_device_cards()
    roots = config["protocol"]["seed_roots"]
    allocator = config["protocol"]["allocator"]
    confidence = config["protocol"]["confidence"]
    stochastic_radius = float(
        config["protocol"]["accuracy"]["stochastic_radius_hartree"]
    )

    plans = {}
    for endpoint in config["comparison"]["endpoints"]:
        plans[endpoint["name"]] = compile_block_measurement_plan(
            source["n_qubits"],
            codes,
            int(endpoint["block_size"]),
            spin_conserving_jw_hamiltonian=bank.hamiltonian,
        )

    protocols = {}
    blocking_failures = []
    ratio_low, ratio_high = audit["acceptable_empirical_to_predicted_interval"]
    tolerance = config["protocol"]["matrix_reconstruction"][
        "max_absolute_entry_error_tolerance"
    ]
    frozen = {row["name"]: row for row in config["comparison"]["endpoints"]}
    for protocol_index, name in enumerate(PROTOCOL_NAMES):
        plan = plans[name]
        snapshot = _plan_snapshot(plan, functional.coefficients)
        for field in ("settings", "ritz_touched_assigned_settings", "resource_sums"):
            if snapshot[field] != frozen[name][field]:
                blocking_failures.append(f"{name}: compiled {field} drift")
        reconstruction = exact_reconstruction(bank, result.indices, plan)
        for field in ("max_S_entry_error", "max_H_entry_error"):
            if reconstruction[field] > tolerance:
                blocking_failures.append(f"{name}: {field} exceeds tolerance")
        if reconstruction["max_signed_readout_word_mean_error"] > tolerance:
            blocking_failures.append(f"{name}: signed readout means exceed tolerance")

        headline = _headline_rows(
            bank.reference,
            plan,
            functional,
            protocol_index,
            endpoints,
            seed_root=int(roots["headline"]),
            confidence=confidence,
            stochastic_radius=stochastic_radius,
            min_shots_per_setting=int(allocator["min_shots_per_setting"]),
        )
        certification = _certification(headline, max(frozen_endpoints))
        covariance = _audit(
            bank.reference,
            plan,
            functional,
            protocol_index,
            replicas=replicas,
            total_shots=int(audit["total_physical_shots"]),
            seed_root=int(roots["covariance_audit"]),
            min_shots_per_setting=int(allocator["min_shots_per_setting"]),
        )
        if covariance["max_pooled_weight_sum_error"] > 1e-12:
            blocking_failures.append(f"{name}: pooled weights do not sum to one")
        for estimator, cell in covariance["estimators"].items():
            ratio = cell["empirical_to_predicted_ratio"]
            if not ratio_low <= ratio <= ratio_high:
                blocking_failures.append(
                    f"{name}/{estimator}: covariance ratio {ratio:.6g} outside "
                    f"[{ratio_low}, {ratio_high}]"
                )
        certified_index = certification["certified_endpoint_index"]
        certified_row = None if certified_index is None else headline[certified_index]
        protocols[name] = {
            "protocol_index": protocol_index,
            "compiled_plan": snapshot,
            "exact_reconstruction": reconstruction,
            "headline": headline,
            "certification": certification,
            "covariance_audit": covariance,
            "device_costs": _device_costs(
                cards,
                plan,
                certified_row,
                source["n_qubits"],
                stochastic_radius,
            ),
        }

    return {
        "schema": SCHEMA,
        "phase": config["phase"],
        "status": "sampled_once_from_merged_preregistration",
        "preregistration": {
            "path": str(CONFIG.relative_to(HERE.parent)),
            "schema": config["schema"],
            "merge_commit": PREREGISTRATION_MERGE_COMMIT,
            "sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "unchanged_protocol": True,
        },
        "evidence_tier": "exact_oracle_finite_sample",
        "system": {
            "system": source["system"],
            "mapping": "jw",
            "n_qubits": source["n_qubits"],
            "basis_size": source["basis_size"],
            "retained_rank": int(result.effective_rank),
            "basis_labels": source["basis_labels"],
            "measured_word_universe": len(codes),
            "ritz_functional_measured_support": len(functional.coefficients),
            "exact_subspace_bias_millihartree": source["error_millihartree"],
        },
        "protocol": {
            "primary_estimand": config["protocol"]["primary_estimand"],
            "primary_estimator": "single_assignment",
            "pooled_estimator_role": "covariance_audit_only",
            "allocator": config["protocol"]["allocator"],
            "accuracy": config["protocol"]["accuracy"],
            "confidence": config["protocol"]["confidence"],
            "total_physical_shot_endpoints": list(endpoints),
            "frozen_total_physical_shot_endpoints": list(frozen_endpoints),
            "seed_roots": config["protocol"]["seed_roots"],
            "covariance_audit": config["protocol"]["covariance_audit"],
        },
        "device_cards": [
            {"sha256": card.sha256, **card.to_dict()} for card in cards
        ],
        "protocols": protocols,
        "decision": _decision(protocols, config, blocking_failures),
        "claim_boundary": config["reporting"]["evidence_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = stamp_record(build_record())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in PROTOCOL_NAMES:
        row = record["protocols"][name]
        certified = row["certification"]["certified_total_physical_shots"]
        print(f"{name}: certified_total_physical_shots={certified}")
        for estimator, cell in row["covariance_audit"]["estimators"].items():
            print(
                f"  {estimator}: covariance ratio="
                f"{cell['empirical_to_predicted_ratio']:.6f}"
            )
    print(f"decision: {record['decision']['status']}")
    print(args.out)


if __name__ == "__main__":
    main()
