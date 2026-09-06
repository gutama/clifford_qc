#!/usr/bin/env python3
"""Regenerate and gate the sampled Phase 14b QWC-versus-FC record.

The checker first re-derives every acceptance decision from the committed
record, then repeats the exact seeded execution and compares every scientific,
schedule, seed, resource, and decision field. Provenance metadata is retained
for auditability and ignored by the value comparison.

    python benchmarks/check_phase14b_qwc_vs_fc.py
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys

from clifford_qc.reproducibility import (
    CROSS_MACHINE_ATOL,
    CROSS_MACHINE_RTOL,
    compare_json_records,
    guarded_contract_problems,
    sampling_stream_mismatch,
)

try:  # package import in tests versus direct script execution
    from benchmarks.check_phase14b_preregistration import (
        EXPECTED_CARD_SHA256,
        load_config,
        static_problems,
    )
    from benchmarks.run_phase14b_qwc_vs_fc import (
        CONFIG,
        ESTIMATORS,
        PREREGISTRATION_MERGE_COMMIT,
        PROTOCOL_NAMES,
        REFERENCE,
        SCHEMA,
        _decision,
        build_record,
        derived_seed,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_phase14b_preregistration import (
        EXPECTED_CARD_SHA256,
        load_config,
        static_problems,
    )
    from run_phase14b_qwc_vs_fc import (
        CONFIG,
        ESTIMATORS,
        PREREGISTRATION_MERGE_COMMIT,
        PROTOCOL_NAMES,
        REFERENCE,
        SCHEMA,
        _decision,
        build_record,
        derived_seed,
    )


def _blocking_failures(record: dict, config: dict) -> list[str]:
    failures = []
    tolerance = config["protocol"]["matrix_reconstruction"][
        "max_absolute_entry_error_tolerance"
    ]
    ratio_low, ratio_high = config["protocol"]["covariance_audit"][
        "acceptable_empirical_to_predicted_interval"
    ]
    frozen = {row["name"]: row for row in config["comparison"]["endpoints"]}
    for name in PROTOCOL_NAMES:
        protocol = record["protocols"][name]
        plan = protocol["compiled_plan"]
        for field in ("settings", "ritz_touched_assigned_settings", "resource_sums"):
            if plan[field] != frozen[name][field]:
                failures.append(f"{name}: compiled {field} drift")
        reconstruction = protocol["exact_reconstruction"]
        for field in ("max_S_entry_error", "max_H_entry_error"):
            if reconstruction[field] > tolerance:
                failures.append(f"{name}: {field} exceeds tolerance")
        if reconstruction["max_signed_readout_word_mean_error"] > tolerance:
            failures.append(f"{name}: signed readout means exceed tolerance")
        audit = protocol["covariance_audit"]
        if audit["max_pooled_weight_sum_error"] > 1e-12:
            failures.append(f"{name}: pooled weights do not sum to one")
        for estimator in ESTIMATORS:
            ratio = audit["estimators"][estimator]["empirical_to_predicted_ratio"]
            if not ratio_low <= ratio <= ratio_high:
                failures.append(
                    f"{name}/{estimator}: covariance ratio {ratio:.6g} outside "
                    f"[{ratio_low}, {ratio_high}]"
                )
    return failures


def _row_problems(
    prefix: str,
    row: dict,
    *,
    endpoint_index: int,
    endpoint: int,
    settings: int,
    seed_root: int,
    protocol_index: int,
    stochastic_radius: float,
) -> list[str]:
    problems = []
    if row.get("endpoint_index") != endpoint_index:
        problems.append(f"{prefix}: endpoint index drifted")
    if row.get("total_physical_shots") != endpoint:
        problems.append(f"{prefix}: total shot endpoint drifted")
    schedule = row.get("shot_vector", [])
    if len(schedule) != settings:
        problems.append(f"{prefix}: shot vector has the wrong length")
    elif (not all(isinstance(value, int) and not isinstance(value, bool)
                  and value >= 2 for value in schedule)
          or sum(schedule) != endpoint):
        problems.append(f"{prefix}: shot vector violates its floor or total")
    seed = row.get("seed", {})
    expected_seed = {
        "root": seed_root,
        "spawn_key": [protocol_index, endpoint_index],
        "derived_seed": derived_seed(seed_root, protocol_index, endpoint_index),
    }
    if seed != expected_seed:
        problems.append(f"{prefix}: sampled seed does not match its namespace")
    radius = row.get("empirical_bernstein_radius_hartree")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius < 0:
        problems.append(f"{prefix}: confidence radius is invalid")
    elif row.get("passes_stochastic_radius") != (radius <= stochastic_radius):
        problems.append(f"{prefix}: crossing flag disagrees with its radius")
    variance = row.get("covariance_aware_variance_hartree2")
    if not isinstance(variance, (int, float)) or isinstance(variance, bool) or variance < 0:
        problems.append(f"{prefix}: covariance-aware variance is invalid")
    estimate = row.get("ritz_estimate_hartree")
    exact = row.get("ritz_exact_hartree")
    error = row.get("estimate_error_hartree")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in (estimate, exact, error)):
        problems.append(f"{prefix}: Ritz estimate fields are invalid")
    elif not math.isclose(error, estimate - exact, rel_tol=0.0, abs_tol=1e-15):
        problems.append(f"{prefix}: Ritz estimate error was not re-derived")
    return problems


def _certification_problems(prefix: str, protocol: dict, max_endpoint: int) -> list[str]:
    problems = []
    rows = protocol["headline"]
    first = next((row for row in rows if row["passes_stochastic_radius"]), None)
    expected = (
        {
            "status": "right_censored",
            "certified_endpoint_index": None,
            "certified_total_physical_shots": None,
            "right_censored_above": max_endpoint,
        }
        if first is None else
        {
            "status": "certified",
            "certified_endpoint_index": first["endpoint_index"],
            "certified_total_physical_shots": first["total_physical_shots"],
            "right_censored_above": None,
        }
    )
    if protocol.get("certification") != expected:
        problems.append(f"{prefix}: certified endpoint is not the first passing look")
    return problems


def _audit_problems(
    prefix: str,
    protocol: dict,
    *,
    protocol_index: int,
    seed_root: int,
    replicas: int,
    total_shots: int,
    settings: int,
) -> list[str]:
    problems = []
    audit = protocol.get("covariance_audit", {})
    if audit.get("total_physical_shots_per_replica") != total_shots:
        problems.append(f"{prefix}: covariance-audit shot total drifted")
    schedule = audit.get("shot_vector", [])
    if len(schedule) != settings or sum(schedule) != total_shots or min(schedule, default=0) < 2:
        problems.append(f"{prefix}: covariance-audit schedule is invalid")
    seeds = audit.get("replica_seeds", [])
    if len(seeds) != replicas:
        problems.append(f"{prefix}: covariance-audit seed count drifted")
    else:
        for replica_index, seed in enumerate(seeds):
            expected = {
                "root": seed_root,
                "spawn_key": [protocol_index, replica_index],
                "derived_seed": derived_seed(seed_root, protocol_index, replica_index),
            }
            if seed != expected:
                problems.append(
                    f"{prefix}: covariance-audit seed {replica_index} drifted"
                )
                break
    if audit.get("max_pooled_weight_sum_error", math.inf) > 1e-12:
        problems.append(f"{prefix}: pooled weights do not sum to one")
    if set(audit.get("estimators", {})) != set(ESTIMATORS):
        problems.append(f"{prefix}: covariance-audit estimator set drifted")
    else:
        for estimator in ESTIMATORS:
            cell = audit["estimators"][estimator]
            if cell.get("replicas") != replicas:
                problems.append(f"{prefix}/{estimator}: replica count drifted")
            empirical = cell.get("empirical_variance_hartree2", -1.0)
            predicted = cell.get("mean_predicted_variance_hartree2", -1.0)
            ratio = cell.get("empirical_to_predicted_ratio", -1.0)
            if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0
                        for value in (empirical, predicted, ratio))
                    or predicted <= 0):
                problems.append(f"{prefix}/{estimator}: covariance cell is invalid")
            elif not math.isclose(
                ratio, empirical / predicted, rel_tol=1e-12, abs_tol=0.0
            ):
                problems.append(f"{prefix}/{estimator}: covariance ratio drifted")
    return problems


@guarded_contract_problems
def contract_problems(record: dict) -> list[str]:
    problems = []
    config = load_config()
    problems.extend(f"preregistration: {item}" for item in static_problems(config))
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("status") != "sampled_once_from_merged_preregistration":
        problems.append("record does not declare the one authorized sampled execution")
    preregistration = record.get("preregistration", {})
    if preregistration.get("merge_commit") != PREREGISTRATION_MERGE_COMMIT:
        problems.append("record does not descend from the merged preregistration")
    if preregistration.get("sha256") != hashlib.sha256(CONFIG.read_bytes()).hexdigest():
        problems.append("record's preregistration digest drifted")
    if preregistration.get("unchanged_protocol") is not True:
        problems.append("record does not affirm the unchanged frozen protocol")
    if record.get("evidence_tier") != "exact_oracle_finite_sample":
        problems.append("Phase 14b evidence tier drifted")
    if record.get("claim_boundary") != config["reporting"]["evidence_boundary"]:
        problems.append("Phase 14b evidence boundary drifted")

    declared_protocol = record.get("protocol", {})
    frozen_endpoints = config["protocol"]["total_physical_shot_endpoints"]
    if declared_protocol.get("total_physical_shot_endpoints") != frozen_endpoints:
        problems.append("headline shot grid drifted from the preregistration")
    if declared_protocol.get("frozen_total_physical_shot_endpoints") != frozen_endpoints:
        problems.append("frozen shot-grid copy drifted")
    if declared_protocol.get("allocator") != config["protocol"]["allocator"]:
        problems.append("allocator drifted from the preregistration")
    if declared_protocol.get("confidence") != config["protocol"]["confidence"]:
        problems.append("confidence family drifted from the preregistration")
    if declared_protocol.get("accuracy") != config["protocol"]["accuracy"]:
        problems.append("accuracy target drifted from the preregistration")
    if declared_protocol.get("seed_roots") != config["protocol"]["seed_roots"]:
        problems.append("seed roots drifted from the preregistration")
    if declared_protocol.get("covariance_audit") != config["protocol"]["covariance_audit"]:
        problems.append("covariance-audit declaration drifted")
    if declared_protocol.get("primary_estimator") != "single_assignment":
        problems.append("pooled estimator was promoted into the headline")
    if declared_protocol.get("pooled_estimator_role") != "covariance_audit_only":
        problems.append("pooled estimator role drifted")

    system = record.get("system", {})
    for field in (
        "system", "mapping", "n_qubits", "basis_size", "retained_rank",
        "basis_labels", "measured_word_universe", "ritz_functional_measured_support",
    ):
        expected_field = config["system"].get(field)
        if system.get(field) != expected_field:
            problems.append(f"system field {field} drifted")
    if not math.isclose(
        system.get("exact_subspace_bias_millihartree", math.inf),
        config["system"]["inherited_exact_subspace_bias_millihartree"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        problems.append("exact subspace bias drifted")

    cards = {row.get("name"): row for row in record.get("device_cards", [])}
    if set(cards) != set(EXPECTED_CARD_SHA256):
        problems.append("device-card set drifted")
    for name, digest in EXPECTED_CARD_SHA256.items():
        if cards.get(name, {}).get("sha256") != digest:
            problems.append(f"{name}: device-card semantic hash drifted")

    protocols = record.get("protocols", {})
    if set(protocols) != set(PROTOCOL_NAMES):
        problems.append("record must contain exactly QWC and fully commuting protocols")
        return problems
    roots = config["protocol"]["seed_roots"]
    stochastic_radius = config["protocol"]["accuracy"]["stochastic_radius_hartree"]
    audit = config["protocol"]["covariance_audit"]
    for protocol_index, name in enumerate(PROTOCOL_NAMES):
        row = protocols[name]
        if row.get("protocol_index") != protocol_index:
            problems.append(f"{name}: protocol index drifted")
        settings = row.get("compiled_plan", {}).get("settings", 0)
        headline = row.get("headline", [])
        if len(headline) != len(frozen_endpoints):
            problems.append(f"{name}: headline look count drifted")
        else:
            for endpoint_index, (endpoint, look) in enumerate(zip(frozen_endpoints, headline)):
                problems.extend(_row_problems(
                    f"{name}/look-{endpoint_index}",
                    look,
                    endpoint_index=endpoint_index,
                    endpoint=endpoint,
                    settings=settings,
                    seed_root=roots["headline"],
                    protocol_index=protocol_index,
                    stochastic_radius=stochastic_radius,
                ))
        problems.extend(_certification_problems(name, row, max(frozen_endpoints)))
        problems.extend(_audit_problems(
            name,
            row,
            protocol_index=protocol_index,
            seed_root=roots["covariance_audit"],
            replicas=audit["replicas"],
            total_shots=audit["total_physical_shots"],
            settings=settings,
        ))
        if set(row.get("device_costs", {})) != set(EXPECTED_CARD_SHA256):
            problems.append(f"{name}: device-cost card set drifted")

    failures = _blocking_failures(record, config)
    derived = _decision(protocols, config, failures)
    problems.extend(
        f"decision: {problem}"
        for problem in compare_json_records(
            record.get("decision"), derived, atol=0.0, rtol=1e-12
        )
    )
    return problems


def _ancestry_problems() -> list[str]:
    """Check ordering when the merge commit is present in the local checkout."""
    known = subprocess.run(
        ["git", "cat-file", "-e", f"{PREREGISTRATION_MERGE_COMMIT}^{{commit}}"],
        capture_output=True,
    )
    if known.returncode:
        return []
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_MERGE_COMMIT, "HEAD"],
        capture_output=True,
    )
    return [] if ancestor.returncode == 0 else [
        "the sampled implementation does not descend from the preregistration merge"
    ]


def main() -> int:
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    problems = contract_problems(expected)
    problems.extend(_ancestry_problems())
    stream = sampling_stream_mismatch(expected, packages=("numpy", "scipy"))
    if stream:
        print("Phase 14b QWC-versus-FC: FAIL (build environment differs)")
        for problem in stream:
            print(f"  - {problem}")
        for problem in problems:
            print(f"  - {problem}")
        print("  skipped deterministic rebuild under a different sampled environment")
        return 1

    actual = build_record()
    problems.extend(compare_json_records(
        expected,
        actual,
        atol=CROSS_MACHINE_ATOL,
        rtol=CROSS_MACHINE_RTOL,
    ))
    problems.extend(
        f"rebuilt record: {problem}" for problem in contract_problems(actual)
    )
    if problems:
        print("Phase 14b QWC-versus-FC: FAIL")
        for problem in problems[:50]:
            print(f"  - {problem}")
        return 1
    print("Phase 14b QWC-versus-FC: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
