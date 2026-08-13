"""Recompute and gate the Clifford measurement-hierarchy records."""

from __future__ import annotations

import json

from clifford_qc.reproducibility import compare_json_records

from run_clifford_hierarchy import (
    SYSTEMS,
    build_record,
    legacy_v2_projection,
    record_path,
)


def v2_record_path(system: str):
    path = record_path(system)
    return path.with_name(f"{path.stem}_v2.json")


def v3_contract_problems(record: dict) -> list[str]:
    """Checks that are structural rather than tolerant record comparisons.

    Run this against the committed artifact as well as the rebuilt record: the
    tier fields are literals in ``build_record``, so asserting them only on a
    record this process just built restates the producer instead of gating the
    file a reader actually consumes.
    """
    problems: list[str] = []
    if record.get("schema") != "clifford_qc.clifford_measurement_hierarchy.v3":
        problems.append("schema is not hierarchy v3")
    cards = record.get("device_cards", [])
    names = [card.get("name") for card in cards]
    if names != ["ion-like", "logical-alltoall", "superconducting-like"]:
        problems.append(f"unexpected device cards {names}")
    matching = record.get("accuracy_matching", {})
    if matching.get("default_tier_status") != "unavailable":
        problems.append("exact-tier C(epsilon) must remain unavailable")
    if matching.get("reported_evidence_tier") != "asymptotic":
        problems.append("the priced hierarchy must identify its asymptotic tier")
    ordering = record.get("accuracy_matched_ordering", {})
    if set(ordering) != set(names):
        problems.append("accuracy-matched ordering is not reported for every card")
    for row in record.get("rows", []):
        estimators = row.get("estimators", {})
        if set(estimators) != {"single_assignment", "pooled"}:
            problems.append(f"k={row.get('block_size')}: estimator pair is incomplete")
            continue
        assigned = estimators["single_assignment"]
        pooled = estimators["pooled"]
        if pooled["raw_word_shots"]["total"] < assigned["raw_word_shots"]["total"]:
            problems.append(f"k={row['block_size']}: pooling lost compatible observations")
        for card in cards:
            name = card["name"]
            cost = row.get("device_costs", {}).get(name)
            if cost is None:
                problems.append(f"k={row['block_size']}: missing cost for {name}")
            elif cost.get("device_card_sha256") != card.get("sha256"):
                problems.append(f"k={row['block_size']}: card hash drift for {name}")
        matched = row.get("accuracy_matched_costs", {})
        if set(matched) != {"single_assignment", "pooled"}:
            problems.append(f"k={row['block_size']}: accuracy-matched estimator pair is incomplete")
        for estimator, cost in matched.items():
            if cost.get("evidence_tier") != "asymptotic":
                problems.append(
                    f"k={row['block_size']} {estimator}: wrong C(epsilon) tier"
                )
            if cost.get("status") == "bias_floor_exceeds_target" and cost.get("device_costs"):
                problems.append(
                    f"k={row['block_size']} {estimator}: priced an unattainable target"
                )
            # A scalar runtime is only meaningful with the card that attains
            # it, and a rung some declared card cannot run is not "priced".
            scalar = cost.get("C_time_epsilon_us")
            named = cost.get("C_time_epsilon_device_card")
            if (scalar is None) != (named is None):
                problems.append(
                    f"k={row['block_size']} {estimator}: C(epsilon) without a named card"
                )
            if cost.get("status") == "priced" and cost.get("inadmissible_device_cards"):
                problems.append(
                    f"k={row['block_size']} {estimator}: priced despite an "
                    "inadmissible declared card"
                )
    return problems


def v2_projection_problems(system: str, actual: dict) -> list[str]:
    """Compare the legacy projection, reporting rather than raising on setup."""
    try:
        projected = legacy_v2_projection(actual)
    except ValueError as exc:
        return [f"legacy v2 projection failed: {exc}"]
    try:
        expected_v2 = json.loads(v2_record_path(system).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"frozen v2 control is unreadable: {exc}"]
    return compare_json_records(
        expected_v2, projected, path="$.legacy_v2", atol=1e-12, rtol=1e-12,
    )


def main() -> int:
    failures = 0
    for system in sorted(SYSTEMS):
        path = record_path(system)
        expected = json.loads(path.read_text(encoding="utf-8"))
        actual = build_record(system)
        problems = compare_json_records(expected, actual, atol=1e-12, rtol=1e-12)
        problems.extend(v3_contract_problems(actual))
        problems.extend(
            f"committed record: {problem}"
            for problem in v3_contract_problems(expected)
        )
        problems.extend(v2_projection_problems(system, actual))
        if problems:
            failures += 1
            print(f"clifford hierarchy {system}: FAIL")
            for problem in problems[:20]:
                print(f"  {problem}")
        else:
            print(f"clifford hierarchy {system}: PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
