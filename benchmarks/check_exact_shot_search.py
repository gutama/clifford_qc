"""Recompute and gate the R1 oracle-reference nonlinear shot search."""

from __future__ import annotations

import json

from clifford_qc.reproducibility import compare_json_records

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_exact_shot_search import (
        BLOCK_SIZES,
        CONFIRMATORY_REPLICAS,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        REFERENCE,
        SEARCH_ENDPOINTS,
        build_record,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_exact_shot_search import (
        BLOCK_SIZES,
        CONFIRMATORY_REPLICAS,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        REFERENCE,
        SEARCH_ENDPOINTS,
        build_record,
    )


def contract_problems(record: dict) -> list[str]:
    problems = []
    if record.get("schema") != "clifford_qc.exact_shot_search.v1":
        problems.append("schema is not exact-shot-search v1")
    if record.get("evidence_tier") != "exact":
        problems.append("oracle comparator is not labelled exact")
    if record.get("search_uncertainty_evidence") != "heuristic":
        problems.append("Monte Carlo search uncertainty is not labelled heuristic")
    protocol = record.get("protocol", {})
    if protocol.get("exploratory_replicas", 0) < 30:
        problems.append("headline record has fewer than 30 exploratory replicas")
    if protocol.get("confirmatory_replicas", 0) < 100:
        problems.append("headline record has fewer than 100 confirmatory replicas")
    if protocol.get("search_endpoints_effective_shots_per_setting") != list(SEARCH_ENDPOINTS):
        problems.append("shot-search endpoint grid drifted")
    cards = record.get("device_cards", [])
    card_hashes = {card.get("name"): card.get("sha256") for card in cards}
    systems = record.get("systems", {})
    if set(systems) != {"h4", "beh2"}:
        problems.append("system pair is not H4 and BeH2")
        return problems

    h4 = systems["h4"]
    if h4.get("status") != "bias_floor_exceeds_target":
        problems.append("H4 did not stop at its exact subspace bias floor")
    if h4.get("exact_subspace_bias_millihartree", 0.0) <= record.get(
        "accuracy_target_millihartree", float("inf")
    ):
        problems.append("H4 bias-floor status disagrees with its numbers")
    for row in h4.get("rows", []):
        for estimator, arm in row.get("estimators", {}).items():
            if arm.get("device_costs"):
                problems.append(f"H4 k={row.get('block_size')} {estimator} was priced")

    beh2 = systems["beh2"]
    rows = beh2.get("rows", [])
    if [row.get("block_size") for row in rows] != list(BLOCK_SIZES):
        problems.append("BeH2 block-size ladder is incomplete or unordered")
    for row in rows:
        if set(row.get("estimators", {})) != set(ESTIMATORS):
            problems.append(f"k={row.get('block_size')}: estimator pair is incomplete")
            continue
        for estimator, arm in row["estimators"].items():
            exploration = arm.get("exploration", [])
            if [item.get("effective_shots_per_setting") for item in exploration] != list(
                SEARCH_ENDPOINTS
            ):
                problems.append(f"k={row['block_size']} {estimator}: exploration grid drift")
            search = arm.get("shot_to_target", {})
            passing = search.get("confirmed_passing_effective_shots_per_setting")
            confirmation = {
                item.get("effective_shots_per_setting"): item
                for item in arm.get("confirmation", [])
            }
            prices = arm.get("device_costs", {})
            if passing is None:
                if prices:
                    problems.append(
                        f"k={row['block_size']} {estimator}: unconfirmed target was priced"
                    )
                continue
            if not confirmation.get(passing, {}).get("passes_target"):
                problems.append(
                    f"k={row['block_size']} {estimator}: priced endpoint did not pass"
                )
            if set(prices) != set(card_hashes):
                problems.append(
                    f"k={row['block_size']} {estimator}: device-card prices incomplete"
                )
            for name, cost in prices.items():
                device = cost.get("device_card", {})
                if device.get("sha256") != card_hashes.get(name):
                    problems.append(
                        f"k={row['block_size']} {estimator}: {name} hash drift"
                    )
                if cost.get("accuracy", {}).get("evidence_tier") != "exact":
                    problems.append(
                        f"k={row['block_size']} {estimator}: {name} wrong tier"
                    )
    return problems


def main() -> int:
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record(
        exploratory_replicas=EXPLORATORY_REPLICAS,
        confirmatory_replicas=CONFIRMATORY_REPLICAS,
        workers=4,
    )
    problems = compare_json_records(expected, actual, atol=1e-12, rtol=1e-12)
    problems.extend(contract_problems(expected))
    problems.extend(f"rebuilt record: {problem}" for problem in contract_problems(actual))
    if problems:
        print("exact shot search: FAIL")
        for problem in problems[:30]:
            print(f"  {problem}")
        return 1
    print("exact shot search: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
