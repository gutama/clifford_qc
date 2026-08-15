"""Recompute and gate the R1 oracle-reference nonlinear shot search."""

from __future__ import annotations

import json

from clifford_qc.reproducibility import (
    compare_json_records,
    sampling_stream_mismatch,
)

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_exact_shot_search import (
        BLOCK_SIZES,
        CONFIRMATORY_REPLICAS,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        MARGINAL_TARGET_FRACTION,
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
        MARGINAL_TARGET_FRACTION,
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

    target = record.get("accuracy_target_millihartree", float("inf"))
    for system_name, system in systems.items():
        bias = system.get("exact_subspace_bias_millihartree", float("inf"))
        expected_status = "bias_floor_exceeds_target" if bias >= target else "searched"
        if system.get("status") != expected_status:
            problems.append(
                f"{system_name} status disagrees with its exact subspace bias"
            )
        if expected_status == "bias_floor_exceeds_target":
            for row in system.get("rows", []):
                for estimator, arm in row.get("estimators", {}).items():
                    if arm.get("device_costs"):
                        problems.append(
                            f"{system_name} k={row.get('block_size')} "
                            f"{estimator} was priced above its bias floor"
                        )

    for system_name, system in systems.items():
        rows = system.get("rows", [])
        if [row.get("block_size") for row in rows] != list(BLOCK_SIZES):
            problems.append(
                f"{system_name} block-size ladder is incomplete or unordered"
            )
        for row in rows:
            if set(row.get("estimators", {})) != set(ESTIMATORS):
                problems.append(
                    f"{system_name} k={row.get('block_size')}: "
                    "estimator pair is incomplete"
                )
                continue
            if system.get("status") != "searched":
                continue
            for estimator, arm in row["estimators"].items():
                prefix = f"{system_name} k={row['block_size']} {estimator}"
                exploration = arm.get("exploration", [])
                if [
                    item.get("effective_shots_per_setting") for item in exploration
                ] != list(SEARCH_ENDPOINTS):
                    problems.append(f"{prefix}: exploration grid drift")
                search = arm.get("shot_to_target", {})
                passing = search.get("confirmed_passing_effective_shots_per_setting")
                failing = search.get("confirmed_failing_effective_shots_per_setting")
                confirmation = {
                    item.get("effective_shots_per_setting"): item
                    for item in arm.get("confirmation", [])
                }
                prices = arm.get("device_costs", {})
                if passing is None:
                    if prices:
                        problems.append(f"{prefix}: unconfirmed target was priced")
                    continue
                if not confirmation.get(passing, {}).get("passes_target"):
                    problems.append(f"{prefix}: priced endpoint did not pass")
                passing_endpoints = sorted(
                    endpoint for endpoint, summary in confirmation.items()
                    if summary.get("passes_target")
                )
                if passing_endpoints and passing != passing_endpoints[0]:
                    problems.append(
                        f"{prefix}: priced endpoint is not the smallest confirmed pass"
                    )
                if any(
                    endpoint > passing and not summary.get("passes_target")
                    for endpoint, summary in confirmation.items()
                ):
                    problems.append(f"{prefix}: priced crossing is nonmonotone")
                if failing is not None and confirmation.get(failing, {}).get(
                    "passes_target"
                ) is not False:
                    problems.append(
                        f"{prefix}: reported failing endpoint did not fail"
                    )
                # A crossing decided within the environment-marginal band is
                # not a resolved shot count, and must say so rather than being
                # read as one.
                marginal = search.get("environment_marginal_endpoints")
                if marginal is None:
                    problems.append(f"{prefix}: crossing margin was not recorded")
                elif bool(marginal) != bool(
                    search.get("crossing_is_environment_marginal")
                ):
                    problems.append(
                        f"{prefix}: marginal endpoints disagree with the flag"
                    )
                else:
                    for label in ("passing", "failing"):
                        fraction = search.get(f"{label}_target_margin_fraction")
                        if fraction is None:
                            continue
                        near = abs(fraction) <= MARGINAL_TARGET_FRACTION
                        if near != (label in marginal):
                            problems.append(
                                f"{prefix}: {label} margin {fraction:+.3f} "
                                "disagrees with its marginal listing"
                            )
                if set(prices) != set(card_hashes):
                    problems.append(f"{prefix}: device-card prices incomplete")
                for name, cost in prices.items():
                    device = cost.get("device_card", {})
                    if device.get("sha256") != card_hashes.get(name):
                        problems.append(f"{prefix}: {name} hash drift")
                    if cost.get("accuracy", {}).get("evidence_tier") != "exact":
                        problems.append(f"{prefix}: {name} wrong tier")
    return problems


def main() -> int:
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    # Every value here descends from sampled shots fed through a projected
    # eigensolve, so the environment check comes before the rebuild rather than
    # after it.  Under a different NumPy the ill-conditioned rank-5 solves land
    # elsewhere -- measured here as a 0.11 mHa shift on individual BeH2
    # replicas from identical shot histograms -- and comparing that reports a
    # moved quantile as though it were drift, inviting a widened tolerance.
    # That is the wrong repair, and the full search would be spent to reach it.
    stream = sampling_stream_mismatch(expected)
    if stream:
        print("exact shot search: FAIL (build environment differs)")
        for problem in stream:
            print(f"  {problem}")
        print("  skipped the rebuild: under a different environment it answers a "
              "different question rather than verifying this record")
        problems = contract_problems(expected)
        if problems:
            print("  the committed record also fails its own contracts:")
            for problem in problems[:30]:
                print(f"    {problem}")
        else:
            print("  the committed record still passes every contract check "
                  "that does not require a rebuild")
        return 1
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
