"""Recompute and gate the R2b raw-pool fermion-mapping record."""

from __future__ import annotations

import json

from clifford_qc.reproducibility import compare_json_records

try:  # package import in tests versus direct script execution
    from benchmarks.run_mapping_axis import (
        REFERENCE,
        SCHEMA,
        build_record,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_mapping_axis import REFERENCE, SCHEMA, build_record


ARMS = ["jw", "parity", "parity+2q", "bk", "bk+2q"]
SYSTEMS = ["h4", "beh2", "h2o_cas8e6o", "hubbard_2x2"]


def _summary_problems(summary: dict, *, prefix: str) -> list[str]:
    problems = []
    total = summary.get("sum")
    mean = summary.get("mean")
    maximum = summary.get("max")
    if not isinstance(total, int) or total < 0:
        problems.append(f"{prefix}: invalid sum")
    if not isinstance(mean, (int, float)) or mean < 0:
        problems.append(f"{prefix}: invalid mean")
    if not isinstance(maximum, int) or maximum < 0:
        problems.append(f"{prefix}: invalid maximum")
    return problems


def contract_problems(record: dict) -> list[str]:
    """Return structural, tier, and invariant-ledger violations."""
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append("schema is not mapping-axis v1")
    if record.get("phase") != "R2b_raw_pool_mapping_axis":
        problems.append("phase label drifted")
    if record.get("scope") != "complete":
        problems.append("committed record is not complete")
    if record.get("mapping_arms") != ARMS:
        problems.append("mapping-arm order drifted")
    if not str(record.get("qr2", "")).startswith("PASS"):
        problems.append("QR2 is not a pass")

    measurement = record.get("measurement_contract", {})
    target = measurement.get("accuracy_target_millihartree")
    uniform_shots = measurement.get("uniform_raw_shots_per_setting")
    if measurement.get("estimator") != "single_assignment":
        problems.append("mapping axis is not single-assignment")
    if measurement.get("accuracy_evidence_tier") != "asymptotic":
        problems.append("accuracy-matched tier is not asymptotic")
    if not isinstance(target, (int, float)) or target <= 0:
        problems.append("invalid accuracy target")
    if not isinstance(uniform_shots, int) or uniform_shots <= 0:
        problems.append("invalid fixed-shot schedule")

    cards = record.get("device_cards", [])
    card_names = [card.get("name") for card in cards]
    card_hashes = {card.get("name"): card.get("sha256") for card in cards}
    if card_names != ["ion-like", "logical-alltoall", "superconducting-like"]:
        problems.append(f"unexpected device cards {card_names}")

    systems = record.get("systems", [])
    if [system.get("system") for system in systems] != SYSTEMS:
        problems.append("system scope or order drifted")
        return problems

    eligible = []
    for system in systems:
        key = system["system"]
        prefix = key
        arms = system.get("arms", [])
        if [arm.get("mapping") for arm in arms] != ARMS:
            problems.append(f"{prefix}: arm order or coverage drifted")
            continue
        source_words = arms[0].get("invariants", {}).get("word_universe_before")
        source_energy = arms[0].get("basis", {}).get("ground_energy")
        all_priced = True
        for arm in arms:
            mapping = arm["mapping"]
            here = f"{prefix}/{mapping}"
            invariant = arm.get("invariants", {})
            encoding = arm.get("encoding", {})
            basis = arm.get("basis", {})
            weights = arm.get("weights", {}).get(
                "deduplicated_word_universe", {}
            )
            ledger = arm.get("measurement", {})
            reduced = mapping.endswith("+2q")

            if invariant.get("mapping_name") != mapping:
                problems.append(f"{here}: invariant mapping label drifted")
            expected_measured = encoding.get("source_qubits", 0) - (2 if reduced else 0)
            if encoding.get("measured_qubits") != expected_measured:
                problems.append(f"{here}: measured qubit count is inconsistent")
            if invariant.get("word_universe_before") != source_words:
                problems.append(f"{here}: physical source universe changed")
            if reduced:
                if invariant.get("word_bijection_checked") is not False:
                    problems.append(f"{here}: reduced arm claims a word bijection")
            elif (
                invariant.get("word_bijection_checked") is not True
                or invariant.get("word_universe_after") != source_words
            ):
                problems.append(f"{here}: pure encoding broke the word bijection")
            if invariant.get("max_generator_leakage", float("inf")) > invariant.get(
                "leakage_tolerance", -1.0
            ):
                problems.append(f"{here}: generator leakage exceeds tolerance")
            if abs(float(basis.get("ground_energy", float("inf"))) - source_energy) > 1e-9:
                problems.append(f"{here}: Ritz energy changed across mappings")
            if basis.get("labels") != system.get("selected_domain", {}).get("labels"):
                problems.append(f"{here}: physical selected domain changed")
            if weights.get("distinct_words") != invariant.get("word_universe_after"):
                problems.append(f"{here}: weight and invariant universes disagree")

            identity_words = int(weights.get("histogram", {}).get("0", 0))
            measured_words = weights.get("distinct_words", 0) - identity_words
            group_size = ledger.get("group_size", {})
            if group_size.get("sum") != measured_words:
                problems.append(f"{here}: QWC cover is not an exact measured-word partition")
            if ledger.get("protocol") != "qwc_basis_cover":
                problems.append(f"{here}: protocol drifted")
            if not str(ledger.get("validity", "")).startswith("PASS"):
                problems.append(f"{here}: QWC validity is not a pass")
            problems.extend(_summary_problems(group_size, prefix=f"{here}/group_size"))

            costs = arm.get("device_costs", {})
            if set(costs) != set(card_names):
                problems.append(f"{here}: fixed-shot device ledger is incomplete")
            for card_name, cost_pair in costs.items():
                if cost_pair.get("device_card_sha256") != card_hashes.get(card_name):
                    problems.append(f"{here}/{card_name}: device-card hash drifted")
                fixed = cost_pair.get("fixed_uniform_raw_shots", {})
                if fixed.get("raw_shots") != ledger.get("settings", 0) * uniform_shots:
                    problems.append(f"{here}/{card_name}: fixed-shot total drifted")
                if fixed.get("accuracy", {}).get("evidence_tier") != "exact":
                    problems.append(f"{here}/{card_name}: fixed-shot tier drifted")

            matched = arm.get("accuracy_matched_cost", {})
            bias = matched.get("exact_subspace_bias_millihartree", float("inf"))
            expected = "priced" if bias < target else "bias_floor_exceeds_target"
            if matched.get("status") != expected:
                problems.append(f"{here}: accuracy status disagrees with bias floor")
            if matched.get("evidence_tier") != "asymptotic":
                problems.append(f"{here}: accuracy tier drifted")
            if expected == "priced":
                if set(matched.get("device_costs", {})) != set(card_names):
                    problems.append(f"{here}: accuracy device ledger is incomplete")
            elif matched.get("device_costs"):
                problems.append(f"{here}: unattainable target was priced")
            all_priced &= expected == "priced"
        if all_priced:
            eligible.append(key)

    accuracy = record.get("qr3", {}).get("accuracy_matched", {})
    if accuracy.get("eligible_systems") != eligible:
        problems.append("QR3 accuracy eligibility disagrees with arm bias floors")
    expected_verdict = (
        "eligible_for_cross_instance_comparison"
        if len(eligible) >= 2
        else "insufficient_eligible_instances"
    )
    if accuracy.get("verdict") != expected_verdict:
        problems.append("QR3 accuracy verdict disagrees with eligibility")
    return problems


def main() -> int:
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(expected, actual, atol=1e-10, rtol=1e-10)
    problems.extend(contract_problems(expected))
    problems.extend(
        f"rebuilt record: {problem}" for problem in contract_problems(actual)
    )
    if problems:
        print("mapping axis: FAIL")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1
    print("mapping axis: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
