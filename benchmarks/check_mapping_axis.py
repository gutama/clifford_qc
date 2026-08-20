"""Recompute and gate the R2b raw-pool fermion-mapping record."""

from __future__ import annotations

import json
import math

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
# Stated here rather than read from the config, deliberately: this list is
# the checker's independent claim about the record's scope, so a config edit
# alone must not be able to change it silently. The two H4 rows are the same
# instance at two subspace budgets and are ordered as a pair.
SYSTEMS = ["h4", "h4_converged", "beh2", "h2o_cas8e6o", "hubbard_2x2"]

# Both fields are (subspace energy - exact energy) in millihartree: a residue of
# order 1e-3 mHa left by two energies of order 1e4 mHa (BeH2 sits at -15566 mHa),
# so seven significant digits are gone to cancellation before the comparison
# starts.  Their reproducible scale is set by the energies, not by the residue --
# the measured spread between OMP_NUM_THREADS=1 and =8 is 1.5e-10 mHa, which the
# default atol=1e-10 sits directly on top of.  1e-8 mHa clears that by ~70x while
# staying five orders below the smallest bias the record reports and eight below
# the 1.6 mHa target, so a scientifically real change still fails.
ENERGY_DIFFERENCE_TOLERANCES = {
    "error_millihartree": (1e-10, 1e-8),
    "exact_subspace_bias_millihartree": (1e-10, 1e-8),
}
GROUPING_PROTOCOLS = {
    "h4": "qwc_groups",
    "h4_converged": "qwc_groups",
    "beh2": "qwc_groups",
    "h2o_cas8e6o": "qwc_basis_cover",
    "hubbard_2x2": "qwc_groups",
}
DEFERRED_TO_R3 = [
    "G(k) across protocol rungs",
    "coverage across protocol rungs",
]


def _is_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _nonnegative_number(value, *, label: str, problems: list[str]) -> float | None:
    if not _is_number(value) or float(value) < 0.0:
        problems.append(f"{label}: invalid non-negative number")
        return None
    return float(value)


def _summary_problems(summary: dict, *, prefix: str) -> list[str]:
    problems = []
    if not isinstance(summary, dict):
        return [f"{prefix}: summary is not an object"]
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


def _invariant_problems(invariant: dict, *, prefix: str) -> list[str]:
    problems: list[str] = []
    if not isinstance(invariant, dict):
        return [f"{prefix}: invariant ledger is not an object"]
    relative = _nonnegative_number(
        invariant.get("relative_tolerance"),
        label=f"{prefix}/relative_tolerance",
        problems=problems,
    )
    absolute = _nonnegative_number(
        invariant.get("absolute_tolerance"),
        label=f"{prefix}/absolute_tolerance",
        problems=problems,
    )
    leakage_tolerance = _nonnegative_number(
        invariant.get("leakage_tolerance"),
        label=f"{prefix}/leakage_tolerance",
        problems=problems,
    )
    _nonnegative_number(
        invariant.get("zero_tolerance"),
        label=f"{prefix}/zero_tolerance",
        problems=problems,
    )
    if relative == 0.0 and absolute == 0.0:
        problems.append(f"{prefix}: relative and absolute tolerances are both zero")

    numerical_gates = (
        ("max_overlap_matrix_error", "overlap_matrix_scale"),
        ("max_hamiltonian_matrix_error", "hamiltonian_matrix_scale"),
        ("max_operator_transport_error", "operator_transport_scale"),
        ("max_ritz_error", "ritz_scale"),
        ("reference_energy_error", "reference_energy_scale"),
    )
    for error_key, scale_key in numerical_gates:
        error = _nonnegative_number(
            invariant.get(error_key), label=f"{prefix}/{error_key}", problems=problems
        )
        scale = _nonnegative_number(
            invariant.get(scale_key), label=f"{prefix}/{scale_key}", problems=problems
        )
        if None not in (error, scale, relative, absolute):
            allowed = absolute + relative * scale
            if error > allowed:
                problems.append(
                    f"{prefix}: {error_key} exceeds its stored tolerance "
                    f"({error:.3e} > {allowed:.3e})"
                )

    leakage = _nonnegative_number(
        invariant.get("max_generator_leakage"),
        label=f"{prefix}/max_generator_leakage",
        problems=problems,
    )
    if None not in (leakage, leakage_tolerance) and leakage > leakage_tolerance:
        problems.append(f"{prefix}: generator leakage exceeds tolerance")

    dense_checked = invariant.get("dense_spectrum_checked")
    dense_limit = invariant.get("dense_spectrum_max_qubits")
    source_qubits = invariant.get("source_qubits")
    if (
        not isinstance(dense_limit, int)
        or isinstance(dense_limit, bool)
        or dense_limit < 0
    ):
        problems.append(f"{prefix}: invalid dense-spectrum qubit limit")
        dense_limit = None
    if not isinstance(source_qubits, int) or isinstance(source_qubits, bool):
        problems.append(f"{prefix}: invalid source qubit count")
        source_qubits = None
    if dense_checked is True:
        if invariant.get("dense_spectrum_reason") is not None:
            problems.append(
                f"{prefix}: checked dense spectrum carries an exclusion reason"
            )
        error = _nonnegative_number(
            invariant.get("max_spectrum_error"),
            label=f"{prefix}/max_spectrum_error",
            problems=problems,
        )
        scale = _nonnegative_number(
            invariant.get("spectrum_scale"),
            label=f"{prefix}/spectrum_scale",
            problems=problems,
        )
        if None not in (error, scale, relative, absolute):
            allowed = absolute + relative * scale
            if error > allowed:
                problems.append(
                    f"{prefix}: max_spectrum_error exceeds its stored tolerance"
                )
    elif dense_checked is False:
        if invariant.get("dense_spectrum_reason") != (
            "source qubits exceed the declared dense-spectrum materialization limit"
        ):
            problems.append(f"{prefix}: dense-spectrum omission has no declared reason")
        if invariant.get("max_spectrum_error") is not None:
            problems.append(f"{prefix}: unchecked dense spectrum carries an error")
        if invariant.get("spectrum_scale") is not None:
            problems.append(f"{prefix}: unchecked dense spectrum carries a scale")
        if (
            source_qubits is not None
            and dense_limit is not None
            and source_qubits <= dense_limit
        ):
            problems.append(
                f"{prefix}: dense spectrum omitted below its declared limit"
            )
    else:
        problems.append(f"{prefix}: invalid dense-spectrum checked flag")
    return problems


def _metric_value(arm: dict, metric: str) -> float | None:
    if metric == "qwc_settings":
        value = arm.get("measurement", {}).get("settings")
    else:
        value = (
            arm.get("weights", {})
            .get("deduplicated_word_universe", {})
            .get("mean_weight")
        )
    if not _is_number(value) or float(value) <= 0.0:
        return None
    return float(value)


def _recompute_spread(
    systems: list[dict], metric: str, *, problems: list[str]
) -> dict | None:
    within = {}
    for system in systems:
        values = [_metric_value(arm, metric) for arm in system.get("arms", [])]
        if len(values) != len(ARMS) or any(value is None for value in values):
            problems.append(
                f"QR3/{metric}: invalid arm metric for {system.get('system')}"
            )
            return None
        minimum = min(values)
        maximum = max(values)
        within[system["system"]] = {
            "minimum": minimum,
            "maximum": maximum,
            "mapping_spread_factor": maximum / minimum,
        }
    by_mapping = {}
    for index, mapping in enumerate(ARMS):
        values = [_metric_value(system["arms"][index], metric) for system in systems]
        if any(value is None for value in values):
            problems.append(
                f"QR3/{metric}: invalid cross-instance metric for {mapping}"
            )
            return None
        minimum = min(values)
        maximum = max(values)
        by_mapping[mapping] = {
            "minimum": minimum,
            "maximum": maximum,
            "instance_spread_factor": maximum / minimum,
        }
    max_mapping = max(item["mapping_spread_factor"] for item in within.values())
    min_instance = min(item["instance_spread_factor"] for item in by_mapping.values())
    return {
        "systems_included": [system["system"] for system in systems],
        "within_system": within,
        "across_instances": by_mapping,
        "max_mapping_spread_factor": max_mapping,
        "min_instance_spread_factor": min_instance,
        "margin_factor": min_instance / max_mapping,
        "verdict": (
            "mapping_spread_smaller_than_instance_spread"
            if max_mapping < min_instance
            else "mapping_spread_not_smaller_than_instance_spread"
        ),
    }


def _contract_problems(record: dict) -> list[str]:
    """Return structural, tier, invariant, and recomputed-QR3 violations."""
    problems: list[str] = []
    if not isinstance(record, dict):
        return ["record is not an object"]
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
    if not isinstance(measurement, dict):
        return [*problems, "measurement contract is not an object"]
    target = measurement.get("accuracy_target_millihartree")
    uniform_shots = measurement.get("uniform_raw_shots_per_setting")
    if measurement.get("estimator") != "single_assignment":
        problems.append("mapping axis is not single-assignment")
    if measurement.get("accuracy_evidence_tier") != "asymptotic":
        problems.append("accuracy-matched tier is not asymptotic")
    if not _is_number(target) or float(target) <= 0:
        problems.append("invalid accuracy target")
        target = None
    else:
        target = float(target)
    if (
        not isinstance(uniform_shots, int)
        or isinstance(uniform_shots, bool)
        or uniform_shots <= 0
    ):
        problems.append("invalid fixed-shot schedule")
        uniform_shots = None
    if measurement.get("grouping_protocol_by_system") != GROUPING_PROTOCOLS:
        problems.append("grouping protocol policy drifted")
    if measurement.get("deferred_to_r3") != DEFERRED_TO_R3:
        problems.append("R2b-to-R3 deferral ledger drifted")

    cards = record.get("device_cards", [])
    if not isinstance(cards, list) or any(not isinstance(card, dict) for card in cards):
        problems.append("device cards are malformed")
        cards = []
    card_names = [card.get("name") for card in cards]
    card_hashes = {card.get("name"): card.get("sha256") for card in cards}
    if card_names != ["ion-like", "logical-alltoall", "superconducting-like"]:
        problems.append(f"unexpected device cards {card_names}")

    systems = record.get("systems", [])
    if not isinstance(systems, list) or any(
        not isinstance(system, dict) for system in systems
    ):
        problems.append("systems ledger is malformed")
        return problems
    if [system.get("system") for system in systems] != SYSTEMS:
        problems.append("system scope or order drifted")
        return problems

    eligible = []
    for system in systems:
        key = system["system"]
        prefix = key
        expected_protocol = GROUPING_PROTOCOLS[key]
        if system.get("grouping_protocol") != expected_protocol:
            problems.append(f"{prefix}: grouping protocol drifted")
        arms = system.get("arms", [])
        if not isinstance(arms, list) or any(not isinstance(arm, dict) for arm in arms):
            problems.append(f"{prefix}: arms ledger is malformed")
            continue
        if [arm.get("mapping") for arm in arms] != ARMS:
            problems.append(f"{prefix}: arm order or coverage drifted")
            continue
        source_words = arms[0].get("invariants", {}).get("word_universe_before")
        source_energy = arms[0].get("basis", {}).get("ground_energy")
        if not _is_number(source_energy):
            problems.append(f"{prefix}: invalid source ground energy")
            source_energy = None
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

            problems.extend(_invariant_problems(invariant, prefix=here))

            if invariant.get("mapping_name") != mapping:
                problems.append(f"{here}: invariant mapping label drifted")
            source_qubits = encoding.get("source_qubits")
            expected_measured = (
                source_qubits - (2 if reduced else 0)
                if isinstance(source_qubits, int)
                and not isinstance(source_qubits, bool)
                else None
            )
            if (
                expected_measured is None
                or encoding.get("measured_qubits") != expected_measured
            ):
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
            ground_energy = basis.get("ground_energy")
            if not _is_number(ground_energy):
                problems.append(f"{here}: invalid ground energy")
            elif (
                source_energy is not None
                and abs(float(ground_energy) - source_energy) > 1e-9
            ):
                problems.append(f"{here}: Ritz energy changed across mappings")
            if basis.get("labels") != system.get("selected_domain", {}).get("labels"):
                problems.append(f"{here}: physical selected domain changed")
            if weights.get("distinct_words") != invariant.get("word_universe_after"):
                problems.append(f"{here}: weight and invariant universes disagree")

            histogram = weights.get("histogram", {})
            identity_words = (
                histogram.get("0", 0) if isinstance(histogram, dict) else None
            )
            distinct_words = weights.get("distinct_words")
            if (
                not isinstance(identity_words, int)
                or isinstance(identity_words, bool)
                or not isinstance(distinct_words, int)
                or isinstance(distinct_words, bool)
            ):
                problems.append(f"{here}: invalid measured-word counts")
                measured_words = None
            else:
                measured_words = distinct_words - identity_words
            group_size = ledger.get("group_size", {})
            if measured_words is None or group_size.get("sum") != measured_words:
                problems.append(
                    f"{here}: QWC cover is not an exact measured-word partition"
                )
            if ledger.get("protocol") != expected_protocol:
                problems.append(f"{here}: protocol drifted")
            if not str(ledger.get("validity", "")).startswith("PASS"):
                problems.append(f"{here}: QWC validity is not a pass")
            problems.extend(_summary_problems(group_size, prefix=f"{here}/group_size"))
            settings_count = ledger.get("settings")
            if (
                not isinstance(settings_count, int)
                or isinstance(settings_count, bool)
                or settings_count <= 0
            ):
                problems.append(f"{here}: invalid setting count")
                settings_count = None
            resources = ledger.get("resource_metrics", {})
            if not isinstance(resources, dict):
                problems.append(f"{here}: invalid resource metrics")
                resources = {}
            for name in ("N_1q", "N_2q", "D_1q", "D_2q"):
                summary = resources.get(name, {})
                problems.extend(
                    _summary_problems(summary, prefix=f"{here}/resource_metrics/{name}")
                )
            for name in ("N_2q", "D_2q"):
                summary = resources.get(name, {})
                if not isinstance(summary, dict) or any(
                    summary.get(field) != 0 for field in ("sum", "mean", "max")
                ):
                    problems.append(
                        f"{here}: QWC {name} ledger is not identically zero"
                    )

            costs = arm.get("device_costs", {})
            if set(costs) != set(card_names):
                problems.append(f"{here}: fixed-shot device ledger is incomplete")
            for card_name, cost_pair in costs.items():
                if cost_pair.get("device_card_sha256") != card_hashes.get(card_name):
                    problems.append(f"{here}/{card_name}: device-card hash drifted")
                fixed = cost_pair.get("fixed_uniform_raw_shots", {})
                if (
                    uniform_shots is not None
                    and settings_count is not None
                    and fixed.get("raw_shots") != settings_count * uniform_shots
                ):
                    problems.append(f"{here}/{card_name}: fixed-shot total drifted")
                if fixed.get("accuracy", {}).get("evidence_tier") != "exact":
                    problems.append(f"{here}/{card_name}: fixed-shot tier drifted")

            matched = arm.get("accuracy_matched_cost", {})
            bias = matched.get("exact_subspace_bias_millihartree", float("inf"))
            if not _is_number(bias) or float(bias) < 0.0:
                problems.append(f"{here}: invalid exact subspace bias")
                expected = None
            elif target is None:
                expected = None
            else:
                expected = (
                    "priced" if float(bias) < target else "bias_floor_exceeds_target"
                )
            if expected is not None and matched.get("status") != expected:
                problems.append(f"{here}: accuracy status disagrees with bias floor")
            if matched.get("evidence_tier") != "asymptotic":
                problems.append(f"{here}: accuracy tier drifted")
            if expected == "priced":
                if set(matched.get("device_costs", {})) != set(card_names):
                    problems.append(f"{here}: accuracy device ledger is incomplete")
            elif (
                expected == "bias_floor_exceeds_target"
                and matched.get("device_costs")
            ):
                problems.append(f"{here}: unattainable target was priced")
            all_priced &= expected == "priced"
        if all_priced:
            eligible.append(key)

    qr3 = record.get("qr3", {})
    if not isinstance(qr3, dict):
        problems.append("QR3 ledger is malformed")
        return problems
    by_key = {system["system"]: system for system in systems}
    expected_structural = {
        "qwc_settings_matched_greedy": _recompute_spread(
            # The matched-greedy QWC set, named rather than filtered: H2O uses
            # the scalable cover and is excluded by GROUPING_PROTOCOLS above,
            # and naming the rest keeps a new system from joining this verdict
            # without an edit here.
            [
                by_key[key]
                for key in ("h4", "h4_converged", "beh2", "hubbard_2x2")
            ],
            "qwc_settings",
            problems=problems,
        ),
        "mean_word_weight": _recompute_spread(
            systems, "mean_word_weight", problems=problems
        ),
    }
    if all(value is not None for value in expected_structural.values()):
        problems.extend(
            f"QR3 structural: {problem}"
            for problem in compare_json_records(
                expected_structural,
                qr3.get("structural"),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        expected_structural_verdict = (
            "mapping_spread_smaller_than_instance_spread_on_both_independent_metrics"
            if all(
                summary["verdict"]
                == "mapping_spread_smaller_than_instance_spread"
                for summary in expected_structural.values()
            )
            else "mapping_spread_not_smaller_on_every_independent_metric"
        )
        if qr3.get("structural_verdict") != expected_structural_verdict:
            problems.append("QR3 structural verdict disagrees with recomputed ratios")
    exclusion = qr3.get("qwc_exclusion", {})
    if not isinstance(exclusion, dict) or exclusion.get("systems") != ["h2o_cas8e6o"]:
        problems.append("QR3 QWC exclusion drifted")
    projections = qr3.get("device_card_projections", {})
    if (
        not isinstance(projections, dict)
        or projections.get("independent_evidence") is not False
        or projections.get("cards") != card_names
    ):
        problems.append("QR3 device-card projection boundary drifted")

    accuracy = qr3.get("accuracy_matched", {})
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


def contract_problems(record: dict) -> list[str]:
    """Reject malformed records with diagnostics rather than a traceback."""
    try:
        return _contract_problems(record)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return [f"malformed record reached a guarded checker path: {exc}"]


def main() -> int:
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(
        expected, actual, atol=1e-10, rtol=1e-10,
        key_tolerances=ENERGY_DIFFERENCE_TOLERANCES,
    )
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
