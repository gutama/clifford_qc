"""Regenerate and gate the preregistered R4a contextual structural record.

The checker compares every scientific/resource field with a fresh deterministic
build and independently derives the admission decision from the record.  It
also refuses cost, sampling, interaction-ratio, and QR5-classification fields:
passing this screen can only make a later sampled preregistration eligible.

    python benchmarks/check_r4a_contextual_screen.py
"""

from __future__ import annotations

import json
import math
import sys

from clifford_qc.reproducibility import compare_json_records

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    import benchmarks.check_r4a_preregistration as preregistration
    from benchmarks.run_r4a_contextual_screen import (
        CONFIG,
        PREREGISTRATION_COMMIT,
        REFERENCE,
        SCHEMA,
        _git_blob_sha1,
        _sha256,
        build_record,
        load_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    import check_r4a_preregistration as preregistration
    from run_r4a_contextual_screen import (
        CONFIG,
        PREREGISTRATION_COMMIT,
        REFERENCE,
        SCHEMA,
        _git_blob_sha1,
        _sha256,
        build_record,
        load_config,
    )


NUMERIC_TOLERANCES = {
    "exact_sector_energy": (1e-11, 1e-11),
    "ground_energy": (1e-11, 1e-11),
    "signed_error_millihartree": (1e-10, 1e-8),
    "bias_millihartree": (1e-10, 1e-8),
    "hamiltonian_removed_hs_fraction": (1e-10, 1e-12),
}

FORBIDDEN_KEYS = {
    "C_time",
    "Delta",
    "bootstrap",
    "confirmatory",
    "cost_bracket",
    "cost_symbol",
    "cost_us",
    "device_costs",
    "effective_shots_per_setting",
    "exploratory",
    "k_star",
    "qr5_classification",
    "r_A",
    "r_CS",
    "r_joint",
    "replicas",
    "samples",
    "shot_to_target",
    "shots",
}


def _forbidden_paths(node, path: str = "$") -> list[str]:
    if isinstance(node, dict):
        found = []
        for key, value in node.items():
            if key in FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_forbidden_paths(value, f"{path}.{key}"))
        return found
    if isinstance(node, list):
        return [
            found
            for index, value in enumerate(node)
            for found in _forbidden_paths(value, f"{path}[{index}]")
        ]
    return []


def _arm_map(rung: dict) -> dict[str, dict]:
    return {row["name"]: row for row in rung.get("arms", [])}


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _arm_problems(
    arm: dict, *, gates: dict, source_qubits: int, fixed_qubits: int
) -> list[str]:
    problems: list[str] = []
    name = arm.get("name", "<unnamed>")
    required = preregistration.EXPECTED_STRUCTURAL_FIELDS
    missing = sorted(required - set(arm))
    if missing:
        problems.append(f"{name}: missing required structural fields {missing}")
        return problems

    before = arm["basis_size_before_deduplication"]
    after = arm["basis_size_after_deduplication"]
    if (
        not _is_plain_int(before)
        or not _is_plain_int(after)
        or not 1 <= after <= before
    ):
        problems.append(f"{name}: basis sizes must be positive integer counts")
        return problems
    annihilated = arm["annihilated_candidate_indices"]
    duplicates = arm["duplicate_projected_candidate_indices"]
    for label, indices in (("annihilated", annihilated), ("duplicate", duplicates)):
        if indices != sorted(set(indices)):
            problems.append(f"{name}: {label} indices are not sorted and unique")
        if any(not _is_plain_int(index) or not 0 <= index < before for index in indices):
            problems.append(f"{name}: {label} index lies outside the source basis")
        if 0 in indices:
            problems.append(f"{name}: identity at candidate index 0 may not be removed")
    if set(annihilated) & set(duplicates):
        problems.append(f"{name}: one candidate is both annihilated and duplicate")
    if after != before - len(annihilated) - len(duplicates):
        problems.append(f"{name}: projected basis-size arithmetic does not close")
    before_labels = arm.get("basis_labels_before_deduplication", [])
    after_labels = arm.get("basis_labels_after_deduplication", [])
    if len(before_labels) != before:
        problems.append(f"{name}: source label count differs from its basis size")
    if len(after_labels) != after:
        problems.append(f"{name}: retained label count differs from its basis size")
    if not before_labels or before_labels[0] != "I":
        problems.append(f"{name}: identity is not first in the source basis")
    if not after_labels or after_labels[0] != "I":
        problems.append(f"{name}: identity is not first in the retained basis")
    rows = arm.get("duplicate_projected_candidates", [])
    if [row.get("candidate_index") for row in rows] != duplicates:
        problems.append(f"{name}: duplicate detail rows disagree with their index list")

    contextual = name.startswith("cs_")
    expected_qubits = source_qubits - fixed_qubits if contextual else source_qubits
    if not _is_plain_int(arm["active_qubits"]) or arm["active_qubits"] != expected_qubits:
        problems.append(
            f"{name}: active_qubits={arm['active_qubits']}, expected {expected_qubits}"
        )
    energy = arm.get("ground_energy")
    signed = arm.get("signed_error_millihartree")
    bias = arm["bias_millihartree"]
    if not all(isinstance(value, (int, float)) for value in (energy, signed, bias)):
        problems.append(f"{name}: energy fields must be numeric")
    elif not all(math.isfinite(float(value)) for value in (energy, signed, bias)):
        problems.append(f"{name}: energy fields must be finite")
    elif not math.isclose(bias, abs(signed), rel_tol=1e-12, abs_tol=1e-10):
        problems.append(
            f"{name}: bias is not the absolute error required for a non-variational arm"
        )

    fraction = arm["hamiltonian_removed_hs_fraction"]
    if not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0 + 1e-12:
        problems.append(f"{name}: Hamiltonian removed fraction is outside [0, 1]")
    if not contextual and fraction != 0.0:
        problems.append(f"{name}: an unrestricted control reports Hamiltonian removal")

    words = arm["word_universe"]
    if not _is_plain_int(words) or not 0 <= words <= 4 ** arm["active_qubits"] - 1:
        problems.append(f"{name}: invalid non-identity word universe {words!r}")
    if arm.get("word_universe_convention") != "non_identity_words_only":
        problems.append(f"{name}: word-universe convention drifted")
    expected_blocks = [str(value) for value in gates["block_sizes"]]
    settings = arm["settings_by_block_size"]
    if list(settings) != expected_blocks:
        problems.append(f"{name}: setting grid differs from {expected_blocks}")
    else:
        counts = [settings[key] for key in expected_blocks]
        if any(not _is_plain_int(value) or value < 0 for value in counts):
            problems.append(f"{name}: setting counts must be non-negative integers")
        if words and any(not 1 <= value <= words for value in counts):
            problems.append(f"{name}: a setting count lies outside [1, word_universe]")

    clears_bias = bias <= gates["admissible_bias_millihartree"]
    clears_words = words <= gates["word_universe_ceiling"]
    if arm.get("clears_bias_margin") != clears_bias:
        problems.append(f"{name}: bias gate flag contradicts its measured bias")
    if arm.get("clears_word_universe_ceiling") != clears_words:
        problems.append(f"{name}: word gate flag contradicts its measured universe")
    if arm.get("admissible") != (clears_bias and clears_words):
        problems.append(f"{name}: admissible flag is not the conjunction of both gates")
    rank = arm.get("retained_overlap_rank")
    if not _is_plain_int(rank) or not 1 <= rank <= after:
        problems.append(f"{name}: retained overlap rank is outside its basis size")
    return problems


def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_tier") != "structural":
        problems.append("R4a screen must remain structural evidence")
    if record.get("status") != "structural_screen_complete_no_samples":
        problems.append("R4a structural status drifted")
    prereg = record.get("preregistration", {})
    if prereg.get("path") != "benchmarks/configs/r4a_contextual_interaction.json":
        problems.append("the structural record points to the wrong preregistration")
    if prereg.get("sha256") != _sha256(CONFIG):
        problems.append("the structural record's preregistration SHA-256 drifted")
    if prereg.get("git_blob_sha1") != _git_blob_sha1(CONFIG):
        problems.append("the structural record's preregistration blob id drifted")
    if prereg.get("merged_commit") != PREREGISTRATION_COMMIT:
        problems.append("the structural record does not descend from the merged declaration")
    authority = record.get("execution_authority", {})
    if authority.get("structural_execution_consumed") is not True:
        problems.append("the structural record does not consume its one execution")
    if authority.get("sampled_execution_performed") is not False:
        problems.append("the structural record may not contain a sampled execution")
    if authority.get("sampled_execution_authorized_by_this_record") is not False:
        problems.append("the structural record may not authorize sampling by itself")
    forbidden = _forbidden_paths(record)
    if forbidden:
        problems.append(f"sampled/cost field in structural record: {forbidden[0]}")

    config = preregistration.load_config()
    gates = record.get("gates", {})
    frozen = config["structural_screen"]
    for key in (
        "accuracy_target_millihartree",
        "margin_factor",
        "admissible_bias_millihartree",
        "word_universe_ceiling",
        "word_universe_convention",
        "block_sizes",
        "grouping_rule",
        "hamiltonian_removed_hs_fraction_denominator",
    ):
        if gates.get(key) != frozen.get(key):
            problems.append(f"structural gate {key} differs from the preregistration")
    source_qubits = record.get("system", {}).get("n_qubits")
    if source_qubits != config["system"]["n_qubits"]:
        problems.append("the structural record's source qubit count drifted")
        source_qubits = config["system"]["n_qubits"]
    system = record.get("system", {})
    labels = system.get("complete_basis_labels", [])
    if not labels or labels[0] != "I" or len(labels) != len(set(labels)):
        problems.append("the complete basis labels must be unique and identity-first")
    if system.get("complete_basis_size") != len(labels):
        problems.append("the complete basis size disagrees with its label list")
    if system.get("frozen_acase_basis_labels") != config["system"]["acase_bank"][
        "basis_labels"
    ]:
        problems.append("the structural record reopened the frozen A-CASE bank")

    rungs = record.get("contextual_rungs", [])
    expected_rungs = config["contextual_restriction"]["fixed_qubit_ladder"]
    if [row.get("fixed_qubits") for row in rungs] != expected_rungs:
        problems.append("the contextual rung ladder drifted")
    expected_arms = [row["name"] for row in config["comparison"]["arms"]]
    if record.get("arm_order") != expected_arms:
        problems.append("the top-level arm order drifted")

    first_selection = None
    control_rows: dict[str, dict] = {}
    removal_fractions: list[float] = []
    derived_passing: list[int] = []
    for rung in rungs:
        fixed = rung["fixed_qubits"]
        plan = rung.get("contextual_plan", {})
        selection = plan.get("selection", {})
        if selection.get("requested_count") != fixed:
            problems.append(f"rung {fixed}: selected stabilizer count drifted")
        if plan.get("active_qubits") != source_qubits - fixed:
            problems.append(f"rung {fixed}: contextual plan active-qubit count drifted")
        stabilizers = selection.get("stabilizers", [])
        if len(stabilizers) != fixed:
            problems.append(f"rung {fixed}: stabilizer ledger length drifted")
        if selection.get("visited_terms") != fixed + sum(selection.get("skipped", {}).values()):
            problems.append(f"rung {fixed}: selection visit arithmetic does not close")
        current_selection = [row.get("packed_word_code") for row in stabilizers]
        if first_selection is not None and current_selection[:-1] != first_selection:
            problems.append(f"rung {fixed}: stabilizer selection is not prefix-nested")
        first_selection = current_selection

        arms = rung.get("arms", [])
        if [row.get("name") for row in arms] != expected_arms:
            problems.append(f"rung {fixed}: four-arm order drifted")
            continue
        for arm in arms:
            problems.extend(
                f"rung {fixed}/{problem}"
                for problem in _arm_problems(
                    arm,
                    gates=gates,
                    source_qubits=source_qubits,
                    fixed_qubits=fixed,
                )
            )
        mapped = _arm_map(rung)
        for control in ("full_qse", "acase"):
            if control not in control_rows:
                control_rows[control] = mapped[control]
            elif mapped[control] != control_rows[control]:
                problems.append(f"rung {fixed}: unrestricted {control} changed across rungs")
        cs_fractions = {
            mapped[name]["hamiltonian_removed_hs_fraction"]
            for name in ("cs_qse", "cs_acase")
        }
        if len(cs_fractions) != 1:
            problems.append(f"rung {fixed}: contextual arms report different Hamiltonian loss")
        else:
            removal_fractions.append(next(iter(cs_fractions)))

        failed_bias = [
            row["name"] for row in arms if not row["clears_bias_margin"]
        ]
        failed_words = [
            row["name"] for row in arms if not row["clears_word_universe_ceiling"]
        ]
        if rung.get("failed_bias_margin") != failed_bias:
            problems.append(f"rung {fixed}: failed-bias arm list is not derived")
        if rung.get("failed_word_universe_ceiling") != failed_words:
            problems.append(f"rung {fixed}: failed-word arm list is not derived")
        admitted = all(row["admissible"] for row in arms)
        if rung.get("all_four_arms_admissible") != admitted:
            problems.append(f"rung {fixed}: four-arm admission flag is not derived")
        if admitted:
            derived_passing.append(fixed)

    if any(
        right + 1e-12 < left
        for left, right in zip(removal_fractions, removal_fractions[1:])
    ):
        problems.append("Hamiltonian removed fraction falls as stabilizers are added")

    gate = record.get("structural_gate", {})
    selected = max(derived_passing) if derived_passing else None
    if gate.get("passing_contextual_rungs") != derived_passing:
        problems.append("passing-rung summary is not derived from the arm gates")
    if gate.get("selected_contextual_rung") != selected:
        problems.append("selected rung is not the largest four-arm passing rung")
    if gate.get("all_four_arms_admitted_at_selected_rung") != (selected is not None):
        problems.append("selected-rung admission flag contradicts the rung result")
    if gate.get("sampled_execution_authorized") is not False:
        problems.append("the structural result may not directly authorize sampling")
    expected_next = (
        "eligible_for_separate_sampled_preregistration"
        if selected is not None
        else "R4a_stops_without_sampling"
    )
    if gate.get("next_step") != expected_next:
        problems.append("the structural next step does not follow from its gate")
    if not record.get("claim_boundary"):
        problems.append("the structural record carries no claim boundary")
    return problems


def main() -> int:
    config = load_config()
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record(config)
    problems = compare_json_records(
        expected,
        actual,
        atol=1e-12,
        rtol=1e-12,
        key_tolerances=NUMERIC_TOLERANCES,
    )
    problems.extend(contract_problems(expected))
    problems.extend(f"rebuilt record: {row}" for row in contract_problems(actual))
    for problem in problems:
        print(f"  {problem}")
    print("R4a contextual structural screen:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
