"""Recompute and gate the R3S priceability-screen record.

Two independent jobs, as the project's other checkers do. The record is rebuilt
from scratch and compared field by field, and the contracts it must satisfy are
re-derived from the record's own contents rather than trusted:

* the two monotonicities the early exit rests on -- bias non-increasing and the
  word universe non-decreasing along the greedy prefix -- hold on every walk, so
  a walk that stopped short really had nothing left to find;
* ``margin_stop`` is the *smallest* clearing prefix: the last walked row clears
  the margin and no earlier one does;
* the per-arm biases at ``margin_stop`` agree, which is what makes the claim
  that the prefix was chosen mapping-blind checkable rather than asserted;
* the binding word universe is the maximum over arms, not a chosen arm;
* every verdict follows mechanically from the two declared gates, so no verdict
  can be written into the record by hand;
* the omitted ``h4`` candidate really is a prefix of ``h4_converged``, which is
  the whole reason it was omitted;
* BeH2's intrinsic word universe still reproduces the value the ceiling is
  calibrated on, binding the gate to the record that justifies it;
* no cost-record field appears anywhere: this tier prices nothing.

    python benchmarks/check_priceability_screen.py
"""

from __future__ import annotations

import json
import math
import sys

from clifford_qc.reproducibility import compare_json_records

try:  # package import in tests versus direct script execution
    from benchmarks.run_mapping_axis import load_config as load_mapping_config
    from benchmarks.run_priceability_screen import (
        REFERENCE,
        SCHEMA,
        WORD_UNIVERSE_CONVENTION,
        build_record,
        load_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_mapping_axis import load_config as load_mapping_config
    from run_priceability_screen import (
        REFERENCE,
        SCHEMA,
        WORD_UNIVERSE_CONVENTION,
        build_record,
        load_config,
    )


# ``bias_millihartree`` is (subspace energy - exact energy) in millihartree: a
# residue of order 1e-3 mHa left by two energies of order 1e3 mHa, so several
# significant digits are gone to cancellation before the comparison starts.
# Same tolerance, for the same reason, that check_mapping_axis.py documents.
ENERGY_DIFFERENCE_TOLERANCES = {"bias_millihartree": (1e-10, 1e-8)}

# What the arms may disagree by at one prefix. They compute one physical bias
# through five encodings, so the spread is accumulated rounding, not physics.
# Seven orders below the 0.533 mHa gate the selection turns on, which is what
# makes this a real check rather than a formality.
ARM_BIAS_AGREEMENT_MILLIHARTREE = 1e-8

# Fields that would make this a cost record. The tier is structural; if one of
# these ever appears the record has quietly changed what it claims.
COST_RECORD_KEYS = {
    "C_time",
    "cost_bracket",
    "device_costs",
    "k_star",
    "qr3_accuracy_matched",
    "shot_to_target",
}


def _cost_key_paths(node, path: str = "$") -> list[str]:
    if isinstance(node, dict):
        found = []
        for key, value in node.items():
            if key in COST_RECORD_KEYS:
                found.append(f"{path}.{key}")
            found += _cost_key_paths(value, f"{path}.{key}")
        return found
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _cost_key_paths(v, f"{path}[{i}]")]
    return []


def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_tier") != "structural":
        problems.append(
            f"evidence tier is {record.get('evidence_tier')!r}; this record "
            "samples nothing and must not claim a stronger tier"
        )

    gates = record.get("acceptance_gates", {})
    target = float(gates.get("accuracy_target_millihartree", 0.0))
    margin = float(gates.get("margin_factor", 0.0))
    ceiling = int(gates.get("word_universe_ceiling", 0))
    derived = record.get("derived_gates", {})
    admissible_bias = float(derived.get("admissible_bias_millihartree", -1.0))
    if not margin or abs(admissible_bias - target / margin) > 1e-12:
        problems.append(
            f"admissible bias {admissible_bias} is not the declared "
            f"{target}/{margin}"
        )
    allowance = math.sqrt(max(target**2 - admissible_bias**2, 0.0))
    if abs(float(derived.get("statistical_allowance_millihartree", -1.0)) - allowance) > 1e-12:
        problems.append("the statistical allowance is not the quadrature remainder")

    for candidate in record.get("candidates", []):
        key = candidate["candidate"]
        walk = candidate["prefix_walk"]
        if not walk:
            problems.append(f"{key}: empty prefix walk")
            continue

        sizes = [row["basis_size"] for row in walk]
        if sizes != list(range(sizes[0], sizes[0] + len(sizes))):
            problems.append(f"{key}: prefix walk is not contiguous: {sizes}")

        biases = [row["bias_millihartree"] for row in walk]
        if any(b - a > 1e-9 for a, b in zip(biases, biases[1:])):
            problems.append(
                f"{key}: bias rose along the greedy prefix ({biases}); a Ritz "
                "value cannot rise as the span grows, and the walk's early exit "
                "depends on it not doing so"
            )
        words = [row["source_word_universe"] for row in walk]
        if any(b < a for a, b in zip(words, words[1:])):
            problems.append(
                f"{key}: word universe fell along the greedy prefix ({words}); a "
                "longer prefix adds matrix-element pairs and removes none"
            )

        for row in walk:
            if row["clears_margin"] != (row["bias_millihartree"] <= admissible_bias):
                problems.append(
                    f"{key}: clears_margin at M={row['basis_size']} disagrees "
                    "with the declared bias gate"
                )
            if row["within_ceiling"] != (row["source_word_universe"] <= ceiling):
                problems.append(
                    f"{key}: within_ceiling at M={row['basis_size']} disagrees "
                    "with the declared ceiling"
                )

        reason = candidate["walk_terminated_because"]
        stop = candidate["margin_stop"]
        if stop is None:
            if any(row["clears_margin"] for row in walk):
                problems.append(
                    f"{key}: a walked prefix clears the margin but no margin_stop "
                    "was recorded"
                )
            if reason == "word_universe_exceeded_before_bias_cleared" and (
                walk[-1]["within_ceiling"] or walk[-1]["clears_margin"]
            ):
                problems.append(
                    f"{key}: walk claims the ceiling stopped it, but its last row "
                    "is within the ceiling or already clearing"
                )
            if reason == "bias_margin_unreachable_within_the_frozen_ordering" and (
                walk[-1]["basis_size"] != candidate["frozen_ordering_size"]
            ):
                problems.append(
                    f"{key}: walk claims the ordering was exhausted at "
                    f"M={walk[-1]['basis_size']} of {candidate['frozen_ordering_size']}"
                )
        else:
            if reason != "margin_cleared":
                problems.append(f"{key}: margin_stop recorded but walk ended {reason!r}")
            if not walk[-1]["clears_margin"] or stop["basis_size"] != walk[-1]["basis_size"]:
                problems.append(f"{key}: margin_stop is not the walk's final row")
            if any(row["clears_margin"] for row in walk[:-1]):
                problems.append(
                    f"{key}: an earlier prefix clears the margin, so "
                    f"M={stop['basis_size']} is not the smallest one"
                )
            if len(stop["labels"]) != stop["basis_size"]:
                problems.append(f"{key}: margin_stop labels do not match its size")

            arm_biases = [arm["bias_millihartree"] for arm in stop["arms"]]
            spread = max(arm_biases) - min(arm_biases)
            if spread > ARM_BIAS_AGREEMENT_MILLIHARTREE:
                problems.append(
                    f"{key}: per-arm bias spread {spread:.3e} mHa at margin_stop "
                    "exceeds rounding, so the prefix choice was not encoding-blind"
                )
            binding = max(arm["word_universe"] for arm in stop["arms"])
            if stop["binding_word_universe"] != binding:
                problems.append(
                    f"{key}: binding word universe {stop['binding_word_universe']} "
                    f"is not the maximum over arms ({binding})"
                )
            named = sorted(
                arm["mapping"] for arm in stop["arms"] if arm["word_universe"] == binding
            )
            if sorted(stop["binding_arms"]) != named:
                problems.append(f"{key}: binding_arms does not name the widest arms")
            if stop["within_ceiling"] != (binding <= ceiling):
                problems.append(f"{key}: margin_stop within_ceiling disagrees with the gate")
            # The walk's early exit gates on the source-side count while the
            # verdict gates on the binding one. They coincide because a pure
            # encoding preserves the word universe and a +2q arm can only
            # shrink it -- but if that ever stopped holding, the walk would be
            # stopping on a different quantity than the one being gated, and a
            # candidate could be rejected early on a count no arm carries.
            if binding != walk[-1]["source_word_universe"]:
                problems.append(
                    f"{key}: binding word universe {binding} differs from the "
                    f"source-side {walk[-1]['source_word_universe']} the walk's "
                    "early exit gates on; the two must be the same quantity"
                )

        intrinsic = candidate["intrinsic_stop"]
        if intrinsic["basis_size"] != candidate["frozen_ordering_size"]:
            problems.append(f"{key}: intrinsic stop is not the full frozen ordering")
        intrinsic_binding = max(arm["word_universe"] for arm in intrinsic["arms"])
        if intrinsic["binding_word_universe"] != intrinsic_binding:
            problems.append(f"{key}: intrinsic binding word universe is not the maximum")

        verdict = candidate["verdict"]
        expected = stop is not None and stop["within_ceiling"]
        if verdict["admissible_for_a_probe"] != expected:
            problems.append(
                f"{key}: verdict {verdict['admissible_for_a_probe']} does not follow "
                "from the two declared gates"
            )
        if verdict["rule_change_is_what_admits_it"] != bool(
            expected and intrinsic_binding > ceiling
        ):
            problems.append(
                f"{key}: rule_change_is_what_admits_it does not follow from the "
                "intrinsic stop's own word universe"
            )

    summary = record.get("summary", {})
    admitted = sorted(
        c["candidate"]
        for c in record.get("candidates", [])
        if c["verdict"]["admissible_for_a_probe"]
    )
    if sorted(summary.get("admissible_for_a_probe", [])) != admitted:
        problems.append("summary admissions disagree with the per-candidate verdicts")
    rejected = sorted(
        c["candidate"]
        for c in record.get("candidates", [])
        if not c["verdict"]["admissible_for_a_probe"]
    )
    if sorted(summary.get("rejected", {})) != rejected:
        problems.append("summary rejections disagree with the per-candidate verdicts")

    problems += [
        f"cost-record field on a structural tier: {path}"
        for path in _cost_key_paths(record)
    ]
    return problems


def omission_problems(record: dict) -> list[str]:
    """The declared ``h4`` omission has to be true, not merely stated.

    ``h4``'s frozen labels are the first nine of ``h4_converged``'s, so over a
    shared prefix the two banks are the same object and the walked rows are
    literally ``h4``'s own. The omission is sound when that shared walk rejects
    ``h4`` inside its own ordering -- which the ceiling exit does, because ``W``
    is non-decreasing: an exit at ``M`` puts every later prefix above the
    ceiling too, whatever its bias does. What would break the omission is a
    walk that cleared the margin at or below ``h4``'s size, since ``h4`` would
    then be a candidate in its own right.
    """
    problems: list[str] = []
    omitted = record.get("omitted_candidates", {})
    if "h4" not in omitted:
        return problems
    systems = {spec["key"]: spec for spec in load_mapping_config()["systems"]}
    short = systems["h4"]["selected_labels"]
    long = systems["h4_converged"]["selected_labels"]
    if list(long[: len(short)]) != list(short):
        problems.append(
            "h4 is omitted as a prefix of h4_converged, but its frozen labels "
            "are not that prefix"
        )
        return problems

    covering = next(
        (c for c in record.get("candidates", []) if c["candidate"] == "h4_converged"),
        None,
    )
    if covering is None:
        problems.append("h4 is omitted as covered by h4_converged, which is not screened")
        return problems

    walk = covering["prefix_walk"]
    exit_at = walk[-1]["basis_size"]
    if covering["walk_terminated_because"] == "margin_cleared":
        if exit_at <= len(short):
            problems.append(
                f"the shared walk clears the margin at M={exit_at}, inside h4's own "
                f"M={len(short)} ordering, so h4 is a candidate rather than an omission"
            )
        return problems
    if covering["walk_terminated_because"] != "word_universe_exceeded_before_bias_cleared":
        problems.append(
            f"the shared walk ended {covering['walk_terminated_because']!r}, which "
            "settles nothing about h4"
        )
        return problems
    if exit_at > len(short):
        problems.append(
            f"the shared walk left the ceiling at M={exit_at}, past h4's own "
            f"M={len(short)}, so h4's rejection does not follow from it"
        )
    if any(row["clears_margin"] for row in walk):
        problems.append(
            "the shared walk clears the margin before its ceiling exit, so h4's "
            "rejection does not follow from the exit"
        )
    return problems


def calibration_problems(record: dict) -> list[str]:
    """BeH2 is the single success the word-universe ceiling is calibrated on.

    Calibrated against ``protocol_axis.json`` rather than ``mapping_axis.json``,
    because the two count differently: ``mapping_axis`` includes the identity
    word and the other two records do not. The ceiling came from the second
    convention, so the check has to use it -- comparing across the two would
    fail by exactly one on every arm and say nothing about the gate.
    """
    problems: list[str] = []
    if record.get("word_universe_convention") != WORD_UNIVERSE_CONVENTION:
        problems.append(
            f"record counts words as {record.get('word_universe_convention')!r}; "
            f"the ceiling is calibrated on {WORD_UNIVERSE_CONVENTION!r}"
        )
    beh2 = next(
        (c for c in record.get("candidates", []) if c["candidate"] == "beh2"), None
    )
    if beh2 is None:
        return problems
    frozen_record = json.loads(
        (REFERENCE.parent / "protocol_axis.json").read_text(encoding="utf-8")
    )
    frozen = {
        arm["mapping"]: arm["word_universe"]
        for system in frozen_record["systems"]
        if system["system"] == "beh2"
        for arm in system["arms"]
    }
    if not frozen:
        problems.append("no frozen BeH2 arms found to calibrate the ceiling against")
    for arm in beh2["intrinsic_stop"]["arms"]:
        expected = frozen.get(arm["mapping"])
        if expected is not None and arm["word_universe"] != expected:
            problems.append(
                f"beh2/{arm['mapping']}: intrinsic word universe "
                f"{arm['word_universe']} does not reproduce the frozen R3 "
                f"{expected} the ceiling is calibrated on"
            )
    return problems


def main() -> int:
    load_config()  # the config's own preregistration gates
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(
        expected, actual, atol=1e-10, rtol=1e-10,
        key_tolerances=ENERGY_DIFFERENCE_TOLERANCES,
    )
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {p}" for p in contract_problems(actual)]
    problems += omission_problems(expected)
    problems += calibration_problems(expected)
    for problem in problems:
        print(f"  {problem}")
    print("priceability screen:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
