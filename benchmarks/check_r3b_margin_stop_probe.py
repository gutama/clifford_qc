"""Regenerate and validate the R3b LiH margin-stop scope-decision record.

Two jobs, as the project's other checkers have: rebuild the record from scratch
and compare it field by field, and re-derive its contracts from its own contents
rather than trusting them. The contracts specific to this phase:

* it is preregistered, and says which config fixed it before any sampling;
* the bank is the one the R3S margin rule selects, re-derived here -- a prefix
  chosen for cheapness fails in the producer before a shot is spent;
* the protocol is QR3b's, endpoint for endpoint, so the two records differ in
  the bank and nothing else and the comparison between them is like for like;
* the decision re-derives from the probe's own cells, and authorizes nothing;
* ``screen_prediction`` follows from the resolution gate rather than being
  written in, since this probe is the ceiling's first test on an admission;
* QR3b's own record is untouched by any of it.

    python benchmarks/check_r3b_margin_stop_probe.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys

from clifford_qc.reproducibility import (
    CROSS_MACHINE_ATOL,
    CROSS_MACHINE_RTOL,
    compare_json_records,
    guarded_contract_problems,
    sampling_stream_mismatch,
)

try:
    from benchmarks.check_r3b_preregistration import load_config
    from benchmarks.run_qr3b_instance_preflight import (
        COST_DERIVATIVE_KEYS,
        resolution_decision,
    )
    from benchmarks.run_r3b_margin_stop_probe import (
        QR3B_RECORD,
        REFERENCE,
        SCHEMA,
        build_record,
        screen_prediction,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_r3b_preregistration import load_config
    from run_qr3b_instance_preflight import COST_DERIVATIVE_KEYS, resolution_decision
    from run_r3b_margin_stop_probe import (
        QR3B_RECORD,
        REFERENCE,
        SCHEMA,
        build_record,
        screen_prediction,
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


@guarded_contract_problems
def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    config = load_config()

    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_role") != "scope_decision_only":
        problems.append("probe is not labelled scope-decision-only evidence")

    prereg = record.get("preregistration", {})
    if prereg.get("landed_before_any_sampling") is not True:
        problems.append("the record does not claim a preregistration")
    if prereg.get("config") != "benchmarks/configs/r3b_margin_stop_probe.json":
        problems.append("the record names the wrong preregistration config")

    # The bank, and the rule that picked it.
    selection = record.get("selection", {})
    gates = record.get("acceptance_gates", {})
    if selection.get("stopping_reason") != "smallest prefix clearing the accuracy margin":
        problems.append("selection did not stop on the declared margin rule")
    if list(selection.get("labels", [])) != list(config["candidate"]["selected_labels"]):
        problems.append("selected labels drifted from the preregistration")
    if selection.get("basis_size") != config["candidate"]["expected_basis_size"]:
        problems.append("selected basis size drifted from the preregistration")
    if selection.get("bias_millihartree", float("inf")) > selection.get(
        "admissible_bias_millihartree", 0.0
    ):
        problems.append("the selected bank does not clear the margin it was chosen by")
    if selection.get("bias_millihartree", float("inf")) >= gates.get(
        "accuracy_target_millihartree", 0.0
    ):
        problems.append("the selected bank does not clear the bias gate")
    if selection.get("source_word_universe") != config["candidate"][
        "expected_binding_word_universe"
    ]:
        problems.append("the bank's word universe drifted from the preregistration")
    # The point of the phase: a shorter bank than the one QR3b probed.
    if selection.get("basis_size", 0) >= selection.get("intrinsic_stop_basis_size", 0):
        problems.append(
            "the margin bank is not shorter than the intrinsic-stop bank QR3b "
            "probed, so this record is not testing what it claims to"
        )

    # QR3b's protocol, unchanged, so the two records are comparable.
    protocol = record.get("protocol", {})
    frozen = json.loads(
        (REFERENCE.parent.parent / "configs" / "qr3b_instance_preflight.json").read_text(
            encoding="utf-8"
        )
    )["protocol"]
    for field in ("block_sizes", "estimators", "exploratory_replicas",
                  "confirmatory_replicas",
                  "search_endpoints_effective_shots_per_setting"):
        if protocol.get(field) != frozen[field]:
            problems.append(f"protocol.{field} differs from QR3b's")
    if set(protocol.get("seed_roots", {}).values()) & set(frozen["seed_roots"].values()):
        problems.append("a seed root collides with QR3b's, so the streams could alias")

    probe = record.get("scoping_probe", {})
    if probe.get("is_a_cost_record") is not False:
        problems.append("scoping probe does not disclaim cost-record status")
    if probe.get("executed") is not True:
        problems.append("committed record did not execute its declared probe")

    decision = record.get("decision", {})
    if decision.get("full_run_authorized") is not False:
        problems.append("probe silently authorizes the full 30+100 run")
    if decision.get("qr3b_verdict") != "not_evaluated_by_this_probe":
        problems.append("probe reports a QR3b verdict")
    if "screen_would_have_rejected_before_probe" in decision:
        problems.append(
            "the record carries QR3b's hindsight field; here the screen ran first"
        )
    sampling = probe.get("sampling_evidence")
    if isinstance(sampling, dict):
        expected = dict(resolution_decision(sampling, config))
        expected.pop("screen_would_have_rejected_before_probe", None)
        expected["qr3b_verdict"] = "not_evaluated_by_this_probe"
        if expected != decision:
            problems.append("scope decision does not rederive from the probe cells")
    else:
        problems.append("probe carries no sampling evidence")

    # The screen's own first test, derived rather than asserted.
    prediction = record.get("screen_prediction")
    if not isinstance(prediction, dict):
        problems.append("the record makes no statement about the screen it tests")
    elif isinstance(sampling, dict):
        if prediction != screen_prediction(decision, selection):
            problems.append("screen_prediction does not follow from the resolution gate")
        if prediction.get("resolved_strictly_inside_the_grid") is not decision.get(
            "resolution_gate_passes"
        ):
            problems.append("screen_prediction disagrees with the resolution gate")

    forbidden = sorted(set(_walk_keys(record)) & COST_DERIVATIVE_KEYS)
    if forbidden:
        problems.append(f"probe leaks cost/verdict derivatives: {forbidden}")
    if any(key.startswith("C_time") for key in _walk_keys(record)):
        problems.append("probe contains a C_time value")

    problems.extend(
        compare_json_records(
            config["acceptance_gates"], gates,
            path="$.acceptance_gates", atol=0.0, rtol=0.0,
        )
    )
    return problems


def qr3b_untouched_problems() -> list[str]:
    """The record this phase extends must not have moved."""
    record = json.loads(QR3B_RECORD.read_text(encoding="utf-8"))
    problems: list[str] = []
    if record.get("decision", {}).get("status") != "rejected_unresolved_at_frozen_grid":
        problems.append(
            "QR3b's recorded rejection changed; this phase extends that record and "
            "may not reopen it"
        )
    if record.get("decision", {}).get("full_run_authorized") is not False:
        problems.append("QR3b's record now authorizes a full run")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    stream = sampling_stream_mismatch(expected)
    if stream:
        print("R3b margin-stop probe: FAIL (build environment differs)")
        for problem in stream:
            print(f"  {problem}")
        print(
            "  skipped the rebuild: under a different NumPy the sampled "
            "projected eigensolves do not verify this record"
        )
        for problem in contract_problems(expected):
            print(f"  committed record: {problem}")
        return 1
    actual = build_record(workers=args.workers)
    problems = compare_json_records(
        expected, actual,
        atol=CROSS_MACHINE_ATOL, rtol=CROSS_MACHINE_RTOL,
        key_tolerances={
            "absolute_tolerance": (0.0, 0.0),
            "leakage_tolerance": (0.0, 0.0),
            "relative_tolerance": (0.0, 0.0),
            "zero_tolerance": (0.0, 0.0),
        },
    )
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {item}" for item in contract_problems(actual)]
    problems += qr3b_untouched_problems()
    for problem in problems[:30]:
        print(f"  {problem}")
    if len(problems) > 30:
        print(f"  ... and {len(problems) - 30} more")
    print("R3b margin-stop probe:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
