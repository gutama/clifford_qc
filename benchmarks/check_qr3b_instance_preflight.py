"""Regenerate and validate the QR3b LiH scope-decision record."""

from __future__ import annotations

import argparse
import json
import sys

from clifford_qc.reproducibility import compare_json_records

try:
    from benchmarks.run_qr3b_instance_preflight import (
        REFERENCE,
        SCHEMA,
        build_record,
        load_config,
        resolution_decision,
    )
except ImportError:  # pragma: no cover
    from run_qr3b_instance_preflight import (
        REFERENCE,
        SCHEMA,
        build_record,
        load_config,
        resolution_decision,
    )


FORBIDDEN_COST_KEYS = {
    "cost_bracket",
    "device_costs",
    "k_star",
    "qr3_accuracy_matched",
}


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_role") != "scope_decision_only":
        problems.append("preflight is not labelled scope-decision-only evidence")
    parent = record.get("parent_lineage", {})
    if parent.get("merged_pull_request") != 67:
        problems.append("parent lineage does not name PR #67")
    if "right-censored" not in parent.get("preserved_result", ""):
        problems.append("parent lineage does not preserve H4 right-censoring")

    protocol = record.get("protocol", {})
    probe = record.get("scoping_probe", {})
    if probe.get("is_a_cost_record") is not False:
        problems.append("scoping probe does not disclaim cost-record status")
    if probe.get("executed") is not True:
        problems.append("committed preflight did not execute its declared probe")
    if probe.get("exploratory_replicas") != 2:
        problems.append("exploratory replica count drifted from 2")
    if probe.get("confirmatory_replicas") != 2:
        problems.append("confirmatory replica count drifted from 2")
    if probe.get("search_endpoints_effective_shots_per_setting") != protocol.get(
        "search_endpoints_effective_shots_per_setting"
    ):
        problems.append("probe endpoints differ from the preregistration")

    decision = record.get("decision", {})
    if decision.get("full_run_authorized") is not False:
        problems.append("preflight silently authorizes the full 30+100 run")
    if decision.get("qr3b_verdict") != "not_evaluated_by_preflight":
        problems.append("preflight reports a QR3b verdict")
    sampling = probe.get("sampling_evidence")
    if isinstance(sampling, dict):
        expected = resolution_decision(sampling, load_config())
        if expected != decision:
            problems.append("scope decision does not rederive from probe cells")
    else:
        problems.append("probe carries no sampling evidence")

    forbidden = sorted(set(_walk_keys(probe)) & FORBIDDEN_COST_KEYS)
    if forbidden:
        problems.append(f"scope probe leaks cost/verdict derivatives: {forbidden}")
    if any(key.startswith("C_time") for key in _walk_keys(probe)):
        problems.append("scope probe contains a C_time value")

    selection = record.get("selection", {})
    gates = record.get("acceptance_gates", {})
    if selection.get("budget_is_nonbinding") is not True:
        problems.append("selection implementation budget is binding")
    if selection.get("stopping_reason") != "predicted lowering below threshold":
        problems.append("selection did not stop intrinsically")
    if selection.get("bias_millihartree", float("inf")) >= gates.get(
        "accuracy_target_millihartree", 0.0
    ):
        problems.append("selected LiH bank does not clear the bias gate")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record(workers=args.workers)
    problems = compare_json_records(
        expected,
        actual,
        atol=1e-10,
        rtol=1e-10,
        key_tolerances={
            "error_millihartree": (1e-10, 1e-8),
            "exact_subspace_bias_millihartree": (1e-10, 1e-8),
            "bias_millihartree": (1e-10, 1e-8),
        },
    )
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {item}" for item in contract_problems(actual)]
    for problem in problems:
        print(f"  {problem}")
    print("QR3b instance preflight:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
