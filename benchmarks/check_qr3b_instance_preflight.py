"""Regenerate and validate the QR3b LiH scope-decision record."""

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
    from benchmarks.run_qr3b_instance_preflight import (
        REFERENCE,
        SCHEMA,
        COST_DERIVATIVE_KEYS,
        build_record,
        load_config,
        resolution_decision,
    )
except ImportError:  # pragma: no cover
    from run_qr3b_instance_preflight import (
        REFERENCE,
        SCHEMA,
        COST_DERIVATIVE_KEYS,
        build_record,
        load_config,
        resolution_decision,
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

    forbidden = sorted(set(_walk_keys(record)) & COST_DERIVATIVE_KEYS)
    if forbidden:
        problems.append(f"preflight leaks cost/verdict derivatives: {forbidden}")
    if any(key.startswith("C_time") for key in _walk_keys(record)):
        problems.append("preflight contains a C_time value")

    selection = record.get("selection", {})
    gates = record.get("acceptance_gates", {})
    problems.extend(
        compare_json_records(
            load_config()["acceptance_gates"],
            gates,
            path="$.acceptance_gates",
            atol=0.0,
            rtol=0.0,
        )
    )
    if selection.get("budget_is_nonbinding") is not True:
        problems.append("selection implementation budget is binding")
    if selection.get("stopping_reason") != "predicted lowering below threshold":
        problems.append("selection did not stop intrinsically")
    if selection.get("bias_millihartree", float("inf")) >= gates.get(
        "accuracy_target_millihartree", 0.0
    ):
        problems.append("selected LiH bank does not clear the bias gate")
    if not isinstance(gates.get("word_universe_ceiling"), int):
        problems.append("acceptance gates declare no word-universe ceiling")
    elif decision.get("word_universe_ceiling") != gates["word_universe_ceiling"]:
        problems.append("decision prices a different ceiling than the gates declare")
    elif decision.get("word_universe_gate_passes") is not (
        decision.get("maximum_word_universe", 0) <= gates["word_universe_ceiling"]
    ):
        problems.append("word-universe verdict disagrees with its own ceiling")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    stream = sampling_stream_mismatch(expected)
    if stream:
        print("QR3b instance preflight: FAIL (build environment differs)")
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
    # No per-key widening -- the same choice check_protocol_cost.py makes and
    # for the same reason: every millihartree field here is a difference against
    # an exact sector reference taken from the dense eigensolve rather than
    # ARPACK, precisely so it reproduces bit for bit, and a per-key 1e-8 would
    # absorb a nondeterministic reference instead of failing on it.
    #
    # The floor below is the cross-machine one, which is a separate quantity:
    # this record rebuilds bit-identically here and drifts only against a
    # differently-kernelled runner.
    problems = compare_json_records(
        expected,
        actual,
        atol=CROSS_MACHINE_ATOL,
        rtol=CROSS_MACHINE_RTOL,
        key_tolerances={
            "absolute_tolerance": (0.0, 0.0),
            "leakage_tolerance": (0.0, 0.0),
            "relative_tolerance": (0.0, 0.0),
            "zero_tolerance": (0.0, 0.0),
        },
    )
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {item}" for item in contract_problems(actual)]
    for problem in problems[:30]:
        print(f"  {problem}")
    if len(problems) > 30:
        print(f"  ... and {len(problems) - 30} more")
    print("QR3b instance preflight:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
