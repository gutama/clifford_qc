#!/usr/bin/env python3
"""Regenerate and validate the sampled R3d within-grid QR3 refinement.

The checker separately re-derives every endpoint decision, interval update,
cost interval, support spread, and allowed readout from the committed record.
By default it then redraws all seven frozen cells and compares every scientific
field with the committed evidence.

    python benchmarks/check_r3d_qr3_refinement.py --workers 4
    python benchmarks/check_r3d_qr3_refinement.py --skip-rebuild
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import subprocess
import sys

from clifford_qc.reproducibility import (
    CROSS_MACHINE_ATOL,
    CROSS_MACHINE_RTOL,
    compare_json_records,
    sampling_stream_mismatch,
)

try:  # package import in tests versus direct benchmark execution
    from benchmarks.check_r3d_preregistration import load_config, static_problems
    from benchmarks.run_r3d_qr3_refinement import (
        CONFIG,
        PREREGISTRATION_MERGE,
        PREREGISTRATION_SHA256,
        REFERENCE,
        RESULT_CLAIM_BOUNDARY,
        SCHEMA,
        _file_sha256,
        _parent_rung,
        _prepare_cell,
        _scalar_cost,
        build_record,
        declared_environment,
        derive_readout,
        endpoint_decision,
        load_device_cards,
        refined_shot_interval,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_r3d_preregistration import load_config, static_problems
    from run_r3d_qr3_refinement import (
        CONFIG,
        PREREGISTRATION_MERGE,
        PREREGISTRATION_SHA256,
        REFERENCE,
        RESULT_CLAIM_BOUNDARY,
        SCHEMA,
        _file_sha256,
        _parent_rung,
        _prepare_cell,
        _scalar_cost,
        build_record,
        declared_environment,
        derive_readout,
        endpoint_decision,
        load_device_cards,
        refined_shot_interval,
    )


CELL_COORDINATES = (
    "support",
    "parent_record",
    "system",
    "mapping",
    "measured_qubits",
    "block_size",
    "estimator",
    "card",
)


def _same_float(actual, expected) -> bool:
    return (
        isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
        and math.isclose(
            float(actual),
            float(expected),
            rel_tol=CROSS_MACHINE_RTOL,
            abs_tol=CROSS_MACHINE_ATOL,
        )
    )


@functools.lru_cache(maxsize=16)
def _rebuilt_costs(index: int, lower: int, upper: int) -> tuple[float | None, float | None]:
    """Reprice an interval through the bound instrument, including rounding.

    Fidelity inflation rounds raw shots per setting, so device cost is nearly
    but not exactly linear in effective shots.  Rebuilding the resources is the
    only exact audit; a ratio extrapolated from a parent endpoint is not.
    """
    target = load_config()["refinement"]["target_cells"][index]
    problem, _, resources, _ = _prepare_cell(target)
    cards = load_device_cards()
    card = target["card"]
    return (
        _scalar_cost(cards, resources, problem["n_qubits"], lower, card),
        _scalar_cost(cards, resources, problem["n_qubits"], upper, card),
    )


def _cell_problems(record: dict, config: dict) -> list[str]:
    problems = []
    run = record.get("refinement_run", {})
    cells = run.get("cells")
    targets = config["refinement"]["target_cells"]
    if not isinstance(cells, list):
        return ["the refinement run has no cell list"]
    if len(cells) != len(targets):
        problems.append(f"the record has {len(cells)} target cells, expected {len(targets)}")
    if run.get("target_cell_count") != len(targets):
        problems.append("the reported target-cell count drifted")
    endpoint_count = sum(
        len(row["new_endpoints_effective_shots_per_setting"]) for row in targets
    )
    if run.get("new_endpoint_cell_count") != endpoint_count:
        problems.append("the reported new-endpoint count drifted")
    if run.get("parent_evidence_reused_as_samples") is not False:
        problems.append("parent evidence is not explicitly read-only")

    protocol = config["protocol"]
    for index, (cell, target) in enumerate(zip(cells, targets)):
        label = f"cell {index}"
        if cell.get("target_cell_index") != index:
            problems.append(f"{label}: target-cell index drifted")
        for key in CELL_COORDINATES:
            if cell.get(key) != target[key]:
                problems.append(f"{label}: {key} drifted from the preregistration")
        endpoints = target["new_endpoints_effective_shots_per_setting"]
        if cell.get("new_endpoints_effective_shots_per_setting") != endpoints:
            problems.append(f"{label}: midpoint endpoint list drifted")
        if cell.get("inherited_endpoint_interval") != target["inherited_endpoint_interval"]:
            problems.append(f"{label}: inherited endpoint interval drifted")

        parent = _parent_rung(target)
        structural = cell.get("structural_instrument", {})
        if structural.get("settings") != parent["settings"]:
            problems.append(f"{label}: setting count differs from the parent cell")
        parent_estimator = parent["estimators"][target["estimator"]]
        parent_ranks = {
            int(rank)
            for row in parent_estimator["confirmation"]
            for rank, count in row.get("rank_histogram", {}).items()
            if count
        }
        if structural.get("retained_rank") not in parent_ranks:
            problems.append(f"{label}: retained rank is absent from parent confirmations")

        for phase, expected_replicas in (
            ("exploration", protocol["exploratory_replicas_per_new_endpoint"]),
            ("confirmation", protocol["confirmatory_replicas_per_new_endpoint"]),
        ):
            rows = cell.get(phase)
            if not isinstance(rows, list):
                problems.append(f"{label}: {phase} block is missing")
                continue
            if [row.get("effective_shots_per_setting") for row in rows] != endpoints:
                problems.append(f"{label}: {phase} did not report every frozen endpoint")
            for row in rows:
                if row.get("replicas") != expected_replicas:
                    problems.append(f"{label}: {phase} replica count drifted")

        confirmation = cell.get("confirmation", [])
        decisions = cell.get("endpoint_decisions")
        if not isinstance(decisions, list):
            problems.append(f"{label}: endpoint decisions are missing")
            decisions = []
        expected_decisions = [endpoint_decision(row, protocol) for row in confirmation]
        if decisions != expected_decisions:
            problems.append(f"{label}: endpoint decisions do not re-derive")
        expected_interval = refined_shot_interval(target, expected_decisions)
        if cell.get("refined_endpoint_interval") != expected_interval:
            problems.append(f"{label}: refined endpoint interval does not re-derive")
        if not cell.get("refined_endpoint_interval", {}).get(
            "parent_interval_only_narrowed"
        ):
            problems.append(f"{label}: the parent interval was widened")

        bracket = parent_estimator["cost_bracket"]
        parent_card = bracket["cards"][target["card"]]
        inherited_cost = cell.get("inherited_cost_interval", {})
        for side in ("lower", "upper"):
            field = f"C_time_{side}_us"
            if not _same_float(inherited_cost.get(field), parent_card[field]):
                problems.append(f"{label}: inherited {side} cost differs from its parent")

        refined_cost = cell.get("refined_cost_interval", {})
        if refined_cost.get("device_card") != target["card"]:
            problems.append(f"{label}: refined interval uses another device card")
        lower = expected_interval["lower_effective_shots_per_setting"]
        upper = expected_interval["upper_effective_shots_per_setting"]
        rebuilt = _rebuilt_costs(index, lower, upper)
        for side, expected_cost in zip(("lower", "upper"), rebuilt):
            if not _same_float(refined_cost.get(f"C_time_{side}_us"), expected_cost):
                problems.append(f"{label}: refined {side} cost does not re-derive")
    return problems


def contract_problems(record: dict) -> list[str]:
    config = load_config()
    problems = [f"preregistration: {item}" for item in static_problems(config)]
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("phase") != config["phase"]:
        problems.append("the record phase drifted")
    if record.get("evidence_role") != "targeted_accuracy_matched_cost_refinement":
        problems.append("the R3d evidence role drifted")
    if record.get("evidence_tier") != "exact":
        problems.append("the oracle-reference tier is not labelled exact")
    if record.get("search_uncertainty_evidence") != "heuristic":
        problems.append("the Monte Carlo uncertainty is not labelled heuristic")
    if record.get("config_file_sha256") != PREREGISTRATION_SHA256:
        problems.append("the record does not bind the landed config bytes")
    if _file_sha256(CONFIG) != PREREGISTRATION_SHA256:
        problems.append("the committed preregistration changed after landing")
    if record.get("producer_sha256") != _file_sha256(
        REFERENCE.parent.parent / "run_r3d_qr3_refinement.py"
    ):
        problems.append("the record's producer digest differs from the source")

    preregistration = record.get("preregistration", {})
    if preregistration.get("merge_commit") != PREREGISTRATION_MERGE:
        problems.append("the record does not name the merged preregistration commit")
    if preregistration.get("result_commit_is_separate") is not True:
        problems.append("the sampled result is not declared separate from preregistration")
    if preregistration.get(
        "supports_and_endpoints_selected_before_every_r3d_draw"
    ) is not True:
        problems.append("the prospective support/endpoint boundary is missing")

    for field in (
        "parent_lineage",
        "estimand",
        "protocol",
        "future_readout_contract",
        "reporting_contract",
    ):
        config_field = "future_readout" if field == "future_readout_contract" else field
        if record.get(field) != config[config_field]:
            problems.append(f"{field} drifted from the preregistration")
    if record.get("claim_boundary") != RESULT_CLAIM_BOUNDARY:
        problems.append("the result claim boundary drifted")

    run = record.get("refinement_run", {})
    if run.get("executed") is not True:
        problems.append("the committed R3d record does not report an execution")
    problems += _cell_problems(record, config)
    cells = run.get("cells", []) if isinstance(run.get("cells"), list) else []
    expected_readout = derive_readout(cells)
    if record.get("readout") != expected_readout:
        problems.append("the R3d support spreads or classification do not re-derive")
    readout = record.get("readout", {})
    if readout.get("classification") not in {
        "qr3_mapping_spread_larger_on_preregistered_support",
        "indeterminate_after_refinement",
    }:
        problems.append("the record reports an unlicensed R3d classification")
    if readout.get("negative_readout_allowed") is not False:
        problems.append("the record licenses a negative QR3 direction")
    if readout.get("global_extrema_reselected") is not False:
        problems.append("the record reselected an extremal support after the draw")

    environment = record.get("execution_environment", {})
    if environment.get("matches_preregistration") is not True or environment.get(
        "problems"
    ) != []:
        problems.append("the record does not attest the frozen execution stack")
    for key, value in config["protocol"]["execution_environment"].items():
        if environment.get(key) != value:
            problems.append(f"the recorded {key} version drifted")
    provenance = record.get("provenance", {})
    dependencies = provenance.get("dependencies", {})
    built_under = {
        "python": ".".join(str(provenance.get("python", "")).split(".")[:2]),
        "numpy": dependencies.get("numpy"),
        "scipy": dependencies.get("scipy"),
        "stim": dependencies.get("stim"),
    }
    if built_under != declared_environment(config):
        problems.append("the provenance stamp differs from the frozen R3d stack")
    return problems


def ancestry_problems() -> list[str]:
    """Check result ordering when the preregistration commit is available."""
    known = subprocess.run(
        ["git", "cat-file", "-e", f"{PREREGISTRATION_MERGE}^{{commit}}"],
        capture_output=True,
    )
    if known.returncode:
        return []
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_MERGE, "HEAD"],
        capture_output=True,
    )
    return [] if ancestor.returncode == 0 else [
        "the sampled implementation does not descend from the R3d preregistration merge"
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args(argv)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    problems = contract_problems(expected) + ancestry_problems()
    if not args.skip_rebuild:
        stream = sampling_stream_mismatch(expected)
        if stream:
            print("R3d QR3 refinement: FAIL (build environment differs)")
            for problem in stream:
                print(f"  {problem}")
            return 1
        actual = build_record(workers=args.workers)
        problems += compare_json_records(
            expected,
            actual,
            atol=CROSS_MACHINE_ATOL,
            rtol=CROSS_MACHINE_RTOL,
        )
        problems += [f"rebuilt record: {item}" for item in contract_problems(actual)]
    for problem in problems[:30]:
        print(f"  {problem}")
    if len(problems) > 30:
        print(f"  ... and {len(problems) - 30} more")
    print("R3d QR3 refinement:", "FAIL" if problems else "PASS")
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
