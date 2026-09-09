#!/usr/bin/env python3
"""Execute the frozen R3d within-grid QR3 refinement.

R3d adds exactly seven midpoint cells to five immutable R3c point-support
cells.  It does not redraw a parent endpoint, widen the grid, reselect a
support, or license a negative QR3 result.  Every midpoint receives the frozen
30 exploratory and 100 confirmatory replicas on fresh streams.  Exploration is
reported but never selects which confirmatory cells are run.

    python benchmarks/run_r3d_qr3_refinement.py --workers 4
    python benchmarks/run_r3d_qr3_refinement.py --skip-run
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np

from clifford_qc.measurement.block_commuting import block_commuting_partition
from clifford_qc.measurement.cache import GroupedWordCache
from clifford_qc.measurement.compiled import CompiledMeasurementSampler
from clifford_qc.measurement.planning import compile_block_measurement_plan
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.reproducibility import stamp_record

try:  # package import in tests versus direct benchmark execution
    from benchmarks.check_r3d_preregistration import load_config, static_problems
    from benchmarks.run_clifford_hierarchy import (
        ACCURACY_TARGET_MILLIHARTREE,
        _synthesize,
    )
    from benchmarks.run_exact_shot_search import (
        BOOTSTRAP_REPLICATES,
        DELTA,
        ESTIMATORS,
        MARGINAL_TARGET_FRACTION,
        RANK_SELECTION_GAMMA,
        _device_prices,
        _seed_sequence,
        _solve,
        _summary,
    )
    from benchmarks.run_mapping_axis import (
        _canonical_sha256,
        load_device_cards,
    )
    from benchmarks.run_protocol_cost import _system_specs, arm_problem
    from benchmarks.run_r3b_margin_stop_probe import (
        _downstream_spec,
        derive_selection as derive_margin_bank,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_r3d_preregistration import load_config, static_problems
    from run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE, _synthesize
    from run_exact_shot_search import (
        BOOTSTRAP_REPLICATES,
        DELTA,
        ESTIMATORS,
        MARGINAL_TARGET_FRACTION,
        RANK_SELECTION_GAMMA,
        _device_prices,
        _seed_sequence,
        _solve,
        _summary,
    )
    from run_mapping_axis import _canonical_sha256, load_device_cards
    from run_protocol_cost import _system_specs, arm_problem
    from run_r3b_margin_stop_probe import (
        _downstream_spec,
        derive_selection as derive_margin_bank,
    )


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "r3d_qr3_refinement.json"
R3C_CONFIG = HERE / "configs" / "r3c_lih_full_cost.json"
R3C_PARENT = HERE / "reference_results" / "r3c_lih_full_cost.json"
PROTOCOL_PARENT = HERE / "reference_results" / "protocol_cost.json"
REFERENCE = HERE / "reference_results" / "r3d_qr3_refinement.json"
SCHEMA = "clifford_qc.r3d_qr3_refinement.v1"
PREREGISTRATION_MERGE = "c9fe494f98a4994380a1fb50cf13bb7783ea4f9c"
PREREGISTRATION_SHA256 = "26bd9c2ca025112cba44163f331229c3ddb11cac18c6afd70db42921aad1791e"
RESULT_CLAIM_BOUNDARY = (
    "This record reports only the seven midpoint cells frozen by R3d under the "
    "declared 30+100 protocol and re-derives the two support spreads without "
    "altering either parent record. It may establish only "
    "qr3_mapping_spread_larger_on_preregistered_support or remain "
    "indeterminate_after_refinement; it cannot establish a negative or "
    "mapping-wide QR3 direction, widen the grid, or reselect an extremal cell."
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_environment(config: dict) -> dict[str, str]:
    declared = config["protocol"]["execution_environment"]
    return {
        "python": declared["python_minor"],
        "numpy": declared["numpy"],
        "scipy": declared["scipy"],
        "stim": declared["stim"],
    }


def installed_environment() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": ".".join(platform.python_version().split(".")[:2])
    }
    for name in ("numpy", "scipy", "stim"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_problems(config: dict) -> list[str]:
    installed = installed_environment()
    return [
        f"{name} {installed.get(name)!r} is not the frozen {expected}"
        for name, expected in sorted(declared_environment(config).items())
        if installed.get(name) != expected
    ]


def source_order_problems() -> list[str]:
    """Best-effort proof that the sampled source follows the declaration."""
    current = (
        os.environ.get("GITHUB_SHA")
        or os.environ.get("CI_COMMIT_SHA")
    )
    if current is None:
        resolved = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        current = resolved.stdout.strip() if resolved.returncode == 0 else None
    if current == PREREGISTRATION_MERGE:
        return ["R3d sampling source is the preregistration merge itself"]

    known = subprocess.run(
        ["git", "cat-file", "-e", f"{PREREGISTRATION_MERGE}^{{commit}}"],
        capture_output=True,
    )
    if known.returncode or current is None:
        return []
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_MERGE, current],
        capture_output=True,
    )
    return [] if ancestor.returncode == 0 else [
        "R3d sampling source does not descend from the preregistration merge"
    ]


def preregistration_problems(config: dict) -> list[str]:
    problems = [f"preregistration: {item}" for item in static_problems(config)]
    actual = _file_sha256(CONFIG)
    if actual != PREREGISTRATION_SHA256:
        problems.append(
            f"the landed R3d config digest is {actual}, not {PREREGISTRATION_SHA256}"
        )
    problems.extend(source_order_problems())
    return problems


def _parent_system(target: dict) -> dict:
    if target["system"] == "beh2":
        return _read(PROTOCOL_PARENT)["systems"]["beh2"]
    return _read(R3C_PARENT)["full_cost_run"]["sampling_evidence"]


def _parent_rung(target: dict) -> dict:
    system = _parent_system(target)
    arm = next(row for row in system["arms"] if row["mapping"] == target["mapping"])
    return next(
        row for row in arm["rungs"] if row["block_size"] == target["block_size"]
    )


def _lih_spec() -> dict:
    config = _read(R3C_CONFIG)
    selection = derive_margin_bank(
        {
            "candidate": config["candidate"],
            "acceptance_gates": {
                "accuracy_target_millihartree": config["acceptance_rule"][
                    "accuracy_target_millihartree"
                ]
            },
        }
    )
    return _downstream_spec(config["candidate"], selection)


def _target_spec(target: dict) -> dict:
    return _system_specs()["beh2"] if target["system"] == "beh2" else _lih_spec()


def _prepare_cell(target: dict):
    problem = arm_problem(target["system"], target["mapping"], _target_spec(target))
    if problem["n_qubits"] != target["measured_qubits"]:
        raise ValueError("target measured-qubit count drifted from the reconstructed bank")
    codes = problem["codes"]
    block_size = int(target["block_size"])
    groups = block_commuting_partition(problem["n_qubits"], codes, block_size)
    plan = compile_block_measurement_plan(
        problem["n_qubits"], codes, block_size, groups=groups
    )
    structural, resources, compatibility, _ = _synthesize(
        problem["n_qubits"],
        codes,
        groups,
        block_size,
        synthesis=plan.synthesis,
    )
    parent = _parent_rung(target)
    if structural["settings"] != parent["settings"]:
        raise ValueError(
            f"{target['system']}/{target['mapping']} k={block_size}: rebuilt "
            f"{structural['settings']} settings, parent froze {parent['settings']}"
        )
    if any(
        len(setting.readouts) != int(compatibility[index].sum())
        for index, setting in enumerate(plan.settings)
    ):
        raise AssertionError("compiled readouts disagree with hierarchy compatibility")
    return problem, plan.settings, resources, compatibility


def _sample_phase(
    *,
    problem: dict,
    settings,
    target: dict,
    target_cell_index: int,
    phase_index: int,
    replicas: int,
    seed_root: int,
    bootstrap_root: int,
) -> list[dict]:
    endpoints = tuple(target["new_endpoints_effective_shots_per_setting"])
    estimator = target["estimator"]
    by_code = {word.code: word for word in problem["bank"].words(problem["result"].indices)}
    groups = [
        [by_code[code] for code in setting.assigned_word_codes if code != 0]
        for setting in settings
    ]
    session = SharedMeasurement(
        problem["bank"],
        problem["result"].indices,
        groups=groups,
        pooling="assigned" if estimator == "single_assignment" else "shots",
    )
    raw = {endpoint: [] for endpoint in endpoints}
    sampler = CompiledMeasurementSampler(seed_root)
    sampler.prepare(problem["bank"].reference, settings)
    for replica_index in range(replicas):
        sampler.reseed(_seed_sequence(seed_root, target_cell_index, replica_index))
        cache = GroupedWordCache(
            problem["bank"].reference.n,
            pooling="assigned" if estimator == "single_assignment" else "shots",
        )
        previous = 0
        for endpoint in endpoints:
            batch = sampler.sample_from_state(
                problem["bank"].reference, settings, endpoint - previous
            )
            previous = endpoint
            cache.add_batch(batch)
            raw[endpoint].append(_solve(session, cache))

    estimator_index = ESTIMATORS.index(estimator)
    summaries = []
    for endpoint_index, endpoint in enumerate(endpoints):
        summary = _summary(
            raw[endpoint],
            problem["exact_ground_energy"],
            bootstrap_seed=_seed_sequence(
                bootstrap_root,
                target_cell_index,
                phase_index,
                endpoint_index,
                estimator_index,
            ),
        )
        summaries.append({"effective_shots_per_setting": endpoint, **summary})
    return summaries


def endpoint_decision(summary: dict, protocol: dict) -> dict:
    upper = summary.get("rmse_one_sided_95pct_upper_millihartree")
    valid = (
        isinstance(upper, (int, float))
        and math.isfinite(float(upper))
        and summary.get("replicas") == protocol["confirmatory_replicas_per_new_endpoint"]
    )
    marginal = bool(
        valid
        and abs(float(upper) - protocol["accuracy_target_millihartree"])
        / protocol["accuracy_target_millihartree"]
        <= protocol["environment_marginal_target_fraction"]
    )
    if not valid:
        classification = "invalid"
    elif marginal:
        classification = "environment_marginal"
    elif summary.get("passes_target") is True:
        classification = "confirmed_pass"
    else:
        classification = "confirmed_fail"
    return {
        "effective_shots_per_setting": summary.get("effective_shots_per_setting"),
        "valid": valid,
        "passes_target": summary.get("passes_target") if valid else None,
        "environment_marginal": marginal,
        "informative_for_interval": valid and not marginal,
        "classification": classification,
    }


def refined_shot_interval(target: dict, decisions: Sequence[dict]) -> dict:
    inherited = target["inherited_endpoint_interval"]
    lower = int(inherited["lower"])
    upper = int(inherited["upper"])
    expected = list(target["new_endpoints_effective_shots_per_setting"])
    actual = [row.get("effective_shots_per_setting") for row in decisions]
    problems = []
    if actual != expected:
        problems.append("the endpoint list is missing, reordered, or contains an extra cell")
    if any(not row.get("valid") for row in decisions):
        problems.append("at least one endpoint has a missing or non-finite required statistic")

    informative = [row for row in decisions if row.get("informative_for_interval")]
    seen_pass = False
    for row in informative:
        if row["classification"] == "confirmed_pass":
            seen_pass = True
            upper = min(upper, int(row["effective_shots_per_setting"]))
        elif row["classification"] == "confirmed_fail":
            if seen_pass:
                problems.append("non-marginal endpoint decisions are nonmonotone")
            lower = max(lower, int(row["effective_shots_per_setting"]))
        else:
            problems.append("an informative endpoint is neither a pass nor a fail")
    if lower >= upper:
        problems.append("the refined lower endpoint is not strictly below the upper")
    return {
        "valid": not problems,
        "problems": problems,
        "lower_effective_shots_per_setting": lower,
        "upper_effective_shots_per_setting": upper,
        "parent_interval_only_narrowed": (
            lower >= inherited["lower"] and upper <= inherited["upper"]
        ),
    }


def _scalar_cost(cards, resources, n_qubits: int, shots: int, card_name: str):
    priced = _device_prices(cards, resources, shots, n_qubits).get(card_name)
    return None if priced is None else priced["accuracy"]["C_time_epsilon_us"]


def _run_cell(arguments) -> dict:
    target_cell_index, target, protocol = arguments
    cards = load_device_cards()
    problem, settings, resources, compatibility = _prepare_cell(target)
    exploration = _sample_phase(
        problem=problem,
        settings=settings,
        target=target,
        target_cell_index=target_cell_index,
        phase_index=0,
        replicas=protocol["exploratory_replicas_per_new_endpoint"],
        seed_root=protocol["seed_roots"]["exploratory"],
        bootstrap_root=protocol["seed_roots"]["bootstrap"],
    )
    confirmation = _sample_phase(
        problem=problem,
        settings=settings,
        target=target,
        target_cell_index=target_cell_index,
        phase_index=1,
        replicas=protocol["confirmatory_replicas_per_new_endpoint"],
        seed_root=protocol["seed_roots"]["confirmatory"],
        bootstrap_root=protocol["seed_roots"]["bootstrap"],
    )
    decisions = [endpoint_decision(row, protocol) for row in confirmation]
    interval = refined_shot_interval(target, decisions)
    lower = interval["lower_effective_shots_per_setting"]
    upper = interval["upper_effective_shots_per_setting"]
    card = target["card"]
    refined_cost = {
        "device_card": card,
        "device_card_sha256": next(item.sha256 for item in cards if item.name == card),
        "C_time_lower_us": _scalar_cost(
            cards, resources, problem["n_qubits"], lower, card
        ),
        "C_time_upper_us": _scalar_cost(
            cards, resources, problem["n_qubits"], upper, card
        ),
    }
    parent = _parent_rung(target)["estimators"][target["estimator"]]["cost_bracket"]
    parent_cost = parent["cards"][card]
    return {
        "target_cell_index": target_cell_index,
        **{key: target[key] for key in (
            "support", "parent_record", "system", "mapping", "measured_qubits",
            "block_size", "estimator", "card",
        )},
        "new_endpoints_effective_shots_per_setting": list(
            target["new_endpoints_effective_shots_per_setting"]
        ),
        "structural_instrument": {
            "settings": len(settings),
            "compatible_readouts": int(compatibility.sum()),
            "mean_reader_count": float(compatibility.sum(axis=0).mean()),
            "word_universe": len(problem["codes"]),
            "retained_rank": int(problem["result"].effective_rank),
            "exact_ground_energy": problem["exact_ground_energy"],
        },
        "exploration": exploration,
        "confirmation": confirmation,
        "endpoint_decisions": decisions,
        "inherited_endpoint_interval": target["inherited_endpoint_interval"],
        "refined_endpoint_interval": interval,
        "inherited_cost_interval": {
            "device_card": card,
            "C_time_lower_us": parent_cost["C_time_lower_us"],
            "C_time_upper_us": parent_cost["C_time_upper_us"],
        },
        "refined_cost_interval": refined_cost,
    }


def derive_readout(cells: Sequence[dict]) -> dict:
    mapping = [row for row in cells if row.get("support") == "mapping"]
    instance = [row for row in cells if row.get("support") == "instance"]
    valid = (
        len(mapping) == 3
        and len(instance) == 2
        and all(row.get("refined_endpoint_interval", {}).get("valid") for row in cells)
    )
    try:
        mapping_minimum = max(
            1.0,
            max(row["refined_cost_interval"]["C_time_lower_us"] for row in mapping)
            / min(row["refined_cost_interval"]["C_time_upper_us"] for row in mapping),
        )
        instance_maximum = (
            max(row["refined_cost_interval"]["C_time_upper_us"] for row in instance)
            / min(row["refined_cost_interval"]["C_time_lower_us"] for row in instance)
        )
        finite = math.isfinite(mapping_minimum) and math.isfinite(instance_maximum)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        mapping_minimum = None
        instance_maximum = None
        finite = False
    valid = valid and finite
    separated = bool(valid and mapping_minimum > instance_maximum)
    return {
        "status": "compared" if valid else "invalid_refinement",
        "mapping_support_minimum": mapping_minimum if valid else None,
        "instance_support_maximum": instance_maximum if valid else None,
        "strict_separation": separated,
        "classification": (
            "qr3_mapping_spread_larger_on_preregistered_support"
            if separated
            else "indeterminate_after_refinement"
        ),
        "negative_readout_allowed": False,
        "global_extrema_reselected": False,
    }


def build_record(*, workers: int = 1, run: bool = True) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = load_config()
    problems = preregistration_problems(config)
    if problems:
        raise ValueError("R3d preregistration/lineage failed:\n  " + "\n  ".join(problems))
    environment = environment_problems(config)
    if run and environment:
        raise ValueError("R3d execution environment failed:\n  " + "\n  ".join(environment))
    protocol = config["protocol"]
    if (
        protocol["accuracy_target_millihartree"] != ACCURACY_TARGET_MILLIHARTREE
        or protocol["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
        or protocol["one_sided_delta"] != DELTA
        or protocol["environment_marginal_target_fraction"]
        != MARGINAL_TARGET_FRACTION
    ):
        raise ValueError("R3d protocol constants drifted from the bound sampler")
    cards = load_device_cards()
    declared_cards = {row["name"]: row["sha256"] for row in protocol["device_cards"]}
    if {card.name: card.sha256 for card in cards} != declared_cards:
        raise ValueError("the device cards differ from the R3d declaration")

    base = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_role": "targeted_accuracy_matched_cost_refinement",
        "evidence_tier": config["estimand"]["evidence_tier"],
        "search_uncertainty_evidence": "heuristic",
        "config_sha256": _canonical_sha256(config),
        "config_file_sha256": _file_sha256(CONFIG),
        "producer_sha256": _file_sha256(Path(__file__)),
        "preregistration": {
            "config": "benchmarks/configs/r3d_qr3_refinement.json",
            "merge_commit": PREREGISTRATION_MERGE,
            "result_commit_is_separate": True,
            "supports_and_endpoints_selected_before_every_r3d_draw": True,
        },
        "parent_lineage": config["parent_lineage"],
        "estimand": config["estimand"],
        "protocol": protocol,
        "future_readout_contract": config["future_readout"],
        "reporting_contract": config["reporting_contract"],
        "claim_boundary": RESULT_CLAIM_BOUNDARY,
        "execution_environment": {
            **protocol["execution_environment"],
            "matches_preregistration": not environment,
            "problems": environment,
        },
        "device_cards": [{"sha256": card.sha256, **card.to_dict()} for card in cards],
    }
    if not run:
        return stamp_record(
            {
                **base,
                "refinement_run": {
                    "executed": False,
                    "reason": "explicitly skipped by caller",
                    "target_cell_count": len(config["refinement"]["target_cells"]),
                    "new_endpoint_cell_count": sum(
                        len(row["new_endpoints_effective_shots_per_setting"])
                        for row in config["refinement"]["target_cells"]
                    ),
                },
                "readout": {"classification": "run_not_executed"},
            }
        )

    arguments = [
        (index, target, protocol)
        for index, target in enumerate(config["refinement"]["target_cells"])
    ]
    if workers == 1:
        cells = [_run_cell(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cells = list(pool.map(_run_cell, arguments))
    return stamp_record(
        {
            **base,
            "refinement_run": {
                "executed": True,
                "target_cell_count": len(cells),
                "new_endpoint_cell_count": sum(
                    len(row["new_endpoints_effective_shots_per_setting"])
                    for row in cells
                ),
                "parent_evidence_reused_as_samples": False,
                "cells": cells,
            },
            "readout": derive_readout(cells),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record(workers=args.workers, run=not args.skip_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("R3d:", record["readout"]["classification"])
    if record["refinement_run"]["executed"]:
        print(
            f"  {record['refinement_run']['target_cell_count']} parent cells, "
            f"{record['refinement_run']['new_endpoint_cell_count']} new endpoints"
        )
        print(
            "  mapping minimum:", record["readout"]["mapping_support_minimum"],
            "instance maximum:", record["readout"]["instance_support_maximum"],
        )
    print(args.out)


if __name__ == "__main__":
    main()
