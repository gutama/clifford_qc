"""Validate the result-free R3d within-grid QR3 refinement declaration.

This checker reads only committed configs, code, and historical records.  It
does not construct a bank, draw a replica, or classify QR3.  The R3d producer,
sampled record, and regenerating checker belong in a later commit.

    python benchmarks/check_r3d_preregistration.py
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "configs" / "r3d_qr3_refinement.json"
R3C_CONFIG = HERE / "configs" / "r3c_lih_full_cost.json"
R3C_RECORD = HERE / "reference_results" / "r3c_lih_full_cost.json"
PROTOCOL_COST_RECORD = HERE / "reference_results" / "protocol_cost.json"

CONFIG_SCHEMA = "clifford_qc.r3d_qr3_refinement_config.v1"
EXPECTED_STATUS = "preregistration_only_no_refinement_run_yet"
EXPECTED_BASE = "eaddbb454cdf867c013839a97f6541600b07a6c2"
EXPECTED_R3C_RESULT_COMMIT = "4d664e2857c57452e9e06c0f15ab6396e115e540"
EXPECTED_CLAIM_BOUNDARY = (
    "This configuration contains historical R3c support coordinates and inherited "
    "parent endpoint intervals, but no R3d sample, endpoint classification, refined "
    "interval, refined spread, execution stamp, or QR3 classification. A later "
    "record may report only the seven frozen within-grid endpoint cells under the "
    "declared 30+100 protocol. It may establish the positive QR3 direction on the "
    "preregistered supports or remain indeterminate; it may not report a negative "
    "QR3 direction, widen the grid, reselect an extremal cell, or alter either "
    "parent record."
)

PARENT_GRID = [64, 256, 1024, 4096, 16384, 65536]
GLOBAL_MIDPOINTS = [128, 512, 2048, 8192, 32768]
COMBINED_GRID = sorted(PARENT_GRID + GLOBAL_MIDPOINTS)
EXPECTED_SEED_ROOTS = {
    "exploratory": 160814000,
    "confirmatory": 160914000,
    "bootstrap": 161014000,
}
EXPECTED_CARD_SHA256 = {
    "ion-like": "9cc7e53e935e4ec3c265efbd9b6218f17a68c51d727d658c5a5bcaf98d34f453",
    "logical-alltoall": "07e9cc637e590668bc89b7d407535f34f509155e08c95d2ecd0f8f931738265b",
    "superconducting-like": "cf78e3240ad48839a8113a1570a43e614c230a8428fe8ceda7783e7ff75e1a13",
}
EXPECTED_POSITIVE_READOUT = (
    "qr3_mapping_spread_larger_on_preregistered_support, if and only if both "
    "supports are valid and mapping_support_minimum is strictly greater than "
    "instance_support_maximum"
)
EXPECTED_INDETERMINATE_READOUT = (
    "indeterminate_after_refinement, including overlap, equality, marginal-only "
    "non-resolution, missing evidence, nonmonotonicity, or an invalid interval"
)
EXPECTED_REFINEMENT_RULES = {
    "endpoint_rule": (
        "For each target cell, sample every global geometric midpoint strictly inside "
        "that cell's inherited conservative endpoint interval and no other endpoint."
    ),
    "ceiling_policy": (
        "No endpoint above 65536 is licensed. R3d refines resolution between existing "
        "endpoints and is not a wider-grid search."
    ),
    "parent_evidence_policy": (
        "The two parent records and their endpoint decisions are immutable and read-only. "
        "R3d does not redraw, overwrite, pool replicas with, or reinterpret a parent "
        "endpoint. It begins from each parent's already-conservative interval."
    ),
    "new_evidence_policy": (
        "Every declared midpoint receives its own R3d exploratory and confirmatory "
        "blocks. Only the confirmatory block can tighten an inherited interval; "
        "exploration is reported as a protocol diagnostic and cannot select or "
        "suppress an endpoint."
    ),
    "interval_update_rule": (
        "Start from the inherited conservative (lower, upper] endpoint interval. A "
        "non-marginal confirmed R3d failure raises lower to max(lower, endpoint); a "
        "non-marginal confirmed R3d pass lowers upper to min(upper, endpoint). An "
        "environment-marginal R3d endpoint is reported but does not tighten either "
        "side. Parent intervals may only narrow, never widen."
    ),
    "invalid_refinement_rule": (
        "If an endpoint is missing, a required statistic is non-finite, the combined "
        "non-marginal decisions are nonmonotone, or the updated lower is not strictly "
        "below the updated upper, that support is invalid and the only allowed readout "
        "is indeterminate_after_refinement."
    ),
}
EXPECTED_PROTOCOL_RULES = {
    "seed_derivation": (
        "For exploratory and confirmatory sampling use numpy.SeedSequence(root, "
        "spawn_key=(target_cell_index, replica_index)); larger R3d endpoints reuse "
        "the prefix shots of the same cell/phase/replica stream. For bootstrap use "
        "spawn_key=(target_cell_index, phase_index, endpoint_index, estimator_index). "
        "Target-cell order is the array order above. Parent samples are never part "
        "of an R3d stream."
    ),
    "pass_rule": (
        "A new endpoint passes only when all 100 confirmatory solves succeed and the "
        "one-sided 95% bootstrap upper confidence bound on replica RMSE is at most "
        "1.6 mHa."
    ),
    "environment_marginal_rule": (
        "When the confirmatory RMSE upper bound is finite and abs(UCB - 1.6 mHa) / "
        "1.6 mHa is at most 0.10, label that R3d endpoint environment-marginal and "
        "do not use it to tighten either interval side."
    ),
}
EXPECTED_LINEAGE = {
    "benchmarks/configs/r3c_lih_full_cost.json": (
        "ad59b2f997d85e3856adb106ee2f808c45484a39",
        "cfbc97f3225ebdf8bdd8bcbddd2141214649664ef238a7cc98188db6f4160daa",
    ),
    "benchmarks/reference_results/r3c_lih_full_cost.json": (
        "5f2faba83d1533d5a8ac4e40432cc294d9aff676",
        "b3d302ae2bc8717658145022c8757d32543b51993e4ca2265d6d66c5ae9be673",
    ),
    "benchmarks/reference_results/protocol_cost.json": (
        "5466fda9e083c2d55b388446e03670b7026f5eec",
        "a1b5bea91d5924ae6c6878baf83e9d8b6b10ad107f4c3a2ab1644ddee1adff59",
    ),
    "benchmarks/run_exact_shot_search.py": (
        "0e2c93a28ab5c9b3517c3e21c0186dff510e8c36",
        "82f28e77e9e97fe57dae49c865dfed95c3c4f74138889a0cd01c9967454a17c7",
    ),
    "benchmarks/run_protocol_cost.py": (
        "59862efd9d074338d76663a07265a940ab914eea",
        "8f94cbd285ce23720131fb5b185197d0fa998cb230dbd1316d19f5845a76724e",
    ),
    "benchmarks/run_r3c_lih_full_cost.py": (
        "ec37472ec5013292c233142fc8d0b5e8cdc90415",
        "1a2ee7380115b67f687ae1919900f8f6e56e012819daf98b0185070abf4fed9c",
    ),
    "benchmarks/configs/device_cards/ion-like.json": (
        "45ab9fd9b59638d2b5d0a94dafe3a5bcb311cb26",
        "1ac47a1a80b621bfeb3d5cc36dd9ab859bf94dc782da4c6fee8aebfb3d9c53c2",
    ),
    "benchmarks/configs/device_cards/logical-alltoall.json": (
        "21fab44375442ce632d12c2d6f3ba9bdad3815f7",
        "5eb4668daa124b4f8f375d37839ab4eeb31f10f8cf27612869e2aebcb9f30fe3",
    ),
    "benchmarks/configs/device_cards/superconducting-like.json": (
        "befb3fb9f09087174f7d07f59d2e1f4e8e4f7316",
        "99fac3de98a34324f31dd734918bc6448ce48dd315752c12dbcc503b1d82e248",
    ),
}

FORBIDDEN_RESULT_KEYS = {
    "result",
    "results",
    "outcome",
    "verdict",
    "r3d_sample",
    "r3d_samples",
    "sampling_evidence",
    "endpoint_classification",
    "refined_endpoint_interval",
    "refined_cost_interval",
    "refined_mapping_spread",
    "refined_instance_spread",
    "qr3_classification",
    "provenance",
    "executed_at",
}
TEMPORAL_BOUNDARY_PHRASES = (
    "no sampling has been performed",
    "has not been sampled",
    "has not run",
    "not run yet",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _result_key_paths(value, prefix=()):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = prefix + (str(key),)
            if key in FORBIDDEN_RESULT_KEYS:
                paths.append(".".join(here))
            paths.extend(_result_key_paths(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_result_key_paths(child, prefix + (str(index),)))
    return paths


def _card_map(rows, label: str, problems: list[str]) -> dict[str, str]:
    cards = {}
    if not isinstance(rows, list):
        problems.append(f"{label} device_cards must be a list")
        return cards
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"{label} device_cards[{index}] must be an object")
            continue
        name = row.get("name")
        digest = row.get("sha256")
        if not isinstance(name, str) or not name:
            problems.append(f"{label} device_cards[{index}] has no valid name")
            continue
        if name in cards:
            problems.append(f"{label} device_cards contains duplicate name {name!r}")
            continue
        if not isinstance(digest, str) or len(digest) != 64:
            problems.append(f"{label} device card {name!r} has no valid SHA-256")
            continue
        cards[name] = digest
    return cards


def load_config(path: Path = CONFIG) -> dict:
    config = _read(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported R3d preregistration schema")
    if config.get("status") != EXPECTED_STATUS:
        raise ValueError("the R3d sampled record must land in a later commit")
    leaked = _result_key_paths(config)
    if leaked:
        raise ValueError(f"result field in R3d preregistration: {leaked[0]}")
    return config


def lineage_problems(config: dict) -> list[str]:
    problems = []
    lineage = config.get("parent_lineage", {})
    if lineage.get("repository") != "gutama/clifford_qc":
        problems.append("the R3d repository lineage drifted")
    if lineage.get("base_commit") != EXPECTED_BASE:
        problems.append("R3d does not descend from the frozen post-R4a main head")
    merge = lineage.get("r3c_result_merge", {})
    if merge != {"pull_request": 79, "commit": EXPECTED_R3C_RESULT_COMMIT}:
        problems.append("the R3c result merge lineage drifted")

    rows = lineage.get("files", [])
    declared = {row.get("path"): row for row in rows if isinstance(row, dict)}
    if len(declared) != len(rows):
        problems.append("R3d lineage files must have unique valid paths")
    if set(declared) != set(EXPECTED_LINEAGE):
        problems.append("R3d lineage file set differs from the frozen inputs")
    for relative, (blob, digest) in EXPECTED_LINEAGE.items():
        row = declared.get(relative, {})
        if row.get("git_blob_sha1") != blob:
            problems.append(f"{relative} declared git_blob_sha1 drifted")
        if row.get("sha256") != digest:
            problems.append(f"{relative} declared sha256 drifted")
        path = ROOT / relative
        if not path.exists():
            problems.append(f"{relative} is missing")
            continue
        if _git_blob_sha1(path) != blob or _sha256(path) != digest:
            problems.append(f"{relative} content differs from the frozen R3d lineage")
    return problems


def _support_snapshot(comparison: dict, field: str) -> dict:
    row = comparison[field]
    identity_fields = (
        ("system", "card", "estimator", "measured_qubits", "block_size", "arms")
        if field == "widest_mapping_spread"
        else ("card", "estimator", "mapping", "measured_qubits", "block_size", "systems")
    )
    return {
        **{key: row[key] for key in identity_fields},
        "minimum_possible": row["minimum_possible"],
        "point": row["point"],
        "maximum_possible": row["maximum_possible"],
    }


def trigger_problems(config: dict) -> list[str]:
    problems = []
    comparison = _read(R3C_RECORD)["qr3_second_instance"]["comparison"]
    trigger = config.get("r3c_trigger", {})
    if comparison.get("status") != "compared":
        problems.append("the parent R3c record no longer contains a QR3 comparison")
    if comparison.get("verdict") != "indeterminate_at_this_shot_grid":
        problems.append("the parent R3c readout no longer motivates resolution")
    expected = {
        "comparison_status": comparison.get("status"),
        "parent_readout": comparison.get("verdict"),
        "priced_instances": comparison.get("priced_instances"),
        "widest_mapping_point_support": _support_snapshot(comparison, "widest_mapping_spread"),
        "narrowest_instance_point_support": _support_snapshot(
            comparison, "narrowest_instance_spread"
        ),
    }
    for field, value in expected.items():
        if trigger.get(field) != value:
            problems.append(f"r3c_trigger.{field} differs from the frozen R3c comparison")
    if "selected after R3c" not in trigger.get("selection_role", ""):
        problems.append("R3d must label its support-cell selection as post-R3c")
    if "before any R3d sample" not in trigger.get("selection_role", ""):
        problems.append("R3d must freeze support coordinates before R3d sampling")
    return problems


@functools.lru_cache(maxsize=1)
def _systems() -> dict[str, dict]:
    """Load the two immutable parent system payloads once per checker process."""
    r3c = _read(R3C_RECORD)
    cost = _read(PROTOCOL_COST_RECORD)
    return {
        "beh2": cost["systems"]["beh2"],
        "lih_cas4e4o_margin_stop": r3c["full_cost_run"]["sampling_evidence"],
    }


def _parent_cell(system: str, mapping: str, block_size: int, estimator: str):
    arm = next(row for row in _systems()[system]["arms"] if row["mapping"] == mapping)
    rung = next(row for row in arm["rungs"] if row["block_size"] == block_size)
    return arm, rung["estimators"][estimator]


def _expected_target_coordinates(config: dict) -> list[dict]:
    trigger = config["r3c_trigger"]
    mapping = trigger["widest_mapping_point_support"]
    instance = trigger["narrowest_instance_point_support"]
    rows = []
    for arm in mapping["arms"]:
        rows.append(
            {
                "support": "mapping",
                "parent_record": "benchmarks/reference_results/r3c_lih_full_cost.json",
                "system": mapping["system"],
                "mapping": arm,
                "measured_qubits": mapping["measured_qubits"],
                "block_size": mapping["block_size"],
                "estimator": mapping["estimator"],
                "card": mapping["card"],
            }
        )
    for system in instance["systems"]:
        rows.append(
            {
                "support": "instance",
                "parent_record": (
                    "benchmarks/reference_results/protocol_cost.json"
                    if system == "beh2"
                    else "benchmarks/reference_results/r3c_lih_full_cost.json"
                ),
                "system": system,
                "mapping": instance["mapping"],
                "measured_qubits": instance["measured_qubits"],
                "block_size": instance["block_size"],
                "estimator": instance["estimator"],
                "card": instance["card"],
            }
        )
    return rows


def target_problems(config: dict) -> list[str]:
    problems = []
    refinement = config.get("refinement", {})
    if refinement.get("parent_grid_effective_shots_per_setting") != PARENT_GRID:
        problems.append("the inherited R3c endpoint grid drifted")
    if refinement.get("global_geometric_midpoints_effective_shots_per_setting") != GLOBAL_MIDPOINTS:
        problems.append("the one-step geometric midpoint grid drifted")
    if refinement.get("combined_grid_effective_shots_per_setting") != COMBINED_GRID:
        problems.append("the declared combined endpoint grid drifted")

    rows = refinement.get("target_cells", [])
    expected_coordinates = _expected_target_coordinates(config)
    if not isinstance(rows, list) or len(rows) != len(expected_coordinates):
        problems.append("R3d must retain exactly five ordered target cells")
        return problems
    coordinate_fields = tuple(expected_coordinates[0])
    observed_coordinates = [
        {field: row.get(field) for field in coordinate_fields}
        for row in rows
        if isinstance(row, dict)
    ]
    if observed_coordinates != expected_coordinates:
        problems.append("R3d target cells differ from the two R3c point supports")

    endpoint_cells = 0
    for index, (row, expected) in enumerate(zip(rows, expected_coordinates)):
        try:
            arm, cell = _parent_cell(
                expected["system"],
                expected["mapping"],
                expected["block_size"],
                expected["estimator"],
            )
            bracket = cell["cost_bracket"]
            inherited = {
                "lower": bracket["lower_effective_shots_per_setting"],
                "point": bracket["point_effective_shots_per_setting"],
                "upper": bracket["upper_effective_shots_per_setting"],
                "widened_sides": bracket["widened_sides"],
            }
            if arm.get("measured_qubits") != expected["measured_qubits"]:
                problems.append(f"target cell {index} measured width drifted in its parent")
            if row.get("inherited_endpoint_interval") != inherited:
                problems.append(f"target cell {index} inherited endpoint interval drifted")
            card = cell["cost_bracket"].get("cards", {}).get(expected["card"], {})
            if card.get("admissible") is not True:
                problems.append(f"target cell {index} is not admissible on its target card")
            if card.get("device_card_sha256") != EXPECTED_CARD_SHA256[expected["card"]]:
                problems.append(f"target cell {index} parent device-card digest drifted")
            lower, upper = inherited["lower"], inherited["upper"]
            expected_endpoints = [
                endpoint for endpoint in GLOBAL_MIDPOINTS if lower < endpoint < upper
            ]
            if row.get("new_endpoints_effective_shots_per_setting") != expected_endpoints:
                problems.append(f"target cell {index} midpoint endpoints drifted")
            endpoint_cells += len(expected_endpoints)
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            problems.append(f"target cell {index} cannot be tied to its parent: {exc}")

    if refinement.get("target_cell_count") != 5:
        problems.append("R3d target_cell_count must remain five")
    if refinement.get("new_endpoint_cell_count") != endpoint_cells or endpoint_cells != 7:
        problems.append("R3d must retain exactly seven new endpoint cells")
    endpoints = [
        endpoint
        for row in rows
        for endpoint in row.get("new_endpoints_effective_shots_per_setting", [])
    ]
    if any(endpoint not in GLOBAL_MIDPOINTS for endpoint in endpoints):
        problems.append("R3d names an endpoint outside the midpoint grid")
    if any(endpoint > PARENT_GRID[-1] for endpoint in endpoints):
        problems.append("R3d widens the frozen 65536-shot ceiling")

    for field, value in EXPECTED_REFINEMENT_RULES.items():
        if refinement.get(field) != value:
            problems.append(f"refinement.{field} drifted")
    return problems


@functools.lru_cache(maxsize=1)
def _historical_seed_roots() -> frozenset[int]:
    roots = set()
    paths = [
        *sorted((HERE / "configs").rglob("*.json")),
        *sorted((HERE / "reference_results").rglob("*.json")),
    ]
    for path in paths:
        if path == CONFIG:
            continue
        try:
            payload = _read(path)
        except (OSError, json.JSONDecodeError):
            continue

        def visit(value, seed_namespace=False):
            if isinstance(value, dict):
                for key, child in value.items():
                    child_namespace = seed_namespace or (
                        "seed" in key.lower() and key != "seed_derivation"
                    )
                    if child_namespace and isinstance(child, int) and not isinstance(child, bool):
                        roots.add(child)
                    visit(child, child_namespace)
            elif isinstance(value, list):
                for child in value:
                    if seed_namespace and isinstance(child, int) and not isinstance(child, bool):
                        roots.add(child)
                    visit(child, seed_namespace)

        visit(payload)
    return frozenset(roots)


def protocol_problems(config: dict) -> list[str]:
    problems = []
    protocol = config.get("protocol", {})
    expected = {
        "accuracy_target_millihartree": 1.6,
        "exploratory_replicas_per_new_endpoint": 30,
        "confirmatory_replicas_per_new_endpoint": 100,
        "bootstrap_replicates": 10000,
        "one_sided_delta": 0.05,
        "environment_marginal_target_fraction": 0.10,
        "nested_new_endpoints_within_cell_and_phase": True,
        "parent_and_r3d_streams_are_disjoint": True,
        "seed_roots": EXPECTED_SEED_ROOTS,
        "execution_environment": {
            "python_minor": "3.12",
            "numpy": "2.5.2",
            "scipy": "1.18.0",
            "stim": "1.16.0",
        },
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            problems.append(f"protocol.{field} differs from the frozen R3d value")

    roots = protocol.get("seed_roots", {})
    if isinstance(roots, dict):
        values = list(roots.values())
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            problems.append("R3d seed roots must be plain integers")
        elif len(values) != len(set(values)):
            problems.append("R3d seed roots must be distinct")
        elif set(values) & set(_historical_seed_roots()):
            problems.append("an R3d seed root aliases a historical stream")
    for field, value in EXPECTED_PROTOCOL_RULES.items():
        if protocol.get(field) != value:
            problems.append(f"protocol.{field} drifted")

    cards = _card_map(protocol.get("device_cards"), "R3d", problems)
    if cards != EXPECTED_CARD_SHA256:
        problems.append("R3d device-card names or hashes drifted")
    r3c_cards = _card_map(_read(R3C_CONFIG)["protocol"]["device_cards"], "R3c", problems)
    if cards != r3c_cards:
        problems.append("R3d device cards differ from the R3c instrument")
    return problems


def readout_problems(config: dict) -> list[str]:
    problems = []
    estimand = config.get("estimand", {})
    if estimand.get("name") != "qr3_direction_on_preregistered_r3c_point_support":
        problems.append("the R3d estimand name drifted")
    if "strictly greater" not in estimand.get("question", ""):
        problems.append("the R3d estimand lost its strict directional inequality")
    if "cannot establish the opposite QR3 direction" not in estimand.get(
        "asymmetric_limitation", ""
    ):
        problems.append("R3d must retain its asymmetric inference boundary")
    if estimand.get("evidence_tier") != "exact":
        problems.append("R3d must remain an exact oracle-reference evidence tier")

    readout = config.get("future_readout", {})
    if readout.get("mapping_support_minimum_formula") != (
        "max(mapping-cell lower costs) / min(mapping-cell upper costs), clamped below at 1"
    ):
        problems.append("the mapping-support lower-bound formula drifted")
    if readout.get("instance_support_maximum_formula") != (
        "max(instance-cell upper costs) / min(instance-cell lower costs)"
    ):
        problems.append("the instance-support upper-bound formula drifted")
    if readout.get("positive") != EXPECTED_POSITIVE_READOUT:
        problems.append("the positive R3d readout rule drifted")
    if readout.get("otherwise") != EXPECTED_INDETERMINATE_READOUT:
        problems.append("every non-separating R3d readout must be indeterminate")
    for field, expected in (
        ("negative_readout_allowed", False),
        ("point_estimates_are_decisive", False),
        ("global_extrema_may_not_be_reselected_after_r3d", True),
    ):
        if readout.get(field) is not expected:
            problems.append(f"future_readout.{field} drifted")
    return problems


def claim_problems(config: dict) -> list[str]:
    problems = []
    leaked = _result_key_paths(config)
    if leaked:
        problems.append(f"result field in R3d preregistration: {leaked[0]}")
    boundary = config.get("claim_boundary", "")
    if boundary != EXPECTED_CLAIM_BOUNDARY:
        problems.append("the R3d claim boundary differs from the frozen wording")
    lowered = boundary.lower()
    for phrase in TEMPORAL_BOUNDARY_PHRASES:
        if phrase in lowered:
            problems.append(f"the R3d claim boundary is temporal: {phrase!r}")

    authorization = config.get("authorization", {})
    expected_authorization = {
        "exactly_one_later_refinement_execution": True,
        "producer": "benchmarks/run_r3d_qr3_refinement.py",
        "record": "benchmarks/reference_results/r3d_qr3_refinement.json",
        "checker": "benchmarks/check_r3d_qr3_refinement.py",
        "no_sampling_in_this_change": True,
    }
    for field, value in expected_authorization.items():
        if authorization.get(field) != value:
            problems.append(f"authorization.{field} drifted")
    if "later commit" not in authorization.get("temporal_order", ""):
        problems.append("R3d must separate the future sampled result commit")

    reporting = config.get("reporting_contract", {})
    for field in (
        "result_commit_must_be_separate",
        "existing_configs_and_records_are_immutable",
        "r3d_record_must_include_all_seven_endpoint_cells",
        "r3d_record_must_report_exploratory_and_confirmatory_blocks",
        "r3d_record_must_rederive_every_interval_and_spread",
        "sampled_ci_is_manual_dispatch_only",
    ):
        if reporting.get(field) is not True:
            problems.append(f"reporting_contract.{field} drifted")
    return problems


def static_problems(config: dict) -> list[str]:
    problems = []
    for check in (
        lineage_problems,
        trigger_problems,
        target_problems,
        protocol_problems,
        readout_problems,
        claim_problems,
    ):
        try:
            problems.extend(check(config))
        except (
            KeyError,
            TypeError,
            ValueError,
            StopIteration,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            problems.append(f"{check.__name__} could not validate the config: {exc}")
    return problems


def main() -> int:
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("R3d preregistration: FAIL")
        print(f"  - {exc}")
        return 1
    problems = static_problems(config)
    print("R3d preregistration:", "FAIL" if problems else "PASS")
    for problem in problems:
        print(f"  - {problem}")
    if not problems:
        print("  result-free contract; no R3d samples or QR3 classification")
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
