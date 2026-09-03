"""Validate the result-free R3c full-cost preregistration.

R3c is intentionally split across commits.  This checker lands with the config
and performs no sampling.  It binds the future 30+100 run to the LiH bank R3b
piloted, the exact-tier search constants, fresh random streams, and the device
cards already used by the cost layer.  A producer and sampled record belong in
a later commit.

    python benchmarks/check_r3c_preregistration.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

try:  # package import in tests versus direct script execution
    from benchmarks.r3_environment_migration import (
        load_manifest as load_migration_manifest,
        r3b_findings,
        record_successor_problems,
    )
    from benchmarks.check_r3b_preregistration import (
        _result_key_paths,
        digest_problems as r3b_digest_problems,
        lineage_problems as r3b_lineage_problems,
        rederivation_problems as r3b_rederivation_problems,
    )
    from benchmarks.run_exact_shot_search import (
        BOOTSTRAP_REPLICATES,
        CONFIRMATORY_REPLICAS,
        CONFIRMATORY_SEED,
        DELTA,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        EXPLORATORY_SEED,
        SEARCH_ENDPOINTS,
        BOOTSTRAP_SEED,
    )
except ImportError:  # pragma: no cover - direct script execution
    from r3_environment_migration import (
        load_manifest as load_migration_manifest,
        r3b_findings,
        record_successor_problems,
    )
    from check_r3b_preregistration import (
        _result_key_paths,
        digest_problems as r3b_digest_problems,
        lineage_problems as r3b_lineage_problems,
        rederivation_problems as r3b_rederivation_problems,
    )
    from run_exact_shot_search import (
        BOOTSTRAP_REPLICATES,
        CONFIRMATORY_REPLICAS,
        CONFIRMATORY_SEED,
        DELTA,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        EXPLORATORY_SEED,
        SEARCH_ENDPOINTS,
        BOOTSTRAP_SEED,
    )

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "r3c_lih_full_cost.json"
R3B_CONFIG = HERE / "configs" / "r3b_margin_stop_probe.json"
QR3B_CONFIG = HERE / "configs" / "qr3b_instance_preflight.json"
R3B_RECORD = HERE / "reference_results" / "r3b_margin_stop_probe.json"
PROTOCOL_COST_RECORD = HERE / "reference_results" / "protocol_cost.json"

CONFIG_SCHEMA = "clifford_qc.r3c_lih_full_cost_config.v1"
R3B_MERGE_COMMIT = "09c8d68a2e9249d3ae3fed33675cf462310be230"
R3B_CONFIG_CANONICAL_SHA256 = (
    "cfc1037c3067e942c0cd5e004a7aa61acfee4deb3d210db7dad4287edb4a6ca5"
)
R3B_RECORD_GIT_BLOB_SHA1 = "7c7eb9b22990ceab972ce8d5dc5580df63e5ab30"
EXPECTED_CLAIM_BOUNDARY = (
    "This config carries no sampled result, verdict, cost bracket, C(epsilon), "
    "or k-star field. A later record may report only the frozen LiH margin-stop "
    "bank under the declared 30+100 exact-tier protocol and illustrative device "
    "cards; it may not alter the bank, grid, estimators, replica counts, pass "
    "rule, seed roots, or the existing R3b and QR3b findings."
)
TEMPORAL_BOUNDARY_PHRASES = (
    "no sampling has been performed",
    "has not been sampled",
    "has not run",
    "not run yet",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _card_hashes(rows, label: str, problems: list[str]) -> dict[str, str]:
    """Return an unambiguous name-to-digest map without trusting the payload."""
    cards: dict[str, str] = {}
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
        raise ValueError("unsupported R3c preregistration config schema")
    if config.get("status") != "preregistration_only_no_run_yet":
        raise ValueError(
            "this config declares a status other than preregistration; a sampled "
            "record belongs in reference_results in a later commit"
        )
    return config


def lineage_problems(config: dict) -> list[str]:
    """Bind R3c to the exact R3b preregistration and sampled pilot."""
    problems: list[str] = []
    pilot = config["parent_lineage"]["pilot"]
    r3b_config = _read(R3B_CONFIG)
    r3b_record = _read(R3B_RECORD)

    expected = {
        "phase": "R3b",
        "merged_pull_request": 73,
        "merge_commit": R3B_MERGE_COMMIT,
        "config": "benchmarks/configs/r3b_margin_stop_probe.json",
        "config_canonical_sha256": R3B_CONFIG_CANONICAL_SHA256,
        "record": "benchmarks/reference_results/r3b_margin_stop_probe.json",
        "record_git_blob_sha1": R3B_RECORD_GIT_BLOB_SHA1,
    }
    for field, value in expected.items():
        if pilot.get(field) != value:
            problems.append(f"parent_lineage.pilot.{field} does not name frozen R3b")

    config_digest = _canonical_sha256(r3b_config)
    if config_digest != R3B_CONFIG_CANONICAL_SHA256:
        problems.append("the committed R3b config no longer has its frozen canonical digest")
    if config_digest != r3b_record.get("config_sha256"):
        problems.append("the R3b record no longer binds the committed R3b config")
    if config_digest != pilot.get("config_canonical_sha256"):
        problems.append("R3c does not bind the canonical R3b config payload")
    if _git_blob_sha1(R3B_RECORD) != pilot.get("record_git_blob_sha1"):
        problems += [
            f"R3b record migration: {problem}"
            for problem in record_successor_problems(
                "r3b_margin_stop_probe",
                historical_git_blob_sha1=pilot.get("record_git_blob_sha1"),
            )
        ]

    # The R3S and QR3b lineage fields are copied so the existing deterministic
    # R3b digest checker can continue to validate the bank's full ancestry.
    r3b_view = copy.deepcopy(r3b_config)
    r3b_view["parent_lineage"] = {
        key: copy.deepcopy(config["parent_lineage"][key])
        for key in ("repository", "admitted_by", "extends")
    }
    problems += [f"R3b ancestry: {p}" for p in r3b_digest_problems(r3b_view)]
    return problems


def pilot_problems(config: dict) -> list[str]:
    """Keep both the historical pilot and its migrated redraw explicit."""
    problems: list[str] = []
    record = _read(R3B_RECORD)
    current_decision, current_diagnosis = r3b_findings(record)
    migration = load_migration_manifest()["records"]["r3b_margin_stop_probe"]
    historical_decision = migration["before"]["decision"]
    historical_diagnosis = migration["before"]["diagnosis"]

    expected_historical_decision = {
        "status": "rejected_unresolved_at_frozen_grid",
        "evaluated_cells": 40,
        "resolved_cells": 30,
        "unresolved_cells": 10,
        "cells_at_or_beyond_ceiling": 1,
        "full_run_authorized": False,
        "eligible_for_full_run": False,
    }
    for field, value in expected_historical_decision.items():
        if historical_decision.get(field) != value:
            problems.append(f"historical R3b pilot {field} drifted from R3c's basis")
    expected_historical_diagnosis = {
        "grid_fit_failures": 0,
        "confirmation_failures": 10,
        "unclassified_failures": 0,
    }
    for field, value in expected_historical_diagnosis.items():
        if historical_diagnosis.get(field) != value:
            problems.append(f"historical R3b pilot {field} drifted from R3c's basis")

    if current_decision != migration["after"]["decision"]:
        problems.append("the live R3b decision differs from its migrated redraw")
    if current_diagnosis != migration["after"]["diagnosis"]:
        problems.append("the live R3b diagnosis differs from its migrated redraw")
    if current_decision.get("status") != "rejected_unresolved_at_frozen_grid":
        problems.append("the migrated R3b record no longer rejects the 2+2 probe")
    if current_decision.get("full_run_authorized") is not False:
        problems.append("the migrated R3b record authorizes a full run")
    if current_decision.get("eligible_for_full_run") is not False:
        problems.append("the migrated R3b record makes the bank eligible")
    if current_diagnosis.get("grid_fit_failures") != 0:
        problems.append("the migrated R3b record reintroduces a grid-fit failure")
    if record["protocol"].get("exploratory_replicas") != 2:
        problems.append("R3b pilot exploratory replica count is no longer 2")
    if record["protocol"].get("confirmatory_replicas") != 2:
        problems.append("R3b pilot confirmatory replica count is no longer 2")
    if not config["parent_lineage"]["pilot"].get("preserved_finding"):
        problems.append("R3c must state the R3b finding it preserves")
    return problems


def candidate_problems(config: dict) -> list[str]:
    """The full run changes the instrument, never the bank or mapping arms."""
    problems: list[str] = []
    r3b = _read(R3B_CONFIG)
    if config.get("candidate") != r3b["candidate"]:
        problems.append("R3c candidate differs from the bank R3b froze and piloted")
    if config.get("mapping_arms") != r3b["mapping_arms"]:
        problems.append("R3c mapping arms differ from R3b")
    if config["structural_gates"].get("source_qubits") != 8:
        problems.append("R3c must retain the eight-qubit LiH active space")
    if config["structural_gates"].get("word_universe_ceiling") != 2048:
        problems.append("R3c must retain the screen's 2048-word ceiling")
    if config["structural_gates"].get(
        "selection_may_not_use_mapping_cost_direction"
    ) is not True:
        problems.append("R3c may not select the bank using a mapping cost direction")
    if config["structural_gates"].get(
        "full_30_plus_100_run_authorized_by_this_config"
    ) is not True:
        problems.append("R3c must explicitly authorize exactly one later full run")
    return problems


def _historical_seed_roots() -> set[int]:
    r3b = _read(R3B_CONFIG)["protocol"]["seed_roots"]
    qr3b = _read(QR3B_CONFIG)["protocol"]["seed_roots"]
    cost = _read(PROTOCOL_COST_RECORD)["protocol"]
    return {
        *r3b.values(),
        *qr3b.values(),
        EXPLORATORY_SEED,
        CONFIRMATORY_SEED,
        BOOTSTRAP_SEED,
        cost["exploratory_seed"],
        cost["confirmatory_seed"],
        cost["bootstrap_seed"],
    }


def protocol_problems(config: dict) -> list[str]:
    """Freeze the headline exact-tier protocol and independent random streams."""
    problems: list[str] = []
    protocol = config["protocol"]
    expected = {
        "block_sizes": [1, 2, 4, 8],
        "estimators": list(ESTIMATORS),
        "search_endpoints_effective_shots_per_setting": list(SEARCH_ENDPOINTS),
        "exploratory_replicas": EXPLORATORY_REPLICAS,
        "confirmatory_replicas": CONFIRMATORY_REPLICAS,
        "nested_endpoints": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "one_sided_delta": DELTA,
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            problems.append(f"protocol.{field} differs from the frozen exact-tier value")

    roots = protocol.get("seed_roots", {})
    if set(roots) != {"exploratory", "confirmatory", "bootstrap"}:
        problems.append("R3c must declare exploratory, confirmatory, and bootstrap roots")
    elif any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in roots.values()):
        problems.append("R3c seed roots must be non-negative integers")
    elif len(set(roots.values())) != 3:
        problems.append("R3c seed roots must be distinct from each other")
    elif set(roots.values()) & _historical_seed_roots():
        problems.append("an R3c seed root aliases an existing exact-tier stream")
    if not protocol.get("seed_derivation"):
        problems.append("R3c must declare how seed roots derive disjoint streams")

    cost_record = _read(PROTOCOL_COST_RECORD)
    provenance = cost_record["provenance"]
    dependencies = provenance["dependencies"]
    frozen_environment = {
        "python_minor": ".".join(provenance["python"].split(".")[:2]),
        "numpy": dependencies["numpy"],
        "scipy": dependencies["scipy"],
        "stim": dependencies["stim"],
        "reference_record": "benchmarks/reference_results/protocol_cost.json",
    }
    if protocol.get("execution_environment") != frozen_environment:
        problems.append("R3c execution environment differs from the exact-tier baseline")

    cards = _card_hashes(protocol.get("device_cards"), "R3c", problems)
    frozen_cards = _card_hashes(cost_record.get("device_cards"), "protocol_cost", problems)
    if cards != frozen_cards:
        problems.append("R3c device-card names or hashes differ from protocol_cost")
    return problems


def rule_problems(config: dict) -> list[str]:
    """Pin the pass, crossing, censoring, and reporting semantics."""
    problems: list[str] = []
    rule = config["acceptance_rule"]
    if rule.get("accuracy_target_millihartree") != 1.6:
        problems.append("R3c accuracy target must remain 1.6 mHa")
    if rule.get("zero_failure_gate") is not True:
        problems.append("R3c must retain the zero-solver-failure gate")
    if rule.get("statistic") != "one-sided bootstrap upper confidence bound on replica RMSE":
        problems.append("R3c changed the preregistered RMSE statistic")
    if rule.get("one_sided_confidence") != 0.95:
        problems.append("R3c must retain the one-sided 95% confidence level")
    if "confirmed failing predecessor and confirmed passing endpoint" not in rule.get(
        "crossing_policy", ""
    ):
        problems.append("R3c must define a finite crossing interval from confirmed endpoints")
    censoring = rule.get("censoring_policy", "")
    if "right-censored" not in censoring or "does not license a wider grid" not in censoring:
        problems.append("R3c must preserve right-censoring at the frozen ceiling")

    reporting = config["reporting_contract"]
    if reporting.get("evidence_tier") != "exact":
        problems.append("R3c is an exact oracle-reference tier, not certified evidence")
    if reporting.get("full_run_is_a_cost_record") is not True:
        problems.append("the later R3c run must be identified as a cost record")
    if reporting.get("result_commit_must_be_separate") is not True:
        problems.append("the sampled R3c record must land in a later commit")
    if reporting.get("existing_records_are_immutable") is not True:
        problems.append("R3c may not rewrite R3b, QR3b, or another existing record")
    return problems


def claim_problems(config: dict) -> list[str]:
    """Keep this commit result-free and its wording true after a later run."""
    problems = [
        f"result field in an R3c preregistration: {path}"
        for path in _result_key_paths(config)
    ]
    boundary = config.get("claim_boundary", "")
    if boundary != EXPECTED_CLAIM_BOUNDARY:
        problems.append("R3c claim boundary differs from the frozen tenseless wording")
    lowered = boundary.lower()
    for phrase in TEMPORAL_BOUNDARY_PHRASES:
        if phrase in lowered:
            problems.append(f"R3c claim boundary is temporal: {phrase!r}")
    return problems


def static_problems(config: dict) -> list[str]:
    problems = lineage_problems(config)
    problems += pilot_problems(config)
    problems += candidate_problems(config)
    problems += protocol_problems(config)
    problems += rule_problems(config)
    problems += claim_problems(config)
    return problems


def main() -> int:
    config = load_config()
    problems = static_problems(config)

    # Candidate equality above connects this deterministic rebuild to the new
    # config; the upstream checker remains the single implementation of the
    # bank reconstruction and margin-rule proof.
    r3b = _read(R3B_CONFIG)
    problems += [f"R3b bank lineage: {p}" for p in r3b_lineage_problems(r3b)]
    problems += [f"R3b bank rebuild: {p}" for p in r3b_rederivation_problems(r3b)]

    for problem in problems:
        print(f"  {problem}")
    print("R3c preregistration:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
