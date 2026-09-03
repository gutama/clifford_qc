"""Contracts for the completed R3 record-environment migration."""

from __future__ import annotations

import copy
import json

import benchmarks.r3_environment_migration as migration


def test_the_committed_migration_passes_every_gate():
    assert migration.migration_problems() == []


def test_both_live_records_use_the_frozen_target_environment():
    manifest = migration.load_manifest()
    for entry in manifest["records"].values():
        record = json.loads((migration.ROOT / entry["path"]).read_text())
        assert migration._record_environment(record) == migration.TARGET_ENVIRONMENT


def test_the_historical_and_migrated_r3b_findings_are_both_explicit():
    entry = migration.load_manifest()["records"]["r3b_margin_stop_probe"]
    assert entry["before"]["decision"]["resolved_cells"] == 30
    assert entry["before"]["diagnosis"]["confirmation_failures"] == 10
    assert entry["after"]["decision"]["resolved_cells"] == 29
    assert entry["after"]["diagnosis"]["confirmation_failures"] == 11
    for side in ("before", "after"):
        decision = entry[side]["decision"]
        assert decision["status"] == "rejected_unresolved_at_frozen_grid"
        assert decision["full_run_authorized"] is False
        assert decision["eligible_for_full_run"] is False
        assert entry[side]["diagnosis"]["grid_fit_failures"] == 0


def test_a_manifest_that_rewrites_a_reviewed_anchor_is_caught():
    broken = copy.deepcopy(migration.load_manifest())
    broken["records"]["r3b_margin_stop_probe"]["after"]["sha256"] = "0" * 64
    assert any("anchor differs" in problem for problem in migration.migration_problems(broken))


def test_a_preregistration_cannot_name_a_different_history():
    problems = migration.record_successor_problems(
        "r3b_margin_stop_probe", historical_git_blob_sha1="0" * 40
    )
    assert any("historical Git blob" in problem for problem in problems)
