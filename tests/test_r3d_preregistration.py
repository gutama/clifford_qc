"""Contracts for the result-free R3d within-grid QR3 declaration."""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_r3d_preregistration as r3d


@pytest.fixture(scope="module")
def config() -> dict:
    return r3d.load_config()


def _write(tmp_path, payload: dict):
    path = tmp_path / "r3d.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_committed_preregistration_passes_every_static_gate(config):
    assert r3d.static_problems(config) == []


def test_it_is_result_free_and_a_sampled_status_is_refused(tmp_path, config):
    assert config["status"] == r3d.EXPECTED_STATUS
    assert r3d._result_key_paths(config) == []

    broken = copy.deepcopy(config)
    broken["status"] = "sampled_record"
    with pytest.raises(ValueError, match="later commit"):
        r3d.load_config(_write(tmp_path, broken))


def test_every_parent_input_is_bound_by_blob_and_sha256(config):
    assert r3d.lineage_problems(config) == []
    assert set(r3d.EXPECTED_LINEAGE) == {row["path"] for row in config["parent_lineage"]["files"]}

    broken = copy.deepcopy(config)
    broken["parent_lineage"]["files"][1]["git_blob_sha1"] = "0" * 40
    assert any(
        "declared git_blob_sha1 drifted" in problem for problem in r3d.lineage_problems(broken)
    )


def test_the_r3c_trigger_and_point_supports_are_frozen(config):
    trigger = config["r3c_trigger"]
    assert trigger["parent_readout"] == "indeterminate_at_this_shot_grid"
    assert trigger["widest_mapping_point_support"]["point"] == pytest.approx(17.523996431440562)
    assert trigger["narrowest_instance_point_support"]["point"] == pytest.approx(1.0194094267289762)
    assert r3d.trigger_problems(config) == []

    broken = copy.deepcopy(config)
    broken["r3c_trigger"]["widest_mapping_point_support"]["arms"].reverse()
    assert any("frozen R3c comparison" in p for p in r3d.trigger_problems(broken))


def test_exactly_five_parent_cells_and_seven_midpoints_are_declared(config):
    refinement = config["refinement"]
    assert [
        row["new_endpoints_effective_shots_per_setting"] for row in refinement["target_cells"]
    ] == [[32768], [2048, 8192], [2048, 8192], [8192], [2048]]
    assert refinement["target_cell_count"] == 5
    assert refinement["new_endpoint_cell_count"] == 7
    assert r3d.target_problems(config) == []


def test_every_inherited_interval_is_rederived_from_its_parent(config):
    r3d._systems.cache_clear()
    assert r3d._systems() is r3d._systems()

    broken = copy.deepcopy(config)
    broken["refinement"]["target_cells"][1]["inherited_endpoint_interval"]["upper"] = 4096
    assert any("inherited endpoint interval drifted" in p for p in r3d.target_problems(broken))


def test_r3d_only_adds_geometric_midpoints_below_the_old_ceiling(config):
    refinement = config["refinement"]
    assert refinement["parent_grid_effective_shots_per_setting"] == r3d.PARENT_GRID
    assert (
        refinement["global_geometric_midpoints_effective_shots_per_setting"] == r3d.GLOBAL_MIDPOINTS
    )
    assert max(refinement["combined_grid_effective_shots_per_setting"]) == 65536

    broken = copy.deepcopy(config)
    broken["refinement"]["target_cells"][0]["new_endpoints_effective_shots_per_setting"] = [
        32768,
        131072,
    ]
    problems = r3d.target_problems(broken)
    assert any("midpoint endpoints drifted" in p for p in problems)
    assert any("widens" in p for p in problems)


def test_the_30_plus_100_protocol_and_environment_are_frozen(config):
    protocol = config["protocol"]
    assert protocol["exploratory_replicas_per_new_endpoint"] == 30
    assert protocol["confirmatory_replicas_per_new_endpoint"] == 100
    assert protocol["bootstrap_replicates"] == 10000
    assert protocol["one_sided_delta"] == 0.05
    assert protocol["execution_environment"] == {
        "python_minor": "3.12",
        "numpy": "2.5.2",
        "scipy": "1.18.0",
        "stim": "1.16.0",
    }
    assert r3d.protocol_problems(config) == []


def test_seed_roots_are_plain_distinct_and_fresh(config):
    roots = config["protocol"]["seed_roots"]
    assert roots == r3d.EXPECTED_SEED_ROOTS
    assert not set(roots.values()) & set(r3d._historical_seed_roots())

    broken = copy.deepcopy(config)
    broken["protocol"]["seed_roots"]["exploratory"] = True
    assert any("plain integers" in p for p in r3d.protocol_problems(broken))

    aliased = copy.deepcopy(config)
    aliased["protocol"]["seed_roots"]["exploratory"] = next(iter(r3d._historical_seed_roots()))
    assert any("aliases" in p for p in r3d.protocol_problems(aliased))


def test_device_cards_are_unique_and_match_the_parent_instrument(config):
    assert {
        row["name"]: row["sha256"] for row in config["protocol"]["device_cards"]
    } == r3d.EXPECTED_CARD_SHA256

    broken = copy.deepcopy(config)
    broken["protocol"]["device_cards"].append(copy.deepcopy(broken["protocol"]["device_cards"][0]))
    assert any("duplicate name" in p for p in r3d.protocol_problems(broken))


def test_parent_intervals_only_narrow_and_marginal_midpoints_do_not_tighten(config):
    refinement = config["refinement"]
    assert "may only narrow, never widen" in refinement["interval_update_rule"]
    assert "does not tighten either side" in refinement["interval_update_rule"]
    assert "do not use it to tighten" in config["protocol"]["environment_marginal_rule"]

    broken = copy.deepcopy(config)
    broken["refinement"]["interval_update_rule"] = "Replace the parent interval."
    assert any("interval_update_rule" in p for p in r3d.target_problems(broken))


def test_the_readout_is_strict_directional_and_never_negative(config):
    readout = config["future_readout"]
    assert "strictly greater" in readout["positive"]
    assert readout["otherwise"].startswith("indeterminate_after_refinement")
    assert readout["negative_readout_allowed"] is False
    assert readout["point_estimates_are_decisive"] is False
    assert readout["global_extrema_may_not_be_reselected_after_r3d"] is True
    assert r3d.readout_problems(config) == []

    broken = copy.deepcopy(config)
    broken["future_readout"]["negative_readout_allowed"] = True
    assert any("negative_readout_allowed" in p for p in r3d.readout_problems(broken))

    renamed = copy.deepcopy(config)
    renamed["future_readout"]["positive"] = renamed["future_readout"]["positive"].replace(
        "qr3_mapping_spread_larger", "qr3_positive"
    )
    assert any("positive R3d readout" in p for p in r3d.readout_problems(renamed))


def test_claim_boundary_and_separate_result_commit_are_enforced(tmp_path, config):
    assert config["claim_boundary"] == r3d.EXPECTED_CLAIM_BOUNDARY
    assert config["reporting_contract"]["result_commit_must_be_separate"] is True
    assert r3d.claim_problems(config) == []

    result_bearing = copy.deepcopy(config)
    result_bearing["qr3_classification"] = "positive"
    with pytest.raises(ValueError, match="result field"):
        r3d.load_config(_write(tmp_path, result_bearing))

    temporal = copy.deepcopy(config)
    temporal["claim_boundary"] = "No sampling has been performed under this config."
    assert any("temporal" in p for p in r3d.claim_problems(temporal))

    same_commit = copy.deepcopy(config)
    same_commit["reporting_contract"]["result_commit_must_be_separate"] = False
    assert any("result_commit_must_be_separate" in p for p in r3d.claim_problems(same_commit))
