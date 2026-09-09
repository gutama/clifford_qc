"""Contracts for the sampled R3d within-grid QR3 refinement."""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_r3d_qr3_refinement as check
import benchmarks.run_r3d_qr3_refinement as r3d


@pytest.fixture(scope="module")
def config() -> dict:
    return check.load_config()


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(r3d.REFERENCE.read_text(encoding="utf-8"))


def test_the_committed_record_passes_every_static_contract(record):
    assert check.contract_problems(record) == []


def test_result_is_separate_from_the_landed_preregistration(record):
    preregistration = record["preregistration"]
    assert preregistration["merge_commit"] == r3d.PREREGISTRATION_MERGE
    assert preregistration["result_commit_is_separate"] is True
    assert record["config_file_sha256"] == r3d.PREREGISTRATION_SHA256


def test_exactly_five_parent_cells_and_seven_new_endpoints_are_reported(record):
    run = record["refinement_run"]
    assert run["executed"] is True
    assert run["target_cell_count"] == 5
    assert run["new_endpoint_cell_count"] == 7
    assert len(run["cells"]) == 5


def test_every_target_coordinate_and_endpoint_was_frozen(config, record):
    for index, (target, cell) in enumerate(
        zip(config["refinement"]["target_cells"], record["refinement_run"]["cells"])
    ):
        assert cell["target_cell_index"] == index
        for key in check.CELL_COORDINATES:
            assert cell[key] == target[key]
        assert (
            cell["new_endpoints_effective_shots_per_setting"]
            == target["new_endpoints_effective_shots_per_setting"]
        )


def test_each_endpoint_has_the_frozen_30_plus_100_blocks(config, record):
    for cell in record["refinement_run"]["cells"]:
        endpoints = cell["new_endpoints_effective_shots_per_setting"]
        assert [row["effective_shots_per_setting"] for row in cell["exploration"]] == endpoints
        assert [row["replicas"] for row in cell["exploration"]] == [30] * len(endpoints)
        assert [row["effective_shots_per_setting"] for row in cell["confirmation"]] == endpoints
        assert [row["replicas"] for row in cell["confirmation"]] == [100] * len(endpoints)
    assert config["protocol"]["bootstrap_replicates"] == 10_000


def test_endpoint_decisions_rederive_from_confirmation_only(config, record):
    for cell in record["refinement_run"]["cells"]:
        assert cell["endpoint_decisions"] == [
            r3d.endpoint_decision(row, config["protocol"])
            for row in cell["confirmation"]
        ]


def test_marginal_endpoints_never_tighten_an_interval(config):
    target = config["refinement"]["target_cells"][1]
    decisions = [
        {
            "effective_shots_per_setting": endpoint,
            "valid": True,
            "passes_target": endpoint == 8192,
            "environment_marginal": True,
            "informative_for_interval": False,
            "classification": "environment_marginal",
        }
        for endpoint in target["new_endpoints_effective_shots_per_setting"]
    ]
    refined = r3d.refined_shot_interval(target, decisions)
    assert refined["lower_effective_shots_per_setting"] == 1024
    assert refined["upper_effective_shots_per_setting"] == 16384


def test_nonmarginal_failure_raises_only_the_lower_endpoint(config):
    target = config["refinement"]["target_cells"][4]
    decision = {
        "effective_shots_per_setting": 2048,
        "valid": True,
        "passes_target": False,
        "environment_marginal": False,
        "informative_for_interval": True,
        "classification": "confirmed_fail",
    }
    refined = r3d.refined_shot_interval(target, [decision])
    assert refined["valid"] is True
    assert refined["lower_effective_shots_per_setting"] == 2048
    assert refined["upper_effective_shots_per_setting"] == 4096


def test_nonmarginal_pass_lowers_only_the_upper_endpoint(config):
    target = config["refinement"]["target_cells"][3]
    decision = {
        "effective_shots_per_setting": 8192,
        "valid": True,
        "passes_target": True,
        "environment_marginal": False,
        "informative_for_interval": True,
        "classification": "confirmed_pass",
    }
    refined = r3d.refined_shot_interval(target, [decision])
    assert refined["valid"] is True
    assert refined["lower_effective_shots_per_setting"] == 4096
    assert refined["upper_effective_shots_per_setting"] == 8192


def test_nonmonotone_new_evidence_forces_an_invalid_refinement(config):
    target = config["refinement"]["target_cells"][1]
    decisions = [
        {
            "effective_shots_per_setting": 2048,
            "valid": True,
            "passes_target": True,
            "environment_marginal": False,
            "informative_for_interval": True,
            "classification": "confirmed_pass",
        },
        {
            "effective_shots_per_setting": 8192,
            "valid": True,
            "passes_target": False,
            "environment_marginal": False,
            "informative_for_interval": True,
            "classification": "confirmed_fail",
        },
    ]
    refined = r3d.refined_shot_interval(target, decisions)
    assert refined["valid"] is False
    assert any("nonmonotone" in problem for problem in refined["problems"])


def test_support_spreads_and_readout_rederive_from_cells(record):
    cells = record["refinement_run"]["cells"]
    assert record["readout"] == r3d.derive_readout(cells)
    assert record["readout"]["classification"] in {
        "qr3_mapping_spread_larger_on_preregistered_support",
        "indeterminate_after_refinement",
    }


def test_no_negative_or_post_draw_reselection_is_licensed(record):
    readout = record["readout"]
    assert readout["negative_readout_allowed"] is False
    assert readout["global_extrema_reselected"] is False
    assert "negative" not in readout["classification"]


def test_mutating_a_result_field_is_detected(record):
    broken = copy.deepcopy(record)
    broken["refinement_run"]["cells"][0]["endpoint_decisions"][0][
        "classification"
    ] = "confirmed_pass"
    assert any("endpoint decisions do not re-derive" in p for p in check.contract_problems(broken))


def test_mutating_a_cost_or_readout_is_detected(record):
    broken = copy.deepcopy(record)
    broken["refinement_run"]["cells"][0]["refined_cost_interval"][
        "C_time_lower_us"
    ] *= 2
    broken["readout"]["classification"] = "negative"
    problems = check.contract_problems(broken)
    assert any("refined lower cost does not re-derive" in p for p in problems)
    assert any("classification do not re-derive" in p for p in problems)
    assert any("unlicensed" in p for p in problems)
