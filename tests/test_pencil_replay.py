"""Replay accounting, immutable inputs, and explicit campaign boundaries."""

import copy
import hashlib
import json

import numpy as np
import pytest

from benchmarks import replay_pencil_estimators as replay
from benchmarks import run_prd_case_finite_shot as finite
from benchmarks import run_prd_case_paper_suite as suite


@pytest.fixture
def manifest():
    return suite.load_manifest(finite.DEFAULT_CONFIG)


def seed_record(manifest, index=0):
    task = finite.build_tasks(manifest)[index]
    frozen = finite.load_frozen_bases(finite.DEFAULT_CONFIG, manifest)
    S = np.eye(7)
    S[0, 1] = S[1, 0] = 1.2
    return {
        "schema": finite.SCHEMA, "task": task, "M": 7,
        "config_sha256": suite.config_sha256(manifest),
        "basis_sha256": frozen["systems"][task["system"]]["basis_sha256"],
        "physical_shots": task["shot_budget"],
        "exact_energy": -1.0, "exact_packet_energy": -0.9,
        "S_hat": finite._complex_matrix(S),
        "H_hat": finite._complex_matrix(np.diag(np.arange(7) - 1)),
    }


def test_two_arms_read_the_same_matrices_and_retain_sampling_error(manifest):
    record = seed_record(manifest)
    original = copy.deepcopy(record)
    result = replay.evaluate_record(
        record, {"norm_floor": 1e-14}, violation_tolerance=1e-9,
        chemical_accuracy_mha=1.6)
    assert record == original
    assert len(result["arms"]) == 2
    for arm in result["arms"]:
        assert arm["status"] == "ok"
        assert arm["sampling_error_mHa"] == pytest.approx((arm["energy"] + 0.9) * 1000)
    assert result["arms"][1]["repair"]["normalized_min_eigenvalue_after"] >= -1e-12


def test_failed_solves_are_counted_and_errors_are_explicitly_conditional(manifest):
    record = seed_record(manifest)
    # Negative measured norms cannot be preserved by a PSD repair. Both arms
    # fail when there is no live generator; neither failure disappears.
    record["S_hat"] = finite._complex_matrix(-np.eye(7))
    cell = replay.evaluate_record(
        record, {"norm_floor": 1e-14}, violation_tolerance=1e-9,
        chemical_accuracy_mha=1.6)
    for summary in replay.summarize([cell]):
        assert summary["cells"] == summary["failed_solves"] == 1
        assert summary["failure_rate"] == 1.0
        assert summary["chemical_accuracy_rate_all_cells"] == 0.0
        assert summary["median_absolute_error_mHa_successful_solves"] is None
        assert summary["variational_violation_rate_successful_solves"] is None
    json.dumps(cell, allow_nan=False)


def test_source_bytes_are_unchanged_and_a_partial_replay_cannot_claim_completion(
        tmp_path, manifest):
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(seed_record(manifest)))
    before = path.read_bytes()
    frozen = finite.load_frozen_bases(finite.DEFAULT_CONFIG, manifest)
    report = replay.replay_records([path], manifest, frozen)
    assert path.read_bytes() == before
    assert report["sources"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert report["expected_cells"] == 180 and report["replayed_cells"] == 1
    assert len(report["missing_task_ids"]) == 179
    assert not report["complete_campaign"]
    assert report["additional_physical_shots"] == 0
    assert report["interval_coverage"] is None
    json.dumps(report, allow_nan=False)


def test_replay_rejects_duplicates_and_foreign_configs(tmp_path, manifest):
    path = tmp_path / "cell.json"
    record = seed_record(manifest)
    frozen = finite.load_frozen_bases(finite.DEFAULT_CONFIG, manifest)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duplicate campaign task"):
        replay.replay_records([path, path], manifest, frozen)
    record["config_sha256"] = "foreign"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="config hash disagrees"):
        replay.replay_records([path], manifest, frozen)
    with pytest.raises(ValueError, match="no PRD seed records"):
        replay.replay_records([], manifest, frozen)


def test_replay_rejects_undeclared_tasks_and_mixed_bases(tmp_path, manifest):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    first, second = seed_record(manifest), seed_record(manifest, 1)
    frozen = finite.load_frozen_bases(finite.DEFAULT_CONFIG, manifest)
    second["basis_sha256"] = "different-test-fixture"
    left.write_text(json.dumps(first))
    right.write_text(json.dumps(second))
    with pytest.raises(ValueError, match="basis digest disagrees"):
        replay.replay_records([left, right], manifest, frozen)
    first["task"]["shot_budget"] += 1
    left.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="undeclared campaign task"):
        replay.replay_records([left], manifest, frozen)


@pytest.mark.parametrize("field,value,message", [
    ("schema", "other", "expected a PRD"),
    ("S_hat", [], "finite square"),
    ("H_hat", [[[0, 0]]], "dimensions disagree"),
    ("M", 8, "dimensions disagree"),
    ("exact_energy", float("nan"), "energies must be finite"),
    ("physical_shots", 1, "physical shots disagree"),
])
def test_malformed_records_fail_before_aggregation(manifest, field, value, message):
    record = seed_record(manifest)
    record[field] = value
    with pytest.raises(ValueError, match=message):
        replay.evaluate_record(record, {"norm_floor": 1e-14}, violation_tolerance=1e-9,
                               chemical_accuracy_mha=1.6)


def test_cli_writes_a_separate_report_and_refuses_overwrites(tmp_path, manifest):
    record = seed_record(manifest)
    source = tmp_path / f"{record['task']['task_id']}.json"
    source.write_text(json.dumps(record))
    output = tmp_path / "replay.json"
    argv = ["--input-dir", str(tmp_path), "--output", str(output)]
    replay.main(argv)
    assert json.loads(output.read_text())["replayed_cells"] == 1
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        replay.main(argv)
    assert output.read_bytes() == before
