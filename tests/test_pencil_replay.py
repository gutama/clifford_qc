"""Replay accounting, immutable inputs, and explicit campaign boundaries."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

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
    # Both arms fail when there is no live generator; neither failure disappears.
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


@pytest.mark.parametrize("field", ["S_hat", "H_hat"])
def test_nonhermitian_records_are_rejected_before_estimator_accounting(manifest, field):
    record = seed_record(manifest)
    record[field][0][1] = [0.3, 0.0]
    record[field][1][0] = [0.0, 0.0]
    with pytest.raises(ValueError, match=f"{field} must be Hermitian"):
        replay.evaluate_record(record, {"norm_floor": 1e-14}, violation_tolerance=1e-9,
                               chemical_accuracy_mha=1.6)


@pytest.mark.parametrize("ridge,minimum", [([0.0] * 7, .2), ([2.0] + [0.0] * 6, 2.2)])
def test_retained_metric_uses_the_solvers_per_mode_decisions(manifest, ridge, minimum):
    record = seed_record(manifest)
    S = np.eye(7)
    S[0, 1] = S[1, 0] = .8
    record["S_hat"] = finite._complex_matrix(S)
    kwargs = {"norm_floor": 1e-14, "overlap_noise_floor": [.1] + [2.0] * 6,
              "overlap_ridge": ridge}
    cell = replay.evaluate_record(record, kwargs, violation_tolerance=1e-9,
                                  chemical_accuracy_mha=1.6)
    for arm in cell["arms"]:
        assert arm["status"] == "ok"
        assert arm["retained_rank"] == 1
        assert arm["smallest_retained_overlap_eigenvalue"] == pytest.approx(minimum)


def test_shared_prd_decoder_round_trips_complex_hermitian_entries():
    matrix = np.array([[1, .2 + .3j], [.2 - .3j, -.01]])
    np.testing.assert_array_equal(finite.decode_complex_matrix(finite._complex_matrix(matrix)), matrix)


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
    for module in (replay, finite, suite, replay.linalg, replay.pencil, replay.selection):
        name = module.__name__.replace(".", "/") + ".py"
        path = Path(module.__file__).resolve()
        assert report["implementation_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert report["implementation_paths"][name] == str(path)
    json.dumps(report, allow_nan=False)


def test_provenance_resolves_imported_module_paths(tmp_path, manifest, monkeypatch):
    source = tmp_path / "cell.json"
    source.write_text(json.dumps(seed_record(manifest)))
    frozen = finite.load_frozen_bases(finite.DEFAULT_CONFIG, manifest)
    original = replay.replay_records([source], manifest, frozen)
    alternative = tmp_path / "imported_policy.py"
    alternative.write_text("# alternative campaign acceptance source\n")
    monkeypatch.setattr(suite, "__file__", str(alternative))
    changed = replay.replay_records([source], manifest, frozen)
    key = "benchmarks/run_prd_case_paper_suite.py"
    assert changed["implementation_paths"][key] == str(alternative)
    assert changed["implementation_sha256"][key] != original["implementation_sha256"][key]


def test_direct_script_entry_point_works_outside_the_checkout(tmp_path):
    script = Path(replay.__file__).resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    done = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                          env=environment, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    assert "--input-dir" in done.stdout


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
