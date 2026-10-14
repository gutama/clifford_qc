"""Profile audit catches corrupt decisions/counts without timing thresholds."""
from copy import deepcopy
import json
import subprocess

import pytest

from benchmarks import check_sparse_ga_kernel as gate
from benchmarks.metadata_io import read_json


@pytest.fixture
def record():
    return read_json(gate.ROOT / "benchmarks/reference_results/sparse_ga_kernel.json")


def test_committed_profile_and_large_sparse_bounds_are_valid(record):
    gate.check_record(record)
    large = next(row for row in record["fixtures"] if row["fixture"] == "n12_grade12_shared")
    assert large["candidate_count"] == 34 <= large["candidate_bound"] == 36


@pytest.mark.parametrize("failure", ["decision", "count", "definition", "cache", "nonfinite", "order", "source", "scope"])
def test_profile_refuses_corrupt_values_and_bindings(record, failure):
    broken = deepcopy(record)
    row = broken["fixtures"][0]
    measured = row["measurements"]["active"]
    if failure == "decision":
        measured["is_blade"] = False
    elif failure == "count":
        row["candidate_count"] += 1
    elif failure == "definition":
        row["masks"][0] = 0
    elif failure == "cache":
        measured["whole_grade_cache_entries"] = 1
    elif failure == "nonfinite":
        measured["cold_ns"] = float("inf")
    elif failure == "order":
        measured["warm_ns"]["min"] = measured["warm_ns"]["max"] + 1
    elif failure == "source":
        broken["source_files"][0]["sha256"] = "0" * 64
    else:
        broken["scope"] = "quantum speedup established"
    with pytest.raises(ValueError):
        gate.check_record(broken)


def test_timings_are_observations_without_fixed_performance_thresholds(record):
    for row in record["fixtures"]:
        for measured in row["measurements"].values():
            measured["cold_ns"] = 10**15
            measured["warm_ns"] = {"min": 10**12, "median": 10**13, "max": 10**14}
            measured["cold_tracemalloc_peak_bytes"] = 10**12
    gate.check_record(record)


@pytest.mark.parametrize("failure", ["dirty", "missing_commit", "wrong_commit"])
def test_execution_identity_must_be_clean_and_contain_the_sources(record, failure):
    if failure == "dirty":
        record["provenance"]["git_dirty"] = True
    else:
        record["provenance"]["git_sha"] = ("0" * 40 if failure == "missing_commit"
                                              else "5793459cfc3ab59f900af56b4ce30699345b6f1e")
    with pytest.raises(ValueError):
        gate.check_record(record)


def test_later_worktree_changes_do_not_rebind_historical_measurements(record, tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-b", "main")
    git("config", "user.name", "Profile fixture")
    git("config", "user.email", "profile@example.invalid")
    git("config", "commit.gpgsign", "false")
    execution = record["provenance"]["git_sha"]
    for source in record["source_files"]:
        raw = subprocess.check_output(["git", "show", f"{execution}:{source['path']}"], cwd=gate.ROOT)
        target = tmp_path / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    git("add", ".")
    git("commit", "-m", "measured source")
    record["provenance"]["git_sha"] = git("rev-parse", "HEAD")
    (tmp_path / "clifford_qc/multivector.py").write_text("# a later unrelated source edit\n")
    gate.check_record(record, tmp_path)


def test_live_candidate_enumeration_regression_is_observed(record, monkeypatch):
    from clifford_qc import multivector as kernel
    original = kernel._blade_contraction_buckets
    def extra(*args):
        buckets = original(*args)
        buckets[0] = []  # Extra work leaves the blade decision unchanged.
        return buckets
    monkeypatch.setattr(kernel, "_blade_contraction_buckets", extra)
    with pytest.raises(ValueError, match="live kernel candidate enumeration differs"):
        gate.check_record(record)


@pytest.mark.parametrize("warm", [[1, 2, 3], None, "timing"])
def test_malformed_warm_timing_is_a_clean_cli_failure(record, tmp_path, capsys, warm):
    record["fixtures"][0]["measurements"]["active"]["warm_ns"] = warm
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(record))
    assert gate.main(["--record", str(path)]) == 1
    errors = capsys.readouterr().err
    assert "warm timing shape differs" in errors and "Traceback" not in errors


def test_producer_refuses_dirty_sources_before_measurement(monkeypatch):
    from benchmarks import run_sparse_ga_kernel as producer
    original = producer.subprocess.check_output
    def dirty(command, **kwargs):
        if command[:3] == ["git", "status", "--porcelain"]:
            return " M clifford_qc/multivector.py\n"
        return original(command, **kwargs)
    monkeypatch.setattr(producer.subprocess, "check_output", dirty)
    with pytest.raises(ValueError, match="clean committed source tree"):
        producer.clean_execution()
