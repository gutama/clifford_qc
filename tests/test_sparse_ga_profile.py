"""Profile audit catches corrupt decisions/counts without timing thresholds."""
from copy import deepcopy

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
