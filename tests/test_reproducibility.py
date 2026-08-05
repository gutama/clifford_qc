import json

from clifford_qc.reproducibility import (
    compare_json_records, execution_provenance, stamp_record,
)


def test_record_comparison_ignores_only_provenance():
    expected = {"energy": -1.0, "nested": {"count": 3},
                "provenance": {"git_sha": "old"}}
    actual = {"energy": -1.0, "nested": {"count": 3},
              "provenance": {"git_sha": "new", "utc": "later"}}
    assert compare_json_records(expected, actual) == []
    assert compare_json_records(expected, {**actual, "energy": -0.9})


def test_stamp_record_is_non_mutating_and_json_safe(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    source = {"schema": "example.v1", "value": 2}
    provenance = execution_provenance()
    stamped = stamp_record(source, provenance)
    assert "provenance" not in source
    assert stamped["provenance"]["git_sha"] == "a" * 40
    assert stamped["provenance"]["clifford_qc"]
    assert len(stamped["provenance"]["environment_sha256"]) == 64
    json.dumps(stamped)
