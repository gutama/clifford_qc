import json

from clifford_qc.reproducibility import (
    compare_json_records, execution_provenance, sampling_stream_mismatch,
    stamp_record,
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


def test_key_tolerances_widen_only_the_named_field():
    # A residue of two large energies keeps only a few significant digits, so
    # its own magnitude is the wrong error scale; a neighbouring field of the
    # same size that is *not* a difference must stay on the tight default.
    expected = {"error_millihartree": 0.0033014997669056356,
                "condition_number": 0.0033014997669056356}
    actual = {"error_millihartree": 0.0033014999125668965,
              "condition_number": 0.0033014999125668965}
    tight = compare_json_records(expected, actual, atol=1e-10, rtol=1e-10)
    assert len(tight) == 2

    widened = compare_json_records(
        expected, actual, atol=1e-10, rtol=1e-10,
        key_tolerances={"error_millihartree": (1e-10, 1e-8)},
    )
    assert len(widened) == 1
    assert "condition_number" in widened[0]


def test_key_tolerances_still_reject_a_scientifically_real_change():
    expected = {"error_millihartree": 0.0033014997669056356}
    actual = {"error_millihartree": 0.0033114997669056356}   # +1e-5 mHa
    assert compare_json_records(
        expected, actual, atol=1e-10, rtol=1e-10,
        key_tolerances={"error_millihartree": (1e-10, 1e-8)},
    )


def test_key_tolerances_default_leaves_comparison_unchanged():
    expected = {"a": 1.0, "b": {"c": 2.0}}
    assert compare_json_records(expected, {"a": 1.0, "b": {"c": 2.0}},
                                key_tolerances=None) == []
    assert compare_json_records(expected, {"a": 1.0, "b": {"c": 2.5}},
                                key_tolerances={"z": (1.0, 1.0)})


def test_sampling_stream_mismatch_names_the_environment_not_the_arithmetic():
    import importlib.metadata
    installed = importlib.metadata.version("numpy")
    matching = {"provenance": {"dependencies": {"numpy": installed}}}
    assert sampling_stream_mismatch(matching) == []

    # A record built under a different NumPy answers a different question --
    # different bundled LAPACK, possibly different draws -- rather than drifting,
    # so the diagnosis must say so rather than leave a reader widening a
    # tolerance against it.
    drifted = {"provenance": {"dependencies": {"numpy": "0.0.1-not-installed"}}}
    problems = sampling_stream_mismatch(drifted)
    assert len(problems) == 1
    assert "0.0.1-not-installed" in problems[0] and installed in problems[0]
    assert "not comparable across" in problems[0]


def test_sampling_stream_mismatch_is_silent_without_a_dependency_stamp():
    assert sampling_stream_mismatch({}) == []
    assert sampling_stream_mismatch({"provenance": {}}) == []
    assert sampling_stream_mismatch({"provenance": {"dependencies": {}}}) == []
    assert sampling_stream_mismatch("not a record") == []
    # An absent optional package must not be reported as a mismatch.
    assert sampling_stream_mismatch(
        {"provenance": {"dependencies": {"numpy": None}}}) == []
