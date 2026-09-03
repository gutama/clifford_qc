import json

import pytest

from clifford_qc import record_environment, reproducibility
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


def test_path_tolerances_keep_one_fixed_float_exact():
    expected = {
        "arms": {
            "fixed": {"median_overlap_threshold": 1e-10},
            "calibrated": {"median_overlap_threshold": 0.07},
        }
    }
    actual = {
        "arms": {
            "fixed": {"median_overlap_threshold": 0.0},
            "calibrated": {"median_overlap_threshold": 0.0700000005},
        }
    }
    problems = compare_json_records(
        expected,
        actual,
        atol=1e-9,
        rtol=1e-9,
        path_tolerances={
            "$.arms.fixed.median_overlap_threshold": (0.0, 0.0),
        },
    )
    assert len(problems) == 1
    assert "arms.fixed.median_overlap_threshold" in problems[0]


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
    # A record built *without* an optional package claims nothing about it.
    assert sampling_stream_mismatch(
        {"provenance": {"dependencies": {"numpy": None}}}) == []


def test_a_named_package_that_is_missing_here_is_as_disqualifying_as_a_bump():
    # The caller names a package because its absence moves the values, not
    # merely the speed, so silence would defer the diagnosis to whatever the
    # fallback path happens to produce.
    record = {"provenance": {"dependencies": {"nonexistent-package": "1.2.3"}}}
    problems = sampling_stream_mismatch(record, packages=("nonexistent-package",))
    assert len(problems) == 1
    assert "is not installed" in problems[0] and "1.2.3" in problems[0]


def test_stamping_is_refused_under_an_environment_no_record_declares(
        tmp_path, monkeypatch):
    # The producer boundary, not a later cross-record gate, is where an
    # unreproducible environment costs a rebuild instead of a retraction.
    (tmp_path / "committed.json").write_text(json.dumps(
        {"provenance": {"python": "3.12.4",
                        "dependencies": {"numpy": "9.9.9"}}}), encoding="utf-8")
    monkeypatch.delenv(record_environment.ALLOW_MIGRATION_ENV, raising=False)
    monkeypatch.setattr(record_environment, "DEFAULT_DATA", tmp_path)
    monkeypatch.setattr(reproducibility, "_environment_guarded", False)
    with pytest.raises(record_environment.UndeclaredEnvironment):
        stamp_record({"schema": "example.v1"})


def test_a_deliberate_migration_may_stamp_outside_the_declared_set(
        tmp_path, monkeypatch):
    (tmp_path / "committed.json").write_text(json.dumps(
        {"provenance": {"python": "3.12.4",
                        "dependencies": {"numpy": "9.9.9"}}}), encoding="utf-8")
    monkeypatch.delenv(record_environment.ALLOW_MIGRATION_ENV, raising=False)
    monkeypatch.setattr(record_environment, "DEFAULT_DATA", tmp_path)
    monkeypatch.setattr(reproducibility, "_environment_guarded", False)
    assert stamp_record({"schema": "example.v1"},
                        allow_environment_migration=True)["provenance"]
    monkeypatch.setenv(record_environment.ALLOW_MIGRATION_ENV, "1")
    assert stamp_record({"schema": "example.v1"})["provenance"]


def test_an_authorized_stamp_does_not_excuse_the_next_one(tmp_path, monkeypatch):
    # Memoizing the survey must not memoize the authorization with it.
    (tmp_path / "committed.json").write_text(json.dumps(
        {"provenance": {"python": "3.12.4",
                        "dependencies": {"numpy": "9.9.9"}}}), encoding="utf-8")
    monkeypatch.delenv(record_environment.ALLOW_MIGRATION_ENV, raising=False)
    monkeypatch.setattr(record_environment, "DEFAULT_DATA", tmp_path)
    monkeypatch.setattr(reproducibility, "_environment_guarded", False)
    stamp_record({"schema": "example.v1"}, allow_environment_migration=True)
    with pytest.raises(record_environment.UndeclaredEnvironment):
        stamp_record({"schema": "example.v1"})


def test_the_committed_set_is_surveyed_once_per_process(tmp_path, monkeypatch):
    # A JSONL producer stamps every row; the guard must not re-read the whole
    # record directory ten thousand times to say the same thing.
    surveys = []
    monkeypatch.delenv(record_environment.ALLOW_MIGRATION_ENV, raising=False)
    monkeypatch.setattr(record_environment, "DEFAULT_DATA", tmp_path)
    monkeypatch.setattr(reproducibility, "_environment_guarded", False)
    monkeypatch.setattr(record_environment, "guard",
                        lambda *args, **kwargs: surveys.append(1))
    for _ in range(5):
        stamp_record({"schema": "example.v1"})
    assert len(surveys) == 1
