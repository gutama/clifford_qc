"""Contract checks for the environment gate that fronts the record gates."""

import importlib.metadata
import json
import platform

import pytest

import benchmarks.check_record_environment as gate
from benchmarks.check_record_environment import (
    agreed,
    constraints,
    main,
    survey,
    verify,
)


def _write(directory, name, python=None, dependencies=None, **extra):
    provenance = {}
    if python is not None:
        provenance["python"] = python
    if dependencies is not None:
        provenance["dependencies"] = dependencies
    record = dict(extra)
    if provenance:
        record["provenance"] = provenance
    (directory / name).write_text(json.dumps(record), encoding="utf-8")


def _agreed(directory):
    declarations, problems = survey(directory)
    return agreed(declarations, problems), problems


def test_records_built_together_agree_on_every_version(tmp_path):
    _write(tmp_path, "a.json", "3.11.15", {"numpy": "2.4.6", "scipy": "1.17.1"})
    _write(tmp_path, "b.json", "3.11.15", {"numpy": "2.4.6", "scipy": "1.17.1"})
    versions, problems = _agreed(tmp_path)
    assert problems == []
    assert versions == {"python": "3.11", "numpy": "2.4.6", "scipy": "1.17.1"}


def test_a_record_rebuilt_against_a_newer_library_is_named_with_its_siblings(tmp_path):
    _write(tmp_path, "old.json", "3.11.15", {"numpy": "2.4.6"})
    _write(tmp_path, "peer.json", "3.11.15", {"numpy": "2.4.6"})
    _write(tmp_path, "rebuilt.json", "3.11.15", {"numpy": "2.5.2"})
    versions, problems = _agreed(tmp_path)
    # No pin at all: a majority vote would silently bless three records that
    # cannot all be reproduced in one environment.
    assert "numpy" not in versions
    assert len(problems) == 1
    assert "2.4.6 (old.json, peer.json)" in problems[0]
    assert "2.5.2 (rebuilt.json)" in problems[0]


def test_the_interpreter_is_matched_on_the_minor_version_only(tmp_path):
    # Patch releases do not move floating-point results, so records built weeks
    # apart under 3.11.9 and 3.11.15 are still one reproducible environment.
    _write(tmp_path, "a.json", "3.11.9", {"numpy": "2.4.6"})
    _write(tmp_path, "b.json", "3.11.15", {"numpy": "2.4.6"})
    versions, problems = _agreed(tmp_path)
    assert problems == []
    assert versions["python"] == "3.11"


def test_a_different_minor_version_is_still_a_disagreement(tmp_path):
    _write(tmp_path, "a.json", "3.11.15", {"numpy": "2.4.6"})
    _write(tmp_path, "b.json", "3.12.1", {"numpy": "2.4.6"})
    versions, problems = _agreed(tmp_path)
    assert "python" not in versions
    assert len(problems) == 1
    assert "3.11" in problems[0] and "3.12" in problems[0]


def test_a_null_dependency_is_absence_not_a_competing_claim(tmp_path):
    _write(tmp_path, "with.json", "3.11.15", {"numpy": "2.4.6", "pyscf": "2.14.0"})
    _write(tmp_path, "without.json", "3.11.15", {"numpy": "2.4.6", "pyscf": None})
    versions, problems = _agreed(tmp_path)
    assert problems == []
    assert versions["pyscf"] == "2.14.0"


def test_an_unstamped_record_claims_nothing_rather_than_failing(tmp_path):
    _write(tmp_path, "stamped.json", "3.11.15", {"numpy": "2.4.6"})
    _write(tmp_path, "legacy.json", energies=[1.0, 2.0])
    versions, problems = _agreed(tmp_path)
    assert problems == []
    assert versions == {"python": "3.11", "numpy": "2.4.6"}


def test_an_unreadable_record_is_reported_and_does_not_abort_the_survey(tmp_path):
    (tmp_path / "broken.json").write_text("{oops", encoding="utf-8")
    _write(tmp_path, "good.json", "3.11.15", {"numpy": "2.4.6"})
    versions, problems = _agreed(tmp_path)
    assert versions == {"python": "3.11", "numpy": "2.4.6"}
    assert len(problems) == 1
    assert problems[0].startswith("broken.json: unreadable")


def test_verify_flags_an_installed_version_the_records_were_not_built_under(
        monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.5.2")
    problems: list[str] = []
    verify({"numpy": "2.4.6"}, problems)
    assert len(problems) == 1
    assert "numpy 2.5.2" in problems[0] and "2.4.6" in problems[0]


def test_verify_flags_the_running_interpreter_too(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.4")
    problems: list[str] = []
    verify({"python": "3.11"}, problems)
    assert len(problems) == 1
    assert "python 3.12" in problems[0]


def test_verify_accepts_the_environment_that_produced_the_records(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.4.6")
    problems: list[str] = []
    verify({"python": "3.11", "numpy": "2.4.6"}, problems)
    assert problems == []


def test_an_uninstalled_optional_package_is_not_a_mismatch(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    problems: list[str] = []
    verify({"pyscf": "2.14.0"}, problems)
    assert problems == []


def test_constraints_pin_the_libraries_and_leave_python_to_the_runner():
    text = constraints({"python": "3.11", "numpy": "2.4.6", "scipy": "1.17.1"})
    pins = [line for line in text.splitlines() if not line.startswith("#")]
    assert pins == ["numpy==2.4.6", "scipy==1.17.1"]
    assert "# python==3.11" in text
    assert text.endswith("\n")


def test_the_emitted_interpreter_is_the_one_the_records_name(tmp_path,
                                                             monkeypatch, capsys):
    _write(tmp_path, "a.json", "3.11.15", {"numpy": "2.4.6"})
    monkeypatch.setattr(gate, "DATA", tmp_path)
    assert main(["--python"]) == 0
    assert capsys.readouterr().out == "3.11\n"


@pytest.mark.parametrize("mode", ["--python", "--constraints"])
def test_a_disagreement_stops_the_build_before_anything_is_installed(
        tmp_path, monkeypatch, capsys, mode):
    _write(tmp_path, "a.json", "3.11.15", {"numpy": "2.4.6"})
    _write(tmp_path, "b.json", "3.11.15", {"numpy": "2.5.2"})
    monkeypatch.setattr(gate, "DATA", tmp_path)
    assert main([mode]) == 1
    captured = capsys.readouterr()
    # Nothing on stdout: the caller redirects it into a file the install reads.
    assert captured.out == ""
    assert "numpy: the committed records disagree" in captured.err
