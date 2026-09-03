"""Contract checks for the environment gate that fronts the record gates."""

import importlib.metadata
import json
import platform

import pytest

import benchmarks.check_record_environment as gate
import clifford_qc.record_environment as gate_environment
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


def test_a_named_record_selects_its_own_reproducible_environment(tmp_path):
    _write(tmp_path, "old.json", "3.11.15", {"numpy": "2.4.6"})
    _write(tmp_path, "new.json", "3.12.4", {"numpy": "2.5.2"})
    declarations, problems = survey(tmp_path, ["new.json"])
    versions = agreed(declarations, problems)
    assert problems == []
    assert versions == {"python": "3.12", "numpy": "2.5.2"}


def test_a_bad_record_selector_is_reported_without_reading_outside_data(tmp_path):
    declarations, problems = survey(tmp_path, ["../outside.json", "missing.json"])
    assert declarations == {}
    assert len(problems) == 2
    assert "basenames" in problems[0]
    assert "does not exist" in problems[1]


def test_the_cli_can_emit_one_named_records_interpreter(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "old.json", "3.11.15", {"numpy": "2.4.6"})
    _write(tmp_path, "new.json", "3.12.4", {"numpy": "2.5.2"})
    monkeypatch.setattr(gate, "DATA", tmp_path)
    assert main(["--python", "--record", "new.json"]) == 0
    assert capsys.readouterr().out == "3.12\n"


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


# The producer-side half of the same contract: check_record_environment reports
# a split that already exists, and the guard below is what stops one starting.


def _declarations(**by_package):
    return {name: dict(versions) for name, versions in by_package.items()}


def test_a_version_every_record_declares_is_not_reported(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.4")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.5.2")
    declarations = _declarations(python={"3.12": ["a.json"]},
                                 numpy={"2.5.2": ["a.json"]})
    assert gate_environment.undeclared(declarations) == []


def test_a_version_no_record_declares_is_reported_with_the_records(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.4.6")
    declarations = _declarations(python={"3.12": ["a.json", "b.json"]},
                                 numpy={"2.5.2": ["a.json", "b.json"]})
    reports = gate_environment.undeclared(declarations)
    assert len(reports) == 2
    assert "python 3.11 is declared by no committed record" in reports[1]
    assert "3.12 (a.json, b.json)" in reports[1]
    assert "numpy 2.4.6 is declared by no committed record" in reports[0]


def test_only_a_few_records_are_named_before_the_count(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    records = [f"r{index}.json" for index in range(7)]
    reports = gate_environment.undeclared(_declarations(python={"3.12": records}))
    assert "r0.json, r1.json, r2.json, and 4 more" in reports[0]


def test_either_side_of_an_existing_split_can_still_rebuild(monkeypatch):
    # The repair for a split is rebuilding the odd records out, and that run
    # happens under one of the versions in the split. Demanding agreement here
    # would refuse the only run that ends it.
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    split = _declarations(python={"3.11": ["old.json"], "3.12": ["new.json"]})
    assert gate_environment.undeclared(split) == []
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.4")
    assert gate_environment.undeclared(split) == []


def test_a_third_environment_is_refused_even_during_a_split(monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.13.2")
    split = _declarations(python={"3.11": ["old.json"], "3.12": ["new.json"]})
    reports = gate_environment.undeclared(split)
    assert len(reports) == 1
    assert "3.11 (old.json); 3.12 (new.json)" in reports[0]


def test_an_uninstalled_optional_package_is_not_undeclared(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.4")
    declarations = _declarations(python={"3.12": ["a.json"]},
                                 pyscf={"2.14.0": ["a.json"]})
    assert gate_environment.undeclared(declarations) == []


def test_the_guard_raises_and_names_how_to_proceed(tmp_path, monkeypatch):
    _write(tmp_path, "a.json", "3.12.4", {"numpy": "2.5.2"})
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.4.6")
    with pytest.raises(gate_environment.UndeclaredEnvironment) as raised:
        gate_environment.guard(tmp_path)
    message = str(raised.value)
    assert "numpy 2.4.6" in message and "python 3.11" in message
    assert gate_environment.ALLOW_MIGRATION_ENV in message
    assert "--constraints" in message


def test_the_guard_passes_the_environment_that_produced_the_records(
        tmp_path, monkeypatch):
    _write(tmp_path, "a.json", "3.12.4", {"numpy": "2.5.2"})
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.4")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.5.2")
    gate_environment.guard(tmp_path)


def test_the_guard_stands_down_where_there_are_no_records(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    # An installed wheel has no committed record set to disagree with.
    gate_environment.guard(tmp_path / "absent")
    # Neither has an empty one, or a set that stamps nothing.
    gate_environment.guard(tmp_path)
    _write(tmp_path, "legacy.json", energies=[1.0])
    gate_environment.guard(tmp_path)


def test_an_unreadable_record_does_not_stall_the_bench(tmp_path, monkeypatch):
    (tmp_path / "broken.json").write_text("{oops", encoding="utf-8")
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    gate_environment.guard(tmp_path)


@pytest.mark.parametrize("value,allowed", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("  ", False),
])
def test_the_migration_variable_is_read_strictly(value, allowed):
    environ = {gate_environment.ALLOW_MIGRATION_ENV: value}
    assert gate_environment.migration_allowed(environ) is allowed


def test_migration_is_not_allowed_by_default(monkeypatch):
    monkeypatch.delenv(gate_environment.ALLOW_MIGRATION_ENV, raising=False)
    assert gate_environment.migration_allowed() is False


# JSONL records stamp their rows exactly as JSON records stamp their document.
# Surveying only *.json was a silent under-count, and a record it could not see
# could neither fail the gate nor be counted by the producer guard.


def _write_rows(directory, name, rows):
    lines = [json.dumps(row) for row in rows]
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(python=None, dependencies=None, **extra):
    provenance = {}
    if python is not None:
        provenance["python"] = python
    if dependencies is not None:
        provenance["dependencies"] = dependencies
    row = dict(extra)
    if provenance:
        row["provenance"] = provenance
    return row


def _empty_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_environment, "PENDING", tmp_path / "absent.json")


def test_a_jsonl_records_rows_are_surveyed(tmp_path, monkeypatch):
    _empty_manifest(tmp_path, monkeypatch)
    _write_rows(tmp_path, "ladder.jsonl", [
        _row("3.12.4", {"numpy": "2.5.2"}, energy=-1.0),
        _row("3.12.4", {"numpy": "2.5.2"}, energy=-2.0),
    ])
    versions, problems = _agreed(tmp_path)
    assert problems == []
    assert versions == {"python": "3.12", "numpy": "2.5.2"}


def test_a_jsonl_record_declares_once_however_many_rows_it_has(tmp_path,
                                                               monkeypatch):
    _empty_manifest(tmp_path, monkeypatch)
    _write_rows(tmp_path, "ladder.jsonl",
                [_row("3.12.4", {"numpy": "2.5.2"}) for _ in range(50)])
    declarations, _ = survey(tmp_path)
    assert declarations["numpy"]["2.5.2"] == ["ladder.jsonl"]


def test_a_jsonl_record_off_the_agreed_environment_is_reported(tmp_path,
                                                               monkeypatch):
    # The case the *.json glob could not see at all.
    _empty_manifest(tmp_path, monkeypatch)
    _write(tmp_path, "current.json", "3.12.4", {"numpy": "2.5.2"})
    _write_rows(tmp_path, "ladder.jsonl", [_row("3.11.15", {"numpy": "2.4.6"})])
    versions, problems = _agreed(tmp_path)
    assert "numpy" not in versions and "python" not in versions
    assert any("ladder.jsonl" in problem for problem in problems)


def test_unstamped_and_malformed_rows_do_not_abort_the_survey(tmp_path,
                                                              monkeypatch):
    _empty_manifest(tmp_path, monkeypatch)
    _write_rows(tmp_path, "legacy.jsonl", [_row(energy=-1.0), _row(energy=-2.0)])
    (tmp_path / "broken.jsonl").write_text('{"provenance": {}\n', encoding="utf-8")
    _write(tmp_path, "good.json", "3.12.4", {"numpy": "2.5.2"})
    versions, problems = _agreed(tmp_path)
    assert versions == {"python": "3.12", "numpy": "2.5.2"}
    assert len(problems) == 1 and problems[0].startswith("broken.jsonl: unreadable")


def test_a_pending_record_is_passed_over_but_still_readable(tmp_path, monkeypatch):
    _write(tmp_path, "current.json", "3.12.4", {"numpy": "2.5.2"})
    _write_rows(tmp_path, "old.jsonl", [_row("3.11.15", {"numpy": "2.4.6"})])
    manifest = tmp_path / "pending.json"
    manifest.write_text(json.dumps({"records": {"old.jsonl": {"declares": {}}}}),
                        encoding="utf-8")
    monkeypatch.setattr(gate_environment, "PENDING", manifest)

    versions, problems = _agreed(tmp_path)
    assert problems == [] and versions == {"python": "3.12", "numpy": "2.5.2"}

    # Passed over is not hidden: the gate lists it by asking for it.
    declarations, _ = survey(tmp_path, include_pending=True)
    assert declarations["numpy"]["2.4.6"] == ["old.jsonl"]
    assert set(gate_environment.pending(manifest)) == {"old.jsonl"}


def test_a_missing_manifest_exempts_nothing(tmp_path, monkeypatch):
    # Losing the manifest must not silently excuse the records it named.
    monkeypatch.setattr(gate_environment, "PENDING", tmp_path / "gone.json")
    assert gate_environment.pending() == {}
    _write(tmp_path, "current.json", "3.12.4", {"numpy": "2.5.2"})
    _write_rows(tmp_path, "old.jsonl", [_row("3.11.15", {"numpy": "2.4.6"})])
    _, problems = _agreed(tmp_path)
    assert len(problems) == 2


def test_the_guard_does_not_count_a_pending_records_declaration(tmp_path,
                                                                monkeypatch):
    # The property the manifest exists for: a superseded record must not
    # re-authorize the environment the guard is there to refuse.
    _write(tmp_path, "current.json", "3.12.4", {"numpy": "2.5.2"})
    _write_rows(tmp_path, "old.jsonl", [_row("3.11.15", {"numpy": "2.4.6"})])
    manifest = tmp_path / "pending.json"
    manifest.write_text(json.dumps({"records": {"old.jsonl": {}}}), encoding="utf-8")
    monkeypatch.setattr(gate_environment, "PENDING", manifest)
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.4.6")
    with pytest.raises(gate_environment.UndeclaredEnvironment):
        gate_environment.guard(tmp_path)


def test_the_outstanding_record_is_named_on_stderr_not_stdout(tmp_path,
                                                              monkeypatch, capsys):
    _write(tmp_path, "a.json", "3.12.4", {"numpy": "2.5.2"})
    _write_rows(tmp_path, "old.jsonl", [_row("3.11.15", {"numpy": "2.4.6"})])
    # The manifest says nothing about versions, so the note can only be right
    # by reading them back out of the record.
    manifest = tmp_path / "pending.json"
    manifest.write_text(json.dumps({"records": {"old.jsonl": {}}}), encoding="utf-8")
    monkeypatch.setattr(gate_environment, "PENDING", manifest)
    monkeypatch.setattr(gate, "DATA", tmp_path)
    assert main(["--python"]) == 0
    captured = capsys.readouterr()
    # stdout is redirected into the file the install reads: it stays the pin.
    assert captured.out == "3.12\n"
    assert "old.jsonl" in captured.err
    assert "numpy 2.4.6" in captured.err and "python 3.11" in captured.err


def test_an_outstanding_record_that_straddles_versions_reports_both(tmp_path,
                                                                    monkeypatch):
    _write_rows(tmp_path, "old.jsonl", [_row("3.11.15", {"numpy": "2.4.6"}),
                                        _row("3.11.15", {"numpy": "2.4.5"})])
    manifest = tmp_path / "pending.json"
    manifest.write_text(json.dumps({"records": {"old.jsonl": {}}}), encoding="utf-8")
    monkeypatch.setattr(gate_environment, "PENDING", manifest)
    assert gate.outstanding(tmp_path) == {
        "old.jsonl": {"numpy": ["2.4.5", "2.4.6"], "python": ["3.11"]}}
