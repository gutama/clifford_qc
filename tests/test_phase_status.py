"""Contracts for the machine-readable phase ledger and its drift gate."""

from __future__ import annotations

import copy
import json

from benchmarks import check_phase_status as status


def ledger() -> dict:
    return json.loads(status.LEDGER.read_text())


def set_summary(data: dict) -> None:
    fractions = [float(row["implementation_fraction"]) for row in data["numbered_phases"]]
    complete = sum(value == 1.0 for value in fractions)
    count = len(fractions)
    points = sum(fractions)
    data["summary"] = {
        "numbered_phase_count": count,
        "complete_phase_count": complete,
        "strict_complete_fraction": complete / count,
        "progress_weighted_points": points,
        "progress_weighted_fraction": points / count,
    }


def test_current_ledger_and_documents_are_valid():
    data = ledger()
    assert status.validate(data) == []
    assert status.check_docs(data) == []


def test_summary_arithmetic_drift_is_rejected():
    data = ledger()
    data["summary"]["progress_weighted_points"] += 1
    assert any(
        "summary.progress_weighted_points" in problem
        for problem in status.validate(data)
    )


def test_decimal_partial_fractions_use_tolerant_arithmetic():
    data = ledger()
    for phase_id, fraction in (("15", 0.1), ("16", 0.2)):
        row = next(row for row in data["numbered_phases"] if row["id"] == phase_id)
        row["status"] = "partial"
        row["implementation_fraction"] = fraction
    set_summary(data)
    data["summary"]["progress_weighted_points"] = 16.55
    data["summary"]["progress_weighted_fraction"] = 0.8275
    assert status.validate(data) == []


def test_absolute_evidence_path_is_rejected(tmp_path):
    evidence = tmp_path / "outside.txt"
    evidence.write_text("not repository evidence")
    data = ledger()
    data["numbered_phases"][0]["evidence"] = [str(evidence)]
    assert any("repository-relative" in problem for problem in status.validate(data))


def test_parent_evidence_path_is_rejected():
    data = ledger()
    data["numbered_phases"][0]["evidence"] = ["../outside.txt"]
    assert any("escapes the repository" in problem for problem in status.validate(data))


def test_grouped_core_row_is_derived_from_phase_rows():
    data = ledger()
    data["numbered_phases"][0]["status"] = "partial"
    data["numbered_phases"][0]["implementation_fraction"] = 0.5
    block = status.render_summary(data)
    assert "| Phases 0--14 | partial | 14.50 / 15 |" in block


def test_document_summary_drift_is_rejected(tmp_path, monkeypatch):
    data = ledger()
    block = status.render_summary(data)
    plan = tmp_path / "PLAN.md"
    readme = tmp_path / "README.md"
    plan.write_text(block)
    readme.write_text(block.replace("81.25%", "80.00%"))
    monkeypatch.setattr(status, "DOCS", (plan, readme))
    assert status.check_docs(data) == ["README.md: generated phase-status summary drifted"]


def test_stale_status_phrase_is_rejected(tmp_path, monkeypatch):
    data = ledger()
    block = status.render_summary(data)
    plan = tmp_path / "PLAN.md"
    readme = tmp_path / "README.md"
    plan.write_text(block)
    readme.write_text(block + "\nPhases 15–16 and 19 open\n")
    monkeypatch.setattr(status, "DOCS", (plan, readme))
    assert any("stale status phrase" in problem for problem in status.check_docs(data))
