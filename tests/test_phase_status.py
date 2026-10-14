"""Contracts for the machine-readable phase ledger and its drift gate."""

from __future__ import annotations

import copy
import json

import pytest

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
    before = data["summary"]["progress_weighted_points"]
    # Any two unfinished phases will do; the ledger's own progress moves which
    # ones are unstarted, so the fractions they give up are subtracted back.
    unfinished = [row for row in data["numbered_phases"] if row["status"] != "complete"]
    replaced = 0.0
    for row, fraction in zip(unfinished[:2], (0.1, 0.2), strict=True):
        replaced += row["implementation_fraction"]
        row["status"] = "partial"
        row["implementation_fraction"] = fraction
    set_summary(data)
    # Written as a decimal literal, as a person would type it into the ledger;
    # the gate must accept it whatever bits the float sum of 0.1 and 0.2 lands on.
    points = round(before - replaced + 0.3, 3)
    data["summary"]["progress_weighted_points"] = points
    data["summary"]["progress_weighted_fraction"] = round(points / 20, 6)
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


def test_missing_decision_record_is_rejected():
    data = ledger()
    row = next(row for row in data["numbered_phases"] if row["id"] == "16")
    row["decision_records"] = ["benchmarks/reference_results/not_a_record.json"]
    assert any(
        "phase 16 decision record: missing evidence path" in problem
        for problem in status.validate(data)
    )


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
    plan.write_text((status.ROOT / "PLAN.md").read_text())
    weighted = f"{100 * data['summary']['progress_weighted_fraction']:.2f}%"
    assert weighted in block
    readme.write_text(block.replace(weighted, "12.34%"))
    monkeypatch.setattr(status, "DOCS", (plan, readme))
    assert status.check_docs(data) == ["README.md: generated phase-status summary drifted"]


def test_stale_status_phrase_is_rejected(tmp_path, monkeypatch):
    data = ledger()
    block = status.render_summary(data)
    plan = tmp_path / "PLAN.md"
    readme = tmp_path / "README.md"
    plan.write_text((status.ROOT / "PLAN.md").read_text())
    readme.write_text(block + "\nPhases 15–16 and 19 open\n")
    monkeypatch.setattr(status, "DOCS", (plan, readme))
    assert any("stale status phrase" in problem for problem in status.check_docs(data))


def detailed_docs(tmp_path, monkeypatch, data, transform):
    """Mutate real narrative rows while keeping both generated summaries valid."""
    original = (status.ROOT / "PLAN.md").read_text()
    block = status.summary_block(original)
    assert block is not None
    text = original.replace(block, status.render_summary(data))
    plan = tmp_path / "PLAN.md"
    readme = tmp_path / "README.md"
    plan.write_text(transform(text))
    readme.write_text(status.render_summary(data))
    monkeypatch.setattr(status, "DOCS", (plan, readme))
    assert status.summary_block(plan.read_text()) == status.render_summary(data)


def replace_detailed_row(text, phase_id, heading=None, suffix=""):
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(f"| {phase_id} |"):
            before, current, after = line.split("**", 2)
            lines[index] = before + "**" + (heading or current) + "**" + suffix + after
            break
    else:
        raise AssertionError(f"fixture has no detailed phase {phase_id}")
    return "".join(lines)


@pytest.mark.parametrize("phase_id,heading", [
    ("15", "partially implemented (25%)"),
    ("16", "partially implemented (75%)"),
    ("17", "partially implemented (75%)"),
    ("18", "partial (75%)"),
    ("19", "partial (75%)"),
])
def test_detailed_status_drift_is_rejected_with_valid_summaries(
    tmp_path, monkeypatch, phase_id, heading,
):
    data = ledger()
    detailed_docs(tmp_path, monkeypatch, data,
                  lambda text: replace_detailed_row(text, phase_id, heading))
    problems = status.check_docs(data)
    assert any(f"detailed phase {phase_id}" in problem for problem in problems)
    assert not any("generated phase-status summary" in problem for problem in problems)


@pytest.mark.parametrize("heading", [
    "partially implemented (87.5%; rounded to 88% above)",
    "partial (87.5%)",
    "partial (88%)",
])
def test_detailed_fraction_accepts_exact_or_correctly_rounded_display(
    tmp_path, monkeypatch, heading,
):
    data = ledger()
    row = next(row for row in data["numbered_phases"] if row["id"] == "17")
    row["status"] = "partial"
    row["implementation_fraction"] = 0.875
    set_summary(data)
    detailed_docs(tmp_path, monkeypatch, data,
                  lambda text: replace_detailed_row(text, "17", heading))
    assert status.check_docs(data) == []


@pytest.mark.parametrize("fraction,display,accepted", [
    (1 / 3, "33.3", True), (2 / 3, "66.7", True),
    (1 / 3, "33.33", True), (2 / 3, "66.67", True),
    (1 / 3, "33.4", False), (2 / 3, "66.6", False),
])
def test_detailed_percentage_respects_displayed_decimal_precision(
    tmp_path, monkeypatch, fraction, display, accepted,
):
    data = ledger()
    row = next(row for row in data["numbered_phases"] if row["id"] == "17")
    row["status"] = "partial"
    row["implementation_fraction"] = fraction
    set_summary(data)
    detailed_docs(tmp_path, monkeypatch, data,
                  lambda text: replace_detailed_row(text, "17", f"partial ({display}%)"))
    assert (status.check_docs(data) == []) == accepted


@pytest.mark.parametrize("heading", [
    "partial (87%)",
    "partially implemented (87.5%; rounded to 87% above)",
    "partial",
    "pending (88%)",
])
def test_detailed_rounding_or_malformed_status_is_rejected(
    tmp_path, monkeypatch, heading,
):
    data = ledger()
    row = next(row for row in data["numbered_phases"] if row["id"] == "17")
    row["status"] = "partial"
    row["implementation_fraction"] = 0.875
    set_summary(data)
    detailed_docs(tmp_path, monkeypatch, data,
                  lambda text: replace_detailed_row(text, "17", heading))
    assert any("detailed phase 17" in problem for problem in status.check_docs(data))


@pytest.mark.parametrize("phase_id", ["15", "19"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_detailed_phase_row_is_rejected(
    tmp_path, monkeypatch, phase_id, duplicate,
):
    def change(text):
        line = next(line for line in text.splitlines(keepends=True)
                    if line.startswith(f"| {phase_id} |"))
        return text.replace(line, line * 2 if duplicate else "", 1)

    detailed_docs(tmp_path, monkeypatch, ledger(), change)
    assert any("detailed phase " + phase_id in problem
               for problem in status.check_docs(ledger()))


@pytest.mark.parametrize("replacement", ["", "| phase | subject | status |\n" * 2])
def test_missing_or_duplicate_detailed_table_is_rejected(tmp_path, monkeypatch, replacement):
    detailed_docs(tmp_path, monkeypatch, ledger(),
                  lambda text: text.replace("| phase | subject | status |\n", replacement))
    assert any("detailed phase table" in problem for problem in status.check_docs(ledger()))


def test_phase19_completed_scope_cannot_deny_its_shipped_comparison(tmp_path, monkeypatch):
    detailed_docs(tmp_path, monkeypatch, ledger(), lambda text: replace_detailed_row(
        text, "19", suffix="; No sampled comparison, producer or record ships yet."))
    assert any("detailed phase 19" in problem for problem in status.check_docs(ledger()))


def test_phase19_pending_lever2_and_other_historical_percentages_are_allowed(
    tmp_path, monkeypatch,
):
    detailed_docs(tmp_path, monkeypatch, ledger(), lambda text: replace_detailed_row(
        text, "19", suffix="; Historical pilot: 75%. Lever 2: no sampled comparison ships yet."))
    assert status.check_docs(ledger()) == []


def test_detailed_status_follows_updated_ledger_instead_of_frozen_percentages(
    tmp_path, monkeypatch,
):
    data = copy.deepcopy(ledger())
    row = next(row for row in data["numbered_phases"] if row["id"] == "19")
    row["status"] = "partial"
    row["implementation_fraction"] = 0.75
    set_summary(data)
    detailed_docs(tmp_path, monkeypatch, data, lambda text: replace_detailed_row(
        text, "19", "partial (75%)",
        suffix="; No sampled comparison, producer or record ships yet."))
    assert status.check_docs(data) == []
