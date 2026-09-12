"""Validate the machine-readable research-phase ledger and its doc summaries.

``PHASE_STATUS.json`` is the single source for implementation progress.  This
gate validates its vocabulary, 0--19 denominator, score arithmetic and evidence
paths, then requires ``PLAN.md`` and ``README.md`` to carry the same generated
summary block.

    python benchmarks/check_phase_status.py

Exits nonzero on any drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "PHASE_STATUS.json"
DOCS = (ROOT / "PLAN.md", ROOT / "README.md")
START = "<!-- PHASE-STATUS-SUMMARY:START -->"
END = "<!-- PHASE-STATUS-SUMMARY:END -->"
EXPECTED_IDS = tuple(str(i) for i in range(20))
NUMBERED_STATUSES = {"complete", "partial", "open", "proposed"}
ADJUNCT_STATUSES = NUMBERED_STATUSES | {
    "retired",
    "closed_negative",
    "not_authorized",
}


def load_ledger() -> dict:
    return json.loads(LEDGER.read_text())


def validate(data: dict) -> list[str]:
    problems: list[str] = []
    if data.get("schema") != "clifford_qc.phase_status.v1":
        problems.append("unexpected phase-ledger schema")

    phases = data.get("numbered_phases")
    if not isinstance(phases, list):
        return problems + ["numbered_phases must be a list"]

    ids = tuple(row.get("id") for row in phases)
    if ids != EXPECTED_IDS:
        problems.append(f"numbered phase ids must be 0..19 in order, got {ids}")
    declared_ids = tuple(data.get("score_scope", {}).get("phase_ids", ()))
    if declared_ids != EXPECTED_IDS:
        problems.append("score_scope.phase_ids must be exactly 0..19 in order")

    seen: set[str] = set()
    fractions: list[float] = []
    for row in phases:
        phase_id = row.get("id")
        if phase_id in seen:
            problems.append(f"duplicate numbered phase id {phase_id!r}")
        seen.add(phase_id)
        status = row.get("status")
        if status not in NUMBERED_STATUSES:
            problems.append(f"phase {phase_id}: invalid numbered status {status!r}")
        fraction = row.get("implementation_fraction")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            problems.append(f"phase {phase_id}: implementation_fraction is not numeric")
            continue
        fraction = float(fraction)
        fractions.append(fraction)
        if not 0.0 <= fraction <= 1.0:
            problems.append(f"phase {phase_id}: implementation_fraction outside [0,1]")
        if status == "complete" and fraction != 1.0:
            problems.append(f"phase {phase_id}: complete requires fraction 1")
        if status in {"open", "proposed"} and fraction != 0.0:
            problems.append(f"phase {phase_id}: {status} requires fraction 0")
        if status == "partial" and not 0.0 < fraction < 1.0:
            problems.append(f"phase {phase_id}: partial requires a fraction inside (0,1)")
        for evidence in row.get("evidence", ()):
            if not (ROOT / evidence).is_file():
                problems.append(f"phase {phase_id}: missing evidence path {evidence}")

    adjunct_ids: set[str] = set()
    for row in data.get("adjunct_programs", ()):
        phase_id = row.get("id")
        if phase_id in adjunct_ids:
            problems.append(f"duplicate adjunct phase id {phase_id!r}")
        adjunct_ids.add(phase_id)
        status = row.get("status")
        if status not in ADJUNCT_STATUSES:
            problems.append(f"adjunct {phase_id}: invalid status {status!r}")
        for evidence in row.get("evidence", ()):
            if not (ROOT / evidence).is_file():
                problems.append(f"adjunct {phase_id}: missing evidence path {evidence}")

    summary = data.get("summary", {})
    complete = sum(value == 1.0 for value in fractions)
    points = sum(fractions)
    expected = {
        "numbered_phase_count": len(EXPECTED_IDS),
        "complete_phase_count": complete,
        "strict_complete_fraction": complete / len(EXPECTED_IDS),
        "progress_weighted_points": points,
        "progress_weighted_fraction": points / len(EXPECTED_IDS),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            problems.append(
                f"summary.{key} is {summary.get(key)!r}; computed value is {value!r}"
            )
    return problems


def render_summary(data: dict) -> str:
    phases = {row["id"]: row for row in data["numbered_phases"]}
    summary = data["summary"]
    rows = [
        START,
        "| numbered phase scope | lifecycle | implementation |",
        "|---|---|---:|",
        "| Phases 0--14 | complete | 15 / 15 |",
    ]
    for phase_id in ("15", "16", "17", "18", "19"):
        row = phases[phase_id]
        percent = 100.0 * row["implementation_fraction"]
        rows.append(
            f"| Phase {phase_id}: {row['title']} | {row['status']} | {percent:.0f}% |"
        )
    rows.extend(
        [
            "",
            (
                "Strict complete-phase score: "
                f"**{summary['complete_phase_count']} / "
                f"{summary['numbered_phase_count']} = "
                f"{100 * summary['strict_complete_fraction']:.2f}%**."
            ),
            (
                "Progress-weighted score: "
                f"**{summary['progress_weighted_points']:.2f} / "
                f"{summary['numbered_phase_count']} = "
                f"{100 * summary['progress_weighted_fraction']:.2f}%**."
            ),
            (
                "Retired and conditional adjunct phases are tracked separately and do not "
                "change this denominator. Source: `PHASE_STATUS.json`; validate with "
                "`python benchmarks/check_phase_status.py`."
            ),
            END,
        ]
    )
    return "\n".join(rows)


def summary_block(text: str) -> str | None:
    if text.count(START) != 1 or text.count(END) != 1:
        return None
    start = text.index(START)
    end = text.index(END, start) + len(END)
    return text[start:end]


def check_docs(data: dict) -> list[str]:
    problems: list[str] = []
    expected = render_summary(data)
    forbidden = {
        "README.md": (
            "13--18 open",
            "13–18 open",
            "Q1--Q13",
            "Q1–Q13",
            "fifteen-paper",
        ),
        "PLAN.md": (
            "Preregistered, not run.",
            "Still abstaining at the accuracy-matched tier.",
        ),
    }
    for path in DOCS:
        text = path.read_text()
        actual = summary_block(text)
        if actual is None:
            problems.append(f"{path.name}: missing or duplicate phase-status summary markers")
        elif actual != expected:
            problems.append(f"{path.name}: generated phase-status summary drifted")
        for phrase in forbidden[path.name]:
            if phrase in text:
                problems.append(f"{path.name}: stale status phrase remains: {phrase!r}")
    return problems


def main() -> int:
    try:
        data = load_ledger()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"phase status check failed: {exc}", file=sys.stderr)
        return 1
    problems = validate(data)
    if not problems:
        problems.extend(check_docs(data))
    if problems:
        print("Phase status drift:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    summary = data["summary"]
    print(
        "OK: phase ledger and documentation agree; "
        f"strict={100 * summary['strict_complete_fraction']:.2f}%, "
        f"weighted={100 * summary['progress_weighted_fraction']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
