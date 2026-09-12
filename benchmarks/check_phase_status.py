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
import math
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


def check_evidence_path(owner: str, evidence: object, problems: list[str]) -> None:
    if not isinstance(evidence, str) or not evidence:
        problems.append(f"{owner}: evidence path must be a nonempty string")
        return
    relative = Path(evidence)
    if relative.is_absolute():
        problems.append(f"{owner}: evidence path must be repository-relative: {evidence}")
        return
    root = ROOT.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        problems.append(f"{owner}: evidence path escapes the repository: {evidence}")
        return
    if not resolved.is_file():
        problems.append(f"{owner}: missing evidence path {evidence}")


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
            check_evidence_path(f"phase {phase_id}", evidence, problems)

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
            check_evidence_path(f"adjunct {phase_id}", evidence, problems)

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
        actual = summary.get(key)
        equal = actual == value
        if isinstance(value, float):
            equal = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(float(actual), value, rel_tol=1e-12, abs_tol=1e-12)
            )
        if not equal:
            problems.append(
                f"summary.{key} is {actual!r}; computed value is {value!r}"
            )
    return problems


def format_points(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def render_summary(data: dict) -> str:
    phases = {row["id"]: row for row in data["numbered_phases"]}
    summary = data["summary"]
    core = [phases[str(i)] for i in range(15)]
    core_points = sum(float(row["implementation_fraction"]) for row in core)
    if all(row["status"] == "complete" for row in core):
        core_status = "complete"
    elif core_points == 0.0:
        core_status = "open"
    else:
        core_status = "partial"
    rows = [
        START,
        "| numbered phase scope | lifecycle | implementation |",
        "|---|---|---:|",
        (
            f"| Phases 0--14 | {core_status} | "
            f"{format_points(core_points)} / {len(core)} |"
        ),
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
            "15–16 and 19 open",
            "Q1--Q13",
            "Q1–Q13",
            "fifteen-paper",
        ),
        "PLAN.md": (
            "Phases 0–18 of §5",
            "Preregistered, not run.",
            "Still abstaining at the accuracy-matched tier.",
            "preregistered but carries no result",
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
