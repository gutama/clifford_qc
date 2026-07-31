"""Check the reported Krylov widths and their pencil certificates.

The raw zero-threshold support is not stable across chemistry runs because
last-bit SCF differences seed different round-off terms in H^17.  The reported
cutoff is stable, but a stable count alone is insufficient: the truncated
moments must also reproduce the unpruned generalized eigenproblem.  This gate
checks both properties.

    python benchmarks/check_krylov_width.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "reference_results" / "krylov_width.json"


def main() -> int:
    committed = json.loads(RECORD.read_text(encoding="utf-8"))
    if committed.get("schema") != "clifford_qc.krylov_width.v2":
        print("FAIL committed Krylov record is not the certified v2 schema")
        return 1

    expected = {row["rung"]: row for row in committed["rows"]}
    sys.path.insert(0, str(ROOT))
    from run_krylov_width import build_record

    fresh = build_record()
    problems: list[str] = []
    if fresh.get("schema") != committed["schema"]:
        problems.append(
            f"schema moved: {committed['schema']} -> {fresh.get('schema')}")
    if fresh.get("reported_threshold") != committed["reported_threshold"]:
        problems.append(
            f"reported threshold moved: {committed['reported_threshold']} "
            f"-> {fresh.get('reported_threshold')}")
    if fresh.get("certificate") != committed.get("certificate"):
        problems.append("certificate policy differs from the committed record")

    fresh_rows = {row["rung"]: row for row in fresh["rows"]}
    skipped = {row["rung"] for row in fresh.get("skipped", ())}
    for rung, expected_row in expected.items():
        if rung in skipped:
            continue
        if rung not in fresh_rows:
            problems.append(f"{rung}: in the record, not produced by a fresh run")
            continue
        row = fresh_rows[rung]
        if row["word_universe"] != expected_row["word_universe"]:
            problems.append(
                f"{rung}: reported W is {row['word_universe']}, "
                f"record says {expected_row['word_universe']}")
        if not row.get("word_universe_certificate_passed", False):
            problems.append(f"{rung}: fresh reported width is uncertified")

    for rung in fresh_rows:
        if rung not in expected:
            problems.append(f"{rung}: present in a fresh run, absent from the record")

    for problem in problems:
        print(f"FAIL {problem}")
    checked = len(set(fresh_rows) & set(expected))
    skipped_count = len(set(expected) & skipped)
    print(f"{len(problems)} problem(s)" if problems else
          f"OK ({checked} rung(s) checked, {skipped_count} skipped)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
