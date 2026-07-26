"""Verify each committed summary matches the record it summarizes.

``run_benchmark.py`` writes a per-seed JSONL; ``summarize.py`` derives a CSV
and a Markdown table beside it. Those derived files are easy to forget when a
record is regenerated, and a summary that silently disagrees with its own
JSONL is the same defect class as a manuscript table transcribed by hand.

This regenerates every summary in memory and diffs it against the committed
file, so a forgotten regeneration fails loudly:

    python benchmarks/check_summaries.py

Exits nonzero on any mismatch, naming the command that fixes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "reference_results"
SUMMARIZE = ROOT / "benchmarks" / "summarize.py"


def regenerate(record: Path) -> tuple[str, str]:
    """Return the (markdown, csv) that summarize.py would write for ``record``."""
    csv_tmp = record.with_suffix(".summary-check.csv")
    try:
        md = subprocess.run(
            [sys.executable, str(SUMMARIZE), str(record), "--csv", str(csv_tmp)],
            capture_output=True, text=True, check=True).stdout
        return md, csv_tmp.read_text()
    finally:
        csv_tmp.unlink(missing_ok=True)


def main() -> int:
    stale: list[str] = []
    checked = 0
    for record in sorted(DATA.glob("*.jsonl")):
        md_path = DATA / f"{record.stem}_summary.md"
        csv_path = DATA / f"{record.stem}_summary.csv"
        if not md_path.exists() and not csv_path.exists():
            continue  # not every record has a summarize.py-derived twin
        checked += 1
        md, csv = regenerate(record)
        for path, want in ((md_path, md), (csv_path, csv)):
            if path.exists() and path.read_text() != want:
                stale.append(f"{path.relative_to(ROOT)} disagrees with "
                             f"{record.relative_to(ROOT)}")

    for s in stale:
        print(f"FAIL {s}")
    if stale:
        print("\nfix with, for each stale record:")
        print("  python benchmarks/summarize.py RECORD.jsonl \\")
        print("      --csv RECORD_summary.csv > RECORD_summary.md")
    else:
        print(f"OK ({checked} record(s) checked)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
