"""Verify each committed summary matches the record it summarizes.

``run_benchmark.py`` writes a per-seed JSONL and ``summarize.py`` derives a CSV
and a Markdown table beside it; ``run_acase_ladder.py`` writes the §7 ladder
record and ``summarize_ladder.py`` owns that schema. Those derived files are easy to forget when a
record is regenerated, and a summary that silently disagrees with its own
JSONL is the same defect class as a manuscript table transcribed by hand.

Which records are *required* to have summaries is declared, not inferred from
what happens to be on disk: every config in ``benchmarks/configs/`` produces a
record of the same stem, and that record must carry both companions. Inferring
the requirement from existing files cannot distinguish "this record has no
summary by design" from "someone deleted the summary", so a deletion would
pass silently.

Checks, for each required record:
  - the JSONL, the CSV, and the Markdown all exist;
  - each companion is byte-identical to what ``summarize.py`` regenerates.

Any *unrequired* record that nonetheless carries a companion is checked for
content too, so a stray or hand-edited summary is not ignored.

    python benchmarks/check_summaries.py

Exits nonzero on any mismatch, naming the command that fixes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "reference_results"
CONFIGS = ROOT / "benchmarks" / "configs"
SUMMARIZE = ROOT / "benchmarks" / "summarize.py"

# Records whose schema a different summarizer owns. Declared, like the required
# set itself: two record schemas exist (per-seed ADAPT runs and the §7 ladder's
# one-run-per-method rows), and inferring which script wrote a record from its
# contents would guess wrong the first time a third schema appears.
SUMMARIZERS = {"acase_ladder": ROOT / "benchmarks" / "summarize_ladder.py"}


def required_stems() -> set[str]:
    """Record stems that must have both summary companions.

    Derived from the benchmark configs so the manifest maintains itself: add a
    config, and its record's summaries become required automatically.
    """
    return {p.stem for p in CONFIGS.glob("*.json")}


def summarizer_for(record: Path) -> Path:
    return SUMMARIZERS.get(record.stem, SUMMARIZE)


def regenerate(record: Path) -> tuple[str, str, str | None]:
    """Return the (markdown, csv, error) this record's summarizer would produce.

    Each summarizer understands one schema, so a summary sitting beside a record
    written by one of the bespoke scripts makes it exit nonzero. Report that as
    a finding rather than letting a traceback escape a script whose job is to
    gate a release.
    """
    csv_tmp = record.with_suffix(".summary-check.csv")
    try:
        done = subprocess.run(
            [sys.executable, str(summarizer_for(record)), str(record),
             "--csv", str(csv_tmp)],
            capture_output=True, text=True)
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip().splitlines()
            return "", "", detail[-1] if detail else f"exit {done.returncode}"
        return done.stdout, csv_tmp.read_text(), None
    finally:
        csv_tmp.unlink(missing_ok=True)


def main() -> int:
    problems: list[str] = []
    required = required_stems()
    rel = lambda p: p.relative_to(ROOT)

    stems = sorted(required | {p.stem for p in DATA.glob("*.jsonl")})
    checked = 0
    for stem in stems:
        record = DATA / f"{stem}.jsonl"
        md_path = DATA / f"{stem}_summary.md"
        csv_path = DATA / f"{stem}_summary.csv"
        is_required = stem in required

        if not record.exists():
            if is_required:
                problems.append(f"{rel(record)} is missing (a config declares it)")
            continue

        present = [p for p in (md_path, csv_path) if p.exists()]
        if not is_required and not present:
            continue  # a record with no summarize.py-derived twin, by design

        # A record that is required, or that already carries one companion,
        # must carry both: a lone survivor is how a deletion hides.
        for path in (md_path, csv_path):
            if not path.exists():
                why = ("a config declares this record" if is_required
                       else f"{rel(present[0])} exists")
                problems.append(f"{rel(path)} is missing ({why})")

        checked += 1
        md, csv, error = regenerate(record)
        if error:
            problems.append(f"cannot summarize {rel(record)} with "
                            f"{rel(summarizer_for(record))} ({error}); it has "
                            "summary companions but does not match that "
                            "script's schema")
            continue
        for path, want in ((md_path, md), (csv_path, csv)):
            if path.exists() and path.read_text() != want:
                problems.append(f"{rel(path)} disagrees with {rel(record)}")

    for p in problems:
        print(f"FAIL {p}")
    if problems:
        print("\nfix with, for each affected record (summarize_ladder.py for the "
              "ladder record):")
        print("  python benchmarks/summarize.py RECORD.jsonl \\")
        print("      --csv RECORD_summary.csv > RECORD_summary.md")
    else:
        print(f"OK ({checked} record(s) checked, {len(required)} required)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
