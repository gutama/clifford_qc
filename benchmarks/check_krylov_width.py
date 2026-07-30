"""Check the Krylov width record's *reported* numbers, not its bytes.

The other benchmark records get a bit-for-bit CI gate.  This one cannot have
one, and the reason is worth stating because it is the same reason the record
carries a threshold sweep at all.

``word_universe_by_threshold['0.0']`` counts every word the floating-point
product carries, including the thousands that exist only as accumulated
round-off.  Two runs of the chemistry rungs do not agree on that number: PySCF
settles on molecular-orbital coefficients that differ in the last bits, and a
Hamiltonian that differs at 1e-16 raises to a seventeenth power whose
round-off-level support differs.  Observed on ``h4_chain(r=0.9)``: 8184 on one
run and 8180 on the next, with the reported count identical at 4224 both times.

So the raw entry is not a reproducible quantity, and gating on it would give a
CI failure that means nothing.  The reported counts -- the numbers the
manuscript quotes -- are stable, and those are what this checks.

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
    expected = {row["system"]: row["word_universe"] for row in committed["rows"]}

    sys.path.insert(0, str(ROOT))
    from run_krylov_width import build_record

    fresh = build_record()
    problems: list[str] = []

    for row in fresh["rows"]:
        system = row["system"]
        if system not in expected:
            problems.append(f"{system}: present in a fresh run, absent from the record")
            continue
        if row["word_universe"] != expected[system]:
            problems.append(
                f"{system}: reported W is {row['word_universe']}, "
                f"record says {expected[system]}")
    fresh_systems = {row["system"] for row in fresh["rows"]}
    skipped = {row["rung"] for row in fresh.get("skipped", ())}
    for system in expected:
        if system not in fresh_systems and system not in skipped:
            problems.append(f"{system}: in the record, not produced by a fresh run")

    if fresh["reported_threshold"] != committed["reported_threshold"]:
        problems.append(
            f"reported threshold moved: {committed['reported_threshold']} "
            f"-> {fresh['reported_threshold']}")

    for problem in problems:
        print(f"FAIL {problem}")
    checked = len(fresh_systems)
    print(f"{len(problems)} problem(s)" if problems
          else f"OK ({checked} rung(s) checked)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
