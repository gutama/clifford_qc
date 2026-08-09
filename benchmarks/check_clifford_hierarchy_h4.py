"""Recompute and gate the H4 Clifford measurement-hierarchy record."""

from __future__ import annotations

import json
from pathlib import Path

from clifford_qc.reproducibility import compare_json_records

from run_clifford_hierarchy_h4 import build_record


HERE = Path(__file__).resolve().parent
RECORD = HERE / "reference_results" / "clifford_hierarchy_h4.json"


def main() -> int:
    expected = json.loads(RECORD.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(expected, actual, atol=1e-12, rtol=1e-12)
    if problems:
        for problem in problems[:20]:
            print(problem)
        return 1
    print("clifford hierarchy H4: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
