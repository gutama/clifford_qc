"""Recompute and gate the Clifford measurement-hierarchy records."""

from __future__ import annotations

import json

from clifford_qc.reproducibility import compare_json_records

from run_clifford_hierarchy import SYSTEMS, build_record, record_path


def main() -> int:
    failures = 0
    for system in sorted(SYSTEMS):
        path = record_path(system)
        expected = json.loads(path.read_text(encoding="utf-8"))
        actual = build_record(system)
        problems = compare_json_records(expected, actual, atol=1e-12, rtol=1e-12)
        if problems:
            failures += 1
            print(f"clifford hierarchy {system}: FAIL")
            for problem in problems[:20]:
                print(f"  {problem}")
        else:
            print(f"clifford hierarchy {system}: PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
