"""Check committed response records against a fresh deterministic run.

The grouped histograms, replica accounting, seeds, and all discrete metadata
must agree exactly.  Floating-point outputs are compared numerically because
LAPACK and BLAS implementations can move the last few serialized bits without
changing the measured pipeline or any reported digit.  A byte comparison would
therefore reject an independently reproduced record for platform noise while
still saying nothing stronger about its scientific content.

Run from the repository root:

    python paper_acase/check_response_records.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

from clifford_qc.reproducibility import compare_json_records
from run_response_record import build_record

HERE = Path(__file__).resolve().parent
ATOL = 1e-11
RTOL = 1e-11
MAX_REPORTED_PROBLEMS = 20

RECORDS = (
    (HERE / "data" / "response_bootstrap.json", "determinant"),
    (HERE / "data" / "response_bootstrap_illconditioned.json", "krylov"),
)


def main() -> int:
    problems: list[str] = []
    for record_path, family in RECORDS:
        committed = json.loads(record_path.read_text(encoding="utf-8"))
        fresh = build_record(generators=family)
        problems.extend(compare_json_records(
            committed,
            fresh,
            record_path.name,
            atol=ATOL,
            rtol=RTOL,
        ))

    for problem in problems[:MAX_REPORTED_PROBLEMS]:
        print(f"FAIL {problem}")
    if len(problems) > MAX_REPORTED_PROBLEMS:
        print(f"FAIL ... {len(problems) - MAX_REPORTED_PROBLEMS} more")
    if problems:
        print(f"{len(problems)} response-record mismatch(es)")
        return 1
    print(
        "OK (discrete fields exact; floating fields within "
        f"atol={ATOL:g}, rtol={RTOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
