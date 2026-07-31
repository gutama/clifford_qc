"""Check the warm-start record with exact resources and tolerant energies.

The one-thread benchmark reproduces operator choices, gradient counts,
optimizer evaluations, basis labels, ranks, and word universes exactly.
Different BLAS/LAPACK builds can still move the final serialized energy by a
few units in the fifteenth decimal place, so those floating fields use the
same narrow numerical tolerance as the finite-shot response record gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from clifford_qc.reproducibility import compare_json_records

from run_warm_start import build_record

HERE = Path(__file__).resolve().parent
RECORD = HERE / "reference_results" / "warm_start_h4.json"
ATOL = 1e-11
RTOL = 1e-11
MAX_REPORTED_PROBLEMS = 20


def main() -> int:
    committed = json.loads(RECORD.read_text(encoding="utf-8"))
    fresh = build_record()
    problems = compare_json_records(
        committed,
        fresh,
        RECORD.name,
        atol=ATOL,
        rtol=RTOL,
    )
    for problem in problems[:MAX_REPORTED_PROBLEMS]:
        print(f"FAIL {problem}")
    if len(problems) > MAX_REPORTED_PROBLEMS:
        print(f"FAIL ... {len(problems) - MAX_REPORTED_PROBLEMS} more")
    if problems:
        print(f"{len(problems)} warm-start record mismatch(es)")
        return 1
    print(
        "OK (resource and selection fields exact; floating fields within "
        f"atol={ATOL:g}, rtol={RTOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
