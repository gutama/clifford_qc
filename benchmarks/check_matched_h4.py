"""Check the matched-contract record with exact resources and tolerant energies.

Every quantity the comparison actually argues from is discrete and reproduces
exactly: state preparations, ansatz rotors, optimizer evaluations, candidate
scorings, word universes, QWC group counts, basis sizes, effective ranks, and
the selected labels. Those are compared bit for bit.

The floating fields do not survive a change of BLAS. A byte gate on this record
failed CI on differences of a few units in the fifteenth decimal place --
``sector_weight`` 1.0 against 0.9999999999999999, ``error_hartree`` moving by
9e-16 Ha -- with every count identical. The warm-started and ADAPT arms make
that unavoidable rather than incidental: their reference states come from a
parameter optimization whose path depends on reduction order, so the last bits
of the final energy are a property of the runner, not of the method.

This uses the same tolerance as the warm-start and response-record gates, for
the same reason and with the same consequence: a real change to any resource
count, label, or rank still fails.

    python benchmarks/check_matched_h4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from clifford_qc.reproducibility import compare_json_records

from run_matched_h4 import build_record

HERE = Path(__file__).resolve().parent
RECORD = HERE / "reference_results" / "matched_h4.json"
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
        print(f"{len(problems)} matched-contract record mismatch(es)")
        return 1
    print(
        "OK (resource and selection fields exact; floating fields within "
        f"atol={ATOL:g}, rtol={RTOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
