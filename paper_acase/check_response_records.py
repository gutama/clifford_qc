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

import math
import sys
from pathlib import Path
from typing import Any

import json

from run_response_record import build_record

HERE = Path(__file__).resolve().parent
ATOL = 1e-11
RTOL = 1e-11
MAX_REPORTED_PROBLEMS = 20

RECORDS = (
    (HERE / "data" / "response_bootstrap.json", "determinant"),
    (HERE / "data" / "response_bootstrap_illconditioned.json", "krylov"),
)


def _compare(expected: Any, actual: Any, path: str, problems: list[str]) -> None:
    """Recursively compare a JSON record, tolerating only floating round-off."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(
                f"{path}: expected object, got {type(actual).__name__}")
            return
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            problems.append(f"{path}: missing keys {missing}")
        if extra:
            problems.append(f"{path}: unexpected keys {extra}")
        for key in sorted(set(expected) & set(actual)):
            _compare(expected[key], actual[key], f"{path}.{key}", problems)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            problems.append(
                f"{path}: expected array, got {type(actual).__name__}")
            return
        if len(expected) != len(actual):
            problems.append(
                f"{path}: expected {len(expected)} entries, got {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]", problems)
        return

    # bool is a subclass of int, so exact discrete types must be handled first.
    if expected is None or isinstance(expected, (bool, str, int)):
        if type(actual) is not type(expected) or actual != expected:
            problems.append(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            problems.append(
                f"{path}: expected floating value, got {actual!r}")
            return
        if not math.isclose(
                expected, float(actual), rel_tol=RTOL, abs_tol=ATOL):
            problems.append(
                f"{path}: expected {expected:.17g}, got {float(actual):.17g}")
        return

    problems.append(f"{path}: unsupported value type {type(expected).__name__}")


def main() -> int:
    problems: list[str] = []
    for record_path, family in RECORDS:
        committed = json.loads(record_path.read_text(encoding="utf-8"))
        fresh = build_record(generators=family)
        _compare(committed, fresh, record_path.name, problems)

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
