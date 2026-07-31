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
import math
import sys
from pathlib import Path

from clifford_qc.reproducibility import compare_json_records

from run_matched_h4 import build_record

HERE = Path(__file__).resolve().parent
RECORD = HERE / "reference_results" / "matched_h4.json"
ATOL = 1e-11
RTOL = 1e-11
MAX_REPORTED_PROBLEMS = 20
# A computed condition number carries relative error of order kappa*eps: it is
# the ratio of extreme singular values, and the smallest of them is what the
# conditioning is large *because of*. The Krylov arm's kappa_S is 6.6e10, so
# only about five of its digits mean anything, and the flat tolerance above
# asks it for seventeen. CI duly failed on a 4e-10 relative wobble -- 36000x
# inside the bound. Condition numbers therefore get the bound itself, which
# stays far tighter than any real change: the arms in this table differ by ten
# orders of magnitude, not by parts per million.
DOUBLE_EPS = 2.220446049250313e-16
CONDITION_KEY = "condition_number"


def _condition_numbers(node, path="$", out=None):
    """Every ``condition_number`` in the record, keyed by its path."""
    out = {} if out is None else out
    if isinstance(node, dict):
        for key, value in node.items():
            if key == CONDITION_KEY and isinstance(value, (int, float)):
                out[f"{path}.{key}"] = float(value)
            else:
                _condition_numbers(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _condition_numbers(value, f"{path}[{index}]", out)
    return out


def _blank_condition_numbers(node):
    """Replace condition numbers with a sentinel so the generic pass skips them."""
    if isinstance(node, dict):
        return {key: ("<checked separately>" if key == CONDITION_KEY
                      and isinstance(value, (int, float))
                      else _blank_condition_numbers(value))
                for key, value in node.items()}
    if isinstance(node, list):
        return [_blank_condition_numbers(value) for value in node]
    return node


def compare_condition_numbers(committed: dict, fresh: dict) -> list[str]:
    problems: list[str] = []
    left = _condition_numbers(committed)
    right = _condition_numbers(fresh)
    for path in sorted(set(left) | set(right)):
        if path not in left or path not in right:
            problems.append(f"{path}: present in only one record")
            continue
        expected, actual = left[path], right[path]
        tolerance = max(RTOL, abs(expected) * DOUBLE_EPS)
        if not math.isclose(expected, actual, rel_tol=tolerance, abs_tol=ATOL):
            problems.append(
                f"{path}: expected {expected!r}, got {actual!r} "
                f"(rel_tol {tolerance:.3g} from kappa*eps)")
    return problems


def main() -> int:
    committed = json.loads(RECORD.read_text(encoding="utf-8"))
    fresh = build_record()
    problems = compare_condition_numbers(committed, fresh)
    problems += compare_json_records(
        _blank_condition_numbers(committed),
        _blank_condition_numbers(fresh),
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
        f"atol={ATOL:g}, rtol={RTOL:g}; condition numbers within kappa*eps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
