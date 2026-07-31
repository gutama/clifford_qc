"""Small helpers for reproducibility gates on versioned JSON records."""

from __future__ import annotations

import math
from typing import Any


def compare_json_records(
        expected: Any,
        actual: Any,
        path: str = "$",
        *,
        atol: float = 1e-11,
        rtol: float = 1e-11,
) -> list[str]:
    """Compare JSON-like values with exact discrete and tolerant float fields.

    Keys, list lengths, strings, booleans, integers, and nulls are exact.
    Floating fields alone use :func:`math.isclose`, which isolates harmless
    BLAS/LAPACK last-bit changes without weakening resource or replica counts.
    """
    problems: list[str] = []

    def compare(left: Any, right: Any, here: str) -> None:
        if isinstance(left, dict):
            if not isinstance(right, dict):
                problems.append(
                    f"{here}: expected object, got {type(right).__name__}")
                return
            missing = sorted(set(left) - set(right))
            extra = sorted(set(right) - set(left))
            if missing:
                problems.append(f"{here}: missing keys {missing}")
            if extra:
                problems.append(f"{here}: unexpected keys {extra}")
            for key in sorted(set(left) & set(right)):
                compare(left[key], right[key], f"{here}.{key}")
            return

        if isinstance(left, list):
            if not isinstance(right, list):
                problems.append(
                    f"{here}: expected array, got {type(right).__name__}")
                return
            if len(left) != len(right):
                problems.append(
                    f"{here}: expected {len(left)} entries, got {len(right)}")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{here}[{index}]")
            return

        # bool is a subclass of int, so exact discrete types come first.
        if left is None or isinstance(left, (bool, str, int)):
            if type(right) is not type(left) or right != left:
                problems.append(f"{here}: expected {left!r}, got {right!r}")
            return

        if isinstance(left, float):
            if not isinstance(right, (int, float)) or isinstance(right, bool):
                problems.append(
                    f"{here}: expected floating value, got {right!r}")
                return
            if not math.isclose(
                    left, float(right), rel_tol=rtol, abs_tol=atol):
                problems.append(
                    f"{here}: expected {left:.17g}, got {float(right):.17g}")
            return

        problems.append(
            f"{here}: unsupported value type {type(left).__name__}")

    compare(expected, actual, path)
    return problems


__all__ = ["compare_json_records"]
