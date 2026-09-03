"""Validate the completed R3 record-environment migration without sampling."""

from __future__ import annotations

import sys

try:
    from benchmarks.r3_environment_migration import migration_problems
except ImportError:  # pragma: no cover - direct script execution
    from r3_environment_migration import migration_problems


def main() -> int:
    problems = migration_problems()
    if problems:
        print("R3 environment migration: FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("R3 environment migration: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
