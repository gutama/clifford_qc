"""Semantically compare a regenerated JSON/JSONL record with its committed peer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clifford_qc.reproducibility import compare_json_records


def _read(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _without_volatile(value):
    """Remove execution metadata and wall-clock telemetry, never result fields."""
    if isinstance(value, dict):
        return {key: _without_volatile(item) for key, item in value.items()
                if key != "provenance" and not key.endswith("_seconds")}
    if isinstance(value, list):
        return [_without_volatile(item) for item in value]
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args(argv)
    problems = compare_json_records(
        _without_volatile(_read(args.expected)),
        _without_volatile(_read(args.actual)),
        ignored_keys=frozenset())
    if problems:
        for problem in problems[:100]:
            print(problem)
        if len(problems) > 100:
            print(f"... {len(problems) - 100} further mismatch(es)")
        return 1
    print(f"OK ({args.actual.name} reproduces {args.expected.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
