#!/usr/bin/env python3
"""Regenerate the finite-shot reconstruction study and compare its record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from clifford_qc.reproducibility import (
    CROSS_MACHINE_ATOL,
    CROSS_MACHINE_RTOL,
    compare_json_records,
)

from run_finite_shot_rethink import run


HERE = Path(__file__).resolve().parent
RECORD = HERE / "reference_results" / "finite_shot_rethink.json"


def main() -> int:
    committed = json.loads(RECORD.read_text(encoding="utf-8"))
    protocol = committed["protocol"]
    fresh = run(
        protocol["replicas"],
        tuple(protocol["shots_per_group"]),
        protocol["seed"],
        gamma=protocol["rank_selection_gamma"],
    )
    problems = compare_json_records(
        committed,
        fresh,
        RECORD.name,
        atol=CROSS_MACHINE_ATOL,
        rtol=CROSS_MACHINE_RTOL,
    )
    for problem in problems[:20]:
        print(f"FAIL {problem}")
    if len(problems) > 20:
        print(f"FAIL ... {len(problems) - 20} more")
    if problems:
        print(f"{len(problems)} finite-shot rethink record mismatch(es)")
        return 1
    print("OK (fixed seeds, shared caches, reconstruction and rank-rule arms match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
