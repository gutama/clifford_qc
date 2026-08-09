#!/usr/bin/env python3
"""Regenerate the finite-shot optimization diagnostic and compare its record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from clifford_qc.reproducibility import compare_json_records

from run_finite_shot_optimization import run


HERE = Path(__file__).resolve().parent
RECORD = HERE / "reference_results" / "finite_shot_optimization.json"


def main() -> int:
    committed = json.loads(RECORD.read_text(encoding="utf-8"))
    budget = committed["budget"]
    fresh = run(
        committed["arms"]["uniform_fixed"]["replicas"],
        budget["uniform_shots_per_group"],
        budget["adaptive_pilot_shots_per_group"],
        seed=9000,
    )
    problems = compare_json_records(
        committed,
        fresh,
        RECORD.name,
        atol=1e-12,
        rtol=1e-12,
    )
    for problem in problems[:20]:
        print(f"FAIL {problem}")
    if len(problems) > 20:
        print(f"FAIL ... {len(problems) - 20} more")
    if problems:
        print(f"{len(problems)} finite-shot record mismatch(es)")
        return 1
    print("OK (fixed seeds, physical budget, allocation, and solver arms match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
