"""Audit sparse fixture decisions and bindings, without asserting timing thresholds."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.metadata_io import binding, read_json
from clifford_qc import MV, blade_from_mask
from clifford_qc.evidence import validate_evidence

EXPECTED = {
    "n3_grade3_shared": (3, 3, [7, 19, 35], [[1, 0], [0, 2], [-3, 0]], True),
    "n12_grade3_shared": (12, 3, [7, 4194307, 8388611], [[1, 0], [0, 2], [-3, 0]], True),
    "n12_grade12_shared": (12, 12, [4095, 4196351, 8390655], [[1, 0], [0, 2], [-3, 0]], True),
    "n12_grade3_disjoint": (12, 3, [7, 56], [[1, 0], [0, 1]], False),
}


def check_record(record, root=ROOT):
    if not isinstance(record, dict):
        raise ValueError("kernel profile must be an object")
    validate_evidence(record)
    if set(record) != {"schema", "fixtures", "scope", "source_files", "evidence_schema", "evidence", "provenance"}:
        raise ValueError("kernel profile key set differs")
    if record["schema"] != "clifford_qc.sparse_ga_kernel.v1":
        raise ValueError("kernel profile schema differs")
    scope = ("Classical maintenance profile; fixed JW frame; no quantum resource claim. "
             "Each fixture/method runs in a fresh process. Untraced cold/warm runtimes exclude "
             "fixture construction; cold tracemalloc peak counts Python allocations, not RSS.")
    if record["scope"] != scope:
        raise ValueError("kernel profile scope differs")
    if [row["fixture"] for row in record["fixtures"]] != list(EXPECTED):
        raise ValueError("kernel profile fixture set/order differs")
    for row in record["fixtures"]:
        n, k, masks, coefficients, decision = EXPECTED[row["fixture"]]
        if (row["n"], row["grade"], row["masks"], row["coefficients"]) != (n, k, masks, coefficients):
            raise ValueError("kernel fixture definition differs")
        value = sum((complex(*c) * blade_from_mask(n, mask)
                     for mask, c in zip(masks, coefficients)), MV(n))
        candidates = {mask ^ (1 << bit) for mask in masks
                      for bit in range(2 * n) if mask & (1 << bit)}
        if (row["stored_terms"] != value.nnz() or row["candidate_count"] != len(candidates)
                or row["candidate_bound"] != k * value.nnz() or len(candidates) > row["candidate_bound"]):
            raise ValueError("active candidate counts differ")
        if value.is_blade() is not decision:
            raise ValueError("rebuilt kernel fixture decision differs")
        methods = {"active", "exhaustive"} if n == 3 else {"active"}
        if set(row["measurements"]) != methods:
            raise ValueError("measurement method set differs")
        for method, measured in row["measurements"].items():
            if set(measured) != {"is_blade", "cold_ns", "warm_ns", "repeats", "cold_tracemalloc_peak_bytes", "whole_grade_cache_entries"}:
                raise ValueError("classical measurement key set differs")
            if measured["is_blade"] is not decision:
                raise ValueError("measured fixture decision differs")
            if type(measured["repeats"]) is not int or measured["repeats"] < 1:
                raise ValueError("invalid repeat count")
            if type(measured["whole_grade_cache_entries"]) is not int or measured["whole_grade_cache_entries"] < 0:
                raise ValueError("invalid cache entry count")
            values = [measured["cold_ns"], measured["cold_tracemalloc_peak_bytes"], *measured["warm_ns"].values()]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("invalid classical measurement")
            warm = measured["warm_ns"]
            if set(warm) != {"min", "median", "max"} or not warm["min"] <= warm["median"] <= warm["max"]:
                raise ValueError("warm timing order differs")
            if method == "active" and measured["whole_grade_cache_entries"] != 0:
                raise ValueError("active method populated a whole-grade cache")
    expected_paths = ["clifford_qc/multivector.py", "clifford_qc/pauli_kernel.py",
                      "benchmarks/run_sparse_ga_kernel.py"]
    if [row["path"] for row in record["source_files"]] != expected_paths:
        raise ValueError("kernel profile source set differs")
    for row in record["source_files"]:
        if row != {"path": row["path"], **binding(root / row["path"])}:
            raise ValueError("kernel profile source digest differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "benchmarks/reference_results/sparse_ga_kernel.json")
    args = parser.parse_args(argv)
    try:
        check_record(read_json(args.record))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Sparse GA profile failed: {exc}", file=sys.stderr)
        return 1
    print("OK: named sparse GA fixtures, active-support bounds and source bindings; timings are observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
