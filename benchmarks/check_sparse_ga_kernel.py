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
from benchmarks.archive_provenance import (
    archive_notice, commit_binding, history_kind, safe_path, verify_archive,
)
from benchmarks.sparse_ga_audit import audit_blade
from clifford_qc import MV, blade_from_mask
from clifford_qc.evidence import validate_evidence

EXPECTED = {
    "n3_grade3_shared": (3, 3, [7, 19, 35], [[1, 0], [0, 2], [-3, 0]], True),
    "n12_grade3_shared": (12, 3, [7, 4194307, 8388611], [[1, 0], [0, 2], [-3, 0]], True),
    "n12_grade12_shared": (12, 12, [4095, 4196351, 8390655], [[1, 0], [0, 2], [-3, 0]], True),
    "n12_grade3_disjoint": (12, 3, [7, 56], [[1, 0], [0, 1]], False),
}
RECORD_PATH = "benchmarks/reference_results/sparse_ga_kernel.json"
# This profile freezes its fixtures in the producer rather than a separate
# configuration file. Archives bind that definition as well as execution bytes.
CONFIG_PATH = "benchmarks/run_sparse_ga_kernel.py"


def check_record(record, root=ROOT, *, archive=False, record_path=None):
    root = Path(root).resolve()
    history_kind(root, archive=archive)
    if not isinstance(record, dict):
        raise ValueError("kernel profile must be an object")
    validate_evidence(record)
    if set(record) != {"schema", "fixtures", "scope", "source_files", "evidence_schema", "evidence", "provenance"}:
        raise ValueError("kernel profile key set differs")
    if record["schema"] != "clifford_qc.sparse_ga_kernel.v2":
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
        observed, generated, visited = audit_blade(value)
        if observed is not decision:
            raise ValueError("rebuilt kernel fixture decision differs")
        if set(generated) != candidates or not set(visited).issubset(candidates):
            raise ValueError("live kernel candidate enumeration differs")
        if decision and set(visited) != candidates:
            raise ValueError("live kernel skipped required contraction masks")
        methods = {"active", "exhaustive"} if n == 3 else {"active"}
        if set(row["measurements"]) != methods:
            raise ValueError("measurement method set differs")
        for method, measured in row["measurements"].items():
            if set(measured) != {"is_blade", "cold_ns", "warm_ns", "repeats", "cold_tracemalloc_peak_bytes", "whole_grade_cache_entries", "candidate_masks", "visited_candidate_masks"}:
                raise ValueError("classical measurement key set differs")
            if measured["is_blade"] is not decision:
                raise ValueError("measured fixture decision differs")
            if type(measured["repeats"]) is not int or measured["repeats"] < 1:
                raise ValueError("invalid repeat count")
            if type(measured["whole_grade_cache_entries"]) is not int or measured["whole_grade_cache_entries"] < 0:
                raise ValueError("invalid cache entry count")
            warm = measured["warm_ns"]
            if not isinstance(warm, dict) or set(warm) != {"min", "median", "max"}:
                raise ValueError("warm timing shape differs")
            values = [measured["cold_ns"], measured["cold_tracemalloc_peak_bytes"], *warm.values()]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("invalid classical measurement")
            if set(warm) != {"min", "median", "max"} or not warm["min"] <= warm["median"] <= warm["max"]:
                raise ValueError("warm timing order differs")
            if method == "active" and measured["whole_grade_cache_entries"] != 0:
                raise ValueError("active method populated a whole-grade cache")
            if method == "active":
                masks_observed, visits = measured["candidate_masks"], measured["visited_candidate_masks"]
                if (not isinstance(masks_observed, list) or not isinstance(visits, list)
                        or masks_observed != sorted(candidates) or len(visits) != len(set(visits))
                        or not set(visits).issubset(candidates) or (decision and set(visits) != candidates)):
                    raise ValueError("recorded kernel candidate observations differ")
            elif measured["candidate_masks"] is not None or measured["visited_candidate_masks"] is not None:
                raise ValueError("exhaustive baseline cannot claim active candidate observations")
    expected_paths = ["clifford_qc/multivector.py", "clifford_qc/pauli_kernel.py", "clifford_qc/clifford.py",
                      "benchmarks/run_sparse_ga_kernel.py", "benchmarks/sparse_ga_audit.py"]
    if [row["path"] for row in record["source_files"]] != expected_paths:
        raise ValueError("kernel profile source set differs")
    provenance = record["provenance"]
    if not isinstance(provenance, dict) or provenance.get("git_dirty") is not False:
        raise ValueError("kernel profile requires clean execution provenance")
    execution = provenance.get("git_sha")
    if (not isinstance(execution, str) or len(execution) != 40
            or any(letter not in "0123456789abcdef" for letter in execution)):
        raise ValueError("execution identity must be a full commit SHA")
    if archive:
        source_bindings = {}
        for row in record["source_files"]:
            if set(row) != {"path", "sha256", "git_blob_sha1"}:
                raise ValueError("kernel profile source binding keys differ")
            source_bindings[row["path"]] = {key: row[key] for key in ("sha256", "git_blob_sha1")}
        path = root / RECORD_PATH if record_path is None else Path(record_path)
        if not path.is_absolute():
            path = root / path
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("archive record must stay inside the verification root") from exc
        if read_json(safe_path(root, relative)) != record:
            raise ValueError("archive record differs from the inspected record contents")
        if binding(safe_path(root, CONFIG_PATH)) != source_bindings[CONFIG_PATH]:
            raise ValueError("archive fixture configuration differs from recorded execution")
        verify_archive(root, relative, CONFIG_PATH, execution, source_bindings)
    else:
        for row in record["source_files"]:
            if row != {"path": row["path"], **commit_binding(root, execution, row["path"])}:
                raise ValueError("kernel profile source digest differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / RECORD_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--archive", action="store_true",
                      help="verify packaged content bindings; chronology remains unverified")
    mode.add_argument("--require-git", action="store_true",
                      help="require full Git history (the default)")
    args = parser.parse_args(argv)
    try:
        check_record(read_json(args.record), archive=args.archive, record_path=args.record.resolve())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Sparse GA profile failed: {exc}", file=sys.stderr)
        return 1
    if args.archive:
        archive_notice()
    print("OK: named sparse GA fixtures, active-support bounds and source bindings; timings are observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
