"""Isolated classical runtime/allocation measurements on named sparse blades.

This maintenance profile does not sample a quantum campaign. Timings exclude
fixture construction, and tracemalloc measures Python allocations, not RSS.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.metadata_io import binding
from benchmarks.archive_provenance import commit_binding
from benchmarks.sparse_ga_audit import audit_blade
from clifford_qc import MV, blade_from_mask, blade_mask
from clifford_qc.evidence import declare_evidence
from clifford_qc.pauli_kernel import _word_mul_unchecked
from clifford_qc.reproducibility import execution_provenance, stamp_record

FIXTURES = {
    "n3_grade3_shared": (3, 3, [7, 19, 35], [1, 2j, -3]),
    "n12_grade3_shared": (12, 3, [7, 3 | (1 << 22), 3 | (1 << 23)], [1, 2j, -3]),
    "n12_grade12_shared": (12, 12, [(1 << 12) - 1, 2047 | (1 << 22), 2047 | (1 << 23)], [1, 2j, -3]),
    "n12_grade3_disjoint": (12, 3, [7, 56], [1, 1j]),
}
SOURCES = ("clifford_qc/multivector.py", "clifford_qc/pauli_kernel.py", "clifford_qc/clifford.py",
           "benchmarks/run_sparse_ga_kernel.py", "benchmarks/sparse_ga_audit.py")


@lru_cache(maxsize=1)
def baseline_codes(n, grade):
    """The legacy cache exists only in this bounded n<=3 benchmark baseline."""
    if n > 3:
        raise ValueError("whole-grade baseline is restricted to n <= 3")
    return tuple(code for code in range(4 ** n) if blade_mask(n, code).bit_count() == grade)


def fixture(name):
    n, _, masks, coefficients = FIXTURES[name]
    return sum((coefficient * blade_from_mask(n, mask)
                for mask, coefficient in zip(masks, coefficients)), MV(n))


def exhaustive(value):
    """Previous whole-grade check, restricted to the small comparison fixture."""
    if value.n > 3:
        raise ValueError("the exhaustive baseline is restricted to n <= 3")
    k = next(iter(value.grades()))
    for code in baseline_codes(value.n, k - 1):
        vector = (MV(value.n, {code: 1}) * value).grade(1)
        if not vector.wedge(value).is_zero(1e-9):
            return False
    return True


def clear_caches():
    blade_mask.cache_clear()
    baseline_codes.cache_clear()
    _word_mul_unchecked.cache_clear()


def measure(name, method, repeats):
    value = fixture(name)
    check = value.is_blade if method == "active" else lambda: exhaustive(value)
    clear_caches()
    start = time.perf_counter_ns()
    decision = check()
    cold = time.perf_counter_ns() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        assert check() == decision
        samples.append(time.perf_counter_ns() - start)
    clear_caches()
    tracemalloc.start()
    try:
        assert check() == decision
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    cache_entries = baseline_codes.cache_info().currsize
    candidates, visited = None, None
    if method == "active":
        observed, candidates, visited = audit_blade(value)
        if observed is not decision:
            raise ValueError("untimed candidate audit changed the decision")
    return {"is_blade": decision, "cold_ns": cold,
            "warm_ns": {"min": min(samples), "median": statistics.median(samples), "max": max(samples)},
            "repeats": repeats, "cold_tracemalloc_peak_bytes": peak,
            "whole_grade_cache_entries": cache_entries,
            "candidate_masks": candidates, "visited_candidate_masks": visited}


def build_record(repeats):
    rows = []
    for name, (n, k, masks, coefficients) in FIXTURES.items():
        measurements = {}
        for method in (("active", "exhaustive") if n == 3 else ("active",)):
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--worker", name, "--method", method, "--repeats", str(repeats)],
                                  cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
            measurements[method] = json.loads(done.stdout)
        rows.append({"fixture": name, "n": n, "grade": k, "stored_terms": len(masks),
                     "masks": masks, "coefficients": [[complex(c).real, complex(c).imag] for c in coefficients],
                     "candidate_count": len(measurements["active"]["candidate_masks"]), "candidate_bound": k * len(masks),
                     "measurements": measurements,
                     "baseline_scope": "measured n=3 only; exhaustive n=12 intentionally not executed"})
    record = {"schema": "clifford_qc.sparse_ga_kernel.v2", "fixtures": rows,
              "scope": "Classical maintenance profile; fixed JW frame; no quantum resource claim. "
                       "Each fixture/method runs in a fresh process. Untraced cold/warm runtimes exclude "
                       "fixture construction; cold tracemalloc peak counts Python allocations, not RSS.",
              "source_files": [{"path": path, **binding(ROOT / path)} for path in SOURCES]}
    return declare_evidence(record, {
        "fixture_decisions_and_candidates": {"tier": "exact_algebra", "scope": "named deterministic fixture decisions and active-mask counts"},
        "runtime_ns": {"tier": "classical_process_measurement", "scope": "isolated cold call and warm repeat timings on this recorded environment"},
        "tracemalloc_peak_bytes": {"tier": "classical_process_measurement", "scope": "cold-call Python allocation peak, excluding fixture construction; not total process memory"},
    })


def clean_execution():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("kernel profiling requires a clean committed source tree")
    provenance = execution_provenance()
    if provenance["git_sha"] != sha or provenance["git_dirty"] is not False:
        raise ValueError("kernel profile execution identity differs from the clean checkout")
    for path in SOURCES:
        if binding(ROOT / path) != commit_binding(ROOT, sha, path):
            raise ValueError(f"uncommitted execution source bytes: {path}")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reproductions/sparse_ga_kernel.json")
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--worker", choices=tuple(FIXTURES), help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("active", "exhaustive"), default="active", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        print(json.dumps(measure(args.worker, args.method, args.repeats)))
        return 0
    provenance = clean_execution()
    record = build_record(args.repeats)
    if clean_execution()["git_sha"] != provenance["git_sha"]:
        raise ValueError("execution source changed while profiling")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp_record(record, provenance), indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
