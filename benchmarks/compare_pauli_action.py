"""Compare R8 packed Pauli action with dense BLAS and dense ``einsum``.

This is a diagnostic benchmark rather than a committed scientific record.  It
times repeated operator application after construction and reports construction
time separately.  The ``einsum`` case intentionally consumes the same dense
matrix as BLAS: it answers whether changing the contraction spelling removes
the exponential operator storage (it does not).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from clifford_qc.dense_reference import to_matrix
from clifford_qc.multivector import MV
from clifford_qc.pauli_action import PauliLinearOperator


def _median_seconds(fn, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), result


def benchmark(n: int, terms: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + n)
    count = min(int(terms), 4 ** n)
    codes = rng.choice(4 ** n, size=count, replace=False)
    operator = MV(n, {int(code): float(rng.normal()) for code in codes})
    state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    state /= np.linalg.norm(state)

    started = time.perf_counter()
    packed = PauliLinearOperator(operator)
    packed_build = time.perf_counter() - started
    started = time.perf_counter()
    dense = to_matrix(operator)
    dense_build = time.perf_counter() - started

    # Warm up NumPy/BLAS dispatch before taking the medians.
    packed.matvec(state)
    dense @ state
    np.einsum("ij,j->i", dense, state, optimize=True)

    packed_seconds, packed_result = _median_seconds(
        lambda: packed.matvec(state), repeats)
    blas_seconds, blas_result = _median_seconds(lambda: dense @ state, repeats)
    einsum_seconds, einsum_result = _median_seconds(
        lambda: np.einsum("ij,j->i", dense, state, optimize=True), repeats)
    memory = packed.memory_estimate()
    packed_operator_bytes = memory["index_bytes"] + memory["term_bytes"]
    return {
        "n": n,
        "terms": count,
        "dimension": 2 ** n,
        "build_seconds": {"packed": packed_build, "dense": dense_build},
        "apply_seconds": {
            "packed": packed_seconds,
            "dense_blas": blas_seconds,
            "dense_einsum": einsum_seconds,
        },
        "operator_bytes": {
            "packed_numeric": packed_operator_bytes,
            "dense": memory["dense_operator_bytes"],
        },
        "dense_to_packed_memory_ratio": (
            memory["dense_operator_bytes"] / max(packed_operator_bytes, 1)),
        "max_abs_error": {
            "packed_vs_blas": float(np.max(np.abs(packed_result - blas_result))),
            "einsum_vs_blas": float(np.max(np.abs(einsum_result - blas_result))),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[6, 8],
                        help="qubit counts (dense reference limits practical size)")
    parser.add_argument("--terms", type=int, default=64,
                        help="maximum random Pauli terms per operator")
    parser.add_argument("--repeats", type=int, default=7,
                        help="timed applications per method")
    parser.add_argument("--seed", type=int, default=8128)
    args = parser.parse_args(argv)
    if args.terms < 1 or args.repeats < 1 or any(n < 1 for n in args.n):
        parser.error("n, terms, and repeats must all be positive")
    for n in args.n:
        print(json.dumps(benchmark(n, args.terms, args.repeats, args.seed),
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
