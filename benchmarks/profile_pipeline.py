"""Small, isolated-process A-CASE profile; not a molecular performance gate.

Run each storage/policy arm in a fresh process with BLAS threads pinned to one.
Copy this script to a baseline checkout to compare an earlier implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import MatrixElementBank, commutator_response, pauli_orbit, run_acase
from clifford_qc.subspace.packed import PackedRowStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', choices=('object', 'packed'), default='object')
    parser.add_argument('--policy', choices=('retain_all', 'stream_recompute'), default='retain_all')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--max-size', type=int, default=5)
    parser.add_argument('--frontier-pairs', type=int, default=4)
    args = parser.parse_args()
    model = tfim(args.n)
    rho = ExactMVBackend().state(model.reference, ())
    words = [op.word for op in odd_y_filter(local_pool(args.n))]
    pool = pauli_orbit(words) + commutator_response(model.hamiltonian, words)
    if args.policy == 'stream_recompute':
        from clifford_qc.subspace.streaming import StreamingMatrixElementBank
        bank = StreamingMatrixElementBank(rho, model.hamiltonian, storage=args.storage,
                                          frontier_pairs=args.frontier_pairs)
    else:
        bank = MatrixElementBank(rho, model.hamiltonian, storage=args.storage)
    calls = 0
    original = PackedRowStore.materialize

    def materialize(*positional, **keywords):
        nonlocal calls
        calls += 1
        return original(*positional, **keywords)

    started = time.perf_counter()
    with patch.object(PackedRowStore, 'materialize', materialize):
        result = run_acase(rho, model.hamiltonian, pool, bank=bank, max_size=args.max_size)
    seconds = time.perf_counter() - started
    matrices = bank.matrices(result.indices)
    digest = hashlib.sha256(json.dumps([result.labels, result.energy_history,
                                       result.stopped_reason]).encode())
    for matrix in matrices:
        digest.update(matrix.tobytes())
    counters = bank.resources(result.indices)
    print(json.dumps({
        'scope': 'TFIM smoke profile; process RSS includes interpreter and setup',
        'python': platform.python_version(), 'options': vars(args),
        'solve_seconds': seconds, 'materialize_calls_during_solve': calls,
        'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == 'darwin' else 1024),
        'numerical_sha256': digest.hexdigest(),
        'resources': {key: counters[key] for key in (
            'pairs_built', 'resident_operator_rows', 'coefficient_occurrences',
            'evicted_rows', 'recomputed_rows')},
    }, indent=2))


if __name__ == '__main__':
    main()
