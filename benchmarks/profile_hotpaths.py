"""Deterministic micro-profile for the R7/R8 computational hot paths.

This is a diagnostic benchmark, not a scientific result writer: it prints one
JSON record and does not modify committed benchmark artifacts.  The workloads
are deliberately small enough for CI/developer laptops while still exercising
the same kernels used by larger studies.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend, FiniteShotBackend
from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.measurement.grouping import _qwc_partition, qwc_groups
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.models import hubbard, tfim
from clifford_qc.ir import PauliWord
from clifford_qc.pauli_action import PauliLinearOperator
from clifford_qc.pauli_kernel import _word_mul_unchecked, word_mul
from clifford_qc.subspace import MatrixElementBank, bootstrap_ritz
from clifford_qc.subspace.generators import (
    commutator_response,
    identity_generator,
    pauli_orbit,
)


def _median_seconds(action, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples))


def _word_mul_profile(repeats: int, n: int = 12, pairs: int = 10_000):
    rng = np.random.default_rng(7)
    left = rng.integers(0, 4 ** n, size=pairs, dtype=np.int64)
    right = rng.integers(0, 4 ** n, size=pairs, dtype=np.int64)

    def cold_batch():
        _word_mul_unchecked.cache_clear()
        for a, b in zip(left, right):
            word_mul(n, int(a), int(b))

    def internal_batch():
        _word_mul_unchecked.cache_clear()
        for a, b in zip(left, right):
            _word_mul_unchecked(n, int(a), int(b))

    validated = _median_seconds(cold_batch, repeats)
    internal = _median_seconds(internal_batch, repeats)
    return {
        "pairs": pairs,
        "validated_seconds": validated,
        "validated_ns_per_pair": 1e9 * validated / pairs,
        "internal_seconds": internal,
        "internal_ns_per_pair": 1e9 * internal / pairs,
        "internal_speedup": validated / internal,
    }


def _subspace_fixture(n: int = 5):
    model = tfim(n, J=1.0, h=1.0)
    rho = ExactMVBackend().state(model.reference, ())
    words = [op.word for op in odd_y_filter(local_pool(n))][:8]
    generators = ([identity_generator(n)] + pauli_orbit(words)
                  + commutator_response(model.hamiltonian, words))
    return model, rho, generators


def _element_profile(repeats: int):
    model, rho, generators = _subspace_fixture()
    latest = None

    def cold_build():
        nonlocal latest
        _word_mul_unchecked.cache_clear()
        latest = MatrixElementBank(rho, model.hamiltonian, generators)
        latest.matrices()

    seconds = _median_seconds(cold_build, repeats)
    resources = latest.resources()
    return {
        "seconds": seconds,
        "generators": len(latest),
        "pairs_built": resources["pairs_built"],
        "word_universe": resources["word_universe"],
    }, latest


def _grouping_profile(bank: MatrixElementBank, repeats: int):
    words = [PauliWord(bank.n, code) for code in sorted(bank.word_set())]

    def cold_grouping():
        _qwc_partition.cache_clear()
        qwc_groups(words)

    seconds = _median_seconds(cold_grouping, repeats)
    groups = qwc_groups(words)
    return {"seconds": seconds, "words": len(words), "groups": len(groups)}


def _projected_solve_profile(bank: MatrixElementBank, repeats: int):
    seconds = _median_seconds(lambda: bank.solve(), repeats)
    return {"seconds": seconds, "basis_size": len(bank)}


def _matvec_profiles(repeats: int, sites: int):
    model = hubbard(sites)
    electrons = model.metadata["n_electrons"]
    sz = model.metadata["sz"]
    sector = SectorStatevectorBackend(model.n, electrons, sz)
    sector_operator = sector.operator(model.hamiltonian, precompute=True)
    full_operator = PauliLinearOperator(model.hamiltonian)
    rng = np.random.default_rng(11)
    sector_state = rng.normal(size=sector.dimension) + 1j * rng.normal(
        size=sector.dimension)
    sector_state /= np.linalg.norm(sector_state)
    full_state = rng.normal(size=full_operator.dimension) + 1j * rng.normal(
        size=full_operator.dimension)
    full_state /= np.linalg.norm(full_state)

    sector_seconds = _median_seconds(
        lambda: sector_operator.matvec(sector_state), repeats)
    full_seconds = _median_seconds(lambda: full_operator.matvec(full_state), repeats)
    return {
        "sector_matvec": {
            "seconds": sector_seconds,
            "dimension": sector.dimension,
            **sector_operator.memory_estimate(),
        },
        "pauli_linear_operator_matvec": {
            "seconds": full_seconds,
            "dimension": full_operator.dimension,
            **full_operator.memory_estimate(),
        },
        "lanczos_full_reorth_memory": {
            "sector_100_vectors_bytes": 16 * sector.dimension * 101,
            "full_100_vectors_bytes": 16 * full_operator.dimension * 101,
        },
    }


def _bootstrap_profile(replicates: int):
    model = tfim(4, J=1.0, h=1.0)
    rho = ExactMVBackend().state(model.reference, ())
    words = [op.word for op in odd_y_filter(local_pool(4))][:3]
    bank = MatrixElementBank(
        rho, model.hamiltonian, [identity_generator(4)] + pauli_orbit(words))
    shared = SharedMeasurement(bank)
    cache = shared.measure(FiniteShotBackend(seed=19), 1_000)
    started = time.perf_counter()
    bootstrap_ritz(shared, cache, replicates=replicates, seed=23)
    seconds = time.perf_counter() - started
    return {"seconds": seconds, "replicates": replicates,
            "seconds_per_replicate": seconds / replicates,
            "basis_size": len(bank), "groups": len(shared.groups)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=30)
    parser.add_argument("--sites", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.bootstrap_replicates < 2:
        raise ValueError("--bootstrap-replicates must be at least two")
    if args.sites < 2:
        raise ValueError("--sites must be at least two")

    element, bank = _element_profile(args.repeats)
    report = {
        "word_mul": _word_mul_profile(args.repeats),
        "element_operator_construction": element,
        "word_union_qwc_grouping": _grouping_profile(bank, args.repeats),
        "repeated_projected_solve": _projected_solve_profile(bank, args.repeats),
        **_matvec_profiles(args.repeats, args.sites),
        "bootstrap_ritz": _bootstrap_profile(args.bootstrap_replicates),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
