"""Development-only cross-checks of the larger-bank measurement accelerations."""
import itertools

import numpy as np
import pytest

pytest.importorskip('scipy')

from benchmarks import run_stretched_molecular_stop as runner
from benchmarks.check_stretched_molecular_stop import occupation_hamiltonian
from benchmarks.check_cross_domain_ratio import _pauli_restriction
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.models.fcidump import model_from_fcidump
from clifford_qc.multivector import MV
from clifford_qc.subspace import MatrixElementBank


def test_two_det_qwc_probabilities_match_mv_all_small_settings():
    rho, sparse = runner.reference({'n': 4, 'reference_bits': ['1100', '0011'], 'theta': .3, 'phi': .2})
    backend, native = runner.TwoDetQWCBackend(sparse), FiniteShotBackend(0)
    # Includes coherent X/Y settings, Z-only and partial traces.
    for label in map(''.join, itertools.product('IXYZ', repeat=4)):
        if label == 'IIII':
            continue
        word = PauliWord.from_label(label)
        a, b = backend._plan(rho, [word], {}), native._plan(rho, [word], {})
        p, q = dict(zip(a.bits, a.probs)), dict(zip(b.bits, b.probs))
        for key in p.keys() | q.keys():
            assert abs(p.get(key, 0.)-q.get(key, 0.)) < 1e-12


def test_integer_prefill_matches_native_point_and_bootstrap():
    rho, sparse = runner.reference({'n': 4, 'reference_bits': ['1100', '0011'], 'theta': .3, 'phi': .2})
    t = cdag_op(4, 2)*cdag_op(4, 3)*c_op(4, 1)*c_op(4, 0)
    h = MV.from_terms(4, {'ZIII': .3, 'IZII': .3, 'IIZI': -.2, 'IIIZ': -.2}) + .2*(t+t.dagger())
    bank = MatrixElementBank(rho, h, [('I', MV.scalar(4, 1)), ('G', t-t.dagger())])
    native = SharedMeasurement(bank)
    fast = runner.HistogramSharedMeasurement(bank, groups=native.groups)
    backend = runner.TwoDetQWCBackend(sparse, seed=7)
    cache = native.measure(backend, 100)
    expected = native.matrices(cache)
    cache._word_sums.clear()
    actual = fast.matrices(cache)
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual))
    from clifford_qc.subspace.uncertainty import bootstrap_ritz_samples
    first = bootstrap_ritz_samples(native, cache, replicates=4, seed=13)
    second = bootstrap_ritz_samples(fast, cache, replicates=4, seed=13)
    assert np.array_equal(first, second)
    assert len(fast.solve_trace) == 4


def test_frozen_core_energy_agrees_with_parent_occupation_projection():
    # Small actual LiH sector; does not generate evaluation outcomes.
    case = runner.declaration()['cases'][0]
    words, matrix = occupation_hamiltonian(case)
    h = model_from_fcidump(runner.active_integrals(case)).hamiltonian.to_mv()
    assert np.allclose(matrix, _pauli_restriction(h, words), atol=1e-9, rtol=0)
