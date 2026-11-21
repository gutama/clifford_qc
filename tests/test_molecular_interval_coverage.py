"""Development-only checks; the preregistered molecular seeds are never drawn here."""
import math

import numpy as np
import pytest

pytest.importorskip('scipy')

from benchmarks import run_molecular_interval_coverage as producer
from benchmarks.check_molecular_interval_coverage import interval, summarize
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.multivector import MV
from clifford_qc.subspace import MatrixElementBank
from clifford_qc.subspace.symmetry_shifts import certify_bank_family, sector_shift_family
from clifford_qc.subspace.uncertainty import bootstrap_ritz_samples


def test_paired_bootstrap_matches_both_public_paths():
    state = np.zeros(16, complex)
    state[12], state[3] = math.cos(.3), np.exp(.2j)*math.sin(.3)
    rho = producer.two_det_density({'reference_bits': ['1100', '0011'], 'n': 4}, state)
    t = cdag_op(4, 2)*cdag_op(4, 3)*c_op(4, 1)*c_op(4, 0)
    h = MV.from_terms(4, {'ZIII': .3, 'IZII': .3, 'IIZI': -.2, 'IIIZ': -.2}) + .2*(t+t.dagger())
    shared = SharedMeasurement(MatrixElementBank(rho, h, [('I', MV.scalar(4, 1)), ('G', t-t.dagger())]))
    family = sector_shift_family(2, 2, one_body=True, sz=0.)
    certify_bank_family(shared.bank, family)
    cache = shared.measure(FiniteShotBackend(seed=11), 300)
    point = shared.solve(cache)
    draws = producer.paired_bootstrap(shared, cache, point, family, 8, 23)
    for arm, control in [('raw', None), ('refitted_cv', family)]:
        values, failure = draws[arm]
        assert failure is None
        expected = bootstrap_ritz_samples(shared, cache, result=point, replicates=8, seed=23,
                                          control_variates=control)
        assert np.allclose(values, expected, atol=1e-12, rtol=0)


def test_missing_interval_counts_as_uncovered():
    d = {'arms': ['raw', 'refitted_cv'], 'intervals': ['delta_normal', 'bootstrap_spread'],
         'levels': [.95], 'shots_per_qwc_group': [100],
         'primary_test': {'cells': 1, 'familywise_alpha': .05}}
    entry = {'estimate': 0., 'delta_scale': .1, 'delta_failure': None, 'draws': [-.1, .1]}
    rows = [{'case': 0, 'budget': 0, 'arms': {'raw': entry, 'refitted_cv': entry}},
            {'case': 0, 'budget': 0, 'arms': {'raw': entry, 'refitted_cv': None}}]
    summaries, _ = summarize({'rows': rows}, d, [{'name': 'toy'}], [{'independent_target': 0.}])
    assert all(r['total'] == 2 for r in summaries)
    assert all(r['coverage'] == .5 for r in summaries if r['arm'] == 'refitted_cv')
    assert interval({**entry, 'draws': None}, 'bootstrap_spread', .95) is None
