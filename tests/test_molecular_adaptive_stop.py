"""Small development-only acquisition checks, separate from evaluation seeds."""
import math

import numpy as np
import pytest

pytest.importorskip('scipy')

from benchmarks import run_molecular_adaptive_stop as producer
from benchmarks.check_molecular_adaptive_stop import bounds, check_row, summarize
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.multivector import MV
from clifford_qc.subspace import MatrixElementBank
from clifford_qc.subspace.symmetry_shifts import sector_shift_family


def test_audited_adaptive_paths(monkeypatch):
    state = np.zeros(16, complex)
    state[12], state[3] = math.cos(.3), np.exp(.2j)*math.sin(.3)
    rho = producer.fixed.two_det_density({'reference_bits': ['1100', '0011'], 'n': 4}, state)
    t = cdag_op(4, 2)*cdag_op(4, 3)*c_op(4, 1)*c_op(4, 0)
    h = MV.from_terms(4, {'ZIII': .3, 'IZII': .3, 'IIZI': -.2, 'IIIZ': -.2}) + .2*(t+t.dagger())
    shared = SharedMeasurement(MatrixElementBank(rho, h, [('I', MV.scalar(4, 1)), ('G', t-t.dagger())]))
    family = sector_shift_family(2, 2, one_body=True, sz=0.)
    backend = FiniteShotBackend(seed=8)
    backend.prepare_grouped_state(shared.bank.reference, shared.groups)
    system = {'groups': len(shared.groups), 'rank': 2, 'shift_labels': [t.label for t in family]}
    monkeypatch.setitem(producer.SYSTEMS, 0, (shared, family, backend, system))
    d = {'seed_root': 17, 'cumulative_average_shots_per_group': [100, 200, 400],
         'target_std_error_Ha': 1e-8, 'delta': .05, 'bootstrap_replicas': 4,
         'arms': [{'key': 'raw', 'allocation': 'ritz_targeted', 'control_variates': False},
                  {'key': 'cv', 'allocation': 'ritz_targeted', 'control_variates': True}]}
    previous = {'solver': {}}
    rows = [producer.run_arm(d, previous, 0, 0, arm) for arm in d['arms']]
    for row, arm in zip(rows, d['arms'], strict=True):
        assert not row['failure']
        check_row(row, d, previous, system, arm)
        assert bounds(row, 'bootstrap_spread_schedule', .95, d) is not None
    assert rows[0]['pilot_fingerprint'] == rows[1]['pilot_fingerprint']
    early = {**d, 'target_std_error_Ha': 100.}
    stopped = producer.run_arm(early, previous, 0, 0, early['arms'][0])
    assert stopped['stopped'] == 'target_met' and len(stopped['looks']) == 1
    check_row(stopped, early, previous, system, early['arms'][0])
    with pytest.raises(ValueError, match='total cost'):
        check_row({**stopped, 'shots': stopped['shots']+1}, early, previous, system, early['arms'][0])


def test_failures_and_exhaustion_count_in_coverage_denominator():
    d = {'arms': [{'key': 'cv'}], 'levels': [.95], 'cumulative_average_shots_per_group': [1, 2],
         'target_std_error_Ha': .1, 'delta': .05,
         'primary_test': {'arm': 'cv', 'method': 'bootstrap_spread_schedule', 'level': .95,
                          'cells': 1, 'familywise_alpha': .05}}
    entry = {'case': 0, 'arm': 'cv', 'failure': None, 'estimate': 0., 'stopped': 'target_met',
             'shots': 10, 'batches': [{}], 'looks': [{'retained_rank': 1}],
             'naive_interval': {'lower': -.2, 'upper': .2}, 'bootstrap_interval': {'std_error': .1}}
    rows = [entry, {**entry, 'stopped': 'schedule_exhausted', 'shots': 20},
            {**entry, 'failure': 'test failure', 'stopped': 'failed', 'shots': 20}]
    summaries, _ = summarize({'rows': rows}, d, [{'name': 'toy'}], [{'independent_target': 0., 'rank': 1}])
    assert all(s['covered'] == 2 and s['total'] == 3 and s['covered_and_target_met'] == 1 for s in summaries)
    assert all(s['median_shots'] == 20 and s['total_shots'] == 50 for s in summaries)
