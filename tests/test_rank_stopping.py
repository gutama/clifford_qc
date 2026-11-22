"""Measured rank guards: normalization derivatives, stop and cost semantics."""
import numpy as np
import pytest

from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.multivector import MV
from clifford_qc.states import ket_density
from clifford_qc.subspace import MatrixElementBank
from clifford_qc.subspace.acquisition import RankStopPolicy, acquire_ritz
from clifford_qc.subspace.linalg import normalize_overlap
from clifford_qc.subspace.rank_stopping import normalized_mode_functional


def bank():
    ident = MV.scalar(2, 1)
    x = MV.from_terms(2, {'XI': 1.})
    # Unequal norms and a weak but nonzero direction. Z on the second qubit
    # makes the norms/overlap read real measured word means.
    mixed = ident+.3*x+.2*MV.from_terms(2, {'IZ': 1.})
    h = MV.from_terms(2, {'ZI': .5, 'XI': .3, 'IZ': .1})
    return SharedMeasurement(MatrixElementBank(ket_density(2, '00'), h, [('I', ident), ('m', mixed)]))


def run(shared, **kwargs):
    return acquire_ritz(shared, FiniteShotBackend(19), pilot_shots_per_group=400,
                        round_budgets=[1500, 3000], target_std_error=10., **kwargs)


def test_normalized_mode_derivative_matches_finite_difference():
    shared = bank()
    cache = shared.measure(FiniteShotBackend(5), 500)
    s, _ = shared.matrices(cache)
    frame = normalize_overlap(s)
    values, vectors = np.linalg.eigh(frame.matrix)
    f = normalized_mode_functional(shared, frame, values[0], vectors[:, 0])
    assert f.coefficients
    means = {w.code: (2*cache._group_word_sums(cache.group_key(w.code), w.code)[0]
                      /cache.shots(w.code)-1) for w in shared.words}
    # Independently perturb word means and rebuild the normalized eigenvalue.
    for code, slope in f.coefficients.items():
        def at(epsilon):
            def value(functional):
                return functional.constant+sum(c*(means[k]+(epsilon if k == code else 0.))
                                                for k, c in functional.coefficients.items())
            overlap, _ = shared._assemble(value)
            return np.linalg.eigvalsh(normalize_overlap(overlap).matrix)[0]
        assert (at(1e-6)-at(-1e-6))/2e-6 == pytest.approx(slope, abs=1e-8)


def test_full_rank_guard_needs_two_resolved_looks():
    shared = bank()
    found = run(shared, rank_stop=RankStopPolicy(), solver_kwargs={'tau_s': .001})
    assert found.stopped == 'target_met'
    assert len(found.looks) == 2
    assert not found.looks[0]['rank_guard']['passed']
    assert found.looks[1]['rank_guard']['passed']
    assert found.shots == 400*len(shared.groups)+1500


def test_stable_truncated_rank_cannot_qualify_and_budget_is_spent():
    shared = bank()
    found = run(shared, rank_stop=RankStopPolicy(), solver_kwargs={'tau_s': .8})
    assert found.stopped == 'schedule_exhausted'
    assert found.shots == 400*len(shared.groups)+4500
    assert all(l['retained_rank'] == 1 and not l['rank_guard']['passed'] for l in found.looks)
    assert [l['next_rank_budget'] for l in found.looks[:-1]] == [1500, 3000]


def test_bootstrap_width_continues_and_reuses_terminal_draws(monkeypatch):
    from clifford_qc.subspace import acquisition
    calls = []
    def draws(*args, **kwargs):
        calls.append(kwargs['seed'])
        return [-10., 10.] if len(calls) == 1 else [0., 0.]
    monkeypatch.setattr(acquisition, 'bootstrap_ritz_samples', draws)
    found = run(bank(), bootstrap_replicates=2, bootstrap_seed=4, max_bootstrap_half_width=.01)
    assert found.stopped == 'target_met' and len(found.looks) == len(calls) == 2
    assert calls[0] != calls[1]
    assert [l['bootstrap_width_passed'] for l in found.looks] == [False, True]
    assert found.bootstrap_interval.std_error == 0.


def test_bootstrap_failure_is_not_a_pass(monkeypatch):
    from clifford_qc.subspace import acquisition
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs['seed'])
        raise ValueError('development failure')
    monkeypatch.setattr(acquisition, 'bootstrap_ritz_samples', fail)
    found = run(bank(), bootstrap_replicates=2, max_bootstrap_half_width=.01)
    assert found.stopped == 'schedule_exhausted' and found.bootstrap_interval is None
    assert len(calls) == len(found.looks) == 3
    assert all(not l['bootstrap_width_passed'] and l['bootstrap_failure'] for l in found.looks)


def test_combined_guard_does_not_bootstrap_before_rank_confirmation(monkeypatch):
    from clifford_qc.subspace import acquisition
    calls = []
    def draws(*args, **kwargs):
        calls.append(kwargs['seed'])
        return [0., 0.]
    monkeypatch.setattr(acquisition, 'bootstrap_ritz_samples', draws)
    found = run(bank(), rank_stop=RankStopPolicy(), solver_kwargs={'tau_s': .001},
                bootstrap_replicates=2, max_bootstrap_half_width=.01)
    assert found.stopped == 'target_met' and len(found.looks) == 2 and len(calls) == 1
    assert 'bootstrap_seed' not in found.looks[0]
    assert found.looks[-1]['bootstrap_width_passed']


@pytest.mark.parametrize('options', [{'rank_stop': True}, {'rank_stop': RankStopPolicy(4)},
                                     {'rank_stop': RankStopPolicy(), 'solver_kwargs': {'calibrate_overlap': True}},
                                     {'max_bootstrap_half_width': .01}, {'max_bootstrap_half_width': float('nan')}])
def test_bad_guard_options_preflight(options):
    class NoShots:
        def sample_grouped_from_state(self, *args, **kwargs):
            pytest.fail('invalid policy consumed shots')
    with pytest.raises(ValueError):
        acquire_ritz(bank(), NoShots(), pilot_shots_per_group=4, round_budgets=[10, 10],
                     target_std_error=.1, **options)
