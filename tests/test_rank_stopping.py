"""Measured rank guards: normalization derivatives, stop and cost semantics."""
import numpy as np
import pytest

from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.multivector import MV
from clifford_qc.states import ket_density
from clifford_qc.subspace import MatrixElementBank, acquire_ritz_precision
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


def precision(shared, **kwargs):
    return acquire_ritz_precision(shared, FiniteShotBackend(19), pilot_shots_per_group=400,
                                  round_budgets=[1500, 3000], target_std_error=10., **kwargs)


def test_primary_controller_matches_explicit_rank_policy_without_bootstrap(monkeypatch):
    from clifford_qc.subspace import acquisition
    def unexpected(*args, **kwargs):
        pytest.fail('rank-only primary controller must not run bootstrap')
    monkeypatch.setattr(acquisition, 'bootstrap_ritz_samples', unexpected)
    shared = bank()
    primary = precision(shared, solver_kwargs={'tau_s': .001})
    explicit = run(shared, rank_stop=RankStopPolicy(), solver_kwargs={'tau_s': .001})
    assert primary.looks == explicit.looks and primary.schedule == explicit.schedule
    assert primary.result.measurement_fingerprint == explicit.result.measurement_fingerprint
    assert primary.interval == explicit.interval
    assert primary.bootstrap_interval is None
    assert primary.stopped == 'target_met' and len(primary.looks) == 2
    # Historical API remains SE-only if no rank policy is explicitly given.
    historical = run(shared, solver_kwargs={'tau_s': .001})
    assert len(historical.looks) == 1 and 'rank_stop' not in historical.schedule


def test_primary_controller_abstains_on_stable_reduced_rank():
    found = precision(bank(), solver_kwargs={'tau_s': .8})
    assert found.stopped == 'schedule_exhausted'
    assert len(found.looks) == 3
    assert all(l['retained_rank'] == 1 and not l['rank_guard']['passed'] for l in found.looks)


def test_primary_controller_width_is_explicit_and_shares_rank_prefix(monkeypatch):
    from clifford_qc.subspace import acquisition
    calls = []
    def draws(*args, **kwargs):
        calls.append(kwargs['seed'])
        return [-10., 10.] if len(calls) == 1 else [0., 0.]
    monkeypatch.setattr(acquisition, 'bootstrap_ritz_samples', draws)
    shared = bank()
    rank_only = precision(shared, solver_kwargs={'tau_s': .001})
    width = precision(shared, solver_kwargs={'tau_s': .001},
                      max_bootstrap_half_width=.01, bootstrap_replicates=2, bootstrap_seed=29)
    assert rank_only.bootstrap_interval is None
    assert width.stopped == 'target_met' and len(width.looks) == 3
    assert len(calls) == 2 and calls[0] != calls[1]
    assert 'bootstrap_seed' not in width.looks[0]
    assert not width.looks[1]['bootstrap_width_passed'] and width.looks[2]['bootstrap_width_passed']
    for plain, guarded in zip(rank_only.looks, width.looks):
        assert plain['energy'] == guarded['energy'] and plain['rank_guard'] == guarded['rank_guard']
    assert width.shots == rank_only.shots+3000


def test_primary_actual_terminal_bootstrap_diagnostic_vs_width_guard():
    shared = bank()
    diagnostic = precision(shared, solver_kwargs={'tau_s': .001}, bootstrap_replicates=8)
    guarded = precision(shared, solver_kwargs={'tau_s': .001}, bootstrap_replicates=8,
                        max_bootstrap_half_width=10.)
    assert diagnostic.stopped == guarded.stopped == 'target_met'
    assert diagnostic.bootstrap_interval is not None and guarded.bootstrap_interval is not None
    assert diagnostic.shots == guarded.shots
    assert 'bootstrap_width_passed' not in diagnostic.looks[-1]
    assert guarded.looks[-1]['bootstrap_width_passed']


@pytest.mark.parametrize('options', [{'rank_stop': None}, {'rank_stop': 'se_only'},
                                     {'max_bootstrap_half_width': .0016},
                                     {'rank_stop': RankStopPolicy(4)},
                                     {'solver_kwargs': {'calibrate_overlap': True}}])
def test_primary_controller_refuses_missing_guard_or_budget_before_sampling(options):
    class NoShots:
        def sample_grouped_from_state(self, *args, **kwargs):
            pytest.fail('invalid primary policy consumed shots')
    with pytest.raises(ValueError):
        acquire_ritz_precision(bank(), NoShots(), pilot_shots_per_group=4,
                               round_budgets=[10, 10], target_std_error=.1, **options)
