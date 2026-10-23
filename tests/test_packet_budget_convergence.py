"""Budget-convergence arms, runner and checker on toy inputs, never campaign inputs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import check_packet_budget_convergence as check
from benchmarks import run_packet_budget_convergence as run
from benchmarks import summarize_packet_budget_convergence as summary
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.subspace import sparse_baselines as baselines
from clifford_qc.subspace import sparse_packets as packets

TOY = {'name': 'toy_2x2', 'kind': 'hubbard', 'model': {'shape': [2, 2], 't': 1., 'U': 4., 'mu': 0., 'periodic': False},
       'n_qubits': 8, 'n_electrons': 4, 'sector_dimension': 36, 'singlet_dimension': 20,
       'active_spatial_orbitals': [1, 2], 'maximum_columns': 20, 'maximum_growth_steps': 17}
TOY_MOLECULE = {'name': 'toy_beh2', 'kind': 'fcidump', 'path': 'benchmarks/data/beh2_sto3g_r1.3264.FCIDUMP',
                'n_qubits': 8, 'n_electrons': 4, 'sector_dimension': 36, 'singlet_dimension': 20,
                'active_spatial_orbitals': [1, 2], 'maximum_columns': 12, 'maximum_growth_steps': 9}


def config():
    return json.loads(Path(run.CONFIG).read_text())


def setup(case):
    model = run.build_input(case)
    backend = SectorStatevectorBackend(case['n_qubits'], case['n_electrons'], 0)
    h = backend.operator(model.hamiltonian).to_csr()
    frame = packets.SparseSingletFrame(backend, tolerance=config()['spin_tolerance'])
    return h, frame, sum(1 << (model.n-1-j) for j in range(case['n_electrons'])), backend.dimension


@pytest.fixture(scope='module')
def produced():
    """Every arm's scientific payload and independent replay data on both toys."""
    c, out = config(), {}
    for case in (TOY, TOY_MOLECULE):
        h, frame, hf, dimension = setup(case)
        data = check.independent_data(case, c)
        reference = check.independent_reference(case, c, data)
        for arm in c['arms']:
            observer = run.PrefixObserver(dimension, c['support_tolerance'])
            result = run.run_arm(arm, h, frame, hf, case, c, observer)
            science = json.loads(run.canonical(run.scientific_result(result, arm, observer)))
            out[case['name'], arm['name']] = science, result, case, data, reference
    return out


def test_design_schedules_declared_cells_and_arms():
    c = config()
    order = run.expected_order(c)
    assert len(order) == 52 == sum(len(c['arms'])*case['repeats'] for case in c['inputs'])
    for case in c['inputs']:
        for repeat in range(case['repeats']):
            arms = [t['arm'] for t in order if t['input'] == case['name'] and t['repeat'] == repeat]
            assert sorted(arms) == sorted(a['name'] for a in c['arms'])
    assert order == run.expected_order(c)


@pytest.mark.parametrize('case', [TOY, TOY_MOLECULE], ids=lambda case: case['name'])
@pytest.mark.parametrize('arm', [a['name'] for a in config()['arms']])
def test_independent_replay_accepts_every_arm(produced, case, arm):
    science, _, case, data, _ = produced[case['name'], arm]
    assert check.replay(science, case, config(), data, full=True) == []


@pytest.mark.parametrize('arm', ['packets_exhaustive', 'packets_early'])
def test_bounded_replay_checks_selected_candidates_between_samples(produced, arm, monkeypatch):
    science, _, case, data, _ = produced['toy_2x2', arm]
    monkeypatch.setattr(check, 'FULL_REPLAY_DIMENSION', 0)
    monkeypatch.setattr(check, 'SAMPLE_STRIDE', 10**6)
    assert len(science['decisions']) > 5  # Some steps fall outside the sample.
    assert check.replay(science, case, config(), data) == []
    altered = deepcopy(science)
    altered['decisions'][3]['score'] += 1e-6
    assert 'selected score' in check.replay(altered, case, config(), data)


@pytest.mark.parametrize('arm,mutate,problem', [
    ('selected_ci', lambda s: s['decisions'][1].update(selected=s['decisions'][1]['selected']+1), 'exact recorded tie-break'),
    ('selected_ci', lambda s: s['decisions'][2]['near_ties'].clear(), 'missing near-tie record'),
    ('packets_exhaustive', lambda s: s['trajectory'][2].update(h_action_calls=s['trajectory'][2]['h_action_calls']+1), 'prefix work'),
    ('packets_early', lambda s: s['decisions'][0].update(admissible=s['decisions'][0]['admissible']-1), 'admissible count'),
    ('davidson', lambda s: s['trajectory'][1].update(energy=s['trajectory'][1]['energy']-1e-6), 'energy'),
    ('davidson', lambda s: s['support'][-1].__setitem__(0, s['support'][-1][0]+10), 'span support'),
    ('packets_exhaustive', lambda s: s.update(final_state=run.encode_array(np.roll(run.decode_array(s['final_state']), 1))), 'final Ritz state'),
    ('selected_ci', lambda s: s.update(stop_reason='column_ceiling'), 'stopping rule')])
def test_replay_rejects_altered_payloads(produced, arm, mutate, problem):
    science, _, case, data, _ = produced['toy_2x2', arm]
    altered = deepcopy(science)
    mutate(altered)
    assert problem in check.replay(altered, case, config(), data, full=True)


def test_support_counts_must_lie_within_one_decade_of_the_threshold():
    span = np.zeros((6, 2)); span[:3, 0] = 1.; span[3, 1], span[4, 1], span[5, 1] = 1., 5e-12, 5e-14
    rows = [{'dimension': 1}, {'dimension': 2}]

    def problems(support):
        found = []
        check.support_problems({'support': support}, span, rows, config(), lambda ok, name: ok or found.append(name))
        return found
    assert problems([[3, 3], [4, 4]]) == problems([[3, 3], [5, 5]]) == []  # 5e-12 sits inside the band.
    assert problems([[3, 3], [6, 6]]) == ['span support', 'amplitude nonzeros']  # 5e-14 is below it.
    assert problems([[2, 3], [4, 4]]) == ['span support']  # No band entries: exact equality.


def test_selected_ci_adds_the_best_column_and_completes_the_singlet_span():
    h, frame, hf, _ = setup(TOY)
    c = {**config(), **TOY, 'residual_tolerance': 1e-300}  # Grow to the full singlet span.
    result = baselines.run_selected_ci(h.dot, h.diagonal().real, frame, hf, TOY['active_spatial_orbitals'], c)
    assert result['stop_reason'] == 'column_ceiling' and result['trajectory'][-1]['dimension'] == 20
    exact = np.linalg.eigvalsh((frame.matrix.conj().T @ (h @ frame.matrix)).toarray())[0]
    assert abs(result['trajectory'][-1]['energy']-exact) < 1e-10
    assert result['materialized_candidates'] == 17 and result['final_h_action_calls'] == 3+17
    for decision in result['decisions']:
        best = max(decision['near_ties'].values())
        assert decision['selected'] == min(int(k) for k, v in decision['near_ties'].items() if v >= best-1e-12)


def test_davidson_adds_the_frame_projected_preconditioned_residual():
    h, frame, hf, _ = setup(TOY_MOLECULE)
    c = {**config(), **TOY_MOLECULE}
    taken = []
    original = baselines.orthogonal_direction

    def recording(vector, basis, **kwargs):
        taken.append((np.array(vector), basis.copy())); return original(vector, basis, **kwargs)

    baselines.orthogonal_direction = recording
    try:
        result = baselines.run_singlet_davidson(h.dot, h.diagonal().real, frame, hf, [1, 2], c)
    finally:
        baselines.orthogonal_direction = original
    vector, basis = taken[0]
    space = packets.ActionSpace(h.dot, basis)
    energy, _, residual = space.solve()
    raw = residual/np.maximum(h.diagonal().real-energy+c['preconditioner_shift'], c['preconditioner_shift'])
    np.testing.assert_allclose(vector, frame.matrix @ (frame.matrix.conj().T @ raw), atol=1e-12)
    assert result['final_h_action_calls'] == 3+len(result['decisions'])


def test_observer_hook_never_changes_packet_science():
    h, frame, hf, dimension = setup(TOY)
    c = {**config(), **TOY}
    plain = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], c, cache_mode='combined')
    observer = run.PrefixObserver(dimension, c['support_tolerance'])
    seen = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], c, cache_mode='combined', observe=observer)
    strip = lambda r: [{k: v for k, v in row.items() if k != 'wall_seconds'} for row in r['trajectory']]
    assert strip(plain) == strip(seen) and plain['decisions'] == seen['decisions'] and 'observations' not in plain
    assert len(seen['observations']) == len(seen['trajectory'])
    assert [o['support'][1] for o in seen['observations']] == sorted(o['support'][1] for o in seen['observations'])


def test_compact_decisions_keep_selection_evidence_and_arrays_round_trip():
    raw = {'step': 0, 'indicators': {'1': 2., '2': 1.}, 'shortlist': [1, 2], 'attempted': [1, 2],
           'scores': {'1': .5, '2': .5-1e-9, '3': .1}, 'selected': 1, 'audit': False, 'fallback': False,
           'audit_missed_winner': False, 'audit_score_loss': None}
    compact = run.compact_decisions({'kind': 'packets', 'policy': 'early'}, [raw])[0]
    assert compact['near_ties'] == {'1': .5, '2': .5-1e-9} and compact['admissible'] == 3 and compact['shortlist'] == [1, 2]
    assert 'shortlist' not in run.compact_decisions({'kind': 'packets', 'policy': 'exhaustive'}, [raw])[0]
    values = np.exp(1j*np.arange(7.)) / 3
    np.testing.assert_array_equal(run.decode_array(run.encode_array(values)), values)


def test_independent_reference_cross_checks_the_sector_ground_state():
    reference = check.independent_reference(TOY, config())
    assert reference['sector_ground_is_singlet'] and abs(reference['singlet_reference']-reference['sector_ground']) < 1e-9


def test_readouts_follow_the_declared_rules():
    def science(errors, calls):
        return {'trajectory': [{'dimension': 64*(i+1), 'energy': e, 'h_action_calls': w} for i, (e, w) in enumerate(zip(errors, calls))]}
    better, worse = science([.5, .05, .009], [10, 20, 30]), science([.6, .2, .05], [5, 10, 15])
    assert summary.hit_columns(better, 0., .1) == 128 and summary.hit_columns(worse, 0., .01) == float('inf')
    assert summary.compactness(better, worse, 0., [.1, .01], [128, 192], .8)
    assert not summary.compactness(worse, better, 0., [.1, .01], [128, 192], .8)
    assert summary.frontier_error(better, 0., 150) == .05
    better['support'] = [[10, 10], [30, 40], [50, 90]]
    assert summary.support_at(better, 150) == 30 and summary.support_at(better, 192) == 50


def test_early_readout_names_the_failing_clause_and_the_sign():
    d = config()['decisions']
    def science(rows):
        return {'trajectory': [{'dimension': k, 'energy': e} for k, e in rows]}
    exhaustive = science([(64, 1.), (128, .1), (256, .01)])
    assert summary.early_readout(exhaustive, science([(64, .95), (128, .1), (256, .0105)]), 0., [1., .1], [128, 256], d) == '**preserves accuracy**.'
    better = summary.early_readout(exhaustive, science([(64, 1.), (128, .05), (256, .01)]), 0., [1., .1], [128, 256], d)
    assert 'outside the declared band' in better and 'at 128' in better and '-0.05 at 128' in better and 'columns above' not in better
    slower = summary.early_readout(exhaustive, science([(64, 1.), (128, 1.), (256, .1)]), 0., [1., .1], [128, 256], d)
    assert 'columns above 1.1× exhaustive at ≤0.1' in slower
    rounding = summary.early_readout(science([(64, 1e-13)]), science([(64, 3e-13)]), 0., [1.], [64], d)
    assert 'Both errors are within 1e-08 of the reference at 64.' in rounding


def test_davidson_note_reports_a_stop_above_the_reference_outside_the_declared_trigger():
    d = config()['decisions']
    def science(stop, energy):
        return {'stop_reason': stop, 'trajectory': [{'energy': energy, 'residual_norm': 9e-9}]}
    note = summary.davidson_note(science('no_admissible_direction', 1.), 0., 1e-12, d)
    assert 'excludes the ground state' in note and 'declared fallback' in note
    assert 'excludes' not in summary.davidson_note(science('column_ceiling', 1.), 0., .9, d)
    assert summary.davidson_note(science('no_admissible_direction', 1e-12), 0., 1., d) == ''
    assert summary.davidson_note(science('residual_tolerance', 1.), 0., 0., d) == ''  # The declared verdict covers it.


def test_report_renders_every_declared_readout_from_toy_cells(produced):
    import base64
    import gzip
    c = config()
    roles = {'toy_2x2': 'primary', 'toy_beh2': 'complete_span_lattice_control'}
    c['inputs'] = [{**case, 'role': roles[case['name']], 'repeats': 1, 'energy_unit': 'Ha'} for case in (TOY, TOY_MOLECULE)]
    workers, payloads, overlaps, references = [], {}, {}, {}
    for (name, arm), (science, result, case, data, reference) in produced.items():
        raw = run.canonical(science); compressed = gzip.compress(raw, mtime=0)
        payloads[run.digest(raw)] = {'sha256': run.digest(raw), 'gzip_sha256': run.digest(compressed),
                                     'encoding': 'gzip+base64', 'payload': base64.b64encode(compressed).decode()}
        workers.append({'input': name, 'arm': arm, 'repeat': 0, 'status': 'ok', 'scientific_sha256': run.digest(raw),
                        'setup_seconds': .1, 'solver_seconds': result['solver_wall_seconds'],
                        'prefix_wall_seconds': [row['wall_seconds'] for row in result['trajectory']],
                        'worker_peak_rss_bytes': 2**27})
        overlaps[run.digest(raw)] = check.final_overlap(science, reference)
        references[name] = {k: v for k, v in reference.items() if k != 'ground_state'}
    record = {'config': c, 'workers': workers, 'scientific_payloads': payloads, 'references': references,
              'final_ground_state_overlaps': overlaps, 'declaration_commit': 'declared', 'source_commit': 'executed'}
    text = summary.render(record)
    for line in ('Readout 1, column compactness', 'Readout 3, early screening', 'Readout 4, Davidson',
                 'Complete-span control, selected_ci', 'Packets add column compactness', "Davidson's last prefix"):
        assert line in text
    assert 'uninformative (confined)' in text  # The symmetric toy lattice traps lowest-root Davidson.
