"""Exact-cache invariants on toy inputs, independent of held-out campaign runs."""
from copy import deepcopy
import base64
import gzip
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from benchmarks.metadata_io import read_json
from benchmarks.run_exact_projection_cache import canonical, decode_science, digest, expected_order, scientific_result
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import sparse_packets as packets
from clifford_qc.subspace.action_space import ActionSpace, orthogonal_direction


def config():
    c = read_json(Path('benchmarks/configs/exact_projection_cache.json'))
    return {**c, 'maximum_columns': 8, 'maximum_growth_steps': 5}


@pytest.fixture
def physical():
    backend = SectorStatevectorBackend(8, 4, 0)
    h = backend.operator(hubbard((2, 2), U=4.).hamiltonian).to_csr()
    frame = packets.SparseSingletFrame(backend)
    hf = sum(1 << (7-j) for j in range(4))
    return h, frame, hf


@pytest.mark.parametrize('policy', ['exhaustive', 'early'])
@pytest.mark.parametrize('mode', ['none', 'basis', 'frame', 'combined'])
def test_cached_growth_is_bitwise_identical_and_refreshes_each_span(physical, policy, mode, monkeypatch):
    h, frame, hf = physical
    baseline = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), policy=policy)
    builds = []
    original = packets._StepProjection.__init__

    def init(self, basis):
        builds.append(basis.shape[1]); original(self, basis)

    monkeypatch.setattr(packets._StepProjection, '__init__', init)
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(),
                                       policy=policy, cache_mode=mode)
    baseline['arm'] = result['arm'] = 'early' if policy == 'early' else 'sparse'
    assert scientific_result(result) == scientific_result(baseline)
    expected = [row['dimension'] for row in result['trajectory'][:-1]] if mode in ('basis', 'combined') else []
    assert builds == expected
    stats = result['cache']
    assert stats['basis_adjoint_builds'] == len(builds)
    assert stats['projected_candidates'] == (result['materialized_candidates'] if builds else 0)
    assert stats['peak_basis_adjoint_bytes'] == (16*h.shape[0]*max(builds) if builds else 0)
    if mode in ('frame', 'combined'):
        dense = frame.matrix.toarray()
        initial = result['trajectory'][0]['selection']['initial_csf_indices']
        assert stats['frame_nnz'] == np.count_nonzero(dense)
        assert stats['group_nnz'] == np.count_nonzero(np.delete(dense, initial, axis=1))
        expected_fixed = ((16+stats['index_itemsize'])*stats['frame_nnz']
                          + stats['indptr_itemsize']*(frame.shape[1]+1)
                          + (16+stats['group_index_itemsize'])*stats['group_nnz']
                          + stats['group_indptr_itemsize']*(stats['group_columns']+stats['group_block_builds']))
        assert stats['fixed_owned_bytes'] == expected_fixed
    else:
        assert stats['fixed_owned_bytes'] == stats['group_block_builds'] == stats['frame_adjoint_builds'] == 0
    assert stats['peak_owned_bytes'] == stats['fixed_owned_bytes']+stats['peak_basis_adjoint_bytes']


def test_combined_path_skips_repeated_frame_slices_and_old_projection(physical, monkeypatch):
    h, frame, hf = physical
    monkeypatch.setattr(frame, 'combine', lambda *a: pytest.fail('uncached group slice used'))
    monkeypatch.setattr(frame, 'adjoint', lambda *a: pytest.fail('uncached adjoint used'))
    original = packets.orthogonal_direction

    def cached_only(vector, basis, *, cutoff, adjoint=None):
        if adjoint is None: pytest.fail('uncached span projection used')
        return original(vector, basis, cutoff=cutoff, adjoint=adjoint)

    monkeypatch.setattr(packets, 'orthogonal_direction', cached_only)
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode='combined')
    assert result['cache']['basis_adjoint_builds'] == len(result['decisions'])


@pytest.mark.parametrize('mode', ['none', 'basis', 'frame', 'combined'])
def test_old_span_adjoint_and_losing_candidates_are_released_before_extension(physical, mode, monkeypatch):
    h, frame, hf = physical
    projections, computed, extensions = [], [], []
    init, action, extend = packets._StepProjection.__init__, packets.ActionSpace.action, packets.ActionSpace.extend

    def tracked_init(self, basis):
        init(self, basis); projections.append(weakref.ref(self))

    def tracked_action(self, vector):
        result = action(self, vector); computed.append(weakref.ref(result)); return result

    def checked_extend(self, columns, *, actions=None):
        # Only the winning action may survive into the solver's extension peak.
        extensions.append((sum(ref() is not None for ref in projections),
                           sum(ref() is not None for ref in computed)))
        return extend(self, columns, actions=actions)

    monkeypatch.setattr(packets._StepProjection, '__init__', tracked_init)
    monkeypatch.setattr(packets.ActionSpace, 'action', tracked_action)
    monkeypatch.setattr(packets.ActionSpace, 'extend', checked_extend)
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode=mode)
    assert len(extensions) == len(result['decisions'])+1 and len(computed) > len(result['decisions'])
    assert all(live_projections == 0 and live_actions <= 1 for live_projections, live_actions in extensions)


def test_default_path_needs_only_the_public_frame_api(physical):
    h, frame, hf = physical
    public = SimpleNamespace(**{name: getattr(frame, name) for name in (
        'shape', 'n', 'representatives', 'spin', 'columns', 'adjoint', 'combine', 'active_columns')})
    expected = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config())
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, public, hf, [1, 2], config())
    assert scientific_result({**result, 'arm': 'sparse'}) == scientific_result({**expected, 'arm': 'sparse'})
    assert result['cache']['construction_wall_seconds'] == 0
    assert result['cache']['index_itemsize'] == result['cache']['indptr_itemsize'] == 0


def test_private_step_cache_uses_same_complex_projection_and_old_span_after_replacement():
    rng = np.random.default_rng(51)
    basis = np.linalg.qr(rng.normal(size=(9, 3))+1j*rng.normal(size=(9, 3)))[0]
    cache = packets._StepProjection(basis)
    vector = rng.normal(size=9)+1j*rng.normal(size=9)
    actual = cache.direction(vector, cutoff=1e-10)
    np.testing.assert_array_equal(actual, orthogonal_direction(vector, basis))
    assert not cache.adjoint.flags.writeable
    space = ActionSpace(lambda v: v, basis)
    old = packets._StepProjection(space.basis)
    direction = old.direction(vector, cutoff=1e-10)
    space.extend(direction[:, None])
    assert old.basis.shape[1] == 3  # ActionSpace replaced its basis without mutating this span.
    assert packets._StepProjection(space.basis).direction(direction, cutoff=1e-10) is None


@pytest.mark.parametrize('vector,cutoff', [(np.ones(2), 1e-9), (np.array([np.nan]*3), 1e-9),
                                         (np.ones(3), 0.), (np.ones(3), np.inf)])
def test_cached_projection_rejects_invalid_inputs(vector, cutoff):
    with pytest.raises(ValueError): packets._StepProjection(np.eye(3)[:, :1]).direction(vector, cutoff=cutoff)


def test_invalid_mode_is_rejected_before_setup(physical):
    h, frame, hf = physical
    with pytest.raises(ValueError, match='cache_mode'):
        packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode='approximate')


def test_checker_rejects_stale_generation_count_and_changed_selection(physical):
    from benchmarks.check_exact_projection_cache import cache_problems, compare_science
    h, frame, hf = physical
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode='combined')
    result['arm'] = 'sparse'
    science = scientific_result(result)
    case = {'n_qubits': 8, 'sector_dimension': 36, 'active_spatial_orbitals': [1, 2]}
    stats = deepcopy(result['cache'])
    assert cache_problems(stats, science, case, frame.matrix.toarray(), frame.words) == []
    stats['basis_adjoint_builds'] -= 1
    assert 'cache formula basis_adjoint_builds' in cache_problems(stats, science, case, frame.matrix.toarray(), frame.words)
    altered = deepcopy(science); altered['trajectory'][1]['selection'] = {'candidate': -1}
    assert 'cross-arm selection' in compare_science(altered, science, 36)


def test_scientific_content_address_detects_corruption_and_order_is_complete():
    raw = canonical({'trajectory': [1., 2.]}); compressed = gzip.compress(raw, mtime=0)
    payload = {'sha256': digest(raw), 'gzip_sha256': digest(compressed),
               'payload': base64.b64encode(compressed).decode()}
    assert decode_science(payload) == {'trajectory': [1., 2.]}
    altered = deepcopy(payload); altered['sha256'] = '0'*64
    with pytest.raises(ValueError, match='payload digest'): decode_science(altered)
    c = read_json(Path('benchmarks/configs/exact_projection_cache.json'))
    order = expected_order(c)
    assert len(order) == 84 and order == expected_order(c)
    assert len({tuple(sorted(task.items())) for task in order}) == 84


def test_worker_does_not_survey_records_or_accept_reference_energy(monkeypatch):
    from benchmarks import run_exact_projection_cache as producer
    from clifford_qc import record_environment
    c = config()
    c['inputs'] = [{'name': 'toy', 'kind': 'hubbard', 'model': {'shape': [2, 2], 't': 1., 'U': 4.,
                  'mu': 0., 'periodic': False}, 'n_qubits': 8, 'n_electrons': 4, 'sector_dimension': 36,
                  'active_spatial_orbitals': [1, 2], 'maximum_columns': 8, 'maximum_growth_steps': 5}]
    provenance = producer.execution_provenance()
    monkeypatch.setattr(producer, 'git', lambda *args: 'source' if args == ('rev-parse', 'HEAD') else '')
    monkeypatch.setattr(producer, 'stamp_record', lambda *a: pytest.fail('worker stamped/surveyed'))
    monkeypatch.setattr(record_environment, 'survey', lambda *a, **k: pytest.fail('worker surveyed'))
    science, metrics = producer.worker({'config': c, 'source_commit': 'source', 'source_sha256': {},
        'provenance': provenance}, {'input': 'toy', 'cache_mode': 'combined', 'policy': 'exhaustive'})
    assert 'wall_seconds' not in science['trajectory'][0]
    assert 'error' not in science['trajectory'][-1]
    assert metrics['setup_peak_rss_bytes'] >= metrics['imports_peak_rss_bytes']
    assert metrics['solver_end_peak_rss_bytes'] >= metrics['setup_peak_rss_bytes']


def test_checker_counts_cache_nonzeros_from_the_independent_frame(physical):
    from benchmarks.check_exact_projection_cache import cache_problems
    h, frame, hf = physical
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode='frame')
    science = scientific_result({**result, 'arm': 'sparse'})
    case = {'n_qubits': 8, 'sector_dimension': 36, 'active_spatial_orbitals': [1, 2]}
    stats = deepcopy(result['cache'])
    # A self-consistent but wrong report: one fewer nonzero, with matching bytes.
    stats['frame_nnz'] -= 1
    stats['fixed_owned_bytes'] -= 16+stats['index_itemsize']; stats['peak_owned_bytes'] -= 16+stats['index_itemsize']
    problems = cache_problems(stats, science, case, frame.matrix.toarray(), frame.words)
    assert 'cache sparse layout' in problems and 'cache formula fixed_owned_bytes' in problems


def published_record():
    from benchmarks.run_exact_projection_cache import RECORD
    return read_json(Path(RECORD))


def rejournal(record):
    record['journal_sha256'] = digest(b''.join(canonical(e)+b'\n' for e in record['journal']))
    return record


def test_published_record_passes_structural_checks():
    from benchmarks.check_exact_projection_cache import verify
    assert verify(published_record(), bound=False, numerical=False) == []


@pytest.mark.parametrize('field,value,problem', [
    ('cache', {'mode': 'none'}, 'historical control identity'),
    ('worker_environment_sha256', '0'*64, 'historical control identity'),
    ('scientific_gzip_sha256', '0'*64, 'historical control identity'),
    ('order_index', 0, 'historical control order'),
    ('cache_mode', 'none', 'historical control order')])
def test_checker_binds_each_historical_control(field, value, problem):
    from benchmarks.check_exact_projection_cache import verify
    record = published_record()
    record['historical_controls'][2][field] = value
    assert problem in verify(record, bound=False, numerical=False)


def test_checker_verifies_journaled_controls_when_present():
    from benchmarks.check_exact_projection_cache import verify
    record = published_record()
    late = max(e['utc_seconds'] for e in record['journal'])
    for control in record['historical_controls']:
        task = {k: control[k] for k in ('input', 'repeat', 'policy', 'cache_mode')}
        record['journal'] += [{'event': 'start', 'order_index': control['order_index'], 'task': task, 'utc_seconds': late},
                              {'event': 'finish', 'order_index': control['order_index'], 'worker': control, 'utc_seconds': late}]
    assert verify(rejournal(deepcopy(record)), bound=False, numerical=False) == []
    del record['journal'][-2:]  # The last control has neither a start nor a finish.
    assert 'partial historical control journal' in verify(rejournal(record), bound=False, numerical=False)


def test_checker_accepts_only_flagged_recoveries_without_supervisor_wall_time():
    from benchmarks.check_exact_projection_cache import verify
    record = published_record()
    worker = record['workers'][0]
    worker['process_wall_seconds'] = None
    next(e for e in record['journal'] if e['event'] == 'finish' and e['order_index'] == 0)['worker'] = worker
    assert 'finite timing process_wall_seconds' in verify(rejournal(deepcopy(record)), bound=False, numerical=False)
    worker['recovered'] = 'test'
    assert verify(rejournal(record), bound=False, numerical=False) == []


def test_unbound_check_does_not_claim_bindings(monkeypatch, capsys):
    from benchmarks import check_exact_projection_cache as checker
    monkeypatch.setattr(checker, 'verify', lambda record, bound: [])
    assert checker.main(['--unbound']) == 0
    assert 'bindings were not checked' in capsys.readouterr().out


def test_resume_recovers_durable_worker_metrics_and_never_reruns(tmp_path):
    from benchmarks.run_exact_projection_cache import recover
    task = {'input': 'toy', 'repeat': 0, 'policy': 'exhaustive', 'cache_mode': 'none'}
    (tmp_path/'cells'/'3').mkdir(parents=True); (tmp_path/'scientific').mkdir()
    assert recover(tmp_path, task, 3)['status'] == 'interrupted'
    compressed = gzip.compress(b'{}', mtime=0)
    (tmp_path/'scientific'/(digest(b'{}')+'.json.gz')).write_bytes(compressed)
    metrics = {'scientific_sha256': digest(b'{}'), 'scientific_gzip_sha256': digest(compressed), 'solver_seconds': 1.}
    (tmp_path/'cells'/'3'/'metrics.json').write_bytes(canonical(metrics))
    value = recover(tmp_path, task, 3)
    assert value['status'] == 'ok' and value['process_wall_seconds'] is None and value['recovered']
    assert value['solver_seconds'] == 1. and value['order_index'] == 3
    (tmp_path/'cells'/'3'/'metrics.json').write_bytes(canonical({**metrics, 'scientific_gzip_sha256': '0'*64}))
    assert recover(tmp_path, task, 3)['status'] == 'interrupted'


@pytest.mark.skipif(not Path('/proc/self/cmdline').exists(), reason='Linux /proc process table')
def test_resume_liveness_ignores_reused_pids():
    import subprocess
    import sys
    from benchmarks.run_exact_projection_cache import campaign_process
    sleep = 'import time; time.sleep(30)'
    other = subprocess.Popen([sys.executable, '-c', sleep])
    ours = subprocess.Popen([sys.executable, '-c', sleep, 'benchmarks/run_exact_projection_cache.py'])
    try:
        assert not campaign_process(other.pid) and campaign_process(ours.pid)
    finally:
        for process in (other, ours): process.kill(); process.wait()
    assert not campaign_process(ours.pid)


def test_preflight_refuses_evolved_historical_dependencies(monkeypatch):
    from benchmarks import run_exact_projection_cache as producer
    c = read_json(Path('benchmarks/configs/exact_projection_cache.json'))
    monkeypatch.setattr(producer, 'commit_binding', lambda root, commit, path: {'sha256': producer.sha256(root/path)})
    producer.historical_dependencies(c)
    monkeypatch.setattr(producer, 'commit_binding', lambda root, commit, path: {'sha256': '0'*64})
    with pytest.raises(ValueError, match='dependency evolved'): producer.historical_dependencies(c)


def test_report_promotion_reads_config_and_requires_every_input_policy():
    from benchmarks.summarize_exact_projection_cache import render
    record = published_record()
    record['config']['promotion']['maximum_median_rss_ratio'] = 2.
    exhaustive = next(row for row in record['config']['comparisons'] if row['policy'] == 'exhaustive')
    exhaustive['cache_modes'] = ['basis', 'none', 'frame', 'combined']  # 'none' is excluded by name.
    text = render(deepcopy(record))
    assert 'Primary combined speed/RSS clause: PASS' in text and 'Primary none' not in text
    assert 'Primary basis speed/RSS clause: PASS' in text
    # A held-out policy that changes its selections blocks promotion of the primary arms.
    worker = next(w for w in record['workers'] if w['input'] == 'hubbard_2x3_u8'
                  and w['policy'] == 'early' and w['cache_mode'] == 'combined')
    science = decode_science(record['scientific_payloads'][worker['scientific_sha256']])
    science['trajectory'][1]['selection'] = {'candidate': -1}
    raw = canonical(science); compressed = gzip.compress(raw, mtime=0)
    record['scientific_payloads'][digest(raw)] = {'sha256': digest(raw), 'gzip_sha256': digest(compressed),
        'encoding': 'gzip+base64', 'payload': base64.b64encode(compressed).decode()}
    worker['scientific_sha256'] = digest(raw)
    text = render(record)
    assert 'Primary combined speed/RSS clause: FAIL' in text
