"""Sparse physical operators/CSFs and early materialization on undeclared small sectors."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from benchmarks.check_h2o_residual_comparison import independent_frame
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from benchmarks.run_packet_work_reuse import trajectory
from benchmarks.metadata_io import read_json
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.sparse_packets import SparseSingletFrame, run_sparse_packets


@pytest.fixture
def physical():
    backend = SectorStatevectorBackend(8, 4, 0)
    operator = backend.operator(hubbard((2, 2), U=4.).hamiltonian)
    return backend, operator


@pytest.mark.parametrize('precompute', [True, False])
def test_direct_csr_preserves_permutations_actions_and_cancellations(physical, precompute, monkeypatch):
    backend, _ = physical
    op = backend.operator(hubbard((2, 2), U=4.).hamiltonian, precompute=precompute)
    indices = [17, 0, 9, 3]
    expected = op.restrict(indices)
    full = op.restrict(np.arange(backend.dimension))
    monkeypatch.setattr(op, 'restrict', lambda *a: pytest.fail('dense restriction called'))
    sparse = op.to_csr()
    np.testing.assert_allclose(sparse.toarray(), full, atol=1e-14)
    np.testing.assert_allclose(op.to_csr(indices).toarray(), expected, atol=1e-14)
    psi = np.arange(backend.dimension) + 1j
    np.testing.assert_allclose(sparse @ psi, op.matvec(psi), atol=1e-13)
    assert sparse.has_sorted_indices and not np.any(sparse.data == 0)


@pytest.mark.parametrize('indices', [[], [True], [0., 1.], [[0]], [0, 0], [-1], [36]])
def test_sparse_restriction_rejects_ambiguous_indices(physical, indices):
    with pytest.raises(ValueError): physical[1].to_csr(indices)


def test_sparse_frame_matches_independent_spin_and_canonical_frame(physical):
    backend, _ = physical
    s2 = independent_spin_squared(backend.basis.tolist(), 8)
    expected, labels = independent_frame(backend.basis.tolist(), 8, s2, 1e-9)
    frame = SparseSingletFrame(backend)
    np.testing.assert_allclose(frame.spin.toarray(), s2, atol=1e-14)
    np.testing.assert_allclose(frame.matrix.toarray(), expected, atol=1e-13)
    assert frame.labels == tuple(labels)
    assert frame.payload_bytes < 16*(36**2 + 36*20)
    coeff = np.arange(frame.shape[1]) + 1j
    np.testing.assert_allclose(frame.combine(list(range(frame.shape[1])), coeff), expected @ coeff)
    np.testing.assert_allclose(frame.adjoint(expected @ coeff), coeff, atol=1e-12)


def test_block_admission_precedes_dense_spin_construction(physical, monkeypatch):
    backend, _ = physical
    monkeypatch.setattr(backend, 'operator', lambda *a: pytest.fail('spin constructed before cap'))
    with pytest.raises(ValueError, match='block exceeds'):
        SparseSingletFrame(backend, max_block_dimension=1)
    with pytest.raises(ValueError): SparseSingletFrame(SectorStatevectorBackend(8, 4))
    with pytest.raises(ValueError): SparseSingletFrame(backend, tolerance=float('nan'))


def small_config():
    config = read_json(Path('benchmarks/configs/sparse_packet_work.json'))
    return {**config, 'maximum_columns': 8, 'maximum_growth_steps': 5, 'audit_interval': 2}


def test_sparse_exhaustive_replays_existing_physical_trajectory(physical):
    backend, operator = physical
    h = operator.to_csr()
    frame = SparseSingletFrame(backend)
    config = small_config()
    hf = sum(1 << (7-j) for j in range(4))
    result = run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config)
    old_config = {**config, 'input': {'active_spatial_orbitals': [1, 2]},
                  'preconditioner_shift_hartree': .1, 'residual_tolerance_hartree': 1e-9}
    old = trajectory(h.dot, h.diagonal().real, backend.basis, 8, hf,
                     frame.matrix.toarray(), old_config, 'original', frame.spin.toarray())
    assert result['stop_reason'] == old['stop_reason']
    for new, previous in zip(result['trajectory'], old['trajectory'], strict=True):
        assert new['selection'] == previous['selection']
        assert new['energy'] == pytest.approx(previous['energy'], abs=1e-11)
        assert new['residual_norm'] == pytest.approx(previous['residual_norm'], abs=1e-11)
        assert new['h_action_calls'] == previous['h_action_calls']


def test_early_policy_materializes_only_marks_until_audit_or_fallback(physical, monkeypatch):
    backend, operator = physical
    frame = SparseSingletFrame(backend)
    h = operator.to_csr()
    config = small_config()
    result = run_sparse_packets(h.dot, h.diagonal().real, frame, 240, [1, 2], config, policy='early')
    assert any(d['audit'] for d in result['decisions'])
    for decision in result['decisions']:
        if not decision['audit'] and not decision['fallback']:
            assert set(decision['attempted']) == set(decision['shortlist'])
    assert result['materialized_candidates'] == sum(len(d['attempted']) for d in result['decisions'])
    assert all(b['energy'] <= a['energy'] + 1e-12
               for a, b in zip(result['trajectory'], result['trajectory'][1:]))
    bad = deepcopy(config); bad['audit_interval'] = 0
    with pytest.raises(ValueError): run_sparse_packets(h.dot, h.diagonal().real, frame, 240, [1, 2], bad)


@pytest.mark.parametrize('arm', ['sparse', 'early'])
def test_independent_replay_rejects_counter_and_score_mutations(physical, arm):
    from benchmarks.check_sparse_packet_work import replay
    backend, operator = physical
    frame = SparseSingletFrame(backend)
    h = operator.to_csr()
    c = small_config()
    case = {'n_qubits': 8, 'n_electrons': 4, 'active_spatial_orbitals': [1, 2],
            'maximum_columns': 8, 'maximum_growth_steps': 5}
    r = run_sparse_packets(h.dot, h.diagonal().real, frame, 240, [1, 2], c,
                           policy='early' if arm == 'early' else 'exhaustive')
    r['arm'] = arm
    spin = independent_spin_squared(backend.basis.tolist(), 8)
    independent = independent_frame(backend.basis.tolist(), 8, spin, 1e-9)
    assert replay(r, case, c, h, backend.basis.tolist(), spin, independent) == []
    changed = deepcopy(r); changed['trajectory'][0]['h_action_calls'] += 1
    assert 'prefix work' in replay(changed, case, c, h, backend.basis.tolist(), spin, independent)
    changed = deepcopy(r)
    key = next(iter(changed['decisions'][0]['scores']))
    changed['decisions'][0]['scores'][key] += 100
    assert replay(changed, case, c, h, backend.basis.tolist(), spin, independent)
    changed = deepcopy(r)
    for component in ('real', 'imag'):
        changed['trajectory'][1]['added_columns'][0][component] = [-v for v in changed['trajectory'][1]['added_columns'][0][component]]
    assert 'recorded canonical phase' in replay(changed, case, c, h, backend.basis.tolist(), spin, independent)


def test_result_free_default_and_no_overwrite(monkeypatch, tmp_path):
    from benchmarks import run_sparse_packet_work as producer
    monkeypatch.setattr(producer, 'execute', lambda c: pytest.fail('science reached'))
    assert producer.main([]) == 0
    output = tmp_path/'existing.json'; output.write_text('old')
    with pytest.raises(FileExistsError): producer.main(['--execute', '--output', str(output)])
    assert output.read_text() == 'old'


@pytest.mark.parametrize('arm', ['dense', 'sparse', 'early'])
def test_isolated_worker_replays_independently_on_small_physical_input(arm, monkeypatch):
    from benchmarks import run_sparse_packet_work as producer
    from benchmarks.check_sparse_packet_work import independent_input, replay
    from clifford_qc.backends.sector_statevector import SectorOperator
    c = small_config()
    case = deepcopy(c['inputs'][1])
    case['model']['shape'] = [2, 2]
    case.update(n_qubits=8, n_electrons=4, sector_dimension=36,
                active_spatial_orbitals=[1, 2], maximum_columns=8, maximum_growth_steps=5)
    monkeypatch.setattr(producer, 'preflight', lambda _: None)
    if arm != 'dense':
        monkeypatch.setattr(SectorOperator, 'restrict', lambda *a: pytest.fail('dense restriction reached'))
    result = producer.worker(case, arm, c)
    h, words = independent_input(case)
    spin = independent_spin_squared(words, 8)
    frame = independent_frame(words, 8, spin, 1e-9)
    assert replay(result, case, c, h, words, spin, frame) == []
    if arm != 'dense':
        assert result['setup']['dense_sector_h_temporary_bytes'] == 0


def test_execution_journal_keeps_completed_cells_and_never_retries_interruption(tmp_path):
    import json
    from benchmarks.run_sparse_packet_work import open_journal, save_journal, expected_order
    c = {'order_seed': 3, 'arms': ['dense', 'sparse'], 'inputs': [{'name': 'toy', 'repeats': 1}]}
    base = {'source_commit': 'a', 'declaration_commit': 'b', 'config': c, 'source_sha256': {},
            'provenance': {'dependencies': {}, 'platform': {}, 'environment_sha256': 'c'}, 'workers': []}
    path = tmp_path/'journal.json'
    state = open_journal(path, deepcopy(base))
    order = expected_order(c)
    entry = lambda i: dict(zip(('input', 'repeat', 'arm'), order[i]), order_index=i)
    state['record']['workers'].append({**entry(0), 'status': 'ok', 'process_wall_seconds': 1.})
    pid_path = tmp_path/'worker.pid'; pid_path.write_text(str(2**30))
    state.update(in_flight={**entry(1), 'worker_pid_path': str(pid_path)}, supervisor_pid=2**30)
    save_journal(path, state)
    resumed = open_journal(path, deepcopy(base), resume=True)
    assert resumed['record']['workers'][0]['status'] == 'ok'
    failed = resumed['record']['workers'][1]
    assert failed['status'] == 'failed' and failed['process_wall_seconds'] is None
    assert len(resumed['record']['workers']) == len(order)
    changed = deepcopy(base); changed['source_commit'] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        open_journal(path, changed, resume=True)
    assert json.loads(path.read_text())['in_flight'] is None
    from benchmarks.run_sparse_packet_work import pack, unpack
    native = {'schema': 'clifford_qc.sparse_packet_work.v1', 'provenance': {},
              'evidence_schema': 'example', 'evidence': {}, 'workers': resumed['record']['workers']}
    assert unpack(pack(native)) == native
    manifest = json.loads(path.read_text())
    cell = path.with_name(path.name+'.cells')/manifest['record']['workers'][0]['checkpoint']
    cell.write_bytes(b'corrupted checkpoint')
    with pytest.raises(ValueError, match='digest'):
        open_journal(path, deepcopy(base), resume=True)


def test_journal_reuses_verified_cells_across_saves_and_resume(tmp_path, monkeypatch):
    from benchmarks import run_sparse_packet_work as producer
    import json
    c = {'order_seed': 3, 'arms': ['dense', 'sparse'], 'inputs': [{'name': 'toy', 'repeats': 1}]}
    base = {'source_commit': 'a', 'declaration_commit': 'b', 'config': c, 'source_sha256': {},
            'provenance': {'dependencies': {}, 'platform': {}, 'environment_sha256': 'c'}, 'workers': []}
    path = tmp_path/'journal.json'
    state = producer.open_journal(path, deepcopy(base))
    compressed = []
    original = producer.gzip.compress

    def compress(data, **kwargs):
        compressed.append(len(data))
        return original(data, **kwargs)

    monkeypatch.setattr(producer.gzip, 'compress', compress)
    for i, cell in enumerate(producer.expected_order(c)):
        state['record']['workers'].append(dict(zip(('input', 'repeat', 'arm'), cell),
            order_index=i, status='ok', process_wall_seconds=1.))
        producer.save_journal(path, state)
        producer.save_journal(path, state)
        assert len(compressed) == i+1
    state['state'] = 'complete'
    producer.save_journal(path, state)
    manifest = json.loads(path.read_text())
    assert '_cell_checkpoint_cache' not in manifest
    resumed = producer.open_journal(path, deepcopy(base), resume=True)
    assert len(resumed['record']['workers']) == 2
    assert len(compressed) == 2  # Resume verified existing files rather than compressing again.
    cell = path.with_name(path.name+'.cells')/manifest['record']['workers'][0]['checkpoint']
    cell.write_bytes(b'corrupted checkpoint')
    with pytest.raises(ValueError, match='digest'):
        producer.open_journal(path, deepcopy(base), resume=True)


def test_report_derives_coverage_and_deduplicates_deterministic_audits():
    from benchmarks.summarize_sparse_packet_work import render
    from benchmarks.run_sparse_packet_work import unpack, RECORD
    record = unpack(read_json(Path(RECORD)))
    text = render(record)
    assert 'Coverage for h2o: 9/9' in text
    medium = text.split('## hubbard_2x4_u4')[1].split('## hubbard_2x3_u8')[0]
    held_out = text.split('## hubbard_2x3_u8')[1]
    assert '(7 timing observations): 12; missed winners: 0' in medium
    assert '(1 timing observation): 5; missed winners: 2' in held_out
    assert '(+12.85%)' in held_out
    record['workers'].remove(next(w for w in record['workers'] if w['input'] == 'h2o' and w['arm'] == 'early'))
    altered = render(record)
    assert 'Coverage for h2o: 8/9' in altered
    assert '| early | 2/3 |' in altered


def test_report_historical_equivalence_detects_selection_drift():
    from benchmarks.summarize_sparse_packet_work import render, PREVIOUS_RECORD
    from benchmarks.run_packet_work_large import unpack_observation
    from benchmarks.run_sparse_packet_work import RECORD
    published = read_json(Path(RECORD))
    previous = unpack_observation(read_json(Path(PREVIOUS_RECORD)))
    assert 'within 1e-9: PASS; 62 retained prefixes' in render(published, previous)
    worker = next(w for w in previous['workers'] if w['status'] == 'ok' and w['arm'] == 'shortlist')
    worker['result']['trajectory'][1]['selection'] = {'candidate': -1}
    assert 'within 1e-9: FAIL' in render(published, previous)
