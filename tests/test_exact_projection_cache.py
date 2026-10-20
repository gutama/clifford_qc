"""Exact-cache invariants on toy inputs, independent of held-out campaign runs."""
from copy import deepcopy
import base64
import gzip
from pathlib import Path

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
        expected_fixed = (20*(stats['frame_nnz']+stats['group_nnz'])
                          + 4*(frame.shape[1]+1+stats['group_columns']+stats['group_block_builds']))
        assert stats['fixed_owned_bytes'] == expected_fixed
    else:
        assert stats['fixed_owned_bytes'] == stats['group_block_builds'] == stats['frame_adjoint_builds'] == 0
    assert stats['peak_owned_bytes'] == stats['fixed_owned_bytes']+stats['peak_basis_adjoint_bytes']


def test_combined_path_skips_repeated_frame_slices_and_old_projection(physical, monkeypatch):
    h, frame, hf = physical
    monkeypatch.setattr(frame, 'combine', lambda *a: pytest.fail('uncached group slice used'))
    monkeypatch.setattr(frame, 'adjoint', lambda *a: pytest.fail('uncached adjoint used'))
    monkeypatch.setattr(packets, 'orthogonal_direction', lambda *a, **k: pytest.fail('uncached span projection used'))
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode='combined')
    assert result['cache']['basis_adjoint_builds'] == len(result['decisions'])


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
