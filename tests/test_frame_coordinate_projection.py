"""Frame-coordinate invariants on independent toys, not campaign inputs."""
from copy import deepcopy
from pathlib import Path
import weakref

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from benchmarks.check_exact_projection_cache import compare_science
from benchmarks.check_frame_coordinate_projection import cache_problems
from benchmarks.metadata_io import read_json
from benchmarks.run_frame_coordinate_projection import (
    canonical, decode_science, digest, expected_order, scientific_result)
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import sparse_packets as packets
from clifford_qc.subspace.action_space import orthogonal_direction


def config():
    c = read_json(Path('benchmarks/configs/frame_coordinate_projection.json'))
    return {**c, 'maximum_columns': 8, 'maximum_growth_steps': 5}


def complex_frame(seed=42):
    rng = np.random.default_rng(seed)
    frame = np.linalg.qr(rng.normal(size=(17, 9)) + 1j*rng.normal(size=(17, 9)))[0]
    coordinates = np.linalg.qr(rng.normal(size=(9, 3)) + 1j*rng.normal(size=(9, 3)))[0]
    return rng, csc_matrix(frame), frame @ coordinates


@pytest.mark.parametrize('seed', range(8))
def test_complex_isometry_matches_independent_physical_projector(seed):
    rng, frame, basis = complex_frame(seed)
    projection = packets._FrameCoordinateProjection(frame, frame.conj().T.dot, basis, tolerance=1e-10)
    coefficients = rng.normal(size=4) + 1j*rng.normal(size=4)
    indices = np.array([0, 2, 5, 8])
    candidate = np.asarray(frame[:, indices] @ coefficients)
    direction = projection.direction(indices, coefficients, cutoff=1e-10)
    expected = orthogonal_direction(candidate, basis)
    assert abs(np.vdot(expected, direction)) == pytest.approx(1, abs=2e-14)
    # An independent dense projector, rather than just the production routine.
    projector = np.eye(17) - basis @ basis.conj().T
    oracle = projector @ (projector @ candidate)
    oracle /= np.linalg.norm(oracle)
    assert abs(np.vdot(oracle, direction)) == pytest.approx(1, abs=2e-14)
    assert np.linalg.norm(basis.conj().T @ direction) < 1e-13
    assert np.linalg.norm(direction) == pytest.approx(1)
    pivot = direction[np.argmax(abs(direction))]
    assert pivot.real > 0 and abs(pivot.imag) < 1e-14
    assert projection.payload_bytes == 32*9*3
    assert not projection.coordinates.flags.writeable and not projection.adjoint.flags.writeable
    assert not np.shares_memory(projection.coordinates, projection.adjoint)


@pytest.mark.parametrize('size, accepted', [(0., False), (1e-12, False), (1e-7, True)])
def test_dependence_uses_physical_cutoff(size, accepted):
    frame = csc_matrix(np.eye(6, 4, dtype=complex))
    basis = frame[:, [0]].toarray()
    projection = packets._FrameCoordinateProjection(frame, frame.conj().T.dot, basis, tolerance=1e-10)
    result = projection.direction([0, 1], [1., size], cutoff=1e-9)
    assert (result is not None) == accepted
    if accepted:
        np.testing.assert_allclose(result, np.eye(6)[:, 1], atol=1e-15)


def test_outside_frame_or_nonorthonormal_span_is_rejected():
    frame = csc_matrix(np.eye(6, 4, dtype=complex))
    for basis in (np.eye(6)[:, [5]], 2*np.eye(6)[:, [0]], np.ones((6, 2))):
        with pytest.raises(ValueError, match='orthonormal'):
            packets._FrameCoordinateProjection(frame, frame.conj().T.dot, basis, tolerance=1e-10)


@pytest.mark.parametrize('indices, values, cutoff', [
    ([0, 0], [1., 2.], 1e-9), ([-1], [1.], 1e-9), ([9], [1.], 1e-9),
    ([1.5], [1.], 1e-9), ([1], [np.nan], 1e-9), ([1], [], 1e-9),
    ([1], [1.], 0.), ([1], [1.], np.nan)])
def test_invalid_candidate_is_rejected(indices, values, cutoff):
    _, frame, basis = complex_frame()
    projection = packets._FrameCoordinateProjection(frame, frame.conj().T.dot, basis, tolerance=1e-10)
    with pytest.raises(ValueError):
        projection.direction(indices, values, cutoff=cutoff)


@pytest.fixture
def physical():
    backend = SectorStatevectorBackend(8, 4, 0)
    h = backend.operator(hubbard((2, 2), U=4.).hamiltonian).to_csr()
    return h, packets.SparseSingletFrame(backend), 240


@pytest.mark.parametrize('policy', ['exhaustive', 'early'])
def test_growth_matches_sector_path_and_storage_formula(physical, policy):
    h, frame, hf = physical
    baseline = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(),
                                         policy=policy, cache_mode='combined')
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(),
                                       policy=policy, cache_mode='coordinates')
    result['arm'] = baseline['arm'] = 'early' if policy == 'early' else 'sparse'
    science = scientific_result(result)
    assert not compare_science(science, scientific_result(baseline), h.shape[0])
    case = {'sector_dimension': h.shape[0]}
    assert not cache_problems(result['cache'], science, case, frame.matrix.toarray(), frame.words)
    corrupted = deepcopy(result['cache']); corrupted['peak_coordinate_bytes'] += 16
    assert cache_problems(corrupted, science, case, frame.matrix.toarray(), frame.words)


def test_coordinate_path_skips_physical_group_materialization_and_old_projector(physical, monkeypatch):
    h, frame, hf = physical
    monkeypatch.setattr(frame, 'combine', lambda *a: pytest.fail('physical candidate materialized'))
    monkeypatch.setattr(packets._StepProjection, '__init__', lambda *a: pytest.fail('sector adjoint built'))
    original = packets._FrameSnapshot.__init__

    def snapshot(self, matrix, groups):
        assert not groups
        original(self, matrix, groups)

    monkeypatch.setattr(packets._FrameSnapshot, '__init__', snapshot)
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode='coordinates')
    assert result['cache']['coordinate_builds'] > 0


def test_coordinate_arrays_are_released_before_extension(physical, monkeypatch):
    h, frame, hf = physical
    references, stages = [], []
    init, extend = packets._FrameCoordinateProjection.__init__, packets.ActionSpace.extend

    def tracked(self, matrix, adjoint, basis, **kwargs):
        init(self, matrix, adjoint, basis, **kwargs)
        stages.append(basis.shape[1])
        references.extend(weakref.ref(a) for a in (self.coordinates, self.adjoint))

    def checked(self, columns, **kwargs):
        assert all(ref() is None for ref in references)
        return extend(self, columns, **kwargs)

    monkeypatch.setattr(packets._FrameCoordinateProjection, '__init__', tracked)
    monkeypatch.setattr(packets.ActionSpace, 'extend', checked)
    result = packets.run_sparse_packets(h.dot, h.diagonal().real, frame, hf, [1, 2], config(), cache_mode='coordinates')
    assert stages == [r['dimension'] for r in result['trajectory'][:-1]]
    assert result['cache']['coordinate_builds'] == len(stages)


def test_design_has_fixed_schedule_and_both_transfer_checks():
    c = config()
    order = expected_order(c)
    assert len(order) == 56 and order == expected_order(c)
    assert {row['cache_mode'] for row in order} == {'combined', 'coordinates'}
    assert {row['policy'] for row in order} == {'exhaustive', 'early'}
    assert {'held_out_molecule', 'held_out_lattice'} <= {case['role'] for case in c['inputs']}
    assert all(p not in c['implementation_paths_absent_at_declaration'] for p in [
        'clifford_qc/subspace/sparse_packets.py'])  # Pre-existing module, checked by baseline bytes.


def test_scientific_payload_digest_is_enforced():
    import base64
    import gzip
    raw = canonical({'test': [1, 2, 3]}); compressed = gzip.compress(raw, mtime=0)
    payload = {'payload': base64.b64encode(compressed).decode(), 'gzip_sha256': digest(compressed), 'sha256': digest(raw)}
    assert decode_science(payload) == {'test': [1, 2, 3]}
    payload['sha256'] = '0'*64
    with pytest.raises(ValueError, match='digest'):
        decode_science(payload)
