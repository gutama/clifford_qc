"""Linear-action identities, isolation, and deterministic residual coverage."""
import numpy as np
import pytest
from clifford_qc.subspace.action_space import ActionSpace, orthogonal_direction
from clifford_qc.subspace.packet_selection import FrameActionCache, orthogonal_action, residual_shortlist


def test_complex_actions_survive_refresh_projection_phase_and_input_mutation():
    rng = np.random.default_rng(74)
    raw = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8))
    h = raw + raw.conj().T
    frame = np.linalg.qr(raw)[0]
    source = frame.copy()
    space = ActionSpace(lambda v: h @ v, frame[:, :2])
    cache = FrameActionCache(source, space.action)
    cache.seed([0, 1], space.actions)
    source[:] = 0  # The cache owns its snapshot.
    for j in range(4):
        coeff = rng.normal(size=3) + 1j*rng.normal(size=3)
        vector, acted = cache.combine([2, 3, 4], coeff)
        np.testing.assert_allclose(acted, h @ vector, atol=1e-13)
        pair = orthogonal_action(vector, acted, space.basis, space.actions)
        np.testing.assert_allclose(pair[0], orthogonal_direction(vector, space.basis), atol=1e-13)
        np.testing.assert_allclose(pair[1], h @ pair[0], atol=1e-13)
        acted[:] = 900  # Returned arrays cannot corrupt cached actions.
    assert space.h_action_calls == 5
    assert cache.indices == (0, 1, 2, 3, 4)
    assert cache.payload_bytes == frame.nbytes + 5*8*16
    assert orthogonal_action(frame[:, 0], h @ frame[:, 0], space.basis, space.actions) is None


@pytest.mark.parametrize('indices', [[-1], [4], [True], [1, 1]])
def test_cache_rejects_invalid_indices_before_callback(indices):
    cache = FrameActionCache(np.eye(4), lambda v: pytest.fail('callback reached'))
    with pytest.raises(ValueError):
        cache.combine(indices, np.ones(len(indices)))


def test_seed_and_callback_are_validated_without_silent_overwrite():
    cache = FrameActionCache(np.eye(3), lambda v: np.full(3, np.nan))
    cache.seed([0], np.eye(3)[:, :1])
    with pytest.raises(ValueError): cache.seed([0], np.eye(3)[:, :1])
    with pytest.raises(ValueError): cache.seed([1], np.zeros((2, 1)))
    with pytest.raises(ValueError): cache.combine([1], [1])
    assert cache.indices == (0,)
    with pytest.raises(ValueError): FrameActionCache(np.ones((3, 2)), lambda v: v)
    with pytest.raises(ValueError): orthogonal_action([1], [1, 2], np.empty((1, 0)), np.empty((1, 0)))


@pytest.mark.parametrize('indicators,fraction,expected', [
    ({9: 1., 2: 4., 1: 4., 3: 1.}, .8, (1, 2)),
    ({3: 0., 1: 0.}, .8, (1, 3)), ({}, .8, ()),
    ({1: 4., 2: 1., 3: 1.}, 1., (1, 2, 3)),
])
def test_shortlist_prefix_coverage_and_ties(indicators, fraction, expected):
    assert residual_shortlist(indicators, fraction=fraction) == expected


@pytest.mark.parametrize('indicators,fraction', [({1: -1}, .8), ({1: float('nan')}, .8),
    ({1: 1}, 0), ({1: 1}, 1.1), ({1: 1e308, 2: 1e308}, .8), ({'1': 1}, .8)])
def test_invalid_shortlist_inputs(indicators, fraction):
    with pytest.raises(ValueError): residual_shortlist(indicators, fraction=fraction)
