"""Cached-action invariants independent of the molecular campaign."""
import numpy as np
import pytest

from clifford_qc.subspace.action_space import ActionSpace, orthogonal_direction


def test_actions_are_reused_and_residual_is_in_full_space():
    h = np.array([[0., 1., 0.], [1., 2., .7], [0., .7, 4.]])
    calls = []

    def apply(v):
        calls.append(v.copy())
        return h @ v

    space = ActionSpace(apply, np.eye(3)[:, :1])
    second = np.eye(3)[:, 1]
    acted = space.action(second)
    space.extend(second[:, None], actions=acted[:, None])
    energy, state, residual = space.solve()
    np.testing.assert_allclose(residual, h @ state - energy*state, atol=1e-14)
    assert np.linalg.norm(residual) > .1
    space.solve()
    assert len(calls) == space.h_action_calls == 2
    space.extend(np.eye(3)[:, 2:])
    energy, state, residual = space.solve()
    assert energy == pytest.approx(np.linalg.eigvalsh(h)[0])
    assert np.linalg.norm(residual) < 1e-12


def test_whole_span_dependence_and_nonhermitian_actions_are_rejected():
    space = ActionSpace(lambda v: v, np.eye(4)[:, :2])
    with pytest.raises(ValueError, match="whole span"):
        space.extend(np.eye(4)[:, 1:2])
    assert orthogonal_direction(np.eye(4)[:, 1], space.basis) is None
    with pytest.raises(ValueError, match="Hermitian"):
        ActionSpace(lambda v: np.array([[0., 1.], [0., 0.]]) @ v, np.eye(2))


@pytest.mark.parametrize("action", [lambda v: v[:-1], lambda v: v*np.nan])
def test_invalid_callback_results_are_rejected(action):
    with pytest.raises(ValueError, match="invalid vector"):
        ActionSpace(action, np.eye(3)[:, :1])
