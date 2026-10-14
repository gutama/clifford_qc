"""Independent occupation/matrix checks for exchange pools and exact ADAPT."""
from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from clifford_qc.algorithms.exchange import (
    exchange_energy_gradient, exchange_pools, run_exchange_adapt,
)
from clifford_qc.dense_reference import to_matrix
from clifford_qc.fermion import total_number_op, total_sz_op
from clifford_qc.models.fcidump import fcidump_model


def occupation_exchange(n, creation, annihilation):
    """Qubit ladders act without Jordan--Wigner signs."""
    out = np.zeros((2**n, 2**n), complex)
    for bits in product((0, 1), repeat=n):
        if any(bits[j] for j in creation) or any(not bits[j] for j in annihilation):
            continue
        target = list(bits)
        for j in creation:
            target[j] = 1
        for j in annihilation:
            target[j] = 0
        i = int(''.join(map(str, bits)), 2)
        k = int(''.join(map(str, target)), 2)
        out[k, i], out[i, k] = 1, -1
    return out


@pytest.mark.parametrize('n', [2, 4, 6, 8])
def test_qe_pool_against_occupation_action(n):
    qe, _ = exchange_pools(n)
    for op in qe:
        expected = occupation_exchange(n, op.creation, op.annihilation)
        assert np.array_equal(to_matrix(op.generator), expected)
    assert len({op.label for op in qe}) == len(qe)


@pytest.mark.parametrize('n,counts', [(2, (0, 0)), (4, (4, 4)), (6, (24, 24)),
                                      (8, (90, 96))])
def test_pool_census(n, counts):
    qe, ceo = exchange_pools(n)
    assert (len(qe), len(ceo)) == counts


def test_ceo_structure_and_exact_exponential():
    qe, ceo = exchange_pools(6)
    parents = {op.label: op for op in qe}
    number = to_matrix(total_number_op(6))
    spin = to_matrix(total_sz_op(6))
    for op in ceo:
        t = to_matrix(op.generator)
        assert np.array_equal(t.conj().T, -t)
        assert np.allclose(t @ t @ t, -t)
        assert np.allclose(number @ t, t @ number)
        assert np.allclose(spin @ t, t @ spin)
        theta = .31
        u = np.eye(64) + np.sin(theta)*t + (1-np.cos(theta))*(t@t)
        assert np.allclose(u, expm(theta*t), atol=1e-13)
        assert np.allclose(u.conj().T@u, np.eye(64))
        if len(op.parents) == 2:
            expected = (parents[op.parents[0]].generator
                        + op.sign*parents[op.parents[1]].generator)
            assert op.generator.terms == expected.terms
            assert len(op.generator.terms) == 4


def test_analytic_gradient_for_noncommuting_sequence():
    qe, ceo = exchange_pools(4)
    mats = [to_matrix(qe[0].generator), to_matrix(ceo[-1].generator)]
    rng = np.random.default_rng(4)
    a = rng.normal(size=(16,16)) + 1j*rng.normal(size=(16,16))
    h = a+a.conj().T
    psi = rng.normal(size=16) + 1j*rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    theta = np.array([.17, -.29])
    energy, gradient, state = exchange_energy_gradient(h, psi, mats, theta)
    exact_state = expm(theta[1]*mats[1])@expm(theta[0]*mats[0])@psi
    assert np.allclose(state, exact_state)
    assert energy == pytest.approx(np.vdot(exact_state, h@exact_state).real)
    finite = []
    for i in range(2):
        delta = np.eye(2)[i]*1e-6
        ep = exchange_energy_gradient(h, psi, mats, theta+delta)[0]
        em = exchange_energy_gradient(h, psi, mats, theta-delta)[0]
        finite.append((ep-em)/2e-6)
    assert np.allclose(gradient, finite, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('arm', ['qe', 'ovp_ceo', 'mixed_ceo'])
def test_exact_run_variational_and_accounted(arm):
    from clifford_qc.models import hubbard
    model = hubbard(2, U=2.1, t=.7)
    result = run_exchange_adapt(model, arm=arm, max_iterations=3, maxiter=100)
    energies = [result['reference_energy']] + [r['energy'] for r in result['trajectory']]
    assert all(a >= b-1e-10 for a,b in zip(energies,energies[1:]))
    assert result['energy'] >= result['sector_ground_energy']-1e-10
    assert result['resources']['screened_gradients'] == sum(
        r['screened_gradients'] for r in result['screening'])
    assert result['resources']['optimizer_energy_gradient_evaluations'] == sum(
        r['optimizer_evaluations'] for r in result['trajectory'])
    assert result['resources']['parameter_derivatives'] == sum(
        len(r['parameters'])*r['optimizer_evaluations'] for r in result['trajectory'])
    assert result['quantum_shots'] is None


@pytest.mark.parametrize('n', [0, -1, 2.5, True])
def test_invalid_register(n):
    with pytest.raises(ValueError):
        exchange_pools(n)


def test_bad_gradient_input():
    with pytest.raises(ValueError):
        exchange_energy_gradient(np.eye(2), np.ones(2), [np.eye(2)], [0.])


def test_selection_uses_first_member_within_declared_tolerance():
    from clifford_qc.algorithms.exchange import _select_exchange
    labels = ['first', 'later', 'largest']
    assert _select_exchange(labels, dict(first=1., later=1.+2e-16, largest=.5), 1e-10) == 'first'
    assert _select_exchange(labels, dict(first=1., later=1.+2e-9, largest=.5), 1e-10) == 'later'


def test_optimizer_cache_and_arm_storage():
    from clifford_qc.models import hubbard
    model = hubbard(2, U=2.1, t=.7)
    run = run_exchange_adapt(model, max_iterations=3, maxiter=100)
    for row in run['trajectory']:
        trace = row['optimizer_trace']
        assert len({tuple(call['parameters']) for call in trace}) == len(trace)
        assert row['optimizer_callback_requests'] == len(row['optimizer_callback_indices'])
        assert row['optimizer_cache_hits'] == row['optimizer_callback_requests']-len(trace)
        assert row['optimizer_cache_hits'] >= 1  # common adapter's final request is cached
        assert row['optimizer_resolved'] == (max(map(abs, trace[row['final_evaluation_index']]['gradient'])) <= 1e-8)
    dimension = len(run['sector_basis'])
    assert run['resources']['dense_operator_bytes'] == (1+len(run['pool_labels']))*dimension**2*16


def test_case_preparation_is_shared_and_matrices_are_lazy():
    from clifford_qc.algorithms.exchange import _prepare_exchange
    from clifford_qc.models import hubbard
    model = hubbard(4, U=2.1, t=.7)
    prepared = _prepare_exchange(model, 256)
    assert not prepared['matrices'] and not prepared['commutators']
    qe = run_exchange_adapt(model, max_iterations=1, _prepared=prepared)
    assert set(prepared['matrices']) == set(qe['pool_labels'])
    ovp = run_exchange_adapt(model, arm='ovp_ceo', max_iterations=1, _prepared=prepared)
    expected = (1+len(ovp['pool_labels']))*len(ovp['sector_basis'])**2*16
    assert ovp['resources']['dense_operator_bytes'] == expected
