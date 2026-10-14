"""Qubit exchange and OVP-CEO pools; exact small-sector ADAPT comparison.

Qubit ladders are (X +/- iY)/2, without Jordan--Wigner Z strings. Pool
members are anti-Hermitian T and evolve as exp(theta T), matching Ramôa
et al., npj Quantum Information 11, 86 (2025), Eqs. 23--27. In the package's
Hermitian convention the generator is G=2iT and the rotor is exp(-i theta G/2).
All pools conserve N and interleaved S_z, but they are not fermionic excitation
pools and do not impose total-spin or point-group symmetry.

This solver stores dense matrices in a bounded occupation sector. Selection
and optimization use exact expectations, with no finite-shot or hardware cost
claim. Mixed CEO uses OVP screening and promotes the nonzero QE parents on the
selected support to independent parameters (the paper's Eq. 29).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from ..backends.sector_statevector import SectorStatevectorBackend
from ..multivector import MV
from ..pauli import X, Y
from .optimize import minimize_energy


@dataclass(frozen=True)
class ExchangeOperator:
    label: str
    generator: MV
    support: tuple[int, ...]
    creation: tuple[int, ...] = ()
    annihilation: tuple[int, ...] = ()
    parents: tuple[str, ...] = ()
    sign: int = 1


def exchange_pools(n: int) -> tuple[list[ExchangeOperator], list[ExchangeOperator]]:
    """Complete spin-conserving generalized QE and OVP-CEO pools.

    Each unordered complementary partition of four distinct sites enters once
    when both sides have the same spin census. Each support has two QEs in
    the mixed-spin case or three in the same-spin case; CEOs are every pair's
    sum and difference. Singles belong to both pools. Canonical index ordering
    fixes signs and ties independently of the reference determinant.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if n > 16:
        raise ValueError('exchange pool enumeration is limited to 16 spin orbitals')
    q = [0.5*(X(n, j)+1j*Y(n, j)) for j in range(n)]

    def make(creation, annihilation):
        term = MV.scalar(n, 1.)
        for j in creation:
            term = term*q[j].dagger()
        for j in annihilation:
            term = term*q[j]
        label = 'qe:' + ','.join(map(str, creation)) + '<-' + ','.join(map(str, annihilation))
        return ExchangeOperator(label, term-term.dagger(),
                                tuple(sorted(creation+annihilation)), creation, annihilation)

    singles = [make((a,), (b,)) for a,b in combinations(range(n), 2) if a%2 == b%2]
    qe, ceo = list(singles), list(singles)
    for support in combinations(range(n), 4):
        group = []
        for creation in combinations(support, 2):
            annihilation = tuple(j for j in support if j not in creation)
            if creation >= annihilation:
                continue
            if sum(j%2 for j in creation) != sum(j%2 for j in annihilation):
                continue
            group.append(make(creation, annihilation))
        qe.extend(group)
        for first, second in combinations(group, 2):
            for sign in (1, -1):
                label = f'ceo:{first.label}:{sign:+d}:{second.label}'
                ceo.append(ExchangeOperator(label, first.generator+sign*second.generator,
                                            support, parents=(first.label, second.label), sign=sign))
    return qe, ceo


def _evolve(t, theta, psi):
    # T^3=-T gives exp(theta T) without an eigendecomposition or Trotter error.
    first = t@psi
    return psi + math.sin(theta)*first + (1-math.cos(theta))*(t@first)


def exchange_energy_gradient(h, reference, generators, parameters):
    """Exact adjoint gradient of an ordered exchange ansatz.

    Matrices must be anti-Hermitian and satisfy T^3=-T; these conditions are
    checked here so the trigonometric exponential cannot silently approximate
    a general generator. The reference is normalized, never rescaled here.
    """
    h = np.asarray(h, complex)
    reference = np.asarray(reference, complex)
    ts = [np.asarray(t, complex) for t in generators]
    theta = np.asarray(parameters, float)
    if (reference.ndim != 1 or h.shape != (reference.size, reference.size)
            or theta.shape != (len(ts),) or not np.isfinite(theta).all()
            or not np.isfinite(h).all() or not np.isfinite(reference).all()
            or not np.allclose(h, h.conj().T, atol=1e-12, rtol=0)
            or abs(np.vdot(reference, reference)-1) > 1e-12):
        raise ValueError('finite Hermitian H, normalized reference and one angle per generator required')
    for t in ts:
        if (t.shape != h.shape or not np.isfinite(t).all()
                or not np.allclose(t.conj().T, -t, atol=1e-12, rtol=0)
                or not np.allclose(t@t@t, -t, atol=1e-12, rtol=0)):
            raise ValueError('generator must be anti-Hermitian with T^3=-T')
    return _energy_gradient(h, reference, ts, theta)


def _energy_gradient(h, reference, ts, theta):
    states = [reference]
    for t, angle in zip(ts, theta):
        states.append(_evolve(t, float(angle), states[-1]))
    adjoint = h@states[-1]
    energy = float(np.vdot(states[-1], adjoint).real)
    gradient = np.empty(len(ts))
    for k in reversed(range(len(ts))):
        gradient[k] = 2*np.vdot(adjoint, ts[k]@states[k+1]).real
        adjoint = _evolve(ts[k], -float(theta[k]), adjoint)
    return energy, gradient, states[-1]


def _prepare_exchange(model, max_sector_dimension):
    """One per-case Hamiltonian, sector and spectrum; pool matrices are lazy."""
    metadata = model.metadata
    if (metadata.get('encoding', 'jw') != 'jw'
            or metadata.get('spin_convention') != 'interleaved'):
        raise ValueError('exchange ADAPT requires interleaved Jordan--Wigner occupations')
    backend = SectorStatevectorBackend(model.n, metadata['n_electrons'], metadata['sz'])
    if backend.dimension > max_sector_dimension:
        raise ValueError('dense exchange benchmark sector exceeds its declared dimension cap')
    h = backend.operator(model.hamiltonian).restrict(np.arange(backend.dimension))
    qe, ceo = exchange_pools(model.n)
    return dict(model=model, backend=backend, h=h,
                reference=backend.state_from_program(model.reference), qe=qe, ceo=ceo,
                h_mv=model.hamiltonian.to_mv(), ground=float(np.linalg.eigvalsh(h)[0]),
                matrices={}, commutators={})


def _select_exchange(pool_labels, scores, tolerance):
    """First member within an absolute derivative tolerance of the maximum."""
    maximum = max(abs(scores[label]) for label in pool_labels)
    return next(label for label in pool_labels if maximum-abs(scores[label]) <= tolerance)


def run_exchange_adapt(model, *, arm='qe', max_iterations=32, maxiter=350,
                       gradient_threshold=1e-6, parent_threshold=None,
                       optimizer_gtol=1e-8, max_sector_dimension=256,
                       selection_tolerance=1e-10, _prepared=None):
    """QE, OVP-only CEO or mixed-CEO ADAPT with common exact optimization.

    Repeated candidates are allowed. Every screening round evaluates the full
    fixed pool. Mixed-CEO parent gradients are linear combinations of the same
    QE expectations: no additional measurement setting is claimed. Operator
    matrices and commutators are built once and their time is outside optimizer
    evaluations. The ground energy is an oracle diagnostic, never a selector.
    """
    if arm not in {'qe', 'ovp_ceo', 'mixed_ceo'}:
        raise ValueError('unknown exchange ADAPT arm')
    for name, value in [('max_iterations',max_iterations),('maxiter',maxiter),
                        ('max_sector_dimension',max_sector_dimension)]:
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if parent_threshold is None:
        parent_threshold = max(10*optimizer_gtol, gradient_threshold/10)
    for value in (gradient_threshold,parent_threshold,optimizer_gtol,selection_tolerance):
        if isinstance(value,bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('thresholds must be finite and positive')
    prepared = _prepare_exchange(model, max_sector_dimension) if _prepared is None else _prepared
    if prepared['model'] is not model:
        raise ValueError('prepared exchange data belongs to another model')
    backend = prepared['backend']
    if backend.dimension > max_sector_dimension:
        raise ValueError('dense exchange benchmark sector exceeds its declared dimension cap')
    h, reference = prepared['h'], prepared['reference']
    qe, ceo = prepared['qe'], prepared['ceo']
    by_label = {op.label:op for op in qe+ceo}
    matrices, commutators = prepared['matrices'], prepared['commutators']
    def matrix(label):
        if label in matrices:
            return matrices[label]
        t = backend.operator(by_label[label].generator).restrict(np.arange(backend.dimension))
        if not np.allclose(t@t@t, -t, atol=1e-12, rtol=0):
            raise ValueError('sector generator violates the exact evolution identity')
        matrices[label] = t
        return t
    pool = qe if arm == 'qe' else ceo
    pool_labels = [op.label for op in pool]
    for op in pool:
        matrix(op.label)
        if op.label not in commutators:
            c = prepared['h_mv']*op.generator-op.generator*prepared['h_mv']
            commutators[op.label] = {word for word, coefficient in c.terms.items()
                                    if word != 0 and coefficient != 0}
    gradient_words = set().union(*(commutators[label] for label in pool_labels))
    exact = prepared['ground']
    psi = reference.copy()
    reference_energy = energy = float(np.vdot(psi, h@psi).real)
    chosen, theta, trajectory, screening = [], (), [], []
    reason = 'iteration_budget'
    for iteration in range(max_iterations):
        hp = h@psi
        scores = {label:float(2*np.vdot(hp, matrices[label]@psi).real) for label in pool_labels}
        norm = float(np.linalg.norm(list(scores.values())))
        if not pool:
            reason = 'empty_pool'
            break
        selected = _select_exchange(pool_labels, scores, selection_tolerance)
        screening.append({'iteration':iteration+1, 'gradient_norm':norm,
                          'max_abs_gradient':abs(scores[selected]),
                          'screened_gradients':len(pool), 'selected_label':selected,
                          'unique_commutator_words':len(gradient_words),
                          'commutator_word_occurrences':sum(len(commutators[label]) for label in pool_labels),
                          'stopped':norm <= gradient_threshold})
        if norm <= gradient_threshold:
            reason = 'gradient_threshold'
            break
        op = by_label[selected]
        appended = [selected]
        parent_scores = {}
        if arm == 'mixed_ceo' and op.parents:
            support_parents = [item for item in qe if item.support == op.support]
            # Recover each QE derivative from the already screened +/- pair.
            # No additional state expectation is performed for MVP selection.
            for item in support_parents:
                plus = next(candidate for candidate in ceo
                            if candidate.sign == 1 and item.label in candidate.parents)
                minus = next(candidate for candidate in ceo
                             if candidate.parents == plus.parents and candidate.sign == -1)
                orientation = 1 if item.label == plus.parents[0] else -1
                parent_scores[item.label] = .5*(scores[plus.label]+orientation*scores[minus.label])
            nonzero = [label for label,value in parent_scores.items() if abs(value)>parent_threshold]
            if len(nonzero) > 1:
                appended = nonzero
        chosen.extend(appended)
        theta = theta + (0.,)*len(appended)
        ts = [matrix(label) for label in chosen]
        previous_energy = energy
        evaluations, cache, callback_indices = [], {}, []
        def objective(x):
            key = tuple(float(v) for v in x)
            if key not in cache:
                e, gradient, state = _energy_gradient(h, reference, ts, key)
                cache[key] = (len(evaluations), e, gradient, state)
                evaluations.append({'parameters':list(key), 'energy':e,
                                    'gradient':gradient.tolist()})
            index, e, gradient, _ = cache[key]
            callback_indices.append(index)
            return e, gradient
        result = minimize_energy(objective,theta,
                                 method='lbfgsb',maxiter=maxiter,gtol=optimizer_gtol,bound=exact)
        theta = result.x
        final_index, energy, final_gradient, psi = cache[theta]
        if callback_indices[-1] != final_index or len(callback_indices) != result.evaluations:
            raise RuntimeError('optimizer evaluation trace does not identify its final state')
        if energy > previous_energy + 1e-10:
            raise RuntimeError('optimization increased the nested ansatz energy')
        if abs(np.vdot(psi,psi)-1)>1e-10:
            raise RuntimeError('exchange evolution lost normalization')
        trajectory.append({'iteration':iteration+1,'selected_label':selected,
                           'appended_labels':appended,'parent_gradients':parent_scores,
                           'labels':list(chosen),'parameters':list(theta),'energy':energy,
                           'optimizer_evaluations':len(evaluations),
                           'optimizer_trace':evaluations,
                           'optimizer_callback_indices':callback_indices,
                           'optimizer_callback_requests':result.evaluations,
                           'optimizer_cache_hits':result.evaluations-len(evaluations),
                           'final_evaluation_index':final_index,
                           'optimizer_iterations':result.iterations,
                           'optimizer_gradient_norm':result.grad_norm,
                           'optimizer_max_abs_gradient':float(np.max(np.abs(final_gradient))),
                           'optimizer_resolved':float(np.max(np.abs(final_gradient))) <= optimizer_gtol,
                           'pauli_rotation_occurrences':sum(len(by_label[label].generator.terms) for label in chosen),
                           'generic_ladder_cnot_upper_bound':sum(2*(sum((code>>(2*j))&3 != 0 for j in range(model.n))-1)
                               for label in chosen for code in by_label[label].generator.terms)})
    resources = {
        'screening_rounds':len(screening),
        'screened_gradients':sum(r['screened_gradients'] for r in screening),
        'screening_unique_words_sum':sum(r['unique_commutator_words'] for r in screening),
        'screening_word_occurrences':sum(r['commutator_word_occurrences'] for r in screening),
        'optimizer_energy_gradient_evaluations':sum(r['optimizer_evaluations'] for r in trajectory),
        'parameter_derivatives':sum(len(r['parameters'])*r['optimizer_evaluations'] for r in trajectory),
        'dense_operator_bytes':h.nbytes + sum(matrices[label].nbytes for label in set(pool_labels+chosen)),
    }
    return {'arm':arm,'sector_basis':backend.basis.tolist(), 'pool_labels':pool_labels,
            'reference_energy':reference_energy,'sector_ground_energy':exact,'energy':energy,
            'trajectory':trajectory,'screening':screening,'stopped_reason':reason,
            'resources':resources,'quantum_shots':None,
            'policy_version':2, 'selection_tolerance':selection_tolerance,
            'parent_promotion_threshold':parent_threshold,
            'evidence_category':'exact_small_sector_simulation',
            'resource_scope':'exact screening and optimizer counts; generic Pauli ladders, no optimized CEO circuits'}
