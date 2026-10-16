"""Uniform-weight multireference ADAPT with a shared QE unitary.

This bounded MORE-ADAPT variant minimizes Tr(Phi† H Phi)/r for orthonormal
reference columns Phi. A common unitary preserves their orthogonality. Sorted
Rayleigh--Ritz eigenvalues of Phi† H Phi, rather than the individual diagonal
expectations, provide the root readout. It is an exact small-sector algorithm,
not a reproduction of a published optimized circuit or finite-shot protocol.
"""
from __future__ import annotations

import math

import numpy as np

from .exchange import _energy_gradient, _prepare_exchange, _select_exchange
from .optimize import minimize_energy


def _gradient(h, references, generators, parameters):
    # The exchange adjoint also accepts a matrix of state columns: vdot sums
    # over its entries. Divide once to obtain the uniform state average.
    total, gradient, states = _energy_gradient(h, references, generators, parameters)
    roots = references.shape[1]
    return total / roots, gradient / roots, states


def more_energy_gradient(h, references, generators, parameters):
    """Return the uniform objective, exact adjoint gradient and evolved columns.

    References are a D-by-r isometry. Exchange generators must be anti-Hermitian
    and satisfy T³ = -T, which makes their trigonometric evolution exact.
    """
    h = np.asarray(h, complex)
    references = np.asarray(references, complex)
    ts = [np.asarray(t, complex) for t in generators]
    theta = np.asarray(parameters, float)
    if (references.ndim != 2 or not references.shape[1]
            or h.shape != (references.shape[0], references.shape[0])
            or theta.shape != (len(ts),) or not np.isfinite(theta).all()
            or not np.isfinite(h).all() or not np.isfinite(references).all()
            or not np.allclose(h, h.conj().T, atol=1e-12, rtol=0)
            or not np.allclose(references.conj().T @ references,
                               np.eye(references.shape[1]), atol=1e-12, rtol=0)):
        raise ValueError('finite Hermitian H, orthonormal reference columns and one angle per generator required')
    for t in ts:
        if (t.shape != h.shape or not np.isfinite(t).all()
                or not np.allclose(t.conj().T, -t, atol=1e-12, rtol=0)
                or not np.allclose(t @ t @ t, -t, atol=1e-12, rtol=0)):
            raise ValueError('generator must be anti-Hermitian with T^3=-T')
    return _gradient(h, references, ts, theta)


def matrix_record(matrix):
    """Retain complex matrix entries without decimal round-trip ambiguity."""
    return [[[float(value.real).hex(), float(value.imag).hex()] for value in row]
            for row in np.asarray(matrix)]


def _roots(h, states):
    projected = states.conj().T @ h @ states
    projected = (projected + projected.conj().T) / 2
    return np.linalg.eigvalsh(projected).tolist(), matrix_record(projected)


def readout_bounds(n, h_terms, reference_bits, ansatz_cnots):
    """Logical bounds for one independent-word sweep of the r-by-r H readout.

    Diagonals use r original references. Each complex upper-triangle transition
    uses (|i>+|j>)/sqrt(2) and (|i>+i|j>)/sqrt(2), followed by the common ansatz.
    A pivot H, optional S and d-1 fanout CX gates prepare each superposition
    from |i>, choosing a differing pivot whose |i> bit is zero. Equal occupation
    counts guarantee such a pivot. Local Pauli readout requires no CX gates.
    These are unoptimized algebraic bounds, not executed shots or total costs.
    """
    from itertools import combinations
    roots = len(reference_bits)
    pairs = list(combinations(reference_bits, 2))
    h_words = [code for code, value in h_terms.items() if code and value != 0]
    fanout = 2 * sum((left ^ right).bit_count() - 1 for left, right in pairs)
    basis_gates = sum(sum(1 if (code >> (2 * j)) & 3 == 1 else
                              2 if (code >> (2 * j)) & 3 == 2 else 0
                              for j in range(n)) for code in h_words)
    families = roots**2
    preparation_x = sum(word.bit_count() for word in reference_bits)
    preparation_x += 2 * sum(left.bit_count() for left, _ in pairs)
    return {'projected_h_entries': roots**2, 'complex_transition_pairs': len(pairs),
            'reference_families': families,
            'independent_word_readout_circuits': len(h_words) * families,
            'generic_cnot_bound': len(h_words) * (families * ansatz_cnots + fanout),
            'reference_x_gate_bound': len(h_words) * preparation_x,
            'reference_hs_gate_bound': len(h_words) * 3 * len(pairs),
            'local_readout_clifford_gate_bound': families * basis_gates}


def run_more_adapt(model, reference_bits, *, max_iterations=32,
                   max_parameters=32, maxiter=350, gradient_threshold=1e-6,
                   optimizer_gtol=1e-8, selection_tolerance=1e-10,
                   max_sector_dimension=256, _prepared=None):
    """Grow a shared QE ansatz against the uniform multireference objective.

    References are distinct occupation integers in the model's N/Sz sector.
    Every screen visits the complete generalized QE pool for every reference.
    Repeated operators are allowed; all parameters are reoptimized, with a new
    zero angle. The spectrum is only a lower-bound check and readout diagnostic.
    """
    for name, value in (('max_iterations', max_iterations), ('max_parameters', max_parameters),
                        ('maxiter', maxiter), ('max_sector_dimension', max_sector_dimension)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for value in (gradient_threshold, optimizer_gtol, selection_tolerance):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('thresholds must be finite and positive')
    bits = list(reference_bits)
    if (not bits or any(isinstance(word, (bool, np.bool_)) or not isinstance(word, (int, np.integer)) for word in bits)
            or len(set(bits)) != len(bits)):
        raise ValueError('references must be distinct integer occupations')
    bits = [int(word) for word in bits]
    prepared = _prepare_exchange(model, max_sector_dimension) if _prepared is None else _prepared
    if prepared['model'] is not model:
        raise ValueError('prepared exchange data belongs to another model')
    backend, h, pool = prepared['backend'], prepared['h'], prepared['qe']
    if backend.dimension > max_sector_dimension:
        raise ValueError('dense MORE-ADAPT sector exceeds its declared dimension cap')
    index = {int(word): i for i, word in enumerate(backend.basis)}
    if any(word not in index for word in bits):
        raise ValueError('reference occupation is outside the declared N/Sz sector')
    references = np.eye(backend.dimension, dtype=complex)[:, [index[word] for word in bits]]
    spectra = np.linalg.eigvalsh(h)[:len(bits)]
    lower_bound = float(np.mean(spectra))
    matrices, commutators = {}, {}
    for op in pool:
        t = backend.operator(op.generator).restrict(np.arange(backend.dimension))
        if not np.allclose(t @ t @ t, -t, atol=1e-12, rtol=0):
            raise ValueError('sector generator violates the exact evolution identity')
        matrices[op.label] = t
        commutator = prepared['h_mv'] * op.generator - op.generator * prepared['h_mv']
        commutators[op.label] = {word for word, value in commutator.terms.items()
                               if word != 0 and value != 0}
    labels = [op.label for op in pool]
    by_label = {op.label: op for op in pool}
    gradient_words = set().union(*commutators.values()) if pool else set()
    word_occurrences = sum(map(len, commutators.values()))
    states = references.copy()
    initial_roots, initial_projected = _roots(h, states)
    initial_readout = readout_bounds(model.n, prepared['h_mv'].terms, bits, 0)
    energy = float(np.mean(initial_roots))
    chosen, theta, trajectory, screening = [], (), [], []
    reason = 'operator_budget'
    for iteration in range(max_iterations):
        if not pool:
            reason = 'empty_pool'
            break
        hp = h @ states
        per_reference = {label: (2 * np.sum(hp.conj() * (matrices[label] @ states), axis=0).real).tolist()
                         for label in labels}
        scores = {label: float(np.mean(values)) for label, values in per_reference.items()}
        norm = float(np.linalg.norm(list(scores.values())))
        selected = _select_exchange(labels, scores, selection_tolerance)
        stopped = norm <= gradient_threshold
        screening.append({'iteration': iteration + 1, 'scores': scores,
                          'per_reference_scores': per_reference, 'gradient_norm': norm,
                          'selected_label': selected, 'stopped': stopped})
        if stopped:
            reason = 'gradient_threshold'
            break
        if len(chosen) >= max_parameters:
            reason = 'parameter_budget'
            break
        chosen.append(selected)
        theta += (0.0,)
        ts = [matrices[label] for label in chosen]
        previous = energy
        evaluations, cache, callbacks = [], {}, []

        def objective(x):
            key = tuple(float(value) for value in x)
            if key not in cache:
                value, gradient, evolved = _gradient(h, references, ts, key)
                cache[key] = (len(evaluations), value, gradient, evolved)
                evaluations.append({'parameters': list(key), 'objective': value,
                                    'gradient': gradient.tolist()})
            call_index, value, gradient, _ = cache[key]
            callbacks.append(call_index)
            return value, gradient

        optimized = minimize_energy(objective, theta, method='lbfgsb', maxiter=maxiter,
                                    gtol=optimizer_gtol, bound=lower_bound)
        theta = optimized.x
        final_index, energy, gradient, states = cache[theta]
        if callbacks[-1] != final_index or len(callbacks) != optimized.evaluations:
            raise RuntimeError('optimizer trace does not identify its final state')
        if energy > previous + 1e-10:
            raise RuntimeError('optimization increased the nested state-average objective')
        if not np.allclose(states.conj().T @ states, np.eye(len(bits)), atol=1e-10, rtol=0):
            raise RuntimeError('shared unitary lost reference orthogonality')
        roots, projected = _roots(h, states)
        if any(value < bound - 1e-9 for value, bound in zip(roots, spectra)):
            raise RuntimeError('multi-root Rayleigh--Ritz bound violated')
        trajectory.append({'iteration': iteration + 1, 'selected_label': selected,
                           'labels': list(chosen), 'parameters': list(theta),
                           'objective': energy, 'root_energies': roots,
                           'projected_h': projected, 'optimizer_trace': evaluations,
                           'optimizer_callback_indices': callbacks,
                           'optimizer_iterations': optimized.iterations,
                           'final_evaluation_index': final_index,
                           'optimizer_max_abs_gradient': float(np.max(np.abs(gradient))),
                           'optimizer_resolved': float(np.max(np.abs(gradient))) <= optimizer_gtol,
                           'pauli_rotation_occurrences': sum(len(by_label[label].generator.terms) for label in chosen),
                           'generic_ladder_cnots_per_reference': sum(
                               2 * (sum((code >> (2 * j)) & 3 != 0 for j in range(model.n)) - 1)
                               for label in chosen for code in by_label[label].generator.terms)})
        trajectory[-1]['readout'] = readout_bounds(
            model.n, prepared['h_mv'].terms, bits, trajectory[-1]['generic_ladder_cnots_per_reference'])
    root_energies, final_projected = ((trajectory[-1]['root_energies'], trajectory[-1]['projected_h'])
                                      if trajectory else (initial_roots, initial_projected))
    calls = sum(len(row['optimizer_trace']) for row in trajectory)
    parameter_derivatives = sum(len(row['parameters']) * len(row['optimizer_trace']) for row in trajectory)
    resources = {
        'screening_rounds': len(screening), 'screened_state_average_gradients': len(pool) * len(screening),
        'screened_reference_gradients': len(bits) * len(pool) * len(screening),
        'screening_unique_words_per_reference_sum': len(gradient_words) * len(screening),
        'screening_distinct_word_state_expectations': len(bits) * len(gradient_words) * len(screening),
        'screening_word_expectations': len(bits) * word_occurrences * len(screening),
        'optimizer_energy_gradient_evaluations': calls,
        'optimizer_reference_energy_evaluations': len(bits) * calls,
        'parameter_derivatives': parameter_derivatives,
        'reference_parameter_derivatives': len(bits) * parameter_derivatives,
        'optimizer_callback_requests': sum(len(row['optimizer_callback_indices']) for row in trajectory),
        'optimizer_cache_hits': sum(len(row['optimizer_callback_indices']) - len(row['optimizer_trace']) for row in trajectory),
        'dense_operator_bytes': h.nbytes + sum(matrix.nbytes for matrix in matrices.values()),
        'reference_preparation_x_gates_per_batch': sum(word.bit_count() for word in bits),
        'root_readout_eigensolves': len(trajectory) + 1,
        'sector_spectrum_eigensolves': 1 + int(_prepared is None),
        'projected_h_entries': (len(trajectory) + 1) * len(bits)**2,
        'complex_transition_pairs': (len(trajectory) + 1) * len(bits) * (len(bits) - 1) // 2,
    }
    return {'arm': 'more_adapt', 'reference_bits': bits, 'roots': len(bits),
            'sector_basis': backend.basis.tolist(), 'pool_labels': labels,
            'initial_root_energies': initial_roots, 'initial_projected_h': initial_projected,
            'initial_readout': initial_readout,
            'sector_exact_root_energies': spectra.tolist(),
            'root_energies': root_energies, 'projected_h': final_projected,
            'objective': energy, 'trajectory': trajectory, 'screening': screening,
            'stopped_reason': reason, 'resources': resources, 'quantum_shots': None}
