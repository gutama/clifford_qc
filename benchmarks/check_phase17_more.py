"""Independently replay MORE optimizer calls and native A-CASE root growth.

The scientific replay uses occupation matrices, spectral exponentials and
letter products. Recorded A-CASE eigenframes are verified by residuals and
S-orthogonality before scoring; their degenerate gauge is not independently
claimed as a reproduction of the native canonical eigensolver's gauge.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.phase17_more_design import RECORD, EVIDENCE, EXECUTION_SOURCES, digest, load_config, target_summary
from benchmarks.phase17_more_oracle import (
    matrices, hamiltonian, decode, replay, root_readout, coefficients,
    candidate_vectors, product, generalized, candidate_score,
)
from benchmarks.metadata_io import read_json
from clifford_qc.evidence import validate_evidence


def near(left, right, tolerance=1e-9):
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=tolerance, rtol=0)


def counts_equal(observed, expected):
    return (set(observed) == set(expected) and all(type(observed[key]) is int
            and observed[key] == value for key, value in expected.items()))


def readout_counts(spec, bits, h_terms, ansatz_cnots):
    """Derive the declared H/S/fanout/local-readout construction independently."""
    pairs = [(bits[i], bits[j]) for i in range(len(bits)) for j in range(i + 1, len(bits))]
    word_count = sum(code != 0 and value != 0 for code, value in h_terms.items())
    gates = sum(sum({0: 0, 1: 1, 2: 2, 3: 0}[(code >> (2 * j)) & 3]
                    for j in range(spec['n'])) for code, value in h_terms.items() if code and value != 0)
    families = len(bits) * len(bits)
    x_gates = sum(word.bit_count() for word in bits) + sum(2 * a.bit_count() for a, _ in pairs)
    return {'projected_h_entries': families, 'complex_transition_pairs': len(pairs),
            'reference_families': families, 'independent_word_readout_circuits': word_count * families,
            'generic_cnot_bound': word_count * (families * ansatz_cnots + sum(2 * ((a ^ b).bit_count() - 1) for a, b in pairs)),
            'reference_x_gate_bound': word_count * x_gates,
            'reference_hs_gate_bound': 3 * word_count * len(pairs),
            'local_readout_clifford_gate_bound': families * gates}


def more_problems(run, case, roots, config, h, basis, ts, h_terms):
    problems = []

    def require(value, message):
        if not value:
            problems.append(message)

    bits = case['structure']['reference_bits'][str(roots)]
    references = np.eye(len(basis), dtype=complex)[:, [basis.index(word) for word in bits]]
    spec = case['structure']
    qe, _ = coefficients(spec, bits)
    pool = list(ts)
    require(run['pool_labels'] == pool and run['sector_basis'] == basis, 'MORE pool/sector order')
    spectra = {label: np.linalg.eigh(1j * t) for label, t in ts.items()}
    initial, initial_h = root_readout(h, references)
    require(near(run['initial_root_energies'], initial) and near(decode(run['initial_projected_h']), initial_h), 'MORE initial roots')
    require(counts_equal(run['initial_readout'], readout_counts(spec, bits, h_terms, 0)), 'MORE initial transition readout')
    words = {}
    for label, terms in qe.items():
        ht, th = product(h_terms, terms, spec['n']), product(terms, h_terms, spec['n'])
        words[label] = {code for code in ht.keys() | th.keys()
                        if code and abs(ht.get(code, 0) - th.get(code, 0)) > 1e-12}
    chosen, parameters, states = [], [], references
    previous = float(np.mean(initial))
    options = config['more']
    require(len(run['trajectory']) <= options['max_iterations'], 'MORE growth budget')
    require(len(run['screening']) in (len(run['trajectory']), len(run['trajectory']) + 1), 'MORE terminal screening count')
    for i, screen in enumerate(run['screening']):
        hp = h @ states
        individual = {label: (2 * np.sum(hp.conj() * (t @ states), axis=0).real).tolist()
                      for label, t in ts.items()}
        scores = {label: float(np.mean(values)) for label, values in individual.items()}
        maximum = max(map(abs, scores.values()))
        selected = next(label for label in pool if maximum - abs(scores[label]) <= options['selection_tolerance'])
        require(set(screen['scores']) == set(pool) and set(screen['per_reference_scores']) == set(pool), 'MORE complete screening')
        require(all(near(screen['scores'][label], scores[label], 1e-8)
                    and near(screen['per_reference_scores'][label], individual[label], 1e-8)
                    for label in pool), 'MORE directional gradients')
        norm = float(np.linalg.norm(list(scores.values())))
        require(near(screen['gradient_norm'], norm, 1e-8) and screen['iteration'] == i + 1, 'MORE screening norm/order')
        require(screen['selected_label'] == selected, 'MORE canonical selection')
        stopped = screen['gradient_norm'] <= options['gradient_threshold']
        require(type(screen['stopped']) is bool and screen['stopped'] == stopped, 'MORE stopping flag')
        if i == len(run['trajectory']):
            require((stopped and run['stopped_reason'] == 'gradient_threshold')
                    or (not stopped and len(chosen) == options['max_parameters']
                        and run['stopped_reason'] == 'parameter_budget'), 'MORE terminal stop')
            continue
        require(not stopped, 'MORE continued after gradient stop')
        row = run['trajectory'][i]
        chosen.append(selected)
        require(row['iteration'] == i + 1 and row['selected_label'] == selected
                and row['labels'] == chosen and len(row['parameters']) == len(chosen), 'MORE ansatz/prefix')
        require(len(chosen) <= options['max_parameters'], 'MORE parameter budget')
        trace, callbacks = row['optimizer_trace'], row['optimizer_callback_indices']
        require(bool(trace), 'MORE missing optimizer work')
        require(trace[0]['parameters'] == parameters + [0.0], 'MORE zero initialization/warm start')
        for call in trace:
            require(len(call['parameters']) == len(chosen), 'MORE parameter shape')
            value, gradient, _ = replay(h, references, spectra, chosen, call['parameters'])
            require(near(call['objective'], value) and near(call['gradient'], gradient, 1e-8), 'MORE optimizer replay')
        require(all(type(index) is int and 0 <= index < len(trace) for index in callbacks)
                and list(dict.fromkeys(callbacks)) == list(range(len(trace)))
                and len({tuple(call['parameters']) for call in trace}) == len(trace), 'MORE callback/cache ledger')
        final = row['final_evaluation_index']
        require(type(final) is int and callbacks[-1] == final
                and trace[final]['parameters'] == row['parameters'], 'MORE final optimizer call')
        parameters = row['parameters']
        value, gradient, states = replay(h, references, spectra, chosen, parameters)
        actual_roots, projected = root_readout(h, states)
        require(near(row['objective'], value) and near(row['root_energies'], actual_roots)
                and near(decode(row['projected_h']), projected), 'MORE projected roots/objective')
        require(near(states.conj().T @ states, np.eye(roots), 1e-10), 'MORE orthogonality')
        require(value <= previous + 1e-9, 'MORE nonmonotone objective')
        previous = value
        stored_maximum = max(map(abs, trace[final]['gradient']))
        require(near(row['optimizer_max_abs_gradient'], max(map(abs, gradient)), 1e-8)
                and row['optimizer_max_abs_gradient'] == stored_maximum
                and type(row['optimizer_resolved']) is bool
                and row['optimizer_resolved'] == (stored_maximum <= options['optimizer_gtol']), 'MORE optimizer resolution')
        require(type(row['optimizer_iterations']) is int and 0 <= row['optimizer_iterations'] <= options['maxiter'], 'MORE optimizer budget')
        rotations = sum(len(qe[label]) for label in chosen)
        cnots = sum(2 * (sum((code >> (2 * j)) & 3 != 0 for j in range(spec['n'])) - 1)
                    for label in chosen for code in qe[label])
        require(type(row['pauli_rotation_occurrences']) is int and row['pauli_rotation_occurrences'] == rotations
                and type(row['generic_ladder_cnots_per_reference']) is int
                and row['generic_ladder_cnots_per_reference'] == cnots, 'MORE circuit accounting')
        require(counts_equal(row['readout'], readout_counts(spec, bits, h_terms, cnots)), 'MORE transition readout accounting')
    if run['stopped_reason'] == 'operator_budget':
        require(len(run['trajectory']) == options['max_iterations']
                and len(run['screening']) == len(run['trajectory']), 'MORE budget termination')
    else:
        require(run['stopped_reason'] in ('gradient_threshold', 'parameter_budget'), 'MORE unknown termination')
        require(len(run['screening']) == len(run['trajectory']) + 1, 'MORE missing terminal screen')
    final_roots, final_h = root_readout(h, states)
    require(near(run['root_energies'], final_roots) and near(run['objective'], previous)
            and near(decode(run['projected_h']), final_h), 'MORE final readout')
    exact = np.linalg.eigvalsh(h)[:roots]
    require(near(run['sector_exact_root_energies'], exact), 'MORE oracle roots')
    require(all(value >= bound - 1e-9 for value, bound in zip(final_roots, exact)), 'MORE per-root variational bounds')
    calls = sum(len(row['optimizer_trace']) for row in run['trajectory'])
    derivatives = sum(len(row['parameters']) * len(row['optimizer_trace']) for row in run['trajectory'])
    rounds = len(run['screening'])
    expected = {
        'screening_rounds': rounds, 'screened_state_average_gradients': len(pool) * rounds,
        'screened_reference_gradients': roots * len(pool) * rounds,
        'screening_unique_words_per_reference_sum': len(set().union(*words.values())) * rounds,
        'screening_distinct_word_state_expectations': roots * len(set().union(*words.values())) * rounds,
        'screening_word_expectations': roots * sum(map(len, words.values())) * rounds,
        'optimizer_energy_gradient_evaluations': calls, 'optimizer_reference_energy_evaluations': roots * calls,
        'parameter_derivatives': derivatives, 'reference_parameter_derivatives': roots * derivatives,
        'optimizer_callback_requests': sum(len(row['optimizer_callback_indices']) for row in run['trajectory']),
        'optimizer_cache_hits': sum(len(row['optimizer_callback_indices']) - len(row['optimizer_trace']) for row in run['trajectory']),
        'dense_operator_bytes': h.nbytes + sum(t.nbytes for t in ts.values()),
        'reference_preparation_x_gates_per_batch': sum(word.bit_count() for word in bits),
        'root_readout_eigensolves': len(run['trajectory']) + 1, 'sector_spectrum_eigensolves': 2,
        'projected_h_entries': (len(run['trajectory']) + 1) * roots**2,
        'complex_transition_pairs': (len(run['trajectory']) + 1) * roots * (roots - 1) // 2,
    }
    require(counts_equal(run['resources'], expected), 'MORE resource ledger')
    return problems


def acase_problems(run, case, roots, config, h, basis, ts, h_terms):
    problems = []

    def require(value, message):
        if not value:
            problems.append(message)

    spec = case['structure']
    bits = spec['reference_bits'][str(roots)]
    _, operators = coefficients(spec, bits)
    vectors = candidate_vectors(spec, bits, basis, ts)
    expected_generators = [{'label': label, 'coefficients': [[code, float(value.real).hex(), float(value.imag).hex()]
                            for code, value in sorted(terms.items())]} for label, terms in operators]
    require(run['generators'] == expected_generators, 'A-CASE generator coefficients/registration')
    candidate_indices = list(range(roots, len(operators)))
    require(run['candidate_indices'] == candidate_indices, 'A-CASE candidate pool')
    pencils = [run['initial_pencil']] + [row['pencil'] for row in run['trajectory']]
    options = config['acase']
    require(len(run['trajectory']) <= options['max_size'], 'A-CASE growth budget')
    pair_data, seen = [], set()
    acted = [product(h_terms, terms, spec['n']) for _, terms in operators]
    for stage, i, j, overlap_count, h_count in run['pair_ledger']:
        require(all(type(value) is int for value in (stage, i, j, overlap_count, h_count))
                and 0 <= stage <= len(pencils) and 0 <= i <= j < len(operators)
                and (i, j) not in seen, 'A-CASE pair history')
        seen.add((i, j))
        adjoint = {code: value.conjugate() for code, value in operators[i][1].items()}
        overlap, element = product(adjoint, operators[j][1], spec['n']), product(adjoint, acted[j], spec['n'])
        require(len(overlap) == overlap_count and len(element) == h_count, 'A-CASE pair support counts')
        pair_data.append((stage, i, j, set(overlap) | set(element), len(overlap) + len(element)))

    def resources(indices, stage):
        pairs = [item for item in pair_data if item[0] <= stage]
        selected = set(indices)
        retained = [item for item in pairs if item[1] in selected and item[2] in selected]
        words = set().union(*(item[3] for item in pairs)) if pairs else set()
        selected_words = set().union(*(item[3] for item in retained)) if retained else set()
        occurrences = sum(item[4] for item in pairs)
        return {'pairs_built': len(pairs), 'coefficient_occurrences': occurrences,
                'cached_operator_bytes': 24 * occurrences, 'resident_operator_rows': 2 * len(pairs),
                'word_universe': len(selected_words), 'resident_word_universe': len(words),
                'retained_block_pairs': len(retained), 'selection_pairs': len(pairs) - len(retained),
                'operator_products': 2 * len(pairs) + len(operators),
                'registered_generators': len(operators), 'basis_size': len(indices),
                'independent_word_readout_settings': len(words - {0}),
                'local_readout_cnot_bound': 0,
                'local_readout_clifford_gate_bound': sum(sum({0: 0, 1: 1, 2: 2, 3: 0}[(code >> (2 * j)) & 3]
                                                            for j in range(spec['n'])) for code in words if code)}

    retained_indices = list(range(roots))
    total_scores, terminal_scores = 0, 0
    previous = None
    for step, pencil in enumerate(pencils):
        require(pencil['indices'] == retained_indices, 'A-CASE basis growth/order')
        b = vectors[:, retained_indices]
        s, projected = b.conj().T @ b, b.conj().T @ h @ b
        require(near(decode(pencil['S']), s, 1e-10) and near(decode(pencil['H']), projected), 'A-CASE direct pencil')
        energies, whitening, condition = generalized(s, projected, options)
        c = decode(pencil['coefficients'])
        require(near(pencil['energies'], energies) and type(pencil['effective_rank']) is int
                and pencil['effective_rank'] == len(energies) and near(pencil['condition_number'], condition, 1e-7), 'A-CASE eigenvalues/rank/condition')
        require(c.shape == (len(retained_indices), len(energies)), 'A-CASE eigenframe shape')
        require(near(c.conj().T @ s @ c, np.eye(len(energies)), 1e-8)
                and near(whitening.conj().T @ (projected @ c - (s @ c) * np.asarray(pencil['energies'])),
                         np.zeros((len(energies), len(energies))), 1e-8), 'A-CASE eigenframe residual/orthogonality')
        require(counts_equal(pencil['resources'], resources(retained_indices, step)), 'A-CASE prefix bank accounting')
        objective = float(np.mean(energies[:roots]))
        if step == 0:
            require(near(run['initial_root_energies'], energies[:roots]), 'A-CASE initial roots')
        else:
            row = run['trajectory'][step - 1]
            require(row['iteration'] == step and near(row['root_energies'], energies[:roots])
                    and near(row['objective'], objective), 'A-CASE prefix roots/objective')
            require(objective <= previous + 1e-9, 'A-CASE nonmonotone mean-root objective')
            require(near(row['actual_lowering'], previous - objective)
                    and type(row['basis_size']) is int and row['basis_size'] == len(retained_indices)
                    and type(row['effective_rank']) is int and row['effective_rank'] == len(energies)
                    and near(row['condition_number'], condition, 1e-7), 'A-CASE growth diagnostics')
            current = resources(retained_indices, step)
            prior = resources(pencils[step - 1]['indices'], step - 1)
            require(row['word_universe'] == current['word_universe']
                    and row['selection_word_universe'] == current['resident_word_universe']
                    and row['new_words'] == current['word_universe'] - prior['word_universe'], 'A-CASE growth word accounting')
        previous = objective
        remaining = [index for index in candidate_indices if index not in retained_indices]
        all_states = b @ c
        tracked = all_states[:, :roots]
        scores = {index: candidate_score(h, all_states, tracked, pencil['energies'][:roots],
                                         vectors[:, index], options) for index in remaining}
        live = {index: value for index, value in scores.items() if value is not None}
        if step == len(run['trajectory']):
            if step < options['max_size'] and remaining:
                terminal_scores = len(remaining)
                if not live:
                    require(run['stopped_reason'] == 'every candidate rejected (conditioning or sector)', 'A-CASE rejection stop')
                else:
                    require(max(value[0] for value in live.values()) < options['min_lowering'] + 1e-12
                            and run['stopped_reason'] == 'predicted lowering below threshold', 'A-CASE lowering stop')
            else:
                require(run['stopped_reason'] == ('basis budget reached' if step == options['max_size'] else 'candidate pool exhausted'), 'A-CASE budget/pool stop')
            continue
        require(bool(live), 'A-CASE missing eligible candidate')
        best = max(value[0] for value in live.values())
        tolerance = 1e-12 + 1e-9 * abs(best)
        selected = min(index for index, value in live.items() if value[0] >= best - tolerance)
        row = run['trajectory'][step]
        score, lowerings, fraction = live[selected]
        require(row['selected_label'] == operators[selected][0] and row['candidates_scored'] == len(remaining), 'A-CASE full screen/selection')
        require(near(row['predicted_lowering'], score, 1e-8) and near(row['per_root_lowering'], lowerings, 1e-8)
                and near(row['orthogonal_fraction'], fraction, 1e-8), 'A-CASE candidate score')
        require(score >= options['min_lowering'] - 1e-12, 'A-CASE grew below threshold')
        conditioning = sum(np.vdot(vectors[:, index], vectors[:, index]).real > 1e-24
                           and value is None for index, value in scores.items())
        candidate = vectors[:, selected]
        coupling = float(np.mean([abs(np.vdot(state, h @ candidate) - energy * np.vdot(state, candidate))
                                  / np.linalg.norm(candidate)
                                  for energy, state in zip(pencil['energies'][:roots], tracked.T)]))
        require(near(row['residual_coupling'], coupling, 1e-8)
                and row['rejected_conditioning'] == conditioning and row['rejected_sector'] == 0
                and row['leakage'] is None and row['selection_criterion'] == 'lowering'
                and near(row['selection_score'], score, 1e-8) and row['target_overlap_gain'] is None, 'A-CASE selection diagnostics')
        total_scores += len(remaining)
        retained_indices.append(selected)
    final_roots = pencils[-1]['energies'][:roots]
    require(near(run['root_energies'], final_roots) and near(run['objective'], previous), 'A-CASE final roots/objective')
    require(all(value >= bound - 1e-9 for value, bound in zip(final_roots, np.linalg.eigvalsh(h)[:roots])), 'A-CASE per-root variational bounds')
    expected = resources(retained_indices, len(pencils))
    expected.update({'candidate_score_forms': total_scores + terminal_scores,
                     'terminal_candidate_score_forms': terminal_scores,
                     'root_score_forms': roots * (total_scores + terminal_scores),
                     'root_readout_eigensolves': len(pencils), 'optimizer_energy_gradient_evaluations': 0,
                     'overlap_eigendecompositions': len(pencils),
                     'projected_h_eigendecompositions': len(pencils),
                     'parameter_derivatives': 0, 'nonlinear_parameters': 0,
                     'reference_preparation_x_gates_per_measurement': spec['n_electrons'],
                     'retained_ritz_coefficient_entries': len(retained_indices) * roots,
                     'reachable_family_dimension': int(np.linalg.matrix_rank(vectors, tol=1e-10)),
                     'retained_generator_support': sum(len(operators[index][1]) for index in retained_indices),
                     'sector_spectrum_eigensolves': 2})
    require(counts_equal(run['resources'], expected), 'A-CASE final accounting')
    return problems


def check_record(record, config, *, history=True, check_verdict=True, root=ROOT):
    problems = []
    validate_evidence(record)
    if record['evidence'] != EVIDENCE:
        problems.append('evidence scope differs')
    if record['schema'] != 'clifford_qc.phase17_more.v1' or record['config_sha256'] != digest(config):
        problems.append('record declaration binding differs')
    if record['claim_boundary'] != config['claim_boundary']:
        problems.append('claim boundary differs')
    if check_verdict and (record['verdict'] != 'VALID' or record['validation_problems']):
        problems.append('campaign is not VALID')
    if 'execution' in record and record['execution'].get('serial_fresh_workers') is not True:
        problems.append('fresh serial worker declaration differs')
    expected = [(case['id'], roots, arm) for case in config['cases'] for roots in config['roots'] for arm in config['arms']]
    if [(cell['case'], cell['roots'], cell['arm']) for cell in record['cells']] != expected:
        return problems + ['missing, repeated or reordered cells']
    contexts, identities = {}, set()
    for cell in record['cells']:
        name = f'{cell["case"]}/{cell["roots"]}/{cell["arm"]}'
        if cell['status'] != 'completed':
            problems.append(name + ': incomplete cell ' + cell['status'])
            continue
        case = next(case for case in config['cases'] if case['id'] == cell['case'])
        roots = cell['roots']
        if case['id'] not in contexts:
            basis, ts = matrices(case['structure'])
            h, h_terms = hamiltonian(root / case['path'], basis)
            contexts[case['id']] = basis, ts, h, h_terms
        basis, ts, h, h_terms = contexts[case['id']]
        run = cell['run']
        bits = case['structure']['reference_bits'][str(roots)]
        if run['arm'] != cell['arm'] or run['roots'] != roots or run['reference_bits'] != bits or run['quantum_shots'] is not None:
            problems.append(name + ': arm/reference/evidence identity')
        exact = np.linalg.eigvalsh(h)[:roots]
        if not near(cell['exact_root_energies'], exact) or cell['target_summary'] != target_summary(run, cell['exact_root_energies'], config['target_hartree']):
            problems.append(name + ': oracle/target readout')
        checker = more_problems if cell['arm'] == 'more_adapt' else acase_problems
        try:
            problems += [name + ': ' + message for message in checker(run, case, roots, config, h, basis, ts, h_terms)]
        except (ValueError, KeyError, IndexError, TypeError, np.linalg.LinAlgError) as error:
            problems.append(name + ': malformed scientific trace: ' + str(error))
        if 'worker' in cell:
            worker, parent = cell['worker'], cell['parent']
            if any(type(block[key]) is not int or block[key] < 1 for block, key in
                   ((worker, 'pid'), (worker, 'parent_pid'), (worker, 'start_ticks'),
                    (parent, 'pid'), (parent, 'worker_pid'))):
                problems.append(name + ': invalid process identity type')
            if any(type(block[key]) is not int or block[key] < 0 for block, key in
                   ((worker, 'peak_rss_bytes'), (parent, 'peak_rss_bytes'), (parent, 'returncode'))):
                problems.append(name + ': invalid observed byte/completion count')
            identity = (worker['pid'], worker['start_ticks'])
            if identity in identities or worker['pid'] != parent['worker_pid'] or worker['parent_pid'] != parent['pid']:
                problems.append(name + ': worker isolation/parent identity')
            identities.add(identity)
            stamp = record['provenance']
            if worker['git_sha'] != stamp['git_sha'] or worker['python'] != stamp['python'] or any(worker[key] != stamp['dependencies'][key] for key in ('numpy', 'scipy')):
                problems.append(name + ': worker source/interpreter/libraries')
            if worker['threads'] != {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}:
                problems.append(name + ': worker BLAS threads')
            if any(parent[key] != config['execution'][key] for key in ('rss_ceiling_bytes', 'timeout_seconds', 'sample_seconds')) or parent['returncode'] != 0:
                problems.append(name + ': execution supervision')
            if max(worker['peak_rss_bytes'], parent['peak_rss_bytes']) > config['execution']['rss_ceiling_bytes'] or parent['wall_seconds'] > config['execution']['timeout_seconds']:
                problems.append(name + ': execution ceiling/deadline')
            if any(isinstance(value, bool) or not np.isfinite(value) or value < 0 for value in
                   (worker['prepare_seconds'], worker['algorithm_seconds'], parent['wall_seconds'])):
                problems.append(name + ': invalid observed time')
        elif history:
            problems.append(name + ': missing process observations')
    # Both methods must begin from the same span, not only the same root count.
    for i in range(0, len(record['cells']), 2):
        left, right = record['cells'][i:i + 2]
        if left['status'] == right['status'] == 'completed' and not near(left['run']['initial_root_energies'], right['run']['initial_root_energies']):
            problems.append(left['case'] + ': unmatched initial reference roots')
    if history:
        from benchmarks.check_phase17_more_declaration import history_problems
        if record.get('execution', {}).get('serial_fresh_workers') is not True:
            problems.append('fresh process execution observations are required')
        if set(record['execution_source_bindings']) != set(EXECUTION_SOURCES):
            problems.append('execution source binding set differs')
        problems += history_problems(config, record, root)
        reference = read_json(root / config['environment_reference'])['provenance']
        stamp = record['provenance']
        if stamp['python'].split('.')[:2] != reference['python'].split('.')[:2] or any(stamp['dependencies'][key] != reference['dependencies'][key] for key in ('numpy', 'scipy')):
            problems.append('record producer is outside the licensed environment')
    return problems


def main():
    from benchmarks.check_phase17_more_declaration import validate
    config = load_config()
    problems = validate(config)
    if RECORD.exists():
        problems += check_record(read_json(RECORD), config)
    if problems:
        raise ValueError('; '.join(problems[:20]))
    print('Phase 17 MORE: PASS (independent calls, pencils, scores and ledgers)' if RECORD.exists()
          else 'Phase 17 MORE: outcome-free design only; campaign pending')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
