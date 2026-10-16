"""Result-free structure and readout policy for Phase 17 MORE-ADAPT breadth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.metadata_io import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'benchmarks/configs/phase17_more.json'
RECORD = ROOT / 'benchmarks/reference_results/phase17_more.json'
ARMS = ('more_adapt', 'acase_mean')
EVIDENCE = {
    'root_energies_and_gradients': {
        'tier': 'exact_small_sector_simulation',
        'scope': 'Uniform-weight shared-QE-unitary ADAPT and native mean-root A-CASE on frozen small N/Sz sectors; local optimizer trajectories, not global optimality.'},
    'screening_and_optimizer_ledgers': {
        'tier': 'classical_process_measurement',
        'scope': 'Observed complete-pool screens, retained distinct optimizer calls, callback/cache requests, and bank-pair creation; separate currencies, not a shot or total-cost estimate.'},
    'generic_circuit_bounds': {
        'tier': 'exact_algebra',
        'scope': 'Commuting-QE Pauli parity-ladder bounds for MORE ansatzes, explicit two-quadrature reference fanouts for its root transitions, and independent local Pauli readouts for A-CASE; no optimized circuits, shot price or measured device cost.'},
    'wall_time_and_rss': {
        'tier': 'classical_process_measurement',
        'scope': 'One fresh interpreter per cell, including preparation and instrumentation; single-host observations, not intrinsic row allocation or universal bounds.'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def structure(path):
    """Freeze operators and occupations without a spectrum or optimizer call."""
    from benchmarks.phase17_ceo_design import structure as ceo_structure
    from clifford_qc.backends.sector_statevector import sector_basis
    result = ceo_structure(path)
    result.pop('ovp_ceo')
    basis = sector_basis(result['n'], result['n_electrons'], result['sz']).tolist()
    hf = sum(1 << (result['n'] - 1 - j) for j in result['reference_occupied'])
    # Frontier singles are a physical reference policy, fixed from occupations
    # alone: HOMO first, LUMO first, then alpha/beta. Ascending sector integers
    # would instead put an all-virtual complement beside HF on these inputs.
    singles = []
    for word in basis:
        if (int(word) ^ hf).bit_count() != 2:
            continue
        removed = next(q for q in range(result['n']) if hf & (1 << (result['n'] - 1 - q))
                       and not int(word) & (1 << (result['n'] - 1 - q)))
        added = next(q for q in range(result['n']) if int(word) & (1 << (result['n'] - 1 - q))
                     and not hf & (1 << (result['n'] - 1 - q)))
        singles.append(((-(removed // 2), added // 2, removed % 2, int(word)), int(word)))
    ordered = [hf] + [word for _, word in sorted(singles)]
    ordered += [int(word) for word in basis if int(word) not in ordered]
    result['sector_basis'] = basis
    result['reference_bits'] = {str(roots): ordered[:roots] for roots in (2, 3)}
    result['response_family_dimensions'] = {
        str(roots): len(reachable(result['n'], result['qe'], ordered[:roots])) for roots in (2, 3)}
    return result


def reachable(n, pool, references):
    """Structural determinant support of the static T_i P_j response family."""
    result = set(references)
    for op in pool:
        creation, annihilation = op['creation'], op['annihilation']
        for word in references:
            occupied = lambda q: bool(word & (1 << (n - 1 - q)))
            forward = all(not occupied(q) for q in creation) and all(occupied(q) for q in annihilation)
            reverse = all(occupied(q) for q in creation) and all(not occupied(q) for q in annihilation)
            if forward or reverse:
                target = word
                for q in creation + annihilation:
                    target ^= 1 << (n - 1 - q)
                result.add(target)
    return result


def load_config(path=CONFIG):
    return read_json(path)


def target_summary(run, exact, target):
    """First retained prefix satisfying every ordered root; no interpolation."""
    initial = run['initial_root_energies']
    rows = [{'iteration': 0, 'root_energies': initial}] + run['trajectory']
    for row in rows:
        errors = [float(value - bound) for value, bound in zip(row['root_energies'], exact)]
        maximum = max(map(abs, errors))
        if len(errors) == len(exact) and maximum <= target:
            return {'status': 'REFERENCE_ALREADY_WITHIN_TARGET' if row['iteration'] == 0 else 'REACHED',
                    'iteration': row['iteration'], 'per_root_errors': errors,
                    'max_root_error': maximum}
    errors = [float(value - bound) for value, bound in zip(run['root_energies'], exact)]
    return {'status': 'RIGHT_CENSORED', 'iteration': None,
            'per_root_errors': errors, 'max_root_error': max(map(abs, errors))}


def source_bindings(paths, root=ROOT):
    return {path: sha256(root / path) for path in paths}


EXECUTION_SOURCES = (
    'benchmarks/run_phase17_more.py', 'benchmarks/check_phase17_more.py',
    'benchmarks/check_phase17_more_declaration.py',
)
