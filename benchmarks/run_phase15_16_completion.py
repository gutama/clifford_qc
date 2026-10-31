"""Reproducible bounded completion of Phase 15 and the Phase 16B Taylor path."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json, write_json
from benchmarks.run_sampled_ratio import JointShotSimulator
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.measurement.ratio import uniform_setting_plan
from clifford_qc.models import tfim
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.reference_states import DeterminantReference, StabilizerReference
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace import (
    SampledSecondMomentProvider, SecondMomentStoragePolicy,
    truncated_time_family, validate_truncated_time_family,
)
from clifford_qc.subspace.providers import ExactProjectedProvider
from clifford_qc.dense_reference import to_matrix

DESIGN = 'docs/restart/phase15_16_completion_design.json'
CONTRACT = 'docs/restart/PHASE15_16_COMPLETION_CONTRACT.md'
RESULT = 'docs/restart/phase15_16_completion_results.json.gz'
HISTORICAL = ('phase15_h2_preflight.json', 'phase15_residual_successor.json',
              'phase15_certifying_screen.json', 'phase15_certifying_krylov.json',
              'phase15_fc_residual.json', 'phase16a_te_qsci.json',
              'phase16b_feasibility.json', 'phase16b_v2_feasibility.json',
              'phase16b_v3_feasibility.json')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def sources():
    return sorted([p for p in git('ls-files', 'clifford_qc').splitlines() if p.endswith('.py')]
            + ['benchmarks/run_phase15_16_completion.py', 'benchmarks/check_phase15_16_completion.py',
               'benchmarks/run_sampled_ratio.py', 'benchmarks/check_observable_noise.py', DESIGN, CONTRACT])


def complex_record(array):
    a = np.asarray(array)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def coefficient_vector(values):
    a = np.asarray(values)
    return a[..., 0]+1j*a[..., 1]


def build_case(name):
    if name not in ('tfim2', 'tfim3', 'hubbard2'):
        raise ValueError('unknown completion case')
    if name.startswith('tfim'):
        n = {'tfim2': 2, 'tfim3': 3}[name]
        h = tfim(n, J=1., h=1.2).hamiltonian.to_mv()
        psi = np.ones(2**n, complex)/np.sqrt(2**n)
        gens = [('I', MV.scalar(n, 1)), ('ZZ0', MV.from_label('ZZ'+'I'*(n-2)))]
        if n == 3:
            gens.append(('ZZ1', MV.from_label('IZZ')))
        exact = StabilizerReference(n, [(1 << (2*q), 1) for q in range(n)], spin_ordering='qubits')
    else:
        n = 4
        h = hubbard(2, t=1., U=4., mu=2.).hamiltonian.to_mv()
        psi = np.zeros(16, complex); psi[9] = 1.
        a = cdag_op(4, 0)*c_op(4, 2)+cdag_op(4, 2)*c_op(4, 0)
        b = cdag_op(4, 1)*c_op(4, 3)+cdag_op(4, 3)*c_op(4, 1)
        gens = [('I', MV.scalar(4, 1)), ('hopA', a), ('hopB', b), ('hopAB', a*b)]
        exact = DeterminantReference('1001')
    return h, psi, gens, exact


def make_provider(reference, h, gens, config):
    return SampledSecondMomentProvider(reference, h, gens,
        max_words=config['max_words'], max_settings=config['max_settings'],
        storage_policy=SecondMomentStoragePolicy(config['max_coefficient_occurrences']))


def settings_record(settings, device=None):
    return [{'assigned': list(s.assigned_word_codes),
             'readouts': {str(w): [int(sign), list(positions)]
                          for w, (sign, positions) in s.readouts.items()},
             'gates': [[name, list(qubits)] for name, qubits in s.clifford.gates],
             **({'sampling_probabilities': device.probabilities(s).tolist()} if device is not None else {})}
            for s in settings]


def observations_record(observations):
    groups = {group.setting_key: group for group in observations.batch.groups}
    # The caller inserts histograms in the compiled setting order below.
    return groups


def matrices_record(problem):
    return {'S': complex_record(problem.overlap), 'H': complex_record(problem.hamiltonian),
            'K': complex_record(problem.second_moment),
            'rS': problem.overlap_radii.tolist(), 'rH': problem.hamiltonian_radii.tolist(),
            'rK': problem.second_moment_radii.tolist()}


def assess_direction(h, basis, coefficients, target):
    coefficients = coefficients/np.linalg.norm(coefficients)
    state = basis @ coefficients
    norm = float(np.vdot(state, state).real)
    state /= np.sqrt(norm)
    energy = float(np.vdot(state, h @ state).real)
    second = float(np.vdot(h @ state, h @ state).real)
    variance = float(np.linalg.norm(h @ state-energy*state)**2)
    folded = float(np.linalg.norm(h @ state-target*state)**2)
    return {'normalization': norm, 'energy': energy, 'second_moment': second,
            'variance': variance, 'residual_norm': float(np.sqrt(variance)),
            'folded_value': folded}


def infer(problem, config, target):
    try:
        solved = problem.solve(tau_s=config['tau_s'], max_condition=config['max_condition'])
        c = solved.ritz_vector()
        report = problem.assess(c, residual_tolerance=config['residual_tolerance'], target=target,
                                normalization_floor=config['normalization_floor'])
        folded = problem.folded_roots(target, residual_tolerance=config['residual_tolerance'],
                    normalization_floor=config['normalization_floor'], count=1,
                    tau_s=config['tau_s'], max_condition=config['max_condition'])
        return {'status': 'solved', 'energy': solved.ground_energy,
                'coefficients': complex_record(c), 'retained_rank': solved.effective_rank,
                'retained_condition': solved.condition_number,
                'convergence': report, 'folded': folded}
    except (ValueError, np.linalg.LinAlgError) as error:
        return {'status': 'solve_failed', 'error_type': type(error).__name__}


def run(config, *, execution_commit, declaration_commit):
    rows, cases, time_cases, integrations = [], [], [], []
    for index, name in enumerate(config['cases']):
        h, psi, gens, exact_reference = build_case(name)
        n = h.n
        device = JointShotSimulator(psi, n)
        reference = PreparedSampledReference(n, 'phase15_16.v1/'+name, device.draw,
                             spin_ordering='interleaved' if name == 'hubbard2' else 'qubits')
        provider = make_provider(reference, h, gens, config)
        dense_h = to_matrix(h)
        basis = np.column_stack([to_matrix(g) @ psi for _, g in gens])
        true_blocks = (basis.conj().T @ basis, basis.conj().T @ dense_h @ basis,
                       (dense_h @ basis).conj().T @ (dense_h @ basis))
        target = config['folded_targets'][name]
        cases.append({'case': name, 'n': n, 'admission': provider.admission,
                      'settings': settings_record(provider.plan.settings, device),
                      'spectrum': np.linalg.eigvalsh(dense_h).tolist()})
        for bi, budget in enumerate(config['shot_budgets']):
            counts = uniform_setting_plan(provider.plan.settings, budget)
            for replica in range(config['replicas']):
                seed = int(np.random.SeedSequence([config['master_seed'], index, bi, replica]).generate_state(1)[0])
                observations = reference.sample(provider.plan.settings, counts, seed=seed)
                problem = provider.reconstruct(observations, delta=config['delta'])
                inference = infer(problem, config, target)
                hist = observations_record(observations)
                row = {'case': name, 'budget': budget, 'replica': replica, 'seed': seed,
                       'counts': list(counts), 'histograms': [dict(hist[s.key].hist) for s in provider.plan.settings],
                       'matrices': matrices_record(problem), 'inference': inference,
                       'observation_identity': observations.identity,
                       'physical_shots': problem.resources['shots'], 'circuits': problem.resources['circuits'],
                       'entry_region_covers': all(np.all(np.abs(a-b) <= radius+1e-10)
                         for a, b, radius in zip((problem.overlap, problem.hamiltonian, problem.second_moment),
                            true_blocks, (problem.overlap_radii, problem.hamiltonian_radii, problem.second_moment_radii)))}
                if inference['status'] == 'solved':
                    row['assessment'] = {
                        'energy_direction': assess_direction(dense_h, basis, coefficient_vector(inference['coefficients']), target),
                        'folded_direction': assess_direction(dense_h, basis,
                             coefficient_vector(inference['folded']['roots'][0]['coefficients']), target)}
                rows.append(row)
        family = truncated_time_family(h, **config['time_family'])
        validation = validate_truncated_time_family(family, psi, max_state_amplitudes=16, method='lanczos')
        exact_provider = ExactProjectedProvider(exact_reference, h, family.generators, storage='packed')
        projected = exact_provider.projected_problem()
        solved = projected.solve(tau_s=config['tau_s'], max_condition=config['max_condition'])
        time_cases.append({'case': name, 'validation': validation,
                           'overlap': complex_record(projected.overlap),
                           'hamiltonian': complex_record(projected.hamiltonian),
                           'projected_energies': list(solved.energies),
                           'exact_reference_identity': exact_reference.metadata.identity})
        # One end-to-end sampled S/H/K reconstruction over the Taylor family.
        # Its potentially unresolved result is retained rather than promoted.
        time_provider = make_provider(reference, h, family.generators, config)
        budget = config['shot_budgets'][-1]
        counts = uniform_setting_plan(time_provider.plan.settings, budget)
        seed = int(np.random.SeedSequence([config['master_seed'], index, 99]).generate_state(1)[0])
        obs = reference.sample(time_provider.plan.settings, counts, seed=seed)
        measured = time_provider.reconstruct(obs, delta=config['delta'])
        by_key = observations_record(obs)
        integrations.append({'case': name, 'budget': budget, 'seed': seed,
             'counts': list(counts), 'histograms': [dict(by_key[s.key].hist) for s in time_provider.plan.settings],
             'settings': settings_record(time_provider.plan.settings, device), 'admission': time_provider.admission,
             'matrices': matrices_record(measured), 'inference': infer(measured, config, target),
             'physical_shots': measured.resources['shots'], 'circuits': measured.resources['circuits']})
    summaries = []
    for name in config['cases']:
        selected = [r for r in rows if r['case'] == name]
        statuses = Counter(r['inference']['convergence']['status'] if r['inference']['status'] == 'solved'
                           else 'solve_failed' for r in selected)
        coverage = sum(r['entry_region_covers'] for r in selected)/len(selected)
        summaries.append({'case': name, 'outcomes': len(selected), 'entry_region_coverage': coverage,
                          'convergence_statuses': dict(statuses),
                          'coverage_acceptance_passed': coverage >= .90})
    return {'schema': 'clifford_qc.phase15_16_completion.v1', 'design': config,
            'declaration_commit': declaration_commit, 'execution_commit': execution_commit,
            'source_sha256': {path: digest(path) for path in sources()},
            'historical_bindings': {'benchmarks/reference_results/'+p: digest('benchmarks/reference_results/'+p) for p in HISTORICAL},
            'execution_environment': config['execution_environment'],
            'observed_environment': {'python': platform.python_version(), 'numpy': np.__version__},
            'evidence': {'sampled_moments': 'finite_sample_conditional_on_iid_shots',
                         'coverage': 'seeded_small_system_simulation',
                         'truncated_time': 'analytic_truncation_plus_numerical_validation',
                         'advantage_claim': False},
            'cases': cases, 'rows': rows, 'time_families': time_cases,
            'sampled_time_integration': integrations, 'summaries': summaries}


def compare(expected, actual, path='record'):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(expected) != set(actual):
            raise ValueError('key mismatch at '+path)
        for key in actual:
            if key not in ('source_sha256', 'execution_commit', 'observed_environment'):
                compare(expected[key], actual[key], path+'/'+key)
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(expected) != len(actual):
            raise ValueError('length mismatch at '+path)
        for j, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, path+'/'+str(j))
    elif isinstance(actual, float):
        if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not np.isclose(expected, actual, atol=2e-9, rtol=1e-8):
            raise ValueError('numeric mismatch at '+path)
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError('value mismatch at '+path)


def preflight(config):
    for name in config['cases']:
        h, psi, gens, _ = build_case(name)
        reference = PreparedSampledReference(h.n, 'preflight/'+name, JointShotSimulator(psi, h.n).draw)
        provider = make_provider(reference, h, gens, config)
        print(name, json.dumps(provider.admission, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/RESULT)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    config = read_json(ROOT/DESIGN)
    if args.preflight_only:
        preflight(config)
        return
    source_paths = sources()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *source_paths], cwd=ROOT, check=True)
    for path in source_paths:
        git('cat-file', '-e', 'HEAD:'+path)
    declaration = git('log', '-1', '--format=%H', '--', DESIGN)
    result = run(config, execution_commit=git('rev-parse', 'HEAD'), declaration_commit=declaration)
    if args.check:
        record = read_json(args.output)
        compare(record, result)
        print(f"OK: {len(record['rows'])} sampled outcomes and three time families replayed")
    else:
        write_json(args.output, result)
        print(f"Published {len(result['rows'])} sampled outcomes and three time families")


if __name__ == '__main__':
    main()
