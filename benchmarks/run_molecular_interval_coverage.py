"""Frozen paired molecular Ritz-interval experiment (no new Hamiltonians)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json, write_json
from benchmarks.run_cross_domain_ratio import build_case
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.backends.protocol import GroupSample, MeasurementBatch
from clifford_qc.matrix import to_matrix
from clifford_qc.measurement.cache import GroupedWordCache
from clifford_qc.measurement.session import SharedMeasurement, grouped_cache_fingerprint
from clifford_qc.multivector import MV
from clifford_qc.states import ket_density
from clifford_qc.subspace import MatrixElementBank
from clifford_qc.subspace.symmetry_shifts import (certify_bank_family, corrected_ritz_functional,
                                                 sector_shift_family)
from clifford_qc.subspace.uncertainty import _point_estimator, ritz_uncertainty

DECLARATION = 'docs/restart/molecular_interval_coverage_declaration.json'
RESULT = 'docs/restart/molecular_interval_coverage_results.json.gz'
SCHEMA = 'clifford_qc.molecular_interval_coverage_results.v1'
SUPPORT = ('benchmarks/run_molecular_interval_coverage.py',
           'benchmarks/check_molecular_interval_coverage.py',
           'benchmarks/run_cross_domain_ratio.py', 'benchmarks/run_sampled_ratio.py',
           'benchmarks/check_cross_domain_ratio.py', 'benchmarks/phase17_ceo_oracle.py',
           'benchmarks/check_orthogonal_active_space.py', 'benchmarks/experiment_artifacts.py',
           'benchmarks/metadata_io.py', 'pyproject.toml')
FAILURES = (ValueError, np.linalg.LinAlgError, FloatingPointError)
SYSTEMS: dict = {}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def declaration():
    value = read_json(ROOT / DECLARATION)
    if value['schema'] != 'clifford_qc.molecular_interval_coverage.v1':
        raise ValueError('unexpected declaration schema')
    source = ROOT / value['case_source']
    if digest(source.read_bytes()) != value['case_source_sha256']:
        raise ValueError('source case declaration drift')
    known = {case['name']: case for case in read_json(source)['cases']}
    cases = [known[name] for name in value['cases']]
    if len(cases) != 2 or any(c['kind'] != 'molecule' or c['n'] != 8 or c['n_electrons'] != 4
                              for c in cases):
        raise ValueError('unexpected molecular cases')
    for case in cases:
        if digest((ROOT / case['fcidump']).read_bytes()) != case['fcidump_sha256']:
            raise ValueError('FCIDUMP drift')
    return value, cases


def seed(root, c, b, r, stream):
    return int(np.random.SeedSequence([root, c, b, r, stream]).generate_state(1, np.uint64)[0])


def environment():
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('set all three numerical thread limits to 1 before import')
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'threads': 1}


def manifest(commit, d, cases, *, live=False):
    paths = git('ls-tree', '-r', '--name-only', commit, 'clifford_qc').decode().splitlines()
    paths = sorted(set(paths + list(SUPPORT) + [DECLARATION, d['case_source']]
                       + [case['fcidump'] for case in cases]))
    bindings = {}
    for path in paths:
        data = git('show', f'{commit}:{path}')
        if live and (ROOT / path).read_bytes() != data:
            raise ValueError(f'live source differs from declaration commit: {path}')
        bindings[path] = digest(data)
    return bindings


def two_det_density(case, state):
    a, b = case['reference_bits']
    n = case['n']
    alpha, beta = state[int(a, 2)], state[int(b, 2)]
    rho_a, rho_b = ket_density(n, a), ket_density(n, b)
    flip = MV.from_terms(n, {''.join('X' if x != y else 'I' for x, y in zip(a, b)): 1})
    cross = flip * rho_b
    rho = (abs(alpha)**2 * rho_a + abs(beta)**2 * rho_b
           + complex(alpha * beta.conjugate()) * cross
           + complex(beta * alpha.conjugate()) * cross.dagger())
    if not np.allclose(to_matrix(rho), np.outer(state, state.conj()), atol=1e-11, rtol=0):
        raise ValueError('two-determinant density mismatch')
    return rho


def build_system(case, d, *, prepare=True):
    h, generators, state = build_case(case)
    rho = two_det_density(case, state)
    shared = SharedMeasurement(MatrixElementBank(rho, h, generators), pooling=d['pooling'])
    exact = shared.bank.solve(**d['solver'])
    if exact.effective_rank != 3:
        raise ValueError('exact three-direction bank is rank-deficient')
    family = sector_shift_family(4, 4, **d['control_variates'])
    residuals = certify_bank_family(shared.bank, family)
    backend = FiniteShotBackend(seed=0)
    if prepare:
        backend.prepare_grouped_state(shared.bank.reference, shared.groups)
    meta = {'target': float(exact.ground_energy), 'rank': int(exact.effective_rank),
            'condition': float(exact.condition_number), 'words': len(shared.words),
            'groups': len(shared.groups), 'shift_labels': [t.label for t in family],
            'certification_residuals': list(residuals)}
    return shared, family, backend, meta


def paired_bootstrap(shared, cache, point, family, replicates, bootstrap_seed):
    """Identical grouped multinomial draws and solves for both arms.

    Reuses one solve for the two estimators; compared to both public bootstrap
    paths on development seeds by the test suite. No failed draw is dropped.
    The family must already have been certified on this bank.
    """
    kwargs = _point_estimator(shared, cache, point)
    rng = np.random.default_rng(bootstrap_seed)
    states = cache.group_states()
    raw, corrected = [], []
    raw_failure = corrected_failure = None
    for i in range(replicates):
        groups = []
        for group in states:
            keys = list(group['hist'])
            counts = np.array([group['hist'][key] for key in keys], dtype=float)
            drawn = rng.multinomial(group['shots'], counts/counts.sum())
            groups.append(GroupSample(support=group['support'], basis=group['basis'],
                                     hist={key: int(count) for key, count in zip(keys, drawn) if count},
                                     shots=group['shots'], word_codes=group['word_codes'],
                                     setting_key=group.get('setting_key'), readouts=group.get('readouts', {})))
        replica = GroupedWordCache(shared.n, pooling=cache.pooling)
        replica.add_batch(MeasurementBatch(n=shared.n, shots={}, plus_counts={}, circuits=len(groups),
                                           groups=tuple(groups), state_key=cache.state_key))
        try:
            result = shared.solve(replica, **kwargs)
            value = float(result.ground_energy)
            if not np.isfinite(value):
                raise ValueError('nonfinite raw energy')
        except FAILURES as exc:
            raw_failure = f'draw {i}: {type(exc).__name__}: {exc}'
            corrected_failure = corrected_failure or raw_failure
            break
        raw.append(value)
        if corrected_failure is None:
            try:
                cv, _ = corrected_ritz_functional(shared, replica, result, family)
                if not np.isfinite(cv.energy):
                    raise ValueError('nonfinite corrected energy')
                corrected.append(float(cv.energy))
            except FAILURES as exc:
                corrected_failure = f'draw {i}: {type(exc).__name__}: {exc}'
    return {'raw': (raw if raw_failure is None else None, raw_failure),
            'refitted_cv': (corrected if corrected_failure is None else None, corrected_failure)}


def row(task):
    d, ci, bi, ri = task
    shared, family, backend, _ = SYSTEMS[ci]
    mseed, bseed = [seed(d['seed_root'], ci, bi, ri, stream) for stream in (0, 1)]
    backend.reseed(mseed)
    cache = shared.measure(backend, d['shots_per_qwc_group'][bi])
    out = {'case': ci, 'budget': bi, 'replica': ri, 'measurement_seed': mseed,
           'bootstrap_seed': bseed, 'shots': int(cache.total_shots),
           'histogram_sha256': grouped_cache_fingerprint(cache), 'rank': None,
           'point_failure': None, 'arms': {'raw': None, 'refitted_cv': None}}
    try:
        point = shared.solve(cache, **d['solver'])
        if not np.isfinite(point.ground_energy):
            raise ValueError('nonfinite raw point')
        out['rank'] = int(point.effective_rank)
    except FAILURES as exc:
        out['point_failure'] = f'{type(exc).__name__}: {exc}'
        return out
    raw = {'estimate': float(point.ground_energy), 'sigma': None, 'delta_scale': None, 'delta_failure': None}
    out['arms']['raw'] = raw
    try:
        raw_interval = ritz_uncertainty(shared, cache, point, delta=0.05)
        z = NormalDist().inv_cdf(0.975)
        sigma, scale = float(raw_interval.std_error), float((raw_interval.upper-raw_interval.lower)/(2*z))
        if not np.all(np.isfinite([sigma, scale])):
            raise ValueError('nonfinite raw delta interval')
        raw.update(sigma=sigma, delta_scale=scale)
    except FAILURES as exc:
        raw['delta_failure'] = f'{type(exc).__name__}: {exc}'
    try:
        cv, _ = corrected_ritz_functional(shared, cache, point, family)
        if not np.all(np.isfinite([cv.energy, cv.std_error])):
            raise ValueError('nonfinite corrected point')
        out['arms']['refitted_cv'] = {'estimate': float(cv.energy), 'sigma': float(cv.std_error),
                                     'delta_scale': float(cv.std_error), 'delta_failure': None, 'used': list(cv.used),
                                     'unreadable': list(cv.unreadable)}
    except FAILURES as exc:
        out['cv_point_failure'] = f'{type(exc).__name__}: {exc}'
    draws = paired_bootstrap(shared, cache, point, family, d['bootstrap_replicas'], bseed)
    for arm, entry in out['arms'].items():
        if entry is not None:
            entry['draws'], entry['bootstrap_failure'] = draws[arm]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        parser.error('jobs must be in [1, 8]')
    d, cases = declaration()
    commit = git('log', '-1', '--format=%H', '--', DECLARATION).decode().strip()
    if not commit or git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('execute only from the result-free declaration commit')
    if environment() != d['environment'] or (ROOT / RESULT).exists():
        raise ValueError('environment mismatch or result already exists; refusing execution')
    bindings = manifest(commit, d, cases, live=True)
    meta = {'schema': SCHEMA, 'declaration': d, 'declaration_commit': commit,
            'declaration_sha256': digest((ROOT / DECLARATION).read_bytes()),
            'source_sha256': bindings, 'environment': environment()}
    folder = args.checkpoint_dir.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume)
    with (folder/'active.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = folder/'meta.json'
        if args.resume:
            if read_json(stamp) != meta:
                raise ValueError('checkpoint declaration or environment mismatch')
        else:
            write_json(stamp, meta)
        for i, case in enumerate(cases):
            SYSTEMS[i] = build_system(case, d)
            print(case['name'], SYSTEMS[i][3], flush=True)
        expected = [(ci, bi, ri) for ci in range(len(cases)) for bi in range(len(d['shots_per_qwc_group']))
                    for ri in range(d['outer_replicas'])]
        completed = {}
        for path in folder.glob('row-*.json'):
            entry = read_json(path)
            key = entry['case'], entry['budget'], entry['replica']
            if key not in expected or key in completed:
                raise ValueError('invalid/duplicate checkpoint row')
            completed[key] = entry
        tasks = [(d, *key) for key in expected if key not in completed]
        with mp.get_context('fork').Pool(args.jobs) as pool:
            for entry in pool.imap_unordered(row, tasks, chunksize=1):
                key = entry['case'], entry['budget'], entry['replica']
                write_json(folder/('row-%d-%d-%d.json' % key), entry)
                completed[key] = entry
                if len(completed) % 8 == 0:
                    print(f'{len(completed)}/{len(expected)} completed', flush=True)
        if manifest(commit, d, cases, live=True) != bindings:
            raise ValueError('source changed during execution')
        result = {**meta, 'systems': [SYSTEMS[i][3] for i in range(len(cases))],
                  'rows': [completed[key] for key in expected]}
        write_json(ROOT / RESULT, result)
        print(f'wrote {RESULT}: {len(completed)} paired rows', flush=True)


if __name__ == '__main__':
    main()
