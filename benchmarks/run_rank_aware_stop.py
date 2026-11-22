"""Preregistered paired rank-aware stopping comparison on existing stress banks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_stretched_molecular_stop as stress
from benchmarks.experiment_artifacts import read_json, write_json
from clifford_qc.measurement.session import grouped_cache_fingerprint
from clifford_qc.subspace.acquisition import AcquisitionError, RankStopPolicy, acquire_ritz

DECLARATION = 'docs/restart/rank_aware_stop_declaration.json'
RESULT = 'docs/restart/rank_aware_stop_results.json.gz'
SCHEMA = 'clifford_qc.rank_aware_stop_results.v1'
SYSTEMS = {}


def declaration():
    d = read_json(ROOT/DECLARATION)
    if d['schema'] != 'clifford_qc.rank_aware_stop.v1':
        raise ValueError('invalid declaration')
    for path, digest in d['historical_bindings'].items():
        if stress.fixed.digest((ROOT/path).read_bytes()) != digest:
            raise ValueError(f'historical binding changed: {path}')
    previous = stress.declaration()
    if d['environment'] != previous['environment']:
        raise ValueError('environment pins changed')
    return d, previous


def manifest(commit, d, previous, *, live=False):
    bindings = stress.manifest(commit, previous, live=live)
    for path in sorted(set(d['historical_bindings']) | {DECLARATION, 'benchmarks/run_rank_aware_stop.py',
                         'benchmarks/check_rank_aware_stop.py', 'tests/test_rank_stopping.py'}):
        content = stress.fixed.git('show', f'{commit}:{path}')
        if live and (ROOT/path).read_bytes() != content:
            raise ValueError(f'live source differs: {path}')
        bindings[path] = stress.fixed.digest(content)
    return bindings


class SolveAudit:
    """Differentiate measured and bootstrap solves, including failed draws.

    Both use the same native controller and solver. The acquisition owns one
    growing cache; a bootstrap draw owns a fresh resampled cache.
    """
    def __init__(self, shared):
        self.shared = shared
        self.measured_cache = None
        self.trace = []

    def __getattr__(self, name):
        return getattr(self.shared, name)

    def solve(self, cache, **kwargs):
        if self.measured_cache is None:
            self.measured_cache = cache
        entry = {'kind': 'look' if cache is self.measured_cache else 'bootstrap',
                 'shots': int(cache.total_shots), 'failure': None}
        try:
            point = self.shared.solve(cache, **kwargs)
        except stress.fixed.FAILURES as error:
            entry['failure'] = f'{type(error).__name__}: {error}'
            self.trace.append(entry)
            raise
        entry.update(self.shared.solve_trace[-1])
        self.trace.append(entry)
        return point


def task(args):
    d, previous, ci, ri, ai = args
    arm, case = d['arms'][ai], previous['cases'][ci]
    shared, cv, backend, meta = SYSTEMS[ci]
    shared.solve_trace = []
    audit = SolveAudit(shared)
    measurement_seed, bootstrap_seed = (stress.adaptive.seed(d, ci, ri, stream) for stream in (0, 1))
    backend.reseed(measurement_seed)
    sampler = stress.adaptive.AuditedBackend(audit, backend)
    shots = d['cumulative_average_shots_per_group']
    budgets = [(b-a)*meta['groups'] for a, b in zip(shots, shots[1:])]
    row = {'case': ci, 'replica': ri, 'arm': arm['key'], 'measurement_seed': measurement_seed,
           'bootstrap_seed': bootstrap_seed, 'failure': None}
    try:
        point = acquire_ritz(audit, sampler, pilot_shots_per_group=shots[0], round_budgets=budgets,
                            allocation='ritz_targeted', solver_kwargs=case['solver'], control_variates=cv,
                            target_std_error=None if arm['fixed'] else d['target_std_error_Ha'], delta=d['delta'],
                            rank_stop=RankStopPolicy(**d['rank_policy']) if arm['rank'] else None,
                            max_bootstrap_half_width=d['bootstrap_half_width_Ha'] if arm['width'] else None,
                            bootstrap_replicates=d['bootstrap_replicas'], bootstrap_seed=bootstrap_seed)
    except stress.fixed.FAILURES as error:
        row.update(failure=f'{type(error).__name__}: {error}', stopped='failed',
                   looks=list(error.looks) if isinstance(error, AcquisitionError) else [],
                   estimate=None, std_error=None, delta_interval=None, naive_interval=None,
                   bootstrap_interval=None, final_fingerprint=None, schedule=None)
    else:
        row.update(stopped=point.stopped, looks=list(point.looks), estimate=point.energy, std_error=point.std_error,
                   delta_interval=asdict(point.interval), naive_interval=asdict(point.naive_interval),
                   bootstrap_interval=asdict(point.bootstrap_interval) if point.bootstrap_interval is not None else None,
                   final_fingerprint=grouped_cache_fingerprint(point.cache), schedule=point.schedule)
        if point.shots != sum(batch['shots'] for batch in sampler.batches):
            raise RuntimeError('physical shot accounting mismatch')
    row.update(batches=sampler.batches, shots=sum(batch['shots'] for batch in sampler.batches),
               pilot_fingerprint=sampler.pilot_fingerprint, solve_trace=audit.trace)
    return row


def execute(d, previous, args):
    fixed = stress.fixed
    commit = fixed.git('log', '-1', '--format=%H', '--', DECLARATION).decode().strip()
    if fixed.git('rev-parse', 'HEAD').decode().strip() != commit or fixed.environment() != d['environment']:
        raise ValueError('execute only at the pinned, result-free declaration commit')
    if (ROOT/RESULT).exists():
        raise ValueError('result already exists')
    bindings = manifest(commit, d, previous, live=True)
    meta = {'schema': SCHEMA, 'declaration': d, 'declaration_commit': commit,
            'declaration_sha256': fixed.digest((ROOT/DECLARATION).read_bytes()),
            'source_sha256': bindings, 'environment': fixed.environment()}
    folder = args.checkpoint_dir.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume)
    with (folder/'active.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = folder/'meta.json'
        if args.resume:
            if read_json(stamp) != meta:
                raise ValueError('checkpoint metadata mismatch')
        else:
            write_json(stamp, meta)
        arm_ids = {a['key']: i for i, a in enumerate(d['arms'])}
        expected = [(ci, ri, ai) for ci in range(len(previous['cases'])) for ri in range(d['outer_replicas'])
                    for ai in range(len(d['arms']))]
        completed = {}
        for path in folder.glob('row-*.json'):
            row = read_json(path)
            key = row['case'], row['replica'], arm_ids[row['arm']]
            if key not in expected or key in completed:
                raise ValueError('unexpected checkpoint row')
            completed[key] = row
        systems = []
        for ci, case in enumerate(previous['cases']):
            SYSTEMS[ci] = stress.build_system(case, previous)
            system = SYSTEMS[ci][3]
            if system['groups'] != case['expected_groups'] or system['words'] != case['expected_words']:
                raise ValueError('bank changed')
            systems.append(system)
            print(case['name'], system, flush=True)
            tasks = [(d, previous, *key) for key in expected if key[0] == ci and key not in completed]
            with mp.get_context('fork').Pool(args.jobs) as pool:
                for entry in pool.imap_unordered(task, tasks, chunksize=1):
                    key = entry['case'], entry['replica'], arm_ids[entry['arm']]
                    write_json(folder/('row-%d-%d-%d.json' % key), entry)
                    completed[key] = entry
                    if len(completed) % 4 == 0:
                        print(f'{len(completed)}/{len(expected)} completed', flush=True)
            del SYSTEMS[ci]
        if manifest(commit, d, previous, live=True) != bindings:
            raise ValueError('source changed during evaluation')
        write_json(ROOT/RESULT, {**meta, 'systems': systems, 'rows': [completed[key] for key in expected]})
        print(f'wrote {RESULT}: {len(completed)} acquisitions', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--jobs', default=1, type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        parser.error('jobs must be in [1,8]')
    execute(*declaration(), args)


if __name__ == '__main__':
    main()
