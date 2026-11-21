"""Preregistered adaptive Ritz stopping on the previously validated molecular banks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_molecular_interval_coverage as fixed
from benchmarks.experiment_artifacts import read_json, write_json
from clifford_qc.measurement.session import grouped_cache_fingerprint
from clifford_qc.subspace.acquisition import AcquisitionError, acquire_ritz

DECLARATION = 'docs/restart/molecular_adaptive_stop_declaration.json'
RESULT = 'docs/restart/molecular_adaptive_stop_results.json.gz'
SCHEMA = 'clifford_qc.molecular_adaptive_stop_results.v1'
SYSTEMS = {}


def declaration():
    d = read_json(ROOT/DECLARATION)
    if d['schema'] != 'clifford_qc.molecular_adaptive_stop.v1':
        raise ValueError('unexpected declaration')
    for field in ('prior_declaration', 'prior_result'):
        if fixed.digest((ROOT/d[field]).read_bytes()) != d[field+'_sha256']:
            raise ValueError(f'frozen {field} changed')
    previous, cases = fixed.declaration()
    if d['environment'] != previous['environment']:
        raise ValueError('fixed and adaptive studies must use the same pinned environment')
    return d, previous, cases


def manifest(commit, d, previous, cases, *, live=False):
    bindings = fixed.manifest(commit, previous, cases, live=live)
    for path in (DECLARATION, d['prior_result'], 'benchmarks/run_molecular_adaptive_stop.py',
                 'benchmarks/check_molecular_adaptive_stop.py', 'tests/test_molecular_adaptive_stop.py'):
        saved = fixed.git('show', f'{commit}:{path}')
        if live and (ROOT/path).read_bytes() != saved:
            raise ValueError(f'live source differs: {path}')
        bindings[path] = fixed.digest(saved)
    return bindings


def seed(d, ci, ri, stream):
    return int(np.random.SeedSequence([d['seed_root'], ci, ri, stream]).generate_state(1, np.uint64)[0])


class AuditedBackend:
    """Record physical batches, including the cost of an unsuccessful solve."""
    def __init__(self, shared, backend):
        self.shared = shared
        self.backend = backend
        self.batches = []
        self.pilot_fingerprint = None

    def sample_grouped_from_state(self, *args, **kwargs):
        batch = self.backend.sample_grouped_from_state(*args, **kwargs)
        self.batches.append({'shots': sum(int(g.shots) for g in batch.groups), 'groups': len(batch.groups)})
        if len(self.batches) == 1:
            pilot = self.shared.new_cache()
            pilot.add_batch(batch)
            self.pilot_fingerprint = grouped_cache_fingerprint(pilot)
        return batch


def run_arm(d, previous, ci, ri, arm):
    shared, family, backend, _ = SYSTEMS[ci]
    measurement_seed, bootstrap_seed = (seed(d, ci, ri, s) for s in (0, 1))
    backend.reseed(measurement_seed)
    audited = AuditedBackend(shared, backend)
    cumulative = d['cumulative_average_shots_per_group']
    groups = len(shared.groups)
    budgets = [(b-a)*groups for a, b in zip(cumulative, cumulative[1:])]
    row = {'case': ci, 'replica': ri, 'arm': arm['key'], 'measurement_seed': measurement_seed,
           'bootstrap_seed': bootstrap_seed, 'failure': None}
    try:
        found = acquire_ritz(shared, audited, pilot_shots_per_group=cumulative[0], round_budgets=budgets,
                             allocation=arm['allocation'], target_std_error=d['target_std_error_Ha'],
                             solver_kwargs=previous['solver'], delta=d['delta'],
                             control_variates=family if arm['control_variates'] else None,
                             bootstrap_replicates=d['bootstrap_replicas'], bootstrap_seed=bootstrap_seed)
    except fixed.FAILURES as exc:
        row.update(failure=f'{type(exc).__name__}: {exc}', stopped='failed',
                   looks=list(exc.looks) if isinstance(exc, AcquisitionError) else [],
                   estimate=None, std_error=None, delta_interval=None, naive_interval=None,
                   bootstrap_interval=None, final_fingerprint=None, schedule=None)
    else:
        row.update(stopped=found.stopped, looks=list(found.looks), estimate=found.energy,
                   std_error=found.std_error, delta_interval=asdict(found.interval),
                   naive_interval=asdict(found.naive_interval),
                   bootstrap_interval=asdict(found.bootstrap_interval) if found.bootstrap_interval is not None else None,
                   final_fingerprint=grouped_cache_fingerprint(found.cache), schedule=found.schedule)
        if found.shots != sum(b['shots'] for b in audited.batches):
            raise RuntimeError('physical shot accounting mismatch')
    row.update(batches=audited.batches, shots=sum(b['shots'] for b in audited.batches),
               pilot_fingerprint=audited.pilot_fingerprint)
    return row


def task(args):
    d, previous, ci, ri, ai = args
    return run_arm(d, previous, ci, ri, d['arms'][ai])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        parser.error('jobs must be in [1,8]')
    d, previous, cases = declaration()
    commit = fixed.git('log', '-1', '--format=%H', '--', DECLARATION).decode().strip()
    if not commit or fixed.git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('execute only at the result-free declaration commit')
    if fixed.environment() != d['environment'] or (ROOT/RESULT).exists():
        raise ValueError('environment mismatch or result already exists')
    bindings = manifest(commit, d, previous, cases, live=True)
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
                raise ValueError('checkpoint metadata differs')
        else:
            write_json(stamp, meta)
        for ci, case in enumerate(cases):
            SYSTEMS[ci] = fixed.build_system(case, previous)
            print(case['name'], SYSTEMS[ci][3], flush=True)
        arm_ids = {a['key']: i for i, a in enumerate(d['arms'])}
        expected = [(ci, ri, ai) for ci in range(len(cases)) for ri in range(d['outer_replicas'])
                    for ai in range(len(d['arms']))]
        completed = {}
        for path in folder.glob('row-*.json'):
            entry = read_json(path)
            key = entry['case'], entry['replica'], arm_ids[entry['arm']]
            if key not in expected or key in completed:
                raise ValueError('invalid checkpoint row')
            completed[key] = entry
        tasks = [(d, previous, *key) for key in expected if key not in completed]
        with mp.get_context('fork').Pool(args.jobs) as pool:
            for entry in pool.imap_unordered(task, tasks, chunksize=1):
                key = entry['case'], entry['replica'], arm_ids[entry['arm']]
                write_json(folder/('row-%d-%d-%d.json' % key), entry)
                completed[key] = entry
                if len(completed) % 8 == 0:
                    print(f'{len(completed)}/{len(expected)} completed', flush=True)
        if manifest(commit, d, previous, cases, live=True) != bindings:
            raise ValueError('source changed during execution')
        write_json(ROOT/RESULT, {**meta, 'systems': [SYSTEMS[i][3] for i in range(len(cases))],
                                 'rows': [completed[key] for key in expected]})
        print(f'wrote {RESULT}: {len(completed)} acquisition runs', flush=True)


if __name__ == '__main__':
    main()
