"""Independent replay, cache-storage formulas and immutable campaign bindings."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.archive_provenance import commit_binding, history_kind
from benchmarks.metadata_io import read_json, sha256
from benchmarks.run_frame_coordinate_projection import (BUNDLE, CONFIG, DECLARATION, RECEIPT, RECORD,
    SOURCES, canonical, decode_science, digest, expected_order)
from benchmarks.check_sparse_packet_work import independent_input, reference_energy, replay
from benchmarks.check_h2o_residual_comparison import independent_frame
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from benchmarks.check_exact_projection_cache import (compare_science, independent_layout,
                                                       cache_problems as sector_cache_problems)
from clifford_qc.evidence import validate_evidence


def cache_problems(stats, science, case, frame, words, layout=None):
    if stats['mode'] != 'coordinates':
        return sector_cache_problems(stats, science, case, frame, words, layout)
    stages = science['trajectory'] if science['stop_reason'] == 'no_admissible_direction' else science['trajectory'][:-1]
    coordinate_bytes = 32 * frame.shape[1] * max((row['dimension'] for row in stages), default=0)
    # Coordinate mode owns one sparse F-adjoint, no duplicate group blocks.
    fixed = ((16 + stats['index_itemsize']) * np.count_nonzero(frame)
             + stats['indptr_itemsize'] * (frame.shape[1]+1))
    expected = {'basis_adjoint_builds': 0, 'peak_basis_adjoint_bytes': 0,
                'frame_adjoint_builds': 1, 'group_block_builds': 0, 'group_columns': 0,
                'group_nnz': 0, 'group_index_itemsize': 0, 'group_indptr_itemsize': 0,
                'frame_nnz': int(np.count_nonzero(frame)),
                'coordinate_builds': len(stages), 'peak_coordinate_bytes': coordinate_bytes,
                'projected_candidates': science['materialized_candidates'],
                'fixed_owned_bytes': int(fixed), 'peak_owned_bytes': int(fixed+coordinate_bytes)}
    problems = ['cache formula '+key for key, value in expected.items() if stats.get(key) != value]
    if any(stats[k] not in (4, 8) for k in ('index_itemsize', 'indptr_itemsize')):
        problems.append('coordinate sparse index widths')
    return problems


def bindings(record):
    history_kind(ROOT)
    receipt = read_json(ROOT/RECEIPT)
    if receipt['schema'] != 'clifford_qc.frame_coordinate_projection_publication.v1':
        raise ValueError('publication schema differs')
    if receipt['record_sha256'] != sha256(ROOT/RECORD) or record != read_json(ROOT/RECORD):
        raise ValueError('observation bytes differ')
    if receipt['bundle_sha256'] != sha256(ROOT/BUNDLE): raise ValueError('source bundle differs')
    subprocess.run(['git', 'bundle', 'verify', str(ROOT/BUNDLE)], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'bundle', 'unbundle', str(ROOT/BUNDLE)], cwd=ROOT, check=True, capture_output=True)
    if commit_binding(ROOT, receipt['observation_commit'], RECORD)['sha256'] != receipt['record_sha256']:
        raise ValueError('observation commit differs')
    for before, after in ((record['declaration_commit'], record['source_commit']),
                          (record['source_commit'], receipt['observation_commit'])):
        if before == after: raise ValueError('declaration/source/observation must differ')
        subprocess.run(['git', 'merge-base', '--is-ancestor', before, after], cwd=ROOT, check=True, capture_output=True)


def verify(record, *, bound=True, numerical=True):
    if bound: bindings(record)
    problems = []
    validate_evidence(record)
    c = read_json(ROOT/CONFIG)
    if record['schema'] != 'clifford_qc.frame_coordinate_projection.v1' or record['config'] != c:
        return ['record/config identity']
    for path, key in ((CONFIG, 'config_sha256'), (DECLARATION, 'declaration_sha256')):
        if sha256(ROOT/path) != record[key] or commit_binding(ROOT, record['declaration_commit'], path)['sha256'] != record[key]:
            problems.append('design binding '+path)
    if set(record['source_sha256']) != set(SOURCES): problems.append('execution source universe')
    for path, value in record['source_sha256'].items():
        if commit_binding(ROOT, record['source_commit'], path)['sha256'] != value:
            problems.append('execution source '+path)
    for path in c['implementation_paths_absent_at_declaration']:
        if subprocess.run(['git', 'cat-file', '-e', f'{record["declaration_commit"]}:{path}'], cwd=ROOT, capture_output=True).returncode == 0:
            problems.append('implementation predates declaration')
    kernel = 'clifford_qc/subspace/sparse_packets.py'
    if commit_binding(ROOT, record['declaration_commit'], kernel) != commit_binding(ROOT, c['baseline_commit'], kernel):
        problems.append('kernel changed before declaration')
    if record['provenance']['git_sha'] != record['source_commit'] or record['provenance']['git_dirty'] is not False:
        problems.append('clean source provenance')
    journal = record['journal']
    if digest(b''.join(canonical(e)+b'\n' for e in journal)) != record['journal_sha256']:
        problems.append('journal digest')
    order = expected_order(c)
    if len(record['workers']) != len(order): return problems+['scheduled-cell count']
    control_tasks = [dict(input=case['name'], repeat=0, policy=policy, cache_mode='historical')
                     for case in c['inputs'] for policy in ('exhaustive', 'early')]
    if len(record['historical_controls']) != len(control_tasks): return problems+['historical control count']
    starts, finishes = {}, {}
    for event in journal:
        if event['event'] not in ('start', 'finish'):
            problems.append('unknown journal event'); continue
        i = event['order_index']; target = starts if event['event'] == 'start' else finishes
        if i in target or i not in range(len(order)+len(control_tasks)): problems.append('duplicate/invalid journal index')
        target[i] = event
    sciences = {key: decode_science(value) for key, value in record['scientific_payloads'].items()}
    used = {w['scientific_sha256'] for w in record['workers']+record['historical_controls'] if w['status'] == 'ok'}
    if set(sciences) != used: problems.append('scientific payload universe')
    if any(key != value['sha256'] for key, value in record['scientific_payloads'].items()):
        problems.append('scientific content address')
    for i, (worker, task) in enumerate(zip(record['workers'], order)):
        if any(worker[k] != v for k, v in task.items()) or worker['order_index'] != i:
            problems.append('worker order')
        if i not in starts or starts[i]['task'] != task or i not in finishes or finishes[i]['worker'] != worker:
            problems.append('durable start/outcome evidence')
        elif finishes[i]['utc_seconds'] < starts[i]['utc_seconds']:
            problems.append('journal time order')
        if worker['status'] not in ('ok', 'failed', 'timeout', 'interrupted'):
            problems.append('worker status')
        if worker['status'] != 'ok': continue
        # A recovered cell kept its durable worker metrics; only the supervisor's wall time was lost.
        recovered = 'recovered' in worker
        if recovered and worker['process_wall_seconds'] is not None: problems.append('recovered process timing')
        bad = [key for key in ('setup_seconds', 'solver_seconds') + (() if recovered else ('process_wall_seconds',))
               if not isinstance(worker[key], (int, float)) or not math.isfinite(worker[key]) or worker[key] < 0]
        problems.extend('finite timing '+key for key in bad)
        if not recovered and not bad and worker['setup_seconds']+worker['solver_seconds'] > worker['process_wall_seconds']+1e-6:
            problems.append('process timing')
        peaks = [worker[key] for key in ('imports_peak_rss_bytes', 'setup_peak_rss_bytes',
                                         'solver_end_peak_rss_bytes', 'worker_peak_rss_bytes')]
        if any(type(v) is not int or v <= 0 for v in peaks) or peaks != sorted(peaks):
            problems.append('RSS phase high-water order')
        payload = record['scientific_payloads'][worker['scientific_sha256']]
        if payload['gzip_sha256'] != worker['scientific_gzip_sha256']:
            problems.append('cell scientific gzip binding')
        science = sciences[worker['scientific_sha256']]
        if science['policy'] != worker['policy'] or worker['cache']['mode'] != worker['cache_mode']:
            problems.append('policy/cache identity')
        times = worker['prefix_wall_seconds']
        if (len(times) != len(science['trajectory']) or times != sorted(times)
                or any(not math.isfinite(t) or t < 0 for t in times) or times[-1] > worker['solver_seconds']):
            problems.append('prefix timing')
        if (not 0 <= worker['cache']['construction_wall_seconds'] <= worker['solver_seconds']
                or worker['worker_environment_sha256'] != record['provenance']['environment_sha256']):
            problems.append('cache timing/environment')
    journaled = [len(order)+k in starts or len(order)+k in finishes for k in range(len(control_tasks))]
    if any(journaled) and not all(journaled): problems.append('partial historical control journal')
    for k, (control, task) in enumerate(zip(record['historical_controls'], control_tasks)):
        i = len(order)+k
        if any(control[key] != value for key, value in task.items()) or control['order_index'] != i:
            problems.append('historical control order')
        if all(journaled) and (i not in starts or starts[i]['task'] != task
                               or i not in finishes or finishes[i]['worker'] != control):
            problems.append('historical control start/outcome evidence')
        if control['status'] != 'ok': continue
        # Historical controls execute the baseline's exact combined-cache path.
        if (not control['cache'] or control['cache']['mode'] != 'combined'
                or control['worker_environment_sha256'] != record['provenance']['environment_sha256']
                or record['scientific_payloads'][control['scientific_sha256']]['gzip_sha256'] != control['scientific_gzip_sha256']
                or sciences[control['scientific_sha256']]['policy'] != control['policy']):
            problems.append('historical control identity')
    for case in c['inputs']:
        name = case['name']; dimension = case['sector_dimension']
        workers = [w for w in record['workers'] if w['input'] == name and w['status'] == 'ok']
        controls = [w for w in record['historical_controls'] if w['input'] == name]
        if {w['policy'] for w in controls} != {'exhaustive', 'early'} or any(w['status'] != 'ok' for w in controls):
            problems.append('historical baseline controls '+name)
        if numerical:
            h, words = independent_input(case)
            spin = independent_spin_squared(words, case['n_qubits'])
            frame_data = independent_frame(words, case['n_qubits'], spin, c['spin_tolerance'])
            frame = frame_data[0]
            if abs(reference_energy(h)-record['references'][name]) > 1e-9: problems.append('reference energy '+name)
        unique, layouts = {}, {}
        for w in workers+controls:
            if w['status'] != 'ok': continue
            key = w['scientific_sha256']; science = sciences[key]
            if numerical and key not in unique:
                result = deepcopy(science)
                for row, elapsed in zip(result['trajectory'], w['prefix_wall_seconds']): row['wall_seconds'] = elapsed
                problems.extend(name+': '+p for p in replay(result, case, c, h, words, spin, frame_data))
                unique[key] = True
            if numerical and w['cache_mode'] != 'historical':
                initial = tuple(science['trajectory'][0]['selection']['initial_csf_indices'])
                if initial not in layouts: layouts[initial] = independent_layout(case, frame, initial, words)
                problems.extend(name+': '+p for p in cache_problems(w['cache'], science, case, frame, words, layouts[initial]))
        for policy in ('exhaustive', 'early'):
            baseline = next((w for w in workers if w['policy'] == policy and w['cache_mode'] == 'combined'), None)
            historical = next((w for w in controls if w['policy'] == policy and w['status'] == 'ok'), None)
            if baseline is None or historical is None:
                problems.append('missing policy baseline '+name+'/'+policy); continue
            base = sciences[baseline['scientific_sha256']]
            problems.extend(name+': '+p for p in compare_science(base, sciences[historical['scientific_sha256']], dimension))
            for w in workers:
                if w['policy'] == policy:
                    problems.extend(name+': '+p for p in compare_science(sciences[w['scientific_sha256']], base, dimension))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=ROOT/RECORD)
    parser.add_argument('--declaration-only', action='store_true', help='check frozen pre-implementation declaration without timing')
    parser.add_argument('--unbound', action='store_true', help='validate unpublished local execution without a publication receipt')
    args = parser.parse_args(argv)
    if args.declaration_only:
        from benchmarks.run_frame_coordinate_projection import preflight, design
        preflight(design())
        print('VALID: frozen frame-coordinate declaration precedes implementation')
        return 0
    problems = verify(read_json(args.record), bound=not args.unbound)
    if problems:
        for problem in problems: print('FAIL '+problem)
        return 1
    print('VALID: independent exact trajectories, historical controls and cache formulas; publication/source bindings '
          + ('were not checked (--unbound)' if args.unbound else 'verified'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
