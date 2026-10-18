"""Preflight MORE breadth; --execute consumes one frozen 12-cell campaign."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.check_phase17_more_declaration import validate
from benchmarks.phase17_more_design import (
    CONFIG, RECORD, EVIDENCE, EXECUTION_SOURCES, digest, load_config, source_bindings, target_summary,
)
from benchmarks.metadata_io import read_json
from benchmarks.run_bank_policy_matrix import require_host_capacity, stop_worker
from clifford_qc.evidence import declare_evidence
from clifford_qc.reproducibility import stamp_record


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def claim_path():
    common = Path(git('rev-parse', '--git-common-dir'))
    return (common if common.is_absolute() else ROOT / common) / 'phase17-more-execution.claim'


def cells(config):
    return [(case, roots, arm) for case in config['cases']
            for roots in config['roots'] for arm in config['arms']]


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, separators=(',', ':'), allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def worker_cell(case, roots, arm, config):
    """Fresh-process observation; callable on undeclared toy fixtures in tests."""
    import numpy as np
    import scipy
    import resource
    from clifford_qc.algorithms.exchange import _prepare_exchange
    from clifford_qc.algorithms.more_adapt import run_more_adapt
    from clifford_qc.models.fcidump import fcidump_model
    from benchmarks.phase17_more_acase import run_acase_mean
    started = time.perf_counter()
    model = fcidump_model(ROOT / case['path'], name=case['id'], integral_tolerance=1e-12)
    bits = case['structure']['reference_bits'][str(roots)]
    prepared = _prepare_exchange(model, config['more']['max_sector_dimension'])
    prepare_seconds = time.perf_counter() - started
    algorithm_started = time.perf_counter()
    if arm == 'more_adapt':
        run = run_more_adapt(model, bits, _prepared=prepared, **config['more'])
        exact = run['sector_exact_root_energies']
        # One sector diagonalization is already charged in preparation.
        run['resources']['sector_spectrum_eigensolves'] += 1
    elif arm == 'acase_mean':
        run = run_acase_mean(model, bits, config['acase'])
        run['resources']['reachable_family_dimension'] = case['structure']['response_family_dimensions'][str(roots)]
        exact = np.linalg.eigvalsh(prepared['h'])[:roots].tolist()
        run['resources']['sector_spectrum_eigensolves'] = 2
    else:
        raise ValueError('unknown breadth arm')
    algorithm_seconds = time.perf_counter() - algorithm_started
    start_ticks = int(Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19])
    return {'case': case['id'], 'roots': roots, 'arm': arm, 'status': 'completed',
            'run': run, 'exact_root_energies': exact,
            'target_summary': target_summary(run, exact, config['target_hartree']),
            'worker': {'pid': os.getpid(), 'parent_pid': os.getppid(), 'start_ticks': start_ticks,
                       'git_sha': git('rev-parse', 'HEAD'), 'python': sys.version.split()[0],
                       'numpy': np.__version__, 'scipy': scipy.__version__,
                       'threads': {key: os.environ.get(key) for key in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                       'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                       'prepare_seconds': prepare_seconds, 'algorithm_seconds': algorithm_seconds}}


def rss_bytes(pid):
    """Return observed resident bytes, or None when the ceiling cannot be checked."""
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                fields = line.split()
                if len(fields) == 3 and fields[2] == 'kB' and int(fields[1]) >= 0:
                    return int(fields[1]) * 1024
                return None
    except (OSError, ValueError, IndexError):
        return None
    return None


def run_cell(index, case, roots, arm, config):
    execution = config['execution']
    started, peak, failure = time.perf_counter(), 0, None
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    '--worker-cell', str(index)], cwd=ROOT,
                                   stdout=output, stderr=errors)
        while process.poll() is None:
            observed_rss = rss_bytes(process.pid)
            if observed_rss is None:
                # An exited child can disappear between poll and /proc read.
                # A live unobservable child must never run without a ceiling.
                if process.poll() is not None:
                    break
                failure = 'rss_unavailable'
                stop_worker(process)
                break
            peak = max(peak, observed_rss)
            if peak > execution['rss_ceiling_bytes']:
                failure = 'memory_limit'
            elif time.perf_counter() - started > execution['timeout_seconds']:
                failure = 'timeout'
            if failure:
                stop_worker(process)
                break
            time.sleep(execution['sample_seconds'])
        process.wait()
        output.seek(0); errors.seek(0)
        raw, error = output.read(), errors.read().decode(errors='replace')
        parent = {'pid': os.getpid(), 'worker_pid': process.pid,
                  'returncode': process.returncode, 'peak_rss_bytes': peak,
                  'wall_seconds': time.perf_counter() - started,
                  'rss_ceiling_bytes': execution['rss_ceiling_bytes'],
                  'timeout_seconds': execution['timeout_seconds'],
                  'sample_seconds': execution['sample_seconds']}
        if failure or process.returncode:
            return {'case': case['id'], 'roots': roots, 'arm': arm,
                    'status': failure or 'worker_failed', 'parent': parent,
                    'error': error[-4000:], 'partial_output': raw[-4000:].decode(errors='replace')}
        result = json.loads(raw)
        result['parent'] = parent
        if result['worker']['peak_rss_bytes'] > execution['rss_ceiling_bytes']:
            result['status'] = 'memory_limit'
        return result


def execute(config):
    from benchmarks.archive_provenance import history_kind
    if RECORD.exists():
        raise ValueError('record exists; another campaign is not authorized')
    # Finding the shared claim may itself need Git for linked worktrees. A
    # consumed record or known claim must still refuse before history checks.
    claim = claim_path()
    if claim.exists():
        raise ValueError('execution claim already consumed; it must not be reset or reused')
    history_kind(ROOT)
    if git('status', '--porcelain'):
        raise ValueError('execution requires clean committed sources')
    head = git('rev-parse', 'HEAD')
    declaration = git('log', '-1', '--format=%H', '--', str(CONFIG.relative_to(ROOT)))
    if not declaration or declaration == head:
        raise ValueError('execution source must strictly follow the declaration')
    subprocess.run(['git', 'merge-base', '--is-ancestor', declaration, head], cwd=ROOT, check=True)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError(f'{name}=1 required')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('PYTHONHASHSEED=0 required')
    capacity = require_host_capacity(config['execution'])
    # The stamp's migration escape hatch is for temporary test records, not a
    # licence to run this declared campaign under another numerical stack.
    from clifford_qc.reproducibility import sampling_stream_mismatch
    from clifford_qc.record_environment import verify
    reference_record = read_json(ROOT / config['environment_reference'])
    environment_errors = sampling_stream_mismatch(reference_record, packages=('numpy', 'scipy'))
    verify({'python': '.'.join(reference_record['provenance']['python'].split('.')[:2])}, environment_errors)
    if environment_errors:
        raise ValueError('; '.join(environment_errors))
    provenance = stamp_record({})['provenance']
    if provenance['git_sha'] != head or provenance['git_dirty']:
        raise ValueError('stamp must identify the clean actual execution source')
    state = {'source': head, 'config_sha256': digest(config), 'parent_pid': os.getpid(),
             'status': 'claimed', 'cells': [], 'active_cell': None}
    with claim.open('x') as handle:
        json.dump(state, handle, allow_nan=False)
    for index, (case, roots, arm) in enumerate(cells(config)):
        if git('rev-parse', 'HEAD') != head or git('status', '--porcelain'):
            raise ValueError('execution tree changed; claim and completed observations retained')
        state['active_cell'] = index
        atomic_json(claim, state)
        print(f'[{index + 1}/{len(cells(config))}] {case["id"]} roots={roots} {arm}', flush=True)
        state['cells'].append(run_cell(index, case, roots, arm, config))
        atomic_json(claim, state)
    record = declare_evidence({
        'schema': 'clifford_qc.phase17_more.v1', 'config_sha256': digest(config),
        'claim_boundary': config['claim_boundary'], 'cells': state['cells'],
        'execution_source_bindings': source_bindings(EXECUTION_SOURCES),
        'execution': {'serial_fresh_workers': True, 'capacity': capacity},
        'verdict': 'PENDING', 'validation_problems': []}, EVIDENCE)
    record = stamp_record(record, provenance)
    diagnostic = claim.with_suffix('.diagnostic.json')
    atomic_json(diagnostic, record)
    from benchmarks.check_phase17_more import check_record
    problems = check_record(record, config, history=False, check_verdict=False)
    complete = all(cell['status'] == 'completed' for cell in record['cells'])
    record['verdict'] = 'VALID' if complete and not problems else 'INVALID' if complete else 'INCOMPLETE'
    record['validation_problems'] = problems
    atomic_json(diagnostic, record)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=RECORD.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, separators=(',', ':'), allow_nan=False); handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, RECORD)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    state.update(status=record['verdict'], active_cell=None)
    atomic_json(claim, state)
    print(f'Phase 17 MORE breadth: {record["verdict"]}; {len(record["cells"])} recorded cells', flush=True)
    return 0 if record['verdict'] == 'VALID' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-cell', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_config()
    problems = validate(config)
    if problems:
        raise ValueError('; '.join(problems))
    if args.worker_cell is not None:
        state = read_json(claim_path())
        if (state['status'] != 'claimed' or state['active_cell'] != args.worker_cell
                or state['parent_pid'] != os.getppid() or state['source'] != git('rev-parse', 'HEAD')
                or state['config_sha256'] != digest(config) or git('status', '--porcelain')):
            raise ValueError('worker requires the active, source-bound parent campaign claim')
        case, roots, arm = cells(config)[args.worker_cell]
        print(json.dumps(worker_cell(case, roots, arm, config), separators=(',', ':'), allow_nan=False))
        return 0
    if not args.execute:
        declare_evidence({}, EVIDENCE)
        print('Phase 17 MORE: frozen design validated; no new spectra, trajectories or costs evaluated')
        return 0
    return execute(config)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
