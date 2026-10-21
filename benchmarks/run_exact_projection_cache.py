"""Declared isolated exact-cache comparison; default invocation is result-free."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.archive_provenance import commit_binding, history_kind
from benchmarks.metadata_io import read_json, sha256
from benchmarks.run_sparse_packet_work import SOURCES as PREVIOUS_SOURCES
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.evidence import declare_evidence
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.reproducibility import execution_provenance, stamp_record
from clifford_qc.subspace import sparse_packets

CONFIG = 'benchmarks/configs/exact_projection_cache.json'
DECLARATION = 'benchmarks/EXACT_PROJECTION_CACHE_DECLARATION.md'
RECORD = 'benchmarks/reference_results/exact_projection_cache.json'
RECEIPT = 'benchmarks/provenance/exact_projection_cache.json'
BUNDLE = 'benchmarks/provenance/exact_projection_cache.bundle'
HISTORICAL = 'clifford_qc/subspace/sparse_packets.py'
SOURCES = tuple(sorted(set(PREVIOUS_SOURCES) | {
    'benchmarks/run_exact_projection_cache.py', 'benchmarks/check_exact_projection_cache.py',
    'benchmarks/summarize_exact_projection_cache.py', 'tests/test_exact_projection_cache.py'}))
SCIENTIFIC_KEYS = ('policy', 'trajectory', 'decisions', 'stop_reason', 'final_h_action_calls',
                   'final_candidates_scored', 'materialized_candidates', 'arm')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def append_event(path, event):
    with path.open('ab') as stream:
        stream.write(canonical(event)+b'\n'); stream.flush(); os.fsync(stream.fileno())


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=20).strip()


def design():
    c = read_json(ROOT/CONFIG)
    if c['schema'] != 'clifford_qc.exact_projection_cache_design.v1':
        raise ValueError('unexpected design schema')
    for case in c['inputs']:
        if case['kind'] == 'fcidump' and sha256(ROOT/case['path']) != case['sha256']:
            raise ValueError('frozen molecular input differs')
    return c


def expected_order(c):
    rng = np.random.default_rng(c['order_seed'])
    combinations = [(row['policy'], mode) for row in c['comparisons'] for mode in row['cache_modes']]
    return [dict(input=case['name'], repeat=repeat, policy=combinations[int(j)][0],
                 cache_mode=combinations[int(j)][1])
            for case in c['inputs'] for repeat in range(case['repeats'])
            for j in rng.permutation(len(combinations))]


def preflight(c):
    history_kind(ROOT)
    if git('status', '--porcelain'):
        raise ValueError('execute only on clean committed sources')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('pin all numerical thread counts to one')
    if os.environ.get('CLIFFORD_QC_ALLOW_ENVIRONMENT_MIGRATION') == '1':
        raise ValueError('this campaign does not authorize an environment migration')
    source = git('rev-parse', 'HEAD')
    declaration = git('log', '-1', '--format=%H', '--', DECLARATION)
    if source == declaration:
        raise ValueError('declaration must precede implementation')
    git('merge-base', '--is-ancestor', declaration, source)
    for path in (CONFIG, DECLARATION):
        if commit_binding(ROOT, declaration, path)['sha256'] != sha256(ROOT/path):
            raise ValueError('frozen design bytes differ')
    for path in c['implementation_paths_absent_at_declaration']:
        if subprocess.run(['git', 'cat-file', '-e', f'{declaration}:{path}'], cwd=ROOT,
                          capture_output=True).returncode == 0:
            raise ValueError('implementation existed at declaration')
    historical_dependencies(c)  # Refuse before timing, not at the post-timing controls.
    started = time.perf_counter()
    # Linux preserves a fork parent's high-water RSS across exec. Survey in a
    # separate supervisory helper so the worker-launching parent stays small.
    survey = json.loads(subprocess.check_output([sys.executable, __file__, '--supervisor-stamp'], cwd=ROOT))
    provenance = survey['provenance']
    if provenance['git_sha'] != source or provenance['git_dirty']:
        raise ValueError('source provenance differs')
    return {'config': c, 'source_commit': source, 'declaration_commit': declaration,
            'source_sha256': {p: sha256(ROOT/p) for p in SOURCES},
            'config_sha256': sha256(ROOT/CONFIG), 'declaration_sha256': sha256(ROOT/DECLARATION),
            'provenance': provenance, 'supervisor_preflight_seconds': time.perf_counter()-started,
            'supervisor_preflight_peak_rss_bytes': survey['peak_rss_bytes']}


def historical_dependencies(c):
    """The pre-change module imports today's natives, so they must equal the baseline's."""
    for native in (p for p in SOURCES if p.startswith('clifford_qc/') and p != HISTORICAL):
        if commit_binding(ROOT, c['baseline_commit'], native)['sha256'] != sha256(ROOT/native):
            raise ValueError('historical baseline dependency evolved: '+native)


def historical_module(c):
    """Execute the exact pre-change sparse module, retaining its relative imports."""
    path = HISTORICAL
    historical_dependencies(c)
    raw = subprocess.check_output(['git', 'show', c['baseline_commit']+':'+path], cwd=ROOT)
    if digest(raw) != commit_binding(ROOT, c['baseline_commit'], path)['sha256']:
        raise ValueError('historical baseline source differs')
    module = types.ModuleType('clifford_qc.subspace._exact_cache_baseline')
    module.__package__ = 'clifford_qc.subspace'
    exec(compile(raw, 'git:'+c['baseline_commit']+':'+path, 'exec'), module.__dict__)
    return module


def scientific_result(result):
    scientific = {k: result[k] for k in SCIENTIFIC_KEYS}
    scientific['trajectory'] = [{k: v for k, v in row.items() if k != 'wall_seconds'}
                                for row in result['trajectory']]
    return scientific


def decode_science(payload):
    compressed = base64.b64decode(payload['payload'], validate=True)
    if digest(compressed) != payload['gzip_sha256']:
        raise ValueError('scientific gzip digest differs')
    raw = gzip.decompress(compressed)
    if digest(raw) != payload['sha256']:
        raise ValueError('scientific payload digest differs')
    return json.loads(raw)


def worker(context, task):
    c = context['config']
    if git('rev-parse', 'HEAD') != context['source_commit'] or git('status', '--porcelain'):
        raise ValueError('worker source identity differs')
    if any(sha256(ROOT/p) != h for p, h in context['source_sha256'].items()):
        raise ValueError('worker source bytes differ')
    actual = execution_provenance()  # Metadata only: deliberately no survey/stamp call.
    for key in ('dependencies', 'environment_sha256', 'platform'):
        if actual[key] != context['provenance'][key]:
            raise ValueError('worker environment differs: '+key)
    if '.'.join(platform.python_version_tuple()[:2]) != '.'.join(context['provenance']['python'].split('.')[:2]):
        raise ValueError('worker Python minor differs')
    case = next(case for case in c['inputs'] if case['name'] == task['input'])
    module = historical_module(c) if task['cache_mode'] == 'historical' else sparse_packets
    imports_peak = peak_rss()
    started = time.perf_counter()
    if case['kind'] == 'fcidump':
        model = fcidump_model(ROOT/case['path'])
    else:
        m = case['model']
        model = hubbard(tuple(m['shape']), t=m['t'], U=m['U'], mu=m['mu'], periodic=m['periodic'])
    backend = SectorStatevectorBackend(case['n_qubits'], case['n_electrons'], 0)
    if backend.dimension != case['sector_dimension']:
        raise ValueError('sector dimension differs')
    h = backend.operator(model.hamiltonian).to_csr()
    frame = module.SparseSingletFrame(backend, tolerance=c['spin_tolerance'],
                                      max_block_dimension=c['max_spin_block_dimension'])
    setup_seconds, setup_peak = time.perf_counter()-started, peak_rss()
    hf = sum(1 << (model.n-1-j) for j in range(case['n_electrons']))
    pc = {**c, **case}
    kwargs = {} if task['cache_mode'] == 'historical' else {'cache_mode': task['cache_mode']}
    result = module.run_sparse_packets(h.dot, h.diagonal().real, frame, hf,
        case['active_spatial_orbitals'], pc, policy=task['policy'], **kwargs)
    result['arm'] = 'early' if task['policy'] == 'early' else 'sparse'
    metrics = {'setup_seconds': setup_seconds, 'solver_seconds': result['solver_wall_seconds'],
        'imports_peak_rss_bytes': imports_peak, 'setup_peak_rss_bytes': setup_peak,
        'solver_end_peak_rss_bytes': peak_rss(), 'cache': result.get('cache'),
        'prefix_wall_seconds': [row['wall_seconds'] for row in result['trajectory']],
        'setup': {'h_nnz': int(h.nnz), 'h_payload_bytes': sparse_packets.sparse_payload(h),
                  'frame_payload_bytes': frame.payload_bytes, 'frame_shape': list(frame.shape),
                  'largest_spin_block': frame.maximum_block_dimension},
        'worker_environment_sha256': actual['environment_sha256']}
    return scientific_result(result), metrics


def run_cell(directory, context, task, order_index):
    cell = directory/'cells'/str(order_index)
    cell.mkdir()
    write_new(cell/'task.json', canonical(task))
    started = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, __file__, '--worker', str(cell),
                               '--directory', str(directory)], cwd=ROOT, capture_output=True,
                              timeout=context['config']['worker_timeout_seconds'])
        if done.returncode:
            raise RuntimeError(done.stderr.decode(errors='replace')[-4000:])
        value = read_json(cell/'metrics.json')
        value.update(status='ok')
    except subprocess.TimeoutExpired:
        value = {'status': 'timeout', 'diagnostic': 'declared worker timeout'}
    except (ValueError, OSError, RuntimeError) as exc:
        value = {'status': 'failed', 'diagnostic': str(exc)}
    value.update(task, order_index=order_index, process_wall_seconds=time.perf_counter()-started)
    return value


def campaign_process(pid):
    """Whether pid is live and still runs this script; finished PIDs get reused."""
    if pid == os.getpid(): return False
    try: return Path(__file__).name.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError: return False


def recover(directory, task, index):
    """Outcome of a started cell whose finish was never journaled; it is never rerun."""
    try:
        value = read_json(directory/'cells'/str(index)/'metrics.json')
        if sha256(directory/'scientific'/(value['scientific_sha256']+'.json.gz')) != value['scientific_gzip_sha256']:
            raise ValueError('scientific file differs')
    except (OSError, ValueError, KeyError):
        return {**task, 'order_index': index, 'status': 'interrupted', 'process_wall_seconds': None,
                'diagnostic': 'Interrupted before durable finish; cell not rerun.'}
    # metrics.json is the worker's last durable write; only the supervisor's wall time is lost.
    value.update(task, order_index=index, status='ok', process_wall_seconds=None,
                 recovered='Durable worker metrics; the supervisor stopped before journaling finish.')
    return value


def execute(directory, *, resume=False):
    c = design()
    fresh = preflight(c)
    if directory.resolve().is_relative_to(ROOT):
        raise ValueError('campaign output must stay outside the Git worktree')
    journal = directory/'events.jsonl'
    events = [json.loads(line) for line in journal.read_bytes().splitlines()] if journal.exists() else []
    finished = {e['order_index']: e['worker'] for e in events if e['event'] == 'finish'}
    starts = {e['order_index'] for e in events if e['event'] == 'start'}
    if not resume:
        directory.mkdir(); (directory/'cells').mkdir(); (directory/'scientific').mkdir()
        context = fresh
        write_new(directory/'context.json', canonical(context))
    else:
        if (directory/'result.json').exists(): raise ValueError('campaign already has a complete publication')
        context = read_json(directory/'context.json')
        for key in ('config', 'source_commit', 'source_sha256'):
            if context[key] != fresh[key]: raise ValueError('resume source/design differs')
        for key in ('platform', 'dependencies', 'environment_sha256'):
            if context['provenance'][key] != fresh['provenance'][key]: raise ValueError('resume environment differs')
        prior = read_json(directory/'supervisor.json')
        namespace = os.readlink('/proc/self/ns/pid') if sys.platform == 'linux' else None
        if namespace is not None and namespace == prior.get('pid_namespace'):
            # Only the supervisor and cells without a journaled finish can still be running.
            pidfiles = [directory/'supervisor.json', *(path for path in directory.glob('cells/*/pid.json')
                                                       if int(path.parent.name) not in finished)]
            if any(campaign_process(read_json(path)['pid']) for path in pidfiles):
                raise ValueError('prior supervisor/worker may still be active')
        # Across namespaces explicit --resume asserts all prior jobs stopped.
        context['resume_assertion'] = 'Explicit resume asserts all prior jobs stopped; started cells are never rerun.'
    state = {'pid': os.getpid(), 'pid_namespace': os.readlink('/proc/self/ns/pid') if sys.platform == 'linux' else None}
    (directory/'supervisor.json').write_bytes(canonical(state))
    order = expected_order(c)
    total = len(order)+2*len(c['inputs'])

    def outcome(i, task):
        if i not in finished:
            if i in starts or (directory/'cells'/str(i)).exists():
                # A started cell is never run a second time, even if its output was lost.
                value = recover(directory, task, i)
            else:
                append_event(journal, {'event': 'start', 'order_index': i, 'task': task, 'utc_seconds': time.time()})
                value = run_cell(directory, context, task, i)
            append_event(journal, {'event': 'finish', 'order_index': i, 'worker': value, 'utc_seconds': time.time()})
            finished[i] = value
            print(f'{i+1}/{total} {task["input"]}/{task["repeat"]}/{task["policy"]}/{task["cache_mode"]}: '
                  f'{value["status"]} ({value["process_wall_seconds"] or 0:.2f}s)', flush=True)
        return finished[i]

    for i, task in enumerate(order):
        outcome(i, task)
    controls = []
    # Frozen pre-change module controls run, and are journaled, only after all timed observations.
    for case in c['inputs']:
        for policy in ('exhaustive', 'early'):
            task = dict(input=case['name'], repeat=0, policy=policy, cache_mode='historical')
            controls.append(outcome(len(order)+len(controls), task))
    payloads = {}
    for entry in list(finished.values())+controls:
        if entry['status'] != 'ok': continue
        name = entry['scientific_sha256']
        raw = (directory/'scientific'/(name+'.json.gz')).read_bytes()
        if digest(raw) != entry['scientific_gzip_sha256']: raise ValueError('immutable scientific file differs')
        payloads[name] = {'sha256': name, 'gzip_sha256': digest(raw), 'encoding': 'gzip+base64',
                          'payload': base64.b64encode(raw).decode()}
    from benchmarks.check_sparse_packet_work import independent_input, reference_energy
    references = {case['name']: reference_energy(independent_input(case)[0]) for case in c['inputs']}
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    result = {**context, 'schema': 'clifford_qc.exact_projection_cache.v1',
              'workers': [finished[i] for i in range(len(order))], 'historical_controls': controls,
              'references': references,
              'scientific_payloads': payloads, 'journal': events, 'journal_sha256': sha256(journal),
              'hardware_execution': False, 'quantum_advantage_claim': False}
    result = declare_evidence(result, {
        'trajectories': {'tier': 'exact_small_sector_simulation', 'scope': 'Fixed classical packet policies and independent occupation replay.'},
        'resources': {'tier': 'classical_process_measurement', 'scope': 'Fresh single-threaded processes; supervisor-only survey; measured times/RSS and owned cache arrays.'}})
    write_new(directory/'result.json', canonical(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--directory', type=Path); parser.add_argument('--worker', type=Path)
    parser.add_argument('--supervisor-stamp', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.supervisor_stamp:
        provenance = stamp_record({})['provenance']
        print(canonical({'provenance': provenance, 'peak_rss_bytes': peak_rss()}).decode())
    elif args.worker:
        directory, cell = args.directory, args.worker
        write_new(cell/'pid.json', canonical({'pid': os.getpid()}))
        context, task = read_json(directory/'context.json'), read_json(cell/'task.json')
        science, metrics = worker(context, task)
        raw = canonical(science); compressed = gzip.compress(raw, mtime=0); name = digest(raw)
        target = directory/'scientific'/(name+'.json.gz')
        if target.exists():
            if sha256(target) != digest(compressed): raise ValueError('scientific content address collision')
        else: write_new(target, compressed)
        metrics.update(scientific_sha256=name, scientific_gzip_sha256=digest(compressed),
                       worker_peak_rss_bytes=peak_rss())
        write_new(cell/'metrics.json', canonical(metrics))
    elif args.execute:
        if args.directory is None: parser.error('--execute requires a new external --directory')
        execute(args.directory, resume=args.resume)
    else:
        c = design(); print(f'Result-free exact cache design: {len(expected_order(c))} scheduled cells.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
