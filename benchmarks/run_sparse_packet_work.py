"""Declared sparse-setup comparison. Default invocation is result-free."""
from __future__ import annotations

import argparse
import gzip
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.metadata_io import read_json, sha256
from benchmarks.archive_provenance import commit_binding, history_kind
from benchmarks.run_packet_work_reuse import trajectory, SOURCES as OLD_SOURCES
from benchmarks.run_h2o_residual_comparison import singlet_frame, peak_rss_bytes
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.spin_packets import spin_squared_operator, spatial_pattern
from clifford_qc.subspace.sparse_packets import SparseSingletFrame, run_sparse_packets, sparse_payload
from clifford_qc.reproducibility import stamp_record
from clifford_qc.evidence import declare_evidence

CONFIG = 'benchmarks/configs/sparse_packet_work.json'
DECLARATION = 'benchmarks/SPARSE_PACKET_WORK_DECLARATION.md'
RECORD = 'benchmarks/reference_results/sparse_packet_work.json'
RECEIPT = 'benchmarks/provenance/sparse_packet_work.json'
BUNDLE = 'benchmarks/provenance/sparse_packet_work.bundle'
SOURCES = tuple(sorted(set(OLD_SOURCES) | {
    'clifford_qc/subspace/sparse_packets.py', 'benchmarks/run_sparse_packet_work.py',
    'benchmarks/check_sparse_packet_work.py', 'benchmarks/summarize_sparse_packet_work.py',
    'benchmarks/check_packet_work_large.py', 'clifford_qc/models/lattice.py',
    'clifford_qc/capabilities.py'}))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=15).strip()


def load_design():
    return read_json(ROOT / CONFIG)


def preflight(c):
    history_kind(ROOT)
    provenance = stamp_record({})['provenance']
    source = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or provenance['git_dirty'] or provenance['git_sha'] != source:
        raise ValueError('execute only on clean committed sources')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('set all numerical thread counts to one')
    declaration = git('log', '-1', '--format=%H', '--', DECLARATION)
    if declaration == source:
        raise ValueError('declaration must precede implementation')
    git('merge-base', '--is-ancestor', declaration, source)
    for path in (CONFIG, DECLARATION):
        if commit_binding(ROOT, declaration, path)['sha256'] != sha256(ROOT / path):
            raise ValueError('frozen design differs')
    for path in c['implementation_paths_absent_at_declaration']:
        if subprocess.run(['git', 'cat-file', '-e', f'{declaration}:{path}'], cwd=ROOT,
                          capture_output=True, timeout=15).returncode == 0:
            raise ValueError('implementation predates declaration')
    for case in c['inputs']:
        if case['kind'] == 'fcidump' and sha256(ROOT / case['path']) != case['sha256']:
            raise ValueError('molecular input differs')
    return source, declaration, provenance


def policy_config(c, case):
    return {**c, **case, 'input': {'active_spatial_orbitals': case['active_spatial_orbitals']},
            'preconditioner_shift_hartree': c['preconditioner_shift'],
            'residual_tolerance_hartree': c['residual_tolerance']}


def probes(dimension):
    index = np.arange(dimension)+1
    vectors = np.column_stack((np.sin(index), np.cos(index*np.sqrt(2)))).astype(complex)
    return vectors/np.linalg.norm(vectors, axis=0)


def worker(case, arm, c):
    preflight(c)
    start = time.perf_counter()
    if case['kind'] == 'fcidump':
        model = fcidump_model(ROOT / case['path'])
    else:
        m = case['model']
        model = hubbard(tuple(m['shape']), t=m['t'], U=m['U'], mu=m['mu'], periodic=m['periodic'])
    backend = SectorStatevectorBackend(case['n_qubits'], case['n_electrons'], 0)
    if backend.dimension != case['sector_dimension']:
        raise ValueError('declared sector dimension differs')
    operator = backend.operator(model.hamiltonian)
    hf = sum(1 << (model.n - 1 - j) for j in range(case['n_electrons']))
    if arm == 'dense':
        h = csr_matrix(operator.restrict(np.arange(backend.dimension)))
        h.eliminate_zeros(); h.sort_indices()
        spin = backend.operator(spin_squared_operator(model.n)).restrict(np.arange(backend.dimension))
        frame, _ = singlet_frame(backend.basis, model.n, spin, c['spin_tolerance'])
        setup = {'frame_dimension': frame.shape[1], 'frame_payload_bytes': frame.nbytes,
                 'spin_payload_bytes': spin.nbytes, 'largest_spin_block': None,
                 'dense_sector_h_temporary_bytes': 16*backend.dimension**2}
    elif arm in ('sparse', 'early'):
        h = operator.to_csr()
        frame = SparseSingletFrame(backend, tolerance=c['spin_tolerance'],
                                  max_block_dimension=c['max_spin_block_dimension'])
        setup = {'frame_dimension': frame.shape[1], 'frame_payload_bytes': sparse_payload(frame.matrix),
                 'spin_payload_bytes': sparse_payload(frame.spin),
                 'largest_spin_block': frame.maximum_block_dimension,
                 'dense_sector_h_temporary_bytes': 0}
    else:
        raise ValueError('unknown arm')
    acted = h @ probes(backend.dimension)
    setup.update(h_probe_real=acted.real.tolist(), h_probe_imag=acted.imag.tolist(),
                 wall_seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
                 sector_dimension=backend.dimension, h_payload_bytes=sparse_payload(h),
                 h_nnz=int(h.nnz), compiled_operator_bytes=operator.memory_estimate()['compiled_bytes'])
    pc = policy_config(c, case)
    if arm == 'dense':
        result = trajectory(h.dot, h.diagonal().real, backend.basis, model.n, hf, frame, pc, 'original', spin)
        initial = result['trajectory'][0]['selection']['initial_csf_indices']
        outside = ((1 << model.n)-1) ^ sum(1 << (model.n-1-2*p-s)
            for p in case['active_spatial_orbitals'] for s in (0, 1))
        groups = {spatial_pattern(int(backend.basis[np.argmax(abs(column))]), outside, model.n)
                  for j, column in enumerate(frame.T) if j not in initial}
        for row in result['trajectory']:
            row['materialized_candidates'] = row['step']*len(groups)
        result['materialized_candidates'] = len(groups)*(len(result['decisions'])
            + int(result['stop_reason'] == 'no_admissible_direction'))
    else:
        result = run_sparse_packets(h.dot, h.diagonal().real, frame, hf,
                                    case['active_spatial_orbitals'], pc,
                                    policy='early' if arm == 'early' else 'exhaustive')
    result.update(arm=arm, setup=setup, final_peak_rss_bytes=peak_rss_bytes())
    return result


def expected_order(c):
    rng = np.random.default_rng(c['order_seed'])
    return [(case['name'], repeat, str(arm)) for case in c['inputs']
            for repeat in range(case['repeats']) for arm in rng.permutation(c['arms'])]


def save_journal(path, state):
    """Checkpoint append-only completed entries once, then replace a small manifest.

    Completed entry objects must not be mutated. The private in-memory cache
    is excluded from the manifest; resume verifies every cell before reusing it.
    """
    if path is None:
        return
    directory = path.with_name(path.name+'.cells')
    directory.mkdir(exist_ok=True)
    references = []
    record = state['record']
    identity = json.dumps([str(directory.resolve()), record['source_commit'], record['source_sha256']], sort_keys=True)
    cached_identity, cached = state.get('_cell_checkpoint_cache', (None, []))
    if cached_identity != identity:
        cached = []
    updated = []
    for index, entry in enumerate(record['workers']):
        if index < len(cached) and cached[index][0] is entry:
            reference = cached[index][1]
            references.append(reference)
            updated.append((entry, reference))
            continue
        payload = {'source_commit': record['source_commit'], 'source_sha256': record['source_sha256'],
                   'entry': entry}
        compressed = gzip.compress(json.dumps(payload, allow_nan=False, separators=(',', ':')).encode(), mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        name = f'{entry["order_index"]:05d}-{digest}.json.gz'
        target = directory/name
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(compressed); stream.flush(); os.fsync(stream.fileno())
        elif sha256(target) != digest:
            raise ValueError('immutable cell checkpoint digest differs')
        reference = {'checkpoint': name, 'sha256': digest}
        references.append(reference)
        updated.append((entry, reference))
    snapshot = {k: v for k, v in state.items() if k != '_cell_checkpoint_cache'}
    snapshot['record'] = {**record, 'workers': references}
    encoded = json.dumps(snapshot, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as stream:
        stream.write(encoded)
        stream.flush(); os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    state['_cell_checkpoint_cache'] = identity, updated


def open_journal(path, base, *, resume=False):
    try:
        namespace = os.readlink('/proc/self/ns/pid')
    except OSError:  # Non-Linux callers must explicitly assert stopped jobs on resume.
        namespace = None
    state = {'schema': 'clifford_qc.sparse_packet_journal.v2', 'record': base,
             'in_flight': None, 'supervisor_pid': os.getpid(), 'pid_namespace': namespace, 'state': 'running'}
    if path is None:
        if resume: raise ValueError('resume requires a journal')
        return state
    if path.resolve().is_relative_to(ROOT):
        raise ValueError('execution journal must be outside the Git worktree')
    if not resume:
        with path.open('x') as stream:
            stream.write(json.dumps(state, allow_nan=False))
        return state
    old = read_json(path)
    if old.get('schema') != state['schema']:
        raise ValueError('journal schema differs')
    entries = []
    references = old['record']['workers']
    for reference in old['record']['workers']:
        name = reference['checkpoint']
        if Path(name).name != name: raise ValueError('checkpoint must be a cell basename')
        target = path.with_name(path.name+'.cells')/name
        if sha256(target) != reference['sha256']: raise ValueError('cell checkpoint digest differs')
        cell = json.loads(gzip.decompress(target.read_bytes()))
        if cell['source_commit'] != base['source_commit'] or cell['source_sha256'] != base['source_sha256']:
            raise ValueError('journal execution identity differs')
        entries.append(cell['entry'])
    old['record']['workers'] = entries
    identity = json.dumps([str(path.with_name(path.name+'.cells').resolve()),
                           base['source_commit'], base['source_sha256']], sort_keys=True)
    old['_cell_checkpoint_cache'] = identity, list(zip(entries, references))
    for key in ('source_commit', 'declaration_commit', 'config', 'source_sha256'):
        if old['record'][key] != base[key]: raise ValueError('journal execution identity differs')
    for key in ('dependencies', 'platform', 'environment_sha256'):
        if old['record']['provenance'][key] != base['provenance'][key]:
            raise ValueError('journal environment differs')
    pid = old['supervisor_pid']
    if type(pid) is not int or pid <= 0: raise ValueError('invalid prior supervisor identity')
    same_namespace = namespace is not None and old.get('pid_namespace') == namespace
    if old['state'] != 'complete' and same_namespace:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('prior supervisor may still be active; refusing concurrent resume')
    elif old['state'] != 'complete':
        old['record']['recovery_assertion'] = 'Explicit resume asserts the prior supervisor and worker stopped across process namespaces.'
    order = expected_order(base['config'])
    if [(e['input'], e['repeat'], e['arm']) for e in entries] != order[:len(entries)]:
        raise ValueError('journal order is not a declared prefix')
    if old['in_flight'] is not None:
        entry = old['in_flight']
        if len(entries) >= len(order) or (entry['input'], entry['repeat'], entry['arm']) != order[len(entries)]:
            raise ValueError('journal in-flight identity differs')
        if same_namespace:
            pid_path = Path(entry['worker_pid_path'])
            if not pid_path.exists():
                raise ValueError('interrupted worker identity unavailable; verify it has stopped before recovery')
            child_pid = int(pid_path.read_text())
            if child_pid <= 0: raise ValueError('invalid interrupted worker identity')
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('interrupted worker may still be active; refusing concurrent resume')
        # Never rerun an interrupted cell or invent its lost process duration.
        entry.update(status='failed', process_wall_seconds=None,
                     diagnostic='Supervisor interrupted before checkpoint; process duration unavailable; cell not rerun.')
        entries.append(entry)
        old['in_flight'] = None
    old['supervisor_pid'] = os.getpid()
    old['pid_namespace'] = namespace
    save_journal(path, old)
    return old


def execute(c, *, journal=None, resume=False):
    source, declaration, provenance = preflight(c)
    base = {'schema': 'clifford_qc.sparse_packet_work.v1', 'config': c,
        'source_commit': source, 'declaration_commit': declaration,
        'source_sha256': {p: sha256(ROOT/p) for p in SOURCES},
        'config_sha256': sha256(ROOT/CONFIG), 'declaration_sha256': sha256(ROOT/DECLARATION),
        'provenance': provenance, 'references': {}, 'workers': [],
        'hardware_execution': False, 'quantum_advantage_claim': False}
    state = open_journal(journal, base, resume=resume)
    base, workers = state['record'], state['record']['workers']
    cases = {case['name']: case for case in c['inputs']}
    with tempfile.TemporaryDirectory(prefix='sparse-packet-workers-') as directory:
        for name, repeat, arm in expected_order(c)[len(workers):]:
            output = Path(directory) / f'{name}-{repeat}-{arm}.json'
            entry = {'input': name, 'repeat': repeat, 'arm': arm, 'order_index': len(workers),
                     'worker_pid_path': str(output.with_suffix('.pid'))}
            state['in_flight'] = entry
            save_journal(journal, state)
            start = time.perf_counter()
            try:
                done = subprocess.run([sys.executable, __file__, '--worker', arm, '--input', name,
                                       '--output', str(output)], cwd=ROOT, text=True, capture_output=True,
                                      timeout=c['worker_timeout_seconds'])
                if done.returncode:
                    entry.update(status='failed', diagnostic=(done.stderr or done.stdout)[-4000:])
                else:
                    entry.update(status='ok', result=read_json(output))
            except subprocess.TimeoutExpired:
                entry.update(status='timeout', diagnostic='declared worker timeout')
            entry['process_wall_seconds'] = time.perf_counter()-start
            workers.append(entry)
            state['in_flight'] = None
            save_journal(journal, state)
            print(f'{name}/{repeat}/{arm}: {entry["status"]} ({entry["process_wall_seconds"]:.2f}s)', flush=True)
    # Independent energies are computed only after every timed cell finishes.
    from benchmarks.check_sparse_packet_work import independent_input, reference_energy
    references = {}
    for name, case in cases.items():
        h, words = independent_input(case)
        references[name] = reference_energy(h)
        for entry in workers:
            if entry['input'] == name and entry['status'] == 'ok':
                for row in entry['result']['trajectory']:
                    row['error'] = row['energy'] - references[name]
                    row['target_reached'] = abs(row['error']) <= case['accuracy_target']
    base['references'] = references
    result = declare_evidence(base, {
        'trajectories': {'tier': 'exact_small_sector_simulation', 'scope': 'Declared classical sector packet spans and independent occupation replay.'},
        'resources': {'tier': 'classical_process_measurement', 'scope': 'Isolated single-threaded time/RSS; numerical array payloads exclude Python containers and temporary arrays.'}})
    state.update(record=result, state='complete')
    save_journal(journal, state)
    return result


def unpack(record):
    if record.get('schema') != 'clifford_qc.sparse_packet_work_archive.v1':
        return record
    payload = gzip.decompress(base64.b64decode(record['payload'], validate=True))
    if hashlib.sha256(payload).hexdigest() != record['payload_sha256']:
        raise ValueError('observation archive digest differs')
    native = json.loads(payload)
    if (native.get('schema') != 'clifford_qc.sparse_packet_work.v1'
            or any(native.get(k) != record.get(k) for k in ('provenance', 'evidence_schema', 'evidence'))):
        raise ValueError('archive metadata differs from observations')
    return native


def pack(record):
    """Lossless publication envelope; serialize once before a single file write."""
    payload = json.dumps(record, allow_nan=False, separators=(',', ':')).encode()
    return {'schema': 'clifford_qc.sparse_packet_work_archive.v1', 'encoding': 'gzip+base64',
            'payload_sha256': hashlib.sha256(payload).hexdigest(),
            **{k: record[k] for k in ('provenance', 'evidence_schema', 'evidence')},
            'payload': base64.b64encode(gzip.compress(payload, mtime=0)).decode()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', choices=('dense', 'sparse', 'early'))
    p.add_argument('--input')
    p.add_argument('--journal', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/RECORD)
    args = p.parse_args(argv)
    c = load_design()
    if not args.execute and not args.worker:
        print(json.dumps(c, indent=2)); return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.worker:
        args.output.with_suffix('.pid').write_text(str(os.getpid()))
    result = (worker(next(case for case in c['inputs'] if case['name'] == args.input), args.worker, c)
              if args.worker else execute(c, journal=args.journal, resume=args.resume))
    if not args.worker:
        result = pack(result)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
