"""Frozen larger-Hubbard work comparison; default invocation is result-free."""
from __future__ import annotations

import argparse
import base64
import gzip
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
from benchmarks.archive_provenance import commit_binding, history_kind, safe_path
from benchmarks.metadata_io import read_json, sha256
from benchmarks.run_packet_work_reuse import ARMS, SOURCES as SMALL_SOURCES, trajectory
from benchmarks.run_h2o_residual_comparison import singlet_frame, peak_rss_bytes
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.evidence import declare_evidence
from clifford_qc.models.lattice import hubbard
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.spin_packets import spin_squared_operator

CONFIG = 'benchmarks/configs/packet_work_large.json'
DECLARATION = 'benchmarks/PACKET_WORK_LARGE_DECLARATION.md'
RECORD = 'benchmarks/reference_results/packet_work_large.json'
RECEIPT = 'benchmarks/provenance/packet_work_large.json'
BUNDLE = 'benchmarks/provenance/packet_work_large.bundle'
SOURCES = tuple(sorted(set(SMALL_SOURCES) | {
    'benchmarks/run_packet_work_large.py', 'benchmarks/check_packet_work_large.py',
    'benchmarks/summarize_packet_work_large.py', 'clifford_qc/models/lattice.py',
    'benchmarks/check_packet_work_reuse.py', 'benchmarks/check_h2o_residual_comparison.py',
}))


def unpack_observation(record):
    """Decode a lossless publication envelope; native producer records pass through."""
    if record.get('schema') != 'clifford_qc.packet_work_large_archive.v1':
        return record
    if record.get('encoding') != 'gzip+base64':
        raise ValueError('unsupported observation archive encoding')
    payload = gzip.decompress(base64.b64decode(record['payload'], validate=True))
    if hashlib.sha256(payload).hexdigest() != record['payload_sha256']:
        raise ValueError('observation archive payload digest differs')
    native = json.loads(payload)
    if (native.get('schema') != 'clifford_qc.packet_work_large.v1'
            or any(native.get(k) != record.get(k) for k in ('provenance', 'evidence_schema', 'evidence'))):
        raise ValueError('observation archive metadata differs')
    return native


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,timeout=15).strip()


def load_design():
    c = read_json(ROOT/CONFIG)
    if (c['schema'] != 'clifford_qc.packet_work_large_design.v1'
            or tuple(c['arms']) != ARMS or c['model'] != {'shape':[2,4],'t':1.,'U':4.,'mu':0.,'periodic':False}
            or c['sector_dimension'] != 4900 or c['singlet_dimension'] != 1764
            or c['maximum_columns'] != 64 or c['repeats'] != 3
            or c['shortlist_fraction'] != .8 or c['audit_interval'] != 5
            or c['energy_unit'] != 't'):
        raise ValueError('frozen larger-sector design differs')
    return c


def preflight(config):
    history_kind(ROOT)
    provenance = stamp_record({})["provenance"]
    source = git("rev-parse", "HEAD")
    if git("status", "--porcelain") or provenance["git_dirty"] or provenance["git_sha"] != source:
        raise ValueError("execution requires clean committed sources, including untracked files")
    if any(os.environ.get(k) != "1" for k in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("declare one numerical-library thread before execution")
    declaration = git("log", "-1", "--format=%H", "--", DECLARATION)
    if declaration == source:
        raise ValueError("declaration must precede implementation")
    subprocess.run(["git", "merge-base", "--is-ancestor", declaration, source],
                   cwd=ROOT, check=True, capture_output=True, timeout=15)
    for path in (CONFIG, DECLARATION):
        if commit_binding(ROOT, declaration, path)["sha256"] != sha256(ROOT / path):
            raise ValueError("declared design bytes changed")
    for path in config["implementation_paths_absent_at_declaration"]:
        if subprocess.run(["git", "cat-file", "-e", f"{declaration}:{path}"],
                          cwd=ROOT, capture_output=True, timeout=15).returncode == 0:
            raise ValueError("implementation exists at declaration")
    return source, declaration, provenance


def trajectory_config(c):
    # Compatibility names belong to the unchanged H2O trajectory API. Numeric
    # values here use hopping units t throughout, never Hartree conversion.
    keys = ('maximum_columns','maximum_growth_steps','direction_absolute_tolerance',
            'direction_relative_tolerance','orthogonality_tolerance','spin_tolerance',
            'score_tie_tolerance','serialization_tolerance','shortlist_fraction','audit_interval')
    return {**{k:c[k] for k in keys},
        'input':{'active_spatial_orbitals':c['active_spatial_orbitals']},
        'preconditioner_shift_hartree':c['preconditioner_shift'],
        'residual_tolerance_hartree':c['residual_tolerance']}


def probes(dimension):
    index = np.arange(dimension, dtype=float)
    result = np.column_stack((np.sin(index + 1), np.cos((index + 1)*np.sqrt(2)))).astype(complex)
    return result / np.linalg.norm(result,axis=0)


def probe_digest(h):
    return hashlib.sha256(np.ascontiguousarray(h @ probes(h.shape[0])).tobytes()).hexdigest()


def csr_bytes(h):
    return h.data.nbytes + h.indices.nbytes + h.indptr.nbytes


def worker(arm,c):
    preflight(c)
    start = time.perf_counter()
    m = c['model']
    model = hubbard(tuple(m['shape']),t=m['t'],U=m['U'],mu=m['mu'],periodic=m['periodic'])
    backend = SectorStatevectorBackend(model.n,c['n_electrons'],c['sz'])
    if backend.dimension != c['sector_dimension']:
        raise ValueError('sector dimension differs')
    operator = backend.operator(model.hamiltonian)
    h = csr_matrix(operator.restrict(np.arange(backend.dimension)))
    h.eliminate_zeros(); h.sort_indices()
    s2 = backend.operator(spin_squared_operator(model.n)).restrict(np.arange(backend.dimension))
    frame,labels = singlet_frame(backend.basis,model.n,s2,c['spin_tolerance'])
    if frame.shape[1] != c['singlet_dimension']:
        raise ValueError('singlet dimension differs')
    diagonal = h.diagonal()
    setup = {'wall_seconds':time.perf_counter()-start,'peak_rss_bytes':peak_rss_bytes(),
        'sector_dimension':backend.dimension,'singlet_dimension':frame.shape[1],
        'spin_matrix_columns':backend.dimension,'csf_labels':[list(label) for label in labels],
        'array_payload_bytes':s2.nbytes+frame.nbytes+diagonal.nbytes,
        'csr_payload_bytes':csr_bytes(h),'csr_nnz':int(h.nnz),
        'dense_h_temporary_bytes':16*backend.dimension**2,
        'hamiltonian_operator_storage':operator.memory_estimate(),
        'native_probe_sha256':probe_digest(h)}
    setup['wall_seconds'] = time.perf_counter()-start
    hf = sum(1 << (model.n-1-j) for j in range(c['n_electrons']))
    result = trajectory(h.dot,diagonal.real,backend.basis,model.n,hf,frame,trajectory_config(c),arm,s2)
    result['setup'] = setup
    return result


def execute(c):
    source,declaration,provenance = preflight(c)
    identity = {path:sha256(ROOT/path) for path in SOURCES}
    for path,digest in identity.items():
        if commit_binding(ROOT,source,path)['sha256'] != digest:
            raise ValueError('uncommitted source: '+path)
    workers = []
    with tempfile.TemporaryDirectory(prefix='packet-large-') as tmp:
        for repeat in range(c['repeats']):
            for arm in c['arms'][repeat:] + c['arms'][:repeat]:
                output = Path(tmp)/f'{repeat}-{arm}.json'
                entry = {'repeat':repeat,'arm':arm,'order_index':len(workers)}
                start = time.perf_counter()
                try:
                    done = subprocess.run([sys.executable,__file__,'--worker',arm,'--output',str(output)],
                        cwd=ROOT,text=True,capture_output=True,timeout=c['worker_timeout_seconds'])
                    if done.returncode:
                        entry.update(status='failed',diagnostic=(done.stderr or done.stdout)[-4000:])
                    else:
                        entry.update(status='ok',result=read_json(output))
                except subprocess.TimeoutExpired:
                    entry.update(status='timeout',diagnostic='declared 300-second worker timeout')
                entry['process_wall_seconds'] = time.perf_counter()-start
                workers.append(entry)
                print(f"{repeat}/{arm}: {entry['status']} ({entry['process_wall_seconds']:.2f}s)",flush=True)
    # Independent occupation reference is first evaluated after all workers.
    from benchmarks.check_packet_work_large import occupation_hubbard, ground_reference
    start = time.perf_counter()
    h,words = occupation_hubbard(c)
    reference = ground_reference(h,c,words)
    reference['wall_seconds'] = time.perf_counter()-start
    for w in workers:
        if w['status'] == 'ok':
            for row in w['result']['trajectory']:
                row['energy_error'] = row['energy']-reference['energy']
                row['target_reached'] = abs(row['energy_error']) <= c['accuracy_target']
    record = {'schema':'clifford_qc.packet_work_large.v1','config':c,
        'config_sha256':sha256(ROOT/CONFIG),'declaration_commit':declaration,
        'declaration_sha256':sha256(ROOT/DECLARATION),'source_commit':source,
        'source_sha256':identity,'workers':workers,'reference':reference,
        'hardware_execution':False,'quantum_advantage_claim':False}
    record = declare_evidence(record,{
        'energies_selection_residuals':{'tier':'exact_small_sector_simulation',
            'scope':'One 4900-determinant Hubbard sector; explicit dense singlet setup and compiled CSR H; fixed retained-column budget, not asymptotic scaling.'},
        'costs':{'tier':'classical_process_measurement',
            'scope':'Three isolated processes per arm, one numerical thread. Setup and transient full-H construction are disclosed; cache payload is not process RSS.'}})
    if preflight(c)[0] != source:
        raise ValueError('source changed during execution')
    return stamp_record(record,provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',choices=ARMS,help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path,default=ROOT/RECORD)
    args = parser.parse_args(argv)
    c = load_design()
    if not args.execute and not args.worker:
        print('VALID larger-system design; no outcomes computed')
        return 0
    if args.output.exists():
        raise FileExistsError('refusing to overwrite '+str(args.output))
    result = worker(args.worker,c) if args.worker else execute(c)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print('Wrote '+str(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
