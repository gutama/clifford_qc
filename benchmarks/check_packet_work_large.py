"""Independent occupation-sign Hubbard reference and larger-sector replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from benchmarks.archive_provenance import commit_binding, history_kind
from benchmarks.metadata_io import read_json, sha256
from benchmarks.phase17_ceo_oracle import _fermionic_action
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from benchmarks.check_h2o_residual_comparison import independent_frame
from benchmarks.check_packet_work_reuse import verify_worker
from benchmarks.run_packet_work_large import (
    ARMS,BUNDLE,CONFIG,DECLARATION,RECEIPT,RECORD,SOURCES,
    trajectory_config,csr_bytes,probe_digest,unpack_observation,
)
from clifford_qc.evidence import validate_evidence


def occupation_hubbard(c):
    rows,cols = c['model']['shape']
    sites,n = rows*cols,2*rows*cols
    pairs = c['n_electrons']//2
    words = sorted(sum(1 << (n-1-2*p-spin) for spin,occupied in enumerate((a,b)) for p in occupied)
        for a in combinations(range(sites),pairs) for b in combinations(range(sites),pairs))
    lookup = {word:j for j,word in enumerate(words)}
    bonds = [(r*cols+p,r*cols+p+1) for r in range(rows) for p in range(cols-1)]
    bonds += [(r*cols+p,(r+1)*cols+p) for r in range(rows-1) for p in range(cols)]
    rr,cc,data = [],[],[]
    for j,word in enumerate(words):
        double = sum(bool(word & (1 << (n-1-2*p))) and bool(word & (1 << (n-2-2*p))) for p in range(sites))
        diagonal = c['model']['U']*double-c['model']['mu']*c['n_electrons']
        if diagonal:
            rr.append(j);cc.append(j);data.append(diagonal)
        for p,q in bonds:
            for a,b in ((p,q),(q,p)):
                for spin in (0,1):
                    action = _fermionic_action(word,[(2*a+spin,True),(2*b+spin,False)],n)
                    if action is not None:
                        target,sign = action
                        rr.append(lookup[target]);cc.append(j);data.append(-c['model']['t']*sign)
    h = csr_matrix((np.asarray(data,complex),(rr,cc)),shape=(len(words),len(words)))
    h.sum_duplicates();h.eliminate_zeros();h.sort_indices()
    return h,words


def ground_reference(h,c,words):
    # Shift above the entire Gershgorin spectrum: LM then selects the lowest
    # original eigenvalue, rather than relying on an unshifted SA ordering.
    from benchmarks.run_packet_work_large import probes
    diagonal = h.diagonal().real
    upper = float(np.max(diagonal + np.asarray(abs(h).sum(axis=1)).ravel()-abs(diagonal)))
    shift = upper+1.
    shifted = h.real-eye(h.shape[0],format='csr')*shift
    s2 = independent_spin_squared(words,c['n_qubits'])
    energies,residuals,spins = [],[],[]
    for start in probes(h.shape[0]).real.T:
        value,vector = eigsh(shifted,k=1,which='LM',v0=start,tol=1e-12,maxiter=20000)
        energy,psi = float(value[0]+shift),vector[:,0]
        energies.append(energy)
        residuals.append(float(np.linalg.norm(h @ psi-energy*psi)))
        spins.append(float(np.linalg.norm(s2 @ psi)))
    if abs(energies[0]-energies[1]) > 1e-9 or max(residuals)>1e-9 or max(spins)>1e-8:
        raise ValueError('independent reference did not meet the declared agreement/residual/spin checks')
    return {'energy':energies[0],'starting_vector_energies':energies,'residual_norms':residuals,
        'spin_action_norms':spins,'gershgorin_upper':upper,'shift':shift,
        'role':'independent_occupation_reference_after_all_workers'}


COST_KEYS = {'wall_seconds','peak_rss_bytes','solver_wall_seconds','final_peak_rss_bytes'}


def scientific_view(value):
    if isinstance(value,dict):
        return {k:scientific_view(v) for k,v in value.items() if k not in COST_KEYS}
    if isinstance(value,list):
        return [scientific_view(v) for v in value]
    return value


def validate_costs(entry):
    problems=[]
    process=entry['process_wall_seconds']
    if not np.isfinite(process) or process<=0:
        problems.append('process time')
    if entry['status']!='ok':
        return problems
    result=entry['result']; setup=result['setup']
    previous, rss = -1.,setup['peak_rss_bytes']
    if not np.isfinite(setup['wall_seconds']) or setup['wall_seconds']<0 or type(rss) is not int or rss<=0:
        problems.append('setup cost')
    for row in result['trajectory']:
        if (not np.isfinite(row['wall_seconds']) or row['wall_seconds']<previous
                or type(row['peak_rss_bytes']) is not int or row['peak_rss_bytes']<rss):
            problems.append('prefix costs')
        previous,rss=row['wall_seconds'],row['peak_rss_bytes']
    if (not np.isfinite(result['solver_wall_seconds']) or result['solver_wall_seconds']<previous
            or result['final_peak_rss_bytes']<rss or process+1e-6<setup['wall_seconds']+result['solver_wall_seconds']):
        problems.append('terminal costs')
    return problems


def restore_and_bind(record):
    published = record
    history_kind(ROOT)
    if published != read_json(ROOT / RECORD):
        raise ValueError("inspected object differs from the preserved observations")
    record = unpack_observation(published)
    receipt = read_json(ROOT / RECEIPT)
    if (receipt["schema"] != "clifford_qc.packet_work_large_publication.v1"
            or receipt["bundle_sha256"] != sha256(ROOT / BUNDLE)
            or receipt["archive_sha256"] != sha256(ROOT / RECORD)
            or receipt["observation_sha256"] != published["payload_sha256"]):
        raise ValueError("publication/observation byte binding differs")
    native_bytes = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    if (receipt.get("serialization") != "gzip_envelope_lossless"
            or hashlib.sha256(native_bytes).hexdigest() != receipt.get("producer_json_sha256")):
        raise ValueError("native producer serialization binding differs")
    subprocess.run(["git", "bundle", "verify", str(ROOT / BUNDLE)], cwd=ROOT,
                   check=True, capture_output=True, timeout=15)
    subprocess.run(["git", "bundle", "unbundle", str(ROOT / BUNDLE)], cwd=ROOT,
                   check=True, capture_output=True, timeout=15)
    if commit_binding(ROOT, receipt["observation_commit"], RECORD)["sha256"] != receipt["observation_sha256"]:
        raise ValueError("observation bytes differ from their observation commit")
    if set(record["source_sha256"]) != set(SOURCES):
        raise ValueError("source manifest differs")
    for path, digest in record["source_sha256"].items():
        if commit_binding(ROOT, record["source_commit"], path)["sha256"] != digest:
            raise ValueError(f"source binding differs: {path}")
    for path, key in ((CONFIG, "config_sha256"), (DECLARATION, "declaration_sha256")):
        if (commit_binding(ROOT, record["declaration_commit"], path)["sha256"] != record[key]
                or sha256(ROOT / path) != record[key]):
            raise ValueError("declared design binding differs")
    for earlier, later in ((record["declaration_commit"], record["source_commit"]),
                           (record["source_commit"], receipt["observation_commit"])):
        if earlier == later:
            raise ValueError("declaration/execution/observation must be distinct")
        subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later], cwd=ROOT,
                       check=True, capture_output=True, timeout=15)
    for path in record["config"]["implementation_paths_absent_at_declaration"]:
        if subprocess.run(["git", "cat-file", "-e", f"{record['declaration_commit']}:{path}"],
                          cwd=ROOT, capture_output=True, timeout=15).returncode == 0:
            raise ValueError("implementation predates declaration")
    provenance = record["provenance"]
    if provenance["git_sha"] != record["source_commit"] or provenance["git_dirty"] is not False:
        raise ValueError("execution provenance differs")


def verify(record,*,bindings=True):
    if bindings:
        restore_and_bind(record)
    record = unpack_observation(record)
    validate_evidence(record)
    c=read_json(ROOT/CONFIG)
    if (record['config']!=c or record['schema']!='clifford_qc.packet_work_large.v1'
            or record['hardware_execution'] is not False or record['quantum_advantage_claim'] is not False):
        raise ValueError('larger-system identity differs')
    h,words=occupation_hubbard(c)
    reference=ground_reference(h,c,words)
    s2=independent_spin_squared(words,c['n_qubits'])
    frame_data=independent_frame(words,c['n_qubits'],s2,c['spin_tolerance'])
    # The frame needs dense occupation blocks; repeated spin diagnostics only
    # need matvecs. Preserve every nonzero entry in a sparse verification copy.
    s2=csr_matrix(s2)
    problems=[]
    if (len(words)!=c['sector_dimension'] or frame_data[0].shape[1]!=c['singlet_dimension']
            or abs(record['reference']['energy']-reference['energy'])>1e-9):
        problems.append('reference dimensions/energy')
    for key in ('starting_vector_energies','residual_norms','spin_action_norms','gershgorin_upper','shift'):
        if not np.allclose(record['reference'][key],reference[key],atol=1e-9,rtol=0):
            problems.append('reference '+key)
    if record['reference']['role']!=reference['role'] or not np.isfinite(record['reference']['wall_seconds']) or record['reference']['wall_seconds']<0:
        problems.append('reference metadata')
    expected=[(r,a) for r in range(c['repeats']) for a in c['arms'][r:]+c['arms'][:r]]
    if [(w['repeat'],w['arm']) for w in record['workers']]!=expected:
        problems.append('repeat/arm denominator and order')
    replayed={}
    for i,w in enumerate(record['workers']):
        if w['order_index']!=i or w['status'] not in ('ok','failed','timeout'):
            problems.append('worker identity/status')
        problems.extend(validate_costs(w))
        if w['status']!='ok':
            if not w.get('diagnostic') or 'result' in w:
                problems.append('failure disclosure')
            continue
        result=w['result']; setup=result['setup']
        if result['arm']!=w['arm']:
            problems.append('worker arm')
        if (setup['csr_payload_bytes']!=csr_bytes(h) or setup['csr_nnz']!=h.nnz
                or setup['dense_h_temporary_bytes']!=16*len(words)**2
                or setup['native_probe_sha256']!=probe_digest(h)):
            problems.append('independent native-H probes/CSR setup')
        science=scientific_view(result)
        if replayed.get(w['arm'])!=science:
            problems.extend(f"{w['arm']}/{w['repeat']}: {p}" for p in verify_worker(
                result,trajectory_config(c),h,words,s2,frame_data=frame_data,
                n_qubits=c['n_qubits'],n_electrons=c['n_electrons']))
            replayed[w['arm']]=science
        for row in result['trajectory']:
            error=row['energy']-reference['energy']
            if (abs(row['energy_error']-error)>1e-9 or error < -1e-8
                    or row['target_reached']!=(abs(error)<=c['accuracy_target'])):
                problems.append('energy/target/variational readout')
    return problems


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--science-only',action='store_true')
    parser.add_argument('--record',type=Path,default=ROOT/RECORD)
    args=parser.parse_args(argv)
    try:
        problems=verify(read_json(args.record),bindings=not args.science_only)
    except (OSError,ValueError,TypeError,KeyError,IndexError,subprocess.SubprocessError) as exc:
        problems=[str(exc)]
    for p in problems:print('FAIL large packet work: '+p)
    if not problems:print('OK independent larger-sector replay'+('; chronology/bytes UNVERIFIED' if args.science_only else '; observation bytes and chronology verified'))
    return int(bool(problems))


if __name__=='__main__':
    raise SystemExit(main())
