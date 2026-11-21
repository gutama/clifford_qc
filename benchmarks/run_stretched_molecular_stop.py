"""Development and frozen acquisition runs on larger stretched molecular banks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_molecular_adaptive_stop as adaptive
from benchmarks import run_molecular_interval_coverage as fixed
from benchmarks.experiment_artifacts import read_json, write_json
from clifford_qc.backends.finite_shot import FiniteShotBackend, _GroupPlan
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.measurement.grouping import qwc_basis_cover, shared_basis
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.models.fcidump import model_from_fcidump, read_fcidump
from clifford_qc.multivector import MV
from clifford_qc.states import ket_density
from clifford_qc.subspace import MatrixElementBank
from clifford_qc.subspace.linalg import normalize_overlap
from clifford_qc.subspace.symmetry_shifts import certify_bank_family, sector_shift_family

DECLARATION = 'docs/restart/stretched_molecular_stop_declaration.json'
RESULT = 'docs/restart/stretched_molecular_stop_results.json.gz'
SCHEMA = 'clifford_qc.stretched_molecular_stop_results.v1'


def parity_bits(values):
    """Parity on packed integer arrays, including NumPy versions before 2.0."""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(values) & 1
    result = values.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        result ^= result >> shift
    return result & 1


def reference(case):
    """Two closed-shell determinants, differing by a singlet pair transfer."""
    n = case['n']
    a, b = case['reference_bits']
    alpha = math.cos(case['theta'])
    beta = np.exp(1j*case['phi'])*math.sin(case['theta'])
    ket_a, ket_b = ket_density(n, a), ket_density(n, b)
    flip = MV.from_terms(n, {''.join('X' if x != y else 'I' for x, y in zip(a, b)): 1})
    cross = flip * ket_b
    rho = (abs(alpha)**2 * ket_a + abs(beta)**2 * ket_b
           + complex(alpha * beta.conjugate()) * cross
           + complex(beta * alpha.conjugate()) * cross.dagger())
    return rho, ((a, complex(alpha)), (b, complex(beta)))


class TwoDetQWCBackend(FiniteShotBackend):
    """Compile exact sparse two-determinant product-basis probabilities.

    The native grouped sampler, histogram format and cache are unchanged.
    Only deterministic plan construction differs: avoid a dense 2**n square
    density matrix, and trace unmeasured computational bits by summing over
    their mutually orthogonal labels. Keep joint outcomes, never independent
    Pauli draws. The toy tests compare every marginal with the native MV path.
    """
    def __init__(self, determinants, seed=0):
        super().__init__(seed)
        self.determinants = tuple(determinants)

    def _plan(self, rho, group, table):
        signature = tuple(sorted(w.code for w in group))
        if signature in table:
            return table[signature]
        basis = shared_basis(group)
        keep = tuple(sorted(basis))
        position = {q: i for i, q in enumerate(keep)}
        xy = [q for q in keep if basis[q] != 'Z']
        hidden = [q for q in range(rho.n) if q not in position]
        grid = np.arange(1 << len(xy), dtype=np.int64)
        varied = np.zeros_like(grid)
        for i, q in enumerate(xy):
            varied |= ((grid >> i) & 1) << (len(keep)-1-position[q])
        amplitudes = {}
        for bits, weight in self.determinants:
            key = tuple(bits[q] for q in hidden)
            state = amplitudes.setdefault(key, np.zeros(1 << len(keep), complex))
            fixed_bits = sum(1 << (len(keep)-1-position[q]) for q in keep
                             if basis[q] == 'Z' and bits[q] == '1')
            sign_mask = sum(1 << i for i, q in enumerate(xy) if bits[q] == '1')
            phase = (-1j)**sum(bits[q] == '1' for q in xy if basis[q] == 'Y')
            parity = 1-2*parity_bits(grid & sign_mask).astype(np.int64)
            state[varied | fixed_bits] += weight*phase*parity/(math.sqrt(2)**len(xy))
        prob = sum(np.abs(state)**2 for state in amplitudes.values())
        if not np.isclose(prob.sum(), 1., atol=1e-12, rtol=0):
            raise ValueError('group probability lost normalization')
        outcomes = np.flatnonzero(prob > 0.)
        probs = prob[outcomes]/prob[outcomes].sum()
        bits = tuple(format(int(i), f'0{len(keep)}b') for i in outcomes)
        parity = {}
        for word in group:
            mask = sum(1 << (len(keep)-1-position[q]) for q in word.support())
            parity[word.code] = parity_bits(outcomes & mask) == 0
        plan = _GroupPlan(keep, tuple(sorted(basis.items())), bits, probs, parity)
        table[signature] = plan
        return plan


class HistogramSharedMeasurement(SharedMeasurement):
    """Exact integer parity prefill of the native cache's assigned-word sums.

    The controller, solver, allocation, covariance and bootstrap algorithms
    remain the public ones. Before each matrix assembly (including every
    bootstrap solve), compute the same integer plus counts in vectorized
    blocks. Unassigned/readable controls retain the native fallback.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parity_masks = {}
        for group in self.groups:
            basis = tuple(sorted(shared_basis(group).items()))
            support = [q for q, _ in basis]
            codes = [w.code for w in group]
            masks = np.array([sum(1 << (len(support)-1-support.index(q)) for q in w.support())
                              for w in group], dtype=np.int64)
            self._parity_masks[basis] = codes, masks
        if len(self._parity_masks) != len(self.groups):
            raise ValueError('duplicate QWC setting basis')
        self.solve_trace = []

    def matrices(self, cache):
        if cache.pooling != 'assigned':
            raise ValueError('parity prefill requires assigned-word pooling')
        for key, group in cache._groups.items():
            codes, masks = self._parity_masks[key]
            if all((key, code) in cache._word_sums for code in codes):
                continue
            hist = group['hist']
            outcomes = np.fromiter((int(bits, 2) for bits in hist), dtype=np.int64)
            counts = np.fromiter(hist.values(), dtype=np.int64)
            plus = (parity_bits(masks[:, None] & outcomes) == 0) @ counts
            for code, value in zip(codes, plus, strict=True):
                cache._word_sums[key, code] = int(value), group['N']
        return super().matrices(cache)

    def solve(self, cache, **kwargs):
        point = super().solve(cache, **kwargs)
        self.solve_trace.append({'rank': int(point.effective_rank), 'spectrum': list(point.overlap_eigenvalues),
                                 'cutoff': float(point.resources['overlap_threshold']),
                                 'raw_energy': float(point.energies[0])})
        return point


def active_integrals(case):
    """Closed-shell frozen-core contraction of the recorded parent integrals."""
    parent = read_fcidump(ROOT/case['fcidump'])
    core = case['frozen_doubly_occupied']
    active = case['active_orbitals']
    if (set(core) & set(active) or len(set(core)) != len(core) or len(set(active)) != len(active)
            or any(not 0 <= p < parent.n_orbitals for p in core+active)):
        raise ValueError('invalid frozen-core partition')
    h, g = parent.one_body, parent.two_body
    enuc = parent.core_energy + 2*sum(h[i, i] for i in core)
    enuc += sum(2*g[i, i, j, j]-g[i, j, j, i] for i in core for j in core)
    effective = h.copy()
    for p in active:
        for q in active:
            effective[p, q] += sum(2*g[p, q, i, i]-g[p, i, i, q] for i in core)
    return replace(parent, n_orbitals=len(active), n_electrons=parent.n_electrons-2*len(core),
                   one_body=effective[np.ix_(active, active)], two_body=g[np.ix_(active, active, active, active)],
                   core_energy=float(enuc), orbsym=tuple(parent.orbsym[i] for i in active))


def build_system(case, d, *, prepare=True):
    model = model_from_fcidump(active_integrals(case))
    n = case['n']
    if model.n != n or model.metadata['n_electrons'] != case['n_electrons']:
        raise ValueError('FCIDUMP register/sector mismatch')
    rho, determinants = reference(case)
    ident = MV.scalar(n, 1)
    generators = [('I', ident)]
    source = case['pair_source']
    for i, target in enumerate(case['pair_targets']):
        t = cdag_op(n, 2*target)*cdag_op(n, 2*target+1)*c_op(n, 2*source+1)*c_op(n, 2*source)
        g = t-t.dagger()
        if i == 0:
            g = ident+case['weak_mix']*g
        generators.append((f'G{source}{target}', g))
    bank = MatrixElementBank(rho, model.hamiltonian.to_mv(), generators)
    s, h = bank.matrices()
    values = np.linalg.eigvalsh(normalize_overlap(s).matrix)
    target = float(eigh(h, s, eigvals_only=True)[0])
    groups = qwc_basis_cover([w for w in bank.words() if w.code])
    shared = HistogramSharedMeasurement(bank, groups=groups, pooling='assigned')
    family = sector_shift_family(n//2, case['n_electrons'], one_body=True, sz=0.)
    # Diagonal one-body and number/spin controls only. This bounded family
    # keeps the refitted-bootstrap work tractable on the larger registers.
    labels = {'N-Ne', '(N-Ne)N', 'Sz-m', '(Sz-m)N'} | {f'(N-Ne)E{i}{i}' for i in range(n//2)}
    family = [x for x in family if x.label in labels]
    cert = certify_bank_family(bank, family)
    backend = TwoDetQWCBackend(determinants)
    if prepare:
        backend.prepare_grouped_state(rho, shared.groups)
    exact = bank.solve(**case['solver'])
    meta = {'target': target, 'rank': len(generators), 'exact_policy_rank': int(exact.effective_rank),
            'exact_policy_target': float(exact.ground_energy), 'overlap_spectrum': values.tolist(),
            'groups': len(shared.groups), 'words': len(shared.words), 'shift_labels': [x.label for x in family],
            'certification_residuals': list(cert)}
    return shared, family, backend, meta


def declaration(path=None):
    d = read_json(ROOT/DECLARATION if path is None else Path(path))
    if d['schema'] != 'clifford_qc.stretched_molecular_stop.v1':
        raise ValueError('invalid declaration')
    for file, digest in d['historical_bindings'].items():
        if fixed.digest((ROOT/file).read_bytes()) != digest:
            raise ValueError(f'historical binding changed: {file}')
    for case in d['cases']:
        if fixed.digest((ROOT/case['fcidump']).read_bytes()) != case['fcidump_sha256']:
            raise ValueError('FCIDUMP changed')
    return d


def manifest(commit, d, *, live=False):
    paths = fixed.git('ls-tree', '-r', '--name-only', commit, 'clifford_qc').decode().splitlines()
    paths += list(fixed.SUPPORT) + ['benchmarks/run_molecular_adaptive_stop.py',
             'benchmarks/check_molecular_adaptive_stop.py', 'benchmarks/run_stretched_molecular_stop.py',
             'benchmarks/check_stretched_molecular_stop.py', 'tests/test_stretched_molecular_stop.py', DECLARATION]
    paths += list(d['historical_bindings']) + [case['fcidump'] for case in d['cases']]
    bindings = {}
    for path in sorted(set(paths)):
        content = fixed.git('show', f'{commit}:{path}')
        if live and (ROOT/path).read_bytes() != content:
            raise ValueError(f'live source differs: {path}')
        bindings[path] = fixed.digest(content)
    return bindings


def task(args):
    d, ci, ri, ai = args
    shared = adaptive.SYSTEMS[ci][0]
    shared.solve_trace = []
    row = adaptive.run_arm(d, d['cases'][ci], ci, ri, d['arms'][ai])
    row['solve_trace'] = shared.solve_trace
    return row


def execute(d, args):
    commit = fixed.git('log', '-1', '--format=%H', '--', DECLARATION).decode().strip()
    if fixed.git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('run only at the result-free declaration commit')
    if fixed.environment() != d['environment'] or (ROOT/RESULT).exists():
        raise ValueError('environment mismatch or result exists')
    bindings = manifest(commit, d, live=True)
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
        expected = [(ci, ri, ai) for ci in range(len(d['cases'])) for ri in range(d['outer_replicas'])
                    for ai in range(len(d['arms']))]
        completed = {}
        for path in folder.glob('row-*.json'):
            row = read_json(path)
            key = row['case'], row['replica'], arm_ids[row['arm']]
            if key not in expected or key in completed:
                raise ValueError('unexpected checkpoint row')
            completed[key] = row
        systems = []
        # Isolate per-molecule pools: never keep two large bank/cache plans in
        # every worker. No forked worker consumes another molecule's memory.
        for ci, case in enumerate(d['cases']):
            adaptive.SYSTEMS[ci] = build_system(case, d)
            systems.append(adaptive.SYSTEMS[ci][3])
            if (systems[-1]['groups'] != case['expected_groups'] or systems[-1]['words'] != case['expected_words']
                    or systems[-1]['exact_policy_rank'] != systems[-1]['rank']):
                raise ValueError('declared deterministic bank inventory mismatch')
            print(case['name'], systems[-1], flush=True)
            tasks = [(d, *key) for key in expected if key[0] == ci and key not in completed]
            with mp.get_context('fork').Pool(args.jobs) as pool:
                for entry in pool.imap_unordered(task, tasks, chunksize=1):
                    key = entry['case'], entry['replica'], arm_ids[entry['arm']]
                    write_json(folder/('row-%d-%d-%d.json' % key), entry)
                    completed[key] = entry
                    if len(completed) % 4 == 0:
                        print(f'{len(completed)}/{len(expected)} completed', flush=True)
            del adaptive.SYSTEMS[ci]
        if manifest(commit, d, live=True) != bindings:
            raise ValueError('source changed during run')
        write_json(ROOT/RESULT, {**meta, 'systems': systems, 'rows': [completed[key] for key in expected]})
        print(f'wrote {RESULT}: {len(completed)} acquisitions', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--development', type=Path, help='separate non-evaluation development declaration')
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args(argv)
    if args.development:
        if args.execute or args.checkpoint_dir:
            parser.error('development never writes evaluation rows')
        d = declaration(args.development)
        if d['seed_root'] != 317:
            parser.error('development seed namespace is 317 only')
        for ci, case in enumerate(d['cases']):
            start = time.monotonic()
            adaptive.SYSTEMS[ci] = build_system(case, d)
            print(case['name'], adaptive.SYSTEMS[ci][3], 'setup_seconds', time.monotonic()-start, flush=True)
            for ri in range(d['outer_replicas']):
                for ai in range(len(d['arms'])):
                    start = time.monotonic()
                    row = task((d, ci, ri, ai))
                    print(row['arm'], ri, row['stopped'], [(l['retained_rank'], l['std_error'], l['energy']) for l in row['looks']],
                          'failure', row['failure'], 'seconds', time.monotonic()-start, flush=True)
            del adaptive.SYSTEMS[ci]
        return
    if not args.execute or args.checkpoint_dir is None or not 1 <= args.jobs <= 8:
        parser.error('execute requires checkpoint directory and 1-8 jobs')
    execute(declaration(), args)


if __name__ == '__main__':
    main()
