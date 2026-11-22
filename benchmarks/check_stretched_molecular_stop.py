"""Independent occupation-space anchors and rank-aware stopping assessment."""
from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_stretched_molecular_stop as producer
from benchmarks.check_cross_domain_ratio import _pair_matrix, _pauli_restriction
from benchmarks.check_molecular_adaptive_stop import bounds, check_row, close, summarize
from benchmarks.check_molecular_interval_coverage import require
from benchmarks.experiment_artifacts import read_json
from benchmarks.phase17_ceo_oracle import _fermionic_action
from clifford_qc.models.fcidump import read_fcidump
from clifford_qc.reproducibility import compare_json_records


def occupation_hamiltonian(case):
    """Restrict the *parent* determinant Hamiltonian, without the core formula.

    Embed every active determinant into the source spin register, with the
    declared occupied and inactive orbitals held fixed. Discard matrix
    elements out of this projected sector. This independently checks the
    frozen-core energy shift and one-body contraction in the producer.
    """
    parent = read_fcidump(ROOT/case['fcidump'])
    n, ns = case['n'], 2*parent.n_orbitals
    words = [w for w in range(1 << n) if w.bit_count() == case['n_electrons']
             and sum((1 if q % 2 == 0 else -1)*((w >> (n-1-q)) & 1) for q in range(n)) == 0]
    active, core = case['active_orbitals'], case['frozen_doubly_occupied']
    embedding = []
    for w in words:
        state = sum(3 << (ns-2-2*p) for p in core)
        for q in range(n):
            state |= ((w >> (n-1-q)) & 1) << (ns-1-2*active[q//2]-q%2)
        embedding.append(state)
    lookup = {w: i for i, w in enumerate(embedding)}
    h = np.eye(len(words), dtype=complex)*parent.core_energy
    monomials = []
    for p, q in product(range(parent.n_orbitals), repeat=2):
        coefficient = parent.one_body[p, q]
        if abs(coefficient) > 1e-12:
            for s in (0, 1):
                monomials.append((coefficient, [(2*p+s, True), (2*q+s, False)]))
    for p, q, r, s in product(range(parent.n_orbitals), repeat=4):
        coefficient = .5*parent.two_body[p, q, r, s]
        if abs(coefficient) > 1e-12:
            for a, b in product((0, 1), repeat=2):
                monomials.append((coefficient, [(2*p+a, True), (2*r+b, True),
                                                (2*s+b, False), (2*q+a, False)]))
    for coefficient, sequence in monomials:
        for column, w in enumerate(embedding):
            acted = _fermionic_action(w, sequence, ns)
            if acted is not None and acted[0] in lookup:
                h[lookup[acted[0]], column] += coefficient*acted[1]
    require(np.allclose(h, h.conj().T, atol=1e-10, rtol=0), 'independent Hamiltonian not Hermitian')
    return words, h


def independent_anchor(case):
    words, h = occupation_hamiltonian(case)
    state = np.zeros(len(words), complex)
    for bits, amplitude in zip(case['reference_bits'], [math.cos(case['theta']),
                             np.exp(1j*case['phi'])*math.sin(case['theta'])], strict=True):
        state[words.index(int(bits, 2))] = amplitude
    columns = [state]
    for i, target in enumerate(case['pair_targets']):
        acted = _pair_matrix(words, case['n'], case['pair_source'], target)@state
        columns.append(state+case['weak_mix']*acted if i == 0 else acted)
    basis = np.column_stack(columns)
    for w, vector in zip(words, basis, strict=True):
        if np.any(abs(vector) > 1e-12):
            require(all(((w >> (2*p)) & 3) in (0, 3) for p in range(case['n']//2)), 'bank is not closed-shell singlet')
    s, projected = basis.conj().T@basis, basis.conj().T@h@basis
    norm = np.sqrt(s.diagonal().real)
    spectrum = np.linalg.eigvalsh(s/norm[:, None]/norm[None, :])
    return words, h, s, projected, spectrum, float(eigh(projected, s, eigvals_only=True)[0]), float(eigh(h, eigvals_only=True)[0])


def anchors(d, *, full=False):
    systems = []
    for case in d['cases']:
        words, h, s, projected, spectrum, target, ground = independent_anchor(case)
        tau = case['solver']['tau_s']
        require(spectrum[0] > tau and spectrum[0]-tau < .001, 'stress case does not straddle declared cutoff')
        system = {'independent_target': target, 'active_sector_ground': ground,
                  'sector_dimension': len(words), 'rank': len(spectrum), 'overlap_spectrum': spectrum.tolist()}
        if full:
            shared, _, _, meta = producer.build_system(case, d, prepare=False)
            native_s, native_h = shared.bank.matrices()
            native_sector = _pauli_restriction(shared.bank.hamiltonian, words)
            require(np.allclose(h, native_sector, atol=1e-9, rtol=0), 'frozen-core Hamiltonian mismatch')
            require(np.allclose(s, native_s, atol=1e-10, rtol=0) and
                    np.allclose(projected, native_h, atol=1e-9, rtol=0), 'projected matrix mismatch')
            require(close(target, meta['target']) and meta['exact_policy_rank'] == len(spectrum), 'target/rank mismatch')
            system.update(meta)
        systems.append(system)
    return systems


def check_trace(row, case, d):
    trace, looks = row['solve_trace'], row['looks']
    require(len(trace) >= len(looks), 'missing solve trace')
    for step in trace:
        values = step['spectrum']
        require(values and all(math.isfinite(v) for v in values) and values == sorted(values, reverse=True),
                'bad normalized spectrum')
        cutoff = max(case['solver']['tau_s'], values[0]/case['solver']['max_condition'])
        require(close(cutoff, step['cutoff']) and step['rank'] == sum(v > cutoff for v in values), 'rank decision mismatch')
        require(math.isfinite(step['raw_energy']), 'nonfinite raw trace energy')
    for look, step in zip(looks, trace):
        require(look['retained_rank'] == step['rank'] and close(look.get('raw_energy', look['energy']), step['raw_energy']),
                'measured look/solve trace mismatch')
        require(close(look['smallest_retained_overlap'], step['spectrum'][step['rank']-1]), 'retained eigenvalue mismatch')
    if row['failure']:
        require(len(trace) <= len(row['batches'])+d['bootstrap_replicas'], 'too many failed solves')
        return
    boot = trace[len(looks):]
    require(len(boot) <= d['bootstrap_replicas'], 'too many bootstrap solves')
    if row['bootstrap_interval'] is not None:
        require(len(boot) == d['bootstrap_replicas'], 'incomplete successful bootstrap')
        if row['arm'] == 'targeted_raw':
            lo, hi = np.quantile([x['raw_energy'] for x in boot], [.16, .84])
            require(close((hi-lo)/2, row['bootstrap_interval']['std_error']), 'raw bootstrap spread mismatch')


def check(record, *, verify_history=False, full=False):
    d = producer.declaration()
    require(record['schema'] == producer.SCHEMA and record['declaration'] == d, 'declaration mismatch')
    fixed = producer.fixed
    require(record['declaration_sha256'] == fixed.digest((ROOT/producer.DECLARATION).read_bytes()), 'declaration digest mismatch')
    require(record['environment'] == d['environment'], 'environment mismatch')
    commit = record['declaration_commit']
    require(re.fullmatch(r'[0-9a-f]{40}', commit), 'invalid declaration commit')
    require(fixed.git('show', f'{commit}:{producer.DECLARATION}') == (ROOT/producer.DECLARATION).read_bytes(), 'declaration not committed')
    require(subprocess.run(['git', 'cat-file', '-e', f'{commit}:{producer.RESULT}'], cwd=ROOT,
                           capture_output=True).returncode != 0, 'result existed at declaration')
    require(record['source_sha256'] == producer.manifest(commit, d), 'manifest mismatch')
    if verify_history:
        final = fixed.git('log', '-1', '--format=%H', '--', producer.RESULT).decode().strip()
        require(final and final != commit and subprocess.run(['git', 'merge-base', '--is-ancestor', commit, final],
                         cwd=ROOT).returncode == 0, 'result not a separate descendant commit')
        require(fixed.git('show', f'{final}:{producer.RESULT}') == (ROOT/producer.RESULT).read_bytes(), 'uncommitted result')
    systems = anchors(d, full=full)
    for case, anchor, actual in zip(d['cases'], systems, record['systems'], strict=True):
        require(close(actual['target'], anchor['independent_target']) and actual['rank'] == anchor['rank'] and
                actual['exact_policy_rank'] == anchor['rank'] and close(actual['exact_policy_target'], anchor['independent_target']),
                'anchor target/rank mismatch')
        require(np.allclose(actual['overlap_spectrum'], anchor['overlap_spectrum'], atol=1e-10, rtol=0), 'anchor spectrum mismatch')
        require(actual['groups'] == case['expected_groups'] and actual['words'] == case['expected_words'], 'group inventory mismatch')
        require(max(actual['certification_residuals']) <= 1e-9, 'shift certification failed')
        if full:
            expected = {k: anchor[k] for k in actual}
            require(not compare_json_records(expected, actual, 'system', atol=1e-9, rtol=1e-9), 'full system mismatch')
        anchor.update(actual)
    arms = {a['key']: a for a in d['arms']}
    expected = {(ci, ri, arm) for ci in range(len(d['cases'])) for ri in range(d['outer_replicas']) for arm in arms}
    seen, pilots = set(), {}
    for row in record['rows']:
        ci, ri, a = key = row['case'], row['replica'], row['arm']
        require(key in expected and key not in seen, 'unexpected/duplicate row')
        seen.add(key)
        check_row(row, d, d['cases'][ci], systems[ci], arms[a])
        check_trace(row, d['cases'][ci], d)
        if row['batches']:
            require(pilots.setdefault((ci, ri), row['pilot_fingerprint']) == row['pilot_fingerprint'], 'paired pilots differ')
    require(seen == expected, 'incomplete experiment')
    return d, systems


def rank_assessment(record, d, systems):
    results = []
    for ci, case in enumerate(d['cases']):
        full_rank, target = systems[ci]['rank'], systems[ci]['independent_target']
        for arm in d['arms']:
            rows = [r for r in record['rows'] if r['case'] == ci and r['arm'] == arm['key']]
            reduced = lambda r: bool(r['looks']) and not r['failure'] and r['looks'][-1]['retained_rank'] < full_rank
            changed = lambda r: len({l['retained_rank'] for l in r['looks']}) > 1
            def covered(row):
                interval = bounds(row, 'bootstrap_spread_schedule', .95, d)
                return interval is not None and interval[0] <= target <= interval[1]
            correct = lambda r: not r['failure'] and abs(r['estimate']-target) <= d['target_std_error_Ha']
            bins = {}
            for key, subset in [('terminal_reduced', [r for r in rows if reduced(r)]),
                                ('terminal_full', [r for r in rows if not r['failure'] and not reduced(r)]),
                                ('rank_changed', [r for r in rows if changed(r)]),
                                ('rank_stable', [r for r in rows if not changed(r)])]:
                bins[key] = {'attempts': len(subset), 'covered': sum(covered(r) for r in subset),
                             'target_met': sum(r['stopped'] == 'target_met' for r in subset),
                             'within_abs_error_target': sum(correct(r) for r in subset)}
            results.append({'case': case['name'], 'arm': arm['key'], 'rank_changed': sum(changed(r) for r in rows),
                            'ever_reduced': sum(any(l['retained_rank'] < full_rank for l in r['looks']) for r in rows),
                            'false_stops': sum(r['stopped'] == 'target_met' and not correct(r) for r in rows),
                            'bootstrap_rank_mixed': sum(len({x['rank'] for x in r['solve_trace'][len(r['looks']):]}) > 1
                                                        for r in rows if not r['failure']), 'strata': bins})
    stress = {case['name']: sum(r['rank_changed'] for r in results if r['case'] == case['name'])
              for case in d['cases']}
    return results, {'rank_change_counts_across_arms': stress,
                     'qualified_cases': [k for k, n in stress.items() if n >= d['stress_min_changed_trajectories_per_case']]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration-only', action='store_true')
    parser.add_argument('--verify-history', action='store_true')
    parser.add_argument('--full-anchor', action='store_true')
    parser.add_argument('--spot-replay', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args(argv)
    if args.declaration_only:
        print(json.dumps(anchors(producer.declaration(), full=args.full_anchor), indent=2))
        return
    record = read_json(ROOT/producer.RESULT)
    d, systems = check(record, verify_history=args.verify_history, full=args.full_anchor)
    summaries, primary = summarize(record, d, d['cases'], systems)
    ranks, stress = rank_assessment(record, d, systems)
    if args.spot_replay:
        require(producer.fixed.environment() == d['environment'], 'spot replay environment mismatch')
        producer.manifest(record['declaration_commit'], d, live=True)
        for ci, case in enumerate(d['cases']):
            producer.adaptive.SYSTEMS[ci] = producer.build_system(case, d)
            for ai, arm in enumerate(d['arms']):
                saved = next(r for r in record['rows'] if (r['case'], r['replica'], r['arm']) == (ci, 0, arm['key']))
                actual = producer.task((d, ci, 0, ai))
                problems = compare_json_records(actual, saved, 'replay', atol=1e-7, rtol=1e-7)
                require(not problems, f'spot replay mismatch: {problems[:3]}')
            del producer.adaptive.SYSTEMS[ci]
        print('Four declared spot replays passed', file=sys.stderr)
    if args.report:
        print(json.dumps({'systems': systems, 'summaries': summaries, 'primary': primary,
                          'rank_assessment': ranks, 'stress': stress}, indent=2, allow_nan=False))
    else:
        print(f'Verified {len(record["rows"])} acquisitions; primary {primary}; stress {stress}')


if __name__ == '__main__':
    main()
