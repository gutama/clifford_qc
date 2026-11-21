"""Independent molecular anchors, unconditional coverage accounting and spot replay."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from statistics import NormalDist
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_molecular_interval_coverage as producer
from benchmarks.check_cross_domain_ratio import case_data
from benchmarks.experiment_artifacts import read_json
from clifford_qc.reproducibility import compare_json_records


def require(condition, message):
    if not condition:
        raise ValueError(message)


def anchors(d, cases):
    """Independently build occupation-sign projected matrices and singlet ground."""
    systems = []
    for case in cases:
        independent = case_data(case)
        s, h = independent['overlap'], independent['projected']
        exact = float(eigh(h, s, eigvals_only=True)[0])
        shared, _, _, meta = producer.build_system(case, d, prepare=False)
        measured_s, measured_h = shared.bank.matrices()
        require(np.allclose(measured_s, s, atol=1e-10, rtol=0), 'independent overlap mismatch')
        require(np.allclose(measured_h, h, atol=1e-9, rtol=0), 'independent projected Hamiltonian mismatch')
        require(abs(meta['target']-exact) < 1e-9, 'independent Ritz target mismatch')
        require(exact >= independent['ground']-1e-9, 'projected target below singlet ground')
        systems.append({**meta, 'independent_target': exact, 'exact_singlet_ground': independent['ground']})
    return systems


def check(record, *, verify_history=False):
    d, cases = producer.declaration()
    require(record['schema'] == producer.SCHEMA and record['declaration'] == d, 'declaration mismatch')
    require(record['declaration_sha256'] == producer.digest((ROOT/producer.DECLARATION).read_bytes()),
            'declaration digest mismatch')
    require(record['environment'] == d['environment'], 'execution environment mismatch')
    commit = record['declaration_commit']
    require(re.fullmatch(r'[0-9a-f]{40}', commit), 'invalid declaration commit')
    require(producer.git('show', f'{commit}:{producer.DECLARATION}') == (ROOT/producer.DECLARATION).read_bytes(),
            'declaration was not present at the named commit')
    if subprocess.run(['git', 'cat-file', '-e', f'{commit}:{producer.RESULT}'], cwd=ROOT,
                      capture_output=True).returncode == 0:
        raise ValueError('result already existed at declaration commit')
    require(record['source_sha256'] == producer.manifest(commit, d, cases), 'source manifest mismatch')
    if verify_history:
        result_commit = producer.git('log', '-1', '--format=%H', '--', producer.RESULT).decode().strip()
        require(result_commit and result_commit != commit, 'result does not have a separate commit')
        require(subprocess.run(['git', 'merge-base', '--is-ancestor', commit, result_commit], cwd=ROOT).returncode == 0,
                'declaration is not an ancestor of the result commit')
        require(producer.git('show', f'{result_commit}:{producer.RESULT}') == (ROOT/producer.RESULT).read_bytes(),
                'uncommitted result bytes')
    systems = anchors(d, cases)
    for stated, actual in zip(record['systems'], systems, strict=True):
        expected = {k: actual[k] for k in stated}
        require(not compare_json_records(expected, stated, 'system', atol=1e-9, rtol=1e-9),
                'recorded system metadata differs from anchor')
    expected_keys = {(ci, bi, r) for ci in range(len(cases)) for bi in range(len(d['shots_per_qwc_group']))
                     for r in range(d['outer_replicas'])}
    seen = set()
    for row in record['rows']:
        ci, bi, ri = key = row['case'], row['budget'], row['replica']
        require(key in expected_keys and key not in seen, 'missing/duplicate/undeclared row')
        seen.add(key)
        for stream, field in enumerate(('measurement_seed', 'bootstrap_seed')):
            # Derive independently from the declared rule, not from the runner helper.
            expected = int(np.random.SeedSequence([d['seed_root'], ci, bi, ri, stream]).generate_state(1, np.uint64)[0])
            require(row[field] == expected, 'seed mismatch')
        require(row['shots'] == systems[ci]['groups']*d['shots_per_qwc_group'][bi], 'physical shot mismatch')
        require(re.fullmatch(r'[0-9a-f]{64}', row['histogram_sha256']), 'invalid histogram digest')
        arms = row['arms']
        require(set(arms) == set(d['arms']), 'arm mismatch')
        if row['point_failure']:
            require(row['rank'] is None and all(x is None for x in arms.values()), 'point failure not retained')
            continue
        require(1 <= row['rank'] <= 3 and arms['raw'] is not None, 'invalid solved row')
        require((arms['refitted_cv'] is None) == bool(row.get('cv_point_failure')), 'missing correction without failure')
        for arm, entry in arms.items():
            if entry is None:
                continue
            require(math.isfinite(entry['estimate']), 'nonfinite point')
            if entry['delta_failure']:
                require(entry['sigma'] is None and entry['delta_scale'] is None, 'delta failure not retained')
            else:
                require(all(math.isfinite(entry[k]) and entry[k] >= 0 for k in ('sigma', 'delta_scale')),
                        'nonfinite/negative interval scale')
            require((entry['draws'] is None) == bool(entry['bootstrap_failure']), 'failed draw not retained')
            if entry['draws'] is not None:
                require(len(entry['draws']) == d['bootstrap_replicas'] and np.isfinite(entry['draws']).all(),
                        'incomplete/nonfinite bootstrap draws')
            if arm == 'refitted_cv':
                require(entry['delta_scale'] == entry['sigma'], 'correction interval scale mismatch')
                require(set(entry['used']).isdisjoint(entry['unreadable']) and
                        set(entry['used']+entry['unreadable']) <= set(systems[ci]['shift_labels']), 'invalid shift inventory')
    require(seen == expected_keys, 'incomplete declared experiment')
    return d, cases, systems


def interval(entry, method, level):
    if entry is None:
        return None
    if method == 'bootstrap_spread':
        if entry['draws'] is None:
            return None
        lo, hi = np.quantile(entry['draws'], [0.16, 0.84])
        scale = float((hi-lo)/2)
    else:
        if entry['delta_failure']:
            return None
        scale = entry['delta_scale']
    width = NormalDist().inv_cdf((1+level)/2)*scale
    return entry['estimate']-width, entry['estimate']+width


def wilson(k, n):
    z = NormalDist().inv_cdf(0.975)
    p = k/n
    middle = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., middle-radius), min(1., middle+radius)


def summarize(record, d, cases, systems):
    summaries = []
    for ci, case in enumerate(cases):
        target = systems[ci]['independent_target']
        for bi, shots in enumerate(d['shots_per_qwc_group']):
            rows = [r for r in record['rows'] if r['case'] == ci and r['budget'] == bi]
            for arm in d['arms']:
                entries = [r['arms'][arm] for r in rows]
                errors = [e['estimate']-target for e in entries if e is not None]
                for method in d['intervals']:
                    for level in d['levels']:
                        bounds = [interval(e, method, level) for e in entries]
                        valid = [x for x in bounds if x is not None]
                        count = sum(lo <= target <= hi for lo, hi in valid)
                        n = len(rows)
                        cp = float(binom.cdf(count, n, level))
                        lo, hi = wilson(count, n)
                        summaries.append({'case': case['name'], 'shots_per_group': shots,
                                          'arm': arm, 'method': method, 'level': level,
                                          'covered': count, 'total': n, 'valid': len(valid),
                                          'coverage': count/n, 'wilson_95': [lo, hi],
                                          'undercoverage_p': cp,
                                          'median_width_Ha': float(np.median([b-a for a,b in valid])) if valid else None,
                                          'rmse_Ha': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                                          'bias_Ha': float(np.mean(errors)) if errors else None})
    primary = [s for s in summaries if (s['arm'], s['method'], s['level']) == ('refitted_cv','bootstrap_spread',0.95)]
    limit = d['primary_test']['familywise_alpha']/d['primary_test']['cells']
    return summaries, {'threshold': limit, 'rejected_cells': [s['case']+':'+str(s['shots_per_group'])
                                                             for s in primary if s['undercoverage_p'] < limit]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration-only', action='store_true')
    parser.add_argument('--verify-history', action='store_true')
    parser.add_argument('--spot-replay', action='store_true', help='replay replica zero in each cell')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args(argv)
    if args.declaration_only:
        d, cases = producer.declaration()
        print('Result-free independent anchors:', anchors(d, cases))
        return
    record = read_json(ROOT/producer.RESULT)
    d, cases, systems = check(record, verify_history=args.verify_history)
    summaries, verdict = summarize(record, d, cases, systems)
    if args.spot_replay:
        require(producer.environment() == d['environment'], 'spot replay environment differs')
        producer.manifest(record['declaration_commit'], d, cases, live=True)
        for ci, case in enumerate(cases):
            producer.SYSTEMS[ci] = producer.build_system(case, d)
        for stored in record['rows']:
            if stored['replica'] == 0:
                actual = producer.row((d, stored['case'], stored['budget'], 0))
                problems = compare_json_records(actual, stored, 'replay', atol=1e-7, rtol=1e-7)
                require(not problems, f'spot replay differs: {problems[:3]}')
        print('Four declared spot replays passed')
    if args.report:
        import json
        print(json.dumps({'systems': systems, 'summaries': summaries, 'primary': verdict}, indent=2, allow_nan=False))
    else:
        print(f'Independent verification passed: {len(record["rows"])} paired rows; primary {verdict}')


if __name__ == '__main__':
    main()
