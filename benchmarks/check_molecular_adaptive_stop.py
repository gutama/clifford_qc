"""Verify adaptive molecular acquisition history, shot ledgers and coverage."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import subprocess
import sys

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_molecular_adaptive_stop as producer
from benchmarks.check_molecular_interval_coverage import anchors, require, wilson
from benchmarks.experiment_artifacts import read_json
from clifford_qc.reproducibility import compare_json_records

METHODS = ('delta_schedule', 'delta_single', 'bootstrap_spread_schedule', 'bootstrap_spread_single')


def z(level, family):
    return NormalDist().inv_cdf(1-(1-level)/(2*family))


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)


def fingerprint(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)


def check_interval(entry, estimate, scale, delta, family, evidence):
    require(entry['evidence'] == evidence and entry['delta'] == delta, 'interval label/delta mismatch')
    require(close(entry['estimate'], estimate), 'interval center mismatch')
    radius = z(1-delta, family)*scale
    require(close(entry['lower'], estimate-radius) and close(entry['upper'], estimate+radius),
            'interval radius mismatch')


def check_row(row, d, previous, system, arm):
    groups = system['groups']
    cumulative = d['cumulative_average_shots_per_group']
    family = len(cumulative)
    batches, looks = row['batches'], row['looks']
    for stream, field in enumerate(('measurement_seed', 'bootstrap_seed')):
        expected = int(np.random.SeedSequence([d['seed_root'], row['case'], row['replica'], stream])
                       .generate_state(1, np.uint64)[0])
        require(row[field] == expected, 'seed mismatch')
    require(len(looks) <= len(batches) <= family, 'invalid batch/solve count')
    budgets = np.diff([0]+cumulative)*groups
    for i, batch in enumerate(batches):
        require(batch['shots'] == budgets[i] and 1 <= batch['groups'] <= groups, 'batch cost mismatch')
    require(row['shots'] == sum(b['shots'] for b in batches), 'total cost mismatch')
    if batches:
        require(batches[0]['groups'] == groups and fingerprint(row['pilot_fingerprint']), 'pilot mismatch')
    else:
        require(row['pilot_fingerprint'] is None, 'unexpected pilot digest')
    target = d['target_std_error_Ha']
    for i, look in enumerate(looks):
        require(look['look'] == i and look['shots'] == cumulative[i]*groups, 'look ledger mismatch')
        require(1 <= look['retained_rank'] <= system['rank'], 'invalid retained rank')
        require(math.isfinite(look['energy']) and math.isfinite(look['std_error']) and look['std_error'] >= 0,
                'invalid point or standard error')
        if i < len(looks)-1:
            require(not look['stopped'] and look['std_error'] > target, 'missed earlier stopping condition')
        if arm['control_variates']:
            require(0 <= look['control_variates_used'] <= len(system['shift_labels']) and
                    math.isfinite(look['raw_energy']) and math.isfinite(look['raw_std_error']) and
                    look['raw_std_error'] >= 0, 'invalid control-variate look')
    fields = ('estimate', 'std_error', 'delta_interval', 'naive_interval', 'bootstrap_interval',
              'final_fingerprint', 'schedule')
    if row['failure']:
        require(row['stopped'] == 'failed' and all(row[k] is None for k in fields), 'invalid failure record')
        return
    require(looks and len(looks) == len(batches) and fingerprint(row['final_fingerprint']), 'incomplete successful run')
    last = looks[-1]
    require(last['stopped'] and close(last['energy'], row['estimate']) and
            close(last['std_error'], row['std_error']), 'terminal point mismatch')
    if row['stopped'] == 'target_met':
        require(last['std_error'] <= target, 'premature target stop')
    elif row['stopped'] == 'schedule_exhausted':
        require(len(looks) == family and last['std_error'] > target, 'incorrect exhaustion')
    else:
        require(row['stopped'] == 'deterministic' and len(looks) < family and last['std_error'] > target,
                'invalid deterministic exit')
    schedule = {'pilot_shots_per_group': cumulative[0], 'round_budgets': budgets[1:].tolist(),
                'target_std_error': target, 'delta': d['delta'], 'interval_family': family,
                'solver_kwargs': previous['solver'], 'bootstrap': {'replicates': d['bootstrap_replicas'],
                                                                  'seed': row['bootstrap_seed']}}
    if arm['control_variates']:
        schedule['control_variates'] = system['shift_labels']
        schedule['bootstrap']['estimator'] = 'refitted_control_variates'
    require(row['schedule'] == schedule, 'schedule mismatch')
    estimate, delta = row['estimate'], d['delta']
    naive = row['naive_interval']
    # The raw functional has a variance floor for the radius, not for its SE.
    scale = (naive['upper']-naive['lower'])/(2*z(1-delta, 1))
    require(math.isfinite(scale) and scale >= 0, 'invalid delta scale')
    for entry, looks_paid in [(naive, 1), (row['delta_interval'], family)]:
        require(close(entry['std_error'], row['std_error']), 'delta SE mismatch')
        check_interval(entry, estimate, scale, delta, looks_paid, 'asymptotic')
    if arm['control_variates']:
        require(close(scale, row['std_error']), 'corrected radius SE mismatch')
    bootstrap = row['bootstrap_interval']
    if bootstrap is None:
        require(bool(last.get('bootstrap_failure')) and 'bootstrap_spread' not in last, 'lost bootstrap failure')
    else:
        spread = last['bootstrap_spread']
        require(math.isfinite(spread) and spread >= 0 and close(bootstrap['std_error'], spread), 'bootstrap spread mismatch')
        check_interval(bootstrap, estimate, spread, delta, family, 'heuristic')


def check(record, *, verify_history=False):
    d, previous, cases = producer.declaration()
    require(record['schema'] == producer.SCHEMA and record['declaration'] == d, 'declaration mismatch')
    require(record['declaration_sha256'] == producer.fixed.digest((ROOT/producer.DECLARATION).read_bytes()),
            'declaration digest mismatch')
    require(record['environment'] == d['environment'], 'environment mismatch')
    git, commit = producer.fixed.git, record['declaration_commit']
    require(re.fullmatch(r'[0-9a-f]{40}', commit), 'invalid declaration commit')
    require(git('show', f'{commit}:{producer.DECLARATION}') == (ROOT/producer.DECLARATION).read_bytes(),
            'declaration not committed as stated')
    require(subprocess.run(['git', 'cat-file', '-e', f'{commit}:{producer.RESULT}'], cwd=ROOT,
                           capture_output=True).returncode != 0, 'result existed at declaration commit')
    require(record['source_sha256'] == producer.manifest(commit, d, previous, cases), 'source manifest mismatch')
    if verify_history:
        final = git('log', '-1', '--format=%H', '--', producer.RESULT).decode().strip()
        require(final and final != commit, 'result must be a separate commit')
        require(subprocess.run(['git', 'merge-base', '--is-ancestor', commit, final], cwd=ROOT).returncode == 0,
                'declaration not ancestor of result')
        require(git('show', f'{final}:{producer.RESULT}') == (ROOT/producer.RESULT).read_bytes(), 'uncommitted result')
    systems = anchors(previous, cases)
    for actual, saved in zip(systems, record['systems'], strict=True):
        expected = {k: v for k, v in actual.items() if k not in ('independent_target', 'exact_singlet_ground')}
        require(not compare_json_records(expected, saved, 'system', atol=1e-9, rtol=1e-9), 'anchor metadata mismatch')
    arms = {a['key']: a for a in d['arms']}
    expected = {(ci, ri, a) for ci in range(len(cases)) for ri in range(d['outer_replicas']) for a in arms}
    seen, pilots = set(), {}
    for row in record['rows']:
        ci, ri, a = key = row['case'], row['replica'], row['arm']
        require(key in expected and key not in seen, 'unexpected/duplicate row')
        seen.add(key)
        check_row(row, d, previous, systems[ci], arms[a])
        if row['batches']:
            value = pilots.setdefault((ci, ri), row['pilot_fingerprint'])
            require(value == row['pilot_fingerprint'], 'paired pilots differ')
    require(seen == expected, 'incomplete acquisition experiment')
    return d, cases, systems


def bounds(row, method, level, d):
    if row['failure']:
        return None
    if method.startswith('bootstrap'):
        if row['bootstrap_interval'] is None:
            return None
        scale = row['bootstrap_interval']['std_error']
    else:
        entry = row['naive_interval']
        scale = (entry['upper']-entry['lower'])/(2*z(1-d['delta'], 1))
    family = len(d['cumulative_average_shots_per_group']) if method.endswith('_schedule') else 1
    radius = z(level, family)*scale
    return row['estimate']-radius, row['estimate']+radius


def summarize(record, d, cases, systems):
    summaries = []
    for ci, case in enumerate(cases):
        target = systems[ci]['independent_target']
        for a in d['arms']:
            arm = a['key']
            rows = [r for r in record['rows'] if r['case'] == ci and r['arm'] == arm]
            n = len(rows)
            met = [r['stopped'] == 'target_met' for r in rows]
            errors = [r['estimate']-target for r in rows if not r['failure']]
            shots = [r['shots'] for r in rows]
            stops = {r: sum(x['stopped'] == r for x in rows)
                     for r in ('target_met', 'schedule_exhausted', 'deterministic', 'failed')}
            common = {'case': case['name'], 'arm': arm, 'total': n, 'stops': stops,
                      'median_shots': float(np.median(shots)), 'mean_shots': float(np.mean(shots)),
                      'total_shots': sum(shots), 'shot_range': [min(shots), max(shots)],
                      'look_counts': [sum(len(r['batches']) == i for r in rows)
                                      for i in range(len(d['cumulative_average_shots_per_group'])+1)],
                      'ranks': {str(i): sum(bool(r['looks']) and not r['failure'] and r['looks'][-1]['retained_rank'] == i
                                           for r in rows) for i in range(1, systems[ci]['rank']+1)},
                      'within_target_absolute_error': sum(abs(e) <= d['target_std_error_Ha'] for e in errors),
                      'point_rmse_Ha': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                      'point_bias_Ha': float(np.mean(errors)) if errors else None}
            for method in METHODS:
                for level in d['levels']:
                    intervals = [bounds(r, method, level, d) for r in rows]
                    valid = [x for x in intervals if x is not None]
                    covered = [x is not None and x[0] <= target <= x[1] for x in intervals]
                    k = sum(covered)
                    joint = sum(c and m for c, m in zip(covered, met))
                    summaries.append({**common, 'method': method, 'level': level,
                                      'covered': k, 'valid': len(valid), 'coverage': k/n,
                                      'covered_and_target_met': joint,
                                      'coverage_given_target_met': joint/sum(met) if any(met) else None,
                                      'coverage_given_valid': k/len(valid) if valid else None,
                                      'wilson_95': wilson(k, n), 'undercoverage_p': float(binom.cdf(k, n, level)),
                                      'median_width_Ha': float(np.median([b-a for a, b in valid])) if valid else None})
    p = d['primary_test']
    primary = [s for s in summaries if (s['arm'], s['method'], s['level']) == (p['arm'], p['method'], p['level'])]
    require(len(primary) == p['cells'], 'primary test family mismatch')
    alpha = p['familywise_alpha']/p['cells']
    return summaries, {'threshold': alpha, 'rejected_cells': [s['case'] for s in primary if s['undercoverage_p'] < alpha]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration-only', action='store_true')
    parser.add_argument('--verify-history', action='store_true')
    parser.add_argument('--spot-replay', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args(argv)
    if args.declaration_only:
        _, previous, cases = producer.declaration()
        print('Result-free independent anchors:', anchors(previous, cases))
        return
    record = read_json(ROOT/producer.RESULT)
    d, cases, systems = check(record, verify_history=args.verify_history)
    summaries, verdict = summarize(record, d, cases, systems)
    if args.spot_replay:
        require(producer.fixed.environment() == d['environment'], 'spot replay environment differs')
        _, previous, _ = producer.declaration()
        producer.manifest(record['declaration_commit'], d, previous, cases, live=True)
        for ci, case in enumerate(cases):
            producer.SYSTEMS[ci] = producer.fixed.build_system(case, previous)
        arms = {a['key']: a for a in d['arms']}
        for stored in record['rows']:
            if stored['replica'] == 0:
                actual = producer.run_arm(d, previous, stored['case'], 0, arms[stored['arm']])
                problems = compare_json_records(actual, stored, 'replay', atol=1e-7, rtol=1e-7)
                require(not problems, f'spot replay differs: {problems[:3]}')
        print('Four declared spot replays passed', file=sys.stderr)
    if args.report:
        print(json.dumps({'systems': systems, 'summaries': summaries, 'primary': verdict}, indent=2, allow_nan=False))
    else:
        print(f'Independent verification passed: {len(record["rows"])} acquisitions; primary {verdict}')


if __name__ == '__main__':
    main()
