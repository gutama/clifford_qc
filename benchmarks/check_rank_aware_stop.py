"""Independently verify guarded stops, shot/solve costs, and paired outcomes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rank_aware_stop as producer
from benchmarks.check_molecular_adaptive_stop import check_interval, close, fingerprint, z
from benchmarks.check_molecular_interval_coverage import require, wilson
from benchmarks.check_stretched_molecular_stop import anchors
from benchmarks.experiment_artifacts import read_json
from clifford_qc.reproducibility import compare_json_records


def check_row(row, d, case, system, arm):
    groups = system['groups']
    shots = d['cumulative_average_shots_per_group']
    family = 1 if arm['fixed'] else len(shots)
    budgets = np.diff([0]+shots)*groups
    target = d['target_std_error_Ha']
    for stream, field in enumerate(('measurement_seed', 'bootstrap_seed')):
        require(row[field] == int(np.random.SeedSequence([d['seed_root'], row['case'], row['replica'], stream])
                                  .generate_state(1, np.uint64)[0]), 'seed mismatch')
    batches, looks, trace = row['batches'], row['looks'], row['solve_trace']
    require(len(looks) <= len(batches) <= len(shots), 'batch count mismatch')
    for i, b in enumerate(batches):
        require(b['shots'] == budgets[i] and 1 <= b['groups'] <= groups, 'batch budget mismatch')
    require(row['shots'] == sum(b['shots'] for b in batches), 'physical shot mismatch')
    if batches:
        require(batches[0]['groups'] == groups and fingerprint(row['pilot_fingerprint']), 'invalid pilot')
    else:
        require(row['pilot_fingerprint'] is None, 'unexpected pilot')
    measured = [s for s in trace if s['kind'] == 'look']
    require(len(measured) <= len(batches) and len(measured) >= len(looks), 'measured solve count mismatch')
    observed_shots = {int(x*groups) for x in shots[:len(batches)]}
    trace_locs = {}
    for s in trace:
        require(s['kind'] in ('look', 'bootstrap') and s['shots'] in observed_shots, 'undeclared solve')
        if s['kind'] == 'look':
            require(s['shots'] not in trace_locs, 'duplicate measured solve')
            trace_locs[s['shots']] = []
        else:
            require(s['shots'] in trace_locs and s['shots'] == max(trace_locs), 'bootstrap detached from look')
            trace_locs[s['shots']].append(s)
        if s['failure']:
            continue
        values = s['spectrum']
        require(values and all(math.isfinite(v) for v in values) and values == sorted(values, reverse=True), 'invalid spectrum')
        cutoff = max(case['solver']['tau_s'], values[0]/case['solver']['max_condition'])
        require(close(s['cutoff'], cutoff) and s['rank'] == sum(v > cutoff for v in values), 'rank decision mismatch')
        require(math.isfinite(s['raw_energy']), 'nonfinite raw energy')
    streak = 0
    eligible = []
    for i, look in enumerate(looks):
        step = measured[i]
        require(not step['failure'] and look['look'] == i and look['shots'] == shots[i]*groups == step['shots'], 'look trace mismatch')
        require(look['retained_rank'] == step['rank'] and close(look['raw_energy'], step['raw_energy']), 'look rank/energy mismatch')
        require(close(look['smallest_retained_overlap'], step['spectrum'][step['rank']-1]), 'retained eigenvalue mismatch')
        require(math.isfinite(look['energy']) and math.isfinite(look['std_error']) and look['std_error'] >= 0 and
                0 <= look['control_variates_used'] <= len(system['shift_labels']), 'invalid measured estimate')
        se_ok = not arm['fixed'] and look['std_error'] <= target
        rank_ok = True
        if arm['rank']:
            guard = look.get('rank_guard')
            # A diagnosed numerical guard failure may retain a partially completed look.
            if guard is None and row['failure'] and i == len(looks)-1:
                break
            require(guard is not None and guard['required_rank'] == system['rank'] and
                    guard['live_rank'] == len(step['spectrum']) and guard['evidence'] == 'heuristic', 'rank guard metadata mismatch')
            comparisons = guard['comparisons']
            if comparisons:
                require(len(comparisons) == 3 and close(guard['weakest'], step['spectrum'][-1]), 'margin spectrum mismatch')
                expected = [('absolute', case['solver']['tau_s']), ('relative', 0.),
                            ('condition', step['spectrum'][0]/case['solver']['max_condition'])]
                for c, (name, threshold) in zip(comparisons, expected, strict=True):
                    require(c['cutoff'] == name and close(c['threshold'], threshold) and c['radius'] >= 0 and
                            math.isfinite(c['radius']) and close(c['margin'], guard['weakest']-threshold) and
                            close(c['lower_margin'], c['margin']-c['radius']), 'margin arithmetic mismatch')
                lower = min(c['lower_margin'] for c in comparisons)
                require(close(lower, guard['lower_margin']), 'margin minimum mismatch')
                resolved = step['rank'] == system['rank'] and lower > 0.
            else:
                require(guard['lower_margin'] is None, 'unresolved margin has bound')
                resolved = False
            streak = streak+1 if resolved else 0
            rank_ok = streak >= d['rank_policy']['consecutive_looks']
            require(guard['resolved'] == resolved and guard['consecutive_resolved'] == streak and
                    guard['passed'] == rank_ok, 'rank guard decision mismatch')
            if not resolved and i < len(batches)-1:
                expected_budget = int(budgets[i+1]) if se_ok else (int(budgets[i+1])+1)//2
                require(look.get('next_rank_budget') == expected_budget, 'rank allocation budget mismatch')
            else:
                require('next_rank_budget' not in look, 'unexpected rank allocation')
        else:
            require('rank_guard' not in look, 'undeclared rank guard')
        boot = trace_locs[look['shots']]
        width_ok = not arm['width']
        decision = arm['width'] and se_ok and rank_ok
        # Width checks run only at eligible looks; at a terminal ineligible
        # look bootstrap is diagnostic only and cannot authorize a stop.
        if arm['width'] and 'bootstrap_seed' in look:
            seed = int(np.random.SeedSequence([row['bootstrap_seed'], i]).generate_state(1, np.uint64)[0])
            require(look['bootstrap_seed'] == seed and look['bootstrap_attempted_replicates'] == d['bootstrap_replicas'],
                    'width resampling seed/count mismatch')
        if decision:
            require('bootstrap_seed' in look and 'bootstrap_width_passed' in look, 'missing width assessment')
            width_ok = ('bootstrap_spread' in look and
                        z(1-d['delta'], family)*look['bootstrap_spread'] <= d['bootstrap_half_width_Ha']+1e-13)
            require(look['bootstrap_width_passed'] == width_ok, 'width guard decision mismatch')
        else:
            require('bootstrap_width_passed' not in look, 'spurious width pass')
        if 'bootstrap_spread' in look:
            require(math.isfinite(look['bootstrap_spread']) and look['bootstrap_spread'] >= 0 and
                    len(boot) == d['bootstrap_replicas'] and not any(s['failure'] for s in boot), 'incomplete bootstrap')
        elif 'bootstrap_failure' in look:
            require(len(boot) <= d['bootstrap_replicas'], 'failed bootstrap count mismatch')
        else:
            require(not boot, 'unreported bootstrap solves')
        eligible.append(se_ok and rank_ok and width_ok)
        if i < len(looks)-1:
            require(not look['stopped'] and not eligible[-1], 'missed earlier eligible stop')
    fields = ('estimate', 'std_error', 'delta_interval', 'naive_interval', 'bootstrap_interval', 'final_fingerprint', 'schedule')
    if row['failure']:
        require(row['stopped'] == 'failed' and all(row[k] is None for k in fields), 'invalid failure record')
        return
    require(looks and len(looks) == len(batches) == len(measured) and fingerprint(row['final_fingerprint']), 'invalid successful ledger')
    last = looks[-1]
    require(last['stopped'] and close(last['energy'], row['estimate']) and close(last['std_error'], row['std_error']), 'terminal point mismatch')
    require((row['stopped'] == 'target_met') == eligible[-1], 'stop rule mismatch')
    if row['stopped'] == 'schedule_exhausted':
        require(len(looks) == len(shots), 'incorrect exhaustion')
    elif row['stopped'] == 'deterministic':
        require(not arm['rank'] and not arm['width'] and len(looks) < len(shots), 'unexpected deterministic stop')
    else:
        require(row['stopped'] == 'target_met', 'invalid stop code')
    schedule = {'pilot_shots_per_group': shots[0], 'round_budgets': budgets[1:].tolist(),
                'target_std_error': None if arm['fixed'] else target, 'delta': d['delta'], 'interval_family': family,
                'solver_kwargs': case['solver'], 'control_variates': system['shift_labels'],
                'bootstrap': {'replicates': d['bootstrap_replicas'], 'seed': row['bootstrap_seed'], 'estimator': 'refitted_control_variates'}}
    if arm['rank']:
        schedule['rank_stop'] = d['rank_policy']
    if arm['width']:
        schedule['max_bootstrap_half_width'] = d['bootstrap_half_width_Ha']
    require(row['schedule'] == schedule, 'schedule differs')
    for entry, looks_paid in [(row['delta_interval'], family), (row['naive_interval'], 1)]:
        require(close(entry['std_error'], row['std_error']), 'delta SE mismatch')
        check_interval(entry, row['estimate'], row['std_error'], d['delta'], looks_paid, 'asymptotic')
    if row['bootstrap_interval'] is not None:
        entry = row['bootstrap_interval']
        require(close(entry['std_error'], last['bootstrap_spread']), 'terminal spread mismatch')
        check_interval(entry, row['estimate'], last['bootstrap_spread'], d['delta'], family, 'heuristic')
    else:
        require('bootstrap_failure' in last, 'missing terminal bootstrap')


def check(record, *, verify_history=False, full=False):
    d, previous = producer.declaration()
    fixed = producer.stress.fixed
    require(record['schema'] == producer.SCHEMA and record['declaration'] == d and record['environment'] == d['environment'], 'declaration mismatch')
    commit = record['declaration_commit']
    require(re.fullmatch(r'[0-9a-f]{40}', commit), 'invalid declaration commit')
    require(record['declaration_sha256'] == fixed.digest((ROOT/producer.DECLARATION).read_bytes()) and
            fixed.git('show', f'{commit}:{producer.DECLARATION}') == (ROOT/producer.DECLARATION).read_bytes(), 'declaration not committed')
    require(subprocess.run(['git', 'cat-file', '-e', f'{commit}:{producer.RESULT}'], cwd=ROOT,
                           capture_output=True).returncode != 0, 'result existed at declaration')
    require(record['source_sha256'] == producer.manifest(commit, d, previous), 'manifest mismatch')
    if verify_history:
        final = fixed.git('log', '-1', '--format=%H', '--', producer.RESULT).decode().strip()
        require(final and final != commit and subprocess.run(['git', 'merge-base', '--is-ancestor', commit, final],
                         cwd=ROOT).returncode == 0, 'result not a separate descendant commit')
        require(fixed.git('show', f'{final}:{producer.RESULT}') == (ROOT/producer.RESULT).read_bytes(), 'uncommitted result')
    systems = anchors(previous, full=full)
    for case, anchor, actual in zip(previous['cases'], systems, record['systems'], strict=True):
        require(close(actual['target'], anchor['independent_target']) and actual['rank'] == anchor['rank'] and
                actual['exact_policy_rank'] == anchor['rank'] and actual['groups'] == case['expected_groups'] and
                actual['words'] == case['expected_words'] and max(actual['certification_residuals']) <= 1e-9, 'anchor mismatch')
        require(np.allclose(actual['overlap_spectrum'], anchor['overlap_spectrum'], atol=1e-10, rtol=0), 'spectrum mismatch')
        if full:
            require(not compare_json_records({k: anchor[k] for k in actual}, actual, 'system', atol=1e-9, rtol=1e-9), 'full anchor mismatch')
        anchor.update(actual)
    arms = {a['key']: a for a in d['arms']}
    expected = {(ci, ri, arm) for ci in range(len(systems)) for ri in range(d['outer_replicas']) for arm in arms}
    seen, pilots = set(), {}
    for row in record['rows']:
        ci, ri, a = key = row['case'], row['replica'], row['arm']
        require(key in expected and key not in seen, 'unexpected or duplicate row')
        seen.add(key)
        check_row(row, d, previous['cases'][ci], systems[ci], arms[a])
        if row['batches']:
            require(pilots.setdefault((ci, ri), row['pilot_fingerprint']) == row['pilot_fingerprint'], 'paired pilots differ')
    require(seen == expected, 'incomplete experiment')
    return d, previous, systems


def summarize(record, d, previous, systems):
    summaries, criteria = [], []
    for ci, case in enumerate(previous['cases']):
        exact, rank = systems[ci]['independent_target'], systems[ci]['rank']
        n, target = d['outer_replicas'], d['target_std_error_Ha']
        for arm in d['arms']:
            rows = [r for r in record['rows'] if r['case'] == ci and r['arm'] == arm['key']]
            successful = [r for r in rows if not r['failure']]
            errors = [r['estimate']-exact for r in successful]
            met = [r for r in successful if r['stopped'] == 'target_met']
            true = sum(abs(r['estimate']-exact) <= target for r in met)
            false = len(met)-true
            summary = {'case': case['name'], 'arm': arm['key'], 'attempts': n, 'target_met': len(met),
                       'true_precision_stops': true, 'false_precision_stops': false,
                       'true_stop_wilson95': wilson(true, n), 'false_stop_wilson95': wilson(false, n),
                       'exhausted': sum(r['stopped'] == 'schedule_exhausted' for r in rows), 'failed': n-len(successful),
                       'terminal_within_target': sum(abs(e) <= target for e in errors),
                       'reduced_rank_stops': sum(r['looks'][-1]['retained_rank'] < rank for r in met),
                       'terminal_reduced_rank': sum(r['looks'][-1]['retained_rank'] < rank for r in successful),
                       'mean_shots': float(np.mean([r['shots'] for r in rows])), 'total_shots': sum(r['shots'] for r in rows),
                       'mean_looks': float(np.mean([len(r['batches']) for r in rows])),
                       'measured_solves': sum(s['kind'] == 'look' for r in rows for s in r['solve_trace']),
                       'bootstrap_solves': sum(s['kind'] == 'bootstrap' for r in rows for s in r['solve_trace']),
                       'rmse_Ha': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                       'bias_Ha': float(np.mean(errors)) if errors else None}
            for name in ('delta_interval', 'bootstrap_interval'):
                available = [r[name] for r in successful if r[name] is not None]
                count = sum(v['lower'] <= exact <= v['upper'] for v in available)
                summary[name] = {'covered': count, 'valid': len(available), 'coverage': count/n,
                                 'covered_among_stops': sum(r[name] is not None and r[name]['lower'] <= exact <= r[name]['upper'] for r in met),
                                 'median_width_Ha': float(np.median([v['upper']-v['lower'] for v in available])) if available else None}
            summaries.append(summary)
        p = d['primary_criteria']
        reference = next(s for s in summaries if s['case'] == case['name'] and s['arm'] == 'se_only')
        rank_guard = next(s for s in summaries if s['case'] == case['name'] and s['arm'] == p['arm'])
        passed = (rank_guard['true_precision_stops'] >= p['minimum_true_precision_stops'] and
                  rank_guard['false_precision_stops'] <= p['maximum_false_precision_stops'] and
                  rank_guard['false_precision_stops'] < reference['false_precision_stops'])
        criteria.append({'case': case['name'], 'passed': passed, 'baseline_false_stops': reference['false_precision_stops'],
                         'guard_false_stops': rank_guard['false_precision_stops'], 'guard_true_stops': rank_guard['true_precision_stops']})
    return summaries, criteria


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration-only', action='store_true')
    parser.add_argument('--verify-history', action='store_true')
    parser.add_argument('--full-anchor', action='store_true')
    parser.add_argument('--spot-replay', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args(argv)
    if args.declaration_only:
        _, previous = producer.declaration()
        print(json.dumps(anchors(previous, full=args.full_anchor), indent=2))
        return
    record = read_json(ROOT/producer.RESULT)
    d, previous, systems = check(record, verify_history=args.verify_history, full=args.full_anchor)
    summaries, criteria = summarize(record, d, previous, systems)
    if args.spot_replay:
        require(producer.stress.fixed.environment() == d['environment'], 'replay environment mismatch')
        producer.manifest(record['declaration_commit'], d, previous, live=True)
        for ci, case in enumerate(previous['cases']):
            producer.SYSTEMS[ci] = producer.stress.build_system(case, previous)
            for ai, arm in enumerate(d['arms']):
                saved = next(r for r in record['rows'] if (r['case'], r['replica'], r['arm']) == (ci, 0, arm['key']))
                actual = producer.task((d, previous, ci, 0, ai))
                problems = compare_json_records(actual, saved, 'replay', atol=1e-7, rtol=1e-7)
                require(not problems, f'replay differs: {problems[:3]}')
            del producer.SYSTEMS[ci]
        print('Eight designated spot replays passed', file=sys.stderr)
    if args.report:
        print(json.dumps({'systems': systems, 'summaries': summaries, 'criteria': criteria}, indent=2, allow_nan=False))
    else:
        print(f'Verified {len(record["rows"])} acquisitions; primary {criteria}')


if __name__ == '__main__':
    main()
