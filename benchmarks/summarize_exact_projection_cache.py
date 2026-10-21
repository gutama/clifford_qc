"""Generate exact-cache comparisons from retained per-cell observations."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from benchmarks.metadata_io import read_json
from benchmarks.run_exact_projection_cache import RECORD, decode_science
from benchmarks.check_exact_projection_cache import compare_science

REPORT = ROOT/'benchmarks/results/exact_projection_cache.md'


def median(values):
    return statistics.median(values)


def iqr(values):
    if len(values) < 2: return '—'
    q = statistics.quantiles(values, n=4, method='inclusive')
    return f'[{q[0]:.3f}, {q[2]:.3f}]'


def render(record):
    c = record['config']; workers = record['workers']
    sciences = {k: decode_science(v) for k, v in record['scientific_payloads'].items()}
    good = [w for w in workers if w['status'] == 'ok']
    complete = len(good) == len(workers) and all(w['status'] == 'ok' for w in record['historical_controls'])
    lines = ['# Exact packet projection caching', '',
        'Exploratory exact classical sector-vector comparisons; caches are opt-in and defaults remain unchanged.',
        f'{len(good)}/{len(workers)} scheduled cells retained; '
        f'{sum(w["status"] == "ok" for w in record["historical_controls"])}/{len(record["historical_controls"])} post-timing historical controls retained. '
        f'{len(sciences)} content-addressed scientific payloads are stored once; every cell retains its own timing/memory observations.', '',
        'Basis caching owns one V† for each fixed accepted span; frame caching owns group CSC blocks and a fixed frame adjoint. '
        'It caches neither candidate vectors nor H actions. Construction is included in solver and total times.',
        'The supervisor alone surveys committed records. Worker RSS includes imports, setup, solver and final serialization. '
        'Phase RSS values are cumulative OS high-water marks, not isolated phase allocation peaks. Owned cache bytes exclude Python/container overhead.', '']
    equivalence = {}
    for case in c['inputs']:
        for comparison in c['comparisons']:
            group = [w for w in good if w['input'] == case['name'] and w['policy'] == comparison['policy']]
            baseline = [w for w in group if w['cache_mode'] == 'none']
            controls = [w for w in record['historical_controls'] if w['input'] == case['name']
                        and w['policy'] == comparison['policy'] and w['status'] == 'ok']
            base = sciences[baseline[0]['scientific_sha256']] if baseline else None
            equivalence[case['name'], comparison['policy']] = (
                base is not None and bool(controls)
                and not compare_science(base, sciences[controls[0]['scientific_sha256']], case['sector_dimension'])
                and all(not compare_science(sciences[w['scientific_sha256']], base, case['sector_dimension']) for w in group))
    all_equivalent = all(equivalence.values())
    promotion = c['promotion']
    # Promotion is one joint decision: every cell and every input/policy, then primary speed and RSS.
    eligible = ((complete or not promotion['all_scheduled_cells_required'])
                and (all_equivalent or not promotion['all_inputs_and_policies_must_preserve_trajectory']))
    for case in c['inputs']:
        name = case['name']; unit = case['energy_unit']
        lines += [f'## {name} ({case["role"]}, {unit})', '',
            f'{case["sector_dimension"]} determinants; {case["maximum_columns"]} columns; target {case["accuracy_target"]} {unit}.', '']
        for comparison in c['comparisons']:
            policy = comparison['policy']; modes = comparison['cache_modes']
            group = [w for w in good if w['input'] == name and w['policy'] == policy]
            baseline = [w for w in group if w['cache_mode'] == 'none']
            equivalent = equivalence[name, policy]
            lines += [f'### {policy}', '', f'Physical trajectories/selections/counters versus the pre-change module: {"PASS" if equivalent else "FAIL"}.', '',
                '| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |',
                '|---|---:|---:|---:|---|---|---:|---:|']
            for mode in modes:
                ws = [w for w in group if w['cache_mode'] == mode]
                if not ws:
                    lines.append(f'| {mode} | 0/{case["repeats"]} | — | — | — | — | — | 0/{case["repeats"]} |'); continue
                totals = [w['setup_seconds']+w['solver_seconds'] for w in ws]
                ratios = [(b['setup_seconds']+b['solver_seconds'])/(w['setup_seconds']+w['solver_seconds'])
                          for w in ws for b in baseline if b['repeat'] == w['repeat']]
                errors = [sciences[w['scientific_sha256']]['trajectory'][-1]['energy']-record['references'][name] for w in ws]
                targets = sum(abs(error) <= case['accuracy_target'] for error in errors)
                lines.append(f'| {mode} | {len(ws)}/{case["repeats"]} | {median([w["setup_seconds"] for w in ws]):.3f} | '
                    f'{median([w["solver_seconds"] for w in ws]):.3f} | {median(totals):.3f}, {iqr(totals)} | '
                    f'{median(ratios):.3f}, {iqr(ratios)} | {median(errors):.9g} | {targets}/{case["repeats"]} |')
            lines += ['', '| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |',
                '|---|---:|---:|---:|---:|---:|---:|']
            for mode in modes:
                ws = [w for w in group if w['cache_mode'] == mode]
                if not ws: lines.append(f'| {mode} | — | — | — | — | — | — |'); continue
                peaks = [median([w[k]/2**20 for w in ws]) for k in ('imports_peak_rss_bytes', 'setup_peak_rss_bytes',
                    'solver_end_peak_rss_bytes', 'worker_peak_rss_bytes')]
                lines.append(f'| {mode} | '+' | '.join(f'{v:.3f}' for v in peaks)
                    +f' | {median([w["cache"]["peak_owned_bytes"]/2**20 for w in ws]):.3f} | '
                    f'{median([w["cache"]["construction_wall_seconds"] for w in ws]):.4f} |')
            unique = {w['scientific_sha256'] for w in group}
            audits = [d for key in unique for d in sciences[key]['decisions'] if d['audit']]
            lines += ['', f'{len(unique)} distinct scientific trajectory/trajectories; '
                f'{len(audits)} unique audits, {sum(d["audit_missed_winner"] for d in audits)} missed winners.']
            if case['role'] == 'primary' and policy == 'exhaustive' and baseline:
                base_time = median([w['setup_seconds']+w['solver_seconds'] for w in baseline])
                base_rss = median([w['worker_peak_rss_bytes'] for w in baseline])
                for mode in (m for m in modes if m != 'none'):
                    ws = [w for w in group if w['cache_mode'] == mode]
                    if not ws: continue
                    reduction = 1-median([w['setup_seconds']+w['solver_seconds'] for w in ws])/base_time
                    rss_ratio = median([w['worker_peak_rss_bytes'] for w in ws])/base_rss
                    passes = (eligible and reduction >= promotion['primary_minimum_setup_plus_solver_reduction']
                              and rss_ratio <= promotion['maximum_median_rss_ratio'])
                    lines += [f'Primary {mode} speed/RSS clause: {"PASS" if passes else "FAIL"}; '
                        f'{100*reduction:.2f}% total-time reduction, {rss_ratio:.4f}× worker RSS.']
            lines += ['']
    lines += [f'All reported policies preserve the pre-change physical trajectories: {"PASS" if all_equivalent else "FAIL"}.', '',
        'Medians of setup and solver are separate; the total is the median of paired sums. Ratios pair cache and baseline cells within each repetition. '
        'Repeated deterministic paths do not increase independent scientific/audit sample size. A fixed-budget speedup is not a matched-accuracy claim. '
        'BeH2 tests a different small molecular active space, not larger-molecule scaling. No optimized-SCI, hardware or quantum advantage is established.', '',
        f'Declaration: `{record["declaration_commit"]}`. Execution: `{record["source_commit"]}`.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true'); parser.add_argument('--record', type=Path, default=ROOT/RECORD)
    args = parser.parse_args(argv); text = render(read_json(args.record))
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text: raise SystemExit('exact projection report is stale')
    else: REPORT.write_text(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
