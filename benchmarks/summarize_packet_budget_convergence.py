"""Generate the budget-convergence report and its declared readouts from retained observations."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from benchmarks.metadata_io import read_json
from benchmarks.run_packet_budget_convergence import RECORD, decode_science

REPORT = ROOT/'benchmarks/results/packet_budget_convergence.md'


def first_hit(science, reference, threshold):
    """Index of the first prefix at or below the threshold, else None."""
    return next((i for i, row in enumerate(science['trajectory']) if row['energy']-reference <= threshold), None)


def frontier_error(science, reference, columns):
    """Error at the last prefix with at most ``columns`` columns."""
    rows = [row for row in science['trajectory'] if row['dimension'] <= columns]
    return rows[-1]['energy']-reference if rows else None


def support_at(science, columns):
    """Span determinant support at the last prefix with at most ``columns`` columns."""
    return next(support for row, support in zip(reversed(science['trajectory']), reversed(science['support']))
                if row['dimension'] <= columns)[0]


def hit_columns(science, reference, threshold):
    i = first_hit(science, reference, threshold)
    return math.inf if i is None else science['trajectory'][i]['dimension']


def compactness(a, b, reference, thresholds, frontiers, ratio):
    """Whether arm a is more column-compact than arm b under the declared rule."""
    reached = [t for t in thresholds if math.isfinite(hit_columns(a, reference, t)) or math.isfinite(hit_columns(b, reference, t))]
    columns = all(hit_columns(a, reference, t) <= ratio*hit_columns(b, reference, t) for t in reached)
    errors = all(frontier_error(a, reference, k) <= frontier_error(b, reference, k) for k in frontiers)
    return columns and errors


def early_readout(packets, early, reference, thresholds, frontiers, d):
    """Readout 3 under the declared band, with signed differences and the clause that fails."""
    reached = [t for t in thresholds if math.isfinite(hit_columns(packets, reference, t))]
    slow = [t for t in reached if hit_columns(early, reference, t) > d['early_column_ratio']*hit_columns(packets, reference, t)]
    gaps = [(k, frontier_error(early, reference, k), frontier_error(packets, reference, k)) for k in frontiers]
    outside = [k for k, e, x in gaps if abs(e-x) > d['early_relative_frontier_error']*abs(x)]
    if not slow and not outside: return '**preserves accuracy**.'
    rounding = [k for k, e, x in gaps if max(abs(e), abs(x)) <= d['complete_span_tolerance']]
    failed = ([f'columns above {d["early_column_ratio"]}× exhaustive at ≤{", ≤".join(map(str, slow))}'] if slow else []) + (
        [f'frontier error outside {d["early_relative_frontier_error"]:.0%} of exhaustive at {", ".join(map(str, outside))}'] if outside else [])
    return (f'**outside the declared band** ({"; ".join(failed)}). Early minus exhaustive error (negative: early lower) '
            + ', '.join(f'{e-x:+.4g} at {k}' for k, e, x in gaps) + '.'
            + (f' Both errors are within {d["complete_span_tolerance"]:g} of the reference at {", ".join(map(str, rounding))}.' if rounding else ''))


def davidson_note(davidson, reference, overlap, d):
    """Factual note when Davidson ends above the reference without the declared confinement trigger."""
    last = davidson['trajectory'][-1]; error = last['energy']-reference
    if davidson['stop_reason'] == 'residual_tolerance' or error <= d['davidson_uninformative_if_converged_above_reference_by']: return ''
    return (f' It stopped by {davidson["stop_reason"]} at residual {last["residual_norm"]:.3g}, {error:.6g} above the reference with '
            f'ground overlap {overlap:.2g}' + (', so its span excludes the ground state' if overlap <= 1e-6 else '')
            + '. The declared confinement rule names residual-tolerance stops only, so the verdict above follows the declared fallback.')


def fmt(value, digits=6):
    if value is None: return '—'
    if isinstance(value, float) and math.isinf(value): return 'not reached'
    return f'{value:.{digits}g}' if isinstance(value, float) else str(value)


def render(record):
    c = record['config']; d = c['decisions']; workers = record['workers']
    sciences = {k: decode_science(v) for k, v in record['scientific_payloads'].items()}
    good = [w for w in workers if w['status'] == 'ok']
    complete = len(good) == len(workers)
    thresholds = c['thresholds']
    lines = ['# Column-budget convergence', '',
        'Exact classical sector-vector comparison of residual packets, greedy spin-adapted selected CI and a singlet-projected '
        'Davidson reference, each adding one column per step from the same three-column CAS(2e,2o) singlet span.',
        f'{len(good)}/{len(workers)} scheduled cells retained; {len(sciences)} content-addressed scientific payloads. '
        'Errors are above the lowest singlet-frame eigenvalue, in t for lattices and Ha for molecules.', '',
        'Times are medians of setup plus solver across repetitions and are descriptive. Worker RSS is the OS high-water mark. '
        'Span support counts determinants with any retained amplitude above 1e-12. On the molecular inputs, packet and '
        'Davidson directions carry rounding-level amplitudes near that threshold: their counts reproduce exactly but can differ '
        'by a few percent from an independent computation of the same span, which brackets each between its own counts at '
        '1e-11 and 1e-13. Selected-CI columns are exact CSFs and carry none.', '']
    verdicts = {}
    for case in c['inputs']:
        name = case['name']; ref = record['references'][name]; reference = ref['singlet_reference']
        budget = case['maximum_columns']
        frontiers = [k for k in c['frontiers'] if k < budget] + [budget]
        group = {arm['name']: [w for w in good if w['input'] == name and w['arm'] == arm['name']] for arm in c['arms']}
        science = {arm: sciences[ws[0]['scientific_sha256']] if ws else None for arm, ws in group.items()}
        lines += [f'## {name} ({case["role"]}, {case["energy_unit"]})', '',
            f'{case["sector_dimension"]} determinants, {case["singlet_dimension"]} singlet columns, budget {budget}. '
            f'Singlet reference {reference:.10g}; sector ground state {"is" if ref["sector_ground_is_singlet"] else "is not"} a singlet.', '',
            '| Arm | Completed | Stop | Columns | Final error | Ground overlap | H actions | Support | Time (s) | Worker RSS (MiB) |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm, ws in group.items():
            s = science[arm]
            if s is None:
                lines.append(f'| {arm} | 0/{case["repeats"]} | — | — | — | — | — | — | — | — |'); continue
            last = s['trajectory'][-1]
            overlap = record['final_ground_state_overlaps'][ws[0]['scientific_sha256']]
            time = statistics.median(w['setup_seconds']+w['solver_seconds'] for w in ws)
            rss = statistics.median(w['worker_peak_rss_bytes'] for w in ws)/2**20
            lines.append(f'| {arm} | {len(ws)}/{case["repeats"]} | {s["stop_reason"]} | {last["dimension"]} | '
                f'{last["energy"]-reference:.6g} | {overlap:.4f} | {s["final_h_action_calls"]} | {s["support"][-1][0]} | {time:.2f} | {rss:.1f} |')
        lines += ['', '| Arm | '+' | '.join(f'Error at {k}' for k in frontiers)+' |', '|---|'+'---:|'*len(frontiers)]
        for arm, s in science.items():
            if s is None: continue
            lines.append(f'| {arm} | '+' | '.join(fmt(frontier_error(s, reference, k)) for k in frontiers)+' |')
        lines += ['', '| Arm | '+' | '.join(f'≤{t}: K / W / s' for t in thresholds)+' |', '|---|'+'---|'*len(thresholds)]
        for arm, s in science.items():
            if s is None: continue
            cells = []
            for t in thresholds:
                i = first_hit(s, reference, t)
                if i is None: cells.append('not reached'); continue
                row = s['trajectory'][i]
                seconds = statistics.median(w['setup_seconds']+w['prefix_wall_seconds'][i] for w in group[arm])
                cells.append(f'{row["dimension"]} / {row["h_action_calls"]} / {seconds:.2f}')
            lines.append(f'| {arm} | '+' | '.join(cells)+' |')
        if 'control' in case['role']:
            lines.append('')
            for arm, s in science.items():
                if s is None: continue
                last = s['trajectory'][-1]
                full = last['dimension'] == case['singlet_dimension']
                lines.append(f'Complete-span control, {arm}: ' + (
                    f'{"PASS" if abs(last["energy"]-reference) <= d["complete_span_tolerance"] else "FAIL"} at the full span.'
                    if full else f'stopped at {last["dimension"]} columns ({s["stop_reason"]}, final error {last["energy"]-reference:.3g}).'))
        packets, sci, early, davidson = (science.get(k) for k in ('packets_exhaustive', 'selected_ci', 'packets_early', 'davidson'))
        lines += ['']
        if packets and sci:
            limited = [k for k in frontiers if k >= d['frontier_minimum_columns']]
            p_more = compactness(packets, sci, reference, thresholds, limited, d['column_compactness_ratio'])
            s_more = compactness(sci, packets, reference, thresholds, limited, d['column_compactness_ratio'])
            verdict = 'mixed' if p_more == s_more else 'packets more compact' if p_more else 'selected CI more compact'
            verdicts[case['role']] = verdict
            lines.append(f'Readout 1, column compactness (packets_exhaustive vs selected_ci): **{verdict}**.')
            if verdict == 'mixed':
                lines.append('Readout 2, work: not applicable; columns favour neither arm.')
            else:
                winner, other = (packets, sci) if p_more else (sci, packets)
                compared = [t for t in thresholds if math.isfinite(hit_columns(winner, reference, t)) and math.isfinite(hit_columns(other, reference, t))]
                costly = [t for t in compared if winner['trajectory'][first_hit(winner, reference, t)]['h_action_calls']
                          >= d['work_ratio']*other['trajectory'][first_hit(other, reference, t)]['h_action_calls']]
                lines.append('Readout 2, work: ' + ('no threshold reached by both arms to compare.' if not compared else
                    f'compression at higher classical work (≥{d["work_ratio"]}× H actions at ≤{", ≤".join(map(str, costly))}).' if costly else
                    f'no H-action penalty at the {len(compared)} compared threshold(s).'))
        if packets and early:
            lines.append('Readout 3, early screening: ' + early_readout(packets, early, reference, thresholds, frontiers, d))
        if davidson:
            last = davidson['trajectory'][-1]
            overlap = record['final_ground_state_overlaps'][group['davidson'][0]['scientific_sha256']]
            confined = davidson['stop_reason'] == 'residual_tolerance' and last['energy']-reference > d['davidson_uninformative_if_converged_above_reference_by']
            if confined:
                lines.append(f'Readout 4, Davidson: **uninformative (confined)**; it converged {last["energy"]-reference:.6g} above the '
                             f'reference with ground overlap {overlap:.4f}.')
            else:
                fewer = packets and sci and all(hit_columns(davidson, reference, t) < min(hit_columns(packets, reference, t), hit_columns(sci, reference, t))
                                                for t in thresholds)
                lines.append('Readout 4, Davidson: ' + ('reaches every threshold in fewer columns than both restricted arms, so the packets-versus-selected-CI '
                    'comparison concerns bounded-support compactness only.' if fewer else 'does not reach every threshold in fewer columns than both restricted arms.')
                    + davidson_note(davidson, reference, overlap, d))
            matched = davidson['trajectory'][-1]['dimension']
            lines.append(f'Span support at {matched} columns, Davidson\'s last prefix: '
                         + ', '.join(f'{arm} {support_at(s, matched)}' for arm, s in science.items() if s is not None) + ' determinants.')
        lines += ['']
    decisive = all(verdicts.get(role) == 'packets more compact' for role in d['decisive_roles'])
    lines += [f'All scheduled cells retained: {"PASS" if complete else "FAIL"}.',
        f'Packets add column compactness (primary and both held-out inputs): **{"YES" if decisive and complete else "NO"}**.', '',
        'Repeated deterministic trajectories count once. Two or three timing repetitions are descriptive. No quantum, hardware, '
        'shot-cost, optimized-selected-CI or asymptotic-scaling claim follows, and no solver default changes.', '',
        'Revised after observation: readout 3 names the failing clause and the sign of each difference instead of calling every '
        'miss a loss; readout 4 adds a factual note when Davidson ends above the reference without the declared trigger, and the '
        'declared span support at matched columns, which the executed summarizer omitted; and the support caveat above. No verdict '
        'rule changed.', '',
        f'Declaration: `{record["declaration_commit"]}`. Execution: `{record["source_commit"]}`.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true'); parser.add_argument('--record', type=Path, default=ROOT/RECORD)
    args = parser.parse_args(argv); text = render(read_json(args.record))
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text: raise SystemExit('budget convergence report is stale')
    else: REPORT.write_text(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
