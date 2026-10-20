"""Generate the declared sparse packet comparison report from preserved observations."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import statistics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from benchmarks.metadata_io import read_json
from benchmarks.run_sparse_packet_work import RECORD, unpack

REPORT = ROOT/'benchmarks/results/sparse_packet_work.md'


def render(published):
    r = unpack(published)
    lines = ['# Sparse packet setup and early screening', '',
        'Exploratory exact classical comparisons. Times are fresh-process observations; units are input-specific.',
        'Early marks precede candidate admissibility. No solver default is changed.', '']
    recovery = published.get('publication_recovery')
    if recovery:
        lines += [f'Publication recovery: {recovery["recovered_observations"]} complete observations retained; '
            f'{recovery["unavailable_observations"]} secondary-input observations were unavailable and count as failures. '
            'All H2O and all seven 2x4 Hubbard repetitions survived. No cells were selectively rerun.', '']
    for case in r['config']['inputs']:
        lines += [f'## {case["name"]} ({case["energy_unit"]})', '',
            f'{case["sector_dimension"]} determinants; {case["maximum_columns"]}-column cap; target {case["accuracy_target"]} {case["energy_unit"]}.', '',
            '| Arm | Completed | Target | Final error | H calls | Materialized | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        medians = {}
        for arm in r['config']['arms']:
            entries = [w for w in r['workers'] if w['input'] == case['name'] and w['arm'] == arm]
            ok = [w for w in entries if w['status'] == 'ok']
            if not ok:
                lines.append(f'| {arm} | 0/{len(entries)} | 0/{len(entries)} | — | — | — | — | — | — | — |'); continue
            median = lambda f: statistics.median(f(w) for w in ok)
            error = median(lambda w: w['result']['trajectory'][-1]['error'])
            target = sum(any(row['target_reached'] for row in w['result']['trajectory']) for w in ok)
            elapsed = median(lambda w: w['result']['setup']['wall_seconds']+w['result']['solver_wall_seconds'])
            rss = median(lambda w: w['result']['final_peak_rss_bytes']/2**20)
            medians[arm] = elapsed, rss, len(ok) == len(entries)
            times = [w['result']['setup']['wall_seconds']+w['result']['solver_wall_seconds'] for w in ok]
            quartiles = statistics.quantiles(times, n=4, method='inclusive') if len(times) > 1 else None
            iqr = f'[{quartiles[0]:.3f}, {quartiles[2]:.3f}]' if quartiles else '—'
            materialized = median(lambda w: w['result'].get('materialized_candidates',
                w['result']['final_candidates_scored']))
            lines.append(f'| {arm} | {len(ok)}/{len(entries)} | {target}/{len(entries)} | {error:.9g} | '
                f'{median(lambda w: w["result"]["final_h_action_calls"]):g} | {materialized:g} | '
                f'{elapsed:.3f} | {iqr} | {median(lambda w: w["process_wall_seconds"]):.3f} | {rss:.3f} |')
        if 'dense' in medians and 'sparse' in medians:
            speed, rss = medians['dense'][0]/medians['sparse'][0], medians['sparse'][1]/medians['dense'][1]
            lines += ['', f'Exhaustive sparse/dense: {speed:.3f}× speed ratio; {rss:.3f}× RSS ratio. '
                f'Timing/RSS promotion clause: {"PASS" if speed >= 1.25 and rss <= 1 and medians["dense"][2] and medians["sparse"][2] else "FAIL"}.']
        early = [w['result']['decisions'] for w in r['workers'] if w['input'] == case['name'] and w['arm'] == 'early' and w['status'] == 'ok']
        audits = [d for ds in early for d in ds if d['audit']]
        lines += ['', f'Early audits across repeats: {len(audits)}; missed winners: '
            f'{sum(d["audit_missed_winner"] for d in audits)}; largest audited score loss: '
            f'{max((d["audit_score_loss"] for d in audits), default=0):.9g} {case["energy_unit"]}.', '']
    lines += ['Materialized counts include rejected directions; H calls and exact scores count admissible candidates only.', '',
        'Sparse setup avoids full-sector dense H/S² and a dense global CSF frame; it retains bounded dense spin blocks and explicit accepted sector vectors. Payloads are not peak RSS.',
        'Fixed-budget errors do not establish matched-accuracy speedups. Repeats measure timing variability, not independent scientific generalization. No optimized-SCI, hardware or quantum-advantage claim follows.', '',
        f'Declaration: `{r["declaration_commit"]}`. Execution: `{r["source_commit"]}`.', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--check', action='store_true')
    args = p.parse_args(argv); text = render(read_json(ROOT/RECORD))
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text: raise SystemExit('sparse packet report is stale')
    else: REPORT.write_text(text)
    return 0


if __name__ == '__main__': raise SystemExit(main())
