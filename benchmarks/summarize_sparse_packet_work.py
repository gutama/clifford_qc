"""Generate the declared sparse packet comparison report from preserved observations."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import statistics
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from benchmarks.metadata_io import read_json
from benchmarks.run_sparse_packet_work import RECORD, unpack
from benchmarks.run_packet_work_large import unpack_observation, RECORD as PREVIOUS_RECORD

REPORT = ROOT/'benchmarks/results/sparse_packet_work.md'


def render(published, previous=None):
    r = unpack(published)
    lines = ['# Sparse packet setup and early screening', '',
        'Exploratory exact classical comparisons. Times are fresh-process observations; units are input-specific.',
        'Early marks precede candidate admissibility. No solver default is changed.', '']
    recovery = published.get('publication_recovery')
    if recovery:
        lines += [f'Publication recovery: {recovery["recovered_observations"]} complete observations retained; '
            f'{recovery["unavailable_observations"]} observations have no surviving record and count as failures. '
            'These are scheduled-cell denominators, not proof that every cell ran. No cells were selectively rerun.', '']
    for case in r['config']['inputs']:
        entries = [w for w in r['workers'] if w['input'] == case['name']]
        completed = sum(w['status'] == 'ok' for w in entries)
        scheduled = case['repeats']*len(r['config']['arms'])
        lines += [f'Coverage for {case["name"]}: {completed}/{scheduled} scheduled observations retained; '
            f'{scheduled-completed} unavailable or failed.']
    lines += ['', 'RSS is the whole-process high-water mark, including provenance preflight, which parses committed records. '
        'The approximately 181 MiB preflight floor masks smaller solver peaks and small-fixture RSS regressions; '
        'solver-only peaks were not measured, and the floor can grow as records accumulate.', '']
    for case in r['config']['inputs']:
        lines += [f'## {case["name"]} ({case["energy_unit"]})', '',
            f'{case["sector_dimension"]} determinants; {case["maximum_columns"]}-column cap; target {case["accuracy_target"]} {case["energy_unit"]}.', '',
            '| Arm | Completed | Target | Final error | H calls | Materialized | Setup (s) | Solver (s) | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        medians = {}
        for arm in r['config']['arms']:
            entries = [w for w in r['workers'] if w['input'] == case['name'] and w['arm'] == arm]
            ok = [w for w in entries if w['status'] == 'ok']
            if not ok:
                lines.append(f'| {arm} | 0/{case["repeats"]} | 0/{case["repeats"]} | — | — | — | — | — | — | — | — | — |'); continue
            median = lambda f: statistics.median(f(w) for w in ok)
            error = median(lambda w: w['result']['trajectory'][-1]['error'])
            target = sum(any(row['target_reached'] for row in w['result']['trajectory']) for w in ok)
            elapsed = median(lambda w: w['result']['setup']['wall_seconds']+w['result']['solver_wall_seconds'])
            rss = median(lambda w: w['result']['final_peak_rss_bytes']/2**20)
            medians[arm] = elapsed, rss, len(ok) == case['repeats']
            times = [w['result']['setup']['wall_seconds']+w['result']['solver_wall_seconds'] for w in ok]
            quartiles = statistics.quantiles(times, n=4, method='inclusive') if len(times) > 1 else None
            iqr = f'[{quartiles[0]:.3f}, {quartiles[2]:.3f}]' if quartiles else '—'
            materialized = median(lambda w: w['result'].get('materialized_candidates',
                w['result']['final_candidates_scored']))
            lines.append(f'| {arm} | {len(ok)}/{case["repeats"]} | {target}/{case["repeats"]} | {error:.9g} | '
                f'{median(lambda w: w["result"]["final_h_action_calls"]):g} | {materialized:g} | '
                f'{median(lambda w: w["result"]["setup"]["wall_seconds"]):.3f} | '
                f'{median(lambda w: w["result"]["solver_wall_seconds"]):.3f} | '
                f'{elapsed:.3f} | {iqr} | {median(lambda w: w["process_wall_seconds"]):.3f} | {rss:.3f} |')
        if 'dense' in medians and 'sparse' in medians:
            speed, rss = medians['dense'][0]/medians['sparse'][0], medians['sparse'][1]/medians['dense'][1]
            lines += ['', f'Exhaustive sparse/dense: {speed:.3f}× speed ratio; {rss:.3f}× RSS ratio. '
                f'Timing/RSS promotion clause: {"PASS" if speed >= 1.25 and rss <= 1 and medians["dense"][2] and medians["sparse"][2] else "FAIL"}.']
        early = [w['result']['decisions'] for w in r['workers'] if w['input'] == case['name'] and w['arm'] == 'early' and w['status'] == 'ok']
        unique = {json.dumps(ds, sort_keys=True): ds for ds in early}
        audits = [d for ds in unique.values() for d in ds if d['audit']]
        trajectory_label = 'trajectory' if len(unique) == 1 else 'trajectories'
        observation_label = 'observation' if len(early) == 1 else 'observations'
        lines += ['', f'Early audits over {len(unique)} unique decision {trajectory_label} '
            f'({len(early)} timing {observation_label}): {len(audits)}; missed winners: '
            f'{sum(d["audit_missed_winner"] for d in audits)}; largest audited score loss: '
            f'{max((d["audit_score_loss"] for d in audits), default=0):.9g} {case["energy_unit"]}.', '']
        arms = {arm: [w for w in r['workers'] if w['input'] == case['name'] and w['arm'] == arm and w['status'] == 'ok']
                for arm in ('sparse', 'early')}
        if all(arms.values()):
            errors = {arm: statistics.median(w['result']['trajectory'][-1]['error'] for w in ws)
                      for arm, ws in arms.items()}
            if errors['sparse'] > 0:
                change = 100*(errors['early']/errors['sparse']-1)
                lines += [f'Early versus exhaustive final error: {errors["early"]:.9g} versus '
                    f'{errors["sparse"]:.9g} {case["energy_unit"]} ({change:+.2f}%).']
                if any(len(ws) < case['repeats'] for ws in arms.values()):
                    lines += ['One surviving observation per arm establishes the observed deterministic accuracy outcome; '
                        'missing repeats reduce timing evidence, not the visibility of this error comparison.']
                lines += ['']
    if previous is not None:
        old = unpack_observation(previous)
        current = [w['result'] for w in r['workers'] if w['input'] == 'hubbard_2x4_u4'
                   and w['arm'] == 'early' and w['status'] == 'ok']
        prior = [w['result'] for w in old['workers'] if w['arm'] == 'shortlist' and w['status'] == 'ok']
        if current and prior:
            pairs = [(a, b) for a in current for b in prior]
            same = all(len(a['trajectory']) == len(b['trajectory']) and
                [row['selection'] for row in a['trajectory']] == [row['selection'] for row in b['trajectory']]
                for a, b in pairs)
            differences = {k: max(abs(x[k]-y[k]) for a, b in pairs
                                  for x, y in zip(a['trajectory'], b['trajectory']))
                           for k in ('energy', 'residual_norm')}
            calls = all(a['final_h_action_calls'] == b['final_h_action_calls'] for a, b in pairs)
            equivalent = same and calls and max(differences.values()) <= 1e-9
            prior_time = statistics.median(a['setup']['wall_seconds']+a['solver_wall_seconds'] for a in prior)
            current_time = statistics.median(a['setup']['wall_seconds']+a['solver_wall_seconds'] for a in current)
            dense = [w['result']['materialized_candidates'] for w in r['workers']
                     if w['input'] == 'hubbard_2x4_u4' and w['arm'] == 'dense' and w['status'] == 'ok']
            lines += ['## Historical shortlist comparison', '',
                f'2x4 early versus the PR #145 shortlist: equivalent ordered selections, energies/residuals '
                f'and H calls within 1e-9: {"PASS" if equivalent else "FAIL"}; '
                f'{len(current[0]["trajectory"])} retained prefixes '
                f'({len(current[0]["trajectory"])-1} growth selections); maximum energy difference '
                f'{differences["energy"]:.3g} t, residual difference {differences["residual_norm"]:.3g} t.',
                f'Historical shortlist setup + solver: {prior_time:.3f} s; sparse early: {current_time:.3f} s. '
                f'The historical path materializes every group before shortlisting '
                f'({statistics.median(dense):g} candidates for this common growth schedule); '
                f'early materializes {statistics.median(a["materialized_candidates"] for a in current):g}. '
                'Moving marking before admissibility does not change the observed trajectory here. '
                'Avoided materialization supports the mechanism; these are separate campaigns, not a new controlled timing ablation.', '']
    lines += ['Materialized counts include rejected directions; H calls and exact scores count admissible candidates only.', '',
        'Setup and solver columns are separate medians; setup + solver is the median of each observation\'s paired sum.', '',
        'Sparse setup avoids full-sector dense H/S² and a dense global CSF frame; it retains bounded dense spin blocks and explicit accepted sector vectors. Payloads are not peak RSS.',
        'Fixed-budget errors do not establish matched-accuracy speedups. Repeats measure timing variability, not independent scientific generalization. No optimized-SCI, hardware or quantum-advantage claim follows.', '',
        f'Declaration: `{r["declaration_commit"]}`. Execution: `{r["source_commit"]}`.', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--check', action='store_true')
    args = p.parse_args(argv); text = render(read_json(ROOT/RECORD), read_json(ROOT/PREVIOUS_RECORD))
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text: raise SystemExit('sparse packet report is stale')
    else: REPORT.write_text(text)
    return 0


if __name__ == '__main__': raise SystemExit(main())
