"""Render only observed packet-reuse results; no molecular calculations."""
import argparse
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.metadata_io import read_json
from benchmarks.run_packet_work_reuse import ARMS, RECORD

REPORT = ROOT / 'benchmarks/results/packet_work_reuse.md'


def first_target(worker):
    return next((r for r in worker['result']['trajectory'] if r['target_reached']), None) if worker['status'] == 'ok' else None


def render(record):
    repeats, workers = record['config']['repeats'], record['workers']
    lines = ['# Residual packet action reuse and screening', '',
        'Exploratory exact classical H2O STO-3G CAS(8e,6o): 225 determinants, 105 singlet-frame columns.',
        'Every arm starts from the same singlet CAS(2e,2o) on [3,4]. Target: 1.6 mHa on this fixed Hamiltonian.',
        f"Fresh processes: {len(workers)}; failed/timed out: {sum(w['status'] != 'ok' for w in workers)}.", '',
        '## First target prefix', '',
        'Medians over successful target-reaching runs. Denominators include failures. Time includes symmetry/model setup and cumulative solver work; imports/startup are excluded here and included in the terminal process time below.', '',
        '| Arm | Target runs | Columns | H actions | Exact scores | Time incl. setup (s) | Peak RSS (MiB) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        hits = [(w, first_target(w)) for w in workers if w['arm'] == arm and first_target(w) is not None]
        if not hits:
            lines.append(f'| {arm} | 0/{repeats} | — | — | — | — | — |')
            continue
        values = [f'{median([r[k] for _,r in hits]):g}' for k in ('dimension','h_action_calls','candidates_scored')]
        elapsed = median([w['result']['setup']['wall_seconds'] + r['wall_seconds'] for w,r in hits])
        rss = median([r['peak_rss_bytes']/2**20 for _,r in hits])
        lines.append(f'| {arm} | {len(hits)}/{repeats} | ' + ' | '.join(values) + f' | {elapsed:.6f} | {rss:.3f} |')
    lines += ['', '## Partial array payloads at first target', '',
        'These are partial array payloads, not total resident memory. The SCI candidate counter includes borrowed frame-column views plus owned H arrays; its views already belong to the common frame and are not additional owned RAM. Setup additionally owns dense S², the full singlet frame and diagonal; Python containers, operator storage and temporary arrays are excluded. RSS above is the whole process high-water mark.', '',
        '| Arm | V/HV (KiB) | Extra frame snapshot (KiB) | Cached frame actions (KiB) | SCI candidate cache (KiB) |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        hits = [first_target(w) for w in workers if w['arm'] == arm and first_target(w) is not None]
        values = [f"{median([r[k] for r in hits])/1024:.3f}" if hits else '—' for k in
            ('retained_payload_bytes','frame_snapshot_payload_bytes','frame_actions_payload_bytes','candidate_cache_payload_bytes')]
        lines.append(f'| {arm} | ' + ' | '.join(values) + ' |')
    lines += ['', '## Full-run audits and terminal costs', '',
        'Audit losses concern exact candidate energy-lowering scores. Audits select the full-screen winner. Non-audit steps with negligible shortlisted gain fall back to full screening; both costs are charged.', '',
        '| Arm | Completed runs | Audits | Missed winners | Largest audit loss (Ha) | Fallbacks | Final H actions | Median process time (s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        group = [w for w in workers if w['arm'] == arm and w['status'] == 'ok']
        decisions = [d for w in group for d in w['result']['decisions']]
        audits = [d for d in decisions if d['audit']]
        calls = f"{median([w['result']['final_h_action_calls'] for w in group]):g}" if group else '—'
        elapsed = f"{median([w['process_wall_seconds'] for w in group]):.6f}" if group else '—'
        lines.append(f"| {arm} | {len(group)}/{repeats} | {len(audits)} | {sum(d['audit_missed_winner'] for d in audits)} | {max([d['audit_score_loss'] for d in audits], default=0):.9g} | {sum(d['fallback'] for d in decisions)} | {calls} | {elapsed} |")
    lines += ['', '## Trajectory equivalence', '',
        'Original versus reuse-only (same repeat); compare selected group identities and entire energy/residual trajectories, including later prefixes. A mismatch must be reported, not hidden by target-only agreement.', '']
    for repeat in range(repeats):
        pair = {w['arm']: w for w in workers if w['repeat'] == repeat and w['arm'] in ('original','reuse')}
        if len(pair) != 2 or any(w['status'] != 'ok' for w in pair.values()):
            lines.append(f'- Repeat {repeat}: unavailable (failed/missing arm).')
            continue
        a,b = [pair[k]['result']['trajectory'] for k in ('original','reuse')]
        same = len(a) == len(b) and all(x['selection'] == y['selection'] for x,y in zip(a,b))
        energy = max((abs(x['energy']-y['energy']) for x,y in zip(a,b)), default=0)
        residual = max((abs(x['residual_norm']-y['residual_norm']) for x,y in zip(a,b)), default=0)
        lines.append(f'- Repeat {repeat}: same prefix count/selections = {same}; maximum paired energy difference {energy:.3g} Ha; residual-norm difference {residual:.3g} Ha.')
    for w in workers:
        if w['status'] != 'ok':
            lines += ['', f"Failure: {w['arm']}/{w['repeat']}: {w['status']}", '```text', w['diagnostic'], '```']
    lines += ['', '## Scope', '',
        'The cache retains all populated HF columns; its 105-action bound trades repeated operator applications for storage approaching a complete singlet action table. No full H or H² is built by the timed solver, but HF itself is a dense sector-by-frame table when populated.',
        'Shortlisting uses a fixed 80% residual-mass prefix and every-fifth-decision full audits. It is heuristic and does not inherit finite-element convergence guarantees. Full residual stopping is independent of candidate screening.',
        'Reference energies are evaluated after all timed workers. Independent occupation-H replay verifies packet spans, decisions, audits, cache population, costs and spin. Three-repeat tiny-sector timings are descriptive; this study establishes no general scaling, quantum shots/preparation, hardware, optimized SHCI/DMRG superiority or quantum advantage.', '',
        f"Declaration: `{record['declaration_commit']}`.", f"Execution: `{record['source_commit']}`.", '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    text = render(read_json(ROOT / RECORD))
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text:
            print('FAIL packet report differs from observations')
            return 1
        print('OK packet report matches observations')
    else:
        REPORT.write_text(text)
        print(f'Wrote {REPORT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
