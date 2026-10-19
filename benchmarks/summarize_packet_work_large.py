"""Observation-driven larger-Hubbard packet work report."""
import argparse
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.metadata_io import read_json
from benchmarks.run_packet_work_large import ARMS,RECORD,unpack_observation
REPORT=ROOT/'benchmarks/results/packet_work_large.md'


def render(record):
    record=unpack_observation(record)
    c=record['config'];workers=record['workers'];repeats=c['repeats']
    good=lambda arm:[w for w in workers if w['arm']==arm and w['status']=='ok']
    lines=['# Larger-sector residual packet work comparison','',
        'Open 2x4 Hubbard, t=1, U=4, mu=0, half filling: 16 qubits, 4,900 determinants and 1,764 singlet-frame columns.',
        'The sector is 21.8 times the H2O pilot size. Every energy/error/residual below uses hopping units t, not Hartree.',
        'Same classical CAS(2e,2o) starting protocol on sites [3,4]; this is not the lattice builder\'s Neel reference.',
        f"Budget: {c['maximum_columns']} total retained columns; target: {c['accuracy_target']:g} t. Fresh processes: {len(workers)}; failed/timed out: {sum(w['status']!='ok' for w in workers)}.",'',
        '## Target attainment','',
        'First-target costs are shown only when the fixed accuracy target is reached. Denominators include failures.','',
        '| Arm | Target runs | Columns | H actions | Time including setup (s) |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        hits=[]
        for w in good(arm):
            hit=next((r for r in w['result']['trajectory'] if r['target_reached']),None)
            if hit is not None:hits.append((w,hit))
        if not hits:
            lines.append(f'| {arm} | 0/{repeats} | — | — | — |')
        else:
            columns=median([r['dimension'] for w,r in hits]);calls=median([r['h_action_calls'] for w,r in hits])
            elapsed=median([w['result']['setup']['wall_seconds']+r['wall_seconds'] for w,r in hits])
            lines.append(f'| {arm} | {len(hits)}/{repeats} | {columns:g} | {calls:g} | {elapsed:.6f} |')
    lines+=['','## Final retained prefix','',
        'Medians over completed runs; this is a budget comparison when target attainment above is zero.','',
        '| Arm | Completed | Columns | Error (t) | Full residual (t) | H calls | Exact scores | Stop reasons |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        group=good(arm);rows=[w['result']['trajectory'][-1] for w in group]
        if not rows:
            lines.append(f'| {arm} | 0/{repeats} | — | — | — | — | — | failed/timeout |');continue
        values=[f'{median([r[k] for r in rows]):.9g}' for k in ('dimension','energy_error','residual_norm','h_action_calls','candidates_scored')]
        reasons=', '.join(sorted({w['result']['stop_reason'] for w in group}))
        lines.append(f'| {arm} | {len(rows)}/{repeats} | '+' | '.join(values)+f' | {reasons} |')
    lines+=['','## Fixed-column error frontiers','',
        'Median error (t) at the largest observed prefix within each cap; no interpolation.','',
        '| Column cap | Original | Reuse | Shortlist | Combined | Selected CI |','|---|---:|---:|---:|---:|---:|']
    for cap in (8,16,32,64):
        values=[]
        for arm in ARMS:
            rows=[r for w in good(arm) if (r:=next((r for r in reversed(w['result']['trajectory']) if r['dimension']<=cap),None)) is not None]
            values.append(f"{median([r['energy_error'] for r in rows]):.9g}" if rows else '—')
        lines.append(f'| {cap} | '+' | '.join(values)+' |')
    lines+=['','## Time and memory','',
        'Time-to-final includes setup plus solver; whole-process time also includes imports/startup. RSS is OS process high-water memory. V/HV and cache frame/actions are owned arrays; selected-CI direction columns are borrowed views of the common frame and are excluded from the owned-H-array column.','',
        '| Arm | Setup (s) | Setup + solver (s) | Whole process (s) | Peak RSS (MiB) | V/HV (MiB) | Extra cache frame/actions (MiB) | SCI owned H arrays (MiB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        group=good(arm)
        if not group:
            lines.append(f'| {arm} | — | — | — | — | — | — | — |');continue
        setup=median([w['result']['setup']['wall_seconds'] for w in group])
        elapsed=median([w['result']['setup']['wall_seconds']+w['result']['solver_wall_seconds'] for w in group])
        process=median([w['process_wall_seconds'] for w in group])
        rss=median([w['result']['final_peak_rss_bytes']/2**20 for w in group])
        ends=[w['result']['trajectory'][-1] for w in group]
        vhv=median([r['retained_payload_bytes']/2**20 for r in ends])
        frame=median([(r['frame_snapshot_payload_bytes']+r['frame_actions_payload_bytes'])/2**20 for r in ends])
        sci=median([r['candidate_cache_payload_bytes']/2/2**20 for r in ends])
        lines.append(f'| {arm} | {setup:.3f} | {elapsed:.3f} | {process:.3f} | {rss:.3f} | {vhv:.3f} | {frame:.3f} | {sci:.3f} |')
    lines+=['','## Audits and trajectory equivalence','',
        '| Arm | Full-screen audits | Missed winners | Largest score loss (t) | Fallbacks |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        decisions=[d for w in good(arm) for d in w['result']['decisions']];audits=[d for d in decisions if d['audit']]
        lines.append(f"| {arm} | {len(audits)} | {sum(d['audit_missed_winner'] for d in audits)} | {max([d['audit_score_loss'] for d in audits],default=0):.9g} | {sum(d['fallback'] for d in decisions)} |")
    lines.append('')
    for a,b in (('original','reuse'),('shortlist','combined')):
        for repeat in range(repeats):
            pair={w['arm']:w for w in workers if w['repeat']==repeat and w['arm'] in (a,b) and w['status']=='ok'}
            if len(pair)!=2:
                lines.append(f'- {a}/{b}, repeat {repeat}: unavailable (failed/missing arm).');continue
            x,y=[pair[k]['result']['trajectory'] for k in (a,b)]
            same=len(x)==len(y) and all(p['selection']==q['selection'] for p,q in zip(x,y))
            energy=max((abs(p['energy']-q['energy']) for p,q in zip(x,y)),default=0)
            residual=max((abs(p['residual_norm']-q['residual_norm']) for p,q in zip(x,y)),default=0)
            lines.append(f'- {a}/{b}, repeat {repeat}: same prefix count/selections = {same}; maximum paired energy difference {energy:.3g} t, residual-norm difference {residual:.3g} t.')
    for w in workers:
        if w['status']!='ok':lines+=['',f"Failure: {w['arm']}/{w['repeat']}: {w['status']}",'```text',w['diagnostic'],'```']
    any_good=next((w for w in workers if w['status']=='ok'),None)
    if any_good:
        s=any_good['result']['setup']
        lines+=['',f"Common dense S²/frame/diagonal arrays: {s['array_payload_bytes']/2**20:.3f} MiB. The temporary full-H payload during CSR compilation is {s['dense_h_temporary_bytes']/2**20:.3f} MiB; retained CSR H is {s['csr_payload_bytes']/2**20:.3f} MiB ({s['csr_nnz']} nonzeros). These counters are not summed into a peak-RSS claim."]
    lines+=['','## Reference and scope','',
        f"Independent occupation-sign reference: {record['reference']['energy']:.12f} t. Two shifted sparse solves agree; full residual and singlet checks are retained in the record.",
        'This tests one larger correlated lattice Hamiltonian at a fixed budget. It does not establish general scaling or larger molecular performance. Native CSR compilation temporarily builds full sector H, and dense spin/frame setup remains. Cache savings in H calls can coexist with worse runtime or memory. The shortlist is heuristic; all audits and fallback costs are charged. Earlier H2O evidence is unchanged.',
        'The published gzip/base64 JSON envelope preserves the complete observations. Its digest and the uncompressed observation commit are bound separately; native producer JSON bytes are exactly reconstructible.',
        'The independent checker rebuilds Hubbard signs, symmetry, trajectory decisions and cost counts. Exact repeated scientific trajectories may reuse a full replay, while every repeat has its measured times/RSS validated. No quantum measurement, hardware or optimized-SCI superiority claim follows.','',
        f"Declaration: `{record['declaration_commit']}`.",f"Execution: `{record['source_commit']}`.",'']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args(argv)
    text=render(read_json(ROOT/RECORD))
    if args.check:
        if not REPORT.exists() or REPORT.read_text()!=text:print('FAIL large packet report differs');return 1
        print('OK large packet report matches observations')
    else:REPORT.write_text(text);print('Wrote '+str(REPORT))
    return 0


if __name__=='__main__':raise SystemExit(main())
