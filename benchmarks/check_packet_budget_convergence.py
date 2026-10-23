"""Independent references, replay and campaign bindings for the budget-convergence record.

Every arm's prefix energies, residuals, spin, work counters, support and final
state are replayed at every prefix from independently built occupation-sign
Hamiltonians and singlet frames. Candidate scoring is replayed at every step
with ``--full``, and always for inputs up to ``FULL_REPLAY_DIMENSION``
determinants; larger packet trajectories otherwise rescore every candidate at
``sampled_steps`` and check only the selected candidate elsewhere.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.archive_provenance import commit_binding, history_kind
from benchmarks.metadata_io import read_json, sha256
from benchmarks.run_packet_budget_convergence import (BUNDLE, CONFIG, DECLARATION, RECEIPT, RECORD,
    SOURCES, canonical, decode_array, decode_science, digest, expected_order)
from benchmarks.check_sparse_packet_work import independent_input
from benchmarks.check_h2o_residual_comparison import cas_indices, independent_frame, spatial_key
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from clifford_qc.evidence import validate_evidence

FULL_REPLAY_DIMENSION = 1225
SAMPLE_STRIDE = 64
NEAR_TIE = 1e-8  # The producer records scores this close to a step's best.


def sampled_steps(decision_count):
    """Steps whose every candidate is rescored when a packet replay is bounded."""
    return {0, 1, 2, decision_count-1, decision_count-2} | set(range(0, decision_count, SAMPLE_STRIDE))


def independent_data(case, c):
    h, words = independent_input(case)
    n = case['n_qubits']
    dense_spin = independent_spin_squared(words, n)
    frame = independent_frame(words, n, dense_spin, c['spin_tolerance'])[0]
    spin = csr_matrix(dense_spin)
    del dense_spin  # 384 MB on the 4,900-determinant sectors.
    hf = sum(1 << (n-1-j) for j in range(case['n_electrons']))
    initial = cas_indices(frame, words, n, hf, case['active_spatial_orbitals'], c['spin_tolerance'])
    groups = {}
    for j, column in enumerate(frame.T):
        if j not in initial:
            counts = spatial_key(words[np.argmax(abs(column))], n, range(n//2))
            key = sum((counts[p] if p not in case['active_spatial_orbitals'] else 0)*3**p for p in range(n//2))
            groups.setdefault(key, []).append(j)
    return {'h': h, 'words': words, 'spin': spin, 'frame': frame, 'initial': initial, 'groups': groups}


def independent_reference(case, c, data=None):
    """Lowest singlet-frame eigenvalue, cross-checked with the full-sector ground state."""
    data = data or independent_data(case, c)
    h, frame, spin = data['h'], data['frame'], data['spin']
    values, vectors = np.linalg.eigh(frame.conj().T @ (h @ frame))
    singlet = frame @ vectors[:, 0]
    # Overlaps use the whole ground space, so a degenerate level has no arbitrary basis.
    ground_space = frame @ vectors[:, values <= values[0] + 1e-8]
    upper = float(np.max(np.asarray(abs(h).sum(axis=1)).ravel())) + 1
    shifted = h.real - upper*eye(h.shape[0], format='csr')
    sector = []
    for offset in (1., np.sqrt(2)):
        value, vector = eigsh(shifted, k=1, which='LM', v0=np.sin((np.arange(h.shape[0])+1)*offset),
                              tol=1e-12, maxiter=20000)
        sector.append((float(value[0]+upper), vector[:, 0]))
    if abs(sector[0][0]-sector[1][0]) > 1e-9:
        raise ValueError('independent sector reference starts disagree')
    energy, vector = sector[0]
    if np.linalg.norm(h @ vector - energy*vector) > 1e-9 or np.linalg.norm(h @ singlet - values[0]*singlet) > 1e-9:
        raise ValueError('reference residual exceeds tolerance')
    if values[0] < energy - 1e-9:
        raise ValueError('singlet reference lies below the sector ground state')
    spin_norm = float(np.linalg.norm(spin @ vector))
    return {'singlet_reference': float(values[0]), 'sector_ground': energy,
            'sector_ground_spin_action_norm': spin_norm, 'sector_ground_is_singlet': spin_norm <= 1e-8,
            'singlet_gap_to_next': float(values[1]-values[0]) if len(values) > 1 else None,
            'ground_space_dimension': int(ground_space.shape[1]), 'ground_state': ground_space}


def final_overlap(science, reference):
    """Norm of the final Ritz state's projection onto the singlet ground space."""
    return float(np.linalg.norm(reference['ground_state'].conj().T @ decode_array(science['final_state'])))


def lowering(energy, diagonal, coupling):
    gap = diagonal - energy
    return np.maximum(0.0, (np.hypot(gap, 2*np.abs(coupling)) - gap)/2)


def orthonormalize_columns(vectors, span, cutoff):
    """Batched two-pass projection, normalization and canonical phase; inadmissible columns become None."""
    result = np.array(vectors, complex, copy=True)
    for _ in (0, 1):
        result -= span @ (span.conj().T @ result)
    lengths = np.linalg.norm(result, axis=0)
    columns = []
    for j, length in enumerate(lengths):
        if length <= cutoff:
            columns.append(None); continue
        column = result[:, j]/length
        pivot = column[np.argmax(abs(column))]
        columns.append(column*pivot.conjugate()/abs(pivot))
    return columns


def replay(science, case, c, data, *, full=False):
    """Independent prefix-by-prefix replay of one arm; returns problem strings."""
    arm = next(a for a in c['arms'] if a['name'] == science['arm'])
    h, spin, frame, initial, groups = data['h'], data['spin'], data['frame'], data['initial'], data['groups']
    problems = []

    def require(ok, message):
        if not ok: problems.append(message)

    def close(actual, expected, message, atol=1e-9):
        require(actual is not None and np.shape(actual) == np.shape(expected)
                and np.allclose(actual, expected, atol=atol, rtol=0), message)

    rows, decisions = science['trajectory'], science['decisions']
    require(len(science['support']) == len(rows), 'support prefix count')
    span = np.asarray(frame[:, initial], complex)
    actions = np.column_stack([h @ column for column in span.T])
    projected = span.conj().T @ actions
    diagonal = h.diagonal().real
    calls, scored, materialized = len(initial), 0, 0
    choice, selected_score, terminal, state, sci = {'initial_csf_indices': initial}, None, None, None, None
    remaining = [j for j in range(frame.shape[1]) if j not in initial]
    every = full or case['sector_dimension'] <= FULL_REPLAY_DIMENSION
    sample = sampled_steps(len(decisions))
    for step, row in enumerate(rows):
        values, vectors = np.linalg.eigh((projected+projected.conj().T)/2)
        energy, state = float(values[0]), span @ vectors[:, 0]
        residual = actions @ vectors[:, 0] - energy*state
        close(row['energy'], energy, 'energy')
        close(row['residual_norm'], np.linalg.norm(residual), 'true residual')
        close(row['spin_action_norm'], np.linalg.norm(spin @ state), 'spin diagnostic')
        require(np.linalg.norm(spin @ state) < c['spin_tolerance'], 'singlet state')
        require(row['step'] == step and row['dimension'] == span.shape[1], 'prefix identity')
        require((row['h_action_calls'], row['candidates_scored'], row['materialized_candidates'])
                == (calls, scored, materialized), 'prefix work')
        require(row['selection'] == choice, 'selection identity')
        if selected_score is None: require(row['selection_score'] is None, 'initial score')
        else: close(row['selection_score'], selected_score, 'selection score')
        require(row['retained_payload_bytes'] == 32*span.size, 'retained array payload')
        terminal = ('residual_tolerance' if np.linalg.norm(residual) <= c['residual_tolerance']
                    else 'column_ceiling' if span.shape[1] >= case['maximum_columns']
                    else 'growth_step_ceiling' if step == case['maximum_growth_steps'] else None)
        if terminal:
            require(step == len(rows)-1, 'continued after terminal prefix')
            break
        raw = residual/np.maximum(diagonal-energy+c['preconditioner_shift'], c['preconditioner_shift'])
        cutoff = max(c['direction_absolute_tolerance'], c['direction_relative_tolerance']*np.linalg.norm(raw))
        if arm['kind'] == 'davidson':
            materialized += 1
            direction = orthonormalize_columns((frame @ (frame.conj().T @ raw))[:, None], span, cutoff)[0]
            if direction is None:
                require(step == len(decisions), 'missing nonterminal decision')
                terminal = 'no_admissible_direction'; break
            if step >= len(decisions): return problems+['missing recorded decision']
            key, score = 'residual', float(lowering(energy, np.vdot(direction, h @ direction).real,
                                                    np.vdot(direction, residual)))
            require(decisions[step]['step'] == step, 'decision step')
            close(decisions[step]['score'], score, 'Davidson score')
            calls += 1; scored += 1
        elif arm['kind'] == 'selected_ci':
            if sci is None:
                actions_all = h @ frame
                sci = actions_all, np.einsum('ij,ij->j', frame.conj(), actions_all).real
                materialized += len(remaining); calls += len(remaining)
            if not remaining:
                terminal = 'no_admissible_direction'; break
            if step >= len(decisions): return problems+['missing recorded decision']
            keys = np.array(remaining)
            scores = dict(zip(keys.tolist(), lowering(energy, sci[1][keys], sci[0][:, keys].T @ state.conj()).tolist()))
            key = check_selection(decisions[step], scores, c, require, close, step)
            if key not in scores: return problems+['inadmissible recorded selection']
            score, direction = scores[key], np.asarray(frame[:, key], complex)
            remaining.remove(key); scored += len(scores)
        else:
            key, direction, score, admissible, attempted = packet_step(
                decisions, step, arm, energy, residual, raw, cutoff, span, h, frame, groups, c,
                full_scoring=every or step in sample, require=require, close=close)
            materialized += attempted
            if key is None:
                terminal = 'no_admissible_direction'; break
            calls += admissible; scored += admissible
        if direction is None: return problems+['inadmissible recorded selection']
        acted = h @ direction
        projected = np.block([[projected, (span.conj().T @ acted)[:, None]],
                              [(direction.conj() @ actions)[None, :], np.vdot(direction, acted)]])
        span, actions = np.column_stack((span, direction)), np.column_stack((actions, acted))
        choice, selected_score = {'candidate': key}, score
    support_problems(science, span, rows, c, require)
    require(science['stop_reason'] == terminal and terminal is not None, 'stopping rule')
    require(len(decisions) == len(rows)-1, 'decision count')
    require((science['final_h_action_calls'], science['final_candidates_scored'], science['materialized_candidates'])
            == (calls, scored, materialized), 'terminal work')
    close(span.conj().T @ span, np.eye(span.shape[1]), 'retained orthogonality')
    final = decode_array(science['final_state'])
    overlap = np.vdot(state, final)
    require(abs(abs(overlap)-1) <= 1e-8 and np.allclose(final, state*overlap/abs(overlap), atol=1e-8, rtol=0),
            'final Ritz state')
    return problems


def support_problems(science, span, rows, c, require):
    """Span determinant support and amplitude nonzeros at each prefix, to 0.1%."""
    mask, nonzeros = np.zeros(span.shape[0], bool), 0
    for row, (support, count) in zip(rows, science['support']):
        new = abs(span[:, :row['dimension']]) > c['support_tolerance']
        mask = new.any(axis=1); nonzeros = int(new.sum())
        for recorded, independent, name in ((support, int(mask.sum()), 'span support'), (count, nonzeros, 'amplitude nonzeros')):
            require(abs(recorded-independent) <= max(2, 1e-3*independent), name)


def check_selection(decision, scores, c, require, close, step):
    """Exact recorded tie-break, with every independent near-best score recorded."""
    require(decision['step'] == step, 'decision step')
    near = {int(k): v for k, v in decision['near_ties'].items()}
    best = max(scores.values())
    require(decision['admissible'] == len(scores), 'admissible count')
    for key, value in scores.items():
        if value >= best - NEAR_TIE + 2e-9:
            require(key in near, 'missing near-tie record')
    for key, value in near.items():
        close(value, scores.get(key, float('nan')), 'near-tie score')
    key = decision['selected']
    require(near and key == min(k for k in near if near[k] >= max(near.values()) - c['score_tie_tolerance']),
            'exact recorded tie-break')
    require(key in scores and best - scores[key] <= 2e-9, 'independent best score')
    close(decision['score'], scores.get(key, float('nan')), 'selected score')
    return key


def packet_step(decisions, step, arm, energy, residual, raw, cutoff, span, h, frame, groups, c, *,
                full_scoring, require, close):
    """Replay one packet decision; returns (key, direction, score, admissible, attempted)."""
    early = arm['policy'] == 'early'
    res_coefficients, raw_coefficients = frame.conj().T @ residual, frame.conj().T @ raw
    mass = {key: float(np.linalg.norm(res_coefficients[ids])**2) for key, ids in groups.items()}
    audit = early and (step+1) % c['audit_interval'] == 0
    if step >= len(decisions):
        # Terminal: every group must be inadmissible, after any early fallback.
        directions = orthonormalize_columns(np.column_stack(
            [frame[:, ids] @ raw_coefficients[ids] for ids in groups.values()]), span, cutoff)
        require(all(d is None for d in directions), 'missing nonterminal decision')
        return None, None, None, 0, len(groups)
    decision = decisions[step]
    require(decision['step'] == step and decision['audit'] == audit, 'audit policy')
    if early:
        shortlist = decision.get('shortlist')
        require(isinstance(shortlist, list) and len(set(shortlist)) == len(shortlist)
                and set(shortlist) <= set(groups), 'shortlist identity')
        shortlist = [k for k in (shortlist or []) if k in groups]
        total, inside = sum(mass.values()), sum(mass[k] for k in shortlist)
        outside = [mass[k] for k in groups if k not in set(shortlist)]
        slack = 1e-10*max(total, 1e-300)
        require(not outside or min((mass[k] for k in shortlist), default=0.) >= max(outside)-slack, 'shortlist mass order')
        require(total == 0 or inside >= c['shortlist_fraction']*total-slack*len(groups), 'shortlist coverage')
        lightest = min((mass[k] for k in shortlist), default=0.)
        require(total == 0 or inside-lightest < c['shortlist_fraction']*total+slack*len(groups), 'shortlist minimality')
    else:
        shortlist = sorted(groups)
    attempted = sorted(groups) if audit or decision['fallback'] else shortlist
    require(not decision['fallback'] or (early and not audit), 'fallback policy')
    require(decision['attempted'] == len(attempted), 'attempted count')
    key = decision['selected']
    if full_scoring:
        primary = sorted(groups) if audit else shortlist
        directions = dict(zip(groups, orthonormalize_columns(np.column_stack(
            [frame[:, ids] @ raw_coefficients[ids] for ids in groups.values()]), span, cutoff)))
        exact = {}
        for k, d in directions.items():
            if d is not None:
                exact[k] = float(lowering(energy, np.vdot(d, h @ d).real, np.vdot(d, residual)))
        first = {k: exact[k] for k in primary if k in exact}
        fallback = early and not audit and len(primary) < len(groups) and (
            not first or max(first.values()) <= c['score_tie_tolerance'])
        require(decision['fallback'] == fallback, 'fallback policy')
        evaluated = exact if fallback or not early or audit else first
        if not evaluated: return None, None, None, 0, len(groups)
        check_selection(decision, evaluated, c, require, close, step)
        require(decision['audit_missed_winner'] == bool(audit and key not in shortlist), 'audit winner miss')
        if audit:
            marked = [exact[k] for k in shortlist if k in exact]
            close(decision['audit_score_loss'], max(0., max(exact.values())-max(marked, default=0.)), 'audit loss')
        else: require(decision['audit_score_loss'] is None, 'unaudited loss')
        direction = directions.get(key)
        return key, direction, evaluated.get(key, float('nan')), len(evaluated), len(attempted)
    # Bounded replay: verify the recorded selection's internal consistency and its own candidate.
    near = {int(k): v for k, v in decision['near_ties'].items()}
    require(key in near and key == min(k for k in near if near[k] >= max(near.values()) - c['score_tie_tolerance']),
            'exact recorded tie-break')
    require(key in groups and key in attempted, 'selected group attempted')
    require(decision['audit_missed_winner'] == bool(audit and key not in shortlist), 'audit winner miss')
    ids = groups.get(key, [])
    direction = orthonormalize_columns((frame[:, ids] @ raw_coefficients[ids])[:, None], span, cutoff)[0] if ids else None
    score = float(lowering(energy, np.vdot(direction, h @ direction).real, np.vdot(direction, residual))) if direction is not None else float('nan')
    close(decision['score'], score, 'selected score')
    require(1 <= decision['admissible'] <= len(attempted), 'admissible count')
    return key, direction, score, decision['admissible'], len(attempted)


def bindings(record):
    history_kind(ROOT)
    receipt = read_json(ROOT/RECEIPT)
    if receipt['schema'] != 'clifford_qc.packet_budget_convergence_publication.v1':
        raise ValueError('publication schema differs')
    if receipt['record_sha256'] != sha256(ROOT/RECORD) or record != read_json(ROOT/RECORD):
        raise ValueError('observation bytes differ')
    if receipt['bundle_sha256'] != sha256(ROOT/BUNDLE): raise ValueError('source bundle differs')
    subprocess.run(['git', 'bundle', 'verify', str(ROOT/BUNDLE)], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'bundle', 'unbundle', str(ROOT/BUNDLE)], cwd=ROOT, check=True, capture_output=True)
    if commit_binding(ROOT, receipt['observation_commit'], RECORD)['sha256'] != receipt['record_sha256']:
        raise ValueError('observation commit differs')
    for before, after in ((record['declaration_commit'], record['source_commit']),
                          (record['source_commit'], receipt['observation_commit'])):
        if before == after: raise ValueError('declaration/source/observation must differ')
        subprocess.run(['git', 'merge-base', '--is-ancestor', before, after], cwd=ROOT, check=True, capture_output=True)


def verify(record, *, bound=True, numerical=True, full=False):
    if bound: bindings(record)
    problems = []
    validate_evidence(record)
    c = read_json(ROOT/CONFIG)
    if record['schema'] != 'clifford_qc.packet_budget_convergence.v1' or record['config'] != c:
        return ['record/config identity']
    for path, key in ((CONFIG, 'config_sha256'), (DECLARATION, 'declaration_sha256')):
        if sha256(ROOT/path) != record[key] or commit_binding(ROOT, record['declaration_commit'], path)['sha256'] != record[key]:
            problems.append('design binding '+path)
    if set(record['source_sha256']) != set(SOURCES): problems.append('execution source universe')
    for path, value in record['source_sha256'].items():
        if commit_binding(ROOT, record['source_commit'], path)['sha256'] != value:
            problems.append('execution source '+path)
    for path in c['implementation_paths_absent_at_declaration']:
        if subprocess.run(['git', 'cat-file', '-e', f'{record["declaration_commit"]}:{path}'], cwd=ROOT, capture_output=True).returncode == 0:
            problems.append('implementation predates declaration')
    if record['provenance']['git_sha'] != record['source_commit'] or record['provenance']['git_dirty'] is not False:
        problems.append('clean source provenance')
    journal = record['journal']
    if digest(b''.join(canonical(e)+b'\n' for e in journal)) != record['journal_sha256']:
        problems.append('journal digest')
    order = expected_order(c)
    if len(record['workers']) != len(order): return problems+['scheduled-cell count']
    starts, finishes = {}, {}
    for event in journal:
        if event['event'] not in ('start', 'finish'):
            problems.append('unknown journal event'); continue
        i = event['order_index']; target = starts if event['event'] == 'start' else finishes
        if i in target or i not in range(len(order)): problems.append('duplicate/invalid journal index')
        target[i] = event
    sciences = {key: decode_science(value) for key, value in record['scientific_payloads'].items()}
    used = {w['scientific_sha256'] for w in record['workers'] if w['status'] == 'ok'}
    if set(sciences) != used: problems.append('scientific payload universe')
    if set(record['final_ground_state_overlaps']) != set(sciences): problems.append('overlap universe')
    if any(key != value['sha256'] for key, value in record['scientific_payloads'].items()):
        problems.append('scientific content address')
    for i, (worker, task) in enumerate(zip(record['workers'], order)):
        if any(worker[k] != v for k, v in task.items()) or worker['order_index'] != i:
            problems.append('worker order')
        if i not in starts or starts[i]['task'] != task or i not in finishes or finishes[i]['worker'] != worker:
            problems.append('durable start/outcome evidence')
        elif finishes[i]['utc_seconds'] < starts[i]['utc_seconds']:
            problems.append('journal time order')
        if worker['status'] not in ('ok', 'failed', 'timeout', 'interrupted'):
            problems.append('worker status')
        if worker['status'] != 'ok': continue
        recovered = 'recovered' in worker
        if recovered and worker['process_wall_seconds'] is not None: problems.append('recovered process timing')
        bad = [key for key in ('setup_seconds', 'solver_seconds') + (() if recovered else ('process_wall_seconds',))
               if not isinstance(worker[key], (int, float)) or not math.isfinite(worker[key]) or worker[key] < 0]
        problems.extend('finite timing '+key for key in bad)
        if not recovered and not bad and worker['setup_seconds']+worker['solver_seconds'] > worker['process_wall_seconds']+1e-6:
            problems.append('process timing')
        peaks = [worker[key] for key in ('imports_peak_rss_bytes', 'setup_peak_rss_bytes',
                                         'solver_end_peak_rss_bytes', 'worker_peak_rss_bytes')]
        if any(type(v) is not int or v <= 0 for v in peaks) or peaks != sorted(peaks):
            problems.append('RSS phase high-water order')
        payload = record['scientific_payloads'][worker['scientific_sha256']]
        if payload['gzip_sha256'] != worker['scientific_gzip_sha256']:
            problems.append('cell scientific gzip binding')
        science = sciences[worker['scientific_sha256']]
        if science['arm'] != worker['arm']: problems.append('arm identity')
        times, rss = worker['prefix_wall_seconds'], worker['prefix_peak_rss_bytes']
        if (len(times) != len(science['trajectory']) or times != sorted(times)
                or any(not math.isfinite(t) or t < 0 for t in times) or times[-1] > worker['solver_seconds']):
            problems.append('prefix timing')
        if len(rss) != len(times) or rss != sorted(rss) or rss[0] < worker['setup_peak_rss_bytes'] or rss[-1] > worker['solver_end_peak_rss_bytes']:
            problems.append('prefix RSS high-water order')
        if worker['worker_environment_sha256'] != record['provenance']['environment_sha256']:
            problems.append('worker environment')
    for case in c['inputs']:
        name = case['name']
        workers = [w for w in record['workers'] if w['input'] == name and w['status'] == 'ok']
        for arm in c['arms']:
            if len({w['scientific_sha256'] for w in workers if w['arm'] == arm['name']}) > 1:
                problems.append(f'{name}: repeated {arm["name"]} payloads differ')
        if not numerical: continue
        data = independent_data(case, c)
        reference = independent_reference(case, c, data)
        recorded = record['references'][name]
        for key in ('singlet_reference', 'sector_ground'):
            if abs(reference[key]-recorded[key]) > 1e-9: problems.append(f'{name}: {key}')
        if reference['sector_ground_is_singlet'] != recorded['sector_ground_is_singlet']:
            problems.append(f'{name}: sector ground spin')
        for key in sorted({w['scientific_sha256'] for w in workers}):
            science = sciences[key]
            problems.extend(f'{name}/{science["arm"]}: {p}' for p in replay(science, case, c, data, full=full))
            if abs(final_overlap(science, reference)-record['final_ground_state_overlaps'][key]) > 1e-8:
                problems.append(f'{name}/{science["arm"]}: final ground-state overlap')
            last = science['trajectory'][-1]
            if (last['dimension'] == case['singlet_dimension']
                    and abs(last['energy']-reference['singlet_reference']) > c['decisions']['complete_span_tolerance']):
                problems.append(f'{name}/{science["arm"]}: complete span misses the reference')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=ROOT/RECORD)
    parser.add_argument('--unbound', action='store_true', help='validate unpublished local execution without a publication receipt')
    parser.add_argument('--full', action='store_true', help='rescore every candidate at every step of every trajectory')
    args = parser.parse_args(argv)
    problems = verify(read_json(args.record), bound=not args.unbound, full=args.full)
    if problems:
        for problem in problems: print('FAIL '+problem)
        return 1
    print(f'VALID: independent references and {"full" if args.full else "bounded"} replay; publication/source bindings '
          + ('were not checked (--unbound)' if args.unbound else 'verified'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
