"""Independent occupation-sign replay of sparse/exhaustive and early packet growth."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
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
from benchmarks.check_h2o_residual_comparison import (
    occupation_basis, independent_frame, orthogonalize, decode, cas_indices, spatial_key, best_key)
from benchmarks.check_orthogonal_active_space import independent_spin_squared
from benchmarks.check_packet_work_large import occupation_hubbard
from benchmarks.check_packet_work_reuse import verify_worker as verify_dense
from benchmarks.phase17_ceo_oracle import hamiltonian
from benchmarks.run_sparse_packet_work import (
    CONFIG, DECLARATION, RECORD, RECEIPT, BUNDLE, SOURCES, unpack, policy_config, expected_order, probes)
from clifford_qc.evidence import validate_evidence


def independent_input(case):
    if case['kind'] == 'hubbard':
        return occupation_hubbard(case)
    words = occupation_basis(case['n_qubits']//2, case['n_electrons']//2)
    h, _ = hamiltonian(ROOT/case['path'], words)
    return csr_matrix(h), words


def reference_energy(h):
    upper = float(np.max(np.asarray(abs(h).sum(axis=1)).ravel())) + 1
    shifted = h.real - upper*eye(h.shape[0], format='csr')
    energies = []
    for offset in (1., np.sqrt(2)):
        start = np.sin((np.arange(h.shape[0])+1)*offset)
        value, vector = eigsh(shifted, k=1, which='LM', v0=start, tol=1e-12, maxiter=20000)
        energy = float(value[0]+upper)
        if np.linalg.norm(h @ vector[:, 0] - energy*vector[:, 0]) > 1e-9:
            raise ValueError('reference residual exceeds tolerance')
        energies.append(energy)
    if abs(energies[0]-energies[1]) > 1e-9:
        raise ValueError('independent reference starts disagree')
    return energies[0]


def columns_agree(actual, expected, tolerance=1e-8):
    """Compare ordered physical directions, allowing one global phase/column.

    Equal-magnitude opposite-sign pivots can swap at roundoff between dense
    and sparse reductions. Their canonical arrays differ, but not their rays.
    The replay separately enforces each recorded column's own phase convention.
    """
    if actual.shape != expected.shape:
        return False
    for a, b in zip(actual.T, expected.T):
        overlap = np.vdot(b, a)
        phase = overlap/abs(overlap) if abs(overlap) > 1e-30 else 1
        if not np.allclose(a, b*phase, atol=tolerance, rtol=0):
            return False
    return True


def replay(result, case, c, h, words, spin, frame_data):
    if result['arm'] == 'dense':
        original = deepcopy(result)
        original['arm'] = 'original'
        original.pop('setup')  # Setup accounting has a different explicit schema.
        problems = verify_dense(original, policy_config(c, case), h, words, spin,
                            frame_data=frame_data, n_qubits=case['n_qubits'],
                            n_electrons=case['n_electrons'])
        frame, _ = frame_data
        initial = result['trajectory'][0]['selection']['initial_csf_indices']
        counts = {spatial_key(words[np.argmax(abs(column))], case['n_qubits'],
                             [p for p in range(case['n_qubits']//2) if p not in case['active_spatial_orbitals']])
                  for j, column in enumerate(frame.T) if j not in initial}
        if any(row['materialized_candidates'] != row['step']*len(counts) for row in result['trajectory']):
            problems.append('dense materialization work')
        if result['materialized_candidates'] != len(counts)*(len(result['decisions'])
                + int(result['stop_reason'] == 'no_admissible_direction')):
            problems.append('dense terminal materializations')
        return problems
    problems = []

    def require(ok, message):
        if not ok: problems.append(message)

    def close(actual, expected, message, atol=1e-9):
        require(np.shape(actual) == np.shape(expected)
                and np.allclose(actual, expected, atol=atol, rtol=0), message)

    frame, _ = frame_data
    n, d = case['n_qubits'], len(words)
    hf = sum(1 << (n-1-j) for j in range(case['n_electrons']))
    initial = cas_indices(frame, words, n, hf, case['active_spatial_orbitals'], c['spin_tolerance'])
    groups = {}
    for j, column in enumerate(frame.T):
        if j not in initial:
            counts = spatial_key(words[np.argmax(abs(column))], n, range(n//2))
            key = sum((counts[p] if p not in case['active_spatial_orbitals'] else 0)*3**p for p in range(n//2))
            groups.setdefault(key, []).append(j)
    expected = frame[:, initial]
    span = np.empty((d, 0), complex)
    choice, selected_score = {'initial_csf_indices': initial}, None
    scored = materialized = decision_count = 0
    calls = len(initial)
    terminal = None
    previous_energy, previous_time = float('inf'), -1.
    for step, row in enumerate(result['trajectory']):
        added = decode(row['added_columns'], d)
        require(columns_agree(added, expected), 'selected physical direction')
        for column in added.T:
            pivot = column[np.argmax(abs(column))]
            require(pivot.real >= 0 and abs(pivot.imag) <= 1e-9, 'recorded canonical phase')
        require(row['selection'] == choice, 'selection identity')
        if selected_score is None: require(row['selection_score'] is None, 'initial score')
        else: close(row['selection_score'], selected_score, 'selection score')
        span = np.column_stack((span, added))
        close(span.conj().T @ span, np.eye(span.shape[1]), 'retained orthogonality')
        actions = h @ span
        projected = span.conj().T @ actions
        values, vectors = np.linalg.eigh((projected+projected.conj().T)/2)
        energy, state = float(values[0]), span @ vectors[:, 0]
        residual = h @ state-energy*state
        close(row['energy'], energy, 'energy')
        close(row['residual_norm'], np.linalg.norm(residual), 'true residual')
        close(row['spin_action_norm'], np.linalg.norm(spin @ state), 'spin diagnostic')
        require(np.linalg.norm(spin @ state) < c['spin_tolerance'], 'singlet state')
        require(row['step'] == step and row['dimension'] == span.shape[1], 'prefix identity')
        require((row['h_action_calls'], row['candidates_scored'], row['materialized_candidates'])
                == (calls, scored, materialized), 'prefix work')
        require(row['retained_payload_bytes'] == 32*span.size, 'retained array payload')
        require(energy <= previous_energy+1e-9, 'nested monotonicity')
        require(np.isfinite(row['wall_seconds']) and row['wall_seconds'] >= previous_time, 'prefix timing')
        previous_energy, previous_time = energy, row['wall_seconds']
        terminal = ('residual_tolerance' if np.linalg.norm(residual) <= c['residual_tolerance']
                    else 'column_ceiling' if span.shape[1] >= case['maximum_columns']
                    else 'growth_step_ceiling' if step == case['maximum_growth_steps'] else None)
        if terminal:
            require(step == len(result['trajectory'])-1, 'continued after terminal prefix')
            break
        raw = residual/np.maximum(h.diagonal().real-energy+c['preconditioner_shift'], c['preconditioner_shift'])
        cutoff = max(c['direction_absolute_tolerance'], c['direction_relative_tolerance']*np.linalg.norm(raw))
        candidates, indicators, exact_scores = {}, {}, {}
        for key, ids in groups.items():
            block = frame[:, ids]
            indicators[key] = float(np.linalg.norm(block.conj().T @ residual)**2)
            direction = orthogonalize(block @ (block.conj().T @ raw), span, cutoff)
            if direction is not None:
                candidates[key] = direction
                gap = np.vdot(direction, h @ direction).real-energy
                exact_scores[key] = max(0., float((np.hypot(gap, 2*abs(np.vdot(direction, residual)))-gap)/2))
        if step >= len(result['decisions']):
            require(not candidates, 'missing nonterminal decision')
            materialized += len(groups)
            terminal = 'no_admissible_direction'
            break
        decision = result['decisions'][step]
        decision_count += 1
        require(set(decision['indicators']) == {str(k) for k in indicators}, 'indicator universe')
        if set(decision['indicators']) != {str(k) for k in indicators}: return problems
        for key, value in indicators.items():
            close(decision['indicators'][str(key)], value, 'group residual mass',
                  max(1e-24, 1e-10*sum(indicators.values())))
        # Validate masses independently, then use their recorded last-bit order
        # for the declared exact marking rule, as in the historical checker.
        observed_mass = {int(k): v for k, v in decision['indicators'].items()}
        require(all(np.isfinite(v) and v >= 0 for v in observed_mass.values()), 'finite masses')
        ordered = sorted(observed_mass, key=lambda k: (-observed_mass[k], k))
        short = sorted(groups)
        early = result['arm'] == 'early'
        if early:
            total, covered, short = sum(observed_mass.values()), 0., []
            for key in ordered:
                short.append(key); covered += observed_mass[key]
                if total > 0 and covered >= c['shortlist_fraction']*total: break
        audit = early and (step+1) % c['audit_interval'] == 0
        attempted = sorted(groups) if audit else short.copy()
        evaluated = {k: exact_scores[k] for k in attempted if k in candidates}
        fallback = early and not audit and len(attempted) < len(groups) and (
            not evaluated or max(evaluated.values()) <= c['score_tie_tolerance'])
        if fallback:
            attempted = sorted(groups)
            evaluated = exact_scores.copy()
        require(decision['step'] == step and decision['audit'] == audit
                and decision['fallback'] == fallback, 'audit/fallback policy')
        require(decision['shortlist'] == short and decision['attempted'] == sorted(attempted), 'mark/attempt identity')
        require(set(decision['scores']) == {str(k) for k in evaluated}, 'score universe')
        for key, value in evaluated.items(): close(decision['scores'].get(str(key), float('nan')), value, 'exact score')
        if not evaluated: return problems+['empty recorded decision']
        observed_scores = {int(k): v for k, v in decision['scores'].items()}
        key = decision['selected']
        require(key == best_key(observed_scores, c['score_tie_tolerance']), 'exact recorded tie-break')
        require(key in evaluated and max(evaluated.values())-evaluated.get(key, -1) <= 2e-9, 'independent best score')
        require(decision['audit_missed_winner'] == bool(audit and key not in short), 'audit winner miss')
        if audit:
            marked = [observed_scores[k] for k in short if k in observed_scores]
            close(decision['audit_score_loss'], max(0., max(observed_scores.values())-max(marked, default=0.)), 'audit loss', 1e-14)
        else: require(decision['audit_score_loss'] is None, 'unaudited loss')
        calls += len(evaluated); scored += len(evaluated); materialized += len(attempted)
        expected, choice, selected_score = candidates[key][:, None], {'candidate': key}, evaluated[key]
    require(result['stop_reason'] == terminal and terminal is not None, 'stopping rule')
    require(len(result['decisions']) == decision_count, 'decision count')
    require((result['final_h_action_calls'], result['final_candidates_scored'], result['materialized_candidates'])
            == (calls, scored, materialized), 'terminal work')
    return problems


def restore_and_bind(published):
    history_kind(ROOT)
    receipt = read_json(ROOT/RECEIPT)
    if receipt.get('schema') != 'clifford_qc.sparse_packet_work_publication.v1':
        raise ValueError('publication receipt schema differs')
    if published != read_json(ROOT/RECORD) or receipt['observation_sha256'] != sha256(ROOT/RECORD):
        raise ValueError('observation bytes differ')
    if receipt['bundle_sha256'] != sha256(ROOT/BUNDLE): raise ValueError('bundle digest differs')
    subprocess.run(['git', 'bundle', 'verify', str(ROOT/BUNDLE)], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'bundle', 'unbundle', str(ROOT/BUNDLE)], cwd=ROOT, check=True, capture_output=True)
    if commit_binding(ROOT, receipt['observation_commit'], RECORD)['sha256'] != sha256(ROOT/RECORD):
        raise ValueError('observation commit differs')
    record = unpack(published)
    if receipt.get('publication_recovery') != published.get('publication_recovery'):
        raise ValueError('publication recovery metadata differs')
    if published.get('publication_recovery'):
        recipe_path = ROOT/'benchmarks/provenance/sparse_packet_work_recovery.json'
        if sha256(recipe_path) != receipt.get('recovery_recipe_sha256'):
            raise ValueError('recovery recipe digest differs')
        recipe = read_json(recipe_path)
        recovery = published['publication_recovery']
        retained = recipe['retained_count']
        if (recipe['schema'] != 'clifford_qc.sparse_packet_recovery_recipe.v1'
                or retained != recovery['recovered_observations']
                or retained != sum(w['status'] == 'ok' for w in record['workers'])
                or len(record['workers'])-retained != recovery['unavailable_observations']):
            raise ValueError('recovery counts differ')
        checkpoint_record = deepcopy(record)
        checkpoint_record['workers'] = checkpoint_record['workers'][:retained]
        checkpoint_record['references'] = {}
        checkpoint_record.pop('evidence_schema'); checkpoint_record.pop('evidence')
        for entry in checkpoint_record['workers']:
            for row in entry['result']['trajectory']:
                row.pop('error'); row.pop('target_reached')
        checkpoint = {k: checkpoint_record if k == 'record' else recipe['state_header'][k]
                      for k in recipe['state_key_order']}
        digest = hashlib.sha256(json.dumps(checkpoint, allow_nan=False).encode()).hexdigest()
        if digest != recipe['checkpoint_sha256'] or digest != recovery['checkpoint_sha256']:
            raise ValueError('recovered observation checkpoint bytes differ')
    if set(record['source_sha256']) != set(SOURCES): raise ValueError('source universe differs')
    for path, digest in record['source_sha256'].items():
        if commit_binding(ROOT, record['source_commit'], path)['sha256'] != digest:
            raise ValueError('execution source differs: '+path)
    for path, key in ((CONFIG, 'config_sha256'), (DECLARATION, 'declaration_sha256')):
        if (sha256(ROOT/path) != record[key]
                or commit_binding(ROOT, record['declaration_commit'], path)['sha256'] != record[key]):
            raise ValueError('design bytes differ')
    for before, after in ((record['declaration_commit'], record['source_commit']),
                          (record['source_commit'], receipt['observation_commit'])):
        if before == after: raise ValueError('declaration/execution/observation must be distinct')
        subprocess.run(['git', 'merge-base', '--is-ancestor', before, after], cwd=ROOT, check=True, capture_output=True)
    for path in record['config']['implementation_paths_absent_at_declaration']:
        if subprocess.run(['git', 'cat-file', '-e', f'{record["declaration_commit"]}:{path}'], cwd=ROOT,
                          capture_output=True).returncode == 0:
            raise ValueError('implementation predates declaration')
    if record['provenance']['git_sha'] != record['source_commit'] or record['provenance']['git_dirty'] is not False:
        raise ValueError('execution provenance differs')


def verify(published, *, bindings=True):
    if bindings: restore_and_bind(published)
    record = unpack(published)
    validate_evidence(record)
    c = read_json(ROOT/CONFIG)
    if (record['schema'] != 'clifford_qc.sparse_packet_work.v1' or record['config'] != c
            or record['hardware_execution'] is not False or record['quantum_advantage_claim'] is not False):
        raise ValueError('campaign identity differs')
    problems = []
    order = [(w['input'], w['repeat'], w['arm']) for w in record['workers']]
    if order != expected_order(c): problems.append('complete randomized worker order')
    for case in c['inputs']:
        h, words = independent_input(case)
        reference = reference_energy(h)
        if abs(record['references'][case['name']]-reference) > 1e-9: problems.append('reference')
        spin = independent_spin_squared(words, case['n_qubits'])
        frame_data = independent_frame(words, case['n_qubits'], spin, c['spin_tolerance'])
        spin_action = csr_matrix(spin)
        seen = {}
        for entry in record['workers']:
            if entry['input'] != case['name']: continue
            if entry['status'] not in ('ok', 'failed', 'timeout'):
                problems.append('worker status'); continue
            if entry['status'] != 'ok':
                if not entry.get('diagnostic') or 'result' in entry: problems.append('failure disclosure')
                continue
            r = entry['result']; setup = r['setup']
            if r['arm'] != entry['arm']: problems.append('arm identity')
            if (not np.isfinite(entry['process_wall_seconds']) or entry['process_wall_seconds'] <= 0
                    or not np.isfinite(setup['wall_seconds']) or setup['wall_seconds'] < 0
                    or not np.isfinite(r['solver_wall_seconds']) or r['solver_wall_seconds'] < 0
                    or entry['process_wall_seconds']+1e-6 < setup['wall_seconds']+r['solver_wall_seconds']
                    or type(r['final_peak_rss_bytes']) is not int
                    or r['final_peak_rss_bytes'] < max(1, setup['peak_rss_bytes'])):
                problems.append('isolated cost observation')
            if setup['sector_dimension'] != len(words) or setup['frame_dimension'] != frame_data[0].shape[1]:
                problems.append('setup dimensions')
            if (type(setup['h_nnz']) is not int or not 0 < setup['h_nnz'] <= len(words)**2
                    or setup['h_payload_bytes'] != 20*setup['h_nnz']+4*(len(words)+1)):
                problems.append('sparse H payload')
            native_probe = np.asarray(setup['h_probe_real'])+1j*np.asarray(setup['h_probe_imag'])
            if not np.allclose(native_probe, h @ probes(len(words)), atol=1e-10, rtol=0):
                problems.append('independent H probes')
            if entry['arm'] != 'dense' and (setup['dense_sector_h_temporary_bytes'] != 0
                    or not 0 < setup['largest_spin_block'] <= c['max_spin_block_dimension']):
                problems.append('sparse setup bound')
            for row in r['trajectory']:
                if (abs(row['error']-(row['energy']-reference)) > 1e-9
                        or row['target_reached'] != (abs(row['error']) <= case['accuracy_target'])
                        or row['error'] < -1e-8): problems.append('post-run accuracy')
            scientific = {k: v for k, v in r.items() if k not in ('setup', 'solver_wall_seconds', 'final_peak_rss_bytes')}
            scientific = deepcopy(scientific)
            for row in scientific['trajectory']:
                row.pop('wall_seconds', None); row.pop('peak_rss_bytes', None)
            if entry['arm'] not in seen or scientific != seen[entry['arm']]:
                problems.extend(f'{case["name"]}/{entry["arm"]}: {p}' for p in
                                replay(r, case, c, h, words, spin_action, frame_data))
                seen[entry['arm']] = scientific
        for repeat in range(case['repeats']):
            paired = {w['arm']: w['result'] for w in record['workers']
                      if w['input'] == case['name'] and w['repeat'] == repeat and w['status'] == 'ok'}
            if 'dense' in paired and 'sparse' in paired:
                left, right = paired['dense']['trajectory'], paired['sparse']['trajectory']
                if len(left) != len(right): problems.append('exhaustive prefix equivalence')
                for a, b in zip(left, right):
                    if (a['selection'] != b['selection'] or abs(a['energy']-b['energy']) > 1e-9
                            or abs(a['residual_norm']-b['residual_norm']) > 1e-9
                            or not columns_agree(decode(a['added_columns'], len(words)),
                                                  decode(b['added_columns'], len(words)))):
                        problems.append('exhaustive trajectory equivalence')
        del spin, frame_data
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', type=Path, default=ROOT/RECORD)
    args = p.parse_args(argv)
    problems = verify(read_json(args.record))
    if problems: raise SystemExit('\n'.join(problems))
    print('Sparse packet work: independent replay and observation bindings pass')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
