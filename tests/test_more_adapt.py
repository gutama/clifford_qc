"""Independent spectral/finite-difference checks on undeclared small systems."""
from copy import deepcopy
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from clifford_qc.algorithms import more_energy_gradient, run_more_adapt
from clifford_qc.algorithms.exchange import exchange_pools
from clifford_qc.dense_reference import to_matrix
from clifford_qc.evidence import declare_evidence
from clifford_qc.models import hubbard
from clifford_qc.models.fcidump import fcidump_model
from benchmarks.phase17_more_acase import run_acase_mean
from benchmarks.phase17_more_design import EVIDENCE, digest, structure, target_summary
from benchmarks.phase17_more_oracle import matrices, hamiltonian
from benchmarks.check_phase17_more import check_record


def test_shared_adjoint_matches_spectral_evolution_and_finite_differences():
    pool, _ = exchange_pools(4)
    ts = [to_matrix(op.generator) for op in (pool[0], pool[-1], pool[1])]
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    h = raw + raw.conj().T
    refs = np.linalg.qr(rng.normal(size=(16, 3)) + 1j * rng.normal(size=(16, 3)))[0]
    theta = np.array([.23, -.37, .11])
    value, gradient, states = more_energy_gradient(h, refs, ts, theta)
    oracle = expm(theta[2] * ts[2]) @ expm(theta[1] * ts[1]) @ expm(theta[0] * ts[0]) @ refs
    np.testing.assert_allclose(states, oracle, atol=1e-12)
    assert value == pytest.approx(np.trace(oracle.conj().T @ h @ oracle).real / 3)
    finite = []
    for j in range(3):
        delta = np.eye(3)[j] * 1e-6
        finite.append((more_energy_gradient(h, refs, ts, theta + delta)[0]
                       - more_energy_gradient(h, refs, ts, theta - delta)[0]) / 2e-6)
    np.testing.assert_allclose(gradient, finite, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(states.conj().T @ states, np.eye(3), atol=1e-12)
    # The equal-weight objective depends on the span, not a reference gauge.
    mixing = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    mixed_value, mixed_gradient, _ = more_energy_gradient(h, refs @ mixing, ts, theta)
    assert mixed_value == pytest.approx(value, abs=1e-12)
    np.testing.assert_allclose(mixed_gradient, gradient, atol=1e-12)


@pytest.mark.parametrize('bad', ['nonorthogonal', 'nonfinite', 'nonhermitian', 'noncubic', 'angle_shape'])
def test_shared_gradient_validates_its_mathematical_contract(bad):
    h = np.diag([0., 1.]).astype(complex)
    refs = np.eye(2, dtype=complex)
    t = np.array([[0., 1.], [-1., 0.]], complex)
    theta = [.1]
    if bad == 'nonorthogonal': refs[:, 1] = refs[:, 0]
    elif bad == 'nonfinite': refs[0, 0] = np.nan
    elif bad == 'nonhermitian': h[0, 1] = 1
    elif bad == 'noncubic': t *= 2
    else: theta = []
    with pytest.raises(ValueError):
        more_energy_gradient(h, refs, [t], theta)


def test_transition_reference_fanouts_recover_complex_projected_elements():
    from clifford_qc.algorithms.more_adapt import readout_bounds
    n, size = 4, 16
    rng = np.random.default_rng(193)
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    h = raw + raw.conj().T
    pool, _ = exchange_pools(n)
    u = expm(.23 * to_matrix(pool[0].generator)) @ expm(-.17 * to_matrix(pool[-2].generator))
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for left, right in ((12, 3), (12, 6), (3, 6), (6, 9)):
        pivot = next(q for q in range(n) if not left & (1 << (n - 1 - q))
                     and right & (1 << (n - 1 - q)))
        local_h = np.array([[1.]])
        for q in range(n): local_h = np.kron(local_h, hadamard if q == pivot else np.eye(2))
        references = []
        for imaginary in (False, True):
            state = local_h @ np.eye(size)[:, left]
            if imaginary:
                state = state * np.array([1j if word & (1 << (n - 1 - pivot)) else 1 for word in range(size)])
            for q in range(n):
                if q == pivot or not (left ^ right) & (1 << (n - 1 - q)): continue
                image = np.zeros(size, complex)
                for word, amplitude in enumerate(state):
                    target = word ^ (1 << (n - 1 - q)) if word & (1 << (n - 1 - pivot)) else word
                    image[target] += amplitude
                state = image
            expected = (np.eye(size)[:, left] + (1j if imaginary else 1) * np.eye(size)[:, right]) / np.sqrt(2)
            np.testing.assert_allclose(state, expected, atol=1e-12)
            references.append(u @ state)
        a, b = u[:, left], u[:, right]
        diagonal_mean = (np.vdot(a, h @ a).real + np.vdot(b, h @ b).real) / 2
        real = np.vdot(references[0], h @ references[0]).real - diagonal_mean
        imag = diagonal_mean - np.vdot(references[1], h @ references[1]).real
        assert complex(real, imag) == pytest.approx(np.vdot(a, h @ b), abs=1e-12)
    # Separate three-word Hamiltonian: identity needs no readout; the two
    # complex quadratures add six fanout CX gates per complete reference batch.
    bound = readout_bounds(n, {0: .3, 1: .2, 2: -.4, 3: .1}, [12, 3], 17)
    assert bound['reference_families'] == 4
    assert bound['independent_word_readout_circuits'] == 12
    assert bound['generic_cnot_bound'] == 3 * (4 * 17 + 6)
    assert bound['reference_x_gate_bound'] == 24


def test_full_reference_span_stops_on_cancellation_but_ritz_roots_are_exact():
    model = hubbard(2, U=2.1, t=.7)
    run = run_more_adapt(model, [3, 6, 9, 12], max_iterations=2, max_parameters=2)
    assert run['stopped_reason'] == 'gradient_threshold' and not run['trajectory']
    assert any(abs(value) > 1e-3 for values in run['screening'][0]['per_reference_scores'].values() for value in values)
    assert run['screening'][0]['gradient_norm'] < 1e-12
    np.testing.assert_allclose(run['root_energies'], run['sector_exact_root_energies'], atol=1e-12)
    assert run['resources']['screened_reference_gradients'] == 4 * run['resources']['screened_state_average_gradients']
    assert run['resources']['optimizer_energy_gradient_evaluations'] == 0


@pytest.mark.parametrize('bits', [[], [3, 3], [True, 3], [0, 3]])
def test_reference_occupations_are_distinct_and_in_the_declared_sector(bits):
    with pytest.raises(ValueError):
        run_more_adapt(hubbard(2), bits)


def test_numpy_occupations_are_accepted_without_losing_reference_identity():
    run = run_more_adapt(hubbard(2), np.array([3, 6, 9, 12], dtype=np.int64),
                         max_iterations=1, max_parameters=1)
    assert run['reference_bits'] == [3, 6, 9, 12]


def test_parameter_budget_charges_a_terminal_screen_without_an_extra_operator(toy, monkeypatch):
    from clifford_qc.algorithms import more_adapt as driver
    from clifford_qc.algorithms.optimize import OptimizeResult
    config, _ = toy
    model = fcidump_model(config['cases'][0]['path'], integral_tolerance=1e-12)
    def unresolved(objective, x, **kwargs):
        value, gradient = objective(x)
        return OptimizeResult(tuple(x), value, float(np.linalg.norm(gradient)), 0, 1, 'undeclared_test_stub')
    monkeypatch.setattr(driver, 'minimize_energy', unresolved)
    run = driver.run_more_adapt(model, [12, 3], max_iterations=3, max_parameters=1)
    assert run['stopped_reason'] == 'parameter_budget'
    assert len(run['trajectory']) == 1 and len(run['screening']) == 2
    assert not run['trajectory'][0]['optimizer_resolved']
    assert run['resources']['optimizer_energy_gradient_evaluations'] == 1
    assert run['resources']['screened_reference_gradients'] == 2 * len(run['pool_labels']) * 2


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    directory = tmp_path_factory.mktemp('more-undeclared-toy')
    path = directory / 'toy.FCIDUMP'
    path.write_text('''&FCI NORB=2, NELEC=2, MS2=0,
 ORBSYM=1,1, ISYM=1,
&END
 0.11 1 1 0 0
 0.21 2 2 0 0
 -0.7 1 2 0 0
 2.1 1 1 1 1
 1.6 2 2 2 2
 0.3 1 1 2 2
 0.15 1 2 1 2
 0.02 0 0 0 0
''')
    case = {'id': 'undeclared_toy', 'path': str(path), 'structure': structure(path)}
    config = {'schema': 'clifford_qc.phase17_more_config.v1', 'cases': [case],
              'arms': ['more_adapt', 'acase_mean'], 'roots': [2, 3],
              'more': {'max_iterations': 2, 'max_parameters': 2, 'maxiter': 60,
                       'gradient_threshold': 1e-6, 'optimizer_gtol': 1e-8,
                       'selection_tolerance': 1e-10, 'max_sector_dimension': 256},
              'acase': {'max_size': 2, 'min_lowering': 1e-10,
                        'min_orthogonality': 1e-8, 'tau_s': 1e-10,
                        'rel_tau': 0., 'max_condition': 1e12},
              'target_hartree': 1.6e-3, 'claim_boundary': 'undeclared_toy_only'}
    model = fcidump_model(path, integral_tolerance=1e-12)
    basis, _ = matrices(case['structure'])
    h, _ = hamiltonian(path, basis)
    cells = []
    for roots in config['roots']:
        bits = case['structure']['reference_bits'][str(roots)]
        for arm in config['arms']:
            run = (run_more_adapt(model, bits, **config['more']) if arm == 'more_adapt'
                   else run_acase_mean(model, bits, config['acase']))
            if arm == 'acase_mean':
                run['resources']['sector_spectrum_eigensolves'] = 2
                run['resources']['reachable_family_dimension'] = case['structure']['response_family_dimensions'][str(roots)]
            exact = np.linalg.eigvalsh(h)[:roots].tolist()
            cells.append({'case': case['id'], 'roots': roots, 'arm': arm,
                          'status': 'completed', 'run': run, 'exact_root_energies': exact,
                          'target_summary': target_summary(run, exact, config['target_hartree'])})
    record = declare_evidence({'schema': 'clifford_qc.phase17_more.v1',
                              'config_sha256': digest(config), 'claim_boundary': config['claim_boundary'],
                              'cells': cells, 'verdict': 'VALID', 'validation_problems': []}, EVIDENCE)
    return config, record


def test_both_arms_replay_from_occupation_matrices_and_charge_full_work(toy):
    config, record = toy
    assert check_record(record, config, history=False) == []
    for more, acase in (record['cells'][:2], record['cells'][2:]):
        np.testing.assert_allclose(more['run']['initial_root_energies'], acase['run']['initial_root_energies'], atol=1e-12)
        assert more['run']['resources']['optimizer_reference_energy_evaluations'] == more['roots'] * more['run']['resources']['optimizer_energy_gradient_evaluations']
        assert acase['run']['resources']['candidate_score_forms'] > 0
        assert acase['run']['resources']['optimizer_energy_gradient_evaluations'] == 0


def test_reference_manifest_uses_frontier_singles_and_declares_static_capacity(toy):
    config, _ = toy
    spec = config['cases'][0]['structure']
    assert spec['reference_bits'] == {'2': [12, 6], '3': [12, 6, 9]}
    assert spec['response_family_dimensions'] == {'2': 4, '3': 4}


@pytest.mark.parametrize('field', [
    'missing_cell', 'reordered_cells', 'root_energy', 'target', 'reference', 'quantum_claim',
    'more_gradient', 'more_per_reference_gradient', 'more_choice', 'more_call',
    'more_callback', 'more_warm_start', 'more_count', 'more_bool_count', 'more_circuit',
    'acase_pencil', 'acase_frame', 'acase_choice', 'acase_score', 'acase_pair',
    'acase_count', 'acase_generator', 'evidence', 'config_binding',
    'acase_actual_lowering', 'acase_rejections', 'acase_coupling', 'acase_word_count',
    'more_readout', 'acase_readout',
])
def test_independent_replay_rejects_corrupted_science_and_ledgers(toy, field):
    config, original = toy
    record = deepcopy(original)
    more, acase = record['cells'][:2]
    if field == 'missing_cell': record['cells'].pop()
    elif field == 'reordered_cells': record['cells'].reverse()
    elif field == 'root_energy': more['run']['root_energies'][0] += .1
    elif field == 'target': more['target_summary']['status'] = 'MADE_UP'
    elif field == 'reference': more['run']['reference_bits'].reverse()
    elif field == 'quantum_claim': more['run']['quantum_shots'] = 100
    elif field == 'more_gradient':
        label = next(iter(more['run']['screening'][0]['scores']))
        more['run']['screening'][0]['scores'][label] += .1
    elif field == 'more_per_reference_gradient':
        label = next(iter(more['run']['screening'][0]['per_reference_scores']))
        more['run']['screening'][0]['per_reference_scores'][label][0] += .1
    elif field == 'more_choice': more['run']['screening'][0]['selected_label'] = 'forged'
    elif field == 'more_call': more['run']['trajectory'][0]['optimizer_trace'][0]['objective'] += .1
    elif field == 'more_callback': more['run']['trajectory'][0]['optimizer_callback_indices'][-1] = -1
    elif field == 'more_warm_start': more['run']['trajectory'][0]['optimizer_trace'][0]['parameters'][-1] = .1
    elif field == 'more_count': more['run']['resources']['screened_reference_gradients'] += 1
    elif field == 'more_bool_count': more['run']['resources']['root_readout_eigensolves'] = True
    elif field == 'more_circuit': more['run']['trajectory'][0]['generic_ladder_cnots_per_reference'] += 2
    elif field == 'more_readout': more['run']['initial_readout']['generic_cnot_bound'] += 1
    elif field == 'acase_pencil': acase['run']['initial_pencil']['H'][0][0][0] = float(100).hex()
    elif field == 'acase_frame': acase['run']['initial_pencil']['coefficients'][0][0][0] = float(2).hex()
    elif field == 'acase_choice': acase['run']['trajectory'][0]['selected_label'] = 'forged'
    elif field == 'acase_score': acase['run']['trajectory'][0]['predicted_lowering'] += .1
    elif field == 'acase_pair': acase['run']['pair_ledger'][0][-1] += 1
    elif field == 'acase_count': acase['run']['resources']['root_score_forms'] += 1
    elif field == 'acase_actual_lowering': acase['run']['trajectory'][0]['actual_lowering'] += .1
    elif field == 'acase_rejections': acase['run']['trajectory'][0]['rejected_conditioning'] += 1
    elif field == 'acase_coupling': acase['run']['trajectory'][0]['residual_coupling'] += .1
    elif field == 'acase_word_count': acase['run']['trajectory'][0]['new_words'] += 1
    elif field == 'acase_readout': acase['run']['resources']['local_readout_cnot_bound'] = 1
    elif field == 'acase_generator': acase['run']['generators'][-1]['coefficients'][0][1] = float(1).hex()
    elif field == 'evidence': record['evidence']['root_energies_and_gradients']['scope'] = 'hardware advantage'
    elif field == 'config_binding': record['config_sha256'] = '0' * 64
    with_evidence_error = field == 'evidence'
    problems = check_record(record, config, history=False)
    assert problems, (field, with_evidence_error)


def test_outcome_free_cli_and_execution_refusals_do_not_run_a_solver(monkeypatch, tmp_path):
    from benchmarks import run_phase17_more as producer
    monkeypatch.setattr(producer, 'load_config', lambda: {})
    monkeypatch.setattr(producer, 'validate', lambda config: [])
    monkeypatch.setattr(producer, 'worker_cell', lambda *a: pytest.fail('default/refusal evaluated a new cell'))
    monkeypatch.setattr(producer, 'execute', lambda config: pytest.fail('default executed campaign'))
    assert producer.main([]) == 0
    assert not list(tmp_path.iterdir())


def test_missing_active_claim_refuses_internal_worker_before_science(monkeypatch, tmp_path):
    from benchmarks import run_phase17_more as producer
    monkeypatch.setattr(producer, 'load_config', lambda: {})
    monkeypatch.setattr(producer, 'validate', lambda config: [])
    monkeypatch.setattr(producer, 'claim_path', lambda: tmp_path / 'missing.claim')
    monkeypatch.setattr(producer, 'worker_cell', lambda *a: pytest.fail('unclaimed worker ran'))
    with pytest.raises(FileNotFoundError): producer.main(['--worker-cell', '0'])


def test_consumed_execution_claim_is_never_reset(monkeypatch, tmp_path):
    from benchmarks import archive_provenance
    from benchmarks import run_phase17_more as producer
    claim = tmp_path / 'consumed.claim'
    original = '{"status":"INVALID","observations":"retained"}'
    claim.write_text(original)
    monkeypatch.setattr(producer, 'claim_path', lambda: claim)
    monkeypatch.setattr(producer, 'RECORD', tmp_path / 'absent-record.json')
    monkeypatch.setattr(producer, 'worker_cell', lambda *a: pytest.fail('consumed campaign ran'))
    monkeypatch.setattr(archive_provenance, 'history_kind',
                        lambda *a, **kw: pytest.fail('consumed claim reached history inspection'))
    with pytest.raises(ValueError, match='already consumed'):
        producer.execute({})
    assert claim.read_text() == original


def test_published_more_record_refuses_before_claim_or_git_inspection(monkeypatch, tmp_path):
    from benchmarks import archive_provenance
    from benchmarks import run_phase17_more as producer
    record = tmp_path / 'published.json'
    record.write_text('{"retained":"observations"}')
    monkeypatch.setattr(producer, 'RECORD', record)
    monkeypatch.setattr(producer, 'claim_path',
                        lambda: pytest.fail('published record reached the Git claim lookup'))
    monkeypatch.setattr(archive_provenance, 'history_kind',
                        lambda *a, **kw: pytest.fail('published record reached history inspection'))
    with pytest.raises(ValueError, match='record exists'):
        producer.execute({})
    assert record.read_text() == '{"retained":"observations"}'


def test_unused_more_campaign_still_requires_full_clean_git_before_science(monkeypatch, tmp_path):
    from benchmarks import archive_provenance
    from benchmarks import run_phase17_more as producer
    monkeypatch.setattr(producer, 'RECORD', tmp_path / 'absent-record.json')
    monkeypatch.setattr(producer, 'claim_path', lambda: tmp_path / 'absent.claim')
    monkeypatch.setattr(producer, 'worker_cell', lambda *a: pytest.fail('unlicensed campaign ran'))

    def no_history(root, *, archive=False):
        assert archive is False
        raise ValueError('full Git history required')

    monkeypatch.setattr(archive_provenance, 'history_kind', no_history)
    with pytest.raises(ValueError, match='full Git history required'):
        producer.execute({})
    monkeypatch.setattr(archive_provenance, 'history_kind', lambda root: 'git')
    monkeypatch.setattr(producer, 'git', lambda *a: ' M dirty-source.py')
    with pytest.raises(ValueError, match='clean committed sources'):
        producer.execute({})
    assert not list(tmp_path.iterdir())


def test_fresh_process_receipt_matches_supervising_parent(monkeypatch):
    from benchmarks import run_phase17_more as producer
    original = producer.subprocess.Popen
    code = ('import json,os; print(json.dumps({"case":"undeclared_process_toy",'
            '"roots":2,"arm":"more_adapt","status":"completed",'
            '"worker":{"pid":os.getpid(),"parent_pid":os.getppid(),"peak_rss_bytes":1024}}))')
    def receipt(command, **kwargs):
        return original([sys.executable, '-c', code], **kwargs)
    monkeypatch.setattr(producer.subprocess, 'Popen', receipt)
    config = {'execution': {'rss_ceiling_bytes': 1024**3, 'timeout_seconds': 3, 'sample_seconds': .01}}
    result = producer.run_cell(0, {'id': 'undeclared_process_toy'}, 2, 'more_adapt', config)
    assert result['status'] == 'completed'
    assert result['worker']['pid'] == result['parent']['worker_pid']
    assert result['worker']['parent_pid'] == result['parent']['pid']


@pytest.mark.parametrize('limit,status', [('timeout_seconds', 'timeout'), ('rss_ceiling_bytes', 'memory_limit')])
def test_supervisor_records_limits_without_restarting_a_toy_worker(monkeypatch, limit, status):
    from benchmarks import run_phase17_more as producer
    original, launches = producer.subprocess.Popen, []
    def sleeper(command, **kwargs):
        launches.append(command)
        return original([sys.executable, '-c', 'import time; time.sleep(1)'], **kwargs)
    monkeypatch.setattr(producer.subprocess, 'Popen', sleeper)
    config = {'execution': {'rss_ceiling_bytes': 1024**3, 'timeout_seconds': 3, 'sample_seconds': .01}}
    config['execution'][limit] = .03 if limit == 'timeout_seconds' else 1
    result = producer.run_cell(0, {'id': 'undeclared_process_toy'}, 2, 'more_adapt', config)
    assert result['status'] == status and len(launches) == 1
    assert result['parent']['returncode'] != 0


@pytest.mark.parametrize('field', ['rss_type', 'pid_type', 'worker_reuse', 'threads',
                                  'python', 'source', 'negative_time', 'ceiling'])
def test_worker_identity_environment_and_observed_costs_are_checked(toy, field):
    import scipy
    config, original = toy
    config = deepcopy(config)
    config['execution'] = {'rss_ceiling_bytes': 1024 * 1024, 'timeout_seconds': 60,
                           'sample_seconds': .05}
    record = deepcopy(original)
    record['config_sha256'] = digest(config)
    record['provenance'] = {'git_sha': 'a' * 40, 'python': sys.version.split()[0],
                            'dependencies': {'numpy': np.__version__, 'scipy': scipy.__version__}}
    for i, cell in enumerate(record['cells']):
        cell['worker'] = {'pid': i + 100, 'parent_pid': 1, 'start_ticks': i + 1,
                          'git_sha': 'a' * 40, 'python': sys.version.split()[0],
                          'numpy': np.__version__, 'scipy': scipy.__version__,
                          'threads': {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                          'peak_rss_bytes': 1024, 'prepare_seconds': .01, 'algorithm_seconds': .02}
        cell['parent'] = {'pid': 1, 'worker_pid': i + 100, 'returncode': 0,
                          'peak_rss_bytes': 1024, 'wall_seconds': .03, **config['execution']}
    assert check_record(record, config, history=False) == []
    worker, parent = record['cells'][0]['worker'], record['cells'][0]['parent']
    if field == 'rss_type': worker['peak_rss_bytes'] = False
    elif field == 'pid_type': worker['pid'] = True
    elif field == 'worker_reuse': record['cells'][1]['worker'] = deepcopy(worker)
    elif field == 'threads': worker['threads']['OMP_NUM_THREADS'] = '2'
    elif field == 'python': worker['python'] = '3.11.1'
    elif field == 'source': worker['git_sha'] = 'b' * 40
    elif field == 'negative_time': parent['wall_seconds'] = -1
    else: parent['peak_rss_bytes'] = 2 * config['execution']['rss_ceiling_bytes']
    assert check_record(record, config, history=False)


@pytest.mark.parametrize('budget', [0, True, -1, 1.5])
def test_invalid_growth_budgets(budget):
    with pytest.raises(ValueError):
        run_more_adapt(hubbard(2), [3, 12], max_parameters=budget)
