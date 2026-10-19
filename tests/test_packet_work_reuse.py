"""Five-arm replay, whole-trajectory equivalence and adversarial evidence."""
from copy import deepcopy
import json
import numpy as np
import pytest
from benchmarks import run_packet_work_reuse as producer
from benchmarks import check_packet_work_reuse as checker
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.spin_packets import spin_squared_operator


@pytest.fixture(scope='module')
def synthetic():
    model = hubbard(4, t=1., U=3.)
    backend = SectorStatevectorBackend(8, 4, 0)
    operator = backend.operator(model.hamiltonian)
    h = operator.restrict(np.arange(backend.dimension))
    s2 = backend.operator(spin_squared_operator(8)).restrict(np.arange(backend.dimension))
    frame, _ = producer.singlet_frame(backend.basis, 8, s2, 1e-9)
    config = producer.load_design()
    config['input']['active_spatial_orbitals'] = [1, 2]
    config.update(maximum_columns=frame.shape[1], maximum_growth_steps=40,
                  test_n_qubits=8, test_n_electrons=4)
    results = {arm: producer.trajectory(operator.matvec, h.diagonal().real, backend.basis,
                   8, 240, frame, config, arm, s2) for arm in producer.ARMS}
    return config,h,backend.basis.tolist(),s2,frame,results


@pytest.mark.parametrize('arm', producer.ARMS)
def test_independent_replay_and_true_convergence(synthetic, arm):
    config,h,words,s2,frame,results = synthetic
    assert checker.verify_worker(results[arm], config,h,words,s2) == []
    assert abs(results[arm]['trajectory'][-1]['energy'] - np.linalg.eigvalsh(h)[0]) < 1e-8
    if arm in ('reuse','combined'):
        assert results[arm]['final_h_action_calls'] <= frame.shape[1]


def test_cache_preserves_entire_scientific_trajectory(synthetic):
    results = synthetic[-1]
    for direct,cached in [('original','reuse'),('shortlist','combined')]:
        a,b = [results[k]['trajectory'] for k in (direct,cached)]
        assert len(a) == len(b)
        assert [r['selection'] for r in a] == [r['selection'] for r in b]
        np.testing.assert_allclose([r['energy'] for r in a], [r['energy'] for r in b], atol=1e-11, rtol=0)
        np.testing.assert_allclose([r['residual_norm'] for r in a], [r['residual_norm'] for r in b], atol=1e-10, rtol=0)
    assert results['reuse']['final_h_action_calls'] < results['original']['final_h_action_calls']
    assert any(d['audit'] for d in results['combined']['decisions'])


@pytest.mark.parametrize('field', ['energy','residual','actions','cache','payload','indicator','shortlist','audit','evaluated','score','winner','loss','stop'])
def test_replay_rejects_tampering(synthetic, field):
    config,h,words,s2,_,results = synthetic
    result = deepcopy(results['combined'])
    row,decision = result['trajectory'][1],result['decisions'][0]
    if field == 'energy': row['energy'] += .01
    elif field == 'residual': row['residual_norm'] = 0
    elif field == 'actions': row['h_action_calls'] -= 1
    elif field == 'cache': row['frame_cache_indices'] = []
    elif field == 'payload': row['frame_actions_payload_bytes'] += 16
    elif field == 'indicator': decision['indicators'][next(iter(decision['indicators']))] += 1
    elif field == 'shortlist': decision['shortlist'] = []
    elif field == 'audit': decision['audit'] = not decision['audit']
    elif field == 'evaluated': decision['evaluated'] = []
    elif field == 'score': decision['scores'][next(iter(decision['scores']))] += 1
    elif field == 'winner': decision['selected'] = -1
    elif field == 'loss': decision['audit_score_loss'] = 1
    elif field == 'stop': result['stop_reason'] = 'fake_convergence'
    assert checker.verify_worker(result,config,h,words,s2)


def test_result_free_default_and_refuse_overwrite(monkeypatch,tmp_path):
    monkeypatch.setattr(producer, 'execute', lambda c: pytest.fail('science reached'))
    assert producer.main([]) == 0
    path = tmp_path/'existing.json'
    path.write_text('old')
    with pytest.raises(FileExistsError): producer.main(['--execute','--output',str(path)])
    assert path.read_text() == 'old'


def test_committed_campaign():
    if not (producer.ROOT / producer.RECEIPT).exists():
        pytest.skip('observation publication not yet committed')
    assert checker.verify(json.loads((producer.ROOT / producer.RECORD).read_text())) == []


def test_negligible_shortlist_gain_falls_back_and_prices_all_candidates(synthetic):
    config,h,words,s2,frame,_ = synthetic
    config = deepcopy(config)
    config.update(shortlist_fraction=.1, score_tie_tolerance=100., maximum_growth_steps=1)
    result = producer.trajectory(lambda v: h @ v, h.diagonal().real, np.asarray(words),
                                 8,240,frame,config,'shortlist',s2)
    decision = result['decisions'][0]
    assert decision['fallback'] and not decision['audit']
    assert len(decision['evaluated']) > len(decision['shortlist'])
    assert result['final_h_action_calls'] == result['trajectory'][0]['dimension'] + len(decision['evaluated'])
    assert checker.verify_worker(result,config,h,words,s2) == []


def test_timeout_campaign_keeps_denominators(monkeypatch):
    import subprocess
    from benchmarks import phase17_ceo_oracle as oracle
    from benchmarks import summarize_packet_work_reuse as report
    from clifford_qc.backends import sector_statevector
    monkeypatch.setattr(producer,'preflight', lambda c: ('1'*40,'2'*40,{}))
    monkeypatch.setattr(producer,'commit_binding', lambda root,commit,path: {'sha256': producer.sha256(root/path)})
    monkeypatch.setattr(producer,'stamp_record', lambda r,**kwargs:r)
    monkeypatch.setattr(oracle,'hamiltonian',lambda path,words:(np.diag([0.,1.]),{}))
    monkeypatch.setattr(sector_statevector,'sector_basis',lambda *args:np.array([0,1]))
    def timeout(*args,**kwargs): raise subprocess.TimeoutExpired('worker',120)
    monkeypatch.setattr(producer.subprocess,'run',timeout)
    record = producer.execute(producer.load_design())
    assert len(record['workers']) == 15
    assert all(w['status'] == 'timeout' and 'result' not in w for w in record['workers'])
    text = report.render(record)
    assert 'failed/timed out: 15' in text
    assert text.count('| 0/3 |') == 10


@pytest.mark.parametrize('field',['timing','rss','provenance','source'])
def test_observation_byte_binding_rejects_metadata_tampering(field):
    if not (producer.ROOT/producer.RECEIPT).exists():
        pytest.skip('observation publication not yet committed')
    record = json.loads((producer.ROOT/producer.RECORD).read_text())
    if field == 'timing': record['workers'][0]['process_wall_seconds'] += 1
    elif field == 'rss': record['workers'][0]['result']['final_peak_rss_bytes'] += 1
    elif field == 'provenance': record['provenance']['utc'] = 'changed'
    elif field == 'source': record['source_commit'] = '0'*40
    with pytest.raises(ValueError,match='inspected object'):
        checker.verify(record)


def test_bundle_restores_declared_history_from_baseline(monkeypatch,tmp_path):
    import shutil
    import subprocess
    if not (producer.ROOT/producer.RECEIPT).exists():
        pytest.skip('observation publication not yet committed')
    receipt = json.loads((producer.ROOT/producer.RECEIPT).read_text())
    root = tmp_path/'fresh'
    subprocess.run(['git','init','-q',str(root)],check=True)
    subprocess.run(['git','fetch','--quiet',str(producer.ROOT),receipt['base_commit']],cwd=root,check=True)
    subprocess.run(['git','checkout','--quiet','--detach','FETCH_HEAD'],cwd=root,check=True)
    for relative in set(producer.SOURCES) | {producer.CONFIG,producer.DECLARATION,producer.RECORD,producer.RECEIPT,producer.BUNDLE}:
        target = root/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(producer.ROOT/relative,target)
    refs = subprocess.run(['git','show-ref'],cwd=root,capture_output=True).stdout
    monkeypatch.setattr(checker,'ROOT',root)
    record = json.loads((root/producer.RECORD).read_text())
    checker.restore_and_bind(record)
    assert subprocess.run(['git','show-ref'],cwd=root,capture_output=True).stdout == refs
    bundle = root/producer.BUNDLE
    bundle.write_bytes(bundle.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='binding differs'):
        checker.restore_and_bind(record)
