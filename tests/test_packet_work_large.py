"""Larger-run plumbing checked on small physical occupation sectors."""
from copy import deepcopy
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from benchmarks import run_packet_work_large as producer
from benchmarks import check_packet_work_large as checker
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard


def small_config():
    c=deepcopy(producer.load_design())
    c['model']['shape']=[2,2]
    c.update(n_qubits=8,n_electrons=4,sector_dimension=36,singlet_dimension=20,
             active_spatial_orbitals=[1,2],maximum_columns=8,maximum_growth_steps=5)
    return c


def test_independent_sparse_hubbard_matches_native_and_reference():
    c=small_config();h,words=checker.occupation_hubbard(c)
    model=hubbard((2,2),t=1.,U=4.,mu=0.)
    backend=SectorStatevectorBackend(8,4,0)
    native=backend.operator(model.hamiltonian).restrict(np.arange(36))
    assert words==backend.basis.tolist()
    np.testing.assert_allclose(h.toarray(),native,atol=1e-14)
    reference=checker.ground_reference(h,c,words)
    assert reference['energy']==pytest.approx(np.linalg.eigvalsh(native)[0],abs=1e-11)
    assert max(reference['residual_norms'])<1e-9
    assert max(reference['spin_action_norms'])<1e-8


@pytest.fixture(scope='module')
def worker_results():
    c=small_config()
    # Only the clean-history boundary is bypassed; build the native model,
    # singlet setup, CSR and original five policy paths on a physical sector.
    original=producer.preflight
    producer.preflight=lambda config:None
    try:results={a:producer.worker(a,c) for a in producer.ARMS}
    finally:producer.preflight=original
    return c,results


@pytest.mark.parametrize('arm',producer.ARMS)
def test_worker_uses_same_policy_and_independent_frame(worker_results,arm):
    c,results=worker_results;h,words=checker.occupation_hubbard(c)
    s2=checker.independent_spin_squared(words,c['n_qubits'])
    frame=checker.independent_frame(words,c['n_qubits'],s2,c['spin_tolerance'])
    result=results[arm]
    for spin in (s2,csr_matrix(s2)):
        assert checker.verify_worker(result,producer.trajectory_config(c),h,words,spin,
            frame_data=frame,n_qubits=8,n_electrons=4)==[]
    assert result['setup']['native_probe_sha256']==producer.probe_digest(h)
    assert result['setup']['csr_payload_bytes']==producer.csr_bytes(h)


def test_repeat_reuse_strips_only_observed_costs(worker_results):
    _,results=worker_results
    original=results['combined'];changed=deepcopy(original)
    changed['solver_wall_seconds']+=1;changed['trajectory'][0]['peak_rss_bytes']+=1
    assert checker.scientific_view(original)==checker.scientific_view(changed)
    changed['trajectory'][0]['h_action_calls']+=1
    assert checker.scientific_view(original)!=checker.scientific_view(changed)


@pytest.mark.parametrize('field',['process','setup','prefix','terminal','rss'])
def test_repeat_costs_are_validated_separately(worker_results,field):
    _,results=worker_results
    entry={'status':'ok','process_wall_seconds':100.,'result':deepcopy(results['combined'])}
    if field=='process':entry['process_wall_seconds']=float('nan')
    elif field=='setup':entry['result']['setup']['wall_seconds']=-1
    elif field=='prefix':entry['result']['trajectory'][1]['wall_seconds']=-1
    elif field=='terminal':entry['result']['solver_wall_seconds']=-1
    elif field=='rss':entry['result']['trajectory'][1]['peak_rss_bytes']=0
    assert checker.validate_costs(entry)


def test_default_is_result_free_and_never_overwrites(monkeypatch,tmp_path):
    monkeypatch.setattr(producer,'execute',lambda c:pytest.fail('science reached'))
    assert producer.main([])==0
    path=tmp_path/'old.json';path.write_text('old')
    with pytest.raises(FileExistsError):producer.main(['--execute','--output',str(path)])
    assert path.read_text()=='old'


def test_committed_large_campaign_and_report():
    if not (producer.ROOT/producer.RECEIPT).exists():pytest.skip('large observations not published yet')
    from benchmarks import summarize_packet_work_large as report
    record=json.loads((producer.ROOT/producer.RECORD).read_text())
    assert checker.verify(record)==[]
    assert report.REPORT.read_text()==report.render(record)
