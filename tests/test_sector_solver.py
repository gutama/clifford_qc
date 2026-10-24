"""Root recovery, cached residuals, failed attempts and matched historical spans."""
import base64
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from clifford_qc.backends.sector_reference import SectorSpec,SectorStateReference,SectorActionProvider
from clifford_qc.backends.sector_statevector import SectorStatevectorBackend,SectorOperator
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.pipeline import solve_prepared_sector,main
from clifford_qc.prepared import PreparedProblem
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.subspace.bounded_actions import ActionMemoryPlan,BoundedActionSpace
from clifford_qc.subspace.convergence import ConvergenceConfig,GroundStateReference,residual_convergence_report
from clifford_qc.subspace.fermionic_generators import determinant_excitations
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.providers import ExactProjectedProvider
from clifford_qc.subspace.sector_solver import action_residual,solve_sector,SectorSolveConfig,root_tracking_overlap

ROOT=Path(__file__).resolve().parents[1]


def model_provider(model):
    spec=SectorSpec(model.n,model.metadata['n_electrons'],model.metadata['sz'])
    backend=SectorStatevectorBackend(spec.n,spec.n_electrons,spec.sz)
    return SectorActionProvider(SectorStateReference(spec,backend.state_from_program(model.reference)),model.hamiltonian)


def test_direct_action_residual_is_normalized_offset_safe_and_uses_shared_report():
    offset=1e7;h=np.array([[offset,1.],[1.,offset+2.]])
    state=np.array([2.,0.]);r=action_residual(state,h@state,offset)
    assert r.residual_norm==1.
    report=residual_convergence_report(r,ConvergenceConfig(2.),basis_size=1,effective_rank=1,
        condition_number=1.,residual_resolution=r.resolution_norm)
    assert report.residual_status=='within_tolerance' and report.ground_state_status=='not_assessed'
    assert report.as_dict()['residual']['source']=='cached_actions'
    exact=action_residual(np.array([1.,0.]),np.array([offset,0.]),offset)
    unresolved=residual_convergence_report(exact,ConvergenceConfig(1e-12),basis_size=1,
        effective_rank=1,condition_number=1.,residual_resolution=exact.resolution_norm)
    assert unresolved.residual_status=='unresolved'
    with pytest.raises(ValueError):action_residual(np.zeros(2),np.zeros(2),0)


@pytest.fixture(scope='module')
def recovered_primary():
    archive=json.loads((ROOT/'benchmarks/reference_results/packet_budget_convergence.json').read_text())
    worker=next(w for w in archive['workers'] if w['input']=='hubbard_2x4_u4' and w['arm']=='davidson')
    payload=archive['scientific_payloads'][worker['scientific_sha256']]
    old=json.loads(gzip.decompress(base64.b64decode(payload['payload'])))
    final=old['final_state']
    state=np.frombuffer(gzip.decompress(base64.b64decode(final['gzip_base64'])),dtype='<c16').copy()
    model=hubbard((2,4),t=1.,U=4.,mu=0.,periodic=False)
    provider=SectorActionProvider(SectorStateReference(SectorSpec(16,8,0.),state),model.hamiltonian)
    result=solve_sector(provider,config=SectorSolveConfig(ncv=48,residual_tolerance=2e-8))
    return old,archive['references']['hubbard_2x4_u4'],provider,result


def test_observed_2x4_excited_singlet_is_retained_then_recovered_without_an_oracle(recovered_primary):
    old,spectrum,provider,result=recovered_primary;initial=result.attempts[0]
    assert initial['energies'][0]==pytest.approx(old['trajectory'][-1]['energy'],abs=1e-12)
    assert initial['residual_norms'][0]<2e-8
    assert initial['numerically_converged'] and initial['spin_consistent']
    assert initial['ground_state_status']=='not_established'
    assert initial['energies'][0]-spectrum['singlet_reference']>.99
    assert result.energies[0]==pytest.approx(spectrum['singlet_reference'],abs=1e-9)
    assert len(result.attempts)==3
    assert all(a['numerically_converged'] for a in result.attempts[1:])
    assert all(a['spin_consistent'] for a in result.attempts)
    assert result.status=='numerically_converged_root_unresolved'
    assert result.as_dict()['ground_state_status']=='not_established'
    assert result.resources['recovery_attempts_run']==2
    assert result.resources['h_action_columns']==1+sum(a['h_action_columns'] for a in result.attempts[1:])
    assert result.resources['spin_action_columns']>result.resources['h_action_columns']
    assert result.resources['memory_plan']['planned_bytes']<=result.resources['config']['max_owned_bytes']
    assert max(a['spin_action_defect'] for a in result.spin)<1e-8
    ref=GroundStateReference(spectrum['singlet_reference'],
        spectrum['singlet_reference']+spectrum['singlet_gap_to_next'],
        'unchanged packet-budget independent singlet reference, Hubbard 2x4 U4')
    assessed=result.as_dict(ground_reference=ref)
    assert assessed['convergence'][0]['ground_state_status']=='ground_dominated'
    assert not assessed['stopping_uses_spectral_oracle']
    np.testing.assert_allclose(provider.apply_hamiltonian(result.states[:,0]),result.actions[:,0],atol=1e-12)


def test_sector_solve_forbids_density_h_squared_and_dense_sector_construction(monkeypatch):
    provider=model_provider(hubbard((2,2),U=4,mu=0))
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.subspace.second_moment import SecondMomentBank
    def forbidden(*args,**kwargs):raise AssertionError('density/H squared/dense/oracle construction')
    for cls,name in [(ExactMVBackend,'__init__'),(SecondMomentBank,'__init__'),(MV,'__mul__'),
        (SectorOperator,'restrict'),(SectorOperator,'to_csr'),(SectorStatevectorBackend,'ground_state')]:
        monkeypatch.setattr(cls,name,forbidden)
    result=solve_sector(provider,config=SectorSolveConfig(ncv=16,recovery_attempts=1))
    assert result.status=='numerically_converged_root_unresolved'
    assert result.residuals[0].residual_norm<1e-8
    before=provider.h_action_columns;result.as_dict();result.as_dict()
    assert provider.h_action_columns==before
    inconsistent=replace(result,spin=({'particle_number_defect':0.,'spin_z_defect':0.,'spin_action_defect':1.},))
    with pytest.raises(ValueError,match='quantum-number'):
        inconsistent.as_dict(ground_reference=GroundStateReference(-3.,-2.,'independent test spectrum'))
    with pytest.raises(ValueError):result.states.flags.writeable=True


def test_recovery_starts_and_subspace_tracking_are_deterministic():
    p=model_provider(hubbard((2,2),U=4,mu=0));cfg=SectorSolveConfig(roots=2,ncv=20,recovery_attempts=2)
    one=solve_sector(p,config=cfg);two=solve_sector(p,config=cfg)
    np.testing.assert_allclose(one.energies,two.energies,atol=1e-11)
    assert one.status==two.status=='numerically_converged_root_unresolved'
    np.testing.assert_allclose(np.linalg.svd(one.states.conj().T@two.states,compute_uv=False),1.,atol=1e-10)
    assert one.attempts[-1]['tracking_principal_cosines'] is not None
    assert one.resources['solve_h_action_columns']==two.resources['solve_h_action_columns']


def test_iteration_failures_and_zero_spin_starts_remain_in_unconditional_results(monkeypatch):
    import scipy.sparse.linalg as sparse
    def fail(*args,**kwargs):
        raise sparse.ArpackNoConvergence('controlled iteration limit',np.array([]),np.empty((36,0),complex))
    monkeypatch.setattr(sparse,'eigsh',fail)
    p=model_provider(hubbard((2,2),U=4,mu=0));result=solve_sector(p,config=SectorSolveConfig(recovery_attempts=2))
    assert result.status=='accuracy_unresolved'
    assert [a['solver_status'] for a in result.attempts[1:]]==['iteration_limit']*2
    assert result.resources['recovery_attempts_run']==2
    assert result.as_dict()['ground_state_status']=='not_established'


def test_monitored_rss_breach_is_a_failed_result_with_work_preserved():
    p=model_provider(hubbard((2,2),U=4,mu=0))
    result=solve_sector(p,config=SectorSolveConfig(max_rss_bytes=1000),
                        read_rss=lambda:100 if p.h_action_columns==0 else 2000)
    assert result.status=='resource_limited' and result.attempts[-1]['kind']=='limit_breach'
    assert result.resources['rss']['peak_observed_bytes']==2000
    assert result.resources['solve_h_action_columns']>0


@pytest.mark.parametrize('kind',['h4','lih','hubbard'])
def test_old_small_spans_match_sector_action_projection_and_admission(kind):
    if kind=='hubbard':model=hubbard((2,2),U=4,mu=0)
    else:model=fcidump_model(ROOT/'benchmarks/data'/('h4_sto3g_r0.9.FCIDUMP' if kind=='h4' else 'lih_sto3g_r1.5949_cas4e4o.FCIDUMP'))
    p=model_provider(model);ref=DeterminantReference.from_program(model.reference,arithmetic='legacy_trace')
    generators=[identity_generator(model.n),*determinant_excitations(model.n,ref.occupied)[:6]]
    expected=ExactProjectedProvider(ref,model.hamiltonian,generators)
    columns=np.column_stack([p.reference.apply_operator(g.mv) for g in generators])
    plan=ActionMemoryPlan.admit(p.dimension,len(generators),input_bytes=p.input_payload_bytes)
    space=BoundedActionSpace(p.apply_hamiltonian,plan);space.extend(columns)
    for actual,target in zip(space.matrices(),expected.matrices()):
        np.testing.assert_allclose(actual,target,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(space.solve().energies,expected.solve().energies,atol=1e-10,rtol=1e-12)
    assert space.payload_bytes<=plan.planned_bytes


def test_prepared_workflow_cli_and_preallocation_failure(tmp_path,monkeypatch):
    model=fcidump_model(ROOT/'benchmarks/data/h4_sto3g_r0.9.FCIDUMP')
    prepared=PreparedProblem.from_model(model,preparation={'source':'sector acceptance'})
    path=tmp_path/'prepared.json';prepared.save(path);output=tmp_path/'solve.json'
    main(['solve-sector',str(path),'--output',str(output),'--recovery-attempts','1','--ncv','16'])
    record=json.loads(output.read_text())
    assert record['status']=='numerically_converged_root_unresolved'
    assert record['preparation_seconds']>=0 and not record['stopping_uses_spectral_oracle']
    def forbidden(*args,**kw):raise AssertionError('sector allocated before owned admission')
    monkeypatch.setattr(SectorStatevectorBackend,'__init__',forbidden)
    result,record=solve_prepared_sector(prepared,config=SectorSolveConfig(max_owned_bytes=1))
    assert result is None and record['status']=='resource_limited'


def test_small_sector_fallback_is_explicit_and_its_reference_is_still_separate():
    p=SectorActionProvider(SectorStateReference(SectorSpec(2,2,0.),np.ones(1,complex)),MV.scalar(2,1))
    result=solve_sector(p)
    assert result.attempts[-1]['kind']=='complete_small_sector'
    assert result.status=='numerically_converged_root_unresolved'
    assert result.resources['memory_plan']['dimension']==1
    assert result.resources['projection_resets']==2
    assert result.resources['projection_restarts']==1


@pytest.mark.parametrize('options',[{'ncv':2},{'recovery_attempts':17},{'max_iterations':0},
    {'roots':True},{'residual_tolerance':-1},{'max_owned_bytes':0}])
def test_invalid_solver_configuration(options):
    with pytest.raises(ValueError):SectorSolveConfig(**options)


def test_zero_spin_initial_block_is_priced_and_recovered_in_the_requested_triplet_sector():
    m=hubbard((2,2),U=4,mu=0);a=np.zeros(36,complex);a[0]=1
    p=SectorActionProvider(SectorStateReference(SectorSpec(8,4,0.),a),m.hamiltonian,target_spin=1.)
    result=solve_sector(p,config=SectorSolveConfig(ncv=20,recovery_attempts=1))
    assert result.attempts[0]['solver_status']=='zero_spin_weight'
    assert result.status=='numerically_converged_root_unresolved'
    assert result.spin[0]['spin_squared_mean']==pytest.approx(2,abs=1e-10)
    assert result.spin[0]['spin_action_defect']<1e-9


def test_degenerate_block_tracking_survives_unitary_mixing_and_ray_phases():
    q=np.eye(5,dtype=complex)[:,:2];u=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    tracked=root_tracking_overlap(q,q@u)
    np.testing.assert_allclose(tracked['principal_cosines'],1,atol=1e-14)
    np.testing.assert_allclose(tracked['ray_overlaps_squared'],.5,atol=1e-14)
    assert root_tracking_overlap(q[:,:1],q[:,:1]*1j)['principal_cosines']==[1.]
