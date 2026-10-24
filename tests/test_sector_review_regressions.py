"""PR 156 controls for complete roots, offsets, failure accounting and admission."""
from dataclasses import replace
import sys

import numpy as np
import pytest

from clifford_qc.backends.sector_reference import (
    SectorActionProvider, SectorSpec, SectorStateReference, sector_input_plan,
)
from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.pipeline import main, solve_prepared_sector
from clifford_qc.prepared import PreparedProblem
from clifford_qc.subspace.bounded_actions import BoundedActionSpace
from clifford_qc.subspace.sector_solver import (
    SectorSolveConfig, sector_memory_plan, solve_sector, spin_diagnostics,
)


def provider(shape=(2,2), offset=0):
    model=hubbard(shape,U=4,mu=0)
    spec=SectorSpec(model.n,model.metadata['n_electrons'],model.metadata['sz'])
    backend=SectorStatevectorBackend(spec.n,spec.n_electrons,spec.sz)
    ref=SectorStateReference(spec,backend.state_from_program(model.reference))
    return SectorActionProvider(ref,model.hamiltonian.to_mv()+MV.scalar(model.n,offset),precompute=True)


@pytest.mark.parametrize('columns',[1,2])
def test_complete_converged_recovery_beats_lower_energy_partial_or_unconverged_initial(monkeypatch,columns):
    p=provider()
    matrix=p.reference.compile_operator(p.hamiltonian).restrict(range(p.dimension))
    _,vectors=np.linalg.eigh(matrix)
    initial=vectors[:,:1]
    if columns==2:
        # Keep a true ground vector, but include a non-eigenvector second direction.
        other=np.random.default_rng(81).normal(size=p.dimension)
        other=p.project_spin(other);other-=initial[:,0]*np.vdot(initial[:,0],other)
        initial=np.column_stack((initial,other/np.linalg.norm(other)))
    original=BoundedActionSpace.solve;calls=0
    def lower_first(space):
        nonlocal calls
        result=original(space);calls+=1
        if calls==1:
            return replace(result,energies=(result.energies[0]-1e-12,*result.energies[1:]))
        return result
    monkeypatch.setattr(BoundedActionSpace,'solve',lower_first)
    result=solve_sector(p,initial=initial,config=SectorSolveConfig(roots=2,ncv=20))
    assert result.attempts[0]['energies'][0]<result.energies[0]
    assert not result.attempts[0]['numerically_converged']
    assert len(result.energies)==result.states.shape[1]==2
    assert result.status=='numerically_converged_root_unresolved'
    assert all(a['numerically_converged'] for a in result.attempts[1:])


def test_large_identity_offset_leaves_centered_arpack_problem_and_acceptance_unchanged():
    cfg=SectorSolveConfig(ncv=32,recovery_attempts=1)
    base=solve_sector(provider((2,3)),config=cfg)
    shifted=solve_sector(provider((2,3),1e4),config=cfg)
    assert base.status==shifted.status=='numerically_converged_root_unresolved'
    np.testing.assert_allclose(np.array(shifted.energies)-1e4,base.energies,atol=1e-10,rtol=0)
    assert shifted.resources['centered_spectral_bound']==base.resources['centered_spectral_bound']
    assert shifted.resources['arpack_relative_tolerance']==base.resources['arpack_relative_tolerance']
    assert shifted.resources['solve_h_action_columns']==base.resources['solve_h_action_columns']
    assert shifted.residuals[0].residual_norm+shifted.residuals[0].resolution_norm<cfg.residual_tolerance


@pytest.mark.parametrize('platform',['darwin','win32'])
def test_unavailable_rss_is_optional_but_a_configured_ceiling_fails_closed(monkeypatch,platform):
    import scipy.sparse.linalg  # Load platform-specific imports before simulation.
    p=provider();monkeypatch.setattr(sys,'platform',platform)
    result=solve_sector(p,config=SectorSolveConfig(recovery_attempts=0))
    assert result.status=='accuracy_unresolved'
    assert result.resources['rss']['available'] is False
    assert result.resources['rss']['peak_observed_bytes'] is None
    limited=solve_sector(p,config=SectorSolveConfig(max_rss_bytes=10**12))
    assert limited.status=='resource_limited'
    assert 'cannot be monitored' in limited.attempts[-1]['reason']


@pytest.mark.parametrize('stage',['arpack','projection','tracking','extension','initial'])
def test_failed_attempts_keep_charged_work_and_continue_to_a_successful_recovery(monkeypatch,stage):
    import scipy.sparse.linalg as sparse
    import clifford_qc.subspace.sector_solver as solver
    p=provider();calls=0
    if stage=='arpack':
        original=sparse.eigsh
        def fail(operator,**kwargs):
            nonlocal calls
            calls+=1
            if calls==1:
                operator.matvec(kwargs['v0'])
                raise sparse.ArpackError(3)
            return original(operator,**kwargs)
        monkeypatch.setattr(sparse,'eigsh',fail)
    elif stage in ('projection','initial'):
        original=BoundedActionSpace.solve
        def fail(space):
            nonlocal calls
            calls+=1
            if calls==(1 if stage=='initial' else 2):raise np.linalg.LinAlgError('controlled projected failure')
            return original(space)
        monkeypatch.setattr(BoundedActionSpace,'solve',fail)
    elif stage=='extension':
        original=BoundedActionSpace.extend
        def fail(space,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise ValueError('controlled extension failure')
            return original(space,*args,**kwargs)
        monkeypatch.setattr(BoundedActionSpace,'extend',fail)
    else:
        original=solver.root_tracking_overlap
        def fail(*args):
            nonlocal calls
            calls+=1
            if calls==1:raise ValueError('controlled tracking failure')
            return original(*args)
        monkeypatch.setattr(solver,'root_tracking_overlap',fail)
    result=solve_sector(p,config=SectorSolveConfig(ncv=20,recovery_attempts=2))
    assert result.status=='numerically_converged_root_unresolved'
    assert len(result.attempts)==3 and result.attempts[-1]['numerically_converged']
    failure=next(a for a in result.attempts if a['solver_status']=='numerical_failure')
    assert failure['error_type'] in ('ArpackError','LinAlgError','ValueError')
    assert result.resources['recovery_attempts_run']==2
    for key in ('h_action_columns','spin_action_columns','spin_projections','diagonal_word_scans'):
        assert result.resources[key]==sum(a[key] for a in result.attempts)
    assert failure['h_action_columns']>0


def test_resets_count_even_when_recovery_has_zero_spin_weight(monkeypatch):
    import scipy.sparse.linalg as sparse
    p=provider()
    monkeypatch.setattr(sparse,'eigsh',lambda *a,**kw:(np.zeros(1),np.zeros((p.dimension,1))))
    result=solve_sector(p,config=SectorSolveConfig(recovery_attempts=2))
    assert [a['solver_status'] for a in result.attempts[1:]]==['zero_spin_weight']*2
    assert result.resources['projection_resets']==3
    assert result.resources['projection_restarts']==2


def test_projection_report_uses_assessed_span_and_provider_costs_are_per_solve():
    p=provider();cfg=SectorSolveConfig(ncv=20,recovery_attempts=0)
    block=np.eye(p.dimension,dtype=complex)[:,:6]
    first=solve_sector(p,initial=block,config=cfg)
    second=solve_sector(p,initial=block,config=cfg)
    projection=first.attempts[0]['projection']
    assert projection['basis_size']>len(first.energies)
    report=first.as_dict()['convergence'][0]
    for name,value in projection.items():assert report[name]==value
    for key in ('h_action_columns','spin_action_columns','spin_projections','diagonal_word_scans'):
        assert first.resources[key]==second.resources[key]
        assert second.resources['provider_lifetime'][key]==2*first.resources[key]
    assert p.reference==p.reference and first==first
    another=SectorStateReference(p.reference.spec,p.reference.amplitudes)
    assert another!=p.reference and replace(first)!=first
    assert len({p.reference,another,first,replace(first)})==4


def test_operator_sector_projection_is_an_explicit_opt_in():
    p=provider()
    operator=MV.word(p.reference.n,'X'+'I'*(p.reference.n-1))
    with pytest.raises(ValueError,match='does not conserve'):p.reference.apply_operator(operator)
    np.testing.assert_array_equal(p.reference.apply_operator(operator,project_to_sector=True),0)


@pytest.mark.parametrize('roots,spin',[(40,0),(2,2),(1,3)])
def test_prepared_dimension_and_spin_validation_precedes_basis_construction(monkeypatch,roots,spin):
    prepared=PreparedProblem.from_model(hubbard((2,2),U=4,mu=0),preparation={"source":"review regression"})
    def forbidden(*a,**kw):raise AssertionError('basis allocated before validation')
    monkeypatch.setattr(SectorStatevectorBackend,'__init__',forbidden)
    result,record=solve_prepared_sector(prepared,target_spin=spin,config=SectorSolveConfig(roots=roots,ncv=50))
    assert result is None and record['status']=='invalid_configuration'
    assert record['admission_plan'] is None


def test_invalid_cli_rss_headroom_produces_a_clear_record(tmp_path):
    path=tmp_path/'prepared.json';out=tmp_path/'run.json'
    PreparedProblem.from_model(hubbard((2,2),U=4,mu=0),preparation={"source":"review regression"}).save(path)
    main(['solve-sector',str(path),'--output',str(out),'--max-rss-bytes','100','--rss-headroom-bytes','100'])
    import json
    record=json.loads(out.read_text())
    assert record['status']=='invalid_configuration' and 'headroom' in record['reason']
    main(['solve-sector',str(path),'--output',str(out),'--precompute','off','--recovery-attempts','0'])
    assert json.loads(out.read_text())['resources']['precompute'] is False


def test_auto_precompute_and_fallback_share_the_direct_solver_admission_plan():
    model=hubbard((2,2),U=4,mu=0);prepared=PreparedProblem.from_model(model,preparation={"source":"review regression"})
    spec=SectorSpec(8,4,0);cfg=SectorSolveConfig(recovery_attempts=0)
    plans=[]
    for precompute in (False,True):
        payload,_=sector_input_plan(spec,model.hamiltonian,precompute=precompute,max_owned_bytes=cfg.max_owned_bytes)
        plans.append(sector_memory_plan(spec,cfg,target_spin=0,input_bytes=payload['admission_bytes']))
    budget=(plans[0].planned_bytes+plans[1].planned_bytes)//2
    for limit,mode in ((cfg.max_owned_bytes,True),(budget,False)):
        result,record=solve_prepared_sector(prepared,config=replace(cfg,max_owned_bytes=limit))
        assert result.resources['precompute'] is mode
        assert record['preparation_memory_plan']==result.resources['memory_plan']
        assert result.resources['input_payload_bytes']<=result.resources['input_plan']['retained_bound_bytes']
    result,record=solve_prepared_sector(prepared,config=replace(cfg,max_owned_bytes=budget),precompute=True)
    assert result is None and record['status']=='resource_limited'


def test_spin_validation_avoids_mv_cross_products_and_number_diagnostics_avoid_basis_scans(monkeypatch):
    import clifford_qc.subspace.spin_packets as spin_module
    from clifford_qc.pauli_kernel import _word_mul_unchecked
    from clifford_qc.backends.sector_reference import _spin_conservation
    p=provider();spin=spin_module.spin_squared_operator(p.reference.n)
    before=_word_mul_unchecked.cache_info().currsize
    details=_spin_conservation(p.hamiltonian,spin.to_mv())
    assert _word_mul_unchecked.cache_info().currsize==before
    assert details['workspace_coefficient_model_bytes']<=p.resources()['input_plan']['spin_validation_workspace_bytes']
    monkeypatch.setattr(spin_module,'spin_squared_operator',lambda n:spin)
    def forbidden(*a,**kw):raise AssertionError('MV product/basis scan')
    monkeypatch.setattr(MV,'__mul__',forbidden)
    other=SectorActionProvider(p.reference,p.hamiltonian,precompute=True)
    with pytest.raises(ValueError,match='total-spin'):
        SectorActionProvider(p.reference,p.hamiltonian+MV.word(8,'ZIIIIIII'),precompute=True)
    monkeypatch.setattr(np,'fromiter',forbidden)
    assert spin_diagnostics(other,other.reference.amplitudes)['particle_number_defect']==0
