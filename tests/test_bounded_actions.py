"""Admission, fixed buffers, incremental borders, snapshots and RSS boundaries."""
import numpy as np
import pytest

from clifford_qc.subspace.bounded_actions import ActionMemoryLimit,ActionMemoryPlan,BoundedActionSpace,RSSMonitor


def test_admission_precedes_any_array_allocation(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('allocation before admission')
    monkeypatch.setattr(np,'empty',forbidden)
    with pytest.raises(ActionMemoryLimit,match='before allocation'):
        ActionMemoryPlan.admit(853776,512,max_owned_bytes=2**30)


def test_fixed_buffers_reuse_actions_and_form_only_new_borders(monkeypatch):
    rng=np.random.default_rng(12)
    a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8));h=a+a.conj().T
    q,_=np.linalg.qr(rng.normal(size=(8,4))+1j*rng.normal(size=(8,4)))
    calls=[]
    def apply(v): calls.append(v.copy());return h@v
    plan=ActionMemoryPlan.admit(8,4,roots=2)
    space=BoundedActionSpace(apply,plan,monitor=RSSMonitor(plan,read_rss=lambda:100))
    buffers=(id(space._basis),id(space._actions),space.payload_bytes)
    original=np.column_stack
    def forbidden(columns,*args,**kwargs):
        if any(np.shape(c)[0]==8 for c in columns):raise AssertionError('history stacked')
        return original(columns,*args,**kwargs)
    monkeypatch.setattr(np,'column_stack',forbidden)
    space.extend(q[:,:2]);old=space.matrices();old_result=space.solve()
    space.extend(q[:,2:],actions=h@q[:,2:])
    assert (id(space._basis),id(space._actions),space.payload_bytes)==buffers
    s,projected=space.matrices()
    np.testing.assert_allclose(s,q.conj().T@q,atol=1e-14)
    np.testing.assert_allclose(projected,q.conj().T@h@q,atol=1e-14)
    np.testing.assert_array_equal(s[:2,:2],old[0])
    np.testing.assert_array_equal(projected[:2,:2],old[1])
    assert len(calls)==space.h_action_columns==2
    result=space.solve();states,actions=space.ritz_actions(result,2)
    np.testing.assert_allclose(actions,h@states,atol=1e-13)
    snapshot=states.copy();space.restart(states,actions)
    assert space.restarts==1 and space.size==2 and len(calls)==2
    np.testing.assert_array_equal(states,snapshot)
    assert old_result.coefficients.shape==(2,2)
    with pytest.raises(ValueError):space.basis[:]=1


def test_capacity_failure_cannot_apply_more_actions():
    calls=[]
    space=BoundedActionSpace(lambda v:calls.append(1) or v,ActionMemoryPlan.admit(4,1))
    space.extend(np.eye(4)[:,:1])
    with pytest.raises(ActionMemoryLimit):space.extend(np.eye(4)[:,1:2])
    assert len(calls)==1


def test_bad_extension_and_restart_do_not_change_the_live_span():
    space=BoundedActionSpace(lambda v:v,ActionMemoryPlan.admit(4,3,roots=2))
    space.extend(np.eye(4)[:,:1]);original=space.matrices()
    with pytest.raises(ValueError):space.extend(np.eye(4)[:,:1])
    with pytest.raises(ValueError):space.restart(np.ones((4,2)),np.ones((4,2)))
    assert space.size==1
    np.testing.assert_array_equal(space.matrices(),original)


def test_rss_and_owned_admission_are_separate_and_sampled():
    plan=ActionMemoryPlan.admit(4,2,max_owned_bytes=100000,max_rss_bytes=1000,rss_headroom_bytes=100)
    values=iter([800,850,1001]);monitor=RSSMonitor(plan,read_rss=lambda:next(values))
    monitor.check(admission=True);monitor.check()
    with pytest.raises(ActionMemoryLimit,match='RSS'):monitor.check()
    assert monitor.as_dict()['peak_observed_bytes']==1001 and monitor.as_dict()['samples']==3
    assert plan.planned_bytes<=100000
    with pytest.raises(ActionMemoryLimit):RSSMonitor(plan,read_rss=lambda:901).check(admission=True)


@pytest.mark.parametrize('options',[{'dimension':True,'capacity':1},{'dimension':4,'capacity':5},
    {'dimension':4,'capacity':2,'roots':3},{'dimension':4,'capacity':2,'ncv':2},
    {'dimension':4,'capacity':2,'max_owned_bytes':0},
    {'dimension':4,'capacity':2,'max_rss_bytes':10,'rss_headroom_bytes':10}])
def test_invalid_memory_plans(options):
    with pytest.raises(ValueError):ActionMemoryPlan.admit(**options)
