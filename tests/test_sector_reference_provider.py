"""Sector ordering, immutable inputs, exact actions and multiplicity defects."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest

from clifford_qc.backends.sector_reference import SectorSpec,SectorStateReference,SectorActionProvider
from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.reference_states import SectorActionReference,UnsupportedReferenceOperation,require_exact_reference
from clifford_qc.subspace.bounded_actions import ActionMemoryLimit
from clifford_qc.subspace.providers import ActionProvider
from clifford_qc.subspace.sector_solver import spin_diagnostics


def provider(spin=0):
    m=hubbard((2,2),U=4,mu=0);spec=SectorSpec(8,4,0.)
    a=np.zeros(spec.dimension,complex);a[0]=1
    return SectorActionProvider(SectorStateReference(spec,a),m.hamiltonian,target_spin=spin)


def test_metadata_conventions_owned_arrays_and_capabilities():
    p=provider();r=p.reference
    assert isinstance(r,SectorActionReference) and isinstance(p,ActionProvider)
    assert r.metadata.n==8 and r.sector==(4,0.) and r.spec.dimension==36
    assert r.metadata.bit_order=='q0-leftmost'
    with pytest.raises(ValueError):r.amplitudes.flags.writeable=True
    with pytest.raises(ValueError):r.occupation_words.flags.writeable=True
    with pytest.raises(FrozenInstanceError):r.spec=SectorSpec(8,2,0.)
    with pytest.raises(UnsupportedReferenceOperation):require_exact_reference(r)
    with pytest.raises(AttributeError):p.target_spin=2
    with pytest.raises(TypeError):p.hamiltonian.terms[0]=3
    op=r.compile_operator(p.hamiltonian);op.backend.basis=np.array([0])
    assert len(r.occupation_words)==36


@pytest.mark.parametrize('precompute',[False,True])
def test_block_actions_match_independent_sector_restriction(precompute):
    p=provider();r=p.reference;p=SectorActionProvider(r,p.hamiltonian,precompute=precompute)
    rng=np.random.default_rng(42);block=rng.normal(size=(36,3))+1j*rng.normal(size=(36,3))
    matrix=SectorStatevectorBackend(8,4,0.).operator(p.hamiltonian).restrict(range(36))
    np.testing.assert_allclose(p.apply_hamiltonian(block),matrix@block,atol=1e-13)
    assert p.h_action_columns==3
    np.testing.assert_allclose(p.diagonal(),matrix.diagonal().real,atol=1e-14)
    assert r.storage()['density_coefficients']==0


def test_spin_expectation_can_hide_a_mixture_but_action_defect_cannot():
    singlet,quintet,triplet=provider(0),provider(2),provider(1)
    raw=np.random.default_rng(61).normal(size=36)
    s0=singlet.project_spin(raw);s0/=np.linalg.norm(s0)
    s2=quintet.project_spin(raw);s2/=np.linalg.norm(s2)
    diag=spin_diagnostics(triplet,np.sqrt(2/3)*s0+np.sqrt(1/3)*s2)
    assert diag['spin_squared_mean']==pytest.approx(2,abs=1e-12)
    assert diag['spin_action_defect']>2
    assert diag['particle_number_defect']==diag['spin_z_defect']==0
    projected=triplet.project_spin(raw);projected/=np.linalg.norm(projected)
    assert spin_diagnostics(triplet,projected)['spin_action_defect']<1e-12


@pytest.mark.parametrize('spec',[lambda:SectorSpec(64,2,0),lambda:SectorSpec(7,2,0),
    lambda:SectorSpec(8,3,0),lambda:SectorSpec(8,4,.25),lambda:SectorSpec(8,4,3),
    lambda:SectorSpec(8,4,0,'blocked'),lambda:SectorSpec(True,0,0)])
def test_bad_sectors(spec):
    with pytest.raises(ValueError):spec()


def test_reference_admission_precedes_sector_construction(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('basis allocated before admission')
    monkeypatch.setattr(SectorStatevectorBackend,'__init__',forbidden)
    with pytest.raises(ActionMemoryLimit):SectorStateReference(SectorSpec(24,12,0.),None,max_owned_bytes=2**20)


def test_reference_and_hamiltonian_snapshots_are_independent_of_callers():
    h=hubbard((2,2),mu=0).hamiltonian.to_mv();a=np.zeros(36,complex);a[0]=1
    r=SectorStateReference(SectorSpec(8,4,0.),a);p=SectorActionProvider(r,h)
    initial=p.apply_hamiltonian(r.amplitudes);a[:]=0;h.terms.clear()
    np.testing.assert_array_equal(p.apply_hamiltonian(r.amplitudes),initial)
    assert r.amplitudes[0]==1


@pytest.mark.parametrize('spin',[-1,.5,3,float('nan'),True])
def test_bad_total_spin_targets(spin):
    with pytest.raises(ValueError):provider(spin)
