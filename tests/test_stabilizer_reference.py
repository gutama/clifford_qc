"""Signed stabilizer membership, reference ownership and fermionic boundaries."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.fermion import number_op_pauli
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.graph_state import NativeClifford
from clifford_qc.multivector import MV
from clifford_qc.reference_states import StabilizerReference,require_exact_reference
from clifford_qc.subspace.providers import ExactProjectedProvider
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.spin_packets import spin_squared_operator


@pytest.mark.parametrize('n',range(1,5))
def test_every_small_word_matches_independent_evolved_density(n):
    rng=np.random.default_rng(531+n);gates=[]
    for _ in range(5*n):
        q=int(rng.integers(n));gates.append(('H' if rng.integers(2) else 'S',(q,)))
        if n>1:gates.append(('CZ',tuple(map(int,rng.choice(n,2,replace=False)))))
    native=NativeClifford(n,tuple(gates))
    generators=[]
    for q in range(n):
        sign,word=native.conjugate(PauliWord(n,3 << (2*q)))
        generators.append((word,int(sign.real)))
    ref=StabilizerReference(n,generators)
    rho=ExactMVBackend().state(native.program())
    assert require_exact_reference(ref) is ref
    for code in range(4**n):
        expected=(2**n)*MV(n,{code:1}).trace_pairing(rho)
        assert ref.word_expectation(code)==pytest.approx(expected,abs=1e-10)
    assert ref.word_expectation(0)==1


def test_physical_identity_is_canonical_across_basis_changes_and_signs_are_exact():
    first=StabilizerReference(2,[('XX',1),('ZZ',1)])
    second=StabilizerReference(2,[('YY',-1),('XX',1)])
    assert first.metadata.identity==second.metadata.identity
    assert first.word_expectation(PauliWord.from_label('YY').code)==-1
    assert first.metadata.identity!=StabilizerReference(2,[('XX',-1),('ZZ',1)]).metadata.identity
    assert first.metadata.identity!=StabilizerReference(2,[('XX',1),('ZZ',1)],spin_ordering='qubits').metadata.identity
    with pytest.raises(FrozenInstanceError):first._rows=()
    with pytest.raises(FrozenInstanceError):first.metadata=None


@pytest.mark.parametrize('bits',['','0','1','101','0011'])
def test_determinants_are_a_stabilizer_special_case(bits):
    from clifford_qc.reference_states import DeterminantReference
    stab=StabilizerReference.from_determinant(bits)
    determinant=DeterminantReference(bits)
    for code in range(4**len(bits)):
        assert stab.word_expectation(code)==determinant.evaluate_terms([(code,1)],row_nnz=1).value


def test_large_reference_uses_rows_without_density_or_hilbert_space_allocation(monkeypatch):
    import clifford_qc.states as states
    def forbidden(*a,**kw):raise AssertionError('density/Hilbert allocation')
    monkeypatch.setattr(states,'ket_density',forbidden)
    monkeypatch.setattr(ExactMVBackend,'__init__',forbidden)
    monkeypatch.setattr(MV,'__init__',forbidden)
    for name in ('zeros','ones','empty','eye'):monkeypatch.setattr(np,name,forbidden)
    ref=StabilizerReference.from_determinant('1'+'0'*126+'1')
    assert ref.evaluate_terms([(3,.3),(3 << 254,.4),(2 << 128,7)],row_nnz=3).value==pytest.approx(-.7)
    assert ref.storage()['density_coefficients']==0
    assert ref.storage()['tableau_payload_bits']==128*257


def test_exact_projection_consumes_stabilizer_rows_without_density_conversion():
    from clifford_qc.states import bell_density
    from clifford_qc.subspace.projection import MatrixElementBank
    ref=StabilizerReference(2,[('XX',1),('ZZ',1)])
    h=MV.from_terms(2,{'XX':.4,'YY':.2,'ZI':-.3,'ZZ':.1})
    generators=[identity_generator(2),PauliWord.from_label('ZI'),PauliWord.from_label('XI')]
    provider=ExactProjectedProvider(ref,h,generators,storage='packed')
    baseline=MatrixElementBank(bell_density(),h,generators)
    for actual,expected in zip(provider.matrices(),baseline.matrices()):
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert provider.execution_metadata()['reference_evaluation']['arithmetic']=='stabilizer_analytic'


def fermionic_moments(ref):
    n=ref.n
    number=sum((number_op_pauli(n,q) for q in range(n)),MV(n))
    sz=sum(((.5 if q%2==0 else -.5)*number_op_pauli(n,q) for q in range(n)),MV(n))
    s2=spin_squared_operator(n).to_mv()
    return ref.expectation(number),ref.expectation(number*number),ref.expectation(sz),ref.expectation(sz*sz),ref.expectation(s2),ref.expectation(s2*s2)


def test_alpha_beta_bell_pair_has_fixed_number_but_not_sz_or_singlet_spin():
    ref=StabilizerReference(2,[('XX',-1),('ZZ',-1)],spin_ordering='interleaved')
    number,number2,sz,sz2,s2,_=fermionic_moments(ref)
    assert number==1 and number2==1 and sz==0
    assert sz2==.25 and s2==.75


def test_four_mode_ghz_type_valence_bond_is_a_fixed_sector_singlet():
    # (|1001>-|0110>)/sqrt(2): two determinants, not a product of two-mode Bell pairs.
    ref=StabilizerReference(4,[('ZZII',-1),('IIZZ',-1),('ZIZI',-1),('XXXX',-1)],spin_ordering='interleaved')
    assert fermionic_moments(ref)==(2,4,0,0,0,0)
    # Pairing alpha with alpha and beta with beta fixes N/Sz, but still does
    # not make an arbitrary product of two mode-Bell pairs a physical singlet.
    same_spin_pairs=StabilizerReference(4,[('XIXI',-1),('ZIZI',-1),('IXIX',-1),('IZIZ',-1)],spin_ordering='interleaved')
    assert fermionic_moments(same_spin_pairs)==(2,4,0,0,1,2)


def test_canonical_graph_state_has_number_variance_and_no_parity_eigenvalue():
    ref=StabilizerReference(2,[('XZ',1),('ZX',1)],spin_ordering='interleaved')
    number,number2,*_=fermionic_moments(ref)
    assert number2-number**2==.5
    assert ref.word_expectation(PauliWord.from_label('ZZ').code)==0


@pytest.mark.parametrize('n,gens',[(2,[('XX',1)]),(1,[('X',1),('Z',1)]),
    (2,[('XX',1),('XX',-1)]),(1,[('I',-1)]),(1,[('X',1j)]),(1,[('X',True)]),
    (True,[]),(1,[(True,1)])])
def test_invalid_or_non_pure_stabilizers_fail(n,gens):
    with pytest.raises(ValueError):StabilizerReference(n,gens)
