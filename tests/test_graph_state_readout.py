"""Exact graph-normal-form, phase, scheduling and no-Stim execution checks."""
import json
import subprocess
import sys

import numpy as np
import pytest

from clifford_qc.ir import PauliWord,conjugate_pauli_word
from clifford_qc.measurement.graph_state import NativeClifford,graph_diagonalizer,matching_layers,native_stats
from clifford_qc.measurement.planning import compile_block_measurement_plan
from clifford_qc.measurement.compiled import CompiledMeasurementSampler
from clifford_qc.measurement.cache import GroupedWordCache
from clifford_qc.states import bell_density


@pytest.mark.parametrize('gates',[(('H',(0,)),),(('S',(0,)),),(('SDG',(0,)),),
    (('CZ',(0,1)),),(('H',(1,)),('S',(0,)),('CZ',(0,1)),('H',(0,)))])
def test_native_scalar_vectorized_and_independent_mv_conjugations_agree(gates):
    native=NativeClifford(2,gates)
    signs,codes=native.transform_codes(range(16))
    for code in range(16):
        word=PauliWord(2,code);phase,image=native.conjugate(word)
        exact=conjugate_pauli_word(native.program(),word)
        assert exact[1]==image and exact[0]==pytest.approx(phase,abs=1e-12)
        assert phase==signs[code] and image.code==codes[code]
        inverse_phase,original=native.inverse().conjugate(image)
        assert phase*inverse_phase==1 and original==word


@pytest.mark.parametrize('size',range(1,6))
def test_random_isotropic_rank_deficient_groups_and_every_span_member(size):
    rng=np.random.default_rng(400+size)
    for rank in range(size+1):
        gates=[]
        for _ in range(4*size):
            q=int(rng.integers(size));gates.append((('H' if rng.integers(2) else 'S'),(q,)))
            if size>1:
                a,b=map(int,rng.choice(size,2,replace=False));gates.append(('CZ',(a,b)))
        preparation=NativeClifford(size,tuple(gates))
        generators=[preparation.conjugate(PauliWord(size,3 << (2*q)))[1].code for q in range(rank)]
        span=[0]
        for g in generators:span += [c^g for c in list(span)]
        diagonalizer=graph_diagonalizer(span,size)
        for code in span:
            _,image=diagonalizer.conjugate(PauliWord(size,code))
            assert all(image.letter(q) in ('I','Z') for q in range(size))
        nonpivots=set(range(size))-set(diagonalizer.graph.pivots)
        assert not any(a in nonpivots and b in nonpivots for a,b in diagonalizer.graph.edges)
        assert diagonalizer==graph_diagonalizer(list(reversed(span)),size)


def test_z_spectators_do_not_create_unnecessary_jw_tail_entanglers():
    codes=[PauliWord.from_label(v).code for v in ('XZZZZ','YZZZZ')]
    # One X word with a common Z tail needs only its local H; the two listed
    # X/Y words together anticommute and must not form a commuting group.
    d=graph_diagonalizer(codes[:1],5)
    assert d.gates==( ('H',(0,)), )
    assert d.conjugate(PauliWord(5,codes[0]))[1].label=='ZZZZZ'
    with pytest.raises(ValueError,match='commute|isotropic'):graph_diagonalizer(codes,5)


@pytest.mark.parametrize('size',[2,3,4,8])
def test_matching_schedule_is_disjoint_complete_and_obeys_even_odd_bound(size):
    edges=tuple((a,b) for a in range(size) for b in range(a+1,size))
    layers=matching_layers(size,edges)
    assert sorted(e for layer in layers for e in layer)==list(edges)
    assert all(len({q for e in layer for q in e})==2*len(layer) for layer in layers)
    assert len(layers)==size-1 if size%2==0 else len(layers)==size
    if size==3:assert len(layers)>size-1  # Vizing alone does not give k-1.


@pytest.mark.parametrize('label,count',[('X',1),('Y',2),('Z',0)])
def test_graph_k1_matches_qwc_rotation_cost(label,count):
    d=graph_diagonalizer([PauliWord.from_label(label).code],1)
    counts,critical,d1,d2=native_stats(d)
    assert sum(counts.values())==count and critical==d2==0
    assert d1==count


def test_graph_compilation_and_actual_joint_sampling_need_no_stim():
    script='''
import builtins,sys,json
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=='stim' or name.startswith('stim.'):
        raise AssertionError('Stim imported on graph path')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import compile_block_measurement_plan,CompiledMeasurementSampler,GroupedWordCache
from clifford_qc.states import bell_density
codes=[PauliWord.from_label(s).code for s in ('XX','YY','ZZ')]
p=compile_block_measurement_plan(2,codes,2,groups=[[0,1,2]],method='graph_state')
cache=GroupedWordCache(2)
cache.add_batch(CompiledMeasurementSampler(91).sample_from_state(bell_density(),p.settings,100))
assert [cache.candidate_estimate({c:1}) for c in codes]==[1,-1,1]
assert 'stim' not in sys.modules
assert p.synthesis.gate_counts['CZ']==1
# Simulate package absence as well as guarding imports, even in a Stim-enabled
# parent environment, so capability discovery exercises the real missing-extra path.
import importlib.util
find_spec=importlib.util.find_spec
importlib.util.find_spec=lambda name,*a,**kw: None if name=='stim' else find_spec(name,*a,**kw)
from clifford_qc.measurement.block_synthesis import synthesize_block_settings
for call in (lambda: compile_block_measurement_plan(2,codes,2),
             lambda: synthesize_block_settings(2,codes,[[0,1,2]],2)):
    try:
        call()
    except ModuleNotFoundError as error:
        assert 'clifford-qc[stim]' in str(error),str(error)
    else:
        raise AssertionError('elimination unexpectedly succeeded without Stim')
print('ok')
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='ok'


def test_compiled_signed_readouts_match_sampling_and_separate_cache_keys():
    pytest.importorskip("stim")
    codes=[PauliWord.from_label(s).code for s in ('XX','YY','ZZ')]
    p=compile_block_measurement_plan(2,codes,2,method='graph_state')
    old=compile_block_measurement_plan(2,codes,2)
    assert p.settings[0].key!=old.settings[0].key
    cache=GroupedWordCache(2)
    cache.add_batch(CompiledMeasurementSampler(1).sample_from_state(bell_density(),p.settings,50))
    assert [cache.candidate_estimate({c:1}) for c in codes]==[1,-1,1]
    assert p.synthesis.settings[0].n_2q==1
    with pytest.raises(ValueError):p.synthesis.compatibility.flags.writeable=True


@pytest.mark.parametrize('options',[{'method':'unknown'},{'groups':[[0],[0]],'method':'graph_state'}])
def test_invalid_methods_and_groups_fail(options):
    with pytest.raises(ValueError):compile_block_measurement_plan(2,[5,15],2,**options)


def test_unsplit_double_excitation_uses_only_a_three_edge_active_star():
    from clifford_qc.subspace.fermionic_generators import determinant_excitations
    from clifford_qc.pauli_structure import gf2_rank
    generator=next(g for g in determinant_excitations(8,(0,3)) if g.label=='E(4,7<-0,3)')
    assert generator.mv.nnz()==8
    d=graph_diagonalizer(list(generator.mv.terms),8)
    assert len(d.graph.edges)==3
    assert set(q for edge in d.graph.edges for q in edge)=={0,3,4,7}
    adjacency=[sum(1 << b for a,b in d.graph.edges if a==q) | sum(1 << a for a,b in d.graph.edges if b==q) for q in range(8)]
    assert gf2_rank(adjacency,8)==2  # A nonempty star's adjacency rank is not one.
    assert native_stats(d)[1]==3    # Its three incident CZs cannot share a layer.


def test_32_qubit_packed_boundary_preserves_high_bit_signs():
    codes=[0,1 << 62,2 << 62,3 << 62,1 | (2 << 62)]
    d=NativeClifford(32,(('H',(31,)),('S',(31,)),('CZ',(0,31))))
    signs,images=d.transform_codes(codes)
    for code,sign,image in zip(codes,signs,images):
        expected=d.conjugate(PauliWord(32,code))
        assert expected==(complex(sign),PauliWord(32,int(image)))


def test_assigned_joint_distribution_is_preserved_despite_changed_extra_readouts():
    pytest.importorskip("stim")
    from clifford_qc.ir import Program,Rotor
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.states import computational_probabilities
    from clifford_qc.subspace.restriction import Restriction
    program=Program(3);program.clifford('H',0);program.clifford('CZ',0,2)
    program.append(Rotor(PauliWord.from_label('YXI'),.37))
    program.append(Rotor(PauliWord.from_label('IZY'),.23))
    rho=ExactMVBackend().state(program)
    codes=[PauliWord.from_label(s).code for s in ('XXI','ZZI','YYI')]
    distributions=[]
    for method in ('elimination','graph_state'):
        setting=compile_block_measurement_plan(3,codes,3,groups=[[0,1,2]],method=method).settings[0]
        probabilities=computational_probabilities(Restriction.encoding(setting.clifford).state(rho))
        joint={}
        for bits,probability in probabilities.items():
            outcomes=tuple(sign*(-1 if sum(bits[q]=='1' for q in qubits)%2 else 1)
                           for sign,qubits in (setting.readouts[c] for c in codes))
            joint[outcomes]=joint.get(outcomes,0)+probability
        distributions.append(joint)
    for key in distributions[0].keys() | distributions[1].keys():
        assert distributions[0].get(key,0)==pytest.approx(distributions[1].get(key,0),abs=1e-12)


def test_qwc_words_never_acquire_unnecessary_entanglers_from_normal_form():
    codes=[PauliWord.from_label(s).code for s in ('YYZ','YIZ','IYI')]
    d=graph_diagonalizer(codes,3)
    assert d.graph.kind=='qwc' and not d.graph.edges
    assert not any(name=='CZ' for name,_ in d.gates)
    assert all(all(d.conjugate(PauliWord(3,c))[1].letter(q) in ('I','Z') for q in range(3)) for c in codes)
