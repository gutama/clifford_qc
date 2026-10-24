"""Duration, replay and resource controls from the graph-readout review."""
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.compare_graph_readout import compare_audits
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.block_synthesis import gate_stats, synthesize_block_settings
from clifford_qc.measurement.graph_state import graph_diagonalizer, native_stats
from clifford_qc.measurement.planning import compile_block_measurement_plan
from clifford_qc.reference_states import StabilizerReference


def codes(*labels):
    return [PauliWord.from_label(label).code for label in labels]


def test_disjoint_blocks_are_priced_in_parallel_without_cross_block_clock_inflation():
    words=codes('XZXX','ZXZZ')
    plan=compile_block_measurement_plan(4,words,2,groups=[[0,1]],method='graph_state')
    blocks=[graph_diagonalizer([((c >> (2*start)) & 15) for c in words],2) for start in (0,2)]
    local=[native_stats(block) for block in blocks]
    resource=plan.resources[0]
    assert resource.d_2q==1
    assert (resource.d_1q,resource.d_2q)==(max(row[2] for row in local),max(row[3] for row in local))
    assert resource.n_2q==sum(row[0]['CZ'] for row in local)
    assert plan.synthesis.two_qubit_depth_per_setting==(max(row[1] for row in local),)


def test_surviving_local_gates_do_not_split_a_disjoint_cz_matching():
    diagonalizer=graph_diagonalizer(codes('XZII','ZXII','IIYZ','IIZX'),4)
    assert diagonalizer.graph.matching_layers==(((0,1),(2,3)),)
    counts,critical,d1,d2=native_stats(diagonalizer)
    assert counts['CZ']==2 and critical==d2==1
    assert d1>0


@pytest.mark.parametrize('size',[2,4,8])
def test_matching_layer_bound_survives_local_gate_cancellation(size):
    from clifford_qc.measurement.graph_state import NativeClifford
    rng=np.random.default_rng(603+size)
    for _ in range(20):
        gates=[]
        for _ in range(4*size):
            q=int(rng.integers(size));gates.append(('S' if rng.integers(2) else 'H',(q,)))
            a,b=map(int,rng.choice(size,2,replace=False));gates.append(('CZ',(a,b)))
        preparation=NativeClifford(size,tuple(gates))
        words=[preparation.conjugate(PauliWord(size,3 << (2*q)))[1].code for q in range(size)]
        result=graph_diagonalizer(words,size)
        assert native_stats(result)[3]<=len(result.graph.matching_layers)<=size-1


def test_shared_scheduler_has_dependency_sensitive_one_and_two_qubit_layers():
    # Same topology and dependencies for the two entangling gate names.
    gates=(('H',(0,)),('CX',(0,1)),('S',(1,)),('CX',(1,2)))
    cx=gate_stats(gates)
    cz=gate_stats(tuple(('CZ' if name=='CX' else name,qubits) for name,qubits in gates))
    assert cx[1:]==cz[1:]==(2,2,2)
    assert cx[0]['CX']==cz[0]['CZ']==2


def test_full_bank_code_validation_is_linear_in_bank_and_blocks(monkeypatch):
    import clifford_qc.measurement.graph_state as graph
    original=graph.validate_word_code;calls=0
    def counted(*args):
        nonlocal calls
        calls+=1
        return original(*args)
    monkeypatch.setattr(graph,'validate_word_code',counted)
    words=list(range(4**4))
    plan=compile_block_measurement_plan(4,words,1,method='graph_state')
    assert len(plan.settings)>20
    # No per-setting scan of the full bank through Python validation.
    assert calls<len(words)*(1+6*4)


@pytest.mark.parametrize('bad',[[True],[-1],[16],[1.5]])
def test_public_vectorized_conjugation_still_rejects_invalid_codes(bad):
    from clifford_qc.measurement.graph_state import NativeClifford
    with pytest.raises(ValueError):NativeClifford(2,()).transform_codes(bad)


def test_zero_gate_counts_are_read_only_and_have_counter_semantics():
    plan=compile_block_measurement_plan(2,codes('XI','IZ'),2,method='graph_state')
    counts=plan.synthesis.gate_counts;keys=tuple(counts)
    assert counts['CZ']==counts['CX']==0 and tuple(counts)==keys
    with pytest.raises(TypeError):counts['CZ']=1
    copied=counts.copy();copied['CZ']=9
    assert counts['CZ']==0


def test_stabilizer_reference_never_grows_the_shared_mv_product_cache():
    from clifford_qc.pauli_kernel import _word_mul_unchecked
    before=_word_mul_unchecked.cache_info()
    bits='10'*20;reference=StabilizerReference.from_determinant(bits)
    rng=np.random.default_rng(717)
    for _ in range(1000):
        selected=rng.integers(2,size=40)
        code=sum(3 << (2*q) for q in range(40) if selected[q])
        expected=(-1)**sum(int(bits[q]) for q in range(40) if selected[q])
        assert reference.word_expectation(code)==expected
    assert _word_mul_unchecked.cache_info()==before


def test_manual_and_determinant_stabilizer_constructors_share_default_conventions():
    determinant=StabilizerReference.from_determinant('10')
    manual=StabilizerReference(2,[('ZI',-1),('IZ',1)])
    assert determinant.metadata.spin_ordering==manual.metadata.spin_ordering=='interleaved'
    assert determinant.metadata.identity==manual.metadata.identity
    assert manual.metadata.identity!=StabilizerReference(2,manual.generators,spin_ordering='qubits').metadata.identity


def test_audit_stamps_are_advisory_while_inputs_and_results_are_strict(capsys):
    old={'implementation_fingerprint':'NumPy 2.5.2','source_sha256':{'code.py':'old'},
         'inputs':{'record_sha256':'historical'},'rows':[{'counts':3,'duration_us':.5}]}
    current=deepcopy(old);current.update(implementation_fingerprint='NumPy 2.5.4',source_sha256={'code.py':'new'})
    compare_audits(old,current)
    assert capsys.readouterr().err.count('WARNING:')==2
    for field in ('counts','duration_us'):
        changed=deepcopy(current);changed['rows'][0][field]+=1
        with pytest.raises(SystemExit,match='inputs or results differ'):compare_audits(old,changed)
    changed=deepcopy(current);changed['inputs']['record_sha256']='changed'
    with pytest.raises(SystemExit):compare_audits(old,changed)


def test_public_synthesis_apis_share_method_validation():
    messages=[]
    for call in (lambda:compile_block_measurement_plan(2,codes('XX'),2,method='unknown'),
                 lambda:synthesize_block_settings(2,codes('XX'),[[0]],2,method='unknown')):
        with pytest.raises(ValueError) as error:call()
        messages.append(str(error.value))
    assert messages[0]==messages[1]=='method must be elimination or graph_state'
