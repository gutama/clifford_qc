"""Independent occupation/spectral replay of the frozen exact CEO record.

The replay does not call the producer, exchange solver, native Pauli product or
FCIDUMP-to-JW mapper. It shares FCIDUMP text parsing and digest utilities.
Occupation signs build H independently; letter products rebuild its Pauli words
and every pool coefficient. Spectral propagation checks the observed calls.
Optimization itself is not rerun: this verifies an observed trajectory and its
call trace, not that a local optimizer found a global optimum.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.check_phase17_ceo_declaration import load_config, validate, paths
from benchmarks.phase17_ceo_design import CONFIG, LEGACY_CONFIG, digest
from benchmarks.phase17_ceo_oracle import hamiltonian, pool_coefficients


def matrices(structural):
    n=structural['n']
    basis=[word for word in range(2**n)
           if word.bit_count()==structural['n_electrons']
           and sum((1 if j%2==0 else -1)*((word>>(n-1-j))&1) for j in range(n))==round(2*structural['sz'])]
    index={word:i for i,word in enumerate(basis)}
    out={}
    for op in structural['qe']:
        t=np.zeros((len(basis),len(basis)),complex)
        for word,i in index.items():
            if any((word>>(n-1-j))&1 for j in op['creation']): continue
            if any(not ((word>>(n-1-j))&1) for j in op['annihilation']): continue
            target=word
            for j in op['creation']+op['annihilation']: target ^= 1<<(n-1-j)
            k=index[target]
            t[k,i],t[i,k]=1,-1
        out[op['label']]=t
    for op in structural['ovp_ceo']:
        if op['parents']:
            out[op['label']]=out[op['parents'][0]]+op['sign']*out[op['parents'][1]]
    return basis,out


def replay(h,reference,spectra,labels,parameters):
    states=[reference]
    unitaries=[]
    for label,theta in zip(labels,parameters):
        eig,vec=spectra[label]
        u=(vec*np.exp(-1j*theta*eig))@vec.conj().T
        unitaries.append(u); states.append(u@states[-1])
    lam=h@states[-1]
    energy=float(np.vdot(states[-1],lam).real)
    gradients=[]
    for i in reversed(range(len(labels))):
        eig,vec=spectra[labels[i]]
        derivative=vec@((-1j*eig)*(vec.conj().T@states[i+1]))
        gradients.append(float(2*np.vdot(lam,derivative).real))
        lam=unitaries[i].conj().T@lam
    return energy,list(reversed(gradients)),states[-1]


def _near(a,b,tolerance=1e-9):
    return np.asarray(a).shape==np.asarray(b).shape and np.allclose(a,b,atol=tolerance,rtol=0)


def commutator_support(h_terms, op, n):
    """Independent letter multiplication; preserve the substrate's staged 1e-12 pruning."""
    t_terms={code:complex(float.fromhex(real),float.fromhex(imag))
             for code,real,imag in op['coefficients']}
    def multiply(left,right):
        out={}
        for a,ca in left.items():
            for b,cb in right.items():
                phase=1+0j
                for j in range(n):
                    x,y=(a>>(2*j))&3,(b>>(2*j))&3
                    if x and y and x!=y:
                        phase*=1j if (x,y) in ((1,2),(2,3),(3,1)) else -1j
                out[a^b]=out.get(a^b,0)+ca*cb*phase
        return {code:c for code,c in out.items() if abs(c)>1e-12}
    ht,th=multiply(h_terms,t_terms),multiply(t_terms,h_terms)
    return {code for code in ht.keys()|th.keys()
            if code and abs(ht.get(code,0)-th.get(code,0))>1e-12}


def _target(run,target):
    base=run['reference_energy']-run['sector_ground_energy']
    if base<=target:
        return dict(status='REFERENCE_ALREADY_WITHIN_TARGET',iteration=0,error=base,
                    parameters=0,screened_gradients=0,optimizer_evaluations=0,
                    parameter_derivatives=0,generic_ladder_cnots=0)
    calls=derivatives=0
    for row in run['trajectory']:
        calls+=len(row['optimizer_trace'])
        derivatives+=len(row['parameters'])*len(row['optimizer_trace'])
        error=row['energy']-run['sector_ground_energy']
        if error<=target:
            return dict(status='REACHED',iteration=row['iteration'],error=error,
                        parameters=len(row['parameters']),
                        screened_gradients=sum(len(run['pool_labels']) for s in run['screening']
                                               if s['iteration']<=row['iteration']),
                        optimizer_evaluations=calls,parameter_derivatives=derivatives,
                        generic_ladder_cnots=row['generic_ladder_cnot_upper_bound'])
    return dict(status='RIGHT_CENSORED',iteration=None,error=run['energy']-run['sector_ground_energy'])


def check_record(record,config):
    problems=[]
    def require(test,message):
        if not test: problems.append(message)
    reviewed = 'selection_tolerance' in config['solver']
    require(record.get('schema')==('clifford_qc.phase17_ceo.v2' if reviewed else 'clifford_qc.phase17_ceo.v1'),'schema')
    if 'publication_replay_manifest' in config:
        manifest=json.loads((ROOT/config['publication_replay_manifest']).read_text())
        require(record.get('publication_replay')==manifest,'publication replay identity')
        require(digest(record['cases'])==manifest['cases_sha256'],'publication replay scientific equality')
    require(record.get('config_sha256')==digest(config),'declaration identity')
    require(record.get('claim_boundary')==config['claim_boundary'],'claim boundary')
    require([c['id'] for c in record['cases']]==[c['id'] for c in config['cases']],'case census/order')
    for declared,case in zip(config['cases'],record['cases']):
        prefix=case['id']
        require(case['role']==declared['role'],prefix+': role')
        structural=declared['structure']; n=structural['n']
        basis,ts=matrices(structural)
        h,h_terms=hamiltonian(ROOT/declared['path'],basis)
        ground=float(np.linalg.eigvalsh(h)[0])
        reference=np.zeros(len(basis),complex)
        word=sum(1<<(n-1-j) for j in structural['reference_occupied'])
        reference[basis.index(word)]=1.
        spectra={label:np.linalg.eigh(1j*t) for label,t in ts.items()}
        ops={op['label']:op for op in structural['qe']+structural['ovp_ceo']}
        independent_terms=pool_coefficients(structural)
        for label,op in ops.items():
            declared_terms={code:complex(float.fromhex(real),float.fromhex(imag)) for code,real,imag in op['coefficients']}
            require(declared_terms==independent_terms[label],prefix+': independent pool coefficients '+label)
        supports={label:commutator_support(h_terms,op,n)
                  for label,op in ops.items()}
        require([a['run']['arm'] for a in case['arms']]==config['arms'],prefix+': arm census/order')
        for armrow in case['arms']:
            run=armrow['run']; arm=run['arm']; name=prefix+':'+arm
            labels=[op['label'] for op in structural['qe' if arm=='qe' else 'ovp_ceo']]
            unique_words=len(set().union(*(supports[label] for label in labels)))
            word_occurrences=sum(len(supports[label]) for label in labels)
            require(run['pool_labels']==labels and run['sector_basis']==basis,name+': basis/pool')
            require(_near(run['sector_ground_energy'],ground),name+': spectrum')
            require(_near(run['reference_energy'],np.vdot(reference,h@reference).real),name+': reference')
            require(run['quantum_shots'] is None and run['evidence_category']=='exact_small_sector_simulation',name+': evidence')
            if reviewed:
                require(run.get('policy_version')==2 and run.get('selection_tolerance')==config['solver']['selection_tolerance'],name+': selection policy')
                require(run.get('parent_promotion_threshold')==config['solver']['parent_threshold'],name+': promotion policy')
            psi=reference.copy(); chosen=[]; previous=run['reference_energy']
            require(len(run['trajectory'])<=config['solver']['max_iterations'],name+': iteration budget')
            require(len(run['screening']) in (len(run['trajectory']),len(run['trajectory'])+1),name+': screening census')
            for i,screen in enumerate(run['screening']):
                scores={label:float(2*np.vdot(h@psi,ts[label]@psi).real) for label in labels}
                norm=float(np.linalg.norm(list(scores.values())))
                selected=screen['selected_label']
                require(screen['iteration']==i+1 and selected in labels,name+': screening index/label')
                require(_near(screen['gradient_norm'],norm) and _near(screen['max_abs_gradient'],abs(scores[selected])),name+': screening score')
                if reviewed:
                    maximum=max(map(abs,scores.values()))
                    first=next(label for label in labels if maximum-abs(scores[label])<=config['solver']['selection_tolerance'])
                    require(selected==first,name+': first member within selection tolerance')
                else:
                    require(abs(scores[selected])>=max(map(abs,scores.values()))-1e-10,name+': legacy float-max selection')
                stopped=norm<=config['solver']['gradient_threshold']
                require(screen['stopped']==stopped,name+': stopping threshold')
                require(screen['screened_gradients']==len(labels),name+': screening count')
                require(screen['unique_commutator_words']==unique_words and
                        screen['commutator_word_occurrences']==word_occurrences,name+': screening word ledger')
                if i>=len(run['trajectory']):
                    require(stopped and run['stopped_reason']=='gradient_threshold',name+': terminal screening')
                    continue
                require(not stopped,name+': continued after stop')
                row=run['trajectory'][i]
                require(row['iteration']==i+1 and row['selected_label']==selected,name+': row selection')
                appended=[selected]; parents={}
                if arm=='mixed_ceo' and ops[selected]['parents']:
                    parents={op['label']:float(2*np.vdot(h@psi,ts[op['label']]@psi).real)
                             for op in structural['qe'] if op['support']==ops[selected]['support']}
                    nonzero=[label for label,value in parents.items() if abs(value)>config['solver']['parent_threshold']]
                    if (len(nonzero)>1 if reviewed else len(nonzero)!=1): appended=nonzero
                require(row['appended_labels']==appended,name+': CEO/MVP choice')
                require(row['parent_gradients'].keys()==parents.keys() and all(_near(row['parent_gradients'][label],value) for label,value in parents.items()),name+': parent gradients')
                chosen+=appended
                require(row['labels']==chosen and len(row['parameters'])==len(chosen),name+': ansatz')
                trace=row['optimizer_trace']
                require(bool(trace) and row['optimizer_evaluations']==len(trace),name+': observed call count')
                require(len(trace[0]['parameters'])==len(chosen) and all(x==0 for x in trace[0]['parameters'][-len(appended):]),name+': zero initialized new parameters')
                old=[] if i==0 else run['trajectory'][i-1]['parameters']
                require(trace[0]['parameters'][:-len(appended)]==old,name+': retained parameter warm start')
                for call in trace:
                    require(len(call['parameters'])==len(chosen),name+': call shape')
                    e,g,_=replay(h,reference,spectra,chosen,call['parameters'])
                    require(_near(call['energy'],e) and _near(call['gradient'],g,1e-8),name+': optimizer call replay')
                    require(e>=ground-1e-9,name+': variational call')
                e,g,psi=replay(h,reference,spectra,chosen,row['parameters'])
                final_index=row['final_evaluation_index'] if reviewed else len(trace)-1
                require(trace[final_index]['parameters']==row['parameters'] and _near(row['energy'],e),name+': final call/state')
                if reviewed:
                    indices=row['optimizer_callback_indices']
                    require(len(indices)==row['optimizer_callback_requests'] and
                            row['optimizer_cache_hits']==len(indices)-len(trace),name+': callback/cache ledger')
                    require(indices[-1]==final_index and set(indices)==set(range(len(trace))) and
                            list(dict.fromkeys(indices))==list(range(len(trace))),name+': callback indices')
                    require(len({tuple(c['parameters']) for c in trace})==len(trace),name+': distinct computation ledger')
                require(_near(row['optimizer_gradient_norm'],np.linalg.norm(g),1e-8),name+': optimizer gradient')
                if reviewed:
                    maximum=max(map(abs,trace[final_index]['gradient']))
                    require(_near(row['optimizer_max_abs_gradient'],max(map(abs,g)),1e-8),name+': optimizer maximum gradient')
                    require(row['optimizer_max_abs_gradient']==maximum,name+': retained maximum gradient')
                    require(row['optimizer_resolved']==(maximum<=config['solver']['optimizer_gtol']),name+': optimizer resolution')
                else:
                    require(row['optimizer_resolved']==(row['optimizer_gradient_norm']<=config['solver']['optimizer_gtol']),name+': legacy L2 optimizer resolution')
                require(type(row['optimizer_iterations']) is int and 0<=row['optimizer_iterations']<=config['solver']['maxiter'],name+': optimizer iteration budget')
                require(e<=previous+1e-9 and e>=ground-1e-9 and _near(np.vdot(psi,psi).real,1),name+': monotonicity/norm/bound')
                previous=e
                rotations=sum(len(ops[label]['coefficients']) for label in chosen)
                cnots=sum(2*(sum((code>>(2*j))&3 != 0 for j in range(n))-1)
                          for label in chosen for code,*_ in ops[label]['coefficients'])
                require(row['pauli_rotation_occurrences']==rotations and row['generic_ladder_cnot_upper_bound']==cnots,name+': generic lowering accounting')
            require(_near(run['energy'],previous),name+': final energy')
            if run['stopped_reason']=='iteration_budget':
                require(len(run['trajectory'])==config['solver']['max_iterations'] and len(run['screening'])==len(run['trajectory']),name+': budget stop')
            elif run['stopped_reason']=='gradient_threshold':
                require(len(run['screening'])==len(run['trajectory'])+1 and bool(run['screening']) and
                        run['screening'][-1]['stopped'] is True and
                        run['screening'][-1]['gradient_norm']<=config['solver']['gradient_threshold'],name+': convergence requires terminal screening')
            elif run['stopped_reason']=='empty_pool':
                require(not labels and not run['screening'] and not run['trajectory'],name+': empty pool stop')
            else:
                require(False,name+': unknown stop')
            resources=run['resources']
            expected={'screening_rounds':len(run['screening']),
                      'screened_gradients':sum(len(labels) for _ in run['screening']),
                      'screening_unique_words_sum':len(run['screening'])*unique_words,
                      'screening_word_occurrences':len(run['screening'])*word_occurrences,
                      'optimizer_energy_gradient_evaluations':sum(len(r['optimizer_trace']) for r in run['trajectory']),
                      'parameter_derivatives':sum(len(r['parameters'])*len(r['optimizer_trace']) for r in run['trajectory']),
                      'dense_operator_bytes':h.nbytes+sum(ts[label].nbytes for label in (set(labels+chosen) if reviewed else ts))}
            for key,value in expected.items(): require(resources[key]==value,name+': '+key)
            expected_target=_target(run,config['target_hartree'])
            require(armrow['target_summary'].keys()==expected_target.keys(),name+': target keys')
            require(all((_near(armrow['target_summary'][key],value) if key=='error'
                         else armrow['target_summary'][key]==value) for key,value in expected_target.items()),name+': target crossing/accounting')
    return problems


def main():
    for path in (LEGACY_CONFIG, CONFIG):
        config=load_config(path); notes=[]
        problems=validate(config,notes=notes)
        _, record_path=paths(config)
        record=json.loads(record_path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        problems+=check_record(record,config)
        for note in notes: print(note)
        if problems: raise ValueError('; '.join(problems[:30]))
        suffix='; provenance unverified' if any('SKIP' in note for note in notes) else ''
        print(f'Phase 17 CEO {record_path.name}: PASS (occupation/spectral trace replay{suffix})')


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,KeyError,IndexError) as error:
        print(f'FAIL: {error}'); sys.exit(1)
