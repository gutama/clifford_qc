"""Independent occupation/spectral replay of the frozen exact CEO record.

Imports neither the producer nor the exchange solver. Rebuilds QE action from
occupation changes, CEO sums/differences, full-space Kronecker H, sector spectrum,
every optimizer call, every selected support, target crossings and accounting.
Optimization itself is not rerun: this verifies an observed trajectory and its
call trace, not that a local optimizer found a global optimum.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.check_phase17_ceo_declaration import load_config, validate
from benchmarks.phase17_ceo_design import RECORD, digest, file_digest
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models.fcidump import fcidump_model


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
    require(record.get('schema')=='clifford_qc.phase17_ceo.v1','schema')
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
        model=fcidump_model(ROOT/declared['path'],integral_tolerance=1e-12)
        h=to_matrix(model.hamiltonian.to_mv())[np.ix_(basis,basis)]
        ground=float(np.linalg.eigvalsh(h)[0])
        reference=np.zeros(len(basis),complex)
        word=sum(1<<(n-1-j) for j in structural['reference_occupied'])
        reference[basis.index(word)]=1.
        spectra={label:np.linalg.eigh(1j*t) for label,t in ts.items()}
        ops={op['label']:op for op in structural['qe']+structural['ovp_ceo']}
        supports={label:commutator_support(model.hamiltonian.to_mv().terms,op,n)
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
            psi=reference.copy(); chosen=[]; previous=run['reference_energy']
            require(len(run['trajectory'])<=config['solver']['max_iterations'],name+': iteration budget')
            require(len(run['screening']) in (len(run['trajectory']),len(run['trajectory'])+1),name+': screening census')
            for i,screen in enumerate(run['screening']):
                scores={label:float(2*np.vdot(h@psi,ts[label]@psi).real) for label in labels}
                norm=float(np.linalg.norm(list(scores.values())))
                selected=screen['selected_label']
                require(screen['iteration']==i+1 and selected in labels,name+': screening index/label')
                require(_near(screen['gradient_norm'],norm) and _near(screen['max_abs_gradient'],abs(scores[selected])),name+': screening score')
                require(abs(scores[selected])>=max(map(abs,scores.values()))-1e-10,name+': selection')
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
                    if len(nonzero)!=1: appended=nonzero
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
                require(trace[-1]['parameters']==row['parameters'] and _near(row['energy'],e),name+': final call/state')
                require(_near(row['optimizer_gradient_norm'],np.linalg.norm(g),1e-8),name+': optimizer gradient')
                require(row['optimizer_resolved']==(row['optimizer_gradient_norm']<=config['solver']['optimizer_gtol']),name+': optimizer resolution')
                require(type(row['optimizer_iterations']) is int and 0<=row['optimizer_iterations']<=config['solver']['maxiter'],name+': optimizer iteration budget')
                require(e<=previous+1e-9 and e>=ground-1e-9 and _near(np.vdot(psi,psi).real,1),name+': monotonicity/norm/bound')
                previous=e
                rotations=sum(len(ops[label]['coefficients']) for label in chosen)
                cnots=sum(2*(sum((code>>(2*j))&3 != 0 for j in range(n))-1)
                          for label in chosen for code,*_ in ops[label]['coefficients'])
                require(row['pauli_rotation_occurrences']==rotations and row['generic_ladder_cnot_upper_bound']==cnots,name+': generic lowering accounting')
            require(_near(run['energy'],previous),name+': final energy')
            if run['stopped_reason']=='iteration_budget':
                require(len(run['trajectory'])==config['solver']['max_iterations'],name+': budget stop')
            elif run['stopped_reason']!='gradient_threshold':
                require(False,name+': unknown stop')
            resources=run['resources']
            expected={'screening_rounds':len(run['screening']),
                      'screened_gradients':sum(len(labels) for _ in run['screening']),
                      'screening_unique_words_sum':len(run['screening'])*unique_words,
                      'screening_word_occurrences':len(run['screening'])*word_occurrences,
                      'optimizer_energy_gradient_evaluations':sum(len(r['optimizer_trace']) for r in run['trajectory']),
                      'parameter_derivatives':sum(len(r['parameters'])*len(r['optimizer_trace']) for r in run['trajectory']),
                      'dense_operator_bytes':h.nbytes+sum(t.nbytes for t in ts.values())}
            for key,value in expected.items(): require(resources[key]==value,name+': '+key)
            expected_target=_target(run,config['target_hartree'])
            require(armrow['target_summary'].keys()==expected_target.keys(),name+': target keys')
            require(all((_near(armrow['target_summary'][key],value) if key=='error'
                         else armrow['target_summary'][key]==value) for key,value in expected_target.items()),name+': target crossing/accounting')
    return problems


def main():
    config=load_config(); problems=validate(config)
    record=json.loads(RECORD.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    problems+=check_record(record,config)
    for path,expected in record['execution_source_bindings'].items():
        if file_digest(ROOT/path)!=expected: problems.append('execution source drift: '+path)
    source=record['provenance']['git_sha']
    for path,expected in record['execution_source_bindings'].items():
        historical=subprocess.check_output(['git','show',f'{source}:{path}'],cwd=ROOT)
        import hashlib
        if hashlib.sha256(historical).hexdigest()!=expected:
            problems.append('source commit does not bind execution file: '+path)
    if record['provenance']['git_dirty']: problems.append('dirty execution')
    first=subprocess.check_output(['git','log','--format=%H','--diff-filter=A','--',str(RECORD.relative_to(ROOT))],cwd=ROOT,text=True).strip().splitlines()
    if first and (first[-1]==source or subprocess.run(['git','merge-base','--is-ancestor',source,first[-1]],cwd=ROOT).returncode):
        problems.append('record must follow execution source in history')
    if problems: raise ValueError('; '.join(problems[:30]))
    print('Phase 17 CEO: PASS (independent occupation/spectral trace replay)')


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,KeyError,IndexError,subprocess.SubprocessError) as error:
        print(f'FAIL: {error}'); sys.exit(1)
