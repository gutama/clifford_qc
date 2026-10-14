"""Undeclared toy comparisons, corruption and outcome-free design checks."""
from copy import deepcopy
import json

import pytest

from benchmarks import check_phase17_ceo as checker
from benchmarks import check_phase17_ceo_declaration as declaration
from benchmarks.phase17_ceo_design import structure, target_summary
from benchmarks.run_phase17_ceo import run_campaign


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    path=tmp_path_factory.mktemp('exchange')/'toy.FCIDUMP'
    path.write_text('&FCI NORB=2, NELEC=2, MS2=0 /\n'
                    '0.7 1 2 0 0\n-0.4 1 1 0 0\n0.2 2 2 0 0\n'
                    '1.3 1 1 1 1\n1.1 2 2 2 2\n0.1 1 1 2 2\n0.0 0 0 0 0\n')
    config={'arms':['qe','ovp_ceo','mixed_ceo'],'target_hartree':.0016,
            'claim_boundary':'toy exact simulation',
            'solver':dict(max_iterations=12,maxiter=100,gradient_threshold=1e-6,
                          parent_threshold=1e-7,optimizer_gtol=1e-8,max_sector_dimension=256,selection_tolerance=1e-10),
            'cases':[dict(id='toy',path=str(path),role='undeclared_test',structure=structure(path))]}
    return config,run_campaign(config)


def test_independent_complete_toy_replay(toy):
    config,record=toy
    assert checker.check_record(record,config)==[]


@pytest.mark.parametrize('kind',['energy','gradient','count','words','parent','target','missing_arm','ordering','bound','parent_gradient','terminal','false_stop','tie','callbacks','resolution','coefficients'])
def test_corrupt_record_refused(toy,kind):
    config,original=toy
    record=deepcopy(original)
    arms=record['cases'][0]['arms']
    run=arms[2 if kind in {'parent','parent_gradient'} else 0]['run']; row=run['trajectory'][0]
    if kind in {'parent','parent_gradient'}:
        row=next(r for r in run['trajectory'] if r['parent_gradients'] and len(r['appended_labels'])>1)
    if kind=='energy': row['energy']+=.01
    elif kind=='gradient': row['optimizer_trace'][0]['gradient'][0]+=.01
    elif kind=='count': row['optimizer_evaluations']+=1
    elif kind=='words': run['resources']['screening_unique_words_sum']+=1
    elif kind=='parent': row['appended_labels']=[]
    elif kind=='parent_gradient':
        label=next(iter(row['parent_gradients'])); row['parent_gradients'][label]+=.01
    elif kind=='target': arms[0]['target_summary']['status']='invented'
    elif kind=='missing_arm': arms.pop()
    elif kind=='ordering': run['pool_labels']=list(reversed(run['pool_labels']))
    elif kind=='bound': run['sector_ground_energy']-=.1
    elif kind=='terminal':
        assert run['stopped_reason']=='gradient_threshold'
        run['screening'].pop()
        for key in ('screening_rounds','screened_gradients','screening_unique_words_sum','screening_word_occurrences'):
            run['resources'][key]=sum(s[{'screening_rounds':'iteration','screened_gradients':'screened_gradients','screening_unique_words_sum':'unique_commutator_words','screening_word_occurrences':'commutator_word_occurrences'}[key]] for s in run['screening']) if key!='screening_rounds' else len(run['screening'])
    elif kind=='false_stop': run['stopped_reason']='gradient_threshold'; run['screening']=run['screening'][:len(run['trajectory'])]
    elif kind=='tie':
        run['screening'][0]['selected_label']=run['pool_labels'][1]
    elif kind=='callbacks': row['optimizer_cache_hits']+=1
    elif kind=='resolution': row['optimizer_resolved']=not row['optimizer_resolved']
    elif kind=='coefficients':
        config=deepcopy(config); config['cases'][0]['structure']['qe'][0]['coefficients'][0][1]=float(1).hex()
    errors=checker.check_record(record,config)
    assert errors
    if kind=='tie': assert any('first member within selection tolerance' in error for error in errors)
    if kind in {'terminal','false_stop'}: assert any('convergence requires terminal screening' in error for error in errors)


def test_outcome_free_config_and_structures():
    config=declaration.load_config()
    assert declaration.validate(config,provenance=False,check_sources=False)==[]
    declaration._walk(config)


@pytest.mark.parametrize('mutation',['outcome','threshold','duplicate','nonfinite'])
def test_declaration_drift_refused(tmp_path,mutation):
    value=deepcopy(declaration.load_config())
    if mutation=='outcome': value['cases'][0]['nested']={'energy':0.}
    elif mutation=='threshold': value['target_hartree']*=2
    text=json.dumps(value)
    if mutation=='duplicate': text=text[:-1]+',"arms": []}'
    elif mutation=='nonfinite': text=text[:-1]+',"bad": NaN}'
    path=tmp_path/'config.json'; path.write_text(text)
    with pytest.raises(ValueError): declaration.load_config(path)


def test_first_prefix_and_reference_floor_are_separate():
    run=dict(reference_energy=1.,sector_ground_energy=0.,energy=.1,
             trajectory=[dict(iteration=1,energy=.1,optimizer_evaluations=2,
                              parameters=[.4],generic_ladder_cnot_upper_bound=6)],
             screening=[dict(iteration=1,screened_gradients=4)])
    assert target_summary(run,.0016)['status']=='RIGHT_CENSORED'
    assert target_summary(run,.2)['iteration']==1
    assert target_summary(run,1.)['iteration']==0


def test_legacy_declaration_remains_immutable():
    from benchmarks.phase17_ceo_design import LEGACY_CONFIG
    assert declaration.load_config(LEGACY_CONFIG)['design_status'] == 'publication_replay_of_frozen_local_design'


def test_historical_binding_and_last_change(tmp_path):
    import subprocess
    import hashlib
    import shutil
    if shutil.which('git') is None:
        pytest.skip('Git lineage fixtures require git')
    from benchmarks.phase17_ceo_provenance import history_problems
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q'); git('config', 'user.email', 'test@example.com'); git('config', 'user.name', 'Test')
    config = {'source_bindings': {'source.txt': hashlib.sha256(b'source').hexdigest()}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    git('add', 'config.json'); git('commit', '-qm', 'declaration')
    (tmp_path/'source.txt').write_text('source')
    git('add', 'source.txt'); git('commit', '-qm', 'source')
    source = git('rev-parse', 'HEAD')
    record = {'provenance': {'git_sha': source, 'git_dirty': False}, 'execution_source_bindings': {}}
    (tmp_path/'record.json').write_text(json.dumps(record))
    notes = []
    assert 'record has no committed adding history' in history_problems(config,record,tmp_path,'config.json','record.json',notes)
    git('add', 'record.json'); git('commit', '-qm', 'record')
    assert history_problems(config,record,tmp_path,'config.json','record.json',[]) == []
    (tmp_path/'source.txt').write_text('future shared-core fix')
    assert history_problems(config,record,tmp_path,'config.json','record.json',[]) == []
    # A semantic no-op edit still counts as the latest declaration change.
    (tmp_path/'config.json').write_text(json.dumps(config,indent=2))
    git('add', 'config.json'); git('commit', '-qm', 'late config change')
    assert 'declaration last change must strictly precede execution' in history_problems(config,record,tmp_path,'config.json','record.json',[])
    # An uncommitted declaration has no lineage; it cannot index an empty log.
    assert 'declaration has no committed history' in history_problems(config,record,tmp_path,'uncommitted.json','record.json',[])


def test_provenance_export_and_shallow_are_explicitly_unverified(tmp_path,monkeypatch):
    from benchmarks import phase17_ceo_provenance as lineage
    notes = []
    assert lineage.history_problems({}, {}, tmp_path, 'config.json','record.json',notes) == []
    assert 'SKIP' in notes[0]
    class Result:
        returncode = 0
    monkeypatch.setattr(lineage,'_git',lambda root,*args: Result())
    monkeypatch.setattr(lineage,'_shallow_boundary',lambda **kwargs: {'boundary'})
    monkeypatch.setattr(lineage,'_last_commit',lambda *args,**kwargs: 'boundary')
    monkeypatch.setattr(lineage,'_first_commit',lambda *args,**kwargs: 'boundary')
    notes = []
    assert lineage.history_problems({}, {'provenance': {'git_sha': 'source'}},tmp_path,'config.json','record.json',notes) == []
    assert 'shallow' in notes[0]
