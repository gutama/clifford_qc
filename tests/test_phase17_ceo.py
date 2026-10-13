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
            'solver':dict(max_iterations=3,maxiter=100,gradient_threshold=1e-6,
                          parent_threshold=1e-12,optimizer_gtol=1e-8,max_sector_dimension=256),
            'cases':[dict(id='toy',path=str(path),role='undeclared_test',structure=structure(path))]}
    return config,run_campaign(config)


def test_independent_complete_toy_replay(toy):
    config,record=toy
    assert checker.check_record(record,config)==[]


@pytest.mark.parametrize('kind',['energy','gradient','count','words','parent','target','missing_arm','ordering','bound'])
def test_corrupt_record_refused(toy,kind):
    config,original=toy
    record=deepcopy(original)
    arms=record['cases'][0]['arms']
    run=arms[0]['run']; row=run['trajectory'][0]
    if kind=='energy': row['energy']+=.01
    elif kind=='gradient': row['optimizer_trace'][0]['gradient'][0]+=.01
    elif kind=='count': row['optimizer_evaluations']+=1
    elif kind=='words': run['resources']['screening_unique_words_sum']+=1
    elif kind=='parent': row['appended_labels']=[]
    elif kind=='target': arms[0]['target_summary']['status']='invented'
    elif kind=='missing_arm': arms.pop()
    elif kind=='ordering': run['pool_labels']=list(reversed(run['pool_labels']))
    elif kind=='bound': run['sector_ground_energy']-=.1
    assert checker.check_record(record,config)


def test_outcome_free_config_and_structures():
    config=declaration.load_config()
    assert declaration.validate(config)==[]
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
