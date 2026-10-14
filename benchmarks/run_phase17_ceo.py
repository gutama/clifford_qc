"""Run the frozen exact CEO pool comparison; default CLI is outcome-free."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.check_phase17_ceo_declaration import load_config, validate
from benchmarks.phase17_ceo_design import CONFIG, RECORD, digest, file_digest, target_summary
from clifford_qc.algorithms.exchange import run_exchange_adapt, _prepare_exchange
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.reproducibility import stamp_record


def run_campaign(config):
    cases=[]
    for case in config['cases']:
        model=fcidump_model(ROOT/case['path'],name=case['id'],integral_tolerance=1e-12)
        prepared=_prepare_exchange(model,config['solver']['max_sector_dimension'])
        arms=[]
        for arm in config['arms']:
            run=run_exchange_adapt(model,arm=arm,_prepared=prepared,**config['solver'])
            arms.append({'run':run,'target_summary':target_summary(run,config['target_hartree'])})
        cases.append({'id':case['id'],'role':case['role'],'arms':arms})
    return {'schema':'clifford_qc.phase17_ceo.v2','config_sha256':digest(config),
            'claim_boundary':config['claim_boundary'],'cases':cases,
            'execution_source_bindings':{path:file_digest(ROOT/path) for path in
                ['benchmarks/run_phase17_ceo.py','benchmarks/check_phase17_ceo.py',
                 'benchmarks/check_phase17_ceo_declaration.py']}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    config=load_config()
    problems=validate(config)
    if problems: raise ValueError('; '.join(problems))
    if not args.execute:
        print('Phase 17 CEO: frozen declaration validated; no solver run')
        return
    if RECORD.exists(): raise ValueError('record already exists; another campaign is not authorized')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('execution requires clean committed sources')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    declaration=subprocess.check_output(['git','log','-1','--format=%H','--',
                                        str(CONFIG.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    if head == declaration: raise ValueError('declaration must precede execution source')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name) != '1': raise ValueError(f'{name}=1 required')
    provenance=stamp_record({})['provenance']  # environment refusal precedes the exclusive claim
    if provenance['git_sha'] != head or provenance['git_dirty']:
        raise ValueError('execution provenance must identify the clean actual HEAD')
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=ROOT,text=True).strip())
    if not common.is_absolute(): common=ROOT/common
    claim=common/'phase17-ceo-review-execution.claim'
    with claim.open('x') as handle:
        json.dump({'source':head,'config_sha256':digest(config)},handle)
    record=stamp_record(run_campaign(config),provenance)
    # Retain the complete observed trajectory before a checker can refuse it.
    # Failure consumes the claim but does not discard its scientific evidence.
    with claim.with_suffix('.diagnostic.json').open('x') as handle:
        json.dump(record,handle,allow_nan=False)
    from benchmarks.check_phase17_ceo import check_record
    problems=check_record(record,config)
    if problems: raise ValueError('independent validation failed: '+'; '.join(problems[:10]))
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=RECORD.parent,delete=False) as handle:
            temporary=Path(handle.name)
            json.dump(record,handle,separators=(',',':'),allow_nan=False); handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,RECORD)  # fails atomically if a record already exists
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    print(f'Phase 17 CEO: recorded and independently checked {len(config["cases"])*len(config["arms"])} arms')


if __name__ == '__main__':
    try: main()
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        print(f'FAIL: {error}'); sys.exit(1)
