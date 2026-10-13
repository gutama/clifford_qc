"""Outcome-free Phase 17 CEO design gate; does not solve or optimize."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.phase17_ceo_design import CONFIG, RECORD, digest, file_digest, structure

EXPECTED_DIGEST = 'c60140c5ffc3daa98f11303ffcf0d9a280aae6b4c5f660d1db62c4430bcc7a6e'
FORBIDDEN = {'energy','energy_error','results','trajectory','winner','target_summary',
             'optimizer_trace','sector_ground_energy','reference_energy'}


def _walk(value):
    if isinstance(value,dict):
        if FORBIDDEN.intersection(value):
            raise ValueError('declaration contains an outcome')
        for child in value.values(): _walk(child)
    elif isinstance(value,list):
        for child in value: _walk(child)


def load_config(path=CONFIG):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError('duplicate declaration key')
            result[key]=value
        return result
    config = json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                        parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    _walk(config)
    if digest(config) != EXPECTED_DIGEST:
        raise ValueError('frozen CEO declaration drifted')
    return config


def validate(config, root=ROOT):
    problems = []
    for relative,expected in config['source_bindings'].items():
        if file_digest(root/relative) != expected:
            problems.append(f'source drift: {relative}')
    for case in config['cases']:
        if file_digest(root/case['path']) != case['sha256']:
            problems.append(f'input drift: {case["id"]}')
        if structure(root/case['path']) != case['structure']:
            problems.append(f'pool/reference structure drift: {case["id"]}')
    if (root/RECORD.relative_to(ROOT)).exists():
        record=json.loads((root/RECORD.relative_to(ROOT)).read_text())
        source=record['provenance']['git_sha']
        historic=subprocess.run(['git','show',f'{source}:{CONFIG.relative_to(ROOT)}'],
                                 cwd=root,capture_output=True,text=True,check=True)
        if digest(json.loads(historic.stdout)) != digest(config):
            problems.append('execution commit lacks the frozen declaration')
        commit=subprocess.check_output(['git','log','--format=%H','--diff-filter=A','--',
                                        str(CONFIG.relative_to(ROOT))],cwd=root,text=True).strip().splitlines()[-1]
        if commit == source or subprocess.run(['git','merge-base','--is-ancestor',commit,source],cwd=root).returncode:
            problems.append('declaration must strictly precede execution')
    return problems


if __name__ == '__main__':
    try:
        errors=validate(load_config())
        for error in errors: print(error)
        if errors: sys.exit(1)
        print('Phase 17 CEO declaration: PASS (outcome-free structure only)')
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
        print(f'FAIL: {error}'); sys.exit(1)
