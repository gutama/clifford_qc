"""Outcome-free Phase 17 CEO design gate; does not solve or optimize."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.phase17_ceo_design import CONFIG, RECORD, LEGACY_CONFIG, LEGACY_RECORD, digest, file_digest, structure

EXPECTED_DIGEST = 'd26536081ba219ea083abb17a8335dcf04d56c8981cc245cce05fc79c1aa4970'
LEGACY_DIGEST = 'c60140c5ffc3daa98f11303ffcf0d9a280aae6b4c5f660d1db62c4430bcc7a6e'
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
    if digest(config) not in {EXPECTED_DIGEST, LEGACY_DIGEST}:
        raise ValueError('frozen CEO declaration drifted')
    return config


def paths(config):
    return (CONFIG, RECORD) if config.get('design_status') == 'post_review_correction' else (LEGACY_CONFIG, LEGACY_RECORD)


def validate(config, root=ROOT, *, provenance=True, check_sources=True, notes=None):
    """Static checks are portable; dedicated CLI also audits historical sources."""
    from benchmarks.phase17_ceo_provenance import history_problems
    notes = [] if notes is None else notes
    problems = []
    config_path, record_path = (p.relative_to(ROOT) for p in paths(config))
    record_exists = (root/record_path).exists()
    if check_sources and not record_exists:
        for relative, expected in config['source_bindings'].items():
            if file_digest(root/relative) != expected:
                problems.append(f'source drift before execution: {relative}')
    for case in config['cases']:
        if file_digest(root/case['path']) != case['sha256']:
            problems.append(f'input drift: {case["id"]}')
        if structure(root/case['path']) != case['structure']:
            problems.append(f'pool/reference structure drift: {case["id"]}')
    if provenance and record_exists:
        record = json.loads((root/record_path).read_text())
        problems += history_problems(config, record, root, config_path, record_path, notes)
    return problems


def main():
    for path in (LEGACY_CONFIG, CONFIG):
        notes = []
        errors = validate(load_config(path), notes=notes)
        for note in notes: print(note)
        if errors: raise ValueError('; '.join(errors))
        suffix = '; provenance unverified' if any('SKIP' in note for note in notes) else ''
        print(f'Phase 17 CEO declaration {path.name}: PASS (outcome-free structure{suffix})')


if __name__ == '__main__':
    try: main()
    except (OSError, ValueError, KeyError) as error:
        print(f'FAIL: {error}'); sys.exit(1)
