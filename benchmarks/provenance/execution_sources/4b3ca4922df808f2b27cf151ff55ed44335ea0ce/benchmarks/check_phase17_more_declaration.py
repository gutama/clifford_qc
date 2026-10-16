"""Outcome-free gate for the frozen Phase 17 MORE-ADAPT breadth design."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.phase17_more_design import CONFIG, RECORD, digest, load_config, structure
from benchmarks.metadata_io import read_json, sha256
from benchmarks.archive_provenance import history_kind

EXPECTED_DIGEST = '61046d36a933f9c4a50ea85437aa29b1f2c89e1421ab4b2c2301385b96cf643c'


def history_problems(config, record, root=ROOT):
    """Require full history and a record-only first publication commit."""
    from benchmarks.phase17_ceo_provenance import history_problems as audit
    history_kind(root)
    notes = []
    problems = audit(config, record, root, str(CONFIG.relative_to(ROOT)),
                     str(RECORD.relative_to(ROOT)), notes)
    if any('SKIP' in note for note in notes):
        problems.append('full Git execution history is required')
    first = subprocess.check_output(['git', 'log', '--diff-filter=A', '--format=%H', '--',
                                     str(RECORD.relative_to(ROOT))], cwd=root, text=True).splitlines()
    if not first:
        problems.append('record has no committed publication')
    else:
        changed = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only',
                                           '-r', first[-1]], cwd=root, text=True).strip()
        if changed != str(RECORD.relative_to(ROOT)):
            problems.append('record must first be committed alone')
    return problems


def validate(config, root=ROOT, *, frozen=True, sources=True):
    problems = []
    if config.get('schema') != 'clifford_qc.phase17_more_config.v1':
        problems.append('configuration schema differs')
    if frozen and digest(config) != EXPECTED_DIGEST:
        problems.append('frozen MORE declaration drifted')
    if config['arms'] != ['more_adapt', 'acase_mean']:
        problems.append('arm order differs')
    if config['more']['max_parameters'] != config['more']['max_iterations'] or config['acase']['max_size'] != config['more']['max_iterations']:
        problems.append('operator/parameter budgets must match')
    if (config['common_caps']['new_operator_directions'] != config['more']['max_iterations']
            or config['common_caps']['nonlinear_parameters'] != config['more']['max_parameters']):
        problems.append('common budget caps differ from arm budgets')
    if sha256(root / config['environment_reference']) != config['environment_reference_sha256']:
        problems.append('licensed environment reference changed')
    for case in config['cases']:
        if sha256(root / case['path']) != case['sha256']:
            problems.append('input changed: ' + case['id'])
        if structure(root / case['path']) != case['structure']:
            problems.append('reference/pool structure changed: ' + case['id'])
    if sources and not (root / RECORD.relative_to(ROOT)).exists():
        for path, expected in config['source_bindings'].items():
            if sha256(root / path) != expected:
                problems.append('source changed before execution: ' + path)
    return problems


def main():
    config = load_config()
    problems = validate(config)
    if RECORD.exists():
        problems += history_problems(config, read_json(RECORD))
    if problems:
        raise ValueError('; '.join(problems))
    print('Phase 17 MORE declaration: PASS (outcome-free structure; committed history checked when recorded)')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
