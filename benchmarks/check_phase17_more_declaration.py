"""Outcome-free gate for the frozen Phase 17 MORE-ADAPT breadth design."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.phase17_more_design import CONFIG, RECORD, EXECUTION_SOURCES, digest, load_config, structure
from benchmarks.metadata_io import read_json, sha256
from benchmarks.archive_provenance import history_kind, verify_archive, archive_notice

EXPECTED_DIGEST = '61046d36a933f9c4a50ea85437aa29b1f2c89e1421ab4b2c2301385b96cf643c'


def history_problems(config, record, root=ROOT, *, archive=False):
    """Audit full history, or explicitly verify packaged historical content.

    Archive bindings preserve execution sources without making any claim about
    declaration ancestry or the record-only first publication commit.
    """
    from benchmarks.phase17_ceo_provenance import history_problems as audit
    kind = history_kind(root, archive=archive)
    if record.get('config_sha256') != digest(config):
        return ['record declaration binding differs']
    if set(record['execution_source_bindings']) != set(EXECUTION_SOURCES):
        return ['execution source binding set differs']
    if record['provenance']['git_dirty'] is not False:
        return ['dirty execution']
    if kind == 'archive':
        if record != read_json(root / RECORD.relative_to(ROOT)):
            return ['inspected archive record differs from packaged record']
        if config != read_json(root / CONFIG.relative_to(ROOT)):
            return ['inspected archive configuration differs from packaged configuration']
        sources = {path: {'sha256': expected} for path, expected in
                   (config['source_bindings'] | record['execution_source_bindings']).items()}
        verify_archive(root, str(RECORD.relative_to(ROOT)), str(CONFIG.relative_to(ROOT)),
                       record['provenance']['git_sha'], sources)
        return []
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--archive', action='store_true',
                      help='verify packaged historical bindings; chronology remains unverified')
    mode.add_argument('--require-git', action='store_true',
                      help='require full Git ancestry (the default)')
    args = parser.parse_args(argv)
    history_kind(ROOT, archive=args.archive)
    if args.archive and not RECORD.is_file():
        raise ValueError('archive verification requires the published MORE record')
    config = load_config()
    problems = validate(config)
    if RECORD.exists():
        problems += history_problems(config, read_json(RECORD), archive=args.archive)
    if problems:
        raise ValueError('; '.join(problems))
    if args.archive and RECORD.exists():
        archive_notice()
    print('Phase 17 MORE declaration: PASS (outcome-free structure; '
          + ('archive content checked)' if args.archive else 'committed history checked when recorded)'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
