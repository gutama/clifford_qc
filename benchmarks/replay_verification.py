"""Cross-platform verification of recorded scientific fields, not plan bytes.

The original same-platform --check and its exact identity criterion remain
unchanged. This separately labelled maintenance contract reports plan-identity
and verified execution-provenance drift and delegates all scientific-value
comparisons to that original checker. It never
rewrites observations or claims equivalence of unrecorded plan bytes.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json, write_json


def compare_recorded_values(expected, actual, original_compare):
    """Isolate execution fingerprints on a copy; retain all other old checks."""
    if 'worker_receipts' in expected or 'worker_receipts' in actual:
        if set(expected['source_sha256']) != set(actual['source_sha256']):
            raise AssertionError('execution source path sets differ')
        for record in (expected, actual):
            manifest = hashlib.sha256(json.dumps(record['source_sha256'], sort_keys=True).encode()).hexdigest()
            for receipt in record['worker_receipts']:
                if receipt['source_manifest_sha256'] != manifest:
                    raise AssertionError('worker source manifest does not match execution source map')
    candidate = deepcopy(actual)
    drift = []

    def visit(left, right, path='record'):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in left.keys() & right.keys():
                plan = key == 'plan_identity' and re.fullmatch(r'record\.rows\[\d+\]', path)
                provenance = key == 'source_manifest_sha256' and re.fullmatch(r'record\.worker_receipts\[\d+\]', path)
                if plan or provenance:
                    a, b = left[key], right[key]
                    if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                           for v in (a, b)):
                        raise AssertionError(f'{path}: invalid execution fingerprint')
                    if a != b:
                        drift.append({'path': path+'.'+key, 'recorded': a, 'replayed': b})
                        right[key] = a
                else:
                    visit(left[key], right[key], path+'.'+key)
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            for i, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f'{path}[{i}]')
    visit(expected, candidate)
    # This retains the producer's existing tolerances, exact counts/statuses,
    # pilot/observation identities and shape checks. No comparison is copied.
    original_compare(expected, candidate)
    return drift


def verify_source_bindings(record, root):
    """Bind each execution's source map to raw Git bytes at its own commit."""
    sources = record['source_sha256']
    commit = record['execution_commit']
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise AssertionError('invalid execution commit')
    if not isinstance(sources, dict) or not sources:
        raise AssertionError('missing execution source map')
    for path, digest in sources.items():
        if (not isinstance(path, str) or '\n' in path or '\r' in path
                or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise AssertionError('invalid execution source binding')
    refs = ''.join(f'{commit}:{path}\n' for path in sources)
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=root, input=refs.encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    data = io.BytesIO(result.stdout)
    for path, digest in sources.items():
        header = data.readline().split()
        if len(header) != 3 or header[1] != b'blob' or not header[2].isdigit():
            raise AssertionError(f'execution source blob unavailable: {path}')
        raw = data.read(int(header[2]))
        if data.read(1) != b'\n' or hashlib.sha256(raw).hexdigest() != digest:
            raise AssertionError(f'execution source digest differs: {path}')
    if data.read():
        raise AssertionError('unexpected execution source output')


def publication_binding(module):
    path = module.ROOT/module.RESULT
    first = subprocess.check_output(['git', 'log', '--diff-filter=A', '--format=%H',
                                     '--', module.RESULT], cwd=module.ROOT, text=True).splitlines()
    if not first:
        raise ValueError('full publication history is required')
    published = subprocess.check_output(['git', 'show', f'{first[-1]}:{module.RESULT}'],
                                        cwd=module.ROOT)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(published).hexdigest() != digest:
        raise ValueError('published observation bytes changed')
    return path, digest, first[-1]


def verify(module, checker, *, campaign, output):
    if Path(output).exists():
        raise FileExistsError('verification receipt already exists')
    record_path, digest, published = publication_binding(module)
    if Path(output).resolve() == record_path.resolve():
        raise ValueError('verification cannot overwrite observations')
    record = read_json(record_path)
    checker.check(record)  # Independent scientific, cost and ancestry checks.
    if 'source_sha256' in record:
        verify_source_bindings(record, module.ROOT)
    original = module.compare_records
    summary = {'schema': 'clifford_qc.recorded_numerical_replay.v1', 'campaign': campaign,
               'record_sha256': digest, 'publication_commit': published,
               'comparison': 'original_recorded_values_and_tolerances',
               'exact_plan_bytes_claim': False, 'scientific_evidence_claim': False}

    def compare(expected, actual):
        if 'source_sha256' in actual:
            verify_source_bindings(actual, module.ROOT)
        drift = compare_recorded_values(expected, actual, original)
        plans = [row for row in drift if row['path'].endswith('.plan_identity')]
        provenance = [row for row in drift if row['path'].endswith('.source_manifest_sha256')]
        summary.update(status='recorded_values_verified', outcomes=len(actual['rows']),
                       exact_plan_fingerprints_match=not plans,
                       original_strict_replay_passed=not drift,
                       plan_fingerprint_drift_count=len(plans), plan_fingerprint_drift=plans,
                       execution_provenance_drift_count=len(provenance), execution_provenance_drift=provenance)

    module.compare_records = compare
    try:
        # Suppress the old main's generic "replay passed" sentence; this
        # contract emits its own precise status including fingerprint drift.
        with redirect_stdout(io.StringIO()):
            module.main(['--check', str(record_path)])
        if summary.get('status') != 'recorded_values_verified':
            raise AssertionError('producer did not invoke recorded-value verification')
        if hashlib.sha256(record_path.read_bytes()).hexdigest() != digest:
            raise AssertionError('verification changed observations')
    except BaseException as error:
        summary.update(status='verification_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        module.compare_records = original
        write_json(Path(output), summary)
    print(f"Recorded numerical replay verified: {summary['outcomes']} outcomes; "
          f"{summary['plan_fingerprint_drift_count']} exact plan fingerprint differences; "
          f"{summary['execution_provenance_drift_count']} execution provenance differences")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('sampled_ratio', 'cross_domain_ratio'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.campaign == 'sampled_ratio':
        from benchmarks import run_sampled_ratio as producer, check_sampled_ratio as checker
    else:
        from benchmarks import run_cross_domain_ratio as producer, check_cross_domain_ratio as checker
    verify(producer, checker, campaign=args.campaign, output=args.output)


if __name__ == '__main__':
    main()
