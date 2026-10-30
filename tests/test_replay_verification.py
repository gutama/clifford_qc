"""Fingerprint diagnostics must never mask a recorded scientific difference."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.replay_verification import compare_recorded_values, publication_binding, verify, verify_source_bindings
from benchmarks.run_sampled_ratio import compare_records


def records():
    expected = {'rows': [{'plan_identity': 'a'*64, 'pilot_identity': 'p',
                          'observation_identity': 'o', 'estimate': 1.,
                          'evaluation_counts': [10, 20], 'status': 'resolved'}]}
    actual = deepcopy(expected)
    actual['rows'][0]['plan_identity'] = 'b'*64
    return expected, actual


def test_identity_drift_is_reported_and_neither_record_is_rewritten():
    expected, actual = records()
    saved = deepcopy(actual)
    with pytest.raises(AssertionError):
        compare_records(expected, actual)  # Original strict control still fails.
    drift = compare_recorded_values(expected, actual, compare_records)
    assert drift == [{'path': 'record.rows[0].plan_identity', 'recorded': 'a'*64, 'replayed': 'b'*64}]
    assert actual == saved and expected['rows'][0]['plan_identity'] == 'a'*64


@pytest.mark.parametrize('field,value', [('estimate', 1.001), ('evaluation_counts', [11, 20]),
                                       ('pilot_identity', 'changed'),
                                       ('observation_identity', 'changed'),
                                       ('status', 'failed'), ('plan_identity', 'invalid')])
def test_drift_does_not_excuse_numeric_discrete_observation_or_invalid_hash_tampering(field, value):
    expected, actual = records()
    actual['rows'][0][field] = value
    with pytest.raises(AssertionError):
        compare_recorded_values(expected, actual, compare_records)


def test_original_numeric_tolerance_is_used_without_broadening():
    expected, actual = records()
    actual['rows'][0]['estimate'] += 1e-12
    assert len(compare_recorded_values(expected, actual, compare_records)) == 1
    actual['rows'][0]['estimate'] += 1e-7
    with pytest.raises(AssertionError):
        compare_recorded_values(expected, actual, compare_records)


def make_publication(tmp_path, payload):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q');git('config', 'user.name', 'Fixture');git('config', 'user.email', 'fixture@example.invalid')
    (tmp_path/'record.json').write_text(json.dumps(payload))
    git('add', '.');git('commit', '-qm', 'published')
    return SimpleNamespace(ROOT=tmp_path, RESULT='record.json')


def test_even_fingerprint_only_edits_to_published_observations_are_rejected(tmp_path):
    expected, actual = records()
    module = make_publication(tmp_path, expected)
    publication_binding(module)
    (tmp_path/'record.json').write_text(json.dumps(actual))
    with pytest.raises(ValueError, match='observation bytes changed'):
        publication_binding(module)


def test_wrapper_runs_old_preflights_labels_mode_and_restores_original_comparator(tmp_path, capsys):
    expected, actual = records()
    module = make_publication(tmp_path, expected)
    original_bytes = (tmp_path/'record.json').read_bytes()
    module.compare_records = compare_records
    def original_main(argv):
        assert argv == ['--check', str(tmp_path/'record.json')]
        module.compare_records(expected, actual)
        print('generic replay passed')
    module.main = original_main
    checked = []
    checker = SimpleNamespace(check=lambda record: checked.append(record))
    summary = verify(module, checker, campaign='toy', output=tmp_path/'verification.json')
    assert summary['status'] == 'recorded_values_verified'
    assert summary['exact_plan_fingerprints_match'] is False
    assert summary['original_strict_replay_passed'] is False
    assert summary['plan_fingerprint_drift_count'] == 1
    assert summary['exact_plan_bytes_claim'] is False
    assert checked == [expected] and module.compare_records is compare_records
    assert (tmp_path/'record.json').read_bytes() == original_bytes
    output = capsys.readouterr().out
    assert 'generic replay passed' not in output and '1 exact plan fingerprint differences' in output


def worker_record(digest):
    sources = {'source.py': digest}
    return {'source_sha256': sources, 'worker_receipts': [{
        'source_manifest_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}]}


def test_validated_worker_provenance_drift_is_retained_in_receipt():
    from benchmarks.run_cross_domain_ratio import compare_records as compare
    expected, actual = worker_record('a'*64), worker_record('b'*64)
    saved = deepcopy(actual)
    drift = compare_recorded_values(expected, actual, compare)
    assert len(drift) == 1 and drift[0]['path'] == 'record.worker_receipts[0].source_manifest_sha256'
    assert drift[0]['recorded'] != drift[0]['replayed'] and actual == saved


def test_forged_worker_source_manifest_cannot_be_excused():
    expected, actual = worker_record('a'*64), worker_record('b'*64)
    actual['worker_receipts'][0]['source_manifest_sha256'] = 'c'*64
    with pytest.raises(AssertionError, match='worker source manifest'):
        compare_recorded_values(expected, actual, compare_records)


def test_changed_execution_source_path_set_cannot_be_excused():
    expected, actual = worker_record('a'*64), worker_record('b'*64)
    actual['source_sha256']['other.py'] = 'c'*64
    with pytest.raises(AssertionError, match='source path sets differ'):
        compare_recorded_values(expected, actual, compare_records)


def test_execution_source_hash_must_match_raw_blob_at_named_commit(tmp_path):
    module = make_publication(tmp_path, {})
    raw = (tmp_path/'record.json').read_bytes()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=module.ROOT, text=True).strip()
    record = {'execution_commit': commit, 'source_sha256': {'record.json': hashlib.sha256(raw).hexdigest()}}
    verify_source_bindings(record, module.ROOT)
    record['source_sha256']['record.json'] = 'a'*64
    with pytest.raises(AssertionError, match='execution source digest differs'):
        verify_source_bindings(record, module.ROOT)


def test_both_names_of_the_same_frozen_plan_fingerprint_are_reported():
    expected, actual = records()
    for record in (expected, actual):
        record['rows'][0]['frozen_plan_identity'] = record['rows'][0]['plan_identity']
    drift = compare_recorded_values(expected, actual, compare_records)
    assert {row['path'] for row in drift} == {
        'record.rows[0].plan_identity', 'record.rows[0].frozen_plan_identity'}


def test_disagreeing_plan_fingerprint_aliases_are_rejected():
    expected, actual = records()
    actual['rows'][0]['frozen_plan_identity'] = 'c'*64
    with pytest.raises(AssertionError, match='plan fingerprint aliases disagree'):
        compare_recorded_values(expected, actual, compare_records)
