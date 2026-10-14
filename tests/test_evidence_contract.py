"""Evidence labels bind metadata; they never change scientific quantities."""
import hashlib
import json

import pytest

from benchmarks import evidence_metadata as metadata
from clifford_qc.evidence import SCHEMA, declare_evidence, validate_evidence
from clifford_qc.reproducibility import stamp_record


def labels():
    return {"energy": {"tier": "exact_small_sector_simulation", "scope": "fixed two-qubit simulation"}}


def test_producer_declaration_preserves_data_and_does_not_mutate_input():
    original = {"schema": "example.v1", "energy": -2.0}
    result = declare_evidence(original, labels())
    assert original == {"schema": "example.v1", "energy": -2.0}
    assert result["energy"] == original["energy"]
    assert stamp_record(result, {}, allow_environment_migration=True)["evidence_schema"] == SCHEMA


@pytest.mark.parametrize("change", [
    {"evidence_schema": "unknown"}, {"evidence": {}}, {"evidence": "exact"},
    {"evidence": {"energy": {"tier": "hardware_demonstrated", "scope": "claim"}}},
    {"evidence": {"energy": {"tier": "exact_algebra", "scope": ""}}},
    {"evidence": {"energy": {"tier": "exact_algebra"}}},
    {"evidence": {"": {"tier": "exact_algebra", "scope": "claim"}}},
])
def test_malformed_evidence_fails_at_producer_boundary(change):
    record = declare_evidence({"schema": "example.v1"}, labels()) | change
    with pytest.raises(ValueError):
        stamp_record(record, {}, allow_environment_migration=True)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    records = tmp_path / "benchmarks/reference_results"
    records.mkdir(parents=True)
    path = records / "old.json"
    path.write_text('{"schema":"old.v1","energy":-1}')
    manifest = tmp_path / metadata.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema": "clifford_qc.legacy_evidence.v1",
                                    "records": {path.name: metadata.sha256(path)}}))
    monkeypatch.setattr(metadata, "LEGACY_MANIFEST_SHA256", metadata.sha256(manifest))
    return tmp_path, path


def annotation(root, path):
    payload = {"schema": "clifford_qc.evidence_annotation.v1", "record": path.name,
               "record_sha256": metadata.sha256(path), "evidence_schema": SCHEMA, "evidence": labels()}
    target = root / metadata.ANNOTATIONS / f"{path.name}.evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    return target, payload


def test_sidecar_labels_are_read_and_original_bytes_remain_unchanged(bundle):
    root, path = bundle
    before = path.read_bytes()
    annotation(root, path)
    assert metadata.check_records(root) == []
    assert metadata.evidence_view(path, root=root)["evidence"] == labels()
    assert path.read_bytes() == before


def test_new_record_without_contract_fails_and_valid_contract_passes(bundle):
    root, path = bundle
    new = path.with_name("new.json")
    new.write_text('{"schema":"new.v1"}')
    assert metadata.check_records(root)
    new.write_text(json.dumps(declare_evidence({"schema": "new.v1"}, labels())))
    assert metadata.check_records(root) == []


def test_changed_legacy_bytes_are_not_grandfathered(bundle):
    root, path = bundle
    path.write_text('{"schema":"old.v1","energy":-2}')
    assert metadata.check_records(root)


@pytest.mark.parametrize("top_level", [False, True])
def test_valid_evidence_cannot_rebind_a_modified_historical_record(bundle, top_level):
    root, path = bundle
    path.write_text(json.dumps({"schema": "old.v1", "energy": -999}))
    if top_level:
        path.write_text(json.dumps(declare_evidence(json.loads(path.read_text()), labels())))
    else:
        annotation(root, path)  # Correct digest for the tampered bytes.
    assert any("frozen legacy record digest differs" in error
               for error in metadata.check_records(root))


def test_annotation_for_existing_non_json_record_is_rejected(bundle):
    root, path = bundle
    jsonl = path.with_suffix(".jsonl")
    jsonl.write_text('{"schema":"rows.v1"}\n')
    annotation(root, jsonl)
    assert any("target must be a JSON record" in error for error in metadata.check_records(root))


def test_metadata_reads_utf8_independently_of_locale(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / "unicode.json"
    path.write_text('{"scope":"H₂O"}', encoding="utf-8")
    original = Path.read_text

    def locale_read(self, *args, **kwargs):
        # Simulate a non-UTF-8 default without mutating the process locale.
        kwargs.setdefault("encoding", "ascii")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", locale_read)
    assert metadata.read_json(path) == {"scope": "H₂O"}


@pytest.mark.parametrize("raw", ['{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_shared_metadata_reader_rejects_ambiguous_or_nonfinite_values(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="nonfinite|duplicate"):
        metadata.read_json(path)


def test_sidecar_cannot_hide_record_drift_or_override_evidence(bundle):
    root, path = bundle
    target, payload = annotation(root, path)
    path.write_text(json.dumps(declare_evidence({"schema": "old.v1"}, labels())))
    assert metadata.check_records(root)
    payload["record_sha256"] = metadata.sha256(path)
    target.write_text(json.dumps(payload))
    assert any("override" in error for error in metadata.check_records(root))


def test_unknown_sidecar_tier_and_duplicate_keys_fail(bundle):
    root, path = bundle
    target, payload = annotation(root, path)
    payload["evidence"]["energy"]["tier"] = "quantum_advantage"
    target.write_text(json.dumps(payload))
    assert metadata.check_records(root)
    target.write_text('{"schema":"x","schema":"y"}')
    assert any("duplicate" in error for error in metadata.check_records(root))


def test_unchanged_legacy_nonfinite_diagnostic_is_not_reinterpreted(bundle, monkeypatch):
    root, path = bundle
    path.write_text('{"schema":"old.v1","unreached":Infinity}')
    manifest = root / metadata.MANIFEST
    declared = json.loads(manifest.read_text())
    declared["records"][path.name] = metadata.sha256(path)
    manifest.write_text(json.dumps(declared))
    monkeypatch.setattr(metadata, "LEGACY_MANIFEST_SHA256", metadata.sha256(manifest))
    assert metadata.check_records(root) == []


def test_legacy_manifest_cannot_grandfather_a_new_unlabelled_record(bundle):
    root, path = bundle
    manifest = root / metadata.MANIFEST
    payload = json.loads(manifest.read_text())
    payload["records"]["new.json"] = metadata.sha256(path)
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen legacy"):
        metadata.check_records(root)


def test_current_records_and_annotations_pass():
    assert metadata.check_records() == []


def test_census_recognizes_digest_bound_annotations(bundle, monkeypatch):
    from paper_architecture import make_tables
    root, path = bundle
    annotation(root, path)
    monkeypatch.setattr(make_tables, "ROOT", root)
    monkeypatch.setattr(make_tables, "RECORDS", path.parent)
    census = make_tables.record_census()
    assert census["forms"]["per_quantity_mapping"] == 1
    assert census["forms"]["none"] == 0
    assert census["declarations"][path.name] != hashlib.sha256(b"{}").hexdigest()[:16]
