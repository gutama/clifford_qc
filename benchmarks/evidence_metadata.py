"""Read digest-bound evidence sidecars without rewriting historical records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from clifford_qc.evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "benchmarks/evidence/legacy_records.json"
LEGACY_MANIFEST_SHA256 = "dadd116852a2aecc5457011988aaef43f4727abddba0c30d303ec4b9edd2268e"
ANNOTATIONS = "benchmarks/evidence/annotations"


def read_json(path: Path, *, allow_nonfinite: bool = False):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        if allow_nonfinite:
            return {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": -float("inf")}[value]
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_view(path: Path, *, root: Path = ROOT) -> dict:
    """Overlay a validated, hash-bound label map for census/gate use only."""
    # Scientific-value validation belongs to each record's own gate. Some
    # unchanged historical diagnostics intentionally encode Infinity.
    record = read_json(path, allow_nonfinite=True)
    if not isinstance(record, dict):
        raise ValueError(f"{path.name}: JSON record must be an object")
    sidecar = root / ANNOTATIONS / f"{path.name}.evidence.json"
    if not sidecar.exists():
        return record
    annotation = read_json(sidecar)
    if (not isinstance(annotation, dict)
            or set(annotation) != {"schema", "record", "record_sha256", "evidence_schema", "evidence"}
            or annotation["schema"] != "clifford_qc.evidence_annotation.v1"
            or annotation["record"] != path.name
            or annotation["record_sha256"] != sha256(path)):
        raise ValueError(f"{path.name}: evidence annotation identity/digest differs")
    validate_evidence(annotation)
    if "evidence_schema" in record or "evidence" in record:
        raise ValueError(f"{path.name}: sidecar may not override top-level evidence")
    return dict(record, evidence_schema=annotation["evidence_schema"], evidence=annotation["evidence"])


def check_records(root: Path = ROOT) -> list[str]:
    """Grandfather only byte-identical historical files; enforce v1 on new ones."""
    if sha256(root / MANIFEST) != LEGACY_MANIFEST_SHA256:
        raise ValueError("frozen legacy evidence manifest digest differs")
    manifest = read_json(root / MANIFEST)
    if set(manifest) != {"schema", "records"} or manifest["schema"] != "clifford_qc.legacy_evidence.v1":
        raise ValueError("legacy evidence manifest identity differs")
    legacy = manifest["records"]
    if not isinstance(legacy, dict) or any(
        Path(name).name != name or not name.endswith(".json")
        or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in legacy.items()
    ):
        raise ValueError("legacy evidence manifest entries are malformed")
    records = root / "benchmarks/reference_results"
    problems = [f"legacy record missing: {name}" for name in legacy if not (records / name).is_file()]
    for path in sorted(records.glob("*.json")):
        try:
            payload = evidence_view(path, root=root)
            if "evidence_schema" in payload:
                validate_evidence(payload)
            elif legacy.get(path.name) != sha256(path):
                raise ValueError("new/changed record requires a validated v1 evidence declaration")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f"{path.name}: {exc}")
    for annotation in (root / ANNOTATIONS).glob("*.evidence.json"):
        if not (records / annotation.name.removesuffix(".evidence.json")).is_file():
            problems.append(f"orphan evidence annotation: {annotation.name}")
    return problems
