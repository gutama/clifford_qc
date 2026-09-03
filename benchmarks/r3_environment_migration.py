"""Auditable lineage for the R3 record-environment migration.

The R3c preregistration cites the R3S and R3b records that existed when it
landed. Rebuilding those records under the already-declared Python 3.12 stack
must not rewrite that history, but the live records also cannot retain false
3.11 provenance. This module keeps both identities explicit: the frozen
historical digests and the exact successor records produced by the migration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = HERE / "migrations" / "r3_environment_3_12.json"
SCHEMA = "clifford_qc.record_environment_migration.v1"
SOURCE_MAIN_COMMIT = "44cd16fb8ea6263901deb81eb5ba4c2af1de6283"
TARGET_ENVIRONMENT = {
    "python_minor": "3.12",
    "numpy": "2.5.2",
    "scipy": "1.18.0",
    "stim": "1.16.0",
}
FROZEN_CONFIG_SHA256 = {
    "priceability_screen": "f43fac4ad8386ecdfa279cfbf5e61661c7a24f3459c7f2f229ad10b0b7f9ec3e",
    "r3b_margin_stop_probe": "319af3cfd047d336b0d01fea21fe5a980e8bff84777b7ba9e67dffb439da8ef1",
    "r3c_lih_full_cost": "cfbc97f3225ebdf8bdd8bcbddd2141214649664ef238a7cc98188db6f4160daa",
}
RECORD_ANCHORS = {
    "priceability_screen": {
        "before": {
            "sha256": "06669c5047fbe16e56ab829e98cc9325b0ad52f94dd89be8eb33f373239477d9",
            "git_blob_sha1": "8ac3a1b62b2aef4560ff4d0dc81c692820a39bcd",
        },
        "after": {
            "sha256": "61ee716540b9aad7be49193255dac12a097a1c46000f6b0f8534139a646a175d",
            "git_blob_sha1": "bf648f05ce60a0e5213d00f3bffae256090b6b7d",
        },
    },
    "r3b_margin_stop_probe": {
        "before": {
            "sha256": "f68213315b203420db6947711fcd94bef93ec202dc2734ef4c7c94f38a9c7623",
            "git_blob_sha1": "7c7eb9b22990ceab972ce8d5dc5580df63e5ab30",
        },
        "after": {
            "sha256": "108dde4e95d2df08aa6ebca01f3fe179b628f4d080feb800bf6eeeb9bbff8b83",
            "git_blob_sha1": "8d28b6187d1213e5294f3dd82c279bade403196a",
        },
    },
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _minor(version: object) -> object:
    return ".".join(version.split(".")[:2]) if isinstance(version, str) else version


def load_manifest(path: Path = MANIFEST) -> dict:
    payload = _read(path)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported R3 environment-migration manifest schema")
    return payload


def record_successor_problems(
    name: str,
    *,
    historical_sha256: str | None = None,
    historical_git_blob_sha1: str | None = None,
    manifest: dict | None = None,
) -> list[str]:
    """Verify that a live record is the declared successor of a frozen one."""
    problems: list[str] = []
    manifest = load_manifest() if manifest is None else manifest
    entry = manifest.get("records", {}).get(name)
    anchors = RECORD_ANCHORS.get(name)
    if not isinstance(entry, dict) or not isinstance(anchors, dict):
        return [f"migration has no unambiguous record entry for {name!r}"]

    before = entry.get("before", {})
    after = entry.get("after", {})
    for side in ("before", "after"):
        declared = entry.get(side, {})
        if {key: declared.get(key) for key in ("sha256", "git_blob_sha1")} != anchors[side]:
            problems.append(f"{name} {side} anchor differs from the reviewed migration")
    if historical_sha256 is not None and before.get("sha256") != historical_sha256:
        problems.append(f"{name} historical SHA-256 no longer matches its preregistration")
    if (
        historical_git_blob_sha1 is not None
        and before.get("git_blob_sha1") != historical_git_blob_sha1
    ):
        problems.append(f"{name} historical Git blob no longer matches its preregistration")

    path_value = entry.get("path")
    if not isinstance(path_value, str):
        return problems + [f"{name} migration path is missing"]
    path = ROOT / path_value
    if not path.is_file():
        return problems + [f"{name} migrated record is missing at {path_value}"]
    if _sha256(path) != after.get("sha256"):
        problems.append(f"{name} migrated record SHA-256 drifted")
    if _git_blob_sha1(path) != after.get("git_blob_sha1"):
        problems.append(f"{name} migrated record Git blob drifted")
    return problems


def _record_environment(record: dict) -> dict[str, object]:
    provenance = record.get("provenance", {})
    dependencies = provenance.get("dependencies", {})
    return {
        "python_minor": _minor(provenance.get("python")),
        "numpy": dependencies.get("numpy"),
        "scipy": dependencies.get("scipy"),
        "stim": dependencies.get("stim"),
    }


def _priceability_summary(record: dict) -> dict:
    summary = record.get("summary", {})
    return {
        "admissible_for_a_probe": summary.get("admissible_for_a_probe"),
        "admitted_only_by_the_rule_change": summary.get(
            "admitted_only_by_the_rule_change"
        ),
        "rejected": sorted(summary.get("rejected", {})),
    }


def _r3b_decision(record: dict) -> dict:
    decision = record.get("decision", {})
    keys = (
        "status",
        "evaluated_cells",
        "resolved_cells",
        "unresolved_cells",
        "cells_at_or_beyond_ceiling",
        "full_run_authorized",
        "eligible_for_full_run",
    )
    return {key: decision.get(key) for key in keys}


def _r3b_diagnosis(record: dict) -> dict:
    prediction = record.get("screen_prediction", {})
    keys = (
        "grid_fit_failures",
        "confirmation_failures",
        "unclassified_failures",
        "verdict",
    )
    return {key: prediction.get(key) for key in keys}


def r3b_findings(record: dict) -> tuple[dict, dict]:
    """Return the decision and diagnostic fields whose history R3c preserves."""
    return _r3b_decision(record), _r3b_diagnosis(record)


def migration_problems(manifest: dict | None = None) -> list[str]:
    """Validate hashes, environments, and the scientific migration boundary."""
    problems: list[str] = []
    manifest = load_manifest() if manifest is None else manifest
    if manifest.get("status") != "completed_rebuild_not_relabel":
        problems.append("migration does not declare a completed genuine rebuild")
    if manifest.get("source_main_commit") != SOURCE_MAIN_COMMIT:
        problems.append("migration no longer names the exact source main commit")
    if manifest.get("target_environment") != TARGET_ENVIRONMENT:
        problems.append("migration target environment drifted")

    configs = manifest.get("frozen_configs", {})
    for name, expected in FROZEN_CONFIG_SHA256.items():
        entry = configs.get(name, {})
        if entry.get("sha256") != expected:
            problems.append(f"{name} manifest config anchor drifted")
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str) or _sha256(ROOT / path_value) != expected:
            problems.append(f"{name} frozen config changed during the migration")

    for name in RECORD_ANCHORS:
        problems += record_successor_problems(name, manifest=manifest)
        entry = manifest.get("records", {}).get(name, {})
        path_value = entry.get("path")
        if isinstance(path_value, str) and (ROOT / path_value).is_file():
            record = _read(ROOT / path_value)
            if _record_environment(record) != TARGET_ENVIRONMENT:
                problems.append(f"{name} was not rebuilt under the target environment")

    entries = manifest.get("records", {})
    screen_entry = entries.get("priceability_screen", {})
    screen_path = screen_entry.get("path")
    if isinstance(screen_path, str) and (ROOT / screen_path).is_file():
        screen = _read(ROOT / screen_path)
        current = _priceability_summary(screen)
        if current != screen_entry.get("after", {}).get("summary"):
            problems.append("priceability-screen verdicts differ from the migration record")
        if screen_entry.get("before", {}).get("summary") != current:
            problems.append("priceability-screen scientific summary moved across migration")

    r3b_entry = entries.get("r3b_margin_stop_probe", {})
    r3b_path = r3b_entry.get("path")
    if isinstance(r3b_path, str) and (ROOT / r3b_path).is_file():
        r3b = _read(ROOT / r3b_path)
        if _r3b_decision(r3b) != r3b_entry.get("after", {}).get("decision"):
            problems.append("R3b migrated decision differs from the recorded redraw")
        if _r3b_diagnosis(r3b) != r3b_entry.get("after", {}).get("diagnosis"):
            problems.append("R3b migrated diagnosis differs from the recorded redraw")
        before = r3b_entry.get("before", {})
        if before.get("decision", {}).get("status") != "rejected_unresolved_at_frozen_grid":
            problems.append("R3b historical rejection is not preserved")
        if before.get("diagnosis", {}).get("grid_fit_failures") != 0:
            problems.append("R3b historical grid-fit finding is not preserved")
        decision = _r3b_decision(r3b)
        diagnosis = _r3b_diagnosis(r3b)
        if decision.get("status") != "rejected_unresolved_at_frozen_grid":
            problems.append("R3b migration changed the preregistered gate's rejection")
        if decision.get("full_run_authorized") is not False:
            problems.append("R3b migration silently authorizes the full run")
        if decision.get("eligible_for_full_run") is not False:
            problems.append("R3b migration silently makes the bank eligible")
        if diagnosis.get("grid_fit_failures") != 0:
            problems.append("R3b migration changed the zero grid-fit-failure finding")
    if not manifest.get("claim_boundary"):
        problems.append("migration has no claim boundary")
    return problems
