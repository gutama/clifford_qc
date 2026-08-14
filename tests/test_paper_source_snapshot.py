"""Regression gates for the restored P1 source manifest."""

from __future__ import annotations

import json
from pathlib import Path

from paper_a_case_subspaces.check_snapshot import MANIFEST, snapshot_problems


SNAPSHOT = Path(__file__).resolve().parents[1] / "paper_a_case_subspaces"


def test_published_snapshot_manifest_covers_every_restored_source():
    problems, matched_text, generated = snapshot_problems()
    assert problems == []
    assert matched_text == 16
    assert len(generated) == 4


def test_snapshot_gate_rejects_a_manifest_with_one_entry_removed(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    removed = manifest["files"].pop()
    truncated = tmp_path / "SOURCE_SNAPSHOT.json"
    truncated.write_text(json.dumps(manifest), encoding="utf-8")

    problems, matched_text, _ = snapshot_problems(SNAPSHOT, truncated)
    assert matched_text == 15
    assert any("has 19 entries, expected 20" in problem for problem in problems)
    assert any(
        problem == f"unlisted snapshot file: {removed['path']}"
        for problem in problems
    )
