"""The retained P2 tables bind historical inputs without requiring Git."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from paper_acase import source_snapshot


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def exported_paper(tmp_path):
    """Only the paper and its declared inputs; no repository history."""
    shutil.copytree(ROOT / "paper_acase", tmp_path / "paper_acase",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = json.loads((tmp_path / source_snapshot.MANIFEST_RELATIVE).read_text())
    for entry in manifest["sources"]:
        relative = Path(entry["path"])
        destination = tmp_path / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
    return tmp_path


def _manifest(root):
    return json.loads((root / source_snapshot.MANIFEST_RELATIVE).read_text())


def _write_manifest(root, manifest):
    (root / source_snapshot.MANIFEST_RELATIVE).write_text(json.dumps(manifest))


def _run(root, script):
    env = dict(os.environ, PATH="")
    return subprocess.run([sys.executable, f"paper_acase/{script}"], cwd=root,
                          env=env, capture_output=True, text=True)


def test_checkout_and_export_verify_the_same_content(exported_paper):
    assert source_snapshot.snapshot_problems() == []
    assert not (exported_paper / ".git").exists()
    assert source_snapshot.snapshot_problems(exported_paper) == []


def test_export_regenerates_and_checks_all_tables_without_git(exported_paper):
    tables = exported_paper / "paper_acase" / "tables"
    before = {path.name: path.read_bytes() for path in tables.glob("*.tex")}
    regenerated = _run(exported_paper, "make_tables.py")
    assert regenerated.returncode == 0, regenerated.stdout + regenerated.stderr
    assert {path.name: path.read_bytes() for path in tables.glob("*.tex")} == before
    checked = _run(exported_paper, "check_manuscript.py")
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.parametrize("original", [
    "benchmarks/reference_results/clifford_hierarchy_h4.json",
    "benchmarks/reference_results/acase_ladder_summary.csv",
])
def test_changed_archived_or_unchanged_input_is_detected(exported_paper, original):
    tables = exported_paper / "paper_acase" / "tables"
    before = {path.name: path.read_bytes() for path in tables.glob("*.tex")}
    path = source_snapshot.source_path(original, exported_paper)
    path.write_bytes(path.read_bytes() + b"\n")
    problems = source_snapshot.snapshot_problems(exported_paper)
    assert any("table snapshot source Git blob drifted" in p for p in problems)
    assert any("table snapshot source SHA-256 drifted" in p for p in problems)
    refused = _run(exported_paper, "make_tables.py")
    assert refused.returncode != 0
    assert "table snapshot verification failed" in refused.stdout
    assert "Traceback" not in refused.stderr
    assert {path.name: path.read_bytes() for path in tables.glob("*.tex")} == before


def test_live_record_changes_cannot_migrate_the_manuscript(exported_paper):
    manifest = _manifest(exported_paper)
    for entry in manifest["sources"]:
        if entry["source_path"] != entry["path"]:
            live = exported_paper / entry["source_path"]
            live.parent.mkdir(parents=True, exist_ok=True)
            live.write_text('{"replacement_live_record": true}\n')
    assert source_snapshot.snapshot_problems(exported_paper) == []
    regenerated = _run(exported_paper, "make_tables.py")
    assert regenerated.returncode == 0, regenerated.stdout + regenerated.stderr
    assert source_snapshot.snapshot_problems(exported_paper) == []


def test_changed_scientific_cell_is_detected(exported_paper):
    table = exported_paper / "paper_acase" / "tables" / "clifford_hierarchy.tex"
    table.write_bytes(table.read_bytes().replace(b"2006", b"1886"))
    assert any("retained manuscript table body drifted" in p
               for p in source_snapshot.snapshot_problems(exported_paper))


def test_historical_generator_is_bound_as_inert_bytes(exported_paper):
    recipe = exported_paper / source_snapshot.HISTORICAL_GENERATOR
    recipe.write_bytes(recipe.read_bytes() + b"# changed historical recipe\n")
    assert any("historical generator SHA-256 drifted" in p
               for p in source_snapshot.snapshot_problems(exported_paper))


@pytest.mark.parametrize("field,value", [
    ("schema", "clifford_qc.paper_table_snapshot.v2"),
    ("arxiv", "2608.00560"),
    ("sources", []),
    ("tables", []),
])
def test_manifest_boundary_tampering_is_detected(exported_paper, field, value):
    manifest = _manifest(exported_paper)
    manifest[field] = value
    _write_manifest(exported_paper, manifest)
    assert source_snapshot.snapshot_problems(exported_paper)


def test_manifest_cannot_redirect_a_source_outside_the_export(exported_paper):
    manifest = _manifest(exported_paper)
    manifest["sources"][0]["path"] = "../unbound.json"
    _write_manifest(exported_paper, manifest)
    assert any("snapshot path escapes" in p
               for p in source_snapshot.snapshot_problems(exported_paper))


def test_manifest_digest_tampering_is_detected(exported_paper):
    manifest = _manifest(exported_paper)
    manifest["sources"][0]["sha256"] = "0" * 64
    _write_manifest(exported_paper, manifest)
    assert any("SHA-256 drifted" in p
               for p in source_snapshot.snapshot_problems(exported_paper))


def test_unknown_source_cannot_fall_back_to_a_live_record(exported_paper):
    with pytest.raises(ValueError, match="undeclared table source"):
        source_snapshot.source_path("benchmarks/reference_results/new.json", exported_paper)


@pytest.mark.parametrize("failure", [
    'out.append("unexpected scientific cell")',
    'raise ValueError("late generation failure")',
])
def test_failed_late_regeneration_preserves_every_original_table(exported_paper, failure):
    tables = exported_paper / "paper_acase" / "tables"
    before = {path.name: path.read_bytes() for path in tables.glob("*.tex")}
    generator = exported_paper / "paper_acase" / "make_tables.py"
    source = generator.read_text()
    marker = '    _write("finite_shot.tex", out)'
    assert source.count(marker) == 1
    generator.write_text(source.replace(marker, f"    {failure}\n{marker}"))
    refused = _run(exported_paper, "make_tables.py")
    assert refused.returncode == 1
    assert "FAIL" in refused.stdout
    assert "Traceback" not in refused.stderr
    assert {path.name: path.read_bytes() for path in tables.glob("*.tex")} == before
    assert not list((exported_paper / "paper_acase").glob(".tables-stage-*"))


@pytest.mark.parametrize("kind", ["manifest", "source", "recipe", "table"])
def test_symlink_escapes_are_refused_before_read_or_publish(exported_paper, kind):
    paths = {
        "manifest": exported_paper / source_snapshot.MANIFEST_RELATIVE,
        "source": source_snapshot.source_path(
            "benchmarks/reference_results/clifford_hierarchy_h4.json", exported_paper),
        "recipe": exported_paper / source_snapshot.HISTORICAL_GENERATOR,
        "table": exported_paper / "paper_acase/tables/clifford_hierarchy.tex",
    }
    path = paths[kind]
    original = path.read_bytes()
    outside = exported_paper.parent / f"{exported_paper.name}-outside"
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    assert any("snapshot path escapes" in p
               for p in source_snapshot.snapshot_problems(exported_paper))
    if kind == "source":
        with pytest.raises(ValueError, match="snapshot path escapes"):
            source_snapshot.source_path(
                "benchmarks/reference_results/clifford_hierarchy_h4.json", exported_paper)
    for script in ("make_tables.py", "check_manuscript.py"):
        refused = _run(exported_paper, script)
        assert refused.returncode == 1, refused.stdout + refused.stderr
        assert "snapshot path escapes" in refused.stdout
        assert "Traceback" not in refused.stderr
        assert outside.read_bytes() == original
        assert path.is_symlink()


@pytest.mark.parametrize("sources", [None, {}, "invalid", 17])
def test_malformed_source_lists_produce_clean_cli_failures(exported_paper, sources):
    manifest = _manifest(exported_paper)
    manifest["sources"] = sources
    _write_manifest(exported_paper, manifest)
    for script in ("source_snapshot.py", "make_tables.py", "check_manuscript.py"):
        refused = _run(exported_paper, script)
        assert refused.returncode == 1, refused.stdout + refused.stderr
        assert "table snapshot sources must be a list" in refused.stdout
        assert "Traceback" not in refused.stderr
