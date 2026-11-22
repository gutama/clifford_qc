"""Regression tests for paper_weighted_precision tables, figures, and manuscript gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from paper_weighted_precision import check_manuscript, make_figures, make_tables

ROOT = Path(__file__).resolve().parents[1]


def test_tables_and_macros_are_deterministic_and_current():
    rendered = make_tables.render_all()
    assert set(rendered) == set(make_tables.TABLE_FILES)
    for name, expected in rendered.items():
        actual = (make_tables.TABLES_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert actual == expected.replace("\r\n", "\n"), f"stale table/macro: {name}"
    manifest = json.loads(make_tables.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == make_tables.build_manifest(rendered)


def test_figure_manifest_and_pdfs_are_current():
    manifest = json.loads(make_figures.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == make_figures.build_manifest()
    for fname in make_figures.FIGURE_NAMES:
        fpath = make_figures.ASSETS / fname
        assert fpath.exists() and fpath.stat().st_size > 1000


def test_manuscript_gate_and_self_test_pass():
    proc = subprocess.run(
        [sys.executable, "paper_weighted_precision/check_manuscript.py", "--self-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "passed" in proc.stdout


def test_manuscript_checker_rejects_column_mismatch_and_missing_caveat():
    base_tex = check_manuscript.TEX.read_text(encoding="utf-8")
    base_bib = check_manuscript.BIB.read_text(encoding="utf-8")

    # Wrong column count in Table I preamble
    bad_cols = base_tex.replace(r"\begin{tabular}{l r l r l r r r r}", r"\begin{tabular}{l r l r l r r r}", 1)
    errs = check_manuscript.check_tex_text(bad_cols, base_bib)
    assert any("tabular column mismatch" in e for e in errs)

    # Removing the honest P1a failure disclosure must fail
    no_caveat = base_tex.replace(r"all\_passed = false", "all passed")
    errs2 = check_manuscript.check_tex_text(no_caveat, base_bib)
    assert any("missing required provenance/caveat marker" in e for e in errs2)
