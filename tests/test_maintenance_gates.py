"""Regression contracts for standalone docs and declared summary ownership."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import check_summaries as summaries
from benchmarks import check_docs as docs

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_gate_works_without_pythonpath():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    done = subprocess.run([sys.executable, str(ROOT / "benchmarks/check_docs.py")],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    # Other doc drift can fail the gate, but standalone package discovery must work.
    assert "ModuleNotFoundError" not in done.stderr
    assert "Traceback" not in done.stderr
    assert "optional dependency 'clifford_qc' not installed" not in done.stdout
    assert "OK (" in done.stdout or "FAIL " in done.stdout


def test_documented_help_does_not_get_an_injected_import_path(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    scripts = root / "benchmarks"
    scripts.mkdir(parents=True)
    package = root / "paper_architecture"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n")
    script = scripts / "run_probe.py"
    script.write_text("import paper_architecture\nimport argparse\nargparse.ArgumentParser().parse_args()\n")
    monkeypatch.setattr(docs, "ROOT", root)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    flags, reason = docs.script_flags(script)
    assert flags is None
    assert "paper_architecture" in reason
    assert not reason.startswith("optional dependency")


@pytest.mark.parametrize("module", ["benchmarks", "clifford_qc", "paper_architecture"])
def test_repository_import_failure_is_a_docs_defect(monkeypatch, module):
    monkeypatch.setattr(docs.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, "", f"ModuleNotFoundError: No module named '{module}'"))
    flags, reason = docs.script_flags(Path("run_broken.py"))
    assert flags is None
    assert not reason.startswith("optional dependency")


def config_fixture(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    data = tmp_path / "data"
    configs.mkdir()
    data.mkdir()
    monkeypatch.setattr(summaries, "ROOT", tmp_path)
    monkeypatch.setattr(summaries, "CONFIGS", configs)
    monkeypatch.setattr(summaries, "DATA", data)
    return configs, data


def sweep():
    return {"seeds": 1, "models": [{"model": {"type": "tfim", "n": 2}}],
            "methods": {"exact": {"kind": "exact"}}}


def test_only_declared_jsonl_schemas_require_summaries(tmp_path, monkeypatch):
    configs, _ = config_fixture(tmp_path, monkeypatch)
    (configs / "sweep.json").write_text(json.dumps(sweep()))
    (configs / "acase_ladder.json").write_text(json.dumps({"methods": {"exact": {}}, "ladder": [{}]}))
    (configs / "future.json").write_text(json.dumps({"schema": "clifford_qc.future_config.v1"}))
    assert summaries.required_stems() == {"sweep", "acase_ladder"}


@pytest.mark.parametrize("change", [{"seeds": False}, {"models": []}, {"methods": {}},
                                    {"schema": ""}, {"seeds": 0}, {"schema": "typo"},
                                    {"schema": "clifford_qc.benchmark_sweep_config.v99"}])
def test_malformed_sweep_cannot_remove_summary_requirement(tmp_path, monkeypatch, change):
    configs, _ = config_fixture(tmp_path, monkeypatch)
    (configs / "sweep.json").write_text(json.dumps(sweep() | change))
    assert summaries.main() == 1


@pytest.mark.parametrize("missing", ["jsonl", "csv", "md"])
def test_deleted_record_or_summary_fails(tmp_path, monkeypatch, missing):
    configs, data = config_fixture(tmp_path, monkeypatch)
    (configs / "sweep.json").write_text(json.dumps(sweep()))
    files = {"jsonl": data / "sweep.jsonl", "csv": data / "sweep_summary.csv",
             "md": data / "sweep_summary.md"}
    for kind, path in files.items():
        if kind != missing:
            path.write_text(kind)
    monkeypatch.setattr(summaries, "regenerate", lambda record: ("md", "csv", None))
    assert summaries.main() == 1


def test_unrequired_orphan_companion_is_not_ignored(tmp_path, monkeypatch):
    _, data = config_fixture(tmp_path, monkeypatch)
    (data / "orphan_summary.csv").write_text("stray")
    assert summaries.main() == 1


def test_stale_summary_is_rejected(tmp_path, monkeypatch):
    configs, data = config_fixture(tmp_path, monkeypatch)
    (configs / "sweep.json").write_text(json.dumps(sweep()))
    for name in ("sweep.jsonl", "sweep_summary.csv", "sweep_summary.md"):
        (data / name).write_text("stale")
    monkeypatch.setattr(summaries, "regenerate", lambda record: ("new md", "new csv", None))
    assert summaries.main() == 1
