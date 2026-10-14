"""Real Git/worktree boundaries and archive checksum tamper regressions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks.archive_provenance import MANIFEST, binding, history_kind, verify_archive


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True,
                                   stderr=subprocess.PIPE).strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Archive fixture")
    git(root, "config", "user.email", "archive@example.invalid")
    (root / "source").write_text("one")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root


def test_full_clone_and_linked_worktree_require_real_history(repository, tmp_path):
    assert history_kind(repository) == "git"
    linked = tmp_path / "linked"
    git(repository, "worktree", "add", "--detach", str(linked), "HEAD")
    assert (linked / ".git").is_file()
    assert history_kind(linked) == "git"
    with pytest.raises(ValueError, match="cannot bypass"):
        history_kind(linked, archive=True)


def test_shallow_clone_cannot_be_downgraded_to_archive(repository, tmp_path):
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "--depth=1", repository.as_uri(), str(shallow)],
                   capture_output=True, check=True)
    for archive in (False, True):
        with pytest.raises(ValueError, match="full git history"):
            history_kind(shallow, archive=archive)


def test_archive_requires_explicit_mode_and_broken_git_never_skips(tmp_path):
    with pytest.raises(ValueError, match="full Git history required"):
        history_kind(tmp_path)
    assert history_kind(tmp_path, archive=True) == "archive"
    (tmp_path / ".git").write_text("gitdir: missing")
    with pytest.raises(ValueError, match="unavailable/incomplete"):
        history_kind(tmp_path, archive=True)


@pytest.fixture
def archive_bundle(tmp_path):
    (tmp_path / "record.json").write_text('{"execution":"historical"}')
    (tmp_path / "config.json").write_text('{"frozen":true}')
    (tmp_path / "historical.py").write_text("value = 1\n")
    expected = binding(tmp_path / "historical.py")
    manifest = {"schema": "clifford_qc.archive_bindings.v1", "records": {
        "record.json": {"record_sha256": binding(tmp_path / "record.json")["sha256"],
                        "config": {"path": "config.json", **binding(tmp_path / "config.json")},
                        "execution_commit": "a" * 40,
                        "sources": {"source.py": {"snapshot": "historical.py", **expected}}}}}
    target = tmp_path / MANIFEST
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(manifest))
    return tmp_path, expected, manifest


def verify(bundle):
    root, expected, _ = bundle
    verify_archive(root, "record.json", "config.json", "a" * 40, {"source.py": expected})


def test_packaged_historical_sources_verify_without_current_source(archive_bundle):
    verify(archive_bundle)


@pytest.mark.parametrize("path", ["record.json", "config.json", "historical.py"])
def test_archive_tampered_record_input_or_execution_source_fails(archive_bundle, path):
    root, *_ = archive_bundle
    (root / path).write_text("tampered")
    with pytest.raises(ValueError, match="digest differs"):
        verify(archive_bundle)


@pytest.mark.parametrize("change", ["identity", "sha1", "sha256", "escape", "missing"])
def test_manifest_cannot_replace_recorded_source_bindings(archive_bundle, change):
    root, _, manifest = archive_bundle
    entry = manifest["records"]["record.json"]
    if change == "identity":
        entry["execution_commit"] = "b" * 40
    elif change in {"sha1", "sha256"}:
        entry["sources"]["source.py"]["git_blob_sha1" if change == "sha1" else "sha256"] = "0" * 64
    elif change == "escape":
        entry["sources"]["source.py"]["snapshot"] = "../outside.py"
    else:
        entry["sources"]["source.py"]["snapshot"] = "missing.py"
    (root / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(archive_bundle)


def test_both_gates_explicitly_verify_exported_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    exported = tmp_path / "exported"
    shutil.copytree(root, exported, ignore=shutil.ignore_patterns(
        ".git", ".venv", "__pycache__", ".pytest_cache", ".hypothesis"))
    for name in ("check_phase19_preregistration", "check_bank_policy_matrix"):
        script = exported / "benchmarks" / f"{name}.py"
        done = subprocess.run([sys.executable, str(script), "--archive"], cwd=exported,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "commit chronology UNVERIFIED" in done.stdout
        assert "commit provenance verified" not in done.stdout
        strict = subprocess.run([sys.executable, str(script), "--require-git"], cwd=exported,
                                capture_output=True, text=True)
        assert strict.returncode != 0
