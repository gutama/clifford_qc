"""Real Git/worktree boundaries and archive checksum tamper regressions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks.archive_provenance import MANIFEST, binding, history_kind, verify_archive
from benchmarks import archive_provenance


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


@pytest.mark.parametrize("call", [1, 2])
@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("git", 10),
                                   subprocess.CalledProcessError(128, "git")])
def test_history_subprocess_failures_are_reportable(repository, monkeypatch, call, error):
    original = subprocess.run
    calls = 0

    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == call:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(archive_provenance.subprocess, "run", failing)
    with pytest.raises(archive_provenance.HistoryError, match="Git history inspection failed"):
        history_kind(repository)


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


@pytest.fixture
def exported_checkout(tmp_path):
    from benchmarks import bank_policy_contract, phase19_structure
    root = Path(__file__).resolve().parents[1]
    exported = tmp_path / "exported"
    paths = set()
    for directory in ("clifford_qc", "molecular", "benchmarks"):
        paths.update(path.relative_to(root) for path in (root / directory).rglob("*.py")
                     if "provenance" not in path.parts)
    paths.update(path.relative_to(root) for path in (root / "benchmarks/configs").rglob("*.json"))
    paths.update(Path(path) for path in phase19_structure.INPUTS)
    config = bank_policy_contract.load_config()
    paths.add(Path(config["environment_reference"]))
    for system in config["systems"]:
        paths.update(Path(system[key]["path"]) for key in ("source", "reference"))
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    paths.add(Path(MANIFEST))
    for record_path, entry in manifest["records"].items():
        paths.update((Path(record_path), Path(entry["config"]["path"])))
        paths.update(Path(item["snapshot"]) for item in entry["sources"].values())
        paths.update(Path(item["snapshot"]) for item in entry.get("lineage_sources", {}).values())
    for relative in paths:
        target = exported / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return exported


def test_both_gates_explicitly_verify_exported_checkout(exported_checkout):
    exported = exported_checkout
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


def test_archive_manifest_cannot_hide_wrong_declaration_binding(exported_checkout):
    from benchmarks import phase19_structure
    record_path = exported_checkout / phase19_structure.RECORD_PATH
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["declaration_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    manifest_path = exported_checkout / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][phase19_structure.RECORD_PATH]["record_sha256"] = binding(record_path)["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    done = subprocess.run([sys.executable, str(exported_checkout / "benchmarks/check_phase19_preregistration.py"),
                           "--archive"], cwd=exported_checkout, capture_output=True, text=True)
    assert done.returncode == 1
    assert "record does not bind this declaration" in done.stderr
    assert "content bindings verified" not in done.stdout
