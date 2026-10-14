"""Explicit content verification for archives; never certifies commit chronology."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

MANIFEST = "benchmarks/provenance/archive_manifest.json"


def history_kind(root: Path, *, archive: bool = False) -> str:
    """Require a full Git worktree by default; forbid archive-mode downgrades."""
    root = root.resolve()
    try:
        done = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                              capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        done = None
    if done is not None and done.returncode == 0:
        if Path(done.stdout.strip()).resolve() != root:
            raise ValueError("verification root is inside a different Git worktree")
        shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=root,
                                 capture_output=True, text=True, check=True, timeout=10)
        if shallow.stdout.strip() != "false":
            raise ValueError("record chronology requires full git history")
        if archive:
            raise ValueError("--archive cannot bypass a Git worktree's ancestry checks")
        return "git"
    if any((parent / ".git").exists() or (parent / ".git").is_symlink()
           for parent in (root, *root.parents)):
        raise ValueError("Git metadata is present but its worktree is unavailable/incomplete")
    if not archive:
        raise ValueError("full Git history required; use --archive for unverified archive chronology")
    return "archive"


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("binding path must be a nonempty relative string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("binding path escapes the verification root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("binding symlink escapes the verification root")
    if not resolved.is_file():
        raise ValueError(f"missing packaged binding: {relative}")
    return resolved


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw,
                                          usedforsecurity=False).hexdigest()}


def verify_archive(root: Path, record_path: str, config_path: str,
                   execution: str, sources: dict[str, dict]) -> None:
    """Verify manifest, current inputs, and packaged historical execution bytes.

    A checksum manifest verifies content consistency, not who created the archive.
    The source identities must agree with the original record's own bindings.
    """
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate archive manifest key: {key}")
            result[key] = value
        return result
    manifest = json.loads(safe_path(root, MANIFEST).read_text(), object_pairs_hook=pairs)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "records"}
            or manifest["schema"] != "clifford_qc.archive_bindings.v1"):
        raise ValueError("archive manifest identity differs")
    entry = manifest["records"][record_path]
    if set(entry) != {"record_sha256", "config", "execution_commit", "sources"}:
        raise ValueError("archive record binding keys differ")
    if entry["record_sha256"] != binding(safe_path(root, record_path))["sha256"]:
        raise ValueError("archive record digest differs")
    if entry["execution_commit"] != execution:
        raise ValueError("archive execution identity differs")
    config = entry["config"]
    if config != {"path": config_path, **binding(safe_path(root, config_path))}:
        raise ValueError("archive configuration digest differs")
    packaged = entry["sources"]
    if not isinstance(packaged, dict) or set(packaged) != set(sources):
        raise ValueError("archive execution source set differs")
    for path, expected in sources.items():
        item = packaged[path]
        if set(item) != {"snapshot", "sha256", "git_blob_sha1"}:
            raise ValueError(f"archive source binding keys differ: {path}")
        actual = binding(safe_path(root, item["snapshot"]))
        if any(actual[key] != item[key] for key in actual):
            raise ValueError(f"archive execution source digest differs: {path}")
        if any(item[key] != value for key, value in expected.items()):
            raise ValueError(f"archive source differs from recorded execution: {path}")


def archive_notice() -> None:
    print("ARCHIVE: content bindings verified; commit chronology UNVERIFIED; "
          "preregistration ancestry is not verified")
