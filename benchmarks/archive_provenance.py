"""Explicit content verification for archives; never certifies commit chronology."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

from benchmarks.metadata_io import binding, byte_binding, read_json

MANIFEST = "benchmarks/provenance/archive_manifest.json"


class HistoryError(ValueError):
    """Git history cannot be verified."""


class ShallowHistoryError(HistoryError):
    """Chronology requires history omitted by a shallow checkout."""


def commit_binding(root: Path, commit: str, relative: str) -> dict:
    """Hash raw Git blob bytes, independent of working-tree newline filters."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("execution identity must be a full commit SHA")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("execution source path must stay inside the repository")
    try:
        done = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                              capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HistoryError(f"execution source unavailable: {relative}: {type(exc).__name__}") from exc
    return byte_binding(done.stdout)


def load_archive_manifest(root: Path, *, cache: dict | None = None) -> dict:
    if cache is not None and "manifest" in cache:
        manifest = cache["manifest"]
    else:
        manifest = read_json(safe_path(root, MANIFEST))
        if cache is not None:
            cache["manifest"] = manifest
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "records"}
            or manifest["schema"] != "clifford_qc.archive_bindings.v1"
            or not isinstance(manifest["records"], dict)):
        raise ValueError("archive manifest identity differs")
    for entry in manifest["records"].values():
        keys = {"record_sha256", "config", "execution_commit", "sources"}
        if (not isinstance(entry, dict) or set(entry) not in (keys, keys | {"lineage_sources"})
                or not isinstance(entry["config"], dict) or not isinstance(entry["sources"], dict)
                or ("lineage_sources" in entry and not isinstance(entry["lineage_sources"], dict))):
            raise ValueError("archive record binding keys/types differ")
    return manifest


def history_kind(root: Path, *, archive: bool = False) -> str:
    """Require a full Git worktree by default; forbid archive-mode downgrades."""
    root = root.resolve()
    try:
        done = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                              capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        done = None
    except subprocess.SubprocessError as exc:
        raise HistoryError(f"Git history inspection failed: {exc}") from exc
    if done is not None and done.returncode == 0:
        if Path(done.stdout.strip()).resolve() != root:
            raise ValueError("verification root is inside a different Git worktree")
        try:
            shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=root,
                                     capture_output=True, text=True, check=True, timeout=10)
        except subprocess.SubprocessError as exc:
            raise HistoryError(f"Git history inspection failed: {exc}") from exc
        if shallow.stdout.strip() != "false":
            raise ShallowHistoryError("record chronology requires full git history")
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


def verify_archive(root: Path, record_path: str, config_path: str,
                   execution: str, sources: dict[str, dict], *,
                   lineage_sources: dict[str, dict] | None = None,
                   manifest: dict | None = None) -> None:
    """Verify manifest, current inputs, and packaged historical execution bytes.

    A checksum manifest verifies content consistency, not who created the archive.
    The source identities must agree with the original record's own bindings.
    """
    manifest = load_archive_manifest(root, cache=None if manifest is None else {"manifest": manifest})
    entry = manifest["records"][record_path]
    expected_keys = {"record_sha256", "config", "execution_commit", "sources"}
    if lineage_sources is not None:
        expected_keys.add("lineage_sources")
    if set(entry) != expected_keys:
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
    if lineage_sources is not None:
        if set(entry["lineage_sources"]) != set(lineage_sources):
            raise ValueError("archive lineage source set differs")
        for path, expected in lineage_sources.items():
            lineage_source_path(root, record_path, {"path": path, **expected}, manifest=manifest)


def lineage_source_path(root: Path, record_path: str, expected: dict, *, manifest: dict | None = None) -> Path:
    """Read a packaged execution source pinned by the frozen declaration."""
    manifest = load_archive_manifest(root, cache=None if manifest is None else {"manifest": manifest})
    item = manifest["records"][record_path]["lineage_sources"][expected["path"]]
    if not isinstance(item, dict) or set(item) != {"snapshot", "sha256", "git_blob_sha1"}:
        raise ValueError("archive lineage source binding keys differ")
    source = safe_path(root, item["snapshot"])
    actual = binding(source)
    if actual != {key: expected[key] for key in actual} or actual != {key: item[key] for key in actual}:
        raise ValueError(f"frozen execution source digest differs: {expected['path']}")
    return source


def archive_notice() -> None:
    print("ARCHIVE: content bindings verified; commit chronology UNVERIFIED; "
          "preregistration ancestry is not verified")
