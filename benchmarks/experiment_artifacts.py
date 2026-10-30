"""Versioned execution artifacts; historical scientific formats stay untouched.

Objects identify exact bytes. Receipts are maintenance records, not scientific
evidence declarations or replacements for independent numerical checkers.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from benchmarks.metadata_io import unique_object


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def decode_json(payload: bytes):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    value = json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject)
    # Also reject overflow such as 1e999, including in nested containers.
    json_bytes(value)
    return value


def read_json(path: Path):
    payload = Path(path).read_bytes()
    return decode_json(gzip.decompress(payload) if str(path).endswith(".gz") else payload)


def atomic_write(path: Path, payload: bytes, *, exclusive: bool = True) -> None:
    """Same-directory atomic publication; exclusive writes never replace bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value, *, exclusive: bool = True) -> None:
    payload = json_bytes(value)
    if str(path).endswith(".gz"):
        payload = gzip.compress(payload, mtime=0)
    atomic_write(path, payload, exclusive=exclusive)


def local_path(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlink escapes from the run root."""
    candidate = Path(relative)
    if (not relative or candidate.is_absolute() or ".." in candidate.parts
            or "\\" in relative):
        raise ValueError(f"invalid relative artifact path: {relative!r}")
    root = Path(root).resolve()
    path = (root / candidate).resolve()
    if path == root or root not in path.parents:
        raise ValueError("artifact path escapes root")
    return path


class ArtifactStore:
    """Immutable SHA-256 objects, with bounded reads and verified deduplication."""

    def __init__(self, root: Path, *, max_object_bytes: int = 16 * 2**20):
        if type(max_object_bytes) is not int or max_object_bytes <= 0:
            raise ValueError("max_object_bytes must be a positive integer")
        self.root = Path(root)
        self.max_object_bytes = max_object_bytes

    def _path(self, digest: str) -> Path:
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid object digest")
        return local_path(self.root, f"objects/{digest[:2]}/{digest}")

    def put(self, payload: bytes) -> dict:
        if len(payload) > self.max_object_bytes:
            raise ValueError("artifact exceeds object byte budget")
        digest = hashlib.sha256(payload).hexdigest()
        ref = {"sha256": digest, "size_bytes": len(payload)}
        try:
            atomic_write(self._path(digest), payload)
        except FileExistsError:
            if self.get(ref) != payload:
                raise ValueError("object collision")
        return ref

    def get(self, ref: dict) -> bytes:
        if set(ref) != {"sha256", "size_bytes"}:
            raise ValueError("invalid artifact reference")
        size = ref["size_bytes"]
        if type(size) is not int or not 0 <= size <= self.max_object_bytes:
            raise ValueError("artifact exceeds object byte budget")
        path = self._path(ref["sha256"])
        if path.stat().st_size != size:
            raise ValueError("artifact size differs")
        with path.open("rb") as stream:
            payload = stream.read(size + 1)
        if len(payload) != size:
            raise ValueError("artifact size differs")
        if hashlib.sha256(payload).hexdigest() != ref["sha256"]:
            raise ValueError("artifact digest differs")
        return payload

    def put_json(self, value) -> dict:
        return self.put(json_bytes(value))
