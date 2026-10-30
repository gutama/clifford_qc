"""Strict UTF-8 JSON and byte digests for maintenance metadata."""
from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import tempfile


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def decode_json(payload: str | bytes, *, allow_nonfinite: bool = False):
    """Reject duplicate keys and nonfinite numbers unless explicitly permitted.

    Only unchanged historical scientific diagnostics may opt into nonfinite
    values; declarations, sidecars and archive manifests remain strict.
    """
    data = json.loads(payload, object_pairs_hook=unique_object)

    def finite(value):
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON value")

    if not allow_nonfinite:
        finite(data)
    return data


def read_json(path: Path, *, allow_nonfinite: bool = False):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return decode_json(raw.decode("utf-8"), allow_nonfinite=allow_nonfinite)


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes, *, exclusive: bool = True) -> None:
    """Atomically publish exact bytes; historical serializers retain their format."""
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


def sha256(path: Path) -> str:
    return byte_digest(path.read_bytes())


def byte_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def binding(path: Path) -> dict:
    return byte_binding(path.read_bytes())


def byte_binding(raw: bytes) -> dict:
    return {"sha256": byte_digest(raw),
            "git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw,
                                         usedforsecurity=False).hexdigest()}
