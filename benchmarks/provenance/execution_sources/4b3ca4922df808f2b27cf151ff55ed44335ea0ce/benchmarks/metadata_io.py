"""Strict UTF-8 JSON and byte digests for maintenance metadata."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path: Path, *, allow_nonfinite: bool = False):
    """Reject duplicate keys and nonfinite numbers unless explicitly permitted.

    Only unchanged historical scientific diagnostics may opt into nonfinite
    values; declarations, sidecars and archive manifests remain strict.
    """
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)

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
