"""Small helpers for reproducibility gates on versioned JSON records."""

from __future__ import annotations

import math
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from typing import Any


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True,
            timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def execution_provenance() -> dict[str, Any]:
    """JSON-safe execution identity for benchmark and manuscript records.

    The block records the source revision, package/runtime versions, platform,
    numerical-library configuration, and a digest of the complete installed
    distribution set.  Volatile fields are intentionally metadata: record
    comparisons ignore ``provenance`` while still comparing every scientific
    and resource field.
    """
    import numpy as np

    from . import __version__

    sha = (os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")
           or _git_output("rev-parse", "HEAD") or "unknown")
    dirty_text = _git_output("status", "--porcelain")
    distributions = sorted(
        f"{dist.metadata.get('Name', 'unknown')}=={dist.version}"
        for dist in importlib.metadata.distributions())
    freeze_digest = hashlib.sha256(
        "\n".join(distributions).encode("utf-8")).hexdigest()

    try:
        blas = np.__config__.show(mode="dicts")
    except TypeError:  # NumPy 1.x has only the printing form of show().
        stream = StringIO()
        with redirect_stdout(stream):
            np.__config__.show()
        blas = {"summary": stream.getvalue().strip()}

    return {
        "schema": "clifford_qc.execution_provenance.v1",
        "git_sha": sha,
        "git_dirty": None if dirty_text is None else bool(dirty_text),
        "clifford_qc": __version__,
        "python": platform.python_version(),
        "dependencies": {
            name: _version(name)
            for name in ("numpy", "scipy", "pyscf", "openfermion",
                         "openfermionpyscf")
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
        },
        "blas_lapack": blas,
        "environment_sha256": freeze_digest,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def stamp_record(record: dict[str, Any],
                 provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a shallow copy of one record with execution provenance attached."""
    if not isinstance(record, dict):
        raise TypeError("record must be a mapping object")
    stamped = dict(record)
    stamped["provenance"] = (execution_provenance() if provenance is None
                             else provenance)
    # Fail at the producer boundary, not after a long benchmark has written an
    # unreadable artifact.
    json.dumps(stamped["provenance"])
    return stamped


def compare_json_records(
        expected: Any,
        actual: Any,
        path: str = "$",
        *,
        atol: float = 1e-11,
        rtol: float = 1e-11,
        ignored_keys: frozenset[str] = frozenset({"provenance"}),
) -> list[str]:
    """Compare JSON-like values with exact discrete and tolerant float fields.

    Keys, list lengths, strings, booleans, integers, and nulls are exact.
    Floating fields alone use :func:`math.isclose`, which isolates harmless
    BLAS/LAPACK last-bit changes without weakening resource or replica counts.
    """
    problems: list[str] = []

    def compare(left: Any, right: Any, here: str) -> None:
        if isinstance(left, dict):
            if not isinstance(right, dict):
                problems.append(
                    f"{here}: expected object, got {type(right).__name__}")
                return
            left_keys = set(left) - ignored_keys
            right_keys = set(right) - ignored_keys
            missing = sorted(left_keys - right_keys)
            extra = sorted(right_keys - left_keys)
            if missing:
                problems.append(f"{here}: missing keys {missing}")
            if extra:
                problems.append(f"{here}: unexpected keys {extra}")
            for key in sorted(left_keys & right_keys):
                compare(left[key], right[key], f"{here}.{key}")
            return

        if isinstance(left, list):
            if not isinstance(right, list):
                problems.append(
                    f"{here}: expected array, got {type(right).__name__}")
                return
            if len(left) != len(right):
                problems.append(
                    f"{here}: expected {len(left)} entries, got {len(right)}")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{here}[{index}]")
            return

        # bool is a subclass of int, so exact discrete types come first.
        if left is None or isinstance(left, (bool, str, int)):
            if type(right) is not type(left) or right != left:
                problems.append(f"{here}: expected {left!r}, got {right!r}")
            return

        if isinstance(left, float):
            if not isinstance(right, (int, float)) or isinstance(right, bool):
                problems.append(
                    f"{here}: expected floating value, got {right!r}")
                return
            if not math.isclose(
                    left, float(right), rel_tol=rtol, abs_tol=atol):
                problems.append(
                    f"{here}: expected {left:.17g}, got {float(right):.17g}")
            return

        problems.append(
            f"{here}: unsupported value type {type(left).__name__}")

    compare(expected, actual, path)
    return problems


__all__ = ["compare_json_records", "execution_provenance", "stamp_record"]
