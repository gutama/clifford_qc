"""Outcome-free contracts for the process-isolated Phase 2M policy matrix."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "benchmarks/configs/bank_policy_matrix.json"
RECORD = ROOT / "benchmarks/reference_results/bank_policy_matrix.json"
SCHEMA = "clifford_qc.bank_policy_matrix.v1"
CONFIG_SCHEMA = "clifford_qc.bank_policy_matrix_config.v1"
EXPECTED_CONFIG_SHA256 = "132a9efcb17480c760fa8556a03acd57e4d1012dbd399ef21e2af2b45ac72e80"
ARMS = (
    ("object_retain", "object", "retain_all"),
    ("packed_retain", "packed", "retain_all"),
    ("object_stream", "object", "stream_recompute"),
    ("packed_stream", "packed", "stream_recompute"),
)
SOURCES = (
    "benchmarks/bank_policy_contract.py", "benchmarks/bank_policy_worker.py",
    "benchmarks/run_bank_policy_matrix.py", "benchmarks/check_bank_policy_matrix.py",
    "clifford_qc/subspace/projection.py", "clifford_qc/subspace/packed.py",
    "clifford_qc/subspace/streaming.py", "clifford_qc/subspace/adaptive.py",
    "clifford_qc/subspace/fermionic_generators.py", "clifford_qc/models/fcidump.py",
    "clifford_qc/subspace/linalg.py", "clifford_qc/multivector.py",
    "clifford_qc/pauli_kernel.py", "clifford_qc/backends/exact_mv.py",
)
CLAIM_BOUNDARY = (
    "Exact-arithmetic storage-policy equivalence and measured classical process costs "
    "on five fixed FCIDUMPs, full candidate pools and frozen stopping rules. "
    "Fresh workers isolate allocator history. RSS includes preparation, reference, "
    "support history, scalar caches, instrumentation and coefficient rows; it is "
    "not an attributed coefficient-storage byte count. Packed live/reserved bytes "
    "and 24-byte coefficient payload estimates are separate. No sampling, "
    "hardware-time price, universal memory bound or quantum advantage is established. "
    "Historical records and the object default remain unchanged."
)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value, *, exclusive=False):
    """Serialize first; publish a complete file without a check/write race."""
    path = Path(path)
    data = canonical(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".policy-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def git(*args, root=ROOT):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def load_config(path=CONFIG):
    path = Path(path)
    if file_sha(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen policy configuration differs")
    config = read_json(path)
    if set(config) != {"schema", "phase", "systems", "arms", "execution", "options",
                       "environment_reference", "claim_boundary"}:
        raise ValueError("configuration key set differs")
    if config["schema"] != CONFIG_SCHEMA or config["phase"] != "2M-C/D":
        raise ValueError("configuration identity differs")
    if config["arms"] != [list(arm) for arm in ARMS]:
        raise ValueError("policy arms/order differ")
    if config["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("claim boundary differs")
    for system in config["systems"]:
        for key in ("source", "reference"):
            item = system[key]
            path = Path(item["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("input escapes repository")
            if file_sha(ROOT / path) != item["sha256"]:
                raise ValueError(f"input binding drift: {path}")
        reference = read_json(ROOT / system["reference"]["path"])
        for key in system["reference"]["energy_key"]:
            reference = reference[key]
        if type(reference) is not float or reference != system["exact_ground_energy"]:
            raise ValueError("reference stopping input differs")
    return config


def matrix_record(matrix):
    import numpy as np
    data = np.asarray(matrix, dtype="<c16", order="C")
    if not np.isfinite(data).all():
        raise ValueError("non-finite projected matrix")
    return {"shape": list(data.shape), "dtype": "<c16",
            "data": base64.b64encode(data.tobytes()).decode()}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)
