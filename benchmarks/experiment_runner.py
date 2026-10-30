"""Bounded subprocess runner with failure-preserving, resumable receipts.

The transport executes argv without a shell. Campaign-specific declarations,
shot accounting and scientific validation remain the consumer's responsibility.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from benchmarks.experiment_artifacts import (
    ArtifactStore, json_bytes, local_path, read_json, write_json,
)

SCHEMA = "clifford_qc.execution_receipt.v1"
TERMINAL = {"passed", "failed", "timed_out", "output_limited", "interrupted"}


@dataclass(frozen=True)
class Task:
    key: str
    argv: tuple[str, ...]
    timeout_seconds: float = 600
    max_output_bytes: int = 2**20
    artifacts: tuple[str, ...] = ()

    def __post_init__(self):
        if (not self.key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.key)
                or not isinstance(self.argv, tuple) or not self.argv
                or any(not isinstance(x, str) or not x or "\0" in x for x in self.argv)):
            raise ValueError("invalid task key or argv")
        if (type(self.timeout_seconds) not in (float, int)
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
                or type(self.max_output_bytes) is not int
                or not 0 < self.max_output_bytes <= 16 * 2**20):
            raise ValueError("invalid task resource bounds")
        if (not isinstance(self.artifacts, tuple) or len(self.artifacts) > 64
                or len(set(self.artifacts)) != len(self.artifacts)):
            raise ValueError("invalid artifact schedule")


def source_bindings(root: Path, paths: tuple[str, ...]) -> dict:
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate source path")
    return {p: hashlib.sha256(local_path(root, p).read_bytes()).hexdigest() for p in sorted(paths)}


def context(root: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    libraries = {}
    for name in ("numpy", "scipy", "stim"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {"git_commit": commit, "python": platform.python_version(), "libraries": libraries,
            "transport_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name("experiment_artifacts.py"),
                                           Path(__file__).with_name("metadata_io.py"))},
            "threads": {p: os.environ.get(p) for p in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def _terminate(proc):
    # Kill the whole subprocess group: a timed-out checker must not leave a
    # sampling child spending resources after its receipt records a failure.
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait()
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextmanager
def _run_lock(path):
    # Advisory locks are released by the kernel on crashes. Keep the inode:
    # unlinking a lock while another process holds it permits two writers.
    import fcntl
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _execute(task: Task, root: Path, log_path: Path) -> dict:
    started = time.monotonic()
    status, reason, returncode = "passed", None, None
    with log_path.open("xb") as log:
        try:
            proc = subprocess.Popen(task.argv, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=(os.name == "posix"))
        except OSError as error:
            return {"status": "failed", "reason": str(error), "returncode": None,
                    "elapsed_seconds": time.monotonic() - started}
        try:
            while True:
                returncode = proc.poll()
                if log_path.stat().st_size > task.max_output_bytes:
                    status, reason = "output_limited", "log byte budget exceeded"
                    break
                if returncode is not None:
                    status = "passed" if returncode == 0 else "failed"
                    break
                if time.monotonic() - started >= task.timeout_seconds:
                    status, reason = "timed_out", "wall-time budget exceeded"
                    break
                time.sleep(min(.05, task.timeout_seconds))
        finally:
            _terminate(proc)
            returncode = proc.wait()
    # Polling limits are monitored, not OS-enforced. Keep only the declared
    # prefix; a transient larger file is disclosed as an output-limit failure.
    observed_bytes = log_path.stat().st_size
    with log_path.open("r+b") as stream:
        stream.truncate(min(observed_bytes, task.max_output_bytes))
    return {"status": status, "reason": reason, "returncode": returncode,
            "observed_log_bytes": observed_bytes,
            "elapsed_seconds": time.monotonic() - started}


def run(tasks: tuple[Task, ...], *, root: Path, destination: Path,
        sources: tuple[str, ...], resume: bool = False) -> dict:
    """Run the frozen schedule once; resume never retries an interrupted cell.

    A destination is single-writer locked. A persistent started row becomes an
    interrupted failure after a crash; all not-yet-started rows still execute.
    Existing final receipts are verified and returned only with explicit resume.
    """
    if not tasks or len({t.key for t in tasks}) != len(tasks):
        raise ValueError("empty or duplicate task schedule")
    root, destination = Path(root).resolve(), Path(destination).resolve()
    bindings = source_bindings(root, sources)
    for task in tasks:
        for relative in task.artifacts:
            local_path(root, relative)
    plan = {"tasks": [asdict(t) for t in tasks], "source_sha256": bindings,
            "context": context(root)}
    # JSON roundtrip normalizes tuples before equality checks after a restart.
    import json
    plan = json.loads(json_bytes(plan))
    identity = hashlib.sha256(json_bytes(plan)).hexdigest()
    if resume:
        if not destination.is_dir():
            raise ValueError("resume requires an existing run")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    with _run_lock(destination / "active.lock"):
        checkpoint = destination / "checkpoint.json"
        if resume:
            receipt = read_json(checkpoint)
            if (receipt.get("schema") != SCHEMA or receipt.get("plan_sha256") != identity
                    or receipt.get("plan") != plan):
                raise ValueError("resume plan, sources or environment differ")
            _validate_rows(receipt, tasks)
        else:
            receipt = {"schema": SCHEMA, "kind": "execution_transport_only",
                       "plan_sha256": identity, "plan": plan, "rows": [],
                       "scientific_evidence_claim": False}
            write_json(checkpoint, receipt)
        store = ArtifactStore(destination)
        for index, task in enumerate(tasks):
            if index < len(receipt["rows"]):
                row = receipt["rows"][index]
                if row["status"] in TERMINAL:
                    if row.get("log"):
                        store.get(row["log"])
                    for ref in row.get("artifacts", {}).values():
                        store.get(ref)
                    continue
                row.update(status="interrupted", reason="started task has no terminal receipt",
                           returncode=None, elapsed_seconds=None, costs_complete=False)
                partial = destination / f"{task.key}.log"
                if partial.exists():
                    row["observed_log_bytes"] = partial.stat().st_size
                    with partial.open("rb") as stream:
                        row["log"] = store.put(stream.read(task.max_output_bytes))
                    partial.unlink()
                write_json(checkpoint, receipt, exclusive=False)
                continue
            if source_bindings(root, sources) != bindings or context(root) != plan["context"]:
                raise ValueError("bound source changed during execution")
            row = {"key": task.key, "status": "started", "costs_complete": False}
            receipt["rows"].append(row)
            write_json(checkpoint, receipt, exclusive=False)
            log_path = destination / f"{task.key}.log"
            row.update(_execute(task, root, log_path))
            row["log"] = store.put(log_path.read_bytes())
            row["artifacts"] = {}
            for relative in task.artifacts:
                try:
                    artifact = local_path(root, relative)
                    if artifact.stat().st_size > store.max_object_bytes:
                        raise ValueError("artifact exceeds object byte budget")
                    with artifact.open("rb") as stream:
                        row["artifacts"][relative] = store.put(stream.read(store.max_object_bytes + 1))
                except (OSError, ValueError) as error:
                    row.setdefault("artifact_errors", {})[relative] = str(error)
                    if row["status"] == "passed":
                        row.update(status="failed", reason="required artifact unavailable")
            row["costs_complete"] = row["status"] != "interrupted"
            # Consumer artifacts remain separate; log objects and this receipt
            # never overwrite campaign records or change scientific labels.
            log_path.unlink()
            print(f'{task.key}: {row["status"]}', flush=True)
            if row["status"] != "passed":
                print(store.get(row["log"]).decode("utf-8", errors="replace"), flush=True)
            write_json(checkpoint, receipt, exclusive=False)
        receipt["summary"] = {"scheduled": len(tasks), "attempted": len(receipt["rows"]),
                              "passed": sum(r["status"] == "passed" for r in receipt["rows"]),
                              "failed": sum(r["status"] != "passed" for r in receipt["rows"])}
        write_json(checkpoint, receipt, exclusive=False)
        final = destination / "receipt.json"
        if final.exists():
            if read_json(final) != receipt:
                raise ValueError("immutable final receipt differs")
        else:
            write_json(final, receipt)
        return receipt


def _validate_rows(receipt, tasks):
    rows = receipt.get("rows")
    if (not isinstance(rows, list) or len(rows) > len(tasks)
            or receipt.get("scientific_evidence_claim") is not False):
        raise ValueError("invalid execution receipt")
    for index, row in enumerate(rows):
        if row.get("key") != tasks[index].key or row.get("status") not in TERMINAL | {"started"}:
            raise ValueError("receipt schedule differs")
        if row["status"] == "started" and index != len(rows)-1:
            raise ValueError("nonterminal row is not the last row")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    spec = read_json(args.spec)
    if (set(spec) != {"schema", "sources", "tasks"}
            or spec.get("schema") != "clifford_qc.execution_plan.v1"):
        raise ValueError("unknown execution plan")
    tasks = tuple(Task(**{**row, "argv": tuple(row["argv"]),
                          "artifacts": tuple(row.get("artifacts", ()))}) for row in spec["tasks"])
    receipt = run(tasks, root=Path.cwd(), destination=args.output,
                  sources=tuple(spec["sources"]), resume=args.resume)
    return int(receipt["summary"]["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
