"""Launch and watch the exact-H4 and certification-calibration reproductions.

The watcher writes an atomic ``status.json`` heartbeat until both jobs finish.
It deliberately keeps independent logs and outputs so a failure in one job
does not destroy the other job's result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _tail(path: Path, lines: int = 12) -> list[str]:
    if not path.exists():
        return []
    with path.open(errors="replace") as fh:
        return fh.read().splitlines()[-lines:]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _versions() -> dict[str, str]:
    names = ["clifford-qc", "numpy", "scipy", "openfermion", "pyscf", "openfermionpyscf"]
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "missing"
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--calibration-seeds", type=int, default=200)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument(
        "--jobs", nargs="+", choices=("h4_exact", "calibration"),
        default=("h4_exact", "calibration"),
    )
    parser.add_argument("--status-file", default="status.json")
    parser.add_argument("--marker-prefix", default="")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(exist_ok=True)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    _atomic_json(
        run_dir / "manifest.json",
        {
            "created_unix": time.time(),
            "git_commit": commit,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _versions(),
            "calibration_seeds": args.calibration_seeds,
        },
    )

    common_env = os.environ.copy()
    common_env.update(
        PYTHONUNBUFFERED="1",
        MPLCONFIGDIR=str(runtime_dir / "matplotlib"),
        XDG_CACHE_HOME=str(runtime_dir / "cache"),
    )
    specs = {
        "h4_exact": {
            "command": [
                sys.executable,
                str(repo / "benchmarks" / "reproduce_exact_h4.py"),
                "--out",
                str(run_dir / "h4_exact.jsonl"),
            ],
            "output": run_dir / "h4_exact.jsonl",
            "threads": "4",
        },
        "calibration": {
            "command": [
                sys.executable,
                str(repo / "benchmarks" / "run_calibration.py"),
                "--seeds",
                str(args.calibration_seeds),
                "--out",
                str(run_dir / "calibration.jsonl"),
            ],
            "output": run_dir / "calibration.jsonl",
            "threads": "1",
        },
    }
    specs = {name: specs[name] for name in args.jobs}

    jobs = {}
    started = time.time()
    for name, spec in specs.items():
        log_path = run_dir / f"{name}.log"
        log_fh = log_path.open("w")
        env = common_env.copy()
        env.update(
            OMP_NUM_THREADS=spec["threads"],
            OPENBLAS_NUM_THREADS=spec["threads"],
            MKL_NUM_THREADS=spec["threads"],
        )
        proc = subprocess.Popen(
            spec["command"],
            cwd=repo,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        jobs[name] = {
            **spec,
            "process": proc,
            "log_fh": log_fh,
            "log": log_path,
            "started_unix": time.time(),
        }

    while True:
        snapshot = {
            "watcher_pid": os.getpid(),
            "git_commit": commit,
            "heartbeat_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "jobs": {},
        }
        all_done = True
        any_failed = False
        for name, job in jobs.items():
            proc = job["process"]
            code = proc.poll()
            if code is None:
                all_done = False
                state = "running"
            elif code == 0:
                state = "completed"
            else:
                state = "failed"
                any_failed = True
            snapshot["jobs"][name] = {
                "state": state,
                "pid": proc.pid,
                "returncode": code,
                "elapsed_seconds": time.time() - job["started_unix"],
                "command": job["command"],
                "log": str(job["log"]),
                "log_tail": _tail(job["log"]),
                "output": str(job["output"]),
                "output_rows": _row_count(job["output"]),
                "output_sha256": _sha256(job["output"]),
            }
        _atomic_json(run_dir / args.status_file, snapshot)
        if all_done:
            for job in jobs.values():
                job["log_fh"].close()
            marker = args.marker_prefix + ("FAILED" if any_failed else "DONE")
            (run_dir / marker).write_text(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
            )
            return 1 if any_failed else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
