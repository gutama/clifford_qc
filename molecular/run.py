"""Run reproducible molecular comparisons, one isolated process per molecule."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from clifford_qc.prepared import _atomic_json, _digest, implementation_fingerprint
from molecular.catalog import MOLECULES
from molecular.chemistry import environment, file_digest


def request_identity(key, options):
    return {"schema": "clifford_qc.molecular_request.v1", "key": key,
            "molecule": MOLECULES[key], "options": options, "environment": environment(),
            "package": implementation_fingerprint(),
            "workflow": {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def seal_record(record, request, fcidump):
    record = {**record, "provenance": {"request": request, "fcidump_sha256": file_digest(fcidump)}}
    return {**record, "record_digest": _digest(record)}


def load_records(directory):
    """Recover the summary from per-molecule records; never discard broken JSON."""
    records = {}
    for path in sorted(Path(directory).glob("*_results.json")):
        record = json.loads(path.read_text())
        key = path.name.removesuffix("_results.json")
        if not isinstance(record, dict) or record.get("key") != key or key not in MOLECULES:
            raise ValueError(f"{path}: invalid molecular record key")
        if "provenance" in record or "record_digest" in record:
            payload = {k: v for k, v in record.items() if k != "record_digest"}
            if record.get("record_digest") != _digest(payload):
                raise ValueError(f"{path}: record digest mismatch")
        records[key] = record
    return records


def can_resume(record, request, fcidump):
    provenance = record.get("provenance", {})
    payload = {k: v for k, v in record.items() if k != "record_digest"}
    return (record.get("record_digest") == _digest(payload)
            and provenance.get("request") == request and Path(fcidump).is_file()
            and provenance.get("fcidump_sha256") == file_digest(fcidump))


@contextmanager
def output_lock(directory):
    # RSS accounting already requires Linux. flock also releases on a killed
    # driver, so an interrupted run does not leave a stale exclusive lock.
    import fcntl
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another molecular run is writing {directory}") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--molecules", nargs="+", choices=MOLECULES, default=list(MOLECULES))
    p.add_argument("--output-directory", type=Path, default=ROOT / "molecular" / "runs")
    p.add_argument("--max-candidates", type=int, default=0, help="0 uses the full SD candidate pool")
    p.add_argument("--max-additions", type=int, default=None,
                   help="additions after the identity; overrides catalog defaults (0 adds none)")
    p.add_argument("--target-error-mha", type=float, default=1.5936,
                   help="oracle-assisted stopping tolerance; 0 disables the oracle stop")
    p.add_argument("--storage", choices=("object", "packed"), default="object")
    p.add_argument("--policy", choices=("retain_all", "stream_recompute"), default="retain_all")
    p.add_argument("--frontier-pairs", type=int, default=32)
    p.add_argument("--complete-sd", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--reference-method", choices=("auto", "dense", "eigsh", "lanczos"), default="dense")
    p.add_argument("--resume", action="store_true", help="reuse only matching verified records")
    p.add_argument("--refresh-chemistry", action="store_true")
    p.add_argument("--worker", choices=MOLECULES, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.max_candidates < 0 or (args.max_additions is not None and args.max_additions < 0):
        p.error("candidate and addition limits must be nonnegative")
    if args.frontier_pairs < 1:
        p.error("--frontier-pairs must be positive")
    if not math.isfinite(args.target_error_mha) or args.target_error_mha < 0:
        p.error("--target-error-mha must be finite and nonnegative")
    options = {key: getattr(args, key) for key in
               ("max_candidates", "max_additions", "target_error_mha", "storage", "policy",
                "frontier_pairs", "complete_sd", "reference_method")}
    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        from molecular.simulation import simulate
        record = simulate(args.worker, MOLECULES[args.worker], directory,
                          **options, refresh_chemistry=args.refresh_chemistry)
        _atomic_json(directory / f"{args.worker}_results.json",
                     seal_record(record, request_identity(args.worker, options),
                                 directory / f"{args.worker}.fcidump"))
        return
    with output_lock(directory) as lock:
        records = load_records(directory)
        _atomic_json(directory / "results_summary.json", records)
        for key in dict.fromkeys(args.molecules):
            if (args.resume and not args.refresh_chemistry
                    and can_resume(records.get(key, {}), request_identity(key, options),
                                   directory / f"{key}.fcidump")):
                print(f"{key}: resumed verified result", flush=True)
                continue
            command = [sys.executable, "-u", "-m", "molecular.run", "--worker", key,
                       "--output-directory", str(directory)]
            for name, value in options.items():
                if value is not None:
                    command.extend(["--" + name.replace("_", "-"), str(value)])
            if args.refresh_chemistry:
                command.append("--refresh-chemistry")
            subprocess.run(command, cwd=ROOT, check=True, pass_fds=(lock.fileno(),))
            records = load_records(directory)
            _atomic_json(directory / "results_summary.json", records)
        print(f"Results: {directory / 'results_summary.json'}")


if __name__ == "__main__":
    main()
