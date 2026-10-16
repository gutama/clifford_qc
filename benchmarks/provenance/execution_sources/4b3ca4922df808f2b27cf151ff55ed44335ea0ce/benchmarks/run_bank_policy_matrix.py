"""Preflight Phase 2M; --execute runs every frozen arm in a fresh process."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import bank_policy_contract as contract


def host_capacity():
    """Available memory within both Linux host and current cgroup limits."""
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    limits = []
    # The default cgroup v2 mount and ancestors all constrain this process.
    relative = next((line.split("::", 1)[1] for line in
                     Path("/proc/self/cgroup").read_text().splitlines() if "::" in line), "")
    current = Path("/sys/fs/cgroup") / relative.lstrip("/")
    root = Path("/sys/fs/cgroup")
    if not current.exists():
        current = root
    while True:
        maximum, usage = current / "memory.max", current / "memory.current"
        if maximum.exists() and maximum.read_text().strip() != "max":
            cap = int(maximum.read_text())
            used = int(usage.read_text())
            available = min(available, max(0, cap - used))
            limits.append(cap)
        if current == root:
            break
        current = current.parent
    # Fall back to cgroup v1's conventional memory controller mount.
    if not limits:
        base = Path("/sys/fs/cgroup/memory")
        maximum, usage = base / "memory.limit_in_bytes", base / "memory.usage_in_bytes"
        if maximum.exists() and usage.exists():
            cap, used = int(maximum.read_text()), int(usage.read_text())
            available = min(available, max(0, cap - used))
            limits.append(cap)
    return {"available_bytes": available, "cgroup_limits_bytes": limits}


def require_host_capacity(execution):
    capacity = host_capacity()
    required = execution["rss_ceiling_bytes"] + execution["host_reserve_bytes"]
    if capacity["available_bytes"] < required:
        raise ValueError(f"insufficient host capacity: {capacity['available_bytes']} available bytes; "
                         f"{required} required for the unchanged 15 GiB campaign plus reserve")
    return capacity


def stop_worker(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_cell(system, arm, config):
    """No fork inheritance or reused worker; parent enforces time and RSS limits."""
    execution = config["execution"]
    started = time.perf_counter()
    peak = solve_peak = 0
    baseline = None
    status = None
    with tempfile.TemporaryDirectory(prefix="phase2m-arm-") as directory:
        directory = Path(directory)
        request, output, ready, log = [directory / name for name in
                                      ("request.json", "cell.json", "ready.json", "stderr.log")]
        contract.atomic_json(request, {"system": system, "arm": list(arm),
                                      "options": config["options"],
                                      "frontier_pairs": execution["frontier_pairs"],
                                      "sample_seconds": execution["sample_seconds"]})
        env = dict(os.environ)
        env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
        env["PYTHONHASHSEED"] = "0"
        command = [sys.executable, str(ROOT / "benchmarks/bank_policy_worker.py"),
                   "--request", str(request), "--output", str(output), "--ready", str(ready)]
        with log.open("wb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                                       stderr=stderr, start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        rss = int(Path(f"/proc/{process.pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                    except (OSError, IndexError):
                        rss = 0
                    peak = max(peak, rss)
                    if ready.exists():
                        signal = contract.read_json(ready)
                        if signal["pid"] != process.pid:
                            raise ValueError("worker ready identity differs")
                        baseline = signal["baseline_rss_bytes"]
                        # Self-reported RSS also covers command wrappers whose
                        # externally visible PID is a small transport process.
                        rss = max(rss, signal["rss_bytes"], signal["peak_rss_bytes"])
                        peak = max(peak, rss)
                    if baseline is not None:
                        solve_peak = max(solve_peak, rss)
                    if rss > execution["rss_ceiling_bytes"]:
                        status = "memory_limit"
                        stop_worker(process)
                    elif time.perf_counter() - started > execution["timeout_seconds"]:
                        status = "timeout"
                        stop_worker(process)
                    if status is None:
                        time.sleep(execution["sample_seconds"])
            except BaseException:
                if process.poll() is None:
                    stop_worker(process)
                raise
        if status is None:
            status = "completed" if process.returncode == 0 and output.exists() else "worker_failed"
        parent = {"pid": os.getpid(), "launched_pid": process.pid,
                  "returncode": process.returncode, "wall_seconds": time.perf_counter() - started,
                  "sampled_peak_rss_bytes": peak, "sampled_solve_peak_rss_bytes": solve_peak,
                  "sampled_solve_delta_rss_bytes": None if baseline is None else max(0, solve_peak - baseline),
                  "rss_ceiling_bytes": execution["rss_ceiling_bytes"],
                  "timeout_seconds": execution["timeout_seconds"],
                  "sample_seconds": execution["sample_seconds"], "launch_method": "exec_fresh_python"}
        if status == "completed":
            cell = contract.read_json(output)
            # A high-water crossing missed by polling is still a failed arm.
            if cell["costs"]["worker_high_water_rss_bytes"] > execution["rss_ceiling_bytes"]:
                status = "memory_limit"
            cell["status"] = status
        else:
            cell = {"system": system["id"], "arm": list(arm), "status": status,
                    "diagnostic": log.read_text(errors="replace")[-4000:]}
        cell["parent"] = parent
        return cell


def execution_preflight():
    config = contract.load_config()
    if contract.RECORD.exists():
        raise ValueError("campaign record already exists")
    if contract.git("rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("execution requires full Git history")
    if contract.git("status", "--porcelain"):
        raise ValueError("execution requires a clean committed tree")
    revision = contract.git("rev-parse", "HEAD")
    config_path = str(contract.CONFIG.relative_to(ROOT))
    declaration = contract.git("log", "--diff-filter=A", "--format=%H", "--", config_path).splitlines()[-1]
    contract.git("merge-base", "--is-ancestor", declaration, revision)
    if contract.git("show", f"{declaration}:{config_path}") != contract.CONFIG.read_text().strip():
        raise ValueError("declaration changed after its first commit")
    from clifford_qc.record_environment import guard
    guard()
    capacity = require_host_capacity(config["execution"])
    from clifford_qc.reproducibility import execution_provenance
    provenance = execution_provenance()
    provenance.update(git_sha=revision, git_dirty=False, declaration_commit=declaration)
    provenance["source_blobs"] = {path: contract.git("rev-parse", f"{revision}:{path}")
                                  for path in contract.SOURCES}
    for path, blob in provenance["source_blobs"].items():
        if contract.git("hash-object", path) != blob:
            raise ValueError(f"execution source differs from committed blob: {path}")
    common = Path(contract.git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = ROOT / common
    claim = common / f"phase2m-policy-{contract.EXPECTED_CONFIG_SHA256}.execution.json"
    if claim.exists():
        raise ValueError("the campaign was already claimed; preserve its diagnostics")
    return config, provenance, capacity, claim


def run_campaign(config, checkpoint):
    cells = []
    for system in config["systems"]:
        for arm in contract.ARMS:
            print(f"{system['id']} / {arm[0]}: fresh worker", flush=True)
            cell = run_cell(system, arm, config)
            cells.append(cell)
            checkpoint(cells)
            print(f"  {cell['status']}", flush=True)
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    claim = None
    claimed = False
    try:
        if not args.execute:
            config = contract.load_config()
            print(f"OK: frozen {len(config['systems'])} x 4 policy matrix; no energies or costs evaluated")
            return 0
        config, provenance, capacity, claim = execution_preflight()
        contract.atomic_json(claim, {"status": "claimed", "provenance": provenance,
                                     "completed_cells": []}, exclusive=True)
        claimed = True
        def checkpoint(cells):
            contract.atomic_json(claim, {"status": "running", "provenance": provenance,
                                         "completed_cells": cells})
        cells = run_campaign(config, checkpoint)
        from benchmarks.check_bank_policy_matrix import comparison_problems
        # The producer labels equality directly; the checker derives every gate.
        completed = all(cell["status"] == "completed" for cell in cells)
        equivalent = completed and all(
            all(cell["scientific"] == block[0]["scientific"] for cell in block)
            for block in [cells[i:i + 4] for i in range(0, len(cells), 4)])
        verdict = "INCOMPLETE" if not completed else "VALID" if equivalent else "INVALID"
        record = {"schema": contract.SCHEMA, "phase": "2M-C/D",
                  "configuration_sha256": contract.EXPECTED_CONFIG_SHA256,
                  "claim_boundary": contract.CLAIM_BOUNDARY, "cells": cells,
                  "verdict": verdict, "execution": {"capacity": capacity,
                  "parent_pid": os.getpid(), "serial_workers": True}, "provenance": provenance}
        if verdict == "VALID" and comparison_problems(record, config):
            record["verdict"] = "INVALID"
        from clifford_qc.reproducibility import stamp_record
        record = stamp_record(record, provenance)
        contract.atomic_json(contract.RECORD, record, exclusive=True)
        contract.atomic_json(claim, {"status": "completed", "provenance": provenance,
                                     "record": str(contract.RECORD.relative_to(ROOT)),
                                     "verdict": record["verdict"]})
        print(f"Wrote {contract.RECORD}: {record['verdict']}")
        return 0 if record["verdict"] == "VALID" else 1
    except Exception as exc:
        if claimed:
            try:
                state = contract.read_json(claim)
                state.update(status="failed", error=str(exc))
                contract.atomic_json(claim, state)
            except Exception as persistence_error:
                # The exclusive claim remains consumed even if storage itself
                # failed. Never replace the original error with this failure.
                print(f"Could not update consumed claim: {persistence_error}", file=sys.stderr)
        print(f"Phase 2M campaign refused/failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
