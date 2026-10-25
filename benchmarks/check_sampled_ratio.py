"""Check Step 5 declaration chronology, every outcome, costs and verdict.

This short gate draws no experiment outcomes. Full seed replay is explicit:
python benchmarks/run_sampled_ratio.py --check docs/restart/sampled_ratio_results.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_sampled_ratio import (
    DECLARATION, RESULT, array_admission_bound, device_seconds, digest, git,
    load_declaration, setup, summarize, verdict,
)
from clifford_qc.measurement.ratio import uniform_setting_plan


def check(record):
    config = load_declaration()
    if record["schema"] != "clifford_qc.sampled_ratio_results.v1":
        raise AssertionError("unknown result schema")
    if record["declaration_sha256"] != digest(DECLARATION):
        raise AssertionError("declaration changed after execution")
    committed = subprocess.check_output(
        ["git", "show", f'{record["declaration_commit"]}:{DECLARATION}'], cwd=ROOT)
    import hashlib
    if hashlib.sha256(committed).hexdigest() != record["declaration_sha256"]:
        raise AssertionError("declaration commit does not bind the declaration")
    subprocess.run(["git", "merge-base", "--is-ancestor", record["declaration_commit"],
                    record["execution_commit"]], cwd=ROOT, check=True)
    # The execution HEAD must contain the frozen declaration as well.
    at_execution = subprocess.check_output(
        ["git", "show", f'{record["execution_commit"]}:{DECLARATION}'], cwd=ROOT)
    if at_execution != committed:
        raise AssertionError("execution declaration differs")
    for path, expected_digest in record["source_sha256"].items():
        observed = subprocess.check_output(
            ["git", "show", f'{record["execution_commit"]}:{path}'], cwd=ROOT)
        if hashlib.sha256(observed).hexdigest() != expected_digest:
            raise AssertionError(f"execution source binding differs: {path}")
    # After publication, declaration must precede the first committed record.
    first = git("log", "--diff-filter=A", "--format=%H", "--", RESULT).splitlines()
    if first:
        if first[-1] == record["declaration_commit"]:
            raise AssertionError("declaration and outcomes were published in the same commit")
        subprocess.run(["git", "merge-base", "--is-ancestor", record["execution_commit"], first[-1]],
                       cwd=ROOT, check=True)
    expected = {(c["name"], b, r, arm) for c in config["cases"]
                for b in config["total_shot_budgets"] for r in range(config["replicas"])
                for arm in ("ratio", "bank")}
    keys = [(r["case"], r["budget"], r["replica"], r["arm"]) for r in record["rows"]]
    if len(keys) != len(expected) or set(keys) != expected:
        raise AssertionError("replicas missing/duplicated, including failed attempts")
    for ci, case in enumerate(config["cases"]):
        case_rows = [r for r in record["rows"] if r["case"] == case["name"]]
        if all(r["status"] == "resource_failed" and r["total_shots"] == 0 for r in case_rows):
            if any(r["estimate"] is not None or not r.get("failure_reason") for r in case_rows):
                raise AssertionError("invalid failed admission accounting")
            continue
        _, _, _, _, provider, _, admitted, _ = setup(case, config)
        for row in (r for r in record["rows"] if r["case"] == case["name"]):
            budget, arm = row["budget"], row["arm"]
            bi = config["total_shot_budgets"].index(budget)
            seed = config["master_seed"]+100000*ci+1000*bi+3*row["replica"]
            if row["pilot_seed"] != seed or row["evaluation_seed"] != seed+(1 if arm == "ratio" else 2):
                raise AssertionError("independent streams differ from preregistration")
            if row["status"] not in ("resolved", "denominator_unresolved", "pilot_failed", "resource_failed"):
                raise AssertionError("unknown failure status")
            if row["total_shots"] != row["pilot_shots"]+row["evaluation_shots"]:
                raise AssertionError("physical shots were double-counted or omitted")
            if not row["pilot_shots"] <= row["total_shots"] <= budget:
                raise AssertionError("shot budget breached")
            if row["status"] == "resolved" and row["total_shots"] != budget:
                raise AssertionError("resolved nondeterministic experiment did not spend its frozen budget")
            if row["admitted_array_bytes"] != admitted or admitted > config["owned_array_cap_bytes"]:
                raise AssertionError("array admission model differs or cap breached")
            pilot_counts = uniform_setting_plan(provider.plan.settings, row["pilot_shots"])
            device = device_seconds(provider.plan.settings, pilot_counts, config, case["n"])
            if "evaluation_counts" in row:
                if min(row["evaluation_counts"]) < 2:
                    raise AssertionError("new evaluation group lacks minimum coverage")
                if sum(row["evaluation_counts"]) != row["evaluation_shots"]:
                    raise AssertionError("evaluation counts do not match shots")
                settings = [provider.plan.settings[i] for i in row["evaluation_settings"]]
                device += device_seconds(settings, row["evaluation_counts"], config, case["n"])
            if not math.isclose(device, row["device_seconds"], abs_tol=1e-12):
                raise AssertionError("preparation/readout/compiled-gate cost differs")
            if not math.isclose(row["end_to_end_seconds"], device+row["classical_seconds"], abs_tol=1e-12):
                raise AssertionError("classical inference or device cost was omitted")
            if row["status"] == "resolved":
                for key, truth in (("estimation_error", "true_direction_energy"),
                                   ("retained_space_error", "retained_space_energy"),
                                   ("ground_error", "ground_energy")):
                    if not math.isclose(row[key], abs(row["estimate"]-row[truth]), abs_tol=1e-12):
                        raise AssertionError("estimation/retained-space/ground error accounting differs")
                if row["denominator_interval"][0] <= config["denominator_floor"]:
                    raise AssertionError("denominator failure reported as resolved")
                if row["process_peak_rss_bytes"] > config["process_rss_cap_bytes"]:
                    raise AssertionError("RSS breach reported as resolved")
            elif row["status"] == "denominator_unresolved":
                if row["estimate"] is not None or row["interval"] is not None:
                    raise AssertionError("unresolved ratio presented as an estimate")
    summaries = summarize(record["rows"], config)
    if summaries != record["summaries"]:
        raise AssertionError("unconditional summaries differ from outcomes")
    if verdict(summaries, config) != record["verdict"]:
        raise AssertionError("frozen promotion criterion/verdict differs")
    return len(keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT/RESULT)
    args = parser.parse_args(argv)
    count = check(json.loads(args.record.read_text()))
    print(f"sampled ratio chronology/accounting passed: {count} unconditional outcomes")


if __name__ == "__main__":
    main()
