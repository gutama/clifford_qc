"""Screen: bootstrap intervals at the stopping look of the declared acquisition runs.

Development data: the declared study's own seeds, already used to screen other
stopping rules. Every stopped run is regenerated with the declared arguments and
checked against the record (estimate and looks taken), then its cache at the
stop is bootstrapped with the declared solver policy (bootstrap_ritz_samples,
bound to the point result). The stopping rule itself is unchanged; only the
interval reported at the stop differs.
"""
import gzip
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/user/clifford_qc")
import numpy as np

from benchmarks.run_interval_coverage import build_system, derived_seed
from benchmarks.run_target_acquisition import (load_declaration, measurement_seed, schedule,
                                               target_key)
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.subspace.acquisition import AcquisitionError, acquire_ritz
from clifford_qc.subspace.uncertainty import bootstrap_ritz_samples

D = load_declaration()
RECORD = "/home/user/clifford_qc/docs/restart/target_acquisition_results.json.gz"
QUANTILES = (0.025, 0.5, 0.975)
_STORED = None


def stored():
    global _STORED
    if _STORED is None:
        rec = json.load(gzip.open(RECORD))
        _STORED = {(r["system"], r["allocation"], r["replica"]): r["stopping"]
                   for r in rec["rows"]}
    return _STORED


def work(task):
    system, replica, replicates = task
    spec = D["systems"][system]
    shared, design = build_system(spec)
    pilot, budgets = schedule(D, len(shared.groups))
    seed = measurement_seed(D, system, replica)
    out = []
    for a, allocation in enumerate(D["allocations"]):
        for t, goal in enumerate(D["target_std_errors"]):
            record = stored()[(spec["key"], allocation, replica)][target_key(goal)]
            row = {"system": spec["key"], "allocation": allocation, "replica": replica,
                   "target_std_error": goal, "exact_rank": design["exact_rank"]}
            try:
                found = acquire_ritz(shared, FiniteShotBackend(seed=seed),
                                     pilot_shots_per_group=pilot, round_budgets=budgets,
                                     allocation=allocation, target_std_error=goal,
                                     solver_kwargs=dict(D["solver_kwargs"]), delta=D["delta"])
            except AcquisitionError as error:
                row.update(outcome="solve_failed", matches=record["outcome"] == "solve_failed")
                out.append(row)
                continue
            point = found.result
            estimate = float(point.energies[0])
            row.update(outcome="ok", stopped=found.stopped, looks=len(found.looks),
                       shots=found.shots, estimate=estimate, error=estimate - design["target"],
                       sigma=found.std_error, rank=int(point.effective_rank),
                       matches=(record["estimate"] == estimate
                                and record["looks_taken"] == len(found.looks)))
            boot_seed = derived_seed(D["seed_root"], 2_000_000 + system, replica, 10 * a + t)
            started = time.time()
            try:
                values = bootstrap_ritz_samples(shared, found.cache, result=point,
                                                replicates=replicates, seed=boot_seed)
            except (ValueError, np.linalg.LinAlgError) as error:
                row.update(boot_failed=f"{type(error).__name__}: {error}")
            else:
                row.update(samples=[float(v) for v in values],
                           boot_seconds=time.time() - started)
            out.append(row)
    return out


def main(replicas, replicates, workers, path):
    tasks = [(s, r, replicates) for s in range(len(D["systems"])) for r in range(replicas)]
    # Longest first: the six-qubit bank dominates.
    tasks.sort(key=lambda task: -task[0])
    rows, started = [], time.time()
    with ProcessPoolExecutor(workers) as pool:
        for done, chunk in enumerate(pool.map(work, tasks, chunksize=2), start=1):
            rows.extend(chunk)
            if done % 50 == 0:
                print(f"{done}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    json.dump({"replicas": replicas, "bootstrap_replicates": replicates, "rows": rows},
              open(path, "w"))
    print("done", time.time() - started, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
