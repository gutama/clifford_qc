"""Fresh-seed check of bootstrap intervals at the stop (criteria in CRITERIA.md).

The look loop mirrors clifford_qc.subspace.acquisition.acquire_ritz exactly (no
stopping rule, so a stopped run is this run truncated); `verify` checks that the
arm-A stop equals acquire_ritz's for the same seed.
"""
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/user/clifford_qc")
import numpy as np

from benchmarks.run_interval_coverage import build_system, derived_seed
from benchmarks.run_target_acquisition import load_declaration, schedule
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.measurement.functionals import ritz_functional
from clifford_qc.subspace.acquisition import (acquire_ritz, neyman_group_plan,
                                              uniform_group_plan)
from clifford_qc.subspace.uncertainty import bootstrap_ritz_samples

D = load_declaration()
ROOT = 2026111901
B = 500
FAILURES = (ValueError, np.linalg.LinAlgError)


def run(system, replica, allocation_index):
    spec = D["systems"][system]
    shared, design = build_system(spec)
    pilot, budgets = schedule(D, len(shared.groups))
    allocation = D["allocations"][allocation_index]
    seed = derived_seed(ROOT, system, replica)
    targets = list(D["target_std_errors"])
    backend = FiniteShotBackend(seed=seed)
    cache = shared.measure(backend, pilot)
    arm_a, arm_b = {}, {}          # target -> record
    last = None

    def snapshot(index, point, sigma, t, tag):
        draws = bootstrap_ritz_samples(
            shared, cache, result=point, replicates=B,
            seed=derived_seed(ROOT, 2_000_000 + system, replica,
                              10 * allocation_index + t, index))
        q = np.quantile(draws, [0.025, 0.16, 0.5, 0.84, 0.975])
        return {"look": index, "shots": int(cache.total_shots),
                "estimate": float(point.energies[0]), "sigma": sigma,
                "rank": int(point.effective_rank), "robust": float((q[3] - q[1]) / 2),
                "quantiles": [float(v) for v in q], "stopped": tag}

    for index in range(len(budgets) + 1):
        try:
            point = shared.solve(cache, **D["solver_kwargs"])
        except FAILURES as error:
            return {"system": spec["key"], "allocation": allocation, "replica": replica,
                    "failed": f"look {index}: {type(error).__name__}: {error}",
                    "A": arm_a, "B": arm_b}
        functional = ritz_functional(shared.bank, point.indices, point.ritz_vector(0),
                                     point.energies[0])
        sigma = math.sqrt(max(0.0, functional.variance(cache)))
        last = (index, point, sigma)
        for t, goal in enumerate(targets):
            if sigma > goal:
                continue
            boot = None
            if goal not in arm_a:
                boot = snapshot(index, point, sigma, t, "target_met")
                arm_a[goal] = boot
            if goal not in arm_b:
                boot = boot or snapshot(index, point, sigma, t, "target_met")
                if boot["robust"] <= goal:
                    arm_b[goal] = boot
        if all(g in arm_a and g in arm_b for g in targets) or index == len(budgets):
            break
        if allocation == "uniform":
            plan = uniform_group_plan(shared.groups, budgets[index])
        else:
            plan = neyman_group_plan(shared.groups, cache, functional, budgets[index])
        if not plan:
            break
        shared.measure_plan(backend, plan, cache)
    index, point, sigma = last
    for t, goal in enumerate(targets):
        for arm in (arm_a, arm_b):
            if goal not in arm:
                arm[goal] = snapshot(index, point, sigma, t, "schedule_exhausted")
    return {"system": spec["key"], "allocation": allocation, "replica": replica,
            "seed": seed, "target": design["target"], "exact_rank": design["exact_rank"],
            "A": {repr(k): v for k, v in arm_a.items()},
            "B": {repr(k): v for k, v in arm_b.items()}}


def verify(system, replica, allocation_index):
    """The arm-A stop must be acquire_ritz's stop for the same seed."""
    spec = D["systems"][system]
    shared, _ = build_system(spec)
    pilot, budgets = schedule(D, len(shared.groups))
    row = run(system, replica, allocation_index)
    for goal in D["target_std_errors"]:
        found = acquire_ritz(shared, FiniteShotBackend(seed=derived_seed(ROOT, system, replica)),
                             pilot_shots_per_group=pilot, round_budgets=budgets,
                             allocation=D["allocations"][allocation_index],
                             target_std_error=goal, solver_kwargs=dict(D["solver_kwargs"]),
                             delta=D["delta"])
        a = row["A"][repr(goal)]
        assert a["estimate"] == float(found.result.energies[0]), (system, replica, goal)
        assert a["look"] == len(found.looks) - 1 and a["sigma"] == found.std_error
    return True


def work(task):
    system, replica = task
    return [run(system, replica, a) for a in range(len(D["allocations"]))]


def main(replicas, workers, path):
    tasks = sorted(((s, r) for s in range(len(D["systems"])) for r in range(replicas)),
                   key=lambda task: -task[0])
    rows, started = [], time.time()
    with ProcessPoolExecutor(workers) as pool:
        for done, chunk in enumerate(pool.map(work, tasks, chunksize=2), start=1):
            rows.extend(chunk)
            if done % 50 == 0:
                print(f"{done}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    json.dump({"root": ROOT, "replicas": replicas, "bootstrap_replicates": B, "rows": rows},
              open(path, "w"))
    print("done", time.time() - started, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "verify":
        for s in range(len(D["systems"])):
            for a in range(len(D["allocations"])):
                for r in range(2):
                    verify(s, r, a)
        print("verified")
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
