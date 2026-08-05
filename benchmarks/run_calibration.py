"""Calibration of strict selection (PRA revision headline).

The per-call certification claim is that a *strict* selector -- one that returns a
resolved operator only when the best-arm rule fires, and abstains otherwise
-- selects the wrong operator with probability at most ``delta`` at one
fixed ansatz state. This
experiment measures the empirical wrong-selection probability against the
nominal ``delta``, together with interval coverage, selection regret, the
abstention rate, and the measurement cost.

Method. We build selection *instances* with a known unique argmax of
|g_j|: for a range of random-field Ising and displaced-TFIM Hamiltonians we
evaluate the exact commutator gradients on a fixed generic state, and keep
instances whose top gradient is separated from the runner-up by a real gap
(so "wrong selection" is unambiguous). For each instance, each nominal
``delta`` on a grid, each confidence ``bound`` (normal / empirical-Bernstein),
and many measurement seeds, we run one strict selection and record whether
it resolved, whether the resolved operator was the true argmax, whether it
abstained, the regret |g*|-|g_selected|, and the shots and circuits used.
Interval coverage is the fraction of (candidate, run) pairs whose true |g_j|
lies within the reported bounds at the final round.

Each measurement seed is derived from *both* the instance index and the
seed index, so different instances never share an RNG realization; the runs
are then mutually independent rather than correlated at a fixed seed index
(an instance-independent seed would couple the instances and inflate the
effective sample size). Output is one JSON row per (bound, delta, instance)
with ``scope="instance"`` plus a pooled row with ``scope="pooled"``, so
calibration can be read per instance and not only in aggregate.

Run: python benchmarks/run_calibration.py --out results.jsonl [--seeds 150]
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import median

import numpy as np

from clifford_qc.reproducibility import execution_provenance, stamp_record

from clifford_qc.models import tfim, random_ising
from clifford_qc.algorithms import ConfidenceSelector, local_pool
from clifford_qc.algorithms.adapt import _ansatz_program
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import CommutatorBank, GroupedWordCache, UniformDoubling, qwc_groups

DELTAS = (0.01, 0.05, 0.10, 0.20)
BOUNDS = ("normal", "eb")
GAP = 0.03          # minimum |g(1)| - |g(2)| for a well-posed instance
# A capped budget stresses the strict rule: small-gap instances abstain more
# as delta shrinks, exposing the abstention/cost cost of tighter certification.
BASE, MAX_FACTOR = 256, 64


def instances(n: int = 3):
    """(bank, pool, rho, argmax, gaps) for well-posed selection instances."""
    out = []
    for seed in range(60):
        model = random_ising(n, seed=seed)
        pool = local_pool(n, periodic_context=False)
        bank = CommutatorBank(model.hamiltonian, [op.word for op in pool])
        # displace off the symmetric reference to break exact mirror ties
        prog = _ansatz_program(model, [pool[seed % len(pool)], pool[(seed + 1) % len(pool)]])
        rho = FiniteShotBackend(seed=100 + seed).state(prog, (0.31, -0.47))
        scores = sorted(((abs(bank.exact_score(j, rho)), j) for j in range(len(bank))),
                        reverse=True)
        if scores[0][0] - scores[1][0] >= GAP:
            out.append((model.name, bank, pool, rho, scores[0][1]))
        if len(out) >= 16:
            break
    return out


def one_selection(bank, rho, delta, bound, seed):
    """Run one strict selection; return diagnostics."""
    n = bank.n
    cache = GroupedWordCache(n)
    backend = FiniteShotBackend(seed=seed)
    candidates = list(range(len(bank)))
    fixed_groups = qwc_groups(bank.words_for(candidates))
    sampler = lambda words, plan: backend.sample_grouped_from_state(rho, fixed_groups, plan)
    selector = ConfidenceSelector(delta=delta, threshold=1e-4, bound=bound)
    allocator = UniformDoubling(base=BASE, max_factor=MAX_FACTOR)
    idx, status, diag = selector.select(bank, cache, sampler, allocator, candidates)
    # coverage: does every candidate's interval cover the true |g_j| at the end?
    bounds = selector._bounds(bank, cache, candidates,
                              selector._planned_rounds(allocator),
                              family_size=len(candidates))
    covered = 0
    for j in candidates:
        _, lo, up = bounds[j]
        if lo - 1e-9 <= abs(bank.exact_score(j, rho)) <= up + 1e-9:
            covered += 1
    return idx, status.value, diag, cache.total_shots, cache.total_circuits, \
        covered / len(candidates)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)

    insts = instances(args.n)
    print(f"{len(insts)} well-posed instances (gap >= {GAP})", flush=True)

    def new_agg():
        return {"resolved": 0, "wrong": 0, "abstained": 0, "regret": [],
                "cov": [], "shots": [], "circuits": [], "runs": 0}

    def record(agg, idx, status, cov, shots, circ, argmax, gstar, bank, rho):
        agg["runs"] += 1
        agg["cov"].append(cov)
        agg["shots"].append(shots)
        agg["circuits"].append(circ)
        if status == "resolved_best":
            agg["resolved"] += 1
            if idx != argmax:
                agg["wrong"] += 1
            agg["regret"].append(gstar - abs(bank.exact_score(idx, rho)))
        elif status == "budget_exhausted_ambiguous":
            agg["abstained"] += 1

    def summarize(agg, **extra):
        runs = agg["runs"]
        return {
            **extra, "runs": runs,
            "simultaneous_method": "bonferroni",
            "candidate_family": "fixed_pre_measurement",
            "allocation": "uniform_doubling",
            "decision_rounds": int(math.log2(MAX_FACTOR)) + 1,
            "base_shots": BASE,
            "max_factor": MAX_FACTOR,
            "wrong_selection_rate": agg["wrong"] / runs,
            "wrong_given_resolved": (agg["wrong"] / agg["resolved"]
                                     if agg["resolved"] else None),
            "resolved_rate": agg["resolved"] / runs,
            "abstention_rate": agg["abstained"] / runs,
            "coverage": float(np.mean(agg["cov"])),
            "median_regret": median(agg["regret"]) if agg["regret"] else 0.0,
            "median_shots": median(agg["shots"]),
            "median_circuits": median(agg["circuits"]),
        }

    provenance = execution_provenance()
    with open(args.out, "w") as fh:
        for bound in BOUNDS:
            for delta in DELTAS:
                pooled = new_agg()
                for inst_idx, (name, bank, pool, rho, argmax) in enumerate(insts):
                    per_inst = new_agg()
                    gstar = abs(bank.exact_score(argmax, rho))
                    for s in range(args.seeds):
                        # Instance-specific, seed-specific measurement stream:
                        # distinct instances never share an RNG realization, so
                        # the runs are mutually independent (not correlated at
                        # fixed s as an instance-independent seed would make them).
                        seed = ((inst_idx + 1) * 1_000_003 + s * 1009 + 7) % (2 ** 31)
                        idx, status, diag, shots, circ, cov = one_selection(
                            bank, rho, delta, bound, seed=seed)
                        record(per_inst, idx, status, cov, shots, circ,
                               argmax, gstar, bank, rho)
                        record(pooled, idx, status, cov, shots, circ,
                               argmax, gstar, bank, rho)
                    inst_row = summarize(per_inst, scope="instance", bound=bound,
                                         delta=delta, instance=name,
                                         instance_index=inst_idx)
                    fh.write(json.dumps(stamp_record(inst_row, provenance)) + "\n")
                    fh.flush()
                row = summarize(pooled, scope="pooled", bound=bound, delta=delta)
                fh.write(json.dumps(stamp_record(row, provenance)) + "\n")
                fh.flush()
                print(f"bound={bound} delta={delta:.2f}: wrong={row['wrong_selection_rate']:.4f} "
                      f"(<= delta? {row['wrong_selection_rate'] <= delta}) "
                      f"resolved={row['resolved_rate']:.2f} abstain={row['abstention_rate']:.2f} "
                      f"cover={row['coverage']:.3f} regret={row['median_regret']:.2e} "
                      f"shots={row['median_shots']:.0f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
