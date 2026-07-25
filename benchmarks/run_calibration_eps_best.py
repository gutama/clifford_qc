"""Calibration of the certified eps-best rule (PRA revision, Priority 3).

The eps-best certificate claims that a strict eps-best selector returns an
operator whose gradient is within ``eps`` of the maximum with probability at
least ``1 - delta`` at one fixed ansatz state:

    Pr(resolved and |g_selected| < max_k|g_k| - eps)  <=  delta.

This experiment recalibrates *wrong selection under the eps-best definition*
(``|g_selected| < gmax - eps``), which differs from the exact-argmax
definition used in ``run_calibration.py``: an eps-best selection on a tied or
near-tied instance is correct even when it is not the unique argmax. That is
exactly the regime the eps-best rule was built for, so the instance set here
is deliberately broad -- symmetric (exact-tie) reference states, displaced
generic states, and actual exact-ADAPT trajectory states -- rather than the
well-posed unique-argmax instances of the exact-best calibration.

For each (bound, delta, eps) it reports, pooled and stratified by the
eps-boundary gap:
  - the empirical eps-best wrong-selection rate and its one-sided 95%
    Clopper-Pearson upper confidence bound (conditional on resolving);
  - for reference, the exact-argmax wrong rate (how often eps-best commits a
    non-argmax operator -- correct under eps-best, would be "wrong" under the
    stricter definition);
  - the resolved and abstention rates, simultaneous interval coverage, and
    measurement cost.

Run: python benchmarks/run_calibration_eps_best.py --out results.jsonl [--seeds 60]
Requires the ``chemistry`` extra only if molecular instances are enabled
(off by default).
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import median

import numpy as np

from clifford_qc.models import tfim, random_ising, xxz
from clifford_qc.algorithms import ConfidenceSelector, local_pool, run_adapt
from clifford_qc.algorithms.adapt import _ansatz_program
from clifford_qc.backends import FiniteShotBackend, ExactMVBackend
from clifford_qc.measurement import (
    CommutatorBank, GroupedWordCache, UniformDoubling, qwc_groups)

DELTAS = (0.05, 0.10, 0.20)
BOUNDS = ("normal", "eb")
EPS_GRID = (0.15, 0.30)
BASE, MAX_FACTOR = 512, 64
TIGHT_GAP = 0.10  # eps-boundary gap below this is a "tight" (hard) instance


def _clopper_pearson_upper(wrong: int, total: int, alpha: float = 0.05) -> float:
    """One-sided upper 1-alpha confidence bound on a binomial rate."""
    if total == 0:
        return 1.0
    if wrong == 0:
        return 1.0 - alpha ** (1.0 / total)
    try:
        from scipy.stats import beta
        return float(beta.ppf(1.0 - alpha, wrong + 1, total - wrong))
    except Exception:  # pragma: no cover - scipy always present in test env
        return min(1.0, wrong / total + 1.96 * math.sqrt(
            (wrong / total) * (1 - wrong / total) / total))


def _instance(name, bank, rho):
    g = [abs(bank.exact_score(j, rho)) for j in range(len(bank))]
    return {"name": name, "bank": bank, "rho": rho, "g": g, "gmax": max(g)}


def make_instances(n: int = 3):
    """Diverse selection instances: symmetric (tie) references, displaced
    generic states, and actual exact-ADAPT trajectory states."""
    insts = []
    exact = ExactMVBackend()
    # 1. symmetric reference states -> exact gradient ties
    for h in (0.5, 1.0, 1.5, 2.0):
        m = tfim(n, 1.0, h, periodic=False)
        bank = CommutatorBank(m.hamiltonian, [op.word for op in local_pool(n, periodic_context=False)])
        insts.append(_instance(f"tfim_h{h}_ref", bank, m.reference.state()))
    for seed in (0, 1, 2):
        m = random_ising(n, seed=seed)
        bank = CommutatorBank(m.hamiltonian, [op.word for op in local_pool(n, periodic_context=False)])
        insts.append(_instance(f"rising_s{seed}_ref", bank, m.reference.state()))
    m = xxz(n)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in local_pool(n, periodic_context=False)])
    insts.append(_instance("xxz_ref", bank, m.reference.state()))
    # 2. displaced generic states (break exact ties)
    for seed in range(18):
        m = random_ising(n, seed=seed)
        pool = local_pool(n, periodic_context=False)
        bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
        prog = _ansatz_program(m, [pool[seed % len(pool)], pool[(seed + 1) % len(pool)]])
        rho = FiniteShotBackend(seed=100 + seed).state(prog, (0.31, -0.47))
        insts.append(_instance(f"rising_s{seed}_displaced", bank, rho))
    # 3. actual trajectory states: run exact ADAPT, snapshot each prefix state
    for tag, m in (("tfim_h1", tfim(n, 1.0, 1.0, periodic=False)),
                   ("rising_s3", random_ising(n, seed=3))):
        pool = local_pool(n, periodic_context=False)
        bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
        res = run_adapt(m, pool, max_operators=4, threshold=1e-6)
        chosen = [next(op for op in pool if op.label == lab) for lab in res.labels]
        for k in range(1, len(chosen) + 1):
            prog = _ansatz_program(m, chosen[:k])
            rho = exact.state(prog, res.parameters[:k])
            insts.append(_instance(f"{tag}_traj{k}", bank, rho))
    return insts


def eps_best_selection(inst, delta, bound, eps, seed):
    bank, rho = inst["bank"], inst["rho"]
    n = bank.n
    cache = GroupedWordCache(n)
    backend = FiniteShotBackend(seed=seed)
    candidates = list(range(len(bank)))
    fixed_groups = qwc_groups(bank.words_for(candidates))
    sampler = lambda words, plan: backend.sample_grouped_from_state(rho, fixed_groups, plan)
    selector = ConfidenceSelector(delta=delta, threshold=1e-4, bound=bound, near_tol=eps)
    idx, status, diag = selector.select(bank, cache, sampler, allocator := UniformDoubling(base=BASE, max_factor=MAX_FACTOR), candidates)
    bounds = selector._bounds(bank, cache, candidates,
                              selector._planned_rounds(allocator), family_size=len(candidates))
    covered = sum(1 for j in candidates
                  if bounds[j][1] - 1e-9 <= inst["g"][j] <= bounds[j][2] + 1e-9)
    return idx, status.value, cache.total_shots, covered / len(candidates)


def eps_boundary_gap(g, gmax, eps):
    thr = gmax - eps
    sub = [x for x in g if x < thr - 1e-12]
    return float("inf") if not sub else thr - max(sub)


def stratum_of(gap):
    if gap == float("inf"):
        return "all_within_eps"
    return "tight" if gap < TIGHT_GAP else "clear"


def new_agg():
    return {"runs": 0, "resolved": 0, "eps_wrong": 0, "argmax_wrong": 0,
            "abstained": 0, "cov": [], "shots": []}


def record(agg, idx, status, shots, cov, inst, eps):
    agg["runs"] += 1
    agg["cov"].append(cov)
    agg["shots"].append(shots)
    if status in ("resolved_best", "resolved_eps_best"):
        agg["resolved"] += 1
        gsel = inst["g"][idx]
        if gsel < inst["gmax"] - eps - 1e-12:
            agg["eps_wrong"] += 1
        if idx != int(np.argmax(inst["g"])):
            agg["argmax_wrong"] += 1
    elif status == "budget_exhausted_ambiguous":
        agg["abstained"] += 1


def summarize(agg, **extra):
    r = agg["runs"]
    return {
        **extra, "runs": r, "resolved": agg["resolved"],
        "eps": extra.get("eps"),
        "eps_best_wrong_rate": agg["eps_wrong"] / r if r else 0.0,
        "eps_best_wrong_given_resolved": (agg["eps_wrong"] / agg["resolved"]
                                          if agg["resolved"] else None),
        "eps_best_wrong_upper95": _clopper_pearson_upper(agg["eps_wrong"], agg["resolved"]),
        "argmax_wrong_given_resolved": (agg["argmax_wrong"] / agg["resolved"]
                                        if agg["resolved"] else None),
        "resolved_rate": agg["resolved"] / r if r else 0.0,
        "abstention_rate": agg["abstained"] / r if r else 0.0,
        "coverage": float(np.mean(agg["cov"])) if agg["cov"] else None,
        "median_shots": median(agg["shots"]) if agg["shots"] else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)

    insts = make_instances(args.n)
    print(f"{len(insts)} instances (references/displaced/trajectory)", flush=True)

    with open(args.out, "w") as fh:
        for bound in BOUNDS:
            for delta in DELTAS:
                for eps in EPS_GRID:
                    pooled = new_agg()
                    strata = {}
                    for inst_idx, inst in enumerate(insts):
                        gap = eps_boundary_gap(inst["g"], inst["gmax"], eps)
                        strat = stratum_of(gap)
                        strata.setdefault(strat, new_agg())
                        for s in range(args.seeds):
                            seed = ((inst_idx + 1) * 1_000_003 + s * 1009 + 7) % (2 ** 31)
                            idx, status, shots, cov = eps_best_selection(inst, delta, bound, eps, seed)
                            record(pooled, idx, status, shots, cov, inst, eps)
                            record(strata[strat], idx, status, shots, cov, inst, eps)
                    row = summarize(pooled, scope="pooled", bound=bound, delta=delta, eps=eps)
                    fh.write(json.dumps(row) + "\n")
                    for strat, agg in sorted(strata.items()):
                        fh.write(json.dumps(summarize(
                            agg, scope="stratum", stratum=strat, bound=bound,
                            delta=delta, eps=eps)) + "\n")
                    fh.flush()
                    print(f"bound={bound} d={delta:.2f} eps={eps}: "
                          f"eps_wrong={row['eps_best_wrong_rate']:.4f} "
                          f"(<=d? {row['eps_best_wrong_rate'] <= delta}) "
                          f"up95(cond)={row['eps_best_wrong_upper95']:.4f} "
                          f"resolved={row['resolved_rate']:.2f} "
                          f"argmax_wrong|res={row['argmax_wrong_given_resolved']:.3f} "
                          f"cover={row['coverage']:.3f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
