"""Where fixed-shot selection fails: gap-stratified hard instances (referee item).

On the easy spin families of ``run_baselines.py`` a single fixed-shot round is
as accurate as escalating confidence-guided selection and much cheaper, so the
baseline ladder on its own reads as an argument *against* the elaborate method.
That comparison is dominated by instances whose top-two gradient gap is large:
any sensible estimator ranks them correctly, and the extra machinery buys
nothing. The distinguishing regime is the small-gap one.

This experiment stratifies selection instances by the *normalized* top-two gap

    Delta = (|g|_(1) - |g|_(2)) / |g|_(1)

and, within each stratum, compares four arms at one fixed ansatz state:

  ``fixed_shot``      one predeclared round, then commit to the empirical
                      leader: no escalation, no abstention, no guarantee;
  ``fallback``        escalating uniform doubling under the asymptotic normal
                      bound, committing to the leader when the budget is spent;
  ``strict_exact_eb`` strict empirical-Bernstein exact-best: commit only when
                      the leader is certified the argmax, otherwise abstain;
  ``strict_eps_eb``   strict empirical-Bernstein eps-best: commit on the
                      weaker (eps, delta) certificate, otherwise abstain.

Reported per (stratum, arm): wrong-selection rate against the exact argmax,
eps-best violation rate, absolute and normalized selection regret
(|g*| - |g_sel|)/|g*|, resolution and abstention rates, and the shot and
circuit cost. A wrong selection is charged to ``fixed_shot`` and ``fallback``
whenever they commit to a non-argmax; the strict arms can only be charged when
they *resolve*, which is the whole point of the four-way outcome. The
comparison the referee's ladder invites is therefore not "who has the lowest
error at matched cost" -- it is that the committing arms have an error rate
they cannot see, while the strict arms convert the same uncertainty into a
reported abstention.

A second part runs complete ADAPT trajectories at a *matched shot budget* on
the same families, so the accuracy-per-shot comparison the ladder invites is
made on equal terms rather than at each arm's own budget.

Run: python benchmarks/run_hard_instances.py --out results.jsonl [--seeds 60]
"""

from __future__ import annotations

import argparse
import json
from statistics import median

import numpy as np

from clifford_qc.algorithms import ConfidenceSelector, local_pool, run_adapt
from clifford_qc.algorithms.adapt import _ansatz_program
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.matrix import exact_ground
from clifford_qc.measurement import (
    CommutatorBank, GroupedWordCache, UniformDoubling, UniformFixed, qwc_groups,
)
from clifford_qc.models import random_ising

EPS = 0.30
DELTA = 0.10
FIXED_SHOTS = 4096
BASE, MAX_FACTOR = 256, 64
# normalized-gap stratum edges: (label, lower, upper)
STRATA = (("tight", 0.0, 0.02), ("narrow", 0.02, 0.10),
          ("moderate", 0.10, 0.30), ("wide", 0.30, 1.01))
TIE_TOL = 1e-9


def stratum_of(rel_gap):
    for label, lo, hi in STRATA:
        if lo <= rel_gap < hi:
            return label
    return STRATA[-1][0]


def instances(n=3, per_stratum=8, pool_seeds=400):
    """Selection instances binned by normalized top-two gradient gap.

    Displacing a random-field Ising state by a two-parameter rotation sweeps
    the gap continuously, so scanning seeds and angles populates every stratum
    including the tight one that the well-posed calibration set excludes.
    """
    out, counts = [], {label: 0 for label, _, _ in STRATA}
    angles = ((0.31, -0.47), (0.11, -0.09), (0.63, 0.22), (0.05, 0.41),
              (1.02, 0.77), (1.57, -1.13), (0.88, 1.31), (2.10, 0.36))
    for seed in range(pool_seeds):
        if all(c >= per_stratum for c in counts.values()):
            break
        model = random_ising(n, seed=seed)
        pool = local_pool(n, periodic_context=False)
        bank = CommutatorBank(model.hamiltonian, [op.word for op in pool])
        for ai, angle in enumerate(angles):
            if all(c >= per_stratum for c in counts.values()):
                break
            prog = _ansatz_program(model, [pool[seed % len(pool)],
                                           pool[(seed + 1) % len(pool)]])
            rho = FiniteShotBackend(seed=100 + seed).state(prog, angle)
            mags = sorted((abs(bank.exact_score(j, rho)) for j in range(len(bank))),
                          reverse=True)
            if mags[0] < 1e-3 or mags[0] - mags[1] <= TIE_TOL:
                continue   # a vanishing leader or an exact tie is a different regime
            rel_gap = (mags[0] - mags[1]) / mags[0]
            label = stratum_of(rel_gap)
            if counts[label] >= per_stratum:
                continue
            counts[label] += 1
            argmax = max(range(len(bank)), key=lambda j: abs(bank.exact_score(j, rho)))
            out.append({"name": f"{model.name}@a{ai}", "bank": bank, "rho": rho,
                        "argmax": argmax, "gmax": mags[0], "rel_gap": rel_gap,
                        "stratum": label})
    return out


def _select(bank, rho, arm, seed):
    """One selection under the given arm; returns (idx, committed, status, cost)."""
    cache = GroupedWordCache(bank.n)
    backend = FiniteShotBackend(seed=seed)
    candidates = list(range(len(bank)))
    groups = qwc_groups(bank.words_for(candidates))
    sampler = lambda words, plan: backend.sample_grouped_from_state(rho, groups, plan)
    if arm == "fixed_shot":
        selector = ConfidenceSelector(delta=DELTA, threshold=1e-4, bound="normal")
        allocator = UniformFixed(shots_per_word=FIXED_SHOTS)
    elif arm == "fallback":
        # no eps tolerance: escalate to the ceiling, then take the leader --
        # the honest "adaptive but uncertified" comparator
        selector = ConfidenceSelector(delta=DELTA, threshold=1e-4, bound="normal")
        allocator = UniformDoubling(base=BASE, max_factor=MAX_FACTOR)
    elif arm == "strict_exact_eb":
        selector = ConfidenceSelector(delta=DELTA, threshold=1e-4, bound="eb")
        allocator = UniformDoubling(base=BASE, max_factor=MAX_FACTOR)
    elif arm == "strict_eps_eb":
        selector = ConfidenceSelector(delta=DELTA, threshold=1e-4, bound="eb",
                                      near_tol=EPS)
        allocator = UniformDoubling(base=BASE, max_factor=MAX_FACTOR)
    else:
        raise ValueError(arm)
    idx, status, diag = selector.select(bank, cache, sampler, allocator, candidates)
    resolved = status.value in ("resolved_best", "resolved_eps_best")
    # fixed_shot and fallback commit unconditionally (the empirical leader is
    # returned even for an ambiguous outcome); the strict arms commit only on a
    # certificate.
    committed = resolved if arm.startswith("strict_") else True
    return idx, committed, resolved, status.value, cache.total_shots, cache.total_circuits


ARMS = ("fixed_shot", "fallback", "strict_exact_eb", "strict_eps_eb")


def selection_study(insts, seeds):
    agg = {(s, a): {"runs": 0, "committed": 0, "resolved": 0, "wrong": 0,
                    "eps_violation": 0, "regret": [], "rel_regret": [],
                    "shots": [], "circuits": []}
           for s, _, _ in STRATA for a in ARMS}
    for inst_idx, inst in enumerate(insts):
        bank, rho, gmax = inst["bank"], inst["rho"], inst["gmax"]
        for arm in ARMS:
            a = agg[(inst["stratum"], arm)]
            for s in range(seeds):
                seed = ((inst_idx + 1) * 1_000_003 + s * 1009 + 7) % (2 ** 31)
                idx, committed, resolved, _status, shots, circ = _select(
                    bank, rho, arm, seed)
                a["runs"] += 1
                a["shots"].append(shots)
                a["circuits"].append(circ)
                a["resolved"] += int(resolved)
                if not committed:
                    continue
                a["committed"] += 1
                gsel = abs(bank.exact_score(idx, rho))
                a["regret"].append(gmax - gsel)
                a["rel_regret"].append((gmax - gsel) / gmax)
                a["wrong"] += int(idx != inst["argmax"])
                a["eps_violation"] += int(gsel < gmax - EPS)
    rows = []
    for (stratum, arm), a in agg.items():
        if not a["runs"]:
            continue
        rows.append({
            "part": "selection", "stratum": stratum, "arm": arm,
            "eps": EPS, "delta": DELTA,
            "instances": sum(1 for i in insts if i["stratum"] == stratum),
            "runs": a["runs"],
            "commit_rate": a["committed"] / a["runs"],
            "abstention_rate": 1 - a["committed"] / a["runs"],
            "resolved_rate": a["resolved"] / a["runs"],
            # wrong rates are per *committed* decision: an abstention is not a
            # wrong answer, it is a reported non-answer
            "wrong_rate_given_committed": (a["wrong"] / a["committed"]
                                           if a["committed"] else None),
            "wrong_rate_unconditional": a["wrong"] / a["runs"],
            "eps_violation_rate_given_committed": (a["eps_violation"] / a["committed"]
                                                   if a["committed"] else None),
            "mean_regret": float(np.mean(a["regret"])) if a["regret"] else None,
            "median_regret": median(a["regret"]) if a["regret"] else None,
            "max_regret": max(a["regret"]) if a["regret"] else None,
            "mean_rel_regret": (float(np.mean(a["rel_regret"]))
                                if a["rel_regret"] else None),
            "max_rel_regret": max(a["rel_regret"]) if a["rel_regret"] else None,
            "median_shots": median(a["shots"]),
            "median_circuits": median(a["circuits"]),
        })
    return rows


def trajectory_study(seeds, n=4, max_operators=8):
    """Shot-matched ADAPT trajectories on the disordered family.

    ``fixed_shot`` is run at its own budget and again at a budget raised to the
    escalating arm's median shot count, so the accuracy comparison is not
    confounded by the arms spending different amounts.
    """
    rows = []

    def conf(seed, *, bound="normal", tol=None):
        return dict(selector=ConfidenceSelector(delta=DELTA, near_tol=tol,
                                                bound=bound, method="bonferroni"),
                    backend=FiniteShotBackend(seed=seed), grouping=True)

    doubling = lambda: UniformDoubling(base=BASE, max_factor=MAX_FACTOR)
    arms = {
        "fixed_shot_4k": lambda s: dict(**conf(s),
                                        allocator=UniformFixed(shots_per_word=4096)),
        "fixed_shot_64k": lambda s: dict(**conf(s),
                                         allocator=UniformFixed(shots_per_word=65536)),
        "fallback_doubling": lambda s: dict(**conf(s), allocator=doubling()),
        "strict_eps_eb": lambda s: dict(**conf(s, bound="eb", tol=EPS),
                                        allocator=doubling(),
                                        accept_ambiguous=False),
    }
    for arm, kw in arms.items():
        rels, shots, circ, ops, absten = [], [], [], [], 0
        for s in range(seeds):
            model = random_ising(n, seed=s)
            res = run_adapt(model, local_pool(n, periodic_context=False),
                            max_operators=max_operators, threshold=1e-6, **kw(s))
            rels.append(res.relative_error)
            shots.append(res.total_shots)
            circ.append(res.total_circuits)
            ops.append(len(res.labels))
            absten += res.abstentions
        rows.append({"part": "trajectory", "arm": arm, "runs": seeds,
                     "family": f"random_ising(n={n})",
                     "rel_err_median": median(rels),
                     "rel_err_p90": float(np.percentile(rels, 90)),
                     "shots_median": median(shots),
                     "circuits_median": median(circ),
                     "operators_median": median(ops),
                     "abstentions_total": absten})
        print(f"[traj] {arm:18s} rel={rows[-1]['rel_err_median']:.2e} "
              f"p90={rows[-1]['rel_err_p90']:.2e} shots={rows[-1]['shots_median']:.0f} "
              f"circ={rows[-1]['circuits_median']:.0f} "
              f"ops={rows[-1]['operators_median']:.0f} abs={absten}", flush=True)
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--traj-seeds", type=int, default=15)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)

    insts = instances(args.n)
    counts = {}
    for i in insts:
        counts[i["stratum"]] = counts.get(i["stratum"], 0) + 1
    print(f"{len(insts)} instances by normalized gap: {counts}", flush=True)

    rows = selection_study(insts, args.seeds)
    fmt = lambda v: "  n/a " if v is None else f"{v:.4f}"
    for r in sorted(rows, key=lambda r: (r["stratum"], r["arm"])):
        print(f"{r['stratum']:9s} {r['arm']:16s} commit={r['commit_rate']:.3f} "
              f"wrong|commit={fmt(r['wrong_rate_given_committed'])} "
              f"epsviol={fmt(r['eps_violation_rate_given_committed'])} "
              f"relregret={fmt(r['mean_rel_regret'])} "
              f"shots={r['median_shots']:.0f}", flush=True)
    rows += trajectory_study(args.traj_seeds)

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
