"""Strengthened baseline comparison (PRA revision).

Compares, on a common set of n=4 selection problems at a matched budget:

- exact          : exact commutator-gradient ADAPT (selection upper bound);
- random         : uniform-random selection (zero-measurement floor);
- fixed_shot     : one fixed-budget round, then decide (no escalation);
- shared_only    : confidence selection with the global commutator-bank
                   reuse but no QWC grouping (one circuit per unique word);
- shared_grouped : bank reuse + QWC grouping (this work);
- variance_reuse : bank reuse + grouping + variance-proportional allocation
                   (the reuse-and-variance-allocation family, cf. Ikhtiarudin
                   et al.);
- strict         : shared_grouped, abstain on ambiguity (normal-bound,
                   asymptotically resolved rather than finite-sample certified);
- fallback       : shared_grouped, accept the empirical leader when ambiguous.

It also reports the per-step measurement-plan size for three reuse regimes,
computed exactly from the commutator bank (no sampling):

- naive   = sum_j |supp(G_j)|     (measure every candidate independently);
- shared  = |union_j supp(G_j)|   (each unique word once);
- grouped = number of QWC groups of the shared word set.

Run: python benchmarks/run_baselines.py --out results.jsonl [--seeds 15]
"""

from __future__ import annotations

import argparse
import json
from statistics import median

from clifford_qc.matrix import exact_ground
from clifford_qc.models import tfim, random_ising
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import (
    CommutatorBank, UniformFixed, UniformDoubling, VarianceProportional, qwc_groups,
)
from clifford_qc.algorithms import ConfidenceSelector, RandomSelector, local_pool, run_adapt

MAX_OPERATORS = 8


def plan_sizes(model):
    """Exact naive / shared / grouped measurement-plan sizes at the reference."""
    pool = local_pool(model.n, periodic_context=False)
    bank = CommutatorBank(model.hamiltonian, [op.word for op in pool])
    naive = sum(len(row) for row in bank.coeffs)
    shared = len(bank.words)
    grouped = len(qwc_groups(bank.words))
    return {"naive_words": naive, "shared_words": shared, "qwc_groups": grouped,
            "shared_reuse": naive / shared, "grouped_reuse": naive / grouped}


def arm_kwargs(arm, seed):
    # Preserve the published normal-bound trajectories explicitly. Sidak is
    # an asymptotic heuristic here; finite-sample certification uses the
    # Bonferroni empirical-Bernstein calibration instead.
    conf = dict(selector=ConfidenceSelector(delta=0.05, near_tol=0.05,
                                             method="sidak"),
                backend=FiniteShotBackend(seed=seed), grouping=True)
    if arm == "exact":
        return {}
    if arm == "random":
        return dict(selector=RandomSelector(seed=seed))
    if arm == "fixed_shot":
        return dict(selector=ConfidenceSelector(delta=0.05, near_tol=0.05,
                                                method="sidak"),
                    backend=FiniteShotBackend(seed=seed), grouping=True,
                    allocator=UniformFixed(shots_per_word=4096))
    if arm == "shared_only":
        return dict(selector=ConfidenceSelector(delta=0.05, near_tol=0.05,
                                                method="sidak"),
                    backend=FiniteShotBackend(seed=seed), grouping=False,
                    allocator=UniformDoubling(base=256, max_factor=64))
    if arm == "shared_grouped":
        return dict(**conf, allocator=UniformDoubling(base=256, max_factor=64))
    if arm == "variance_reuse":
        return dict(**conf, allocator=VarianceProportional(round_budget=4096,
                                                           growth=2.0, max_rounds=7))
    if arm == "strict":
        return dict(**conf, allocator=UniformDoubling(base=256, max_factor=64),
                    accept_ambiguous=False)
    if arm == "fallback":
        return dict(**conf, allocator=UniformDoubling(base=256, max_factor=64),
                    accept_ambiguous=True)
    raise ValueError(arm)


ARMS = ["exact", "random", "fixed_shot", "shared_only", "shared_grouped",
        "variance_reuse", "strict", "fallback"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=15)
    args = parser.parse_args(argv)

    families = [("tfim_crit", lambda s: tfim(4, 1.0, 1.0)),
                ("random_ising", lambda s: random_ising(4, seed=s))]
    with open(args.out, "w") as fh:
        for fam, build in families:
            E0 = exact_ground(build(0).hamiltonian.to_mv())[0]
            sizes = plan_sizes(build(0))
            for arm in ARMS:
                rels, shots, circ, absten = [], [], [], 0
                seeds = (0,) if arm == "exact" else tuple(range(args.seeds))
                for s in seeds:
                    m = build(s)
                    e0 = exact_ground(m.hamiltonian.to_mv())[0] if fam == "random_ising" else E0
                    res = run_adapt(m, local_pool(4, periodic_context=False),
                                    max_operators=MAX_OPERATORS, threshold=1e-6,
                                    **arm_kwargs(arm, s))
                    rels.append(res.relative_error)
                    shots.append(res.total_shots)
                    circ.append(res.total_circuits)
                    absten += res.abstentions
                row = {"family": fam, "arm": arm, "runs": len(seeds),
                       "rel_err_median": median(rels),
                       "shots_median": median(shots),
                       "circuits_median": median(circ),
                       "abstentions_total": absten, **sizes}
                row["selection_metadata"] = res.metadata
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"{fam:14s} {arm:15s} rel={row['rel_err_median']:.2e} "
                      f"shots={row['shots_median']:.0f} circ={row['circuits_median']:.0f} "
                      f"abstain={absten}", flush=True)
        print(f"reuse accounting (per step): naive={sizes['naive_words']} "
              f"shared={sizes['shared_words']} grouped={sizes['qwc_groups']} "
              f"(shared {sizes['shared_reuse']:.1f}x, grouped {sizes['grouped_reuse']:.1f}x)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
