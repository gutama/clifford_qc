"""Shot-ceiling sweep for strict empirical-Bernstein selection (referee item).

The headline calibration (``run_calibration.py``) runs the strict
empirical-Bernstein selector at a single capped budget, where it abstains on
96--100% of calls. That number alone cannot say *why*: an abstention may be

  (i) structural -- an exact gradient tie the exact-best rule can never
      resolve at any budget;
  (ii) budget-limited -- a real but small gap that would resolve with more
      shots; or
  (iii) bound-conservatism -- a gap large enough that a tighter interval would
      resolve it, but the union-bounded empirical-Bernstein radius does not
      shrink fast enough within the ceiling.

This experiment separates the three. It sweeps the cumulative shot ceiling
over a geometric ladder and reports, per stratum and per rule (exact-best vs
eps-best):

  - resolution and wrong-return rates, with Clopper-Pearson upper bounds
    reported both unconditionally (the event Proposition 1 controls) and
    conditional on resolving;
  - the median cumulative shots at the resolving round -- the shot *price* of
    finite-sample certification;
  - the terminal confidence radius and the leader's realized separation, so
    the residual gap between "resolved" and "not yet resolved" is visible;
  - ``eta_required``, the strongest scale-free multiplicative certificate
    ``|g_sel| >= (1 - eta) max_k |g_k|`` the same intervals support.

Three strata are built explicitly:

  ``exact_tie``   symmetric (reference) states where several operators share
                  the top gradient exactly -- exact-best can never resolve;
  ``small_gap``   unique argmax with 0 < gap < ``GAP_SMALL``;
  ``clear_gap``   unique argmax with gap >= ``GAP_SMALL``.

Run: python benchmarks/run_ceiling_sweep.py --out results.jsonl [--seeds 60]
"""

from __future__ import annotations

import argparse
import json

from clifford_qc.reproducibility import execution_provenance, stamp_record
import math
from statistics import median

import numpy as np

from clifford_qc.algorithms import ConfidenceSelector, local_pool
from clifford_qc.algorithms.adapt import _ansatz_program
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import (
    CommutatorBank, GroupedWordCache, UniformDoubling, qwc_groups,
)
from clifford_qc.models import random_ising, tfim

BASE = 256
CEILINGS = (64, 256, 1024, 4096, 16384)   # cumulative max_factor over base
EPS = 0.30
DELTA = 0.10
GAP_SMALL = 0.03
TIE_TOL = 1e-9


def _instance(name, model, rho, stratum):
    pool = local_pool(model.n, periodic_context=False)
    bank = CommutatorBank(model.hamiltonian, [op.word for op in pool])
    mags = sorted((abs(bank.exact_score(j, rho)) for j in range(len(bank))),
                  reverse=True)
    gap = mags[0] - mags[1]
    argmax = max(range(len(bank)), key=lambda j: abs(bank.exact_score(j, rho)))
    return {"name": name, "bank": bank, "rho": rho, "argmax": argmax,
            "gmax": mags[0], "gap": gap, "stratum": stratum}


def instances(n: int = 3, per_stratum: int = 6):
    """Instances grouped into exact-tie / small-gap / clear-gap strata."""
    out: list[dict] = []
    counts = {"exact_tie": 0, "small_gap": 0, "clear_gap": 0}

    # exact ties: the symmetric |+...+> reference of a uniform TFIM, where
    # reflection symmetry makes several pool gradients exactly equal.
    for h in (0.5, 1.0, 1.5, 2.0, 0.75, 1.25):
        model = tfim(n, 1.0, h)
        rho = FiniteShotBackend(seed=0).state(_ansatz_program(model, []), ())
        inst = _instance(f"tfim(h={h})@ref", model, rho, "exact_tie")
        if inst["gap"] <= TIE_TOL and counts["exact_tie"] < per_stratum:
            out.append(inst)
            counts["exact_tie"] += 1

    # displaced random-field Ising states: gap depends on the displacement, so
    # sweeping seeds and angles populates the small- and clear-gap strata.
    for seed in range(120):
        if counts["small_gap"] >= per_stratum and counts["clear_gap"] >= per_stratum:
            break
        model = random_ising(n, seed=seed)
        pool = local_pool(n, periodic_context=False)
        prog = _ansatz_program(model, [pool[seed % len(pool)],
                                       pool[(seed + 1) % len(pool)]])
        angles = (0.31, -0.47) if seed % 2 == 0 else (0.11, -0.09)
        rho = FiniteShotBackend(seed=100 + seed).state(prog, angles)
        inst = _instance(model.name, model, rho, "")
        gap = inst["gap"]
        if gap <= TIE_TOL:
            continue
        stratum = "small_gap" if gap < GAP_SMALL else "clear_gap"
        if counts[stratum] >= per_stratum:
            continue
        inst["stratum"] = stratum
        out.append(inst)
        counts[stratum] += 1
    return out


def one_selection(bank, rho, ceiling, eps, seed):
    """One strict empirical-Bernstein selection at a given cumulative ceiling."""
    cache = GroupedWordCache(bank.n)
    backend = FiniteShotBackend(seed=seed)
    candidates = list(range(len(bank)))
    groups = qwc_groups(bank.words_for(candidates))
    sampler = lambda words, plan: backend.sample_grouped_from_state(rho, groups, plan)
    selector = ConfidenceSelector(delta=DELTA, threshold=1e-4, bound="eb",
                                  near_tol=eps)
    allocator = UniformDoubling(base=BASE, max_factor=ceiling)
    idx, status, diag = selector.select(bank, cache, sampler, allocator, candidates)
    return idx, status.value, diag, cache.total_shots, cache.total_circuits


def _cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    if n == 0:
        return None
    if k == 0:                       # closed form, no SciPy needed
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    try:
        from scipy.stats import beta
    except ImportError:              # pragma: no cover - optional dependency
        return None
    return float(beta.ppf(conf, k + 1, n - k)) if k < n else 1.0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)

    insts = instances(args.n)
    by_stratum: dict[str, int] = {}
    for i in insts:
        by_stratum[i["stratum"]] = by_stratum.get(i["stratum"], 0) + 1
    print(f"{len(insts)} instances: {by_stratum}", flush=True)

    # Both rules are evaluated on every run: exact-best is correct only if the
    # returned operator is the unique argmax, eps-best if it is within eps.
    RULES = (("exact_best", None), ("eps_best", EPS))

    provenance = execution_provenance()
    with open(args.out, "w") as fh:
        for rule, eps in RULES:
            for ceiling in CEILINGS:
                agg = {s: {"runs": 0, "resolved": 0, "wrong": 0, "shots_res": [],
                           "radius": [], "eta": [], "shots": [], "circuits": []}
                       for s in ("exact_tie", "small_gap", "clear_gap")}
                for inst_idx, inst in enumerate(insts):
                    bank, rho = inst["bank"], inst["rho"]
                    a = agg[inst["stratum"]]
                    for s in range(args.seeds):
                        seed = ((inst_idx + 1) * 1_000_003 + s * 1009 + 7) % (2 ** 31)
                        idx, status, diag, shots, circ = one_selection(
                            bank, rho, ceiling, eps, seed)
                        a["runs"] += 1
                        a["shots"].append(shots)
                        a["circuits"].append(circ)
                        lo, up = diag.get("lower_bound"), diag.get("upper_bound")
                        if lo is not None and up is not None:
                            a["radius"].append((up - lo) / 2)
                        resolved = status in ("resolved_best", "resolved_eps_best")
                        if rule == "exact_best":
                            resolved = status == "resolved_best"
                        if not resolved:
                            continue
                        a["resolved"] += 1
                        a["shots_res"].append(shots)
                        if diag.get("eta_required") is not None:
                            a["eta"].append(diag["eta_required"])
                        gsel = abs(bank.exact_score(idx, rho))
                        wrong = (gsel < inst["gmax"] - EPS if rule == "eps_best"
                                 else idx != inst["argmax"])
                        # an exact tie has several true argmaxes; correctness
                        # under the exact-best rule means attaining the maximum
                        if rule == "exact_best" and gsel >= inst["gmax"] - TIE_TOL:
                            wrong = False
                        a["wrong"] += int(wrong)
                for stratum, a in agg.items():
                    if not a["runs"]:
                        continue
                    n_runs, n_res, n_wrong = a["runs"], a["resolved"], a["wrong"]
                    row = {
                        "rule": rule, "eps": eps, "delta": DELTA,
                        "ceiling_factor": ceiling,
                        "max_cumulative_shots_per_group": BASE * ceiling,
                        "decision_rounds": int(math.log2(ceiling)) + 1,
                        "stratum": stratum, "instances": by_stratum[stratum],
                        "runs": n_runs,
                        "resolved_rate": n_res / n_runs,
                        "abstention_rate": 1 - n_res / n_runs,
                        "wrong_rate_unconditional": n_wrong / n_runs,
                        "wrong_rate_given_resolved": (n_wrong / n_res if n_res else None),
                        "cp_upper_unconditional": _cp_upper(n_wrong, n_runs),
                        "cp_upper_given_resolved": _cp_upper(n_wrong, n_res),
                        "median_shots_to_resolution": (median(a["shots_res"])
                                                       if a["shots_res"] else None),
                        "median_shots": median(a["shots"]),
                        "median_circuits": median(a["circuits"]),
                        "median_terminal_radius": (median(a["radius"])
                                                   if a["radius"] else None),
                        "median_eta_required": median(a["eta"]) if a["eta"] else None,
                    }
                    fh.write(json.dumps(stamp_record(row, provenance)) + "\n")
                    fh.flush()
                    print(f"{rule:10s} ceil={ceiling:>6} {stratum:10s} "
                          f"res={row['resolved_rate']:.3f} "
                          f"wrong={row['wrong_rate_unconditional']:.4f} "
                          f"shots@res={row['median_shots_to_resolution']} "
                          f"rad={row['median_terminal_radius']:.4f} "
                          f"eta={row['median_eta_required']}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
