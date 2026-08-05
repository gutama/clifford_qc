"""Repaired chemistry analysis (PRA revision).

Three additions the referee asked for:

1. Infinite-shot proxy analysis. At the Hartree-Fock reference of each
   molecule/geometry we rank candidates by the exact commutator gradient and
   by the FAST-inspired determinant-population proxy evaluated at N -> infinity
   (exact populations). If the proxy's top pick has a small true gradient, the
   proxy failure is intrinsic to the population signal, not shot noise.

2. Optional strict H4. A grouped empirical-Bernstein selector allocates a
   trajectory-wide error budget across the requested operator slots and
   abstains unless the exact best arm is resolved. This may stop immediately;
   that is a valid certified outcome. It is opt-in because n=8 finite-shot
   simulation is exceptionally expensive.

3. Geometry sweep. The infinite-shot diagnostic is run along the H4 bond
   length, mapping where the proxy's ranking breaks down as correlation grows.

Run: python benchmarks/run_chemistry_repair.py --out results.jsonl
Requires the ``chemistry`` extras.
"""

from __future__ import annotations

import argparse
import json
import time

from clifford_qc.matrix import exact_ground
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import CommutatorBank, UniformDoubling
from clifford_qc.algorithms import ConfidenceSelector, FastInspiredSelector, run_adapt
from clifford_qc.models.chemistry import beh2, excitation_pool, h2, h4_chain, lih
from clifford_qc.reproducibility import execution_provenance, stamp_record


def infinite_shot_ranking(model) -> dict:
    """Compare exact-gradient and infinite-shot proxy top picks at the HF ref."""
    pool = excitation_pool(model.n, model.metadata["n_electrons"])
    bank = CommutatorBank(model.hamiltonian, [op.word for op in pool])
    rho = model.reference.state()
    gexact = {j: abs(bank.exact_score(j, rho)) for j in range(len(pool))}
    true_best = max(gexact, key=gexact.get)
    gmax = gexact[true_best]
    sel = FastInspiredSelector(None, infinite_shot=True)
    proxy_best, _ = sel.pick(rho, pool, list(range(len(pool))), model.hamiltonian, bank)
    ranked = sorted(range(len(pool)), key=lambda j: -gexact[j])
    return {
        "model": model.name, "kind": "infinite_shot_ranking",
        "n": model.n, "candidates": len(pool),
        "exact_top": pool[true_best].label, "exact_top_grad": gmax,
        "proxy_top": pool[proxy_best].label,
        "proxy_top_grad": gexact[proxy_best],
        "proxy_top_grad_frac": gexact[proxy_best] / gmax if gmax else 0.0,
        "proxy_top_rank_by_gradient": ranked.index(proxy_best) + 1,
        "proxy_picks_true_argmax": proxy_best == true_best,
    }


def strict_h4_selector(max_operators: int, trajectory_delta: float):
    """Construct the selector and its predeclared per-call budget."""
    if max_operators < 1:
        raise ValueError("max_operators must be positive")
    if not (0.0 < trajectory_delta < 1.0):
        raise ValueError("trajectory_delta must be in (0, 1)")
    per_call_delta = trajectory_delta / max_operators
    return (ConfidenceSelector(delta=per_call_delta, near_tol=None, bound="eb",
                               method="bonferroni"), per_call_delta)


def certified_h4(max_operators: int = 10, seed: int = 0,
                 trajectory_delta: float = 0.05) -> dict:
    """Run a genuinely strict finite-sample H4 trajectory.

    ``trajectory_delta`` is split uniformly across the predeclared maximum
    number of ADAPT selections.  Each call uses empirical-Bernstein bounds,
    a fixed candidate family and fixed doubling endpoints; ambiguity and
    near-optimality both cause abstention rather than fallback.
    """
    selector, per_call_delta = strict_h4_selector(max_operators, trajectory_delta)
    model = h4_chain()
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    pool = excitation_pool(model.n, model.metadata["n_electrons"])
    t0 = time.perf_counter()
    res = run_adapt(model, pool, backend=FiniteShotBackend(seed=seed),
                    selector=selector,
                    allocator=UniformDoubling(base=256, max_factor=64),
                    max_operators=max_operators, threshold=1e-6, maxiter=200,
                    grouping=True, accept_ambiguous=False)
    status_counts = {}
    for record in res.records:
        key = record.status.value
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "model": model.name, "kind": "strict_certified_h4", "n": model.n,
        "final_error_mha": abs(res.energy - E0) * 1000.0,
        "operators": len(res.labels), "total_shots": res.total_shots,
        "total_circuits": res.total_circuits,
        "certification_mode": res.metadata["certification_mode"],
        "bound": res.metadata["bound"],
        "trajectory_delta": trajectory_delta,
        "per_call_delta": per_call_delta,
        "certified_selections": sum(r.certified for r in res.records),
        "status_counts": status_counts,
        "selection_records": [
            {
                "step": r.step,
                "selected_label": r.selected_label,
                "status": r.status.value,
                "certification": r.certification,
                "certified": r.certified,
                "shots_added": r.shots_added,
                "cumulative_shots": r.cumulative_shots,
            }
            for r in res.records
        ],
        "labels": list(res.labels),
        "parameters": list(res.parameters),
        "stopped_reason": res.stopped_reason,
        "abstentions": res.abstentions,
        "reached_chemical_accuracy": abs(res.energy - E0) <= 1.6e-3,
        "wall_seconds": time.perf_counter() - t0,
    }


def h4_geometry_sweep(lengths) -> list:
    rows = []
    for r in lengths:
        model = h4_chain(spacing=r)
        row = infinite_shot_ranking(model)
        row["kind"] = "geometry_sweep"
        row["bond_length"] = r
        rows.append(row)
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--h4-operators", type=int, default=10)
    parser.add_argument("--strict-h4", action="store_true",
                        help="run the expensive strict empirical-Bernstein H4 arm")
    parser.add_argument("--trajectory-delta", type=float, default=0.05,
                        help="familywise budget split across H4 selection calls")
    args = parser.parse_args(argv)

    provenance = execution_provenance()
    with open(args.out, "w") as fh:
        # 1. infinite-shot ranking at equilibrium for all molecules
        for builder in (h2, lih, beh2, h4_chain):
            row = infinite_shot_ranking(builder())
            fh.write(json.dumps(stamp_record(row, provenance)) + "\n"); fh.flush()
            print(f"[inf-shot] {row['model']:16s} proxy picks argmax? "
                  f"{row['proxy_picks_true_argmax']} "
                  f"(proxy top grad = {row['proxy_top_grad_frac']*100:.0f}% of max, "
                  f"rank {row['proxy_top_rank_by_gradient']})", flush=True)
        # 3. H4 geometry sweep (do before the slow certified run)
        for row in h4_geometry_sweep([0.7, 0.9, 1.1, 1.5, 2.0]):
            fh.write(json.dumps(stamp_record(row, provenance)) + "\n"); fh.flush()
            print(f"[geom] H4 r={row['bond_length']}: proxy argmax? "
                  f"{row['proxy_picks_true_argmax']} "
                  f"(top grad {row['proxy_top_grad_frac']*100:.0f}% of max)", flush=True)
        # 2. strict certified H4 (opt-in; slow at n=8 and may validly abstain)
        if args.strict_h4:
            row = certified_h4(max_operators=args.h4_operators,
                               trajectory_delta=args.trajectory_delta)
            fh.write(json.dumps(stamp_record(row, provenance)) + "\n"); fh.flush()
            print(f"[strict] H4: err={row['final_error_mha']:.3f} mHa "
                  f"ops={row['operators']} certified={row['certified_selections']} "
                  f"stop={row['stopped_reason']} ({row['wall_seconds']:.0f}s)",
                  flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
