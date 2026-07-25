"""Complete certified eps-best ADAPT trajectories (PRA revision, Priority 1-2).

A strict exact-best selector abstains forever at symmetric ansatz states,
where several pool operators share the top gradient (an exact tie the
best-arm rule cannot resolve at any budget). The certified eps-best rule
resolves such ties: it commits the empirical leader once its lower bound
clears every rival's upper bound up to a tolerance ``eps``, certifying (on
the 1-delta interval event) that the selected operator's gradient is within
``eps`` of the maximum. With the empirical-Bernstein bound each such
resolution is a genuine finite-sample certificate.

This script runs complete strict eps-best trajectories -- so every appended
operator is finite-sample certified -- on spin systems and molecules, and
records each step's status, resolution kind, gradient, confidence radius,
cumulative shots, and energy. The trajectory-level error budget is split
over the operator budget by a union bound (delta_per_call = delta/K).

Run: python benchmarks/run_certified_trajectories.py --out results.jsonl
Requires the ``chemistry`` extras for the molecular systems.
"""
from __future__ import annotations

import argparse
import json

from clifford_qc.models import tfim, random_ising
from clifford_qc.algorithms import ConfidenceSelector, local_pool, run_adapt
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import UniformDoubling
from clifford_qc.matrix import exact_ground

TRAJ_DELTA = 0.10
EPS = 0.30


def _step_rows(records):
    rows = []
    for r in records:
        radius = (None if r.lower_bound is None or r.upper_bound is None
                  else (r.upper_bound - r.lower_bound) / 2)
        rows.append({
            "step": r.step,
            "label": r.selected_label,
            "status": r.status.value,
            "resolution": r.resolution,
            "certification": r.certification,
            "certified": r.certified,
            "grad_selected": (None if r.exact_gradient is None
                              else abs(r.exact_gradient)),
            "grad_max": r.exact_gradient_max,
            "radius": radius,
            "cumulative_shots": r.cumulative_shots,
            "energy": r.energy,
        })
    return rows


def run_traj(name, model, pool, *, eps=EPS, base=1024, max_factor=128,
             max_ops=6, seed=0, delta=TRAJ_DELTA):
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    per_call = delta / max_ops
    selector = ConfidenceSelector(delta=per_call, bound="eb",
                                  method="bonferroni", near_tol=eps)
    res = run_adapt(model, pool, backend=FiniteShotBackend(seed=seed),
                    allocator=UniformDoubling(base=base, max_factor=max_factor),
                    max_operators=max_ops, grouping=True, threshold=1e-4,
                    selector=selector, accept_ambiguous=False)
    steps = _step_rows(res.records)
    certified_steps = sum(1 for s in steps if s["label"] and s["certified"])
    return {
        "system": name, "n": model.n, "pool_size": len(pool),
        "mode": "strict_eps_best", "bound": "eb", "eps": eps,
        "trajectory_delta": delta, "delta_per_selection": per_call,
        "exact_ground_energy": E0, "final_energy": res.energy,
        "final_rel_error": abs(res.energy - E0) / max(abs(E0), 1e-12),
        "operators": len(res.labels), "certified_steps": certified_steps,
        "all_appended_certified": certified_steps == len(res.labels) and bool(res.labels),
        "total_shots": res.total_shots, "abstentions": res.abstentions,
        "stopped_reason": res.stopped_reason, "trajectory": steps,
    }


def build_systems():
    systems = [
        ("tfim_n4_h1", lambda: tfim(4, 1.0, 1.0, periodic=False),
         lambda m: local_pool(4, periodic_context=False), dict(base=512, max_factor=64)),
        ("random_ising_n4", lambda: random_ising(4, seed=1),
         lambda m: local_pool(4, periodic_context=False), dict(base=1024, max_factor=128)),
    ]
    try:
        from clifford_qc.models.chemistry import h2, lih, excitation_pool
        systems += [
            ("h2_sto3g", h2, lambda m: excitation_pool(4, 2), dict(base=512, max_factor=64, max_ops=4)),
            ("lih_2e2o", lih, lambda m: excitation_pool(4, 2), dict(base=1024, max_factor=128, max_ops=4)),
        ]
    except ImportError:
        print("[warn] chemistry extras missing; skipping molecular systems", flush=True)
    return systems


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.out, "w") as fh:
        for name, build_model, build_pool, kw in build_systems():
            model = build_model()
            row = run_traj(name, model, build_pool(model), **kw)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{row['system']:18s} ops={row['operators']} "
                  f"certified={row['certified_steps']}/{row['operators']} "
                  f"rel={row['final_rel_error']:.2e} "
                  f"shots={row['total_shots']:.2e} "
                  f"all_certified={row['all_appended_certified']}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
