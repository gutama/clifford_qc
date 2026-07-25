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


def _step_rows(records, eps):
    """Per-step rows including the *scale* diagnostics for the eps certificate.

    The eps-best guarantee |g_sel| >= max_k |g_k| - eps is absolute, so its
    practical strength depends on the gradient scale at that step. We therefore
    record, alongside the raw gradients, the normalized tolerance
    ``eps_over_gmax`` (eps as a fraction of the largest gradient: <1 means the
    certificate is tight relative to the step's own scale, >1 means it is
    nominally weaker than the whole gradient range) and the realized relative
    shortfall (max|g| - |g_sel|)/max|g|, together with the selected operator's
    post-hoc exact-gradient rank.
    """
    rows = []
    for r in records:
        radius = (None if r.lower_bound is None or r.upper_bound is None
                  else (r.upper_bound - r.lower_bound) / 2)
        gsel = None if r.exact_gradient is None else abs(r.exact_gradient)
        gmax = r.exact_gradient_max
        scale = None if not gmax else eps / gmax
        shortfall = (None if gsel is None or not gmax
                     else (gmax - gsel) / gmax)
        rows.append({
            "step": r.step,
            "label": r.selected_label,
            "status": r.status.value,
            "resolution": r.resolution,
            "certification": r.certification,
            "certified": r.certified,
            "grad_selected": gsel,
            "grad_max": gmax,
            "eps_over_gmax": scale,
            "rel_shortfall": shortfall,
            "exact_rank": r.exact_rank,
            "exact_top_gap": r.exact_top_gap,
            "eta_required": r.eta_required,
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
    steps = _step_rows(res.records, eps)
    certified_steps = sum(1 for s in steps if s["label"] and s["certified"])
    appended = [s for s in steps if s["label"]]
    exact_argmax = sum(1 for s in appended if s["exact_rank"] == 1)
    return {
        "system": name, "n": model.n, "pool_size": len(pool),
        "mode": "strict_eps_best", "bound": "eb", "eps": eps,
        "trajectory_delta": delta, "delta_per_selection": per_call,
        "exact_ground_energy": E0, "final_energy": res.energy,
        "final_rel_error": abs(res.energy - E0) / max(abs(E0), 1e-12),
        "operators": len(res.labels), "certified_steps": certified_steps,
        "all_appended_certified": certified_steps == len(res.labels) and bool(res.labels),
        # how many certified picks were in fact the exact argmax, and the
        # worst-case scale of the eps certificate over the appended steps
        "appended_exact_argmax": exact_argmax,
        "max_eps_over_gmax": max((s["eps_over_gmax"] for s in appended
                                  if s["eps_over_gmax"] is not None), default=None),
        "max_rel_shortfall": max((s["rel_shortfall"] for s in appended
                                  if s["rel_shortfall"] is not None), default=None),
        "max_eta_required": max((s["eta_required"] for s in appended
                                 if s["eta_required"] is not None), default=None),
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
