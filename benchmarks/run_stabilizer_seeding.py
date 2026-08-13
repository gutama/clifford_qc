"""Paper B go/no-go experiment: does a stabilizer scaffold reduce the
non-Clifford correction needed by ADAPT-VQE?

Three arms per model:
- baseline: exact ADAPT from the model's reference state;
- stab: ADAPT from the stabilizer-Hamiltonian-approximation ground state
  (Variant B);
- cliffpt: ADAPT from the best discrete Clifford point of the depth-2 HVA
  (Variant A; recorded as 'same_as_reference' when the search returns the
  bare reference, which exhaustive enumeration shows is the case for the
  TFIM HVA).

Metrics per arm: scaffold/start energy, operators to reach relative error
1e-2 and 1e-3, final relative error at the fixed operator budget, and
optimizer evaluations — the quantities in PLAN.md section 9.8.

Run: python benchmarks/run_stabilizer_seeding.py --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.stabilizer import StimBackend
from clifford_qc.dense_reference import exact_ground
from clifford_qc.models import tfim, random_ising
from clifford_qc.algorithms import (
    bound_hva_program, clifford_point_search, hva_program, local_pool,
    run_adapt, seed_model, stabilizer_ground_program,
    stabilizer_hamiltonian_approximation,
)
from clifford_qc.reproducibility import execution_provenance, stamp_record

MAX_OPERATORS = 12
MAXITER = 150
THRESHOLDS = (1e-2, 1e-3)


def ops_to_threshold(records, E0: float, threshold: float):
    used = 0
    for rec in records:
        if rec.selected_label is None or rec.energy is None:
            continue
        used += 1 + len(rec.layer_labels)
        if abs(rec.energy - E0) / abs(E0) <= threshold:
            return used
    return None


def run_arm(model, arm: str, E0: float) -> dict:
    pool = local_pool(model.n, periodic_context=model.metadata.get("periodic", False))
    start = ExactMVBackend().expectation(model.reference, model.hamiltonian, ())
    t0 = time.perf_counter()
    res = run_adapt(model, pool, max_operators=MAX_OPERATORS, maxiter=MAXITER,
                    allow_repeats=True)
    row = {
        "arm": arm, "start_energy": start, "final_energy": res.energy,
        "relative_error": res.relative_error, "operators": len(res.labels),
        "optimizer_evaluations": res.optimizer_evaluations,
        "support_peak": res.support_peak, "stopped_reason": res.stopped_reason,
        "wall_seconds": time.perf_counter() - t0,
    }
    for threshold in THRESHOLDS:
        row[f"ops_to_{threshold:g}"] = ops_to_threshold(res.records, E0, threshold)
    return row


def run_family(model) -> list[dict]:
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    rows = []

    rows.append({"model": model.name, "exact_ground_energy": E0,
                 **run_arm(model, "baseline", E0)})

    approx = stabilizer_hamiltonian_approximation(model.hamiltonian)
    stab_model = seed_model(model, stabilizer_ground_program(approx), "stab")
    rows.append({"model": model.name, "exact_ground_energy": E0,
                 "scaffold_energy": approx.scaffold_energy,
                 "scaffold_generators": len(approx.generators),
                 "scaffold_excluded": len(approx.excluded),
                 **run_arm(stab_model, "stab", E0)})

    search = clifford_point_search(hva_program(model, 2), model.hamiltonian,
                                   StimBackend(), seed=0, restarts=8)
    if all(v == 0.0 for v in search.values):
        rows.append({"model": model.name, "exact_ground_energy": E0,
                     "arm": "cliffpt", "note": "same_as_reference",
                     "search_energy": search.energy,
                     "search_evaluations": search.evaluations})
    else:
        cliff_model = seed_model(model, bound_hva_program(model, 2, search.values),
                                 "cliffpt")
        rows.append({"model": model.name, "exact_ground_energy": E0,
                     "search_energy": search.energy,
                     "search_evaluations": search.evaluations,
                     **run_arm(cliff_model, "cliffpt", E0)})
    return rows


def large_n_scaffold_demo(n: int = 24) -> list[dict]:
    """Stabilizer-only scaffolds at qubit counts far beyond exact methods:
    scaffold energies and search cost via stim alone."""
    rows = []
    backend = StimBackend()
    for h in (0.5, 1.5):
        model = tfim(n, 1.0, h)
        ref_energy = backend.expectation(model.reference, model.hamiltonian, ())
        t0 = time.perf_counter()
        approx = stabilizer_hamiltonian_approximation(model.hamiltonian)
        stab_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        search = clifford_point_search(hva_program(model, 2), model.hamiltonian,
                                       backend, seed=0, restarts=4)
        rows.append({
            "model": model.name, "n": n, "reference_energy": ref_energy,
            "scaffold_energy": approx.scaffold_energy,
            "stab_seconds": stab_seconds,
            "cliffpt_energy": search.energy,
            "cliffpt_evaluations": search.evaluations,
            "cliffpt_seconds": time.perf_counter() - t0,
        })
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--disorder-seeds", type=int, default=5)
    parser.add_argument("--large-n", type=int, default=24)
    args = parser.parse_args(argv)

    models = [tfim(6, 1.0, h) for h in (0.5, 1.0, 1.5)]
    models.append(tfim(6, 1.0, 1.0, periodic=True))
    models.extend(random_ising(6, seed=s) for s in range(args.disorder_seeds))

    provenance = execution_provenance()
    with open(args.out, "w") as fh:
        for model in models:
            for row in run_family(model):
                fh.write(json.dumps(stamp_record(row, provenance)) + "\n")
                fh.flush()
                print(f"{row['model']:42s} {row['arm']:8s} "
                      f"rel={row.get('relative_error', float('nan')) or float('nan'):.2e} "
                      f"ops@1e-2={row.get('ops_to_0.01')} "
                      f"evals={row.get('optimizer_evaluations')} "
                      f"{row.get('note', '')}", flush=True)
        for row in large_n_scaffold_demo(args.large_n):
            row["arm"] = "large_n_demo"
            fh.write(json.dumps(stamp_record(row, provenance)) + "\n")
            print(f"{row['model']:42s} demo     ref={row['reference_energy']:.1f} "
                  f"stab={row['scaffold_energy']:.1f} "
                  f"cliffpt={row['cliffpt_energy']:.1f} "
                  f"({row['cliffpt_seconds']:.1f}s)", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
