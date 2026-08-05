"""Chemistry benchmark (backlog 17): H2, H4 chain, LiH and BeH2 active
spaces, measured against chemical accuracy (1.6 mHa) of the active-space
FCI energy.

Arms:
- exact: exact commutator-gradient ADAPT (upper bound on selection quality);
- confidence: confidence-gated finite-shot selection with QWC grouping
  (this work);
- fast: the FAST-inspired determinant-population proxy (one Z-basis
  circuit per step) — the natural chemistry competitor;
- random: uniform-random selection.

The qubit-ADAPT pool is the odd-Y word set of JW singles and doubles.
Requires the ``chemistry`` extras (openfermion, pyscf, openfermionpyscf).

Run: python benchmarks/run_chemistry.py --out results.jsonl [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import time

from clifford_qc.matrix import exact_ground
from clifford_qc.backends import ExactMVBackend, FiniteShotBackend
from clifford_qc.diagnostics import fermionic_sector_diagnostics
from clifford_qc.ir import Parameter, Program, Rotor
from clifford_qc.measurement import UniformDoubling
from clifford_qc.algorithms import (
    ConfidenceSelector, FastInspiredSelector, RandomSelector, run_adapt,
)
from clifford_qc.models.chemistry import beh2, excitation_pool, h2, h4_chain, lih
from clifford_qc.reproducibility import execution_provenance, stamp_record

CHEMICAL_ACCURACY = 1.6e-3  # Hartree
MAX_OPERATORS = 12
FAST_SHOTS = 4096


def ops_and_shots_to_accuracy(records, E0: float):
    used = 0
    for rec in records:
        if rec.selected_label is None or rec.energy is None:
            continue
        used += 1 + len(rec.layer_labels)
        if abs(rec.energy - E0) <= CHEMICAL_ACCURACY:
            return used, rec.cumulative_shots, rec.circuits_executed
    return None, None, None


def _final_program(model, pool, labels) -> Program:
    """Reconstruct the optimized word-level ADAPT program for diagnostics."""
    by_label = {op.label: op for op in pool}
    prog = Program(model.n)
    for op in model.reference.ops:
        prog.append(op)
    for k, label in enumerate(labels):
        prog.append(Rotor(by_label[label].word, Parameter(f"t{k}")))
    return prog


def _trajectory(records, E0: float) -> list[dict]:
    out = []
    for rec in records:
        out.append({
            "step": rec.step,
            "selected_label": rec.selected_label,
            "status": rec.status.value,
            "certification": rec.certification,
            "certified": rec.certified,
            "estimate": rec.estimate,
            "lower_bound": rec.lower_bound,
            "upper_bound": rec.upper_bound,
            "exact_gradient": rec.exact_gradient,
            "exact_gradient_max": rec.exact_gradient_max,
            "energy_ha": rec.energy,
            "error_mha": None if rec.energy is None else abs(rec.energy - E0) * 1000.0,
            "shots_added": rec.shots_added,
            "cumulative_shots": rec.cumulative_shots,
            "circuits_executed": rec.circuits_executed,
            "active_candidates": rec.active_candidates,
        })
    return out


def run_arm(model, pool, arm: str, seed: int, E0: float) -> dict:
    kwargs = dict(max_operators=MAX_OPERATORS, threshold=1e-6, maxiter=200)
    if arm == "exact":
        pass
    elif arm == "fast":
        kwargs["selector"] = FastInspiredSelector(shots=FAST_SHOTS, seed=seed)
    elif arm == "random":
        kwargs["selector"] = RandomSelector(seed=seed)
    elif arm == "confidence":
        kwargs.update(selector=ConfidenceSelector(delta=0.05, near_tol=0.05,
                                                   method="sidak"),
                      allocator=UniformDoubling(base=256, max_factor=64),
                      backend=FiniteShotBackend(seed=seed), grouping=True)
    else:
        raise ValueError(arm)
    t0 = time.perf_counter()
    res = run_adapt(model, pool, **kwargs)
    ops_acc, shots_acc, circuits_acc = ops_and_shots_to_accuracy(res.records, E0)
    final_rho = ExactMVBackend().state(
        _final_program(model, pool, res.labels), res.parameters)
    sector = fermionic_sector_diagnostics(
        final_rho, model.metadata["n_electrons"], target_sz=0.0)
    return {
        "model": model.name, "arm": arm, "seed": seed, "n": model.n,
        "pool_size": len(pool),
        "hf_energy": model.metadata["hf_energy"],
        "active_space_fci": E0,
        "final_error_mha": abs(res.energy - E0) * 1000.0,
        "chemical_accuracy_reached": ops_acc is not None,
        "ops_to_accuracy": ops_acc, "shots_to_accuracy": shots_acc,
        "circuits_to_accuracy": circuits_acc,
        "operators": len(res.labels), "total_shots": res.total_shots,
        "total_circuits": res.total_circuits,
        "optimizer_evaluations": res.optimizer_evaluations,
        "stopped_reason": res.stopped_reason,
        "labels": list(res.labels),
        "parameters": list(res.parameters),
        "trajectory": _trajectory(res.records, E0),
        "selection_metadata": res.metadata,
        "pool_semantics": "individual_pauli_words_from_conserving_generators",
        **sector,
        "wall_seconds": time.perf_counter() - t0,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--skip-confidence-above", type=int, default=6,
                        help="skip the confidence arm for n above this "
                             "(support growth makes it slow at n=8 here)")
    args = parser.parse_args(argv)

    provenance = execution_provenance()
    with open(args.out, "w") as fh:
        for model in (h2(), lih(), beh2(), h4_chain()):
            E0, _ = exact_ground(model.hamiltonian.to_mv())
            pool = excitation_pool(model.n, model.metadata["n_electrons"])
            for arm in ("exact", "fast", "confidence", "random"):
                if arm == "confidence" and model.n > args.skip_confidence_above:
                    continue
                seeds = (0,) if arm == "exact" else tuple(range(args.seeds))
                for seed in seeds:
                    row = run_arm(model, pool, arm, seed, E0)
                    fh.write(json.dumps(stamp_record(row, provenance)) + "\n")
                    fh.flush()
                    print(f"{row['model']:18s} {arm:10s} seed={seed} "
                          f"err={row['final_error_mha']:.4f} mHa "
                          f"ops@acc={row['ops_to_accuracy']} "
                          f"shots@acc={row['shots_to_accuracy']} "
                          f"({row['wall_seconds']:.0f}s)", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
