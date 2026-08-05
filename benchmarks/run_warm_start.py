"""A-CASE grown around an exact-simulation ADAPT-VQE reference.

Every arm uses the same frozen FCIDUMP, A-CASE candidate family, and
nine-vector A-CASE budget.  The ADAPT stage is additional work, so the record
reports its selected rotors, pool-gradient evaluations, optimizer evaluations,
and state-preparation operator count.  Selection and optimization are exact in
this benchmark; the zero shot/circuit fields are therefore simulation labels,
not an end-to-end hardware-cost claim.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Exact ADAPT optimization contains BLAS reductions.  One thread removes their
# completion-order freedom so operator choices and resource counts reproduce;
# the record gate separately tolerates last-bit cross-BLAS energy variation.
_THREADS = os.environ.get("CLIFFORD_QC_BENCHMARK_THREADS", "1")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _THREADS

from clifford_qc.algorithms.pools import PoolOperator, is_odd_y
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.ir import PauliWord
from clifford_qc.models import fcidump_model
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import (
    adapt_warm_start,
    determinant_excitations,
    occupied_spin_orbitals,
    run_acase,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "reference_results" / "warm_start_h4.json"
FCIDUMP = ROOT / "data" / "h4_sto3g_r0.9.FCIDUMP"
PROVENANCE = ROOT / "data" / "h4_sto3g_r0.9.provenance.json"

ADAPTIVE_ADDITIONS = 8
WARM_OPERATORS = (2, 4, 6)
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


def word_pool(model, candidates) -> list[PoolOperator]:
    """Qubit-ADAPT pool: unique odd-Y words of determinant excitations."""
    pool: dict[int, PoolOperator] = {}
    for generator in candidates:
        for code in sorted(generator.mv.terms):
            word = PauliWord(model.n, code)
            if code and is_odd_y(word) and code not in pool:
                pool[code] = PoolOperator(word.label, word)
    return list(pool.values())


def _row(result, exact_energy: float) -> dict:
    subspace = result.result
    error = result.energy - exact_energy
    return {
        "basis_size": len(subspace.basis_labels),
        "labels": list(subspace.basis_labels),
        "ground_energy": result.energy,
        "error_hartree": error,
        "error_millihartree": error * 1e3,
        "chemical_accuracy": bool(abs(error) < CHEMICAL_ACCURACY_HARTREE),
        "condition_number": subspace.condition_number,
        "effective_rank": subspace.effective_rank,
        "word_universe": subspace.resources.get("word_universe"),
    }


def build_record() -> dict:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    model = fcidump_model(FCIDUMP, name=provenance["name"])
    if model.metadata["source_sha256"] != provenance["fcidump_sha256"]:
        raise ValueError("FCIDUMP digest disagrees with its provenance record")

    sector = SectorStatevectorBackend(
        model.n, model.metadata["n_electrons"], model.metadata["sz"])
    exact_energy = float(
        sector.ground_state(model.hamiltonian, k=4, method="dense")[0][0])
    if abs(exact_energy - provenance["reference_energies"]["fci"]) > 1e-10:
        raise ValueError("sector-exact energy disagrees with external FCI provenance")

    occupied = occupied_spin_orbitals(model)
    candidates = determinant_excitations(model.n, occupied, max_rank=2)
    pool = word_pool(model, candidates)

    started = time.perf_counter()
    cold = run_acase(
        ExactMVBackend().state(model.reference, ()),
        model.hamiltonian, candidates,
        max_size=ADAPTIVE_ADDITIONS, leakage_tol=1e-10,
        exact_ground_energy=exact_energy)
    cold_row = _row(cold, exact_energy)
    cold_row["reference"] = "Hartree-Fock determinant"
    print(f"cold: {cold_row['error_millihartree']:.6f} mHa "
          f"M={cold_row['basis_size']} W={cold_row['word_universe']} "
          f"[{time.perf_counter() - started:.1f}s]", flush=True)

    warm_rows = []
    for operators in WARM_OPERATORS:
        started = time.perf_counter()
        rho, adapt = adapt_warm_start(
            model, pool, max_operators=operators,
            compute_exact_reference=False)
        warm = run_acase(
            rho, model.hamiltonian, candidates,
            max_size=ADAPTIVE_ADDITIONS, leakage_tol=1e-10,
            exact_ground_energy=exact_energy)
        row = _row(warm, exact_energy)
        adapt_error = adapt.energy - exact_energy
        row.update({
            "reference": f"ADAPT-VQE state, {operators} operators",
            "adapt_operators": len(adapt.labels),
            "adapt_labels": list(adapt.labels),
            "adapt_energy": adapt.energy,
            "adapt_error_hartree": adapt_error,
            "adapt_error_millihartree": adapt_error * 1e3,
            "adapt_chemical_accuracy":
                bool(abs(adapt_error) < CHEMICAL_ACCURACY_HARTREE),
            "adapt_selection_mode": "exact statevector",
            "adapt_selection_shots": adapt.total_shots,
            "adapt_selection_circuits": adapt.total_circuits,
            "adapt_gradient_evaluations":
                sum(record.active_candidates for record in adapt.records),
            "adapt_optimizer_evaluations": adapt.optimizer_evaluations,
            "adapt_support_peak": adapt.support_peak,
            "adapt_state_preparation_operators": len(adapt.labels),
        })
        warm_rows.append(row)
        print(f"warm k={operators}: {row['error_millihartree']:.6f} mHa "
              f"M={row['basis_size']} kappa={row['condition_number']:.4g} "
              f"W={row['word_universe']} grad={row['adapt_gradient_evaluations']} "
              f"opt={row['adapt_optimizer_evaluations']} "
              f"[{time.perf_counter() - started:.1f}s]", flush=True)

    return {
        "schema": "clifford_qc.acase_warm_start.v2",
        "evidence": {
            "energies": "exact statevector simulation",
            "resource_counts": "exact algorithmic counts",
            "physical_shot_budget": "not estimated",
            "quantum_advantage_claim": False,
        },
        "input": {
            "path": str(FCIDUMP.relative_to(ROOT.parent)),
            "sha256": model.metadata["source_sha256"],
            "provenance_schema": provenance["schema"],
            "generator": provenance["generator"],
            "blas_threads": int(_THREADS),
        },
        "system": {
            "name": model.name,
            "n_qubits": model.n,
            "n_electrons": model.metadata["n_electrons"],
            "sz": model.metadata["sz"],
            "sector_dimension": 36,
            "candidate_generators": len(candidates),
            "adapt_pool_size": len(pool),
            "energy_unit": "hartree",
        },
        "reference_energy": exact_energy,
        "predeclared_additions": ADAPTIVE_ADDITIONS,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "resource_boundary": (
            "A-CASE M and W are held fixed/comparable across rows. The ADAPT "
            "rotors, gradient evaluations, and optimizer evaluations are "
            "additional hybrid costs. Exact simulation assigns no physical "
            "shot count to those evaluations."),
        "cold": cold_row,
        "warm": warm_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    record = stamp_record(build_record())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
