"""A-CASE grown around an ADAPT-VQE reference instead of a bare determinant.

``clifford_qc.subspace.adapt_warm_start`` has been in the package since the
subspace work landed, with a test and an example, and no benchmark.  It is the
one place where the two methods compose rather than compete: ADAPT-VQE produces
an optimized state, A-CASE takes that state as its single reference rho, and the
subspace is grown around something that already carries correlation.

That gap mattered, because the H4 rung's headline negative result -- adaptive
A-CASE misses chemical accuracy at the predeclared nine-vector budget -- has an
obvious candidate explanation the paper never tested: the reference is a single
determinant, so every correlation effect has to be paid for out of the eight
adaptive additions.

The arms are matched on the quantity the paper's ledger is about.  Every row
uses the same nine-vector budget, the same candidate pool, and the same frozen
FCIDUMP; only rho changes.  The ADAPT stage's own energy is reported beside the
A-CASE result so the improvement cannot be mistaken for ADAPT having done the
work: a two-operator ADAPT state is far *worse* than cold A-CASE on its own.

ADAPT's operator pool is built here from the odd-Y words of the determinant
excitations rather than through ``models.chemistry.excitation_pool``, which
imports OpenFermion.  That keeps the rung's NumPy-only property, which the
manuscript claims for exactly this benchmark.

    python benchmarks/run_warm_start.py
    python benchmarks/run_warm_start.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from clifford_qc.algorithms.pools import PoolOperator, is_odd_y
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.ir import PauliWord
from clifford_qc.models import fcidump_model
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

ADAPTIVE_ADDITIONS = 8          # the paper's predeclared budget, unchanged
WARM_OPERATORS = (2, 4, 6)      # ADAPT depth used only to prepare rho
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


def word_pool(model, candidates) -> list[PoolOperator]:
    """Qubit-ADAPT pool: the odd-Y words of the determinant excitations.

    The same construction ``run_acase_ladder.word_pool`` uses for its fermionic
    lattice rungs, inlined so this benchmark does not need chemistry extras.
    """
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
    # Wall-clock timings stay on stdout and out of the record: CI gates this
    # file bit-for-bit, and a duration is the one field guaranteed to differ
    # between two correct runs.
    print(f"cold: {cold_row['error_millihartree']:.6f} mHa "
          f"M={cold_row['basis_size']} W={cold_row['word_universe']} "
          f"[{time.perf_counter() - started:.1f}s]", flush=True)

    warm_rows = []
    for operators in WARM_OPERATORS:
        started = time.perf_counter()
        rho, adapt = adapt_warm_start(
            model, pool, max_operators=operators, compute_exact_reference=False)
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
            "adapt_chemical_accuracy": bool(
                abs(adapt_error) < CHEMICAL_ACCURACY_HARTREE),
        })
        warm_rows.append(row)
        print(f"warm k={operators}: {row['error_millihartree']:.6f} mHa "
              f"M={row['basis_size']} kappa={row['condition_number']:.4g} "
              f"W={row['word_universe']} (ADAPT alone "
              f"{row['adapt_error_millihartree']:.6f} mHa) "
              f"[{time.perf_counter() - started:.1f}s]", flush=True)

    return {
        "schema": "clifford_qc.acase_warm_start.v1",
        "evidence": {
            "energies": "exact",
            "resource_counts": "exact",
            "quantum_advantage_claim": False,
        },
        "input": {
            "path": str(FCIDUMP.relative_to(ROOT.parent)),
            "sha256": model.metadata["source_sha256"],
            "provenance_schema": provenance["schema"],
            "generator": provenance["generator"],
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
        "cold": cold_row,
        "warm": warm_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    record = build_record()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
