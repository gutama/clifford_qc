"""Complete molecular simulation pipeline for LiH, BeH2, HF, and H2O using PySCF FCIDUMP and clifford_qc A-CASE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf, tools

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.observables import (double_occupancy, magnetization,
                                             total_spin_squared)
from clifford_qc.subspace import (MatrixElementBank, determinant_excitations,
                                   identity_generator, lehmann_spectrum,
                                   occupied_spin_orbitals, run_acase,
                                   solve_subspace, static_susceptibility)


MOLECULES = {
    "lih": {
        "name": "Lithium Hydride (LiH)",
        "atom": "Li 0 0 0; H 0 0 1.595",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
    "beh2": {
        "name": "Beryllium Hydride (BeH2)",
        "atom": "Be 0 0 0; H 0 0 1.326; H 0 0 -1.326",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
    "hf": {
        "name": "Hydrogen Fluoride (HF)",
        "atom": "F 0 0 0; H 0 0 0.917",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
    "h2o": {
        "name": "Water (H2O)",
        "atom": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
}


def run_pipeline(max_candidates: int | None = 25, max_subspace: int = 15) -> None:
    out_dir = Path("molecular_results")
    out_dir.mkdir(exist_ok=True)

    summary_records = {}

    print("==================================================================")
    print("STARTING MOLECULAR SIMULATION PIPELINE (PySCF -> FCIDUMP -> A-CASE)")
    print("==================================================================")

    for key, spec in MOLECULES.items():
        t0 = time.time()
        print(f"\n---> Processing {spec['name']} ({key.upper()})...")

        # 1. PySCF RHF calculation
        mol = gto.M(
            atom=spec["atom"],
            basis=spec["basis"],
            charge=spec["charge"],
            spin=spec["spin"],
            verbose=0,
        )
        mf = scf.RHF(mol).run()
        e_rhf = float(mf.e_tot)
        print(f"  PySCF RHF Energy: {e_rhf:+.9f} Ha")

        # 2. Dump FCIDUMP
        fcidump_path = out_dir / f"{key}.fcidump"
        tools.fcidump.from_scf(mf, str(fcidump_path))
        print(f"  Exported FCIDUMP: {fcidump_path}")

        # 3. Ingest FCIDUMP with clifford_qc
        model = fcidump_model(fcidump_path)
        model.metadata["kind"] = "fermionic_lattice"
        model.metadata["sites"] = model.n // 2
        model.metadata["n_orbitals"] = 1
        n_electrons = model.metadata["n_electrons"]
        sz = model.metadata["sz"]
        occupied = occupied_spin_orbitals(model)
        rho0 = ExactMVBackend().state(model.reference, ())
        all_candidates = determinant_excitations(model.n, occupied, max_rank=2)
        candidates = (
            all_candidates[:max_candidates]
            if max_candidates and max_candidates > 0
            else all_candidates
        )

        print(f"  Qubits: {model.n}, Electrons: {n_electrons}, Sz: {sz:+g}")
        print(f"  Candidate SD Excitations: {len(candidates)} (total available: {len(all_candidates)})")

        # 4. Sector-Exact FCI Reference via SectorStatevectorBackend
        exact_energies, _ = SectorStatevectorBackend(
            model.n, n_electrons=n_electrons, sz=sz
        ).ground_state(model.hamiltonian, k=4, method="dense")
        e_exact = float(exact_energies[0])
        print(f"  Exact Sector FCI E0: {e_exact:+.9f} Ha")

        # 5. Adaptive A-CASE Subspace Eigensolver
        adaptive = run_acase(
            rho0,
            model.hamiltonian,
            candidates,
            max_size=min(max_subspace, len(candidates)),
            leakage_tol=1e-10,
            exact_ground_energy=e_exact,
        )
        e_acase = float(adaptive.result.ground_energy)
        err_ha = e_acase - e_exact
        err_mha = err_ha * 1000.0
        print(f"  Adaptive A-CASE E0: {e_acase:+.9f} Ha (Error: {err_mha:+.4f} mHa, M={len(adaptive.labels)})")

        # 6. Complete SD Coordinate Subspace Eigensolver (Chemical Accuracy Check)
        t_sd_start = time.time()
        sd_spectrum = solve_subspace(
            rho0, model.hamiltonian, [identity_generator(model.n), *all_candidates]
        )
        t_sd_elapsed = time.time() - t_sd_start
        e_sd = float(sd_spectrum.ground_energy)
        err_sd_ha = e_sd - e_exact
        err_sd_mha = err_sd_ha * 1000.0
        chem_acc = err_sd_mha <= 1.5936
        print(f"  Complete SD Subspace E0: {e_sd:+.9f} Ha (Error: {err_sd_mha:+.4f} mHa, M={len(all_candidates)+1})")
        print(f"  SD Subspace Wall-Clock Time: {t_sd_elapsed:.1f}s ({t_sd_elapsed/60:.1f} min)")
        print(f"  Chemical Accuracy (< 1.6 mHa): {'ACHIEVED [YES]' if chem_acc else 'NO'}")

        # 7. Response Analysis & Observables (on adaptive result)
        spectrum = adaptive.result

        double_occ = float(spectrum.expectation(double_occupancy(model)))
        spin2 = float(spectrum.expectation(total_spin_squared(model)))

        # Lehmann spectrum for Mz response
        mz_obs = magnetization(model, 0, axis="z")
        lehmann_lines = lehmann_spectrum(spectrum, mz_obs, min_weight=1e-12)
        chi0 = float(static_susceptibility(lehmann_lines))

        t_elapsed = time.time() - t0

        # Coefficients of ground state
        ritz_coeffs = {}
        for label, coeff in zip(adaptive.labels, spectrum.ritz_vector(0)):
            if abs(coeff) > 1e-6:
                ritz_coeffs[label] = {"real": float(coeff.real), "imag": float(coeff.imag)}

        # Format Lehmann response lines
        lehmann_data = [
            {
                "final_state": int(line.final_state),
                "excitation_energy_ha": float(line.excitation_energy),
                "weight": float(line.weight),
            }
            for line in lehmann_lines
        ]

        record = {
            "key": key,
            "name": spec["name"],
            "qubits": model.n,
            "n_electrons": n_electrons,
            "sz": sz,
            "e_rhf": e_rhf,
            "e_exact_fci": e_exact,
            "e_acase_adaptive": e_acase,
            "error_adaptive_mha": err_mha,
            "e_sd_complete": e_sd,
            "error_sd_mha": err_sd_mha,
            "chemical_accuracy_achieved": chem_acc,
            "subspace_size_m": len(adaptive.labels),
            "complete_sd_candidates": len(all_candidates),
            "effective_rank": int(spectrum.effective_rank),
            "condition_number": float(spectrum.condition_number),
            "word_universe": int(adaptive.resources.get("word_universe", 0)),
            "double_occupancy": double_occ,
            "total_spin_squared": spin2,
            "static_susceptibility_chi0": chi0,
            "elapsed_seconds": t_elapsed,
            "sd_subspace_seconds": t_sd_elapsed,
            "ground_state_coefficients": ritz_coeffs,
            "lehmann_response": lehmann_data,
        }

        # Write per-molecule result JSON
        with open(out_dir / f"{key}_results.json", "w") as f:
            json.dump(record, f, indent=2)

        summary_records[key] = record

    # Master summary JSON
    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\n==================================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results saved in: {out_dir.resolve()}")
    print("==================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run molecular benchmark pipeline")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=25,
        help="Max candidate excitations to evaluate (0 for full candidate pool)",
    )
    parser.add_argument(
        "--max-subspace",
        type=int,
        default=15,
        help="Max adaptive subspace basis size M",
    )
    args = parser.parse_args()
    run_pipeline(
        max_candidates=args.max_candidates, max_subspace=args.max_subspace
    )


if __name__ == "__main__":
    main()
