"""Complete molecular simulation pipeline for LiH, BeH2, HF, and H2O using PySCF FCIDUMP and clifford_qc A-CASE."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from pyscf import cc, ci, gto, scf, tools

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


def run_pipeline(
    max_candidates: int | None = 0,
    max_subspace: int | None = 0,
    target_error_mha: float = 1.5936,
) -> None:
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

        # 1b. Classical correlated baselines in the same orbital space.
        # Without these the A-CASE column has nothing to be judged against:
        # CISD is the classical method the complete singles/doubles subspace
        # reproduces by construction, and CCSD is the one a chemist would
        # actually run at this cost.
        e_cisd = float(ci.CISD(mf).run().e_tot)
        e_ccsd = float(cc.CCSD(mf).run().e_tot)
        print(f"  PySCF CISD Energy: {e_cisd:+.9f} Ha")
        print(f"  PySCF CCSD Energy: {e_ccsd:+.9f} Ha")

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

        # The number of *distinct* Pauli words in H, after the word-code keyed
        # collection the MV product does on construction. This is not the
        # A-CASE ``word_universe``, which is the union over element operators
        # ``A_i'A_j`` and ``A_i'HA_j`` and is legitimately far larger because
        # those are triple products. Reporting one under the other's name was
        # what made the resource column look ~100x the literature value.
        hamiltonian_pauli_terms = len(model.hamiltonian.to_mv().terms)

        # Sector dimension: C(norb, n_alpha) * C(norb, n_beta).
        n_spatial = model.metadata["n_spatial_orbitals"]
        n_alpha = (n_electrons + int(round(2 * sz))) // 2
        n_beta = n_electrons - n_alpha
        sector_dimension = math.comb(n_spatial, n_alpha) * math.comb(n_spatial, n_beta)

        print(f"  Qubits: {model.n}, Electrons: {n_electrons}, Sz: {sz:+g}")
        print(f"  Distinct Pauli words in H: {hamiltonian_pauli_terms}")
        print(f"  Sector dimension (N={n_electrons}, Sz={sz:+g}): {sector_dimension}")
        print(f"  Candidate SD Excitations: {len(candidates)} (total available: {len(all_candidates)})")

        # 4. Sector-Exact FCI Reference via SectorStatevectorBackend
        exact_energies, _ = SectorStatevectorBackend(
            model.n, n_electrons=n_electrons, sz=sz
        ).ground_state(model.hamiltonian, k=4, method="dense")
        e_exact = float(exact_energies[0])
        print(f"  Exact Sector FCI E0: {e_exact:+.9f} Ha")

        # 5. Adaptive A-CASE Subspace Eigensolver (Optimal Path Selection)
        max_size = (
            max_subspace
            if max_subspace and max_subspace > 0
            else len(candidates)
        )
        target_error = target_error_mha / 1000.0 if target_error_mha > 0 else None

        adaptive = run_acase(
            rho0,
            model.hamiltonian,
            candidates,
            max_size=max_size,
            leakage_tol=1e-10,
            exact_ground_energy=e_exact,
            target_error=target_error,
        )
        e_acase = float(adaptive.result.ground_energy)
        err_ha = e_acase - e_exact
        err_mha = err_ha * 1000.0
        print(f"  Adaptive A-CASE E0: {e_acase:+.9f} Ha (Error: {err_mha:+.4f} mHa, Optimal M={len(adaptive.labels)})")
        print(f"  Adaptive Stop Reason: {adaptive.stopped_reason}")

        # 6. Complete SD Coordinate Subspace Eigensolver (Chemical Accuracy Check)
        t_sd_start = time.time()
        sd_spectrum = solve_subspace(
            rho0, model.hamiltonian, [identity_generator(model.n), *all_candidates]
        )
        t_sd_elapsed = time.time() - t_sd_start
        e_sd = float(sd_spectrum.ground_energy)
        err_sd_ha = e_sd - e_exact
        err_sd_mha = err_sd_ha * 1000.0
        sd_size = len(all_candidates) + 1
        # When the SD basis already spans the whole sector, "exact to machine
        # precision" is arithmetic, not accuracy: there is nothing left to
        # miss. HF/STO-3G is the case here (35 candidates + identity = 36 = the
        # sector). Recording the flag stops that row being read as a result.
        sd_spans_full_sector = sd_size >= sector_dimension
        print(f"  Complete SD Subspace E0: {e_sd:+.9f} Ha (Error: {err_sd_mha:+.4f} mHa, M={sd_size})")
        print(f"  SD Subspace Wall-Clock Time: {t_sd_elapsed:.1f}s ({t_sd_elapsed/60:.1f} min)")
        if sd_spans_full_sector:
            print(f"  NOTE: SD basis ({sd_size}) spans the full sector "
                  f"({sector_dimension}); exactness here is tautological.")

        # 7. Response Analysis & Observables (on adaptive result)
        #
        # Every number below is read off ``adaptive.result`` -- the same solve
        # that produced ``e_acase`` above. Splicing observables from one run
        # beside energies from another produced a record in which the reported
        # Ritz vectors could not lower the reference energy by anything like
        # the reported amount, so this coupling is the point rather than an
        # implementation detail.
        spectrum = adaptive.result

        # An M-dimensional subspace cannot have effective rank above M. The
        # committed record briefly claimed rank 11 in a 6-dimensional space,
        # which is how the hand-edit was caught; assert it at the source.
        m_adaptive = len(adaptive.labels)
        if int(spectrum.effective_rank) > m_adaptive:
            raise AssertionError(
                f"{key}: effective_rank {spectrum.effective_rank} exceeds "
                f"basis size {m_adaptive}")

        double_occ = float(spectrum.expectation(double_occupancy(model)))
        spin2 = float(spectrum.expectation(total_spin_squared(model)))
        # Reference values that make <d> legible: the uncorrelated closed-shell
        # fraction. A correlated state sits strictly below it, so <d> equal to
        # this to many digits means no correlation was recovered.
        double_occ_hf = (n_electrons / 2) / n_spatial

        # Lehmann spectrum for the site-0 S_z response.
        mz_obs = magnetization(model, 0, axis="z")
        lehmann_lines = lehmann_spectrum(spectrum, mz_obs, min_weight=1e-12)
        chi0 = float(static_susceptibility(lehmann_lines))
        # S_z couples the singlet ground state to triplets. The adaptive basis
        # is grown from a closed-shell determinant through number- and
        # S_z-conserving singlet excitations, so it contains no triplet for the
        # operator to reach and every weight comes out at the numerical floor.
        # chi0 is therefore zero as a property of the *basis*, not of the
        # molecule, and must not be reported as a measured susceptibility.
        significant_lines = [ln for ln in lehmann_lines if abs(float(ln.weight)) > 1e-8]
        response_diagnostic = (
            "no significant weight: the S_z operator couples the singlet "
            "reference to triplet states that the singlet-preserving "
            "excitation basis does not span, so chi0 is a basis artifact "
            "rather than a physical response"
            if not significant_lines else
            f"{len(significant_lines)} lines above weight 1e-8"
        )

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
            "sector_dimension": sector_dimension,
            "e_rhf": e_rhf,
            "e_cisd": e_cisd,
            "e_ccsd": e_ccsd,
            "e_exact_fci": e_exact,
            "error_cisd_mha": (e_cisd - e_exact) * 1000.0,
            "error_ccsd_mha": (e_ccsd - e_exact) * 1000.0,
            "e_acase_adaptive": e_acase,
            "error_adaptive_mha": err_mha,
            "e_sd_complete": e_sd,
            "error_sd_mha": err_sd_mha,
            "adaptive_chemical_accuracy": bool(abs(err_mha) <= 1.5936),
            "sd_chemical_accuracy": bool(abs(err_sd_mha) <= 1.5936),
            "sd_spans_full_sector": bool(sd_spans_full_sector),
            # The adaptive stop is fed the exact energy, so "M at chemical
            # accuracy" is an oracle-assisted diagnostic and not a cost the
            # method could reproduce without knowing the answer.
            "adaptive_stop_reason": adaptive.stopped_reason,
            "adaptive_oracle_stop_used": bool(target_error is not None),
            "adaptive_target_error_mha": target_error_mha if target_error else None,
            "subspace_size_m": m_adaptive,
            "complete_sd_candidates": len(all_candidates),
            "complete_sd_basis_size": sd_size,
            "effective_rank": int(spectrum.effective_rank),
            "condition_number": float(spectrum.condition_number),
            "hamiltonian_pauli_terms": hamiltonian_pauli_terms,
            "element_word_universe": int(adaptive.resources.get("word_universe", 0)),
            "double_occupancy": double_occ,
            "double_occupancy_uncorrelated": double_occ_hf,
            "total_spin_squared": spin2,
            "static_susceptibility_chi0": chi0,
            "response_diagnostic": response_diagnostic,
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
        default=0,
        help="Max candidate excitations to evaluate (0 for full candidate pool)",
    )
    parser.add_argument(
        "--max-subspace",
        type=int,
        default=0,
        help="Max adaptive subspace basis size M (0 for unconstrained optimal growth)",
    )
    parser.add_argument(
        "--target-error-mha",
        type=float,
        default=1.5936,
        help="Target energy error threshold in mHa for early stopping (default 1.5936 mHa)",
    )
    args = parser.parse_args()
    run_pipeline(
        max_candidates=args.max_candidates,
        max_subspace=args.max_subspace,
        target_error_mha=args.target_error_mha,
    )


if __name__ == "__main__":
    main()

