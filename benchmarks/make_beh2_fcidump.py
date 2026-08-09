"""Generate the frozen BeH2 active-space FCIDUMP benchmark input.

Linear BeH2 at the equilibrium Be--H distance, STO-3G, RHF orbitals, with the
Be 1s core frozen and the next four canonical orbitals kept active: CAS(4e,4o),
i.e. eight spin orbitals.  The register width matches the committed H4 input on
purpose, so the dyadic measurement hierarchy can be compared across two
chemically different Hamiltonians at the same block structure.

The emitted FCIDUMP and its provenance record are the immutable benchmark
input; this script exists so the file can be regenerated and audited, not so
that it is rerun by the benchmarks.  It requires the ``chemistry`` extra.

    python benchmarks/make_beh2_fcidump.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_STEM = HERE / "data" / "beh2_sto3g_r1.3264"

BOND_LENGTH = 1.3264
FROZEN_SPATIAL_ORBITALS = 1
ACTIVE_SPATIAL_ORBITALS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", type=Path, default=DEFAULT_STEM,
                        help="output path stem; suffixes are appended to it")
    args = parser.parse_args()

    import pyscf
    from pyscf import gto, mcscf, scf
    from pyscf.tools import fcidump

    geometry = [("Be", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, -BOND_LENGTH)),
                ("H", (0.0, 0.0, BOND_LENGTH))]
    mol = gto.M(atom=[(sym, xyz) for sym, xyz in geometry], basis="sto-3g",
                charge=0, spin=0, symmetry=False, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise SystemExit("RHF did not converge")

    n_active_electrons = mol.nelectron - 2 * FROZEN_SPATIAL_ORBITALS
    casci = mcscf.CASCI(mf, ACTIVE_SPATIAL_ORBITALS, n_active_electrons)
    casci.ncore = FROZEN_SPATIAL_ORBITALS
    h1e, core_energy = casci.get_h1eff()
    h2e = casci.get_h2eff()
    casci_energy = casci.kernel()[0]

    fcidump_path = args.stem.with_name(args.stem.name + ".FCIDUMP")
    fcidump_path.parent.mkdir(parents=True, exist_ok=True)
    fcidump.from_integrals(str(fcidump_path), h1e, h2e,
                           ACTIVE_SPATIAL_ORBITALS, n_active_electrons,
                           nuc=core_energy, ms=0,
                           orbsym=[1] * ACTIVE_SPATIAL_ORBITALS, tol=0.0)

    header = (
        f"! Restricted BeH2 active space: {n_active_electrons} electrons in "
        f"{ACTIVE_SPATIAL_ORBITALS} STO-3G spatial orbitals "
        f"(Be 1s core frozen).\n"
        f"! Geometry (angstrom): Be(0,0,0), H(0,0,-{BOND_LENGTH}), "
        f"H(0,0,{BOND_LENGTH}).\n"
        f"! Generated with PySCF {pyscf.__version__} RHF; all values are "
        f"Hartree.\n")
    fcidump_path.write_text(header + fcidump_path.read_text(encoding="utf-8"),
                            encoding="utf-8")

    digest = hashlib.sha256(fcidump_path.read_bytes()).hexdigest()
    provenance = {
        "schema": "clifford_qc.active_space_benchmark.v1",
        "name": f"linear BeH2 STO-3G CAS({n_active_electrons}e,"
                f"{ACTIVE_SPATIAL_ORBITALS}o)",
        "geometry_angstrom": [[sym, list(xyz)] for sym, xyz in geometry],
        "basis": "sto-3g",
        "orbital_method": "RHF",
        "active_space": {
            "electrons": int(n_active_electrons),
            "spatial_orbitals": ACTIVE_SPATIAL_ORBITALS,
            "selection": "Be 1s core frozen; next four canonical RHF orbitals",
        },
        "generator": {
            "software": "PySCF",
            "version": pyscf.__version__,
            "scf_convergence_tolerance": 1e-12,
            "script": "benchmarks/make_beh2_fcidump.py",
        },
        "fcidump_sha256": digest,
        "energy_unit": "hartree",
        "reference_energies": {
            "rhf": float(mf.e_tot),
            "fci": float(casci_energy),
        },
        "notes": [
            "The committed FCIDUMP is the immutable benchmark input.",
            "The reference energy is PySCF's CASCI(4e,4o) determinant-space "
            "solution in the same active space, computed independently of "
            "clifford_qc's Jordan-Wigner and projected-subspace paths.",
            "Molecular-orbital signs are gauge choices; the committed file "
            "fixes one gauge through its SHA-256 digest.",
            "The active-space size matches the committed H4 input so that the "
            "eight-qubit dyadic measurement hierarchy is comparable across "
            "two different Hamiltonians.",
        ],
    }
    provenance_path = args.stem.with_name(args.stem.name + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n",
                               encoding="utf-8")
    print(f"RHF   {mf.e_tot:.12f}")
    print(f"CASCI {casci_energy:.12f}")
    print(fcidump_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
