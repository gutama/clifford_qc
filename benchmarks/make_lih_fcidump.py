"""Generate the frozen LiH CAS(4e,4o) FCIDUMP for the QR3b preflight.

LiH at its equilibrium bond length, STO-3G, RHF orbitals, with all four
electrons active in the lowest four canonical spatial orbitals.  The resulting
eight-spin-orbital register matches the H4 and BeH2 mapping-axis instances.

The emitted FCIDUMP and provenance record are immutable benchmark inputs.  The
script requires the ``chemistry`` extra and is not called by record checkers.

    python benchmarks/make_lih_fcidump.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_STEM = HERE / "data" / "lih_sto3g_r1.5949_cas4e4o"

BOND_LENGTH = 1.5949
ACTIVE_ELECTRONS = 4
ACTIVE_SPATIAL_ORBITALS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stem",
        type=Path,
        default=DEFAULT_STEM,
        help="output path stem; suffixes are appended to it",
    )
    args = parser.parse_args()

    import pyscf
    from pyscf import gto, mcscf, scf
    from pyscf.tools import fcidump

    geometry = [
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, BOND_LENGTH)),
    ]
    mol = gto.M(
        atom=[(symbol, xyz) for symbol, xyz in geometry],
        basis="sto-3g",
        charge=0,
        spin=0,
        symmetry=False,
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise SystemExit("RHF did not converge")

    casci = mcscf.CASCI(mf, ACTIVE_SPATIAL_ORBITALS, ACTIVE_ELECTRONS)
    casci.ncore = 0
    h1e, core_energy = casci.get_h1eff()
    h2e = casci.get_h2eff()
    casci_energy = casci.kernel()[0]

    fcidump_path = args.stem.with_name(args.stem.name + ".FCIDUMP")
    fcidump_path.parent.mkdir(parents=True, exist_ok=True)
    fcidump.from_integrals(
        str(fcidump_path),
        h1e,
        h2e,
        ACTIVE_SPATIAL_ORBITALS,
        ACTIVE_ELECTRONS,
        nuc=core_energy,
        ms=0,
        orbsym=[1] * ACTIVE_SPATIAL_ORBITALS,
        tol=0.0,
    )

    header = (
        f"! Restricted LiH active space: {ACTIVE_ELECTRONS} electrons in "
        f"{ACTIVE_SPATIAL_ORBITALS} STO-3G spatial orbitals "
        "(lowest four canonical RHF orbitals; no frozen electrons).\n"
        f"! Geometry (angstrom): Li(0,0,0), H(0,0,{BOND_LENGTH}).\n"
        f"! Generated with PySCF {pyscf.__version__} RHF; all values are Hartree.\n"
    )
    fcidump_path.write_text(
        header + fcidump_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    digest = hashlib.sha256(fcidump_path.read_bytes()).hexdigest()
    provenance = {
        "schema": "clifford_qc.active_space_benchmark.v1",
        "name": "LiH STO-3G CAS(4e,4o)",
        "geometry_angstrom": [[symbol, list(xyz)] for symbol, xyz in geometry],
        "basis": "sto-3g",
        "orbital_method": "RHF",
        "active_space": {
            "electrons": ACTIVE_ELECTRONS,
            "spatial_orbitals": ACTIVE_SPATIAL_ORBITALS,
            "selection": "lowest four canonical RHF orbitals; no frozen electrons",
        },
        "generator": {
            "software": "PySCF",
            "version": pyscf.__version__,
            "scf_convergence_tolerance": 1e-12,
            "script": "benchmarks/make_lih_fcidump.py",
        },
        "fcidump_sha256": digest,
        "energy_unit": "hartree",
        "reference_energies": {
            "rhf": float(mf.e_tot),
            "fci": float(casci_energy),
        },
        "notes": [
            "The committed FCIDUMP is the immutable QR3b candidate input.",
            "The reference energy is PySCF CASCI(4e,4o) in the same active space, computed independently of clifford_qc's mapping and projected-subspace paths.",
            "Molecular-orbital signs are gauge choices; the committed file fixes one gauge through its SHA-256 digest.",
            "The eight-qubit width matches H4 and BeH2, but this chemically independent Hamiltonian is an extension rather than a replacement for H4's right-censored exact-tier row.",
        ],
    }
    provenance_path = args.stem.with_name(args.stem.name + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"RHF   {mf.e_tot:.12f}")
    print(f"CASCI {casci_energy:.12f}")
    print(fcidump_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
