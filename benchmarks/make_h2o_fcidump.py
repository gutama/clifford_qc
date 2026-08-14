"""Generate the frozen H2O CAS(8e,6o) FCIDUMP used by R2b.

The geometry, STO-3G basis, RHF orbitals, and active-space convention are the
ones frozen by the original A-CASE ladder: the oxygen 1s-like canonical
orbital is frozen and the remaining six spatial orbitals are active.  The
committed FCIDUMP and provenance JSON are benchmark inputs; ordinary R2b
reproduction reads those files and does not require PySCF or OpenFermion.

Run from the repository root with the chemistry extra installed::

    python benchmarks/make_h2o_fcidump.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_STEM = HERE / "data" / "h2o_sto3g_cas8e6o"
FROZEN_SPATIAL_ORBITALS = 1
ACTIVE_SPATIAL_ORBITALS = 6
BOND_LENGTH_ANGSTROM = 0.9584
BOND_ANGLE_DEGREES = 104.45


def geometry() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the exact equilibrium geometry frozen by the ladder config."""
    angle = math.radians(BOND_ANGLE_DEGREES)
    x = BOND_LENGTH_ANGSTROM * math.sin(angle / 2.0)
    z = BOND_LENGTH_ANGSTROM * math.cos(angle / 2.0)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (x, 0.0, z)),
        ("H", (-x, 0.0, z)),
    ]


def _patch_hidden_proc_memory_probe() -> bool:
    """Keep PySCF usable in sandboxes that hide ``/proc/<pid>/statm``."""
    if Path(f"/proc/{os.getpid()}/statm").exists():
        return False
    import pyscf.lib
    import pyscf.lib.misc

    def _zero_memory() -> tuple[float, float]:
        return 0.0, 0.0

    pyscf.lib.current_memory = _zero_memory
    pyscf.lib.misc.current_memory = _zero_memory
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stem",
        type=Path,
        default=DEFAULT_STEM,
        help="output path stem; .FCIDUMP and .provenance.json are appended",
    )
    args = parser.parse_args()

    import pyscf
    from pyscf import gto, mcscf, scf
    from pyscf.tools import fcidump

    memory_probe_fallback = _patch_hidden_proc_memory_probe()
    atoms = geometry()
    molecule = gto.M(
        atom=atoms,
        basis="sto-3g",
        charge=0,
        spin=0,
        symmetry=False,
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.kernel()
    if not mean_field.converged:
        raise SystemExit("RHF did not converge")

    active_electrons = molecule.nelectron - 2 * FROZEN_SPATIAL_ORBITALS
    casci = mcscf.CASCI(mean_field, ACTIVE_SPATIAL_ORBITALS, active_electrons)
    casci.ncore = FROZEN_SPATIAL_ORBITALS
    one_body, core_energy = casci.get_h1eff()
    two_body = casci.get_h2eff()
    casci_energy = float(casci.kernel()[0])

    fcidump_path = args.stem.with_name(args.stem.name + ".FCIDUMP")
    fcidump_path.parent.mkdir(parents=True, exist_ok=True)
    fcidump.from_integrals(
        str(fcidump_path),
        one_body,
        two_body,
        ACTIVE_SPATIAL_ORBITALS,
        active_electrons,
        nuc=core_energy,
        ms=0,
        orbsym=[1] * ACTIVE_SPATIAL_ORBITALS,
        tol=0.0,
    )
    header = (
        f"! H2O STO-3G CAS({active_electrons}e,{ACTIVE_SPATIAL_ORBITALS}o); "
        "lowest canonical RHF orbital frozen.\n"
        f"! Geometry (angstrom): r(OH)={BOND_LENGTH_ANGSTROM}, "
        f"angle(HOH)={BOND_ANGLE_DEGREES} degrees.\n"
        f"! Generated with PySCF {pyscf.__version__}; energies in Hartree.\n"
    )
    fcidump_path.write_text(
        header + fcidump_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    digest = hashlib.sha256(fcidump_path.read_bytes()).hexdigest()
    provenance = {
        "schema": "clifford_qc.active_space_benchmark.v1",
        "name": f"H2O STO-3G CAS({active_electrons}e,{ACTIVE_SPATIAL_ORBITALS}o)",
        "geometry_angstrom": [[symbol, list(xyz)] for symbol, xyz in atoms],
        "basis": "sto-3g",
        "orbital_method": "RHF",
        "active_space": {
            "electrons": int(active_electrons),
            "spatial_orbitals": ACTIVE_SPATIAL_ORBITALS,
            "selection": "lowest canonical RHF orbital frozen; next six active",
        },
        "generator": {
            "software": "PySCF",
            "version": pyscf.__version__,
            "scf_convergence_tolerance": 1e-12,
            "script": "benchmarks/make_h2o_fcidump.py",
            "memory_probe_fallback": memory_probe_fallback,
        },
        "fcidump_sha256": digest,
        "energy_unit": "hartree",
        "reference_energies": {
            "rhf": float(mean_field.e_tot),
            "fci": casci_energy,
        },
        "source_ladder": {
            "config": "benchmarks/configs/acase_ladder.json",
            "result": "benchmarks/reference_results/acase_ladder.jsonl",
            "rung_name": "h2o_cas8e6o",
            "geometry_scale": 1.0,
        },
        "notes": [
            "The committed FCIDUMP, not a live chemistry rebuild, is the R2b input.",
            "The CASCI energy independently checks the sector-exact clifford_qc solve.",
            "Molecular-orbital signs are gauge choices; the SHA-256 fixes this gauge.",
        ],
    }
    provenance_path = args.stem.with_name(args.stem.name + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"RHF   {mean_field.e_tot:.12f}")
    print(f"CASCI {casci_energy:.12f}")
    print(fcidump_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
