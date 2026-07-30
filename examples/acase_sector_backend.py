"""A-CASE Phase-6 exact tier: the sector-restricted statevector backend.

Run from the repository root: PYTHONPATH=. python examples/acase_sector_backend.py

The Witt/minimal-left-ideal representation of ACASE_RESEARCH_PLAN.md §3 with a
plain surface: a pure state is stored on the occupation words of one
(N, S_z) sector, so memory is C(n,k) rather than 2^n, and a Pauli word acts by
bit-mask gather with the words grouped by X-mask.

This is the *reference* tier, explicitly not the compactness claim. What it
buys: exactness where ``matrix.exact_ground`` cannot reach, no
``(#terms) x 2^n`` sparse matrix, and -- for a DMFT-style outer loop -- an
operator whose expensive part is compiled once and reused.
"""
import time

import numpy as np

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.backends.sector_statevector import (lanczos_ground,
                                                     sector_basis,
                                                     sector_projector)
from clifford_qc.matrix import exact_ground
from clifford_qc.models.lattice import hubbard
from clifford_qc.pauli import comm
from clifford_qc.sparse import sparse_ground_in_sector, to_sparse
from clifford_qc.subspace import (determinant_excitations,
                                  occupied_spin_orbitals, run_acase)

print("== the sector is the storage unit ==")
header = (f"{'sites':>6}{'n':>4}{'sector dim':>12}{'2^n':>14}{'ratio':>7}"
          f"{'state':>10}{'full state':>12}{'sparse nnz':>14}")
print(header)
print("-" * len(header))
for sites in (4, 6, 8, 10, 12):
    model = hubbard(sites, t=1.0, U=4.0)
    backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], 0.0)
    report = backend.memory_estimate()
    nnz = (f"{len(model.hamiltonian.terms) * 2 ** model.n:14,}"
           if model.n <= 20 else f"{'too large':>14}")
    print(f"{sites:>6}{model.n:>4}{backend.dimension:>12,}{2 ** model.n:>14,}"
          f"{report['full_space_amplitudes'] / report['sector_amplitudes']:>6.1f}x"
          f"{report['sector_bytes'] / 1e6:>9.2f}M"
          f"{report['full_space_bytes'] / 1e6:>11.1f}M{nnz}")
print("  'sparse nnz' is what sparse.to_sparse would have to hold for the same")
print("  Hamiltonian; the backend holds none of it.")

print("\n== agreement with the tiers it replaces ==")
for sites in (2, 4, 6):
    model = hubbard(sites, t=1.0, U=4.0)
    backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], 0.0)
    banked = backend.ground_state(model.hamiltonian, k=1)[0][0]
    lanczos = backend.ground_state(model.hamiltonian, k=1, method="lanczos")[0][0]
    restricted = sparse_ground_in_sector(model.hamiltonian,
                                         model.metadata["n_electrons"], 0.0)[0][0]
    dense = exact_ground(model.hamiltonian.to_mv())[0] if model.n <= 12 else float("nan")
    print(f"  {sites} sites: eigsh {banked:.10f}  lanczos {lanczos:.10f}  "
          f"sector-sparse {restricted:.10f}  dense {dense:.10f}")

print("\n== the sector projector is a projector (the ideal, at small n) ==")
for sites in (2, 3):
    model = hubbard(sites, U=4.0)
    n = model.n
    electrons, sz = model.metadata["n_electrons"], model.metadata["sz"]
    projector = sector_projector(n, electrons, sz)
    commutes = comm(model.hamiltonian.to_mv(), projector).is_zero(1e-10)
    print(f"  {model.name}: N={electrons} Sz={sz}  "
          f"P^2==P {(projector * projector).is_close(projector, 1e-10)}  "
          f"tr P = {projector.trace().real:.0f} = |sector| "
          f"{sector_basis(n, electrons, sz).size}  [H,P]=0 {commutes}")

print("\n== compile once, solve many (the DMFT-style access pattern) ==")
model = hubbard(10, t=1.0, U=4.0)
backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], 0.0)
started = time.perf_counter()
eager = backend.operator(model.hamiltonian, precompute=True)
compile_time = time.perf_counter() - started
lean = backend.operator(model.hamiltonian, precompute=False)
psi = np.zeros(backend.dimension, dtype=complex)
psi[0] = 1.0
for name, operator in (("compiled", eager), ("lean", lean)):
    started = time.perf_counter()
    for _ in range(5):
        operator.matvec(psi)
    per_matvec = (time.perf_counter() - started) / 5
    memory = operator.memory_estimate()
    print(f"  {name:9s} matvec {per_matvec * 1e3:7.1f} ms   "
          f"held {memory['compiled_bytes'] / 1e6:6.1f} MB   "
          f"{memory['groups']} X-mask groups from {memory['words']} words")
print(f"  compilation of the grouped passes: {compile_time * 1e3:.0f} ms")
started = time.perf_counter()
values = backend.ground_state(model.hamiltonian, k=1)[0]
print(f"  10-site (n=20) ground energy {values[0]:.8f} in "
      f"{time.perf_counter() - started:.2f} s")

print("\n== what it is for: a reference A-CASE can be scored against ==")
model = hubbard(6, t=1.0, U=4.0)
backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], 0.0)
roots = backend.ground_state(model.hamiltonian, k=3)[0]
rho = ExactMVBackend().state(model.reference, ())
candidates = determinant_excitations(model.n, occupied_spin_orbitals(model))
print(f"  6-site cluster (n=12): sector roots {np.round(roots, 6)}")
print(f"  {len(candidates)} sector-preserving candidates")
for size in (4, 7, 10):
    grown = run_acase(rho, model.hamiltonian, candidates, max_size=size - 1,
                      exact_ground_energy=roots[0], leakage_tol=1e-9)
    print(f"    A-CASE M={grown.basis_size:2d}  E={grown.energy:.6f}  "
          f"gap={grown.energy - roots[0]:.3e}  W={grown.resources['word_universe']}")
print("  The reference tier is what makes that gap column meaningful at a size")
print("  where the dense route is already gone.")
