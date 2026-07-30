"""A-CASE Phase-5 materials layer: Hubbard and Kitaev clusters.

Run from the repository root: PYTHONPATH=. python examples/acase_materials.py

The §1 scope in practice. A downfolding step hands over a small correlated
cluster; A-CASE grows a subspace on it, and every material observable is read
through the projected-matrix route of §8 -- the Ritz state is never formed.

Four things are on display:

1. the sparse reference tier (``clifford_qc.sparse``), including the
   sector-restricted solve that a grand-canonical cluster needs;
2. adaptive growth with the natively built, sector-preserving determinant
   excitations, so the subspace never leaves the reference's (N, S_z) sector;
3. projected observables along the growth trajectory -- double occupancy, spin
   correlations, the antiferromagnetic structure factor;
4. excited states by state-averaged growth against several Ritz roots.
"""
import numpy as np

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models.lattice import hubbard, kitaev_honeycomb
from clifford_qc.models.observables import (double_occupancy, link_correlations,
                                            occupation, spin_correlation,
                                            structure_factor, total_spin_squared)
from clifford_qc.sparse import (sparse_ground, sparse_ground_in_sector,
                                to_sparse)
from clifford_qc.subspace import (MatrixElementBank, determinant_excitations,
                                  identity_generator, krylov_response,
                                  occupied_spin_orbitals, run_acase)

backend = ExactMVBackend()


def dense(observable, psi):
    return float((psi.conj() @ (to_sparse(observable) @ psi)).real)


print("== sparse reference tier ==")
for model in (hubbard(4, U=4.0), hubbard((2, 2), U=4.0), kitaev_honeycomb(2, 2)):
    matrix = to_sparse(model.hamiltonian)
    global_value = sparse_ground(model.hamiltonian, k=1)[0][0]
    line = (f"  {model.name:34s} n={model.n:2d} terms={len(model.hamiltonian.terms):3d} "
            f"nnz={matrix.nnz:6d} E0={global_value:10.6f}")
    if model.metadata["kind"] == "fermionic_lattice":
        sector = sparse_ground_in_sector(model.hamiltonian,
                                         model.metadata["n_electrons"],
                                         model.metadata["sz"])[0][0]
        line += f"  sector(N={model.metadata['n_electrons']},Sz=0) {sector:10.6f}"
    print(line)

print("\n== adaptive growth on the 2x2 Hubbard cluster, observables projected ==")
model = hubbard((2, 2), t=1.0, U=4.0)
rho = backend.state(model.reference, ())
occupied = occupied_spin_orbitals(model)
candidates = determinant_excitations(model.n, occupied)
sector = sparse_ground_in_sector(model.hamiltonian, model.metadata["n_electrons"],
                                 model.metadata["sz"])
reference_energy, reference_state = sector[0][0], sector[1][:, 0]
print(f"  reference determinant occupies {occupied}; {len(candidates)} candidates")
print(f"  sector ground energy {reference_energy:.6f}")

observables = {
    "double occ": double_occupancy(model),
    "<S_0.S_1>": spin_correlation(model, 0, 1),
    "S(pi,pi)": structure_factor(model),
    "<S^2>": total_spin_squared(model),
    "<n_0>": occupation(model, 0),
}
header = (f"{'M':>3}{'generator':>18}{'E':>12}{'gap':>10}"
          + "".join(f"{name:>12}" for name in observables))
print(header)
print("-" * len(header))

bank = MatrixElementBank(rho, model.hamiltonian)
basis = bank.extend([identity_generator(model.n)])
for size in (1, 3, 5, 8, 12):
    grown = run_acase(rho, model.hamiltonian, candidates, max_size=size - 1,
                      exact_ground_energy=reference_energy, leakage_tol=1e-9)
    values = [grown.result.expectation(q) for q in observables.values()]
    label = grown.labels[-1] if len(grown.labels) > 1 else "-"
    print(f"{grown.basis_size:>3}{label:>18}{grown.energy:>12.6f}"
          f"{grown.energy - reference_energy:>10.2e}"
          + "".join(f"{v:>12.5f}" for v in values))

exact_values = {name: dense(q, reference_state) for name, q in observables.items()}
print(f"{'exact':>3}{'':>18}{reference_energy:>12.6f}{0.0:>10.2e}"
      + "".join(f"{exact_values[name]:>12.5f}" for name in observables))
print("  Every observable above is a projected matrix contracted with the Ritz")
print("  coefficients; no state vector of the subspace is ever built.")

print("\n== excited states by state-averaged growth (3 roots) ==")
averaged = run_acase(rho, model.hamiltonian, candidates, max_size=12, roots=3,
                     leakage_tol=1e-9)
block = run_acase(rho, model.hamiltonian, candidates, max_size=12, roots=3,
                  aggregation="max", leakage_tol=1e-9)
exact_roots = sparse_ground_in_sector(model.hamiltonian,
                                      model.metadata["n_electrons"],
                                      model.metadata["sz"], k=3)[0]
print(f"  exact sector roots      {np.round(exact_roots, 6)}")
print(f"  state-averaged (mean)   {np.round(averaged.root_energies, 6)}  "
      f"M={averaged.basis_size}")
print(f"  block (max)             {np.round(block.root_energies, 6)}  "
      f"M={block.basis_size}")
print("  Each Ritz root stays above its exact counterpart (Cauchy interlacing);")
print("  the average is monotone once all three roots exist.")

print("\n== Kitaev honeycomb cluster: the spin-liquid signature ==")
kitaev = kitaev_honeycomb(2, 2)
values, vectors = sparse_ground(kitaev.hamiltonian, k=1)
psi = vectors[:, 0]
print(f"  E0 = {values[0]:.6f}")
print("  link correlations (the only surviving spin correlations):")
for kind, operator in sorted(link_correlations(kitaev).items()):
    print(f"    <{kind.upper()}_i {kind.upper()}_j> on {kind}-links = "
          f"{dense(operator, psi):+.6f}")
far = spin_correlation(kitaev, 0, 3)
print(f"  <S_0.S_3> across a non-link pair = {dense(far, psi):+.6f}")
rho_kitaev = backend.state(kitaev.reference, ())
krylov = run_acase(rho_kitaev, kitaev.hamiltonian,
                   krylov_response(kitaev.hamiltonian, 8), max_size=6,
                   exact_ground_energy=values[0])
print(f"  A-CASE from |0...0> with Krylov candidates: M={krylov.basis_size} "
      f"E={krylov.energy:.6f} gap={krylov.energy - values[0]:.2e}")
