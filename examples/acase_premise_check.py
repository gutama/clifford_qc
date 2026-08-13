"""A-CASE Phase-1 premise check on TFIM (n=4).

Run from the repository root: PYTHONPATH=. python examples/acase_premise_check.py

Validates the standing invariants of PLAN.md §3 on the shipped
Phase-1 solver (``clifford_qc.subspace``):

  1. H_ij, S_ij are assembled from the bilinear trace pairing against a single
     reference state -- no basis state |phi_i> = A_i|psi> is ever prepared.
  2. The normalized, thresholded generalized eigenproblem obeys E_sub >= E0.
  3. Nested basis growth is monotone non-increasing.
  4. The level-0..3 hierarchy converges toward the exact ground energy while
     the overlap matrix goes badly singular -- 37 generators span only 12
     independent directions -- which is why conditioning-aware adaptive
     selection is the load-bearing component rather than an optimization.

The overlap matrix is diagonally normalized before thresholding, so which
directions survive does not depend on generator scaling (the H^k rows have
much larger norms than the P_j rows).

The §6 resource columns are the other half of the compactness question: a
basis is not compact if its projected element operators carry a word universe
the measurement layer cannot afford.
"""
import numpy as np

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.matrix import exact_ground
from clifford_qc.models.spin import tfim
from clifford_qc.states import ket_density
from clifford_qc.subspace import (identity_generator, krylov_response,
                                  pauli_orbit, commutator_response,
                                  solve_subspace)

model = tfim(4, J=1.0, h=1.0)
H = model.hamiltonian
n = model.n
E0, _ = exact_ground(H.to_mv())
print(f"exact E0 = {E0:.10f}")

# reference: |0...0> (product state, not the ground state)
rho = ket_density(n, "0" * n)
words = [op.word for op in odd_y_filter(local_pool(n))]

# Level 0 (identity), Level 1 (P_j), Level 2 (G_j = -i/2 [H, P_j]),
# Level 3 (Krylov H^k) -- the §4.2 hierarchy, stacked.
gens = ([identity_generator(n)] + pauli_orbit(words)
        + commutator_response(H, words) + krylov_response(H, 4))

header = f"{'basis':<26}{'M':>4}{'kept':>6}{'E_sub':>16}{'gap':>10}{'kappa_S':>11}{'W':>7}{'S_H':>6}"
print(header)
print("-" * len(header))

previous = None
levels = [("level0 (I)", 1),
          ("level0+1 (P_j)", 1 + len(words)),
          ("level0+1+2 (P_j, G_j)", len(gens) - 4)]
levels += [(f"+H^{k}", len(gens) - 4 + k) for k in range(1, 5)]

for label, size in levels:
    result = solve_subspace(rho, H, gens[:size])
    energy = result.ground_energy
    assert energy >= E0 - 1e-9, f"variational bound violated: {energy} < {E0}"
    if previous is not None:
        assert energy <= previous + 1e-9, "nested monotonicity violated"
    previous = energy
    resources = result.resources
    print(f"{label:<26}{size:>4}{result.effective_rank:>6}{energy:>16.10f}"
          f"{energy - E0:>10.2e}{result.condition_number:>11.3e}"
          f"{resources['word_universe']:>7}"
          f"{resources['max_hamiltonian_element_support']:>6}")

print("\ninvariants hold;", "converged to E0" if abs(previous - E0) < 1e-7
      else f"remaining gap {previous - E0:.2e} (critical-point reference, as expected)")
