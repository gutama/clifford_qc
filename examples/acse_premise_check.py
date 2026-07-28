"""ACSE Phase-1 premise check on TFIM (n=4).

Run from the repository root: PYTHONPATH=. python examples/acse_premise_check.py

Validates the standing invariants of ACSE_RESEARCH_PLAN.md before any
subspace/ module exists:
  1. H_ij, S_ij are computable via states.expectation with non-Hermitian O.
  2. The thresholded generalized eigenproblem obeys E_sub >= E0.
  3. Nested basis growth is monotone non-increasing.
  4. The level-0..3 hierarchy converges toward the exact ground energy,
     with heavy linear dependence (few kept modes) motivating adaptive
     selection and conditioning-aware growth.
"""
import numpy as np

from clifford_qc import MV
from clifford_qc.states import expectation, ket_density
from clifford_qc.matrix import exact_ground
from clifford_qc.models.spin import tfim
from clifford_qc.measurement.bank import CommutatorBank
from clifford_qc.algorithms.pools import local_pool, odd_y_filter

model = tfim(4, J=1.0, h=1.0)
H = model.hamiltonian.to_mv()
n = model.n
E0, _ = exact_ground(H)
print(f"exact E0 = {E0:.10f}")

# reference: |0...0> (product state, not the ground state)
rho = ket_density(n, "0" * n)

pool = odd_y_filter(local_pool(n))
bank = CommutatorBank(model.hamiltonian, [p.word for p in pool])

# generators: Level 0 (I), Level 1 (P_j), Level 2 (G_j from the bank rows)
gens = [MV.scalar(n, 1.0)]
gens += [MV(n, {p.word.code: 1.0}) for p in pool]
gens += [MV(n, dict(row)) for row in bank.coeffs if row]


def ritz(gens, tau=1e-10):
    """Lowest Ritz value of the thresholded generalized eigenproblem."""
    m = len(gens)
    S = np.zeros((m, m), complex)
    Hm = np.zeros((m, m), complex)
    for i in range(m):
        for j in range(m):
            S[i, j] = expectation(rho, gens[i].dagger() * gens[j])
            Hm[i, j] = expectation(rho, gens[i].dagger() * H * gens[j])
    S = 0.5 * (S + S.conj().T)
    Hm = 0.5 * (Hm + Hm.conj().T)
    lam, U = np.linalg.eigh(S)
    keep = lam > tau
    X = U[:, keep] / np.sqrt(lam[keep])
    Ht = X.conj().T @ Hm @ X
    return float(np.linalg.eigvalsh(Ht)[0]), int(keep.sum())


# nested growth: I -> +P_j -> +G_j
prev = None
for k, label in ((1, "level0 (I)"),
                 (1 + len(pool), "level0+1 (P_j)"),
                 (len(gens), "level0+1+2 (P_j, G_j)")):
    E, kept = ritz(gens[:k])
    assert E >= E0 - 1e-9, f"variational bound violated: {E} < {E0}"
    if prev is not None:
        assert E <= prev + 1e-9, "nested monotonicity violated"
    prev = E
    print(f"{label:24s} basis={k:3d} kept={kept:3d}  E_sub={E: .10f}  gap={E - E0:.2e}")

# Level 3: Krylov enrichment H^k
Hk = MV.scalar(n, 1.0)
for k in range(1, 5):
    Hk = Hk * H
    gens.append(Hk.copy())
    E, kept = ritz(gens)
    assert E >= E0 - 1e-9 and E <= prev + 1e-9
    prev = E
    print(f"+H^{k:<21d} basis={len(gens):3d} kept={kept:3d}  E_sub={E: .10f}  gap={E - E0:.2e}")

print("invariants hold;", "converged to E0" if abs(prev - E0) < 1e-7
      else f"remaining gap {prev - E0:.2e} (critical-point reference, as expected)")
