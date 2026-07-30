"""A-CASE Phase-4 finite-shot layers on TFIM (n=4).

Run from the repository root: PYTHONPATH=. python examples/acase_finite_shot.py

Three studies, one per stage of ACASE_RESEARCH_PLAN.md §5 Phase 4, kept apart
because the whole point of the staging is that an asymptotic error bar must
never be mistaken for a certificate.

4A -- shared grouped measurement: one QWC-grouped measurement of the word
universe estimates every entry of (S, H). The infinite-shot limit of that
reconstruction is the exact matrix.

4B -- asymptotic uncertainty: the delta-method sigma is compared against the
actual spread of the measured Ritz value over independent measurement seeds.
The comparison is the only thing that separates a correct variance from a
plausible-looking one, and it is also where the honest limits show: with a
well-conditioned overlap the prediction is accurate, and with an
ill-conditioned one the *typical* run is described while a rare run is not.

4C -- sample-split growth certificate: a construction batch fixes the
subspace, its Ritz pair, and the candidate norms; an independent certification
batch bounds each candidate's residual coupling with empirical-Bernstein
radii; growth happens only when a candidate is certified above threshold, and
the run abstains otherwise.
"""
import numpy as np

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.matrix import exact_ground
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (MatrixElementBank, SharedMeasurement,
                                  bootstrap_ritz, commutator_response,
                                  identity_generator, krylov_response,
                                  pauli_orbit, ritz_uncertainty,
                                  run_certified_acase)

SEEDS = 40
model = tfim(4, J=1.0, h=1.0)
H = model.hamiltonian
n = model.n
rho = ExactMVBackend().state(model.reference, ())
E0, _ = exact_ground(H.to_mv())
words = [op.word for op in odd_y_filter(local_pool(n))]

cases = {
    "kappa_S = 1 (I + 3 words)":
        [identity_generator(n)] + pauli_orbit(words[:3]),
    "kappa_S ~ 2e2 (I + 6 words + H, H^2)":
        [identity_generator(n)] + pauli_orbit(words[:6]) + krylov_response(H, 2),
}

print(f"exact E0 = {E0:.10f}\n")
print("== 4A: one shared grouped measurement reconstructs every entry ==")
for name, generators in cases.items():
    bank = MatrixElementBank(rho, H, generators)
    shared = SharedMeasurement(bank)
    S_limit, H_limit = shared.exact_matrices()
    S_exact, H_exact = bank.matrices()
    print(f"  {name}: M={len(generators)} words={len(shared.words)} "
          f"circuits={len(shared.groups)} "
          f"infinite-shot vs exact = "
          f"{max(np.abs(S_limit - S_exact).max(), np.abs(H_limit - H_exact).max()):.1e}")

print(f"\n== 4B: delta-method sigma against the spread over {SEEDS} seeds ==")
header = (f"{'basis':>38}{'shots/grp':>11}{'MC std':>11}{'mean sigma':>12}"
          f"{'median sigma':>14}{'bias':>11}{'cover':>8}")
print(header)
print("-" * len(header))
for name, generators in cases.items():
    bank = MatrixElementBank(rho, H, generators)
    exact = bank.solve()
    shared = SharedMeasurement(bank)
    for shots in (2000, 20000):
        energies, sigmas, covered = [], [], 0
        for seed in range(SEEDS):
            cache = shared.measure(FiniteShotBackend(seed=7000 + seed), shots)
            result = shared.solve(cache)
            interval = ritz_uncertainty(shared, cache, result)
            energies.append(result.ground_energy)
            sigmas.append(interval.std_error)
            covered += interval.lower <= exact.ground_energy <= interval.upper
        print(f"{name:>38}{shots:>11}{np.std(energies, ddof=1):>11.2e}"
              f"{np.mean(sigmas):>12.2e}{np.median(sigmas):>14.2e}"
              f"{np.mean(energies) - exact.ground_energy:>+11.1e}"
              f"{covered / SEEDS:>8.2f}")

print("\n  The ill-conditioned row at 2000 shots is the finding, not an outlier:")
print("  the MC std is dominated by a handful of runs that admit a near-null")
print("  overlap mode, while the median sigma stays at ~7e-3. A mean-and-variance")
print("  description of the error is inadequate there -- hence 'asymptotic',")
print("  never 'certified', and hence conditioning-aware growth.")

bank = MatrixElementBank(rho, H, cases["kappa_S = 1 (I + 3 words)"])
shared = SharedMeasurement(bank)
cache = shared.measure(FiniteShotBackend(seed=3), 4000)
result = shared.solve(cache)
delta_method = ritz_uncertainty(shared, cache, result)
resampled = bootstrap_ritz(shared, cache, replicates=200, seed=1)
print(f"\n  cross-check at kappa_S = 1, 4000 shots/group:")
print(f"    delta method  sigma = {delta_method.std_error:.3e}  ({delta_method.evidence})")
print(f"    grouped bootstrap   = {resampled.std_error:.3e}  ({resampled.evidence})")

print("\n== 4C: sample-split growth certificate (delta = 0.05, EB bounds) ==")
candidates = (pauli_orbit(words) + commutator_response(H, words)
              + krylov_response(H, 4))
for shots, threshold in ((4000, 0.05), (40000, 0.05), (40000, 0.4)):
    certified = run_certified_acase(rho, H, candidates, FiniteShotBackend(seed=11),
                                    max_size=5, construction_shots=shots,
                                    certification_shots=shots, delta=0.05,
                                    threshold=threshold, bound="eb",
                                    exact_ground_energy=E0)
    print(f"\n  shots/group={shots}, threshold={threshold}: {certified.labels}")
    print(f"    gap={certified.energy - E0:+.2e}  shots={certified.total_shots:,}  "
          f"circuits={certified.total_circuits}  stop={certified.stopped_reason}")
    for record in certified.records:
        if record.coupling is None:
            print(f"      step {record.step}: ABSTAIN over {record.candidates_scored} "
                  f"candidates -- {record.reason}")
            continue
        bound = record.coupling
        print(f"      step {record.step}: {record.selected_label:10s} "
              f"|r| = {bound.estimate:.3f} in [{bound.lower:.3f}, {bound.upper:.3f}]  "
              f"{record.resolution}  evidence={record.evidence}")

print("\n  Certified growth is expensive by construction: two independent batches")
print("  per step over the whole universe, and abstention as soon as no candidate")
print("  clears the threshold. That is the trade the plan accepted for a")
print("  defensible statement rather than a reused-shot heuristic.")
