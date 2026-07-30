"""A-CASE Phase-3 adaptive growth on TFIM (n=4), against the fixed baselines.

Run from the repository root: PYTHONPATH=. python examples/acase_adaptive.py

Grows a subspace one generator at a time from the model's own reference state
|++++> -- the same state ADAPT-VQE starts from -- choosing each addition by the
generalized 2x2 predicted lowering of ACASE_RESEARCH_PLAN.md §4.3, and prints
the §5 matched-budget comparison against the fixed constructions.

Two columns deserve attention beyond the energies. ``kappa_S`` is the retained
condition number: the fixed Krylov basis buys its accuracy with an overlap
spectrum that degrades exponentially in the power, which is the regime where
finite-shot certification (Phase 4) is least likely to survive. ``W`` is the
word universe the projected matrices span -- the measurement cost that decides
whether a small basis is actually compact.
"""
import numpy as np

from clifford_qc.algorithms.adapt import run_adapt
from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.matrix import exact_ground
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (commutator_response, identity_generator,
                                  krylov_response, pauli_orbit, run_acase,
                                  solve_subspace)

model = tfim(4, J=1.0, h=1.0)
H = model.hamiltonian
n = model.n
E0, _ = exact_ground(H.to_mv())
rho = ExactMVBackend().state(model.reference, ())  # |++++>
pool_ops = odd_y_filter(local_pool(n))
words = [op.word for op in pool_ops]

# The candidate pool is the whole §4.2 hierarchy: the adaptive method chooses
# among the families rather than committing to one of them in advance.
candidates = (pauli_orbit(words) + commutator_response(H, words)
              + krylov_response(H, 10))
print(f"exact E0 = {E0:.10f}   candidates = {len(candidates)}")

result = run_acase(rho, H, candidates, max_size=8, exact_ground_energy=E0)
header = (f"{'step':>4}{'generator':>14}{'M':>4}{'gap':>11}{'predicted':>12}"
          f"{'actual':>12}{'orth':>10}{'kappa_S':>10}{'new W':>8}{'W':>7}")
print(header)
print("-" * len(header))
for record in result.records:
    print(f"{record.step:>4}{record.selected_label:>14}{record.basis_size:>4}"
          f"{record.energy - E0:>11.2e}{record.predicted_lowering:>12.3e}"
          f"{record.actual_lowering:>12.3e}{record.orthogonal_fraction:>10.2e}"
          f"{record.condition_number:>10.2e}{record.new_words:>8}"
          f"{record.word_universe:>7}")
    assert record.actual_lowering >= record.predicted_lowering - 1e-12
print(f"stopped: {result.stopped_reason}")

print("\nmatched budget (M counts the identity generator):")
print(f"{'M':>3}{'A-CASE':>12}{'kappa_S':>10}{'Krylov':>12}{'kappa_S':>10}"
      f"{'QSE':>12}{'ADAPT-VQE':>12}{'opt evals':>11}")
for size in (3, 5, 7):
    adaptive = run_acase(rho, H, candidates, max_size=size - 1,
                         exact_ground_energy=E0)
    krylov = solve_subspace(rho, H, [identity_generator(n)]
                            + krylov_response(H, size - 1))
    qse = solve_subspace(rho, H, [identity_generator(n)]
                         + pauli_orbit(words[:size - 1]))
    vqe = run_adapt(model, pool_ops, max_operators=size - 1,
                    compute_exact_reference=False)
    print(f"{size:>3}{adaptive.energy - E0:>12.2e}"
          f"{adaptive.result.condition_number:>10.2e}"
          f"{krylov.ground_energy - E0:>12.2e}{krylov.condition_number:>10.2e}"
          f"{qse.ground_energy - E0:>12.2e}{vqe.energy - E0:>12.2e}"
          f"{vqe.optimizer_evaluations:>11}")

print("\nADAPT warm start: growing from a 2-operator ADAPT state instead of |++++>")
from clifford_qc.subspace import adapt_warm_start  # noqa: E402  (late, for the demo)

warm_rho, adapt_result = adapt_warm_start(model, pool_ops, max_operators=2,
                                          compute_exact_reference=False)
warm = run_acase(warm_rho, H, candidates, max_size=4, exact_ground_energy=E0)
cold = run_acase(rho, H, candidates, max_size=4, exact_ground_energy=E0)
print(f"  ADAPT(2 operators) alone      gap = {adapt_result.energy - E0:.2e}")
print(f"  A-CASE M=5 from |++++>        gap = {cold.energy - E0:.2e}")
print(f"  A-CASE M=5 from the ADAPT state gap = {warm.energy - E0:.2e}")
