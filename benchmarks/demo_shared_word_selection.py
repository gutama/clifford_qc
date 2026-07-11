"""Demonstration benchmark for the Paper A measurement machinery.

Part 1 (H2, structural): how many Pauli-word measurements the global
commutator bank deduplicates, per model and pool.

Part 2 (H3 + allocation): finite-shot ADAPT-VQE with confidence-certified
selection on the n=4 TFIM, comparing allocation policies at equal
wrong-selection control. Selection quality is measured against the exact
commutator gradients recorded alongside each step (simulator diagnostic).

Run: python benchmarks/demo_shared_word_selection.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import statistics

from clifford_qc.models import tfim, xxz, random_ising
from clifford_qc.algorithms import (
    ConfidenceSelector, local_pool, odd_y_filter, all_words_pool, run_adapt,
)
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import CommutatorBank, UniformDoubling, VarianceProportional


def word_reuse_report():
    print("=" * 72)
    print("H2 — shared Pauli-word structure of the commutator bank")
    print("=" * 72)
    rows = []
    for model, pool_name, pool in [
        (tfim(6), "local", local_pool(6, periodic_context=False)),
        (tfim(8), "local", local_pool(8, periodic_context=False)),
        (xxz(6), "local", local_pool(6, periodic_context=False)),
        (random_ising(6, seed=0), "odd-Y w<=2", odd_y_filter(all_words_pool(6, max_weight=2))),
    ]:
        bank = CommutatorBank(model.hamiltonian, [op.word for op in pool])
        per_candidate = sum(len(row) for row in bank.coeffs)
        unique = len(bank.words)
        rows.append((model.name, pool_name, len(pool), per_candidate, unique,
                     per_candidate / unique))
    print(f"{'model':38s} {'pool':11s} {'cands':>5s} {'words':>6s} {'unique':>6s} {'reuse':>6s}")
    for name, pool_name, cands, per, uniq, ratio in rows:
        print(f"{name:38s} {pool_name:11s} {cands:5d} {per:6d} {uniq:6d} {ratio:5.2f}x")
    print("(reuse = per-candidate word measurements / shared unique words;")
    print(" every factor above 1 is measurement the shared cache pays once)\n")


def allocation_comparison(seeds: int):
    print("=" * 72)
    print("H3/H4 — confidence-gated ADAPT (TFIM n=4, delta=0.05, 6 operators)")
    print("=" * 72)
    model = tfim(4, 1.0, 1.0)
    pool = local_pool(4, periodic_context=False)
    policies = {
        "uniform_doubling": lambda: UniformDoubling(base=256, max_factor=64),
        "variance_proportional": lambda: VarianceProportional(round_budget=4096,
                                                              growth=2.0, max_rounds=7),
    }
    print(f"{'policy':22s} {'rel.err (median)':>16s} {'shots (median)':>15s} "
          f"{'circuits':>9s} {'regret>5%':>9s} {'ambiguous':>9s}")
    for name, make in policies.items():
        rels, shots, circuits, regret, ambiguous, steps = [], [], [], 0, 0, 0
        for seed in range(seeds):
            res = run_adapt(model, pool, backend=FiniteShotBackend(seed=seed),
                            selector=ConfidenceSelector(delta=0.05, near_tol=0.05),
                            allocator=make(), max_operators=6)
            rels.append(res.relative_error)
            shots.append(res.total_shots)
            circuits.append(res.total_circuits)
            for rec in res.records:
                if rec.selected_label is None or rec.exact_gradient is None:
                    continue
                steps += 1
                if abs(rec.exact_gradient) < 0.95 * rec.exact_gradient_max:
                    regret += 1
                if rec.status.value == "budget_exhausted_ambiguous":
                    ambiguous += 1
        print(f"{name:22s} {statistics.median(rels):16.2e} "
              f"{statistics.median(shots):15,.0f} {statistics.median(circuits):9,.0f} "
              f"{regret:4d}/{steps:<4d} {ambiguous:4d}/{steps:<4d}")
    print("(regret>5%: selected operator whose exact |g| < 95% of the best;")
    print(" ambiguity concentrates on exact symmetry ties, where any tied")
    print(" choice is equally good — the selector reports rather than hides it)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    word_reuse_report()
    allocation_comparison(args.seeds)
