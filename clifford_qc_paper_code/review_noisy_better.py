"""Adversarial review battery for adapt_vqe_noisy_better.py."""
import math
import statistics
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from adapt_vqe_noisy_better import (
    CandidateAccumulator, qubit_pool, selection_observable, word_expectations,
    selection_gradient, adapt_state, adapt_vqe_noisy, adapt_vqe,
)
from vqe_tfim_rotor_better import tfim_hamiltonian, exact_ground_space


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(name)


n, J = 4, 1.0
H, _, _ = tfim_hamiltonian(n, J, 1.0)
pool = qubit_pool(n)
rho_t = adapt_state(n, [pool[3].generator, pool[7].generator], (0.4, -0.7))

print("== R1: cumulative accumulator is statistically exact ==")
lab, P = pool[5].label, pool[5].generator
G = selection_observable(H, P)
entries = word_expectations(rho_t, G)
g_true = selection_gradient(H, rho_t, P)
reps, N1, N2 = 4000, 96, 160          # incremental 96 + 160 vs single 256
rng = np.random.default_rng(3)
inc, single = [], []
for _ in range(reps):
    a = CandidateAccumulator(lab, P, G, list(entries))
    a.add_shots(N1, rng); a.add_shots(N2, rng)
    inc.append(a.estimate()[0])
    b = CandidateAccumulator(lab, P, G, list(entries))
    b.add_shots(N1 + N2, rng)
    single.append(b.estimate()[0])
sig_pred = CandidateAccumulator(lab, P, G, list(entries))
sig_pred.add_shots(N1 + N2, rng)
_, sig_o, sig_e = sig_pred.estimate()
check(f"incremental unbiased (|mean-exact| = "
      f"{abs(statistics.fmean(inc)-g_true):.2e} < 4σ/√reps)",
      abs(statistics.fmean(inc) - g_true) < 4 * sig_o / math.sqrt(reps))
r = statistics.stdev(inc) / statistics.stdev(single)
check(f"incremental σ == single-batch σ (ratio {r:.3f})", 0.93 < r < 1.07)
r2 = statistics.stdev(inc) / sig_o
check(f"matches oracle formula at N_total (ratio {r2:.3f})", 0.93 < r2 < 1.07)
check(f"empirical σ tracks oracle σ (ratio {sig_e/sig_o:.3f})",
      0.8 < sig_e / sig_o < 1.25)

print("== R2: no-repeat rule at h/J = 0.2 (baseline needed a repeat) ==")
res = adapt_vqe_noisy(n, J, 0.2, shots=None, max_ops=12, eps=1e-9)
print(f"    ops: {', '.join(res.labels)}")
print(f"    rel.err {res.rel_error:.2e}  stopped: {res.stopped_reason}")
check("exact no-repeat path still reaches rel.err < 1e-8 at h = 0.2",
      res.rel_error < 1e-8)
check("no operator repeated", len(set(res.labels)) == len(res.labels))

print("== R3: exit-logic anatomy (the eps-vacuity defect) ==")
r_rank = adapt_vqe_noisy(n, J, 1.0, shots=64,
                         rng=np.random.default_rng(5), max_ops=10,
                         require_rank_resolution=True)
r_norank = adapt_vqe_noisy(n, J, 1.0, shots=64,
                           rng=np.random.default_rng(5), max_ops=10,
                           require_rank_resolution=False)
r_norank_big = adapt_vqe_noisy(n, J, 1.0, shots=8192,
                               rng=np.random.default_rng(5), max_ops=10,
                               require_rank_resolution=False)
print(f"    rank ON,  base 64  : rel {r_rank.rel_error:.2e}  "
      f"shots {r_rank.total_shots/1e6:.2f}M  "
      f"ambiguous {sum(s.ambiguous for s in r_rank.record)}/{len(r_rank.record)}")
print(f"    rank OFF, base 64  : rel {r_norank.rel_error:.2e}  "
      f"shots {r_norank.total_shots/1e6:.2f}M   <- escalation never fires")
print(f"    rank OFF, base 8192: rel {r_norank_big.rel_error:.2e}  "
      f"shots {r_norank_big.total_shots/1e6:.2f}M")
tied_steps = sum(1 for s in r_rank.record if s.ambiguous)
check("symmetric ties leave accepted picks rank-ambiguous "
      "(escalated to ceiling)", tied_steps >= len(r_rank.record) - 1)
check("DEFECT confirmed: rank OFF at low base stops prematurely "
      "(rel > 1e-2) because 'ghat >= eps' short-circuits significance",
      r_norank.rel_error > 1e-2)
check("mechanism confirmed: without escalation, only a large BASE "
      "budget recovers accuracy (rel < 1e-3 at 8192)",
      r_norank_big.rel_error < 1e-3)
check("rank ON remains accurate (rel < 1e-3) — the default path is safe",
      r_rank.rel_error < 1e-3)

print("== R4: determinism under fixed seed ==")
a1 = adapt_vqe_noisy(n, J, 1.0, shots=64, rng=np.random.default_rng(42),
                     max_ops=8)
a2 = adapt_vqe_noisy(n, J, 1.0, shots=64, rng=np.random.default_rng(42),
                     max_ops=8)
check("identical labels, energy, shot count",
      a1.labels == a2.labels and a1.energy == a2.energy
      and a1.total_shots == a2.total_shots)

print("== R5: input validation ==")
for bad in (0, -5):
    try:
        adapt_vqe_noisy(n, J, 1.0, shots=bad)
        ok = False
    except ValueError:
        ok = True
    check(f"shots={bad} rejected", ok)

print("\nReview battery passed.")
