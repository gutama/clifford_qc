# Column-budget convergence: residual packets, selected CI and Davidson

Result-free declaration, frozen before implementation or any new trajectory, on
baseline `abcb02d8e58fc9be02a960a81a772850211535bc` (merged PR #148). The
numerical design is `configs/packet_budget_convergence.json`.

## Question and scope

The 64-column campaigns left the decisive comparison open. On the open 2x4
Hubbard sector, exhaustive residual packets end 1.935913 t above the sector
reference and greedy spin-adapted selected CI 2.077673 t above it, both far from
0.0016 t. Within larger column budgets, does either reach declared accuracy
thresholds, which needs fewer columns and Hamiltonian applications to do so, and
how do both compare with an unrestricted Davidson correction? This remains exact
classical sector-vector computation. There is no quantum preparation, measured
matrix entry, finite-shot estimate, optimized selected CI (CIPSI, HCI, SHCI) or
DMRG comparison, and no claim about larger systems.

Known outcomes inform the design and are not re-claimed: the published 64-column
trajectories, H2O CAS(8e,6o) converging within 32 columns, the 2x3 Hubbard errors
at 32 columns (0.341441/0.385303 t exhaustive/early), and Phase PRD's
Davidson-first method hierarchy at small M. The timeout uses a per-step cost
model fitted only to the published 64-column prefix timings.

## Arms

Every arm uses the same `SparseSingletFrame`, CAS(2e,2o) initial span, Ritz solve,
stopping rules and tolerances, and adds exactly one column per growth step.
Candidate work and cache construction are charged to solver time.

1. `packets_exhaustive`: `run_sparse_packets(policy="exhaustive",
   cache_mode="combined")` on merged code. Cache modes are bitwise identical
   (PR #147, #148); the checker replays independently regardless.
2. `packets_early`: the same with `policy="early"`, its 80% residual-mass
   shortlist, every-fifth audit and low-gain fallback. It is a heuristic; its
   accuracy loss is a measured outcome.
3. `selected_ci`: greedy spin-adapted selected CI, as in the earlier
   `selected_ci` arm. Every unselected singlet-frame column is a candidate. Its
   H action is computed once when first scored, cached and charged. Each step
   adds the candidate with the largest exact two-dimensional energy lowering,
   ties within 1e-12 by column index. This is not an optimized selected CI.
4. `davidson`: reference arm. Each step adds the singlet-frame projection of the
   packets' preconditioned residual, `F F^T [r / max(diag(H) - E + 0.1, 0.1)]`,
   twice orthogonalized against the span with the packets' cutoffs. It is the
   packet direction with every group merged, so it bounds what the one-group
   restriction costs. It is neither bounded-support nor measurement-priced.

Sparse implementations of arms 3 and 4 go in
`clifford_qc/subspace/sparse_baselines.py`, which is absent at this declaration,
as are the runner, checker, summarizer and tests.

## Inputs and budgets

| Input | Role | Determinants | Singlet columns | Budget | Repeats |
|---|---|---:|---:|---:|---:|
| `hubbard_2x4_u4` | primary | 4,900 | 1,764 | 512 | 2 |
| `hubbard_1x8_u4` | held-out lattice | 4,900 | 1,764 | 512 | 2 |
| `beh2_stretched_full` | held-out molecule | 1,225 | 490 | 490 | 3 |
| `hubbard_2x3_u8` | complete-span lattice control | 400 | 175 | 175 | 3 |
| `h2o_cas8e6o` | development, complete-span control | 225 | 105 | 105 | 3 |

Both 4,900-determinant inputs are open-boundary, t=1, U=4, mu=0, N=8, Sz=0, with
572 packet groups; the 8-site chain has never been run by these arms. BeH2 is
full STO-3G at the 2x symmetric stretch (Be-H 2.652 A in `molecular/catalog.py`),
N=6, Sz=0, from the tracked `molecular/results/beh2_stretched.fcidump`. A
different, sampled packet family ran on a CAS(4e,6o) of this geometry; these
arms have never run on its full sector. The 2x3 lattice is known at 32 columns.

Every input starts from the singlet part of a CAS(2e,2o): sites/orbitals [3,4]
for the 8-site lattices and H2O, [2,3] for the 2x3 lattice and BeH2. Other
occupations stay frozen to the first-N-spin-orbital closed-shell determinant,
continuing the earlier protocol, so every input starts from three columns.
Energies are in t for lattices and Ha for molecules.

Stop at true residual <=1e-9, the column budget, budget-3 growth steps, or no
admissible direction. Reference energies are never consulted for selection or
stopping. Reaching the budget without a threshold is an outcome; budgets are not
widened after observation.

## Execution

There are 52 scheduled cells: four arms for each of 13 input/repetition pairs.
Within each pair a seeded permutation fixes arm order. Run cells sequentially in
fresh processes with numerical threads pinned to one, under the repository
environment, with a 3600-second timeout per cell (about four times the fitted
512-column estimate for the slowest packet arm). The supervisor alone surveys
records. Every start and outcome is durably journaled. Failed, timed-out and
interrupted cells stay in denominators and are never rerun selectively; a started
cell whose worker metrics are durable is recovered as in the exact-cache runner.

At every prefix record dimension, energy, true residual norm, spin action norm,
cumulative H actions, candidates scored and materialized, span determinant
support and amplitude nonzeros above 1e-12, cumulative solver wall time and the
process high-water RSS. Record setup, solver and whole-worker time and the
import, setup, solver-end and worker RSS high-water marks separately. Persist
each distinct scientific payload once by SHA-256 with all per-cell observations.

## Validation

Only after all timed cells, build independent occupation-sign Hamiltonians and an
independent singlet frame. The reference is the lowest eigenvalue of H restricted
to that frame, cross-checked against the full-sector ground state and its S2;
report whether the sector ground state is a singlet. Independently replay every
distinct trajectory: selections, admissibility, scores, H calls, support counts
and stopping. Repeated cells of one arm must give bitwise-identical scientific
payloads. On complete-span controls, every arm that reaches the full singlet
span must lie within 1e-8 of the reference; an arm that stops earlier reports its
stopping reason.

## Readouts and decisions

Thresholds are 1, 0.1, 0.01 and 0.0016 in each input's unit. For each arm and
threshold report the first prefix at or below it: columns K, cumulative H actions
W and setup-plus-solver time, or "not reached within budget". Report frontiers at
64, 128 and 256 columns and at the budget.

1. Column compactness, `packets_exhaustive` against `selected_ci`, per input.
   "Packets more compact" if, for every threshold reached by at least one of the
   two, K_packets <= 0.8 K_SCI (an unreached threshold counts as beyond budget),
   and packet error is no higher at every frontier from 128 columns. "Selected CI
   more compact" is symmetric; anything else is "mixed". Packets add column
   compactness only if the primary and both held-out inputs read "packets more
   compact".
2. Work. Wherever columns favour one arm, compare W at the same thresholds. A
   column advantage that costs at least 1.25 times the other arm's H actions at
   any compared threshold is compression at higher classical work, not a
   resource advantage.
3. Early screening. Report early minus exhaustive error at each frontier and K
   and W at each threshold. "Early preserves accuracy" if early reaches every
   threshold that exhaustive reaches within 1.1 times its columns and its
   frontier errors are within 10% of exhaustive's; otherwise state the loss.
4. Davidson reference. Report its K, W and time per threshold and its span
   support at matched columns. If it reaches every threshold in fewer columns
   than both restricted arms, the packets-versus-selected-CI comparison concerns
   bounded-support compactness only, and is reported as such.

Timing from two or three fresh processes is descriptive. No quantum, hardware,
shot-cost or asymptotic-scaling claim follows, and no default changes.

Bind declaration and config hashes, the clean source commit, input digests, the
start/outcome journal, content-addressed scientific payloads and observation
publication to distinct commits, and preserve required Git objects in a
digest-bound bundle. Neither a lost observation nor a later reporting change may
be relabeled as new measurement.
