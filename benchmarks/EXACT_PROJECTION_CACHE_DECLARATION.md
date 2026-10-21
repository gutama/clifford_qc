# Exact packet projection caching: result-free declaration

Frozen before implementation or timing, on baseline
`1cb2d5e259c15220c077ee8661fa51858c6ad9ec` (merged PR #146).
The numerical design is `configs/exact_projection_cache.json`.

## Question and scope

Can reusable exact projections reduce total packet-growth cost while preserving
the baseline's physical directions, ordered selections, energies, residuals,
audits and work counters? Reviewer profiling motivates the question but supplies
no campaign result. This remains exact classical sector-vector computation;
there is no action cache, changed candidate heuristic, quantum preparation,
finite-shot matrix entry or optimized-SCI superiority claim.

The baseline is sparse setup with the existing packet policy. Exhaustive growth
has four arms: `none`, a basis-adjoint cache (`basis`), fixed group CSC blocks
plus a fixed frame adjoint (`frame`), and their `combined` use. The early policy
has `none` and `combined` arms, compared only within that policy. Every arm uses
the same arithmetic, two orthogonalization passes, canonical-phase rule,
candidate order, cutoffs and budget. No default changes automatically.

The basis cache borrows the solver-owned accepted basis for one growth step and
owns its conjugate transpose. The span is never mutated during candidate
evaluation; a new cache must be built after every accepted extension. Shape and
finiteness checks can move to cache construction; vector/cutoff checks remain.
Fixed frame/group snapshots are built once per run, charged to solver startup,
and must not cache candidate vectors or H actions. Storage reports charge every
owned dense/sparse numerical array, including duplicate group/adjoint storage.

## Frozen inputs and execution

H2O CAS(8e,6o), 225 determinants, is the development molecular input (three
repetitions). Open 2x4 Hubbard, U/t=4, 4,900 determinants, is primary (five
repetitions). BeH2 CAS(4e,4o), 36 determinants, is the held-out molecule; open
2x3 Hubbard, U/t=8, 400 determinants, is the held-out lattice (three repetitions
each). Molecular FCIDUMP bytes, active orbitals, budgets, tolerances and units
are fixed by the config. The BeH2 active space is small: it tests transfer of
correctness, not larger-molecule scaling. Implementation tests use toy inputs,
not held-out timing or accuracy feedback.

There are 84 scheduled cells: six policy/cache combinations for each of 14
input/repetition pairs. Within each pair, a seeded permutation fixes order.
Run cells sequentially in fresh processes with numerical threads pinned to one,
using the committed Python-minor/NumPy/SciPy environment. The supervisor alone
runs the record survey and passes a digest-bound execution stamp. Workers
verify source/environment identity without loading the reference-record corpus.
Timeout is 300 seconds per cell. Every start and outcome is durably journaled;
failed/interrupted cells remain in denominators and are never selectively rerun.

Measure model/operator/frame setup, solver, cache construction, end-to-end worker
time, OS whole-process peak RSS, and import/setup high-water baselines. A
high-water increase is not an allocation count or an isolated solver peak.
Report median and inclusive IQR, plus paired time ratios within each repetition.
Repeated deterministic trajectories are scientific replays, not independent
generalization or audit samples. Persist each distinct scientific payload once
by SHA-256, with all per-cell timings, memory and checkpoint identities retained.

## Validation and decisions

The checker independently rebuilds occupation-sign Hamiltonians and spin frames,
replays each distinct scientific trajectory, and verifies exact selections,
admissibility, scores, H calls, materializations, stopping and audits. It compares
each cache arm to its within-policy baseline, with 1e-8 physical-direction and
1e-9 energy/residual/score absolute tolerances; global phase is aligned and each
stored canonical pivot is separately enforced. Discrete selections, dimensions,
counters and audit flags must agree exactly. Cache generation and byte counts
must satisfy independent shape/storage formulas. Cached results must also match
the pre-change baseline, rather than trusting the new `none` branch alone.

Promotion requires every scheduled cell, correctness on every input/policy, at
least 20% lower primary median setup-plus-solver time, and no increase in primary
median whole-process RSS. State failures of either speed or memory honestly;
owned cache bytes are reported even if an OS floor hides their peak effect.
Performance arms remain opt-in unless separately approved after these results.
No matched-accuracy claim follows when a fixed budget misses the target.

Declaration/config hashes, clean source commit, execution dependencies, input
digests, start/outcome journal, immutable checkpoints, scientific payload hashes,
and observation publication are bound separately. Reference energies and error
readouts are computed after all timed cells. Preserve original source Git
objects for exact reproduction after publication or squashing. Neither a lost
observation nor a later reporting change may be relabeled as new measurement.
