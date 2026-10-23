# Frame-coordinate packet projection: result-free declaration

Frozen before implementation or campaign timing on merged PR #149,
`259bd213978cba5b77b85e73c28399ec727d8c6c`. Numerical parameters, molecular
input hashes, budgets, order seed and decisions are fixed by
`configs/frame_coordinate_projection.json`.

## Question and mathematical contract

Can projection in the orthonormal singlet frame reduce packet-growth cost while
preserving today's physical trajectory? For the fixed sparse frame F, accepted
orthonormal physical basis V and frame coordinates C = F†V, candidate frame
coefficients a are projected twice by a ← a − C(C†a), then reconstructed as F a.
The physical norm supplies the existing dependence cutoff and normalization;
the existing physical canonical-phase rule is applied afterward. This relies
on F†F = I and V lying in the image of F. The physical ActionSpace extension
continues to enforce orthonormality. No full determinant-space projector is
built. Coordinate caches are rebuilt and released at every accepted extension.

The new opt-in `coordinates` mode owns C, C† and one sparse frame adjoint.
It borrows the immutable-for-the-run sparse frame, and stores neither group
block copies nor candidate/H-action histories. Account for every owned numerical
array, including both coordinate arrays. Existing `none`, `basis`, `frame` and
`combined` modes retain their meanings. Candidate order, two passes, cutoffs,
tie handling, phase convention, policy and H-action accounting are unchanged.
This is a classical sector-vector optimization, with no quantum, finite-shot,
convergence, asymptotic-scaling or optimized-selected-CI advantage claim.

## Frozen comparison and execution

Compare `combined` with `coordinates` separately under exhaustive and early
policies. Inputs are the same four previously published cache fixtures: H2O
CAS(8e,6o) development (three repetitions), open 2x4 Hubbard U/t=4 primary
(five), BeH2 CAS(4e,4o) molecular transfer check (three), and open 2x3 Hubbard
U/t=8 lattice transfer check (three). These are held out of implementation
tests, not previously unseen data; their historical outcomes are already known.
Tests use independent toy matrices and the undeclared 2x2 Hubbard fixture.

There are 56 timed cells. A seeded permutation sets the four mode/policy cells
within each input/repetition. Run sequentially in fresh single-threaded processes
under the committed record environment; no environment migration is permitted.
After timing, run eight historical `combined` controls from the exact baseline
Git source, one per input/policy. Native dependencies must match baseline bytes.
Failed, timed-out and interrupted cells remain in the denominator; do not rerun
started cells or silently replace a frozen design. The timeout is 300 s/cell.

Use a separate supervisor helper for the reference-record survey, so its peak
does not contaminate worker RSS. Report setup, solver, cache construction,
whole-process wall time, import/setup/solver-end/worker peak RSS and owned cache
bytes separately. Summaries give median and inclusive IQR, plus within-repeat
paired ratios. RSS is a process high-water mark, not an allocation counter or an
isolated solver peak. Keep a durable start/outcome journal and each distinct
scientific payload once by content hash. Record clean source/environment and
declaration/config/input digests. Preserve source Git objects on publication.

## Correctness and decision rules

Every input and policy must preserve exact ordered selections, dimensions,
stopping, materializations, H calls, audit flags and shortlist/fallback choices.
Compare physical directions up to global phase at 1e-8 absolute tolerance and
energies, residuals and scores at 1e-9. Independently enforce each stored
column's physical phase convention. Bitwise trajectory identity is not required:
the coordinate arithmetic deliberately changes floating-point reductions.

An independent occupation-sign Hamiltonian/spin-frame checker replays every
distinct trajectory, including scoring and physical admissibility, and checks
both modes against the pre-change historical control. Owned cache arrays must
satisfy shape-based storage formulas. Unit tests cover complex isometries,
dependent/near-dependent directions, non-singlet inputs, cache lifetimes and
unchanged baseline modes.

The joint performance bar is at least 20% lower primary median setup+solver time
and no increase in primary median worker RSS versus `combined`, separately by
policy, with all 56 cells successful and all correctness checks passing. Report
speed and memory clauses separately, including failures. Repeated deterministic
trajectories do not supply independent generalization evidence. Missing a fixed
accuracy target permits no matched-accuracy claim. Defaults remain unchanged
regardless of this comparison; promotion requires a later explicit decision.

Declaration, implementation, observation and publication are separate commits.
A declaration gate must verify ordering and frozen bytes before a producer runs.
Do not revise the phase pivot, selection heuristic, numbered-phase status or
historical records as part of this experiment.
