# Sparse packet setup and early screening (exploratory)

Declared after inspecting PR #145's H2O and 2x4 Hubbard outcomes. The dense
setup and action-cache overheads are known. This freezes a new engineering
comparison before its implementation or outcomes; it is not a claim that the
fixtures or research question were selected without prior results.

Preserve every earlier declaration, producer and observation. Add three arms:
`dense` uses the existing original packet trajectory and dense setup; `sparse`
uses direct CSR H and S2, a block-built sparse singlet frame, and exhaustive
packet screening; `early` uses that same sparse setup but marks groups before
materializing/orthogonalizing packet directions. No action cache is used.

All arms use the existing deterministic singlet frame, the closed-shell
CAS(2e,2o) start, 0.1 diagonal denominator shift/floor, two-pass
orthogonalization, 1e-12 score ties and the configured cutoffs. Sparse/exhaustive
must preserve the dense trajectory to 1e-9 energy/residual and 1e-8 column
tolerances, including selection identities. No default is changed.

Early screening marks the smallest descending prefix of group residual mass
covering 80%, with integer-key ties. Unlike the old shortlist, the marks are
computed before candidate admissibility. Every fifth decision scores all
admissible groups; other decisions fall back to all groups if the marked set
is dependent or its best gain is <=1e-12. Charge construction, audits and
fallbacks. Record missing audit winners and score losses, full residuals and
every retained prefix. This changed heuristic inherits no convergence theorem.

Freeze three inputs: the existing H2O CAS(8e,6o), the known open 2x4 Hubbard
U/t=4, and a held-out open 2x3 Hubbard U/t=8. The last input is new for this
implementation, not a blinded scientific dataset. Units, starts, budgets and
repeat counts are in configs/sparse_packet_work.json. No exact ground state
or energy is used by the selection code. Obtain independent occupation-sign
reference energies only after timed workers finish.

Run each cell in a fresh single-threaded process, one worker at a time, in
seeded randomized order. Retain failures/timeouts in denominators; do not
selectively rerun failed cells. Record setup/solver/whole-process time, OS peak
RSS, H actions, materialized candidates, exact scores, sparse setup payload,
and target and fixed-budget accuracy. Seven repeats on the medium fixture
measure timing variability, not seven independent scientific instances.

The first implementation succeeds structurally when it removes full-sector
D-by-D H/S2 and dense D-by-R frame allocations while preserving the exhaustive
trajectory. A practical speed promotion additionally requires >=20% lower
median setup-plus-solver time with no RSS regression, evaluated independently
per fixture. Report failures of either clause. Early screening must disclose
its error/time frontier and audit losses; it is not trajectory-equivalent.

The checker independently rebuilds occupation-sign Hamiltonians, spin and
singlet frames and replays candidate construction, selection, pencils and
counters. It may use dense small/medium reference objects; that checker cost
is outside timed solver claims. Bind source/config/declaration hashes and
clean committed execution to an observation digest and distinct commits.
The record is exact classical evidence with empirical resource observations,
not measured quantum entries, scalable state preparation or quantum advantage.

Deferred to later comparisons: bounded/adaptive action caches, block or
Jacobi-Davidson corrections, rotated measured-pencil thresholding, optimized
external SCI/DMRG, and ten-site production solves. Existing packed coefficient
storage and other solver defaults are unaffected.
