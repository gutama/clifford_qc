# Larger-sector packet work comparison

Exploratory follow-up to the known 225-determinant H2O outcome in PR #145.
Freeze this file and configs/packet_work_large.json before the new runner,
checker and report exist or any new Hubbard trajectories are observed.
Preserve the H2O design and all earlier outcomes byte-for-byte.

Use the existing open-boundary 2x4 Hubbard builder at t=1, U=4, mu=0, half
filling N=8 and Sz=0: 16 qubits, 4900 determinants, 1764 singlet-frame columns.
Retain the deterministic spatial-occupation singlet frame and closed-shell
CAS(2e,2o) initial span on sites [3,4], with the other occupations fixed to
the first-eight-spin-orbital closed-shell determinant. This deliberately
continues the earlier classical-span protocol; it does not use the lattice
builder's Neel reference or claim the seed is a good Hubbard ground state.
All energy quantities and the target 0.0016 are in hopping units t, not Ha.

Run the unchanged five-arm trajectory implementation from run_packet_work_reuse:
original, reuse, shortlist, combined and selected_ci. Keep the 0.1 denominator
shift/floor, 80% residual-mass prefix, every-fifth full audit, <=1e-12 low-gain
fallback, orthogonalization, scoring and direction cutoffs unchanged. Stop at
true residual <=1e-9 t, 64 total columns, 61 additions or no admissible direction.
Record every prefix, not just successful target hits. Missing the accuracy
target at this budget is an outcome, not a reason to widen the budget.

Compile the native Pauli sector Hamiltonian once to SciPy CSR during each
worker's setup. Construction temporarily materializes the complete sector H;
report its dense temporary bound, final CSR payload and setup time separately.
The timed trajectory uses CSR matvec for every arm. This is a common compiled
classical kernel, not a matrix-free or scalable chemistry preparation claim.
Dense S2 and the full singlet frame remain explicit setup costs. Reuse keeps
all populated actions and its frame snapshot exactly as in the H2O pilot.
Use three fresh single-threaded processes per arm in rotated order, 300-second
timeout per worker, one worker at a time. Keep failed/time-limited workers in
all denominators. Do not rerun a failed cell selectively or change parameters.

Only after all timed workers finish, build an independent sparse occupation-
sign Hubbard Hamiltonian without native Pauli construction. Obtain its lowest
root by a Gershgorin upper shift and largest-magnitude sparse solves from two
fixed distinct starting vectors. Require agreement <=1e-9 t, full residual
<=1e-9 t and singlet action norm <=1e-8 before pricing accuracy against it.
Validate the native and occupation Hamiltonians on fixed deterministic probes.

Independently reconstruct the singlet frame, all selected correction spans,
energies, true residuals, spin, indicators, exact scores, shortlist/audit/fallback
decisions, frame-cache population and all H-call/payload counters. The checker
may reuse one independently constructed frame across workers; it must not
obtain the frame or Hamiltonian from the producer. Repeated observations may
reuse exact scientific replay only when their complete scientific trajectories
and setup identities agree, with every repeat's costs separately validated.

Report first-target costs only for actual target hits, and fixed-column
frontiers (8,16,32,64), final errors/residuals/stopping reasons for every arm,
cache arrays, full-process RSS, total times and all audit misses. Compare
original/reuse and shortlist/combined trajectories, exposing any differences.
Small repeat counts make wall-time comparisons descriptive. No hardware,
quantum shots, optimized-SCI superiority or asymptotic scaling claim follows.

Require full history, clean committed sources including untracked files, the
repository numerical environment and unchanged declaration/config bytes before
science. Bind source digests and full observation bytes to distinct declaration,
execution and observation commits. Preserve required original Git objects in a
digest-bound bundle, with a publication receipt. Run strict independent replay,
report freshness and the affected test/documentation/evidence/manuscript gates.
