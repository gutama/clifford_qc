# Phase 17: MORE-ADAPT breadth

This is a separate, result-free comparison for the final Phase 17 breadth
deliverable. Existing CEO trajectories, ground-state target crossings and
Phase 17's 87.5% implementation fraction were known before this design. They
are retained as historical context; this is not a blind preregistration or a
replica of that ground-state comparison. Only undeclared Hubbard/two-orbital
fixtures were used to develop the new code. No new molecular multi-root
spectrum, trajectory, target crossing or comparative cost preceded this freeze.

The binding design is `configs/phase17_more.json`. It contains input digests,
complete QE coefficients/order, reference occupations, numerical policies and
source digests, but no multi-root outcome.

## Matched comparison

Use the committed H4, BeH2 and LiH CAS(4e,4o) inputs from the CEO programme.
These are the programme's eight-qubit, four-active-electron fixtures; the scope
does not expand to full molecular orbital spaces.
Every instance is required at two and three roots, with two arms: twelve cells.
Stay in the interleaved Jordan--Wigner N/Sz sector, with dimension at most 256.
Total spin and point-group symmetry are not imposed. The reference order is HF
first, then spin-conserving single excitations in frontier-orbital order:
occupied spatial index descending, virtual spatial index ascending, alpha then
beta, with occupation integer resolving any remaining tie. This uses orbital
indices, not measured orbital energies. If needed, remaining sector occupations
follow in ascending order. No Hamiltonian diagonal or exact spectrum chooses
those references. Both arms
start from precisely the same orthogonal reference span and use the complete
generalized spin-conserving QE family already validated in Phase 17.

`more_adapt` is the explicitly declared uniform-weight shared-QE-unitary
MORE-ADAPT variant: minimize `Tr(Phi† H Phi)/r`, where `Phi` has the r orthogonal
reference columns after a common ansatz. Screen every QE on every reference,
average its signed derivatives, then append the first operator within absolute
`1e-10` of the largest absolute average. Repeated QEs are allowed. Append one
zero angle and reoptimize every angle with L-BFGS-B. State-average gradient
cancellation is legitimate; it is not proof that individual reference columns
are eigenstates. Sorted eigenvalues of `Phi† H Phi` provide the root readout.
This is a bounded QE instantiation, not a claim to reproduce every pool,
reference choice, enhanced subroutine or optimized circuit of the MORE-ADAPT
literature discussed in PLAN (arXiv:2409.11210).

`acase_mean` reuses native `run_acase(roots=r, aggregation='mean')`. On one fixed
HF ket, initial Pauli configuration maps `P_j` represent the same references;
candidates are `T_i P_j` for every QE and reference. No observed-state pruning
occurs. Exact duplicate operator maps are coalesced by the native bank; null or
already-retained state directions are rejected by native conditioning. These
particular responses are determinants, so this is a selected-determinant
special case, not evidence for generic A-CASE compactness. The frozen input
manifest reports the static response family's dimension separately from the
full sector; the two variational families need not have equal expressivity.
Configuration maps
need not conserve the sector globally: their action on the fixed reference is
in the matched sector, which the independent occupation oracle verifies.

Both arms have 32 new operator/direction additions and a cap of 32 nonlinear
angles. MORE uses one angle per addition. A-CASE uses zero nonlinear angles and
at most `r + 32` basis directions; its linear Ritz coefficient entries and both
overlap/H eigendecompositions are reported separately. These are matched caps,
not matched variational manifolds or a licence to treat eigensolver work as free.

## Stopping, targets and currencies

MORE stops only at its full averaged-gradient L2 threshold `1e-6`, the operator
budget, or the angle budget. Each optimization is capped at 350 iterations,
with `gtol=1e-8` and the common adapter's `ftol=1e-13`. Report the final maximum
absolute gradient and whether it meets gtol, not a SciPy success message or
global-optimality guarantee. Every distinct computed parameter tuple, including
the final tuple, and every callback/cache index is retained.

A-CASE retains its mean-root 2-by-2 lowering rule, conditioning floor `1e-8`,
lowering stop `1e-10`, normalized-overlap cutoff `1e-10`, zero relative cutoff,
condition cap `1e12` and canonical ties `1e-12 + 1e-9 * abs(best)`. Its existing
budget, all-rejected and exhausted-pool stops remain intact. No exact energy
target drives either arm's selection or stopping.

Read the first retained prefix for which every sorted root has absolute error
at most `1.6e-3` Ha against the same exact N/Sz spectrum. An initial span already
meeting the target is a reference-floor diagnostic; otherwise label reached or
right-censored. Do not interpolate, stop on the oracle, or aggregate a winner
across instances/root counts. Completion means the declared comparison and its
checks ship, not that every target is reached or one method wins.

Charge full screening, reference-dependent gradient/word forms, distinct
optimizer computations, callback requests, parameter derivatives, native bank
pair creation and cumulative/selected word support independently. MORE's dense
operator bytes count H and the full screened QE matrices, not the process.
A-CASE reports bank coefficient occurrences and a 24-byte payload convention,
not attributed allocator bytes. Report every retained pencil and root readout.
Generic Pauli parity-ladder bounds apply to the MORE ansatz on each reference.
Its root readout additionally charges every complex projected-H entry: two
quadrature superpositions per off-diagonal pair, prepared with a pivot H,
optional S and `HammingDistance(i,j)-1` fanout CX gates. Diagonals plus those
superpositions give `r**2` reference families per independent H-word sweep.
The common ansatz must run on every family. Each Pauli word is then read in
local bases, with zero readout CX gates and H / S-dagger-then-H basis gates.
A-CASE similarly reports independent local-word settings and local basis gates,
with zero readout CX gates on its one HF reference. Identity words need no
readout. X preparations and both generalized-eigenproblem decompositions are
separate. All these are logical constructions, not executed quantum circuits;
optimizer/gradient measurement circuits and finite-shot precision are unpriced.
No weighted total-cost estimate, shot count, optimized CEO circuit or measured
hardware advantage is inferred. Evidence uses the v1 per-quantity contract.

## One execution and independent verification

Commit the declaration/config before a clean implementation-source commit.
The producer defaults to outcome-free validation. Explicit execution requires
full Git history, the protocol-cost record's Python minor and exact NumPy/SciPy
versions, one BLAS thread, hash seed zero, and at least 15 GiB plus 512 MiB free
within Linux host/cgroup limits. Environment/tree/capacity refusal precedes
consuming the exclusive persistent claim in the common Git directory.

Launch each declared cell once in a fresh interpreter, in case/root/arm order.
The parent samples RSS every 0.05 s, enforces a 15 GiB ceiling and six-hour
deadline, and records exit status and wall time. Worker kernel high-water RSS
also rejects a crossing missed by polling. Preparation and instrumentation are
included; these are single-host observations, not a universal memory bound.
Timeout, memory and worker failures remain visible. Preserve checkpointed
observations; never erase/reset a consumed claim, overwrite a record or quietly
retry/shrink an arm. A missing/failed cell cannot complete Phase 17.

The independent scientific checker calls neither producer driver, MORE driver,
native A-CASE selector/bank, native Pauli product nor native generalized
eigensolver. FCIDUMP text parsing and the previously validated independent
occupation/letter Hamiltonian oracle are shared. Rebuild all QE coefficients
and actions, spectrally replay every observed optimizer call, and derive the
screening/cache/circuit counters. Rebuild A-CASE pencils from occupation columns
and eigenvalues from an independent whitening solve. Validate the recorded
eigenframes by residuals and S-orthogonality, then re-evaluate all remaining
candidates and the selected 2-by-2 lowerings. Degenerate recorded eigenframes
are checked as valid frames; the native canonical gauge is not independently
replicated or advertised as cross-platform bitwise identity. Independently
rebuild the full bank pair/support ledger, including rejected-candidate work.
Energy/pencil tolerances are `1e-9` Ha (overlap `1e-10`), gradient/score/frame
tolerances `1e-8`; counters and declared policies must match exactly.

Persist complete observations before validation, and publish VALID, INVALID or
INCOMPLETE honestly. Commit `reference_results/phase17_more.json` alone after
execution, then require the declaration, independent replay, evidence and
history gates before changing the Phase 17 ledger or interpretation. Preserve
declaration -> execution -> record ancestry with a merge commit.
