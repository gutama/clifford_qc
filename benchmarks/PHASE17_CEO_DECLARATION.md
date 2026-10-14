# Phase 17 CEO pool comparison

The active binding design is `configs/phase17_ceo_review.json`.
The original `configs/phase17_ceo.json` and its record remain unchanged for audit.
Review of PR #131 identified floating-point tie selection, a very small parent
cutoff, redundant optimizer callbacks and incomplete convergence checks. The
active design corrects these before a separately committed, single campaign.
Original outcomes were already known: this is a post-review diagnostic, not
a blind preregistration, an independent statistical replica or a robustness survey.

The original design was `configs/phase17_ceo.json`. The declaration and its
outcome-free gate precede the full execution record in commit history.
Three-iteration BeH2 development tests preceded this freeze; their purpose was
variational and accounting validation. No full trajectory, target crossing or
comparative resource outcome was inspected. The original local design was labelled
`declared_after_bounded_development_tests`. The original publication design is
labelled `publication_replay_of_frozen_local_design`, as explained below;
neither is advertised as a blind preregistration.

## Operators and comparison

The primary source is Ramôa et al., *npj Quantum Information* **11**, 86
(2025), https://doi.org/10.1038/s41534-025-01039-4, especially Eqs. 23--29.
Implementations here were derived from occupation action and those definitions;
the authors' code is not a runtime dependency.

`Q_j=(X_j+iY_j)/2` annihilates a qubit occupation without a Jordan--Wigner
parity string. A QE is `T=Q†_a Q_i-h.c.` or
`T=Q†_a Q†_b Q_i Q_j-h.c.`. Enumerate generalized singles and all balanced
complementary partitions of four distinct sites. With interleaved spins, each
partition preserves the number of alpha and beta occupations separately.
Every pair of same-support double QEs contributes `T_1+T_2` and `T_1-T_2`
to the OVP-CEO pool. Singles enter both pools. These are anti-Hermitian operators;
the package's Hermitian equivalent is `G=2iT`, so `exp(theta T)=exp(-i theta G/2)`.
No total-spin or point-group restriction is imposed.

The arms use the same exact dense-sector solver and L-BFGS-B optimizer:

| arm | selection | appended parameters |
|---|---|---|
| `qe` | largest absolute QE derivative | one QE |
| `ovp_ceo` | largest absolute OVP derivative | one OVP |
| `mixed_ceo` | largest absolute OVP derivative | that OVP if one same-support QE derivative is resolved; otherwise all resolved QE parents, separately parameterized |

Mixed-parent derivatives are reconstructed from the already screened plus/minus
OVP derivatives. The active promotion cutoff is `1e-7` (ten times optimizer `gtol`, one tenth
of the pool stopping threshold). At least two parents must exceed it to be
promoted; otherwise keep the selected OVP. This is a declared numerical policy,
not a certificate of finite-shot uncertainty or optimizer residual resolution.
The original `1e-12` policy is retained only in the archived record. All candidates remain
available for reuse; no reference-only or observed-outcome pool pruning occurs.
Selection takes the first pool member within an absolute derivative tolerance
of `1e-10` of the maximum. Both producer and checker enforce this order.
The original record used an exact floating-point maximum, which can break
mathematical ties differently across CPU kernels; it is not evidence for the
corrected policy or for cross-platform bitwise identity.
Each arm performs at most 32 growth rounds and 350 optimizer iterations per
round. The full gradient-vector norm threshold is `1e-6`; optimizer `gtol` is
`1e-8`. New parameters start at zero and all retained parameters are reoptimized.
The ground-state oracle is a diagnostic and a numerical bound check, not an
operator-selection input or an energy stopping rule.

## Inputs and readout

H4 and BeH2 are required, using the committed eight-qubit FCIDUMPs. LiH is a
reference-floor diagnostic: the source choice is retained even if its reference
already reaches the target. All input digests, pool coefficients, signs, order,
reference occupations and numerical options are frozen in the config.
No reference spectrum or comparison outcome is contained in that config.

Read the first retained prefix whose energy error is at most `1.6e-3` Ha.
Report a reference already meeting the target separately, and report an unreached
target as right-censored. Do not interpolate or pool a winner across systems.

The record retains every screened score summary, optimizer call (parameters,
energy and analytic gradient), callback-to-computation indices, final parameters at every prefix, stopping reason,
and derived counters. It separately reports gradient forms, distinct commutator
words per reference state, energy/gradient optimizer calls, parameter derivatives,
dense operator bytes, and a generic Pauli-ladder CNOT upper bound. No weighted
sum of these currencies is a total-cost estimate. Active optimizer counts
charge distinct computed parameter tuples within each round; repeated callback
requests, including the common adapter's final request, are cached and reported
separately. The original counts charge actual calls including one repeated final
evaluation per growth round. The active resolution flag tests max absolute
gradient against `gtol`; the archived flag tests the stricter L2 norm. Neither
flag records a SciPy termination message. Dense bytes count the H and generator
matrices required by that arm, not process RSS; per-case preparation and its
cache are shared across arms. The CNOT upper bound uses each
commuting Pauli term's independent parity ladder; optimized CEO circuits and
the published paper's enhanced subroutines are not implemented or priced.

## Independent verification and execution

The scientific replay calls neither producer, exchange solver, native Pauli
product nor FCIDUMP-to-JW mapper. FCIDUMP text parsing and digest utilities are
shared, so parser defects remain a shared boundary. Occupation swaps rebuild
QE matrices; sums/differences rebuild CEOs; independent fermionic occupation
signs build sector H. A separate letter-product implementation reconstructs
Hamiltonian words and every pool coefficient, including the eight-qubit pools,
before accounting. Spectral exponentials replay each computation, with energies
checked at `1e-9` Ha and gradients at `1e-8`. Word counts preserve the declared
staged `1e-12` pruning. The structural identity gate separately calls the native
pool builder; its agreement alone is not an independent coefficient check.
Counts must be integers derived from the retained trace. This checks an observed
optimization trajectory, not that optimization found a global minimum.

Default producer invocation only validates the declaration and structural inputs.
`--execute` requires clean committed sources, the declared numerical environment,
one BLAS thread and an execution commit strictly after the declaration. It
claims one persistent exclusive execution slot before solving and publishes the
complete record atomically without overwrite after independent verification.
The declaration must precede execution, which must precede the record's first
commit. The record also binds its producer and checker source digests. After execution,
bindings are verified at that historical commit, allowing current shared core
modules to evolve. Default unit tests check static declarations without Git or
current-code freezing. Dedicated gates audit last config change < execution <
first record commit using the existing lineage helpers. Empty logs in complete
history fail; exported trees and shallow boundaries explicitly report provenance
as unverified. Preserve source ancestry on merge; a squash cannot prove lineage.

This is exact small-sector simulation evidence. It carries no finite-shot
certificate, device runtime, enhanced CEO-ADAPT-VQE* or quantum-advantage claim.
The dedicated MORE-ADAPT multi-root comparison remains separate Phase 17 work.

## Publication replay

The original local declaration/execution/result history is preserved byte-for-byte
in `migrations/phase17_ceo_local_history.bundle`. Its manifest identifies the
original config, source, record and scientific-case digests. The full local
comparison was already observed before API publication, whose commit identities
differ from local Git commits. The published source is therefore replayed without
changing any numerical option or candidate/input design, and every scientific
and resource field must match the original case digest exactly. This is a
provenance/reproducibility replay, not another independent statistical replica or
a blind preregistration. The frozen config discloses this explicitly.

To inspect the original local execution in a clone with the base history:
`git fetch benchmarks/migrations/phase17_ceo_local_history.bundle feat/phase17-breadth`.
The original record binds the actual published source and its own record commit.
