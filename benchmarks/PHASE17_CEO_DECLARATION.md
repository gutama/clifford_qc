# Phase 17 CEO pool comparison

The binding design is `configs/phase17_ceo.json`. The declaration and its
outcome-free gate precede the full execution record in commit history.
Three-iteration BeH2 development tests preceded this freeze; their purpose was
variational and accounting validation. No full trajectory, target crossing or
comparative resource outcome was inspected. The design is explicitly labelled
`declared_after_bounded_development_tests`, not a blind preregistration.

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
OVP derivatives. The parent cutoff is frozen at `1e-12`. All candidates remain
available for reuse; no reference-only or observed-outcome pool pruning occurs.
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
energy and analytic gradient), final parameters at every prefix, stopping reason,
and derived counters. It separately reports gradient forms, distinct commutator
words per reference state, energy/gradient optimizer calls, parameter derivatives,
dense operator bytes, and a generic Pauli-ladder CNOT upper bound. No weighted
sum of these currencies is a total-cost estimate. The CNOT upper bound uses each
commuting Pauli term's independent parity ladder; optimized CEO circuits and
the published paper's enhanced subroutines are not implemented or priced.

## Independent verification and execution

The scientific replay calls neither producer nor exchange solver. The
structural identity gate separately rebuilds the frozen pool manifests. It reconstructs QE
matrices from occupation swaps, CEO matrices by addition/subtraction, the
Hamiltonian from independent Kronecker matrices, and exact sector spectra. Dense
spectral exponentials replay every optimizer call and trajectory, with energies
checked at `1e-9` Ha and gradients at `1e-8`. Independent letter multiplication
rebuilds screening word counts under the substrate's staged `1e-12` pruning.
Counts must be integers derived from the retained trace. This checks an observed
optimization trajectory, not that optimization found a global minimum.

Default producer invocation only validates the declaration and structural inputs.
`--execute` requires clean committed sources, the declared numerical environment,
one BLAS thread and an execution commit strictly after the declaration. It
claims one persistent exclusive execution slot before solving and publishes the
complete record atomically without overwrite after independent verification.
The declaration must precede execution, which must precede the record's first
commit. The record also binds its producer and checker source digests.

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
The active record binds the actual published source and its own record commit.
