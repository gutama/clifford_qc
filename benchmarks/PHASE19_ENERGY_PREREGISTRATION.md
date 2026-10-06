# Phase 19 lever 1: fixed-reference Hamiltonian-energy measurement

## Status and authority

This is a result-free declaration. No Phase 19 outcomes have been sampled, and
no reference energy, variance, confidence crossing or device runtime has been
evaluated for this campaign. Only Hamiltonian coefficients, covers, rotations,
readouts, circuit resources and allocation schedules have been rebuilt.

The binding protocol is `configs/phase19_energy_comparison.json`. Its three
structural bank manifests are `configs/phase19_banks/{h4,lih,beh2}.json`.
The JSON, rather than this explanatory note, fixes every rule and tolerance.
`check_phase19_preregistration.py` pins its canonical SHA-256, rejects outcome
fields and unknown changes, verifies file hashes, and rebuilds the structures.

The compilation parent is PR #124's merge,
`45966af20ca99f99e70daa0e075f456e48e1f71e`. Its §3.6 invariant tests passed
before this declaration. The three molecular fixtures were chosen in `PLAN.md`
using an exploratory setting-count probe; that selection and the existing
RHF/FCI energies in their provenance were already known. The old cover counts
were not reproducible under a declared ordering and are not used here. No claim
that cliques will improve shot cost or runtime is made before this execution.

## Question and scope

For the same fixed Hamiltonian and reference state, does the frozen clique arm
need at least two times fewer effective shots than frozen QWC to certify its
reference energy within `1.6e-3` Hartree? What logical runtime does each endpoint
imply under each of the three existing illustrative device cards?

This measures one fixed linear functional, `Tr(rho H)`. It does not estimate a
ground eigenvalue, a Ritz derivative, projected matrix elements, or the
individual word means of a clique. No Phase 14b word bank, contextual restriction,
operator pool, state optimization or additional instance is admitted.

## Hamiltonians and reference states

| id | committed input | active space | nonidentity terms | QWC settings | clique settings |
|---|---|---|---:|---:|---:|
| `h4` | `data/h4_sto3g_r0.9.FCIDUMP` | 4e, 4o; 8 qubits | 184 | 68 | 45 |
| `lih` | `data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP` | 4e, 4o; 8 qubits | 192 | 42 | 54 |
| `beh2` | `data/beh2_sto3g_r1.3264.FCIDUMP` | 4e, 4o; 8 qubits | 60 | 9 | 36 |

All use the existing restricted FCIDUMP-to-Jordan–Wigner builder with
`integral_tolerance=1e-12`. This upstream integral/assembled-term tolerance is
explicit; after assembly, both arms retain every exactly nonzero nonidentity
coefficient. The code-zero coefficient is an exact offset, receiving no shots
and no confidence radius. There is no further coefficient cutoff and no
analytic nonidentity shortcut for a known Hartree–Fock mean.

Each state is the fixture's Hartree–Fock determinant, occupying interleaved
spin orbitals `[0,1,2,3]`, with four electrons and `MS2=0`. Apply `X` on those
qubits to `|00000000>`. The basis is `|q0 q1 ... q7>` with `q0` leftmost; a
packed Pauli code instead places `q0` in its lowest two-bit lane.

These simple states delimit the first comparison. A result on them will not
establish performance on correlated variational or ground states. Their
energies are classically accessible; this is measurement-capability validation,
not a computational advantage experiment. Changing the states requires a new
declaration before new outcomes.

## Frozen compilation and readout

**QWC.** Pass ascending packed codes to `qwc_groups`. Its descending QWC
conflict degree has stable code-ascending ties and first-compatible placement.
Sort the members within each group by code. `shared_basis` supplies local
readout changes in ascending qubit order: `h` for X, `sdg` then `h` for Y,
and no gate for Z. Each shot's contribution is the weighted sum of all assigned
word parities. Every word is assigned exactly once; no pooling is used.

**Clique.** Use `commuting-degree-desc/code-asc/first-compatible-v1`, as shipped
in `measurement/cliques.py`. Code order fixes the pivot and Givens sequence.
The signed generator `i A1 Ak` is represented by an unsigned Pauli word with
its sign folded into the IR angle. The multiword rotation sends the unit
weighted combination to `+A1`; the negative-pivot pi rotation is retained.
A singleton is unrotated and keeps the sign in its measured coefficient.

Lower every rotor with the existing parity-ladder `lower_rotor`, then apply
the pivot's local readout changes. Measure all eight output Z bits. A clique
contribution is its signed weight times the parity on the pivot support.
No readout CX ladder is added. The clique's individual word means cannot be
recovered from that observation.

Manifests bind the coefficients in hexadecimal float form, the exact
partitions, signed rotor angles, readout supports, gate/qubit topology hashes
and per-setting resource vectors. Angles have a separate numerical comparison
(`rtol=1e-13`, `atol=1e-14`); omitting them from the topology hash avoids a
libm last bit changing a gate-topology identity. All keys, integer fields,
resource counts, supports and coefficients remain exact. The manifest files
themselves are hash-bound and cannot be refitted within that tolerance.

## Resources and device boundary

Counts come from the complete emitted measurement circuit, including local
readout changes. One shared per-qubit clock schedules each operation at its
earliest type-homogeneous layer after all its predecessors. It preserves
dependencies through both gate kinds and permits disjoint gates to share a
layer. It claims a feasible schedule, not optimal scheduling or synthesis.

The following are sums across a single sweep's settings, not the depth of one
circuit or a runtime:

| bank / arm | sum `n_1q` | sum `n_2q` | sum `d_1q` | sum `d_2q` |
|---|---:|---:|---:|---:|
| H4 / QWC | 480 | 0 | 126 | 0 |
| H4 / clique | 1693 | 1236 | 679 | 1230 |
| LiH / QWC | 319 | 0 | 78 | 0 |
| LiH / clique | 1444 | 906 | 655 | 898 |
| BeH2 / QWC | 88 | 0 | 16 | 0 |
| BeH2 / clique | 322 | 166 | 120 | 164 |

All three existing cards are frozen by file hashes: `ion-like`,
`logical-alltoall`, and `superconducting-like`. At each finite certified
effective endpoint, apply `inflate_shots_for_fidelity`, using
`ceil(n_g/F_g^2)`, then `cost_schedule` with the same resource vector.
If any setting falls below a card's fidelity floor, report that arm/card as
inadmissible with no `C_time(epsilon)`. It does not invalidate the ideal-shot
comparison and does not authorize dropping a setting.

These are illustrative logical prices. The cards add their existing routing,
readout, reset and preparation charges; each arbitrary `rz` is one generic 1q
operation. Ideal reference availability is assumed and `t_prep` is charged
per shot. The four reference X gates and their errors are not separately added
to the measurement vector. No fault-tolerant T synthesis, routed layout,
calibration, mitigation or noisy-hardware certificate is supplied. Fidelity
inflation is a cost surrogate, not a proof that noisy outcomes satisfy the
ideal confidence intervals. No additional outcomes are drawn per device card.

## Estimator, allocation and fixed endpoints

For QWC, a setting contribution is
`Y_g=sum_w h_w parity_w` with `|Y_g|<=L_g=sum_w |h_w|`.
For a clique it is `Y_g=weight_g parity_pivot`, with
`L_g=abs(weight_g)` (the coefficient L2 norm except for a signed singleton).
The estimate is the exact identity offset plus independent setting sample
means. Preserve joint QWC outcomes when computing a contribution or variance.

Both arms use coefficient-range Neyman allocation. Give every setting two
shots, then apportion the remaining total in proportion to its frozen `L_g`.
Convert binary float scores to exact `Fraction` values for largest-remainder
apportionment; equal remainders use the frozen setting index. Neither sampled
nor oracle variance, pilot outcomes or later endpoint data can alter allocation.

The effective-shot grid is every power of two from `2^16` through `2^32`,
17 endpoints. Draw every endpoint on an independent fixed schedule, even when
an earlier endpoint has certified. There is no cumulative reuse, adaptive
allocation, replacement seed or extra endpoint. Effective/ideal sampled shots
are distinct from the cards' modelled inflated raw shots.

Use NumPy `PCG64` with `SeedSequence(root, spawn_key=(system_index, arm_index,
endpoint_or_replica_index, setting_index))`. Order systems H4, LiH, BeH2;
arms QWC, clique; endpoints ascending; settings as in the manifests. The
headline root is `190814000`; the audit root is `190914000`. They are disjoint
from older config namespaces. The future execution uses the pinned environment
of `reference_results/protocol_cost.json` and records its own full provenance.

## Confidence and numerical allowance

Use the existing AMS two-sided empirical-Bernstein bound on each scalar
contribution, with range `2 L_g` and unbiased sample variance `s_g^2`:

```text
delta_g = 0.05 / (3 systems * 2 arms * 17 endpoints * G_this_arm)
r_g = sqrt(2 s_g^2 log(3/delta_g)/n_g) + 3 (2 L_g) log(3/delta_g)/n_g
r_total = sum_g r_g + 1e-10 Hartree
```

The unbiased variance is at least the theorem's variance with denominator
`n_g`, so this substitution is conservative. Sum group radii without clipping.
A union bound covers all six system/arm pairs and the 17 fixed looks at
overall `delta=0.05`. This is a fixed-grid guarantee for ideal bounded outcomes,
not an anytime bound or a hard gate on the finite-shot spin-factor ball.

The total target is `0.0016` Hartree. Reserve `1e-10` for deterministic numerical
bias, so the stochastic radius must be at most `0.0015999999`. The future law
gate compares its mean to the exact rational Hartree–Fock functional formed
from the frozen coefficients; that energy has not been evaluated here.

Reject raw probabilities outside `[-1e-12,1+1e-12]` or normalization error
above `1e-12`. Clip negative roundoff to zero and normalize once, logging the
clipped mass and normalization change. Apply no nonzero-probability cutoff.
After correction, the exact law mean must satisfy the `1e-10` energy gate.
The allowance does not repair a failed gate or relax an efficiency threshold.

## Later oracle, reconstruction and covariance gates

The future producer executes each frozen circuit on the fixed exact reference
statevector and samples multinomial histograms from its all-eight-qubit Z law.
This is an oracle simulation of the measurement protocol. Stim alone cannot
carry the arbitrary clique rotations. Circuit probabilities are diagnostics
and sampling laws; they never select a cover or improve allocation.

Before the campaign, an independent dense gate interpreter must reconstruct
the full Hamiltonian from each arm's weighted diagonal readouts to maximum
absolute entry error `<=1e-10` Hartree. Each unitary must satisfy a max-entry
unitarity error `<=2e-12`. The corrected circuit laws must reproduce the exact
reference functional within the reserved allowance. These molecular operator
and state-law checks belong to the later implementation, not this declaration's
structural gate.

Audit all six system/arm cells with 1000 independent replicas each at `2^18`
effective shots, using the same range allocation and the audit root. Predict
energy-estimate variance from the exact executed circuit laws as
`sum_g Var(Y_g)/n_g`; compute empirical replica variance with `ddof=1`.
Require every empirical/predicted ratio in `[0.8,1.2]`. If the prediction is
exactly zero, require exactly zero empirical variance and report a null ratio.
Never divide by zero. These exact variances are audit-only and were not
computed before the declaration.

## Decisions and reporting

For each arm/instance, the certified endpoint is the first frozen grid point
with `r_total<=0.0016` Hartree; otherwise it is right-censored above `2^32`.
The decision concerns this grid and allocator, not a continuous optimal cost.

| endpoints | per-instance shot decision |
|---|---|
| both finite, clique `<=QWC/2` | `MATERIAL_REDUCTION` |
| both finite, clique `>QWC/2` | `NO_MATERIAL_REDUCTION` |
| QWC censored, clique finite and `<=2^31` | `MATERIAL_REDUCTION_CENSORED_BASELINE` |
| QWC censored, remaining cases | `INDETERMINATE` |
| clique alone censored | `NO_MATERIAL_REDUCTION` |
| both censored | `INDETERMINATE` |

Any blocking failure makes the entire campaign `INVALID`, suppressing all
efficiency decisions while retaining the failed cells and diagnostics. Do not
rerun or replace a seed. Device inadmissibility is reported separately per card.
There is no pooled instance result, global winner or cross-card claim.

Report every endpoint's allocation, estimate, stochastic and total radii,
exact-reference error and censoring; all six audits; operator/law checks;
all resource vectors and device admissions; certified effective and modelled
raw shots; and three separate card prices. Audit shots and classical
compilation/simulation time are reported separately, not hidden in the
certified-endpoint price. There is no end-to-end advantage claim.

## Execution order

Merge this declaration without samples. In a later commit, implement the
producer and independent record checker and validate them on undeclared toy
fixtures. Only then execute the frozen campaign once from a clean pinned tree.
Commit `reference_results/phase19_energy_comparison.json` alone, after the
execution-source commit, then update status and reproduction notes.

The record must bind this declaration's canonical digest and name both the
full execution commit and the preregistration merge commit. The gate requires
strict ancestry: declaration source, its merge, execution source, first record.
A squash/rebase that erases this order fails. When a record exists, missing or
shallow history fails rather than skipping; CI uses full history for this gate.
Input, implementation and manifest hashes remain binding. Changes to the
functional, state, cover, allocator, confidence, numerical policy, grid, audit
or cards require a new disclosed declaration before new outcomes.

## Method references

- Izmaylov, Yen, Lang and Verteletskyi, *Unitary partitioning approach to the
  measurement problem in the Variational Quantum Eigensolver method*,
  [arXiv:1907.09040](https://arxiv.org/abs/1907.09040). This is the existing
  measurement method being implemented, not a novelty claim.
- Audibert, Munos and Szepesvári, *Exploration–exploitation tradeoff using
  variance estimates in multi-armed bandits* (2009), Theorem 1,
  [author manuscript](https://imagine.enpc.fr/publications/papers/TCS08.pdf).
  The fixed-sample two-sided bound is the repository's confidence primitive.
