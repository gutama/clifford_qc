# Q18-S3: certify on the ladder's Krylov basis, then allocate

## Status and disclosure

This is the fourth Q18-family declaration, after Q18's `INVALID` record,
Q18-S1's post-hoc uniform `NONE` and Q18-S2's `UNREACHED` screen. It is
declared before its own execution, but its **basis is selected post hoc**.
Q18-S2 found no prefix with a resolved variance on its two screened A-CASE
trajectories that certifies the Hubbard 2×2 ground state. A check after that
record found that the ladder's fixed Krylov basis `{H^k|ψ⟩, k ≤ 8}` does. That
is the only reason this basis is examined, and Q18, Q18-S1 and Q18-S2 stay
byte-for-byte intact.

What was known before this declaration, and is therefore disclosed rather than
predicted:

- every number in the Q18, Q18-S1 and Q18-S2 records, including each Hubbard
  2×2 Neyman ratio they report;
- the post-record statevector check of the order-8 Krylov basis
  (`tests/test_phase15_certifying_screen.py`): `E = −10.100105 t`,
  `σ = 0.0856 t`, `E + σ − E₁ = −0.208 t`, at retained rank 8 of 9;
- the committed ladder row: energy `−10.100105016587381 t`, retained rank 8,
  condition number `9.66e9`;
- the first-moment structure frozen in the config's
  `measured_before_freezing`. It fixes every prefix's `(S, H)` ground energy,
  rank and normalized-overlap extremes;
- on a Krylov basis the second-moment block of prefix `M` is a submatrix of the
  first-moment block of prefix `M + 1`. So the moments `⟨ψ|H^p|ψ⟩`, `p ≤ 17`,
  computed for the `M = 9` first-moment solve, fix every second moment below
  `M = 9`. They were printed during that structural work. No `σ` was formed
  from them;
- development on two undeclared systems, described below.

**Not computed before this declaration.** No `SecondMomentBank`, residual
functional, grouping of a combined universe, setting variance, cost ratio or
pilot on any Hubbard 2×2 Krylov prefix. No `σ` on any prefix other than the
order-8 basis's post-record statevector value.

**What the first moments already decide.** Only `M = 6, 7, 8, 9` lie below the
sector midpoint `(E₀ + E₁)/2`, which every certifying prefix must. `M = 8` keeps
its whole block with a smallest normalized-overlap eigenvalue of `1.47e-10`,
1.47 times the package's floor. `M = 9` keeps only 8 of 9 directions. Both
therefore fail the rank margin below under any margin above 1.47, and the domain
is a subset of `M = 6` and `M = 7`. `M = 6` certifies only if `σ < 0.159 t`,
while its energy alone forces `σ ≥ 0.148 t`.

**Development systems.** The undeclared Hubbard 1×4 chain (8 qubits) was used to
calibrate the eighty-digit validator, the independent routes, the rank margin
and the practical stage's run time. Its prefix statuses were displayed under the
margin of 100 then in use. The one priced prefix that certified (`M = 6`) read
PROHIBITIVE. Of the priced prefixes that did not, `M = 3` and `7` read
AFFORDABLE and `M = 2, 4` and `5` PROHIBITIVE. No ratio value was read. The
margin was then lowered to one decade, because Q18-S1's fixed ladder also passed
there at a condition number of `2.7e8`. On the declared basis the two margins
admit the same prefixes. A last end-to-end dry run of the producer and checker
on the chain, which printed only timings and checker problems, did not reach the
practical stage, so its oracle verdict there was not `OPEN`. The undeclared
Hubbard dimer was used to build the
practical stage. There a pilot that misjudged small setting variances by up to
25 times left the achieved `σ` spread at 2.6 times the target, while the
reported standard errors stayed honest (coverage 0.955). That fixed the
effective-cost statistic and the z-spread validation below; a rule on the raw
spread, tried first, was dropped.

## Why certify first, then allocate

A measured `σ` is worth its shots for convergence reporting only where the exact
interval `[E − σ, E + σ]` certifies the ground state. Q18-S2 showed that the
committed A-CASE prefixes never reach that with a resolved variance. The Krylov
sequence does, but Krylov overlaps are ill-conditioned by construction. So this
declaration first establishes, on each prefix and by three routes, that the
variance is resolved and that the interval certifies. It then checks that the
linearization Q18's estimator relies on holds there. Only then does it price
`σ`, first against the oracle bound and then, if that bound leaves room, under a
practical allocator.

## Question

On every prefix of the committed Hubbard 2×2 Krylov sequence
`{H^k|ψ⟩, k ≤ 8}` that meets three conditions:

- its exact Weinstein interval certifies the sector ground state;
- its variance is resolved and independently reproduced;
- its overlap keeps the whole block with a decade to spare;

can `σ` be estimated from shared QWC word means, at the energy's standard error,
for at most ten times the energy's shots? This is asked first under any
allocation, by the Neyman ratio. Only where that bound leaves room is it asked
again, under a declared pilot-based allocation validated by seeded sampling.

## System, basis and prefixes

- **System.** Q18-S2's `hubbard_2x2`, unchanged: open 2×2, `t = 1`, `U = 4`,
  `μ = U/2`, interleaved Jordan–Wigner, reference occupying `(0, 3, 4, 7)`.
- **Basis.** The ladder's `krylov` row on that rung: `I, H^1, …, H^8`, built by
  `krylov_response`. Its order-8 solve reproduces the committed energy to `1e-9`
  and its retained rank, and the gate checks both.
- **Prefixes.** Every `M` from 1 to 9.
- **Rows.** `KrylovBank` reads `A_i' A_j = H^(i+j)`, `A_i' (H A_j) = H^(i+j+1)`
  and `A_i' (H² A_j) = H^(i+j+2)` from one power table, and `SecondMomentBank`
  uses it unchanged. The identity is exact operator algebra. `H`'s coefficients
  are 1/2, 1 and 4, so every power through `H^17` is exact in double precision.
  The rows are therefore the base bank's products to the bit on every prefix
  the rank margin admits. The dimer tests check that to the bit. The bank's own
  product route would take about 30 s per pair here.

## Stage 1 — certification

A prefix is **certifying** exactly as Q18-S2 defines it: its variance is
resolved (`σ² > RESOLUTION · cancellation_scale`, `SecondMomentBank`'s floor),
`E + σ < E₁`, and `E ≥ E₀` to 100 ulp. `E₀` and `E₁` are the exact sector ground
and first distinct excited energies.

On every priced prefix `σ²` and `E` are established twice more:

- by `DecimalPencil`, the same rows and means at eighty digits, with no
  truncation;
- by the normalized Ritz state of the statevector Krylov pencil,
  `‖(H − E)Ψ‖²`, which never touches the word rows.

Both must reproduce the double-precision `σ²` to `1e-6` relative and `E` to
`1e-9` per unit `|E|`. A failure makes the prefix INVALID, not merely
uncertified.

## Stage 2 — rank margin and linearization

**Rank margin.** A prefix is priced only when `solve_projected` keeps its whole
block at default thresholds with a decade to spare on both retention rules:

- the smallest normalized-overlap eigenvalue is at least `10 τ_S = 1e-9`;
- the normalized overlap's condition number is at most `1e12 / 10`.

Q18's linearization assumes that the retained block does not move. A prefix that
fails the margin is `RANK_UNSTABLE`: its interval is reported, but it is unpriced
and outside the domain. A rank-stable prefix whose variance is not resolved is
`UNRESOLVED`, as in Q18-S2.

**Eighty-digit validation (declared).** Q18-S1's two directions and seed are
kept. Along each, the exact-mean pencil is formed at eighty digits and its
ground root found by Cholesky and shifted inverse iteration. The central
difference of `E` and of `σ²` is taken at `t = 1e-30`. Both `ritz_functional`
and Q18's residual functional must match it within `1e-6` of their own norm. At
eighty digits truncation and rounding are both tens of orders below that
tolerance, so the check tests the analytic functionals and nothing else.

**Fixed-step validation (declared).** Q18-S1's four-step Richardson rule is kept
as well, with its steps of `2e-3` to `2.5e-4` in word-mean units. Every endpoint
must keep the whole block and a nondegenerate ground root. This is the
double-precision pipeline's own derivative at finite perturbations.

## Stage 3 — oracle bound (decision)

This stage is Q18-S2's, unchanged. The estimand, the residual functional with
its Ritz-vector response, the measured universes, both QWC protocols
(`qwc_groups` declared, `qwc_basis_cover` alternative) and the Neyman ratio
`N = (Σ_g √v_g^U)² / (4σ² (Σ_g √v_g^SH)²)` are all as Q18-S2 froze them.
Each priced prefix gets Q18's ladder on both ratios:

- `AFFORDABLE` when both are at most 10;
- `PROHIBITIVE` when both exceed 10;
- `GROUPING_SENSITIVE` otherwise;
- `INVALID` on any failed deterministic check.

To first order, `N` bounds every allocator of residual shots from below,
against the energy's own optimal cost. Q18's uniform `R` and Q18-S1's integer
oracle allocation are reported beside it, and no clause reads them.

## Stage 4 — practical allocation (only if the oracle verdict is OPEN)

The stage runs on every `AFFORDABLE` certifying prefix and on no other, under
`qwc_groups`. The fixed parameters are:

- seed `20261014`;
- 100 replicates, each spawning independent energy and residual streams;
- target standard error `s = σ/100` on both sides.

Within each replicate:

1. **Sampling.** Each setting's shots are one multinomial draw from the exact
   grouped outcome distribution of the reference, through
   `FiniteShotBackend`'s own grouped-sampling plans.
2. **Pilot.** Each side spreads 10% of its oracle Neyman production budget
   uniformly over its settings, never fewer than 64 shots per setting. Pilot
   outcomes never enter an estimate.
3. **Plug-in.** The functional weights come from the pilot's means. The
   per-setting variances are the unbiased sample variances of the pilot's
   shot values. A pilot pencil that is not full rank fails the replicate.
4. **Allocation.** Q18-S1's integer Neyman rule on the pilot's variances, with a
   floor of one shot per setting. The residual side divides by `4σ̂_pilot²`, with
   `σ̂_pilot²` clipped below at `s²`.
5. **Estimate and interval.** Production shots alone give the estimate. Its
   delta-method standard error uses production-mean plug-in weights and
   production per-setting sample variances.
6. **Effective cost.** Pilot shots plus production shots times `(se/s)²`: the
   shots this allocation needs to reach `s`, charging a misallocated pilot.

**Statistic.** The expected effective cost ratio is the mean residual effective
cost over the mean energy effective cost across replicates. At most 10 is
`PRACTICAL_AFFORDABLE`.

**Confidence family.** Two-sided 95% Wald intervals for `E` and for `σ`,
separately. A one-sided 95% upper bound `E + σ + 1.645 √(se_E² + se_σ²) < E₁`
is reported as a certification rate, and no clause reads it. No simultaneous
coverage is claimed.

**Validation.** All of the following must hold, or the prefix is
`VALIDATION_FAILED`:

- every replicate returns;
- each interval covers the exact value in at least 90 of 100 replicates;
- the standard deviation of `(estimate − exact)/se` stays within 1.25 on each
  side.

## Lineage

Any failure of these makes the verdict INVALID:

1. the sector spectrum equals Q18-S2's recorded spectral reference to `1e-12`;
2. Q18-S1's frozen nine-generator bank, priced again through Q18-S2's own path,
   passes Q18-S2's lineage rule against Q18-S1's committed entry. That ties the
   shared grouping, variance and functional code to its committed numbers;
3. the order-8 prefix reproduces the ladder row's energy and retained rank.

## Decision rule

| oracle verdict | condition |
|---|---|
| `INVALID` | the lineage fails, or a certifying priced prefix is `INVALID` |
| `UNREACHED` | no priced prefix certifies |
| `OPEN` | some certifying priced prefix is `AFFORDABLE` |
| `CLOSED` | every certifying priced prefix is `PROHIBITIVE` |
| `GROUPING_SENSITIVE` | otherwise |

The final verdict is the oracle verdict unless that is `OPEN`. When it is OPEN,
the practical stage decides:

- `LICENSED` if any practical prefix is `PRACTICAL_AFFORDABLE`;
- else `VALIDATION_FAILED` if any failed validation;
- else `PRACTICAL_PROHIBITIVE`.

Priced prefixes outside the domain are reported as diagnostics and cannot move
either verdict.

## Consequences

- **`CLOSED`.** Under Q18's partitions and matched-precision statistic, no
  allocation, pilot-estimated or oracle, measures `σ` for ten times the
  energy's optimal cost on any certifying, rank-stable prefix of this Krylov
  sequence. The pilot experiment is not run. Phase 15's practical-allocation
  follow-up closes negative **on this declared scope only**: this system, these
  prefixes, single-assignment QWC and a matched standard error. Exact residuals
  and convergence reports remain available.
- **`UNREACHED`.** No rank-stable prefix certifies with a resolved variance. The
  question is unanswered on this basis family.
- **`LICENSED`.** A pilot-allocated residual estimator is licensed on the
  `PRACTICAL_AFFORDABLE` prefixes by name, with the declared intervals.
- **`PRACTICAL_PROHIBITIVE`.** The oracle bound leaves room, but this allocator
  does not reach it. The follow-up closes negative for this allocator only.
- **`VALIDATION_FAILED`.** The allocator's intervals are not validated. No
  estimator is licensed.
- **`GROUPING_SENSITIVE`.** No practical experiment until a declaration fixes one
  grouping.
- **`INVALID`.** No verdict. The record names the failed check.

A scoped negative is a scientific outcome of this declaration, recorded as such.
It is not a lifecycle value for Phase 15, whose implementation fraction stays
separate, while folded-spectrum roots and other bank scopes remain open.

## Not claimed

The following are outside this declaration:

- fully or block-commuting grouping;
- pooled or multiply-assigned estimators, and shot sharing between the two
  campaigns;
- any other allocator, including sequential top-ups;
- the plug-in bias of `σ²` beyond what the simulation shows;
- device noise, readout error and device time;
- other systems, other bases, the rank-unstable prefixes and excited roots.

## Execution

Land this declaration with its config, gate, helper module, producer,
independent checker and their tests on the undeclared Hubbard dimer. Then run
once from a clean tree in the pinned record environment, commit the record
alone, and only then update the ledger and the reproduction notes. The config's
last commit must precede or be the record's execution commit, and that must
strictly precede the record's first commit. Input hashes stay binding.
Implementation hashes lapse once the record exists, since the record names its
own commit.
