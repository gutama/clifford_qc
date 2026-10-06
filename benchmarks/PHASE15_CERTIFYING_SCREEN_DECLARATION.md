# Q18-S2: oracle allocation screen on ground-certifying Hubbard 2×2 bases

## Status and disclosure

This is a third Q18-family declaration, after Q18's `INVALID` record and
Q18-S1's post-hoc uniform `NONE`. It is declared before its own execution, but
its system is **selected post hoc**. Q18-S1's oracle allocation diagnostic
gave Hubbard 2×2 integer cost ratios of 8.43 and 8.12 against the tenfold bar,
and every molecular bank above 700. Hubbard 2×2 is the only bank this screen
examines for that reason. Q18 and Q18-S1 stay byte-for-byte intact.

What was known before this declaration, and is therefore disclosed rather
than predicted:

- Q18-S1's complete Hubbard 2×2 entry: on the frozen nine-generator bank the
  Ritz energy is `-9.2394 t`, `σ = 1.5198 t`, and the Neyman ratios are 9.03
  (`qwc_groups`) and 8.74 (`qwc_basis_cover`).
- The exact `(N, S_z)` sector spectrum, already frozen in Q17's config:
  `E₀ = -10.1027 t`, `E₁ = -9.8064 t`, gap `0.2963 t`.
- The committed energy histories of both trajectories below
  (`reference_results/acase_ladder.jsonl`).
- An informal extrapolation, stated before this declaration, that holds Q18-S1's
  per-shot variances fixed and scales its Neyman ratio as `1/σ²`. It predicted
  ratios in the hundreds wherever `σ` resolves the gap. It is a guess, not a
  measurement, and no clause reads it.

**No second-moment row, `σ`, residual functional, grouping of a combined
universe or cost ratio was computed on any prefix other than the frozen
nine-generator bank before this declaration was committed.** Structural
quantities were: both trajectories were rebuilt from their labels, their
`(S, H)` ground energies reproduce the committed histories exactly, and their
first nine generators are the frozen bank's.

## Why this screen comes before a pilot experiment

The frozen bank's Weinstein interval `[E - σ, E + σ] = [-10.76, -7.72]`
contains the six lowest sector levels. Its Ritz energy lies above `E₁`. A
measured `σ` there could be affordable and still certify nothing about the
ground state. Q18's statistic also carries an explicit `1/σ²`, so
affordability on a poorly converged basis says little about a converged one.
A pilot-driven allocation experiment on Hubbard 2×2 is worth declaring for
convergence reporting only if some basis that does certify the ground state
could pass at all.

## Question

Along the two committed Hubbard 2×2 A-CASE trajectories that extend the frozen
bank, is there a prefix that meets two conditions? Its exact Weinstein interval
must certify the sector ground state. And there, estimating `σ` from shared QWC
word means at the energy's standard error must cost at most ten times the
energy's shots, with *both* sides at their exact-variance optimal allocation.

## System, trajectories and prefixes

- **System.** `hubbard_2x2` exactly as `configs/mapping_axis.json` builds it:
  open 2×2, `t = 1`, `U = 4`, interleaved Jordan–Wigner, reference occupying
  `(0, 3, 4, 7)`.
- **Trajectories.** The two committed `acase_ladder.jsonl` rows on that rung
  whose first nine labels are the frozen bank's:
  - `acase_exact_m25`: levels 0–3, the 26-generator raw pool, 23 labels,
    stopped on its own lowering threshold;
  - `acase_level4`: levels 0–4, 161 candidates including competing-order
    configurations, 26 labels, ending at the exact ground energy.
  Generators are looked up by label in the ladder's own `build_candidates`
  pool. The labels and energy histories are frozen in the config; the gate
  requires the ladder rows to still carry them.
- **Prefixes.** Every `M` from 9 to each trajectory's final size: 15 prefixes
  on the first, 18 on the second. `M = 9` is the frozen bank on both.

No new growth rule, threshold, pool or prefix is chosen here.

## Domain: certifying prefixes

A prefix is **certifying** when its exact ground Ritz pair satisfies

`E + σ < E₁` and `E ≥ E₀` (to 100 ulp),

with `σ² = c'Kc − E²` from `SecondMomentBank` and `E₀`, `E₁` the exact sector
ground and first distinct excited energies. Weinstein puts an eigenvalue within
`σ` of `E`. Variationality and the first clause leave `E₀` as the only level
the interval can hold, so a measured `(E, σ)` brackets it as `[E − σ, E]`.
Because `σ² ≥ (E − E₀)(E₁ − E)` for a sector state with `E₀ ≤ E ≤ E₁`, a
certifying prefix is also ground-dominated in the sense the package's
convergence report uses. It must lie below the midpoint `(E₀ + E₁)/2`.

From the committed energies, that midpoint rule is already decidable in part.
No prefix of `acase_exact_m25` lies below it (its last is `-9.8475`, above
`-9.9546`). On `acase_level4`, prefixes 23, 24, 25 and 26 do. So the domain is
a subset of those four. Whether each certifies depends on `σ`, which this
declaration has not computed.

**The sector reference is the right one.** The Hamiltonian commutes with `N`
and `S_z`, and every generator maps the reference into its `(N, S_z)` sector,
so each Ritz state lies in that sector. The gate checks both to zero. The
level-4 configuration generators are not sector-conserving as operators: they
move other determinants out of the sector while mapping the reference into it.
Their operator leakage is reported and is not the premise.

**Unresolved prefixes are unpriced.** A prefix whose variance does not clear
`SecondMomentBank`'s resolution floor (`1e-12` of its cancellation scale) has
no linear regime for `dσ = dσ²/(2σ)`, and the statistic's `1/σ²` diverges
there. Such a prefix is reported as `UNRESOLVED`, is outside the domain, and
cannot move the verdict. The committed `acase_level4` energy at `M = 26`
equals the exact ground energy to `1e-14`, so that prefix is expected to be
unresolved.

## Statistic

The decision reads Q18's **Neyman ratio**, already reported by Q18 and Q18-S1:

`N = (Σ_g √v_g^U)² / (4 σ² (Σ_g √v_g^SH)²)`.

The `v_g` are exact per-setting variances of the energy's and the residual's
first-order functionals, on the reference, under single assignment. `N` is the
residual's production shots over the energy's at one matched standard error,
both at variance-optimal allocation across the declared settings. Under the
same partition, no allocation of residual shots costs less than its Neyman
optimum. So against the energy's own optimal cost, `N` is a first-order lower
bound for every practical allocator, pilot-estimated or otherwise. A pilot is a
fixed charge that does not scale with the precision, so it is excluded from the
decision. The threshold is Q18's: `N ≤ 10` is affordable.

Reported beside it and read by no clause: Q18's uniform `R`, and Q18-S1's
integer allocation at `s = σ/100` with 64 charged, unperformed pilot shots per
setting. The integer allocation is reported as totals, since the counts are a
ceiling of the recorded setting variances.

Everything else is Q18's and Q18-S1's, unchanged: the estimand, the residual
functional with its Ritz-vector response, the measured universes, both QWC
protocols (`qwc_groups` declared, `qwc_basis_cover` alternative), and the
four-step Richardson validator with its directions, seed and tolerances.

## Deterministic checks, per priced prefix

Full rank, a nondegenerate ground Ritz root, the energy reproducing the ladder
history to `1e-9`, the energy not below `E₀`, the residual functional
reproducing `SecondMomentBank`'s `σ²` and having zero reference mean (Q18's
tolerances), both Richardson directions passing, valid partitions, and
nonnegative setting variances. A failed check makes that prefix `INVALID`.

**Lineage.** On both trajectories, `M = 9` must reproduce Q18-S1's committed
Hubbard 2×2 entry. That covers the energy, `σ²`, both universes, every
setting count, one-shot and per-setting variance, uniform and Neyman ratio
under both protocols, and every Richardson endpoint. Floats agree to `1e-9`
relative. Neyman sums and ratios agree to `1e-3`, the tolerance Q18's own
checker gives them, because they add square roots of rounding-level setting
variances. Endpoints agree to `1e-12` of the cancellation scale. Failure makes
the verdict `INVALID`.

## Decision rule

Each priced prefix gets Q18's ladder on its two Neyman ratios: `AFFORDABLE`
(both `≤ 10`), `PROHIBITIVE` (both `> 10`) or `GROUPING_SENSITIVE`.

| verdict | condition |
|---|---|
| `INVALID` | a premise or the lineage fails, or any certifying prefix is `INVALID` |
| `UNREACHED` | no prefix certifies |
| `OPEN` | some certifying prefix is `AFFORDABLE` |
| `CLOSED` | every certifying prefix is `PROHIBITIVE` |
| `GROUPING_SENSITIVE` | otherwise |

Prefixes outside the domain are priced and reported as diagnostics only.

## Consequences

- **`CLOSED`.** Under these partitions and Q18's matched-precision statistic,
  no shot allocation measures `σ` for ten times the energy's optimal cost on
  any committed Hubbard 2×2 basis whose interval certifies the ground state.
  The pilot-driven Hubbard experiment is not authorized as support for
  convergence reporting. On the frozen bank it could only test the cost of a
  residual that certifies nothing. Phase 15's finite-shot residual line stays
  negative on the five banks and on these trajectories, and exact residuals
  remain available.
- **`OPEN`.** A pilot-driven allocation experiment may be declared, scoped to
  the `AFFORDABLE` certifying prefixes by name. It still needs its own seeds,
  budgets, replication, coverage targets and sampled validation, with the
  energy side measured through the same pilot pipeline. This screen licenses
  no estimator.
- **`GROUPING_SENSITIVE`.** No experiment is authorized until a later
  declaration fixes one grouping and one prefix before any sampling.
- **`UNREACHED`.** No committed basis certifies the ground state, so the
  question is unanswered for certifying bases. A new trajectory needs a new
  declaration. The frozen-bank experiment is not authorized for convergence
  reporting.
- **`INVALID`.** No verdict. The record names the failed check.

## Not claimed

Fully or block-commuting grouping, pooled or multiply-assigned estimators,
shared shots between the energy and residual campaigns, finite-sample
intervals, the plug-in bias of `σ²`, device time, other systems, other
trajectories and excited roots. This screen is first-order and asymptotic, and
reads exact variances.

## Execution

Land the declaration, the helper module, the producer, the independent checker
and their tests on the undeclared Hubbard dimer first. Then run once from a
clean tree in the pinned record environment, commit the record alone, and only
then update the ledger and the reproduction notes. The config's last commit
must strictly precede the record's first commit. Input hashes stay binding.
Implementation hashes lapse once the record exists, since the record names its
own commit.
