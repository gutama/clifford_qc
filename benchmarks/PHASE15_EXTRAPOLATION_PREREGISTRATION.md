# Phase 15 preregistration: energy-variance extrapolation

`SecondMomentBank` gives the exact variance `σ² = ⟨H²⟩ − ⟨H⟩²` of any Ritz
state. PLAN.md §5 lists variance extrapolation among what that makes
possible. The idea is to read the energy a converging sequence of states would
reach at zero variance. An estimator like that sits beside the variational
Ritz energy with a weaker guarantee, so it should earn its place first. This
declaration fixes, before any intermediate variance is computed, the one test
that decides whether it does. Its config is
`benchmarks/configs/phase15_variance_extrapolation.json`, and its gate is
`benchmarks/check_phase15_extrapolation_preregistration.py`.

## 1. Question (Q17)

On each frozen basis whose final Ritz state satisfies `σ_f ≤ gap/2`, fit a
straight line through the last three points of the prefix trajectory
`(σ²_M, E_M)`. Does its value at `σ² = 0` have at most **half** the final
Ritz energy's absolute error against the exact sector ground energy?

## 2. The trajectory

Each basis is the frozen `selected_labels` of
`benchmarks/configs/mapping_axis.json`, identity first. For the rows the
A-CASE ladder supplies, that is the order A-CASE selected them in. H₄'s labels
match the committed growth history in `acase_ladder.jsonl`.

- **Prefix `M`** is the first `M` generators, for `M = 1, …, M_f`.
- **Energy.** `E_M` is the ground energy of `MatrixElementBank.solve` on that
  prefix, with the solver's default thresholds.
- **Variance.** `σ²_M` is `SecondMomentBank.residual(result, 0).variance`,
  with its `resolved` flag.
- **Nesting.** Every prefix block is the top-left corner of the full block, so
  one bank per basis serves the whole trajectory.

## 3. The fit and why this one

The fit is ordinary least squares `E = a + b σ²` over prefixes `M_f − 2`,
`M_f − 1` and `M_f`. The extrapolated energy is `E_x = a`.

Near convergence, suppose one excited contamination of weight `w`
dominates. Then `E − E₀ ≈ w·gap` and `σ² ≈ w·gap²`, so `E ≈ E₀ + σ²/gap`:
a line whose intercept is `E₀` and whose slope is positive. The last three
prefixes are the most converged, where that picture holds best, and a third
point leaves the line a residual to report.

The fit is **extrapolable** only when all three window variances are
resolved, they are not all equal, and `b > 0`. A non-positive slope
contradicts the model. That is a legitimate outcome, NOT_EXTRAPOLABLE, and
not an error.

The improvement factor is **one half**. An estimator reported beside the Ritz
energy should at least halve its error where it applies; anything less is not
worth a second number with a weaker guarantee than the variational one.

## 4. Domain and banks

The linear model assumes the Ritz state is close to the ground state. The
principled version of "close" is Temple's regime: a residual below half the
gap means the ground state dominates the Ritz state. A bank is **required**
when `σ_f ≤ gap/2`, with `σ_f` read from the committed SecondMomentBank
validation record and `gap = E₁ − E₀` from the exact `(N, S_z)` sector. The
gate recomputes both, and a bank outside the domain is a diagnostic that
cannot promote.

| bank | M | gap | σ_f | final error | role |
|---|---|---|---|---|---|
| h4 | 9 | 0.289 | 0.076 | 3.0e-3 Ha | required |
| h4_converged | 15 | 0.289 | 0.050 | 7.7e-4 Ha | required |
| beh2 | 5 | 0.280 | 0.0023 | 3.3e-6 Ha | required |
| h2o_cas8e6o | 9 | 0.397 | 0.152 | 1.4e-2 Ha | required |
| hubbard_2x2 | 9 | 0.296 | 1.52 | 0.86 t | diagnostic |

H₂O sits closest to the domain edge, with `σ_f` at 0.76 of `gap/2`. `h4` is
a prefix of `h4_converged`: the two share a trajectory and fit different
windows (prefixes 7–9 against 13–15), so they are not independent evidence.
The gate checks that the sector ground energy equals each FCIDUMP's
provenance FCI, and that the final Ritz energy equals the validation record's.

## 5. Decision rule

| bank status | condition |
|---|---|
| IMPROVES | extrapolable and `\|E_x − E₀\| ≤ ½ \|E_f − E₀\|` |
| NO_GAIN | extrapolable and `½ \|E_f − E₀\| < \|E_x − E₀\| ≤ \|E_f − E₀\|` |
| WORSENS | extrapolable and `\|E_x − E₀\| > \|E_f − E₀\|` |
| NOT_EXTRAPOLABLE | an unresolved window variance, equal window variances, or `b ≤ 0` |

A failed deterministic check makes that bank INVALID. The checks are:

- the final prefix reproduces the validation record's ground energy and
  variance;
- `E_M` is non-increasing, as nested subspaces require;
- each window variance equals the matrix-free reconstructed residual;
- some exact eigenvalue lies within `σ_M` of `E_M` at every prefix. That is
  Weinstein's theorem, so a failure is a bug;
- the sector ground energy equals the provenance FCI;
- the record carries clean provenance.

Every required bank IMPROVES gives **GO**, and no required bank IMPROVES gives
**NO_GO**. Anything else is **CONDITIONAL**, and any INVALID required bank
gives **INVALID**. With four required banks, all three verdicts are
reachable.

| verdict | what follows |
|---|---|
| GO | Phase 15 ships this rule as a supported estimator for bases in the domain, reported beside the Ritz energy, never replacing it, and labelled extrapolated rather than variational. |
| CONDITIONAL | The rule ships as a diagnostic only, with the banks where it improved named. |
| NO_GO | It does not ship; residual norms and variances stay diagnostics. |
| INVALID | No verdict; the record names the failed check. |

## 6. What this can and cannot conclude

The test reads exact energies that no device run has, so it judges the rule
on exact inputs: evidence `exact_oracle`. A GO does not say a finite-shot
variance estimate would support the same extrapolation, or that the rule
holds outside the domain or on other bases. The record reports several
diagnostics that decide nothing:

- fits over windows of two and four prefixes, and over every resolved prefix;
- the Temple lower bound, which uses the exact `E₁` and is labelled oracle;
- whether `E_x` reaches `1.6e-3` Ha on a molecule where `E_f` does not.

## 7. Disclosure

Before freezing, the final Ritz energy and variance of every bank were known.
Both are committed in the SecondMomentBank validation record, and the domain
criterion reads them. H₄'s energy history along its growth (`E_M` for
`M = 1…9`) is committed in `acase_ladder.jsonl`, and it was displayed while
confirming the growth order. No intermediate variance `σ²_M` for `M < M_f`
and no fit was computed on any declared bank. The fit and ladder code are
exercised in tests on the Hubbard dimer, which is not declared.

## 8. Revisions and order

**Revision 0** is this declaration. The producer
(`run_phase15_variance_extrapolation.py`) and result checker
(`check_phase15_variance_extrapolation.py`) come after it, and the producer
must re-run this gate and refuse to compute a prefix variance if it fails.
Once a record exists, the gate requires the config's last change to strictly
precede it. It keeps the inputs bound by SHA-256 for good, and the
second-moment and eigensolver code only until the record exists. A later
revision must state that no prefix variance existed when it was made. No
follow-up is permitted: another window, model, factor, domain or bank set
would be a new declaration.
