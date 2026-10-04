# Phase 15 preregistration: the measured-residual cost preflight

`SecondMomentBank` gives exact residual norms on the five licensed banks. What
remains of Phase 15 is a finite-shot estimate of the second-moment block. The
H² preflight priced the words and coefficients that estimate must carry. It
excluded grouping and shots, and said FULL "is not a claim that those are
affordable". This declaration prices them before any estimator is built. It
asks what a residual norm costs in shots, against what its own energy costs,
at one matched standard error. Its config is
`benchmarks/configs/phase15_measured_residual_preflight.json`, and its gate is
`benchmarks/check_phase15_measured_residual_preregistration.py`.

## 1. Question (Q18)

On each of the five licensed banks, estimate the ground Ritz root's residual
norm `σ` from shared QWC word means, to the same standard error as its energy.
Does that cost at most **ten times** the energy's shots?

## 2. What is estimated

The root is the ground root of the retained block's `(S, H)` pencil, solved
under the package's default thresholds and normalized `c†Sc = 1`. Its variance
is `σ² = c†Kc − E²`, with `K` the `SecondMomentBank` block. That is
`‖(H − E)|Ψ⟩‖²/⟨Ψ|Ψ⟩`, the squared residual norm of PLAN.md §4.4. The residual
norm `σ` is in the energy's own units. Weinstein's theorem puts an eigenvalue
within `σ` of `E`, so `(E, σ)` is the pair a measured residual reports.

The estimator reconstructs every `S`, `H` and `K` entry from one set of shared
word means. It solves the measured pencil and reads `σ²` from the measured `K`
at the measured Ritz vector. Nothing is measured per entry.

**The energy shift is not a choice.** Measuring `(H − c)²` instead of `H²`
looks like a way to tame the cancellation in `⟨H²⟩ − E²`. With shared word
means it changes nothing. The rows of `(H − c)²` are combinations of the same
`S`, `H` and `K` rows, so the reconstruction is the same function of the same
means, and so is its linearization. The gate's tests confirm this to rounding.

## 3. The linearization

**Energy.** The existing `ritz_functional` is the word expansion of
`B†(H − E)B`, with `B = Σ c_i A_i`. The energy is stationary in `c`, so no
Ritz-vector term enters.

**Residual.** `σ²` is not stationary in `c`, so the Ritz vector's movement
contributes. With `Q = c†Kc`, the Jacobian with respect to every word mean is
the word expansion of

```text
G = B†(H − E)²B − σ² B†B − [V†(H − E)B + B†(H − E)V],
V = Σ v_i A_i,     v = Σ_{k>0} c_k (c_k†(K − QS)c) / (E_k − E).
```

On the undeclared Hubbard dimer the last term is 3.7% of the weights, and
dropping it fails the finite-difference rule below by orders of magnitude.
`G` is built from `H − E`, so a constant shift of `H` leaves every weight
unchanged, and its reference expectation is zero. The definition is
`residual_functional` in the gate, which the producer imports.

**Finite differences.** The producer checks the linearization on every bank
against central differences of the full nonlinear pipeline, along two frozen
directions:

- `functional`, the weights themselves at unit length;
- `sh_gaussian`, a standard normal draw with seed 20261003 over the
  non-identity `(S, H)` words, normalized. Those words carry the Ritz-vector
  term.

The step is `10⁻³`. A direction passes when
`|difference − g·d| ≤ 10⁻⁶ ‖g‖₂ + 10⁻¹² κ / step`. The second term is the
rounding floor: `σ²` is a difference of numbers of the cancellation scale `κ`
the validation record reports, and the package's `RESOLUTION` is `10⁻¹²`. On
the dimer the step leaves a truncation error near `5 × 10⁻⁹ ‖g‖₂`. That holds
with and without `−75` Ha added to `H` to reproduce H₂O's cancellation scale.

## 4. Banks

The banks are the five the H² preflight licensed and the `SecondMomentBank`
validation covers, each at the `selected_labels` that
`benchmarks/configs/mapping_axis.json` freezes. Every number below is
committed or recomputed from first moments, and the gate recomputes it.

| bank | qubits | M | Ritz gap | `σ` | raw `\|U_SH\|` | raw `\|U\|` | `G_SH` | protocol |
|---|---|---|---|---|---|---|---|---|
| h4 | 8 | 9 | 0.852 | 0.0756 | 7,371 | 8,192 | 913 | greedy |
| h4_converged | 8 | 15 | 0.558 | 0.0498 | 7,927 | 8,192 | 913 | greedy |
| beh2 | 8 | 5 | 0.597 | 0.00234 | 1,815 | 2,048 | 353 | greedy |
| h2o_cas8e6o | 12 | 9 | 1.402 | 0.152 | 143,117 | 868,150 | 24,334 | cover |
| hubbard_2x2 | 8 | 9 | 0.573 | 1.52 *t* | 5,537 | 13,940 | 1,406 | greedy |

Energies and `σ` are in hartree, and in *t* for Hubbard. The Ritz gap is the
distance from the ground root to the next root of the same pencil. The
universe sizes are raw, identity included, as the committed records count
them; §5 defines the measured universes, one word smaller. Every block
is full rank, every gap is far from zero, and every variance is resolved: the
three premises of the linearization. `G_SH` regroups each bank's `(S, H)`
universe under its declared protocol and equals the mapping-axis record's
committed count. The ratio is dimensionless, so Hubbard's units do not matter
and it is decided like the rest.

## 5. Grouping and variance

**Raw and measured universes.** The Phase 2M-A ledger, the H² preflight
record and the mapping-axis record count and hash *raw* universes, identity
included. Every frozen basis starts with `I`, so `S₀₀` carries the identity.
The *measured* universe is the raw one minus the identity, whose mean is 1
and is never measured. Every frozen count and digest whose name starts with
`raw_` is a raw quantity, and every grouping partitions a measured universe.
The gate checks that the identity is in each raw universe, and that the
energy partition covers exactly `raw − 1` words.

The energy is measured on the measured `U_SH`, and the residual on the
measured `U = U_SH ∪ U_K`: every word a measured second-moment bank's rows
carry, except the identity. Both are listed in ascending code order, as the
mapping-axis producer lists its JW arm.

- **Declared protocol.** The mapping-axis record's protocol for that system:
  the largest-degree greedy `qwc_groups` on the 8-qubit banks, and the
  scalable `qwc_basis_cover` on H₂O. The energy denominator therefore
  reproduces a committed setting count.
- **Alternative protocol.** The cover on the 8-qubit banks, where both
  heuristics are feasible, so no status rests on one of them. On H₂O the
  greedy is quadratic in 868,150 words, so there is no alternative.
- **Estimator and allocation.** Single assignment and uniform shots per
  setting, as the committed mapping-axis energy variances use.
- **Variance.** For each setting, `Var_ψ(Σ_{w∈g} f_w P_w)` on the exact
  reference, by `run_mapping_axis._functional_variance`, the function behind
  those committed variances. `V` is their sum, the estimator's variance at
  one shot per setting.

## 6. The statistic, and why ten

```text
R = (G_U · V_σ²) / (4 σ² · G_SH · V_E)
```

At `N` shots per setting a linearized estimator has variance `V/N`. A
standard error `s` therefore costs `G·V/s²` shots in total. `σ` is read from
`σ²` through `dσ = dσ²/(2σ)`. Match the standard errors of `E` and `σ` at
`s`, and the `s²` cancels. So `R` is the residual's shots over the energy's at
any one matched precision, to first order. Matching them makes both ends of
the measured Weinstein interval `[E − σ, E + σ]` carry the two sources
equally.

**The threshold is ten.** A decade is the unit this repository reads as
material, as in Q16's word clause and Phase 16B's 10× modelled-shot rule.
Within a decade of its energy's shots, a residual is a by-product of
measuring that energy well. Beyond it, it is a measurement of its own. A
ratio of exactly ten is affordable.

**Where the ratio applies.** `dσ = dσ²/(2σ)` holds while the standard error of
`σ²` is small against `σ²`. At a matched precision `s` that error is `2σs`, so
the record reports `σ/4`, the precision at which it reaches half of `σ²`. Finer
precisions are where `R` describes the cost; coarser ones do not resolve `σ`
from zero. The record also reports the Neyman ratio, with both sides at
variance-optimal allocation across settings. No clause reads either.

## 7. Decision rule

| bank status | condition |
|---|---|
| AFFORDABLE | `R ≤ 10` under the declared protocol, and under the alternative where one is declared |
| PROHIBITIVE | `R > 10` under the declared protocol, and under the alternative where one is declared |
| GROUPING_SENSITIVE | the two protocols put `R` on opposite sides of 10 |

A failed deterministic check makes that bank INVALID. Every bank AFFORDABLE
gives **FULL**, and no bank AFFORDABLE gives **NONE**. Anything else is
**RESTRICTED**, and any INVALID bank gives **INVALID**. Nothing in the
committed numbers settles a bank in advance, so all three verdicts are
reachable.

| verdict | what follows |
|---|---|
| FULL | Phase 15 builds the finite-shot second-moment estimator on all five banks. It samples `K` over the declared grouping and reports `σ` with a delta-method interval labelled asymptotic, cross-checked by the grouped bootstrap. Its sampled validation is a new preregistration that freezes seeds, budgets and coverage targets. This verdict licenses building the estimator, not any sampled claim. |
| RESTRICTED | The estimator is built and validated only on the AFFORDABLE banks, which become its scope by name. Residuals elsewhere stay exact or matrix-free, and extending the estimator needs a new declaration. |
| NONE | No finite-shot second-moment estimator is built. Phase 15's finite-shot item closes negative on these banks. Exact residuals, convergence reports and anything else built on exact second moments are unaffected. |
| INVALID | No verdict. The record names the failed check. |

## 8. Deterministic checks

A bank is INVALID unless all of these hold:

- every number in §4 recomputes, and every premise flag holds;
- the raw `U`, rebuilt from `SecondMomentBank` rows, has the committed size
  and SHA-256, and contains the identity;
- `G_SH` under the declared protocol equals the mapping-axis JW arm's count;
- where the mapping-axis record committed `V_E` (converged H₄ and BeH₂), the
  recomputed value equals it within `10⁻⁹` relative;
- each grouping assigns every word of its measured universe to exactly one
  setting, and every setting is qubit-wise commuting;
- the pipeline at the exact means returns the validation record's `σ²`, and
  `|⟨ψ|G|ψ⟩|` is at most `10⁻⁹ (1 + κ)`;
- both finite-difference directions pass;
- no per-setting variance is below `−10⁻¹⁰` of its scale;
- the record carries clean provenance.

## 9. What this can and cannot conclude

`R` is a first-order, asymptotic ratio read from exact reference variances.
It prices shots under single-assignment QWC with uniform allocation, and
nothing else:

- not fully commuting or block-commuting grouping, which would change both
  sides;
- not device time. QWC settings carry single-qubit rotations only, so the
  device cards would differ only by per-setting depth and readout;
- not the plug-in bias of `σ²`, or any second-order effect, including the
  nonlinear response of the retained eigenspace;
- not finite-sample intervals, bootstrap coverage or a sampled validation;
- not excited or folded-spectrum roots, or any bank outside the five.

FULL says a measured residual is not ruled out by its shot cost relative to
the energy. It does not say the measured residual certifies anything; the
sampled validation decides that.

## 10. Disclosure

Before freezing, only first moments and committed values were computed on the
declared banks: the numbers in `measured_before_freezing`. `G_SH` regrouped
each `(S, H)` universe under its declared protocol and reproduced the
committed count. `V_E` was not computed on any bank. The two committed values
are read from the mapping-axis record. No second-moment row, residual
functional, grouping of a combined universe or group variance was formed on a
declared bank.

The linearization, its shift invariance, its zero mean and the
finite-difference rule were developed and calibrated on the undeclared
Hubbard dimer. The groupings' feasibility was timed on synthetic universes of
random spin-parity-sector words at the declared banks' sizes. Under
`qwc_groups`, 13,939 words on 8 qubits took 7.2 s; under `qwc_basis_cover`,
868,150 words on 12 qubits took 257 s. Neither used a declared word set.

## 11. Revisions and order

**Revision 0** is this declaration. The producer
(`run_phase15_measured_residual_preflight.py`) and result checker
(`check_phase15_measured_residual_preflight.py`) come after it. The producer
must re-run this gate and refuse to run if it fails. Once a record exists,
the gate requires the config's last change to strictly precede the record's
first commit.

The inputs stay bound by SHA-256 for good. The code that defines a row, a
solve, a grouping, a functional and a group variance stays bound until the
record exists. That includes the gate itself, whose estimator the producer
imports, and the Phase 15 gate it builds banks with.

**Revision 1** was made before any ratio existed, after review of the
declaration. It changes no rule. It names raw and measured universes apart
and renames the frozen counts and digest `raw_…` with unchanged values. It
binds the gate, the bank builder and the energy functional, and it makes the
gate refuse every result-shaped key, whatever its value. A revision after this one must state that no ratio existed when it was
made. No follow-up is permitted: another threshold, protocol, allocation or
bank set would be a new declaration.
