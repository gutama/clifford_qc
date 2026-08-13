# Rethinking finite-shot performance

`PLAN.md` §5 Phase 4 and the A-CASE manuscript's
§"Finite-shot stabilization trades median accuracy for tail control" left the
finite-shot projected eigensolver in a state with one good property and two bad
ones. The good property: everything the subspace needs is a linear functional of
Pauli-word means, measured once through shared QWC groups. The bad ones:

1. **The tail, not the median, is the error.** At 104,000 setting-shots on the
   frozen four-qubit TFIM bank, the median absolute error is `4.48 mHa` and the
   RMSE is `1786 mHa` — four replicas in two hundred land whole Hartrees away
   when a near-null overlap mode is resolved the wrong way.
2. **Every fix cost accuracy.** Thresholding overlap modes at their own shot
   noise removed the tail (`RMSE 13.31 mHa`) but doubled the median error to
   `10.64 mHa`, carried a `+10 mHa` truncation bias, and recovered the exact
   rank in only `71/200` replicas.

The plan's response to that had been acquisition-side: allocate shots better.
That was measured and it did not work — covariance-aware Neyman allocation cut
the summed projected-matrix variance by 68.9% and moved the median error from
`4.48` to `4.44 mHa`. Cutting the variance of the thing being estimated by a
factor of three barely touched the error in the thing being computed.

This note records the rethink that followed, its three experiments, and their
verdicts. The producer is `benchmarks/run_finite_shot_rethink.py`; the record is
`benchmarks/reference_results/finite_shot_rethink.json`; it is regenerated and
compared by `benchmarks/check_finite_shot_rethink.py`. It runs on the *same*
bank, seed, and budget as the published study, and its `assigned`/`fixed` and
`assigned`/`calibrated` arms reproduce that study's rows to the digit, so the
new arms are directly comparable to what is in the manuscript.

## The diagnosis

Allocation failed to help because it optimizes the wrong stage. There are three
stages between shots and an energy, and only the first had been varied:

```text
   acquisition           reconstruction              rank rule
shots -> settings  ->  histograms -> (S_hat, H_hat) -> retained modes -> E_hat
```

The failure mode is a near-null overlap mode. Its *variance* is not what makes
it dangerous — its variance relative to its eigenvalue is, and no reallocation
of a fixed budget changes that ratio by the orders of magnitude needed. So the
two stages that had never been varied are the ones worth attacking, and both
turn out to be changeable **without spending a single additional shot**.

## Experiment 1 — reconstruction: the partition is throwing away readings

QWC grouping partitions the word universe so that each measurement setting
covers many words at once. The partition is a *scheduling* device: it answers
"how few circuits cover everything". It had also been used as an *estimation*
device — each word's mean read from the one group it was assigned to.

Those are different questions, and the second answer is wrong. A setting fixes a
measurement basis on its support; every word supported inside that set with
matching letters is a function of the same recorded bitstrings, so its outcome
is in that setting's histogram whether or not the partition assigned it there.
On the frozen bank, a word is recorded by **3.72 settings on average** and by up
to 23. The estimator was looking at one of them.

Pooling reads every word from every setting that records it, weighted by that
setting's shots. Three properties make this the right combination rather than
merely a bigger one:

- *Unbiased.* Each reading group's estimate is unbiased for the same mean, and
  the weights sum to one over the reading groups — a reweighting, not a second
  copy. (Duplicating a coefficient into every capable group instead of splitting
  it is exactly the multiplicity bug Phase 4 documents; the weights are what
  keep the reported variance exact, and a test pins the prediction against a
  Monte-Carlo spread.)
- *Optimal.* The marginal law of a word's ±1 outcome does not depend on which
  compatible basis was used to read it, so every reading group has the same
  per-shot variance `1 - mu_w^2`. Shot-count weights are therefore exactly the
  inverse-variance weights.
- *Certificate-safe.* The weights depend only on the predeclared shot schedule,
  never on outcomes, so a fixed-endpoint empirical-Bernstein argument survives.

`GroupedWordCache(n, pooling='shots')` and `SharedMeasurement(bank,
pooling='shots')` select it. It is off by default: committed records were
produced under the single-assignment estimator, and their digests are part of
those records.

**Verdict: the largest single improvement in the finite-shot layer to date.**
At the published budget it moves the fixed-cutoff arm from `4.48 mHa` median /
`1786 mHa` RMSE to `2.56` / `11.70`, and the calibrated arm from `10.64` /
`13.31` to `2.56` / `4.86` — with rank recovery going from `71/200` to
`195/200`. The manuscript's trade of median accuracy against tail control does
not survive it: the pooled calibrated arm is better than the published best
median *and* better than the published best RMSE, simultaneously.

## Experiment 2 — rank rule: score the energy, not the overlap

The calibrated rule asks of each overlap mode whether it stands above its own
shot noise. That is a question about `S`. The quantity at risk is the Ritz
value, and a barely resolved direction is harmless if `H` hardly couples to it
and ruinous if it does.

`SharedMeasurement.solve_selected_rank` asks the second question directly: solve
at every attainable rank, pair each solution's Ritz value with the delta-method
standard error of `ritz_functional` there, and take the rank minimizing
`E_hat(k) + gamma*sigma_hat(k)`. Adding a mode lowers `E_hat` by Cauchy
interlacing and raises `sigma_hat` once the mode is noise-dominated, so the
score has an interior minimum and `gamma` prices one against the other.
`gamma = 2` is the default: a two-sigma price.

It is a selection rule, not a certificate — the same cache supplies the
matrices, the rank, and the error bar — and the returned value is not a
variational bound on `E_0`. It costs `M` solves and `M` covariance-vector
products, which is cheap against the shots but not free.

**Verdict: it dominates the calibrated cutoff, and it is the rule that matters
at low budget.** Applied alone, without pooling, it turns the published pair
into `4.56 mHa` median and `7.82 mHa` RMSE with no catastrophic replica and
`162/200` rank recovery — the fixed rule's median with better than the
calibrated rule's tail. At 13,000 setting-shots it beats the calibrated cutoff
on both estimators (`11.19` vs `32.63 mHa` median on assigned; `14.33` vs
`17.08 mHa` RMSE on pooled). At 104,000 setting-shots with pooling the two are
tied, because pooling has already removed the regime in which the rank decision
is hard.

Pooling does not make the rank rule redundant: `pooled`/`fixed` still leaves one
catastrophic replica in two hundred at the full budget.

## Experiment 3 — smooth spectral damping instead of truncation (negative)

Truncation is a binary decision taken on noisy data, which invites the obvious
alternative: keep every mode and damp it. `solve_projected(...,
overlap_ridge=...)` replaces `S_bar` with `S_bar + sum_k delta_k u_k u_k'`, so a
mode contributes `1/(lambda_k + delta_k)` to the inverse metric instead of
`1/lambda_k`, and the whitening identity still holds exactly in the ridged
metric.

**Verdict: worse than truncation at every scale tried, by orders of magnitude.**
At the calibrated radii it gives `50 mHa` median and `6166 mHa` RMSE where
truncation gives `2.56` and `4.86`; sweeping the ridge up by factors of 3 to 300
trades the tail for a uniform positive bias that reaches `3.4 Ha` without ever
passing truncation.

The reason is worth keeping, because it says something about the problem rather
than about the knob. The ridge bounds the *metric*, and the damage comes from
the *numerator*: along a near-null mode the measured `H_bar` is noise, and the
Rayleigh quotient will descend into it for any damping that leaves the direction
in the space at all. A factor of `lambda/(lambda + r)` — one half, at the noise
radius — is nowhere near enough, and a ridge large enough to suppress it also
distorts the well-resolved modes, since it acts on all of them. Removing the
direction is qualitatively, not quantitatively, different from shrinking it.

The knob is retained, off by default, so the negative result stays reproducible.

## Results

One bank (four-qubit TFIM, `M=9`, exact rank 8, `kappa_S = 194.94`, 189 words,
52 QWC groups), uniform allocation, 200 replicas. Every arm at a budget reads
the same caches, so the table compares estimators, not budgets. Errors in mHa
against the exact projected energy.

**104,000 setting-shots per replica** (the published budget):

| reconstruction | rank rule | median | p95 | max | RMSE | bias | >0.1 Ha | rank 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| assigned | fixed | 4.48 | 37.01 | 22760.20 | 1786.21 | −178.08 | 4/200 | 200/200 |
| assigned | calibrated | 10.64 | 23.96 | 34.84 | 13.31 | +10.33 | 0/200 | 71/200 |
| assigned | selected | 4.56 | 16.27 | 21.80 | 7.82 | +1.73 | 0/200 | 162/200 |
| assigned | ridge | 97.42 | 2754.89 | 21589.19 | 2095.93 | −406.15 | 73/200 | 1/200 |
| pooled | fixed | 2.56 | 10.15 | 152.64 | 11.70 | −2.46 | 1/200 | 200/200 |
| **pooled** | **calibrated** | **2.56** | **10.12** | **21.29** | **4.86** | **−1.13** | **0/200** | **195/200** |
| pooled | selected | 2.56 | 10.19 | 21.29 | 4.90 | −1.45 | 0/200 | 198/200 |
| pooled | ridge | 50.02 | 11637.25 | 46061.65 | 6166.13 | −2125.76 | 65/200 | 0/200 |

**13,000 setting-shots per replica** (eight times smaller):

| reconstruction | rank rule | median | p95 | max | RMSE | bias | >0.1 Ha | rank 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| assigned | fixed | 21.61 | 1812.02 | 23859.85 | 2254.42 | −446.49 | 34/200 | 160/200 |
| assigned | calibrated | 32.63 | 93.65 | 119.75 | 53.88 | +37.86 | 7/200 | 5/200 |
| assigned | selected | 11.19 | 42.73 | 84.19 | 20.97 | +1.88 | 0/200 | 64/200 |
| assigned | ridge | 240.42 | 3245.81 | 549587.63 | 39254.96 | −3444.74 | 190/200 | 2/200 |
| pooled | fixed | 11.48 | 173.93 | 14303.83 | 1186.45 | −171.95 | 14/200 | 198/200 |
| pooled | calibrated | 12.73 | 32.05 | 44.56 | 17.08 | +9.72 | 0/200 | 48/200 |
| **pooled** | **selected** | **10.11** | **26.62** | **44.56** | **14.33** | **+1.58** | **0/200** | 130/200 |
| pooled | ridge | 130.07 | 5292.96 | 76476.77 | 5986.66 | −1267.71 | 174/200 | 1/200 |

Read together: **pooling is what to change first, and the rank rule is what
still matters once shots are scarce.** The recommended pair is
`pooling='shots'` with `solve_selected_rank`, which is the best or tied-best arm
at both budgets and is the only arm besides pooled/calibrated with no
catastrophic replica anywhere.

Two things the table does not say. It is one bank at one conditioning, four
qubits, in simulation with no device noise — the recommendation is a default to
try, not an established scaling law. And no arm's value is a variational bound:
the same data supply the matrices, the retained rank, and the error bar, so the
`bias` column stays a measured property of an estimator, not a certified one.

## What was considered and not done

- **Redesigning the settings themselves.** Once pooling exists, the partition is
  no longer the right object — settings could be chosen to *overlap*
  deliberately, maximizing coverage-weighted readings rather than minimizing
  circuit count. Measured on this bank: a greedy cover from the 81 full
  four-qubit bases needs 57 settings (against the partition's 52) to reach mean
  readers `4.09` (against `3.72`). About 10% more readings for 10% more
  settings — much smaller than the 3.72× that pooling already recovers from the
  existing partition, so the partition stays.
- **Completing each setting's basis** on qubits its words do not touch (free on
  hardware, which measures them anyway). Measured: mean readers `3.72 -> 3.78`
  at four qubits and `4.41 -> 4.43` at six. Negligible; not implemented.
- **Anytime-valid confidence sequences** in place of the fixed-schedule
  empirical-Bernstein bounds. This is the remaining large win on the *certified*
  path, where two independent full-universe batches per step put certified
  growth four orders of magnitude above the exact path in shots. A confidence
  sequence would license outcome-dependent allocation and confidence-set reuse
  across growth steps. It is a genuine piece of work, not a knob.
  - Worth recording alongside it: pooling helps the estimator more than it helps
    the current EB certificate. On the Ritz functional at 2,000 shots/group it
    cuts sigma from `5.87e-3` to `3.60e-3` (and the normal radius likewise) but
    the EB radius only from `0.208` to `0.183`, because the bound is a sum of
    per-group radii under a union bound over the groups touched, and pooling
    touches at least as many. Tightening that (bounding the sum directly rather
    than group by group) is the companion piece.
- **A variance-penalized Rayleigh quotient** — the continuous version of
  `solve_selected_rank`, minimizing `R(c) + gamma*sigma(c)` over the coefficient
  vector rather than over the rank. Better motivated than the ridge, since it
  penalizes the numerator noise that the ridge fails to reach, but each
  `sigma(c)` evaluation is a covariance-vector product over the whole word
  universe inside an optimization loop.
- **Wiring pooling into noisy ADAPT selection.** `GroupedWordCache` is also
  what `run_adapt` builds when `grouping=True`, and an ADAPT gradient is the
  same shape of object — a linear functional of word means — so pooling applies
  unchanged. It is deliberately not wired through: `run_adapt` carries a
  structured/legacy keyword reconciliation and a certified selector whose
  calibration records are committed, and a free variance reduction is not worth
  half-changing that surface in the same pass. The cache supports it today;
  only the plumbing is missing.
- **Bias correction** by grouped bootstrap or jackknife. The measured bias of
  the recommended arms is `1.1`–`1.6 mHa` against median errors of `2.6`–`10`,
  so there is little left to correct once pooling and rank selection are in.
