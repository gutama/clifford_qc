# Declared target-aware acquisition study

**Result: targeting was never dearer and was cheaper in five of eight cells,
but the stopped interval failed on the larger held-out bank.** At matched
error, Ritz-targeted allocation needed fewer shots than uniform allocation in
five of the eight system-and-target cells, by factors of 1.4 to 2.7, and more
in none. On the three four-qubit banks the schedule-valid interval
over-covered. On the held-out six-qubit bank it under-covered in three of four
cells. An exploratory split suggests why: runs stopped on solves that had
truncated a direction, and the standard error does not contain that bias.
Details are under [Results](#results).

Everything from "Question and estimand" to "Reproduce" is the declaration. It
and the [declaration file](target_acquisition_declaration.json) were committed
as `f8358f87aa28539ca397b6e7a8074bbc90ea9b20`, before any evaluation seed was
drawn, and are unchanged. `check_target_acquisition.py --verify-history` proves
the order from Git.

## Question and estimand

Uniform shots spend the same effort on every measurement setting.
[`acquire_ritz`](../../clifford_qc/subspace/acquisition.py) measures a uniform
pilot and then spends each further round where the current Ritz value is most
sensitive. Two questions follow.

1. **Cost.** How many physical shots does Ritz-targeted allocation need,
   relative to uniform allocation, to deliver the same error?
2. **Validity.** When a predeclared stopping rule ends a run, does the reported
   interval contain the target at its nominal 95% rate?

The target is the ground energy of the exact projected problem in the declared
subspace, from the canonical exact solver at its default numerical thresholds.
It excludes the subspace's own distance from the true ground energy, which no
measurement interval addresses.

## Systems

All are spin models with ideal simulated shots. Generators are the identity,
the first odd-`Y` words of the local pool, and Krylov powers of the
Hamiltonian. These numbers come from exact arithmetic only.

| System | Role | Qubits | Generators | Exact rank | Exact `kappa_S` | Words | Settings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tfim4_orthogonal` | development | 4 | 4 | 4 | 1.0 | 43 | 8 |
| `tfim4_mixed` | development | 4 | 9 | 8 | 194.9 | 189 | 52 |
| `xxz4_mixed` | held out | 4 | 9 | 5 | 30.1 | 215 | 70 |
| `tfim6_mixed` | held out | 6 | 9 | 8 | 373.7 | 1,035 | 104 |

A setting is one qubit-wise-commuting measurement group, and every shot count
below is a count of physical shots of such circuits.

The two four-qubit TFIM banks are the fixtures this module was developed on.
`xxz4_mixed` was sampled by the [interval-coverage study](INTERVAL_COVERAGE.md)
under uniform allocation only, and never under an adaptive schedule.
`tfim6_mixed` was measured once before this declaration, under a throwaway
seed, to time a run; its estimates were not inspected. Both were available in
the repository, so held out is not a claim of blind external data.

## Schedule

One schedule is used for every system, fixed before any shot.

- **Pilot.** 100 shots of every setting.
- **Rounds.** 18 further rounds. Cumulative shots per setting, averaged over
  settings, are 100, 141, 200, 283, 400, 566, 800, 1,131, 1,600, 2,263, 3,200,
  4,525, 6,400, 9,051, 12,800, 18,102, 25,600, 36,204 and 51,200: a factor
  near `sqrt(2)` per round. A round's budget is the number of settings times
  the step.
- **Looks.** The run solves after the pilot and after every round: 19 looks.
- **Allocation.** `uniform` splits each round equally over settings.
  `ritz_targeted` splits it by Neyman allocation on the current Ritz
  functional `c*(H - E S)c`: shots in proportion to the square root of each
  setting's per-shot variance.
- **Estimator.** Package defaults of `SharedMeasurement.solve`, the only
  policy for which the interval-coverage study found no under-coverage of the
  delta method.

On each replica both allocations start from the same seed, so they read the
same pilot. Each then runs three times:

| Run | Stopping rule | Interval reported |
| --- | --- | --- |
| Full schedule | none | single-look delta interval at the last look |
| Stop at 0.02 | first look whose estimated standard error is at most 0.02 | schedule-valid and naive |
| Stop at 0.01 | first look whose estimated standard error is at most 0.01 | schedule-valid and naive |

A stopped run is the full schedule's seeded run, ended early; the checker
verifies this for every row. The two targets are in energy units of `J = 1`
and are the same for every system. They were chosen on the development banks,
where both allocations met them inside the schedule, and are transferred
unchanged. If a held-out system meets a target at the pilot or never meets it,
that is the result, and the exhausted runs are counted.

**Stopping semantics.** The rule is evaluated only at the 19 predeclared
looks. The schedule-valid interval splits its 5% error budget equally over all
19, so its half-width is 3.008 estimated standard errors instead of 1.960. It
is valid over the schedule to the extent the Gaussian delta method is valid at
each look. It is not anytime-valid, and it is not a finite-sample certificate.
The naive interval ignores the other looks and is reported for comparison.

## Estimands and labels

Every ratio below is uniform over targeted, so a value above one means
targeting is cheaper. Each carries a 95% percentile interval from 2,000 paired
bootstrap resamples of whole replicas, which keeps the pairing through the
shared pilot.

- **At equal shots.** The ratio of mean-squared errors at the last look of the
  full schedule. Labels: `targeted_better` if the interval lies above one,
  `uniform_better` if below, otherwise `indistinguishable`.
- **At matched error.** Under a stopping target the two allocations stop at
  different looks with slightly different realized errors, so shots alone are
  not a comparison at matched error. The figure is the ratio of
  work-normalized mean-squared errors, `mean shots x MSE at stop`: the shots
  uniform would need for the error targeting realized, over the shots
  targeting spent, if error falls as one over root shots. Verdicts:
  `fewer_shots_at_matched_error` if the interval lies above one,
  `more_shots_at_matched_error` if below, otherwise `no_resolved_difference`.
- **Raw shots.** The ratio of mean shots at stop, and each allocation's
  root-mean-squared error at stop, are reported beside the verdict. Shots at
  stop are resolved only to the look spacing.
- **Coverage.** Each interval's rate over all replicas carries a 95% Wilson
  interval and the label of the interval-coverage study: `under` if the Wilson
  upper limit is below 95%, `over` if the lower limit is above, otherwise
  `consistent`.

Error at every look of the full schedule is tabulated for both allocations:
root-mean-squared, median and worst absolute error, and mean retained rank.

## Seeds and the development split

Evaluation seeds derive from the root `20261117` through NumPy `SeedSequence`.
`[root, system, replica]` seeds a replica's measurements and
`[root, 1000000 + system, slot]` a bootstrap, with slot 0 for the comparison at
equal shots and slots 1 and 2 for the two targets. No seed from this root was
drawn before the declaration commit.

Development used only the two four-qubit TFIM banks, with backend seeds below
10,000, seeds 900,000 to 900,039 and 7,700,000 to 7,700,059, and dry runs
under the root `424242`.

## Counting rules

- A look that cannot be solved ends the run as a failure. The failure keeps the
  shots already spent, counts toward mean shots, and counts as not covered.
  Error statistics exclude it and state their own denominator.
- Coverage divides by every replica.
- A run that reaches the last look without meeting its target is
  `schedule_exhausted`. Its shots understate what the target would cost, so the
  exhausted counts are printed beside every ratio.

## Reproducibility of an adaptive schedule

In an adaptive schedule the shot counts decide what is sampled next, so one
shot of rounding changes every later outcome. Development found that this
happened across CPUs. Two settings of `tfim4_orthogonal` are related by a
symmetry and have weights that are equal in exact arithmetic. A
largest-remainder rule ordered by raw floats gave the spare shot to a different
one of them under a different BLAS kernel, in 3 of 30 targeted runs.

[`neyman_group_plan`](../../clifford_qc/subspace/acquisition.py) therefore
treats remainders within `1e-6` of a shot as tied and breaks the tie by group
order. After the change none of 240 development rows differed between four
kernels beyond a relative `2e-11`. The checker's replay tolerance is `1e-7`.
A tie the rule does not catch would show up as a replay failure, not as a
silent change.

Conditioning sets a second limit. A look whose smallest retained overlap
eigenvalue is below `1e-6` amplifies rounding too much for its floats to be
compared, and it decides the next round's shots. The ledger records that
eigenvalue at every look, and the replay compares such a row only up to its
first ill-conditioned look.

## What this can and cannot show

There are 4 labels at equal shots, 8 verdicts at matched error and 40 coverage
labels. With 400 replicas a 95% rate has a standard error of about 1.1 points.
A few coverage labels are expected to be off by chance, so single labels are
not findings; patterns across systems are.

Known limits, stated before the run:

- **Adaptive bias.** All data are pooled, pilot included, and each round's
  allocation depends on earlier outcomes. The estimate therefore carries the
  bias any such design has. The full-schedule coverage measures whether it
  matters here.
- **Stopping on an estimate.** A run stops when its estimated standard error
  is small, which favours runs whose error is underestimated. The naive
  interval is expected to lose coverage for that reason; whether the loss is
  visible at 400 replicas is part of the question.
- **The standard error is not the error.** It describes the subspace the
  current solve retained. Bias from truncated directions is not in it, so a
  met target does not bound the distance to the exact subspace energy.
- **Greedy targeting.** The allocation follows the current Ritz functional. It
  buys nothing for a direction the current solve has truncated away.
- **Matched error assumes a scaling.** The work-normalized ratio is exact only
  if error falls as one over root shots. Bias breaks that.
- **Scope.** Four small banks, ideal shots, no device noise, one estimator
  policy. Nothing here transfers to molecular banks or hardware without a new
  declared study.
- **No certificate.** A `consistent` label reports calibration on this grid. It
  does not make an interval a guarantee.

## Reproduce

The record is produced once, under the pinned record environment:

```bash
python benchmarks/run_target_acquisition.py --jobs 4
python benchmarks/check_target_acquisition.py --verify-history
```

The producer refuses to run on uncommitted sources and never overwrites a
record. The checker verifies that the record embeds this declaration and that
every declared row is present with its derived seed. It checks every run
against the schedule: the shots at its last look, that a stopped run is the
full schedule ended at the first look meeting the target, and that the stopped
interval is widened for 19 looks. It recomputes every summary, comparison and
label from the rows. It then regenerates the first two replicas of every
system. That replay runs only under the record's NumPy and SciPy versions and
while the recorded implementation digests match the live sources; otherwise it
prints the reason and the structural checks stand alone. `--full-replay`
regenerates every row.

## Results

The [record](target_acquisition_results.json.gz) holds all 3,200 rows: four
systems, 400 replicas and two allocations. Each row holds the full schedule's
ledger and both stopped runs. It was produced once, under Python 3.12.3, NumPy
2.5.2 and SciPy 1.18.0, from a clean checkout of the declaration commit: all
thirteen bound source digests equal that commit's bytes. No look failed to
solve. The tables under [Tables](#tables) are generated from the record by
`check_target_acquisition.py --report`, and a test holds this document to them.

### Cost at matched error

Targeting never needed more shots at matched error, and needed fewer in five
of eight cells.

- **Well-conditioned bank.** On `tfim4_orthogonal` targeting needed 2.6 to 2.7
  times fewer shots. Both allocations stopped at nearly the same realized
  error (0.0191 against 0.0193, and 0.0092 against 0.0094), so the raw shot
  ratio and the matched-error figure agree.
- **Ill-conditioned development bank.** On `tfim4_mixed` the figure was 1.42
  at the 0.01 target and unresolved at 0.02 (1.23, interval 0.99 to 1.51),
  where half of the runs of either allocation had stopped by 283 shots per
  setting.
- **Held-out `xxz4_mixed`.** Unresolved at both targets: 1.16 and 1.24, with
  intervals that include one. Targeting did spend fewer shots at the 0.01
  target (ratio 1.17, interval 1.03 to 1.30).
- **Held-out `tfim6_mixed`.** 1.56 and 2.29. These are not costs of reaching
  the target. At 0.01 uniform exhausted the schedule in 117 of 400 runs and
  targeting in 40, and the realized error at stop was 0.0215 and 0.0162, far
  above the target. [Validity](#validity-of-the-stopped-interval) explains why.

### At equal shots

At the last look, with every shot of the schedule spent, targeting had a
mean-squared error 3.1, 2.4 and 3.4 times lower on `tfim4_orthogonal`,
`tfim4_mixed` and `tfim6_mixed`.

On `xxz4_mixed` the comparison is unresolved (0.09, interval 0.00 to 1.95),
because both allocations still had outliers at the last look: a worst error of
1.58 under uniform allocation and 4.55 under targeting, against medians of
0.0016 and 0.0013. The exact overlap of that bank has rank five. The default
solve retained six directions at every look of every run, so one retained
direction was always noise, and no amount of either allocation removed it.

The per-look tables show the same split everywhere. Median error fell
smoothly, and targeting's median was lower than uniform's at 71 of the 72
looks after the pilot. Root-mean-squared error at a fixed look was set by rare
catastrophic solves until late in the schedule: the worst single error was
130,887, on `tfim6_mixed` under targeting at 1,600 shots per setting.

### The stopping rule avoids catastrophic solves

No stopped run was catastrophic. The worst absolute error of any stopped run,
over every system, allocation and target, was 0.121. A catastrophic solve
comes with a very large estimated standard error, so the rule does not stop
there. This holds for both allocations and is a property of the rule, not of
targeting.

### Validity of the stopped interval

On the three four-qubit banks the schedule-valid interval covered in 99.2% to
100% of runs and was labelled `over` in all twelve cells. The naive interval
covered in 94.8% to 98.0% and was never labelled `under`. The loss the naive
interval was expected to show was not visible at 400 replicas, so on these
banks the split over 19 looks cost a half-width 1.53 times larger and bought
nothing measurable.

On the held-out `tfim6_mixed` the stopped interval failed. The schedule-valid
interval covered in 83.2% to 93.0% of runs: `under` in three cells and
`consistent` in one. The naive interval covered in 65.8% to 79.2% and was
`under` in all four. Without any stopping rule, the interval at the last look
covered in 92.0% of uniform runs (`under`) and 93.5% of targeted runs
(`consistent`).

The split by retained rank at the stopping look was not declared and is
exploratory. It points at two mechanisms, both listed as limits before the
run.

- **Truncation.** The exact rank of `tfim6_mixed` is eight. At the 0.02 target,
  85 uniform and 65 targeted runs stopped at a look whose solve had retained
  seven directions. Their mean error was +0.036 and +0.034, against an
  estimated standard error near 0.016, and the schedule-valid interval covered
  73% and 77% of them. At the 0.01 target there were 40 and 17 such runs, and
  it covered 10% and 18%. A solve that drops the eighth direction reports a
  standard error below the target while its energy sits about 0.035 too high.
  The rule can stop there, and the standard error does not contain that shift.
- **Selection.** Runs that stopped at full rank were also biased upward, by
  +0.012 and +0.014 at the 0.02 target. That is what stopping on a small
  estimated standard error would do: it favours runs whose error is
  underestimated.

The development bank `tfim4_mixed` shows the truncation mechanism at a smaller
scale. Its rank-seven stops had a mean error of +0.005 to +0.012, which was
too little to change a label there. So the development banks did not warn of
the failure, and the held-out bank found it.

The stopping semantics are therefore valid in a narrower sense than the name
suggests. Splitting the error budget over the looks accounts for looking many
times. It does not make the interval valid when the solve's truncation can
change between looks. A stopping rule for a truncating estimator needs either
a bound on the truncation bias or a rule that refuses to stop on a rank the
next look does not confirm. Neither is implemented here.

### Adaptive bias

On the well-conditioned bank the full-schedule mean error under targeting was
-0.00083, which is 4.3 of its own standard errors and about a fifth of the
root-mean-squared error of 0.00386. Under uniform allocation it was -0.00025,
or 0.7 standard errors. The targeted estimate is measurably biased where the
uniform one is not. It did not move a coverage label: 94.5% under targeting
against 93.8% under uniform allocation.

### Replay

The checker regenerated the first two replicas of every system, 16 rows, to
within its tolerance. One row of 3,200 has an ill-conditioned look
(`tfim6_mixed`, targeted, replica 239: smallest retained overlap eigenvalue
`1.1e-7` at its ninth look) and is compared only up to that look.

Outside the checker, every row was also regenerated once under a different
BLAS kernel: OpenBLAS's Zen kernel in place of the one it selects on the
producing machine. No row was bitwise equal and none failed the checker's
comparison. The largest relative difference in any ledger energy was `7.4e-9`.
This is one machine and one alternative kernel, not a guarantee for every CPU.

Two bound sources, `clifford_qc/subspace/coverage.py` and
`clifford_qc/subspace/uncertainty.py`, were edited after this record was
produced: an unavailable root now becomes a reported failure instead of an
`IndexError`. Their bytes at the declaration commit are packaged under
`benchmarks/provenance/execution_sources/f8358f87aa28539ca397b6e7a8074bbc90ea9b20/`, and the checker holds the
recorded digests against that snapshot. The replay then runs with the edited
code and still has to reproduce every row it compares, which it does. A bound
source edited without such a snapshot turns the replay off, and a test fails.

### Tables

Cost of reaching a declared standard error. Ratios are uniform over targeted, so above one means targeting is cheaper:

| System | Role | Target | Mean shots per setting (U / T) | Shot ratio | RMSE at stop (U / T) | Shots at matched error | Exhausted (U / T) | Verdict |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| `tfim4_orthogonal` | development | 0.02 | 6,943 / 2,598 | 2.67 [2.61, 2.73] | 0.0191 / 0.0193 | 2.63 [2.20, 3.19] | 0 / 0 | `fewer_shots_at_matched_error` |
| `tfim4_orthogonal` | development | 0.01 | 25,627 / 9,126 | 2.81 [2.79, 2.82] | 0.0092 / 0.0094 | 2.67 [2.21, 3.24] | 0 / 0 | `fewer_shots_at_matched_error` |
| `tfim4_mixed` | development | 0.02 | 504 / 376 | 1.34 [1.17, 1.55] | 0.0128 / 0.0134 | 1.23 [0.99, 1.51] | 0 / 0 | `no_resolved_difference` |
| `tfim4_mixed` | development | 0.01 | 1,561 / 873 | 1.79 [1.57, 2.06] | 0.0072 / 0.0081 | 1.42 [1.10, 1.81] | 0 / 0 | `fewer_shots_at_matched_error` |
| `xxz4_mixed` | held out | 0.02 | 1,281 / 1,154 | 1.11 [0.98, 1.25] | 0.0167 / 0.0163 | 1.16 [0.90, 1.50] | 0 / 0 | `no_resolved_difference` |
| `xxz4_mixed` | held out | 0.01 | 4,068 / 3,487 | 1.17 [1.03, 1.30] | 0.0081 / 0.0078 | 1.24 [0.99, 1.54] | 0 / 0 | `no_resolved_difference` |
| `tfim6_mixed` | held out | 0.02 | 6,501 / 4,453 | 1.46 [1.12, 1.92] | 0.0252 / 0.0244 | 1.56 [1.18, 2.09] | 14 / 0 | `fewer_shots_at_matched_error` |
| `tfim6_mixed` | held out | 0.01 | 25,210 / 19,293 | 1.31 [1.16, 1.48] | 0.0215 / 0.0162 | 2.29 [1.79, 2.96] | 117 / 40 | `fewer_shots_at_matched_error` |

Verdict counts over systems:

| Target | `fewer_shots_at_matched_error` | `no_resolved_difference` | `more_shots_at_matched_error` |
| ---: | ---: | ---: | ---: |
| 0.02 | 2 | 2 | 0 |
| 0.01 | 3 | 1 | 0 |

Mean-squared error at the last look of the full schedule, uniform over targeted at equal shots:

| System | Shots | Uniform RMSE | Targeted RMSE | MSE ratio | Label |
| --- | ---: | ---: | ---: | --- | --- |
| `tfim4_orthogonal` | 409,600 | 0.00683 | 0.00386 | 3.12 [2.53, 3.89] | `targeted_better` |
| `tfim4_mixed` | 2,662,400 | 0.00118 | 0.00077 | 2.36 [1.91, 2.89] | `targeted_better` |
| `xxz4_mixed` | 3,584,000 | 0.09016 | 0.30188 | 0.09 [0.00, 1.95] | `indistinguishable` |
| `tfim6_mixed` | 5,324,800 | 0.01469 | 0.00793 | 3.43 [2.18, 5.26] | `targeted_better` |

Coverage of the exact subspace energy at the nominal 95% level, over all replicas:

| System | Allocation | Full schedule | Stopped at 0.02, schedule-valid | Stopped at 0.02, naive | Stopped at 0.01, schedule-valid | Stopped at 0.01, naive |
| --- | --- | --- | --- | --- | --- | --- |
| `tfim4_orthogonal` | `uniform` | 93.8% consistent | 99.8% over | 95.0% consistent | 99.8% over | 95.8% consistent |
| `tfim4_orthogonal` | `ritz_targeted` | 94.5% consistent | 99.5% over | 95.5% consistent | 100.0% over | 94.8% consistent |
| `tfim4_mixed` | `uniform` | 96.8% consistent | 99.8% over | 98.0% over | 99.8% over | 97.0% consistent |
| `tfim4_mixed` | `ritz_targeted` | 95.2% consistent | 99.8% over | 97.5% over | 99.8% over | 95.8% consistent |
| `xxz4_mixed` | `uniform` | 97.2% over | 99.8% over | 96.0% consistent | 99.8% over | 96.0% consistent |
| `xxz4_mixed` | `ritz_targeted` | 98.8% over | 99.2% over | 95.8% consistent | 99.8% over | 95.2% consistent |
| `tfim6_mixed` | `uniform` | 92.0% under | 92.5% under | 73.5% under | 83.2% under | 65.8% under |
| `tfim6_mixed` | `ritz_targeted` | 93.5% consistent | 93.0% consistent | 79.2% under | 89.8% under | 74.2% under |

Coverage label counts:

| Interval | Under | Consistent | Over |
| --- | ---: | ---: | ---: |
| Full schedule, single look | 1 | 5 | 2 |
| Stopped, schedule-valid | 3 | 1 | 12 |
| Stopped, naive | 4 | 10 | 2 |

Solve failures over all runs: 0.

Error at every look of the full schedule on `tfim4_orthogonal`:

| Shots per setting | Uniform RMSE | Uniform median abs. error | Uniform worst abs. error | Targeted RMSE | Targeted median abs. error | Targeted worst abs. error | Mean retained rank (U / T) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.3442 | 0.2236 | 1.0881 | 0.3442 | 0.2236 | 1.0881 | 4.00 / 4.00 |
| 141 | 0.2570 | 0.1694 | 0.8555 | 0.1760 | 0.1290 | 0.6238 | 4.00 / 4.00 |
| 200 | 0.1887 | 0.1321 | 0.6126 | 0.1097 | 0.0794 | 0.3107 | 4.00 / 4.00 |
| 283 | 0.1470 | 0.0987 | 0.5172 | 0.0801 | 0.0536 | 0.2466 | 4.00 / 4.00 |
| 400 | 0.1144 | 0.0743 | 0.4475 | 0.0611 | 0.0417 | 0.1665 | 4.00 / 4.00 |
| 566 | 0.0867 | 0.0630 | 0.2997 | 0.0488 | 0.0319 | 0.1907 | 4.00 / 4.00 |
| 800 | 0.0678 | 0.0435 | 0.2581 | 0.0379 | 0.0256 | 0.1357 | 4.00 / 4.00 |
| 1,131 | 0.0527 | 0.0354 | 0.1724 | 0.0301 | 0.0215 | 0.1264 | 4.00 / 4.00 |
| 1,600 | 0.0402 | 0.0281 | 0.1337 | 0.0239 | 0.0163 | 0.0986 | 4.00 / 4.00 |
| 2,263 | 0.0342 | 0.0234 | 0.1076 | 0.0203 | 0.0140 | 0.0718 | 4.00 / 4.00 |
| 3,200 | 0.0279 | 0.0213 | 0.0903 | 0.0164 | 0.0108 | 0.0618 | 4.00 / 4.00 |
| 4,525 | 0.0239 | 0.0169 | 0.0650 | 0.0132 | 0.0087 | 0.0540 | 4.00 / 4.00 |
| 6,400 | 0.0199 | 0.0137 | 0.0613 | 0.0110 | 0.0074 | 0.0349 | 4.00 / 4.00 |
| 9,051 | 0.0165 | 0.0115 | 0.0507 | 0.0095 | 0.0063 | 0.0272 | 4.00 / 4.00 |
| 12,800 | 0.0144 | 0.0106 | 0.0405 | 0.0079 | 0.0056 | 0.0223 | 4.00 / 4.00 |
| 18,102 | 0.0117 | 0.0079 | 0.0379 | 0.0067 | 0.0045 | 0.0207 | 4.00 / 4.00 |
| 25,600 | 0.0092 | 0.0065 | 0.0310 | 0.0057 | 0.0039 | 0.0175 | 4.00 / 4.00 |
| 36,204 | 0.0077 | 0.0049 | 0.0243 | 0.0046 | 0.0031 | 0.0146 | 4.00 / 4.00 |
| 51,200 | 0.0068 | 0.0040 | 0.0235 | 0.0039 | 0.0026 | 0.0131 | 4.00 / 4.00 |

Error at every look of the full schedule on `tfim4_mixed`:

| Shots per setting | Uniform RMSE | Uniform median abs. error | Uniform worst abs. error | Targeted RMSE | Targeted median abs. error | Targeted worst abs. error | Mean retained rank (U / T) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 150.9595 | 0.0569 | 3006.2945 | 150.9595 | 0.0569 | 3006.2945 | 7.74 / 7.74 |
| 141 | 23.8320 | 0.0406 | 417.9027 | 25.2177 | 0.0310 | 487.6435 | 7.80 / 7.84 |
| 200 | 3.5016 | 0.0237 | 44.1363 | 6.1690 | 0.0225 | 104.6489 | 7.84 / 7.89 |
| 283 | 14.0328 | 0.0185 | 208.1496 | 25.0616 | 0.0146 | 351.3366 | 7.88 / 7.92 |
| 400 | 4.8397 | 0.0144 | 67.5811 | 66.3093 | 0.0104 | 1324.3754 | 7.92 / 7.95 |
| 566 | 2.9141 | 0.0104 | 42.3150 | 4.9693 | 0.0074 | 75.0626 | 7.96 / 7.97 |
| 800 | 3.7720 | 0.0075 | 72.7689 | 1.1176 | 0.0057 | 17.7645 | 7.99 / 7.98 |
| 1,131 | 5.9707 | 0.0060 | 113.6428 | 0.2247 | 0.0048 | 4.0259 | 7.99 / 7.99 |
| 1,600 | 0.0745 | 0.0048 | 1.0328 | 0.2943 | 0.0034 | 5.8155 | 8.00 / 7.99 |
| 2,263 | 0.2112 | 0.0037 | 4.0740 | 0.5222 | 0.0029 | 10.4382 | 8.00 / 8.00 |
| 3,200 | 0.2661 | 0.0031 | 4.7457 | 0.0052 | 0.0023 | 0.0373 | 8.00 / 8.00 |
| 4,525 | 0.0282 | 0.0029 | 0.5278 | 0.0037 | 0.0021 | 0.0161 | 8.00 / 8.00 |
| 6,400 | 0.0051 | 0.0022 | 0.0406 | 0.0025 | 0.0015 | 0.0116 | 8.00 / 8.00 |
| 9,051 | 0.0041 | 0.0019 | 0.0342 | 0.0020 | 0.0012 | 0.0067 | 8.00 / 8.00 |
| 12,800 | 0.0032 | 0.0015 | 0.0254 | 0.0016 | 0.0011 | 0.0052 | 8.00 / 8.00 |
| 18,102 | 0.0026 | 0.0012 | 0.0192 | 0.0014 | 0.0009 | 0.0050 | 8.00 / 8.00 |
| 25,600 | 0.0018 | 0.0011 | 0.0064 | 0.0011 | 0.0007 | 0.0039 | 8.00 / 8.00 |
| 36,204 | 0.0015 | 0.0010 | 0.0052 | 0.0009 | 0.0006 | 0.0028 | 8.00 / 8.00 |
| 51,200 | 0.0012 | 0.0008 | 0.0041 | 0.0008 | 0.0005 | 0.0027 | 8.00 / 8.00 |

Error at every look of the full schedule on `xxz4_mixed`:

| Shots per setting | Uniform RMSE | Uniform median abs. error | Uniform worst abs. error | Targeted RMSE | Targeted median abs. error | Targeted worst abs. error | Mean retained rank (U / T) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.2811 | 0.0482 | 4.5043 | 0.2811 | 0.0482 | 4.5043 | 6.00 / 6.00 |
| 141 | 0.1198 | 0.0455 | 1.8098 | 0.1383 | 0.0415 | 2.1171 | 6.00 / 6.00 |
| 200 | 0.1114 | 0.0393 | 1.5571 | 0.2772 | 0.0327 | 4.2926 | 6.00 / 6.00 |
| 283 | 0.1269 | 0.0282 | 1.8301 | 0.1214 | 0.0250 | 1.7614 | 6.00 / 6.00 |
| 400 | 0.0381 | 0.0219 | 0.3020 | 0.4617 | 0.0207 | 9.0807 | 6.00 / 6.00 |
| 566 | 0.0276 | 0.0178 | 0.1079 | 0.1909 | 0.0155 | 2.8733 | 6.00 / 6.00 |
| 800 | 0.0235 | 0.0130 | 0.1850 | 0.3580 | 0.0119 | 5.5793 | 6.00 / 6.00 |
| 1,131 | 0.0806 | 0.0113 | 1.2355 | 0.0925 | 0.0099 | 1.6300 | 6.00 / 6.00 |
| 1,600 | 0.0147 | 0.0086 | 0.0696 | 0.3265 | 0.0078 | 4.5945 | 6.00 / 6.00 |
| 2,263 | 0.0409 | 0.0074 | 0.7487 | 0.0819 | 0.0060 | 1.1547 | 6.00 / 6.00 |
| 3,200 | 0.1060 | 0.0060 | 1.5890 | 0.0486 | 0.0057 | 0.6938 | 6.00 / 6.00 |
| 4,525 | 0.0093 | 0.0047 | 0.0614 | 0.3130 | 0.0039 | 6.2546 | 6.00 / 6.00 |
| 6,400 | 0.0531 | 0.0041 | 1.0110 | 0.0385 | 0.0035 | 0.7526 | 6.00 / 6.00 |
| 9,051 | 0.0585 | 0.0034 | 1.1596 | 0.3596 | 0.0029 | 6.9923 | 6.00 / 6.00 |
| 12,800 | 0.0060 | 0.0030 | 0.0511 | 0.0049 | 0.0024 | 0.0298 | 6.00 / 6.00 |
| 18,102 | 0.0059 | 0.0024 | 0.0742 | 0.1148 | 0.0021 | 2.1945 | 6.00 / 6.00 |
| 25,600 | 0.0035 | 0.0021 | 0.0127 | 0.0190 | 0.0017 | 0.2702 | 6.00 / 6.00 |
| 36,204 | 0.0097 | 0.0017 | 0.1825 | 0.1862 | 0.0014 | 2.6550 | 6.00 / 6.00 |
| 51,200 | 0.0902 | 0.0016 | 1.5829 | 0.3019 | 0.0013 | 4.5534 | 6.00 / 6.00 |

Error at every look of the full schedule on `tfim6_mixed`:

| Shots per setting | Uniform RMSE | Uniform median abs. error | Uniform worst abs. error | Targeted RMSE | Targeted median abs. error | Targeted worst abs. error | Mean retained rank (U / T) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 13.6947 | 0.1565 | 173.4298 | 13.6947 | 0.1565 | 173.4298 | 7.60 / 7.60 |
| 141 | 13.3005 | 0.0706 | 158.1990 | 161.4085 | 0.0557 | 3196.1917 | 7.67 / 7.74 |
| 200 | 254.2620 | 0.0521 | 5069.0637 | 39.8681 | 0.0459 | 696.0684 | 7.72 / 7.79 |
| 283 | 19.4025 | 0.0392 | 316.2834 | 79.2796 | 0.0345 | 1422.3671 | 7.73 / 7.84 |
| 400 | 41.0067 | 0.0369 | 513.5405 | 13.4367 | 0.0282 | 144.1055 | 7.77 / 7.87 |
| 566 | 415.6399 | 0.0311 | 8298.5927 | 20.5854 | 0.0281 | 240.0870 | 7.80 / 7.91 |
| 800 | 71.0463 | 0.0262 | 1073.6710 | 99.2544 | 0.0257 | 1594.9116 | 7.87 / 7.93 |
| 1,131 | 25.7713 | 0.0231 | 488.6684 | 32.2479 | 0.0239 | 585.0370 | 7.88 / 7.96 |
| 1,600 | 70.9664 | 0.0243 | 1408.9016 | 6544.3684 | 0.0196 | 130887.0165 | 7.92 / 7.96 |
| 2,263 | 20.5344 | 0.0205 | 261.9925 | 7.5223 | 0.0186 | 111.2076 | 7.92 / 7.98 |
| 3,200 | 20.0134 | 0.0181 | 268.6522 | 36.8855 | 0.0159 | 704.6376 | 7.97 / 7.99 |
| 4,525 | 9.1721 | 0.0182 | 107.0911 | 3.2721 | 0.0144 | 35.5205 | 7.98 / 8.00 |
| 6,400 | 25.7869 | 0.0146 | 511.8763 | 1.5406 | 0.0124 | 15.5503 | 7.99 / 8.00 |
| 9,051 | 3.9878 | 0.0133 | 55.7525 | 0.6855 | 0.0103 | 10.6853 | 8.00 / 8.00 |
| 12,800 | 2.6480 | 0.0119 | 40.8359 | 0.3589 | 0.0095 | 7.1349 | 8.00 / 8.00 |
| 18,102 | 1.6923 | 0.0102 | 31.5441 | 0.0347 | 0.0080 | 0.4813 | 8.00 / 8.00 |
| 25,600 | 0.2859 | 0.0079 | 5.5182 | 0.0329 | 0.0064 | 0.5397 | 8.00 / 8.00 |
| 36,204 | 0.0208 | 0.0073 | 0.1589 | 0.0104 | 0.0056 | 0.0584 | 8.00 / 8.00 |
| 51,200 | 0.0147 | 0.0068 | 0.1214 | 0.0079 | 0.0046 | 0.0309 | 8.00 / 8.00 |
