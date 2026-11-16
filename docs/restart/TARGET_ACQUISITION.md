# Declared target-aware acquisition study

**Status: declared, not yet run.** This document and the
[declaration](target_acquisition_declaration.json) were committed before any
evaluation seed was drawn. The result is added in a later commit; the
declaration is not edited afterwards, and
`check_target_acquisition.py --verify-history` proves the order from Git.

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
