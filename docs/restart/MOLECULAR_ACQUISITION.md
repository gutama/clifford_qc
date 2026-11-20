# Declared molecular acquisition study

**Status: run. C1, C2 and C3 pass.** Everything from "Question and estimand"
to "Reproduce" is the declaration. It and the
[declaration file](molecular_acquisition_declaration.json) were committed in
`324fefd05d402774e5c3342284007f5e962167e9`, before any evaluation seed was
drawn, and were not edited afterwards;
`check_molecular_acquisition.py --verify-history` proves the order from Git.
The [results](#results) follow the declaration.

## Question and estimand

The declared [target-acquisition study](TARGET_ACQUISITION.md) and the
[bootstrap check](BOOTSTRAP_STOP.md) used four small spin banks. Exploratory
runs on molecular banks ([SYMMETRY_SHIFTS.md](SYMMETRY_SHIFTS.md)) found much
larger savings from Ritz-targeted allocation and symmetry-shift control
variates, but their systems, seeds and targets were chosen during development.
This study asks, on molecular banks whose generators conserve particle number
and `S_z`:

1. **Cost.** How many physical shots do Ritz-targeted allocation, control
   variates and both together need, relative to uniform allocation, to deliver
   the same error?
2. **Validity.** When a predeclared stopping rule ends a run, do the
   schedule-valid delta-method interval and the bootstrap interval at the stop
   contain the target at their nominal 95% rate?

The target is the ground energy of the exact projected problem in the bank's
subspace, from the canonical exact solver at its default thresholds. It
excludes the subspace's own distance from the true ground energy.

## Systems

Each bank is the identity plus the spin-adapted singles and doubles of its
active space (`spin_adapted_excitations`), measured on a correlated reference:
the CISD state in that pool for the 8-qubit banks, and a fixed random
two-electron singlet (seed 5) for `tri`. The correction family is
`sector_shift_family(n_spatial, electrons, sz=0)`: particle number, its
one-body multipliers and `S_z`. Every family is certified on its bank before
any shot. Exact arithmetic only:

| System | Role | Qubits | Generators | Exact rank | Exact `kappa_S` | Words | Settings | Family terms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tri` (3 orbitals, 2 electrons) | development | 6 | 6 | 6 | 8.3 | 997 | 169 | 10 |
| `h4` (STO-3G 4e4o) | development | 8 | 15 | 15 | 5.8 | 15,846 | 1,689 | 14 |
| `lih` (STO-3G 4e4o) | held out | 8 | 15 | 15 | 6.0 | 15,846 | 1,689 | 14 |
| `beh2` (STO-3G 4e4o, Be 1s frozen) | held out | 8 | 15 | 15 | 6.0 | 14,350 | 1,697 | 14 |

What "held out" means here, stated plainly:

- `tri` and `h4` were run under acquisition during development (200 and 40
  replicas), with their estimates inspected.
- `lih` was run only as fixed-size caches (control variates after the fact,
  estimates inspected). It was never run under an adaptive schedule.
- `beh2` was never run before this declaration except to time it, under
  throwaway seeds, with no estimate printed or inspected.

All four FCIDUMPs were in the repository, so held out is not a claim of blind
external data.

## Schedule and arms

- **Pilot.** 100 shots of every setting.
- **Rounds.** 14 further rounds; cumulative shots per setting, averaged over
  settings: 100, 141, 200, 283, 400, 566, 800, 1,131, 1,600, 2,263, 3,200,
  4,525, 6,400, 9,051, 12,800. A round's budget is the number of settings times
  the step. This is the exploratory schedule, unchanged.
- **Looks.** 15: after the pilot and after every round.
- **Estimator.** Package defaults of `SharedMeasurement.solve`.

Four arms, from three seeded runs per replica that all read the same pilot:

| Arm | Allocation | Control variates | From run |
| --- | --- | --- | --- |
| `uniform` | uniform | no | the uniform run's raw ledger |
| `uniform_cv` | uniform | refitted at every look | the uniform run |
| `targeted` | Neyman on the raw Ritz functional | no | the targeted run |
| `targeted_cv` | Neyman on the corrected functional | refitted at every look | the corrected targeted run |

Under uniform allocation the correction changes no shot, so `uniform` and
`uniform_cv` share one run's data and differ only in what they report and
stop on.

**Stopping.** Two targets on the reported standard error, 0.01 and 0.005
hartree, the targets used in development on `tri` and `h4`, transferred
unchanged. A run stops at the first look whose standard error (corrected, for
the `_cv` arms) is at most the target. Each run is executed over the whole
schedule and the stop is applied to its ledger, which is exact because a
stopping rule only truncates the run it would otherwise have continued.

**Intervals at the stop.**

- *Delta*: the schedule-valid interval `acquire_ritz` reports, `E +- z sigma`
  with `z = 2.935` (two-sided 0.05 split over 15 looks).
- *Bootstrap*: `E +- z r` with the same `z` and `r = (q84 - q16) / 2` of
  grouped bootstrap draws of the stop's data, each re-solved, with the
  correction refitted on every draw for the `_cv` arms
  (`acquire_ritz(bootstrap_replicates=...)`). The bootstrap sets the compute,
  so it is planned per system:

  | Systems | Arms | Replicas | Draws per stop |
  | --- | --- | ---: | ---: |
  | `tri` | all four | all 200 | 200 |
  | `lih`, `beh2` | `targeted_cv` | the first 100 | 100 |
  | `h4` | none | – | – |

  Measured in development, one replica of an 8-qubit bank costs 110 to 155 s
  for its three runs, and one bootstrap draw about 0.75 s uncorrected and
  1.75 s corrected. On four workers the whole study takes about ten hours.

## Estimands and labels

As in the target-acquisition study. Every ratio is reference over arm, so a
value above one means the arm is cheaper, with a 95% percentile interval from
2,000 paired bootstrap resamples of whole replicas.

- **At matched error**: the work-normalized ratio `mean shots x MSE at stop`.
  Verdicts `fewer_shots_at_matched_error`, `no_resolved_difference`,
  `more_shots_at_matched_error`. Comparisons: `uniform` over each of
  `uniform_cv`, `targeted` and `targeted_cv`; and `targeted` over
  `targeted_cv` (what the correction adds to targeting).
- **At equal shots**: for the same pairs, the ratio of mean-squared errors at
  the last look of the full schedule. Labels `arm_better`, `indistinguishable`,
  `reference_better`.
- **Coverage**: each interval's rate over all replicas, a 95% Wilson interval,
  and the label `under` (upper limit below 95%), `over` (lower limit above) or
  `consistent`.
- Raw shot ratios, RMSE at stop, exhausted counts and error at every look are
  tabulated beside them.

## Criteria, fixed now

On the held-out systems only (`lih`, `beh2`), both targets: four cells.

- **C1, cost.** `targeted_cv` is `fewer_shots_at_matched_error` against
  `uniform` in all four cells.
- **C2, delta validity.** The delta interval of `targeted_cv` is not `under`
  in any of the four cells.
- **C3, bootstrap validity.** The bootstrap interval of `targeted_cv` is not
  `under` in any of the four cells.

Each criterion is reported pass or fail on its own. Everything else is
reported with its label and no threshold, and the development systems are
reported but enter no criterion.

## Seeds and the development split

Evaluation seeds derive from the root `20261120` through NumPy `SeedSequence`:
`[root, system, replica]` for a replica's measurements, `[root, 1000000 +
system, slot]` for the paired comparison bootstrap, and `[root, 2000000 +
system, replica, 10 * arm + target]` for an interval bootstrap. No seed from
this root is drawn before the declaration commit.

Development used the exploratory roots of `explore_control_variates.py`
(`tri`, `h4`, `lih` caches), the timing seeds 777 and 12345 on all four
systems, and the root `424242` for this producer's own development and tests
on `tri`.

## Counting rules

- A look that cannot be solved ends the run as a failure: it keeps its shots,
  counts toward mean shots and counts as not covered by both intervals.
- A bootstrap that fails at a stop counts as not covered by the bootstrap
  interval only.
- Coverage divides by every replica. A run that never meets its target is
  `schedule_exhausted`, reported at its last look and counted.

## Execution

Under the committed record environment (Python 3.12, NumPy 2.5.2, SciPy
1.18.0), one BLAS thread and clean committed sources. Each bank's setting
outcome distributions are compiled once, before the workers fork, and every
replica reseeds that backend (`prepare_grouped_state`, then `reseed`), which
draws the same stream as a fresh backend; a test holds a row to it. Rows
are written as they complete, so the run can resume after an interruption
without changing any row. A checker verifies the record against this
declaration, recomputes every summary and verdict, can replay the first
replicas of every system with `acquire_ritz`, and proves from Git that the
declaration predates the record.

## What this can and cannot show

- Ideal simulated shots on four small active spaces, one schedule, one
  estimator policy. Nothing transfers to hardware noise or larger molecules
  without its own study.
- Every exact bank here keeps full rank, so the truncation failure of the spin
  study is not expected; the validity question is about adaptive bias, the
  refitted correction and stopping on an estimated error.
- The corrected value refits its coefficients from the same data, an `O(1/N)`
  bias with a slightly optimistic standard error. The bootstrap includes the
  refit; the delta interval does not.
- Matched error assumes error falls as one over root shots.
- With 200 replicas a cell resolves coverage below 95% only at 91.5% or less.
- A pass on C1 to C3 is calibration and cost on this grid, not a certificate.

## Replicas, roles and costs at a glance

| System | Role | Replicas | Bootstrap |
| --- | --- | ---: | --- |
| `tri` | development | 200 | every arm, 200 draws |
| `h4` | development | 100 | none |
| `lih` | held out | 200 | `targeted_cv`, first 100 replicas, 100 draws |
| `beh2` | held out | 200 | `targeted_cv`, first 100 replicas, 100 draws |

The replica counts, bootstrap plan and targets were chosen from development
timings and the development results on `tri` and `h4`, before any evaluation
seed, as a trade between coverage resolution and compute.

## Reproduce

The record is `docs/restart/molecular_acquisition_results.json.gz`. The
producer runs only from a clean tree under the record environment, writes rows
to `work/molecular-acquisition/rows.jsonl` as they finish, resumes from that
file if interrupted, and refuses to overwrite the record:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/run_molecular_acquisition.py --jobs 4
python benchmarks/check_molecular_acquisition.py --verify-history
python benchmarks/check_molecular_acquisition.py --replay 1
python benchmarks/check_molecular_acquisition.py --report
```

The checker verifies the record against this declaration, recomputes the
exact facts of every bank and every summary, verdict and criterion, and with
`--replay N` reruns the first `N` replicas of every system with
`acquire_ritz` itself, which must reproduce every stop, interval and
bootstrap. The replay compiles each 8-qubit bank's outcome distributions
first, about ten minutes per bank.

## Results

The record has 700 rows, one per declared replica, with no solve failure, no
bootstrap failure and no run that exhausted its schedule. It was produced
under the record environment from a clean checkout of `330cdd0`, whose only
change after the declaration commit is the order of the report's comparison
rows. On four workers the run took 14.5 hours, against the declared estimate
of about ten. The checker recomputes every summary, verdict and criterion and
finds no problem, and the declaration predates the record in the commit
graph. Under the record environment, `--replay 1 --require-replay` reruns the
first replica of every system with `acquire_ritz` itself. It reproduces every
stop, estimate, standard error, interval and bootstrap spread of those rows.

### Verdicts

On the held-out banks, at both targets, `targeted_cv` against `uniform`:

| Criterion | Result | Cells |
| --- | --- | --- |
| C1, cost | **PASS** | `fewer_shots_at_matched_error` in all four: 16.8x and 20.2x on `lih`, 15.1x and 17.1x on `beh2` |
| C2, delta validity | **PASS** | 200 of 200 replicas covered in all four (`over`) |
| C3, bootstrap validity | **PASS** | 100 of 100 replicas covered in all four (`over`) |

### What the numbers show

1. **Targeting with the correction beat uniform allocation in every cell of
   every system.** On the held-out banks, `targeted_cv` met each target with
   3.3 to 9.7 times fewer shots than `uniform`, and still stopped with an RMSE
   1.4 to 2.1 times lower. Because the uniform error only falls as shots are
   added, uniform would need at least that many shots to match
   `targeted_cv`'s error. Unlike the matched-error ratio, this statement needs
   no model of how error falls with shots. The development banks agree at
   matched error: 6.6x and 8.1x on `tri`, 9.5x and 17.1x on `h4`. On `tri`,
   `targeted_cv` stopped with a higher RMSE than `uniform` (point 5), so there
   the work-normalized ratio is the only cost statement. On `h4` at 0.01, it
   is not resolved from `targeted` alone.
2. **Targeting and the correction contribute separately.** On the held-out
   banks, targeting alone is worth 2.8x to 7.7x at matched error. Adding the
   correction to targeting stops at almost the same shots (raw shot ratio 0.97
   to 1.07) but with 2.5x to 5.4x lower work-normalized error, because
   `targeted` stops on a standard error that understates its error and
   `targeted_cv` does not (point 4).
3. **The declared matched-error magnitudes rest on an assumption these runs
   violate.** The declaration noted that matched error assumes error falls as
   one over root shots. Over the schedule's 128-fold range the raw uniform
   RMSE falls 102-fold on `lih` and 63-fold on `beh2`, where that assumption
   predicts 11.3-fold (the per-look tables). C1's direction holds without the
   assumption (point 1). Its magnitudes, 15x to 20x, are a product of shots and
   squared error and should not be read as shot ratios. Read the raw shot
   ratios and the RMSE at the stop together instead.
4. **The raw Ritz value's reported error is optimistic under targeting; the
   corrected value's is not.** On the held-out banks, the RMSE of `targeted` at
   its stop is 1.47 to 1.65 times its reported standard error, and an
   unadjusted 1.96 sigma interval covers only 77% to 87.5% of replicas
   (`under`). The schedule-valid `z = 2.935` absorbs this: that interval still
   covers at least 98% in every held-out cell of every arm. For `targeted_cv`
   the RMSE at the stop is at or below the reported standard error, and even
   the unadjusted interval covers 96.5% to 98.5%.
5. **The bootstrap was wider than needed on the held-out banks.** Its spread
   was 1.41 to 1.76 times the delta standard error, for an interval that
   already covered every replica. On `tri`, the corrected standard error was
   optimistic: the RMSE of `targeted_cv` was 1.24 to 1.37 times it, and the
   delta interval covered 95.5% and 97.5% (`consistent`). There the bootstrap
   (1.35 to 1.40 times wider) covered all 200. This is the refit optimism the
   declaration anticipated, seen on a development bank and not on the
   held-out ones. The bootstrap remains an opt-in check, not the default
   interval.

What this does not show is listed in the declaration and still applies: ideal
shots, four small active spaces, one schedule, one estimator policy. Every
interval here covers at the stop of one schedule; none is a certificate.

## Results by system

Generated by `check_molecular_acquisition.py --report` and checked against the
record by `tests/test_molecular_acquisition.py`. Ratios are reference over
arm, so above one favours the arm. "RMS reported SE" is the root mean square
of the standard error each run stopped on. "Unadjusted 1.96 sigma" is the
pointwise interval, reported beside the declared ones and entering no
criterion.

### `tri` (development)

| Target | Arm | Exhausted | Mean shots per setting | RMS reported SE | RMSE at stop | Delta interval | Bootstrap interval | r / sigma | Unadjusted 1.96 sigma |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 0.01 | uniform | 0 | 3,209 | 0.00920 | 0.00983 | 0.985 [0.957, 0.995] over | 0.980 [0.950, 0.992] consistent | 0.99 | 0.940 [0.898, 0.965] consistent |
| 0.01 | uniform_cv | 0 | 668 | 0.00863 | 0.01029 | 0.980 [0.950, 0.992] consistent | 1.000 [0.981, 1.000] over | 1.11 | 0.910 [0.862, 0.942] under |
| 0.01 | targeted | 0 | 1,456 | 0.00899 | 0.00895 | 0.995 [0.972, 0.999] over | 1.000 [0.981, 1.000] over | 1.03 | 0.950 [0.910, 0.973] consistent |
| 0.01 | targeted_cv | 0 | 338 | 0.00859 | 0.01179 | 0.955 [0.917, 0.976] consistent | 1.000 [0.981, 1.000] over | 1.40 | 0.860 [0.805, 0.901] under |
| 0.005 | uniform | 0 | 12,800 | 0.00452 | 0.00423 | 1.000 [0.981, 1.000] over | 1.000 [0.981, 1.000] over | 0.98 | 0.950 [0.910, 0.973] consistent |
| 0.005 | uniform_cv | 0 | 2,210 | 0.00446 | 0.00470 | 0.990 [0.964, 0.997] over | 0.990 [0.964, 0.997] over | 1.05 | 0.940 [0.898, 0.965] consistent |
| 0.005 | targeted | 0 | 4,919 | 0.00460 | 0.00456 | 0.995 [0.972, 0.999] over | 0.995 [0.972, 0.999] over | 1.03 | 0.955 [0.917, 0.976] consistent |
| 0.005 | targeted_cv | 0 | 944 | 0.00442 | 0.00548 | 0.975 [0.943, 0.989] consistent | 1.000 [0.981, 1.000] over | 1.35 | 0.895 [0.845, 0.930] under |

| Comparison | Equal shots (MSE ratio) | Matched error at 0.01 | Raw shot ratio at 0.01 | Matched error at 0.005 | Raw shot ratio at 0.005 |
| --- | --- | --- | ---: | --- | ---: |
| uniform over uniform_cv | 5.77x [4.57, 7.39] arm_better | 4.38x [3.32, 5.80] fewer_shots_at_matched_error | 4.80x | 4.70x [3.53, 6.34] fewer_shots_at_matched_error | 5.79x |
| uniform over targeted | 2.51x [1.89, 3.31] arm_better | 2.66x [1.94, 3.61] fewer_shots_at_matched_error | 2.20x | 2.24x [1.67, 2.98] fewer_shots_at_matched_error | 2.60x |
| uniform over targeted_cv | 13.90x [10.52, 18.06] arm_better | 6.58x [4.81, 9.05] fewer_shots_at_matched_error | 9.49x | 8.09x [5.82, 11.22] fewer_shots_at_matched_error | 13.56x |
| targeted over targeted_cv | 5.55x [4.17, 7.26] arm_better | 2.48x [1.80, 3.51] fewer_shots_at_matched_error | 4.31x | 3.61x [2.67, 4.87] fewer_shots_at_matched_error | 5.21x |

RMSE at every look of the full schedule:

| Look | Mean shots per setting | uniform | uniform_cv | targeted | targeted_cv |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 0.08141 | 0.03452 | 0.08141 | 0.03452 |
| 1 | 141 | 0.05903 | 0.02743 | 0.04354 | 0.02375 |
| 2 | 200 | 0.04532 | 0.02004 | 0.03312 | 0.01712 |
| 3 | 283 | 0.03196 | 0.01533 | 0.02441 | 0.01367 |
| 4 | 400 | 0.02698 | 0.01239 | 0.01826 | 0.00941 |
| 5 | 566 | 0.02333 | 0.01055 | 0.01525 | 0.00715 |
| 6 | 800 | 0.01817 | 0.00805 | 0.01224 | 0.00615 |
| 7 | 1,131 | 0.01619 | 0.00653 | 0.01035 | 0.00471 |
| 8 | 1,600 | 0.01311 | 0.00535 | 0.00864 | 0.00358 |
| 9 | 2,263 | 0.01112 | 0.00437 | 0.00725 | 0.00292 |
| 10 | 3,200 | 0.00987 | 0.00347 | 0.00589 | 0.00229 |
| 11 | 4,525 | 0.00797 | 0.00296 | 0.00480 | 0.00184 |
| 12 | 6,400 | 0.00665 | 0.00241 | 0.00384 | 0.00151 |
| 13 | 9,051 | 0.00555 | 0.00211 | 0.00311 | 0.00128 |
| 14 | 12,800 | 0.00423 | 0.00176 | 0.00267 | 0.00113 |

Exact rank 6, 169 settings, 200 replicas.

### `h4` (development)

| Target | Arm | Exhausted | Mean shots per setting | RMS reported SE | RMSE at stop | Delta interval | Bootstrap interval | r / sigma | Unadjusted 1.96 sigma |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 0.01 | uniform | 0 | 1,752 | 0.00928 | 0.01073 | 1.000 [0.963, 1.000] over | – | – | 0.900 [0.826, 0.945] under |
| 0.01 | uniform_cv | 0 | 1,229 | 0.00918 | 0.01003 | 0.990 [0.946, 0.998] consistent | – | – | 0.940 [0.875, 0.972] consistent |
| 0.01 | targeted | 0 | 285 | 0.00820 | 0.00958 | 1.000 [0.963, 1.000] over | – | – | 0.930 [0.863, 0.966] consistent |
| 0.01 | targeted_cv | 0 | 242 | 0.00840 | 0.00936 | 1.000 [0.963, 1.000] over | – | – | 0.930 [0.863, 0.966] consistent |
| 0.005 | uniform | 0 | 6,400 | 0.00463 | 0.00498 | 0.990 [0.946, 0.998] consistent | – | – | 0.950 [0.888, 0.978] consistent |
| 0.005 | uniform_cv | 0 | 4,525 | 0.00449 | 0.00448 | 1.000 [0.963, 1.000] over | – | – | 0.940 [0.875, 0.972] consistent |
| 0.005 | targeted | 0 | 594 | 0.00460 | 0.00537 | 1.000 [0.963, 1.000] over | – | – | 0.880 [0.802, 0.930] under |
| 0.005 | targeted_cv | 0 | 522 | 0.00431 | 0.00422 | 1.000 [0.963, 1.000] over | – | – | 0.950 [0.888, 0.978] consistent |

| Comparison | Equal shots (MSE ratio) | Matched error at 0.01 | Raw shot ratio at 0.01 | Matched error at 0.005 | Raw shot ratio at 0.005 |
| --- | --- | --- | ---: | --- | ---: |
| uniform over uniform_cv | 1.74x [1.39, 2.14] arm_better | 1.63x [1.24, 2.17] fewer_shots_at_matched_error | 1.43x | 1.75x [1.31, 2.28] fewer_shots_at_matched_error | 1.41x |
| uniform over targeted | 23.78x [16.06, 34.83] arm_better | 7.71x [5.17, 11.59] fewer_shots_at_matched_error | 6.14x | 9.27x [6.28, 14.32] fewer_shots_at_matched_error | 10.77x |
| uniform over targeted_cv | 33.94x [23.55, 49.59] arm_better | 9.53x [6.65, 13.69] fewer_shots_at_matched_error | 7.26x | 17.06x [12.03, 24.42] fewer_shots_at_matched_error | 12.26x |
| targeted over targeted_cv | 1.43x [0.95, 2.15] indistinguishable | 1.24x [0.83, 1.86] no_resolved_difference | 1.18x | 1.84x [1.24, 2.72] fewer_shots_at_matched_error | 1.14x |

RMSE at every look of the full schedule:

| Look | Mean shots per setting | uniform | uniform_cv | targeted | targeted_cv |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 0.08994 | 0.05255 | 0.08994 | 0.05255 |
| 1 | 141 | 0.06732 | 0.03988 | 0.02472 | 0.01942 |
| 2 | 200 | 0.04928 | 0.03175 | 0.01393 | 0.01137 |
| 3 | 283 | 0.03681 | 0.02476 | 0.00959 | 0.00769 |
| 4 | 400 | 0.02746 | 0.01989 | 0.00754 | 0.00506 |
| 5 | 566 | 0.02152 | 0.01668 | 0.00549 | 0.00421 |
| 6 | 800 | 0.01725 | 0.01371 | 0.00464 | 0.00319 |
| 7 | 1,131 | 0.01372 | 0.01070 | 0.00366 | 0.00265 |
| 8 | 1,600 | 0.01123 | 0.00834 | 0.00280 | 0.00216 |
| 9 | 2,263 | 0.00827 | 0.00683 | 0.00215 | 0.00182 |
| 10 | 3,200 | 0.00655 | 0.00562 | 0.00182 | 0.00157 |
| 11 | 4,525 | 0.00584 | 0.00448 | 0.00153 | 0.00110 |
| 12 | 6,400 | 0.00498 | 0.00382 | 0.00111 | 0.00092 |
| 13 | 9,051 | 0.00420 | 0.00329 | 0.00088 | 0.00077 |
| 14 | 12,800 | 0.00366 | 0.00277 | 0.00075 | 0.00063 |

Exact rank 15, 1,689 settings, 100 replicas.

### `lih` (held out)

| Target | Arm | Exhausted | Mean shots per setting | RMS reported SE | RMSE at stop | Delta interval | Bootstrap interval | r / sigma | Unadjusted 1.96 sigma |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 0.01 | uniform | 0 | 2,344 | 0.00794 | 0.01053 | 1.000 [0.981, 1.000] over | – | – | 0.965 [0.930, 0.983] consistent |
| 0.01 | uniform_cv | 0 | 1,640 | 0.00805 | 0.00713 | 1.000 [0.981, 1.000] over | – | – | 0.980 [0.950, 0.992] consistent |
| 0.01 | targeted | 0 | 302 | 0.00758 | 0.01141 | 1.000 [0.981, 1.000] over | – | – | 0.850 [0.794, 0.893] under |
| 0.01 | targeted_cv | 0 | 310 | 0.00743 | 0.00705 | 1.000 [0.981, 1.000] over | 1.000 [0.963, 1.000] over | 1.76 | 0.970 [0.936, 0.986] consistent |
| 0.005 | uniform | 0 | 4,942 | 0.00407 | 0.00522 | 1.000 [0.981, 1.000] over | – | – | 0.930 [0.886, 0.958] consistent |
| 0.005 | uniform_cv | 0 | 3,551 | 0.00406 | 0.00377 | 1.000 [0.981, 1.000] over | – | – | 0.980 [0.950, 0.992] consistent |
| 0.005 | targeted | 0 | 509 | 0.00399 | 0.00587 | 0.990 [0.964, 0.997] over | – | – | 0.875 [0.822, 0.914] under |
| 0.005 | targeted_cv | 0 | 508 | 0.00400 | 0.00362 | 1.000 [0.981, 1.000] over | 1.000 [0.963, 1.000] over | 1.63 | 0.970 [0.936, 0.986] consistent |

| Comparison | Equal shots (MSE ratio) | Matched error at 0.01 | Raw shot ratio at 0.01 | Matched error at 0.005 | Raw shot ratio at 0.005 |
| --- | --- | --- | ---: | --- | ---: |
| uniform over uniform_cv | 2.74x [2.24, 3.38] arm_better | 3.12x [2.59, 3.86] fewer_shots_at_matched_error | 1.43x | 2.67x [2.22, 3.24] fewer_shots_at_matched_error | 1.39x |
| uniform over targeted | 48.73x [37.62, 62.28] arm_better | 6.61x [5.56, 7.89] fewer_shots_at_matched_error | 7.76x | 7.69x [6.26, 9.53] fewer_shots_at_matched_error | 9.70x |
| uniform over targeted_cv | 77.12x [59.50, 99.53] arm_better | 16.84x [13.55, 21.42] fewer_shots_at_matched_error | 7.55x | 20.22x [15.84, 25.59] fewer_shots_at_matched_error | 9.72x |
| targeted over targeted_cv | 1.58x [1.27, 1.99] arm_better | 2.55x [2.09, 3.21] fewer_shots_at_matched_error | 0.97x | 2.63x [2.09, 3.38] fewer_shots_at_matched_error | 1.00x |

RMSE at every look of the full schedule:

| Look | Mean shots per setting | uniform | uniform_cv | targeted | targeted_cv |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 0.27486 | 0.10208 | 0.27486 | 0.10208 |
| 1 | 141 | 0.20267 | 0.07366 | 0.06553 | 0.03130 |
| 2 | 200 | 0.13908 | 0.05557 | 0.02522 | 0.01756 |
| 3 | 283 | 0.09993 | 0.05131 | 0.01381 | 0.00863 |
| 4 | 400 | 0.07441 | 0.03644 | 0.00843 | 0.00509 |
| 5 | 566 | 0.05009 | 0.02362 | 0.00557 | 0.00345 |
| 6 | 800 | 0.03725 | 0.01690 | 0.00383 | 0.00253 |
| 7 | 1,131 | 0.02655 | 0.01034 | 0.00280 | 0.00190 |
| 8 | 1,600 | 0.01967 | 0.00785 | 0.00211 | 0.00136 |
| 9 | 2,263 | 0.01397 | 0.00559 | 0.00152 | 0.00107 |
| 10 | 3,200 | 0.00991 | 0.00463 | 0.00117 | 0.00080 |
| 11 | 4,525 | 0.00730 | 0.00367 | 0.00087 | 0.00063 |
| 12 | 6,400 | 0.00527 | 0.00295 | 0.00066 | 0.00051 |
| 13 | 9,051 | 0.00367 | 0.00204 | 0.00049 | 0.00040 |
| 14 | 12,800 | 0.00270 | 0.00163 | 0.00039 | 0.00031 |

Exact rank 15, 1,689 settings, 200 replicas.

### `beh2` (held out)

| Target | Arm | Exhausted | Mean shots per setting | RMS reported SE | RMSE at stop | Delta interval | Bootstrap interval | r / sigma | Unadjusted 1.96 sigma |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 0.01 | uniform | 0 | 467 | 0.00837 | 0.01110 | 0.995 [0.972, 0.999] over | – | – | 0.930 [0.886, 0.958] consistent |
| 0.01 | uniform_cv | 0 | 294 | 0.00852 | 0.00864 | 1.000 [0.981, 1.000] over | – | – | 0.945 [0.904, 0.969] consistent |
| 0.01 | targeted | 0 | 149 | 0.00717 | 0.01177 | 0.990 [0.964, 0.997] over | – | – | 0.770 [0.707, 0.823] under |
| 0.01 | targeted_cv | 0 | 141 | 0.00596 | 0.00519 | 1.000 [0.981, 1.000] over | 1.000 [0.963, 1.000] over | 1.48 | 0.985 [0.957, 0.995] over |
| 0.005 | uniform | 0 | 1,110 | 0.00433 | 0.00534 | 0.995 [0.972, 0.999] over | – | – | 0.910 [0.862, 0.942] under |
| 0.005 | uniform_cv | 0 | 752 | 0.00444 | 0.00476 | 0.985 [0.957, 0.995] over | – | – | 0.910 [0.862, 0.942] under |
| 0.005 | targeted | 0 | 204 | 0.00371 | 0.00613 | 0.980 [0.950, 0.992] consistent | – | – | 0.770 [0.707, 0.823] under |
| 0.005 | targeted_cv | 0 | 191 | 0.00330 | 0.00311 | 1.000 [0.981, 1.000] over | 1.000 [0.963, 1.000] over | 1.41 | 0.965 [0.930, 0.983] consistent |

| Comparison | Equal shots (MSE ratio) | Matched error at 0.01 | Raw shot ratio at 0.01 | Matched error at 0.005 | Raw shot ratio at 0.005 |
| --- | --- | --- | ---: | --- | ---: |
| uniform over uniform_cv | 1.33x [1.16, 1.49] arm_better | 2.63x [2.15, 3.23] fewer_shots_at_matched_error | 1.59x | 1.86x [1.52, 2.32] fewer_shots_at_matched_error | 1.48x |
| uniform over targeted | 41.44x [31.41, 54.33] arm_better | 2.79x [2.36, 3.30] fewer_shots_at_matched_error | 3.14x | 4.12x [3.35, 5.07] fewer_shots_at_matched_error | 5.43x |
| uniform over targeted_cv | 84.42x [63.90, 112.68] arm_better | 15.10x [12.21, 18.72] fewer_shots_at_matched_error | 3.31x | 17.10x [13.38, 22.40] fewer_shots_at_matched_error | 5.80x |
| targeted over targeted_cv | 2.04x [1.60, 2.65] arm_better | 5.42x [4.36, 6.91] fewer_shots_at_matched_error | 1.05x | 4.15x [3.26, 5.31] fewer_shots_at_matched_error | 1.07x |

RMSE at every look of the full schedule:

| Look | Mean shots per setting | uniform | uniform_cv | targeted | targeted_cv |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 0.06342 | 0.02589 | 0.06342 | 0.02589 |
| 1 | 141 | 0.04379 | 0.01817 | 0.01361 | 0.00533 |
| 2 | 200 | 0.03035 | 0.01263 | 0.00638 | 0.00278 |
| 3 | 283 | 0.02285 | 0.00990 | 0.00379 | 0.00179 |
| 4 | 400 | 0.01540 | 0.00716 | 0.00244 | 0.00120 |
| 5 | 566 | 0.01068 | 0.00554 | 0.00172 | 0.00092 |
| 6 | 800 | 0.00761 | 0.00436 | 0.00128 | 0.00076 |
| 7 | 1,131 | 0.00575 | 0.00362 | 0.00089 | 0.00056 |
| 8 | 1,600 | 0.00425 | 0.00287 | 0.00071 | 0.00044 |
| 9 | 2,263 | 0.00338 | 0.00251 | 0.00057 | 0.00036 |
| 10 | 3,200 | 0.00266 | 0.00208 | 0.00043 | 0.00028 |
| 11 | 4,525 | 0.00216 | 0.00171 | 0.00033 | 0.00023 |
| 12 | 6,400 | 0.00173 | 0.00143 | 0.00024 | 0.00018 |
| 13 | 9,051 | 0.00133 | 0.00109 | 0.00019 | 0.00014 |
| 14 | 12,800 | 0.00101 | 0.00088 | 0.00016 | 0.00011 |

Exact rank 15, 1,697 settings, 200 replicas.

### Criteria

| System | Target | Matched-error verdict | Delta interval | Bootstrap interval |
| --- | ---: | --- | --- | --- |
| `lih` | 0.01 | fewer_shots_at_matched_error | over | over |
| `lih` | 0.005 | fewer_shots_at_matched_error | over | over |
| `beh2` | 0.01 | fewer_shots_at_matched_error | over | over |
| `beh2` | 0.005 | fewer_shots_at_matched_error | over | over |

C1: **PASS**. C2: **PASS**. C3: **PASS**.
