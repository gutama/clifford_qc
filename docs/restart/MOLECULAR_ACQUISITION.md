# Declared molecular acquisition study

**Status: declared, not yet run.** Everything from "Question and estimand" to
"Reproduce" is the declaration. It and the
[declaration file](molecular_acquisition_declaration.json) are committed
before any evaluation seed is drawn and are not edited afterwards;
`check_molecular_acquisition.py --verify-history` proves the order from Git.
Results will be added below the declaration, under their own heading.

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
