# Declared interval-coverage study

**Result: no interval construction was calibrated for every estimator.** On
the three thresholded estimators the basic bootstrap came closest: it was never
labelled `under` for the default or the repaired solve. The delta method was
calibrated or conservative with default thresholds and failed after calibrated
truncation. The percentile bootstrap under-covered on the well-conditioned bank
at low shots and for the constrained ratio. The constrained ratio's budgets did
not transfer to the held-out system. Details are under [Results](#results).

Everything from "Question and estimand" to "Reproduce" is the declaration. It
and the [declaration file](interval_coverage_declaration.json) were committed
as `c3129bbeb19d110cc1a16d4a059f7d66baa5dec2`, before any evaluation seed was
drawn, and are unchanged. `check_interval_coverage.py --verify-history` proves
the order from Git.

## Question and estimand

Do intervals on a measured Ritz value contain the exact subspace ground energy
at their nominal 68% and 95% rates?

The target is the ground energy of the exact projected problem in the declared
subspace, from the canonical exact solver at its default numerical thresholds.
It excludes the subspace's own distance from the true ground energy, which no
measurement interval addresses. An estimator whose policy truncates physical
directions is biased against this target by construction; that bias counts
against its coverage.

## Systems

All are four-qubit spin models with ideal simulated shots. Generators are the
identity, the first odd-`Y` words of the local pool, and Krylov powers of the
Hamiltonian. These numbers come from exact arithmetic only.

| System | Role | Generators | Exact rank | Exact `kappa_S` | Words | QWC groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tfim4_orthogonal` | development | 4 | 4 | 1.0 | 43 | 8 |
| `tfim4_mixed` | development | 9 | 8 | 194.9 | 189 | 52 |
| `xxz4_mixed` | held out | 9 | 5 | 30.1 | 215 | 70 |

The two TFIM banks are the fixtures every earlier finite-shot test and
exploratory check in this repository used. `xxz4_mixed` is held out from
tuning: it was never sampled before the declared run. It was previously
available in the repository, so held out is not a claim of blind external data.
Its exact overlap has four null directions, which shot noise turns into small
random eigenvalues.

Each system is measured at 2,000 and 20,000 shots per QWC group, uniformly, for
six cells.

## Estimators and intervals

Every arm is a fixed policy of `SharedMeasurement.solve`. On a given replica all
arms read the same joint histograms.

| Arm | Policy | Systems | Replicas | Bootstrap replicates |
| --- | --- | --- | ---: | ---: |
| `threshold_default` | package defaults (`tau_s = 1e-10`) | all | 200 | 200 |
| `calibrated_per_mode` | `calibrate_overlap=True` | all | 200 | 200 |
| `nearest_correlation_repair` | `overlap_repair="nearest_correlation"` | all | 200 | 200 |
| `constrained_ratio` | `ratio_error_budget` 0.005 at 2,000 shots, 0.0016 at 20,000 | mixed banks | 100 | 100 |

Three interval constructions are evaluated at both levels:

- **`delta_normal`**: the Gaussian delta-method interval of `ritz_uncertainty`.
  It applies only to the first two arms.
- **`bootstrap_percentile`**: quantiles of the grouped bootstrap that reruns
  the whole estimator on resampled joint histograms.
- **`bootstrap_basic`**: the same draws reflected about the point estimate,
  `[2E - q_hi, 2E - q_lo]`.

The constrained-ratio budgets were chosen on development seeds of `tfim4_mixed`,
where they were attainable. They are transferred unchanged to `xxz4_mixed`. If a
budget is unattainable there, the failures are the result.

## Seeds and the development split

Evaluation seeds derive from the root `20261116` through NumPy `SeedSequence`:
`[root, cell, replica]` seeds a measurement and `[root, cell, replica, arm + 1]`
its bootstrap. No seed from this root was drawn before the declaration commit.

Development used only the two TFIM banks, with small integer backend seeds
(below 4,000) and one dry run under the root `999`.

## Counting rules

- A failed solve and a failed interval are counted, never dropped. The reported
  coverage rate divides by every replica, so a failure counts as not covered.
  The rate over successful intervals is reported beside it with its own
  denominator.
- A bootstrap in which any replica fails is a failed interval.
- Each rate carries a 95% Wilson interval and one label:

  | Label | Rule |
  | --- | --- |
  | `under` | the Wilson upper limit is below the nominal level |
  | `over` | the Wilson lower limit is above the nominal level |
  | `consistent` | the nominal level lies inside the Wilson interval |

- Rates are also split by the smallest retained overlap eigenvalue of the
  solved metric, at the fixed edges `1e-3`, `1e-2` and `1e-1`. The constrained
  ratio retains no modes, so it uses the smallest measured normalized overlap
  eigenvalue.
- The headline is the count of `under`, `consistent` and `over` labels per arm
  and method, with solve and interval failures.

## What this can and cannot show

There are 112 labelled rates. With 200 replicas a 95% rate has a standard error
of about 1.5 points, and with 100 about 2.2. A few `under` or `over` labels are
expected by chance alone, so single labels are not findings; patterns across
cells are.

Known limits, stated before the run:

- **Finite bootstrap size.** Percentile endpoints from 200 or 100 draws carry
  their own Monte Carlo error. This lowers bootstrap coverage slightly, more so
  for the constrained ratio.
- **Scope.** Three small banks, uniform allocation, ideal shots, no device
  noise. Nothing here transfers to molecular banks or hardware without a new
  declared study.
- **No certificate.** A `consistent` label reports calibration on this grid. It
  does not make an interval a guarantee.

## Reproduce

The record is produced once, under the pinned record environment:

```bash
python benchmarks/run_interval_coverage.py --jobs 4
python benchmarks/check_interval_coverage.py --verify-history
```

The producer refuses to run on uncommitted sources and never overwrites a
record. The checker verifies that the record embeds this declaration, that
every declared row is present with its derived seeds, and that every summary and
label follows from the rows. It then regenerates the first two replicas of
every cell. That replay runs only under the record's NumPy and SciPy versions
and while the recorded implementation digests match the live sources; otherwise
it prints the reason and the structural checks stand alone.
`--full-replay` regenerates every row.

## Results

The [record](interval_coverage_results.json.gz) holds all 4,000 replica rows.
It was produced once, under Python 3.12.3, NumPy 2.5.2 and SciPy 1.18.0, from
the implementation at the declaration commit: all eleven bound source digests
equal that commit's bytes. The tables below are generated from the record by
`check_interval_coverage.py --report`, and a test holds this document to them.

Label counts over cells and levels, with failures summed over cells:

| Arm | Method | Under | Consistent | Over | Solve failures | Interval failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `threshold_default` | `delta_normal` | 0 | 8 | 4 | 0 | 0 |
| `threshold_default` | `bootstrap_percentile` | 1 | 8 | 3 | 0 | 0 |
| `threshold_default` | `bootstrap_basic` | 0 | 11 | 1 | 0 | 0 |
| `calibrated_per_mode` | `delta_normal` | 5 | 6 | 1 | 0 | 0 |
| `calibrated_per_mode` | `bootstrap_percentile` | 1 | 10 | 1 | 0 | 0 |
| `calibrated_per_mode` | `bootstrap_basic` | 2 | 10 | 0 | 0 | 0 |
| `nearest_correlation_repair` | `bootstrap_percentile` | 2 | 8 | 2 | 0 | 0 |
| `nearest_correlation_repair` | `bootstrap_basic` | 0 | 8 | 4 | 0 | 0 |
| `constrained_ratio` | `bootstrap_percentile` | 8 | 0 | 0 | 145 | 55 |
| `constrained_ratio` | `bootstrap_basic` | 5 | 3 | 0 | 145 | 55 |

Coverage of the exact subspace energy at the nominal 68% level, over all replicas:

| System | Shots per group | Arm | `delta_normal` | `bootstrap_percentile` | `bootstrap_basic` |
| --- | ---: | --- | --- | --- | --- |
| `tfim4_orthogonal` | 2,000 | `threshold_default` | 67.0% consistent | 64.5% consistent | 68.5% consistent |
| `tfim4_orthogonal` | 2,000 | `calibrated_per_mode` | 67.0% consistent | 63.5% consistent | 70.0% consistent |
| `tfim4_orthogonal` | 2,000 | `nearest_correlation_repair` | n/a | 63.0% consistent | 72.0% consistent |
| `tfim4_orthogonal` | 20,000 | `threshold_default` | 66.5% consistent | 66.5% consistent | 68.5% consistent |
| `tfim4_orthogonal` | 20,000 | `calibrated_per_mode` | 66.5% consistent | 66.0% consistent | 65.0% consistent |
| `tfim4_orthogonal` | 20,000 | `nearest_correlation_repair` | n/a | 65.0% consistent | 67.5% consistent |
| `tfim4_mixed` | 2,000 | `threshold_default` | 82.5% over | 71.0% consistent | 76.0% over |
| `tfim4_mixed` | 2,000 | `calibrated_per_mode` | 34.5% under | 67.0% consistent | 40.5% under |
| `tfim4_mixed` | 2,000 | `nearest_correlation_repair` | n/a | 49.5% under | 77.5% over |
| `tfim4_mixed` | 2,000 | `constrained_ratio` | n/a | 21.0% under | 66.0% consistent |
| `tfim4_mixed` | 20,000 | `threshold_default` | 70.0% consistent | 70.0% consistent | 71.0% consistent |
| `tfim4_mixed` | 20,000 | `calibrated_per_mode` | 70.0% consistent | 70.0% consistent | 73.5% consistent |
| `tfim4_mixed` | 20,000 | `nearest_correlation_repair` | n/a | 71.5% consistent | 74.5% over |
| `tfim4_mixed` | 20,000 | `constrained_ratio` | n/a | 51.0% under | 70.0% consistent |
| `xxz4_mixed` | 2,000 | `threshold_default` | 79.0% over | 88.5% over | 72.0% consistent |
| `xxz4_mixed` | 2,000 | `calibrated_per_mode` | 58.5% under | 72.5% consistent | 72.5% consistent |
| `xxz4_mixed` | 2,000 | `nearest_correlation_repair` | n/a | 83.5% over | 75.0% over |
| `xxz4_mixed` | 2,000 | `constrained_ratio` | n/a | 0.0% under | 0.0% under |
| `xxz4_mixed` | 20,000 | `threshold_default` | 77.0% over | 78.5% over | 73.0% consistent |
| `xxz4_mixed` | 20,000 | `calibrated_per_mode` | 63.5% consistent | 74.5% over | 67.5% consistent |
| `xxz4_mixed` | 20,000 | `nearest_correlation_repair` | n/a | 71.0% consistent | 74.5% over |
| `xxz4_mixed` | 20,000 | `constrained_ratio` | n/a | 0.0% under | 0.0% under |

Coverage of the exact subspace energy at the nominal 95% level, over all replicas:

| System | Shots per group | Arm | `delta_normal` | `bootstrap_percentile` | `bootstrap_basic` |
| --- | ---: | --- | --- | --- | --- |
| `tfim4_orthogonal` | 2,000 | `threshold_default` | 95.5% consistent | 90.5% under | 97.5% consistent |
| `tfim4_orthogonal` | 2,000 | `calibrated_per_mode` | 95.5% consistent | 90.5% under | 97.0% consistent |
| `tfim4_orthogonal` | 2,000 | `nearest_correlation_repair` | n/a | 89.0% under | 98.0% consistent |
| `tfim4_orthogonal` | 20,000 | `threshold_default` | 96.5% consistent | 95.5% consistent | 96.0% consistent |
| `tfim4_orthogonal` | 20,000 | `calibrated_per_mode` | 96.5% consistent | 94.5% consistent | 97.0% consistent |
| `tfim4_orthogonal` | 20,000 | `nearest_correlation_repair` | n/a | 95.5% consistent | 96.5% consistent |
| `tfim4_mixed` | 2,000 | `threshold_default` | 98.0% consistent | 96.5% consistent | 95.0% consistent |
| `tfim4_mixed` | 2,000 | `calibrated_per_mode` | 55.0% under | 95.5% consistent | 58.0% under |
| `tfim4_mixed` | 2,000 | `nearest_correlation_repair` | n/a | 92.0% consistent | 95.0% consistent |
| `tfim4_mixed` | 2,000 | `constrained_ratio` | n/a | 58.0% under | 90.0% under |
| `tfim4_mixed` | 20,000 | `threshold_default` | 98.5% over | 95.5% consistent | 96.5% consistent |
| `tfim4_mixed` | 20,000 | `calibrated_per_mode` | 98.5% over | 97.0% consistent | 97.5% consistent |
| `tfim4_mixed` | 20,000 | `nearest_correlation_repair` | n/a | 96.0% consistent | 97.0% consistent |
| `tfim4_mixed` | 20,000 | `constrained_ratio` | n/a | 79.0% under | 92.0% consistent |
| `xxz4_mixed` | 2,000 | `threshold_default` | 97.0% consistent | 99.5% over | 96.5% consistent |
| `xxz4_mixed` | 2,000 | `calibrated_per_mode` | 90.0% under | 98.0% consistent | 96.0% consistent |
| `xxz4_mixed` | 2,000 | `nearest_correlation_repair` | n/a | 98.5% over | 97.0% consistent |
| `xxz4_mixed` | 2,000 | `constrained_ratio` | n/a | 0.0% under | 0.0% under |
| `xxz4_mixed` | 20,000 | `threshold_default` | 98.0% consistent | 97.0% consistent | 98.0% consistent |
| `xxz4_mixed` | 20,000 | `calibrated_per_mode` | 91.5% under | 98.0% consistent | 96.0% consistent |
| `xxz4_mixed` | 20,000 | `nearest_correlation_repair` | n/a | 96.0% consistent | 97.5% consistent |
| `xxz4_mixed` | 20,000 | `constrained_ratio` | n/a | 0.0% under | 0.0% under |

Point-estimate error against the exact subspace energy:

| System | Shots per group | Arm | Replicas | Solve failures | Median abs. error | 90th pct. abs. error | Max abs. error | Mean error | Below target |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tfim4_orthogonal` | 2,000 | `threshold_default` | 200 | 0 | 0.0238 | 0.0585 | 0.0955 | -0.0125 | 65.5% |
| `tfim4_orthogonal` | 2,000 | `calibrated_per_mode` | 200 | 0 | 0.0238 | 0.0585 | 0.0955 | -0.0125 | 65.5% |
| `tfim4_orthogonal` | 2,000 | `nearest_correlation_repair` | 200 | 0 | 0.0238 | 0.0585 | 0.0955 | -0.0125 | 65.5% |
| `tfim4_orthogonal` | 20,000 | `threshold_default` | 200 | 0 | 0.0076 | 0.0162 | 0.0316 | -0.0013 | 53.5% |
| `tfim4_orthogonal` | 20,000 | `calibrated_per_mode` | 200 | 0 | 0.0076 | 0.0162 | 0.0316 | -0.0013 | 53.5% |
| `tfim4_orthogonal` | 20,000 | `nearest_correlation_repair` | 200 | 0 | 0.0076 | 0.0162 | 0.0316 | -0.0013 | 53.5% |
| `tfim4_mixed` | 2,000 | `threshold_default` | 200 | 0 | 0.0038 | 0.0195 | 50.3109 | -0.2748 | 62.5% |
| `tfim4_mixed` | 2,000 | `calibrated_per_mode` | 200 | 0 | 0.0083 | 0.0185 | 0.0256 | +0.0077 | 20.0% |
| `tfim4_mixed` | 2,000 | `nearest_correlation_repair` | 200 | 0 | 0.0045 | 0.0246 | 144.2294 | -0.9753 | 72.0% |
| `tfim4_mixed` | 2,000 | `constrained_ratio` | 100 | 0 | 0.0048 | 0.0107 | 0.0156 | -0.0047 | 90.0% |
| `tfim4_mixed` | 20,000 | `threshold_default` | 200 | 0 | 0.0012 | 0.0029 | 0.0069 | -0.0003 | 50.5% |
| `tfim4_mixed` | 20,000 | `calibrated_per_mode` | 200 | 0 | 0.0012 | 0.0029 | 0.0069 | -0.0003 | 50.5% |
| `tfim4_mixed` | 20,000 | `nearest_correlation_repair` | 200 | 0 | 0.0012 | 0.0029 | 0.0069 | -0.0004 | 53.5% |
| `tfim4_mixed` | 20,000 | `constrained_ratio` | 100 | 0 | 0.0012 | 0.0028 | 0.0037 | -0.0008 | 66.0% |
| `xxz4_mixed` | 2,000 | `threshold_default` | 200 | 0 | 0.0066 | 0.0229 | 0.8463 | -0.0069 | 48.0% |
| `xxz4_mixed` | 2,000 | `calibrated_per_mode` | 200 | 0 | 0.0084 | 0.0227 | 0.0533 | +0.0080 | 21.0% |
| `xxz4_mixed` | 2,000 | `nearest_correlation_repair` | 200 | 0 | 0.0087 | 0.0428 | 7.1037 | -0.1212 | 71.0% |
| `xxz4_mixed` | 2,000 | `constrained_ratio` | 100 | 71 | 4.6895 | 4.7090 | 4.7149 | +4.6624 | 0.0% |
| `xxz4_mixed` | 20,000 | `threshold_default` | 200 | 0 | 0.0022 | 0.0055 | 0.6123 | -0.0060 | 58.0% |
| `xxz4_mixed` | 20,000 | `calibrated_per_mode` | 200 | 0 | 0.0022 | 0.0050 | 0.0112 | +0.0007 | 39.5% |
| `xxz4_mixed` | 20,000 | `nearest_correlation_repair` | 200 | 0 | 0.0026 | 0.0080 | 4.1433 | -0.0672 | 70.5% |
| `xxz4_mixed` | 20,000 | `constrained_ratio` | 100 | 74 | 4.6898 | 4.7035 | 4.7173 | +4.6530 | 0.0% |

Coverage at the nominal 95% level by stratification value, pooled over cells (replicas whose interval exists):

| Arm | Stratum | `delta_normal` | `bootstrap_percentile` | `bootstrap_basic` |
| --- | --- | --- | --- | --- |
| `threshold_default` | below 0.001 | 100.0% of 2 | 100.0% of 2 | 50.0% of 2 |
| `threshold_default` | 0.001 to 0.01 | 100.0% of 95 | 98.9% of 95 | 97.9% of 95 |
| `threshold_default` | 0.01 to 0.1 | 97.6% of 703 | 96.9% of 703 | 96.4% of 703 |
| `threshold_default` | at least 0.1 | 96.0% of 400 | 93.0% of 400 | 96.8% of 400 |
| `calibrated_per_mode` | 0.01 to 0.1 | 80.1% of 483 | 96.5% of 483 | 81.0% of 483 |
| `calibrated_per_mode` | at least 0.1 | 93.0% of 717 | 95.0% of 717 | 96.5% of 717 |
| `nearest_correlation_repair` | below 0.001 | n/a | 100.0% of 2 | 0.0% of 2 |
| `nearest_correlation_repair` | 0.001 to 0.01 | n/a | 95.3% of 212 | 96.7% of 212 |
| `nearest_correlation_repair` | 0.01 to 0.1 | n/a | 95.7% of 586 | 96.9% of 586 |
| `nearest_correlation_repair` | at least 0.1 | n/a | 92.2% of 400 | 97.2% of 400 |
| `constrained_ratio` | below 0.001 | n/a | 68.5% of 200 | 91.0% of 200 |

### What the labels show

Single labels are within chance, as declared. These patterns hold across cells.

- **Delta method, default thresholds.** Never `under`. It was `over` at 68% in
  three of the four ill-conditioned cells (77.0% to 82.5%).
- **Delta method, calibrated truncation.** `under` in 5 of 12 rates, at worst
  55.0% at the nominal 95% (`tfim4_mixed`, 2,000 shots). The estimator's mean
  error there is +0.0077: truncation raises the energy, and the
  fixed-coefficient interval does not include that. The percentile bootstrap,
  which reruns the truncation, was `consistent` in the same cell (95.5%).
- **Percentile bootstrap.** `under` at 95% for all three thresholded arms on the
  well-conditioned bank at 2,000 shots (89.0% to 90.5%), where the estimate's
  mean error is -0.0125. It was `consistent` there at 20,000 shots. It was also
  `under` at 68% for the repaired solve on `tfim4_mixed` at 2,000 shots
  (49.5%). A percentile interval repeats an estimator's bias, which probably
  explains this; the study does not isolate the cause.
- **Basic bootstrap.** No `under` label for the default or the repaired solve.
  It was `under` twice for calibrated truncation (40.5% and 58.0%, the same
  cell as above) and `over` four times at 68% for the repaired solve.
- **Constrained ratio, development bank.** The percentile bootstrap was `under`
  in all four rates (21.0% and 51.0% at 68%; 58.0% and 79.0% at 95%). The basic
  bootstrap was `consistent` in three and `under` once (90.0% at 95%, 2,000
  shots).
- **Constrained ratio, held-out bank.** The transferred budgets were
  unattainable. 71 and 74 of 100 solves failed. The others returned energies
  between -0.63 and -0.48 against a target of -5.19: directions with almost no
  variance and no physical value. Coverage was 0%, and 29 and 26 bootstraps
  failed.

### What the point estimates show

These were not labelled, and are reported because the intervals sit on them.

- **Catastrophic replicas.** On `tfim4_mixed` at 2,000 shots the default solve
  missed by more than 0.1 in 7 of 200 replicas, at worst by 50.3. Its intervals
  still contained the target in 6 of those 7 (basic, 95%), because they were
  very wide.
- **Repair did not control that tail.** With the default `tau_s`, the repaired
  solve missed by more than 0.1 in 10 of 200 replicas on the same cell, at
  worst by 144.2. It also had the larger maximum error in both held-out cells
  (7.10 against 0.85, and 4.14 against 0.61).
- **Calibrated truncation removed the tail and added bias.** Its largest error
  on that cell was 0.0256, with only 20.0% of estimates below the target.
- **The constrained ratio removed the tail where its budget was attainable.**
  Its largest error on that cell was 0.0156, with 90.0% of estimates below the
  target.

### Limits

The limits declared above stand. Two were visible in the outcome:

- The held-out result for the constrained ratio is a statement about absolute
  budgets chosen on another system. It says nothing about a budget chosen for
  that system.
- When the record was stamped, the working tree held uncommitted edits to the
  checker, the CI registry and the tests. No bound implementation file or the
  declaration was among them, which the digests above confirm; the stamp's
  `git_dirty` flag is true for that reason.

No estimator, interval or default is changed by this study.

### Replay on a second machine

This section was added after the result was first published. It changes how
one group of rows is replayed and how their counts should be read.

The record was produced on one machine, and its checker passed there. On a
second machine, under the same pinned Python, NumPy and SciPy, the first twelve
replicas of every cell were regenerated. All 240 rows of the three thresholded
arms, and of the constrained ratio on the development bank, matched the record
to `1e-7`. The constrained ratio on the held-out bank did not. Over its first
40 replicas per cell, the outcome matched in 30 and 32 rows and the whole row
in 17 and 26.

Where the transferred budget is unattainable, whether the search returns a
point, gives up in its line search or reaches its iteration limit depends on
rounding, and a different CPU rounds differently. What the two machines agree
on is the result itself:

| `xxz4_mixed`, first 40 replicas | Failed solves, record | Failed solves, second machine | Error of the returned energies |
| --- | ---: | ---: | --- |
| 2,000 shots per group | 28 | 26 | 4.57 to 4.72 on both |
| 20,000 shots per group | 29 | 31 | 4.57 to 4.70 on both |

So the conclusion stands: the budgets were unattainable on the held-out bank,
about seven solves in ten failed, and the rest were far from the target. The
exact counts quoted above for that arm on `xxz4_mixed` (71 and 74 failed
solves, 29 and 26 failed bootstraps) describe the producing machine and should
not be read to the last unit.

The checker now compares those rows by their seeds and shots only, and prints
a note when it does. The rule is read from the record: a searching arm with
any failed solve in a cell. Every other row is regenerated as declared. This
narrows the replay the declaration describes, which regenerates the first two
replicas of every cell.
