# Declared interval-coverage study

**Status: declared, not yet run.** This document and the
[declaration](interval_coverage_declaration.json) were committed before any
evaluation seed was drawn. The result is added in a later commit; the
declaration is not edited afterwards, and
`check_interval_coverage.py --verify-history` proves the order from Git.

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
