# Bootstrap interval at the acquisition stop

**Result: on fresh seeds the bootstrap interval at the declared stop passed
its criteria, at the cost of width.** The
[target-acquisition study](TARGET_ACQUISITION.md#validity-of-the-stopped-interval)
found that the delta-method interval reported when `acquire_ritz` stops
under-covers on its held-out six-qubit bank. Keeping that stop and reporting
the estimate plus or minus the same `z` times a grouped-bootstrap spread
instead covered in 99.0% to 100% of runs on that bank, where the delta interval
covered 84.5% to 93.5%. Elsewhere both covered in at least 99%. The bootstrap
interval was a median 1.2 to 2.0 times as wide on the six-qubit bank and up to
7.2 times on `tfim4_mixed`, where the delta interval had already covered. A
stricter rule that also waits for the bootstrap spread to meet the target was
explored and costs 1.05 to 3.2 times the shots.

`acquire_ritz(bootstrap_replicates=...)` now reports this interval as an
opt-in, beside the delta-method one. The stopping rule, the defaults and every
frozen record are unchanged.

## Status of the evidence

- The criteria in [`bootstrap_stop_criteria.md`](bootstrap_stop_criteria.md)
  were written after a development screen and before the fresh run, and are
  committed byte for byte. They were fixed in the working session (written at
  08:41 UTC on 2026-11-19, the run started at 08:47 with their SHA-256
  recorded) and committed only afterwards, so unlike the declared studies the
  order cannot be proved from Git.
- The development screen reused the declared study's own seeds, which earlier
  stopping-rule screens had also used. The interval construction was chosen on
  it.
- Both runs used the package at `09dccc5` with a clean tree, from scripts in
  the session's scratch space. Those scripts are packaged byte for byte under
  `benchmarks/provenance/bootstrap_stop/`. `benchmarks/run_bootstrap_stop.py`
  is the committed producer. Replica 0 of every bank regenerated with it
  within the checker's `1e-7` tolerance, bit for bit on the three four-qubit
  banks, and `acquire_ritz`'s new option reproduced every replayed arm-A
  spread.

## The interval

At a look, `bootstrap_ritz_samples` resamples each measurement setting's
outcome histogram at its own shot count and re-solves with the point result's
solver policy, retained-rank threshold included. The spread of the draws is

    r = (q84 - q16) / 2,

which for Gaussian draws is within 0.6% of their standard deviation but
ignores the tails.
The interval is `E +- z r`, centred on the point estimate `E`, with
`z = 3.008`: two-sided `delta = 0.05` split over the schedule's 19 looks, the
quantile `acquire_ritz` already uses for its delta-method interval. It is
labelled `heuristic`.

The delta method linearizes the Ritz value at a fixed retained eigenspace. A
solve that drops a direction reports a standard error below the target while
its energy sits too high, and the linearization cannot see that. Each bootstrap
draw re-solves, so when the data barely support the retained rank, some draws
keep it and some drop it, and the spread grows. Centring on `E` rather than on
the draws means that a draw distribution shifted by truncation widens the
interval rather than moving it.

## Interface

```python
found = acquire_ritz(shared, backend, pilot_shots_per_group=..., round_budgets=...,
                     target_std_error=0.01, bootstrap_replicates=500, bootstrap_seed=7)
found.interval               # delta method, schedule-valid, as before
found.bootstrap_interval     # E +- z r at the same look, same z
found.looks[-1]["bootstrap_spread"]   # r
```

- The bootstrap runs once, after the run has ended. It costs
  `bootstrap_replicates` extra solves and no shot. It uses its own random
  stream, so the shots, the stop and every other reported value are those of
  the same run without it. `schedule["bootstrap"]` records the replicates and
  the seed.
- If a resampled solve fails, `bootstrap_interval` is `None` and the last look
  records `bootstrap_failure`; the acquisition itself is still returned. A
  failed replica is never dropped to produce an interval.
- With `control_variates`, every replica now re-solves the resampled pencil,
  reconstructs the zero-mean sector functionals at that replica's Ritz
  vector and refits their coefficients from its own joint histograms. The
  interval is centred on the corrected point estimate, not the raw Ritz value.
  The schedule records `bootstrap.estimator = "refitted_control_variates"`.
  This combination was **not** included in the fresh-seed check below and
  requires a new declared coverage study on symmetry-conserving banks. The
  [subsequent fixed-shot LiH/H4 study](MOLECULAR_INTERVAL_COVERAGE.md) now
  reports coverage for those two compact molecular banks, but not at an
  adaptive acquisition stop.
- `bootstrap_spread(draws)` and `bootstrap_spread_interval(estimate, draws,
  delta=..., family=...)` in `clifford_qc.subspace.uncertainty` build the same
  interval from draws a caller already has.
- A spread many times `std_error` marks a stop whose retained rank the data do
  not settle. On these banks that was where the extra width went.

For a symmetry-conserving bank, both options can be enabled together:

```python
found = acquire_ritz(shared, backend, pilot_shots_per_group=100,
                     round_budgets=[1000, 2000], target_std_error=0.01,
                     control_variates=certified_family,
                     bootstrap_replicates=500, bootstrap_seed=7)
```

The resampling conditions on the final observed per-setting shot counts; it
does not replay the adaptive allocation or stopping schedule. Re-fitting can
account for some uncertainty of the correction that a fixed-coefficient
delta interval misses, but does not establish finite-sample or optional-stop
coverage. `bootstrap_ritz_samples(..., result=found.result,
control_variates=certified_family)` exposes the corrected draws. No additional
physical shots are required; the extra computation includes a covariance fit
for each replica.

## Development screen

Every stopped run of the declared study (200 replicas, both targets, both
allocations) was regenerated with `acquire_ritz` and matched the committed
record exactly. Its cache at the stop was then bootstrapped with 500 draws.
Four schedule-valid constructions were compared:

- the recorded delta interval `E +- z sigma`;
- the percentile interval between the `delta / 38` and `1 - delta / 38`
  quantiles of the draws;
- the basic interval, which reflects the percentile interval about `E`;
- the spread interval `E +- z r`.

The basic interval failed where it was needed: 83.0% to 92.0% coverage on
`tfim6_mixed`, below the delta interval in all four cells. Truncation biases the
energy upward, and reflecting the draws sends the correction the wrong way.
The percentile interval covered everywhere, but its tails reach the draws that
lost a direction. Its median width was 69 to 1362 times the delta interval's
in the six-qubit cells with more than ten truncated stops, and 322 to 987 times
on `tfim4_mixed`. The spread interval covered 96.5% to 100% on `tfim6_mixed` at
a median 1.15 to 1.88 times the delta width. It was taken to the fresh run.
Other constructions were also looked at: single-look percentile and basic
intervals, and the delta interval with its lower end extended to a bootstrap
quantile or joined with the percentile interval. Each either failed on the
six-qubit bank (the single-look basic interval covered 60% to 69%) or had a
median width at least 19 times the delta width in some six-qubit cell. The
screen record keeps the quantiles all of them need.

## Fresh-seed check

The criteria, as fixed before the run, are in
[`bootstrap_stop_criteria.md`](bootstrap_stop_criteria.md). Seed root
2026111901, 200 replicas per bank, 500 draws per bootstrapped look, everything
else as declared for the target-acquisition study. The true value is the exact
subspace ground energy.

- **Arm A (primary):** the declared delta-method stop, unchanged, reported as
  `E +- z r` at that look. It passes if no cell's coverage is resolved below
  95% (Wilson 95% upper bound at least 0.95) and every `tfim6_mixed` cell
  covers in at least 95% of runs.
- **Arm B (exploratory):** stop at the first look where both the delta-method
  standard error and `r` meet the target; report `E +- z r` there. A run that
  never meets both reports at its last look and counts as exhausted.

The "Delta at A stop" column is `E +- z sigma`, with `sigma` the standard error
the stopping rule tested. The interval `acquire_ritz` itself reports is never
narrower: its per-setting statistics floor each setting's variance at
`range^2 / (4 (N + 1))` (`GroupedWordCache.candidate_group_statistics`), which
binds for settings whose outcomes are nearly unanimous. Rerunning all 3,200
arm-A stops without the bootstrap (`check_bootstrap_stop.py
--reported-intervals`), it was 0 to 2.4% wider and covered in exactly the same
runs. "Wide" means a half-width above `2 z` times the target. The tables are generated by
`check_bootstrap_stop.py --report` from the committed records, and a test holds
this document to them.

### Fresh seeds

| Bank | Allocation | Target | n | Delta at A stop | A: bootstrap at A stop | A median r/σ | A wide | B: bootstrap-confirmed stop | B wide | B shots / A shots | B exhausted |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `tfim4_mixed` | ritz_targeted | 0.01 | 200 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 2.03 | 0.36 | 1.000 [0.981, 1.000] | 0.00 | 2.01 | 0.00 |
| `tfim4_mixed` | ritz_targeted | 0.02 | 200 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 7.24 | 0.77 | 1.000 [0.981, 1.000] | 0.00 | 2.88 | 0.00 |
| `tfim4_mixed` | uniform | 0.01 | 200 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 1.58 | 0.23 | 1.000 [0.981, 1.000] | 0.00 | 1.89 | 0.00 |
| `tfim4_mixed` | uniform | 0.02 | 200 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 6.47 | 0.65 | 1.000 [0.981, 1.000] | 0.00 | 3.07 | 0.00 |
| `tfim4_orthogonal` | ritz_targeted | 0.01 | 200 | 0.995 [0.972, 0.999] | 0.995 [0.972, 0.999] | 1.00 | 0.00 | 0.995 [0.972, 0.999] | 0.00 | 1.05 | 0.00 |
| `tfim4_orthogonal` | ritz_targeted | 0.02 | 200 | 0.995 [0.972, 0.999] | 0.995 [0.972, 0.999] | 1.00 | 0.00 | 0.995 [0.972, 0.999] | 0.00 | 1.07 | 0.00 |
| `tfim4_orthogonal` | uniform | 0.01 | 200 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 1.00 | 0.00 | 1.000 [0.981, 1.000] | 0.00 | 1.09 | 0.00 |
| `tfim4_orthogonal` | uniform | 0.02 | 200 | 0.995 [0.972, 0.999] | 0.995 [0.972, 0.999] | 1.00 | 0.00 | 0.995 [0.972, 0.999] | 0.00 | 1.12 | 0.00 |
| `tfim6_mixed` | ritz_targeted | 0.01 | 200 | 0.895 [0.845, 0.930] | 0.990 [0.964, 0.997] | 1.18 | 0.08 | 0.965 [0.930, 0.983] | 0.01 | 1.51 | 0.10 |
| `tfim6_mixed` | ritz_targeted | 0.02 | 200 | 0.935 [0.892, 0.962] | 1.000 [0.981, 1.000] | 1.68 | 0.30 | 0.990 [0.964, 0.997] | 0.00 | 2.26 | 0.01 |
| `tfim6_mixed` | uniform | 0.01 | 200 | 0.845 [0.788, 0.889] | 0.990 [0.964, 0.997] | 1.16 | 0.21 | 0.950 [0.910, 0.973] | 0.17 | 1.34 | 0.36 |
| `tfim6_mixed` | uniform | 0.02 | 200 | 0.900 [0.851, 0.934] | 1.000 [0.981, 1.000] | 1.97 | 0.42 | 0.995 [0.972, 0.999] | 0.06 | 3.18 | 0.11 |
| `xxz4_mixed` | ritz_targeted | 0.01 | 200 | 1.000 [0.981, 1.000] | 0.990 [0.964, 0.997] | 1.33 | 0.01 | 0.990 [0.964, 0.997] | 0.00 | 1.39 | 0.01 |
| `xxz4_mixed` | ritz_targeted | 0.02 | 200 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 1.38 | 0.00 | 1.000 [0.981, 1.000] | 0.00 | 1.36 | 0.00 |
| `xxz4_mixed` | uniform | 0.01 | 200 | 1.000 [0.981, 1.000] | 1.000 [0.981, 1.000] | 1.20 | 0.00 | 1.000 [0.981, 1.000] | 0.00 | 1.30 | 0.00 |
| `xxz4_mixed` | uniform | 0.02 | 200 | 0.995 [0.972, 0.999] | 1.000 [0.981, 1.000] | 1.26 | 0.00 | 1.000 [0.981, 1.000] | 0.00 | 1.34 | 0.00 |

1600 runs; 0 stops with a solve failure and 0 with a bootstrap failure, each counted as a miss. Arm A criterion: **PASS**.

### Development screen

| Bank | Allocation | Target | n | Truncated stops | Delta | Percentile | Basic | Bootstrap spread | Median width: percentile / delta | spread / delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tfim4_mixed` | ritz_targeted | 0.01 | 200 | 16 | 0.995 | 0.985 | 0.985 | 1.000 | 360.24 | 1.90 |
| `tfim4_mixed` | ritz_targeted | 0.02 | 200 | 29 | 0.995 | 0.985 | 0.985 | 1.000 | 931.22 | 5.79 |
| `tfim4_mixed` | uniform | 0.01 | 200 | 10 | 0.995 | 0.990 | 0.985 | 1.000 | 322.02 | 1.55 |
| `tfim4_mixed` | uniform | 0.02 | 200 | 41 | 1.000 | 0.990 | 0.990 | 1.000 | 987.09 | 4.78 |
| `tfim4_orthogonal` | ritz_targeted | 0.01 | 200 | 0 | 1.000 | 0.995 | 1.000 | 1.000 | 0.96 | 1.01 |
| `tfim4_orthogonal` | ritz_targeted | 0.02 | 200 | 0 | 0.995 | 0.980 | 0.995 | 0.995 | 0.98 | 1.02 |
| `tfim4_orthogonal` | uniform | 0.01 | 200 | 0 | 1.000 | 0.995 | 0.995 | 0.995 | 0.93 | 1.00 |
| `tfim4_orthogonal` | uniform | 0.02 | 200 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.95 | 1.00 |
| `tfim6_mixed` | ritz_targeted | 0.01 | 200 | 8 | 0.905 | 1.000 | 0.855 | 0.965 | 2.79 | 1.15 |
| `tfim6_mixed` | ritz_targeted | 0.02 | 200 | 34 | 0.935 | 1.000 | 0.920 | 0.995 | 556.20 | 1.76 |
| `tfim6_mixed` | uniform | 0.01 | 200 | 19 | 0.840 | 1.000 | 0.830 | 0.990 | 68.94 | 1.19 |
| `tfim6_mixed` | uniform | 0.02 | 200 | 41 | 0.950 | 1.000 | 0.915 | 1.000 | 1361.60 | 1.88 |
| `xxz4_mixed` | ritz_targeted | 0.01 | 200 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 10.26 | 1.31 |
| `xxz4_mixed` | ritz_targeted | 0.02 | 200 | 0 | 0.995 | 1.000 | 1.000 | 1.000 | 12.40 | 1.37 |
| `xxz4_mixed` | uniform | 0.01 | 200 | 0 | 0.995 | 1.000 | 1.000 | 1.000 | 2.70 | 1.20 |
| `xxz4_mixed` | uniform | 0.02 | 200 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 3.97 | 1.28 |

### Reading the fresh result

- **Arm A passed.** On `tfim6_mixed` the delta interval was resolved below 95%
  in three cells, as in the declared study (83.2% to 93.0% there at 400
  replicas). The bootstrap interval covered 99.0% to 100% in all four.
- **One cell lost a little.** On `xxz4_mixed`, targeted at 0.01, the bootstrap
  covered 99.0% against the delta interval's 100%: two misses against none.
  There the spread was below `sigma` in 9% of runs. It still passes.
- **The width goes where the truncation is.** On `tfim4_orthogonal`, whose
  stops never truncate, `r` equals `sigma` to a median ratio of 1.00. On
  `tfim4_mixed` the median ratio was 1.6 to 7.2, and 23% to 77% of intervals
  were wide, although the delta interval had already covered every run there.
  So the bootstrap interval is conservative on banks where truncation is
  frequent but harmless. Its 90th-percentile ratio reached 179 on
  `tfim6_mixed`.
- **Arm B is not a better rule.** It kept coverage at 95.0% to 100% and few
  wide intervals, but spent 1.05 to 3.18 times the shots. On `tfim4_mixed` it
  spent 1.9 to 3.1 times as many for no gain. On `tfim6_mixed` with uniform
  allocation at 0.01, 36% of runs exhausted the schedule, and coverage landed
  at exactly 95.0%. That is why `acquire_ritz` reports the bootstrap interval
  and does not stop on it.

## What this does and does not show

- Four small spin banks under one declared schedule, one fresh seed root, 200
  replicas per bank. With 200 runs a cell resolves coverage below 95% only if
  it falls to 91.5% or less. Molecular banks, other schedules and control
  variates were not tested in this study. The subsequent [preregistered LiH/H4
  adaptive stopping study](MOLECULAR_ADAPTIVE_STOP.md) tests both raw and
  refitted control-variate policies under a separate schedule, on stable-rank
  molecular banks.
- The bootstrap only resamples the histograms that were observed. A direction
  that none of the resampled solves retains is invisible to it, and so is any
  error the data cannot show. The interval is `heuristic`, not a certificate.
- Arm A passing makes the reported interval match its nominal coverage here.
  It does not remove the bias of the stopped estimate, which is unchanged.
- The cost is `bootstrap_replicates` solves at the stop. With 500 draws the
  fresh run, which also bootstrapped every look arm B needed, took 2 h 10 min
  on four workers.
- The stopping rule that the target-acquisition study proposed, one that
  refuses to stop on a rank the next look does not confirm, is still not
  implemented. Arm B is a related rule and was not adopted.

## Reproduce

```bash
python -m pytest tests/test_acquisition.py tests/test_bootstrap_stop.py -q
python benchmarks/check_bootstrap_stop.py --replay 1
python benchmarks/check_bootstrap_stop.py --report
```

`--replay 1` regenerates replica 0 of every bank and allocation with the
committed producer, compares the rows within `1e-7`, and checks that
`acquire_ritz(bootstrap_replicates=...)` reproduces each arm-A stop, spread and
interval. Finite-shot values are comparable only under the NumPy and SciPy the
record names, so under other versions the replay is skipped with a note;
`--require-replay` turns that into a failure. `--reported-intervals --jobs 4`
reruns every arm-A stop to compare `acquire_ritz`'s own interval, in a few
minutes. A full rerun writes a new record and refuses to overwrite an existing
one:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python benchmarks/run_bootstrap_stop.py --mode fresh --jobs 4 \
    --output reproductions/bootstrap_stop.json.gz
python benchmarks/check_bootstrap_stop.py --results reproductions/bootstrap_stop.json.gz
```

`--mode screen` reruns the development screen in about an hour on four
workers. Pass its output to the checker with `--screen`. A failed solve or
bootstrap is recorded on the arm it ends, never dropped, and the checker counts
it as a miss.
