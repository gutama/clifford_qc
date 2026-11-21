# Declared molecular adaptive acquisition stopping

This study extends [fixed-shot molecular interval coverage](MOLECULAR_INTERVAL_COVERAGE.md)
to the existing `acquire_ritz` controller. The [machine-readable declaration](molecular_adaptive_stop_declaration.json)
is frozen in a result-free commit before drawing evaluation outcomes. No new
chemistry inputs or changes to the estimator are made.

## Design

Reuse exactly the LiH and H4 eight-qubit CAS(4e,4o), rank-three projected
banks and ideal grouped measurement model of the fixed-shot experiment. The
estimand is the independently reconstructed exact projected ground energy,
not the full singlet-sector ground state. The pilot is uniform at 1,000
shots per QWC setting. Subsequent Ritz-targeted allocations reach cumulative
physical shot budgets equal to 2,000, 4,000, 8,000, 16,000 and 32,000 times the
number of settings. Stop at the first look with estimated SE <= 1.6 mHa.
Compare the raw and certified symmetry-control-variate (CV) policies with a
paired fresh seed per replica (identical pilot, not identical subsequent data).

Each molecule and arm has 64 independent outer attempts. Re-solve 64 grouped
histogram bootstrap draws at the terminal look, refitting CV when enabled.
Compare single-look and six-look Bonferroni-adjusted delta and bootstrap-spread
intervals at 68% and 95%. The primary endpoint is 95% schedule-adjusted CV
bootstrap-spread coverage, separately on each molecule (one-sided exact
binomial lower-tail tests at 0.025 each). All attempts enter coverage and cost
denominators; failed intervals are uncovered and schedule exhaustion is not
target attainment. Report joint attainment/coverage as well as marginal and
conditional coverage, shot cost, interval widths, rank and point errors.

The SE threshold is not a bound on absolute error or interval half-width.
Bootstrap resampling conditions on the final group counts; it does not replay
the adaptive controller. Gaussian schedule correction and bootstrap spread
remain asymptotic and heuristic respectively, not certificates. The small
outer and inner replica counts limit test power and spread precision, and no
hardware or larger-bank inference follows.

## Reproduction

Use the pinned environment in the declaration. The independent anchor and
ledger checker requires no expensive replay in routine CI. Run the producer
only at the published result-free declaration commit, with a new checkpoint
directory outside the tree (add `--resume` to continue an interrupted run).

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/check_molecular_adaptive_stop.py --declaration-only
python benchmarks/run_molecular_adaptive_stop.py --execute --checkpoint-dir /tmp/lih-h4-stop --jobs 8
python benchmarks/check_molecular_adaptive_stop.py --report
python benchmarks/check_molecular_adaptive_stop.py --spot-replay
python benchmarks/check_molecular_adaptive_stop.py --verify-history
```

## Results

Declaration commit: `bd381c97bd10ae5216f281129371aabe3376fb6e` (published before
evaluation). The [compressed record](molecular_adaptive_stop_results.json.gz)
contains all 256 acquisitions, their physical batch ledgers, looks, terminal
intervals, schedule, seeds and cache fingerprints. There were no failed
acquisitions, bootstrap failures or exhausted schedules: 917 measured solves,
16,384 bootstrap solves and 1,237,644,000 simulated physical shots. All 917
looks retained rank three. The independent occupation-sign matrix construction
reproduces the exact projected targets to within 1e-9 Ha (LiH -7.863048178906949,
H4 -2.1479286780747042 Ha), which are 0.1744 and 32.3879 mHa above the full
singlet-sector grounds respectively. The corrected functional used 10 or 11
readable members of the 14 certified sector shifts.

The independent checker verified all 256 ledgers and the matched pilot
fingerprints. Four designated spot replays (replica zero of each cell)
reproduced the shot ledgers, fingerprints and numerical outputs under the
pinned environment. The focused regression suite passed 203 tests (one
older, differently pinned spin-study replay skipped); the CI-gate test suite
passed 73 tests. Routine CI checks history, anchors and ledgers only.

### Acquisition cost and point accuracy

| Molecule | Policy | SE target met | Mean shots (millions) | Median shots (millions) | RMSE (mHa) | Bias (mHa) | Absolute error <= 1.6 mHa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiH | Raw | 64/64 | 2.801 | 2.676 | 1.515 | +0.181 | 44/64 |
| LiH | CV | 64/64 | 2.342 | 2.676 | 1.210 | +0.267 | 51/64 |
| H4 | Raw | 64/64 | 8.587 | 11.216 | 1.206 | +0.043 | 54/64 |
| H4 | CV | 64/64 | 5.608 | 5.608 | 1.355 | -0.350 | 56/64 |

| Molecule | Policy | 1k | 2k | 4k | 8k | 16k | 32k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiH | Raw | 0 | 0 | 61 | 3 | 0 | 0 |
| LiH | CV | 0 | 16 | 48 | 0 | 0 | 0 |
| H4 | Raw | 0 | 0 | 0 | 30 | 34 | 0 |
| H4 | CV | 0 | 0 | 0 | 64 | 0 | 0 |

The second table counts terminal looks by cumulative average physical shots
per group. CV used 16.4% fewer mean shots on LiH and 34.7% fewer on H4;
in paired runs it was cheaper in 18/64 and 34/64, tied otherwise, and never
dearer. LiH RMSE fell 20.1%, whereas H4 RMSE rose 12.4% at its lower shot
cost. These compare policies at the same **estimated SE** target, not at
matched actual RMSE. Only 44–56 of 64 runs per cell had absolute point error
within 1.6 mHa even though all 64 met the estimated-SE target.

### Interval coverage at the stop

All denominators are 64, with no missing intervals. Each 95% covered attempt
also attained the estimated-SE target. Width is median full interval width.

| Molecule | Policy | Delta single 68% | Delta schedule 68% | Spread single 68% | Spread schedule 68% | Delta single 95% | Delta schedule 95% | Spread single 95% | Spread schedule 95% | Spread schedule 95% width (mHa) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiH | Raw | 31 | 61 | 29 | 61 | 62 | 64 | 61 | 64 | 7.213 |
| LiH | CV | 45 | 59 | 42 | 58 | 59 | 63 | 58 | 62 | 5.621 |
| H4 | Raw | 47 | 61 | 45 | 60 | 61 | 64 | 60 | 62 | 6.530 |
| H4 | CV | 44 | 59 | 40 | 59 | 60 | 62 | 59 | 61 | 6.408 |

### Preregistered primary endpoint

| Molecule | CV 95% schedule spread coverage | Wilson 95% interval | Exact lower-tail p |
| --- | ---: | ---: | ---: |
| LiH | 62/64 (96.9%) | 89.3–99.1% | 0.8361 |
| H4 | 61/64 (95.3%) | 87.1–98.4% | 0.6265 |

Neither primary cell rejects nominal 95% coverage at the preregistered
per-cell alpha of 0.025. The corrected controller reached the target in
every attempt and preserved the empirical 95% coverage endpoint on these
two banks at smaller mean shot cost. Non-rejection is not proof of
calibration or equivalence: the Wilson intervals are wide. Schedule-adjusted
68% intervals were conservative here, whereas single-look spreads had mixed
coverage (29–45/64). At 95%, the schedule-adjusted CV spread is 22.1% narrower
on LiH and 1.9% narrower on H4 than raw. The schedule-adjusted delta also
worked on all four cells, with 62–64/64 coverage. Bootstrap did not consistently
improve coverage or width. All ranks were stable, so this study does not test
the difficult truncation-instability regime seen in the earlier spin study.

The next stronger validation should preregister larger or stretched molecular
banks with unstable retained rank, compare against a non-adaptive equal-shot
or matched-RMSE baseline, and test resampling that replays the complete
acquisition workflow. The present result supports continuing experiments
with the opt-in corrected controller; it does not support changing the
default stop to a claim of chemical accuracy or proving anytime validity.

Follow-up: the [larger stretched molecular rank stress](STRETCHED_MOLECULAR_STOP.md)
has now completed on fresh 10/12-qubit LiH/BeH2 banks. The existing controller
met its estimated-SE target in every attempt but stopped with absolute error
above target in 39/128 runs. Rank-aware stopping remains an open task.
