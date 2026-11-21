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
```

## Results

Pending the frozen run. No evaluation draws are included in this commit.
