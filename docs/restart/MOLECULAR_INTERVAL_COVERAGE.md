# Declared molecular interval coverage: LiH and H₄

The [machine-readable declaration](molecular_interval_coverage_declaration.json)
is frozen before any evaluation draw. The question is whether the refitted
symmetry-control-variate spread interval covers the exact three-direction
projected ground energy under repeated ideal QWC measurements; it is not an
interval for the full physical ground energy.

## Frozen protocol

Reuse the two molecular cases, FCIDUMP bytes, 8-qubit interleaved Jordan--Wigner
ordering, 4e/4o singlet sectors, two-determinant state and three generators
from the [cross-domain declaration](cross_domain_ratio_declaration.json).
The operators are in complex `Cl(16,C)`, and the dense occupation-sign oracle
independently reconstructs the projected overlap and Hamiltonian and solves
the definite generalized eigenproblem. Both banks have exact rank three.
The already validated case source is bound by SHA-256; these cases are reused,
not claimed as new blind held-out molecules.

- Each case is measured at 2,000 and 20,000 shots per QWC setting, assigned
  pooling, 64 independent outer replicas per cell (256 physical cache draws).
- Raw and refitted symmetry-shift estimates use the **same** histogram cache
  and default thresholded solve. The certified 14-term number/one-body/spin
  shift family is fitted on each observed cache using the existing API.
- For each cache, 32 grouped multinomial resamples share their solve between
  arms; the correction is refitted on every draw. The implementation is checked
  against both public bootstrap paths on separate development seeds. Using
  paired resamples changes computation, not the statistic or shot budget.
- Nominal 68% and 95% two-sided normal delta and bootstrap-spread intervals
  are reported. The raw delta scale is inferred from its existing API's
  95% radius (including that API's variance floor); the corrected delta uses
  the fitted standard error. Spread is `(q84-q16)/2`; no percentile-tail
  interval is attempted with only 32 draws.
- Evaluation seeds are derived from the declared root and cell/replica/stream
  coordinates; none were used on development runs. Every declared replica
  contributes to the unconditional coverage denominator; failed intervals
  count as misses. Report failure counts, coverage and Wilson 95% uncertainty,
  median valid width, point bias and RMSE. Point metrics are conditional on
  finite point estimates. No failed draw is dropped to shorten an interval.
- Primary outcome: 95% corrected-bootstrap coverage in four cells. Reject
  nominal coverage in a cell only if its lower-tail exact binomial p-value
  is below 0.0125 (Bonferroni 0.05/4); not rejecting is not proof of calibration.
  All other comparisons are descriptive. No promotion or hardware claim.

This bounded first molecular coverage study is deliberately distinct from
the full CISD-like exploratory banks in `SYMMETRY_SHIFTS.md`. With 64 outer
replicas and 32 inner draws uncertainty is material; conclusions must remain
limited to these three-direction banks, fixed shot budgets, and ideal shots.

## Reproduction

Run the producer only at the published declaration commit in its pinned
Python/NumPy/SciPy environment, not on a later branch containing results.
Each completed paired row is checkpointed, and a previously interrupted run
may resume only the missing declared rows at the same commit/environment.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/check_molecular_interval_coverage.py --declaration-only
python benchmarks/run_molecular_interval_coverage.py --execute --checkpoint-dir /tmp/molecular-coverage --jobs 8
python benchmarks/check_molecular_interval_coverage.py --spot-replay
python benchmarks/check_molecular_interval_coverage.py --verify-history --report
```

The final history check requires the result to be committed separately after
the declaration. CI's short checker does not repeat the full bootstrap study.

## Results

Pending at declaration time.
