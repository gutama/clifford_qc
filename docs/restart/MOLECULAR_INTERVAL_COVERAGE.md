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

Completed once on the pinned stack at the result-free declaration commit
[`0c42005afa266d1704b5f2e13d9a7553a4da1944`](https://github.com/gutama/clifford_qc/commit/0c42005afa266d1704b5f2e13d9a7553a4da1944),
with all 256 paired outer rows and 8,192 shared bootstrap solves in
[molecular_interval_coverage_results.json.gz](molecular_interval_coverage_results.json.gz).
All 256 measured solves retained rank three. Every interval succeeded; no
failed point or bootstrap draw was discarded. Ten of the 14 certified shift
terms were readable per point in each molecule; unavailable terms were not
imputed. Total physical measurement accounting is 1,928,960,000 **simulated**
logical shots, shared across the two estimators; bootstrap adds no shots.

Independent verification confirmed all row coordinates, seeds, recorded
source identities, exact molecular anchors and shot counts. Replica zero in
each of the four cells replayed successfully with matching histogram digests.
As an additional cross-check, the separate raw and corrected public
`bootstrap_ritz_samples` calls on the 2,000-shot replica-zero cache of each
molecule reproduced all 128 compared bootstrap draws with maximum absolute
difference zero on the pinned stack.

### Exact, independent reference energies

| Molecule | QWC settings | Exact projected E (Ha) | Exact singlet ground E (Ha) | Projected gap (mHa) |
| --- | ---: | ---: | ---: | ---: |
| LiH | 669 | -7.8630481789 | -7.8632225509 | 0.174 |
| H₄ | 701 | -2.1479286781 | -2.1803166143 | 32.388 |

The independently formed projected matrices agree with the native matrices,
including the complex reference coherence; the projected gap is not sampled
estimation error, and the intervals do not cover that systematic gap.

### Matched-shot outcomes

| Molecule | Shots/group | Arm | Delta 68% | Spread 68% | Delta 95% | Spread 95% | 95% spread median width (mHa) | Point RMSE (mHa) | Point bias (mHa) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiH | 2,000 | raw | 41/64 (64.1%) | 40/64 (62.5%) | 61/64 (95.3%) | 61/64 (95.3%) | 26.205 | 7.271 | -1.817 |
| LiH | 2,000 | refitted_cv | 43/64 (67.2%) | 39/64 (60.9%) | 62/64 (96.9%) | 62/64 (96.9%) | 14.279 | 3.649 | -0.105 |
| LiH | 20,000 | raw | 48/64 (75.0%) | 45/64 (70.3%) | 61/64 (95.3%) | 59/64 (92.2%) | 7.176 | 1.803 | 0.067 |
| LiH | 20,000 | refitted_cv | 43/64 (67.2%) | 39/64 (60.9%) | 62/64 (96.9%) | 61/64 (95.3%) | 4.181 | 1.057 | -0.053 |
| H₄ | 2,000 | raw | 43/64 (67.2%) | 37/64 (57.8%) | 63/64 (98.4%) | 60/64 (93.8%) | 27.728 | 7.371 | -0.288 |
| H₄ | 2,000 | refitted_cv | 42/64 (65.6%) | 37/64 (57.8%) | 61/64 (95.3%) | 59/64 (92.2%) | 21.191 | 5.994 | 0.446 |
| H₄ | 20,000 | raw | 48/64 (75.0%) | 46/64 (71.9%) | 63/64 (98.4%) | 62/64 (96.9%) | 8.694 | 2.101 | -0.121 |
| H₄ | 20,000 | refitted_cv | 44/64 (68.8%) | 40/64 (62.5%) | 62/64 (96.9%) | 63/64 (98.4%) | 6.739 | 1.726 | -0.405 |

Coverage is unconditional (64 outer replicas per cell). Width is the median
full interval width among valid intervals; RMSE and bias refer to the point
estimate, not to a confidence interval. The reproducible summary is computed
from the raw draws by the checker, not fitted after observing coverage.

### Preregistered primary endpoint

| Molecule | Shots/group | Covered | Wilson 95% interval | Lower-tail p |
| --- | ---: | ---: | ---: | ---: |
| LiH | 2,000 | 62/64 (96.9%) | 89.3–99.1% | 0.8361 |
| LiH | 20,000 | 61/64 (95.3%) | 87.1–98.4% | 0.6265 |
| H₄ | 2,000 | 59/64 (92.2%) | 83.0–96.6% | 0.2156 |
| H₄ | 20,000 | 63/64 (98.4%) | 91.7–99.7% | 0.9625 |

**No cell rejects nominal 95% corrected-spread coverage at the preregistered
per-cell threshold 0.0125.** The finite experiment does not prove calibration:
H₄ at 2,000 shots/group covers in only 59 of 64 trials. Corrected-spread coverage
at 68% is descriptively below nominal in every cell (37–40 of 64); this, plus
the small inner bootstrap size, warrants a separate stronger calibration study.

At matched physical shots, corrected-spread median 95% width decreased by
22.5–45.5%, and point RMSE decreased by
17.8–49.8% across the four cells (descriptive, not a test of superiority).
The strongest reduction appears in LiH. The corrected delta intervals are
also useful; the spread bootstrap does not consistently improve coverage.
There is no hardware, full-CISD-bank, adaptive-stopping or quantum advantage
claim. A separately declared acquisition-stop study with wider bootstrap
replication remains the next scientific step.
