# Declared rank-instability stress test on stretched molecular banks

The earlier [LiH/H4 adaptive stopping experiment](MOLECULAR_ADAPTIVE_STOP.md)
found stable rank in every look, all SE targets attained, and no primary
95% corrected-spread undercoverage rejection. This study asks whether the
same `acquire_ritz` controller remains reliable when measurement noise changes
the retained rank. The [machine-readable declaration](stretched_molecular_stop_declaration.json)
is frozen and published before the first evaluation draw.

## Banks and stress construction

Reuse the repository's stretched STO-3G FCIDUMPs, with the lowest spatial
orbital frozen doubly occupied. No new SCF or chemistry parameters are fitted.
Independent occupation-sign matrix restriction checks the frozen-core
contraction against the parent integrals, including the constant term.

| Molecule | Geometry | Parent qubits | Active space | Bank directions | Nonidentity words | QWC settings |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| LiH | 4.785 Å | 12 | CAS(2e,5o), 10 qubits | 4 | 10,341 | 1,841 |
| BeH2 | 2.652 Å each bond | 14 | CAS(4e,6o), 12 qubits | 4 | 17,001 | 2,820 |

Use a two-closed-shell-determinant singlet reference (`theta=0.3`,
`phi=0.2`), interleaved JW and four operators: identity, a mixed operator
`I+0.5*G1` and two other pair-transfer anti-Hermitian operators. The exact
normalized overlap has eigenvalues 0.10416701207637513, 1, 1 and
1.8958329879236244. The deliberately near-threshold cutoff `tau_s=0.104`
retains all four exact directions, but finite-shot estimates can cross it.
This is an engineered conditioning stress test, not an estimate of how often
natural A-CASE banks exhibit such rank fluctuations. The continuous target is
the full-rank projected ground energy; the active-space sector ground is a
separate reference, not the interval estimand.

Development-only seed root 317 was used to establish feasibility. The initial
five-direction full-parent 12/14-qubit banks were too expensive for the
bootstrap budget, so we use four-direction 10/12-qubit frozen-core banks.
Both selected cases exhibited actual changes between measured looks on the
development stream. No development outcomes are included in the evaluation.

## Experiment and endpoints

Thirty-two fresh outer replicas per molecule and policy (128 acquisition
runs) use the same pilot seeds within each raw/CV pair. The uniform 250-shot
pilot is followed by Ritz-targeted allocation to cumulative average physical
budgets of 500, 2,000, 8,000 and 32,000 shots per group, stopping at the first
look whose estimated SE is at most 1.6 mHa. CV uses certified number, spin and
diagonal one-body sector shifts; the off-diagonal family is not included to
bound computation. Use assigned pooling and the scalable deterministic QWC
basis cover, not the earlier degree-greedy grouping. The joint sampler
computes exact product-basis marginals of the two-determinant reference;
integer parity-prefill accelerates native cache reconstruction. Development
tests agree with the native MV probability and bootstrap paths. The actual
controller, variance, allocation, solver and bootstrap API are unchanged.

At the stopping look re-solve 16 grouped histogram bootstrap draws (refitting
the CV for each draw when applicable). Report 68% and 95% single-look and
five-look Bonferroni delta and bootstrap-spread intervals. The two primary
tests are one-sided exact binomial undercoverage tests of 95% CV
schedule-adjusted spread coverage at per-molecule alpha 0.025. Separately
require at least four trajectories with measured rank changes across the
64 arm attempts in each molecule to qualify it as a dynamic rank stress
case (descriptive; arms share pilot samples). Report all attempts, including
failed and exhausted runs, terminal rank, rank-changing and reduced-rank
strata, every solved overlap spectrum and cutoff, bootstrap rank mixture,
attainment, false precision stops (target met but absolute error >1.6 mHa),
point RMSE, width, and physical shot costs. The CV and raw policies share
only pilot outcomes; subsequent allocation may differ.

Limitations: small outer/inner counts, deliberately chosen near-threshold
stress design, limited compact active spaces, ideal QWC sampling. Estimated
SE is not an error bound. The conditional-on-final-counts bootstrap does not
replay allocation or stopping. Neither asymptotic delta nor heuristic spread
certifies finite-sample or anytime-valid coverage.

## Reproduction

Execute at the published *result-free declaration commit*; use the pinned
Python/NumPy/SciPy stack and an empty checkpoint folder outside the repo.
Resume only with matching metadata and `--resume`. The short checker does
not sample in CI; `--full-anchor` and `--spot-replay` are deeper manual checks.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/check_stretched_molecular_stop.py --declaration-only --full-anchor
python benchmarks/run_stretched_molecular_stop.py --execute --checkpoint-dir /tmp/stretched-stop --jobs 8
python benchmarks/check_stretched_molecular_stop.py --report
python benchmarks/check_stretched_molecular_stop.py --spot-replay
python benchmarks/check_stretched_molecular_stop.py --verify-history
```

## Results

The frozen declaration was published at `888018772d3dcb4eeb0fd364ff95b8e755ff39af`
before the fresh evaluation. The [compressed result](stretched_molecular_stop_results.json.gz)
contains all 128 attempts, all 374 measured looks, 2,048 bootstrap solves,
and 583,151,500 ideal simulated physical shots. No acquisition or bootstrap
failed and no schedule exhausted. Both cases qualify under the preregistered
rank-instability rule: 31/64 LiH arm trajectories and 24/64 BeH2 trajectories
changed rank during measured acquisition. Every solve retained either three
or four directions; 81/128 terminal bootstraps included both ranks. The
independent occupation-space anchors are:

| Molecule | Exact projected ground (Ha) | Active-sector ground (Ha) | Projected gap (mHa) |
| --- | ---: | ---: | ---: |
| LiH | -7.763632943522596 | -7.782368501626121 | 18.736 |
| BeH2 | -15.200464376073494 | -15.341894473941178 | 141.430 |

The independent parent-determinant restrictions agree with the *entire*
native active Hamiltonian to 1e-9 Ha, and the projected matrices and exact
overlap spectrum agree within the declared numerical tolerances. The active
sector here is fixed N and Sz=0; all bank states are closed-shell singlets.
The systematic projection gaps are not covered by the sampled Ritz intervals.

The checker verified every acquisition ledger, paired pilot fingerprint and
all 2,422 recorded rank decisions. All four designated spot replays (replica
zero of each case/policy) reproduced the records under the declared stack.
The focused acquisition, symmetry, molecular and CI-gate suite passed 279
tests, with one older differently pinned replay skipped. Manuscript and
reproduction-document checks passed. Routine CI verifies the short history,
occupation-space anchor and ledger checks without new measurement draws.

### Stopping and accuracy

| Molecule | Policy | SE met | Rank changed | Terminal rank 3 | Abs. error <= 1.6 mHa | Mean shots (M) | RMSE (mHa) | Bias (mHa) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiH | Raw | 32/32 | 13 | 6 | 25/32 | 3.682 | 148.590 | +64.320 |
| LiH | CV | 32/32 | 18 | 5 | 20/32 | 2.733 | 135.912 | +54.497 |
| BeH2 | Raw | 32/32 | 12 | 8 | 21/32 | 6.169 | 133.784 | +66.780 |
| BeH2 | CV | 32/32 | 12 | 5 | 23/32 | 5.640 | 106.006 | +42.093 |

| Molecule | Policy | 250 | 500 | 2,000 | 8,000 | 32,000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LiH | Raw | 0 | 0 | 32 | 0 | 0 |
| LiH | CV | 0 | 11 | 21 | 0 | 0 |
| BeH2 | Raw | 0 | 0 | 31 | 1 | 0 |
| BeH2 | CV | 0 | 0 | 32 | 0 | 0 |

The second table counts terminal looks by cumulative average physical shots
per group. CV uses 25.8% fewer mean shots on LiH and 8.6% fewer on BeH2
(cheaper in 11/32 and 1/32 paired attempts respectively, tied otherwise).
However, this is **not** matched-accuracy efficiency: all 128 runs met the
estimated-SE threshold, but 39/128 (30.5%) stopped with absolute error above
1.6 mHa. All 24 reduced-rank terminal estimates missed the error target, and
15 of 104 full-rank terminal estimates also missed it. The certified
diagonal CV family remains readable (9 terms for LiH, 10 for BeH2); reducing
within-rank sampling variance does not restore the missing direction.

### Interval coverage and width

All denominators are 32. All intervals exist. Each covered run also attained
the estimated-SE target. The width is the median full interval width.

| Molecule | Policy | Delta single 68% | Delta schedule 68% | Spread single 68% | Spread schedule 68% | Delta single 95% | Delta schedule 95% | Spread single 95% | Spread schedule 95% | Schedule spread 95% width (mHa) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiH | Raw | 19 | 23 | 22 | 24 | 23 | 26 | 24 | 30 | 4.468 |
| LiH | CV | 12 | 22 | 22 | 27 | 25 | 27 | 27 | 32 | 531.619 |
| BeH2 | Raw | 16 | 21 | 16 | 23 | 21 | 23 | 23 | 31 | 6.690 |
| BeH2 | CV | 15 | 25 | 21 | 26 | 25 | 27 | 26 | 30 | 5.633 |

| Molecule | Primary CV 95% schedule spread | Wilson 95% interval | Exact lower-tail p |
| --- | ---: | ---: | ---: |
| LiH | 32/32 (100%) | 89.3–100% | 1.0000 |
| BeH2 | 30/32 (93.8%) | 79.9–98.3% | 0.4800 |

Neither primary cell rejects nominal 95% coverage at alpha .025. This is
**not a successful validation of the precision stopping rule**. The
schedule-adjusted delta intervals cover only 23–27/32 attempts, and the
bootstrap achieves its coverage on LiH CV partly by widening the median
interval to 531.6 mHa, more than 100 times the raw median. The spread reports
instability but cannot prevent stopping, because it is computed only after
the SE stop. Sixteen inner draws can also entirely miss one of the two
rank regimes.

### Rank-stratified diagnostic

| Molecule | Policy | Rank 3: covered / attempts | Rank 4: covered / attempts | Rank-changing: covered / attempts | Mixed-rank bootstrap |
| --- | --- | ---: | ---: | ---: | ---: |
| LiH | Raw | 5/6 | 25/26 | 13/13 | 20/32 |
| LiH | CV | 5/5 | 27/27 | 18/18 | 24/32 |
| BeH2 | Raw | 7/8 | 24/24 | 12/12 | 15/32 |
| BeH2 | CV | 3/5 | 27/27 | 11/12 | 22/32 |

Post hoc descriptive magnitudes illustrate the mechanism: conditional on
terminal rank three, median absolute error is 343.5–344.3 mHa for LiH and
267.6–268.4 mHa for BeH2, while median adjusted spread width is 885–886 mHa
and 690–692 mHa respectively. Conditional on full rank, median absolute
error is 0.44–1.03 mHa and median width is 4.04–14.48 mHa across the four
cells. The rank strata are small, not independent tests of superiority.

### Decision

The existing controller is a useful within-retained-space shot allocator,
but its SE-only stopping policy should **not** be used as a precision
guarantee near a rank cutoff. The primary coverage endpoint alone conceals
the very large truncation error and interval width. Keep the existing
controller opt-in; preregister a separate rank/margin-aware stopping rule
that checks rank agreement and the overlap's statistical distance from the
cutoff, and compare to a bootstrap-width guard and a fixed-budget baseline.
The next study should use fresh seeds and account for any extra look or
bootstrap costs, including exhaustion. No change to the production stopping
rule is made by this experiment.

Follow-up: the [opt-in rank/margin and bootstrap-width guards](RANK_AWARE_STOP.md)
have now been implemented and tested in a separate fresh-seed study. The
rank/margin policy passed both preregistered feasibility cells; it eliminated
reduced-rank stops and lowered false stops from 7/32 to 1/32, at extra shot
cost compared with SE-only and lower cost than the full budget.
