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
```

## Results

Pending frozen evaluation. No evaluation outcomes exist in the declaration commit.
