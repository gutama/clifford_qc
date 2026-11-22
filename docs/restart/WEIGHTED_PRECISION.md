# Preregistered weighted measurement and adaptive precision study

The [result-free declaration](weighted_precision_declaration.json) predeclares a
combined two-part experiment evaluating the variance-first measurement layer
([weighted measurement planning](WEIGHTED_MEASUREMENT.md)) end-to-end inside
the density-free rank-aware adaptive precision controller
([unified workflow](UNIFIED_WORKFLOW.md), [rank-aware stopping](RANK_AWARE_STOP.md)).

**Status:** Result-free declaration committed; no evaluation seeds have been run.
Development seed `501` was used solely to validate bank anchors, calibrate shot
schedules, and measure exact infinite-shot noisy biases before freezing the
declaration, and is strictly excluded from evaluation.

## Study design

All arms use the NumPy-only `method="graph_state"` readout compiler, `packed`
coefficient storage, `assigned` pooling, and the raw Ritz `sampled_precision`
controller with `RankStopPolicy(consecutive_looks=2, margin_delta=0.05)` and
target standard error `0.0016 Ha` (`1.6 mHa`).

- **Part A (Ideal-shot grouping × adaptive precision stopping):** Compares
  unweighted largest-degree (`ldf_*`), bank-weighted sorted insertion (`si_*`),
  two-stage Ritz+overlap-weighted replanning (`ritz_si_*`, with its independent
  pilot shots billed unconditionally to `total_shots`), and opt-in bootstrap
  half-width gating (`si_fc_width`) across a **matched total physical-shot
  schedule** on three well-conditioned 8-qubit molecular banks (`beh2_8q`,
  `lih_8q`, `h4_8q`; 32 outer replicas, 32 bootstrap replicates) and the
  engineered near-cutoff 10-qubit stretched bank (`lih_10q_stretched`; 16 outer
  replicas, 16 bootstrap replicates).
- **Part B (Gate-level noisy readout bias vs. variance for GALIC `NoiseGatePolicy`):**
  Evaluates `si_qwc`, unconstrained `si_fc_unconstrained`, and GALIC noise-gated
  `si_fc_galic_tight` / `si_fc_galic_moderate` on `beh2_8q`, `lih_8q`, and
  `h4_8q` (16 outer replicas) under three explicit gate-level noise models in
  `benchmarks/noisy_readout_sim.py`:
  1. `depol2q_p002_r3`: pure 2Q depolarizing $p_{2\mathrm{q}} = 0.002$, routing
     multiplier $3.0$ ($p_{1\mathrm{q}} = p_{\text{ro}} = 0$);
  2. `depol2q_p010_r3`: pure 2Q depolarizing $p_{2\mathrm{q}} = 0.010$, routing
     multiplier $3.0$ (matching `superconducting-like.json` 2Q noise);
  3. `sc_card_full`: full illustrative `superconducting-like.json` noise
     ($p_{1\mathrm{q}} = 0.001$, $p_{2\mathrm{q}} = 0.010$, routing $3.0$,
     symmetric readout flip $p_{\text{ro}} = 0.02$).

Every `(case, noise_regime, arm)` cell also records the exact infinite-shot
noisy Ritz energy $E_\infty^{\text{noisy}}$ so total MSE decomposes cleanly into
systematic bias and sampling variance.

## Predeclared primary criteria

- **P1a (Weighted SI vs. unweighted LDF at matched $k$):** On `beh2_8q`,
  `lih_8q`, and `h4_8q` ($k \in \{1, 2, 8\}$), test whether weighted sorted
  insertion achieves $\le 0.80\times$ median total physical shots (or strictly
  lower mean shots on discrete-grid ties) with $\ge 30/32$ `target_met` stops,
  zero reduced-rank stops, and non-rejected 95% schedule-adjusted
  bootstrap-spread coverage (Bonferroni $\alpha = 0.05/9$).
- **P1b (Two-stage Ritz replanning vs. bank-weighted FC and QWC):** Report
  paired total physical shots (including paid pilot), RMSE, 95% coverage, and
  empirical covariance-aware $R$ versus state-independent $\hat{R}$ without
  forcing FC to beat QWC.
- **P2 (Stretched rank-stress transfer):** On `lih_10q_stretched`, verify that
  `si_k2` and `ritz_si_k2` each achieve $\ge 12/16$ true precision stops
  (`target_met` and $|E - E_{\text{exact}}| \le 1.6\,\text{mHa}$), $\le 1/16$
  false precision stops, and zero reduced-rank stops, while reporting realized
  shot savings against `ldf_qwc` and `si_qwc`.
- **P3 (GALIC noise-gate bias protection):** Under `depol2q_p002_r3` and
  `depol2q_p010_r3`, verify that `si_fc_galic_tight` achieves strictly lower
  infinite-shot bias and finite-shot RMSE than `si_fc_unconstrained` on all
  three molecules, and strictly fewer false precision stops whenever
  unconstrained FC has $|\text{bias}| > 1.6\,\text{mHa}$. Under `sc_card_full`,
  explicitly report readout-dominated bias across all arms including QWC.

All criterion verdicts, 68% and 95% unconditional/stop-conditional coverage and
median widths, and deterministic 2,000-resample shot-ratio bootstrap CIs
(with Holm–Bonferroni adjustment across P1a, seed `2026112399`) are frozen in
`benchmarks/check_weighted_precision.py` and `analysis` before evaluation.
Unconditional coverage counts only eligible `status == "target_met"` stops in
the numerator over all declared attempts in the denominator (exhaustions and
failures count as uncovered); stop-conditional coverage is reported separately.
Execution requires a completely clean Git worktree and binds every tracked
`.py` file under `clifford_qc/` and `benchmarks/` at the declaration commit,
and `check_record` independently recomputes all 36 Part B noisy anchors.

## Reproduction

Run only at the pinned result-free declaration commit under the declared
environment (`python 3.12.14`, `numpy 2.5.2`, `scipy 1.18.0`, single-threaded
BLAS). Use an empty checkpoint directory outside the repository:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/check_weighted_precision.py --declaration-only --full-anchor
python benchmarks/run_weighted_precision.py --execute --checkpoint-dir /tmp/weighted-precision --jobs 8
python benchmarks/check_weighted_precision.py --report --spot-replay --verify-history
```

Routine CI runs only the fast structural declaration gate:

```bash
python benchmarks/check_weighted_precision.py --declaration-only
python -m pytest -q tests/test_noisy_readout_sim.py tests/test_weighted_precision.py
```

## Scope and non-claims

- Ideal statevector and gate-level depolarizing/readout simulation only; not a
  calibrated hardware run or quantum advantage claim.
- Targets are the declared fixed-bank exact projected energies, not full-CI
  chemical accuracy.
- Uses the raw Ritz `sampled_precision` estimator (no exact-oracle-certified
  control variates). Rank-margin and terminal bootstrap intervals remain
  heuristic, not finite-sample or anytime-valid certificates.
