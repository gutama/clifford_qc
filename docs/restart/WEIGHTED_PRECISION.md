# Preregistered weighted measurement and adaptive precision study

The [result-free declaration](weighted_precision_declaration.json) predeclares a
combined two-part experiment evaluating the variance-first measurement layer
([weighted measurement planning](WEIGHTED_MEASUREMENT.md)) end-to-end inside
the density-free rank-aware adaptive precision controller
([unified workflow](UNIFIED_WORKFLOW.md), [rank-aware stopping](RANK_AWARE_STOP.md)).

**Status:** Evaluation completed and independently verified. The result-free
declaration was published at commit `73e48cc6f9fce7d0d0a568f5edf88ae441816423`
before drawing any evaluation seeds; the complete [compressed
record](weighted_precision_results.json.gz) contains all 1,408 acquisitions.
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

Run at the pinned result-free declaration commit
(`73e48cc6f9fce7d0d0a568f5edf88ae441816423`) under the declared environment
(`python 3.12.14`, `numpy 2.5.2`, `scipy 1.18.0`, single-threaded BLAS). Use an
empty checkpoint directory outside the repository:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/check_weighted_precision.py --declaration-only --full-anchor
python benchmarks/run_weighted_precision.py --execute --checkpoint-dir /tmp/weighted-precision --jobs 2
python benchmarks/check_weighted_precision.py --report --spot-replay --full-anchor --verify-history
```

Routine CI runs the committed history, manifest, 36-anchor noisy recomputation,
and 1,408-row ledger audit under the pinned record environment:

```bash
python benchmarks/check_weighted_precision.py --verify-history
python -m pytest -q tests/test_noisy_readout_sim.py tests/test_weighted_precision.py
```

## Results

The result-free declaration was frozen at
`73e48cc6f9fce7d0d0a568f5edf88ae441816423` and executed in a clean checkout
under `python 3.12.14`, `numpy 2.5.2`, `scipy 1.18.0`, and single-threaded BLAS
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`). The [compressed
record](weighted_precision_results.json.gz) retains all **1,408 acquisitions**
(832 Part A, 576 Part B), **3,144,593,750 simulated physical shots** (including
5,914,592 paid pilot shots across `ritz_si_*` replanning runs), **6,272 measured
solves**, and **49,696 bootstrap solves**. Zero acquisitions or bootstraps
failed (`0/1,408`); **1,404/1,408** met their active stop guards (`target_met`)
and **4/1,408** reached `schedule_exhausted` (all four in the opt-in
`beh2_8q` `si_fc_width` bootstrap-width-guard arm).

The independent verifier (`check_weighted_precision.py --report --spot-replay
--full-anchor --verify-history`) returned `record_ok`, auditing all 1,408 rows
(seeds, paid-pilot billing, schedules, per-look cumulative shots, rank-guard
cutoff arithmetic, bootstrap width decisions, and intervals), all 4 independent
occupation-space and native anchors (`beh2_8q`: `-15.5608665176 Ha`, `lih_8q`:
`-7.8630481789 Ha`, `h4_8q`: `-2.1479286781 Ha`, `lih_10q_stretched`:
`-7.7636329435 Ha`), all **36 Part B exact infinite-shot noisy anchors**, and
all **3 designated spot replays** (replica 0 of `beh2_8q` `si_fc`,
`ritz_si_fc`, and `depol2q_p010_r3` `si_fc_unconstrained`).

### Predeclared criterion verdicts

- **P1a (Weighted SI vs. unweighted LDF at matched $k$): 8/9 cells pass (`all_passed = false`).**
  Every passing cell is Holm–Bonferroni significant ($p_{\text{mean} \ge 1} = 0.0000$),
  with **0.064×–0.400× median shots** (**2.5× to 13.6× fewer mean shots**),
  32/32 stops, 0 reduced-rank stops, and non-rejected 95% unconditional
  bootstrap coverage (30/32–32/32). Crucially, on QWC (`k=1`), weighted SI uses
  *more* settings than LDF (`312` vs `281` on `beh2_8q`, `768` vs `669` on
  `lih_8q`, `794` vs `701` on `h4_8q`) yet requires **2.5×–13.6× fewer physical
  shots**, confirming that setting count is a misleading proxy for adaptive shot
  cost. The single failing cell is `lih_8q` $k=8$ (`si_fc` vs. `ldf_fc`), where
  both arms tie on the discrete look grid at `0.625M` shots in 32/32 replicas
  (ratio `1.000 [1.000, 1.000]`): at the preceding `250k`-shot look, `si_fc`
  already has lower and tighter SE (`1.996 mHa` median, range `[1.958, 2.027]`
  vs. `2.064 mHa` `[1.735, 3.809]` for `ldf_fc`, and `13.88` vs. `35.68 mHa` at
  `40k` shots), but both sit above `1.6 mHa` and step together onto `625k`.

| Molecule | $k$ | Pair | Median ratio (95% CI) | Mean ratio (95% CI) | $p_{\text{mean}\ge 1}$ | Holm | Stops | 95% boot cov. ($p_{\text{lo}}$) | Verdict |
| --- | ---: | --- | --- | --- | ---: | --- | ---: | --- | --- |
| `beh2_8q` | 1 | `si_qwc` / `ldf_qwc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 0.0000 | Yes | 32/32 | 32/32 (1.000) | PASS |
| `beh2_8q` | 2 | `si_k2` / `ldf_k2` | 0.160 [0.160, 0.160] | 0.175 [0.160, 0.198] | 0.0000 | Yes | 32/32 | 32/32 (1.000) | PASS |
| `beh2_8q` | 8 | `si_fc` / `ldf_fc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 0.0000 | Yes | 32/32 | 31/32 (0.806) | PASS |
| `lih_8q` | 1 | `si_qwc` / `ldf_qwc` | 0.064 [0.064, 0.064] | 0.074 [0.068, 0.083] | 0.0000 | Yes | 32/32 | 31/32 (0.806) | PASS |
| `lih_8q` | 2 | `si_k2` / `ldf_k2` | 0.160 [0.160, 0.160] | 0.180 [0.147, 0.217] | 0.0000 | Yes | 32/32 | 32/32 (1.000) | PASS |
| `lih_8q` | 8 | `si_fc` / `ldf_fc` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.0000 | No | 32/32 | 32/32 (1.000) | FAIL (tie) |
| `h4_8q` | 1 | `si_qwc` / `ldf_qwc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 0.0000 | Yes | 32/32 | 32/32 (1.000) | PASS |
| `h4_8q` | 2 | `si_k2` / `ldf_k2` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 0.0000 | Yes | 32/32 | 30/32 (0.480) | PASS |
| `h4_8q` | 8 | `si_fc` / `ldf_fc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 0.0000 | Yes | 32/32 | 32/32 (1.000) | PASS |

- **P1b (Paid-pilot Ritz replanning vs. bank-weighted FC and QWC; descriptive):**
  On `beh2_8q`, `ritz_si_fc` (including its `19,992`-shot pilot) cuts total
  physical shots by **2.5×** below both `si_fc` and `si_qwc` (`0.250M` vs.
  `0.625M`, ratio `0.400`, 32/32 95% coverage, `1.260 mHa` RMSE). On `h4_8q`,
  both FC arms cut shots by **2.5×** relative to `si_qwc` (`1.562M` vs.
  `3.906M`, ratio `0.400`), with `ritz_si_fc` matching `si_fc`. On `lih_8q`,
  `ritz_si_fc` edges `si_fc` slightly in mean shots (`0.602M` vs. `0.625M`,
  ratio `0.963 [0.906, 1.000]`), **but both FC arms are 2.5× more expensive than
  `si_qwc` (`0.250M`)**. The ratio of covariance-aware empirical $R$ to
  state-independent $\hat{R}$ explains why: `si_qwc` achieves median
  $R/\hat{R} = 3.799$ on `lih_8q` (favorable negative within-group covariance
  on the reference state), versus `1.397` (`si_fc`) and `1.628` (`ritz_si_fc`).
  Across all banks, $R/\hat{R}$ spans `0.171` to `3.799`, demonstrating why FC
  does not uniformly beat QWC and why state-dependent $R$ must be reported
  alongside $\hat{R}$.

| Molecule | Pair | Median ratio (95% CI) | Mean ratio (95% CI) | Treat. RMSE (mHa) | 95% boot cov. | Median $R/\hat{R}$ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `beh2_8q` | `ritz_si_fc` / `si_fc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 1.260 | 32/32 | 0.348 |
| `beh2_8q` | `ritz_si_fc` / `si_qwc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 1.260 | 32/32 | 0.348 |
| `beh2_8q` | `si_fc` / `si_qwc` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.319 | 31/32 | 0.209 |
| `lih_8q` | `ritz_si_fc` / `si_fc` | 1.000 [1.000, 1.000] | 0.963 [0.906, 1.000] | 1.189 | 32/32 | 1.628 |
| `lih_8q` | `ritz_si_fc` / `si_qwc` | 2.500 [2.500, 2.500] | 2.406 [2.266, 2.500] | 1.189 | 32/32 | 1.628 |
| `lih_8q` | `si_fc` / `si_qwc` | 2.500 [2.500, 2.500] | 2.500 [2.500, 2.500] | 1.113 | 32/32 | 1.397 |
| `h4_8q` | `ritz_si_fc` / `si_fc` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.133 | 32/32 | 0.547 |
| `h4_8q` | `ritz_si_fc` / `si_qwc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 1.133 | 32/32 | 0.547 |
| `h4_8q` | `si_fc` / `si_qwc` | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] | 1.088 | 32/32 | 0.512 |

- **Opt-in bootstrap half-width guard (`si_fc_width` vs. `si_fc`):** Enforcing
  a `1.6 mHa` schedule-adjusted bootstrap half-width costs **5.43×–5.78× mean
  shots** (`6.25×` median) and **2.78×–2.88× bootstrap solves**, cutting RMSE
  from `1.09–1.32 mHa` to `0.45–0.54 mHa` and reducing false stops from `5–6/32`
  to `0–1/32` (`28/28` on `beh2_8q`, `31/32` on `lih_8q`, `32/32` on `h4_8q`).
  On `beh2_8q`, `4/32` attempts exhausted the `3.906M`-shot schedule despite
  small terminal errors (`0.023–0.694 mHa`, SE `0.535–0.537 mHa`) because the
  6-look Bonferroni-scaled 32-resample bootstrap half-width (`1.63–2.03 mHa`)
  slightly exceeded `1.6 mHa`; stop-conditional 95% bootstrap coverage was
  `28/28 (100%)` while strict unconditional coverage was `28/32 (87.5%)`.

- **P2 (Stretched 10-qubit rank-stress transfer on `lih_10q_stretched`): ALL PASSED (`all_passed = true`).**
  Both `si_k2` (`490` settings, $\text{CZ}_{\max} = 5$, depth 1) and
  `ritz_si_k2` (`496–535` settings, paid pilot `249,900` shots) achieved
  **16/16 true precision stops**, **0/16 false precision stops**, and **0
  reduced-rank stops** on the 10,341-word bank straddling $\tau_S = 0.104$.
  Because rank-margin confirmation bounds the median look at `3.125M` shots for
  all four arms, median shot ratios are `1.000`, while `si_k2` trims tail looks
  to cut mean shots by **30.0%** vs. `ldf_qwc` (`3.418M` vs. `4.883M`, mean
  ratio `0.700 [0.542, 0.927]`, $p = 0.0155$) and **14.6%** vs. `si_qwc`
  (`4.004M`, ratio `0.854 [0.660, 1.158]`). `ritz_si_k2` achieves `0.820×`
  mean shots vs. `ldf_qwc` (`4.004M`, CI `[0.593, 1.146]`), the lowest RMSE
  (`0.467 mHa` vs. `0.809 mHa` for `ldf_qwc`), and `16/16` 95% bootstrap
  coverage at the narrowest median width (`2.68 mHa`).

- **P3 (GALIC `NoiseGatePolicy` bias protection under gate-level noise): ALL PASSED (6/6 cells, `all_passed = true`).**
  Across all three 8-qubit molecules under both `depol2q_p002_r3` and
  `depol2q_p010_r3`, `si_fc_galic_tight` ($\text{CZ}_{\max} = 1$) strictly
  reduces exact infinite-shot bias (**5.3×–13.4× smaller**), finite-shot RMSE
  (**2.4×–7.1× smaller**), and false precision stops relative to unconstrained
  `si_fc_unconstrained` ($\text{CZ}_{\max} = 7$–$16$). Under `depol2q_p002_r3`,
  tight gating drops false stops from `13–14/16` down to `1–4/16` and restores
  95% unconditional bootstrap coverage from `8–11/16` to **`16/16` on all three
  molecules**. In contrast, `si_fc_galic_moderate` ($\text{CZ}_{\max} = 3$)
  retains `+2.50` to `+3.27 mHa` bias at $p_{2\mathrm{q}} = 0.002$ (`12–15/16`
  false stops), and at $p_{2\mathrm{q}} = 0.010$ even `galic_tight` leaves
  `1.17–2.97 mHa` bias (`7–14/16` false stops) while zero-2Q `si_qwc` remains
  2Q-unbiased (`0.000 mHa`). Under `sc_card_full` (adding unmitigated readout
  flip $p_{\text{ro}} = 0.02$), readout bias dominates **every** arm including
  `si_qwc` (`+61.78` to `+134.88 mHa` exact bias, `16/16` false stops, `0/16`
  coverage across all 12 cells), confirming that 2Q noise gating and readout error
  mitigation address orthogonal error channels.

### Part A complete summary (ideal shots)

All rows below report unconditional coverage (`covered / attempts`, counting
exhaustions as uncovered) and median interval width (in parentheses, mHa) among
`target_met` stops. Stop-conditional coverage equals unconditional coverage in
every cell except `beh2_8q` `si_fc_width`, where stop-conditional coverage is
`28/28` (95% boot/delta) and `26/28` (68% boot).

| Case | Arm | Stops (exh.) | True / False (1.6 mHa) | RR | Med. / Mean shots (M) | Dev. (s) | RMSE (mHa) | Uncond. 95% boot / delta (w, mHa) | Uncond. 68% boot (w, mHa) | Med. $R/\hat{R}$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `beh2_8q` | `ldf_qwc` | 32/32 | 25 / 7 | 0 | 1.562 / 1.562 | 15.75 | 1.484 | 30/32 (7.11) / 30/32 (6.96) | 29/32 (5.21) | 0.272 |
| `beh2_8q` | `si_qwc` | 32/32 | 30 / 2 | 0 | 0.625 / 0.625 | 6.30 | 0.899 | 32/32 (4.72) / 32/32 (5.43) | 30/32 (3.46) | 0.531 |
| `beh2_8q` | `ldf_k2` | 32/32 | 26 / 6 | 0 | 1.562 / 1.562 | 16.25 | 1.239 | 31/32 (6.26) / 32/32 (6.46) | 31/32 (4.59) | 0.217 |
| `beh2_8q` | `si_k2` | 32/32 | 24 / 8 | 0 | 0.250 / 0.273 | 2.58 | 1.362 | 32/32 (7.54) / 32/32 (7.91) | 32/32 (5.53) | 0.386 |
| `beh2_8q` | `ldf_fc` | 32/32 | 28 / 4 | 0 | 1.562 / 1.562 | 17.26 | 1.075 | 32/32 (5.11) / 32/32 (5.49) | 30/32 (3.74) | 0.171 |
| `beh2_8q` | `si_fc` | 32/32 | 26 / 6 | 0 | 0.625 / 0.625 | 6.89 | 1.319 | 31/32 (6.92) / 31/32 (7.15) | 30/32 (5.06) | 0.209 |
| `beh2_8q` | `ritz_si_fc` | 32/32 | 25 / 7 | 0 | 0.250 / 0.250 | 2.70 | 1.260 | 32/32 (5.87) / 32/32 (6.48) | 28/32 (4.30) | 0.348 |
| `beh2_8q` | `si_fc_width` | 28/32 (4) | 28 / 0 | 0 | 3.906 / 3.613 | 43.05 | 0.506 | 28/32 (2.63) / 28/32 (2.83) | 26/32 (1.92) | 0.208 |
| `lih_8q` | `ldf_qwc` | 32/32 | 26 / 6 | 0 | 3.906 / 3.394 | 39.43 | 1.352 | 30/32 (5.53) / 31/32 (5.77) | 29/32 (4.05) | 1.339 |
| `lih_8q` | `si_qwc` | 32/32 | 25 / 7 | 0 | 0.250 / 0.250 | 2.52 | 1.511 | 31/32 (7.79) / 31/32 (8.13) | 30/32 (5.70) | 3.799 |
| `lih_8q` | `ldf_k2` | 32/32 | 23 / 9 | 0 | 1.562 / 1.782 | 16.18 | 1.571 | 31/32 (7.76) / 32/32 (7.91) | 28/32 (5.69) | 1.367 |
| `lih_8q` | `si_k2` | 32/32 | 25 / 7 | 0 | 0.250 / 0.320 | 2.58 | 1.450 | 32/32 (7.38) / 32/32 (8.17) | 30/32 (5.41) | 2.799 |
| `lih_8q` | `ldf_fc` | 32/32 | 28 / 4 | 0 | 0.625 / 0.625 | 6.87 | 1.167 | 31/32 (5.79) / 32/32 (6.36) | 30/32 (4.24) | 1.501 |
| `lih_8q` | `si_fc` | 32/32 | 26 / 6 | 0 | 0.625 / 0.625 | 6.87 | 1.113 | 32/32 (5.89) / 32/32 (6.42) | 30/32 (4.32) | 1.397 |
| `lih_8q` | `ritz_si_fc` | 32/32 | 27 / 5 | 0 | 0.625 / 0.602 | 6.78 | 1.189 | 32/32 (5.90) / 32/32 (5.76) | 29/32 (4.32) | 1.628 |
| `lih_8q` | `si_fc_width` | 32/32 | 31 / 1 | 0 | 3.906 / 3.394 | 42.92 | 0.535 | 31/32 (2.49) / 32/32 (2.53) | 30/32 (1.83) | 1.396 |
| `h4_8q` | `ldf_qwc` | 32/32 | 25 / 7 | 0 | 9.766 / 9.766 | 98.58 | 1.341 | 30/32 (5.97) / 31/32 (6.21) | 29/32 (4.37) | 0.844 |
| `h4_8q` | `si_qwc` | 32/32 | 21 / 11 | 0 | 3.906 / 3.906 | 39.40 | 1.482 | 32/32 (7.06) / 32/32 (7.90) | 28/32 (5.17) | 0.480 |
| `h4_8q` | `ldf_k2` | 32/32 | 29 / 3 | 0 | 9.766 / 9.766 | 101.11 | 1.044 | 31/32 (5.51) / 32/32 (5.73) | 30/32 (4.03) | 0.669 |
| `h4_8q` | `si_k2` | 32/32 | 27 / 5 | 0 | 3.906 / 3.906 | 40.29 | 1.443 | 30/32 (6.88) / 31/32 (7.62) | 29/32 (5.04) | 0.394 |
| `h4_8q` | `ldf_fc` | 32/32 | 30 / 2 | 0 | 3.906 / 3.906 | 44.18 | 0.992 | 32/32 (5.50) / 32/32 (5.83) | 31/32 (4.03) | 0.468 |
| `h4_8q` | `si_fc` | 32/32 | 27 / 5 | 0 | 1.562 / 1.562 | 17.27 | 1.088 | 32/32 (6.42) / 32/32 (6.62) | 32/32 (4.70) | 0.512 |
| `h4_8q` | `ritz_si_fc` | 32/32 | 26 / 6 | 0 | 1.562 / 1.562 | 17.20 | 1.133 | 32/32 (6.13) / 32/32 (6.18) | 31/32 (4.49) | 0.547 |
| `h4_8q` | `si_fc_width` | 32/32 | 32 / 0 | 0 | 9.766 / 9.033 | 107.82 | 0.446 | 31/32 (2.62) / 32/32 (2.61) | 30/32 (1.92) | 0.511 |
| `lih_10q_stretched` | `ldf_qwc` | 16/16 | 15 / 1 | 0 | 3.125 / 4.883 | 37.80 | 0.809 | 16/16 (4.29) / 16/16 (5.43) | 16/16 (3.08) | 0.792 |
| `lih_10q_stretched` | `si_qwc` | 16/16 | 16 / 0 | 0 | 3.125 / 4.004 | 37.78 | 0.620 | 15/16 (3.17) / 16/16 (3.70) | 14/16 (2.28) | 0.391 |
| `lih_10q_stretched` | `si_k2` | 16/16 | 16 / 0 | 0 | 3.125 / 3.418 | 38.72 | 0.705 | 15/16 (3.21) / 16/16 (3.93) | 14/16 (2.31) | 0.325 |
| `lih_10q_stretched` | `ritz_si_k2` | 16/16 | 16 / 0 | 0 | 3.125 / 4.004 | 38.63 | 0.467 | 16/16 (2.68) / 16/16 (3.11) | 15/16 (1.93) | 0.330 |

### Part B complete summary (gate-level noisy readout)

All 576 Part B acquisitions reached `target_met` (0 exhaustions, 0 failures, 0
reduced-rank stops), illustrating why an unbiased-noise-assuming SE stop can
produce high false-stop rates when readout circuits incur systematic bias.

| Case | Regime | Arm | Settings ($\text{CZ}_{\max}$) | True / False | Exact $E_\infty$ bias (mHa) | Emp. bias ± std (mHa) | RMSE (mHa) | Bias² share | Med. shots (M) | Uncond. 95% boot |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `beh2_8q` | `depol2q_p002_r3` | `si_qwc` | 312 (0) | 16 / 0 | +0.000 | -0.344 ± 0.804 | 0.874 | 15.5% | 0.625 | 16/16 |
| `beh2_8q` | `depol2q_p002_r3` | `si_fc_unconstrained` | 14 (7) | 3 / 13 | +2.841 | +2.828 ± 1.307 | 3.116 | 82.4% | 0.625 | 11/16 |
| `beh2_8q` | `depol2q_p002_r3` | `si_fc_galic_tight` | 307 (1) | 14 / 2 | -0.520 | -0.608 ± 0.856 | 1.050 | 33.5% | 0.625 | 16/16 |
| `beh2_8q` | `depol2q_p002_r3` | `si_fc_galic_moderate` | 150 (3) | 4 / 12 | +2.498 | +2.673 ± 1.384 | 3.010 | 78.9% | 0.625 | 12/16 |
| `beh2_8q` | `depol2q_p010_r3` | `si_qwc` | 312 (0) | 10 / 6 | +0.000 | -0.412 ± 1.484 | 1.540 | 7.2% | 0.625 | 13/16 |
| `beh2_8q` | `depol2q_p010_r3` | `si_fc_unconstrained` | 14 (7) | 0 / 16 | +13.820 | +13.282 ± 0.966 | 13.317 | 99.5% | 0.625 | 0/16 |
| `beh2_8q` | `depol2q_p010_r3` | `si_fc_galic_tight` | 307 (1) | 2 / 14 | -2.578 | -2.946 ± 0.979 | 3.104 | 90.1% | 0.625 | 7/16 |
| `beh2_8q` | `depol2q_p010_r3` | `si_fc_galic_moderate` | 150 (3) | 0 / 16 | +12.147 | +12.302 ± 1.223 | 12.362 | 99.0% | 0.625 | 0/16 |
| `beh2_8q` | `sc_card_full` | `si_qwc` | 312 (0) | 0 / 16 | +61.783 | +61.843 ± 1.215 | 61.855 | 100.0% | 0.625 | 0/16 |
| `beh2_8q` | `sc_card_full` | `si_fc_unconstrained` | 14 (7) | 0 / 16 | +67.753 | +67.621 ± 1.700 | 67.643 | 99.9% | 0.625 | 0/16 |
| `beh2_8q` | `sc_card_full` | `si_fc_galic_tight` | 307 (1) | 0 / 16 | +62.414 | +62.592 ± 0.949 | 62.599 | 100.0% | 0.625 | 0/16 |
| `beh2_8q` | `sc_card_full` | `si_fc_galic_moderate` | 150 (3) | 0 / 16 | +67.857 | +68.097 ± 1.307 | 68.110 | 100.0% | 0.625 | 0/16 |
| `lih_8q` | `depol2q_p002_r3` | `si_qwc` | 768 (0) | 11 / 5 | -0.000 | +0.322 ± 1.510 | 1.544 | 4.4% | 0.250 | 14/16 |
| `lih_8q` | `depol2q_p002_r3` | `si_fc_unconstrained` | 59 (12) | 3 / 13 | +3.250 | +2.792 ± 1.208 | 3.042 | 84.2% | 0.625 | 11/16 |
| `lih_8q` | `depol2q_p002_r3` | `si_fc_galic_tight` | 720 (1) | 15 / 1 | -0.598 | -0.464 ± 0.950 | 1.057 | 19.2% | 0.625 | 16/16 |
| `lih_8q` | `depol2q_p002_r3` | `si_fc_galic_moderate` | 282 (3) | 1 / 15 | +3.274 | +3.021 ± 0.866 | 3.143 | 92.4% | 0.625 | 11/16 |
| `lih_8q` | `depol2q_p010_r3` | `si_qwc` | 768 (0) | 11 / 5 | -0.000 | +0.230 ± 1.684 | 1.699 | 1.8% | 0.250 | 16/16 |
| `lih_8q` | `depol2q_p010_r3` | `si_fc_unconstrained` | 59 (12) | 0 / 16 | +15.555 | +15.798 ± 1.558 | 15.874 | 99.0% | 0.625 | 0/16 |
| `lih_8q` | `depol2q_p010_r3` | `si_fc_galic_tight` | 720 (1) | 4 / 12 | -2.967 | -2.566 ± 1.274 | 2.865 | 80.2% | 0.625 | 8/16 |
| `lih_8q` | `depol2q_p010_r3` | `si_fc_galic_moderate` | 282 (3) | 0 / 16 | +15.663 | +15.215 ± 1.007 | 15.248 | 99.6% | 0.625 | 0/16 |
| `lih_8q` | `sc_card_full` | `si_qwc` | 768 (0) | 0 / 16 | +124.854 | +124.412 ± 1.136 | 124.417 | 100.0% | 0.625 | 0/16 |
| `lih_8q` | `sc_card_full` | `si_fc_unconstrained` | 59 (12) | 0 / 16 | +134.882 | +134.365 ± 1.180 | 134.371 | 100.0% | 1.562 | 0/16 |
| `lih_8q` | `sc_card_full` | `si_fc_galic_tight` | 720 (1) | 0 / 16 | +125.539 | +125.830 ± 1.725 | 125.841 | 100.0% | 0.625 | 0/16 |
| `lih_8q` | `sc_card_full` | `si_fc_galic_moderate` | 282 (3) | 0 / 16 | +134.856 | +135.357 ± 1.578 | 135.366 | 100.0% | 1.562 | 0/16 |
| `h4_8q` | `depol2q_p002_r3` | `si_qwc` | 794 (0) | 8 / 8 | +0.000 | +0.207 ± 1.541 | 1.555 | 1.8% | 3.906 | 16/16 |
| `h4_8q` | `depol2q_p002_r3` | `si_fc_unconstrained` | 61 (16) | 2 / 14 | +3.187 | +3.093 ± 1.123 | 3.290 | 88.4% | 1.562 | 8/16 |
| `h4_8q` | `depol2q_p002_r3` | `si_fc_galic_tight` | 745 (1) | 12 / 4 | +0.237 | +0.702 ± 1.167 | 1.362 | 26.6% | 3.906 | 16/16 |
| `h4_8q` | `depol2q_p002_r3` | `si_fc_galic_moderate` | 297 (3) | 2 / 14 | +2.827 | +2.928 ± 1.164 | 3.151 | 86.4% | 1.562 | 11/16 |
| `h4_8q` | `depol2q_p010_r3` | `si_qwc` | 794 (0) | 9 / 7 | +0.000 | +0.324 ± 1.775 | 1.804 | 3.2% | 3.906 | 14/16 |
| `h4_8q` | `depol2q_p010_r3` | `si_fc_unconstrained` | 61 (16) | 0 / 16 | +14.853 | +14.587 ± 0.963 | 14.619 | 99.6% | 1.562 | 0/16 |
| `h4_8q` | `depol2q_p010_r3` | `si_fc_galic_tight` | 745 (1) | 9 / 7 | +1.172 | +1.548 ± 1.348 | 2.052 | 56.9% | 3.906 | 13/16 |
| `h4_8q` | `depol2q_p010_r3` | `si_fc_galic_moderate` | 297 (3) | 0 / 16 | +13.285 | +13.055 ± 1.790 | 13.177 | 98.2% | 1.562 | 0/16 |
| `h4_8q` | `sc_card_full` | `si_qwc` | 794 (0) | 0 / 16 | +109.736 | +110.032 ± 1.321 | 110.040 | 100.0% | 3.906 | 0/16 |
| `h4_8q` | `sc_card_full` | `si_fc_unconstrained` | 61 (16) | 0 / 16 | +114.980 | +114.376 ± 1.301 | 114.384 | 100.0% | 1.562 | 0/16 |
| `h4_8q` | `sc_card_full` | `si_fc_galic_tight` | 745 (1) | 0 / 16 | +109.806 | +109.439 ± 1.626 | 109.451 | 100.0% | 3.906 | 0/16 |
| `h4_8q` | `sc_card_full` | `si_fc_galic_moderate` | 297 (3) | 0 / 16 | +113.799 | +114.035 ± 0.794 | 114.037 | 100.0% | 3.906 | 0/16 |

## Scope and non-claims

- Ideal statevector and gate-level depolarizing/readout simulation only; not a
  calibrated hardware run or quantum advantage claim.
- Targets are the declared fixed-bank exact projected energies, not full-CI
  chemical accuracy.
- Uses the raw Ritz `sampled_precision` estimator (no exact-oracle-certified
  control variates). Rank-margin and terminal bootstrap intervals remain
  heuristic, not finite-sample or anytime-valid certificates.
- P1a failed its all-9-cells conjunction due to the discrete look-grid tie on
  `lih_8q` $k=8$, and FC did not uniformly beat QWC (`lih_8q` `si_qwc` beat
  both FC arms by `2.5×` due to favorable state-dependent covariance).
- GALIC 2Q noise gating does not protect against unmitigated readout bit-flip
  noise (`sc_card_full`), which requires separate readout error mitigation.
