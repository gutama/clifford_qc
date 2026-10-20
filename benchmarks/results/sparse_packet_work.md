# Sparse packet setup and early screening

Exploratory exact classical comparisons. Times are fresh-process observations; units are input-specific.
Early marks precede candidate admissibility. No solver default is changed.

Publication recovery: 33 complete observations retained; 6 observations have no surviving record and count as failures. These are scheduled-cell denominators, not proof that every cell ran. No cells were selectively rerun.

Coverage for h2o: 9/9 scheduled observations retained; 0 unavailable or failed.
Coverage for hubbard_2x4_u4: 21/21 scheduled observations retained; 0 unavailable or failed.
Coverage for hubbard_2x3_u8: 3/9 scheduled observations retained; 6 unavailable or failed.

RSS is the whole-process high-water mark, including provenance preflight, which parses committed records. The approximately 181 MiB preflight floor masks smaller solver peaks and small-fixture RSS regressions; solver-only peaks were not measured, and the floor can grow as records accumulate.

## h2o (Ha)

225 determinants; 32-column cap; target 0.0016 Ha.

| Arm | Completed | Target | Final error | H calls | Materialized | Setup (s) | Solver (s) | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 3/3 | 3/3 | 1.35026937e-07 | 514 | 1421 | 0.656 | 0.069 | 0.725 | [0.712, 0.754] | 1.971 | 180.902 |
| sparse | 3/3 | 3/3 | 1.35026937e-07 | 514 | 1421 | 0.681 | 0.146 | 0.816 | [0.816, 0.824] | 2.142 | 180.926 |
| early | 3/3 | 3/3 | 1.35026937e-07 | 200 | 349 | 0.686 | 0.049 | 0.735 | [0.725, 0.744] | 1.943 | 180.906 |

Exhaustive sparse/dense: 0.888× speed ratio; 1.000× RSS ratio. Timing/RSS promotion clause: FAIL.

Early audits over 1 unique decision trajectory (3 timing observations): 5; missed winners: 0; largest audited score loss: 0 Ha.

Early versus exhaustive final error: 1.35026937e-07 versus 1.35026937e-07 Ha (+0.00%).

## hubbard_2x4_u4 (t)

4900 determinants; 64-column cap; target 0.0016 t.

| Arm | Completed | Target | Final error | H calls | Materialized | Setup (s) | Solver (s) | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 0/7 | 1.93591273 | 10841 | 34892 | 0.507 | 26.846 | 27.353 | [27.025, 27.937] | 28.769 | 686.059 |
| sparse | 7/7 | 0/7 | 1.93591273 | 10841 | 34892 | 0.305 | 25.515 | 25.815 | [25.058, 26.510] | 27.097 | 180.902 |
| early | 7/7 | 0/7 | 1.94577515 | 4469 | 9123 | 0.279 | 7.475 | 7.767 | [7.677, 8.102] | 9.133 | 180.902 |

Exhaustive sparse/dense: 1.060× speed ratio; 0.264× RSS ratio. Timing/RSS promotion clause: FAIL.

Early audits over 1 unique decision trajectory (7 timing observations): 12; missed winners: 0; largest audited score loss: 0 t.

Early versus exhaustive final error: 1.94577515 versus 1.93591273 t (+0.51%).

## hubbard_2x3_u8 (t)

400 determinants; 32-column cap; target 0.0016 t.

| Arm | Completed | Target | Final error | H calls | Materialized | Setup (s) | Solver (s) | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1/3 | 0/3 | 0.341441046 | 1312 | 2030 | 0.033 | 0.159 | 0.192 | — | 1.737 | 180.902 |
| sparse | 1/3 | 0/3 | 0.341441046 | 1312 | 2030 | 0.038 | 0.228 | 0.266 | — | 1.703 | 180.902 |
| early | 1/3 | 0/3 | 0.385302798 | 545 | 656 | 0.034 | 0.088 | 0.123 | — | 1.382 | 180.902 |

Exhaustive sparse/dense: 0.721× speed ratio; 1.000× RSS ratio. Timing/RSS promotion clause: FAIL.

Early audits over 1 unique decision trajectory (1 timing observation): 5; missed winners: 2; largest audited score loss: 0.049864755 t.

Early versus exhaustive final error: 0.385302798 versus 0.341441046 t (+12.85%).
One surviving observation per arm establishes the observed deterministic accuracy outcome; missing repeats reduce timing evidence, not the visibility of this error comparison.

## Historical shortlist comparison

2x4 early versus the PR #145 shortlist: equivalent ordered selections, energies/residuals and H calls within 1e-9: PASS; 62 retained prefixes (61 growth selections); maximum energy difference 9.77e-15 t, residual difference 5.33e-15 t.
Historical shortlist setup + solver: 26.208 s; sparse early: 7.767 s. The historical path materializes every group before shortlisting (34892 candidates for this common growth schedule); early materializes 9123. Moving marking before admissibility does not change the observed trajectory here. Avoided materialization supports the mechanism; these are separate campaigns, not a new controlled timing ablation.

Materialized counts include rejected directions; H calls and exact scores count admissible candidates only.

Setup and solver columns are separate medians; setup + solver is the median of each observation's paired sum.

Sparse setup avoids full-sector dense H/S² and a dense global CSF frame; it retains bounded dense spin blocks and explicit accepted sector vectors. Payloads are not peak RSS.
Fixed-budget errors do not establish matched-accuracy speedups. Repeats measure timing variability, not independent scientific generalization. No optimized-SCI, hardware or quantum-advantage claim follows.

Declaration: `93813b32a0cc1b8c411ddd4bdaba3a4cca4afcfd`. Execution: `ac624c27346dc6a45337f994c05fdfcede1a6ea0`.
