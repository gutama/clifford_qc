# Sparse packet setup and early screening

Exploratory exact classical comparisons. Times are fresh-process observations; units are input-specific.
Early marks precede candidate admissibility. No solver default is changed.

Publication recovery: 33 complete observations retained; 6 secondary-input observations were unavailable and count as failures. All H2O and all seven 2x4 Hubbard repetitions survived. No cells were selectively rerun.

## h2o (Ha)

225 determinants; 32-column cap; target 0.0016 Ha.

| Arm | Completed | Target | Final error | H calls | Materialized | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 3/3 | 3/3 | 1.35026937e-07 | 514 | 1421 | 0.725 | [0.712, 0.754] | 1.971 | 180.902 |
| sparse | 3/3 | 3/3 | 1.35026937e-07 | 514 | 1421 | 0.816 | [0.816, 0.824] | 2.142 | 180.926 |
| early | 3/3 | 3/3 | 1.35026937e-07 | 200 | 349 | 0.735 | [0.725, 0.744] | 1.943 | 180.906 |

Exhaustive sparse/dense: 0.888× speed ratio; 1.000× RSS ratio. Timing/RSS promotion clause: FAIL.

Early audits across repeats: 15; missed winners: 0; largest audited score loss: 0 Ha.

## hubbard_2x4_u4 (t)

4900 determinants; 64-column cap; target 0.0016 t.

| Arm | Completed | Target | Final error | H calls | Materialized | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 0/7 | 1.93591273 | 10841 | 34892 | 27.353 | [27.025, 27.937] | 28.769 | 686.059 |
| sparse | 7/7 | 0/7 | 1.93591273 | 10841 | 34892 | 25.815 | [25.058, 26.510] | 27.097 | 180.902 |
| early | 7/7 | 0/7 | 1.94577515 | 4469 | 9123 | 7.767 | [7.677, 8.102] | 9.133 | 180.902 |

Exhaustive sparse/dense: 1.060× speed ratio; 0.264× RSS ratio. Timing/RSS promotion clause: FAIL.

Early audits across repeats: 84; missed winners: 0; largest audited score loss: 0 t.

## hubbard_2x3_u8 (t)

400 determinants; 32-column cap; target 0.0016 t.

| Arm | Completed | Target | Final error | H calls | Materialized | Setup + solver (s) | Time IQR (s) | Process (s) | RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1/3 | 0/3 | 0.341441046 | 1312 | 2030 | 0.192 | — | 1.737 | 180.902 |
| sparse | 1/3 | 0/3 | 0.341441046 | 1312 | 2030 | 0.266 | — | 1.703 | 180.902 |
| early | 1/3 | 0/3 | 0.385302798 | 545 | 656 | 0.123 | — | 1.382 | 180.902 |

Exhaustive sparse/dense: 0.721× speed ratio; 1.000× RSS ratio. Timing/RSS promotion clause: FAIL.

Early audits across repeats: 5; missed winners: 2; largest audited score loss: 0.049864755 t.

Materialized counts include rejected directions; H calls and exact scores count admissible candidates only.

Sparse setup avoids full-sector dense H/S² and a dense global CSF frame; it retains bounded dense spin blocks and explicit accepted sector vectors. Payloads are not peak RSS.
Fixed-budget errors do not establish matched-accuracy speedups. Repeats measure timing variability, not independent scientific generalization. No optimized-SCI, hardware or quantum-advantage claim follows.

Declaration: `93813b32a0cc1b8c411ddd4bdaba3a4cca4afcfd`. Execution: `ac624c27346dc6a45337f994c05fdfcede1a6ea0`.
