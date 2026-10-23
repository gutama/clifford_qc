# Column-budget convergence

Exact classical sector-vector comparison of residual packets, greedy spin-adapted selected CI and a singlet-projected Davidson reference, each adding one column per step from the same three-column CAS(2e,2o) singlet span.
52/52 scheduled cells retained; 20 content-addressed scientific payloads. Errors are above the lowest singlet-frame eigenvalue, in t for lattices and Ha for molecules.

Times are medians of setup plus solver across repetitions and are descriptive. Worker RSS is the OS high-water mark. Span support counts determinants with any retained amplitude above 1e-12. On the molecular inputs, packet and Davidson directions carry rounding-level amplitudes near that threshold: their counts reproduce exactly but can differ by a few percent from an independent computation of the same span, which brackets each between its own counts at 1e-11 and 1e-13. Selected-CI columns are exact CSFs and carry none.

## hubbard_2x4_u4 (primary, t)

4900 determinants, 1764 singlet columns, budget 512. Singlet reference -5.012503153; sector ground state is a singlet.

| Arm | Completed | Stop | Columns | Final error | Ground overlap | H actions | Support | Time (s) | Worker RSS (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| packets_exhaustive | 2/2 | column_ceiling | 512 | 0.0762604 | 0.9901 | 215421 | 4108 | 1326.24 | 346.7 |
| packets_early | 2/2 | column_ceiling | 512 | 0.0766238 | 0.9909 | 93131 | 4096 | 609.31 | 350.9 |
| selected_ci | 2/2 | column_ceiling | 512 | 0.247694 | 0.9667 | 1764 | 1952 | 134.28 | 418.3 |
| davidson | 2/2 | no_admissible_direction | 48 | 0.994928 | 0.0000 | 48 | 4900 | 0.68 | 85.2 |

| Arm | Error at 64 | Error at 128 | Error at 256 | Error at 512 |
|---|---:|---:|---:|---:|
| packets_exhaustive | 1.93591 | 1.19452 | 0.362465 | 0.0762604 |
| packets_early | 1.94578 | 1.20449 | 0.364538 | 0.0766238 |
| selected_ci | 2.07767 | 1.27438 | 0.606117 | 0.247694 |
| davidson | 0.994928 | 0.994928 | 0.994928 | 0.994928 |

| Arm | ≤1.0: K / W / s | ≤0.1: K / W / s | ≤0.01: K / W / s | ≤0.0016: K / W / s |
|---|---|---|---|---|
| packets_exhaustive | 145 / 38345 / 126.79 | 473 / 194347 / 1133.02 | not reached | not reached |
| packets_early | 146 / 15809 / 44.02 | 473 / 84601 / 514.06 | not reached | not reached |
| selected_ci | 174 / 1764 / 9.34 | not reached | not reached | not reached |
| davidson | 21 / 21 / 0.37 | not reached | not reached | not reached |

Readout 1, column compactness (packets_exhaustive vs selected_ci): **mixed**.
Readout 2, work: not applicable; columns favour neither arm.
Readout 3, early screening: **preserves accuracy**.
Readout 4, Davidson: does not reach every threshold in fewer columns than both restricted arms. It stopped by no_admissible_direction at residual 9.21e-09, 0.994928 above the reference with ground overlap 1.5e-12, so its span excludes the ground state. The declared confinement rule names residual-tolerance stops only, so the verdict above follows the declared fallback.
Span support at 48 columns, Davidson's last prefix: packets_exhaustive 676, packets_early 654, selected_ci 396, davidson 4900 determinants.

## hubbard_1x8_u4 (held_out_lattice, t)

4900 determinants, 1764 singlet columns, budget 512. Singlet reference -4.235806999; sector ground state is a singlet.

| Arm | Completed | Stop | Columns | Final error | Ground overlap | H actions | Support | Time (s) | Worker RSS (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| packets_exhaustive | 2/2 | column_ceiling | 512 | 0.007621 | 0.9988 | 161778 | 3552 | 1314.11 | 387.0 |
| packets_early | 2/2 | column_ceiling | 512 | 0.00564535 | 0.9993 | 64354 | 3626 | 517.01 | 324.5 |
| selected_ci | 2/2 | column_ceiling | 512 | 0.0496348 | 0.9940 | 1764 | 1792 | 134.39 | 411.5 |
| davidson | 2/2 | no_admissible_direction | 46 | -6.21725e-15 | 1.0000 | 46 | 4900 | 0.70 | 83.4 |

| Arm | Error at 64 | Error at 128 | Error at 256 | Error at 512 |
|---|---:|---:|---:|---:|
| packets_exhaustive | 3.45911 | 0.421937 | 0.0924354 | 0.007621 |
| packets_early | 1.99267 | 0.406074 | 0.0853844 | 0.00564535 |
| selected_ci | 1.61885 | 1.16614 | 0.313775 | 0.0496348 |
| davidson | -6.21725e-15 | -6.21725e-15 | -6.21725e-15 | -6.21725e-15 |

| Arm | ≤1.0: K / W / s | ≤0.1: K / W / s | ≤0.01: K / W / s | ≤0.0016: K / W / s |
|---|---|---|---|---|
| packets_exhaustive | 89 / 9623 / 49.68 | 251 / 57451 / 338.33 | 484 / 149497 / 1176.82 | not reached |
| packets_early | 83 / 3459 / 12.67 | 242 / 21658 / 104.64 | 454 / 54394 / 393.64 | not reached |
| selected_ci | 166 / 1764 / 8.21 | 404 / 1764 / 66.18 | not reached | not reached |
| davidson | 18 / 18 / 0.40 | 20 / 20 / 0.42 | 22 / 22 / 0.43 | 23 / 23 / 0.44 |

Readout 1, column compactness (packets_exhaustive vs selected_ci): **packets more compact**.
Readout 2, work: compression at higher classical work (≥1.25× H actions at ≤1.0, ≤0.1).
Readout 3, early screening: **outside the declared band** (frontier error outside 10% of exhaustive at 64, 512). Early minus exhaustive error (negative: early lower) -1.466 at 64, -0.01586 at 128, -0.007051 at 256, -0.001976 at 512.
Readout 4, Davidson: reaches every threshold in fewer columns than both restricted arms, so the packets-versus-selected-CI comparison concerns bounded-support compactness only.
Span support at 46 columns, Davidson's last prefix: packets_exhaustive 308, packets_early 296, selected_ci 482, davidson 4900 determinants.

## beh2_stretched_full (held_out_molecule, Ha)

1225 determinants, 490 singlet columns, budget 490. Singlet reference -15.34224375; sector ground state is a singlet.

| Arm | Completed | Stop | Columns | Final error | Ground overlap | H actions | Support | Time (s) | Worker RSS (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| packets_exhaustive | 3/3 | residual_tolerance | 94 | 4.9738e-14 | 1.0000 | 3409 | 175 | 6.41 | 76.9 |
| packets_early | 3/3 | residual_tolerance | 94 | 4.9738e-14 | 1.0000 | 1260 | 175 | 2.98 | 76.4 |
| selected_ci | 3/3 | residual_tolerance | 94 | 4.26326e-14 | 1.0000 | 490 | 171 | 1.69 | 84.3 |
| davidson | 3/3 | no_admissible_direction | 18 | 4.79616e-14 | 1.0000 | 18 | 325 | 1.27 | 64.1 |

| Arm | Error at 64 | Error at 128 | Error at 256 | Error at 490 |
|---|---:|---:|---:|---:|
| packets_exhaustive | 1.92396e-07 | 4.9738e-14 | 4.9738e-14 | 4.9738e-14 |
| packets_early | 1.92396e-07 | 4.9738e-14 | 4.9738e-14 | 4.9738e-14 |
| selected_ci | 4.60102e-06 | 4.26326e-14 | 4.26326e-14 | 4.26326e-14 |
| davidson | 4.79616e-14 | 4.79616e-14 | 4.79616e-14 | 4.79616e-14 |

| Arm | ≤1.0: K / W / s | ≤0.1: K / W / s | ≤0.01: K / W / s | ≤0.0016: K / W / s |
|---|---|---|---|---|
| packets_exhaustive | 3 / 3 / 1.19 | 4 / 32 / 1.20 | 14 / 575 / 1.37 | 20 / 928 / 1.47 |
| packets_early | 3 / 3 / 1.11 | 4 / 6 / 1.12 | 14 / 190 / 1.18 | 20 / 297 / 1.22 |
| selected_ci | 3 / 3 / 1.20 | 4 / 490 / 1.33 | 17 / 490 / 1.34 | 22 / 490 / 1.35 |
| davidson | 3 / 3 / 1.25 | 4 / 4 / 1.25 | 5 / 5 / 1.25 | 6 / 6 / 1.25 |

Readout 1, column compactness (packets_exhaustive vs selected_ci): **mixed**.
Readout 2, work: not applicable; columns favour neither arm.
Readout 3, early screening: **preserves accuracy**.
Readout 4, Davidson: does not reach every threshold in fewer columns than both restricted arms.
Span support at 18 columns, Davidson's last prefix: packets_exhaustive 34, packets_early 34, selected_ci 29, davidson 325 determinants.

## hubbard_2x3_u8 (complete_span_lattice_control, t)

400 determinants, 175 singlet columns, budget 175. Singlet reference -2.177793552; sector ground state is a singlet.

| Arm | Completed | Stop | Columns | Final error | Ground overlap | H actions | Support | Time (s) | Worker RSS (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| packets_exhaustive | 3/3 | residual_tolerance | 175 | -3.55271e-15 | 1.0000 | 7642 | 400 | 2.70 | 68.3 |
| packets_early | 3/3 | residual_tolerance | 175 | -4.44089e-16 | 1.0000 | 3326 | 400 | 1.70 | 67.2 |
| selected_ci | 3/3 | residual_tolerance | 175 | -2.04281e-14 | 1.0000 | 175 | 400 | 1.00 | 66.6 |
| davidson | 3/3 | no_admissible_direction | 27 | 3.9968e-15 | 1.0000 | 27 | 400 | 0.06 | 58.0 |

| Arm | Error at 64 | Error at 128 | Error at 175 |
|---|---:|---:|---:|
| packets_exhaustive | 0.0563439 | 7.26803e-05 | -3.55271e-15 |
| packets_early | 0.0546907 | 9.50482e-05 | -4.44089e-16 |
| selected_ci | 0.0850443 | 0.0030179 | -2.04281e-14 |
| davidson | 3.9968e-15 | 3.9968e-15 | 3.9968e-15 |

| Arm | ≤1.0: K / W / s | ≤0.1: K / W / s | ≤0.01: K / W / s | ≤0.0016: K / W / s |
|---|---|---|---|---|
| packets_exhaustive | 21 / 669 / 0.12 | 55 / 2715 / 0.37 | 78 / 4178 / 0.62 | 102 / 5544 / 0.96 |
| packets_early | 21 / 266 / 0.09 | 55 / 1256 / 0.26 | 80 / 1843 / 0.45 | 105 / 2469 / 0.68 |
| selected_ci | 16 / 175 / 0.07 | 61 / 175 / 0.15 | 103 / 175 / 0.31 | 135 / 175 / 0.52 |
| davidson | 10 / 10 / 0.04 | 12 / 12 / 0.05 | 14 / 14 / 0.05 | 15 / 15 / 0.05 |

Complete-span control, packets_exhaustive: PASS at the full span.
Complete-span control, packets_early: PASS at the full span.
Complete-span control, selected_ci: PASS at the full span.
Complete-span control, davidson: stopped at 27 columns (no_admissible_direction, final error 4e-15).

Readout 1, column compactness (packets_exhaustive vs selected_ci): **mixed**.
Readout 2, work: not applicable; columns favour neither arm.
Readout 3, early screening: **outside the declared band** (frontier error outside 10% of exhaustive at 128, 175). Early minus exhaustive error (negative: early lower) -0.001653 at 64, +2.237e-05 at 128, +3.109e-15 at 175. Both errors are within 1e-08 of the reference at 175.
Readout 4, Davidson: reaches every threshold in fewer columns than both restricted arms, so the packets-versus-selected-CI comparison concerns bounded-support compactness only.
Span support at 27 columns, Davidson's last prefix: packets_exhaustive 186, packets_early 182, selected_ci 94, davidson 400 determinants.

## h2o_cas8e6o (development_complete_span_control, Ha)

225 determinants, 105 singlet columns, budget 105. Singlet reference -75.01268131; sector ground state is a singlet.

| Arm | Completed | Stop | Columns | Final error | Ground overlap | H actions | Support | Time (s) | Worker RSS (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| packets_exhaustive | 3/3 | residual_tolerance | 38 | 1.27898e-13 | 1.0000 | 535 | 77 | 0.93 | 61.6 |
| packets_early | 3/3 | residual_tolerance | 38 | 1.27898e-13 | 1.0000 | 214 | 77 | 0.78 | 61.5 |
| selected_ci | 3/3 | residual_tolerance | 38 | 8.52651e-14 | 1.0000 | 105 | 67 | 0.83 | 61.6 |
| davidson | 3/3 | no_admissible_direction | 14 | 1.56319e-13 | 1.0000 | 14 | 113 | 0.81 | 60.1 |

| Arm | Error at 64 | Error at 105 |
|---|---:|---:|
| packets_exhaustive | 1.27898e-13 | 1.27898e-13 |
| packets_early | 1.27898e-13 | 1.27898e-13 |
| selected_ci | 8.52651e-14 | 8.52651e-14 |
| davidson | 1.56319e-13 | 1.56319e-13 |

| Arm | ≤1.0: K / W / s | ≤0.1: K / W / s | ≤0.01: K / W / s | ≤0.0016: K / W / s |
|---|---|---|---|---|
| packets_exhaustive | 3 / 3 / 0.82 | 3 / 3 / 0.82 | 9 / 143 / 0.83 | 12 / 213 / 0.85 |
| packets_early | 3 / 3 / 0.74 | 3 / 3 / 0.74 | 9 / 53 / 0.75 | 12 / 66 / 0.75 |
| selected_ci | 3 / 3 / 0.80 | 3 / 3 / 0.80 | 9 / 105 / 0.81 | 14 / 105 / 0.81 |
| davidson | 3 / 3 / 0.81 | 3 / 3 / 0.81 | 4 / 4 / 0.81 | 5 / 5 / 0.81 |

Complete-span control, packets_exhaustive: stopped at 38 columns (residual_tolerance, final error 1.28e-13).
Complete-span control, packets_early: stopped at 38 columns (residual_tolerance, final error 1.28e-13).
Complete-span control, selected_ci: stopped at 38 columns (residual_tolerance, final error 8.53e-14).
Complete-span control, davidson: stopped at 14 columns (no_admissible_direction, final error 1.56e-13).

Readout 1, column compactness (packets_exhaustive vs selected_ci): **mixed**.
Readout 2, work: not applicable; columns favour neither arm.
Readout 3, early screening: **preserves accuracy**.
Readout 4, Davidson: does not reach every threshold in fewer columns than both restricted arms.
Span support at 14 columns, Davidson's last prefix: packets_exhaustive 30, packets_early 30, selected_ci 24, davidson 113 determinants.

All scheduled cells retained: PASS.
Packets add column compactness (primary and both held-out inputs): **NO**.

Repeated deterministic trajectories count once. Two or three timing repetitions are descriptive. No quantum, hardware, shot-cost, optimized-selected-CI or asymptotic-scaling claim follows, and no solver default changes.

Revised after observation: readout 3 names the failing clause and the sign of each difference instead of calling every miss a loss; readout 4 adds a factual note when Davidson ends above the reference without the declared trigger, and the declared span support at matched columns, which the executed summarizer omitted; and the support caveat above. No verdict rule changed.

Declaration: `09b1918be26a70d5a86f85c3fc3923bd58ea6c14`. Execution: `56c4a5ec82e63e5c6e5b67499d2a8e1a0395e8d5`.
