# Frame-coordinate packet projection

Exploratory exact classical sector-vector comparisons; caches are opt-in and defaults remain unchanged.
56/56 scheduled cells retained; 8/8 post-timing historical controls retained. 14 content-addressed scientific payloads are stored once; every cell retains its own timing/memory observations.

Combined caching owns one V† per accepted span and fixed group blocks/frame adjoint. Coordinate projection owns C=F†V and C† per span and one frame adjoint. It caches neither candidate vectors nor H actions. Construction is included in solver and total times.
The supervisor alone surveys committed records. Worker RSS includes imports, setup, solver and final serialization. Phase RSS values are cumulative OS high-water marks, not isolated phase allocation peaks. Owned cache bytes exclude Python/container overhead.

## h2o (development, Ha)

225 determinants; 32 columns; target 0.0016 Ha.

### exhaustive

Physical trajectories and all declared discrete decisions versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 3/3 | 0.902 | 0.087 | 0.996, [0.968, 0.998] | 1.000, [1.000, 1.000] | 1.35026937e-07 | 3/3 |
| coordinates | 3/3 | 0.910 | 0.105 | 1.012, [1.011, 1.021] | 0.967, [0.947, 0.978] | 1.35026923e-07 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.598 | 63.145 | 64.051 | 64.051 | 0.119 | 0.0031 |
| coordinates | 53.234 | 62.781 | 63.918 | 63.918 | 0.106 | 0.0030 |

2 distinct scientific trajectory/trajectories; 0 audit entries across distinct encoded paths (not independent samples), 0 missed winners.

### early

Physical trajectories and all declared discrete decisions versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 3/3 | 1.137 | 0.054 | 1.188, [1.109, 1.356] | 1.000, [1.000, 1.000] | 1.35026937e-07 | 3/3 |
| coordinates | 3/3 | 1.072 | 0.069 | 1.131, [1.079, 1.181] | 1.050, [1.027, 1.145] | 1.35026923e-07 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.180 | 62.535 | 63.543 | 63.543 | 0.119 | 0.0029 |
| coordinates | 53.180 | 62.664 | 63.738 | 63.738 | 0.106 | 0.0041 |

2 distinct scientific trajectory/trajectories; 10 audit entries across distinct encoded paths (not independent samples), 0 missed winners.

## hubbard_2x4_u4 (primary, t)

4900 determinants; 64 columns; target 0.0016 t.

### exhaustive

Physical trajectories and all declared discrete decisions versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 5/5 | 0.450 | 28.650 | 29.101, [26.987, 33.831] | 1.000, [1.000, 1.000] | 1.93591273 | 0/5 |
| coordinates | 5/5 | 0.413 | 14.072 | 14.485, [14.266, 14.681] | 1.907, [1.892, 2.366] | 1.93591273 | 0/5 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.562 | 65.164 | 96.980 | 96.980 | 5.215 | 0.0657 |
| coordinates | 53.242 | 65.281 | 96.824 | 96.824 | 3.643 | 0.1154 |

2 distinct scientific trajectory/trajectories; 0 audit entries across distinct encoded paths (not independent samples), 0 missed winners.
Primary coordinates: speed clause PASS, RSS clause PASS, global eligibility FAIL; 50.22% total-time reduction, 0.9984× worker RSS.

### early

Physical trajectories and all declared discrete decisions versus the pre-change module: FAIL.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 5/5 | 0.511 | 8.975 | 9.483, [9.425, 10.254] | 1.000, [1.000, 1.000] | 1.94577515 | 0/5 |
| coordinates | 5/5 | 0.483 | 5.184 | 5.991, [5.255, 6.045] | 1.864, [1.569, 1.951] | 1.94577515 | 0/5 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.344 | 65.172 | 96.395 | 96.395 | 5.215 | 0.0671 |
| coordinates | 53.273 | 65.105 | 95.824 | 95.824 | 3.643 | 0.1119 |

2 distinct scientific trajectory/trajectories; 24 audit entries across distinct encoded paths (not independent samples), 0 missed winners.
Primary coordinates: speed clause PASS, RSS clause PASS, global eligibility FAIL; 36.83% total-time reduction, 0.9941× worker RSS.

## beh2_cas4e4o (held_out_molecule, Ha)

36 determinants; 20 columns; target 0.0016 Ha.

### exhaustive

Physical trajectories and all declared discrete decisions versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 3/3 | 0.039 | 0.004 | 0.044, [0.042, 0.050] | 1.000, [1.000, 1.000] | -1.42108547e-14 | 3/3 |
| coordinates | 3/3 | 0.037 | 0.003 | 0.043, [0.041, 0.047] | 0.943, [0.903, 1.181] | -1.42108547e-14 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.148 | 59.512 | 59.637 | 59.637 | 0.005 | 0.0005 |
| coordinates | 53.207 | 59.816 | 59.816 | 59.816 | 0.005 | 0.0003 |

1 distinct scientific trajectory/trajectories; 0 audit entries across distinct encoded paths (not independent samples), 0 missed winners.

### early

Physical trajectories and all declared discrete decisions versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 3/3 | 0.044 | 0.003 | 0.048, [0.043, 0.049] | 1.000, [1.000, 1.000] | -1.42108547e-14 | 3/3 |
| coordinates | 3/3 | 0.048 | 0.003 | 0.052, [0.051, 0.066] | 0.952, [0.714, 0.959] | -1.42108547e-14 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.148 | 59.789 | 59.789 | 59.789 | 0.005 | 0.0007 |
| coordinates | 53.148 | 59.641 | 59.773 | 59.773 | 0.005 | 0.0003 |

1 distinct scientific trajectory/trajectories; 0 audit entries across distinct encoded paths (not independent samples), 0 missed winners.

## hubbard_2x3_u8 (held_out_lattice, t)

400 determinants; 32 columns; target 0.0016 t.

### exhaustive

Physical trajectories and all declared discrete decisions versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 3/3 | 0.061 | 0.220 | 0.277, [0.273, 0.283] | 1.000, [1.000, 1.000] | 0.341441046 | 0/3 |
| coordinates | 3/3 | 0.060 | 0.197 | 0.258, [0.249, 0.311] | 1.122, [0.930, 1.138] | 0.341441046 | 0/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.273 | 59.371 | 60.566 | 60.566 | 0.216 | 0.0067 |
| coordinates | 53.273 | 59.559 | 60.684 | 60.684 | 0.179 | 0.0042 |

2 distinct scientific trajectory/trajectories; 0 audit entries across distinct encoded paths (not independent samples), 0 missed winners.

### early

Physical trajectories and all declared discrete decisions versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| combined | 3/3 | 0.068 | 0.087 | 0.157, [0.156, 0.158] | 1.000, [1.000, 1.000] | 0.385302798 | 0/3 |
| coordinates | 3/3 | 0.072 | 0.131 | 0.193, [0.184, 0.237] | 0.815, [0.692, 0.849] | 0.385302798 | 0/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| combined | 53.738 | 59.984 | 61.180 | 61.180 | 0.216 | 0.0048 |
| coordinates | 53.273 | 60.145 | 61.285 | 61.285 | 0.179 | 0.0041 |

2 distinct scientific trajectory/trajectories; 10 audit entries across distinct encoded paths (not independent samples), 4 missed winners.

All reported policies preserve physical trajectories and every declared discrete decision: FAIL.
Joint correctness, completeness, primary speed and memory bar under both policies: FAIL. Defaults remain unchanged.

Medians of setup and solver are separate; the total is the median of paired sums. Ratios pair cache and baseline cells within each repetition. Repeated deterministic paths do not increase independent scientific/audit sample size. A fixed-budget speedup is not a matched-accuracy claim. BeH2 tests a different small molecular active space, not larger-molecule scaling. No optimized-SCI, hardware or quantum advantage is established.

## Ordered-shortlist differences

- hubbard_2x4_u4/early, step 28: shortlist order differs; membership agrees; selected candidate 2452 versus 2452.
- hubbard_2x4_u4/early, step 31: shortlist order differs; membership agrees; selected candidate 3181 versus 3181.
- hubbard_2x4_u4/early, step 32: shortlist order differs; membership agrees; selected candidate 3163 versus 3163.
- hubbard_2x4_u4/early, step 36: shortlist order differs; membership agrees; selected candidate 3415 versus 3415.

These are representative repeat-zero differences; every repetition is checked. Exact ordered-shortlist equality remains the frozen criterion, so equal memberships and selected candidates do not excuse a failure.

Reporting/checker changes after observation: distinguish valid independently replayed observations from a failed equivalence criterion; report shortlist-order differences explicitly; enforce the frozen scheduled-cell count in the report. No acceptance rule or observation byte changed.

Declaration: `5ce28f984eb34537711d500b00676e0b2ac45078`. Execution: `bd1d80f1bfd9e63840ba8381354f1ea9446d971b`.
