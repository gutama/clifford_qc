# Exact packet projection caching

Exploratory exact classical sector-vector comparisons; caches are opt-in and defaults remain unchanged.
84/84 scheduled cells retained; 8/8 post-timing historical controls retained. 8 content-addressed scientific payloads are stored once; every cell retains its own timing/memory observations.

Basis caching owns one V† for each fixed accepted span; frame caching owns group CSC blocks and a fixed frame adjoint. It caches neither candidate vectors nor H actions. Construction is included in solver and total times.
The supervisor alone surveys committed records. Worker RSS includes imports, setup, solver and final serialization. Phase RSS values are cumulative OS high-water marks, not isolated phase allocation peaks. Owned cache bytes exclude Python/container overhead.

## h2o (development, Ha)

225 determinants; 32 columns; target 0.0016 Ha.

### exhaustive

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 3/3 | 0.678 | 0.142 | 0.822, [0.809, 0.829] | 1.000, [1.000, 1.000] | 1.35026937e-07 | 3/3 |
| basis | 3/3 | 0.696 | 0.131 | 0.820, [0.797, 0.826] | 1.004, [0.987, 1.034] | 1.35026937e-07 | 3/3 |
| frame | 3/3 | 0.748 | 0.074 | 0.844, [0.797, 0.858] | 0.974, [0.966, 1.017] | 1.35026937e-07 | 3/3 |
| combined | 3/3 | 0.691 | 0.058 | 0.756, [0.745, 0.768] | 1.069, [1.061, 1.095] | 1.35026937e-07 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 52.598 | 61.473 | 62.398 | 62.398 | 0.000 | 0.0000 |
| basis | 51.547 | 61.527 | 62.410 | 62.410 | 0.106 | 0.0005 |
| frame | 51.355 | 61.020 | 61.734 | 61.801 | 0.013 | 0.0021 |
| combined | 52.102 | 61.730 | 62.520 | 62.520 | 0.119 | 0.0019 |

1 distinct scientific trajectory/trajectories; 0 unique audits, 0 missed winners.

### early

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 3/3 | 0.736 | 0.056 | 0.792, [0.747, 0.793] | 1.000, [1.000, 1.000] | 1.35026937e-07 | 3/3 |
| combined | 3/3 | 0.667 | 0.033 | 0.699, [0.681, 0.729] | 1.132, [1.029, 1.166] | 1.35026937e-07 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 51.609 | 61.598 | 62.254 | 62.254 | 0.000 | 0.0000 |
| combined | 52.332 | 61.441 | 61.984 | 61.984 | 0.119 | 0.0020 |

1 distinct scientific trajectory/trajectories; 5 unique audits, 0 missed winners.

## hubbard_2x4_u4 (primary, t)

4900 determinants; 64 columns; target 0.0016 t.

### exhaustive

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 5/5 | 0.271 | 24.495 | 24.766, [24.677, 25.179] | 1.000, [1.000, 1.000] | 1.93591273 | 0/5 |
| basis | 5/5 | 0.286 | 15.300 | 15.554, [15.068, 16.733] | 1.619, [1.480, 1.619] | 1.93591273 | 0/5 |
| frame | 5/5 | 0.262 | 22.134 | 22.442, [22.147, 22.944] | 1.103, [1.079, 1.114] | 1.93591273 | 0/5 |
| combined | 5/5 | 0.274 | 13.479 | 13.767, [13.608, 14.009] | 1.820, [1.797, 1.840] | 1.93591273 | 0/5 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 51.359 | 63.355 | 137.988 | 137.988 | 0.000 | 0.0001 |
| basis | 52.188 | 63.512 | 142.656 | 142.656 | 4.710 | 0.0147 |
| frame | 51.426 | 63.312 | 137.863 | 137.863 | 0.504 | 0.0185 |
| combined | 51.480 | 63.281 | 144.020 | 144.020 | 5.215 | 0.0328 |

1 distinct scientific trajectory/trajectories; 0 unique audits, 0 missed winners.
Primary basis speed/RSS clause: FAIL; 37.20% total-time reduction, 1.0338× worker RSS.
Primary frame speed/RSS clause: FAIL; 9.38% total-time reduction, 0.9991× worker RSS.
Primary combined speed/RSS clause: FAIL; 44.41% total-time reduction, 1.0437× worker RSS.

### early

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 5/5 | 0.278 | 7.800 | 8.082, [7.943, 8.512] | 1.000, [1.000, 1.000] | 1.94577515 | 0/5 |
| combined | 5/5 | 0.296 | 4.457 | 4.753, [4.688, 4.888] | 1.694, [1.657, 1.748] | 1.94577515 | 0/5 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 51.301 | 63.055 | 131.395 | 131.395 | 0.000 | 0.0001 |
| combined | 51.379 | 63.402 | 136.926 | 136.926 | 5.215 | 0.0353 |

1 distinct scientific trajectory/trajectories; 12 unique audits, 0 missed winners.

## beh2_cas4e4o (held_out_molecule, Ha)

36 determinants; 20 columns; target 0.0016 Ha.

### exhaustive

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 3/3 | 0.026 | 0.003 | 0.029, [0.029, 0.030] | 1.000, [1.000, 1.000] | -1.42108547e-14 | 3/3 |
| basis | 3/3 | 0.028 | 0.004 | 0.031, [0.030, 0.032] | 0.938, [0.893, 1.000] | -1.42108547e-14 | 3/3 |
| frame | 3/3 | 0.028 | 0.003 | 0.031, [0.030, 0.031] | 0.954, [0.939, 1.000] | -1.42108547e-14 | 3/3 |
| combined | 3/3 | 0.027 | 0.002 | 0.029, [0.029, 0.030] | 0.983, [0.976, 1.031] | -1.42108547e-14 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 51.953 | 58.312 | 58.312 | 58.312 | 0.000 | 0.0000 |
| basis | 52.086 | 58.738 | 58.738 | 58.738 | 0.003 | 0.0000 |
| frame | 52.148 | 57.801 | 57.930 | 57.930 | 0.002 | 0.0006 |
| combined | 52.758 | 58.723 | 58.723 | 58.723 | 0.005 | 0.0004 |

1 distinct scientific trajectory/trajectories; 0 unique audits, 0 missed winners.

### early

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 3/3 | 0.026 | 0.002 | 0.028, [0.028, 0.028] | 1.000, [1.000, 1.000] | -1.42108547e-14 | 3/3 |
| combined | 3/3 | 0.025 | 0.002 | 0.027, [0.026, 0.028] | 1.009, [0.975, 1.053] | -1.42108547e-14 | 3/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 51.723 | 58.461 | 58.461 | 58.461 | 0.000 | 0.0000 |
| combined | 51.387 | 58.625 | 58.625 | 58.625 | 0.005 | 0.0004 |

1 distinct scientific trajectory/trajectories; 0 unique audits, 0 missed winners.

## hubbard_2x3_u8 (held_out_lattice, t)

400 determinants; 32 columns; target 0.0016 t.

### exhaustive

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 3/3 | 0.040 | 0.234 | 0.268, [0.268, 0.276] | 1.000, [1.000, 1.000] | 0.341441046 | 0/3 |
| basis | 3/3 | 0.035 | 0.195 | 0.231, [0.227, 0.233] | 1.201, [1.181, 1.205] | 0.341441046 | 0/3 |
| frame | 3/3 | 0.035 | 0.133 | 0.170, [0.169, 0.171] | 1.574, [1.562, 1.637] | 0.341441046 | 0/3 |
| combined | 3/3 | 0.037 | 0.106 | 0.142, [0.140, 0.145] | 1.884, [1.843, 1.966] | 0.341441046 | 0/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 51.871 | 58.418 | 59.469 | 59.469 | 0.000 | 0.0000 |
| basis | 52.344 | 58.348 | 59.277 | 59.277 | 0.189 | 0.0006 |
| frame | 51.414 | 57.910 | 58.754 | 58.754 | 0.027 | 0.0023 |
| combined | 52.758 | 58.988 | 59.809 | 59.934 | 0.216 | 0.0026 |

1 distinct scientific trajectory/trajectories; 0 unique audits, 0 missed winners.

### early

Physical trajectories/selections/counters versus the pre-change module: PASS.

| Cache | Completed | Setup (s) | Solver (s) | Total (s), IQR | Paired speed ratio, IQR | Final error | Target |
|---|---:|---:|---:|---|---|---:|---:|
| none | 3/3 | 0.036 | 0.105 | 0.143, [0.134, 0.143] | 1.000, [1.000, 1.000] | 0.385302798 | 0/3 |
| combined | 3/3 | 0.034 | 0.054 | 0.088, [0.088, 0.095] | 1.627, [1.423, 1.639] | 0.385302798 | 0/3 |

| Cache | Import peak (MiB) | Setup peak (MiB) | Solver-end peak (MiB) | Worker peak (MiB) | Owned cache (MiB) | Cache construction (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 51.387 | 58.195 | 59.301 | 59.301 | 0.000 | 0.0000 |
| combined | 52.371 | 58.637 | 59.301 | 59.301 | 0.216 | 0.0027 |

1 distinct scientific trajectory/trajectories; 5 unique audits, 2 missed winners.

All reported policies preserve the pre-change physical trajectories: PASS.

Medians of setup and solver are separate; the total is the median of paired sums. Ratios pair cache and baseline cells within each repetition. Repeated deterministic paths do not increase independent scientific/audit sample size. A fixed-budget speedup is not a matched-accuracy claim. BeH2 tests a different small molecular active space, not larger-molecule scaling. No optimized-SCI, hardware or quantum advantage is established.

Declaration: `4942a71f0b4b9415e1c28f6b980fdd530a281278`. Execution: `da5feeccd50c56fa777f8b08a84fc538170d5278`.
