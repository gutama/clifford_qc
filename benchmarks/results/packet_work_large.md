# Larger-sector residual packet work comparison

Open 2x4 Hubbard, t=1, U=4, mu=0, half filling: 16 qubits, 4,900 determinants and 1,764 singlet-frame columns.
The sector is 21.8 times the H2O pilot size. Every energy/error/residual below uses hopping units t, not Hartree.
Same classical CAS(2e,2o) starting protocol on sites [3,4]; this is not the lattice builder's Neel reference.
Budget: 64 total retained columns; target: 0.0016 t. Fresh processes: 15; failed/timed out: 0.

## Target attainment

First-target costs are shown only when the fixed accuracy target is reached. Denominators include failures.

| Arm | Target runs | Columns | H actions | Time including setup (s) |
|---|---:|---:|---:|---:|
| original | 0/3 | — | — | — |
| reuse | 0/3 | — | — | — |
| shortlist | 0/3 | — | — | — |
| combined | 0/3 | — | — | — |
| selected_ci | 0/3 | — | — | — |

## Final retained prefix

Medians over completed runs; this is a budget comparison when target attainment above is zero.

| Arm | Completed | Columns | Error (t) | Full residual (t) | H calls | Exact scores | Stop reasons |
|---|---:|---:|---:|---:|---:|---:|---|
| original | 3/3 | 64 | 1.93591273 | 2.21708111 | 10841 | 10838 | column_ceiling |
| reuse | 3/3 | 64 | 1.93591273 | 2.21708111 | 1301 | 10838 | column_ceiling |
| shortlist | 3/3 | 64 | 1.94577515 | 2.23289212 | 4469 | 4466 | column_ceiling |
| combined | 3/3 | 64 | 1.94577515 | 2.23289212 | 1288 | 4466 | column_ceiling |
| selected_ci | 3/3 | 64 | 2.07767315 | 2.34677846 | 1764 | 105591 | column_ceiling |

## Fixed-column error frontiers

Median error (t) at the largest observed prefix within each cap; no interpolation.

| Column cap | Original | Reuse | Shortlist | Combined | Selected CI |
|---|---:|---:|---:|---:|---:|
| 8 | 7.09663042 | 7.09663042 | 7.09663042 | 7.09663042 | 4.01250315 |
| 16 | 5.8157748 | 5.8157748 | 5.8157748 | 5.8157748 | 3.08603426 |
| 32 | 2.99440637 | 2.99440637 | 3.09616636 | 3.09616636 | 2.77916337 |
| 64 | 1.93591273 | 1.93591273 | 1.94577515 | 1.94577515 | 2.07767315 |

## Time and memory

Time-to-final includes setup plus solver; whole-process time also includes imports/startup. RSS is OS process high-water memory. V/HV and cache frame/actions are owned arrays; selected-CI direction columns are borrowed views of the common frame and are excluded from the owned-H-array column.

| Arm | Setup (s) | Setup + solver (s) | Whole process (s) | Peak RSS (MiB) | V/HV (MiB) | Extra cache frame/actions (MiB) | SCI owned H arrays (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 0.513 | 27.886 | 29.206 | 684.281 | 9.570 | 0.000 | 0.000 |
| reuse | 0.476 | 43.351 | 44.664 | 896.613 | 9.570 | 229.164 | 0.000 |
| shortlist | 0.501 | 26.208 | 27.309 | 684.199 | 9.570 | 0.000 | 0.000 |
| combined | 0.442 | 35.786 | 36.974 | 897.094 | 9.570 | 228.192 | 0.000 |
| selected_ci | 0.466 | 6.862 | 8.402 | 739.539 | 9.570 | 0.000 | 127.106 |

## Audits and trajectory equivalence

| Arm | Full-screen audits | Missed winners | Largest score loss (t) | Fallbacks |
|---|---:|---:|---:|---:|
| original | 0 | 0 | 0 | 0 |
| reuse | 0 | 0 | 0 | 0 |
| shortlist | 36 | 0 | 0 | 0 |
| combined | 36 | 0 | 0 | 0 |
| selected_ci | 0 | 0 | 0 | 0 |

- original/reuse, repeat 0: same prefix count/selections = True; maximum paired energy difference 1.02e-14 t, residual-norm difference 6.66e-15 t.
- original/reuse, repeat 1: same prefix count/selections = True; maximum paired energy difference 1.02e-14 t, residual-norm difference 6.66e-15 t.
- original/reuse, repeat 2: same prefix count/selections = True; maximum paired energy difference 1.02e-14 t, residual-norm difference 6.66e-15 t.
- shortlist/combined, repeat 0: same prefix count/selections = True; maximum paired energy difference 1.02e-14 t, residual-norm difference 7.55e-15 t.
- shortlist/combined, repeat 1: same prefix count/selections = True; maximum paired energy difference 1.02e-14 t, residual-norm difference 7.55e-15 t.
- shortlist/combined, repeat 2: same prefix count/selections = True; maximum paired energy difference 1.02e-14 t, residual-norm difference 7.55e-15 t.

Common dense S²/frame/diagonal arrays: 498.329 MiB. The temporary full-H payload during CSR compilation is 366.364 MiB; retained CSR H is 1.179 MiB (60830 nonzeros). These counters are not summed into a peak-RSS claim.

## Reference and scope

Independent occupation-sign reference: -5.012503152657 t. Two shifted sparse solves agree; full residual and singlet checks are retained in the record.
This tests one larger correlated lattice Hamiltonian at a fixed budget. It does not establish general scaling or larger molecular performance. Native CSR compilation temporarily builds full sector H, and dense spin/frame setup remains. Cache savings in H calls can coexist with worse runtime or memory. The shortlist is heuristic; all audits and fallback costs are charged. Earlier H2O evidence is unchanged.
The published gzip/base64 JSON envelope preserves the complete observations. Its digest and the uncompressed observation commit are bound separately; native producer JSON bytes are exactly reconstructible.
The independent checker rebuilds Hubbard signs, symmetry, trajectory decisions and cost counts. Exact repeated scientific trajectories may reuse a full replay, while every repeat has its measured times/RSS validated. No quantum measurement, hardware or optimized-SCI superiority claim follows.

Declaration: `375d9381c8489df7143a99d3b2ec7b64e9b1503f`.
Execution: `f1238efdce0e1e4702025a3552f7bc3d2e7cf462`.
