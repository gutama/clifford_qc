# H2O spin-matched residual-growth comparison

Exploratory exact simulation of one H2O STO-3G CAS(8e,6o) Hamiltonian: 225 determinants, singlet solver spans.
All arms start with the complete singlet CAS(2e,2o) on orbitals [3,4].
Target: 1.6 mHa versus the full-sector energy.
Fresh processes: 9; failed/timed out: 0.

## First target prefix

Medians over target-reaching runs; success denominators include all declared repeats.
Time includes common numerical setup plus cumulative solver work; interpreter imports/startup are excluded.
RSS is OS process high-water memory including imports/setup, not incremental solver memory.

| Arm | Target repeats | Columns | H actions | Candidates scored | Time incl. setup (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| residual_packets | 3/3 | 12 | 213 | 210 | 0.703172 | 162.445 |
| spin_adapted_selected_ci | 3/3 | 14 | 105 | 1067 | 0.649146 | 162.445 |
| adaptive_active_space | 3/3 | 105 | 105 | 10 | 0.709706 | 162.473 |

## Support and tracked payload at first target

Medians over target-reaching runs. V/HV and the screened-candidate H-action cache are partial array payloads, not total resident storage.
Every arm additionally constructs fixed external-group frame copies; these, temporary candidates and other work arrays are outside these two counters. Use OS RSS for whole-process memory.

| Arm | Determinant support | Basis amplitude nonzeros | V/HV (KiB) | Screened-candidate cache (KiB) |
|---|---:|---:|---:|---:|
| residual_packets | 27 | 27 | 84.375 | 0.000 |
| spin_adapted_selected_ci | 24 | 32 | 98.438 | 639.844 |
| adaptive_active_space | 225 | 285 | 738.281 | 0.000 |

## Retained-column frontiers

Median error (mHa) at the largest observed prefix within each cap; no interpolation across CAS block jumps.
Equal columns do not equalize support, cache memory or selection work.

| Column cap | Residual packets | Spin-adapted selected CI | Adaptive active space |
|---|---:|---:|---:|
| 8 | 10.994126 | 12.080238 | 47.986964 |
| 16 | 0.437455 | 0.666399 | 47.986964 |
| 32 | 0.000135 | 0.001272 | 33.038495 |
| 64 | 0.000000 | 0.000000 | 20.765892 |
| 105 | 0.000000 | 0.000000 | 0.000000 |

## Final readout

| Arm | Completed repeats | Median error (mHa) | Stop reasons | Median full process time (s) |
|---|---:|---:|---|---:|
| residual_packets | 3/3 | 0.000000000 | residual_tolerance | 1.783425 |
| spin_adapted_selected_ci | 3/3 | 0.000000000 | residual_tolerance | 1.652527 |
| adaptive_active_space | 3/3 | 0.000000000 | residual_tolerance | 1.776317 |

## Interpretation and scope

Every arm preserves singlet symmetry. Selected CI selects deterministic configuration-state functions, not spin-contaminated determinants.
Packets are refreshed after growth; active-space enlargement adds complete new singlet CAS blocks.
Cached candidate H actions are charged when first built. Retained V and HV are reused; timed solvers build neither full H nor H².
Dense S², the complete singlet frame, exact diagonals and operator compilation remain common small-sector setup costs. Setup constructs 225 S² columns.
JSON retains array payloads, amplitude support and every prefix. Payload is not RSS; setup plus candidate caches must also be considered.
The full reference energy is obtained after all measured solves. An independent occupation replay checks selection, Ritz energies, residuals, spin, support and action counts.
These three-repeat tiny-system timings are descriptive. Column compression alone establishes no time/memory advantage; no optimized SHCI/DMRG, scalable spin adaptation, quantum preparation, shots or quantum advantage is established.

Declaration: `7cc79ff0df0598da56801b3fc3f79da93f880947`.
Execution: `a54438009fc33fe490949f487eddffd6ab529296`.
