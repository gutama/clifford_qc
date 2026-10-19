# Residual packet action reuse and screening

Exploratory exact classical H2O STO-3G CAS(8e,6o): 225 determinants, 105 singlet-frame columns.
Every arm starts from the same singlet CAS(2e,2o) on [3,4]. Target: 1.6 mHa on this fixed Hamiltonian.
Fresh processes: 15; failed/timed out: 0.

## First target prefix

Medians over successful target-reaching runs. Denominators include failures. Time includes symmetry/model setup and cumulative solver work; imports/startup are excluded here and included in the terminal process time below.

| Arm | Target runs | Columns | H actions | Exact scores | Time incl. setup (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| original | 3/3 | 12 | 213 | 210 | 0.705603 | 166.660 |
| reuse | 3/3 | 12 | 57 | 210 | 0.697407 | 164.508 |
| shortlist | 3/3 | 12 | 66 | 63 | 0.630759 | 164.508 |
| combined | 3/3 | 12 | 57 | 63 | 0.655958 | 166.617 |
| selected_ci | 3/3 | 14 | 105 | 1067 | 0.666353 | 166.684 |

## Partial array payloads at first target

These are partial array payloads, not total resident memory. The SCI candidate counter includes borrowed frame-column views plus owned H arrays; its views already belong to the common frame and are not additional owned RAM. Setup additionally owns dense S², the full singlet frame and diagonal; Python containers, operator storage and temporary arrays are excluded. RSS above is the whole process high-water mark.

| Arm | V/HV (KiB) | Extra frame snapshot (KiB) | Cached frame actions (KiB) | SCI candidate cache (KiB) |
|---|---:|---:|---:|---:|
| original | 84.375 | 0.000 | 0.000 | 0.000 |
| reuse | 84.375 | 369.141 | 200.391 | 0.000 |
| shortlist | 84.375 | 0.000 | 0.000 | 0.000 |
| combined | 84.375 | 369.141 | 200.391 | 0.000 |
| selected_ci | 98.438 | 0.000 | 0.000 | 639.844 |

## Full-run audits and terminal costs

Audit losses concern exact candidate energy-lowering scores. Audits select the full-screen winner. Non-audit steps with negligible shortlisted gain fall back to full screening; both costs are charged.

| Arm | Completed runs | Audits | Missed winners | Largest audit loss (Ha) | Fallbacks | Final H actions | Median process time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 3/3 | 0 | 0 | 0 | 0 | 535 | 1.822672 |
| reuse | 3/3 | 0 | 0 | 0 | 0 | 57 | 1.762317 |
| shortlist | 3/3 | 21 | 0 | 0 | 0 | 214 | 1.641652 |
| combined | 3/3 | 21 | 0 | 0 | 0 | 57 | 1.722348 |
| selected_ci | 3/3 | 0 | 0 | 0 | 0 | 105 | 1.657895 |

## Trajectory equivalence

Original versus reuse-only (same repeat); compare selected group identities and entire energy/residual trajectories, including later prefixes. A mismatch must be reported, not hidden by target-only agreement.

- Repeat 0: same prefix count/selections = True; maximum paired energy difference 7.11e-14 Ha; residual-norm difference 2.11e-15 Ha.
- Repeat 1: same prefix count/selections = True; maximum paired energy difference 7.11e-14 Ha; residual-norm difference 2.11e-15 Ha.
- Repeat 2: same prefix count/selections = True; maximum paired energy difference 7.11e-14 Ha; residual-norm difference 2.11e-15 Ha.

## Scope

The cache retains all populated HF columns; its 105-action bound trades repeated operator applications for storage approaching a complete singlet action table. No full H or H² is built by the timed solver, but HF itself is a dense sector-by-frame table when populated.
Shortlisting uses a fixed 80% residual-mass prefix and every-fifth-decision full audits. It is heuristic and does not inherit finite-element convergence guarantees. Full residual stopping is independent of candidate screening.
Reference energies are evaluated after all timed workers. Independent occupation-H replay verifies packet spans, decisions, audits, cache population, costs and spin. Three-repeat tiny-sector timings are descriptive; this study establishes no general scaling, quantum shots/preparation, hardware, optimized SHCI/DMRG superiority or quantum advantage.

Declaration: `ab42279253ef461fb48bb3f1f3501b78ff0a6c96`.
Execution: `e5869e60659bb17bd41e93d20da98e1c5756f2a4`.
