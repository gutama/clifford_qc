# Phase 12 matched-budget arms: seed-clustered replication

Source: `benchmarks/results/m7_seed_replication.jsonl` — 160 cells across 2 systems x 4 orderings x 20 seeds, at the ladder's own determinant budget.

## Arm errors (Ha), median over cells

An arm with identical min and max does not move with the draw. `matched_selected_ci` is sample-independent by construction, so its constancy is a correctness check rather than a finding.

| system | arm | median | min | max |
| :--- | :--- | ---: | ---: | ---: |
| hubbard_2x2 | `budget_selected_ci` | 1.65326 | 1.27432 | 1.65326 |
| hubbard_2x2 | `matched_selected_ci` | 1.27432 | 1.27432 | 1.27432 |
| hubbard_2x2 | `acase` | 0.94047 | 0.94047 | 0.94047 |
| hubbard_2x2 | `qsci_dressed_acase` | 0.94047 | 0.94047 | 1.01057 |
| hubbard_2x2 | `qsci_haar_dressed_acase` | 0.94904 | 0.53352 | 1.20678 |
| hubbard_2x3 | `budget_selected_ci` | 2.67881 | 2.45704 | 3.61932 |
| hubbard_2x3 | `matched_selected_ci` | 2.45704 | 2.45704 | 2.45704 |
| hubbard_2x3 | `acase` | 2.45704 | 2.45704 | 2.45704 |
| hubbard_2x3 | `qsci_dressed_acase` | 2.45704 | 2.45704 | 2.45704 |
| hubbard_2x3 | `qsci_haar_dressed_acase` | 2.48359 | 1.93301 | 2.72296 |

## `qsci_haar_dressed_acase` vs `qsci_dressed_acase`, paired within draw

Negative median log10 ratio favours the packet arm. Pooled rows exclude the `random` control.

| group | n | seeds | cell win | seed win | seed p | median log10 | 95% CI | verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| hubbard_2x2 — pooled policy | 60 | 20 | 0.59 | 0.53 | 1.000 | -0.0034 | [-0.0235, +0.0046] | no effect |
| hubbard_2x2 — order=graph | 20 | 20 | 0.47 | 0.47 | 1.000 | +0.0022 | [-0.0084, +0.0348] | no effect |
| hubbard_2x2 — order=physics | 20 | 20 | 0.80 | 0.80 | 0.012 | -0.0350 | [-0.0398, -0.0160] | packets better |
| hubbard_2x2 — order=probability | 20 | 20 | 0.47 | 0.47 | 1.000 | +0.0220 | [-0.0491, +0.0802] | no effect |
| hubbard_2x2 — control=random | 20 | 20 | 0.25 | 0.25 | 0.041 | +0.0555 | [+0.0268, +0.0756] | packets worse |
| hubbard_2x3 — pooled policy | 60 | 20 | 0.56 | 0.72 | 0.096 | -0.0190 | [-0.0331, +0.0000] | no effect |
| hubbard_2x3 — order=graph | 20 | 20 | 0.10 | 0.10 | 0.000 | +0.0296 | [+0.0114, +0.0369] | packets worse |
| hubbard_2x3 — order=physics | 20 | 20 | 0.83 | 0.83 | 0.008 | -0.0467 | [-0.0861, -0.0092] | packets better |
| hubbard_2x3 — order=probability | 20 | 20 | 0.79 | 0.79 | 0.019 | -0.0358 | [-0.0608, -0.0157] | packets better |
| hubbard_2x3 — control=random | 20 | 20 | 0.11 | 0.11 | 0.001 | +0.0195 | [+0.0118, +0.0276] | packets worse |

## Provenance

- git `919dadb209104c0b37a5b6c339d6f5d762f3911e` (dirty)
- evidence: `oracle_sampled; subspace construction only`
- boundary: paired within-draw comparison of the ladder's matched-budget arms; fixed_krylov excluded as deterministic
