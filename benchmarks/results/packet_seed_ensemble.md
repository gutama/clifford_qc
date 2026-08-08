# Packet seed-ensemble: seed-clustered dressed-vs-packet inference

Sources: `benchmarks/results/packet_seed_ensemble.jsonl`, `benchmarks/results/packet_seed_ensemble_h2o_qsci.jsonl`, `benchmarks/results/packet_seed_ensemble_beh2_stretched.jsonl` — 1440 treatment cells (unbalanced design spanning 6 systems, 4 orderings, 4 shot settings, and 20 seed clusters).

Orderings and shot settings are repeated conditions within seed. Inference therefore uses one median paired effect per seed and resamples whole seed clusters. Cell win rate is descriptive only.

Negative median log10 ratio favours packets. A confidence interval straddling zero or a seed-level sign-test p >= 0.05 is a no-effect result. Reported p-values are unadjusted across secondary groups.

`random` is the predeclared ordering control and is kept separate from the candidate-policy pool by design.

Shot-setting rows use only systems present at every shot setting (h4_equilibrium, h4_stretched, hubbard_2x2, hubbard_2x3) so tiers remain like-for-like.

| Group | compared / cells | seeds | cell win | seed win | seed sign p | median seed log10 ratio | cluster 95% CI | verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| POOLED (policy orderings) | 1018 / 1080 | 20 | 0.67 | 0.95 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| CONTROL (random) | 341 / 360 | 20 | 0.54 | 0.67 | 0.238 | -0.000 | [-0.000, +0.000] | no effect |
| order=graph | 341 / 360 | 20 | 0.66 | 0.95 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| order=physics | 340 / 360 | 20 | 0.65 | 0.95 | 0.000 | -0.000 | [-0.001, -0.000] | negligible |
| order=probability | 337 / 360 | 20 | 0.69 | 0.95 | 0.000 | -0.000 | [-0.020, -0.000] | negligible |
| system=beh2_stretched (policy only) | 60 / 60 | 20 | 0.78 | 0.85 | 0.003 | -0.051 | [-0.095, -0.034] | packets better |
| system=h2o_qsci (policy only) | 54 / 60 | 18 | 0.44 | 0.53 | 1.000 | +0.000 | [-0.000, +0.000] | no effect |
| system=h4_equilibrium (policy only) | 189 / 240 | 20 | 0.69 | 0.72 | 0.096 | -0.000 | [-0.000, +0.000] | no effect |
| system=h4_stretched (policy only) | 240 / 240 | 20 | 0.83 | 1.00 | 0.000 | -0.016 | [-0.052, -0.003] | packets better |
| system=hubbard_2x2 (policy only) | 240 / 240 | 20 | 0.56 | 0.82 | 0.013 | -0.022 | [-0.033, -0.007] | packets better |
| system=hubbard_2x3 (policy only) | 235 / 240 | 20 | 0.59 | 0.84 | 0.004 | -0.004 | [-0.008, -0.001] | negligible |
| shots=32 (policy only; balanced systems) | 202 / 240 | 20 | 0.46 | 0.36 | 0.424 | +0.000 | [+0.000, +0.000] | no effect |
| shots=64 (policy only; balanced systems) | 222 / 240 | 20 | 0.62 | 0.78 | 0.031 | -0.000 | [-0.000, -0.000] | negligible |
| shots=128 (policy only; balanced systems) | 240 / 240 | 20 | 0.78 | 1.00 | 0.000 | -0.011 | [-0.016, -0.000] | packets better |
| shots=256 (policy only; balanced systems) | 240 / 240 | 20 | 0.76 | 1.00 | 0.000 | -0.048 | [-0.065, -0.040] | packets better |
| beh2_stretched / graph | 20 / 20 | 20 | 0.65 | 0.65 | 0.263 | -0.018 | [-0.043, +0.034] | no effect |
| beh2_stretched / physics | 20 / 20 | 20 | 0.70 | 0.70 | 0.115 | -0.047 | [-0.128, +0.001] | no effect |
| beh2_stretched / probability | 20 / 20 | 20 | 1.00 | 1.00 | 0.000 | -0.098 | [-0.134, -0.073] | packets better |
| beh2_stretched / random | 20 / 20 | 20 | 0.80 | 0.80 | 0.012 | -0.040 | [-0.052, -0.012] | packets better |
| h2o_qsci / graph | 18 / 20 | 18 | 0.53 | 0.53 | 1.000 | +0.000 | [-0.000, +0.000] | no effect |
| h2o_qsci / physics | 18 / 20 | 18 | 0.40 | 0.40 | 0.607 | +0.000 | [-0.000, +0.000] | no effect |
| h2o_qsci / probability | 18 / 20 | 18 | 0.40 | 0.40 | 0.607 | +0.000 | [-0.000, +0.000] | no effect |
| h2o_qsci / random | 18 / 20 | 18 | 0.40 | 0.40 | 0.607 | +0.000 | [-0.000, +0.000] | no effect |
| h4_equilibrium / graph | 63 / 80 | 20 | 0.63 | 0.74 | 0.064 | -0.000 | [-0.000, -0.000] | no effect |
| h4_equilibrium / physics | 63 / 80 | 20 | 0.66 | 0.65 | 0.263 | -0.000 | [-0.000, +0.000] | no effect |
| h4_equilibrium / probability | 63 / 80 | 20 | 0.80 | 0.85 | 0.003 | -0.000 | [-0.000, -0.000] | negligible |
| h4_equilibrium / random | 63 / 80 | 20 | 0.75 | 0.79 | 0.019 | -0.000 | [-0.000, -0.000] | negligible |
| h4_stretched / graph | 80 / 80 | 20 | 0.85 | 0.90 | 0.000 | -0.000 | [-0.006, -0.000] | negligible |
| h4_stretched / physics | 80 / 80 | 20 | 0.74 | 0.85 | 0.003 | -0.012 | [-0.027, -0.005] | packets better |
| h4_stretched / probability | 80 / 80 | 20 | 0.90 | 0.95 | 0.000 | -0.147 | [-0.186, -0.054] | packets better |
| h4_stretched / random | 80 / 80 | 20 | 0.75 | 0.85 | 0.003 | -0.082 | [-0.156, -0.002] | packets better |
| hubbard_2x2 / graph | 80 / 80 | 20 | 0.39 | 0.33 | 0.238 | +0.004 | [-0.006, +0.013] | no effect |
| hubbard_2x2 / physics | 80 / 80 | 20 | 0.63 | 0.95 | 0.000 | -0.034 | [-0.068, -0.021] | packets better |
| hubbard_2x2 / probability | 80 / 80 | 20 | 0.64 | 0.65 | 0.263 | -0.055 | [-0.075, +0.009] | no effect |
| hubbard_2x2 / random | 80 / 80 | 20 | 0.63 | 0.55 | 0.824 | -0.005 | [-0.029, +0.027] | no effect |
| hubbard_2x3 / graph | 80 / 80 | 20 | 0.77 | 1.00 | 0.000 | -0.020 | [-0.030, -0.016] | packets better |
| hubbard_2x3 / physics | 79 / 80 | 20 | 0.61 | 0.67 | 0.238 | -0.006 | [-0.017, +0.001] | no effect |
| hubbard_2x3 / probability | 76 / 80 | 20 | 0.38 | 0.35 | 0.263 | +0.003 | [-0.005, +0.006] | no effect |
| hubbard_2x3 / random | 80 / 80 | 20 | 0.03 | 0.00 | 0.000 | +0.024 | [+0.021, +0.030] | packets worse |

## Cost side

- median extra frontier scoring, policy orderings: `-381` candidates per run
- median packet directions retained, policy orderings: `3.0`
- packet-ineligible cells: `76` total; `57` in policy orderings
- cells dropped for unmatched M: `5` total; `5` in policy orderings

## Molecular/system records

- `h2o_qsci`: h2o_qsci_sto3g_cas6e5o; 10 qubits; sector dimension 100 (`benchmarks/results/packet_seed_ensemble_h2o_qsci.jsonl`)
- `beh2_stretched`: beh2_stretched_sto3g_cas4e6o(r=3.0); 12 qubits; sector dimension 225 (`benchmarks/results/packet_seed_ensemble_beh2_stretched.jsonl`)

## Provenance

- `benchmarks/results/packet_seed_ensemble.jsonl`: git `52eb3a8acb06cb360e8c7bb5111dc078cb30d779` (dirty); evidence `oracle_sampled; selector behaviour only`
- `benchmarks/results/packet_seed_ensemble_h2o_qsci.jsonl`: git `unknown`; evidence `oracle_sampled; selector behaviour only`; result digest verified against provenance record; generation source recorded at git `a635462d316ec5e8cd3aa19fc71419ff6f711983`
- `benchmarks/results/packet_seed_ensemble_beh2_stretched.jsonl`: git `unknown`; evidence `oracle_sampled; selector behaviour only`; result digest verified against provenance record; generation source recorded at git `a635462d316ec5e8cd3aa19fc71419ff6f711983`
- boundary: paired dressed-vs-packet comparison at matched budget; no implementable state-preparation claim
