# Packet seed-ensemble: paired dressed-vs-packet comparison

Source: `benchmarks/results/packet_seed_ensemble.jsonl` — 1280 cells (4 systems x 4 orderings x 4 shot settings).

Negative `median log10 ratio` favours the packet arm. A confidence interval straddling zero is a no-effect result, which is the §11C answer, not a missing one.

Pooled rows exclude the `random` ordering, which is the §11C control rather than a policy: the hierarchy is *expected* to lose under an uninformative order, and averaging that in turns a real ordering-dependent effect into a spurious null. The control is reported on its own row as calibration.

| Group | n | win rate | sign p | median log10 ratio | 95% CI | verdict |
| :--- | ---: | ---: | ---: | ---: | :---: | :--- |
| POOLED (policy orderings) | 904 | 0.67 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| CONTROL (random) | 303 | 0.53 | 0.409 | +0.000 | [-0.000, +0.000] | no effect |
| order=graph | 303 | 0.67 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| order=physics | 302 | 0.66 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| order=probability | 299 | 0.68 | 0.000 | -0.000 | [-0.014, -0.000] | negligible |
| control=random | 303 | 0.53 | 0.409 | +0.000 | [-0.000, +0.000] | no effect |
| system=h4_equilibrium (policy only) | 189 | 0.69 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| system=h4_stretched (policy only) | 240 | 0.83 | 0.000 | -0.011 | [-0.023, -0.000] | packets better |
| system=hubbard_2x2 (policy only) | 240 | 0.56 | 0.105 | -0.009 | [-0.043, +0.000] | no effect |
| system=hubbard_2x3 (policy only) | 235 | 0.59 | 0.011 | -0.002 | [-0.007, +0.000] | no effect |
| shots=32 (policy only) | 202 | 0.46 | 0.356 | +0.000 | [+0.000, +0.000] | no effect |
| shots=64 (policy only) | 222 | 0.62 | 0.001 | -0.000 | [-0.000, -0.000] | negligible |
| shots=128 (policy only) | 240 | 0.78 | 0.000 | -0.010 | [-0.021, -0.000] | negligible |
| shots=256 (policy only) | 240 | 0.76 | 0.000 | -0.069 | [-0.081, -0.028] | packets better |
| h4_equilibrium / graph | 63 | 0.63 | 0.067 | -0.000 | [-0.000, +0.000] | no effect |
| h4_equilibrium / physics | 63 | 0.66 | 0.018 | -0.000 | [-0.000, +0.000] | no effect |
| h4_equilibrium / probability | 63 | 0.80 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| h4_equilibrium / random | 63 | 0.75 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| h4_stretched / graph | 80 | 0.85 | 0.000 | -0.000 | [-0.000, -0.000] | negligible |
| h4_stretched / physics | 80 | 0.74 | 0.000 | -0.000 | [-0.016, -0.000] | negligible |
| h4_stretched / probability | 80 | 0.90 | 0.000 | -0.069 | [-0.164, -0.045] | packets better |
| h4_stretched / random | 80 | 0.75 | 0.000 | -0.047 | [-0.093, -0.001] | packets better |
| hubbard_2x2 / graph | 80 | 0.39 | 0.091 | +0.006 | [+0.000, +0.009] | no effect |
| hubbard_2x2 / physics | 80 | 0.63 | 0.029 | -0.053 | [-0.083, +0.000] | no effect |
| hubbard_2x2 / probability | 80 | 0.64 | 0.027 | -0.051 | [-0.103, +0.000] | no effect |
| hubbard_2x2 / random | 80 | 0.63 | 0.034 | -0.015 | [-0.029, +0.000] | no effect |
| hubbard_2x3 / graph | 80 | 0.77 | 0.000 | -0.012 | [-0.031, -0.007] | packets better |
| hubbard_2x3 / physics | 79 | 0.61 | 0.086 | -0.002 | [-0.006, +0.000] | no effect |
| hubbard_2x3 / probability | 76 | 0.38 | 0.060 | +0.003 | [+0.000, +0.007] | no effect |
| hubbard_2x3 / random | 80 | 0.03 | 0.000 | +0.022 | [+0.019, +0.032] | packets worse |

## Cost side

- median extra frontier scoring for the packet stage: `-381` candidates per run
- median packet directions retained: `3.0`
- packet-ineligible cells (fewer than two non-reference configurations): `51`
- cells dropped for unmatched M: `5`

## Provenance

- git sha: `52eb3a8acb06cb360e8c7bb5111dc078cb30d779` (dirty)
- evidence: `oracle_sampled; selector behaviour only`
- boundary: paired dressed-vs-packet comparison at matched budget; no implementable state-preparation claim
