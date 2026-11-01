# Preregistered noise-aware selection: equal-shot test

**Result: the predeclared criterion failed.** The noise-aware penalty and the
unpenalized baseline selected the same candidate in all 384 paired replicates
across the six case/noise cells. Consequently their fresh-evaluation energies,
median losses and paired ratios are identical; every held-out ratio is 1.0.
This supports no benefit for this selector on this declared grid. It does not
rule out different noise-aware policies or other workloads.

The [result-free declaration](noise_aware_selection_declaration.json) was
committed as `c5bdbcc56620399e6e38f37cac6e23b11147fe8c`, followed by the
producer and initial independent checker in
`f4d3487a6cdd253d9c3b06dde2ae8ccc7b7a78f2`; the observation is a later
commit. The checker verifies this ancestry and the committed producer digest.
The [compressed record](noise_aware_selection_results.json.gz) retains every
word, selection and evaluation count vector, seed, candidate, failure, energy,
loss, penalty and summary. Development and held-out cases were all previously
available in the repository: held out from tuning this policy is not a claim of
blind external data.

| Case | Phase-flip p | Unique words | Shots per arm per replicate | Baseline median error | Noise-aware median error | Ratio, paired 95% bootstrap | Failures, baseline / aware |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TFIM4 development | 0.02 | 45 | 450,000 | 0.211113 J | 0.211113 J | 1.000 [1.000, 1.000] | 0 / 0 |
| TFIM4 development | 0.10 | 45 | 450,000 | 0.425360 J | 0.425360 J | 1.000 [1.000, 1.000] | 0 / 0 |
| H4 held out | 0.02 | 2,055 | 20,550,000 | 0.056548 Ha | 0.056548 Ha | 1.000 [1.000, 1.000] | 0 / 0 |
| H4 held out | 0.10 | 2,055 | 20,550,000 | 0.101967 Ha | 0.101967 Ha | 1.000 [1.000, 1.000] | 0 / 0 |
| LiH held out | 0.02 | 2,171 | 21,710,000 | 0.015668 Ha | 0.015668 Ha | 1.000 [1.000, 1.000] | 0 / 0 |
| LiH held out | 0.10 | 2,171 | 21,710,000 | 0.049320 Ha | 0.049320 Ha | 1.000 [1.000, 1.000] | 0 / 0 |

Each cell has 64 scheduled replicates. The primary metric is absolute difference
between the **fresh evaluation** Ritz energy and the independently constructed
full-space TFIM or fixed-singlet molecular ground energy; it includes subspace
bias, noise bias and shot error. It is not a measurement-only error, a spectral
certificate, or an estimate of hardware performance. The 95% interval is a
paired percentile bootstrap of median-loss ratios, using 2,000 fixed resamples.
Because choices and measurements were identical, the degenerate intervals
express equality in these simulations, not certainty about other systems.

No training candidate pencil was invalid, and no selected evaluation pencil
failed. Thus selection, evaluation and retained-rank failure counts are all
zero on this record; these observed zero counts do not establish a zero failure
probability.

## Baseline, penalty, and accounting

Both arms inspect the same two candidate pencils `[I, A1]` and `[I, A2]`. The
baseline chooses the smaller noisy selection-batch Ritz value. The aware arm
adds 0.1 times a state-independent Pauli-l1 channel-defect score for each
candidate's S and H entries. It uses the true simulated channel and strength;
there is no access to the exact state or ground energy in either selection
score. The full union of nonidentity words is sampled separately, then a fresh
independent evaluation batch is drawn. The pair of counterfactual arms shares
both simulated batches to make a paired comparison; each is charged the full
selection plus evaluation shots. One state preparation is charged per shot.

Each aware arm additionally evaluates two candidate penalties; each cell's
penalty setup evaluates four S and four H entries for each of two candidates,
with an absolute-coefficient sum per entry. This extra classical work is not
converted to seconds or a device-equivalent cost. Knowing the dephasing channel
is assumed free, with no calibration shots, calibration uncertainty or drift.
Grouping, readout error, circuit depth, gate-level noise, scheduling, device
latency and model-learning cost are excluded. Thus the numeric cost is an
unoptimized **equal-shot boundary**, not an end-to-end cost estimate or a
claim that either arm saves time or resources. Scoring with a different model,
using adaptive allocation, or adding calibration must be declared and tested
separately.

## Independent validation and limits

`check_noise_aware_selection.py` independently reconstructs the occupation or
spin Hamiltonians, reference state, Pauli actions, channel attenuation,
selection and evaluation pencils, seeded binomial counts, all losses, bootstrap
and verdict without importing the producer. It shares FCIDUMP parsing and the
existing independent occupation-sign/cross-domain oracle. It is independent of
the producer's native Pauli multiplication and projected solver, but not of
all input definitions or numerical libraries. Recorded source digests cover the
named inputs and implementations, not a complete transitive dependency closure.
The original calibration/design and experimental outcomes are unchanged.

The companion [priority audit](INDEPENDENT_CHECK_PRIORITIES.md) strengthens the
published H4 record: `check_fcidump_h4.py` reconstructs all selected-prefix and
complete-SD energies from an occupation-sector matrix and independently checks
the 7,371-word universe and 3,840 maximum H-row support. It does **not** replay
the original adaptive selection rule or certify reported Python heap bytes.

Reproduce and test with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_fcidump_h4.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_noise_aware_selection.py --verify-history
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_noise_selection_checks.py -v
```

The second command requires the published commit ancestry. A source checkout
without that history can run the checker without `--verify-history` to verify the
numerical content only. Replaying the producer is a verification run, not a
new scientific result; it refuses to overwrite the committed observation.
