# Step 5: sampled projection and independent ratio evaluation

The [frozen declaration](sampled_ratio_declaration.json) specifies one ideal
joint-Pauli-shot experiment before outcomes are drawn. It compares estimation
of the **same pilot-frozen direction** with output-focused allocation and with
complete S/H reconstruction. All three input definitions, seeds, budgets,
failure rules and the 20% promotion criterion are fixed. The two held-out inputs
are never used to choose settings, cutoffs, budgets or variants. They are known
synthetic test definitions, not blind external data.

## Consumer and conventions

`PreparedSampledReference` implements the existing sampled-reference protocol
with a preparation identity and a draw callback. It has no exact expectation or
amplitude API. The callback returns actual joint `MeasurementBatch` histograms;
the adapter checks identity, circuit action, signed readouts, counts and cost.
`SampledProjectedProvider` uses a coefficient-only bank, snapshots H/generators,
and reconstructs numerical S/H plus simultaneous entrywise uncertainty from
observations. Exact evaluation on its bank fails explicitly. The exact provider
continues to reject sampled-only references.

Operators retain the existing complex Clifford/Pauli representation of
`Cl(2n,C)`, isomorphic to n-qubit matrices. Qubit zero is the leftmost tensor
factor and lowest two-bit Pauli lane. Pauli words are Hermitian; complex phases
belong to coefficients. The bank compiles `A_i† A_j` and `A_i† H A_j`.
Quadratic, Ritz and coupling functionals consume `iter_operator_terms`; packed
rows are never reconstructed as MV dictionaries to build these functionals.
Final aggregate support/storage still costs memory.

## Two stages and coverage

The uniform pilot buys the complete bank. The noisy projected solve learns the
overlap rank, retained frame, c and pilot energy. It also supplies covariance
estimates for allocation. Frozen plans own immutable coefficient/frame arrays,
functional maps, settings, integer counts, denominator floor and failure rules.
Their identities bind reference, H, pilot observations and circuits. Evaluation
rejects the pilot seed or a changed schedule. Fresh IID preparations are a
sampler/device assumption; metadata alone cannot prove physical independence.

For the ratio arm, allocate by the pilot variance of `f_H - E_pilot f_S`, with
two shots minimum in every needed setting of the **new** evaluation batch.
The baseline allocates to the sum of entry-functional variances, reconstructs
all S/H, and contracts them with the same c. Both charge the full common pilot
and the same total physical-shot budget. No exact variance or coefficient
oracle enters either allocation. Equal per-shot costs define allocation;
different circuit durations are subsequently priced in the ledger.

H and S use the same evaluation histograms. Their empirical covariance enters
the reported delta-method variance, which is an asymptotic diagnostic. The
finite-sample interval uses a different argument: for a fixed functional,
independent group shots with a-priori range R_g give

`r(delta)^2 = log(2/delta)/2 * sum_g R_g^2 / n_g`.

Each range is the coefficient l1 bound, never an observed range. Apply this
two-sided Hoeffding bound to H and S at delta/2 each. The joint rectangle has
coverage at least 1-delta by a union bound, with no independence assumption
between H and S. Only when S's lower endpoint exceeds the positive floor do
the four rectangle corners give a finite ratio interval. Otherwise the attempt
is unresolved and stays in the unconditional failure denominator. Coverage is
conditional on the independently learned pilot. Pilot normalization never
makes a stochastic denominator deterministic. Algebraically constant overlaps
have zero radius. Neither this interval nor centering removes ratio bias or
certifies a full-space ground energy.

## Reproduction and cost boundary

The declaration must be committed before `--execute`. The runner refuses to
overwrite observations, records declaration/execution commits and source
digests, and performs exact assessment only after observations and decisions.
The record checker validates declaration chronology and recomputes every
aggregate from all replica outcomes. Full replay redraws the declared streams;
timing, process RSS and execution stamps are advisory replay fields, while
scientific values and observed histogram identities are checked.

```bash
python benchmarks/run_sampled_ratio.py --validate
python benchmarks/run_sampled_ratio.py --execute --output docs/restart/sampled_ratio_results.json
python benchmarks/check_sampled_ratio.py
python benchmarks/run_sampled_ratio.py --check docs/restart/sampled_ratio_results.json
```

The ledger charges preparation, compiled gate depth, readout/reset, full pilot,
compilation, freezing/allocation and inference, including failed attempts.
Simulator draw time and independent dense assessment time are separate. The
device durations are illustrative, ideal, serial scenarios, with no hardware
calibration or noise mitigation. Array admission includes retained packed-bank
capacity, coefficient maps and a conservative dense/workspace allowance. Python
heap overhead is outside that array scope; a separate process RSS high-water
monitor covers the declared whole-process ceiling. Neither is a measured
peak-heap claim. The ground/retained-space errors are reported separately from
the primary estimation error around the frozen direction.

Results and the frozen-hypothesis verdict are added after execution. Historical
campaign evidence and the unchanged restart blueprint retain their original
verdicts.
