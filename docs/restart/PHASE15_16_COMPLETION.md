# Phase 15 and Phase 16: bounded implementation completion

The [result-free engineering contract](PHASE15_16_COMPLETION_CONTRACT.md) and
[fixed design](phase15_16_completion_design.json) define this completion.
The [record](phase15_16_completion_results.json.gz) retains 288 sampled outcomes,
three matrix-free time-family validations and three sampled integrations.
This completes the missing **capabilities** on TFIM2, TFIM3 and a Hubbard dimer;
it does not establish a cost advantage or reopen historical negative studies.

## Phase 15: sampled second moments and convergence

`SampledSecondMomentProvider` in `subspace/second_moment.py` compiles S, H and
K=H² rows on a sampled-only reference. The existing coefficient-storage gate
runs before H² construction. Reference metadata, rather than an invented
density, binds determinant, stabilizer and sampled capabilities. Explicit word
and setting caps bound subsequent grouping work. Reference amplitudes, callback
closures and device memory are outside the coefficient metric.

Every matrix entry is reconstructed from the same joint histograms. A single
Bonferroni family covers the independent real and imaginary entry functionals
using bounded-shot Hoeffding radii. The group range is its coefficient-l1 bound,
not the observed range or an estimated variance. Correlations within one shot
are retained; IID preparations across shots/settings remain an assumption.
The schedule and row universe must be fixed before these observations.

For coefficients c, let N=c†Sc, M=c†Hc and Q=c†Kc. If R is an absolute-entry
radius matrix, `|c|†R|c|` bounds the corresponding quadratic-form error on that
simultaneous event. Therefore it covers **all** c, including a Ritz vector
selected from the same measured matrices. This does not license changing the
row universe or optional stopping without a new error budget.

`MeasuredMomentProblem.assess` normalizes the coefficient vector's Euclidean
scale, requires the lower N endpoint to exceed the declared floor, and propagates
intervals through M/N and Q/N−(M/N)². Signed point variances remain visible.
Negative point values are never clipped into convergence. Inconsistent moment
regions return no residual certificate, and unresolved normalization abstains.
A separate cancellation allowance uses the existing `RESOLUTION=1e-12` scale;
this numerical guard is reported separately from the statistical radius.
The IID coverage statement is a real-arithmetic statement about the compiled
functionals; the bounded validation checks numerical agreement independently.

A residual is within tolerance only when its **upper** interval endpoint is
within tolerance; a positive lower endpoint can establish it is above tolerance.
Other outcomes are unresolved. This identifies proximity to some eigenvalue,
not ground-state dominance. An excited eigenstate is a deliberate control.

`folded_roots` solves the measured pencil `K−2ωH+ω²S`, then assesses the physical
direction represented by each returned coefficient vector. An upper bound on
its true folded expectation gives an interval `ω ± sqrt(upper)` containing an
eigenvalue of H. The noisy pencil eigenvalue itself is not a certificate of an
ordered exact folded root. Distinctness, ground-root identity, and global
completeness are not claimed; overlap ridge is rejected for this operation.

## Phase 16B: explicitly truncated real-time responses

`truncated_time_family` in `subspace/time_evolution.py` supplies the plan's
explicitly truncated alternative. It is opt-in and retains the global phase:

`A_p(t)=exp(−it s) sum_{k=0}^p [−it(H−sI)]^k/k!`.

No default exponential MV or density is constructed. Degree, time grid,
coefficient support, total product work, norm error, infidelity, energy error,
retained rank and conditioning have explicit finite budgets. Multiplication
and sum support are checked before their allocations. The returned operators
are immutable and feed the existing exact or sampled projected providers.

For B=sum|h_w| of H−sI and x=B|t|, the uniform Taylor remainder is bounded by
`exp(x)x^(p+1)/(p+1)!`. Sparse-threshold pruning has its own recurrence allowance.
For their sum ε<1, the output norm error is at most ε, normalized infidelity
at most 4ε², and the energy-expectation error at most 4Bε. These describe
truncation and pruning in exact arithmetic, not a formal floating-point proof.
`validate_truncated_time_family` separately checks actual actions, norm,
fidelity and energy against matrix-free propagation with a stated numerical
allowance and an explicit validation-amplitude cap. Production construction
and sampled inference request no exact vector or spectrum.

`admit_overlap` reports the retained rank and condition of the supplied Gram
matrix and refuses too few resolved directions. A numerical admission from a
measured Gram matrix is not itself a confidence certificate; the measured
moment normalization gate still applies. Polynomial generators are not unitary
preparation circuits, and their sparse support can grow quickly. There is no
controlled-time-evolution resource claim.

The mathematics is standard, not a new algorithmic speedup theorem. Related
primary references are [Epperly, Lin and Nakatsukasa, *A theory of quantum
subspace diagonalization*](https://arxiv.org/abs/2110.07492) and
[Berry et al., *Simulating Hamiltonian dynamics with a truncated Taylor
series*](https://arxiv.org/abs/1412.4687). This implementation is a bounded
classical operator-response representation; it does not implement the latter
paper's quantum LCU simulation algorithm.

## Validation and its limits

The three cases use seven, twenty-seven and thirty-one measured nonidentity
words, in three, six and three graph-state settings respectively. All 288
simultaneous entry regions contain the independent exact moments in this fixed
seeded run. At the largest budget, TFIM2's 32 energy-direction reports establish
the declared 0.5-unit residual tolerance. TFIM3 and Hubbard retain unresolved
outcomes. Coverage is a simulation check, not an empirical calibration of a
real device or a universal convergence-rate claim.

The time-family action errors are below 3e-12 on these cases. Retained ranks
are two, three and three; conditioning is explicitly reported. The sampled
end-to-end Taylor integrations retain both resolved and unresolved results,
including Hubbard's unresolved normalization. No unsuccessful outcome is
removed or relabelled as convergence.

The independent checker does not import the producer, either new consumer,
the native confidence routines or the native projected solver. It rebuilds
Pauli/Jordan-Wigner operators, measurement-circuit action, histogram decoding,
S/H/K blocks, Hoeffding radii, retained solves, interval arithmetic and dense
Taylor actions. Published simulator probability tables are checked against an
independent dense circuit, then used as the exact floating-point inputs for
seed replay. Tiny roundoff probabilities can otherwise change multinomial RNG
stream consumption despite equivalent circuits. Support/product counters are
native bookkeeping and are range-checked, not independently recounted.

The initial numerical run did not retain those draw tables. Adding them
strengthened independent reproducibility; the design, seeds, budgets and
acceptance rules were unchanged. All 288 numerical outcomes remain retained.
Source digests and declaration→implementation→observation ancestry are checked.
The original Phase 15/16 decision records have explicit unchanged-byte bindings.

Focused controls cover complex entries, no exact-capability fallback, resource
rejection, scale invariance, cancellation, negative variances, unresolved
normalization, excited roots, coherent time reversal, zero time, numerical rank
loss, and deliberate histogram/readout/probability corruptions.

## Use and reproduce

```python
from clifford_qc.subspace import (
    SampledSecondMomentProvider, SecondMomentStoragePolicy,
    truncated_time_family,
)

# reference supplies only metadata and joint sampling; H and generators are
# sparse operator inputs. The provider never requests a reference density.
provider = SampledSecondMomentProvider(
    reference, H, generators, max_words=256, max_settings=128,
    storage_policy=SecondMomentStoragePolicy(1_000_000),
)
observations = reference.sample(provider.compile_measurement().settings, 100_000, seed=7)
problem = provider.reconstruct(observations, delta=0.05)
solved = problem.solve(tau_s=1e-8, max_condition=1e8)
report = problem.assess(solved.ritz_vector(), residual_tolerance=0.5)
folded = problem.folded_roots(-2.5, residual_tolerance=0.5, count=1)
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/run_phase15_16_completion.py --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_phase15_16_completion.py
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_phase15_16_completion.py -v
```

Use `--output` with a fresh path to regenerate; observations are never overwritten
by the producer. `--no-history` is a local-development mode for the independent
checker, not the registered CI command. The record pins Python 3.12.14,
NumPy 2.3.5 and SciPy 1.17.0. Both numerical checks run through the shared
runner/artifact contract in CI.

Completion is limited to the shipped capabilities and this bounded validation.
Historical molecular/Hubbard cost failures, the Phase 16A NO_GO and Phase 16B v3
CONDITIONAL verdict are unchanged. A circuit-native controlled-evolution backend,
advantage experiment, affordable large-bank measured H², ground-state certificate,
and device-noise study remain separate work with their own premises.
