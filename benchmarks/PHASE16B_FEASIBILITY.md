# Phase 16B feasibility: is a real-time Krylov pencil shot-survivable?

This specifies one experiment, at one scale, to answer one question before any
architecture is committed to it. `PLAN.md` §Phase 16B scopes a circuit-native
real-time generator family and states that it "requires a different
matrix-element backend and resource model — architecture research". That is the
right diagnosis, and it is also expensive. This document defines the cheap
experiment that decides whether the expensive one is worth starting.

Everything below is preregistration. No arm here has been run except the pilot
in Section 3, which is a sizing probe and gates nothing.

## 1. What this decides

**Go/no-go on building the Phase 16B matrix-element backend.**

The backend is not a module, it is a second execution path: `MatrixElementBank`
assembles rows as sparse Pauli operators and reads every entry as an expectation
on the reference, and a real-time family satisfies neither assumption. Ancilla
registers, controlled time evolution, and an estimator whose variance is not a
word-variance are a different machine. That is weeks of work, and it invalidates
the resource model that `resources()` currently reports.

Before paying that, one question has to be answered, and it can be answered at
four qubits in dense arithmetic with no new backend at all:

> **Q8 (operational form).** On a predeclared instance, does the real-time family
> reach a target energy error at an overlap-noise level a finite-shot calculation
> could actually deliver — and does it do so at a *lower total estimand count*
> than the operator-response family already in the package?

Two clauses, both necessary. The first is about conditioning: a basis that only
works in exact arithmetic is not a method. The second is about cost: if
real-time buys accuracy at the same 10⁶-word measurement universe A-CASE already
pays (`molecular/results/results_summary.json`: 350 196 words at m = 11, 1 165 663
at m = 31), it has bought nothing.

## 2. Why the real-time family is structurally different

This is the entire reason the phase is worth pricing, so it goes first and it is
checked numerically rather than asserted.

For `A_k = e^{-iH t_k}` on a **uniform** grid `t_k = k·dt`, and Hermitian `H`
which commutes with its own propagator:

```
S_ij = <psi| e^{+iH i dt}   e^{-iH j dt} |psi>  = c(j - i)
H_ij = <psi| e^{+iH i dt} H e^{-iH j dt} |psi>  = d(j - i)
```

Both matrices are **Toeplitz**. The whole `m × m` pencil is `2m - 1` values of
each of two scalar functions of a single time argument. Contrast the two families
the package already has:

| family | distinct estimands for an `m`-dimensional pencil | grows with `|H|`? |
|---|---|---|
| operator response (`A-CASE`) | `O(m²)` operator rows over a word universe `O(m²·|H|)` | yes |
| power Krylov (`krylov_response`) | union of `supp(H^k)`, `k ≤ 2m+1` (`run_krylov_width.py`) | yes, and superlinearly |
| **real-time (16B)** | **`2m` scalars** (`c`, `d` on `k ≥ 0`; negatives by Hermiticity) | **no** |

`benchmarks/probe_realtime_krylov.py` confirms the Toeplitz identity holds to
exactly zero residual in double precision at `m = 4, 6, 8, 10`.

There is a stronger variant. The **unitary (Prony) pencil** replaces the
Hamiltonian matrix entirely: with `S⁰_ij = c(j-i)` and `S¹_ij = c(j-i+1)`, the
generalized eigenvalues of `S¹ v = λ S⁰ v` are `λ = e^{-iE·dt}`, so energies come
from `arg(λ)` and **`d` is never measured**. The Hamiltonian enters only through
the propagator that the circuit implements anyway. This is the lineage of
quantum filter diagonalization and unitary quantum Krylov (Parrish–McMahon 2019;
Klymko et al. 2022; Stair et al. 2020); the package should position against it
rather than rediscover it.

That is the upside. Section 3 is the downside.

## 3. Pilot: the conditioning wall, quantified

Producer: `benchmarks/probe_realtime_krylov.py` (a probe — no committed record,
gates nothing). Instance: TFIM `n = 4`, `J = h = 1` at criticality, product-state
reference `|0000>`, `dt = π/(E_max - E_min) = 0.330084`. Noise model: i.i.d.
complex Gaussian of width `eps` on each Toeplitz scalar, Hermiticity restored,
hard truncation of the normalized overlap spectrum at `eps`. Chemical accuracy
`1.6e-3 Ha`. 40 replicas.

**Exact arithmetic — accuracy and conditioning rise together:**

| `m` | `cond(S)` | rank | energy error |
|---:|---:|---:|---:|
| 4 | 1.96e+02 | 4 | 3.06e-01 |
| 6 | 5.75e+03 | 6 | 2.70e-02 |
| 8 | 3.38e+06 | 8 | 2.51e-04 |
| 10 | 5.32e+09 | 10 | 2.40e-14 |

The basis that reaches chemical accuracy (`m = 8`) is the one whose overlap
spectrum already spans six decades. This is not a defect of the instance; it is
what real-time Krylov bases do.

**Under noise — Hermitian pencil (`c` and `d`), median error over 40 replicas:**

| `eps` \ `m` | 4 | 6 | 8 | 10 | best resolvable rank |
|---|---:|---:|---:|---:|---:|
| 0 | 3.06e-01 | 2.70e-02 | **2.51e-04** | **2.40e-14** | 10 |
| 1e-5 | 3.06e-01 | 2.22e-02 | 3.08e-03 | **1.23e-03** | 8 |
| 1e-4 | 3.06e-01 | 8.73e-02 | 2.20e-02 | 1.22e-02 | 7 |
| 1e-3 | 3.02e-01 | 2.57e-01 | 8.18e-02 | 1.82e-02 | 7 |
| 1e-2 | 5.20e-01 | 3.84e-01 | 3.16e-01 | 1.85e-01 | 6 |

**Unitary (Prony) pencil (`c` only)** is *better* in the median — `6.17e-04`
against `1.23e-03` at `eps = 1e-5, m = 10` — and needs half the estimands, since
`d` is never measured. But it is a non-Hermitian eigenproblem with no variational
floor, and its tail is far worse:

| `eps`, `m` | Prony median | Prony p90 | Hermitian p90 |
|---|---:|---:|---:|
| 1e-5, 10 | 6.17e-04 | 1.02e-03 | 2.17e-03 |
| 1e-4, 10 | 8.39e-03 | **1.87e+00** | 1.46e-02 |
| 1e-3, 8 | 6.00e-02 | **3.13e+00** | 1.26e-01 |
| 1e-2, 10 | 1.24e-01 | **2.49e+00** | 2.72e-01 |

A p90 of `1.87 Ha` on a spectrum of width `9.5 Ha` is not an inaccurate answer,
it is a wrong root: without a variational floor a noisy replica can return a
phase belonging to a different eigenvalue. Cheaper estimand, unbounded outliers.
Both are arms; neither is the obvious winner, and the choice between them is a
risk decision the experiment should inform rather than assume.

**Three readings, all of which the real experiment must confirm or overturn:**

1. **Chemical accuracy needs `eps ≈ 1e-5`.** For a Hadamard test the ancilla bit
   is Bernoulli, so the standard error on one estimand is `≤ 1/√N` and
   `N ≈ 1/eps² = 1e10` shots *per estimand*. With `~4m = 40` estimands at
   `m = 10`, that is `~4e11` shots. At 10 kHz, roughly a year of device time —
   for a four-qubit problem a laptop solves exactly.
2. **Rank saturates.** At `eps = 1e-3`, `m = 10` resolves only 7 modes. Adding
   time points past the noise floor buys nothing. The resolvable-mode count, not
   `m`, is the real basis dimension, and it is set by the noise.
3. **But the estimand count is 4–5 orders of magnitude below A-CASE's.** 40
   scalars against 350 196 Pauli words. Shots-per-estimand is worse; estimand
   count is dramatically better. **Which wins is exactly what has not been
   measured, and is the whole point of the experiment.**

Reading 3 is why this is a go/no-go and not a refutation. The pilot's crude
regularizer (hard truncation at `eps`) is also the weakest available; the package
already ships `overlap_ridge` and per-mode `overlap_noise_floor` in
`solve_projected`, and either can only improve these rows.

## 4. Scope boundary

What this experiment is **not**, stated before the design so no result can be
read past it:

- **Not a hardware claim.** No device, no calibration, no error model beyond the
  declared estimator variance. Device cards remain illustrative accounting.
- **Not a quantum-advantage claim, and cannot become one.** The reference state
  is classically preparable and the instances are exactly diagonalizable. A pass
  here establishes a *necessary* condition (the basis survives shot noise), never
  a sufficient one. The asymptotic argument for real-time Krylov — `poly(n)`
  circuit depth for `e^{-iHt}` against exponential classical propagation — is not
  tested by this experiment and must not be asserted from it. Records carry
  `"quantum_advantage_claim": false`.
- **Not a scaling claim.** Four and six qubits. Section 8's rungs exist to detect
  a trend, not to extrapolate one.
- **Not a Trotter-resource claim.** The Trotterized arm prices propagation error
  against basis error; it does not cost out a compiled circuit.

## 5. Instances (predeclared)

Three at `n = 4`, one confirmation rung at `n = 6`. All exactly diagonalizable,
so every arm has ground truth.

| id | model | reference | why |
|---|---|---|---|
| `tfim4_crit` | `tfim(4, J=1, h=1)` | `\|0000>` | critical, worst conditioning, pilot instance |
| `tfim4_para` | `tfim(4, J=1, h=3)` | `\|0000>` | gapped control — should be easy; if it is not, the method is dead |
| `h2_sto3g` | H₂/STO-3G FCIDUMP (`benchmarks/configs/acase_ladder.json` molecular rung) | RHF determinant | chemistry, and the instance `run_krylov_width.py` already prices |
| `tfim6_crit` | `tfim(6, J=1, h=1)` | `\|000000>` | one rung, for trend only |

Reference-state spectral support is reported per instance (the pilot's
`|0000>` on `tfim4_crit` carries weight `> 1e-6` on 11 of 16 eigenstates). This
is the quantity that bounds resolvable rank and it must be in every record: a
reference with support on 3 eigenstates makes any Krylov method look good and
says nothing.

## 6. Arms

Five, sharing one instance, one grid rule, and one noise model.

| arm | basis | estimands | notes |
|---|---|---|---|
| `exact_diag` | — | — | ground truth |
| `rt_hermitian` | `e^{-iHkΔt}`, exact propagation | `c`, `d` | Section 2 pencil |
| `rt_unitary` | `e^{-iHkΔt}`, exact propagation | `c` only | Prony pencil; non-variational |
| `rt_trotter` | `trotter2_unitary` at declared step count | `c`, `d` | separates propagation error from basis error |
| `acase_control` | operator response, existing `run_acase` | Pauli words | the incumbent, matched on accuracy |
| `power_krylov_control` | `krylov_response(H, m)` | `supp(H^k)` | the family `run_krylov_width.py` already prices |

**Grid rule (predeclared):** `Δt = π/(E_max - E_min)` from the exact spectrum for
`n = 4`, and from a declared norm bound at `n = 6`. A grid chosen per-instance
after seeing results is a tuned result, not a measured one. A secondary sweep
`Δt ∈ {0.5, 1, 2} × π/span` is reported separately and labelled as a sweep.

**Regularization arms**, crossed with the above: hard truncation at the noise
floor (pilot), per-mode `overlap_noise_floor`, and `overlap_ridge`. All three are
already in `solve_projected`; the experiment supplies the vector-valued floor
derived from the *same* normalized overlap matrix, per that function's contract.

**Noise model.** Additive complex Gaussian of width `eps` per estimand, with
`eps` swept over `{0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}`, 200 replicas, declared
seeds. This is a stand-in for shots and is labelled `heuristic` in the evidence
vocabulary, **not** `finite_sample`. A genuine finite-sample arm needs the
ancilla estimator that does not exist yet — which is precisely what a `go`
verdict authorizes building. Converting `eps` to shots uses the stated Bernoulli
bound `N ≈ 1/eps²` and is reported as an arithmetic consequence of the model, not
a measurement.

## 7. Metrics

Per (instance, arm, `m`, `eps`, regularizer, replica):

- ground-energy error against exact diagonalization; median and p90 (the Prony
  tail is the reason p90 is mandatory, not decorative);
- `cond(S)`, `effective_rank`, full overlap spectrum before truncation;
- **resolvable-mode count**: modes above their own noise floor — the headline
  diagnostic, since Section 3 reading 2 says this and not `m` is the basis size;
- **distinct estimand count** and, for the control arms, word universe and QWC
  setting count, so the cost comparison is like-for-like;
- **shots-to-target** under the declared variance bound, for each arm, at a fixed
  accuracy target — the number the go/no-go turns on;
- propagation fidelity `|<exact|trotter>|` for `rt_trotter`;
- wall time and peak RSS, for parity with the molecular records.

## 8. Decision rule (preregistered)

Fix the target at chemical accuracy, `1.6e-3 Ha`, median over replicas.

**GO** — build the backend — if on `tfim4_crit` **and** `h2_sto3g`, some
real-time arm reaches the target at `eps ≥ 1e-5` **and** its shots-to-target is
at least **10×** below the matched `acase_control` on the same instance at the
same target.

**NO-GO** — do not build it — if no real-time arm reaches the target at any
`eps ≥ 1e-6`, **or** if shots-to-target is within 10× of `acase_control`. The
first says the basis cannot survive noise; the second says it survives but buys
nothing, which for the package's purposes is the same verdict.

**CONDITIONAL** — anything between, including a pass on `tfim4_crit` but not
`h2_sto3g`, or a pass only under `overlap_ridge`. A conditional result publishes
as a negative-with-caveat and authorizes *one* follow-up: the estimator-variance
refinement in Section 10, not the backend.

The `tfim4_para` gapped control is a **sanity gate, not a decision input**: if the
easy instance fails, the implementation is wrong and no other row is readable.

Committing this rule before results is the point. The repo already enforces this
ordering elsewhere (`check_r3b_preregistration.py`: "a preregistration earns its
name from commit order, not from the word"), and the molecular suite's
`adaptive_oracle_stop_used: true` is the standing example of what happens without
it — every one of those rows consumed the FCI answer through the stopping rule.

## 9. Implementation

Nothing here needs the new backend. That is the design constraint that makes the
experiment cheap.

| file | status | ~LOC | content |
|---|---|---|---|
| `benchmarks/probe_realtime_krylov.py` | **written** | 170 | pilot; Section 3 |
| `benchmarks/configs/phase16b_feasibility.json` | new | — | instances, grid rule, `eps` grid, seeds, decision thresholds, **no results** |
| `benchmarks/run_phase16b_feasibility.py` | new | ~400 | producer; writes `reference_results/phase16b_feasibility.json` |
| `benchmarks/check_phase16b_preregistration.py` | new | ~150 | gates the config before any result commit |
| `benchmarks/check_phase16b_feasibility.py` | new | ~200 | rebuilds the record; verifies the decision rule was applied as written |

Reused unchanged: `solve_projected` (including `overlap_noise_floor` and
`overlap_ridge`, which is why the regularization arms cost nothing to add),
`SectorStatevectorBackend.as_linear_operator()` with `scipy.sparse.linalg.expm_multiply`
for matrix-free propagation at `n = 6`, `trotter2_unitary`, `krylov_response`,
`run_acase` for the control arm, `stamp_record` for provenance, and the
`EvidenceLevel` vocabulary.

New primitives needed: none. Dense `expm` at `n = 4` and matrix-free
`expm_multiply` at `n = 6` are the ground-truth propagators, and both exist.

**Record schema** `clifford_qc.phase16b_feasibility.v1`, stamped via
`stamp_record`, carrying `quantum_advantage_claim: false`, the evidence label
`heuristic` on every noise row, the config digest, and the decision-rule
evaluation as a field rather than as prose.

**Commit order, and it matters:** config + preregistration gate first, in their
own commit; producer and record second. The gate must pass against a config
carrying no results.

## 10. If it says no

A NO-GO is the good outcome to get cheaply, and it is publishable. "Real-time
quantum Krylov does not survive realistic overlap noise at four qubits, measured
against a matched operator-response control under a preregistered decision rule"
is a stronger contribution than most of what the subspace-method literature
currently reports — and the package is unusually well set up to say it, because
`selected_ci.py`'s controls and the evidence vocabulary already exist to make the
comparison honest.

It also closes Q8 with evidence instead of leaving it open at 0%, and it protects
the several weeks the backend would have cost.

The one follow-up a NO-GO or CONDITIONAL authorizes is narrowing the noise model:
the Gaussian stand-in ignores that a Hadamard test's variance depends on the
estimand's own magnitude (`(1 - Re<U>²)/N`), and `c(τ)` decays with `τ`, so late
grid points are *cheaper* per unit precision than the flat model assumes. If the
verdict lands within 10× of a threshold, that correction is worth making before
the verdict is final. Beyond that, do not re-litigate: the decision rule is the
decision.

## 11. Effort

Roughly one week: two days for producer and config, one for the preregistration
gate, one to run the sweep, one to write the record and the verdict. The pilot is
already done.

Phase 16B stays at 0% in `PHASE_STATUS.json` until the record is committed and
its gate passes. This document is a specification, not progress.
