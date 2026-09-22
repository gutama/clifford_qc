# Phase 16B feasibility: real-time Krylov under estimator noise

This design asks whether a circuit-native real-time family merits the separate
matrix-element backend scoped by `PLAN.md` Phase 16B and Q8. The pilot below
tests numerical structure and sizes a future experiment; it does not authorize
a backend or establish a measurement saving.

**Status: design specification, not completed preregistration.** This PR contains
the pilot and its regression tests. The result-free config, preregistration
checker, experiment producer, and result checker in Section 9 remain future
work. Their parameters and cost contracts must be committed and checked before
the decision experiment runs. The pilot was observed before this design and is
explicitly exploratory.

## 1. Question and scope

On the **same Hamiltonian, reference, energy target, and estimator cost model**,
can a real-time Krylov family tolerate noise and reduce modeled total shots by
at least 10x relative to A-CASE? Both noise tolerance and cost matter. A small
number of scalar outputs is not by itself a small measurement budget.

`MatrixElementBank` currently builds sparse Pauli rows and measures their
expectations on one reference. Controlled propagation and off-diagonal ancilla
estimators would require a separate execution and resource path. Dense pilot
calculations can test necessary numerical conditions before that work begins.

All noise evidence here is `heuristic`, not `finite_sample`. No hardware,
compiled-circuit cost, quantum advantage, or scaling conclusion follows. Records
must carry `quantum_advantage_claim: false`. A NO-GO is restricted to the frozen
instances, grid, regularizers, and variance model, not real-time Krylov generally.

## 2. Structure, estimator counts, and phase branches

For a Hermitian H and uniform grid, let `A_k = exp(-i H k dt)`. Then

```text
c(k) = <psi|exp(-i H k dt)|psi>
d(k) = <psi|H exp(-i H k dt)|psi>
S_ij = c(j-i),  H_ij = d(j-i),  i,j = 0,...,m-1.
```

Both matrices are Toeplitz because H commutes with its propagator. Negative
lags are conjugates of positive lags, `c(0)=1` is known for a normalized state,
and `d(0)` is real. Count **real scalar components requiring estimation**:

| Family | Data for an m-dimensional pencil | Measurement qualification |
|---|---|---|
| A-CASE | O(m²) operator rows, with overlapping Pauli supports | Count actual word union, settings, covariance, and coefficient weights |
| Power Krylov | moments of H through power 2m-1 | `krylov_response(H, m-1)` plus identity; support growth is instance-dependent |
| Exact real-time Hermitian | 2(m-1) components of c plus 2m-1 of d: **4m-3** | d is Hamiltonian-weighted, not a bounded unitary expectation |
| Exact real-time unitary | c(1),...,c(m): **2m** | Each real/imaginary part is a bounded unitary expectation |

The `order` argument in `run_krylov_width.py` counts powers after identity, so
its size is `order+1`; do not equate that argument with m. An O(m) scalar count
for real-time pencils does not remove Hamiltonian-dependent propagation or
measurement cost. At m=10 the counts above are 37 and 20, not exactly a factor
of two.

The pilot checks the Toeplitz construction against independently propagated
columns `V_k = exp(-i H k dt)|psi>`, comparing to `V†V` and `V†HV`. Checking
adjacent diagonals of a matrix already built from c(j-i) would be tautological.

The unitary pencil uses `S0_ij=c(j-i)` and `S1_ij=c(j-i+1)`. Its generalized
roots are exact eigenphases `exp(-i E dt)` **when the retained subspace is
invariant under the propagator**, for example when the supported distinct
energies are resolved. A finite, truncated subspace gives compressed roots,
which need not lie on the unit circle or equal exact phases.

Energy is determined only modulo `2π/dt`. Choose a declared enclosure [L,U],
set `E_shift=(L+U)/2`, and decode the rephased c values on the branch centered
there, restoring `E_shift` afterward. Require `(U-L)*dt < 2π`, strictly. A span
alone cannot fix aliasing under an additive identity shift. At fixed dt the
decoded energies lie in a bounded branch; large errors are not literally
unbounded. Both noisy pencils can violate the true ground-energy bound.

Relevant prior work includes Parrish and McMahon's
[quantum filter diagonalization (2019)](https://arxiv.org/abs/1909.08925),
Stair, Huang, and Evangelista's
[multireference quantum Krylov method (2020)](https://doi.org/10.1021/acs.jctc.9b01125),
and Klymko et al.'s
[VQPE and unitary formulation (2022)](https://doi.org/10.1103/PRXQuantum.3.020323).
These are related methods, not a claim that every formulation uses the same pencil.

## 3. Corrected exploratory pilot

Run from an installed research environment, with single-thread BLAS:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/probe_realtime_krylov.py --replicas 40 --seed 20260922
```

Instance: open-chain `tfim(4,J=1,h=1)`, explicit `|0000>` reference,
`dt=π/(E_max-E_min)=0.330084`, and energy shift at the spectral midpoint.
The model builder normally supplies `|++++>`; the pilot intentionally overrides
that reference. Errors are in **TFIM J=1 energy units**, not Hartree. The numeric
target is 1.6e-3; only the molecular instance uses chemical accuracy in Ha.

Each independent positive-lag real/imaginary component has Gaussian standard
deviation eps; negative lags reuse its conjugate. `c(0)` remains exact, `d(0)`
gets real noise, and both arms share each c draw. d has noise eps in energy
units. This is an absolute-error stress test, **not equal shot budgets**.
The relative truncation threshold is `max(eps,1e-13)*lambda_max(S)`, a heuristic
cutoff rather than a calibrated per-mode noise floor. Median uses all replicas;
p90 uses the empirical upper order statistic (`method="higher"`). A failed
eigensolve or vanished unitary root has infinite error, not a dropped NaN.

The initial version independently perturbed positive and negative lags in the
Hermitian arm and then averaged them, halving off-diagonal component variance;
the unitary arm did not do this. These paired results replace that comparison.

| m | cond(S) | retained rank | exact Hermitian error |
|---:|---:|---:|---:|
| 4 | 1.96e2 | 4 | 3.056e-1 |
| 6 | 5.75e3 | 6 | 2.700e-2 |
| 8 | 3.38e6 | 8 | 2.511e-4 |
| 10 | 5.32e9 | 10 | <1e-10 |

The direct-pencil residual is below 1e-13 in this run. This example couples
improved accuracy with poor conditioning; it is not a universal conditioning
law for real-time bases.

| eps, m | Hermitian median | Hermitian p90 | Unitary median | Unitary p90 | median retained rank |
|---|---:|---:|---:|---:|---:|
| 1e-5, 8 | 3.151e-3 | 6.698e-3 | 2.493e-3 | 4.801e-3 | 7 |
| 1e-5, 10 | 1.104e-3 | 2.569e-3 | 7.196e-4 | 1.197e-3 | 8 |
| 1e-4, 10 | 1.112e-2 | 1.539e-2 | 8.817e-3 | 2.482e0 | 7 |
| 1e-3, 10 | 2.273e-2 | 5.263e-2 | 8.023e-3 | 1.923e-2 | 7 |
| 1e-2, 10 | 1.720e-1 | 2.890e-1 | 1.261e-1 | 1.795e0 | 6 |

No solves failed in these rows. Printed last digits may depend on the numerical
environment (verified with Python 3.12.14 under both NumPy 2.3.5/SciPy 1.17.0
and the CI library pins NumPy 2.5.2/SciPy 1.18.0).
At eps=1e-5,m=10 both medians pass; the Hermitian p90 does not. At larger noise
the unitary tails can be much worse. Neither arm dominates across the grid.
Ridge or per-mode truncation may improve stability but can also introduce bias
or remove useful directions; improvement must be measured.

The reference has weight >1e-6 in **10 distinct energy eigenspaces**, grouping
degeneracies at atol=1e-10, rtol=1e-12. Counting individual eigenvectors gave 11
and depends on the arbitrary basis in a degenerate space. Exact Krylov rank is
bounded by the number of distinct energies with nonzero support (and phase
aliasing can lower it); thresholded support is only a diagnostic.

## 4. Cost model to freeze before execution

For a Hadamard-test outcome X in {-1,+1}, `Var(mean X)=(1-mu²)/N <= 1/N`.
A real or imaginary component of c therefore has worst-case standard error
at most `1/sqrt(N)`. The conservative allocation `ceil(1/eps²)` is sufficient
under that bound, not a measured requirement or a lower bound.

For `H=sum_l h_l P_l`, d(k) is a weighted sum of overlaps with `P_l U^k`.
Under independent term sampling and coefficient-proportional allocation, a
component has variance bounded by `Lambda²/N`, with `Lambda=sum_l |h_l|`.
Equivalently, a declared normalized LCU estimator for d/Lambda needs its own
circuit assumptions. Reaching absolute d error eps_d costs up to
`ceil(Lambda²/eps_d²)` in this model. Separate any identity term analytically
and retain its covariance with c. Do not price d as a single ±1 observable.

For the traceless TFIM pilot, Lambda=7 and the simple uniform-component bound
at m=10, eps_c=eps_d=1e-5 gives:

```text
unitary:    20 / eps_c²                             = 2.00e11 shots
Hermitian:  18 / eps_c² + 19 Lambda² / eps_d²       = 9.49e12 shots
```

These are illustrative allocations; they omit gate depth, state preparation,
and hardware throughput. They cannot establish savings against A-CASE. Its
350,196-word molecular example is a different instance, and Pauli word counts
are not directly comparable with these scalar counts.

The decision experiment must use a common total-shot grid and a declared
allocation for each arm. For A-CASE and power Krylov, perturb **shared underlying
Pauli estimands**, rebuild the pencil with its coefficients, and propagate QWC
covariance if grouped measurements are priced. Include adaptive construction
and selection costs, or freeze every basis using exact arithmetic and label
the entire comparison as fixed-basis estimation only. Exact-target stopping
cannot be used silently. The chosen contract, pools, stopping policy, budgets,
and covariance rules belong in the result-free config.

"Shots-to-target" means the smallest **tested** budget meeting the declared
error statistic. No interpolation or extrapolation beyond that grid. Record
the component standard errors as well as budgets: a single eps cannot describe
both bounded c and unnormalized d measurements. Treat an unpriced control or
a target beyond the tested budget range as censored, not an infinite saving.

## 5. Instances and reference matching

| ID | Hamiltonian and reference | Role |
|---|---|---|
| `tfim4_crit` | open-chain `tfim(4,J=1,h=1)`, `|0000>` | required decision instance |
| `tfim4_para` | open-chain `tfim(4,J=1,h=3)`, `|0000>` | gapped diagnostic |
| `h2_sto3g` | `chemistry.h2(bond_length=0.7414)`, RHF determinant | required decision instance; STO-3G, Angstrom geometry, JW mapping |
| `tfim6_crit` | open-chain `tfim(6,J=1,h=1)`, `|000000>` | confirmation rung only |

The H2 ladder uses a molecular builder, **not a committed FCIDUMP**. Freeze the
generated Hamiltonian/digest, nuclear-energy convention, chemistry dependency
versions, particle sector, and reference bit ordering before execution. Exact
truth must use the same accessible sector. H2 can have very small spectral
support; report it rather than treating a pass as broad chemistry evidence.

Every control must use the explicit reference in this table, overriding the
TFIM builder's default `|+...+>`. Report distinct-energy support, ground-space
weight, and the zero-noise basis error separately from noisy performance. A
larger gap does not guarantee success from this reference. A noisy failure on
`tfim4_para` is a result, not proof of an implementation bug.

## 6. Arms and numerical contracts

Five solver arms plus exact diagonalization:

| Arm | Basis and pencil | Eligibility |
|---|---|---|
| `exact_diag` | exact spectrum in the declared sector | ground truth |
| `rt_hermitian` | exact uniform real-time basis; c,d Toeplitz pencil | primary decision candidate |
| `rt_unitary` | same basis; S0,S1 from c | primary decision candidate with declared phase branch |
| `rt_trotter` | powers of one fixed `trotter2_unitary` step | propagation diagnostic |
| `acase_control` | `run_acase` with frozen pool and cost policy | matched incumbent |
| `power_krylov_control` | identity plus `krylov_response(H,m-1)` | diagnostic control |

For Trotter powers `V_k=U_T^k psi`, S is Toeplitz but `V†HV` is generally
**not**: `U_T` need not commute with H. Construct and price the full Hermitian
Hamiltonian pencil. Do not reuse the exact d(j-i) shortcut. Independently
approximating each total time with a fixed step count can lose even S's Toeplitz
structure. Declare the repeated-step rule and microstep counts in the config.
Report state fidelity `|<psi_exact|psi_trotter>|²`, not its unsquared amplitude.

For n=4, the proposed primary grid is `dt=π/(U-L)` with exact extremal bounds;
for n=6 use the Pauli coefficient bound about the identity shift. Pair this with
the midpoint branch in Section 2. The pilot's candidate sizes are {4,6,8,10};
the decision grid must be frozen separately. A secondary {0.5,1,2} multiplier
sweep is diagnostic only; multiplier 2 can put enclosure endpoints on an
aliasing boundary and must be marked inadmissible when the strict condition
fails. Never choose the primary grid after inspecting energies.

Cross Hermitian solver arms with hard truncation, per-mode noise floors, and
ridge only after freezing each cutoff, condition cap, and ridge coefficient.
`solve_projected` normalizes by generator norms and takes absolute per-mode
floors in that normalized spectrum. The pilot uses a relative cutoff; those
parameters cannot be substituted unchanged. A per-mode floor requires an
explicit covariance propagation or independent calibration, not just a call
to the solver. `solve_projected` does **not** solve the non-Hermitian unitary
pencil; that arm requires its own declared whitening/eigenvalue policy.

The proposed exploratory eps grid is {0,1e-6,1e-5,1e-4,1e-3,1e-2}, with 200
replicas per noisy cell and named independent seed streams. Couple shared c
draws between real-time arms, retain exact normalization, and reuse conjugate
lags. This stress grid complements, but cannot replace, the budget/covariance
experiment in Section 4. No finite-sample inference is licensed by Gaussian
draws. A classical Bernoulli simulator is possible without hardware; it still
needs a specified ancilla estimator and resource contract.

## 7. Metrics and failure handling

Record each replica's energy error, failure reason, retained rank, overlap
spectrum and conditioning; aggregate median, upper-order-statistic p90, target
success fraction, and failure fraction. Failed solves count as infinite error;
encode them as null plus an explicit failure flag in strict JSON. Do not remove
them with `nanmedian` or `nanpercentile`.

Also record modeled shots, allocations, c/d component errors, words and QWC
settings where applicable, basis-construction costs, propagation fidelity,
wall time, and peak RSS. Distinguish numerically retained rank from modes
resolved under a calibrated noise floor. Report p90 even though the proposed
primary target is median error; a median pass does not establish tail safety.

## 8. Proposed decision rule and precedence

The target is median absolute error <=1.6e-3 in each model's declared units
(Ha for H2; J=1 units for TFIM). Use paired candidates on the same instance and
cost contract. A **qualifying comparison** reaches that target and has modeled
shot ratio `N_acase/N_rt >= 10`, inclusive, on the tested budget grid.

First, failed deterministic checks (independent pencil identity, phase branch,
reference/sector consistency, or required data/provenance) give **INVALID**,
not NO-GO. Ordinary noisy solve failures remain in the statistics.

Then assign each required instance exactly one status, in this order:

1. **PASS:** an exact-propagation real-time arm with a non-ridge regularizer has
   a qualifying comparison at c-component noise >=1e-5.
2. **MARGINAL:** no PASS, but a qualifying comparison exists at c noise >=1e-6,
   including a ridge-only comparison.
3. **UNDETERMINED:** neither above, and a required cost comparison is censored
   or unpriceable within the grid.
4. **FAIL:** otherwise; all comparisons are complete but no qualifying candidate
   exists at c noise >=1e-6.

Combine the statuses for `tfim4_crit` and `h2_sto3g`:

| Required-instance outcomes | Verdict | Next work |
|---|---|---|
| PASS and PASS | GO | prototype the estimator/backend under the declared model |
| FAIL and FAIL | NO-GO | stop backend work for this design |
| any other combination | CONDITIONAL | at most the prespecified refinement below |

This precedence makes the 10x boundary, ridge-only outcomes, mixed-instance
results, and missing comparisons disjoint. The Trotter, gapped, and n=6 arms
are diagnostics; they cannot turn a failed primary comparison into GO.
Freeze this rule and table in the config and test all boundaries before
execution. No verdict is computed from the exploratory pilot in Section 3.

## 9. Implementation and preregistration order

| File | Status | Purpose |
|---|---|---|
| `benchmarks/probe_realtime_krylov.py` | present | exploratory numerical probe |
| `tests/test_realtime_krylov_probe.py` | present | independent pencil, variance, phase, support, and failure regressions |
| `benchmarks/configs/phase16b_feasibility.json` | planned | all inputs, estimator contracts, grids, seeds, decision rule; no results |
| `benchmarks/check_phase16b_preregistration.py` | planned | validate completeness, absence of results, and commit ordering |
| `benchmarks/run_phase16b_feasibility.py` | planned | execute the frozen experiment |
| `benchmarks/check_phase16b_feasibility.py` | planned | verify records and decision-rule evaluation |

Commit config plus preregistration checker first and pass it before running or
committing the decision experiment. This document alone does not satisfy that
gate. Freeze the Section 4 comparison contract and all unresolved parameters in
Sections 5-6 rather than selecting them while writing the producer.

Reuse dense `expm`, `solve_projected` for Hermitian pencils, `trotter2_unitary`,
`krylov_response`, `run_acase`, and `stamp_record` as appropriate. Six-qubit TFIM
can use dense propagation too. Any matrix-free implementation must use a
**full-space** Pauli action with the adjoint required by `expm_multiply`; TFIM
does not conserve particle number and cannot use a fixed-particle sector.
`SectorStatevectorBackend` is appropriate only for a verified invariant sector.

The future schema `clifford_qc.phase16b_feasibility.v1` must include the config
digest, environment, evidence labels, complete failures, modeled cost details,
and the decision evaluation. No phase-completion claim or status-ledger update
is justified by this pilot.

## 10. Prespecified follow-up

Before running the primary experiment, specify whether a NO-GO or CONDITIONAL
result permits one estimator-variance refinement. Publish the primary verdict
unchanged alongside any separately labeled refinement; never move thresholds
after observing results.

For bounded outcomes, variance `(1-mu²)/N` is **largest near mu=0**. If an
overlap component decays toward zero, its cost approaches the worst-case bound;
it does not become cheaper. Components near ±1 can require fewer shots. Real
and imaginary components need separate treatment, and d retains Hamiltonian
coefficient weights. Such refinement could change an allocation estimate but
is not guaranteed to rescue the method. No second follow-up or backend follows
automatically from an inconclusive result.
