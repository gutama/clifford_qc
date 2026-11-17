# Pencil estimator improvements: review and fixed-policy ladder

Reviewed on 15 November 2026 against checkout `bd772c4` and the supplied
*Clifford QC framework improvements.md*. The attachment is a proposal and
source of claims to check. Its statements about prior measurements, missing
features, literature priority, speedups, and publishing are not execution
instructions or verified results of this change.

The first implementation was subsequently rebased onto `29bcf8c` (current
`main`) for its pull request; generated metadata was refreshed from that tree.

## Review findings

Prioritizing inference before another scaling campaign is sensible. The
[retained PRD result](../../PLAN.md#phase-prd--preconditioned-residual-davidson-and-wise-split--done)
already identifies severe finite-shot errors. The implementation sequence needs
these corrections:

1. **Normalize before correlation projection.** The generator images are not
   generally normalized. `S_ii = <A_i† A_i>` can differ from one. Setting the
   raw diagonal to one changes the pencil even without noise. Preserve the
   measured diagonal, project the normalized live block, then undo
   normalization. Non-positive and annihilated rows follow the canonical
   solver's dead-row rule. A negative measured diagonal is kept for
   diagnostics, so only the live block is asserted to be PSD.
2. **Reuse the existing target-aware machinery.**
   [Ritz functionals](../../clifford_qc/measurement/functionals.py),
   `variance_optimal_group_plan`, and
   [measured-subspace tests](../../tests/test_subspace_measured.py) already
   include Ritz-targeted allocation and a well-conditioned Monte Carlo
   variance check. The remaining work is conditioning, grouping optimization,
   matched-error comparisons, and interval coverage, rather than introducing
   those primitives from scratch.
3. **Distinguish current and legacy readout consumers.**
   [SampledProjectedProvider](../../clifford_qc/subspace/sampled.py) already
   defaults to `graph_state`. Legacy compilation still exposes elimination.
   A default migration needs consumer-specific compatibility checks and
   new evidence identities; the logical gate-count result alone does not
   establish a hardware bias or shot-cost improvement.
4. **Review current root recovery before replacing Davidson.**
   [Sector actions and recovery](SECTOR_ACTIONS.md) already address the
   excited-singlet regression. An internal residual cannot establish that a
   converged root is the ground root. The older failure remains evidence;
   it is not an accurate description of every current solver path.
5. **Rotation needs an energy convention and an inverse map.** For
   dimensionless `h = (H - E_shift*S)/E_scale`, rotation gives
   `h_theta = h*cos(theta) - S*sin(theta)` and
   `S_theta = S*cos(theta) + h*sin(theta)`. If its Ritz value is `mu`, then
   `lambda = (mu*cos(theta) + sin(theta))/(cos(theta) - mu*sin(theta))`,
   followed by restoration of scale and shift. Poles, indefinite rotated
   metrics, root ordering, and coefficient normalization need explicit
   handling. An angle chosen from noisy batches is part of the estimator;
   its selection must be repeated in a bootstrap or frozen independently.
6. **Matrix archives do not support joint-shot uncertainty replay.** The
   [v3 producer](../../benchmarks/run_prd_case_finite_shot.py) stores `S_hat`
   and `H_hat`, but no setting histograms. Those matrices support point-estimate
   replay. They do not reconstruct the within-setting covariance required by
   a constrained-ratio estimator or joint bootstrap. The seed-level records
   themselves are external artifacts, as [REPRODUCING.md](../../REPRODUCING.md)
   explains; they are absent from this checkout.
7. **Keep the scientific claims conditional.** PSD overlap repair alone
   cannot restore a variational bound for noisy `H`. A correlated reference
   can still be classically tractable, particularly a low-bond-dimension MPS
   or Gaussian state. Published speedups and statements that no prior work
   exists need separate validation before becoming package claims.

The primary sources checked for the leading recommendation were
[De Vriendt et al., arXiv:2602.11985v1](https://arxiv.org/abs/2602.11985v1)
and [Fischer et al., arXiv:2510.25640v1](https://arxiv.org/abs/2510.25640v1).
They support investigating rotated regularization and covariance-aware ratio
optimization. They do not validate transferring published performance or
coverage guarantees to this shared-bank implementation. This review does not
constitute a full audit of the attachment's literature or service pricing.

## Implemented first step

[repair_overlap](../../clifford_qc/subspace/pencil.py) implements Higham's
alternating projections with Dykstra's correction, extended to complex
Hermitian matrices, plus a certified rank-one shortcut. For
`D = diag(sqrt(S_ii))` on live rows, it minimizes the
Frobenius distance to `C = D^-1 S D^-1` subject to PSD and unit diagonal,
then restores `D`. Thus its objective is the **normalized** Frobenius distance,
not the raw Frobenius distance or a covariance-weighted likelihood. The
measured diagonal is preserved; its statistical uncertainty is not repaired.

`normalize_overlap` supplies the live set and scaling for the numerical solver,
repair, and shot calibration. Inactive rows are decoupled while retaining their
measured diagonals, including negative values. Nonfinite or non-Hermitian input
and nonconvergence raise explicit errors. A final PSD projection and diagonal
congruence remove the numerical
feasibility defect without adding a ridge. Diagnostics report iterations,
normalized minimum eigenvalues, and correction sizes. No new dependency or
top-level public export is introduced.

Some severely misnormalized matrices have a rank-one optimum for which
alternating projections converge slowly. A candidate `X = zz†`, `|z_i| = 1`,
is accepted only if its dual slack satisfies the KKT stationarity, positivity,
and complementarity conditions within tolerance and scale-aware roundoff.
These are sufficient for the convex nearest-correlation objective: for any
other feasible `Y`, the objective difference is
`0.5*||Y-X||_F² + Tr(ZY) >= 0` when `ZX = 0`.
The shortcut introduces no new positive-norm cutoff. Other inputs still use
the bounded alternating-projection iteration and retain explicit failures.

Use the opt-in measured path:

```python
from clifford_qc.subspace.uncertainty import bootstrap_ritz

solver_kwargs = {"overlap_repair": "nearest_correlation", "tau_s": 1e-8}
result = shared.solve(cache, **solver_kwargs)
interval = bootstrap_ritz(shared, cache, result=result, replicates=200, seed=17)
```

This reuses the cache without acquiring shots. Each bootstrap draw resamples
whole group histograms and reruns the repair and solve. The point result stores
the full measured solver policy and a joint-outcome fingerprint. Bootstrap
infers that policy, rejects conflicting kwargs or different outcomes, and
returns the point energy as its estimate. Legacy calls without `result` still
bootstrap their explicitly supplied/default policy; they are not bound to a
separate point result. A failed replica
raises instead of being silently discarded. The interval remains heuristic;
neither the example cutoff nor the nominal confidence level is a calibration
claim. The unrepaired delta method rejects repaired results because it omits
the repair's derivative. Low-level numerical results without measured-estimator
provenance are rejected by both point-bound uncertainty paths, including results
assembled manually from `repair_overlap` and `solve_projected`. Combining repair
with shot-calibrated radii also
fails explicitly until that combination has a validated noise model.

With repair enabled, legacy measurement diagnostics retain the measured
normalized spectrum's minimum, maximum, and negative-mode count. Explicit
`measured_overlap_eigenvalues` and `repaired_overlap_eigenvalues` resources and
`repaired_overlap_*` metrics keep the two spectra distinct.
`SubspaceResult.overlap_eigenvalues` continues to describe the solved metric.

The default numerical policy and the PRD sampling policy are unchanged. The
PRD producer gains a shared paired-real matrix decoder next to its encoder;
no historical record is rewritten.

The historical noise experiments retain their exact execution-time
`subspace/linalg.py` bytes in their existing provenance snapshot directories.
Cross-domain recorded replay keeps every originally declared source and adds
new package modules to its current replay manifest. New-campaign execution
still requires the complete source closure to be declared in advance. These
changes preserve published observation bytes and distinguish current replay
provenance from the original experiment's provenance. The recorded-value
verifier accepts additional sources only after checking their raw Git bytes at
the replay's execution commit, lists them in its maintenance receipt and still
requires every originally recorded path and the original scientific comparisons.

## Fixed-angle rotation

The next opt-in step is implemented by
[`solve_rotated`](../../clifford_qc/subspace/pencil.py) and the measured solver:

```python
point = shared.solve(
    cache, overlap_repair="nearest_correlation",
    pencil_angle=0.1, energy_shift=-1.0, energy_scale=2.0, tau_s=1e-8,
)
interval = bootstrap_ritz(shared, cache, result=point, replicates=200, seed=17)
```

The controls above illustrate the API, not recommended settings. Choose the
angle, shift, scale and thresholds independently of evaluation outcomes.
Angles are radians in `(-pi/2, pi/2)`; `energy_scale` is positive and uses the
same energy units as `H` and `energy_shift`. An angle selected on these data
is a different estimator and is not implemented here. An exploratory angle
sweep does not become independent validation by freezing its winner afterward.

Repair runs first. Original dead rows are removed using `normalize_overlap`;
both live matrices use that same generator scaling. The dimensionless pencil
is then rotated as in review finding 5. The canonical solver normalizes the
**rotated** metric's diagonal and thresholds its spectrum. Thresholding need
not retain the same span as the unrotated solver. The inverse fractional map
restores physical energies; all roots are sorted by those energies and their
coefficients are normalized in the supplied unrotated metric (the repaired
metric when repair is requested). Original dead rows remain exactly zero.
Zero angle delegates directly to the original solver, including its exact
floating-point output, even with nontrivial shift/scale controls.

This first rotation policy is deliberately bounded: the rotated live metric
must be PSD to scale-aware roundoff and have positive, non-dead diagonals.
An indefinite metric, inverse-map pole, nonpositive physical-norm branch or
nonfinite pencil causes an explicit failure. Roots are not silently removed.
`pole_tolerance` is a relative cancellation/branch guard on
`cos(theta) - mu*sin(theta)`, not a statistical radius. Calibrated overlap
floors/ridges are unsupported after rotation until a transformed covariance
model is validated. These restrictions can produce failures on noisy data;
they do not promise that any fixed angle is suitable for a campaign.

The measured result binds all four rotation controls to the joint-outcome
fingerprint and solver policy. Bootstrap reconstructs each joint histogram,
reruns repair and the same fixed rotation, and fails if any replica fails.
The delta-method API rejects rotation because its current functional does not
differentiate the rotated retention rule. All intervals remain heuristic.

Measured, repaired and rotated overlap spectra have separate resources.
`SubspaceResult.overlap_eigenvalues`, retained-mode thresholds, condition
number and smallest retained eigenvalue describe the **rotated normalized
solver metric**. Legacy scalar measurement-spectrum diagnostics continue to
describe the measurement. The physical energies and coefficients have already
been mapped back; `pencil_rotation` records the dimensionless rotated values,
inverse denominators and fixed policy for audit.

## Covariance-aware constrained ratio

The third opt-in step is
[`solve_constrained_ratio`](../../clifford_qc/subspace/pencil.py), reached
through the measured solver:

```python
point = shared.solve(cache, ratio_error_budget=0.005)
interval = bootstrap_ritz(shared, cache, result=point, replicates=200, seed=17)
```

The budget above illustrates the API; it is not a recommended setting.

It implements the constrained program of
[Fischer et al., arXiv:2510.25640v1](https://arxiv.org/abs/2510.25640v1),
Eq. 3: minimize `E(c) = c* H c / c* S c` subject to the ratio estimator's
propagated standard error `sigma(c) <= ratio_error_budget`, in the energy units
of `H`. `S` is never inverted and nothing is thresholded. `sigma(c)` is their
Eq. 5, the first-order ratio variance
`(Var N - 2 E Cov(N, D) + E^2 Var D) / D^2` with `N = c* H c` and `D = c* S c`.

**Covariance.** The variance is evaluated from
[`SharedMeasurement.entry_covariance`](../../clifford_qc/measurement/session.py):
the joint covariance of every measured `S` and `H` coordinate. Coordinates are
upper-triangle real and imaginary parts, in
[`hermitian_parameter_layout`](../../clifford_qc/subspace/pencil.py) order.
[`covariance_matrix`](../../clifford_qc/measurement/functionals.py) builds it
from the cache's joint histograms in one pass. It has the same pooling weights,
readouts and per-group population moments as `WordFunctional.covariance`, so
entries share settings and are correlated exactly as the measurement makes
them. The paper's sample moments use `S - 1`; these use `N`, matching the
existing delta method.

**Where it departs from the paper.**

- **Budget.** The budget is absolute and must be fixed before seeing the
  evaluation data. The paper sets it as a multiple of the measured reference
  energy, a data-dependent rule that is not implemented.
- **Denominator gates.** `c* S c` must be positive, and its coefficient of
  variation must be at most `ratio_max_denominator_cv` (default `0.1`). The
  first-order ratio variance is meaningful only while the denominator is well
  resolved. The paper states no such condition.
- **Starts.** The first start is the first live generator, which
  for an identity first generator is the paper's `c = (1, 0, ...)` and the
  direct reference-energy estimate. The other is the lowest root of the
  canonical thresholded solve. The reference can be an exact stationary point,
  as a Brillouin-decoupled Hartree-Fock reference is for single excitations;
  SLSQP never leaves such a point. The two pure starts can both be stationary
  while the ground-root start violates the error budget. Four additional
  starts mix the reference equally with the normalized component of the
  canonical root orthogonal to it, in the solver's scaled coordinates, using
  relative phases `+1`, `-1`, `+i`, and `-i`. They are omitted when that
  component is zero to roundoff. This gives the search nonstationary starts in
  the diagonal two-direction regression, whose constrained minimum lies
  between two stationary pure directions. The policy is recorded as
  `reference_canonical_phase_mixtures_v1` and each run is reported.
- **Selection.** The lowest feasible accepted endpoint is returned. If the
  reference itself is feasible, the result is never above it. This bounded
  deterministic multistart search does not certify a global minimum.
- **Solver.** SLSQP, with analytic gradients checked against finite
  differences, under a unit-norm, fixed-phase gauge. The paper used COBYLA.
  This needs SciPy (the `research` extra).
- **Coefficients.** Coefficients are complex, because the pencils are complex
  Hermitian; the paper's are real.

**Failures and stopping.** `ratio_tolerance` is SLSQP's absolute objective
precision in units of the budget. A line-search stop (SLSQP status 8) is
accepted only at a feasible endpoint. The objective gradient there must be a
nonnegative combination of the active constraint gradients, to a relative NNLS
residual of `1e-4`. A run that fails otherwise is recorded under `runs`. The
call raises if no declared start reaches an accepted feasible endpoint. The
single-live-generator shortcut enforces the same error-budget and denominator
gates and raises when its sole possible ratio is infeasible, including when
all other generators were dropped as dead. Dead rows stay exactly zero.

**Uncertainty.** The resource `constrained_ratio.standard_error` is
`sigma(c)` at the selected coefficients. It conditions on a choice made from
the same data, so it is a diagnostic, not an interval. The fixed-coefficient
delta method (`ritz_uncertainty`) rejects these results. The bound bootstrap
reconstructs each joint histogram, re-estimates the covariance, reruns all
searches and fails if any replica fails. Thresholds, repair, rotation and
calibrated radii do not apply and are rejected in combination.

**Not in the matrix-only replay.** The PRD archives store `S_hat` and `H_hat`
but no setting histograms (review finding 6), so they cannot supply the
covariance this estimator needs. A replay requires a campaign that retains
joint histograms.

**Exploratory development check.** This is not a record and not
preregistered. It uses the four-qubit TFIM bank of the measured-subspace tests:
nine generators, with the identity, six Pauli words and two Krylov powers. The
exact subspace has `kappa_S = 195` and sits `0.0113` above the true ground
energy. The check ran 40 seeds (`500`–`539`) at 2000 shots per group:

| Estimator | Median abs. error | Max abs. error | Below exact subspace | Below true ground |
| --- | ---: | ---: | ---: | ---: |
| Thresholded solve (`tau_s = 1e-10`) | 0.0050 | 3.1057 | 60% | 20% |
| Constrained ratio, budget 0.005 | 0.0055 | 0.0182 | 87.5% | 7.5% |
| Constrained ratio, budget 0.01 | 0.0157 | 0.0403 | 100% | 72.5% |
| Constrained ratio, budget 0.02 | 0.0347 | 0.0827 | 100% | 92.5% |
| Constrained ratio, budget 0.05 | 0.0802 | 0.2030 | 100% | 95% |

The table uses the fixed multistart policy described above. Errors are relative
to the exact subspace energy. Minimizing a noisy ratio is
biased downward: the search exploits shot noise up to about the budget. The
budget is therefore a bias–tail control, not a precision target. Here it
removed the thresholded solve's catastrophic replica. Above the subspace's own
error, it also pushed most estimates below the true ground energy. The paper
reports the same trade. One bank is not evidence of a general improvement.
Reproduce with:

```python
import numpy as np
from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (MatrixElementBank, SharedMeasurement, identity_generator,
                                  krylov_response, pauli_orbit)

model = tfim(4, J=1.0, h=1.0)
rho = ExactMVBackend().state(model.reference, ())
words = [op.word for op in odd_y_filter(local_pool(4))]
bank = MatrixElementBank(rho, model.hamiltonian, [identity_generator(4)]
                         + pauli_orbit(words[:6]) + krylov_response(model.hamiltonian, 2))
shared, exact = SharedMeasurement(bank), bank.solve().ground_energy
caches = [shared.measure(FiniteShotBackend(seed=500 + s), 2000) for s in range(40)]
for budget in (None, 0.005, 0.01, 0.02, 0.05):
    kwargs = {} if budget is None else {"ratio_error_budget": budget}
    errors = np.array([shared.solve(c, **kwargs).ground_energy - exact for c in caches])
    print(budget, np.median(abs(errors)), abs(errors).max(), (errors < 0).mean())
```

## Interval coverage

[`interval_replica` and `summarize_coverage`](../../clifford_qc/subspace/coverage.py)
measure how often an interval contains a known target over repeated seeded
measurements. They evaluate the delta-method interval and two grouped-bootstrap
constructions, percentile and basic, from one set of draws
(`bootstrap_ritz_samples`). A failed solve or failed interval stays in the
all-replica denominator. Rates carry Wilson intervals and are split by the
smallest retained overlap eigenvalue.

The [declared study](INTERVAL_COVERAGE.md) froze the systems, shots, estimator
arms, seeds, counting rules and labels before any evaluation seed was drawn. Its
producer and checker are `benchmarks/run_interval_coverage.py` and
`benchmarks/check_interval_coverage.py`. Its findings bear on this ladder in
three ways:

- **Repair.** With the default `tau_s`, nearest-correlation repair did not
  control the catastrophic tail. On the ill-conditioned development bank at
  2,000 shots its worst error was 144.2, against 50.3 for the unrepaired solve.
- **Constrained ratio.** Where its budget was attainable it removed that tail
  (worst error 0.0156), and its basic-bootstrap interval was close to nominal.
  Its percentile interval was not. The absolute budgets chosen on the
  development bank were unattainable on the held-out bank. There the search's
  outcome on a given row also differs between machines, so only its failure
  rate is a reproducible result.
- **Intervals.** The basic bootstrap is the better default construction for
  the thresholded solves: it was never labelled `under` for the default or the
  repaired solve.

## Target-aware acquisition

[`acquire_ritz`](../../clifford_qc/subspace/acquisition.py) measures a uniform
pilot of every setting, solves, and spends each further round by Neyman
allocation on the current Ritz functional `c*(H - E S)c`: shots in proportion
to the square root of each setting's per-shot variance. `allocation="uniform"`
is the comparator on the same schedule. Only estimators the delta method
accepts are supported, since the allocation target and the interval are both
its Ritz functional.

```python
from clifford_qc.subspace.acquisition import acquire_ritz

groups = len(shared.groups)
found = acquire_ritz(shared, backend, pilot_shots_per_group=100,
                     round_budgets=[100 * groups, 200 * groups, 400 * groups],
                     target_std_error=0.02)
print(found.stopped, found.shots, found.interval.lower, found.interval.upper)
for look in found.looks:
    print(look["shots"], look["energy"], look["std_error"])
```

**Stopping semantics.** The pilot size, the round budgets and the target are
fixed before any shot. The rule "stop when the estimated standard error is at
most the target" is evaluated only at those predeclared looks.
`found.interval` splits its error budget over every look at which the run
could have stopped, so it is valid over the schedule to the extent the
Gaussian delta method is valid at each look. It is asymptotic, not
anytime-valid and not a certificate. `found.naive_interval` ignores the other
looks. Without a target every round is spent and the two coincide.

That condition failed in the declared study wherever the solve's truncation
changed between looks. A solve that drops a direction reports a small standard
error and a biased energy, and the rule can stop on it. Treat a stopped
interval from a truncating solve as unvalidated, and check
`look["retained_rank"]` across the ledger before relying on one.

**Accounting.** `found.shots` counts every physical shot, pilot included, and
`found.looks` is the ledger. A look that cannot be solved raises
`AcquisitionError`, which carries the cache and the ledger so the shots
already spent are not lost. All data are pooled and each round's allocation
depends on earlier outcomes, so the estimate carries the bias any adaptive
design has.

**Reproducible rounding.** The shot counts decide what is sampled next, so one
shot of rounding changes every later outcome. Settings related by a symmetry
have weights that are equal in exact arithmetic, and a largest-remainder rule
ordered by raw floats then depends on the eigensolver's last bits.
`neyman_group_plan` treats remainders within `1e-6` of a shot as tied and
breaks ties by group order. It is otherwise the rule of
`variance_optimal_group_plan`, which is unchanged.

The [declared study](TARGET_ACQUISITION.md) froze the systems, schedule,
stopping targets, seeds, estimands and labels before any evaluation seed was
drawn. Its producer and checker are `benchmarks/run_target_acquisition.py` and
`benchmarks/check_target_acquisition.py`. Its findings bear on this ladder in
three ways:

- **Allocation.** At matched error targeting was never dearer than uniform
  allocation and was 1.4 to 2.7 times cheaper in five of eight
  system-and-target cells. The two cells of the held-out `xxz4_mixed` were
  unresolved.
- **Stopping.** The rule never stopped on a catastrophic solve: the worst error
  of any stopped run was 0.121, where fixed-look errors reached 130,887. But
  the stopped interval under-covered on the held-out six-qubit bank (83.2% to
  93.0% at a nominal 95%), where runs stopped on solves that had truncated a
  direction. Splitting the error budget over the looks does not cover that
  bias.
- **Truncation is the common cause.** On `xxz4_mixed` the default solve
  retained six directions at every look where the exact rank is five, and
  neither allocation removed the resulting outliers. Allocation cannot repair
  a direction the solve should not have retained.

## Matrix-only replay

With the external v3 seed files available, run from the repository root:

```bash
python -m benchmarks.replay_pencil_estimators \
    --input-dir /path/to/prd-seed-records \
    --output work/pencil-replay.json
```

The tool compares the original measured matrices with and without repair at
the **same fixed numerical thresholds**. It does not reproduce the archived
shot-calibrated policies. It verifies campaign and frozen-basis digests, task
identity, and physical-shot accounting; rejects duplicate cells and
non-Hermitian input before estimator accounting; hashes source bytes and the
actually imported implementation files, including both PRD producer modules;
and marks incomplete campaigns. The direct script entry point also works.
Existing output files cannot be overwritten.

To add fixed-angle controls on the same matrices, use:

```bash
python -m benchmarks.replay_pencil_estimators \
    --input-dir /path/to/prd-seed-records \
    --output work/pencil-rotation-replay.json \
    --pencil-angle 0.1 --energy-shift -1.0 --energy-scale 2.0
```

The v2 report retains the two unrotated arms and adds
`rotated_fixed_threshold` and `rotated_nearest_correlation`. It records the
caller-supplied fixed rotation policy; reference energies enter only the
assessment after solving. No angle is optimized or selected by this tool.
Rotation-only failures count in their own arm's all-cell denominator.
Omitting `--pencil-angle` retains the original two-arm computation. Supplying
nondefault energy or pole controls without an angle is rejected.

Reports include median and 90th-percentile absolute errors, violation rates,
retained rank, conditioning, and repair diagnostics. The minimum retained
overlap eigenvalue comes directly from the solver's actual per-mode/ridged
retention decision. Failed solves remain in
the census and count against the all-cell chemical-accuracy rate; conditional
statistics state their successful-solve denominator. Ground and exact packet
energies are read only for assessment, never supplied to the estimator.
`interval_coverage` is explicitly absent (`null`). A replay is exploratory
and does not supersede the original negative campaign.

## Validation and remaining sequence

The new tests check analytic two-direction complex optima, an independently
derived three-direction optimum, exact Gram fixed points, covariance under
generator rescaling and rephasing, inactive rows, nonconvergence, shot
accounting, joint bootstrap execution, and replay provenance/failure handling.
Review regressions add shot-noisy annihilating generators, certified nearly-dead
live-row projections, point-policy and joint-outcome binding, rejection of
unknown-provenance uncertainty inputs, measured/repaired diagnostics,
non-Hermitian record rejection, actual per-mode/ridged retention, imported
producer hashes, direct script invocation, and NumPy integer controls.
Rotation tests compare complex full-rank results with an independent QR-span
Ritz oracle, check physical Rayleigh quotients after truncation, energy-unit
and generator-coordinate covariance, zero-angle identity, dead rows, invalid
metrics, poles, policy-bound joint bootstrap, truth-independent replay arms,
and failure denominators. Constrained-ratio tests show several identities.
The grouped covariance matrix equals every pairwise functional covariance,
under single assignment and pooling. At a Ritz vector, the propagated ratio
variance equals the existing Ritz-functional variance, and the reference
direction reduces to the direct standard error. Over 48 seeds per bank the
propagated standard error is within 25% of the Monte Carlo spread at fixed
coefficients, on a well-conditioned and an ill-conditioned bank. Further tests
cover negligible covariance recovering the generalized ground root, budget and
gate feasibility, a stationary reference, certified line-search stops, explicit
failures, dead rows, policy-bound bootstrap and rejected combinations. Review
regressions cover infeasible single-live-generator budgets and denominator CVs,
including dead-row padding and an identity-only measured bank, plus the
analytic constrained minimum between stationary two-direction starts at three
error budgets. They
establish implementation identities on their fixtures, not empirical interval
coverage or matched-error improvement.
Acquisition tests check that every declared shot is spent and counted, that a
stopping rule only truncates the seeded full schedule, that the stopped
interval is widened for every look, and that a failed look keeps its cost. A
targeted schedule is unchanged by last-bit jitter of the Ritz vector, while the
raw-float rounding rule is not.
Existing measured-subspace, response, PRD, and compatibility tests are also
run. The tests make no empirical coverage or matched-error claim; those come
only from the two declared studies.

| Step | Status | Acceptance before a scientific claim |
| --- | --- | --- |
| Diagonal-preserving PSD repair and point replay | Implemented, opt-in | Source-verified external 180-cell replay; assess tails and failures |
| Fixed-angle rotation and estimator policy | Implemented, opt-in | Independent-angle evaluation and external equal-data replay; data-selected angle inference remains pending |
| Covariance-aware constrained ratio | Implemented, opt-in, measured path only | Budget fixed before evaluation; a histogram-retaining campaign with tails, failures and below-ground rates; coverage of the rerun-search bootstrap |
| Interval coverage instrument and declared study | Implemented and [run as declared](INTERVAL_COVERAGE.md) on three four-qubit banks | Molecular banks and realistic noise need their own declared study; a budget rule for the constrained ratio that transfers between systems |
| Target-aware acquisition | Implemented, opt-in, and [run as declared](TARGET_ACQUISITION.md) on four spin banks | A stopping rule that stays valid when the solve's truncation changes between looks; molecular banks and realistic noise need their own declared study |
| Symmetry pool and SoA kernel work | Pending | Invariant tests, real structured-bank timings, full selection-cache costs at matched error |
| Hardware bridge, community instances, distribution | Pending | End-to-end local adapter checks, strong classical baselines, reproducible packaging/docs builds |

Before running a new confirmatory campaign, freeze its estimators, thresholds,
selection/validation split, histogram format, seeds, failure policy, and
reporting targets. Keep its outputs separate from historical records.
