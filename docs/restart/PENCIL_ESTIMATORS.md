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
provenance from the original experiment's provenance.

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
and failure denominators. They establish implementation identities on their
fixtures, not empirical interval coverage or matched-error improvement.
Existing measured-subspace, response, PRD, and compatibility tests are also
run. There is no new empirical coverage or matched-error improvement claim.

| Step | Status | Acceptance before a scientific claim |
| --- | --- | --- |
| Diagonal-preserving PSD repair and point replay | Implemented, opt-in | Source-verified external 180-cell replay; assess tails and failures |
| Fixed-angle rotation and estimator policy | Implemented, opt-in | Independent-angle evaluation and external equal-data replay; data-selected angle inference remains pending |
| Covariance-aware constrained ratio | Pending | Retain joint histograms; denominator and feasibility gates; validate full propagated covariance |
| Coverage and target-aware acquisition | Pending | Frozen 68%/95% study, all failure counts, minimum-retained-overlap strata, valid stopping semantics |
| Symmetry pool and SoA kernel work | Pending | Invariant tests, real structured-bank timings, full selection-cache costs at matched error |
| Hardware bridge, community instances, distribution | Pending | End-to-end local adapter checks, strong classical baselines, reproducible packaging/docs builds |

Before running a new confirmatory campaign, freeze its estimators, thresholds,
selection/validation split, histogram format, seeds, failure policy, and
reporting targets. Keep its outputs separate from historical records.
