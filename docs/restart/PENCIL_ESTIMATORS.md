# Pencil estimator improvements: review and first implementation

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
   measured nonnegative diagonal, project the normalized live block, then
   undo normalization. A measured negative norm needs an explicit failure or
   a separately justified model; it cannot be preserved by a PSD repair.
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
Hermitian matrices. For `D = diag(sqrt(S_ii))` on live rows, it minimizes the
Frobenius distance to `C = D^-1 S D^-1` subject to PSD and unit diagonal,
then restores `D`. Thus its objective is the **normalized** Frobenius distance,
not the raw Frobenius distance or a covariance-weighted likelihood. The
measured diagonal is preserved; its statistical uncertainty is not repaired.

The same norm floor as the canonical solver identifies inactive rows. Those
rows are decoupled while retaining their nonnegative diagonals. Nonfinite or
non-Hermitian input, negative diagonals, and nonconvergence raise explicit
errors. A final PSD projection and diagonal congruence remove the numerical
feasibility defect without adding a ridge. Diagnostics report iterations,
normalized minimum eigenvalues, and correction sizes. No new dependency or
top-level public export is introduced.

Use the opt-in measured path:

```python
from clifford_qc.subspace.uncertainty import bootstrap_ritz

solver_kwargs = {"overlap_repair": "nearest_correlation", "tau_s": 1e-8}
result = shared.solve(cache, **solver_kwargs)
interval = bootstrap_ritz(shared, cache, replicates=200, seed=17,
                          solver_kwargs=solver_kwargs)
```

This reuses the cache without acquiring shots. Each bootstrap draw resamples
whole group histograms and reruns the repair and solve. A failed replica
raises instead of being silently discarded. The interval remains heuristic;
neither the example cutoff nor the nominal confidence level is a calibration
claim. The unrepaired delta method rejects repaired results because it omits
the repair's derivative. Combining repair with shot-calibrated radii also
fails explicitly until that combination has a validated noise model.

The default measured solver and the frozen PRD producer are unchanged.

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
identity, and physical-shot accounting; rejects duplicate cells; hashes
source bytes and implementation files; and marks incomplete campaigns.
Existing output files cannot be overwritten.

Reports include median and 90th-percentile absolute errors, violation rates,
retained rank, conditioning, and repair diagnostics. Failed solves remain in
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
Existing measured-subspace, response, PRD, and compatibility tests are also
run. There is no new empirical coverage or matched-error improvement claim.

| Step | Status | Acceptance before a scientific claim |
| --- | --- | --- |
| Diagonal-preserving PSD repair and point replay | Implemented, opt-in | Source-verified external 180-cell replay; assess tails and failures |
| Rotation and estimator protocol | Next | Explicit shift/scale, inverse and pole checks; independent-angle evaluation; equal-data controls |
| Covariance-aware constrained ratio | Pending | Retain joint histograms; denominator and feasibility gates; validate full propagated covariance |
| Coverage and target-aware acquisition | Pending | Frozen 68%/95% study, all failure counts, minimum-retained-overlap strata, valid stopping semantics |
| Symmetry pool and SoA kernel work | Pending | Invariant tests, real structured-bank timings, full selection-cache costs at matched error |
| Hardware bridge, community instances, distribution | Pending | End-to-end local adapter checks, strong classical baselines, reproducible packaging/docs builds |

Before running a new confirmatory campaign, freeze its estimators, thresholds,
selection/validation split, histogram format, seeds, failure policy, and
reporting targets. Keep its outputs separate from historical records.
