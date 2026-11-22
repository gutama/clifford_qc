# Current restart roadmap

For the complete documentation map, including current workflows, scientific
results and historical plans, see the [documentation guide](../README.md).

Start with the [unchanged 24 October blueprint](blueprint_2026-10-24.md),
[frozen v1 evidence bindings](v1_evidence_snapshot.json),
[first implementation contract](CONTRACT.md), and
[checked compatibility scope](COMPATIBILITY.md), and
[sector-action/root-recovery contract and acceptance](SECTOR_ACTIONS.md).
The [graph-readout and stabilizer-reference audit](GRAPH_READOUTS.md) supplies
additional compiler/reference support for the
[sampled consumer and frozen ratio experiment](SAMPLED_RATIO.md).
The separately declared [cross-domain molecular validation](CROSS_DOMAIN_RATIO.md)
extends the sampled workflow while preserving the original negative result.
The [shared runner and artifact contract](RUNNER_ARTIFACTS.md) now supplies
failure-preserving execution receipts, immutable byte-addressed artifacts and
one CI gate registry, with all deterministic checks on every PR, affected
spin replay and weekly or explicit full cross-domain verification. Legacy sampled instruments remain
manual-only under their frozen reporting contracts. Historical scientific runners and evidence remain unchanged.

The [pencil-estimator review and fixed-policy ladder](PENCIL_ESTIMATORS.md)
adds opt-in diagonal-preserving overlap repair, fixed-angle rotation with
physical-energy restoration and joint-bootstrap policy binding, a separate
matrix-only PRD replay tool, and a covariance-aware constrained-ratio estimator
that never inverts `S`. That estimator is biased downward by up to about its
error budget. The ladder carries no interval-coverage or accuracy-improvement claim.

The [declared interval-coverage study](INTERVAL_COVERAGE.md) tested those
estimators' delta-method and bootstrap intervals against the exact subspace
energy at 68% and 95%. No construction was calibrated for every estimator. The
basic bootstrap came closest on the thresholded solves, and the constrained
ratio's budgets did not transfer to the held-out system.

The [declared target-aware acquisition study](TARGET_ACQUISITION.md) compared
Ritz-targeted shot allocation with uniform allocation at matched error and
tested the interval reported by a predeclared stopping rule. Targeting was
never dearer and was 1.4 to 2.7 times cheaper in five of eight cells. The
stopped interval over-covered on the three four-qubit banks and under-covered
on the held-out six-qubit bank, where the rule stopped on solves that had
truncated a direction. A [bootstrap interval at the same stop](BOOTSTRAP_STOP.md)
re-solves every resample and so sees that truncation. On fresh seeds, under
criteria fixed before the run, it covered in 99% to 100% of the six-qubit runs,
at the cost of intervals up to seven times wider where truncation was frequent.

[Symmetry-shift control variates](SYMMETRY_SHIFTS.md) subtract certified
zero-mean sector terms from a measured Ritz value, after the fact or at every
look of an acquisition run. In exploratory runs on spin-adapted molecular banks
they cut mean-squared error 1.4 to 7 times at fixed shots. On a six-qubit bank,
combined with targeting, they reached a standard-error target with 9 to 13
times fewer shots than uniform allocation, and the stopped interval covered.
On H4 targeting did most of the work, 10.5 times fewer shots to the tighter
target, and the correction added a further 1.2 times. None of the declared
spin banks qualifies: their generators break the symmetry.

The [preregistered molecular interval coverage study](MOLECULAR_INTERVAL_COVERAGE.md)
uses the three-direction LiH and H4 banks at matched fixed QWC shot budgets.
The 256 paired replicas reject no primary 95% corrected-spread cell and
have 22–46% narrower median intervals and 18–50% lower point RMSE. The 68%
spread intervals under-cover descriptively; this is not an adaptive-stop or
general calibration guarantee.

The [preregistered adaptive molecular stopping study](MOLECULAR_ADAPTIVE_STOP.md)
tests the same two banks under six predeclared looks and a 1.6 mHa estimated-SE
target. All 256 acquisitions met the target; corrected schedule-adjusted
95% bootstrap-spread coverage is 62/64 on LiH and 61/64 on H4, neither rejecting
the primary endpoint. Mean shot cost is 16.4% and 34.7% lower with correction;
H4 RMSE increases by 12.4%, so this is not a matched-accuracy result.

The [declared stretched molecular rank stress](STRETCHED_MOLECULAR_STOP.md)
uses ten- and twelve-qubit frozen-core LiH/BeH2 banks with an engineered
near-cutoff mode. Rank changes in 55/128 acquisition trajectories; all runs
meet the 1.6 mHa estimated-SE target, but 39/128 exceed it in absolute error
(including all 24 reduced-rank stops). Corrected 95% schedule spread does
not reject either primary coverage endpoint, at a median width of 531.6 mHa
on LiH. Coverage is not a precision-stop validation in this regime.

The [rank-aware stopping comparison](RANK_AWARE_STOP.md) supports
full-rank confirmation and covariance-aware cutoff-margin acquisition, plus
a separate bootstrap half-width guard. On fresh seeds of the stressed banks,
rank/margin passed both feasibility criteria (31/32 true precision stops,
one false, versus 25/32 true and seven false for SE-only). It uses 77–87%
fewer shots than the fixed-budget arm. Width-only yielded 32/32 true stops
with greater shot and bootstrap cost. `acquire_ritz_precision` is now the
preferred fixed-bank precision entry point and defaults to rank/margin;
bootstrap width requires an explicit target and draw budget. The historical
`acquire_ritz` engine defaults and evidence labels are unchanged; these are
engineered stress banks, not general certificates.

The [unified workflow](UNIFIED_WORKFLOW.md) now exposes `sampled_precision` on
the coefficient-only sampled provider. It reuses the primary controller with
compiled joint-shot settings, cumulative receipts and explicit measured and
bootstrap costs. A single-seed integration smoke confirms normal stopping
on H4/LiH and honest exhaustion on the near-cutoff stretched LiH/BeH2 banks;
this is raw Ritz transport, not a new corrected-estimator coverage study.

[Weighted and noise-gated measurement planning](WEIGHTED_MEASUREMENT.md) adds
`plan_measurement` (defaulting to NumPy-only `method="graph_state"`),
coefficient- and Ritz-weighted $k$-commuting sorted insertion, a GALIC-style
per-setting `NoiseGatePolicy`, variance-first reporting (`R_hat`, optional
covariance-aware `R`, and 2Q gate cost across rungs), and opt-in weighted or
Ritz-replanned `SampledProjectedProvider` integration. Frozen historical
`compile_block_measurement_plan` / `synthesize_block_settings` elimination
defaults and default `SampledProjectedProvider` circuit identities are
unchanged.

The [preregistered noise-aware selection experiment](NOISE_AWARE_SELECTION.md)
closed negative on its equal-shot held-out criterion. An independent H4 audit
checks the published energy trajectory and Pauli support counts.

The separate [observable-noise investigation](OBSERVABLE_NOISE.md) adds sparse
Heisenberg diagnostics and an exploratory 224-outcome exact-channel study.
Protected diagonal observables and overlaps do not establish protected energies;
this study does not reopen the closed sampled cost hypotheses below.

| Step | Status | Deliverable |
| --- | --- | --- |
| 1 | Implemented | Frozen evidence identities and one determinant molecular problem/claim contract |
| 2 | Implemented | Reference capabilities, determinant/MV adapters, exact provider, density-free determinant pipeline |
| 3 | Implemented on declared scope | Immutable sector actions, fixed-capacity projection, admitted SciPy recovery, cached residuals, spin defects, separate owned/RSS scopes; primary excited-singlet regression |
| 4 | Implemented on bounded suite | H4, LiH and Hubbard 2×2 same-span S/H and Ritz compatibility, with buffer/admission checks |
| 5 | Implemented; spin and molecular experiments completed on declared scope | Sampled-only provider, streamed functionals and independent pilot/evaluation ratio; 2,304 original spin outcomes plus 2,304 separately declared TFIM/H4/LiH outcomes retained; neither all-input 20% cost hypothesis passed |
| 6 | Closed for the declared experiments | Both frozen all-input cost hypotheses closed negative; the TFIM control improvement does not establish molecular accuracy or a cross-domain benefit |

## Current capabilities

| Capability | Implemented? | Validation scope | Needs an exact oracle to operate? | Demonstrated advantage |
| --- | --- | --- | --- | --- |
| Determinant expectations | Yes | Bit patterns, complex rows, compatibility and large-register bounded rows | No density or spectral oracle | Removes explicit reference-density expansion by construction; no timing/RSS campaign |
| Exact projected provider | Yes | Complex 3-qubit controls; object/packed and retain/stream policies | Exact reference-evaluation capability; no spectrum oracle | Numerical interface and immutable cache inputs; no algorithmic advantage claim |
| Prepared determinant workflow | Yes | H4/LiH S/H, energies and selection/stopping compatibility | No ground-state oracle under default config | Reference representation changed; variational span unchanged |
| MV compatibility adapter | Yes | Owned immutable snapshot and legacy trace comparisons | Exact density supplied explicitly | Compatibility only |
| Sector actions and root recovery | Yes | Primary 2×4 excited-singlet recovery, spin-contamination controls, H4/LiH/Hubbard same-span checks | No spectral oracle during solving; independent assessment afterward | Finite numerical recovery; no ground-root guarantee or timing/RSS advantage claim |
| Graph-state readout compilation | Default in `plan_measurement` (opt-in on legacy replay helpers) | Exact signed Pauli conjugation, assigned joint distributions, no-Stim compilation/sampling, six frozen-bank partitions | No spectral oracle; exact simulator used only in validation and test sampling | Fewer logical two-qubit gates on five of six tested rungs; no matched-accuracy shot or hardware claim |
| Weighted and noise-gated measurement planning | Yes | No-Stim default `graph_state` planning, exact/cached `R_hat` and `R`, GALIC 2Q-gate and fidelity-floor bounds, H4/BeH2 rung diagnostics, density-free `SampledProjectedProvider` Ritz replanning | No exact or density oracle during grouping/replanning; exact reference optional for offline covariance diagnostics | Higher state-independent `R_hat` than unweighted LDF on tested H4 rungs and bounded 2Q gate budgets on BeH2; illustrative variance diagnostics, not a preregistered accuracy or hardware campaign |
| Signed stabilizer expectations | Yes | Pure full-rank canonical signed rows, all small Pauli words, projection compatibility and large-register allocation guard | Analytic stabilizer membership; no density or spectrum oracle | Avoids reference-density expansion; no general fermionic-sector guarantee |
| Sampled projection and correlated ratio estimation | Yes | Complex/packed controls; nonidentity overlap and finite-sample denominator gate; original 3 spin inputs and separately declared TFIM/H4/LiH validation, each at 3 budgets × 128 replicas per arm | No exact oracle in consumer/allocation; ideal statevector shots and separate independent physical-sector assessment | Both all-input 20% cost hypotheses closed negative; new TFIM-only illustrative saving of 73% does not transfer to declared molecular accuracy |
| Sampled precision acquisition | Yes | Identical-histogram controller replay for QWC and non-QWC plans; rank exhaustion and bootstrap/shot receipts; single-seed H4/LiH and stretched LiH/BeH2 smoke | No density or exact oracle in raw Ritz consumer | Primary rank-aware controller integrated; no new coverage or cost-advantage claim |

The supported first workflow is documented in the [molecular pipeline guide](../../molecular/PIPELINE.md).
Historical phase status remains in [PHASE_STATUS.json](../../PHASE_STATUS.json)
and [PLAN.md](../../PLAN.md). In particular, this interface work does not reopen
the packet line or change the failed frame-coordinate shortlist criterion.

Use the numerical API directly when no adaptive selector is needed:

```python
from clifford_qc.multivector import MV
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.providers import ExactProjectedProvider

reference = DeterminantReference("01", arithmetic="legacy_trace")
hamiltonian = MV.from_terms(2, {"ZI": 0.5, "IZ": -0.25, "XX": 0.1})
provider = ExactProjectedProvider(reference, hamiltonian, [identity_generator(2)])
result = provider.solve()
print(result.ground_energy, result.resources["reference_evaluation"])
```

This one-direction example evaluates a projected energy. It does not certify a
ground root or demonstrate a quantum resource benefit.
