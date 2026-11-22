# Unified workflow, costs and evidence export

`clifford_qc.workflows.run_workflow` gives four provider paths one
versioned, JSON-compatible result. It is an opt-in orchestration interface: it
does not replace adaptive A-CASE, the sampled ratio consumer, the measured K
consumer, or the sector solver's richer native result. Construct the provider
and problem before calling it. Unsupported combinations are rejected before a
solve or sample is started.

| Path | Provider | Operation | Output boundary |
| --- | --- | --- | --- |
| `exact_projection` | `ExactProjectedProvider` | Exact-reference S/H and projected solve | No true residual or ground-state certificate |
| `sampled_projection` | `SampledProjectedProvider` | Fixed-endpoint joint shots, reconstruct S/H and solve | Simultaneous entrywise radii under bounded IID assumptions; **no energy interval** |
| `sampled_precision` | `SampledProjectedProvider` | Predeclared pilot and adaptive physical-shot rounds, rank-aware Ritz acquisition | Heuristic rank guard, asymptotic schedule-adjusted energy interval, optional heuristic bootstrap width |
| `sector_action` | `SectorActionProvider` | Existing bounded sector solver (SciPy) | Cached-action residuals and attempts; numerical convergence does not identify the ground root |

For example:

```python
from clifford_qc.multivector import MV
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.subspace.providers import ExactProjectedProvider
from clifford_qc.workflows import run_workflow, export_workflow_evidence, verify_workflow_evidence

h = MV.from_terms(2, {"ZI": 1.0, "IX": 0.4})
gens = [("I", MV.scalar(2, 1)), ("X", MV.from_terms(2, {"IX": 1}))]
provider = ExactProjectedProvider(DeterminantReference("00"), h, gens)
result = run_workflow(provider, path="exact_projection", problem="toy", unit="energy units")
print(result.as_dict()["outcome"])
receipt = export_workflow_evidence(result, "/your/persistent/storage/toy.zip")
verify_workflow_evidence(receipt["path"])
```

For a sampled provider pass `path="sampled_projection", shots=10000, seed=17`;
an integer requests that many shots **per setting**. Alternatively pass one
count per setting. The result keeps shot and circuit counts, the seed, setting
and observation identities, joint histograms and signed readouts. The provider's
measurement plan and preparation are not serialized automatically; include
those inputs as attachments when they are needed for later reconstruction.
The sampled result uses the same observations to reconstruct and solve the
pencil. Its entrywise confidence regions are not a confidence interval for the
selected Ritz energy. Reconstruct/solve `ValueError` and linear-algebra failures
after a successful sample retain spent shots and histograms, with an empty
energy list and a failed status. Invalid requests and failures before a valid
sample raise; they do not generate an automatic failure receipt.

### Rank-aware density-free precision acquisition

For a declared independent bank and a reference exposing only `sample_joint`,
the primary controller is also available directly on the sampled provider:

```python
from clifford_qc.subspace import PrecisionAcquisitionConfig
from clifford_qc.subspace.sampled import SampledProjectedProvider
from clifford_qc.workflows import run_workflow

# reference is a PreparedSampledReference or compatible joint-shot reference.
# h and gens are the Hamiltonian and independent generator bank.
provider = SampledProjectedProvider(reference, h, gens, block_size=2)
settings = len(provider.plan.settings)
config = PrecisionAcquisitionConfig(
    pilot_shots_per_group=200,
    round_budgets=(200 * settings, 500 * settings),
    target_std_error=0.02,
)
result = run_workflow(provider, path="sampled_precision", problem="LiH",
                      unit="Ha", seed=20261122, precision_config=config,
                      solve_options={"tau_s": 0.005})
print(result.as_dict()["outcome"]["status"])
# Native result is also available from provider.acquire_precision(config, seed=20261122).
```

The pilot measures every compiled setting; each later budget is a **total
physical-shot count**, not a per-word or per-setting count. The existing
`acquire_ritz_precision` controller operates on one accumulated joint cache,
with covariance-aware Neyman weights keyed by the actual compiled settings
(including non-QWC settings). Each look draws a distinct deterministic random
stream derived from the run seed. Every batch must match reference and circuit
identities, requested seed, assignments, signed readouts and physical counts.
No reference density, exact expectation, exact projected pencil, or spectral
oracle is used for inference, allocation or bootstrap. Simulated preparation
may use a statevector at the device boundary, not in the consumer.

By default, full declared rank with positive cutoff margins must persist for
two consecutive looks **and** the estimated SE must meet its target. If rank
is unresolved, targeted rounds buy overlap-sensitive shots. For tighter
uncertainty, predeclare `max_bootstrap_half_width` and `bootstrap_replicates`
in the config; without replicates there are no bootstrap solves. Replicates
alone request a terminal diagnostic. With a width guard, eligible looks must
also pass the bootstrap width; an exhausted schedule may still receive a
terminal bootstrap diagnostic. Both guards are heuristic, the delta interval
is asymptotic with a predeclared-look multiplicity adjustment, and bootstrap
resampling conditions on the observed budgets. Neither is an anytime-valid
confidence sequence, finite-sample certificate, or chemical accuracy claim.

The raw Ritz estimator is supported with fixed numerical overlap cutoffs
(`tau_s`, `rel_tau`, `max_condition`, `norm_floor`). Sector control variate
certification currently requires exact-reference information and is **not**
silently enabled on this path. The frozen molecular studies with corrected
estimators remain distinct, as does the independent-pilot finite-sample ratio
consumer. The fixed-endpoint path is unchanged.

`target_met` means the declared guards passed; `schedule_exhausted` returns a
diagnostic estimate, not successful precision attainment; `failed` returns
no promoted energy. Invalid requests before sampling raise. Once a request
is issued, every validated batch and completed look is retained if a later
solve or sampling request fails. A failed/invalid device response is recorded
as `unconfirmed`: its requested shots are retained but its actual physical
cost is unknown, not inferred to be zero. The measurement ledger reports
confirmed physical shots, circuit batch-setting counts, whether those costs
are complete, and actual measured/replica solve attempt counts. The archive
carries per-look requested counts, seeds, joint histograms and signed readouts
for replay; attach preparation, operator bank and circuit definitions
separately.

The focused integration tests compare the same compiled observations with a
separately wired replay of the common controller, exercise genuinely commuting
non-QWC settings, reject invalid sampling receipts, and verify rank exhaustion,
bootstrap cost and evidence export. An opt-in ideal joint-shot integration
smoke using already declared molecular inputs is available separately from CI:

```bash
OPENBLAS_NUM_THREADS=1 python examples/sampled_precision_molecules.py
OPENBLAS_NUM_THREADS=1 python examples/sampled_precision_molecules.py --stretched
```

These are single-seed transport checks with a loose 20 mHa estimated-SE target,
not new coverage, cost-advantage or precision-stop validation campaigns.
The integration smoke (seed 20261122, block size 2) produced:

| Bank | Compiled settings | Retained rank by look | Confirmed shots | Outcome |
| --- | ---: | --- | ---: | --- |
| H4 (8 qubits) | 356 | 3, 3 | 142,400 | `target_met` |
| LiH (8 qubits) | 278 | 3, 3 | 111,200 | `target_met` |
| Stretched LiH (10 qubits) | 394 | 4, 4, 4 | 70,920 | `schedule_exhausted` |
| Stretched BeH2 (12 qubits) | 555 | 3, 4, 4 | 99,900 | `schedule_exhausted` |

Both stretched cases had a final SE below 20 mHa but their statistical
cutoff margins remained negative; the rank guard correctly refused a
precision stop even after BeH2 recovered full numerical rank. This is a
functional integration observation, not a comparative performance claim.

For a sector provider pass `path="sector_action"` and optionally a
`SectorSolveConfig` through `sector_config`. Sector-action and projected
workflows can have different spans and need not produce the same energies.
`problem` and `unit` are caller declarations; identities bind the provider's
reference and Hamiltonian but do not validate the labels or physical units.
`solve_options` accepts the existing projected solver's numerical options.
Scalar thresholds, ridge/noise-floor values and impossible vector lengths are
checked before sampling. The exact per-mode vector length depends on the live
generators and is checked in the post-sampling solve, which retains the sampling
receipt on failure.

## Cost ledger

`clifford_qc.workflow_cost.v1` separates preparation, sampling, solve,
validation and export. The wrapper times only its sample and solve calls; the
provider was already constructed. A caller may report either outside stage as
`preparation_cost={"wall_seconds": 1.2, "scope": "FCIDUMP read and provider setup"}`
or the corresponding `validation_cost`. These remain labelled
`caller_reported` and are not independently measured. No total is calculated:
export is outside the run, stages could overlap, and process peak RSS and
hardware time are not measured by this wrapper. Provider-specific resource
snapshots are retained with their native scopes; some counters may reflect prior
provider use. Circuit batch-setting counts and shots refer to this fixed batch
or the confirmed acquisition batches only. Gate
counts, depth, device time and whole-process RSS remain unknown unless a
separate assessment supplies them. Never infer them from a settings count.

The unified record adds `exact_numerical_simulation` to the evidence vocabulary
for numerical exact-reference or explicit-action calculations without implying
that an arbitrary run was a small-sector simulation or a mathematical proof.
Floating-point and model/sector limitations still apply. The result retains
its outcome status, evidence scope and unassessed ground-state status.

## Portable export and verification

`export_workflow_evidence(result, destination, attachments={"input.FCIDUMP": path})`
atomically publishes a no-overwrite ZIP containing strict JSON result bytes, a
manifest and exact copies of explicitly supplied attachments. Its return value
contains the archive's SHA-256 and size; keep that digest separately when
independent integrity checking is needed. `verify_workflow_evidence` checks
membership, byte sizes and digests without extracting the archive. The combined uncompressed result and attachment payload is bounded by
`max_bytes` (64 MiB by default); the manifest and ZIP overhead are outside that
limit. Attach input data,
declaration, source snapshot, independent checker outputs and environment files
as relevant; the API does not discover the full source or dependency closure.
The result includes a SHA-256 fingerprint of the installed package's Python
source bytes at result creation, but does not copy those bytes into the archive
or bind model inputs and external dependencies automatically.

The destination and its retention, backup, access and eventual publication are
the caller's responsibility. A ZIP written to temporary storage is not durable
storage. Hashes detect accidental or unsynchronized tampering; a party that can
replace both archive and manifest can recompute them. They do not authenticate a
publisher, establish preregistration, independently verify scientific results,
or guarantee reproducibility. Existing campaign records, strict replays,
scientific claims and historical acceptance rules are unchanged.

Run the focused checks with:

```bash
python -m unittest discover -s tests -p test_unified_workflow.py -v
python -m pytest -q tests/test_sampled_precision.py
```
