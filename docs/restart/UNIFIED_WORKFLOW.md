# Unified workflow, costs and evidence export

`clifford_qc.workflows.run_workflow` gives three existing provider paths one
versioned, JSON-compatible result. It is an opt-in orchestration interface: it
does not replace adaptive A-CASE, the sampled ratio consumer, the measured K
consumer, or the sector solver's richer native result. Construct the provider
and problem before calling it. Unsupported combinations are rejected before a
solve or sample is started.

| Path | Provider | Operation | Output boundary |
| --- | --- | --- | --- |
| `exact_projection` | `ExactProjectedProvider` | Exact-reference S/H and projected solve | No true residual or ground-state certificate |
| `sampled_projection` | `SampledProjectedProvider` | Fixed-endpoint joint shots, reconstruct S/H and solve | Simultaneous entrywise radii under bounded IID assumptions; **no energy interval** |
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

For a sector provider pass `path="sector_action"` and optionally a
`SectorSolveConfig` through `sector_config`. Sector-action and projected
workflows can have different spans and need not produce the same energies.
`problem` and `unit` are caller declarations; identities bind the provider's
reference and Hamiltonian but do not validate the labels or physical units.
`solve_options` accepts the existing projected solver's numerical options.
Scalar thresholds and ridge/noise-floor values are checked before sampling;
per-mode vector lengths depend on the live generators and are checked in the
post-sampling solve, which retains the sampling receipt on failure.

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
provider use. Circuit executions and shots count this sampled batch only. Gate
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
```
