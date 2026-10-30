# Shared runner and artifact contract

This implements the shared execution/artifact and CI-maintenance work from
Change 5 of the restart blueprint. The existing campaign producers, declarations,
observations, packaged sources and independent checkers retain their formats and
bytes. The first production consumer is CI: the shared runner invokes the
existing checkers and records their outcomes. Full-seed checks redraw only
frozen streams for verification; they never publish new scientific outcomes.

## Versioned boundary

`benchmarks.experiment_runner.Task` declares a unique key, an argv tuple,
positive wall-time limit, bounded log size and optional artifact paths. `run`
freezes the ordered schedule, declared source SHA-256s, transport source hashes,
Git commit, Python/library versions and thread settings before launching a
command. Source and environment changes stop further execution. Callers must
declare the complete input/source closure; CI binds every tracked input byte.
Commands execute without a shell, in the declared repository root.

The JSON CLI input uses `clifford_qc.execution_plan.v1`:

```json
{
  "schema": "clifford_qc.execution_plan.v1",
  "sources": ["benchmarks/check_sampled_ratio.py"],
  "tasks": [{
    "key": "sampled_evidence",
    "argv": ["python3", "-m", "benchmarks.check_sampled_ratio"],
    "timeout_seconds": 600,
    "max_output_bytes": 1048576,
    "artifacts": []
  }]
}
```

Use the actual interpreter path when reproducibility requires a fixed executable.
The example is a minimal transport illustration, not a complete campaign source
closure. Run from the repository root:

```bash
python -m benchmarks.experiment_runner --spec plan.json --output .ci-receipts/example
python -m benchmarks.experiment_runner --spec plan.json --output .ci-receipts/example --resume
```

`clifford_qc.execution_receipt.v1` contains `plan`, `plan_sha256`, the ordered
`rows` and a summary over every scheduled task. Terminal states are `passed`,
`failed`, `timed_out`, `output_limited` and `interrupted`. A failing command does
not suppress later commands. A required missing or oversized artifact fails
its task. The CLI exits nonzero if any scheduled task fails.

Elapsed time, return code and observed log bytes describe transport work only.
`costs_complete` refers only to this execution accounting; it never establishes
shot, preparation, allocation, inference, device or total scientific costs.
Receipts explicitly carry `scientific_evidence_claim: false`. Campaign-specific
preregistration chronology, physical success denominators, resource admission,
cost models and independent numerical validation remain separate requirements.

## Artifact and interruption invariants

`ArtifactStore` identifies exact payload bytes by SHA-256 and byte length under
`objects/<digest-prefix>/<digest>`. Reading or deduplicating verifies both.
Observed artifacts are copied byte for byte: existing JSON, gzip and scientific
records are never parsed and rewritten as an incidental transfer. Artifact
collection does not prove that the command created an existing file.

New metadata uses sorted strict UTF-8 JSON with a trailing newline; duplicate
keys, nonfinite values and overflow are rejected. Optional gzip uses `mtime=0`;
compressed-byte identity is scoped to the recorded compression runtime.
Same-directory atomic publication rejects existing final paths. Only the live
checkpoint is replaced; the final receipt and content-addressed objects are
immutable. Relative artifact/source paths reject traversal and symlink escapes.

Each object has a 16 MiB limit, each task defaults to a 1 MiB log prefix and
declares at most 64 collected artifacts. Log bytes and timeout are monitored;
polling can observe a transient excess before termination. This is not an
OS-enforced disk, memory or CPU quota. These limits bound retained payloads,
not arbitrary resource use inside a checker. Timeout termination kills the
process group, including descendant work. The current locking/termination
contract requires a POSIX host, as used by CI.

A kernel advisory lock prevents simultaneous writers and releases on process
death. Explicit resume requires identical plan, declared inputs, transport and
environment. Completed tasks are verified and never repeated. A task recorded
as started without a terminal receipt becomes an interrupted failure; available
partial logs are retained and its unobserved costs remain unknown. Only tasks
that never started may then execute. A completed run directory cannot be
overwritten; another maintenance execution requires a separate directory.
This local transport identity is not a global once-only experiment authorization.

## CI registry and routing

`benchmarks/ci_gates.json` is the single list of checker argv, record-local
environment, timeout and affected paths. `benchmarks.ci_gates` validates it,
plans matrices and executes selected commands through the shared runner.
Routing follows transitive local Python imports as well as explicit data and
declaration paths; deleting a source selects full replay.

| Event/change | Required work |
| --- | --- |
| Every PR | Full pytest suite, lint and core evidence/document/manuscript gates |
| Known affected campaign | Its structural checks and, for sampled evidence, its frozen replica replay |
| Shared production code, archive bindings or unknown non-document source | All registered campaigns; missing ancestry also fails closed to full replay |
| Transport/workflow edits | All deterministic gates; unchanged frozen sampled sources remain under their historical checker and replay contracts |
| Main push | All deterministic gates plus sampled replays selected by the actual change |
| Weekly schedule or explicit dispatch | Every registered structural/sampled gate, both restart full-seed replays and global environment consistency |

Full registered archival replay is scheduled for Sunday 09:17 Asia/Jakarta.
It checks existing evidence, never launches a new `--execute` campaign.
Record-local dependency pins, full Git history and disabled matrix fail-fast
are retained. An optional `execution_declaration` registry entry selects the
exact runtime directly from a frozen `execution_environment` block for full-seed
replay. In particular, molecular replay uses its declared Python 3.12.14,
NumPy 2.3.5 and SciPy 1.17.0 rather than the ordinary checker's newer stamp.
The environment action verifies the selected runtime and thread settings.
Receipts and verified logs are uploaded for 14 days, including
failed gates. The census distinguishes every-PR gates, affected-path gates and
full-replay gates; gates historically absent from CI remain explicitly visible.

```bash
python -m benchmarks.ci_gates --plan --event workflow_dispatch
python -m benchmarks.ci_gates --gate check_sampled_ratio --output .ci-receipts/sampled
python -m pytest tests/test_experiment_runner.py tests/test_ci_gates.py
```

Acceptance covers corrupted objects/metadata, atomic no-overwrite publication,
missing artifacts, child-process timeout, log limits, independent task failures,
crash/resume and input drift, deleted/renamed evidence, unknown dependencies,
missing ancestry and preservation of the complete pre-extraction CI command,
environment and timeout inventory. This infrastructure adds no scientific
benefit claim and does not reopen either negative measurement hypothesis.
