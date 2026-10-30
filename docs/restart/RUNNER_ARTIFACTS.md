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
command. Source drift fails the causing task, names changed paths, and blocks remaining work while publishing a complete receipt. Callers must
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
`failed`, `timed_out`, `output_limited`, `interrupted` and `blocked`. Ordinary command failures do
not suppress later commands; changed inputs or an exhausted job budget block
unstarted commands explicitly in the failure denominator. A required missing or oversized artifact fails
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

The registry declares a per-gate log allowance: 16 MiB for deterministic checks
and 64 MiB for sampled replays. The standalone Task default remains 1 MiB.
A run admits objects up to its declared maximum log size (at least 16 MiB),
and each task declares at most 64 collected artifacts. Log bytes and timeout are monitored;
polling can observe a transient excess before termination. This is not an
OS-enforced disk, memory or CPU quota. These limits bound retained payloads,
not arbitrary resource use inside a checker. Timeout termination kills the
process group, including descendant work. Main-thread SIGTERM/SIGINT handling
also performs this cleanup, persists an interrupted row and retains partial logs.
Uncatchable OS termination can only leave the last checkpoint. The current locking/termination
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
All deterministic gates retain their every-PR safety net. Replay selection
includes transitive imports, environment records and explicit live data/declaration
bindings. Test, example and document edits do not trigger sampled work.
Legacy sampled instruments remain manual-only, including R3d as its frozen
reporting contract requires. A bundled, digest-bound pre-extraction workflow
fixture verifies inventory compatibility even in shallow clones and source archives.

| Event/change | Required work |
| --- | --- |
| Every PR and main push | Full pytest/lint, all deterministic gates and global environment consistency |
| Affected restart source/data or runner/CI edits | Both applicable restart full-seed verification paths; runtime setup is exercised before merge |
| Legacy sampled source/data changes | Report affected manual-only replay names; never silently run the frozen instrument automatically |
| Weekly schedule | All deterministic gates and the restart full-seed verifications |
| Explicit dispatch | All registered gates, including legacy manual-only sampled instruments |

Weekly verification is scheduled for Sunday 09:17 Asia/Jakarta. It never
launches a new `--execute` campaign. Each main push has a unique concurrency
identity, so later pushes cannot cancel or displace its replay obligations.
Only superseded PR runs may be cancelled.

Job budgets start before checkout/setup. The runner receives the actual elapsed
budget, limits remaining task time and reserves three minutes for finalisation
and artifact upload. Exhausted budgets create explicit blocked rows. Jobs also
have fifteen minutes of setup headroom, and upload steps use `always()` so
catchable cancellation can preserve interrupted checkpoints and logs.
Bulk input bytes are hashed once; later checks compare inode/size/mtime/ctime/mode
and rehash changed files only. This assumes an ordinary isolated checkout, not
an adversary able to forge filesystem metadata. Git/transport identities remain
checked between tasks.

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
