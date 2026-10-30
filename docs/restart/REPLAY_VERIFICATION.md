# Recorded numerical replay and execution fingerprints

This is a **separate software-verification contract introduced after observing
cross-host fingerprint drift**. It is not a preregistered scientific result,
does not change a historical acceptance rule and does not claim byte-identical
unrecorded plans. The original producer `--check` commands remain unchanged.

The full 2,304-outcome spin replay passes its original strict comparison on the
reference runtime. A GitHub host with the same Python 3.12.14 / NumPy 2.5.2
versions reproduces the first pilot-observation identity but changes learned
coefficients by a few ulps. The frozen plan fingerprint includes coefficient,
retained-frame and functional bytes, so those numerical changes produce
a different SHA-256. A controlled older OpenBLAS profile produces the same
first-pilot fingerprint on both hosts; the archived fingerprint comes from a
different profile.

The diagnostic is available in [run 41035139161](https://github.com/gutama/clifford_qc/actions/runs/41035139161).
Runtime dispatch is documented by [OpenBLAS](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)
and [NumPy](https://numpy.org/doc/stable/user/troubleshooting-importerror.html).
Matching package versions and thread counts alone does not establish exact
floating-point byte portability. The old strict criterion's failure on that
host is preserved, not reinterpreted as a strict pass.

## The CI contract

`benchmarks/replay_verification.py` verifies each fixed, published record against
its first-publication Git blob and invokes its independent scientific/history
checker before drawing any replay outcomes. It then calls the original
producer's `main --check`, retaining its environment, source, chronology,
worker, seed, resource and sampling preflights.

At the comparison boundary only, the verifier makes a deep copy of the replay
record and isolates each row's `plan_identity` field. Both fingerprints must be valid
SHA-256 strings, and every difference is retained with its record path and
both exact values.

The molecular workers also retain an exact `source_manifest_sha256`, which
changes when a declared source such as the shared metadata helper changes.
The original comparator already treats the source map and execution commit as
execution provenance. This maintenance mode separately reports worker-manifest
drift only after validating each manifest against its own complete source map,
the same source-path set across executions, and every source digest against raw
Git blob bytes at that execution's named commit. A forged manifest, missing
source or digest mismatch fails verification.

It delegates comparison of every other field to the
**original comparator**: the existing numerical tolerances, exact allocation
counts, statuses, pilot/observation identities, shapes and existing advisory
timing/provenance rules remain authoritative. A differing observation identity,
shot count or numerical value outside the original tolerance still fails CI.
No tolerance is widened and no original dictionary or file is rewritten.

This verifies all **recorded scientific fields**. It does not verify byte
equality of unrecorded coefficients or claim universal equivalence of plans.
The original strict same-platform replay remains the control for that scope.

The separate maintenance receipt uses
`clifford_qc.recorded_numerical_replay.v1`. It reports the immutable observation
digest and publication commit, number of outcomes, all plan-fingerprint
differences, validated execution-provenance differences, and
`original_strict_replay_passed`. A numerical verification can
pass while that last field remains false. CI labels this mode **recorded
numerical replay** and retains the receipt as a content-addressed artifact.
The producer's generic replay-success sentence is suppressed in this mode so
it cannot be confused with a strict-plan success claim.

The scientific observation records, declaration/execution snapshots, original
comparators, original strict replay command and negative cost-benefit verdicts
remain unchanged. The old all-input 20% promotion rule is not relaxed or
reopened by a maintenance verification mode.
