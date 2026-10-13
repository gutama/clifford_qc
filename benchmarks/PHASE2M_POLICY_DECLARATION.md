# Phase 2M-C/D: process-isolated storage-policy matrix

This declaration precedes the new molecular observations. Existing molecular,
2M-A, 2M-B and stretched-H2O feasibility records motivated the design; their
results are not measurements from this matrix.

## Frozen design

`configs/bank_policy_matrix.json` freezes five committed FCIDUMPs: H4 at 0.9 A,
equilibrium and stretched BeH2, and equilibrium and stretched H2O. Source and
reference-record hashes, reference energies used only for the oracle stop,
complete rank-two generator counts, order and operator hashes are included.
There are 26, 204, 204, 140 and 140 candidates respectively; no pool is capped.

Every system has four arms, in the declared serial order:

| Arm | Coefficient storage | Lifetime policy |
| --- | --- | --- |
| object_retain | Python objects | retain_all |
| packed_retain | interleaved packed arrays | retain_all |
| object_stream | Python objects | stream_recompute |
| packed_stream | interleaved packed arrays | stream_recompute |

All arms use the same ACASEConfig, full candidate ordering, identity start,
maximum 30 additions (M <= 31), and the 1.5936 mHa oracle stopping target.
The streaming frontier is 32 pairs. Numerical libraries use one thread and
PYTHONHASHSEED=0. A new Python interpreter is started for every arm; another
arm's allocator does not share its heap. Model preparation and support
instrumentation within that worker are included in its reported process cost.

The RSS ceiling is 15 GiB, sampled every 0.05 s; the per-arm wall-clock limit is
six hours. Execution requires at least 15 GiB plus 512 MiB currently available
within both host and cgroup limits. An undersized host refuses before consuming
the campaign claim. Ceilings and deadlines do not shrink or retry an arm.
Completed cells are checkpointed in the consumed claim, and every declared cell
is attempted unless the orchestrator itself fails. Timeout, memory-limit and
worker failures remain visible and make the campaign INCOMPLETE.

## Equivalence and row-bound gates

Within each system, scientific traces must agree exactly across all four arms:
every projected S/H pencil in original IEEE representation, energy trajectory,
selected labels and indices, scored candidates and rejection reasons, growth
diagnostics, stopping reason, evidence tier and resource scope. Logical pair
history, cumulative coefficient occurrences, cumulative/selected word universes
and pair/word ownership must also agree. Resident rows, resident coefficients,
recomputations, RSS and time are outcomes rather than invariants.

Every observed streaming residency must stay within the retained block plus
32 frontier pairs. This bounds persistent S/H coefficient rows, not scalar or
support history, Hamiltonian actions, the word table or the reference. A product
can hold two additional transient rows. The record exposes that distinction.

The independent checker imports neither the producer nor worker. It derives
pair and word accounting, validates encodings and process/environment identities,
and compares every scientific trace. `--rebuild` additionally constructs each
system's final pencils by direct products in another fresh interpreter, without
a MatrixElementBank, and compares matrices, energy, labels and word count.
Completion reporting requires both observation checking and this direct rebuild.

## Cost boundary

Report preparation/solve time, first physical pair-build time and recomputation
time, postprocessing and parent wall time separately. The fresh worker observes
its own RSS, providing telemetry even when a command transport exposes a wrapper
PID. The parent enforces limits and the kernel high-water mark also rejects a
crossing missed by polling. RSS includes instrumentation and the entire worker;
no resident coefficient-storage attribution is inferred from it.

Packed live/reserved array bytes and the 24-byte-per-coefficient payload estimate
are separate fields. End-of-solve RSS delta per resident coefficient is a process
density, not a row allocator's exact byte count. Spill I/O is zero for these two
implemented policies. No combined multiplicative packing/eviction saving is
presumed, and disk-backed storage remains conditional on a measured need.

## Commit and execution protocol

Land this declaration, frozen configuration, implementation and toy tests before
any new molecular outcome. Execution requires a clean, non-shallow committed tree
and the pinned record environment. It checks input hashes and execution source
blob identities and exclusively claims the campaign in the common Git directory.
Failures never release that claim or overwrite an existing record.

Commit the complete `reference_results/bank_policy_matrix.json` alone, strictly
after its clean execution-source commit. Then run both checker modes, and only
after they pass update the Phase 2M ledger and the interpretation. A VALID record
means this declared equivalence/performance matrix passes; historical 2M-B
storage evidence is retained separately. Missing or failed cells do not complete
Phase 2M.

No new molecular energies or costs were evaluated for this declaration. The
development container's 8 GiB cgroup cap is below the execution requirement;
the molecular matrix remains pending on a suitable host.
