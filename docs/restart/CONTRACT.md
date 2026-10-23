# Restart v2: determinant reference and exact projection

Implementation contract, 24 October 2026. The [uploaded blueprint](blueprint_2026-10-24.md)
is preserved byte for byte. [The v1 snapshot](v1_evidence_snapshot.json) binds
the unchanged evidence at its reviewed commit, separately from the implementation
base containing the completed provenance-guard migration (PR #154).

## First supported workload and falsifiable claim

Prepare a restricted real FCIDUMP Hamiltonian in the existing interleaved
Jordan–Wigner convention, use its computational determinant reference, and
solve the existing rank-1/rank-2 correction span through the stable projected
Ritz solver. H4 STO-3G at 0.9 Angstrom is the integration regression, with
small additional determinants and complex operators as independent controls.
The numerical baseline is the same workflow with the expanded density-MV
reference, identical candidate budget, storage policy, and solver configuration.

The first deliverable removes the explicit `2**n` reference expansion. A
bounded-row large-register test must forbid density construction and Hilbert-space
allocation. It does not establish a better eigensolver, ground-state certification,
matched-accuracy speedup, or a measured quantum advantage. A-CASE remains the
compatibility consumer during this extraction.

## Conventions and arithmetic

Operators use `Cl(2n,C)` in the existing Pauli representation. Qubit zero is
the leftmost occupation bit and the lowest two-bit Pauli lane. Spin ordering
is explicit metadata; molecular inputs use interleaved alpha/beta orbitals.
Pauli phases belong to row coefficients. Expectations are bilinear, without
conjugating coefficients or applying Clifford reversion.

`DeterminantReference` offers two named arithmetic contracts:

* `analytic`: scan the supplied row in generation order and sum each diagonal
  coefficient times its determinant sign. This supports large registers and
  avoids density normalization and expansion. Equivalence with expanded-MV
  arithmetic is numerical, not universally bitwise.
* `legacy_trace`: reproduce the smaller-operand branch, coefficient multiply
  order, `2**(-n)` scaling, complex accumulation, and final `2**n` multiplication
  of the existing trace pairing. If the density would be the smaller operand,
  sort only present diagonal terms into its implicit generation order. This
  requires bookkeeping proportional to that diagonal support. Reject `n >= 40`:
  the legacy MV cutoff removes the density coefficients at that size.

The prepared determinant workflow defaults to `legacy_trace` for its bounded
compatibility scope. The user can explicitly select `analytic`; the run record
names the choice. `legacy_mv` retains the previous backend, bank, and arithmetic
path for historical consumers and capabilities not yet migrated.

Required small fixtures compare complex scalar and S/H bytes, not just floating
point equality (which overlooks signed zero). Analytic comparisons use explicit
absolute/relative tolerances and report their scope. No arithmetic contract or
frozen campaign outcome is changed retrospectively.

## Ownership, capabilities, and provider boundary

Reference metadata and identities are immutable and include qubit/spin ordering.
MV compatibility references snapshot ordered coefficients, including signed zeros.
The exact provider snapshots its Hamiltonian and every registered generator;
public cached inputs cannot be mutated. Cache identity includes reference,
Hamiltonian, ordering, arithmetic, and storage policy; generator identity includes
ordered coefficient bits. Changing any input creates a different provider.

`ReferenceState` declares metadata and capabilities, not exact expectation.
`ExactReference` is separate; sampled-only references are rejected before exact
projection starts. Sector-action and sampled reference/provider protocols define
future consumer boundaries without implementing those adapters now.
`ExactProjectedProvider` exposes numerical S/H blocks and Ritz solves. Its
explicit bank bridge lets the existing A-CASE selector consume those blocks;
measurement observations are a separate capability and are not fabricated.

The determinant path in this PR supports the existing operator-level selection
checks. Reference-conditioned sector certification and H-squared reporting still
require the explicitly selected legacy MV path until their action consumers land.
Unsupported configurations fail before solving; they never construct a hidden
density as a fallback.

## Cost boundary and acceptance

Report occupation-state payload, reference snapshots, row terms scanned,
diagonal terms, evaluation calls, compatibility sort sizes, and retained index
footprints separately from bank coefficient storage and whole-solve time.
Index footprints include referenced boxes, which can overlap the row store;
transient sorting workspaces and Python allocator effects remain outside them.
These counters do not measure RSS. Model preparation and independent validation
remain separate stages, and their costs must be included in any later end-to-end
comparison. Hamiltonians, generators, and coefficient rows can still grow rapidly.

| Acceptance | Required check |
| --- | --- |
| Determinant semantics | Independent bit-pattern expectations; non-prefix occupations and phases |
| Legacy arithmetic | Both smaller-operand branches, reordered emissions, cancellation, signed zeros, extreme finite scales |
| Packed rows | Object/packed and retain/stream match; scalar reads do not materialize packed rows |
| Bounded reference storage | Large-register bounded-row test forbids density/backend/dense allocation |
| Identity safety | Input snapshots resist mutation; cache keys change with every semantic input |
| Capability safety | Sampled-only exact calls and unmigrated reporting modes fail visibly |
| Molecular compatibility | Same S/H, energies, labels, stopping decisions on declared small fixtures |
| Accounting | Reference and index work priced separately; no unmeasured speed/RSS claim |

Publish the checked scope in [COMPATIBILITY.md](COMPATIBILITY.md) before moving
other workflows. Step 3 (sector consumer, owned-memory admission, root recovery,
spin diagnostics, action residuals) and Step 5 (actual sampled ratio estimation)
remain subsequent work. Existing negative campaign verdicts remain in force.
