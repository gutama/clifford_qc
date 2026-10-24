# Sector actions, bounded solving and root evidence

Step 3 adds a separate classical `solve-sector` workflow. Step 4 checks a
bounded old suite through the action projection. The determinant A-CASE
workflow from Steps 1–2 remains available with its existing correction span.
The [original blueprint](blueprint_2026-10-24.md) and
[v1 evidence bindings](v1_evidence_snapshot.json) remain unchanged.

## Supported problem and interface

Inputs are interleaved Jordan–Wigner Hamiltonians, explicit particle number N,
spin projection Sz and target total spin S. Qubit zero is the leftmost
occupation bit and lowest Pauli-code lane. `SectorStateReference` owns immutable,
normalized amplitudes and sorted occupation words. Its identity binds sector,
conventions and amplitude bytes. It supplies actions, rather than pretending to
be an exact Pauli-row expectation adapter.

`SectorActionProvider` owns its Hamiltonian snapshot, checks Hermiticity and
conservation of N, Sz and S², and applies vectors or blocks. Spin projection is
a finite polynomial in S² over the allowed spin values. No dense spin frame,
density MV or H² operator is constructed. Optional gather-pass precomputation
has separate conservative admission and retained-payload accounting. The
supported workflow defaults to recomputing passes to bound retained storage.

`BoundedActionSpace` allocates V/HV once at admitted capacity, checks
orthonormality, and updates only new S/H borders. The existing stable projected
solver receives numerical blocks. Cached H actions supply Ritz states and
residuals. Matrix and Ritz snapshots cannot alias later growth/restart; borrowed
live-prefix views are read-only. Storage still scales with the combinatorial
sector dimension and retained columns.

## Classical recovery and evidence

`solve_sector` wraps SciPy `eigsh`/ARPACK with explicit `ncv`, iteration limits
and deterministic independent starts. It records the initial Ritz block before
running every configured recovery attempt (at most 16). Starts combine a
lowest-diagonal occupation and seeded random amplitudes, projected into the
requested spin sector. Independent spectral information never supplies starts,
directions, candidate selection or stopping. A penalty places the complement
of the target-spin sector above the wanted spectrum; a coefficient norm bound
supplies the spectral shift. Very small sectors use an explicitly admitted
complete projection rather than an implicit library dense fallback.

Every H/S² action column is charged, including spin projection, recovery and
final diagnostics. Failed starts and iteration-limited attempts retain their
work and time; logs do not keep an unbounded iteration-vector history. Public
ARPACK does not expose its internal restart count: that field is null, configured
`ncv`/iteration limits are recorded, and all callback actions are counted.
Reported resets and orthogonal projections refer to this adapter's own work.
PRIMME is not a dependency or implemented adapter in this slice.

The result separates four kinds of evidence:

| Evidence | Meaning |
| --- | --- |
| Numerical convergence | `norm(HV c - E V c) / norm(V c)` from cached actions, with an explicit numerical-resolution convention |
| Quantum numbers | N/Sz defects and `norm((S²-S(S+1)) psi)`; an S² mean alone is insufficient |
| Tracking | Ray overlaps and block principal cosines against the previous best block, including unitary mixing in degenerate blocks |
| Ground state | Unresolved in production; an independent spectrum can assess the returned state afterward |

A converged excited singlet can pass the first three checks. The status
`numerically_converged_root_unresolved` preserves that distinction.
`accuracy_unresolved` means the requested residual accuracy was not established.
A memory breach stays `resource_limited`, even if an earlier candidate converged.
The direct-action resolution convention is 128 machine epsilons times the
normalized action/energy scale; it is neither a forward-error proof nor a
statistical bound. The shared convergence-report builder now accepts residual
records without a matrix-element/second-moment bank. Legacy second-moment
report arithmetic and serialized fields are unchanged. Spectral assessment
rejects a returned state that fails its quantum-number consistency checks.

## Admission and RSS boundary

`ActionMemoryPlan.admit` rejects the requested plan before solver-array
allocation. Its components include fixed V/HV buffers, projection and
small-solver arrays, candidate/reorthogonalization, Ritz/restart/snapshot blocks,
reference/operator payload, and conservative operator/complex-ARPACK workspace
envelopes. The ARPACK envelope was checked against SciPy 1.18.0; each result
records the actual version. The prepared workflow additionally preflights before
sector-basis/amplitude construction. Reference/provider constructors separately
admit their inputs before construction/compilation.

`max_owned_bytes` admits these declared payload models. Coefficients use the
repository's 24-byte model; Python containers, allocator retention and vendor
library overhead are outside it. This is not a process-memory guarantee.
Linux current RSS is sampled around actions, spin projection and solves, with
separate `max_rss_bytes` and initial headroom. Breaches stop solving, retain work
counters and produce failed results. No OS limit is installed, and boundary
sampling can miss transient peaks. Preparation compilation is sampled at its
boundaries and priced separately; independent spectral validation remains a
separate cost. Overlapping payload scopes are not an additive RSS estimate.

## Executed acceptance scope

| Check | Outcome / scope |
| --- | --- |
| Archived Hubbard 2×4 U=4, mu=0 excited singlet | Decode the unchanged final vector: initial energy −4.0175748274 t and residual 9.21e−9; retain its converged, spin-consistent, unresolved-root attempt |
| Two independent recovery starts | Both reach approximately −5.0125031527 t, with action residuals around 5e−12; consult the archived independent spectrum only after solving |
| Spin contamination | A singlet/quintet mixture has the triplet S² mean 2 but a large action defect; target projection passes the defect test |
| Forbidden constructions | Solving/reporting forbid density backends, H² construction, dense sector restrictions and ground-state oracles |
| Matched old spans (Step 4) | H4 STO-3G r=0.9, LiH CAS(4e,4o), Hubbard 2×2 U=4, mu=0: identity plus six unsplit determinant excitations match S/H at `rtol=atol=1e-12` and Ritz energies at `rtol=1e-12, atol=1e-10` |
| Storage / failure | Fixed buffer identities survive growth/restart; admission and monitored RSS failures remain visible; failed attempts stay in unconditional results |
| Tracking / multiplicity | Two-root recovery, unitary mixing of a degenerate block and ray-phase invariance are tested separately from ground-root evidence |

These are numerical regressions, not a matched-accuracy performance campaign,
RSS advantage or quantum-advantage result. The archived packet/Davidson evidence
and negative conclusions are not rewritten. Wider benchmarks, optimized external
selected-CI comparisons and optional PRIMME integration remain outside this
bounded validation. Step 5 still needs an actual sampled consumer and an
independently declared pilot/evaluation ratio experiment.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_bounded_actions.py tests/test_sector_reference_provider.py \
  tests/test_sector_solver.py --hypothesis-profile=ci
```

Solver documentation: [SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html),
[workspace source at v1.18.0](https://github.com/scipy/scipy/blob/v1.18.0/scipy/sparse/linalg/_eigen/arpack/arpack.py).
