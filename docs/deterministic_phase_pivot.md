# Deterministic phase pivot with relative tie tolerance

Numerical contract declared before implementation on main
`97b5286b713bf050bb24ace9141321201a557775`. The user's request reopens the
previously retired pivot item as numerical-stability work. This does not reopen
the residual-packet performance programme or authorize another frozen campaign.

## Rule

For a finite, nonempty complex vector v and a relative tie tolerance
`0 <= tie_tolerance < 1`, compute magnitudes after scaling the real and imaginary
parts by their largest absolute component. This avoids overflow/underflow in
the pivot comparison. Let m be the largest of those scaled magnitudes. Choose
the lowest index j with nonzero magnitude and

`m - |v_j|_scaled <= tie_tolerance * m`.

The new helper's default relative tolerance is `1e-12`. There is no absolute
amplitude floor. The zero vector has no pivot. Rotate the entire vector by the
conjugate unit phase of the selected component, without normalizing or editing
individual components. Its pivot is real and nonnegative up to floating-point
rounding. A finite input whose phase-rotated output cannot be represented in
complex128 is rejected explicitly.

This fixes the representative of a physical ray. It preserves norm, relative
phase, rank-one projectors and observable expectations up to rounding; changing
column phases gives congruent projected Hamiltonian/overlap matrices and the
same Ritz eigenvalues. It does not make different physical vectors equal.
Deterministic selection is guaranteed for the supplied values. Perturbations
or global-phase multiplication may move a component across a band boundary;
stability requires a margin from that boundary, not a claim of global continuity.

## API and integration

`clifford_qc.subspace.phase` will provide `deterministic_phase_pivot` and
`canonical_phase_tolerant`, also exported from `clifford_qc.subspace`.
Both accept keyword-only `tie_tolerance=1e-12`; the pivot helper returns an
integer or None, and canonicalization returns an independent complex128 array.
Invalid shapes, nonfinite values and non-real, nonfinite, boolean or out-of-range
tolerances are rejected.

`ActionSpace(..., phase_tie_tolerance=...)` and
`orthogonal_direction(..., phase_tie_tolerance=...)` opt into this convention.
Their default zero retains the historical strict-maximum helper and arithmetic.
State and true residual remain in the same phase; no extra H actions, changed
span, dependence cutoff or normalization rule is introduced. The historical
`orthogonal.canonical_phase` function remains byte-identical. Packet APIs and
all recorded shortlist decisions retain their existing conventions.

## Validation

Test explicit exact and near ties, first-index choice, boundary/nextafter cases,
zero vectors, tiny and large amplitudes, invalid inputs, copy independence and
global-phase/positive-scale behavior away from band boundaries. Independently
check norm, projectors, expectations and Ritz spectra on complex toys, plus
true residuals, H-action reuse and dependence rejection in ActionSpace and
orthogonal_direction. Exercise existing default-mode tests and provenance,
documentation and manuscript gates. Unit tests supply correctness evidence;
this task schedules no performance campaign and makes no convergence or
resource-advantage claim. The frame-coordinate study's failed exact ordered
shortlist criterion remains failed.
