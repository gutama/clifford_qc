# Orthogonal corrections around a classical active space

This is a bounded, post-hoc exploratory experiment motivated by the existing
A-CASE/PRD results and the H4 state-action equivalence diagnostic of 18 October
2026. It is not a blind preregistration, a reproduction of CASH-QSE, or a
quantum-advantage experiment. The design/configuration is committed before
implementation and before any new campaign energy or noise outcome. Synthetic
matrices may be used for implementation tests. No campaign parameter is tuned
after the outcomes are read.

## Fixed systems and classical reference

`configs/orthogonal_active_space.json` binds four existing FCIDUMPs by SHA-256:
H4, BeH2, LiH CAS(4e,4o), and H2O CAS(8e,6o). Work in the interleaved JW N/Sz
sector, with at most 256 determinants. Hold all occupations outside the declared
two-spatial-orbital frontier active space at their HF values. The resulting
CAS(2e,2o), including all four N/Sz determinants, is the classical reference
**span** C. Diagonalize its Hamiltonian and fix the lowest reference vector by
making its largest-magnitude component positive. Neither a full-sector energy
nor a full-sector eigenvector enters packet construction, ranking, or stopping.

## Corrections and controls

Compute the full-sector classical residual r=(I-CC†)H psi_C. For each external
occupation pattern, form a packet from r_d / (H_dd-E_C+0.1 Ha), using a positive
0.1 Ha denominator floor when its magnitude is smaller. Each packet has zero
support on C, and different packets have disjoint occupation support. Normalize
and phase-fix each nonzero packet. Rank once, before growth, by the exact
two-dimensional energy lowering with psi_C, resolving ties by external pattern
integer. Append nested prefixes of 0, 1, 2 and 4 packets (cap at availability).
These are classically constructed, explicitly stored sector vectors, not
hardware-prepared states or an implementable quantum selector.

At every prefix compare:

* C plus the orthogonal packets, using direct Hamiltonian action and S=I;
* the same span after adding 4 psi_C to each packet, using a generalized solve;
* C plus the same number of individual external determinants, ranked by their
  exact two-dimensional lowering with psi_C;
* C plus the entire determinant support of the retained packets. This last
  classical control contains the packet span and must have no higher energy.

The same-span control is a deliberately ill-conditioned coordinate change,
not a claim about the conditioning of typical A-CASE bases. Equal added column
counts are not equal work or equal determinant support. Report both. Compute
the exact full-sector ground energy only for accuracy readout. Measure true
residuals by H acting on the reconstructed states, never by a projected residual.
Include a full determinant-complement closure diagnostic: C plus every external
determinant must recover the sector spectrum. A small residual is not a
ground-state certificate without independent spectral information.

## Noise diagnostic and evidence boundary

On the largest declared packet prefix use 64 replicas at each frozen noise
scale. Work with H-E_C S to avoid artificial amplification by the molecular
energy zero. Add real symmetric Gaussian entry perturbations only outside the
classically known C/C block. Compare three estimators: known orthogonal overlap,
the same perturbation congruently transported to the mixed basis with its exact
overlap, and that same mixed pencil with additional independent symmetric
Gaussian overlap noise. The first two must agree replica by replica. The third
tests the stated overlap-noise model; reject a non-positive/near-singular
overlap and retain failures in the denominator. Report every replica, error
against the noiseless retained-span energy, full-sector variational violations,
and failure fractions. Do not discard failures from an unconditional accuracy
claim. Hamiltonian scales are in Ha; overlap scales are dimensionless.

These are illustrative matrix perturbations, **not** Pauli samples, shots,
hardware noise, confidence certificates or experimental cost estimates.
Preparation circuits, transition-measurement cost, optimized readout and
hardware-accessible selection remain unimplemented and unpriced.

## Execution, verification and interpretation

The producer defaults to outcome-free design validation. `--execute` requires a
clean committed source tree, the declaration in an earlier ancestor commit,
validated input hashes, a bounded sector, and a new output path. It refuses to
overwrite evidence. Store execution/configuration/source identities and the
repository's versioned per-quantity evidence declarations. Commit observations
separately from implementation. Existing molecular results are prior evidence;
this is not an independently selected confirmation set.

An independent checker rebuilds sector Hamiltonians from FCIDUMP occupation
signs, without native Pauli multiplication, sector matrix construction, packet
selection, or the new solver. It rebuilds packet supports/order, projected
pencils, residuals, classical controls, noise replicas and derived summaries.
Synthetic tests must falsify span changes, support leakage, incomplete cells,
wrong ordering, altered noise outcomes and optimistic failure accounting.

Success means the declared comparisons execute and pass their invariants.
Whether a prefix reaches 1.6 mHa is reported per molecule, not used to choose a
winner or terminate early. No total-cost superiority follows from this pilot.

Relevant starting points: Izmaylov, CASH-QSE (arXiv:2609.08170, 2026), and
Nation, Saki and Kang, generalized quantum subspace diagonalization
(arXiv:2603.18731, 2026). This pilot uses classical residual packets rather than
claiming their state preparations or reproducing their complete algorithms.
