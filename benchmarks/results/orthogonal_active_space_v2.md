# Revised singlet orthogonal active-space pilot

Exact classical simulation on 4 declared molecular inputs.
Errors are against the independently checked full-sector ground energy.
The reference is the complete four-determinant CAS(2e,2o), with other occupations frozen at HF.

| Input | Classical error (mHa) | Packet error (mHa) | Equal-dimension determinant error (mHa) | Support-containing determinant error (mHa) | Packet gain over equal-dimension (mHa) | Corrections |
|---|---:|---:|---:|---:|---:|---:|
| h4 | 35.060850 | 0.625173 | 9.360386 | 0.136241 | +8.735213 | 4 / 4 |
| beh2 | 5.613372 | 0.001153 | 0.003301 | 0.000000 | +0.002149 | 3 / 4 |
| lih | 0.933965 | 0.000477 | 0.018011 | 0.000333 | +0.017534 | 4 / 4 |
| h2o | 48.312980 | 15.055190 | 24.663242 | 14.605534 | +9.608052 | 4 / 4 |

Final target (1.6 mHa): 3/4 inputs (h4, beh2, lih).
Already within target at the classical reference: lih.
Capped requests: beh2: 3 candidates, retained 3.
Nested energy monotonicity holds on 4/4 inputs (1e-9 Ha tolerance).
Support containment energy bound holds on 4/4 inputs.
Maximum same-span energy difference: 4.97e-13 Ha.
Packet gains over equal-dimension determinants range from +0.002149 to +9.608052 mHa; negative values favor determinants.
Column counts alone do not equate amplitude support, symmetry, selection work or computational cost.

## Spin and measured overlap

| Input | Packet ⟨S²⟩ | Mixed ⟨S²⟩ | Equal-dimension determinant ⟨S²⟩ | Support-containing determinant ⟨S²⟩ | Packet measured Gram defect |
|---|---:|---:|---:|---:|---:|
| h4 | 8.94046e-19 | 7.62932e-30 | 0.0213455 | 8.96457e-19 | 2.22e-16 |
| beh2 | 7.51233e-32 | 4.60135e-27 | 3.13207e-28 | 4.81329e-62 | 2.22e-16 |
| lih | 7.17378e-30 | 1.58321e-27 | 1.27355e-28 | 1.49055e-30 | 2.22e-16 |
| h2o | 1.1351e-18 | -5.15079e-19 | 0.0103363 | 5.18011e-19 | 4.44e-16 |

The revised packet arms enforce singlet S² action norms at every prefix; individual-determinant controls are not constrained to be singlets.
Packet norm cutoff: max(1e-09, 1e-08 × full external preconditioned residual norm), applied before normalization and after singlet projection.
The producer records measured overlaps and spin diagnostics for every arm.

## Illustrative centered matrix-noise diagnostic

RMSE is conditional on a successful solve and measured around the noiseless retained-span energy.
Full-sector variational violations count successful estimates below the exact ground energy by more than 1e-9 Ha.
Unconditional target fractions include failed solves in the denominator.

| Input | H noise (Ha) | S noise | Estimator | RMSE (mHa) | Failures / replicas | Variational violations / successes | Unconditional target fraction |
|---|---:|---:|---|---:|---:|---:|---:|
| h4 | 0 | 0 | orthogonal_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0 | 0 | mixed_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0 | 0 | mixed_noisy_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.0001 | 0.0001 | orthogonal_known_overlap | 0.028581 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.0001 | 0.0001 | mixed_known_overlap | 0.028581 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.0001 | 0.0001 | mixed_noisy_overlap | 0.028909 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.001 | 0.001 | orthogonal_known_overlap | 0.284078 | 0 / 64 | 1 / 64 | 1.000000 |
| h4 | 0.001 | 0.001 | mixed_known_overlap | 0.284078 | 0 / 64 | 1 / 64 | 1.000000 |
| h4 | 0.001 | 0.001 | mixed_noisy_overlap | 0.285874 | 0 / 64 | 2 / 64 | 1.000000 |
| beh2 | 0 | 0 | orthogonal_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| beh2 | 0 | 0 | mixed_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| beh2 | 0 | 0 | mixed_noisy_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| beh2 | 0.0001 | 0.0001 | orthogonal_known_overlap | 0.019522 | 0 / 64 | 34 / 64 | 1.000000 |
| beh2 | 0.0001 | 0.0001 | mixed_known_overlap | 0.019522 | 0 / 64 | 34 / 64 | 1.000000 |
| beh2 | 0.0001 | 0.0001 | mixed_noisy_overlap | 0.019509 | 0 / 64 | 34 / 64 | 1.000000 |
| beh2 | 0.001 | 0.001 | orthogonal_known_overlap | 0.154469 | 0 / 64 | 30 / 64 | 1.000000 |
| beh2 | 0.001 | 0.001 | mixed_known_overlap | 0.154469 | 0 / 64 | 30 / 64 | 1.000000 |
| beh2 | 0.001 | 0.001 | mixed_noisy_overlap | 0.154537 | 0 / 64 | 30 / 64 | 1.000000 |
| lih | 0 | 0 | orthogonal_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| lih | 0 | 0 | mixed_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| lih | 0 | 0 | mixed_noisy_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| lih | 0.0001 | 0.0001 | orthogonal_known_overlap | 0.007985 | 0 / 64 | 33 / 64 | 1.000000 |
| lih | 0.0001 | 0.0001 | mixed_known_overlap | 0.007985 | 0 / 64 | 33 / 64 | 1.000000 |
| lih | 0.0001 | 0.0001 | mixed_noisy_overlap | 0.007985 | 0 / 64 | 33 / 64 | 1.000000 |
| lih | 0.001 | 0.001 | orthogonal_known_overlap | 0.079154 | 0 / 64 | 33 / 64 | 1.000000 |
| lih | 0.001 | 0.001 | mixed_known_overlap | 0.079154 | 0 / 64 | 33 / 64 | 1.000000 |
| lih | 0.001 | 0.001 | mixed_noisy_overlap | 0.079160 | 0 / 64 | 33 / 64 | 1.000000 |
| h2o | 0 | 0 | orthogonal_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0 | 0 | mixed_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0 | 0 | mixed_noisy_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.0001 | 0.0001 | orthogonal_known_overlap | 0.026951 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.0001 | 0.0001 | mixed_known_overlap | 0.026951 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.0001 | 0.0001 | mixed_noisy_overlap | 0.027434 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.001 | 0.001 | orthogonal_known_overlap | 0.254074 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.001 | 0.001 | mixed_known_overlap | 0.254074 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.001 | 0.001 | mixed_noisy_overlap | 0.254886 | 0 / 64 | 0 / 64 | 1.000000 |

Maximum replica difference between exact-overlap coordinate representations: 1.42e-14 Ha.
Gaussian matrix perturbations have no conversion here to Pauli shots, hardware noise or confidence certificates.
A small estimation RMSE does not repair retained-space energy bias or guarantee a variational estimate.

## Scope and provenance

Explicit classical sector vectors, full residuals and sector diagonals are used. Candidate selection and Hamiltonian actions are counted; payload bytes are not peak RSS.
Quantum preparation, physical measurements, scalable selection and total quantum cost remain unimplemented and unpriced.
No superiority over SHCI/DMRG implementations or quantum advantage is established.

- Declaration: `549a55e36d540dfb24a512b2e184809edfe73835`.
- Execution: `32364a91a7ea37af33d58924c85f99b40aea7a87`.
- Configuration SHA-256: `423855acaeb880aa068fd9b7493ea7649499289d924706fbf5776a29be76925f`.
- The checker binds original observation bytes to their observation commit and independently replays the science.
- Both versions have digest-bound execution bundles retaining declaration, execution and observation commits across merging. V2 preserves all v1 observations.
