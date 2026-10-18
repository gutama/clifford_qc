# Original orthogonal active-space pilot

Exact classical simulation on 4 declared molecular inputs.
Errors are against the independently checked full-sector ground energy.
The reference is the complete four-determinant CAS(2e,2o), with other occupations frozen at HF.

| Input | Classical error (mHa) | Packet error (mHa) | Equal-dimension determinant error (mHa) | Support-containing determinant error (mHa) | Packet gain over equal-dimension (mHa) | Corrections |
|---|---:|---:|---:|---:|---:|---:|
| h4 | 35.060850 | 11.222920 | 9.360386 | 11.199344 | -1.862533 | 4 / 4 |
| beh2 | 5.613372 | 0.001153 | 0.003301 | 0.000000 | +0.002149 | 3 / 4 |
| lih | 0.933965 | 0.018031 | 0.018011 | 0.018011 | -0.000020 | 4 / 4 |
| h2o | 48.312980 | 24.647171 | 24.663242 | 24.645153 | +0.016071 | 4 / 4 |

Final target (1.6 mHa): 2/4 inputs (beh2, lih).
Already within target at the classical reference: lih.
Capped requests: beh2: 3 candidates, retained 3.
Nested energy monotonicity holds on 4/4 inputs (1e-9 Ha tolerance).
Support containment energy bound holds on 4/4 inputs.
Maximum same-span energy difference: 3.13e-13 Ha.
Packet gains over equal-dimension determinants range from -1.862533 to +0.016071 mHa; negative values favor determinants.
Column counts alone do not equate amplitude support, symmetry, selection work or computational cost.

## Spin and measured overlap

| Input | Packet ⟨S²⟩ | Mixed ⟨S²⟩ | Equal-dimension determinant ⟨S²⟩ | Support-containing determinant ⟨S²⟩ | Packet measured Gram defect |
|---|---:|---:|---:|---:|---:|
| h4 | 0.00732457 | 0.00732457 | 0.0213455 | 0.00736034 | 2.22e-16 |
| beh2 | 6.57903e-28 | 4.518e-27 | 3.13207e-28 | 4.81329e-62 | 2.22e-16 |
| lih | 1.99067e-28 | 5.70329e-28 | 1.27355e-28 | 5.45943e-30 | 2.22e-16 |
| h2o | 0.0103353 | 0.0103353 | 0.0103363 | 0.0103353 | 3.33e-16 |

These spin and Gram diagnostics are retrospective calculations from the serialized states, not new original observation fields.
The original external spin-orbital partition can split spin partners; its Ritz states are not constrained to be singlets.
The original absolute 1e-12 packet cutoff can normalize round-off-only groups. These limitations remain part of the preserved v1 pilot.
Original orthogonal-arm overlap fields assumed identity; the table above measures the Gram defect from serialized vectors.
The original complement-spectrum scalar is permutation round-off, not independent evidence of partition correctness.

## Illustrative centered matrix-noise diagnostic

RMSE is conditional on a successful solve and measured around the noiseless retained-span energy.
Full-sector variational violations count successful estimates below the exact ground energy by more than 1e-9 Ha.
Unconditional target fractions include failed solves in the denominator.

| Input | H noise (Ha) | S noise | Estimator | RMSE (mHa) | Failures / replicas | Variational violations / successes | Unconditional target fraction |
|---|---:|---:|---|---:|---:|---:|---:|
| h4 | 0 | 0 | orthogonal_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0 | 0 | mixed_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0 | 0 | mixed_noisy_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.0001 | 0.0001 | orthogonal_known_overlap | 0.027416 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.0001 | 0.0001 | mixed_known_overlap | 0.027416 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.0001 | 0.0001 | mixed_noisy_overlap | 0.027613 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.001 | 0.001 | orthogonal_known_overlap | 0.226254 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.001 | 0.001 | mixed_known_overlap | 0.226254 | 0 / 64 | 0 / 64 | 1.000000 |
| h4 | 0.001 | 0.001 | mixed_noisy_overlap | 0.228118 | 0 / 64 | 0 / 64 | 1.000000 |
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
| lih | 0.0001 | 0.0001 | orthogonal_known_overlap | 0.007849 | 0 / 64 | 1 / 64 | 1.000000 |
| lih | 0.0001 | 0.0001 | mixed_known_overlap | 0.007849 | 0 / 64 | 1 / 64 | 1.000000 |
| lih | 0.0001 | 0.0001 | mixed_noisy_overlap | 0.007849 | 0 / 64 | 1 / 64 | 1.000000 |
| lih | 0.001 | 0.001 | orthogonal_known_overlap | 0.077099 | 0 / 64 | 25 / 64 | 1.000000 |
| lih | 0.001 | 0.001 | mixed_known_overlap | 0.077099 | 0 / 64 | 25 / 64 | 1.000000 |
| lih | 0.001 | 0.001 | mixed_noisy_overlap | 0.077106 | 0 / 64 | 25 / 64 | 1.000000 |
| h2o | 0 | 0 | orthogonal_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0 | 0 | mixed_known_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0 | 0 | mixed_noisy_overlap | 0.000000 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.0001 | 0.0001 | orthogonal_known_overlap | 0.024546 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.0001 | 0.0001 | mixed_known_overlap | 0.024546 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.0001 | 0.0001 | mixed_noisy_overlap | 0.024646 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.001 | 0.001 | orthogonal_known_overlap | 0.224815 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.001 | 0.001 | mixed_known_overlap | 0.224815 | 0 / 64 | 0 / 64 | 1.000000 |
| h2o | 0.001 | 0.001 | mixed_noisy_overlap | 0.225443 | 0 / 64 | 0 / 64 | 1.000000 |

Maximum replica difference between exact-overlap coordinate representations: 1.42e-14 Ha.
Gaussian matrix perturbations have no conversion here to Pauli shots, hardware noise or confidence certificates.
A small estimation RMSE does not repair retained-space energy bias or guarantee a variational estimate.

## Scope and provenance

Explicit classical sector vectors, full residuals and sector diagonals are used. Candidate selection and Hamiltonian actions are counted; payload bytes are not peak RSS.
Quantum preparation, physical measurements, scalable selection and total quantum cost remain unimplemented and unpriced.
No superiority over SHCI/DMRG implementations or quantum advantage is established.

- Declaration: `dc99818e744553de3ae0bf3549410ece375e22f7`.
- Execution: `6b3bee6fa23325e9178f081ae6460dd7184bea92`.
- Configuration SHA-256: `a4fbdff72084ab75c11a94accde8db03c9a4d956a52d4a27f97cc46047614550`.
- The checker binds original observation bytes to their observation commit and independently replays the science.
- V1 history is preserved in the digest-bound execution bundle. The separately declared v2 revision preserves all v1 observations.
