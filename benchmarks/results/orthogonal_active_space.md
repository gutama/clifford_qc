# Orthogonal active-space pilot

Exact classical simulation on the four frozen molecular active-space inputs.
All errors in the first table are against the independently checked full-sector ground energy.
The classical reference is the complete four-determinant CAS(2e,2o), with other occupations frozen at HF.

| Input | Classical error (mHa) | Packet error (mHa) | Equal-dimension determinant error (mHa) | Support-containing determinant error (mHa) | Corrections | Columns | Determinant support |
|---|---:|---:|---:|---:|---:|---:|---:|
| h4 | 35.060850 | 11.222920 | 9.360386 | 11.199344 | 4 / 4 | 8 | 9 |
| beh2 | 5.613372 | 0.001153 | 0.003301 | 0.000000 | 3 / 4 | 7 | 8 |
| lih | 0.933965 | 0.018031 | 0.018011 | 0.018011 | 4 / 4 | 8 | 10 |
| h2o | 48.312980 | 24.647171 | 24.663242 | 24.645153 | 4 / 4 | 8 | 9 |

At the final declared budget, 2/4 inputs meet 1.6 mHa: beh2, lih.
The classical reference already meets it on: lih.
BeH2 has only three nonzero candidate packets; its four-packet request is explicitly capped.
All nested packet energies decrease. Exact same-span mixed coordinates reproduce the orthogonal energies,
and the determinant control containing each packet's support has no higher energy (to numerical tolerance).
Packet gains over equal-dimension determinants are small and inconsistent across inputs.
The packet and determinant controls have different amplitude support and selection work; column counts alone do not establish a computational advantage.

## Illustrative centered matrix-noise diagnostic

The table uses the largest frozen scale: 0.001 Ha Hamiltonian-entry noise and 0.001 dimensionless overlap-entry noise.
RMSE is conditional on successful solves and measured around the **noiseless retained-span energy**, not the full-sector ground energy.
Every scale/replica and every failed solve is retained in the JSON; unconditional target fractions count failures as failures.

| Input | Known orthogonal overlap RMSE (mHa) | Mixed exact overlap RMSE (mHa) | Mixed noisy overlap RMSE (mHa) | Noisy-overlap failures / replicas |
|---|---:|---:|---:|---:|
| h4 | 0.226254 | 0.226254 | 0.228118 | 0 / 64 |
| beh2 | 0.154469 | 0.154469 | 0.154537 | 0 / 64 |
| lih | 0.077099 | 0.077099 | 0.077106 | 0 / 64 |
| h2o | 0.224815 | 0.224815 | 0.225443 | 0 / 64 |

Congruently transported noise gives the same result with either exact overlap representation, replica by replica.
The added overlap-noise proxy does not demonstrate a substantial accuracy benefit for orthogonality on this frozen grid.
These Gaussian matrix perturbations are not Pauli samples, hardware noise, shot counts or confidence certificates.
A small estimation RMSE does not repair a retained subspace's energy bias.

## Scope and provenance

The experiment constructs packets classically from full-sector residuals and a full-sector diagonal.
It stores explicit vectors, charges candidate selection and every projected Hamiltonian action, and reports payload bytes rather than peak RSS.
Quantum preparation, physical transition measurements, hardware-accessible selection and total quantum cost remain unimplemented and unpriced.
No superiority over established SHCI/DMRG implementations or quantum advantage is inferred.

- Declaration commit: `dc99818e744553de3ae0bf3549410ece375e22f7`.
- Execution commit: `6b3bee6fa23325e9178f081ae6460dd7184bea92`.
- Configuration SHA-256: `a4fbdff72084ab75c11a94accde8db03c9a4d956a52d4a27f97cc46047614550`.
- The digest-bound [execution bundle](../provenance/orthogonal_active_space.bundle) preserves the original Git objects if publication changes commit IDs; the independent checker restores objects without changing branches.
- Original observation bytes are preserved. A later verification fix handles numerical packet amplitudes crossing the serialization cutoff by comparing reconstructed vectors at the existing tolerance; it changes no result or campaign setting.
- Sources: [design](../ORTHOGONAL_ACTIVE_SPACE_DECLARATION.md), [observations](../reference_results/orthogonal_active_space.json), and [independent checker](../check_orthogonal_active_space.py).
