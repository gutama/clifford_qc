# H2O residual-growth comparison

This exploratory follow-up knows PR #142's singlet-packet result (15.055190
mHa on H2O). It is not an independent confirmation or a quantum experiment.
Freeze this declaration and `configs/h2o_residual_comparison.json` before the
new implementation and before any new molecular outcomes. Preserve all earlier
records. Do not change this design after seeing results; report failures too.

Use only the existing H2O STO-3G CAS(8e,6o) FCIDUMP, in its N=8, Sz=0 sector
(225 determinants). Start every arm from the complete singlet part of the same
CAS(2e,2o), spatial orbitals [3,4], with other HF occupations frozen. Construct
a deterministic spin-adapted configuration basis: partition by complete spatial
occupation, project coordinate axes into each block's S2=0 eigenspace, and
twice orthogonalize in determinant order with positive largest-component phase.
Count its setup and storage. This dense small-sector symmetry construction is
not a scalable spin-adaptation algorithm.

Three independently timed arms share the Hamiltonian, symmetry, starting span,
two-dimensional energy-lowering score and numerical tolerances:

1. `residual_packets`: recompute the Ritz residual after each addition; divide
   by max(diag(H)-E+0.1,0.1), group by the initial external spatial occupations,
   project each group to a singlet and twice orthogonalize against the entire
   retained span. Score each surviving normalized direction with its exact H
   action; add the largest lowering, ties within 1e-12 by integer group key.
2. `spin_adapted_selected_ci`: select individual deterministic singlet
   configuration-state functions, using updated residual couplings and the
   same lowering score. Cache candidate H actions, charge their first
   construction and retained storage. This is a spin-matched selected-CI
   control, not an optimized SHCI implementation.
3. `adaptive_active_space`: sum squared current residual amplitudes on
   determinants whose occupation of each frozen orbital differs from HF.
   Unfreeze the largest-weight orbital, ties within 1e-12 by orbital index,
   and add its complete new singlet CAS span in one block. The orbital step
   is indivisible; disclose its larger column jumps.

Cache retained H actions in every arm. The measured solver builds no full H
matrix or H2 operator. Diagonal H entries come from diagonal Pauli terms and
are separately counted. S2 and its symmetry setup are explicit dense matrices.
Stop at true residual <=1e-9 Ha, 105 singlet columns, 102 growth steps, or no
admissible direction; never consult reference energies for selection/stopping.
Use the absolute/relative direction cutoffs and tolerances in the config.

Run three fresh processes per arm in rotated order, one BLAS thread, 120 s
timeout per worker. Measure setup and cumulative solver wall time, cumulative
H actions, scored candidates, retained/candidate array payloads, amplitude
support and OS process high-water RSS at every prefix. RSS includes imports
and setup; payload is not RSS. Report common/setup costs separately and also
include setup in time-to-target. A timed-out/failed arm remains a failed arm;
do not discard it from success fractions or rerun selectively.

Only after all measured arms finish, build an independent occupation-sign H
and obtain the full-sector reference energy. The 1.6 mHa target refers to this
fixed Hamiltonian, not basis-set completeness or experimental accuracy.
Derive first-target prefix, dimension/action/time frontiers and failure counts
from records. No equal-time or equal-memory superiority follows from column
matching. Tiny-system wall times are descriptive, not scaling evidence.

Before execution require full Git history, clean committed sources including
untracked files, the repository's numerical environment, input digest and
unchanged declared bytes; new implementation paths must be absent at this
declaration commit. Independently reconstruct the occupation Hamiltonian,
singlet projectors, selection decisions, Ritz energies, residuals, nested
spaces, work counts and support. Bind full observation bytes, including costs
and provenance, to a separate observation commit; preserve required Git objects
in a digest-bound bundle for merge-safe replay. No hardware, preparation,
shot-cost, SHCI/DMRG-superiority or quantum-advantage claim is licensed.
