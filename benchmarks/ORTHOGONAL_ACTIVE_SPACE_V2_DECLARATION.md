# Review revision of the orthogonal active-space pilot

This is a post-review exploratory revision. The original four-molecule v1
outcomes and the review's spin and round-off findings are known. Preserve the
original declaration, configuration, observation bytes and execution bundle.
This declaration and `configs/orthogonal_active_space_v2.json` precede the new
`clifford_qc/subspace/spin_packets.py` implementation and revised observations.
It is not an independent confirmation campaign or quantum-advantage claim.

Keep v1's four inputs, complete CAS(2e,2o) reference, preconditioner, fixed
0/1/2/4 correction budgets, ranking rule, controls, noise scales/seed/replicas,
accuracy target and resource scope. Change packet construction as follows:

* Group by external **spatial** occupation counts, allowing all alpha/beta
  orientations in a group. Project each preconditioned group into the exact
  S²=0 eigenspace before normalization. The disjoint groups and the classical
  span are invariant under S². Record S² and its action norm on every Ritz
  state in every arm; require singlet packet-arm states, including prefixes.
* Reject groups whose raw or projected norm is at most
  `max(1e-9, 1e-8 * norm(full external preconditioned residual))`. This cutoff
  is above the numerical noise observed in v1. Do not tune it on v2 outcomes.
  Keep the serialization cutoff separate at the absolute 1e-9 value.
* Individual-determinant controls retain their original definition. They need
  not be singlets; disclose their measured spin alongside energy. Do not infer
  spin-matched superiority from the column-matched comparison.

Record measured overlap matrices and identity defects, rather than assuming
exact identity in the readout. Replace the redundant second full eigensolve
with a direct classical/complement partition check. No full-sector H spectrum
or eigenvector selects packets. The small S² diagonalizations enforce a known
symmetry and are separately counted; they do not establish scalable selection.

Before science, enforce full Git history, clean sources including untracked
files, declared numerical environment, input paths/digests, and absence of the
new implementation path at this declaration commit. The historical v1 replay
continues to bind its original sources and observations; current code may
evolve with explicit historical versus revised identities.

Independently replay both designs, all controls, residuals, spin diagnostics,
noise replicas and failure/variational-violation counts. Generate report
conclusions from observations. Publish the revised outcomes even if accuracy
worsens. Preserve every original v1 observation; never overwrite evidence.
