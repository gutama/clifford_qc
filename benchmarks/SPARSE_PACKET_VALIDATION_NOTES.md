# Post-observation validation notes

The frozen declaration/config bytes are unchanged. The observed numerical
implementation is source `ac624c27346dc6a45337f994c05fdfcede1a6ea0`.

Independent replay found that dense and sparse reductions can choose different
largest-amplitude pivots when opposite-sign components have equal magnitudes
to rounding precision. Both stored vectors follow their own canonical phase
rule, but the coefficient arrays can differ by a global sign. The initial
checker incorrectly compared those arrays without phase alignment.

Validation therefore compares each ordered physical direction after aligning
one global phase, retaining the declared 1e-8 column tolerance. It separately
checks that each recorded vector has a nonnegative real largest pivot. Energy
and true-residual tolerances remain 1e-9; selected group identities, exact
scores, work counters and all other replay conditions are unchanged. This is
a disclosed post-observation correction to validation, not new numerical
data or a changed growth policy. Adversarial tests still reject altered
directions and a flipped recorded canonical phase.

Publication uses a validated complete prefix of the execution checkpoint.
All H2O and all seven larger Hubbard repetitions are retained. Six missing
secondary-input observations are disclosed failures, kept in denominators
without selective reruns. The envelope and receipt bind the recovered data;
independent references/readouts are computed after timed execution. The
subsequent immutable compressed-cell checkpoint implementation was not used
to generate these measurements and is not priced as a performance result.

Review reporting correction: 39 cells were scheduled, not evidenced as all
attempted. The surviving checkpoint is running with order index 33 in flight;
33 complete observations survive and six scheduled observations are unavailable.
The original archive and its bound `publication_recovery.policy` are preserved
byte for byte; the receipt's `reporting_correction` supersedes that policy's
unsupported completion/attempt wording. One held-out observation per arm still
establishes the observed deterministic accuracy loss, while timing coverage is
only 1/3 per arm. RSS includes a provenance-preflight floor, not a solver-only
peak. The report compares the earlier shortlist's scientific trajectory and
separates current setup and solver times without treating separate campaigns
as a controlled timing ablation. Repeated deterministic audits count once per
distinct decision trajectory. No new numerical observations were generated.
