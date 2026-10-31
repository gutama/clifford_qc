# Bounded Phase 15/16 implementation contract

Declared 31 October 2026 on main `8878d760b490d6819fcecb16651e3bbbd39733ed`,
before implementation and the new validation outcomes. This is an engineering
completion contract, not a new cost-benefit hypothesis or quantum-advantage
preregistration. The user explicitly requested completion of both phases.

## Deliverables and acceptance

1. **Phase 15: sampled S/H/K.** Extend the existing coefficient-only sampled
   provider with `K_ij=<A_i†H²A_j>`, guarded before products by the existing
   coefficient-storage preflight. Bind nondensity references by their metadata
   identity without asking them for densities, amplitudes or exact expectations.
   Limit the admitted measured universe and setting count explicitly.
2. Reconstruct the three Hermitian blocks from one set of actual joint
   histograms. A Bonferroni family of bounded-shot Hoeffding intervals covers
   every real and imaginary independent entry at a fixed endpoint. Within-shot
   dependence is retained; the guarantee assumes independent IID preparations
   across physical shots and settings. No normal approximation is a certificate.
3. On that simultaneous event, bound normalization, energy, variance and true
   residual norm for any coefficient vector, including a vector selected from
   the same fixed bank's measured matrices. A positive lower normalization bound
   is required. Preserve signed point variances, report inconsistent moments,
   and never manufacture convergence by clipping a negative estimate to zero.
   This establishes proximity to an eigenvalue, not ground-state dominance.
4. Solve noisy folded pencils and report confidence for the corresponding
   physical direction and an eigenvalue-containing target interval. Do not
   relabel the noisy pencil eigenvalue as an exact spectral certificate.
5. **Phase 16B: explicit polynomial alternative.** Supply opt-in truncated
   real-time operator generators, not a default Pauli expansion of exp(-iHt).
   Require degree, time grid, finite coefficient/product-work limits, norm,
   infidelity, energy-error and overlap-conditioning budgets. Report analytic
   Taylor truncation and sparse-pruning allowances separately from floating-
   point validation. Refuse a family outside its budgets before downstream use.
   Validate against matrix-free exact propagation on the bounded cases.
6. Route the truncated generators through the existing exact/sampled projected
   interfaces, including the new sampled second moments. Retain explicit rank
   loss and conditioning admission; an approximate evolution is not a unitary
   circuit or a claimed efficient quantum preparation.

## Fixed validation scope

The JSON design freezes the numerical grid. Required systems are open TFIM2,
TFIM3 and the two-site Hubbard model, at most four qubits. Analytic controls
include an exact eigenstate, a genuinely nonzero residual, dependent generators,
negative sample variances, unresolved normalization, complex matrix entries,
time reversal, zero time, and resource/truncation rejection. Tests use independent
dense products and matrix exponentials, plus deliberate corruptions.

The sampled campaign uses fresh joint multinomial histograms at three fixed
budgets and 32 independent replicas per system. Every success, unresolved result
and failed solve remains in the record. The independent checker recomputes
physical moments, interval arithmetic, folded inclusion and accounting. Coverage
is empirical validation of an analytic IID guarantee, not a calibration claim.

Completion means these capabilities ship and pass on this declared scope.
It does not require positive cost savings, chemical accuracy, a ground-root
guarantee, or affordable H² measurements on a historical bank.

## Historical boundaries remain

Q18, Q18-S1/S2/S3/S4 and the Phase 16A/16B v1/v2/v3 declarations and observations
are immutable inputs. Their negative/conditional conclusions remain valid on
their original scopes. This new engineering authorization implements the
explicitly truncated Phase 16B alternative; it does not promote v3 to GO or
authorize a circuit-native controlled-evolution backend. The one old variance
refinement remains unspent and is not a completion prerequisite for this path.

No new package file is needed: extend `subspace/second_moment.py`,
`second_moment_storage.py` and `time_evolution.py`. Keep the frozen package
inventory compatible with historical molecular replay. Publish this result-free
contract and design before the implementation commit, then publish observations
in a later commit. Preserve all three in Git history.
