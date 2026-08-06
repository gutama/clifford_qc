# `clifford_qc` integrated research roadmap — QSCI, A-CASE, and multiresolution methods

This document supersedes the former Phases 8–14 literature roadmap. It keeps
QSCI/SQD as the critical scientific comparison for Paper B, while aligning the
implementation plan with the repository after PRs #31–#35:

- orbital bases are explicit and benchmarked; orbital wavelets were a negative
  result rather than a default;
- configuration-space Haar packets are implemented as a classical basis change
  among virtual determinant generators;
- ADAPT-VQE and A-CASE expose separate config/state/step boundaries;
- generator domains are split into response, configuration, and fermionic
  modules;
- full-space and sector state-vector action are matrix-free.

The central research question is:

> **Can sampled determinant subspaces, operator-response dressing, and
> support-pruned multiresolution packets produce a compact, resource-honest
> eigensolver that adds something classical selected CI does not already give?**

The roadmap separates the Paper B critical path from measurement work and
longer-horizon infrastructure. Wavelets enter only where the current code has
already shown a defensible use: configuration-space coarse-to-fine selection.
They do not replace QSCI and are not treated as a universal orbital basis.

---

## 0. Current repository facts that determine the plan

### 0.1 QSCI remains missing

`clifford_qc` has no first-class QSCI/SQD implementation and no QSCI row on the
A-CASE ladder. This remains the principal external-validity gap for Paper B.

QSCI and A-CASE spend different resources:

| | A-CASE | QSCI/SQD |
|---|---|---|
| basis | virtual operator states `A_i|psi>` | sampled basis configurations |
| overlap | measured nonorthogonal `S` | identity |
| projected Hamiltonian | reconstructed from measured Pauli words | built classically |
| main quantum cost | state preparation plus grouped word measurement | state preparation plus computational-basis sampling |
| principal risk | `W`, shot cost, and conditioning | duplicate sampling and determinant compactness |

`W=0` for QSCI projected-matrix measurement is correct, but it is not a complete
resource verdict. The comparison must also report sampling yield, preparation
cost, classical matrix construction, diagonalization cost, and memory.

### 0.2 Configuration-space Haar is already implemented

`subspace.configuration.configuration_haar_packets` builds an orthogonal finite
tree-Haar transform over a caller-ordered configuration list. It is a classical
change of basis among virtual generators, not a quantum wavelet circuit.
Support pruning makes it an opt-in coarse tier rather than a convergence-complete
basis.

The committed `2x2` Hubbard benchmark provides one positive finite instance:
staging Haar packets before the ordinary level-4 pool reaches the exact sector
energy with 33.1% fewer projected Pauli words and one fewer basis direction,
while increasing maximum element support and `kappa(S)`. This justifies packets
as a hybrid ablation, not as a default.

### 0.3 Orbital wavelets are not the integration target

The orbital-basis benchmark found that no basis wins uniformly and that the
tested Daubechies orbital bases lose to site or momentum bases on the relevant
trade-offs. Orbital basis remains a recorded model parameter. This roadmap does
not reopen orbital-wavelet optimization without a new, system-specific,
falsifiable reason.

### 0.4 Matrix-free action changes the feasible boundary

`PauliLinearOperator` and `SectorOperator` provide matrix-free action and Krylov
solves without materializing dense operators. They are the foundation for:

- exact QSCI sampling-state oracles on small and medium systems;
- restriction to sampled index sets;
- propagation diagnostics;
- residual and variance checks.

They do not make real-time operators sparse in the multivector generator bank.
Circuit-native real-time A-CASE remains a separate architectural problem.

### 0.5 Adaptive workflows now have clean extension seams

`AdaptConfig`/`AdaptState`/`adapt_step` and
`ACASEConfig`/`ACASEState`/`acase_step` allow hierarchical or racing policies to
be added without unifying the distinct mathematics of ADAPT-VQE and A-CASE.
Multiresolution selection should target these boundaries rather than introduce
a generic adaptive runner.

---

## 1. Literature index and citation discipline

The roadmap is grounded in the following fifteen papers. The table is retained
because it records which work motivates a build task, constrains positioning,
or should be declined.

| # | arXiv | Theme | Integrated disposition |
|---|---|---|---|
| 1 | 2302.11320 | QSCI | Mandatory sampled-subspace baseline; Track A. |
| 2 | 2411.00468 | ext-SQD excited states | Direct chemistry/excited-state comparator; Track A. |
| 3 | 2407.08696 | CEO-ADAPT-VQE | Constrains Paper A framing; pool benchmark deferred until after Track A. |
| 4 | 2409.03747 | Oscillator-qubit | Qumode layer declined; retain symmetry-post-selection accounting only. |
| 5 | 2301.10196 | Overlap-ADAPT-VQE | Motivates overlap-targeted A-CASE selection; Phase 11. |
| 6 | 2412.13839 | Time-evolved QSCI | Time-evolved QSCI input; Phase 16A. |
| 7 | 2302.03052 | Projection-based embedding | Interface boundary only; Phase 18. |
| 8 | 2606.30551 | Generative-ML QSCI | Withdrawn; do not cite. Use paper 14 for related compact-QSCI content. |
| 9 | 2501.14968 | Measurement review | Fully commuting grouping context; Track B. |
| 10 | 2305.04783 | Folded-spectrum VQE | Second moments, variance, and folded spectrum; Phase 15. |
| 11 | 2606.05968 | BK symmetry trap | Not evidence for the headline claim; mapping regression motivation only. |
| 12 | 2409.11210 | MORE-ADAPT-VQE | Existing multi-root capability deserves a later benchmark; deferred. |
| 13 | 2311.01393 | FLDC barren plateaus | Positioning only; build nothing. |
| 14 | 2607.20585 | ML-compact QSCI subspaces | Compactness comparison structure; unrefereed benchmark claims require reproduction. |
| 15 | 2607.16869 | Correlation rank and Clifford-accessible measurement | Test the explicit invariant first; benchmark claims remain unverified. |

### 1.1 Actionable citation hygiene

- **arXiv:2606.30551** was withdrawn by arXiv administrators because the
  submitter did not have the rights to agree to the licence at submission. It
  must not enter either bibliography. For related RBM/configuration-recovery
  content, cite **arXiv:2607.20585** instead and label its benchmark evidence as
  unrefereed until reproduced.
- **arXiv:2606.05968** is not evidence for an intrinsic Bravyi–Kitaev symmetry
  trap or one-cycle FCI convergence. Its stated fixed-UCCSD derivative and
  ADAPT commutator gradient are the same derivative for the same anti-Hermitian
  generator, while its claimed FCI states retain large commutator gradients.
  Use it only to motivate mapping-consistency regression tests.
- **arXiv:2607.16869** supplies an explicit structural claim that can be tested
  independently. Separate a reproduced parity/X-rank invariant from its
  unreproduced shot-reduction benchmarks.

The mapping regression suite motivated by paper 11 must check:

- exact energy invariance across mappings;
- correctly encoded reference states;
- equality of finite-difference, analytic, and commutator gradients;
- separate measurement of Pauli weight, distinct-word count, grouping cost,
  and compiled circuits.

### 1.2 Explicitly deferred or positioning-only items

- **MORE-ADAPT-VQE:** multi-root A-CASE already exists, but a dedicated
  MORE-ADAPT comparison is deferred until the QSCI/ext-SQD ladder is complete.
- **FLDC barren plateaus:** use only to position finite-depth ADAPT circuits.
  A-CASE itself solves a generalized eigenproblem and has no variational
  trainability landscape; no implementation task follows.
- **Circuit-native real-time A-CASE:** retained as falsifiable Question Q8 and
  architecture research in Phase 16B, not treated as an incremental generator.

---

## 2. Program structure

The work is divided into three tracks.

### Track A — Paper B critical path

1. trusted QSCI/SQD baseline;
2. classical selected-CI controls;
3. QSCI x A-CASE operator dressing;
4. overlap-targeted and multiresolution selection;
5. resource-honest ladder and manuscript repositioning.

### Track B — measurement methods

1. parity/X-rank invariant;
2. fully commuting Clifford-accessible grouping;
3. covariance-correct reconstruction;
4. certified shot-cost comparison.

### Track C — longer-horizon infrastructure

1. second-moment bank;
2. folded-spectrum and variance extrapolation;
3. time-evolved inputs;
4. mapping breadth;
5. embedding interface.

Track A must not wait for Tracks B or C.

---

## Track A — QSCI, classical controls, and the hybrid

## Phase 8 — trusted sampled-subspace baseline

Add `clifford_qc/subspace/qsci.py` as a first-class method rather than a
benchmark stub.

### 8A — sampling contract

Define a result object carrying at least:

- raw and accepted shots;
- unique basis configurations and duplicate fraction;
- discarded or repaired fraction;
- cumulative retained probability;
- sampled subspace dimension `M`;
- projected-matrix measurement words `W=0`;
- classical matrix nonzeros, build time, solve time, and peak memory;
- energy, variational gap, and evidence label.

Initial implementation should sample exact probabilities generated by existing
state backends. Hardware-noise emulation and configuration recovery are later
layers over the same contract.

### 8B — sampled Hamiltonian restriction

Reuse `sector_basis`, sector indexing, and the compiled `SectorOperator`. Add a
safe API that restricts a sector operator to a declared set of sector indices.
An exact row/column restriction of the validated sector operator is preferable
to writing Slater–Condon rules prematurely.

Required invariants:

1. the sampled Hamiltonian is Hermitian;
2. increasing nested sampled sets gives non-increasing Ritz energies;
3. for every retained root `k`, the `k`-th sampled Ritz value is not below the
   `k`-th exact sector eigenvalue, as required by Cauchy interlacing;
4. selecting the entire sector reproduces the sector spectrum;
5. permutation of sampled configuration order changes no eigenvalue.

### 8C — fermionic recovery and generic spin sampling

For fermionic systems, implement post-selection and optional recovery against
particle number and `S_z`, reporting every discarded or repaired sample.

For spin systems such as Kitaev, do **not** report that QSCI has no arm. The
fermionic SQD recovery rule is unavailable, but raw computational-basis sampled
subspace diagonalization remains a valid baseline. The question is whether the
basis is compact, not whether it is definitionally excluded.

### 8D — state inputs

Run the QSCI arm with declared inputs:

- reference determinant;
- exact ground-state sampling oracle, for validation only;
- existing ADAPT-VQE state;
- later, a time-evolved state.

Never mix oracle and implementable inputs in one evidence category.

### 8E — ladder integration

Add QSCI to `benchmarks/run_acase_ladder.py`. Equal-`M` remains one comparison,
but the ladder must also emit Pareto records for:

- error versus quantum shots and unique configurations;
- error versus state-preparation cost;
- error versus classical matrix nonzeros and solve time;
- error versus memory.

**Go/no-go:** the full-sector limit, Hermiticity, interlacing, and permutation
invariants must hold on H4, Hubbard, and at least one spin model before QSCI is
used in manuscript claims.

---

## Phase 9 — classical selected-CI controls

This phase is mandatory. Without it, a successful QSCI x A-CASE hybrid may be
indistinguishable from ordinary determinant-space expansion.

For each sampled determinant set `D`, construct:

1. **QSCI:** diagonalize only `span(D)`.
2. **Excitation closure:** add every unique determinant reached by the same
   singles/doubles used for operator dressing.
3. **One-step selected CI:** add determinants using a declared HCI-, CIPSI-, or
   perturbative-style score.
4. **Budget-matched selected CI:** stop at the same determinant count or
   classical matrix cost as the hybrid.

Record determinant count, Hamiltonian nonzeros, classical selection work,
energy, variance, and memory.

### Span-equivalence diagnostic

For each dressed family compare

`span{E_mu |D_k>}`

with the determinant closure generated from the same `D_k` and excitation
operators. Compute numerical ranks and principal angles.

- Equal spans mean the operator form is a representation or measurement-cost
  choice, not a richer variational space.
- A smaller operator-generated basis spanning a much larger determinant closure
  is a valid compactness result.
- Directions outside the declared closure require an algebraic explanation and
  an independent check.

**Go/no-go:** Phase 10 may claim a hybrid gain only after it beats or differs
structurally from these controls.

---

## Phase 10 — QSCI x A-CASE hybrid

### 10A — sampled configurations as generators

Convert retained QSCI configurations through `configuration_generator`. The
current configuration module already supplies the needed generator type.

### 10B — operator-response dressing

Build declared families such as:

- configuration x conserving excitation;
- configuration x commutator response;
- optional support-bounded compound families.

Every family reports candidate count, generator support `S_A`, projected
element support `S_H`, incremental word universe, and conditioning impact.

### 10C — required hybrid arms

Compare:

1. bare sampled configurations;
2. sampled configurations plus individual dressed generators;
3. sampled configurations plus support-pruned Haar packets, followed by the
   ordinary dressed pool.

The third arm integrates the existing wavelet result without creating a
separate state-vector compression project.

### 10D — honest claim

> A sampled determinant set, enriched by selected operator-response directions,
> may reach a target accuracy with fewer retained variational directions or a
> better measured-resource Pareto point than either bare QSCI or bare A-CASE.

Do not claim that operator dressing is strictly richer until Phase 9 proves it.

**Primary systems:** `hubbard_2x2`, `hubbard_2x3`, H4 equilibrium/stretched, and
one molecular FCIDUMP rung with matched multiplicity.

---

## Phase 11 — overlap-targeted and multiresolution selection

### 11A — overlap target

Extend A-CASE scoring with a target-overlap criterion using the QSCI Ritz vector
or a classical selected-CI vector. Preserve existing scale invariance and
orthogonality rejection. Keep the target outside `CandidateScore` so lowering
and overlap criteria remain independently testable.

### 11B — coarse-to-fine packet selection

Use the existing configuration Haar transform as a hierarchy:

1. order sampled configurations using declared physics metadata;
2. score support-pruned coarse packets first;
3. refine only selected or competitive intervals;
4. hand the retained basis to the convergence-complete individual/dressed pool.

Candidate blocks may use `acase_step`, but the score remains A-CASE-specific.
Do not create a shared ADAPT/A-CASE runner.

### 11C — ordering ablations

Compare at least:

- probability order;
- excitation rank plus occupation-pattern metadata;
- determinant-graph traversal;
- random-order controls.

A packet result is publishable only if it is not an accident of one ordering.

### 11D — stopping and uncertainty

Track unseen or unstable probability mass with simple baselines first:
duplicate-rate stopping, bootstrap set stability, or an unseen-mass estimate.
Add wavelet/block allocation only if it improves those controls.

**Go/no-go:** on `hubbard_2x3`, the new criterion or hierarchy must select useful
configuration/dressed directions that the lowering-only pool misses, or the
blind spot is attributed to the family rather than the selector.

---

## Phase 12 — integrated Paper B ladder

The final Track A ladder contains:

- reference and exact-sector results;
- fixed QSE and Krylov;
- ADAPT-VQE and A-CASE;
- QSCI;
- excitation-closure and selected-CI controls;
- QSCI x dressed A-CASE;
- QSCI x Haar-stage x dressed A-CASE.

Required fields include:

- `M`, retained rank, and `kappa(S)`;
- energy error, variance or true residual where available;
- sampling shots, unique yield, duplicate rate, discard/recovery rate;
- state-preparation metadata;
- `W`, grouping contexts, and certified shot cost for measured arms;
- classical matrix nonzeros, build/solve time, and peak memory;
- generator and element supports;
- evidence category and seed.

Paper B must follow the Pareto frontier that survives. If QSCI and classical
selected CI dominate chemistry, narrow A-CASE to systems and representations
where operator-generated or packet directions add measured value.

---

## Track B — Clifford-accessible measurement

## Phase 13 — structural invariant first

Implement GF(2) rank of Hamiltonian X masks using `word_masks`. Test the
explicit parity/X-rank ceiling

`r_X <= 2(N - 1)`

across spin-conserving Jordan–Wigner Hamiltonians, FCIDUMP models, lattice
models, and effective-Hamiltonian ingestion. Any violation stops the grouping
work until the theorem/model construction mismatch is understood.

## Phase 14 — QWC plus fully commuting groups

Extend measurement grouping with fully commuting groups and Clifford
simultaneous diagonalization.

The metric is not group count alone. Report certified leading shot cost at
fixed word universe, allocator, and confidence target, plus:

- diagonalizing circuit depth and two-qubit gates;
- connectivity assumptions;
- covariance-aware reconstruction;
- Monte Carlo agreement between predicted and empirical uncertainty.

The previous multiply-capable-word variance bug remains the gate: lower group
count with inflated or double-counted variance is failure. This track supports
Paper A or a separate measurement paper and must not block Track A.

---

## Track C — longer-horizon infrastructure

## Phase 15 — second-moment bank

Add a `SecondMomentBank` for

`K_ij = <psi|A_i^dagger H^2 A_j|psi>`.

Before building the full bank, add a support/cost preflight for `H^2`. If the
estimated word universe is prohibitive, keep dense or matrix-free residual
oracles for validation and restrict measured implementation to declared small
systems.

Uses:

- true Ritz residual norms;
- energy variance and variance extrapolation;
- folded-spectrum roots;
- an independent convergence criterion.

## Phase 16 — time-evolved inputs, split by method

### 16A — QSCI input

Use `PauliLinearOperator.as_linear_operator()` with matrix-free
`scipy.sparse.linalg.expm_multiply`, Krylov propagation, or a validated Trotter
circuit to generate time-evolved sampling states. SciPy is a `research` extra;
when `expm_multiply` receives a `LinearOperator`, supply the analytically known
`traceA` rather than asking SciPy to estimate it from a matrix-free object.

### 16B — A-CASE real-time generators

Do not represent `exp(-iHt)` as an `MV` by default: Pauli support can become
dense. A circuit-native generator requires a different matrix-element backend
and resource model. Treat this as architecture research.

A short-time polynomial response may be tested only with explicit truncation,
norm, fidelity, energy-error, and conditioning budgets against matrix-free
propagation. This phase owns Q8 below.

## Phase 17 — mapping validation and breadth

Before using BK or parity in scientific records:

- transform Hamiltonian, reference, and generators consistently;
- verify energy and gradient invariance;
- compare Pauli weight, distinct words, grouping, and circuits separately.

Lower Pauli weight does not imply lower `W` or fewer groups.

A CEO pool and dedicated MORE-ADAPT benchmark are follow-ups after Track A;
they constrain positioning but do not gate the QSCI hybrid experiment.

## Phase 18 — embedding boundary

Keep DMET and projection-based embedding outside the package. Provide a
versioned effective-Hamiltonian schema and a fragment-solver callback returning
energy plus one- and two-particle density matrices. QSCI, selected CI, and the
hybrid should implement the same callback.

---

## 3. Manuscript positioning changes

1. **Paper B must confront QSCI and classical selected CI.** A QSCI comparison
   alone is insufficient once the hybrid dresses determinants.
2. **Paper A's novelty is certification, not generic measurement efficiency.**
   CEO-ADAPT constrains the pool-design claim; a benchmark can follow Track A.
3. **Ancilla-free A-CASE is a resource-boundary statement, not a zero-T-count
   statement.**
4. **Configuration Haar packets are an opt-in coarse basis with a measured
   trade-off, not a universal wavelet advantage.**
5. **Orbital basis is a recorded parameter.** The repository already contains a
   negative result against standardizing wavelet orbitals.
6. **BK-trap claims are excluded unless independent mapping and gradient
   invariants reproduce them.**
7. **FLDC is positioning only.** A-CASE has no parameter-training landscape;
   ADAPT may cite finite-depth trainability without creating a new task.

---

## 4. Falsifiable questions

Continuing the numbering from `ACASE_RESEARCH_PLAN.md` §7. The question numbers
Q5–Q10 preserve the identities used in the earlier literature roadmap; new
integrated questions begin at Q11.

- **Q5 — QSCI dominance:** at equal `M`, does QSCI match or beat A-CASE on
  fermionic chemistry while spending zero measured words on projected matrices,
  once preparation, sampling, and classical costs are also reported?
- **Q6 — hybrid versus classical closure:** does the dressed hybrid reach a
  Pareto point unavailable to bare QSCI, selected CI, and A-CASE, and is its
  span different from or more compact than determinant excitation closure?
- **Q7 — overlap selection:** does a QSCI/selected-CI target expose useful
  candidates that lowering-only growth misses on `hubbard_2x3`?
- **Q8 — real-time family:** can a real-time or controlled short-time family
  recover fixed-Krylov accuracy at a `kappa(S)` and propagation error budget a
  finite-shot calculation could survive?
- **Q9 — Clifford grouping:** does fully commuting grouping reduce certified
  leading shot cost with covariance and circuit overhead accounted for?
- **Q10 — parity ceiling:** does `r_X <= 2(N - 1)` hold across every declared
  spin-conserving Jordan–Wigner Hamiltonian construction path?
- **Q11 — packet gain:** do Haar-stage policies survive ordering ablations and
  improve the final Pareto frontier rather than one finite instance only?
- **Q12 — spin sampled subspaces:** is computational-basis sampled
  diagonalization noncompact on Kitaev, rather than nonexistent?
- **Q13 — mapping invariance:** do JW, BK, and parity reproduce exact energies
  and equivalent fermionic gradients under consistent transforms?

---

## 5. Recommended implementation order

1. Add QSCI contracts and exact sampled-subspace restriction.
2. Integrate QSCI into the ladder, including a raw spin-system arm.
3. Add excitation-closure and selected-CI controls.
4. Add QSCI configuration generators and dressed families.
5. Add span/principal-angle diagnostics.
6. Integrate the existing Haar packet tier as a staged hybrid arm.
7. Add overlap-targeted scoring and ordering ablations.
8. Run the complete Track A ladder and reposition Paper B.
9. Implement the explicit X-rank invariant and fully commuting grouping.
10. Add second moments, time-evolved inputs, mapping breadth, and embedding only
    after the Paper B result is known.
11. Benchmark CEO and MORE-ADAPT only after the critical comparison is stable.

---

## 6. What this roadmap does not claim

It does not claim to originate QSCI, selected CI, overlap-guided adaptation,
Haar transforms, folded-spectrum methods, real-time Krylov, fully commuting
measurement, BK mapping, CEO operators, MORE-ADAPT, or embedding.

The possible contributions are narrower:

- a resource-honest comparison of sampled determinant and measured operator
  subspaces;
- a tested hybrid whose distinction from classical determinant closure is made
  explicit;
- a support-pruned multiresolution staging policy built on the repository's
  existing virtual-configuration transform;
- certified measurement and convergence diagnostics around those methods.

Failure remains acceptable. If QSCI or classical selected CI dominates
chemistry and the hybrid span collapses to ordinary excitation closure without a
compactness advantage, Paper B must narrow its claim rather than hide the
comparison.
