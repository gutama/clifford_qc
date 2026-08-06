# `clifford_qc` integrated research roadmap — QSCI, A-CASE, and multiresolution methods

This document supersedes the former Phases 8–14 literature roadmap. It keeps
QSCI/SQD as the critical scientific comparison for Paper B, but updates the
implementation plan to the repository as it exists after PRs #31–#35:

- orbital bases are explicit and benchmarked; orbital wavelets were a negative
  result rather than a default;
- configuration-space Haar packets are implemented as a classical basis change
  among virtual determinant generators;
- ADAPT-VQE and A-CASE now expose separate config/state/step boundaries;
- generator domains are split into response, configuration, and fermionic
  modules;
- full-space and sector state-vector action are matrix-free.

The central research question is now:

> **Can sampled determinant subspaces, operator-response dressing, and
> support-pruned multiresolution packets produce a compact, resource-honest
> eigensolver that adds something classical selected CI does not already give?**

The roadmap deliberately separates the Paper B critical path from measurement
and long-horizon infrastructure. Wavelets enter only where the current code has
already shown a defensible use: configuration-space coarse-to-fine selection.
They do not replace QSCI, and they are not treated as a universal orbital basis.

---

## 0. Current repository facts that determine the plan

### 0.1 QSCI remains missing

`clifford_qc` still has no first-class QSCI/SQD implementation and no QSCI row
on the A-CASE ladder. This remains the principal external-validity gap for
Paper B.

QSCI and A-CASE spend different resources:

| | A-CASE | QSCI/SQD |
|---|---|---|
| basis | virtual operator states `A_i|psi>` | sampled basis configurations |
| overlap | measured nonorthogonal `S` | identity |
| projected Hamiltonian | reconstructed from measured Pauli words | built classically |
| main quantum cost | state preparation plus grouped word measurement | state preparation plus computational-basis sampling |
| principal risk | `W`, shot cost, and conditioning | duplicate sampling and determinant compactness |

`W=0` for QSCI projected-matrix measurement is correct, but it is not a full
resource verdict. The comparison must also report sampling yield, preparation
cost, classical matrix construction, diagonalization cost, and memory.

### 0.2 Configuration-space Haar is already implemented

`subspace.configuration.configuration_haar_packets` builds an orthogonal finite
tree-Haar transform over a caller-ordered configuration list. It is a classical
change of basis among virtual generators, not a quantum wavelet circuit.
Support pruning makes it an opt-in coarse tier rather than a convergence-complete
basis.

The committed `2x2` Hubbard benchmark already provides one positive finite
instance: staging Haar packets before the ordinary level-4 pool reaches the
exact sector energy with 33.1% fewer projected Pauli words and one fewer basis
direction, while increasing maximum element support and `kappa(S)`. This result
justifies integrating packets into the QSCI hybrid as an ablation; it does not
justify making them the default.

### 0.3 Orbital wavelets are not the integration target

The orbital-basis benchmark found that no basis wins uniformly and that the
tested Daubechies orbital bases lose to site or momentum bases on the relevant
trade-offs. Orbital basis therefore remains a recorded model parameter.

The integrated roadmap does **not** reopen orbital-wavelet optimization unless
a new system provides a specific falsifiable reason.

### 0.4 Matrix-free action changes the feasible boundary

`PauliLinearOperator` and `SectorOperator` now provide matrix-free action and
Krylov solves without materializing dense operators. This is the correct
foundation for:

- exact QSCI sampling-state oracles on small and medium systems;
- restriction to sampled index sets;
- propagation diagnostics;
- residual and variance checks.

It does not make dense real-time operators cheap in the multivector generator
bank. Circuit-native real-time A-CASE remains a separate architectural problem.

### 0.5 Adaptive workflows now have clean extension seams

`AdaptConfig`/`AdaptState`/`adapt_step` and
`ACASEConfig`/`ACASEState`/`acase_step` allow hierarchical or racing policies to
be added without unifying the different mathematics of ADAPT-VQE and A-CASE.
Multiresolution selection should target these workflow boundaries rather than
introduce a generic adaptive runner.

---

## 1. Citation and claim discipline

The literature table contains fifteen papers, not fourteen. The roadmap should
refer to fifteen throughout.

The Bravyi–Kitaev preprint arXiv:2606.05968 must be downgraded further than in
the previous roadmap. Its stated fixed-UCCSD gradient and ADAPT commutator
gradient are the same derivative for the same anti-Hermitian generator, while
its claimed FCI states retain large commutator gradients. It is therefore not
evidence for an intrinsic BK symmetry trap or one-cycle FCI convergence.

Use it only to motivate mapping-consistency regression tests:

- exact energy invariance across mappings;
- correctly encoded reference states;
- equality of finite-difference, analytic, and commutator gradients;
- separate measurement of Pauli weight, word count, and grouping cost.

The withdrawn arXiv:2606.30551 must not enter either bibliography. Recent
unrefereed compact-QSCI and Clifford-measurement results may be cited only with
reproduced invariants separated from unreproduced benchmark claims.

---

## 2. Programme structure

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
3. mapping breadth;
4. time-evolved QSCI inputs;
5. embedding interface.

Track A should not wait for Tracks B or C.

---

# Track A — QSCI, classical controls, and the hybrid

## Phase 8 — trusted sampled-subspace baseline

Add `clifford_qc/subspace/qsci.py` as a first-class method rather than a
benchmark stub.

### 8A — sampling contract

Define a result object carrying at least:

- raw shots;
- accepted shots;
- unique basis configurations;
- duplicate fraction;
- discarded or repaired fraction;
- cumulative retained probability;
- sampled subspace dimension `M`;
- projected-matrix measurement words `W=0`;
- classical matrix nonzeros, build time, solve time, and peak memory;
- energy, variational gap, and evidence label.

Initial implementation should sample from exact probabilities generated by
existing state backends. Hardware-noise emulation and configuration recovery
are later layers over the same contract.

### 8B — sampled Hamiltonian restriction

Reuse `sector_basis`, sector indexing, and the compiled `SectorOperator`. Add a
safe API that restricts a sector operator to a declared set of sector indices.
For the first implementation, an exact row/column restriction of the validated
sector operator is preferable to writing Slater–Condon rules prematurely.

Required invariants:

1. the sampled Hamiltonian is Hermitian;
2. increasing nested sampled sets gives non-increasing Ritz energies;
3. every sampled Ritz value remains above the exact sector value;
4. selecting the entire sector reproduces the sector spectrum;
5. permutation of sampled configuration order changes no eigenvalue.

### 8C — fermionic recovery and generic spin sampling

For fermionic systems, implement post-selection and optional recovery against
particle number and `S_z`, reporting all discarded or repaired samples.

For spin systems such as Kitaev, do **not** report that QSCI has no arm. The
fermionic SQD recovery rule is unavailable, but raw computational-basis sampled
subspace diagonalization is still a valid baseline. The scientific question is
whether that basis is compact, not whether it is definitionally excluded.

### 8D — state inputs

Run the QSCI arm with several declared inputs:

- reference determinant;
- exact ground-state sampling oracle, for method validation only;
- existing ADAPT-VQE state;
- later, a time-evolved state.

Never mix oracle and implementable inputs in the same evidence category.

### 8E — ladder integration

Add QSCI to `benchmarks/run_acase_ladder.py`. Equal-`M` remains one comparison,
but the ladder must also emit Pareto records for:

- error versus quantum shots;
- error versus unique configurations;
- error versus state-preparation cost;
- error versus classical matrix nonzeros and solve time;
- error versus memory.

**Go/no-go:** the full-sector limit and variational bound must hold on H4,
Hubbard, and at least one spin model before QSCI is used in manuscript claims.

---

## Phase 9 — classical selected-CI controls

This phase is mandatory. Without it, a successful QSCI x A-CASE hybrid may be
indistinguishable from ordinary determinant-space expansion.

For each sampled determinant set `D`, construct the following controls:

1. **QSCI:** diagonalize only `span(D)`.
2. **Excitation closure:** add every unique determinant reached by the same
   singles/doubles used for operator dressing.
3. **One-step selected CI:** add determinants using a declared HCI-, CIPSI-, or
   perturbative-style classical score.
4. **Budget-matched selected CI:** stop at the same final determinant count or
   classical matrix cost as the hybrid.

Record determinant count, Hamiltonian nonzeros, classical selection work,
energy, variance, and memory.

### Span-equivalence diagnostic

For each dressed family, compare

`span{E_mu |D_k>}`

with the ordinary determinant closure generated from the same `D_k` and
excitation operators. Compute numerical ranks and principal angles.

Interpretation:

- equal spans mean the operator form is a representation or measurement-cost
  choice, not a richer variational space;
- a smaller operator-generated basis spanning a much larger determinant closure
  is a valid compactness result;
- directions outside the declared determinant closure require a precise
  algebraic explanation and an independent check.

**Go/no-go:** Phase 10 may claim a hybrid gain only after it beats or differs
structurally from these controls.

---

## Phase 10 — QSCI x A-CASE hybrid

### 10A — sampled configurations as generators

Convert retained QSCI configurations through `configuration_generator`. This
uses the current configuration module directly; no new generic generator type
is needed.

### 10B — operator-response dressing

Build declared dressed families such as:

- configuration x conserving excitation;
- configuration x commutator response;
- optional support-bounded compound families.

Every family must report candidate count, generator support `S_A`, projected
element support `S_H`, incremental word universe, and conditioning impact.

### 10C — three required hybrid arms

Compare:

1. bare sampled configurations;
2. sampled configurations plus individual dressed generators;
3. sampled configurations plus support-pruned Haar packets, followed by the
   ordinary dressed pool.

The third arm integrates the existing wavelet result without making wavelets a
separate state-vector compression project.

### 10D — honest claim

The pre-registered claim is narrower than before:

> A sampled determinant set, enriched by selected operator-response directions,
> may reach a target accuracy with fewer retained variational directions or a
> better measured-resource Pareto point than either bare QSCI or bare A-CASE.

Do not claim that operator dressing is strictly richer until the Phase 9 span
analysis proves it.

**Primary systems:** `hubbard_2x2`, `hubbard_2x3`, H4 equilibrium/stretched, and
one molecular FCIDUMP rung with matched multiplicity.

---

## Phase 11 — overlap-targeted and multiresolution selection

### 11A — overlap target

Extend A-CASE scoring with a target-overlap criterion using the QSCI Ritz vector
or a classical selected-CI vector as the target. Preserve existing
scale-invariance and orthogonality rejection.

The target object should expose coefficients in a declared basis and a method
for overlap with a candidate virtual state. Keep it outside `CandidateScore` so
the lowering and overlap criteria remain independently testable.

### 11B — coarse-to-fine packet selection

Use the existing configuration Haar transform as a hierarchy:

1. order sampled configurations using declared physics metadata;
2. score support-pruned coarse packets first;
3. refine only selected or competitive packet intervals;
4. hand the retained basis to the convergence-complete individual/dressed pool.

Candidate blocks may use the new pure `acase_step` boundary, but the score within
a block remains A-CASE-specific. Do not create a shared ADAPT/A-CASE runner.

### 11C — ordering ablations

The Haar transform does not infer locality. Compare at least:

- probability order;
- excitation rank plus occupation-pattern metadata;
- determinant-graph traversal;
- random order controls.

A packet result is publishable only if it is not an accident of one favorable
ordering.

### 11D — stopping and uncertainty

For sampled configurations, track unseen or unstable probability mass using a
simple baseline first: duplicate-rate stopping, bootstrap set stability, or an
unseen-mass estimate. Add wavelet/block allocation only if it demonstrably
improves those controls.

**Go/no-go:** on `hubbard_2x3`, the new criterion or hierarchy must select useful
configuration/dressed directions that the lowering-only individual pool misses,
or the blind spot is attributed to the family rather than the selector.

---

## Phase 12 — integrated Paper B ladder

The final Track A ladder contains:

- reference state;
- exact sector result;
- fixed QSE and Krylov;
- ADAPT-VQE;
- A-CASE;
- QSCI;
- excitation-closure CI;
- one-step/budget-matched selected CI;
- QSCI x dressed A-CASE;
- QSCI x Haar-stage x dressed A-CASE.

Required output fields include:

- `M`, retained rank, and `kappa(S)`;
- energy error, variance or true residual where available;
- sampling shots, unique yield, duplicate rate, discard/recovery rate;
- state-preparation metadata;
- `W`, grouping contexts, and certified shot cost for measured arms;
- classical matrix nonzeros, build/solve time, and peak memory;
- generator and element supports;
- evidence category and seed.

Paper B should be rewritten around the Pareto frontier that survives. If QSCI
and classical selected CI dominate chemistry, narrow A-CASE to the systems and
representations where operator-generated or packet directions add measurable
value.

---

# Track B — Clifford-accessible measurement

## Phase 13 — structural invariant first

Implement GF(2) rank of Hamiltonian X masks using the existing `word_masks`
path. Test the parity/X-rank ceiling across spin-conserving JW Hamiltonians,
FCIDUMP models, lattice models, and effective-Hamiltonian ingestion.

This is cheap and should precede the grouping implementation.

## Phase 14 — QWC plus fully commuting groups

Extend measurement grouping with fully commuting groups and Clifford
simultaneous diagonalization.

The comparison metric is not group count alone. Report certified leading shot
cost at fixed word universe, allocator, and confidence target, plus:

- diagonalizing circuit depth and two-qubit gates;
- connectivity assumptions;
- covariance-aware reconstruction;
- Monte Carlo agreement between predicted and empirical uncertainty.

The previous multiply-capable-word variance bug remains the gate: a lower group
count with inflated or double-counted variance is failure.

This track supports Paper A or a separate measurement paper and must not block
Track A.

---

# Track C — longer-horizon infrastructure

## Phase 15 — second-moment bank

Add a `SecondMomentBank` for

`K_ij = <psi|A_i^dagger H^2 A_j|psi>`.

Before building the full bank, add a support/cost preflight for `H^2`. If the
estimated word universe is prohibitive, keep dense or matrix-free residual
oracles for validation and restrict the measured implementation to declared
small systems.

Uses:

- true Ritz residual norms;
- energy variance;
- variance extrapolation for A-CASE, QSCI, and selected-CI controls;
- folded-spectrum roots;
- an independent convergence criterion.

## Phase 16 — time-evolved inputs, split by method

### 16A — QSCI input

Use matrix-free `expm_multiply`, Krylov propagation, or a validated Trotter
circuit to generate time-evolved sampling states. This is a feasible QSCI input
study and may proceed independently.

### 16B — A-CASE real-time generators

Do not represent `exp(-iHt)` as an `MV` by default: Pauli support can become
dense. A circuit-native generator would require a different matrix-element
backend and resource model. Treat this as architecture research, not as an
incremental generator family.

A short-time polynomial response may be tested only with explicit truncation,
norm, fidelity, and energy-error budgets against matrix-free propagation.

## Phase 17 — mapping validation and breadth

Add mapping-consistency tests before using BK or parity in scientific records:

- transform Hamiltonian, reference, and generators consistently;
- verify energy and gradient invariance;
- compare Pauli weight, distinct words, grouping, and circuits separately.

Lower Pauli weight does not imply lower `W` or fewer groups; all three must be
measured independently.

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
3. **Ancilla-free A-CASE is a circuit/resource-boundary statement, not a
   zero-T-count statement.**
4. **Configuration-space Haar packets are an opt-in coarse basis with a measured
   trade-off, not a universal wavelet advantage.**
5. **Orbital basis is a recorded parameter.** The repository already contains a
   negative result against standardizing wavelet orbitals.
6. **BK-trap claims are excluded unless independent mapping and gradient
   invariants reproduce them.**

---

## 4. Falsifiable questions

- **Q5 — QSCI dominance:** does QSCI match or beat A-CASE on fermionic chemistry
  once both quantum and classical resources are reported?
- **Q6 — classical closure:** is the dressed hybrid span different from, or more
  compact than, the corresponding determinant excitation closure?
- **Q7 — hybrid Pareto gain:** does the hybrid reach an accuracy/resource point
  unavailable to QSCI, selected CI, and A-CASE separately?
- **Q8 — packet gain:** do Haar-stage policies survive ordering ablations and
  improve the final Pareto frontier rather than one finite instance only?
- **Q9 — overlap selection:** does the QSCI target expose useful candidates that
  lowering-only growth misses on `hubbard_2x3`?
- **Q10 — spin sampled subspaces:** is computational-basis sampled
  diagonalization noncompact on Kitaev, rather than nonexistent?
- **Q11 — Clifford grouping:** does fully commuting grouping reduce certified
  shot cost with covariance and circuit overhead accounted for?
- **Q12 — mapping invariance:** do JW, BK, and parity reproduce exact energies and
  equivalent fermionic gradients under consistent state/operator transforms?

---

## 5. Recommended implementation order

1. Add QSCI data contracts and exact sampled-subspace restriction.
2. Integrate QSCI into the ladder, including a raw spin-system arm.
3. Add excitation-closure and selected-CI controls.
4. Add QSCI configuration generators and dressed families.
5. Add span/principal-angle diagnostics.
6. Integrate the existing Haar packet tier as a staged hybrid arm.
7. Add overlap-targeted scoring and ordering ablations.
8. Run the complete Track A ladder and reposition Paper B.
9. Implement the X-rank invariant and fully commuting grouping.
10. Add second moments, time-evolved inputs, mapping breadth, and embedding only
    after the Paper B result is known.

---

## 6. What this roadmap does not claim

It does not claim to originate QSCI, selected CI, overlap-guided adaptation,
Haar transforms, folded-spectrum methods, real-time Krylov, fully commuting
measurement, BK mapping, or embedding.

The possible contributions are narrower:

- a resource-honest comparison of sampled determinant and measured operator
  subspaces;
- a tested hybrid whose distinction from classical determinant closure is made
  explicit;
- a support-pruned multiresolution staging policy built on the repository's
  existing virtual-configuration transform;
- certified measurement and convergence diagnostics around those methods.

Failure remains an acceptable result. In particular, if QSCI or classical
selected CI dominates chemistry and the hybrid span collapses to ordinary
excitation closure without a compactness advantage, Paper B must narrow its
claim rather than hide the comparison.
