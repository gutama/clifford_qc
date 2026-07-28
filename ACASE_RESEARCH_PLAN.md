# `clifford_qc` research plan — A-CASE: Adaptive Clifford-Algebra Subspace Eigensolver

This plan is the successor roadmap to `RESEARCH_PLAN.md` (Paper A:
confidence-certified, measurement-efficient ADAPT-VQE). It fixes the
project's next scientific identity:

> **Do low-energy states of correlated molecular and materials Hamiltonians
> have a compact adaptive representation in Clifford-algebra
> operator-response subspaces — and can that subspace be diagonalized
> reliably from globally shared, finite-shot Pauli measurements with
> explicit certificates?**

It supersedes the free-form notes in `future_clifford_qc_research_plan.pdf`
and revises that document's ordering in one essential way: **A-CASE comes
first**, and matrix-free Lanczos is demoted to a reference baseline built
later on a sector-restricted spinor backend. The architecture audit behind
this reversal is recorded in §2.

**Naming.** The method is A-CASE, never "ACSE": in quantum chemistry ACSE
is the anti-Hermitian contracted Schrödinger equation (Mazziotti and
successors), still active in contracted quantum eigensolver research, and
colliding with it would corrupt literature searches and referee context.
Terminology discipline: the working space is a **Clifford-algebra
operator-response subspace**. `P_j|ψ⟩` are Pauli-orbit directions;
`G_j|ψ⟩` with `G_j = -i/2·[H,P_j]` are commutator-response directions
(Pauli sums, not Clifford transformations); `H^k|ψ⟩` are Krylov response
states; genuine **Clifford-group orbit** states (versor/stabilizer
transforms of a reference) are one candidate family among several, not the
generic case. Compound generators need not be versors.

---

## 1. Scope and positioning

`clifford_qc` becomes a **correlated active-space and lattice-model
solver**, not a general materials suite:

```text
DFT / Wannier / embedding  →  effective many-body Hamiltonian
                           →  clifford_qc (this plan)
                           →  energies, states, correlations, response
```

It does not compute band structures, forces, phonons, or geometry
optimization; established DFT codes own that layer. `clifford_qc` solves the
small-but-hard correlated subproblem (active spaces, Hubbard/Kanamori
clusters, impurity models, spin lattices) that downfolding or embedding
produces, and returns energies, low-lying spectra, and material observables.

**Method stack (three tiers, revised order):**

1. **A-CASE** — Rayleigh–Ritz in an adaptively grown, operator-generated
   subspace. Primary research contribution; Paper B candidate.
2. **Exact references** — dense `exact_ground` (now), PySCF FCI metadata
   (now), a minimal scipy-sparse `eigsh` reference (before larger materials
   validation), and later a sector-restricted spinor backend with
   matrix-free Lanczos for n beyond dense reach. Baselines, not the
   identity.
3. **ADAPT-VQE** (Paper A machinery, complete) — the variational comparison
   point at matched operator budget.

## 2. Why A-CASE first (architecture audit)

The PDF ordering ("implement matrix-free Pauli Lanczos first") predates the
audit of what the merged codebase already provides. The decisive facts:

- **The basis states stay virtual.** `|φ_i⟩ = A_i|ψ⟩` is never prepared;
  every projected element is an expectation **on the single reference
  state**: `S_ij = ⟨ψ|A_i†A_j|ψ⟩`, `H_ij = ⟨ψ|A_i†HA_j|ψ⟩`. One reference,
  one global Pauli-word universe, one measurement cache, every measured
  word reused across many entries and candidates. This — not the mere use
  of Clifford algebra — is the project's strongest computational
  proposition, and it is exactly the `CommutatorBank` architecture
  generalized from candidate-by-word to matrix-entry-by-word.
- `states.expectation(rho, O) = Tr(O·rho)` is complex-valued and does not
  require O Hermitian (`states.py`). With `MV.dagger()` and the word
  product, every matrix element is computable **today**, with zero new
  backends. (`ExactMVBackend.expectation` forces `.real`; A-CASE calls the
  core function, not the backend wrapper.)
- In the Pauli-word coordinates, `Tr(O·ρ) = 2^n Σ_w o_w·r_w` — a
  **bilinear, conjugation-free coefficient pairing**. Exact evaluation
  (against the state's word coefficients) and finite-shot evaluation
  (against measured word means) are the same linear reconstruction.
- Matrix-free Lanczos, by contrast, requires a representation the package
  does not have: a sector-indexed state vector and a Pauli-on-statevector
  kernel. That is a new substrate (Phase 6), not a matvec swap.
- A-CASE reuses the packed word product, QWC grouping, the grouped cache's
  joint histograms, adaptive shot allocation, and the evidence taxonomy
  (exact / finite-sample / asymptotic / abstained) from Paper A — though
  finite-shot *certification* is a genuinely new statistical problem
  (Phase 4), not a drop-in reuse of the gradient confidence code.

**Prior art and the novelty boundary.** Quantum subspace expansion, quantum
Krylov (Stair/Motta; Kirby et al.), residual-driven growth
(Jacobi–Davidson), adaptive generator-coordinate subspaces built from
UCC/ADAPT generators, circuit-generated non-orthogonal VQE subspaces
(NOQE-style), and virtual/noise-stabilized QSE all use nonorthogonal
operator- or circuit-generated bases with generalized eigenproblems. The
defensible novelty is therefore narrow and must stay narrow: **globally
shared Pauli reconstruction of all projected matrices, covariance-aware
noisy generalized eigensolving, and statistically certified adaptive basis
growth with explicit abstention.** Do not dilute it by overemphasizing that
the basis is operator-generated. Epperly, Lin & Nakatsukasa (2022) supply
the exact-arithmetic stability theory for thresholded projection; the
noisy, correlated-error extension is the open problem this project attacks.

## 3. Conceptual layer (geometric-algebra contract)

Every phase below states its algebra, objects, and validation invariants
before its implementation substrate.

**Algebra.** `Cl(2n,ℂ) ≅ M(2^n,ℂ)` throughout. Two representations of the
*same* algebra are used, and they must not be conflated:

| Representation | Object | Storage | Role |
|---|---|---|---|
| Operator-centric (current) | density multivector `ρ ∈ Cl(2n,ℂ)`, Pauli-word basis | up to `4^n` words | operators, ADAPT, A-CASE matrix elements |
| Witt-ideal spinor (Phase 6) | `Ψ ∈ Cl(2n,ℂ)·P₀`, `P₀ = ∏_j c_j c_j†` | `2^n` ideal components; `C(n,k)` in a particle sector | large-n pure states, Lanczos |

The Witt/minimal-left-ideal representation is the GA-native name for the
"symmetry-restricted determinant basis": computational determinants are the
occupation words `(c†)^{x₁}…(c†)^{xₙ}·P₀`, particle-number and S_z sectors
are subspaces of the ideal spanned by fixed-weight occupation words, and
Jordan–Wigner dressing is built into the Witt basis rather than bolted on.
Adding it is not a departure from the operator-centric identity; it is the
minimal left ideal of the same algebra, and it removes the `4^n`
density-word blow-up for strongly correlated pure states. Its engineering
surface should nevertheless be plain (`SectorStatevectorBackend`); the
ideal language belongs to the theory sections, not the API.

**The three pairings.** `MV` now carries two coefficient pairings, and
A-CASE needs a third; conflating them produces silent conjugation or sign
errors:

| Pairing | Formula (word coordinates) | Character |
|---|---|---|
| `scalar_product` | `Σ_w a_w b_w (−1)^{k_w(k_w−1)/2}` | bilinear, reversion sign (`⟨A~B⟩₀`) |
| `hs_product` | `Σ_w conj(a_w) b_w` | sesquilinear (`Tr(A†B)/2^n`) |
| `trace_pairing` (Phase 1) | `Σ_w a_w b_w` | bilinear, no reversion, no conjugation (`Tr(AB)/2^n`) |

Matrix elements need `trace_pairing`: `A_i†HA_j` is non-Hermitian, so its
word coefficients are complex and `hs_product` would conjugate them
wrongly, while Hermitian test cases (real coefficients) would mask the bug.
`trace_pairing` must be a distinct primitive, not an overload, and it is
the exact foundation of the matrix-element bank. It also avoids forming
the full product `O·ρ` just to read one scalar.

**Objects.**

- Pauli words: blades (up to phase) under the JW correspondence; the
  exterior layer (`reverse`, `wedge`, `scalar_product`, `is_blade`) makes
  grade/blade structure first-class.
- A-CASE generators `A_i`: multivectors from the hierarchy in §4.2.
  Stabilizer configurations (genuine Clifford-group orbits of `|0…0⟩`)
  admit tableau-cheap overlaps through the existing Stim bridge; they are
  the "competing mean-field / magnetic-order configurations" of the
  materials strategy.

**Standing invariants** (checked in tests, not prose):

- `E_sub ≥ E₀` for every exact-arithmetic subspace (variational bound);
  monotone non-increasing under nested basis growth.
- `S ⪰ 0`, `S = S†` exactly; `H` Hermitian exactly (enforced structurally,
  not numerically).
- Generator-scaling invariance: replacing `A_i` by `cA_i` must not change
  which physical directions survive thresholding (§4.1 normalization).
- Reproduction: a subspace containing the exact ground state returns the
  FCI energy to solver tolerance (validated against `exact_ground` and
  PySCF `fci_energy` metadata for every chemistry model).
- Do **not** truncate by GA grade. Grade is not a good quantum number for
  JW-dressed Hamiltonians; truncation criteria are particle number, S_z,
  Pauli support, excitation rank, residual coupling, and conditioning.

## 4. The method

### 4.1 Core loop

Given reference state ρ (pure), Hamiltonian H, and current generator set
`{A_i}`:

1. Build `O_ij^S = A_i†A_j` and `O_ij^H = A_i†HA_j` as sparse MVs, cached
   in the bank (products computed once; Hermitian pairs share work).
2. Assemble `S_ij`, `H_ij` by `trace_pairing` (exact) or shared
   measurement reconstruction (finite-shot, Phase 4).
3. Solve the **normalized**, thresholded generalized eigenproblem:
   - drop zero-norm rows; scale `D_ii = √S_ii`, `S̄ = D⁻¹SD⁻¹`,
     `H̄ = D⁻¹HD⁻¹` — thresholding raw `S` would make the retained
     subspace depend on arbitrary generator scaling;
   - hermitize; diagonalize `S̄ = UΛU†`; drop modes by absolute threshold
     `τ_S`, relative threshold, and a maximum retained condition number;
   - form `H̃ = Λ^{-1/2}U†H̄UΛ^{-1/2}`, diagonalize;
   - fix deterministic eigenvector phases and a deterministic order inside
     degenerate overlap eigenspaces (same discipline as the deterministic
     ADAPT selection on main);
   - record effective rank before and after truncation.
4. Select the next generator by certified residual coupling or predicted
   Ritz lowering (§4.3); add it, or abstain and stop.

### 4.2 Basis hierarchy

- Level 0: current reference `|ψ⟩` (HF, ADAPT warm start, or stabilizer
  configuration).
- Level 1 (tangent / Pauli-orbit): `P_j|ψ⟩` — tangent directions of the
  Pauli-rotor ansatz at θ=0.
- Level 2 (commutator response): `G_j|ψ⟩` — rows already materialized by
  `CommutatorBank`.
- Level 3 (Krylov response): `H^k|ψ⟩` as one candidate family among
  others, not the organizing principle.
- Level 4 (compound / Clifford-group orbits): selected `P_iP_j|ψ⟩`,
  `P_iG_j|ψ⟩`, stabilizer configurations `V|0…0⟩` for competing orders.
  Only certified candidates enter; support growth is a tracked resource.

**Chemistry basis modes.** Two distinct modes, with the second as the
chemistry default:

- *Word-level benchmark mode*: individual Pauli words `P_j|ψ⟩`, for direct
  comparison with the current qubit-ADAPT pools.
- *Symmetry-preserving mode*: `T_μ|ψ⟩` where `T_μ` is the **complete** JW
  image of a particle-number- and S_z-conserving fermionic excitation,
  kept as one multivector rather than split into words. The repository
  already documents why this matters (`fermionic_sector_diagnostics`:
  a word split from a conserving generator need not itself conserve `N`
  or `S_z`); without it the subspace can gain energy by leaking into
  unphysical sectors. Candidate records report particle-number and S_z
  leakage (via the existing diagnostic), the sector of each stabilizer
  configuration, and rejection of sector-incompatible candidates.

### 4.3 Selection criterion

For candidate `|χ_a⟩` against current Ritz pair `(E_m, |Ψ_m⟩)`:

- residual coupling `r_a = ⟨χ_a|(H−E_m)|Ψ_m⟩`;
- predicted lowering ΔE_a from the **generalized** 2×2 problem in
  `span{Ψ_m, χ_a}` — both blocks, `(E_m, h_a; h_a*, h_aa)` against
  `(1, s_a; s_a*, s_aa)`. The overlap block is mandatory: without it the
  score is biased exactly in the near-linearly-dependent direction the
  method must reject;
- acceptance score `ΔE_a / cost_a^γ`, with rejection when the candidate's
  S-orthogonal component falls below the conditioning floor.

### 4.4 Residual norms need a second-moment bank

The true Ritz residual `‖(H−E)|Ψ⟩‖² = (c†Kc)/(c†Sc) − E²` requires the
second-moment matrix `K_ij = ⟨ψ|A_i†H²A_j|ψ⟩`, which the projected `(H,S)`
pair cannot supply and whose Pauli support can be much larger. Policy:
`residual_norms` is **optional** — computed either from a later
`SecondMomentBank` or, in small exact-validation runs, from a dense
reconstructed state. Never expose the *projected* residual under that
name: it is zero by construction for a solved Ritz pair and says nothing
about error outside the subspace.

## 5. Implementation plan

Phases are ordered by dependency; each has a go/no-go invariant. The bank
comes **before** adaptive growth: adaptive selection repeatedly adds rows
and columns, and without cached pair products it would recompute `A_i†A_j`
and `A_i†HA_j` many times — making a sound method look uncompetitive
because of a deliberately temporary implementation.

**Phase 0 — done (merged main).** Exterior layer on `MV` (reversion,
wedge, k-vector dot, blade tests), O(1) exact gradients, deterministic
selection, `fermionic_sector_diagnostics`, gate preconditions, generated
paper tables.

**Phase 1 — exact fixed-basis A-CASE (`subspace/solver.py`).**
`MV.trace_pairing`; the normalized, thresholded, deterministic GEP of
§4.1; identity plus explicitly listed generators; `SubspaceResult(
energies, coefficients, basis_labels, overlap_eigenvalues,
condition_number, effective_rank, resources)`; dense-matrix GEP
cross-check.
*Targets:* H₂, equilibrium and stretched H₄, LiH(2e,2o).
*Validate:* `E_sub ≥ E₀`; nested monotonicity; FCI reproduction;
generator-scaling invariance of the retained subspace.
*Go/no-go:* Level-1+2 bases reach chemical accuracy with basis size ≪
sector dimension **and** with the §6 resource metrics staying measurably
below competing fixed QSE/Krylov constructions.
*Preliminary evidence:* `examples/acase_premise_check.py` runs the full
premise on TFIM n=4 from the `|0…0⟩` reference with only existing
machinery: the variational bound and nested monotonicity hold at every
level, the level-0..3 hierarchy closes the gap from 1.76 to 1.8×10⁻²,
and 37 generators yield only 12 independent directions — the
near-singular-S regime that makes conditioning-aware adaptive selection
the load-bearing component rather than an optimization.

**Phase 2 — exact `MatrixElementBank` (`subspace/elements.py`).**
Canonical generator IDs; cached `A_i†A_j` and `A_i†HA_j` with
Hermitian-pair reuse; global word-union tracking; exact `trace_pairing`
assembly reproducing Phase 1 bit-for-bit; support/conditioning metrics
(§6) recorded per build; **projected observable API** (§8):
`project_observable(Q) → Q_sub`, `result.expectation(Q)`,
`result.transition(Q, i, j)`. No finite-shot machinery yet.

**Phase 3 — exact adaptive growth (`subspace/adaptive.py`).**
Residual-coupling and generalized-2×2 selection on top of the bank;
symmetry-preserving chemistry generators as default with leakage
reporting; linear-dependence rejection; `energy_history`; warm start from
an ADAPT state. *Validate:* adaptive ≤ fixed basis at equal size; matches
or beats ADAPT-VQE, fixed QSE, and fixed Krylov at matched operator
budget on H₄ and one spin model.

**Phase 4 — finite-shot layers.** Certification here is a **new nonlinear
statistical problem**: `(H,S)` are estimated, the retained eigenspace is
data-dependent, the Ritz pair `(c,E)` is data-dependent, and residual
couplings and 2×2 lowerings are nonlinear functions of correlated
estimates. The gradient best-arm code assumed linear estimators; it does
not transfer unchanged. Staged route:

- *4A — shared grouped measurement (exact statistics deferred):* measure
  the bank's word universe through QWC groups; verify exact adaptive
  behavior is recovered in the infinite-shot limit; establish support and
  word-growth scaling.
- *4B — asymptotic uncertainty:* delta-method Ritz uncertainties from
  grouped joint histograms; grouped bootstrap cross-check; empirical
  coverage studies; every interval labeled `asymptotic` or `heuristic` —
  never `certified`.
- *4C — finite-sample growth certificate via sample splitting:* a
  construction batch estimates `(H,S)`, fixes the thresholded subspace,
  and freezes `(c,E)`; an independent certification batch estimates
  candidate residual couplings with `(c,E)` treated as constants;
  simultaneous empirical-Bernstein bounds decide growth or **abstention**.
  Less measurement-efficient than unrestricted reuse, but it yields a
  defensible first theorem; confidence-set reuse can follow.

**Covariance discipline:** never materialize a dense covariance tensor
over `(H,S)` entries (`O(M²)` entries, `O(M⁴)` pairs). Keep the grouped
joint histograms as the sufficient statistic and expose
covariance-vector / Jacobian-vector products for arbitrary linear
functionals of word means — a generalization of the candidate-specific
`GroupedWordCache` API — computing only what Ritz linearization, one
candidate's residual, pairwise comparisons, and bootstrap draws need.
The plan keeps its honesty: PSD repair of noisy `S` does not
automatically preserve the variational upper bound; asymptotic
uncertainty and finite-sample certification stay separated in the API.

**Phase 5 — materials models and observables.**
`models/lattice.py`: Hubbard, extended Hubbard, Kanamori, small Anderson
impurity, Kitaev honeycomb cluster (native `PauliSum`s; spin models need
no JW). Observables through the **projected-matrix route** (§8) — the
Ritz state is never materialized. Excited states via state-averaged or
block adaptation (grow the basis against several Ritz roots, not only the
lowest). Ingestion: FCIDUMP and PySCF active-space import via the
OpenFermion bridge, with orbital/site/sector metadata on `Model`. A
minimal scipy-sparse `eigsh` reference (~20 lines) lands before the
larger materials runs, wherever dense `eigh` runs out.

**Phase 6 — `SectorStatevectorBackend` + matrix-free Lanczos.**
Sector-restricted ideal representation (occupation words of fixed particle
number / S_z), Pauli-word action as bit-mask gather with phase
accumulation grouped by X-mask, `scipy.sparse.linalg.LinearOperator` +
`eigsh` (own Lanczos fallback optional). Exact-reference tier for n beyond
dense reach and the substrate DMFT-style repeated solves would need.
*Validate:* matches `exact_ground` for n ≤ 12; sector projector
idempotence; memory `C(n,k)` not `2^n`. Explicitly after the A-CASE
prototype: it serves baselines and large-n extension, not the main claim.

**Phase 7 — validation ladder and Paper B.**
H₂ → H₄ → stretched H₂O CAS(4e,4o) → H₂O CAS(8e,6o) → 2×2/2×3 Hubbard →
one Kitaev cluster with spin correlations. Comparison set: HF, exact
diagonalization/FCI, plain QSE, fixed Krylov, generator-coordinate-style
fixed subspaces, ADAPT-VQE (exact and finite-shot), A-CASE (exact and
finite-shot certified). Report energies **and** the §6 resource metrics,
shots, circuits, and abstention rates.

## 6. Resource accounting (equal partner to basis size)

A 20-dimensional basis is not compact if its projected entries carry
millions of words. Every run records:

- `M` — basis size;
- `W` — size of the global word universe
  `|⋃_ij supp(O_ij^H) ∪ supp(O_ij^S)|`;
- `S_A = max_i |supp(A_i)|` and `S_H = max_ij |supp(A_i†HA_j)|`;
- `r_S` — retained rank at threshold τ; `κ_S` — retained condition number;
- bank build time and peak memory; reused vs. newly introduced words per
  accepted generator; QWC group count.

The compactness question is not "is `M` small?" but: **does the adaptive
basis stay small while its projected operator bank stays measurably
smaller than competing QSE/Krylov constructions?**

## 7. Falsifiable questions

- **Q1 (compactness).** Does adaptive selection reach chemical accuracy
  with materially smaller `M` *and* `W` than fixed QSE/Krylov at equal
  generator budget? *Falsifier:* no gap on the Phase 7 ladder.
- **Q2 (certification).** Do finite-shot Ritz intervals achieve nominal
  coverage while abstention prevents uncertified growth? *Falsifier:*
  coverage collapse or near-total abstention at realistic budgets.
- **Q3 (bound survival).** Under what measurable conditions does the
  variational upper bound survive noisy PSD repair? *Deliverable:* a bias
  bound in terms of τ_S, shot covariance, and conditioning — or a
  documented counterexample family.
- **Q4 (materials reach).** Do competing-order stabilizer configurations
  plus response directions compactly represent low-energy states of the
  Hubbard/Kitaev clusters? *Falsifier:* basis growth tracking sector
  dimension.

## 8. Projected observables

Material observables are conceptually `PauliSum` expectations of Ritz
states, but the Ritz state is never stored natively. For every observable
`Q`, build `Q_sub[i,j] = ⟨ψ|A_i†QA_j|ψ⟩` through the same bank machinery
and evaluate `⟨Q⟩_k = (c_k†Q_sub c_k)/(c_k†Sc_k)` and transition elements
between Ritz roots. The extra word and measurement cost of each projected
observable enters the §6 accounting.

## 9. What this plan does not claim

No claim to: the first subspace/QSE method, the first quantum Krylov or
non-orthogonal eigensolver, a speedup over dense linear algebra at small
n, DFT replacement, or an exponential-complexity escape via geometric
algebra. The Clifford representation is an algebraic backend that makes
the measurement-sharing and certification layers natural; the physics
guarantees (variational bounds, Ritz theory) come from the same
eigensolver principles as always.
