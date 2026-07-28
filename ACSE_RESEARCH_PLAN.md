# `clifford_qc` research plan — Adaptive Clifford Subspace Eigensolver and the correlated-materials solver role

This plan is the successor roadmap to `RESEARCH_PLAN.md` (Paper A:
confidence-certified, measurement-efficient ADAPT-VQE). It fixes the
project's next scientific identity:

> **Do low-energy states of correlated molecular and materials Hamiltonians
> have a compact adaptive Clifford-orbit subspace representation — and can
> that subspace be diagonalized reliably from globally shared, finite-shot
> Pauli measurements with explicit certificates?**

It supersedes the free-form notes in `future_clifford_qc_research_plan.pdf`
and revises that document's ordering in one essential way: the **Adaptive
Clifford Subspace Eigensolver (ACSE) comes first**, and matrix-free Lanczos
is demoted to a reference baseline built later on a sector-restricted spinor
backend. The architecture audit behind this reversal is recorded in §2.

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

1. **Adaptive Clifford Subspace Eigensolver (ACSE)** — Rayleigh–Ritz in an
   adaptively grown, operator-generated subspace. Primary research
   contribution; Paper B candidate.
2. **Exact references** — dense `exact_ground` (now), PySCF FCI metadata
   (now), and later a sector-restricted spinor backend with matrix-free
   Lanczos for n beyond dense reach. Baselines, not the identity.
3. **ADAPT-VQE** (Paper A machinery, complete) — the variational comparison
   point at matched operator budget.

## 2. Why ACSE first (architecture audit)

The PDF ordering ("implement matrix-free Pauli Lanczos first") predates the
audit of what the merged codebase already provides. The decisive facts:

- `states.expectation(rho, O) = Tr(O·rho)` is **complex-valued and does not
  require O Hermitian** (`states.py`). With `MV.dagger()` and the word
  product, every ACSE matrix element
  `H_ij = ⟨ψ|A_i†HA_j|ψ⟩ = Tr(A_i†HA_j·ρ)` and
  `S_ij = Tr(A_i†A_j·ρ)` is computable **today**, with zero new backends.
  (`ExactMVBackend.expectation` forces `.real`; ACSE calls the core
  function, not the backend wrapper.)
- Because `Tr(W·ρ) = 2^n · [coefficient of I in W·ρ]` and `w·w = I` for
  Pauli words, every matrix element reduces to a **word-coefficient dot
  product**: `Tr(O·ρ) = 2^n Σ_w O[w]·ρ[w]`. Exact evaluation (against the
  state's word coefficients) and finite-shot evaluation (against measured
  word means) are the *same* linear reconstruction — the structure
  `CommutatorBank` already implements for gradients, generalized from
  candidate-by-word to matrix-entry-by-word.
- Matrix-free Lanczos, by contrast, requires a representation the package
  does not have: a sector-indexed state vector and a Pauli-on-statevector
  kernel. That is a new substrate (§5, Phase 6), not a matvec swap.
- ACSE reuses, nearly unchanged: packed word product, QWC grouping,
  `GroupedWordCache` covariance, adaptive shot allocation, simultaneous
  confidence bounds, and the evidence taxonomy
  (exact / finite-sample / asymptotic / abstained) from Paper A.

Scientific positioning: quantum subspace expansion, quantum Krylov
(Stair/Motta; Kirby et al.), and residual-driven subspace growth
(Jacobi–Davidson) are established. The defensible novelty is the
**statistical layer**: covariance-aware reconstruction of (H, S) from
globally shared Pauli measurements, a stable noisy generalized eigensolver
with stated conditions, and **finite-sample certificates (with explicit
abstention) for basis-growth decisions**. The subspace construction is the
enabling machinery, not the headline claim. Epperly, Lin & Nakatsukasa
(2022) supply the exact-arithmetic stability theory for thresholded
projection; the noisy, correlated-error extension is the open problem this
project attacks.

## 3. Conceptual layer (geometric-algebra contract)

Every phase below states its algebra, objects, and validation invariants
before its implementation substrate.

**Algebra.** `Cl(2n,ℂ) ≅ M(2^n,ℂ)` throughout. Two representations of the
*same* algebra are used, and they must not be conflated:

| Representation | Object | Storage | Role |
|---|---|---|---|
| Operator-centric (current) | density multivector `ρ ∈ Cl(2n,ℂ)`, Pauli-word basis | up to `4^n` words | operators, ADAPT, ACSE matrix elements |
| Witt-ideal spinor (Phase 6) | `Ψ ∈ Cl(2n,ℂ)·P₀`, `P₀ = ∏_j c_j c_j†` | `2^n` ideal components; `C(n,k)` in a particle sector | large-n pure states, Lanczos |

The Witt/minimal-left-ideal representation is the GA-native name for the
"symmetry-restricted determinant basis": computational determinants are the
occupation words `(c†)^{x₁}…(c†)^{xₙ}·P₀`, particle-number and S_z sectors
are subspaces of the ideal spanned by fixed-weight occupation words, and
Jordan–Wigner dressing is built into the Witt basis rather than bolted on.
Adding it is not a departure from the operator-centric identity; it is the
minimal left ideal of the same algebra, and it removes the `4^n`
density-word blow-up for strongly correlated pure states that the
operator-centric picture cannot avoid.

**Objects.**

- Pauli words: blades (up to phase) under the JW correspondence; the new
  exterior layer (`reverse`, `wedge`, `scalar_product`, `is_blade`) makes
  grade/blade structure first-class.
- ACSE basis generators `A_i`: multivectors from the hierarchy in §4.2 —
  identity, pool words `P_i`, commutator residuals `G_i = -i/2·[H,P_i]`,
  selected products. Compound generators are **versor-like orbits** of the
  reference: products of rotors and Pauli words acting on `|ψ_ref⟩`.
- Stabilizer members of the basis (Clifford transforms of `|0…0⟩`) admit
  tableau-cheap overlaps through the existing Stim bridge; they are the
  "competing mean-field / magnetic-order configurations" of the materials
  strategy.

**Standing invariants** (checked in tests, not prose):

- `E_sub ≥ E₀` for every exact-arithmetic subspace (variational bound);
  monotone non-increasing under nested basis growth.
- `S ⪰ 0`, `S = S†` exactly; `H = H†` exactly (Hermiticity of the
  reconstructed pair is enforced structurally, not numerically).
- Ritz residual `‖(H−E)|Ψ⟩‖² = ⟨H²⟩−⟨H⟩²` matches the independently
  computed energy variance.
- Reproduction: subspace containing the exact ground state returns the FCI
  energy to solver tolerance (validated against `exact_ground` and the
  PySCF `fci_energy` metadata for every chemistry model).
- Do **not** truncate by GA grade. Grade is not a good quantum number for
  JW-dressed Hamiltonians; truncation criteria are particle number, S_z,
  Pauli support, excitation rank, residual coupling, and conditioning.

## 4. The method

### 4.1 Core loop

Given reference state ρ (pure), Hamiltonian H, and current generator set
`{A_i}`:

1. Build `O_ij^S = A_i†A_j` and `O_ij^H = A_i†HA_j` as sparse MVs (cached;
   products computed once, Hermitian pairs share work).
2. Assemble `S_ij`, `H_ij` by word-coefficient pairing (exact) or shared
   measurement reconstruction (finite-shot, §4.4).
3. Solve the thresholded generalized eigenproblem: hermitize, diagonalize
   `S = UΛU†`, drop modes with `λ_k < τ_S`, form
   `H̃ = Λ^{-1/2}U†HUΛ^{-1/2}`, diagonalize `H̃`.
4. Select the next generator by **certified residual coupling or predicted
   Ritz lowering** (§4.3); add it, or abstain and stop.

### 4.2 Basis hierarchy

- Level 0: current reference `|ψ⟩` (HF, ADAPT warm start, or stabilizer
  configuration).
- Level 1 (tangent): `P_j|ψ⟩` — the tangent directions of the Pauli-rotor
  ansatz at θ=0.
- Level 2 (residual): `G_j|ψ⟩` with `G_j = -i/2·[H,P_j]` — rows already
  materialized by `CommutatorBank`.
- Level 3 (Krylov enrichment): `H^k|ψ⟩` as one candidate family among
  others, not the organizing principle.
- Level 4 (compound orbits): selected `P_iP_j|ψ⟩`, `P_iG_j|ψ⟩`,
  stabilizer configurations `V|0…0⟩` for competing orders. Only certified
  candidates enter; support growth is a tracked resource (S_max metric).

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

Finite-shot selection reuses the Paper A best-arm machinery: simultaneous
bounds on `{ΔE_a}` or `{|r_a|}`, four-way outcome, explicit abstention when
growth cannot be certified within budget.

### 4.4 Finite-shot matrix elements

Off-diagonal `O_ij` are generally non-Hermitian. Split
`O^R = (O+O†)/2`, `O^I = (O−O†)/(2i)` — both Hermitian — and reconstruct
`H_ij = ⟨O^R⟩ + i⟨O^I⟩` from the shared word cache. Because all entries
draw on one global word universe, entry errors are **correlated**; the
covariance tensor over (H, S) entries is propagated from the per-group
joint histograms (`GroupSample`) that the grouped cache already retains.
Noisy S needs Hermitian symmetrization, eigenvalue flooring/PSD projection,
and honesty: the Ritz value after noisy PSD repair is **not** automatically
a variational upper bound — quantifying when the bound survives (bias
bounds vs. τ_S, shot budget, and conditioning) is a headline research task,
not an implementation detail.

## 5. Implementation plan

Phases are ordered by dependency; each has a go/no-go invariant. Phase 1 is
deliberately thin: if the exact subspace converges poorly on the target
pools, everything downstream is renegotiated before it is built.

**Phase 0 — done (merged main).** Exterior layer on `MV` (reversion, wedge,
k-vector dot, blade tests), O(1) exact gradients, deterministic selection,
`fermionic_sector_diagnostics`, gate preconditions, generated paper tables.

**Phase 1 — thin exact ACSE (`subspace/solver.py`).**
Fixed, explicitly listed generators; build (H, S) via `states.expectation`;
thresholded GEP; `SubspaceResult(energies, coefficients, basis_labels,
overlap_eigenvalues, condition_number, residual_norms)`.
*Validate:* `E_sub ≥ E₀`; nested monotonicity; FCI reproduction on H₂ and
LiH(2e,2o); agreement with a dense-matrix GEP cross-check.
*Go/no-go:* Level-1+2 bases reach chemical accuracy on H₂/H₄ with basis
size ≪ sector dimension. (≈100 lines + tests; no new abstractions.)
*Preliminary evidence:* `examples/acse_premise_check.py` runs the full
premise on TFIM n=4 from the `|0…0⟩` reference with only existing
machinery: the variational bound and nested monotonicity hold at every
level, the level-0..3 hierarchy closes the gap from 1.76 to 1.8×10⁻²,
and 37 generators yield only 12 independent directions — the
near-singular-S regime that makes adaptive, conditioning-aware selection
(Phase 2) the load-bearing component rather than an optimization.

**Phase 2 — adaptive growth (`subspace/adaptive.py`).**
Residual-coupling and generalized-2×2 selection, cost-aware score,
linear-dependence rejection, `energy_history`, warm start from an ADAPT
state. *Validate:* adaptive ≤ fixed basis at equal size; matches or beats
ADAPT-VQE at matched operator budget on H₄ and one spin model.

**Phase 3 — `MatrixElementBank` (`subspace/elements.py`).**
Generalize `CommutatorBank`: entry-by-word coefficient maps for S and H
with Hermitian/anti-Hermitian channels over one shared word universe;
products `A_i†A_j`, `A_i†HA_j` computed once and cached; exact path
reproduces Phase 1 bit-for-bit. *Validate:* reconstruction equals direct
`states.expectation` on random bases.

**Phase 4 — finite-shot certified ACSE.**
Shared QWC measurement of the bank's word universe; covariance tensor of
(H, S) entries from `GroupSample` histograms; noisy-S regularization with
conditioning diagnostics; uncertainty on Ritz energies (linearization +
bootstrap cross-check); certified basis growth with abstention, reusing
the evidence taxonomy. *Validate:* empirical coverage of Ritz-energy
intervals on H₂/H₄ at known shot budgets; abstention triggers under
adversarial near-degenerate bases. This phase is the Paper B core.

**Phase 5 — materials models and observables.**
`models/lattice.py`: Hubbard, extended Hubbard, Kanamori, small Anderson
impurity, Kitaev honeycomb cluster (native `PauliSum`s; spin models need no
JW). `observables.py`: `⟨n_i⟩`, `⟨n_i n_j⟩`, `⟨S_i·S_j⟩`, structure
factors, charge/spin gaps from Ritz spectra — all plain PauliSum
expectations of ACSE eigenvectors, so they ride on Phases 1–4 unchanged.
Ingestion: FCIDUMP and PySCF active-space import via the OpenFermion
bridge, with orbital/site/sector metadata on `Model`.

**Phase 6 — Witt-ideal spinor backend + matrix-free Lanczos.**
Sector-restricted ideal representation (occupation words of fixed particle
number / S_z), Pauli-word action as bit-mask gather with phase
accumulation grouped by X-mask, `scipy.sparse.linalg.LinearOperator` +
`eigsh` (own Lanczos fallback optional). This is the exact-reference tier
for n beyond dense reach and the substrate DMFT-style repeated solves
would need. *Validate:* matches `exact_ground` for n ≤ 12; sector
projector idempotence; memory `C(n,k)` not `2^n`.
*Explicitly after* the ACSE prototype: it serves baselines and large-n
extension, not the main claim. An interim pragmatic baseline (scipy sparse
matrix + `eigsh`, ~20 lines) is acceptable wherever dense `eigh` runs out
before Phase 6 lands.

**Phase 7 — validation ladder and Paper B.**
H₂ → H₄ → stretched H₂O CAS(4e,4o) → H₂O CAS(8e,6o) → 2×2/2×3 Hubbard →
one Kitaev cluster with spin correlations. Comparison set: HF, exact
diagonalization/FCI, plain QSE, fixed Krylov, ADAPT-VQE (exact and
finite-shot), ACSE (exact and finite-shot certified). Report energies,
basis sizes, shots, circuits, support peaks, and abstention rates.

## 6. Falsifiable questions

- **Q1 (compactness).** Does adaptive selection reach chemical accuracy
  with materially smaller bases than fixed QSE/Krylov at equal generator
  budget? *Falsifier:* no gap on the Phase 7 ladder.
- **Q2 (certification).** Do finite-shot Ritz intervals achieve nominal
  coverage while abstention prevents uncertified growth? *Falsifier:*
  coverage collapse or near-total abstention at realistic budgets.
- **Q3 (bound survival).** Under what measurable conditions does the
  variational upper bound survive noisy PSD repair? *Deliverable:* a
  bias bound in terms of τ_S, shot covariance, and conditioning — or a
  documented counterexample family.
- **Q4 (materials reach).** Do competing-order stabilizer configurations
  plus residual directions compactly represent low-energy states of the
  Hubbard/Kitaev clusters? *Falsifier:* basis growth tracking sector
  dimension.

## 7. What this plan does not claim

No claim to: the first subspace/QSE method, the first quantum Krylov, a
speedup over dense linear algebra at small n, DFT replacement, or an
exponential-complexity escape via geometric algebra. The Clifford
representation is an algebraic backend that makes the measurement-sharing
and certification layers natural; the physics guarantees (variational
bounds, Ritz theory) come from the same eigensolver principles as always.
