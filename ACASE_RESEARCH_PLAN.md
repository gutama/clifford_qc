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
| `trace_pairing` (Phase 1, shipped) | `Σ_w a_w b_w` | bilinear, no reversion, no conjugation (`Tr(AB)/2^n`) |

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

**Phase 1 — done (`clifford_qc/subspace/`).** `MV.trace_pairing`;
`solver.py` with the normalized, thresholded, deterministic GEP of §4.1 and
`SubspaceResult(energies, coefficients, basis_labels, overlap_eigenvalues,
condition_number, effective_rank, resources)`; `generators.py` for the
level-0..3 families of §4.2; `reference.py` for the dense-matrix
cross-check (materialize `|ψ⟩`, materialize every `A_i|ψ⟩`, form the Gram
and Hamiltonian matrices directly); `models.chemistry.excitation_multivectors`
for the symmetry-preserving chemistry mode. Two assembly routes, verified
equal: the element-operator route forms `A_i†A_j` and `A_i†HA_j` (the
operators Phase 2 caches and Phase 4 must measure, and the only route that
can report `W` and `S_H`), the cyclic route contracts
`Tr(A_i†HA_j ρ) = Tr((HA_j)(ρA_i†))` in `2M` products and `M²` sparse
pairings and is correspondingly blind to those metrics.

*Validated* (`tests/test_subspace.py`, `tests/test_subspace_chemistry.py`,
and the `A-CASE` section of `clifford_qc.verify`): `E_sub ≥ E₀` and nested
monotonicity on TFIM, XXZ, and both H₄ legs; exact agreement with the dense
route; `S = S†` bitwise (structural, from the upper-triangle layout, not a
numerical symmetrization); FCI reproduction on H₂ and LiH(2e,2o) to 1e-9
with four generators, and by a ground-state-projector generator on TFIM;
invariance of the retained subspace, its spectrum, and `κ_S` under random
complex generator rescaling; deterministic eigenbases inside degenerate
overlap and Ritz eigenspaces (canonicalized from the spectral projector, so
independent of the LAPACK basis); Ritz states staying in the reference
`(N, S_z)` sector.

*Measured* (H₄ chain, sto-3g, 8 qubits; `E₀ = -2.180317` at r=0.9 and
`-1.924431` at r=1.8; `S_A = max_i |supp(A_i)|`):

| fixed basis | M | rank | ΔE (r=0.9) | ΔE (r=1.8) | κ_S | S_A |
|---|---|---|---|---|---|---|
| A-CASE symmetry-preserving level 1 | 27 | 27 | 7.7×10⁻⁴ | 3.5×10⁻² | 1.0 | 8 |
| word-level QSE, matched budget | 27 | 11 | 4.8×10⁻² | 1.6×10⁻¹ | 8.0 | 1 |
| word-level QSE, full odd-Y pool | 161 | 27 | 7.7×10⁻⁴ | 3.5×10⁻² | 8.0 | 1 |
| fixed Krylov `H^k`, k ≤ 6 | 7 | 7 | 1.5×10⁻⁶ | 2.5×10⁻⁴ | 1.0×10⁸ | 4224 |
| fixed Krylov `H^k`, k ≤ 10 | 11 | 9 | 5.3×10⁻⁹ | 6.4×10⁻⁵ | 3.4×10¹⁰ | 4224 |

*Go/no-go: conditionally met, and not in the way the criterion assumed.*
The symmetry-preserving level-1 basis reaches chemical accuracy at
equilibrium with `M = 27` against a 36-state `(N=4, S_z=0)` sector, spans
the same subspace as the 161-word QSE pool at a sixth the basis size, and
does so at `κ_S = 1` — but it does **not** beat fixed Krylov on energy per
basis vector, at either geometry. Krylov wins that column by orders of
magnitude. What it pays is exactly the §6 currency: generators 500× wider
(`S_A = 4224` vs 8, so wide that the element-operator route is not
affordable on H₄ at all, while the A-CASE basis assembles in seconds) and
`κ_S` of 10⁸–10¹⁰, which is the conditioning regime where noisy PSD repair
and finite-shot certification (Q2, Q3) are least likely to survive. So the
compactness claim Q1 is **not** established by fixed bases: it rests on
adaptive selection, which is Phase 3. Recorded here rather than smoothed
over — a fixed basis was never the claim, and Krylov's energy advantage at
catastrophic conditioning is itself the argument for conditioning-aware
growth.

*TFIM premise check.* `examples/acase_premise_check.py` now runs on the
shipped solver: the variational bound and nested monotonicity hold at every
level, the level-0..3 hierarchy closes the gap from 1.76 to 1.8×10⁻², and 37
generators yield only 12 independent directions — the near-singular-S regime
that makes conditioning-aware adaptive selection the load-bearing component
rather than an optimization.

**Phase 2 — done (`subspace/elements.py`).** `MatrixElementBank`:
canonical generator IDs (a repeated operator returns the id it already has;
a label rebound to a different operator is an error, since labels are what
records report); cached `A_i†A_j` and `A_i†HA_j` with Hermitian-pair reuse
(upper triangle only — `S_ji = conj(S_ij)` is a property of the layout);
global word-union tracking, including new-versus-reused words per accepted
generator; exact `trace_pairing` assembly reproducing Phase 1 **bit for
bit** — the product order is deliberately identical, since floating-point
addition is not associative and a "mathematically equivalent"
rearrangement would make the two routes' records irreproducible;
support/conditioning metrics per build, plus cached-operator bytes and an
opt-in QWC group count (quadratic in `W`, so never a hidden cost). No
finite-shot machinery: the cached coefficient maps *are* the sufficient
statistics Phase 4 will reconstruct from measured word means, so that layer
attaches without disturbing this one.

Pair products are lazy, so a candidate that is never scored costs nothing —
what Phase 3 needs when it evaluates and rejects.

**Projected observables (§8) ship with it.** `project_observable(Q) → Q_sub`
through the same element machinery, then `result.expectation(Q, k)` and
`result.transition(Q, i, j)` contract it with the Ritz coefficients. The
Ritz state is never formed; the tests check the answers against the dense
state A-CASE refuses to store. A Hermitian `Q` is mirrored from its upper
triangle like `(S, H)`; a non-Hermitian one is not (`Q_sub[j,i]` is then an
independent element), and `expectation` refuses it rather than quietly
returning the real part. Each observable's word universe, and the words it
adds beyond what `(S, H)` already require, enter the §6 accounting.

*Measured (the reason the bank comes before adaptive growth).* Solving every
nested prefix of a basis — the access pattern Phase 3 generates — costs
`M(M+1)(M+2)/6` pair products when each solve reassembles, and `M(M+1)/2`
through the bank:

| trajectory | pair products | wall clock |
|---|---|---|
| TFIM n=4, M=37, reassembling | 9139 | 0.28 s |
| TFIM n=4, M=37, banked | 703 | 0.15 s |
| H₄ r=0.9, M=27, reassembling | 3654 | 13.9 s |
| H₄ r=0.9, M=27, banked | 378 | 2.3 s |

The gap widens with `M` (the ratio is `(M+2)/3`) and with generator width,
which is why the TFIM speedup is modest — its solves are dominated by the
eigendecomposition, not the products — while H₄'s is 6×. The cost side is
memory, and it is not small: H₄'s 378 cached element operators hold 15847
distinct words and ~10.6 MB. That figure is the §6 metric to watch as
Phase 3 grows bases, not a footnote.

**Phase 3 — done (`subspace/adaptive.py`).** `run_acase` grows the basis one
generator at a time on top of the bank: generalized-2×2 predicted lowering
(closed form, overlap block carried explicitly), scale-free residual
coupling, linear-dependence rejection, `energy_history`, per-step
`GrowthRecord`s carrying conditioning and word costs, an optional cost-aware
score `ΔE/(1+new words)^γ`, `fermionic_excitation_generators` as the
chemistry default with `sector_leakage` reported per accepted generator (and
sector-breaking candidates rejected when `leakage_tol` is set), and
`adapt_warm_start` for growing around an ADAPT-VQE state.

Three details are load-bearing, and each is pinned by a test that fails
under the obvious alternative:

- *The overlap block is not optional.* Assume the candidate is orthonormal
  to the current Ritz vector, and a candidate that **is** that vector times
  3.5 scores over a Hartree of predicted gain; the generalized 2×2 scores
  exactly zero.
- *Rejection is measured against the retained subspace, not the Ritz
  vector.* A candidate duplicating some other basis direction sits at a
  perfectly healthy angle to the Ritz vector, passes the weaker test, and
  makes `S` singular — the thresholded solve then discards it after it has
  been paid for.
- *The 2×2 deflation needs a floor.* Below an orthogonal fraction of
  ~1e-12 the deflated diagonal is a genuine 0/0, and double precision
  returns noise that is not small: a parallel candidate lands at −6 instead
  of −4, two Hartree of fabricated lowering. The conditioning floor sits
  four orders above it, so live scoring never reaches the cliff.

*Validated:* `E_sub ≥ E₀` and monotone `energy_history`; **predicted
lowering ≤ actual lowering** at every step (`span{Ψ_m, χ}` sits inside
`span{basis ∪ χ}`, so the 2×2 can only underestimate); adaptive ≤ fixed
basis at equal size on TFIM, XXZ, and both H₄ legs; deterministic and
scale-invariant selection; convergence cross-checked against the dense
`dense_residual_norm` that §4.4 keeps out of the projected API.

*Measured, TFIM n=4 from the model's own reference `|++++⟩` — the state
ADAPT-VQE also starts from (`examples/acase_adaptive.py`):*

| M | A-CASE | κ_S | fixed Krylov | κ_S | fixed QSE | ADAPT-VQE |
|---|---|---|---|---|---|---|
| 3 | 4.1×10⁻⁸ | 3.6×10² | 2.3×10⁻² | 1.8×10² | 7.6×10⁻¹ | 2.7×10⁻¹ |
| 5 | 4.0×10⁻¹⁰ | 8.8×10² | 1.6×10⁻⁶ | 1.2×10⁴ | 5.2×10⁻¹ | 1.3×10⁻² |
| 7 | 8.9×10⁻¹⁶ | 9.2×10² | −1.8×10⁻¹⁵ | 1.8×10⁷ | 5.2×10⁻¹ | 7.1×10⁻¹⁵ |

The §5 criterion is met on the spin model: A-CASE matches or beats every
baseline at matched budget, and where fixed Krylov finally catches up it
does so at `κ_S = 1.8×10⁷` against A-CASE's `9.2×10²` — five orders of
conditioning, the currency Q2 and Q3 are denominated in. Warm-starting from
a 2-operator ADAPT state improves M=5 further, 4.0×10⁻¹⁰ → 2.0×10⁻¹¹.

*Measured, H₄ chain (symmetry-preserving candidates, `leakage_tol=1e-9`;
every accepted generator leaks < 1e-12 and `κ_S = 1` throughout):*

| M | A-CASE (r=0.9) | fixed prefix | ADAPT-VQE | A-CASE (r=1.8) | ADAPT-VQE |
|---|---|---|---|---|---|
| 4 | 1.87×10⁻² | 5.61×10⁻² | 1.87×10⁻² | 8.54×10⁻² | 3.86×10⁻² |
| 6 | 9.36×10⁻³ | 5.61×10⁻² | 9.72×10⁻³ | 4.09×10⁻² | 2.33×10⁻² |
| 9 | 3.02×10⁻³ | 5.61×10⁻² | 2.38×10⁻³ | — | — |

Against the fixed prefix the gain is decisive at every size — the natural
ordering emits singles first, and on a closed-shell determinant those
contribute almost nothing, so the fixed basis stalls at 5.6×10⁻² while
adaptive selection takes the doubles that matter. Against ADAPT-VQE the
honest reading is a draw at equilibrium (ahead at M=6, behind at M=9) and a
**loss on the stretched geometry**, where ADAPT reaches 3.9×10⁻² against
A-CASE's 8.5×10⁻² at M=4. A linear span of singles and doubles on an HF
reference is the wrong object for a strongly multireference state; the
plan's answer is compound generators and competing-order references (§4.2
level 4, Phase 5), not more of the same family. Recorded rather than
smoothed over.

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
