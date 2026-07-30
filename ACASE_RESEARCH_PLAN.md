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

### 1.1 Minimum viable materials-facing showcase — shipped

`models/effective.py` is the narrow boundary the diagram above was missing. It
reads the versioned schema `clifford_qc.effective_hamiltonian.v1`: a Hermitian,
spin-independent Wannier one-body matrix, local `U_i`, chemical potential,
energy-unit label, explicit interleaved spin-orbital reference, and optional
provenance. It builds the ordinary second-quantized Hamiltonian and maps it
through the package's existing Jordan-Wigner layer. It does **not** claim to
run DFT, Wannierization, cRPA, or DMFT.

The canonical integration rung is the two-site Hubbard dimer (`t=1`, `U=4`
eV, four qubits, sector `N=2, S_z=0`). It is the smallest system that exercises
the whole output contract while retaining an independent oracle:

| output | A-CASE showcase | independent check |
|---|---:|---:|
| ground energy | `-0.828427125 eV` | `(U-sqrt(U^2+16t^2))/2`, and the sector-exact backend |
| average double occupancy | `0.073223305` | `dE_0/dU / 2` (Hellmann-Feynman) |
| `<S_0.S_1>` | `-0.640165043` | `-3/4 (1-2d)` |
| `<S^2>` | `< 1e-12` | singlet invariant |
| staggered-spin line | `omega=0.828427125 eV` | `0 - E_0` (the `S_z=0` triplet sits at 0) |
| staggered-spin weight | `0.853553391` | `1-2d`, and the Lehmann sum rule |
| `chi(0)` | `2.060660172 1/eV` | `2w/omega` |
| response basis | `M=4`, rank `4`, `kappa(S)=1` | sector dimension `4` |

Every row but the last is checked against arithmetic that shares no code with
the projected-observable route that produced it, which is what makes it an
oracle: comparing a projected observable against another projected observable
would agree with itself under a systematic error. The last row is not a
compactness result and is not offered as one -- `M = 4` **is** the sector
dimension, so the response arm is full configuration interaction in that
sector, chosen deliberately so the Lehmann lines are exact while the adaptive
arm is scored separately on the energy.

The benchmark is intentionally labelled synthetic: it proves that an effective
Hamiltonian can cross the software boundary and produce energies, projected
state coefficients, correlations, and response. It does not establish
materials accuracy or a quantum advantage.

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
- Level 4 (compound / Clifford-group orbits) — **done
  (`subspace/generators.py`, `models/lattice.py`)**: selected `P_iP_j|ψ⟩`,
  `P_iG_j|ψ⟩` via `compound_response` (deduplicated up to a scalar, since the
  solver normalizes and `P_jP_i` is `±P_iP_j`; `max_support` and
  `max_generators` make the width and the pool size declared rather than
  discovered), and stabilizer configurations `V|0…0⟩` via
  `configuration_generator`. A configuration enters as the operator that
  *reaches* it: with `|ψ⟩ = R|0…0⟩` and `|φ⟩ = V|0…0⟩` the generator is
  `A = VR†`, so `A|ψ⟩ = |φ⟩` exactly and no second state is prepared. Between
  two determinants that operator is a single `X`-string, so a whole competing
  order costs `S_A = 1`. `models.lattice.competing_orders` names them for a
  Hubbard cluster (antiferromagnet on the *bond-coloured* sublattice — site
  index parity gets the 2×2 grid wrong — its spin-flipped partner, two charge
  density waves, and the stripe the numbering produces), emitting only
  configurations at the reference's own `(N, S_z)`.

  *Configurations need a different sector test, and this is the trap.*
  `sector_leakage` asks whether the **operator** commutes with `N`; an
  `X`-string does not, so a leakage filter rejects every competing-order
  configuration — the same failure mode as the Kitaev misconfiguration in
  Phase 7, in a new place. What matters is the plan's own wording, the sector
  of the configuration, so `state_sector` reports `⟨N⟩`, `⟨S_z⟩` **and their
  variances** for `A|ψ⟩` as expectations on the reference (nothing prepared).
  The competing orders come out sharp: variance `0` to machine precision.

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

**Phase 4 — done (`subspace/measured.py`).** Certification here is a **new
nonlinear statistical problem**: `(H,S)` are estimated, the retained
eigenspace is data-dependent, the Ritz pair `(c,E)` is data-dependent, and
residual couplings and 2×2 lowerings are nonlinear functions of correlated
estimates. The gradient best-arm code assumed linear estimators; it did not
transfer unchanged. What did transfer is the shape: everything the subspace
needs is a linear functional of Pauli-word means, and `WordFunctional` is
that object — the single place shots enter, with `estimate`, `variance`, a
covariance-vector product, an exact (infinite-shot) evaluation, and
arithmetic, so a difference of functionals is a functional.

- *4A — shared grouped measurement.* `SharedMeasurement` measures a
  subspace's whole word universe through QWC groups and reconstructs every
  entry from the same shots; the lower triangle stays the conjugate by
  construction rather than an independent noisy estimate. Its
  `exact_matrices()` walks the same reconstruction with exact means and
  reproduces the Phase-2 matrices to 5×10⁻¹³, which is the acceptance
  criterion for the infinite-shot limit. The identity word is never
  measured: `⟨I⟩ = 1` is known, and reporting it as measured would inflate
  the empirical-Bernstein range of every group that reads it.
- *4B — asymptotic uncertainty.* `ritz_uncertainty` linearizes:
  `dE = Σ_w q_w dμ_w` with `q` the word coefficients of `B†(H−E)B`,
  `B = Σ_i c_i A_i` — one *real* functional (that operator is Hermitian), so
  the Jacobian of the Ritz value with respect to every word mean is a single
  bank-derived object. `bootstrap_ritz` resamples the grouped histograms and
  reruns the whole nonlinear pipeline as the cross-check. Both are labelled
  `asymptotic` / `heuristic`; `Interval.certified` is False for both.
- *4B-R — nonlinear response uncertainty (`subspace/measured_response.py`).*
  `ResponseMeasurement` extends the shared universe to a Hermitian projected
  observable `Q_sub`; `bootstrap_response` resamples the grouped joint
  histograms and reruns `(S,H,Q_sub)` reconstruction, overlap thresholding,
  the generalized eigensolve, transition amplitudes, squared weights, gaps,
  susceptibility, and optional broadening. Root-resolved lines are emitted only
  for isolated ordered roots; rank changes and root collisions are counted.
  The result is always `heuristic` and never certified. A finite-sample
  response certificate remains a separate matrix-pencil confidence-set problem.
- *4C — finite-sample growth certificate by sample splitting.*
  `run_certified_acase` spends a construction batch on `(S,H)`, freezing the
  thresholded subspace, its Ritz pair, **and the candidate norms**; an
  independent certification batch bounds each candidate's residual coupling
  with those held constant. `|r| = √(Re² + Im²)` is not linear, so the
  interval is a rectangle over the two real functionals with the union bound
  paid explicitly over `2·(#candidates)` events. Growth happens only when a
  candidate's lower bound clears the threshold; otherwise the run
  **abstains** and stops.

The norm subtlety is worth stating because it was easy to get wrong: the
coupling must be normalized by `‖A_a|ψ⟩‖` or the ranking would depend on how
a candidate happens to be scaled, but dividing by a quantity estimated from
the *same* batch would make the statistic a ratio of correlated estimates and
void the certificate. The construction batch is what makes the norm a
constant.

**Covariance discipline** (honored): no dense covariance over `(H,S)` entries
is ever formed. The grouped joint histograms stay the sufficient statistic and
`WordFunctional.covariance` computes one covariance-vector product on demand;
comparisons need no covariance object at all, since a difference of linear
functionals is a linear functional. One bug found this way and worth recording:
several QWC groups can be *able* to read the same word, and attributing it to
each capable group inflated every variance by that multiplicity — caught only
by comparing the predicted σ against a Monte-Carlo spread, which is now the
test.

*Measured (TFIM n=4, 40 measurement seeds, `examples/acase_finite_shot.py`):*

| basis | shots/group | MC std | mean σ̂ | median σ̂ | bias | 95% coverage |
|---|---|---|---|---|---|---|
| κ_S = 1 | 2000 | 3.2×10⁻² | 3.7×10⁻² | 3.6×10⁻² | −1.5×10⁻² | 0.95 |
| κ_S = 1 | 20000 | 1.15×10⁻² | 1.10×10⁻² | 1.10×10⁻² | −4.9×10⁻³ | 0.93 |
| κ_S ≈ 2×10² | 2000 | 1.85 | 2.34 | 6.6×10⁻³ | −3.0×10⁻¹ | 0.97 |
| κ_S ≈ 2×10² | 20000 | 1.65×10⁻³ | 1.95×10⁻³ | 1.81×10⁻³ | −1.7×10⁻⁵ | 0.97 |

With a well-conditioned overlap the delta method is accurate (within ~15% of
the Monte-Carlo spread) and the grouped bootstrap agrees with it to three
digits (2.49×10⁻² vs 2.51×10⁻² at 4000 shots). Two findings cut the other way
and are the reason for the labels. First, the measured Ritz value carries a
**systematic downward bias** that shrinks with shots (−1.5×10⁻² at 2000
shots/group, −4.9×10⁻³ at 20000 for κ_S = 1): the noisy energy is not an upper
bound on `E₀`, and at low budgets the bias is a large fraction of the standard
deviation. Second, at κ_S ≈ 2×10² and 2000 shots/group the error distribution
is **heavy-tailed** — a handful of runs in forty admit a near-null overlap mode
and land whole Hartrees away, giving an MC std of 1.85 Ha against a median σ̂ of
6.6×10⁻³. A mean-and-variance description of the error is inadequate there.
That is the strongest argument in the code base both for conditioning-aware
growth and for never calling these intervals certified. The variational-bound
violation is a test, not a caveat: at 200 shots/group most seeds put `E_sub`
below `E₀`, by up to 2×10⁻². Quantifying it in terms of τ_S, shot covariance,
and conditioning remains **Q3**.

*Measured (4C certified growth, δ = 0.05, EB bounds):*

| shots/group | threshold | certified steps | final gap | shots | circuits |
|---|---|---|---|---|---|
| 4000 | 0.05 | 3, then abstain | 1.2×10⁻¹ | 2.6×10⁶ | 648 |
| 40000 | 0.05 | 4, then abstain | 1.4×10⁻² | 3.2×10⁷ | 810 |
| 40000 | 0.40 | 3, then abstain | 1.1×10⁻¹ | 2.6×10⁷ | 648 |

Every accepted step is `finite_sample`, and every run ends in abstention
rather than uncertified growth — the behavior the plan asked for, at the cost
the plan predicted. The cost is the headline: two independent full-universe
batches per step put certified growth four orders of magnitude above the
exact-arithmetic path in shots, and the certified trajectories stop at
gaps (10⁻²) that Phase 3 reaches at 10⁻¹⁰ exactly. Note also what is *not*
certified: the statement is about the accepted candidate's coupling with the
frozen Ritz pair, conditional on the construction batch. It is not a claim
that the candidate is the best available (`resolution` records separately
whether the leader also cleared every rival's upper bound — at these budgets
it usually does not), and it is emphatically not a bound on the energy.
Confidence-set reuse across steps, which would recover much of the shot cost,
is the obvious next stage and is not attempted here.

Two further gaps are left open on purpose. The certified path ranks candidates
by **residual coupling**, not by the generalized 2×2 lowering Phase 3 uses:
the lowering is a nonlinear function of `s_aa`, `h_aa`, and a square root, so
it admits only a delta-method (asymptotic) treatment, and the certified gate
has to be the linear statistic. And shot allocation is a predeclared uniform
budget per group — fixed endpoints are what the empirical-Bernstein validity
argument needs; policy-driven allocation across groups would need the same
fixed-schedule discipline the Paper A allocators already carry.

**Phase 5 — done (`models/lattice.py`, `models/observables.py`,
`sparse.py`).** `models/lattice.py`: Hubbard, extended Hubbard, Kanamori,
small Anderson impurity, Kitaev honeycomb cluster. The fermionic models are
built from the package's *own* Jordan-Wigner operators, so the materials
layer needs no chemistry extra at all; the Kitaev cluster is a native
`PauliSum` with one qubit per site and no transformation. Hopping is written
once and added to its own adjoint, so hermiticity is structural. Every model
carries site/orbital/bond/sector metadata, which is what lets an observable
be asked for by site rather than by spin-orbital index.

*Observables through the projected-matrix route* (§8):
`models/observables.py` supplies occupations, double occupancy, per-site spin
operators, spin correlations, the antiferromagnetic structure factor, Kitaev
per-link bond operators, and `S²`. Each is a `PauliSum` handed to
`result.expectation(Q)`, so no Ritz state is ever formed, and the tests check
every one against the dense exact state A-CASE refuses to store.

*Excited states*: `run_acase(roots=k, aggregation='mean'|'max')` — state-
averaged growth (objective = average of the tracked roots) or block growth
(whichever root gains most decides). Records carry `root_energies` and
`per_root_lowering`. The per-root variational bound `E_k^sub ≥ E_k` holds by
Cauchy interlacing and is tested; the *objective* is monotone only from the
step where the effective rank first reaches `k`, since before that the average
is taken over fewer roots and can rise as a high new root appears.

*Ingestion*: `models.fcidump.fcidump_model` reads the restricted real FCIDUMP
that a downfolding or embedding step actually hands over with NumPy only. The
parser restores packed one- and two-body symmetries, validates `NORB`, `NELEC`,
`MS2`, sentinel patterns, duplicates, and finite coefficients, rejects
unsupported `IUHF=1`, fixes the interleaved-spin reference sector, records the
source SHA-256, and maps chemist integrals directly with the package's own
fermion operators. The historical `models.chemistry.fcidump_model` delegates
to it. The chemist/physicist ordering remains the trap; the chemistry-extra
test compares all 185 H₄ coefficients against the independent
OpenFermion/PySCF construction (maximum mismatch `8.7×10⁻¹⁶`).

*Committed active-space rung*: the linear H₄ STO-3G CAS(4e,4o) FCIDUMP is
frozen with geometry, PySCF 2.14 provenance, SHA-256, RHF energy, and a
determinant-space FCI oracle. It maps to 8 qubits, 185 Pauli terms, and a
36-state `(N=4,S_z=0)` sector; the mapped sector energy agrees with external
FCI to `3.1×10⁻¹⁵ Ha`. At eight adaptive additions A-CASE has `M=9`,
`W=7,371`, `kappa(S)=1`, and a `3.019 mHa` error. The complete
singles/doubles coordinate space has `M=27`, `kappa(S)=1`, and a
`0.766 mHa` error. This is a reproducible chemistry benchmark, not evidence of
quantum advantage; the adaptive arm does not reach chemical accuracy at the
declared budget.

*Sparse reference tier*: `sparse.py` writes each Pauli word as the signed
permutation matrix it is (`W = i^{n_Y} X^x Z^z`) instead of summing dense
Kronecker products, giving `eigsh` a Hamiltonian with `≤ (#terms)·2^n`
nonzeros — a 12-qubit XXZ ground state in 0.1 s. Two findings are baked in:

- **`which='SA'` is not safe here.** On the `t = 0` Hubbard cluster (diagonal,
  eigenvalues in `{0, U, 2U, …}`, 256-fold zero eigenspace) ARPACK returns `U`,
  converged and residual-free, for a matrix whose minimum is 0. A residual
  check cannot catch it — `U` really is an eigenvalue. The fix is in how the
  problem is posed: solve for the largest-magnitude eigenpair of `H − σI` with
  `σ = Σ_w|h_w| ≥ ‖H‖`, which costs one diagonal and no factorization.
- **A grand-canonical cluster does not minimize at the filling its name
  implies.** With `μ = 0` the 4-site Hubbard chain's global ground state sits
  in the *two*-electron sector. `hubbard` therefore defaults to `μ = U/2` (and
  Kanamori to `U/2 + (M−1)U'`, the interaction's linear residue under
  `n → 1−n`), and `sparse_ground_in_sector` restricts to a `(N, S_z)` block
  when a specific filling is wanted — which is the honest comparison for
  A-CASE, since the subspace stays in its reference's sector.

*Measured (`examples/acase_materials.py`).* Analytic limits first, because
they are what catches a hopping sign or a JW string: the `U = 0` Hubbard chain
reproduces `2Σ_{ε_k<0} ε_k` to 10⁻⁸ for 2, 4, and 6 sites; the `t = 0` cluster
gives `−UN/2`; free-fermion double occupancy is exactly 1/4; the singlet
ground state has `⟨S²⟩ = 0`; and the Kitaev cluster's energy is reproduced by
its three per-link correlators alone (`⟨XX⟩_x = ⟨YY⟩_y = 0.4527`,
`⟨ZZ⟩_z = 0.7879`, non-link pairs at `−0.015`) — the spin-liquid signature.

The load-bearing negative result is on the Hubbard clusters. From the Néel
product reference, the **entire** singles-and-doubles response space saturates
at a gap of 3.2×10⁻¹ (4-site chain) and 2.6×10⁻¹ (2×2) against the sector
ground energy at `U = 4`, and adaptive growth reaches that same limit and then
correctly stops — the space does not contain the state. Adding Krylov
candidates helps the chain (1.8×10⁻¹ at M=11) and not the 2×2. Observables
converge in the right direction along the trajectory (double occupancy
0 → 0.053 against an exact 0.072, `⟨S²⟩` 2.0 → 0.82 against 0) without the
energy converging. This is the clearest case yet for §4.2 level 4: a
symmetry-broken product reference plus singles and doubles is the wrong object
for a strongly correlated cluster, and compound generators and competing-order
(stabilizer) references — not more of the same family — are what the plan owes
these models. `examples/acase_materials.py` prints the whole comparison.

*That debt is now paid, and it settles the span half of Q4.* Level 4 ships
(§4.2 above), and on the 2×2 cluster it does exactly what was predicted here:
levels 0–3 saturate at `−9.8475` against a sector ground energy of `−10.1027`,
adding the competing-order configurations *by themselves* changes nothing
(they are determinants `H` does not connect to the reference at first order),
and adding the compound products `configuration × excitation` reaches
`−10.102748` — the sector ground state, to `1e-8`. The barrier was the span,
and it was the one-reference structure of levels 0–3 that imposed it. Grown
adaptively, the committed record reaches the same energy to `1e-14` with
`M = 26` against a 36-state sector (Q4, below).

**Phase 6 — done (`backends/sector_statevector.py`).**
`SectorStatevectorBackend` stores a pure state on the occupation words of one
`(N, S_z)` sector — `C(n,k)` amplitudes, never `2^n` — and applies a Pauli
word as a bit-mask gather, `W|b⟩ = i^{n_Y}(−1)^{|z∧b|}|b⊕x⟩`. Words are
**grouped by X-mask**: every word in a group shares the permutation
`b → b⊕x`, so the gather is resolved once per group and only the diagonal
phases differ — and those phases do not depend on the state either, so each
group collapses to one coefficient vector and a matvec is a few
gather-multiply-scatter passes. `SectorOperator` exposes that as a
`LinearOperator` for `eigsh`, with a numpy-only `lanczos_ground` fallback.
`sector_projector` builds the ideal's projector as an `MV` for the theory-facing
checks; the backend never forms it.

Three implementation points are load-bearing:

- *Term-wise projection is exact, not approximate.* Individual words of a
  number-conserving Hamiltonian do **not** conserve `N` — the same leakage the
  chemistry pool documents — so most words map part of the sector out of it,
  and the backend drops those components. That is legitimate because `H`
  commutes with the sector projector: `H|ψ⟩ = P H|ψ⟩ = Σ_w h_w (P W_w|ψ⟩)`, and
  `P` distributes over the sum. The out-of-sector pieces cancel in the total;
  projecting each term is the same arithmetic reordered. The test compares the
  matvec against the Hamiltonian's sparse submatrix on the sector.
- *No `2^n` index table.* The permutation `b → b⊕x` is resolved by binary
  search on the sorted sector, not by a lookup array over the full space —
  which would reintroduce exactly the memory the backend exists to avoid.
  Basis construction is combinatorial for the same reason (`C(40,2)` states out
  of `2^40` in milliseconds), in contrast to `sparse.sector_indices`, which
  enumerates `2^n` because it masks an already-dense matrix.
- *Lanczos converges on the residual, not the eigenvalue.* Ritz values converge
  quadratically faster than their vectors, so stopping when the eigenvalue
  settles returns vectors an order of magnitude short of the advertised
  tolerance. The criterion is `β_k|s_k[i]|`, with full reorthogonalization
  (the bare three-term recurrence starts manufacturing duplicate eigenvalues,
  which on a degenerate spectrum is indistinguishable from real degeneracy).

*Validated:* ground energies match `exact_ground` and
`sparse_ground_in_sector` on every lattice model for `n ≤ 12`, through both
`eigsh` and the numpy-only Lanczos; the matvec matches the sparse submatrix;
`P² = P`, `P† = P`, `tr P = |sector|`, `[H,P] = 0`; the sector basis agrees
with the independent dense enumeration; expectations of *non*-conserving
observables agree with the dense restriction (only `P O P` contributes); and
the `t=0` degenerate spectrum that defeats ARPACK's `which='SA'` is handled by
both solvers.

*Measured (`examples/acase_sector_backend.py`, half-filled Hubbard chains):*

| sites | n | sector dim | 2^n | ratio | state | sparse nnz it avoids |
|---|---|---|---|---|---|---|
| 4 | 8 | 36 | 256 | 7.1× | 0.6 kB | 4 352 |
| 6 | 12 | 400 | 4 096 | 10.2× | 6 kB | 110 592 |
| 8 | 16 | 4 900 | 65 536 | 13.4× | 78 kB | 2 424 832 |
| 10 | 20 | 63 504 | 1 048 576 | 16.5× | 1.0 MB | 49 283 072 |
| 12 | 24 | 853 776 | 16 777 216 | 19.7× | 13.7 MB | — |

Ground energies: 20 qubits in 1.8 s, 24 qubits in 54 s. The compiled operator
is 19 X-mask groups from 47 words at `n = 20` (matvec 6.6 ms, 22 MB held)
against 93 ms recomputing per matvec — the compile-once/solve-many trade a
DMFT-style outer loop wants, and `precompute=False` is there because at
`n = 24` the compiled passes want 355 MB against the state's 14 MB.

*A Phase-5 bug this phase caught.* The lattice metadata hardcoded `S_z = 0` at
half filling, which is wrong for an odd site count (three electrons on three
sites sit at `S_z = ±½`). The backend refuses to build an empty sector, which
surfaced it; sector metadata is now read from the reference determinant's own
gates, so it cannot disagree with the state it describes.

**Phase 7 — done (`benchmarks/run_acase_ladder.py`,
`benchmarks/summarize_ladder.py`).**
H₂ → LiH → H₄ (equilibrium and stretched) → stretched H₂O CAS(4e,4o) → H₂O
CAS(8e,6o) → 2×2/2×3 Hubbard → one Kitaev cluster with link correlations.
Comparison set as planned: the reference determinant, exact diagonalization,
plain QSE, fixed Krylov, generator-coordinate-style fixed subspaces,
ADAPT-VQE (exact and finite-shot), A-CASE (exact and finite-shot certified).
Every row carries the §6 metrics, shots, circuits, and abstentions beside the
energy, plus the `evidence` label that says what kind of number it is, and the
record has its own summarizer because its schema is one run per (rung, method)
rather than the per-seed ADAPT schema `summarize.py` owns.

Three conventions decide what the table means, and each is recorded in the row
that made it:

- *The error column is measured against the reference's own symmetry sector.*
  A-CASE never leaves the sector its reference lives in, and a
  grand-canonical Hubbard cluster's global minimum sits at a different
  filling — scoring against the global ground state would charge every method
  for a particle-number difference none of them can cross.
- *The generator-coordinate arm takes an even stride through the candidate
  family, not its prefix.* The excitation family lists singles first and every
  single is Brillouin-dead on a Hartree–Fock reference, so a prefix of eight
  reproduces the reference energy to machine precision. That is a fact about
  the ordering, not about non-adaptive subspaces, and reporting it as the
  baseline would have flattered A-CASE for the wrong reason. With the stride,
  the arm clears HF on H₄ (−2.1307 against −2.1243) and is still beaten.
- *Wide generators fall back to the cyclic contraction.* Above
  `max_tracked_support` the row is solved through `Tr((HA_j)(ρA_i†))` and marks
  its support columns `n/t` rather than reporting guessed ones. Deep Krylov
  generators on eight qubits carry thousands of words each, and the
  element-operator route that makes `W` and `S_H` observable costs
  `O(|A_i| |H| |A_j|)` per pair.

*A configuration error the ladder caught, which would have published as a
method failure.* `sector_leakage` measures `||[A,N]||/||A||` and the same for
`S_z` — the *fermionic* symmetries — and it takes a generator alone, so it
cannot know that the Kitaev cluster is one qubit per site with no
Jordan–Wigner transformation behind it and no particle number to conserve.
Applied there, the §4.2 sector filter rejected all 72 candidates (the whole
pool leaks at 0.94 or above) and A-CASE reported a basis of size one sitting at
the reference energy, `−4.0` against the cluster's `−4.9624`. Read off the
table that is "A-CASE cannot grow on a frustrated spin cluster." It is nothing
of the kind: with the filter correctly not applied, A-CASE reaches the exact
energy to `7e-13` at `M = 6`. The tolerance is now dropped for models with no
fermionic sector and the row records that it was dropped, so the failure cannot
recur silently. On the fermionic rungs the excitation candidates conserve both
symmetries *exactly* (leakage `0.0`), so the filter is a no-op and the
distinction costs nothing there.

That recovered row is worth reading carefully rather than banking as a win,
because it is exactly the reading §6 exists to block. A-CASE's selected basis
is `I, H¹, H², H³, H⁷, H⁸` — six of the nine Krylov powers that were sitting in
its own candidate pool, with every `pauli_orbit` and `commutator_response`
candidate passed over. So it did not find a more compact *kind* of basis; it
pruned the Krylov basis by a third. And the resources do not follow the basis
size down: the word universe is identical (`W = 140`, `S_H = 76`, the same words
either way) and the conditioning is marginally *worse* (`κ_S = 3.8e+04` against
Krylov's `3.3e+04`). Counting basis vectors alone, this is `M = 6` beating
`M = 9`; counting what §6 asks for, it is the same measurement cost and the same
ill-conditioning with three fewer coefficients to fit.

*The measurement layer, not the shot budget, was what confined certified growth
to four qubits — and the first diagnosis of which part was wrong.* The QWC
partition is greedy and quadratic in the word universe, and that was the
obvious suspect. But it is memoized on the word set, and the certified loop's
universe is the same index set at every growth step (the basis gains the index
the candidate pool loses), so it was computed **once**, not per step. It was
also never the binding cost. Measured on the H₄ element universe — 15 846
words, 1 689 groups — the partition was ten seconds of a batch that took hours.

The binding cost was `computational_probabilities`: 11.4 s per eight-qubit
readout, called once per QWC group per batch. A Z-basis readout only sees the
*diagonal* Pauli content of the state, so with `|b⟩⟨b| = Π_j (I + s_j Z_j)/2`,

    p(b) = Σ_z ρ_z (−1)^popcount(z ∧ b),

the Walsh–Hadamard transform of the diagonal coefficients — `O(2^n n)` for all
outcomes at once, against `O(4^n)` word products for building each projector and
multiplying it out. With that, plus holding the shot-independent part of each
group's sampling plan (basis rotation, partial trace, outcome distribution,
per-word parity masks) instead of rebuilding it every batch, a warm batch over
those 1 689 groups is 0.19 s. The partition was separately made
output-preserving-but-fast (vectorized conflict degrees, and placement against
each group's merged basis code rather than a per-word incompatibility bitset):
15 846 words, 77 s → 2.3 s.

The lesson is the one §6 keeps making in a different register: the quantity that
looked expensive by inspection was not the quantity that was expensive, and only
measurement distinguished them.

*Certified growth at eight qubits*, now that it runs (`acase_certified_n8`,
32 000 shots per group, `max_size = 4`):

| rung | M reached | error | shots | outcome |
|---|---|---|---|---|
| h4_equilibrium | 2 | +3.66e-02 | 216 M | abstained |
| h4_stretched | 5 | +4.09e-02 | 461 M | budget reached |
| h2o_cas4e4o_stretched | 4 | +1.03e-01 | 432 M | abstained |
| hubbard_2x2 | 5 | +1.28e+00 | 442 M | budget reached |
| kitaev_2x2 | 2 | +1.20e-01 | 151 M | abstained |

This is the first Q2 evidence above four qubits. Two rungs grow to the full
budget without a single uncertified step; three stop by abstention rather than
by growing on a decision the shots do not support, which is the behaviour Q2
asks for. At 4 000 shots per group — the four-qubit budget — H₄ abstains
immediately, so the certified arm is *shot-limited* at eight qubits rather than
structurally blocked. Every certified energy here sits well above chemical
accuracy, so this establishes that the certificate keeps working at `n = 8`,
not that certified A-CASE is accurate there.

*Measured* (`benchmarks/reference_results/acase_ladder.jsonl`, 72 runs; error in
Hartree against the reference's own sector, `—` where the config does not run
that arm on that rung):

| rung | n | reference | QSE | Krylov | gen-coord | ADAPT | A-CASE |
|---|---|---|---|---|---|---|---|
| h2 | 4 | +2.1e-02 | −2.7e-15 | −1.8e-15 | −2.4e-15 | −2.0e-15 | −2.4e-15 |
| lih_2e2o | 4 | +2.6e-04 | +3.0e-14 | +9.8e-12 | +3.1e-14 | +2.8e-14 | +3.1e-14 |
| h4_equilibrium | 8 | +5.6e-02 | +4.2e-02 | **+1.0e-08** | +5.0e-02 | +2.4e-03 | +3.0e-03 |
| h4_stretched | 8 | +2.6e-01 | +1.8e-01 | **+1.5e-05** | +1.8e-01 | +7.4e-03 | +3.9e-02 |
| h2o_cas4e4o_stretched | 8 | +3.2e-01 | +1.7e-01 | **+7.1e-03** | +1.6e-01 | +1.2e-02 | +5.5e-02 |
| h2o_cas8e6o | 12 | +5.0e-02 | +4.7e-02 | — | +4.2e-02 | — | **+1.4e-02** |
| hubbard_2x2 | 8 | +2.1e+00 | +1.5e+00 | +2.6e-03 | +1.9e+00 | +4.0e-01 | +8.6e-01 |
| hubbard_2x2 (level 4) | 8 | — | — | — | — | — | **−1.1e-14** |
| hubbard_2x3 | 12 | +3.6e+00 | +3.4e+00 | — | +3.2e+00 | — | **+2.2e+00** |
| kitaev_2x2 | 8 | +9.6e-01 | +9.6e-01 | −1.1e-14 | — | +9.6e-01 | **+7.4e-13** |

Read with §6 beside it, at the equal budget `M = 9` on the six rungs where both
non-adaptive and adaptive arms ran:

| rung | gen-coord `W` | A-CASE `W` | gen-coord `κ_S` | Krylov `κ_S` | A-CASE `κ_S` |
|---|---|---|---|---|---|
| h4_equilibrium | 13 646 | 7 371 | 1.0 | 6.6e+10 | 1.0 |
| h4_stretched | 13 646 | 7 715 | 1.0 | 2.6e+10 | 1.0 |
| h2o_cas4e4o_stretched | 12 734 | 7 783 | 1.0 | 6.0e+07 | 1.0 |
| h2o_cas8e6o | 232 515 | 143 117 | 1.0 | — | 1.0 |
| hubbard_2x2 | 6 258 | 5 537 | 1.0 | 9.7e+09 | 1.0 |
| hubbard_2x3 | 27 870 | 5 358 | 1.0 | — | 1.0 |

What the ladder does and does not support:

- **Q1, against blind selection: supported.** At equal `M = 9` and drawing from
  the *same* candidate family, adaptive selection beats the strided
  generator-coordinate subspace on every rung — in error (by 1.5× on the 2×3
  Hubbard up to 17× on equilibrium H₄) and in word universe simultaneously
  (7 371 against 13 646 on H₄; 5 358 against 27 870 on the 2×3 Hubbard, a 5.2×
  saving). That is the comparison Q1 was
  written for, and the `W` column is what makes it a resource claim rather than
  a basis-size claim.
- **Q1, against fixed Krylov: not supported.** Krylov wins the energy on every
  fermionic rung where it ran, at the same `M` and by up to five orders of
  magnitude (1.0e-08 against 3.0e-03 on equilibrium H₄). The compactness claim
  does not survive that comparison and should not be advertised as if it did.
  Only on the Kitaev cluster do the two agree, both exact.
  What Krylov pays is conditioning — `κ_S` from 3.3e+04
  to 6.6e+10, against A-CASE's `1.0` on every fermionic rung, because A-CASE
  rejects candidates that are not sufficiently independent of the span it
  already has. At eight qubits its generators are also too wide to bank at all
  (`n/t` in the record). So the honest summary is a trade, not a win: Krylov
  buys accuracy with an overlap matrix no finite-shot run could invert, and
  A-CASE buys a conditioned, measurable subspace at a worse energy.
- **A-CASE does not beat ADAPT-VQE.** ADAPT is better on stretched H₄
  (7.4e-03 against 3.9e-02), on stretched water (1.2e-02 against 5.5e-02), and
  on the 2×2 Hubbard (4.0e-01 against 8.6e-01), and ties at equilibrium H₄.
  This reproduces the Phase 3 finding rather than overturning it. The one place
  ADAPT collapses is the Kitaev cluster, where its exact gradient is below
  threshold at the reference and it selects *zero* operators: the reference is a
  stationary point, and a first-order selection rule has nothing to see. A-CASE's
  2×2 generalized lowering is not a gradient and does grow there.
- **Q4: the span half is answered, the compactness half only on the smaller
  cluster.** With levels 0–3 the singles-and-doubles family saturates 0.86 Ha
  above the 2×2 sector ground energy and 2.2 Ha above the 2×3, and no arm but
  Krylov comes close — the reading Phase 5 recorded. Level 4 removes the span
  barrier outright: on the 2×2, `{identity, excitations, configurations,
  configuration × excitation}` contains the sector ground state exactly
  (`−10.102748`, agreement to `1e-8`), where levels 0–3 stall at `−9.8475`
  however many directions are added.

  Compactness holds on the 2×2 and is untested on the 2×3. From the committed
  record, both arms grown adaptively at the same budget:

  | rung | family | M | error | `κ_S` | `W` | level-4 picked | stop |
  |---|---|---|---|---|---|---|---|
  | hubbard_2x2 | levels 0–3 | 23 | +2.55e-01 | 1.00 | 13 665 | 0 | saturated |
  | hubbard_2x2 | levels 0–4 | 26 | **−1.07e-14** | 1.00 | 15 191 | 5 | budget |
  | hubbard_2x3 | levels 0–3 | 26 | +1.09e+00 | 1.00 | 85 264 | 0 | budget |
  | hubbard_2x3 | levels 0–4 | 26 | +1.09e+00 | 1.00 | 85 264 | 0 | budget |

  On the 2×2 that is the sector ground state at machine precision with `M = 26`
  against a **36**-state sector, at `κ_S = 1` and 11 % more words than the
  saturated levels-0-3 basis — a compact, well-conditioned, measurable basis
  rather than a re-derivation of full CI. It is the only arm on that rung to
  reach chemical accuracy at all; fixed Krylov gets to `+2.6e-03` and misses,
  at `κ_S = 9.7e+09`. The five generators selected are
  `afm*E(2<-0)`, `afm*E(1<-3)`, `afm*E(1,2<-0,3)`, `afm_flipped*E(6<-4)` and
  `afm_flipped*E(5<-7)` — the antiferromagnet and its spin-flipped partner,
  each dressed by an excitation. The levels-0-3 arm stops on its own at
  `M = 23` with predicted lowering below threshold: it is not budget-limited,
  it is out of directions.

  The 2×3 cluster is the honest negative: identical rows, zero level-4
  generators selected. The compactness half of Q4 is answered on one cluster
  and open on the other.

- **A limitation of §4.3, exposed by level 4 and worth stating separately.**
  The generalized 2×2 score cannot see a bare competing-order configuration.
  Such a determinant has zero overlap with the reference *and* zero Hamiltonian
  matrix element to it — the Hubbard `H` connects determinants differing by one
  hop, and these differ by a spin flip on two sites — so the predicted lowering
  is exactly zero and greedy growth never takes one, however useful it would be
  in combination. Level 4 earns its place entirely through the compound
  products, which do couple at first order. A selector that could weigh
  second-order coupling would change this; the present one cannot, and that is
  a property of the criterion rather than of the family.
- **Q2: supported as far as the ladder reaches.** Certified growth now runs at
  eight qubits on five rungs; no run grows on a step the shots do not certify,
  and abstention rather than silent growth is what stops three of them. Coverage
  itself is calibrated in the Paper A experiments, not here.
- **Q3 has a concrete instance in the record.** The certified H₂ row reports
  `−1.13821296` against a reference of `−1.13727017` — 0.94 mHa *below* the
  exact energy. The variational bound does not survive thresholding a noisy
  overlap matrix, which is why the summarizer refuses to rank on energy without
  the evidence label.

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
