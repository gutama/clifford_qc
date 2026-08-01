# `clifford_qc` literature roadmap — Phases 8–14

This document is the third roadmap in the series. `RESEARCH_PLAN.md` is Paper A
(confidence-certified, measurement-efficient ADAPT-VQE, complete through its
Phase 5). `ACASE_RESEARCH_PLAN.md` is Paper B (A-CASE, complete through
Phase 7). This one starts from **fourteen papers from the 2023–2026
literature** and asks a narrower question than either:

> **Which of the things this project does not do are load-bearing for the
> claims it wants to make — and which of them can be built out of parts the
> repository already has?**

The answer is unflattering in one place and encouraging in three. It is
recorded here in that order, because the unflattering one determines the phase
ordering.

---

## 0. The papers, and what each one costs us

| # | arXiv | Theme | Verdict for this project |
|---|---|---|---|
| 1 | 2302.11320 | QSCI | **Missing baseline.** Blocks Paper B submission. |
| 2 | 2411.00468 | ext-SQD excited states | **Direct competitor claim** against A-CASE's chemistry default. |
| 3 | 2407.08696 | CEO-ADAPT-VQE | **Constrains Paper A's framing** (99.6 % measurement reduction by pool design). |
| 4 | 2409.03747 | Oscillator-qubit | **Declined.** One idea salvaged (§8B). |
| 5 | 2301.10196 | Overlap-ADAPT-VQE | **Fixes a documented A-CASE blind spot.** |
| 6 | 2412.13839 | Time-evolved QSCI | **Fixes the A-CASE/Krylov conditioning trade.** |
| 7 | 2302.03052 | Projection-based embedding | Ingestion boundary only. |
| 8 | 2606.30551 | Generative-ML QSCI | **Do not cite** (see §0.1). |
| 9 | 2501.14968 | Measurement review | Names the one measurement upgrade this codebase should own. |
| 10 | 2305.04783 | Folded-spectrum VQE | **Pays the §4.4 second-moment debt** and buys three things at once. |
| 11 | 2606.05968 | BK symmetry trap | Motivates mapping breadth. Weak authority (§0.1). |
| 12 | 2409.11210 | MORE-ADAPT-VQE | A capability we already ship and have never benchmarked. |
| 13 | 2311.01393 | FLDC barren plateaus | Positioning only. Build nothing. |
| 14 | 2607.20585 | ML-compact QSCI subspaces | Sets the compactness bar QSCI arms are now judged against. |
| 15 | 2607.16869 | Correlation rank, Clifford-accessible measurement | **Prices Phase 12, and hands us a testable invariant.** |

### 0.1 Citation hygiene, checked rather than assumed

Three of the fourteen are 2026 preprints, and they do not all survive contact:

- **2606.30551 has been withdrawn by arXiv administrators** ("the submitter did
  not have the rights to agree to the license at the time of submission"). It
  must not enter the bibliography of either manuscript. Its technical content
  (RBM configuration recovery) is available from 2607.20585, which is not
  withdrawn; cite that instead.
- **2606.05968** (3 citations, unrefereed) claims "instant, exact FCI
  convergence within the very first macro-cycle" on LiH, HF and H₂O. That is an
  extraordinary claim for an adaptive method and it is not independently
  reproduced. Treat it as motivation for mapping breadth (§13) and cite it, if
  at all, as a preprint whose central claim we did not verify.
- **2607.20585** (2 citations, unrefereed) is credible and directly on-topic,
  but its headline — chemical accuracy at ~4 % of the configuration subspace
  against DMET-SQD failing at 20 % — is a *compactness* number of exactly the
  kind §6 of the A-CASE plan exists to interrogate. Reproduce the comparison
  structure, not the number.

- **2607.16869** (18 July 2026, unrefereed, too new to have citations) is in the
  same age class, but it differs from the other two in a way that matters: its
  central structural claim is a *theorem we can test ourselves* on Hamiltonians
  this package builds (§12A). Verify the parity ceiling before citing the shot
  numbers, and cite the two independently — a reproduced invariant and an
  unreproduced benchmark are not the same kind of evidence.

Paper 3 is published (*npj Quantum Information* **11**, 86 (2025)) and is the
strongest authority in the ADAPT cluster.

---

## 1. The unflattering part: this project has no QSCI arm

Five of the fourteen papers are QSCI/SQD. It is the paradigm that produced the
largest electronic-structure quantum simulation to date (77 qubits,
[2Fe-2S], paper 2) and it is the frame every 2026 referee will read Paper B in.
`clifford_qc` has no implementation of it, no baseline for it, and no row for it
on the Phase 7 ladder.

The comparison is sharp, and it runs against us on the axis this project chose
as its own currency:

| | A-CASE | QSCI/SQD |
|---|---|---|
| Basis | operator-generated, nonorthogonal `A_i|ψ⟩` | sampled determinants, orthonormal |
| `S` matrix | measured; `κ_S` is a live risk | identity by construction |
| `H_ij` | **measured**; `W` up to 143 117 words on `h2o_cas8e6o` | **computed classically** from the integrals |
| Quantum cost | full word universe per growth step | computational-basis sampling only |
| Variational bound | broken by noisy PSD repair (Q3, documented) | strict, even under hardware noise |

§6 of the A-CASE plan says compactness is not `M` but `M` *and* `W`. On `W`,
QSCI does not merely win — it scores zero, because it never measures a matrix
element at all. The certified-growth rows in the ladder (216–461 M shots at
eight qubits, four orders above the exact path) are the number that comparison
will be made against.

**This is stated up front rather than discovered at referee time.** The
correct response is not to avoid the comparison. It is to run it, find the two
places where the determinant subspace genuinely fails, and build the hybrid
that the repository is unusually well-positioned to build (§9).

Where the determinant subspace should fail, and why we can test it here:

1. **No fermionic structure.** SQD's configuration recovery is built on
   particle number and `S_z`. The ladder's `kitaev_2x2` rung is one qubit per
   site with no Jordan–Wigner transformation behind it and no particle number
   to recover against — the same fact that made `sector_leakage` reject all 72
   candidates in Phase 7. A-CASE reaches `7e-13` there; ADAPT-VQE selects zero
   operators; QSCI has no configuration-recovery step to run.
2. **States not sparse in the determinant basis.** A determinant subspace can
   only be compact if the target is. The 2×2 Hubbard result already isolates
   the mechanism from the other side: levels 0–3 saturate at `−9.8475` against
   `−10.1027`, and it is the compound `configuration × excitation` generators —
   determinants *dressed by operators* — that reach the sector ground state.
   That is precisely a direction a determinant subspace does not contain.

Neither is a hope. Both are testable on rungs that already exist.

---

## 2. The encouraging part: three gaps close with parts already in the repo

**(a) Sampled determinants are already first-class A-CASE generators.**
`subspace/generators.py::configuration_generator` enters a configuration as the
operator `A = VR†` that *carries* the reference onto it, so `A|ψ⟩ = |φ⟩` with
no second state prepared, and between two determinants that operator is a
single `X`-string — `S_A = 1`. A QSCI sample is therefore not a foreign object
that needs a new subspace type; it is a level-4 generator the bank already
accepts, at the cheapest support in the whole hierarchy.

**(b) The second-moment bank is one object that pays four debts.**
`ACASE_RESEARCH_PLAN.md` §4.4 declares `K_ij = ⟨ψ|A_i†H²A_j|ψ⟩` missing and
`residual_norms` therefore optional. Paper 10 (folded spectrum) needs the same
object. So do energy-variance extrapolation (the standard selected-CI
correction, which QSCI arms need too) and any honest convergence criterion.
One bank, four uses.

**(c) General-commuting measurement grouping is native to this package and
absent from it.** Paper 9 catalogues the measurement-reduction landscape;
`measurement/grouping.py` implements only greedy qubit-wise commuting groups.
The upgrade to fully-commuting groups requires a Clifford diagonalization per
group — which is the one measurement optimization a package built on
`Cl(2n,ℂ)` with a Stim bridge and a stabilizer backend should own rather than
import.

---

## 3. Phases

Each phase states its papers, what it reuses, its go/no-go invariant, and the
result that would falsify it. Ordering is by dependency: Phase 8 unblocks 9,
and 9 is where the research contribution is.

### Phase 8 — the QSCI/SQD arm (papers 1, 2, 6, 14)

New module `clifford_qc/subspace/qsci.py`. Not a baseline stub — a first-class
method with the same §6 accounting as every other arm.

**8A — sampling and diagonalization.**
Reuse, in order: `states.computational_probabilities` (already carrying the
Walsh–Hadamard fast path that took the eight-qubit readout from 11.4 s to
0.19 s in Phase 7), `backends/sector_statevector.sector_basis` and `index_of`
for determinant indexing without a `2^n` table, and restriction of
`SectorOperator` to the sampled index set for the subspace Hamiltonian.

Restricting the existing sector operator is exact and introduces no new
physics code — the sampled set is a row/column selection of a block we already
build correctly and already test against the sparse submatrix. Slater–Condon
rules from `models/fcidump.py::FcidumpData` (which retains full `one_body` and
`two_body` arrays) are a *later* addition, needed only when the determinant
count outruns the sector backend. Do not write them first; the sector route is
the one with an oracle.

*Go/no-go:* on `h4_equilibrium`, sampling the exact sector ground state
reproduces the sector-exact energy once the sampled set reaches the sector
dimension, and the variational bound `E_QSCI ≥ E_sector` holds at every
smaller sample size and every seed — including under shot noise, which is the
claim QSCI makes and A-CASE cannot (Q3).

**8B — configuration recovery, and the one idea salvaged from paper 4.**
SQD's recovery step repairs sampled bitstrings that violate particle number or
`S_z`. `subspace/generators.py::state_sector` already reports `⟨N⟩`, `⟨S_z⟩`
and their variances, and `sector_basis` already enumerates the legal
occupation words. Post-selection is therefore a filter over an enumeration we
have, not a new algorithm.

Paper 4 (oscillator-qubit) is otherwise declined — bosonic modes are outside a
Pauli/qubit package's scope and building a qumode layer would dilute the
identity §9 of the A-CASE plan protects. Its one transferable idea is
*ancilla-free partial error detection via Gauss's law*: symmetry post-selection
on raw shots, with the violation rate reported rather than silently discarded.
Record the discard fraction per rung; a recovery step that quietly drops 60 %
of shots is a resource cost, not a free correction.

**8C — ladder rung and accounting.**
Add QSCI to `benchmarks/run_acase_ladder.py` on every existing rung, carrying
`M` (determinants retained), sampling shots, discard fraction, and — this is
the point — `W = 0` for the projected matrices, stated explicitly rather than
left blank.

*Predicted honest outcome, recorded before the run:* QSCI wins the resource
column outright on every fermionic rung, is competitive or better on energy at
equal `M` on `h4_*` and `h2o_*`, and has **no arm at all** on `kitaev_2x2`.
If it also wins on `hubbard_2x2` and `hubbard_2x3` at equal `M`, the Q1
compactness claim is in serious trouble and the manuscript must say so.

**8D — excited states (paper 2).**
Paper 2 states that ext-SQD "improves over quantum subspace expansion based on
single and double electronic excitations ... in both accuracy and efficiency."
A-CASE's *chemistry default* is a subspace expansion in single and double
excitations. That sentence is aimed at us and must be confronted in the
manuscript's positioning section, not only in code.

Build the comparison: ext-SQD-style excited states against `run_acase(roots=k,
aggregation='mean'|'max')`, which already exists and already satisfies the
per-root Cauchy-interlacing bound `E_k^sub ≥ E_k` under test.

*Falsifier for the whole phase:* if the QSCI arm beats A-CASE on energy **and**
`W` **and** conditioning on every rung including Kitaev, then A-CASE's
positioning as a chemistry method does not survive, and Paper B should be
rewritten around the non-fermionic and dressed-determinant cases where it does.

### Phase 9 — QSCI × A-CASE: the hybrid (papers 1, 5, 14)

This is the research contribution. Phase 8 exists to make it possible and to
make it honest.

**9A — sampled determinants as level-4 generators.**
Feed the QSCI-selected determinants into `configuration_generator`, then into
`compound_response` to form `configuration × excitation`. The Phase 7 record
already establishes the mechanism on `hubbard_2x2`: levels 0–3 stall at
`−9.8475` however many directions are added, bare configurations by themselves
change nothing (the Hamiltonian does not connect them to the reference at first
order), and the compound products reach `−10.102748` — the sector ground state
to `1e-8`, at `M = 26` against a 36-state sector and `κ_S = 1`.

The hybrid claim is therefore specific and pre-registered: **a determinant
subspace plus its operator dressing is strictly richer than either, and the
sampling step is how you find which determinants to dress.** QSCI supplies the
span; A-CASE supplies the directions sampling cannot reach.

**9B — overlap-based selection (paper 5).**
The A-CASE plan records its own blind spot in §4.2/§7: the generalized 2×2
score cannot see a bare competing-order configuration, because such a
determinant has zero overlap with the reference *and* zero Hamiltonian matrix
element to it, so predicted lowering is exactly zero and greedy growth never
takes one however useful it would be in combination. The plan calls this "a
property of the criterion rather than of the family."

Paper 5 is the fix. Overlap-ADAPT-VQE selects by maximizing overlap with an
intermediate target that already carries some correlation — and names classical
selected-CI wavefunctions as the strongest choice of target. Phase 8 produces
exactly such a wavefunction, in the same determinant basis, at zero extra
quantum cost.

Implement `score_candidate(..., criterion='overlap', target=...)` alongside the
existing lowering score: rank by `|⟨χ_a|Ψ_target⟩|` normalized by the
candidate's `S`-orthogonal component against the retained span, so the
rejection floor and scale-invariance discipline of §4.3 carry over unchanged.

*Go/no-go, sharp and already isolated by the record:* on `hubbard_2x3`, the
committed rows for levels 0–3 and levels 0–4 are **identical**, with **zero**
level-4 generators selected — the honest negative that leaves the compactness
half of Q4 open. If overlap-based selection picks level-4 generators there and
moves the error below `+1.09e+00`, the criterion is doing the work claimed for
it. If the rows stay identical, the barrier is the family and not the score,
and that is a different paper.

**9C — resource honesty.**
The hybrid inherits QSCI's sampling cost *and* A-CASE's measured `W`. Report
both. A hybrid that reaches A-CASE's accuracy at QSCI's `W = 0` does not exist;
the claim on offer is accuracy the determinant subspace cannot reach, at a `W`
that is bounded and reported.

### Phase 10 — second-moment bank and folded spectrum (paper 10)

`subspace/second_moment.py`: `SecondMomentBank` with
`K_ij = ⟨ψ|A_i†H²A_j|ψ⟩`, built on the same canonical-ID, lazy-pair-product,
Hermitian-reuse discipline as `MatrixElementBank`.

Four things it buys, all of which the repository currently does without:

1. **True Ritz residual norms.** `‖(H−E)|Ψ⟩‖² = (c†Kc)/(c†Sc) − E²`. The A-CASE
   plan's standing prohibition survives intact: the *projected* residual is
   zero by construction for a solved Ritz pair and must never be exposed under
   this name.
2. **Interior and excited roots by folded spectrum.** Minimize `⟨(H−ω)²⟩` in
   the projected subspace to target states around `ω` without state-averaging —
   the paper-10 route, and a second, independent excited-state arm to check
   `roots=k` against.
3. **Energy-variance extrapolation.** `E` against `σ² = ⟨H²⟩ − ⟨H⟩²`, linearly
   extrapolated to `σ² = 0`. The standard selected-CI correction; it applies to
   the Phase 8 QSCI arm and the A-CASE arm alike, from the same bank.
4. **A convergence criterion that is not the objective.** Growth currently stops
   on predicted lowering — the same quantity it maximizes. A residual-based
   stop is independent of the selector.

*The cost is real and must be measured before it is claimed.* `supp(H²)` is far
larger than `supp(H)`, and Phase 7's lesson was that the quantity which looks
expensive by inspection is not always the one that is — measured both ways.
Paper 10's own contribution is a Pauli-grouping procedure for the squared
Hamiltonian; that belongs in `measurement/grouping.py`, next to the QWC
partition, and Phase 12 is where it pays off.

*Go/no-go:* `dense_residual_norm` (already in `subspace/reference.py`, already
used as the Phase 3 cross-check) agrees with the banked residual to solver
tolerance on TFIM and both H₄ legs, and the variance extrapolation reduces the
`h4_equilibrium` A-CASE error below its `+3.0e-03` at unchanged `M`.

### Phase 11 — real-time (unitary) Krylov generators (paper 6)

The central trade in the Phase 7 ladder, in the plan's own words: fixed Krylov
wins the energy on every fermionic rung by up to five orders of magnitude, and
pays with `κ_S` from `3.3e+04` to `6.6e+10` — "an overlap matrix no finite-shot
run could invert" — against A-CASE's `κ_S = 1.0`.

Real-time evolution is the standard cure for exactly this. `e^{-iHt_k}|ψ⟩` spans
a Krylov-like space through *unitary* operators, so the basis vectors are
norm-preserving and the overlap matrix does not degenerate the way powers of
`H` do. Add `real_time_response(hamiltonian, times)` to
`subspace/generators.py` as a level-3′ family, built on the existing
Trotter/`expm` helpers in `gates.py`.

Two payoffs from one family:

- **A-CASE:** the conditioning-versus-accuracy trade the ladder documents may
  be a false dilemma. Test it.
- **QSCI:** paper 6's input state is a time-evolved state, and the method is
  *optimization-free* — no VQE to converge before sampling. Phase 8 gets a
  second, cheaper input-state route for free.

*Go/no-go:* on `h4_equilibrium` and `h4_stretched`, a real-time family at
matched `M = 9` reaches within one order of magnitude of fixed Krylov's energy
(`1.0e-08` and `1.5e-05`) at `κ_S` below `1e+04`. *Falsifier:* `κ_S` tracks the
`H^k` family, in which case the conditioning is intrinsic to spanning that
space and the trade is real.

### Phase 12 — Clifford-accessible measurement (papers 9, 15)

`measurement/grouping.py` currently partitions into qubit-wise commuting groups
only. Fully-commuting (FC) groups are larger, and diagonalizing one requires a
Clifford circuit — which this package can build natively from
`bridges/stim_bridge` tableau conjugation and `backends/stabilizer`.

Paper 15 arrived after this roadmap was first written and changes three things
about the phase: it prices it, it corrects the metric I had chosen, and it
supplies an invariant we can check without trusting any of its numbers.

**12A — the parity ceiling, as a standing invariant (do this first).**
Paper 15 proves `r_X ≤ 2(N−1)` for spin-conserving Jordan–Wigner molecular
Hamiltonians, for *any* Pauli subset, tight even within commuting subsets,
where `r_X` is the GF(2) rank of the words' X-masks. That is a theorem about
exactly this package's setting: JW throughout, sector-restricted,
spin-conserving.

It is also nearly built. `sparse.py::word_masks` already returns `x_mask` as a
named, tested function, and `backends/sector_statevector.py::SectorOperator`
already **groups the Hamiltonian's words by X-mask** and reports
`self.groups = len(groups)` as a live metric — the backend does it to resolve
the permutation `b → b⊕x` once per group and make the matvec cheap. The paper's
routing diagnostic and the backend's compilation strategy are the same object
seen from two sides: one bounds how many measurement contexts you need, the
other exploits the same structure to avoid a `2^n` lookup table. Computing
`r_X` is a GF(2) rank over masks we already extract.

So the ceiling becomes a *checkable structural invariant* on every fermionic
model the package builds, in the register §3 of the A-CASE plan prefers —
enforced structurally, not to a numerical tolerance. If any spin-conserving JW
Hamiltonian we construct violates `r_X ≤ 2(N−1)`, either the theorem is wrong
or our construction is, and both are worth knowing. Run it across the FCIDUMP
rungs, the lattice models, and the effective-Hamiltonian ingestion path.

This is also the honest way to cite a two-week-old preprint: reproduce the part
that is a theorem, and treat the benchmark numbers separately.

**12B — QWC → QWC+FC, priced.**
My original go/no-go asked for a ≥ 3× drop in *group count*. That is the wrong
metric and paper 15 is careful about precisely this: it reports a **31–70 %
reduction in certified leading shot cost** on four 29–35 qubit f-element
Hamiltonians, and labels it "a QWC-versus-QWC+FC result rather than a
Gaussian-versus-Clifford pricing" — i.e. the gain is attributed to the setting
enlargement, not smuggled in from a different baseline. Group count and shot
cost are not the same quantity, because merging words into larger groups
changes the variance allocation across them as well as the number of readouts.

Restate the gate in the currency that matters: **certified leading shot cost,
QWC against QWC+FC, same allocator, same δ, same word universe.** The
comparable target here is the committed H₄ element universe (15 846 words,
1 689 QWC groups) and the `h2o_cas8e6o` universe (`W = 143 117` at `M = 9`).
Our ladder tops out at 12 qubits against their 29–35, so a 31–70 % band is a
reference point and not a prediction — say which it is when reporting.

*Go/no-go:* certified shot cost falls materially at fixed δ; reconstruction of
`(S, H)` from the FC cache reproduces `exact_matrices()` to the same `5e-13`
criterion already used for the QWC route; and — the trap — variance accounting
stays correct. Phase 4 found that attributing a word to every *capable* group
inflated every variance by that multiplicity, caught only by comparing
predicted σ against a Monte-Carlo spread. FC groups make more words
multiply-capable, so that test is the gate, not a formality.

**12C — what the separation theorem does and does not say.**
Paper 15's Bell-diagonal Heisenberg-type witness has correlation rank three: it
needs at least three orbital-rotation contexts, while one explicit physical
Clifford circuit measures its commuting Pauli representatives. Orbital
rotations are therefore *provably* insufficient, and Clifford contexts strictly
beat them. That is an argument for this package's identity rather than a
feature request — a `Cl(2n,ℂ)`-native engine with a Stim bridge and a
stabilizer backend should own Clifford-accessible measurement.

Two precision points, because both are easy to get wrong in a positioning
paragraph. First, the paper is explicit that **X-rank is a routing diagnostic
and the strict separation is carried by the correlation-rank theorem** — do not
present `r_X` as the source of the separation. Second, the correlation-rank
result is proved in the fixed `(1,1)` sector of two spatial orbitals per spin,
with the best `K`-context approximation exactly the Eckart–Young singular-value
tail; it is a sharp small-system statement, not a general-molecule bound, and
citing it as the latter would be the same overreach §6 of the A-CASE plan keeps
catching in resource claims.

### Phase 13 — pool and mapping breadth (papers 3, 11)

**13A — CEO pool as an ADAPT baseline (paper 3).**
Coupled Exchange Operators report up to 88 % CNOT reduction, 96 % depth
reduction and **99.6 % measurement-cost reduction** on 12–14 qubit molecules,
published in *npj QI*. That last number is a direct constraint on how Paper A
frames itself: measurement efficiency achieved by *pool design* is a solved
problem in the literature, and "measurement-efficient ADAPT-VQE" is not
available as a novelty claim.

`RESEARCH_PLAN.md` §1 already anticipated this shape of constraint for FAST-VQE
and Long et al. Extend the same discipline: add CEO to `algorithms/pools.py`
alongside the odd-Y pool, benchmark against it, and sharpen the Paper A claim
to what survives — **certification** (a δ-controlled wrong-selection rate with
an explicit ambiguity outcome), which CEO does not provide, on top of whatever
pool is in use. Note that CEO's multiple-operators-one-parameter structure maps
onto the existing `algorithms/layering.py` rather than needing new machinery.

**13B — Bravyi–Kitaev and parity mappings (paper 11).**
The package is Jordan–Wigner throughout. Two independent reasons to add BK and
parity through `bridges/openfermion_bridge`:

- *Measurement cost.* BK gives `O(log n)`-weight strings where JW gives `O(n)`.
  Pauli weight drives QWC group structure directly, so this is a `W` and
  group-count lever, not a cosmetic option — and it interacts with Phase 12.
- *A validation target.* Paper 11's claim is that fixed-ansatz UCCSD-VQE
  stagnates at zero energy shift under BK because of global phase cancellations
  in the BK tree, while adaptive selection escapes. Whether or not the paper's
  headline convergence claim holds up (§0.1), the *trap* is a falsifiable
  phenomenon this package could reproduce or refute, and the odd-Y pool
  reduction is explicitly derived for the antiunitary-real sector under JW —
  whether it survives BK is an open question in our own theory.

*Go/no-go for 13B:* the mapped Hamiltonian reproduces JW energies to `1e-12` on
`h4_equilibrium` (mapping is a unitary relabelling; anything else is a bug),
and the odd-Y exactness theorem is either extended to BK or explicitly
documented as JW-only.

### Phase 14 — embedding boundary (papers 7, 14)

Paper 7 uses projection-based embedding (VQE-in-DFT) and paper 14 uses DMET,
with QSCI as the fragment solver. Both hand a fragment Hamiltonian to a
correlated solver — which is precisely the interface `models/effective.py`
already defines and `models/fcidump.py` already implements for the CAS case.

Scope discipline, unchanged from §1 of the A-CASE plan: **do not build DMET.**
Build the boundary so an external driver can call `clifford_qc` as its fragment
solver:

- `effective_hamiltonian.v2` — a schema variant carrying an embedded one-body
  matrix plus a fragment/bath partition and full two-body integrals on the
  fragment, versioned and validated like `v1`.
- A fragment-solver callback returning energy **and** the one- and two-particle
  density matrices an embedding self-consistency loop needs. The projected-
  observable route (`result.expectation`, `result.transition`) already computes
  these without materializing a Ritz state; the RDMs are the natural consumer.

Paper 14's DMET-QSCI-RBM result — chemical accuracy at ~4 % of the subspace
against DMET-SQD failing at 20 % — then becomes a structure we can reproduce
with the Phase 8 and 9 arms in place of the RBM. The RBM itself stays out: a
generative model is a dependency the numpy-only core does not want, and the
comparison that matters is compact-subspace-versus-compact-subspace, not
architecture-versus-architecture.

---

## 4. Positioning changes to the manuscripts

Independent of any code, four statements in the drafts need to change.

1. **Paper B must cite and confront QSCI/SQD.** Papers 1, 2, 6, 14 define the
   frame. A subspace-eigensolver manuscript submitted in 2026 without a QSCI
   comparison will be desk-rejected on prior art.
2. **Paper 2's sentence is aimed at us.** "Improves over quantum subspace
   expansion based on single and double electronic excitations ... in both
   accuracy and efficiency" describes A-CASE's chemistry default. Quote it,
   answer it with the ladder, or narrow the claim.
3. **Paper A cannot claim measurement efficiency as novelty.** CEO reports
   99.6 % (paper 3, published). The defensible claim is certification with an
   explicit wrong-selection probability and an ambiguity outcome — which is
   what the code actually implements and what `RESEARCH_PLAN.md` §4 already
   describes.
4. **The A-CASE manuscript's ancilla-free claim is correct as written and needs
   a citation, not a correction.** `paper_acase/manuscript.tex` already says
   that reconstructing every entry from expectations on the same `ρ` — with
   "neither an ancilla-based Hadamard test nor a separately prepared
   `A_j|ψ⟩`" — "does not remove the cost: it moves the relevant cost boundary
   to the number and grouping of distinct Pauli words." Paper 15 is the work
   that makes that boundary quantitative, and belongs on that sentence.

   It also pins a claim we should never drift into making. Paper 15 notes that
   controlled-Pauli insertions in Hadamard tests **are Clifford**, so the
   ancilla-based route is not paying a T-count penalty; its own zero-`T`
   statements "concern measurement circuitry only, while shot counts and state
   preparation retain their full costs." A-CASE's ancilla-free property must
   therefore be argued in ancilla, connectivity and depth terms. The draft has
   never claimed a T-count advantage; this is to keep a later one from
   acquiring it.
5. **Paper 13 is a positioning asset, not a task.** FLDC establishes absence of
   barren plateaus for finite-local-depth circuits on local Hamiltonians.
   A-CASE has no trainability problem to begin with: the subspace solve is a
   generalized eigenproblem, not a circuit optimization, so barren plateaus are
   structurally absent rather than conditionally avoided. Say that once, in the
   positioning section, and build nothing. It also gives the ADAPT-VQE arm a
   principled warm-start argument.

---

## 5. Falsifiable questions

Continuing the numbering from `ACASE_RESEARCH_PLAN.md` §7.

- **Q5 (QSCI dominance).** At equal subspace dimension `M`, does the sampled
  determinant subspace match or beat A-CASE on energy while spending **zero**
  measured words on its projected matrices?
  *Falsifier for A-CASE:* yes on every fermionic rung. *Expected:* yes on the
  chemistry rungs; no on `kitaev_2x2`, where it has no arm.
- **Q6 (hybrid gain).** Does `QSCI determinants × operator dressing` reach
  accuracy neither arm reaches alone, at a `W` bounded well below the full
  A-CASE universe?
  *Falsifier:* the compound generators add nothing the determinant span did not
  already contain, on both Hubbard clusters.
- **Q7 (overlap selection).** Does an overlap-to-target criterion select
  level-4 generators that the generalized 2×2 lowering provably cannot see?
  *Falsifier:* `hubbard_2x3` rows stay identical between levels 0–3 and 0–4.
- **Q8 (unitary Krylov).** Does a real-time family recover fixed Krylov's
  accuracy at a condition number a finite-shot run could survive?
  *Falsifier:* `κ_S` tracks the `H^k` family within an order of magnitude.
- **Q9 (grouping).** Does Clifford-diagonalized fully-commuting grouping cut the
  **certified leading shot cost** — not the group count — at fixed δ, allocator
  and word universe, with variance accounting intact?
  *Falsifier:* group count falls while measured variance inflates, i.e. the
  multiply-capable-word bug of Phase 4 in a harder form. *Reference point, not
  a prediction:* 31–70 % (paper 15, at 29–35 qubits against our 12).
- **Q10 (parity ceiling).** Does `r_X ≤ 2(N−1)` hold across every
  spin-conserving Jordan–Wigner Hamiltonian this package constructs — FCIDUMP
  active spaces, lattice models, and the effective-Hamiltonian ingestion path?
  *Falsifier:* any violation, which convicts either the theorem or our
  construction. This is the one question here that costs almost nothing to
  answer and is worth answering first.

---

## 6. What this roadmap does not claim

No claim to: originating QSCI, SQD, configuration recovery, overlap-based
adaptive selection, folded-spectrum methods, real-time quantum Krylov,
fully-commuting or Clifford-accessible measurement grouping, the correlation-rank
and parity-ceiling results of paper 15, coupled exchange operators, or DMET.
Every one of those is prior art and is cited as such. The contributions on
offer are narrower and are stated as such above: the hybrid of a sampled
determinant span with operator-response dressing (§9), an overlap criterion
targeting the specific blind spot this project documented in its own selector
(§9B), and the resource accounting that makes the comparison between these
paradigms a measured statement rather than a rhetorical one (§6 of the A-CASE
plan, extended to arms that measure nothing at all).

Nor does this roadmap claim any of the phases will succeed. Three of the nine
go/no-go gates above are written so that the likely outcome is failure, and
§1 states in advance which comparison this project is expected to lose.
