# `clifford_qc` research plan — WISE–A-CASE (PRD)

**One document.** It consolidates the former `ACASE_RESEARCH_PLAN.md` (identity,
method, Phases 0–7), `LITERATURE_ROADMAP.md` (Phases 8–18, tracks, literature),
`RESEARCH_PLAN.md` (Paper A: certified ADAPT-VQE, §9), and `FINITE_SHOT_RETHINK.md`
(the Phase 4R lab note) — four documents that existed on `main`. The hardware-aware
resource accounting of §6 and Phases R1–R4 is **new design work**, introduced on this
branch (briefly as a separate `RESOURCE_ACCOUNTING_PLAN.md`, then merged here); it has
no provenance in the base tree and should be reviewed as new rather than as carried
over. The **GA structural preconditioner** of §3.5 and Phases G1–G3 is likewise new
design work; G1 now carries a measured result and G2–G3 are retired on it (§5),
so §3.5 should be read as a contract whose first phase has reported rather than
as an unbuilt proposal. Phase numbers, question numbers, and the
section numbers cited from code docstrings are unchanged, so existing references still
resolve; Paper A's own phases are relabelled A0–A5 to keep them distinct from the
Phases 0–19 of §5.

**Two papers are public as arXiv preprints** (§1.2). Everything they report is now prior art
*for this project's own future claims*: the plan may extend or contradict them, but it
may not re-claim them. Read §1.2 before writing any novelty statement.

The project's scientific identity:

> **Do low-energy states of correlated molecular and materials Hamiltonians have
> a compact adaptive representation in Clifford-algebra operator-response
> subspaces — and can that subspace be diagonalized reliably from globally
> shared, finite-shot Pauli measurements with explicit certificates?**

and the comparison that decides whether it is worth publishing:

> **Can sampled determinant subspaces, operator-response dressing, and
> support-pruned multiresolution packets produce a compact, resource-honest
> eigensolver that adds something classical selected CI does not already give?**

**Naming and current architecture.** The current programme is
**WISE–A-CASE (PRD)**: *Word-Reusing, Inference-Stabilized A-CASE* is the
measurement-and-inference architecture, while preconditioned residual Davidson
(PRD) is the default accuracy and basis-growth engine established by the completed
PRD-CASE suite. A-CASE remains the operator-response span, shared-word bank, and
matched comparator; it is no longer presented as the strongest accuracy engine.
DA-CASE is the published name of the dyadic measurement hierarchy (§1.2), and
historical source identifiers and record labels are not renamed in place.

**The architecture the plan is being restructured onto.** Structural admissibility
moves *before* the fermion-to-qubit encoding, so that what JW or BK receives is
already a symmetry-, reference-, and redundancy-reduced operator domain:

```text
fermionic problem
  → GA structural restriction        (§3.5, Phases G1–G3)  — mapping-independent
  → JW / BK encoding                 (§6.2, Phase R2)      — mapping choice enters here
  → A-CASE / PRD basis selection     (§4)
  → WISE measurement and inference   (§6)                  — cost inherits encoded locality
```

The division of labour that makes this worth doing is **GA = structural
admissibility, PRD = dynamical usefulness**. A candidate that is forbidden by
symmetry, leaks the sector, annihilates the reference, or duplicates another
candidate's physical action is not a question for a residual score to settle
statistically; it should never reach the ranking. Conversely GA says nothing about
which admissible direction lowers the energy — that is PRD's job and only PRD's.

This is *not* a claim that geometric algebra finds a smaller Hilbert space. It
restricts the candidate operator domain while leaving the Hilbert-space dimension
unchanged. What it supplies is a compact algebraic language for symmetry sectors,
ideals, parity, annihilated directions, and equivalence classes **before they
expand into Pauli words**. Action equivalence (§3.5E) is especially direct in this
language, although an equivalent projected-action test remains possible after
encoding.

The method is A-CASE, never "ACSE": in quantum chemistry ACSE is the
anti-Hermitian contracted Schrödinger equation (Mazziotti and successors), still
active in contracted quantum eigensolver research, and colliding with it would
corrupt literature searches and referee context. Terminology discipline: the
working space is a **Clifford-algebra operator-response subspace**. `P_j|ψ⟩` are
Pauli-orbit directions; `G_j|ψ⟩` with `G_j = -i/2·[H,P_j]` are commutator-response
directions (Pauli sums, not Clifford transformations); `H^k|ψ⟩` are Krylov
response states; genuine **Clifford-group orbit** states (versor/stabilizer
transforms of a reference) are one candidate family among several, not the
generic case. Compound generators need not be versors.

### How to read this

| you want | go to |
|---|---|
| **what is already public, and what that forecloses** | §1.2 |
| what the project is and is not | §1 |
| why the architecture is what it is | §2 |
| the algebra contract and standing invariants | §3 |
| **the GA structural preconditioner (new)** | §3.5 |
| **anticommuting cliques as spin factors (new)** | §3.6 |
| the method itself | §4 |
| **status: what is built, and what it measured** | §5, Phases 0–14 and PRD |
| **post-core programmes and remaining work** | §5, Phases 15–19, G1–G3, R1–R4 |
| how cost is counted, and the device model | §6 |
| the validation ladder and benchmark inventory | §7 |
| **Paper A — certified ADAPT-VQE, written but unpublished** | §9 |
| what would falsify each claim | §10 (A-CASE), §9.2 (Paper A) |
| what is deliberately not claimed | §14 |

Status at a glance:

<!-- PHASE-STATUS-SUMMARY:START -->
| numbered phase scope | lifecycle | implementation |
|---|---|---:|
| Phases 0--14 | complete | 15 / 15 |
| Phase 15: Second-moment bank | open | 0% |
| Phase 16: Time-evolved inputs | open | 0% |
| Phase 17: Mapping validation and breadth | partial | 75% |
| Phase 18: Embedding boundary | partial | 50% |
| Phase 19: Anticommuting-clique partitioning | proposed | 0% |

Strict complete-phase score: **15 / 20 = 75.00%**.
Progress-weighted score: **16.25 / 20 = 81.25%**.
Retired and conditional adjunct phases are tracked separately and do not change this denominator. Source: `PHASE_STATUS.json`; validate with `python benchmarks/check_phase_status.py`.
<!-- PHASE-STATUS-SUMMARY:END -->

The machine-readable block above reports implementation progress only. The detailed
ledger below preserves scientific outcomes, evidence boundaries, retired work, and
conditional phases.

| phase | subject | status |
|---|---|---|
| — | **arXiv:2608.00560** — A-CASE, operator-generated subspaces | **public arXiv preprint** (§1.2) |
| — | **arXiv:2608.08739** — DA-CASE, reusable measurements | **public arXiv preprint** (§1.2) |
| A0–A5 | Paper A: exact research layer, measurement/confidence layer, scaling and layering, stabilizer initialization, chemistry | **done**; the manuscript is written and checked (§9.6.1), and is the one manuscript not yet submitted |
| 0–7 | exterior layer, subspace solver, bank, adaptive growth, finite-shot certification, lattice models, sector backend, validation ladder | **done** |
| 4R | pooled reconstruction and rank selection | **done**, off by default |
| 8–12 | QSCI baseline, selected-CI controls, hybrid, overlap/multiresolution selection, Paper B ladder | **done and read**; the result motivated the PRD programme below |
| PRD | orthogonal-residual regression, Davidson/preconditioned expansion, packet pricing, matched A-CASE, exact and finite-shot suites | **done and read**; exact compactness is positive, complete-bank QWC finite-shot energy accuracy is negative |
| 13 | parity/X-rank invariant | **done**; one shared GF(2) implementation is tested across every declared spin-conserving JW construction path before grouping |
| 14 | fully commuting grouping | **done on the frozen BeH₂/JW comparison** — the public compiled-plan API, exact joint sampler, preregistered finite-sample record, covariance audit, and device-card costing ship; Q9 is positive within the declared one-bank oracle boundary |
| 15 | second-moment bank | **open, unimplemented**; no `SecondMomentBank` or H-squared support/cost preflight ships |
| 16 | time-evolved inputs | **open, unimplemented**; matrix-free action exists, but neither the QSCI time-evolved input nor a circuit-native/truncated A-CASE real-time family ships |
| 17 | mapping validation and breadth | **partially implemented (75%)**; the JW/BK/parity transformation, invariance gates, and mapping-axis records ship through R2, while the CEO-pool and dedicated MORE-ADAPT breadth benchmarks remain open |
| 18 | embedding boundary | **partially implemented (50%)**; the versioned effective-Hamiltonian schema ships in `models/effective.py`, while the common fragment-solver callback returning energy and one- and two-particle density matrices does not |
| 19 | anticommuting-clique (spin-factor) partitioning | **proposed, unexecuted**; the algebra is §3.6 and the scope is §5, Phase 19. Lever 1 is scoped to a fixed-coefficient Hamiltonian-energy estimand, *not* to Phase 14b's matrix-element word bank; lever 2 needs a non-Clifford transport primitive that does not exist yet. An in-session structural probe supplies the sizing numbers and is explicitly **not** a committed record — no producer, config, record or checker exists, so nothing there licenses a rung, a price, or an arm |
| G1 | GA structural preconditioner: Majorana pool and filters A–E | **done**; both gates pass and QG1's falsifier does not fire, but E's content is pool-dependent — `522` of `549` removed on the §3.5 Majorana pool, `0` on the excitation pool the mapping records use |
| G2–G3 | mapping-invariance test on the restricted pool, PRD/WISE integration with a cost decomposition | **retired by G1's own result**; the G1-admissible pool reconstructs the pool R2b already builds, so QG2's falsifier holds by construction rather than by measurement (§5, Phase G2) |
| R1 | hardware-aware cost model and pooled-estimator ledger | **done**; asymptotic and exact-oracle nonlinear shot-search tiers are recorded |
| R2a | shared restriction primitive (`subspace/restriction.py`) | **shipped and consumed by the completed R2b record** |
| R2b | raw-pool mapping axis | **done, negative QR3 result** — QR2 passes, but mapping spread is not smaller than instance spread on either independent fixed-QWC metric |
| R3 | protocol axis and accuracy-matched cost regions | **done**; the structural and exact-tier records ship, and `protocol_cost.json` abstains on QR3 because H4-converged is right-censored at the frozen endpoint ceiling. That abstention stands as a statement about that record; R3S opened the route to a second priced instance without widening the grid, and R3c walked it |
| QR3b | chemically independent LiH exact-tier extension preflight | **done, negative scope decision** *for the intrinsic-stop bank* (`M = 13`, `W = 7740`); the bias gate passes, but 21/40 cells are unresolved and 9 more resolve only at the frozen ceiling, so no full run is authorized on that bank. The screened `M = 2` bank is a separate candidate — R3b |
| R3S | priceability screen — which candidates a declared screen admits for a probe | **done, positive**; under a stopping rule that reserves the accuracy target for shot noise, LiH is admissible at a two-generator prefix (`W = 1439` against BeH₂'s `1814`), so QR3b rejected the greedy's stopping point rather than the instance |
| R3b | LiH `margin_stop` probe — the QR3b 2+2 preflight re-run on the bank R3S admitted | **done, mixed**; the target-environment redraw rejects the bank (29/40 cells resolve, versus 30/40 in the historical 3.11 draw; the gate needs 40), so that probe authorized no full run and left the exact tier with one priced instance — but both draws reduce QR3b's grid-fit failures `15 → 0`, leaving the `2+2` probe's confirmation power rather than `W`, which is what R3c went on to test directly |
| R3c | LiH `margin_stop` headline exact-tier cost run | **done, positive**; executed once under the frozen 3.12 / 2.5.2 stack. 38 of 40 cells supply a confirmed finite interval and 2 stay right-censored, so LiH is the exact tier's second priced instance. QR3 is re-derived on two instances and comes back `indeterminate_at_this_shot_grid`: the point estimates order 17.5x against 1.02x, and the cost brackets overlap |
| R3d | QR3 within-grid resolution | **done, indeterminate**. The one frozen 30+100 execution narrows four of five parent cells; BeH2's `8192` midpoint is environment-marginal and cannot tighten. The mapping-support minimum is `2.1906x` against an instance-support maximum of `3.9998x`, so strict positive separation fails. The targeted design licenses no negative or mapping-wide readout |
| R4 | contextual-subspace comparator | **R4a structural screen complete, negative**; no contextual rung admits all four arms, so the frozen rule stops R4a without sampling and leaves QR5 undetermined |

"Shipped" means the module, its tests, and where applicable its benchmark
producer exist. It does not mean the phase's go/no-go has been read: those
verdicts live in the committed records and in §7, and a phase can be built and
still be waiting on the comparison that decides it.

---

## 1. Scope and positioning

`clifford_qc` is a **correlated active-space and lattice-model solver**, not a
general materials suite:

```text
DFT / Wannier / embedding  →  effective many-body Hamiltonian
                           →  clifford_qc (this plan)
                           →  energies, states, correlations, response
```

It does not compute band structures, forces, phonons, or geometry optimization;
established DFT codes own that layer. `clifford_qc` solves the small-but-hard
correlated subproblem (active spaces, Hubbard/Kanamori clusters, impurity
models, spin lattices) that downfolding or embedding produces, and returns
energies, low-lying spectra, and material observables.

**Method stack (three tiers).**

1. **WISE–A-CASE (PRD)** — Davidson/preconditioned residual growth is the
   primary accuracy engine; the WISE layer reuses the global Pauli-word bank and
   owns stabilized measurement and inference. A-CASE supplies the
   operator-response span machinery and the matched infrastructure comparator.
2. **Exact references** — dense `exact_ground`, PySCF FCI metadata, the
   scipy-sparse `eigsh` tier (`sparse.py`), and the sector-restricted spinor
   backend with matrix-free Lanczos for `n` beyond dense reach. Baselines, not
   the identity.
3. **ADAPT-VQE** (Paper A machinery, complete) — the variational comparison
   point at matched operator budget.

### 1.1 Minimum viable materials-facing showcase — shipped

`models/effective.py` is the narrow boundary the diagram above was missing. It
reads the versioned schema `clifford_qc.effective_hamiltonian.v1`: a Hermitian,
spin-independent Wannier one-body matrix, local `U_i`, chemical potential,
energy-unit label, explicit interleaved spin-orbital reference, and optional
provenance. It builds the ordinary second-quantized Hamiltonian and maps it
through the package's existing Jordan-Wigner layer. It does **not** claim to run
DFT, Wannierization, cRPA, or DMFT.

The canonical integration rung is the two-site Hubbard dimer (`t=1`, `U=4` eV,
four qubits, sector `N=2, S_z=0`) — the smallest system that exercises the whole
output contract while retaining an independent oracle:

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
compactness result and is not offered as one — `M = 4` **is** the sector
dimension, so the response arm is full configuration interaction in that sector,
chosen deliberately so the Lehmann lines are exact while the adaptive arm is
scored separately on the energy.

The benchmark is intentionally labelled synthetic: it proves that an effective
Hamiltonian can cross the software boundary and produce energies, projected
state coefficients, correlations, and response. It does not establish materials
accuracy or a quantum advantage.

### 1.2 Public preprints — what is already reported

Two manuscripts from this repository are public. They are the project's own prior
art: a future manuscript may **extend, sharpen, or contradict** them, but it may
not present their content as new, and any claim of novelty has to say what it adds
on top of these two.

| # | arXiv | Title | Submitted | Source in tree |
|---|---|---|---|---|
| P1 | [2608.00560](https://arxiv.org/abs/2608.00560) | Adaptive operator-generated subspaces for effective many-body Hamiltonians | 1 Aug 2026 | `paper_a_case_subspaces/`, restored from `67ea0dd` |
| P2 | [2608.08739](https://arxiv.org/abs/2608.08739) | DA-CASE: reusable measurements for adaptive quantum subspaces | 9 Aug 2026 | `paper_acase/` (current) |

Both are Utama & Dipojono. The two originally shared one manuscript directory in
sequence: `paper_acase/manuscript.tex` carried P1's title through commit `67ea0dd`
and was retitled to P2 afterwards. The published P1 text sources are now restored
byte-for-byte under `paper_a_case_subspaces/`, with their source-commit/blob
manifest and independent snapshot/manuscript checks. `paper_acase/` remains P2.

**What P1 established.** A-CASE as a single-reference, operator-generated
Rayleigh–Ritz method: matrices reconstructed from one shared Pauli-expectation
bank; adaptive growth scoring overlap-aware local pencils and rejecting symmetry
leakage and near-linear dependence; the mapped sector of linear H₄ / STO-3G
CAS(4e,4o) matching independent determinant FCI; comparison against ADAPT-VQE
variants at matched budget; and the executable path from an effective Hamiltonian
to energies, correlations, and response (§1.1). Explicitly **not** claimed there:
materials accuracy, favourable scaling, quantum advantage.

**What P2 established.** The measurement-reuse layer: basis states generated from
a single reference; Hamiltonian and observable matrices reconstructed from cached
Pauli expectations rather than from separately prepared states; identical
nine-dimensional subspaces and energy precision across generator resolutions while
the required word bank falls 7371 → 2240; commuting settings 913 → 64 across the
dyadic hierarchy; and covariance-aware shot allocation cutting projected-matrix
variance by 68.9%. Explicitly **not** claimed there: hardware demonstration,
scaling result, quantum advantage — the authors' own words are "small-instance
exact and Monte Carlo results".

**Consequences for the rest of this plan, stated once so they are not re-litigated
per section:**

1. **Word-bank and setting-count reduction on the frozen H₄ bank is already
   reported in P2.**
   `7371 → 2240` and `913 → 64` are P2's numbers. §6.6 and the
   `clifford_hierarchy_*` records reproduce them; they are a regression baseline
   now, not a finding. A new claim in this area has to be about *accuracy-matched
   cost* (§6.1), not about counts.
2. **Covariance-aware allocation is already reported in P2.** The 68.9% variance
   reduction is P2's. Phase 4R and the WISE inference layer may be cited, not
   re-announced.
3. **The exact-arithmetic compactness of operator-generated subspaces is
   already reported in P1.** What remains open — and what the PRD suite's
   negative finite-shot result (§5, Phase PRD) makes the live question — is
   whether that compactness
   survives finite-shot energy accuracy. That, not compactness, is the forward
   claim.
4. **Paper A (§9) is the one written manuscript still unsubmitted.** Its subject,
   certified ADAPT-VQE operator selection, is disjoint from P1 and P2, so it is
   unaffected by the two publications except that it may now cite them as
   companion work rather than as forthcoming.
5. **The GA structural preconditioner (§3.5) is the first newly proposed
   scientific direction in this plan after P2.** Phase G1 is now built and
   measured (`g1_structural_preconditioner.json`); G2 and G3 are not, and are
   retired rather than pending. What G1 established is a structural agreement
   and one pool-dependent marginal, not a resource result: nothing in §3.5 has
   been shown to reduce any cost, any word universe, or any Hilbert space, and
   no part of it may be described as one.
6. **"Paper B" is a manuscript line, not a manuscript.** It named the A-CASE/QSCI
   work of §1–§8; P1 and P2 are its public arXiv outputs. The label survives in
   Phase 12, §7, and the committed records, so it is not renamed — but where the
   text says "Paper B" it now means the unwritten successor in that line (§9,
   naming collision 3).

---

## 2. Repository facts that determine the plan

### 2.1 Why A-CASE comes before matrix-free Lanczos (architecture audit)

An earlier ordering put matrix-free Pauli Lanczos first. The audit of what the
merged codebase already provides reversed it:

- **The basis states stay virtual.** `|φ_i⟩ = A_i|ψ⟩` is never prepared; every
  projected element is an expectation **on the single reference state**:
  `S_ij = ⟨ψ|A_i†A_j|ψ⟩`, `H_ij = ⟨ψ|A_i†HA_j|ψ⟩`. One reference, one global
  Pauli-word universe, one measurement cache, every measured word reused across
  many entries and candidates. This — not the mere use of Clifford algebra — is
  the project's strongest computational proposition, and it is exactly the
  `CommutatorBank` architecture generalized from candidate-by-word to
  matrix-entry-by-word.
- `states.expectation(rho, O) = Tr(O·rho)` is complex-valued and does not require
  `O` Hermitian (`states.py`). With `MV.dagger()` and the word product, every
  matrix element is computable today, with zero new backends.
  (`ExactMVBackend.expectation` forces `.real`; A-CASE calls the core function,
  not the backend wrapper.)
- In Pauli-word coordinates, `Tr(O·ρ) = 2^n Σ_w o_w·r_w` — a bilinear,
  conjugation-free coefficient pairing.

A large-`n` spinor/Lanczos substrate is a genuinely new backend (Phase 6), not a
matvec swap, and finite-shot certification of a nonlinear projected eigenproblem
is a new statistical problem (Phase 4), not a reuse of the gradient confidence
code.

### 2.2 Matrix-free action changes the feasible boundary

`PauliLinearOperator` and `SectorOperator` provide matrix-free action and Krylov
solves without materializing dense operators: exact QSCI sampling-state oracles
on small and medium systems, restriction to sampled index sets, propagation
diagnostics, residual and variance checks. They do **not** make real-time
operators sparse in the multivector generator bank; circuit-native real-time
A-CASE remains a separate architectural problem (Phase 16B).

### 2.3 Adaptive workflows have clean extension seams

`AdaptConfig`/`AdaptState`/`adapt_step` and `ACASEConfig`/`ACASEState`/
`acase_step` allow hierarchical or racing policies without unifying the distinct
mathematics of ADAPT-VQE and A-CASE. Multiresolution selection targets these
boundaries; it does not introduce a generic adaptive runner.

### 2.4 QSCI is the external-validity comparison, and it now exists

This subsection was written when `clifford_qc` had no first-class QSCI/SQD
implementation and no QSCI row on the ladder — the principal external-validity
gap for Paper B. Phase 8 closed it (`subspace/qsci.py`, ladder arm `qsci`); what
survives is the contract that made it a gap, because it governs how the
comparison must be reported. QSCI and A-CASE spend different resources:

| | A-CASE | QSCI/SQD |
|---|---|---|
| basis | virtual operator states `A_i|psi>` | sampled basis configurations |
| overlap | measured nonorthogonal `S` | identity |
| projected Hamiltonian | reconstructed from measured Pauli words | built classically |
| main quantum cost | state preparation plus grouped word measurement | state preparation plus computational-basis sampling |
| principal risk | `W`, shot cost, and conditioning | duplicate sampling and determinant compactness |

`W=0` for QSCI projected-matrix measurement is correct, but it is not a complete
resource verdict. The comparison must also report sampling yield, preparation
cost, classical matrix construction, diagonalization cost, and memory. (Code
citing "§0.1 of the literature roadmap" refers to this subsection.)

### 2.5 Configuration-space Haar is implemented; orbital wavelets are not the target

`subspace.configuration.configuration_haar_packets` builds an orthogonal finite
tree-Haar transform over a caller-ordered configuration list — a classical change
of basis among virtual generators, not a quantum wavelet circuit. Support pruning
makes it an opt-in coarse tier rather than a convergence-complete basis. The
committed 2×2 Hubbard benchmark gives one positive finite instance: staging Haar
packets before the ordinary level-4 pool reaches the exact sector energy with
33.1% fewer projected Pauli words and one fewer basis direction, while increasing
maximum element support and `kappa(S)`. That justifies packets as a hybrid
ablation, not as a default.

The orbital-basis benchmark found that no basis wins uniformly and that the
tested Daubechies orbital bases lose to site or momentum bases on the relevant
trade-offs. Orbital basis stays a recorded model parameter; orbital-wavelet
optimization is not reopened without a new, system-specific, falsifiable reason.

---

## 3. Conceptual layer (geometric-algebra contract)

Every phase states its algebra, objects, and validation invariants before its
implementation substrate.

### 3.1 Algebra and its two representations

**Algebra.** `Cl(2n,ℂ) ≅ M(2^n,ℂ)` throughout. Two representations of the *same*
algebra are used, and they must not be conflated:

| Representation | Object | Storage | Role |
|---|---|---|---|
| Operator-centric (current) | density multivector `ρ ∈ Cl(2n,ℂ)`, Pauli-word basis | up to `4^n` words | operators, ADAPT, A-CASE matrix elements |
| Witt-ideal spinor (Phase 6) | `Ψ ∈ Cl(2n,ℂ)·P₀`, `P₀ = ∏_j c_j c_j†` | `2^n` ideal components; `C(n,k)` in a particle sector | large-n pure states, Lanczos |

The Witt/minimal-left-ideal representation is the GA-native name for the
"symmetry-restricted determinant basis": computational determinants are the
occupation words `(c†)^{x₁}…(c†)^{xₙ}·P₀`, particle-number and `S_z` sectors are
subspaces of the ideal spanned by fixed-weight occupation words, and
Jordan–Wigner dressing is built into the Witt basis rather than bolted on. Adding
it is not a departure from the operator-centric identity; it is the minimal left
ideal of the same algebra, and it removes the `4^n` density-word blow-up for
strongly correlated pure states. Its engineering surface is nevertheless plain
(`SectorStatevectorBackend`); the ideal language belongs to the theory sections,
not the API.

### 3.2 The three pairings

Conflating them produces silent conjugation or sign errors:

| Pairing | Formula (word coordinates) | Character |
|---|---|---|
| `scalar_product` | `Σ_w a_w b_w (−1)^{k_w(k_w−1)/2}` | bilinear, reversion sign (`⟨A~B⟩₀`) |
| `hs_product` | `Σ_w conj(a_w) b_w` | sesquilinear (`Tr(A†B)/2^n`) |
| `trace_pairing` (Phase 1, shipped) | `Σ_w a_w b_w` | bilinear, no reversion, no conjugation (`Tr(AB)/2^n`) |

Matrix elements need `trace_pairing`: `A_i†HA_j` is non-Hermitian, so its word
coefficients are complex and `hs_product` would conjugate them wrongly, while
Hermitian test cases (real coefficients) would mask the bug. `trace_pairing` is a
distinct primitive, not an overload, and it is the exact foundation of the
matrix-element bank. It also avoids forming the full product `O·ρ` just to read
one scalar.

### 3.3 Objects

- Pauli words: blades (up to phase) under the JW correspondence; the exterior
  layer (`reverse`, `wedge`, `scalar_product`, `is_blade`) makes grade/blade
  structure first-class.
- A-CASE generators `A_i`: multivectors from the hierarchy in §4.2. Stabilizer
  configurations (genuine Clifford-group orbits of `|0…0⟩`) admit tableau-cheap
  overlaps through the Stim bridge; they are the "competing mean-field /
  magnetic-order configurations" of the materials strategy.

### 3.4 Standing invariants

Checked in tests, not prose:

- `E_sub ≥ E₀` for every exact-arithmetic subspace (variational bound); monotone
  non-increasing under nested basis growth.
- `S ⪰ 0`, `S = S†` exactly; `H` Hermitian exactly (enforced structurally, not
  numerically).
- Generator-scaling invariance: replacing `A_i` by `cA_i` must not change which
  physical directions survive thresholding (§4.1 normalization).
- Reproduction: a subspace containing the exact ground state returns the FCI
  energy to solver tolerance (validated against `exact_ground` and PySCF
  `fci_energy` metadata for every chemistry model).
- Do **not** truncate by GA grade. Grade is not a good quantum number for
  JW-dressed Hamiltonians; truncation criteria are particle number, `S_z`, Pauli
  support, excitation rank, residual coupling, and conditioning. §3.5A restates
  this in Majorana language, where it is the same statement about products of
  creation and annihilation operators mixing Clifford grades.

### 3.5 The GA structural preconditioner (contract; G1 measured, G2–G3 retired)

The filters below run **before** the fermion-to-qubit encoding. That placement is
the whole point: after JW or BK, the restrictions become encoded Pauli/symmetry
conditions and action equivalence is less transparent and usually more expensive
to test, whereas before encoding both are direct algebraic conditions on a few
hundred abstract operators. This
section is the contract Phase G1 was built against; its measured outcome is in
§5 (Phases G1–G3) and §10 (QG1), and the boundary on what that outcome licenses
is §1.2(5) and §14.

**Where the algebra actually lives.** For `n` fermionic modes introduce `2n`
Majorana generators with

```text
γ_μ γ_ν + γ_ν γ_μ = 2 δ_μν ,      a_p = (γ_2p + i γ_2p+1)/2 ,
                                  a_p† = (γ_2p − i γ_2p+1)/2 .
```

The Majorana generators generate the **real** `Cl(2n,0)`. The `i` in `a_p`
does not live there, so ladder operators and their complex linear combinations
require its complexification `Cl(2n,ℂ)` — the same algebra §3.1 already fixes,
reached through a Majorana rather than a Pauli generating set. Sector projectors
do not by themselves force that extension: when every `S_α` is real, the
projector in §3.5C is already an element of the real algebra. The complex working
algebra is fixed by the ladder operators, complex coefficients, and Hilbert-space
adjoint structure, not by the projector formula.

That real-projector branch is nevertheless empty for the stabilizers this plan
actually uses. A product of `k` distinct Majoranas squares to `(−1)^{k(k−1)/2}`,
so real involutions exist only for `k ≡ 0, 1 (mod 4)` — while occupation,
per-mode parity, and `S_z` are built from `k = 2` products,
`n_p = (1 + i γ_2p γ_2p+1)/2`, which square to `−1` and therefore carry the `i`
explicitly. So the conditional above is correct in general and vacuous here: a G1
implementation that chooses the real algebra on the strength of it meets the
complexification at its first stabilizer.

JW and BK then become two qubit representations of one fermionic algebra rather
than the place where the physics is defined (P-BK, P-ENC in §11).

**A — parity / even-subalgebra restriction.** Electronic Hamiltonians satisfy
`[H, (−1)^N̂] = 0`, so parity-preserving candidate operators lie in the even
subalgebra `𝒜⁺ ⊂ Cl(2n,ℂ)`. Whole classes of odd candidates disappear before
encoding.
Parity is the robust invariant here; **grade is not** — products of creation and
annihilation operators generally resolve into mixtures of Clifford grades, so
grade may be recorded as an additional structural descriptor but never used as
excitation rank (§3.4).

**B — conserved-quantity centralizer.** For conserved `Q_α` (particle number,
`S_z`, fermionic parity, point-group and molecular symmetries), the centralizer

```text
𝒞_Q = { A ∈ 𝒜 : [A, Q_α] = 0  ∀α }
```

is a mapping-independent **sufficient** restriction when the pool contract requires
each operator to conserve every `Q_α` globally. It is not necessary for
reference-conditioned admissibility. If `Q_α ψ_ref = q_α ψ_ref`, the exact
same-sector condition is `(Q_α − q_α) A ψ_ref = 0`; §§3.5C–D test that action
directly. G1 must not discard a candidate merely because global commutation fails
when its projected action on the reference remains in sector, or it cannot satisfy
the gate requiring agreement with the existing reference-aware leakage filter.

For a deliberately targeted *different* irrep — the excited-state case of §7.4 —
the requirement is a declared target character (or target eigenvalue) on
`A ψ_ref`, not unconditional commutation. Any implementation that hard-codes
`[A,Q]=0` forecloses the excited-state track, so the character is a parameter
from the start.

**C — sector idempotents.** For commuting `S_α` with `S_α² = 1` and signs
`s_α = ±1`,

```text
P_s = ∏_α (1 + s_α S_α)/2 ,        A_phys = P_s A P_s ,     keep iff A_phys ≠ 0 .
```

This is the CS-QSE structural rule `{A, S_k} = 0 ⟹ π_ν(A) = 0` expressed as a
Clifford projection: an operator anticommuting with a selected stabilizer moves
the state to an orthogonal sector and vanishes under projection. Projections and
idempotents are native structures in Clifford formulations of quantum mechanics
rather than bolted-on matrix constructions (P-IDEM, §11). This filter is the
same object R4 studies as the contextual-subspace comparator, which is why R4 and
G1 must share one implementation (§5, Phase G1).

**D — reference ideal.** Choose a primitive idempotent `f` generating the
minimal left ideal `Cl(2n,ℂ) f`, and represent the reference spinor as
`ψ_ref = Ψ_ref f` in that ideal. The idempotent, the ideal it generates, and a
particular spinor in the ideal are distinct objects; `ψ_ref = f` is only the
special case where the chosen primitive idempotent itself represents the reference.
A candidate's action is `A ψ_ref`. Then

```text
discard  A ψ_ref = 0                 (A may be nonzero but annihilates the reference)
require  P_phys A ψ_ref = A ψ_ref   (does not leave the target sector)
```

This is the GA form of the reference-aware leakage restriction the project has
been converging on from the Pauli side.

**E — action-equivalence quotient.** Inside a fixed sector
`S_α ψ_ref = s_α ψ_ref`, hence `A S_α ψ_ref = s_α A ψ_ref`: the operators `A`
and `A S_α` generate the same physical direction up to sign. More generally define

```text
A ∼ B   iff   P_phys A ψ_ref = λ P_phys B ψ_ref ,   λ ≠ 0
```

and carry one canonical representative of `𝒜_candidate / ∼`. This is **not**
Pauli-word deduplication: it removes candidates whose *physical action on the
reference* coincides. The equivalence is representation-invariant and can also be
tested after encoding by comparing projected actions; GA exposes it before Pauli
expansion, and no existing code path performs that quotient. It is the one proposed
filter in this section with no current analogue in the codebase. If any
part of §3.5 justifies the work, it is this one — so G1 reports its marginal
contribution separately from A–D (§5, Phase G1).

**The ordering, and what each stage removes:**

```text
𝒜_raw
  → A, B   symmetry / parity              → 𝒜_sym
  → D      reference ideal                → 𝒜_ref
  → C      sector projection              → 𝒜_phys
  → E      action-equivalence quotient    → 𝒜_unique
  → PRD    residual ranking               → 𝒜_selected
```

PRD then spends no residual evaluations on candidates that are forbidden,
sector-leaking, annihilating, or redundant.

**The congruence rule, inherited from CS-QSE and non-negotiable.** Whatever
restriction is introduced must be applied congruently to the Hamiltonian, the
reference and ansatz operators, and the expansion operators. Inconsistent
projection generates symmetry contamination and spurious or ill-conditioned
directions — this is exactly the failure mode §5's `Restriction` object exists to
prevent for R2/R4, and G1 uses the same object rather than a second one.

**The leakage certificate exists on both sides of the mapping.** The GA filter is
the structural certificate *before* encoding; the existing Pauli reference-aware
test becomes a **regression** certificate that the encoding preserved it. Neither
replaces the other, and disagreement between them is a bug in the transport, not a
finding.

**Deliberately out of scope here.** Pauli words are blades up to phase and Clifford
transformations have a clean GA description (P-GAGATE, §11), which could eventually
inform the dyadic measurement hierarchy of §6.6. That is kept out of §3.5 on
purpose: *should this operator be in the basis?* and *how should surviving
operators be measured together?* are different questions, and merging them would
make both harder to defend. Measurement grouping stays in §6 and Phase 14.

---

### 3.6 Anticommuting cliques are spin factors (contract; Phase 19)

§3.5 asks which *candidates* survive before encoding. This subsection is about a
different structure in the same algebra: what a set of **pairwise anticommuting**
Pauli words is, and what may and may not be concluded from it.

**Conceptual object.** By `A_iA_j + A_jA_i = 2δ_ij`, `m` pairwise anticommuting
Hermitian involutions `{A_1, …, A_m}` are an orthonormal frame in the vector
grade of a real `Cl(m,0)` sitting inside `Cl(2n,ℂ)`. Two sharp yes/no questions
are mutually unbiased exactly when their involutions anticommute, so a clique is a
set of mutually unbiased questions and its state space is the `m`-ball of the spin
factor `V_m` (§11, row 18). **Substrate:** ordinary Pauli words — nothing new is
stored. **Domain interpretation:** the operator layer only, for the reason in the
no-go below.

Three facts follow, and Phase 19 uses each for a different thing.

**(i) Closure — one setting per clique.** For real `|c| = 1`,
`(Σ_i c_i A_i)² = 1`: the combination is *itself* a Hermitian involution, and for
`m ≥ 2` a product of `m − 1` rotors with generators `i A_1 A_k` carries it to
`A_1`.

```text
V (Σ_i c_i A_i) V†  =  A_1,        V = Π_k rotor(i A_1 A_k, θ_k),   m ≥ 2
```

The `m ≥ 2` precondition is not decoration. At `m = 1` the product is empty and
the identity is simply false for `c_1 = −1`: `V(−A_1)V† = −A_1 ≠ A_1`. A singleton
clique is not rotated at all — it carries its sign in the measured coefficient, and
an implementation that assumes `+A_1` reports the negated mean. For `m ≥ 2` a
Givens sequence whose accumulated first component is taken as `+√(c_1² + c_k²)`
lands on `+A_1` for every sign pattern, including `c = (−1, 0, …)`, because the
`θ = π` rotation still fires. Which convention is in force is part of the
estimator contract, not an implementation detail.

This needs no new primitive. `clifford_qc.rotor` *is* the rotor, and `i A_1 A_k`
is Hermitian with square `1` by construction. Verified in-session at `m = 7`,
`n = 3` against the dense bridge to `1e-9`. The rotors are **Pauli rotations, not
Clifford**, and their cost may not be read off `m`. `SettingResources` carries
two-qubit *count* (`n_2q`) and *critical-path depth* (`d_2q`) as separate fields,
each rotation's contribution depends on its Pauli support and on the synthesis
used, and independent rotations may partly parallelise — so `m − 1` bounds the
number of rotations and nothing else. The per-setting resource is synthesised and
measured like any other compiled setting (§6.4), never inferred from the clique
size.

**(ii) The ball bound — an exact variance floor.** `Σ_i ⟨A_i⟩² ≤ 1` on every
state — this is `|x| ≤ 1` in the ball, the Brukner–Zeilinger invariant — so

```text
Σ_i Var(A_i)  =  m − Σ_i ⟨A_i⟩²  ≥  m − 1
```

exactly, for every state. Two things this is not. It is a statement about the
**intrinsic** `Var_ρ(A_i)`, not about the sampling variance of any estimator of
it; and it is therefore a floor on what a perfect estimator faces, not a floor on
what a finite-shot record reports. Unlike the intervals in
`subspace/uncertainty.py` it is not asymptotic in the shot count — it is an
algebraic identity plus one inequality — but that exactness transfers only to
**exact** reconstructions.

Its contrapositive is a consistency gate on that same boundary: a reconstruction
in exact arithmetic returning `Σ_i ⟨A_i⟩² > 1` is inconsistent whatever it was
estimating, and belongs with the structural checks. A finite-shot estimate may
exceed `1` from statistical noise alone and must be gated against a declared
tolerance derived from its own shot count and confidence family, never against
`1` itself — a hard gate there would reject valid noisy data.

**(iii) The ceiling — `2n+1`, already constructible in the package.** On `n`
qubits the largest anticommuting set has `2n+1` members, and the package builds
it: `gamma(n, i)` for `i < 2n` together with
`hermitian_majorana_monomial(n, range(2n))`. Verified as `2n+1` pairwise
anticommuting Hermitian involutions for `n = 1, 2, 3`, and verified *maximal* at
`n = 2` by exhaustion over all sixteen Pauli words; for general `n` the
maximality is the `2q+1` result of §11, row 18, not an in-repo measurement. The
same exhaustion shows the four Majoranas alone extend by exactly one word — `ZZ`,
which is `parity_operator(2)`. The four-word set is of course perfectly
expressible; what that extension rules out is a *closed* four-question theory,
because the fifth question is the product of the four and is therefore already
present in any algebra containing them. A four-question model must forbid
products, not merely omit the fifth — which is what makes §11 row 18's hyperbit a
foundations construction rather than something a Pauli algebra can host.

This is also why `hermitian_majorana_monomial` carries `i^{k(k−1)/2}`. That is the reversion sign
of §3.2, and a bare `k`-fold Majorana product is Hermitian iff `k ≡ 0, 1 (mod 4)`.

**The no-go, which bounds every use above.** The `m`-ball is the state space of
the questions, and the image of this package's *fermionic* states in it is the
single centre point. Odd-grade Majorana monomials anticommute with parity, so
every parity-commuting state — every physical state of a fermionic problem — has
`⟨γ_i⟩ = 0`. Verified: `max |⟨γ_i⟩| = 0` over 200 random parity-commuting
two-mode states, and `hubbard(1×3)` carries **zero** terms of odd Majorana grade,
so no such question appears in a fermionic Hamiltonian this package builds.

The scope of that sentence is exactly the fermionic models, and stating it
loosely would be wrong in two directions. `kitaev_honeycomb` also carries no
odd-grade term, but it is a **direct spin model with no Jordan–Wigner
transformation at all** (`models/lattice.py`), so its even blade grades are a
property of how it was written down and witness nothing about parity
superselection; it is not evidence here. In the other direction the spin builders
`tfim` and `random_ising` carry single-qubit `X_i` field terms, which *are*
odd — four of seven terms at `n = 4` in both — so "no odd-grade term" is false of
the package as a whole and true only of its parity-conserving fermionic sector.

Within that sector the reading is settled: the hyperbit and the `m`-ball are not
a state space of anything this package computes, and §14 blocks wording that
treats them as one. What survives is the operator content: (i), (ii), (iii).

## 4. The method

### 4.1 Core loop

Given reference state `ρ` (pure), Hamiltonian `H`, and current generator set
`{A_i}`:

1. Build `O_ij^S = A_i†A_j` and `O_ij^H = A_i†HA_j` as sparse MVs, cached in the
   bank (products computed once; Hermitian pairs share work).
2. Assemble `S_ij`, `H_ij` by `trace_pairing` (exact) or shared measurement
   reconstruction (finite-shot, Phase 4).
3. Solve the **normalized**, thresholded generalized eigenproblem:
   - drop zero-norm rows; scale `D_ii = √S_ii`, `S̄ = D⁻¹SD⁻¹`, `H̄ = D⁻¹HD⁻¹` —
     thresholding raw `S` would make the retained subspace depend on arbitrary
     generator scaling;
   - hermitize; diagonalize `S̄ = UΛU†`; drop modes by absolute threshold `τ_S`,
     relative threshold, and a maximum retained condition number;
   - form `H̃ = Λ^{-1/2}U†H̄UΛ^{-1/2}`, diagonalize;
   - fix deterministic eigenvector phases and a deterministic order inside
     degenerate overlap eigenspaces;
   - record effective rank before and after truncation.
4. Select the next generator by certified residual coupling or predicted Ritz
   lowering (§4.3); add it, or abstain and stop.

### 4.2 Basis hierarchy

- **Level 0**: current reference `|ψ⟩` (HF, ADAPT warm start, or stabilizer
  configuration).
- **Level 1** (tangent / Pauli-orbit): `P_j|ψ⟩` — tangent directions of the
  Pauli-rotor ansatz at `θ=0`.
- **Level 2** (commutator response): `G_j|ψ⟩` — rows already materialized by
  `CommutatorBank`.
- **Level 3** (Krylov response): `H^k|ψ⟩`, one candidate family among others, not
  the organizing principle.
- **Level 4** (compound / Clifford-group orbits) — **done**
  (`subspace/generators.py`, `models/lattice.py`): selected `P_iP_j|ψ⟩`,
  `P_iG_j|ψ⟩` via `compound_response` (deduplicated up to a scalar, since the
  solver normalizes and `P_jP_i` is `±P_iP_j`; `max_support` and `max_generators`
  make the width and pool size declared rather than discovered), and stabilizer
  configurations `V|0…0⟩` via `configuration_generator`. A configuration enters as
  the operator that *reaches* it: with `|ψ⟩ = R|0…0⟩` and `|φ⟩ = V|0…0⟩` the
  generator is `A = VR†`, so `A|ψ⟩ = |φ⟩` exactly and no second state is prepared.
  Between two determinants that operator is a single `X`-string, so a whole
  competing order costs `S_A = 1`. `models.lattice.competing_orders` names them
  for a Hubbard cluster (antiferromagnet on the *bond-coloured* sublattice — site
  index parity gets the 2×2 grid wrong — its spin-flipped partner, two charge
  density waves, and the stripe the numbering produces), emitting only
  configurations at the reference's own `(N, S_z)`.

  *Configurations need a different sector test, and this is the trap.*
  `sector_leakage` asks whether the **operator** commutes with `N`; an `X`-string
  does not, so a leakage filter rejects every competing-order configuration — the
  same failure mode as the Kitaev misconfiguration in Phase 7, in a new place.
  What matters is the sector of the configuration, so `state_sector` reports
  `⟨N⟩`, `⟨S_z⟩` **and their variances** for `A|ψ⟩` as expectations on the
  reference (nothing prepared). The competing orders come out sharp: variance `0`
  to machine precision.

**Chemistry basis modes.** Two distinct modes, with the second as the chemistry
default:

- *Word-level benchmark mode*: individual Pauli words `P_j|ψ⟩`, for direct
  comparison with qubit-ADAPT pools.
- *Symmetry-preserving mode*: `T_μ|ψ⟩` where `T_μ` is the **complete** JW image of
  a particle-number- and `S_z`-conserving fermionic excitation, kept as one
  multivector rather than split into words. A word split from a conserving
  generator need not itself conserve `N` or `S_z`
  (`fermionic_sector_diagnostics`); without this the subspace can gain energy by
  leaking into unphysical sectors. Candidate records report `N`/`S_z` leakage, the
  sector of each stabilizer configuration, and rejection of sector-incompatible
  candidates.

### 4.3 Selection criterion

For candidate `|χ_a⟩` against current Ritz pair `(E_m, |Ψ_m⟩)`:

- residual coupling `r_a = ⟨χ_a|(H−E_m)|Ψ_m⟩`;
- predicted lowering `ΔE_a` from the **generalized** 2×2 problem in
  `span{Ψ_m, χ_a}` — both blocks, `(E_m, h_a; h_a*, h_aa)` against
  `(1, s_a; s_a*, s_aa)`. The overlap block is mandatory: without it the score is
  biased exactly in the near-linearly-dependent direction the method must reject;
- acceptance score `ΔE_a / cost_a^γ`, with rejection when the candidate's
  `S`-orthogonal component falls below the conditioning floor.

A known limitation, exposed by level 4 (Phase 7): the generalized 2×2 score
cannot see a bare competing-order configuration. Such a determinant has zero
overlap with the reference *and* zero Hamiltonian matrix element to it, so the
predicted lowering is exactly zero and greedy growth never takes one, however
useful it would be in combination. A selector weighing second-order coupling
would change this; the present one cannot, and that is a property of the
criterion rather than of the family.

### 4.4 Residual norms need a second-moment bank

The true Ritz residual `‖(H−E)|Ψ⟩‖² = (c†Kc)/(c†Sc) − E²` requires
`K_ij = ⟨ψ|A_i†H²A_j|ψ⟩`, which the projected `(H,S)` pair cannot supply and whose
Pauli support can be much larger. Policy: `residual_norms` is **optional** —
computed either from a later `SecondMomentBank` (Phase 15) or, in small exact
runs, from a dense reconstructed state. Never expose the *projected* residual
under that name: it is zero by construction for a solved Ritz pair and says
nothing about error outside the subspace.

---

## 5. Phase ledger

Phases are ordered by dependency; each has a go/no-go invariant. The bank comes
**before** adaptive growth: adaptive selection repeatedly adds rows and columns,
and without cached pair products it would recompute `A_i†A_j` and `A_i†HA_j` many
times — making a sound method look uncompetitive because of a deliberately
temporary implementation.

### Phase 0 — done (merged main)

Exterior layer on `MV` (reversion, wedge, k-vector dot, blade tests), O(1) exact
gradients, deterministic selection, `fermionic_sector_diagnostics`, gate
preconditions, generated paper tables.

### Phase 1 — done (`clifford_qc/subspace/`)

`MV.trace_pairing`; `solver.py` with the normalized, thresholded, deterministic
GEP of §4.1 and `SubspaceResult(energies, coefficients, basis_labels,
overlap_eigenvalues, condition_number, effective_rank, resources)`;
`generators.py` for the level-0..3 families; `reference.py` for the dense-matrix
cross-check; `models.chemistry.excitation_multivectors` for the
symmetry-preserving chemistry mode. Two assembly routes, verified equal: the
element-operator route forms `A_i†A_j` and `A_i†HA_j` (the operators Phase 2
caches and Phase 4 must measure, and the only route that can report `W` and
`S_H`); the cyclic route contracts `Tr(A_i†HA_j ρ) = Tr((HA_j)(ρA_i†))` in `2M`
products and `M²` sparse pairings and is correspondingly blind to those metrics.

*Validated* (`tests/test_subspace.py`, `tests/test_subspace_chemistry.py`, and
the `A-CASE` section of `clifford_qc.verify`): `E_sub ≥ E₀` and nested
monotonicity on TFIM, XXZ, and both H₄ legs; exact agreement with the dense
route; `S = S†` bitwise (structural, from the upper-triangle layout, not a
numerical symmetrization); FCI reproduction on H₂ and LiH(2e,2o) to 1e-9 with
four generators, and by a ground-state-projector generator on TFIM; invariance of
the retained subspace, its spectrum, and `κ_S` under random complex generator
rescaling; deterministic eigenbases inside degenerate overlap and Ritz
eigenspaces (canonicalized from the spectral projector, so independent of the
LAPACK basis); Ritz states staying in the reference `(N, S_z)` sector.

*Measured* (H₄ chain, sto-3g, 8 qubits; `E₀ = -2.180317` at r=0.9 and `-1.924431`
at r=1.8; `S_A = max_i |supp(A_i)|`):

| fixed basis | M | rank | ΔE (r=0.9) | ΔE (r=1.8) | κ_S | S_A |
|---|---|---|---|---|---|---|
| A-CASE symmetry-preserving level 1 | 27 | 27 | 7.7×10⁻⁴ | 3.5×10⁻² | 1.0 | 8 |
| word-level QSE, matched budget | 27 | 11 | 4.8×10⁻² | 1.6×10⁻¹ | 8.0 | 1 |
| word-level QSE, full odd-Y pool | 161 | 27 | 7.7×10⁻⁴ | 3.5×10⁻² | 8.0 | 1 |
| fixed Krylov `H^k`, k ≤ 6 | 7 | 7 | 1.5×10⁻⁶ | 2.5×10⁻⁴ | 1.0×10⁸ | 4224 |
| fixed Krylov `H^k`, k ≤ 10 | 11 | 9 | 5.3×10⁻⁹ | 6.4×10⁻⁵ | 3.4×10¹⁰ | 4224 |

*Go/no-go: conditionally met, and not in the way the criterion assumed.* The
symmetry-preserving level-1 basis reaches chemical accuracy at equilibrium with
`M = 27` against a 36-state `(N=4, S_z=0)` sector, spans the same subspace as the
161-word QSE pool at a sixth the basis size, and does so at `κ_S = 1` — but it
does **not** beat fixed Krylov on energy per basis vector, at either geometry.
Krylov wins that column by orders of magnitude. What it pays is exactly the §6
currency: generators 500× wider (`S_A = 4224` vs 8, so wide that the
element-operator route is not affordable on H₄ at all, while the A-CASE basis
assembles in seconds) and `κ_S` of 10⁸–10¹⁰, the conditioning regime where noisy
PSD repair and finite-shot certification (Q2, Q3) are least likely to survive. So
the compactness claim Q1 is **not** established by fixed bases: it rests on
adaptive selection, which is Phase 3.

*TFIM premise check.* `examples/acase_premise_check.py` runs on the shipped
solver: the variational bound and nested monotonicity hold at every level, the
level-0..3 hierarchy closes the gap from 1.76 to 1.8×10⁻², and 37 generators
yield only 12 independent directions — the near-singular-`S` regime that makes
conditioning-aware adaptive selection load-bearing rather than an optimization.

### Phase 2 — done (`subspace/elements.py`)

`MatrixElementBank`: canonical generator IDs (a repeated operator returns the id
it already has; a label rebound to a different operator is an error, since labels
are what records report); cached `A_i†A_j` and `A_i†HA_j` with Hermitian-pair
reuse (upper triangle only — `S_ji = conj(S_ij)` is a property of the layout);
global word-union tracking, including new-versus-reused words per accepted
generator; exact `trace_pairing` assembly reproducing Phase 1 **bit for bit** —
the product order is deliberately identical, since floating-point addition is not
associative and a "mathematically equivalent" rearrangement would make the two
routes' records irreproducible; support/conditioning metrics per build, plus
cached-operator bytes and an opt-in QWC group count (quadratic in `W`, so never a
hidden cost). No finite-shot machinery: the cached coefficient maps *are* the
sufficient statistics Phase 4 reconstructs from measured word means, so that
layer attaches without disturbing this one. Pair products are lazy, so a
candidate that is never scored costs nothing.

**Projected observables (§8) ship with it.** `project_observable(Q) → Q_sub`
through the same element machinery, then `result.expectation(Q, k)` and
`result.transition(Q, i, j)` contract it with the Ritz coefficients.

*Measured (the reason the bank comes before adaptive growth).* Solving every
nested prefix of a basis — the access pattern Phase 3 generates — costs
`M(M+1)(M+2)/6` pair products when each solve reassembles, and `M(M+1)/2` through
the bank:

| trajectory | pair products | wall clock |
|---|---|---|
| TFIM n=4, M=37, reassembling | 9139 | 0.28 s |
| TFIM n=4, M=37, banked | 703 | 0.15 s |
| H₄ r=0.9, M=27, reassembling | 3654 | 13.9 s |
| H₄ r=0.9, M=27, banked | 378 | 2.3 s |

The gap widens with `M` (ratio `(M+2)/3`) and with generator width, which is why
the TFIM speedup is modest — its solves are dominated by the eigendecomposition,
not the products — while H₄'s is 6×. The cost side is memory, and it is not
small: H₄'s 378 cached element operators hold 15 847 distinct words and ~10.6 MB.
That figure is the §6 metric to watch as Phase 3 grows bases, not a footnote.

### Phase 2M — memory-bounded matrix-element bank (adjunct; 2M-A done, B--D open)

Phase 2 made repeated adaptive solves computationally credible by retaining every
built pair. The larger molecular records expose the other side of that decision:
the resident Python-object bank, rather than the numerical `H` and `S` pencils or
the fixed A-CASE reference, is now the dominant classical-memory allocation. Phase
2M changes only storage and lifetime policy; it may not change the generator family,
selection rule, measurement estimand, or scientific evidence category.

**Evidence and claim boundary.** The committed molecular records provide one
measured anchor and two independent OOM failure witnesses.

- Stretched BeH2 at 14 qubits, candidate pool 204 and final `M = 31` built 5,890
  cached pairs and reached `11,543,863,296` bytes = 10.75 GiB peak RSS
  (`adaptive_peak_rss_bytes`) with a 9.98 GiB adaptive delta
  (`adaptive_peak_rss_delta_bytes`). Its
  `cached_operator_bytes = 2,306,582,856` corresponds exactly to
  `T_coeff = 96,107,619` stored nonzero coefficient occurrences under the current
  24-byte payload estimator.
- The attempted stretched-H2O `M = 31` endpoint was killed by the OOM reaper on a
  15 GiB machine. This is a calibrated extrapolation, not a measured endpoint:
  H2O carries 1,086 Hamiltonian words against BeH2's 666, and
  `(1086/666) * 10.75 GiB = 17.5 GiB`, consistent with the approximately 18 GiB
  requirement recorded in `run_molecular_pipeline.py`.
- A separate sequential run retained BeH2's 11.2 GiB bank as H2O's starting point
  and was also killed on the 15 GiB machine. The producer now writes each molecule
  before continuing and explicitly deletes the preceding bank and runs garbage
  collection. That operational fix is evidence that bank lifetime, not merely
  Hamiltonian input size, controls the process peak.

The A-CASE reference in these runs is not the exponential culprit:
`rho0 = ExactMVBackend().state(model.reference, ())` is a fixed Clifford-stabilizer
density operator with exactly `2^n` Pauli terms and is never evolved during A-CASE
growth. A possible `4^n` density-multivector support belongs only to exact classical
arms with non-Clifford rotor evolution, such as simulated VQE or ADAPT-VQE.

**Aligned storage currencies.** For the cached element-operator family
`O_alpha in {A_i^dagger A_j, A_i^dagger H A_j}`, define

`W_res = | union_(resident alpha) supp(O_alpha) |`,

`T_coeff = sum_(resident alpha) nnz(O_alpha)`, and

`R_reuse = T_coeff / W_res`.

`W_sel` remains the distinct Pauli expectations needed by the selected subspace.
`T_coeff` prices every resident nonzero, including rejected/frontier rows kept by
`retain_all`. `R_reuse` is therefore cross-element word reuse over the *same*
resident population, not removable duplication: the same word normally has a
different coefficient for every pair. The mixed ratio `T_coeff/W_sel` is reported
separately as resident coefficients per selected word; it is not called reuse.
The seven old molecular records preserve `W_sel` but not `W_res`, so their true
reuse multiplicity cannot be recovered without rerunning them. Packed storage
removes representation overhead around intrinsic nonzeros; it does not claim a
reuse-factor deduplication gain. Under the current code
`cached_operator_bytes = 24*T_coeff` exactly, but that estimate excludes Python
dictionary, integer, complex-object and
allocator overhead. The 116-to-24 comparison entered this plan as a packing
hypothesis rather than a promised RSS ratio, and 2M-A has since measured it: one
resident coefficient costs `85.24`--`96.19` bytes, `89.92` pooled, so the headroom
is `3.75x` and not the `4.8x` that estimate implied. The measured figure is the one
2M-B is gated against.

**2M-A — storage ledger and frozen baseline. Done.** The declared extension ships
(`clifford_qc/subspace/projection.py`): `resources()` carries `T_coeff` as
`coefficient_occurrences`, `resident_word_universe`, their aligned reuse
multiplicity, the separate coefficients-per-selected-word ratio, retained and
selection pair counts with their fraction, resident and peak rows, the policy label, and
the evicted/recomputed/spill counters — structurally zero under `retain_all`,
which the checker requires rather than assumes. Measured bytes are deliberately
*not* in `resources()`: the walk is linear in `T_coeff` with an identity set
beside it, and `resources()` runs once per adaptive step, so
`measured_storage_bytes()` is opt-in on the same grounds `qwc_group_count()`
already is. The triple is `run_bank_storage_ledger.py`,
`reference_results/bank_storage_ledger.json` and `check_bank_storage_ledger.py`,
on the five frozen mapping-axis banks, two small exact-A-CASE arms, and the seven
committed molecular records read rather than rerun.

*Three results, one of which corrects this plan.* First, the packing hypothesis
measures **3.75x pooled** (`85.24`--`96.19` bytes per resident coefficient against
the packed `24`, so `3.552`--`4.008x` per bank), not the `116`-to-`24` figure
below: it clears 2M-B's `3x` go/no-go with less margin than that estimate implied,
and the working number is now a measurement. Second, the rate is deduplicated by
object identity — the memoized word product hands every row carrying a word the
same integer object, and a per-row sum overstates the recoverable bytes by `4.7%`,
in the direction that flatters the phase. Third, applying the minimum *observed*
small-bank rate to each committed row's recovered `T_coeff` attributes
`63.4%`--`72.4%` of recorded peak RSS to resident coefficient payload on all four
large banks. This is a measurement-calibrated extrapolation, not a direct
measurement and not a proven lower bound for a larger CPython dictionary. Read
beside the two independent OOM witnesses above, it supports the diagnosis that the
bank, not the pencils or fixed A-CASE reference, dominated the failed runs. The
three small rows are reported separately — `hf` carries a `2.94` GiB peak against
`0.06` GiB of attributed payload — because a per-run baseline sets their peak.

*Peak, not delta, is the denominator used by that attribution.*
`adaptive_peak_rss_delta_bytes` subtracts the resident size at the start of the
adaptive block, so a process that had already allocated and freed memory hands the
bank pages the allocator still holds and the delta understates it. That is the
sequential-run mechanism recorded above, not a hypothetical, so the delta fraction
is a lower bound whose denominator depends on run history.

*And one allocation neither remaining lever reaches.* `_word_mul_unchecked` is
memoized at `maxsize=1_000_000`. Counting only the per-entry key and value tuple
containers gives a `120,000,000`-byte = `0.112` GiB floor; boxed referents are
excluded because they may be shared, as are the cache hash table and LRU nodes.
That allocation is a ceiling of at least `0.112` GiB that 2M-B's
packing and 2M-C's eviction both leave in place. On the small banks it exceeds the
resident coefficient payload; on the committed molecular banks it is a constant beside rows one
to three decades bigger, so it is not a competing explanation for the two OOM
failures — but a packed bank meeting its `3x` target still carries it, and the
end-to-end 15 GiB test is where that shows up.

Recovering `T_coeff` for the existing records from `cached_operator_bytes/24` is
exact — the checker fails a nonzero remainder rather than rounding — and none of
their scientific results is rerun or relabelled. Retained-pair fraction is reported
separately and re-derived per row: the committed molecular rows retain
`M(M+1)/2 = 3.0--8.8%` of the pairs they built, so 91.2--97.0% are frontier or
rejected-pair storage, and the two adaptive arms reproduce the effect at `16.7%`
and `10.2%` on a live frontier. That ratio is eviction headroom, not an achieved
speedup. The record carries no go/no-go outcome at all — its verdict is
`baseline_only_no_go_no_go_evaluated` and the checker fails any other value —
because 2M's gate asks for a measured reduction from an implementation that does
not exist yet.

**2M-B — packed CSR/SoA coefficient bank.** Introduce one canonical global word
table and packed row storage for the overlap and Hamiltonian functionals:
`indptr`, word indices/codes, and contiguous coefficient data; benchmark interleaved
complex data against structure-of-arrays real/imaginary storage. Preserve row
identity, upper-triangle Hermitian ownership, canonical word order, and the product
and accumulation order required by the current reproducibility contract. The
`MatrixElementBank` public semantics remain unchanged.

**2M-C — explicit frontier lifetime policies.** Implement and compare three named
policies under one interface:

| policy | retained data | intended use |
|---|---|---|
| `retain_all` | every materialized operator row | current baseline; minimum recomputation |
| `stream_recompute` | scalar `S,H` entries, retained-basis rows, and one bounded candidate batch | exact selection when memory is binding |
| `disk_backed_csr` | packed functionals in an mmap/spill store plus the active batch | finite-shot compilation or repeated reconstruction beyond RAM |

For exact selection, an operator may be discarded after its scalar entry, support
ledger and canonical packed row have served the chosen policy. For finite-shot
selection it may be discarded only after its word functional is durably compiled;
measurement reconstruction still needs every intrinsic coefficient. Eviction must
therefore expose recomputation and I/O as costs rather than present memory reduction
as free.

**2M-D — equivalence and performance matrix.** On H4, equilibrium and stretched
BeH2, and equilibrium and stretched H2O, compare all policies at identical candidate
ordering and basis budget. Require exact equality of `W_sel`, `W_res`, `T_coeff`, pair ownership,
selected labels, rejection decisions, stopping reason, evidence label and resource
scope. Require bitwise `S`, `H` and energies where canonical accumulation order is
preserved; otherwise use a declared tight tolerance and record the first source of
rounding-order divergence. Report bank-build time, recomputation time, spill I/O,
peak and delta RSS, packed bytes and actual bytes per coefficient.

**Go/no-go.** The packed representation must reduce resident coefficient-storage
bytes by at least 3x at unchanged `T_coeff`. A streaming policy must bound live
operator rows by the retained block plus its declared candidate batch, rather than
by all historically scored candidates. The primary end-to-end feasibility test is
the previously failing stretched-H2O `M = 31` configuration completing below 15 GiB
without shrinking its candidate pool, word universe, or basis budget. Packing and
eviction effects are reported separately and together; no multiplicative
`4.8x * 12x` or approximately 58x claim is allowed until the combined implementation
is measured.

Phase 15 may run its `H^2` support/cost preflight in parallel, but full
`SecondMomentBank` construction is gated on Phase 2M passing or on an explicit
small-system exception: otherwise it would knowingly multiply the allocation that
already caused the two OOM failures.

### Phase 3 — done (`subspace/adaptive.py`)

`run_acase` grows the basis one generator at a time on top of the bank:
generalized-2×2 predicted lowering (closed form, overlap block carried
explicitly), scale-free residual coupling, linear-dependence rejection,
`energy_history`, per-step `GrowthRecord`s carrying conditioning and word costs,
an optional cost-aware score `ΔE/(1+new words)^γ`,
`fermionic_excitation_generators` as the chemistry default with `sector_leakage`
reported per accepted generator (and sector-breaking candidates rejected when
`leakage_tol` is set), and `adapt_warm_start` for growing around an ADAPT-VQE
state.

Three details are load-bearing, and each is pinned by a test that fails under the
obvious alternative:

- *The overlap block is not optional.* Assume the candidate is orthonormal to the
  current Ritz vector, and a candidate that **is** that vector times 3.5 scores
  over a Hartree of predicted gain; the generalized 2×2 scores exactly zero.
- *Rejection is measured against the retained subspace, not the Ritz vector.* A
  candidate duplicating some other basis direction sits at a perfectly healthy
  angle to the Ritz vector, passes the weaker test, and makes `S` singular — the
  thresholded solve then discards it after it has been paid for.
- *The 2×2 deflation needs a floor.* Below an orthogonal fraction of ~1e-12 the
  deflated diagonal is a genuine 0/0, and double precision returns noise that is
  not small: a parallel candidate lands at −6 instead of −4, two Hartree of
  fabricated lowering. The conditioning floor sits four orders above it, so live
  scoring never reaches the cliff.

*Validated:* `E_sub ≥ E₀` and monotone `energy_history`; **predicted lowering ≤
actual lowering** at every step (`span{Ψ_m, χ}` sits inside `span{basis ∪ χ}`, so
the 2×2 can only underestimate); adaptive ≤ fixed basis at equal size on TFIM,
XXZ, and both H₄ legs; deterministic and scale-invariant selection; convergence
cross-checked against the dense `dense_residual_norm` that §4.4 keeps out of the
projected API.

*Measured, TFIM n=4 from the model's own reference `|++++⟩`
(`examples/acase_adaptive.py`):*

| M | A-CASE | κ_S | fixed Krylov | κ_S | fixed QSE | ADAPT-VQE |
|---|---|---|---|---|---|---|
| 3 | 4.1×10⁻⁸ | 3.6×10² | 2.3×10⁻² | 1.8×10² | 7.6×10⁻¹ | 2.7×10⁻¹ |
| 5 | 4.0×10⁻¹⁰ | 8.8×10² | 1.6×10⁻⁶ | 1.2×10⁴ | 5.2×10⁻¹ | 1.3×10⁻² |
| 7 | 8.9×10⁻¹⁶ | 9.2×10² | −1.8×10⁻¹⁵ | 1.8×10⁷ | 5.2×10⁻¹ | 7.1×10⁻¹⁵ |

The criterion is met on the spin model: A-CASE matches or beats every baseline at
matched budget, and where fixed Krylov finally catches up it does so at
`κ_S = 1.8×10⁷` against A-CASE's `9.2×10²` — five orders of conditioning, the
currency Q2 and Q3 are denominated in. Warm-starting from a 2-operator ADAPT state
improves M=5 further, 4.0×10⁻¹⁰ → 2.0×10⁻¹¹.

*Measured, H₄ chain (symmetry-preserving candidates, `leakage_tol=1e-9`; every
accepted generator leaks < 1e-12 and `κ_S = 1` throughout):*

| M | A-CASE (r=0.9) | fixed prefix | ADAPT-VQE | A-CASE (r=1.8) | ADAPT-VQE |
|---|---|---|---|---|---|
| 4 | 1.87×10⁻² | 5.61×10⁻² | 1.87×10⁻² | 8.54×10⁻² | 3.86×10⁻² |
| 6 | 9.36×10⁻³ | 5.61×10⁻² | 9.72×10⁻³ | 4.09×10⁻² | 2.33×10⁻² |
| 9 | 3.02×10⁻³ | 5.61×10⁻² | 2.38×10⁻³ | — | — |

Against the fixed prefix the gain is decisive at every size — the natural ordering
emits singles first, and on a closed-shell determinant those contribute almost
nothing, so the fixed basis stalls at 5.6×10⁻² while adaptive selection takes the
doubles that matter. Against ADAPT-VQE the honest reading is a draw at equilibrium
(ahead at M=6, behind at M=9) and a **loss on the stretched geometry**, where
ADAPT reaches 3.9×10⁻² against A-CASE's 8.5×10⁻² at M=4. A linear span of singles
and doubles on an HF reference is the wrong object for a strongly multireference
state; the answer is compound generators and competing-order references (§4.2
level 4), not more of the same family.

### Phase 4 — done (`subspace/measured.py`)

Certification here is a **new nonlinear statistical problem**: `(H,S)` are
estimated, the retained eigenspace is data-dependent, the Ritz pair `(c,E)` is
data-dependent, and residual couplings and 2×2 lowerings are nonlinear functions
of correlated estimates. The gradient best-arm code assumed linear estimators; it
did not transfer unchanged. What did transfer is the shape: everything the
subspace needs is a linear functional of Pauli-word means, and `WordFunctional`
is that object — the single place shots enter, with `estimate`, `variance`, a
covariance-vector product, an exact (infinite-shot) evaluation, and arithmetic,
so a difference of functionals is a functional.

- *4A — shared grouped measurement.* `SharedMeasurement` measures a subspace's
  whole word universe through QWC groups and reconstructs every entry from the
  same shots; the lower triangle stays the conjugate by construction rather than
  an independent noisy estimate. Its `exact_matrices()` walks the same
  reconstruction with exact means and reproduces the Phase-2 matrices to
  5×10⁻¹³, which is the acceptance criterion for the infinite-shot limit. The
  identity word is never measured: `⟨I⟩ = 1` is known, and reporting it as
  measured would inflate the empirical-Bernstein range of every group reading it.
- *4B — asymptotic uncertainty.* `ritz_uncertainty` linearizes:
  `dE = Σ_w q_w dμ_w` with `q` the word coefficients of `B†(H−E)B`,
  `B = Σ_i c_i A_i` — one *real* functional (that operator is Hermitian), so the
  Jacobian of the Ritz value with respect to every word mean is a single
  bank-derived object. `bootstrap_ritz` resamples the grouped histograms and
  reruns the whole nonlinear pipeline as the cross-check. Both are labelled
  `asymptotic` / `heuristic`; `Interval.certified` is False for both.
- *4B-R — nonlinear response uncertainty (`subspace/measured_response.py`).*
  `ResponseMeasurement` extends the shared universe to a Hermitian projected
  observable `Q_sub`; `bootstrap_response` resamples the grouped joint histograms
  and reruns `(S,H,Q_sub)` reconstruction, overlap thresholding, the generalized
  eigensolve, transition amplitudes, squared weights, gaps, susceptibility, and
  optional broadening. Root-resolved lines are emitted only for isolated ordered
  roots; rank changes and root collisions are counted. The result is always
  `heuristic` and never certified. A finite-sample response certificate remains a
  separate matrix-pencil confidence-set problem.
- *4C — finite-sample growth certificate by sample splitting.*
  `run_certified_acase` spends a construction batch on `(S,H)`, freezing the
  thresholded subspace, its Ritz pair, **and the candidate norms**; an independent
  certification batch bounds each candidate's residual coupling with those held
  constant. `|r| = √(Re² + Im²)` is not linear, so the interval is a rectangle
  over the two real functionals with the union bound paid explicitly over
  `2·(#candidates)` events. Growth happens only when a candidate's lower bound
  clears the threshold; otherwise the run **abstains** and stops.

The norm subtlety is worth stating because it was easy to get wrong: the coupling
must be normalized by `‖A_a|ψ⟩‖` or the ranking would depend on how a candidate
happens to be scaled, but dividing by a quantity estimated from the *same* batch
would make the statistic a ratio of correlated estimates and void the
certificate. The construction batch is what makes the norm a constant.

**Covariance discipline** (honored): no dense covariance over `(H,S)` entries is
ever formed. The grouped joint histograms stay the sufficient statistic and
`WordFunctional.covariance` computes one covariance-vector product on demand;
comparisons need no covariance object at all, since a difference of linear
functionals is a linear functional. One bug found this way and worth recording:
several QWC groups can be *able* to read the same word, and attributing it to
each capable group inflated every variance by that multiplicity — caught only by
comparing the predicted `σ` against a Monte-Carlo spread, which is now the test.

*Measured (TFIM n=4, 40 measurement seeds, `examples/acase_finite_shot.py`):*

| basis | shots/group | MC std | mean σ̂ | median σ̂ | bias | 95% coverage |
|---|---|---|---|---|---|---|
| κ_S = 1 | 2000 | 3.2×10⁻² | 3.7×10⁻² | 3.6×10⁻² | −1.5×10⁻² | 0.95 |
| κ_S = 1 | 20000 | 1.15×10⁻² | 1.10×10⁻² | 1.10×10⁻² | −4.9×10⁻³ | 0.93 |
| κ_S ≈ 2×10² | 2000 | 1.85 | 2.34 | 6.6×10⁻³ | −3.0×10⁻¹ | 0.97 |
| κ_S ≈ 2×10² | 20000 | 1.65×10⁻³ | 1.95×10⁻³ | 1.81×10⁻³ | −1.7×10⁻⁵ | 0.97 |

With a well-conditioned overlap the delta method is accurate (within ~15% of the
Monte-Carlo spread) and the grouped bootstrap agrees to three digits (2.49×10⁻²
vs 2.51×10⁻² at 4000 shots). Two findings cut the other way and are the reason
for the labels. First, the measured Ritz value carries a **systematic downward
bias** that shrinks with shots (−1.5×10⁻² at 2000 shots/group, −4.9×10⁻³ at 20000
for `κ_S = 1`): the noisy energy is not an upper bound on `E₀`, and at low budgets
the bias is a large fraction of the standard deviation. Second, at `κ_S ≈ 2×10²`
and 2000 shots/group the error distribution is **heavy-tailed** — a handful of
runs in forty admit a near-null overlap mode and land whole Hartrees away, giving
an MC std of 1.85 Ha against a median `σ̂` of 6.6×10⁻³. A mean-and-variance
description of the error is inadequate there. That is the strongest argument in
the code base both for conditioning-aware growth and for never calling these
intervals certified. The variational-bound violation is a test, not a caveat: at
200 shots/group most seeds put `E_sub` below `E₀`, by up to 2×10⁻². Quantifying
it in terms of `τ_S`, shot covariance, and conditioning remains **Q3**.

*Measured (4C certified growth, δ = 0.05, EB bounds):*

| shots/group | threshold | certified steps | final gap | shots | circuits |
|---|---|---|---|---|---|
| 4000 | 0.05 | 3, then abstain | 1.2×10⁻¹ | 2.6×10⁶ | 648 |
| 40000 | 0.05 | 4, then abstain | 1.4×10⁻² | 3.2×10⁷ | 810 |
| 40000 | 0.40 | 3, then abstain | 1.1×10⁻¹ | 2.6×10⁷ | 648 |

Every accepted step is `finite_sample`, and every run ends in abstention rather
than uncertified growth — the behavior the plan asked for, at the cost the plan
predicted. The cost is the headline: two independent full-universe batches per
step put certified growth four orders of magnitude above the exact-arithmetic
path in shots, and the certified trajectories stop at gaps (10⁻²) that Phase 3
reaches at 10⁻¹⁰ exactly. Note also what is *not* certified: the statement is
about the accepted candidate's coupling with the frozen Ritz pair, conditional on
the construction batch. It is not a claim that the candidate is the best available
(`resolution` records separately whether the leader also cleared every rival's
upper bound — at these budgets it usually does not), and it is emphatically not a
bound on the energy. Confidence-set reuse across steps, which would recover much
of the shot cost, is the obvious next stage and is not attempted here.

Two further gaps are left open on purpose. The certified path ranks candidates by
**residual coupling**, not by the generalized 2×2 lowering Phase 3 uses: the
lowering is a nonlinear function of `s_aa`, `h_aa`, and a square root, so it
admits only a delta-method treatment, and the certified gate has to be the linear
statistic. And shot allocation is a predeclared uniform budget per group — fixed
endpoints are what the empirical-Bernstein validity argument needs.

### Phase 4R — the acquisition stage was the wrong one to optimize

Producer `benchmarks/run_finite_shot_rethink.py`; record
`reference_results/finite_shot_rethink.json`; regenerated and compared by
`benchmarks/check_finite_shot_rethink.py`. It runs on the *same* bank, seed, and
budget as the published study, and its `assigned`/`fixed` and
`assigned`/`calibrated` arms reproduce that study's rows to the digit, so the new
arms are directly comparable to what is in the manuscript.

Phase 4 left the finite-shot projected eigensolver with one good property and two
bad ones. Good: everything the subspace needs is a linear functional of Pauli-word
means, measured once through shared QWC groups. Bad: (1) **the tail, not the
median, is the error** — at 104 000 setting-shots the median absolute error is
`4.48 mHa` and the RMSE is `1786 mHa`, because four replicas in two hundred land
whole Hartrees away when a near-null overlap mode is resolved the wrong way; and
(2) **every fix cost accuracy** — thresholding overlap modes at their own shot
noise removed the tail (`RMSE 13.31 mHa`) but doubled the median to `10.64 mHa`,
carried a `+10 mHa` truncation bias, and recovered the exact rank in only `71/200`
replicas.

Covariance-aware allocation cut the summed projected-matrix variance by 68.9% and
moved the median error from `4.48` to `4.44 mHa`. That is the finding that
reframes the problem: a near-null overlap mode is dangerous through its variance
*relative to its eigenvalue*, and no reallocation of a fixed budget changes that
ratio by the orders of magnitude needed. There are three stages between shots and
an energy, and only the first had ever been varied:

```text
   acquisition           reconstruction              rank rule
shots -> settings  ->  histograms -> (S_hat, H_hat) -> retained modes -> E_hat
```

The two that had not are the ones worth attacking, and both change without
spending a shot.

- *Reconstruction.* QWC grouping partitions the word universe to answer "how few
  circuits cover everything" — a scheduling question. It had also been answering
  "which shots estimate this word" — an estimation question, and wrongly: a
  setting's histogram records **every** word supported inside its basis with
  matching letters, not only the one the partition assigned there. On the frozen
  bank a word is recorded by 3.72 settings on average and by up to 23.
  `pooling='shots'` reads all of them, weighted by shot count — exactly the
  inverse-variance weighting, since a word's per-shot variance `1 - mu_w^2` does
  not depend on which compatible basis read it. Unbiased, exact in the reported
  covariance (the coefficient is *split* across reading groups, not duplicated
  into each — that duplication is the multiplicity bug above), and
  outcome-independent, so fixed-endpoint bounds survive.
- *Rank rule.* The calibrated cutoff asks whether an overlap mode stands above its
  own noise, which is a question about `S`. `solve_selected_rank` asks the one
  that matters — solve at every attainable rank and take the minimizer of
  `E_hat(k) + gamma·sigma_hat(k)`, with `sigma_hat` the delta-method error of
  `ritz_functional` at that solution. Ties resolve to the smaller rank.

Three properties make pooling the right combination rather than merely a bigger
one. *Unbiased:* each reading group's estimate is unbiased for the same mean and
the weights sum to one over the reading groups — a reweighting, not a second copy.
*Optimal:* the marginal law of a word's ±1 outcome does not depend on which
compatible basis read it, so every reading group has the same per-shot variance
and shot-count weights are exactly the inverse-variance weights.
*Certificate-safe:* the weights depend only on the predeclared shot schedule,
never on outcomes, so a fixed-endpoint empirical-Bernstein argument survives.

- *Third experiment — smooth spectral damping instead of truncation (negative).*
  Truncation is a binary decision taken on noisy data, which invites the obvious
  alternative: keep every mode and damp it. `solve_projected(..., overlap_ridge=)`
  replaces `S_bar` with `S_bar + Σ_k δ_k u_k u_k'`, so a mode contributes
  `1/(λ_k + δ_k)` to the inverse metric and the whitening identity still holds
  exactly in the ridged metric. **Worse than truncation at every scale tried, by
  orders of magnitude:** at the calibrated radii it gives `50 mHa` median and
  `6166 mHa` RMSE where truncation gives `2.56` and `4.86`, and sweeping the ridge
  up by factors of 3 to 300 trades the tail for a uniform positive bias reaching
  `3.4 Ha` without ever passing truncation. The reason says something about the
  problem rather than the knob: the ridge bounds the *metric*, and the damage is
  in the *numerator* — along a near-null mode the measured `H_bar` is noise, and
  the Rayleigh quotient descends into it for any damping that leaves the direction
  in the space. A factor of `λ/(λ + r)` — one half, at the noise radius — is
  nowhere near enough, and a ridge large enough to suppress it also distorts the
  well-resolved modes, since it acts on all of them. Removing a direction is
  qualitatively, not quantitatively, different from shrinking it. The knob is
  retained, off by default, so the negative stays reproducible.

*Measured.* One bank (four-qubit TFIM, `M=9`, exact rank 8, `κ_S = 194.94`, 189
words, 52 QWC groups), uniform allocation, 200 replicas. Every arm at a budget
reads the same caches, so the table compares estimators, not budgets. Errors in
mHa against the exact projected energy.

**104 000 setting-shots per replica** (the published budget):

| reconstruction | rank rule | median | p95 | max | RMSE | bias | >0.1 Ha | rank 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| assigned | fixed | 4.48 | 37.01 | 22760.20 | 1786.21 | −178.08 | 4/200 | 200/200 |
| assigned | calibrated | 10.64 | 23.96 | 34.84 | 13.31 | +10.33 | 0/200 | 71/200 |
| assigned | selected | 4.56 | 16.27 | 21.80 | 7.82 | +1.73 | 0/200 | 162/200 |
| assigned | ridge | 97.42 | 2754.89 | 21589.19 | 2095.93 | −406.15 | 73/200 | 1/200 |
| pooled | fixed | 2.56 | 10.15 | 152.64 | 11.70 | −2.46 | 1/200 | 200/200 |
| **pooled** | **calibrated** | **2.56** | **10.12** | **21.29** | **4.86** | **−1.13** | **0/200** | **195/200** |
| pooled | selected | 2.56 | 10.19 | 21.29 | 4.90 | −1.45 | 0/200 | 198/200 |
| pooled | ridge | 50.02 | 11637.25 | 46061.65 | 6166.13 | −2125.76 | 65/200 | 0/200 |

**13 000 setting-shots per replica** (eight times smaller):

| reconstruction | rank rule | median | p95 | max | RMSE | bias | >0.1 Ha | rank 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| assigned | fixed | 21.61 | 1812.02 | 23859.85 | 2254.42 | −446.49 | 34/200 | 160/200 |
| assigned | calibrated | 32.63 | 93.65 | 119.75 | 53.88 | +37.86 | 7/200 | 5/200 |
| assigned | selected | 11.19 | 42.73 | 84.19 | 20.97 | +1.88 | 0/200 | 64/200 |
| assigned | ridge | 240.42 | 3245.81 | 549587.63 | 39254.96 | −3444.74 | 190/200 | 2/200 |
| pooled | fixed | 11.48 | 173.93 | 14303.83 | 1186.45 | −171.95 | 14/200 | 198/200 |
| pooled | calibrated | 12.73 | 32.05 | 44.56 | 17.08 | +9.72 | 0/200 | 48/200 |
| **pooled** | **selected** | **10.11** | **26.62** | **44.56** | **14.33** | **+1.58** | **0/200** | 130/200 |
| pooled | ridge | 130.07 | 5292.96 | 76476.77 | 5986.66 | −1267.71 | 174/200 | 1/200 |

Read together: **pooling is what to change first, and the rank rule is what still
matters once shots are scarce.** The recommended pair is `pooling='shots'` with
`solve_selected_rank` — the best or tied-best arm at both budgets, and the only
arm besides pooled/calibrated with no catastrophic replica anywhere. Pooling does
not make the rank rule redundant: `pooled`/`fixed` still leaves one catastrophic
replica in two hundred at the full budget. The manuscript's trade of median
accuracy against tail control does not survive pooling — the pooled arms beat the
published best median *and* the published best RMSE at once. Both are off by
default; every committed record predates them.

Two things the tables do not say. It is one bank at one conditioning, four qubits,
in simulation with no device noise — the recommendation is a default to try, not
an established scaling law. And no arm's value is a variational bound: the same
data supply the matrices, the retained rank, and the error bar, so `bias` stays a
measured property of an estimator, not a certified one.

*Considered and not done, with the measurement that decided it.*

- **Redesigning the settings themselves.** Once pooling exists the partition is no
  longer the right object — settings could be chosen to *overlap* deliberately,
  maximizing coverage-weighted readings rather than minimizing circuit count.
  Measured on this bank: a greedy cover from the 81 full four-qubit bases needs 57
  settings (against the partition's 52) to reach mean readers `4.09` (against
  `3.72`). About 10% more readings for 10% more settings — much smaller than the
  3.72× pooling already recovers from the existing partition, so the partition
  stays.
- **Completing each setting's basis** on qubits its words do not touch (free on
  hardware, which measures them anyway). Measured: mean readers `3.72 → 3.78` at
  four qubits and `4.41 → 4.43` at six. Negligible; not implemented.
- **Anytime-valid confidence sequences** in place of fixed-schedule
  empirical-Bernstein bounds. This is the remaining large win on the *certified*
  path, where two independent full-universe batches per step put certified growth
  four orders of magnitude above the exact path in shots; a confidence sequence
  would license outcome-dependent allocation and confidence-set reuse across
  growth steps. It is a genuine piece of work, not a knob. Alongside it: pooling
  helps the estimator more than it helps the current certificate — on the Ritz
  functional at 2 000 shots/group it cuts `sigma` from `5.87e-3` to `3.60e-3` (and
  the normal radius likewise) but the EB radius only from `0.208` to `0.183`,
  because that bound is a sum of per-group radii under a union bound over the
  groups touched, and pooling touches at least as many. Bounding the sum directly
  is the companion piece.
- **A variance-penalized Rayleigh quotient** — the continuous version of
  `solve_selected_rank`, minimizing `R(c) + γ·σ(c)` over the coefficient vector
  rather than over the rank. Better motivated than the ridge, since it penalizes
  the numerator noise the ridge fails to reach, but each `σ(c)` evaluation is a
  covariance-vector product over the whole word universe inside an optimization
  loop.
- **Wiring pooling into noisy ADAPT selection.** `GroupedWordCache` is also what
  `run_adapt` builds when `grouping=True`, and an ADAPT gradient is the same shape
  of object — a linear functional of word means — so pooling applies unchanged. It
  is deliberately not wired through: `run_adapt` carries a structured/legacy
  keyword reconciliation and a certified selector whose calibration records are
  committed, and a free variance reduction is not worth half-changing that surface
  in the same pass. The cache supports it today; only the plumbing is missing.
- **Bias correction** by grouped bootstrap or jackknife. The measured bias of the
  recommended arms is `1.1`–`1.6 mHa` against median errors of `2.6`–`10`, so
  there is little left to correct once pooling and rank selection are in.

### Phase 5 — done (`models/lattice.py`, `models/observables.py`, `sparse.py`)

`models/lattice.py`: Hubbard, extended Hubbard, Kanamori, small Anderson
impurity, Kitaev honeycomb cluster. The fermionic models are built from the
package's *own* Jordan-Wigner operators, so the materials layer needs no
chemistry extra at all; the Kitaev cluster is a native `PauliSum` with one qubit
per site and no transformation. Hopping is written once and added to its own
adjoint, so hermiticity is structural. Every model carries
site/orbital/bond/sector metadata, which is what lets an observable be asked for
by site rather than by spin-orbital index.

*Observables through the projected-matrix route* (§8): occupations, double
occupancy, per-site spin operators, spin correlations, the antiferromagnetic
structure factor, Kitaev per-link bond operators, and `S²` — each a `PauliSum`
handed to `result.expectation(Q)`, so no Ritz state is ever formed, and every one
checked against the dense exact state A-CASE refuses to store.

*Excited states*: `run_acase(roots=k, aggregation='mean'|'max')` — state-averaged
growth (objective = average of the tracked roots) or block growth (whichever root
gains most decides). Records carry `root_energies` and `per_root_lowering`. The
per-root variational bound `E_k^sub ≥ E_k` holds by Cauchy interlacing and is
tested; the *objective* is monotone only from the step where the effective rank
first reaches `k`.

*Ingestion*: `models.fcidump.fcidump_model` reads the restricted real FCIDUMP that
a downfolding or embedding step actually hands over, with NumPy only. It restores
packed one- and two-body symmetries, validates `NORB`, `NELEC`, `MS2`, sentinel
patterns, duplicates, and finite coefficients, rejects unsupported `IUHF=1`, fixes
the interleaved-spin reference sector, records the source SHA-256, and maps
chemist integrals directly with the package's own fermion operators. The
chemist/physicist ordering remains the trap; the chemistry-extra test compares all
185 H₄ coefficients against the independent OpenFermion/PySCF construction
(maximum mismatch `8.7×10⁻¹⁶`).

*Committed active-space rung*: the linear H₄ STO-3G CAS(4e,4o) FCIDUMP is frozen
with geometry, PySCF 2.14 provenance, SHA-256, RHF energy, and a determinant-space
FCI oracle. It maps to 8 qubits, 185 Pauli terms, and a 36-state `(N=4,S_z=0)`
sector; the mapped sector energy agrees with external FCI to `3.1×10⁻¹⁵ Ha`. At
eight adaptive additions A-CASE has `M=9`, `W=7,371`, `kappa(S)=1`, and a
`3.019 mHa` error. The complete singles/doubles coordinate space has `M=27`,
`kappa(S)=1`, and a `0.766 mHa` error. This is a reproducible chemistry benchmark,
not evidence of quantum advantage; the adaptive arm does not reach chemical
accuracy at the declared budget.

*Sparse reference tier*: `sparse.py` writes each Pauli word as the signed
permutation matrix it is (`W = i^{n_Y} X^x Z^z`) instead of summing dense
Kronecker products, giving `eigsh` a Hamiltonian with `≤ (#terms)·2^n` nonzeros —
a 12-qubit XXZ ground state in 0.1 s. Two findings are baked in:

- **`which='SA'` is not safe here.** On the `t = 0` Hubbard cluster (diagonal,
  eigenvalues in `{0, U, 2U, …}`, 256-fold zero eigenspace) ARPACK returns `U`,
  converged and residual-free, for a matrix whose minimum is 0. A residual check
  cannot catch it — `U` really is an eigenvalue. The fix is in how the problem is
  posed: solve for the largest-magnitude eigenpair of `H − σI` with
  `σ = Σ_w|h_w| ≥ ‖H‖`, which costs one diagonal and no factorization.
- **A grand-canonical cluster does not minimize at the filling its name implies.**
  With `μ = 0` the 4-site Hubbard chain's global ground state sits in the *two*-
  electron sector. `hubbard` therefore defaults to `μ = U/2` (and Kanamori to
  `U/2 + (M−1)U'`, the interaction's linear residue under `n → 1−n`), and
  `sparse_ground_in_sector` restricts to a `(N, S_z)` block when a specific filling
  is wanted — the honest comparison for A-CASE, since the subspace stays in its
  reference's sector.

*Measured (`examples/acase_materials.py`).* Analytic limits first, because they
are what catches a hopping sign or a JW string: the `U = 0` Hubbard chain
reproduces `2Σ_{ε_k<0} ε_k` to 10⁻⁸ for 2, 4, and 6 sites; the `t = 0` cluster
gives `−UN/2`; free-fermion double occupancy is exactly 1/4; the singlet ground
state has `⟨S²⟩ = 0`; and the Kitaev cluster's energy is reproduced by its three
per-link correlators alone (`⟨XX⟩_x = ⟨YY⟩_y = 0.4527`, `⟨ZZ⟩_z = 0.7879`,
non-link pairs at `−0.015`) — the spin-liquid signature.

The load-bearing negative result is on the Hubbard clusters. From the Néel product
reference, the **entire** singles-and-doubles response space saturates at a gap of
3.2×10⁻¹ (4-site chain) and 2.6×10⁻¹ (2×2) against the sector ground energy at
`U = 4`, and adaptive growth reaches that same limit and then correctly stops —
the space does not contain the state. Adding Krylov candidates helps the chain
(1.8×10⁻¹ at M=11) and not the 2×2. Observables converge in the right direction
along the trajectory (double occupancy 0 → 0.053 against an exact 0.072, `⟨S²⟩`
2.0 → 0.82 against 0) without the energy converging.

*That debt is now paid, and it settles the span half of Q4.* Level 4 ships (§4.2),
and on the 2×2 cluster it does exactly what was predicted: levels 0–3 saturate at
`−9.8475` against a sector ground energy of `−10.1027`, adding the competing-order
configurations *by themselves* changes nothing (they are determinants `H` does not
connect to the reference at first order), and adding the compound products
`configuration × excitation` reaches `−10.102748` — the sector ground state, to
`1e-8`. The barrier was the span, and it was the one-reference structure of levels
0–3 that imposed it.

### Phase 6 — done (`backends/sector_statevector.py`)

`SectorStatevectorBackend` stores a pure state on the occupation words of one
`(N, S_z)` sector — `C(n,k)` amplitudes, never `2^n` — and applies a Pauli word as
a bit-mask gather, `W|b⟩ = i^{n_Y}(−1)^{|z∧b|}|b⊕x⟩`. Words are **grouped by
X-mask**: every word in a group shares the permutation `b → b⊕x`, so the gather is
resolved once per group and only the diagonal phases differ — and those phases do
not depend on the state either, so each group collapses to one coefficient vector
and a matvec is a few gather-multiply-scatter passes. `SectorOperator` exposes
that as a `LinearOperator` for `eigsh`, with a numpy-only `lanczos_ground`
fallback. `sector_projector` builds the ideal's projector as an `MV` for
theory-facing checks; the backend never forms it.

Three implementation points are load-bearing:

- *Term-wise projection is exact, not approximate.* Individual words of a
  number-conserving Hamiltonian do **not** conserve `N`, so most words map part of
  the sector out of it, and the backend drops those components. That is legitimate
  because `H` commutes with the sector projector: `H|ψ⟩ = P H|ψ⟩ = Σ_w h_w
  (P W_w|ψ⟩)`, and `P` distributes over the sum. The out-of-sector pieces cancel in
  the total; projecting each term is the same arithmetic reordered.
- *No `2^n` index table.* The permutation `b → b⊕x` is resolved by binary search on
  the sorted sector, not by a lookup array over the full space — which would
  reintroduce exactly the memory the backend exists to avoid. Basis construction is
  combinatorial for the same reason (`C(40,2)` states out of `2^40` in
  milliseconds).
- *Lanczos converges on the residual, not the eigenvalue.* Ritz values converge
  quadratically faster than their vectors, so stopping when the eigenvalue settles
  returns vectors an order of magnitude short of the advertised tolerance. The
  criterion is `β_k|s_k[i]|`, with full reorthogonalization (the bare three-term
  recurrence starts manufacturing duplicate eigenvalues, which on a degenerate
  spectrum is indistinguishable from real degeneracy).

*Validated:* ground energies match `exact_ground` and `sparse_ground_in_sector` on
every lattice model for `n ≤ 12`, through both `eigsh` and the numpy-only Lanczos;
the matvec matches the sparse submatrix; `P² = P`, `P† = P`, `tr P = |sector|`,
`[H,P] = 0`; the sector basis agrees with the independent dense enumeration;
expectations of *non*-conserving observables agree with the dense restriction; and
the `t=0` degenerate spectrum that defeats ARPACK's `which='SA'` is handled by both
solvers.

*Measured (`examples/acase_sector_backend.py`, half-filled Hubbard chains):*

| sites | n | sector dim | 2^n | ratio | state | sparse nnz it avoids |
|---|---|---|---|---|---|---|
| 4 | 8 | 36 | 256 | 7.1× | 0.6 kB | 4 352 |
| 6 | 12 | 400 | 4 096 | 10.2× | 6 kB | 110 592 |
| 8 | 16 | 4 900 | 65 536 | 13.4× | 78 kB | 2 424 832 |
| 10 | 20 | 63 504 | 1 048 576 | 16.5× | 1.0 MB | 49 283 072 |
| 12 | 24 | 853 776 | 16 777 216 | 19.7× | 13.7 MB | — |

Ground energies: 20 qubits in 1.8 s, 24 qubits in 54 s. The compiled operator is
19 X-mask groups from 47 words at `n = 20` (matvec 6.6 ms, 22 MB held) against
93 ms recomputing per matvec — the compile-once/solve-many trade a DMFT-style outer
loop wants, and `precompute=False` is there because at `n = 24` the compiled passes
want 355 MB against the state's 14 MB.

*A Phase-5 bug this phase caught.* The lattice metadata hardcoded `S_z = 0` at half
filling, which is wrong for an odd site count. The backend refuses to build an
empty sector, which surfaced it; sector metadata is now read from the reference
determinant's own gates, so it cannot disagree with the state it describes.

### Phase 7 — done (`benchmarks/run_acase_ladder.py`, `summarize_ladder.py`)

The validation ladder and everything it settled are in §7.

### Phase 8 — trusted sampled-subspace baseline — shipped

`subspace/qsci.py`, `tests/test_qsci.py`, ladder arm `qsci`. QSCI is a
first-class method rather than a benchmark stub. The contract below is what it
implements and what its rows must carry.

**8A — sampling contract.** A result object carrying at least: raw and accepted
shots; unique basis configurations and duplicate fraction; discarded or repaired
fraction; cumulative retained probability; sampled subspace dimension `M`;
projected-matrix measurement words `W=0`; classical matrix nonzeros, build time,
solve time, and peak memory; energy, variational gap, and evidence label. The
initial implementation samples exact probabilities from existing state backends;
hardware-noise emulation and configuration recovery are later layers over the same
contract.

**8B — sampled Hamiltonian restriction.** Reuse `sector_basis`, sector indexing,
and the compiled `SectorOperator`; add a safe API that restricts a sector operator
to a declared set of sector indices. An exact row/column restriction of the
validated sector operator is preferable to writing Slater–Condon rules
prematurely. Required invariants: (1) the sampled Hamiltonian is Hermitian; (2)
increasing nested sampled sets give non-increasing Ritz energies; (3) for every
retained root `k`, the `k`-th sampled Ritz value is not below the `k`-th exact
sector eigenvalue (Cauchy interlacing); (4) selecting the entire sector reproduces
the sector spectrum; (5) permutation of sampled configuration order changes no
eigenvalue.

**8C — fermionic recovery and generic spin sampling.** For fermionic systems,
post-selection and optional recovery against particle number and `S_z`, reporting
every discarded or repaired sample. For spin systems such as Kitaev, do **not**
report that QSCI has no arm: the fermionic SQD recovery rule is unavailable, but
raw computational-basis sampled subspace diagonalization remains a valid baseline.
The question is whether the basis is compact, not whether it is definitionally
excluded.

**8D — state inputs.** Declared inputs: reference determinant; exact ground-state
sampling oracle (validation only); existing ADAPT-VQE state; later, a time-evolved
state. Never mix oracle and implementable inputs in one evidence category.

**8E — ladder integration.** Add QSCI to the ladder. Equal-`M` remains one
comparison, but the ladder must also emit Pareto records for error versus quantum
shots and unique configurations; versus state-preparation cost; versus classical
matrix nonzeros and solve time; and versus memory.

**Go/no-go:** the full-sector limit, Hermiticity, interlacing, and permutation
invariants must hold on H₄, Hubbard, and at least one spin model before QSCI is
used in manuscript claims.

### Phase 9 — classical selected-CI controls — shipped (mandatory)

`subspace/selected_ci.py`, `tests/test_selected_ci.py`. Without it, a successful QSCI × A-CASE hybrid may be indistinguishable from
ordinary determinant-space expansion. For each sampled determinant set `D`:

1. **QSCI:** diagonalize only `span(D)`.
2. **Excitation closure:** add every unique determinant reached by the same
   singles/doubles used for operator dressing.
3. **One-step selected CI:** add determinants using a declared HCI-, CIPSI-, or
   perturbative-style score.
4. **Budget-matched selected CI:** stop at the same determinant count or classical
   matrix cost as the hybrid.

Record determinant count, Hamiltonian nonzeros, classical selection work, energy,
variance, and memory.

**Span-equivalence diagnostic.** For each dressed family compare
`span{E_mu |D_k>}` with the determinant closure generated from the same `D_k` and
excitation operators; compute numerical ranks and principal angles.

- Equal spans mean the operator form is a representation or measurement-cost
  choice, not a richer variational space.
- A smaller operator rank means the operator family spans strictly *less* than the
  closure. This is not by itself a compactness result and must not be reported as
  one: a closure's columns are distinct determinants, hence independent, so no
  smaller set of vectors spans a larger-dimensional space. Compactness is an
  energy-at-matched-size claim against the controls above, not a span property.
- Directions outside the declared closure require an algebraic explanation and an
  independent check.

The comparator must be the closure of the **declared generator family** — the
determinants those operators actually reach — not a re-derivation of singles and
doubles from each determinant's own occupancy. A fixed pool built relative to the
reference annihilates many sampled determinants and moves different electrons in
the rest: on the 2×2 Hubbard sector three sampled determinants reach 15
determinants under the pool and all 36 under the per-determinant rule. Against the
larger comparator every operator direction is trivially contained and the
diagnostic decides nothing.

Containment is directional and principal angles alone cannot answer it: there are
only `min(rank A, rank D)` of them, so a rank-2 operator span sharing one direction
with a rank-1 closure yields the single angle `0` and reads as contained. Measure
`||(I - Q_D Q_D^dagger) Q_A||` and require `rank(A) <= rank(D)`.

**Go/no-go:** Phase 10 may claim a hybrid gain only after it beats or differs
structurally from these controls.

### Phase 10 — QSCI × A-CASE hybrid — shipped

`subspace/hybrid.py`, `tests/test_hybrid.py`, `tests/test_phase10_driver.py`.

**10A** — convert retained QSCI configurations through `configuration_generator`.
**10B** — operator-response dressing in declared families: configuration ×
conserving excitation; configuration × commutator response; optional
support-bounded compound families. Every family reports candidate count, generator
support `S_A`, projected element support `S_H`, incremental word universe, and
conditioning impact. **10C** — required arms: bare sampled configurations; sampled
configurations plus individual dressed generators; sampled configurations plus
support-pruned Haar packets followed by the ordinary dressed pool (the third arm
integrates the existing wavelet result without creating a separate state-vector
compression project). **10D — honest claim:**

> A sampled determinant set, enriched by selected operator-response directions,
> may reach a target accuracy with fewer retained variational directions or a
> better measured-resource Pareto point than either bare QSCI or bare A-CASE.

Do not claim that operator dressing is strictly richer until Phase 9 proves it.
Primary systems: `hubbard_2x2`, `hubbard_2x3`, H₄ equilibrium/stretched, and one
molecular FCIDUMP rung with matched multiplicity.

### Phase 11 — overlap-targeted and multiresolution selection — shipped

`subspace/multiresolution.py` and the `OverlapTarget`/`score_target_overlap`
path in `subspace/adaptive.py`; `tests/test_phase11.py`.

**11A** — extend A-CASE scoring with a target-overlap criterion using the QSCI Ritz
vector or a classical selected-CI vector. Preserve existing scale invariance and
orthogonality rejection; keep the target outside `CandidateScore` so lowering and
overlap criteria remain independently testable. **11B** — coarse-to-fine packet
selection using the existing configuration Haar transform: order sampled
configurations using declared physics metadata; score support-pruned coarse
packets first; refine only selected or competitive intervals; hand the retained
basis to the convergence-complete individual/dressed pool. Candidate blocks may use
`acase_step`, but the score stays A-CASE-specific — do not create a shared
ADAPT/A-CASE runner. **11C** — ordering ablations: probability order; excitation
rank plus occupation-pattern metadata; determinant-graph traversal; random-order
controls. A packet result is publishable only if it is not an accident of one
ordering. **11D** — stopping and uncertainty: duplicate-rate stopping, bootstrap set
stability, or an unseen-mass estimate first; wavelet/block allocation only if it
improves those controls.

**Go/no-go:** on `hubbard_2x3`, the new criterion or hierarchy must select useful
configuration/dressed directions that the lowering-only pool misses, or the blind
spot is attributed to the family rather than the selector.

### Phase 12 — integrated Paper B ladder — producer shipped

`benchmarks/run_phase12_paper_b.py`, normalizing every arm onto one schema so a
Paper B comparison cannot silently drop an inconvenient cost column, with Pareto
frontiers computed only inside one evidence category — an oracle-sampled QSCI
hybrid never dominates an implementable arm merely because the oracle has no
state-preparation cost to report.

Arms: reference and exact-sector results; fixed QSE and Krylov; ADAPT-VQE and
A-CASE; QSCI; excitation-closure and selected-CI controls; QSCI × dressed A-CASE;
QSCI × Haar-stage × dressed A-CASE. Required fields: `M`, retained rank, and
`kappa(S)`; energy error, variance or true residual where available; sampling
shots, unique yield, duplicate rate, discard/recovery rate; state-preparation
metadata; `W`, grouping contexts, and certified shot cost for measured arms;
classical matrix nonzeros, build/solve time, and peak memory; generator and element
supports; evidence category and seed.

Paper B must follow the Pareto frontier that survives. If QSCI and classical
selected CI dominate chemistry, narrow A-CASE to systems and representations where
operator-generated or packet directions add measured value.

### Phase PRD — preconditioned residual Davidson and WISE split — done

PRs #54 and #55 replace the proposed residual-oracle phase with the question that
survived the theorem and the data. Unpreconditioned orthogonal residual expansion is
Lanczos in an orthogonal basis and remains only a regression arm. Davidson is the
principal accuracy method: its shift is selected by the projected Ritz energy, the
entire shift curve and its work are retained, and the exact ground energy is not used
for selection. Packet Davidson is separately priced because a cheap classical
preconditioner is not automatically a bounded-support measurable generator.

The completed exact suite fixes the method hierarchy. At matched `M = 7`, Davidson
reaches the 1.6 mHa threshold on 9/12 systems, orthogonal residual on 6/12, and
A-CASE and matched selected CI on 0/12. Across the 49 valid exact points, Davidson
is in the minimum-error set at 46 points (ties included), orthogonal residual at 6,
and A-CASE and matched selected CI at 0. The 12 effective-rank rejections remain
explicit negative feasibility evidence; results must not be collapsed onto one
nominal-`M` axis when the effective ranks differ.

Word resolution is measurement infrastructure rather than a competing accuracy
method: it preserves the determinant-resolution A-CASE span and energy while
reducing retained-bank `W` by 58.2–91.4% (median 79.8%). The full selection cache
falls by only 5.2–56.8% (median 36.5%), so retained-bank compression must never be
reported as the cost of adaptive selection.

The preregistered finite-shot extension is a retained negative result. Frozen packet
Davidson (`M = 7`, `K = 16`) was measured with complete shared-QWC banks on H₄
stretched, Hubbard 2×2 at `U/t = 4`, and stretched H₂O, with 20 seeds at 10k, 100k,
and 1M aggregate state-preparation shots (180 cells). At 1M shots the primary median
errors are 6.35, 28.00, and 196.80 mHa; variational-violation rates are 65%, 50%,
and 95%; chemical-accuracy rates are 5%, 0%, and 0%. Packet selection and
coefficients came from exact simulation, the regularization floors are diagnostics
rather than finite-sample energy certificates, and no hardware-readiness or
implementable-selection claim follows.

**Architectural consequence.** PRD owns accuracy and basis growth; WISE owns word
reuse, measurement design, and inference stabilization; A-CASE remains the matched
span/measurement infrastructure and comparator. The next study is measurement-first:
price and reduce word/group width, then optimize a centered energy/Ritz functional
under a newly frozen protocol. R1 is the first accounting step and does not repair or
supersede the finite-shot negative result.

### Phase 13 — structural invariant first (Track B, done)

`clifford_qc.pauli_structure` now computes the GF(2) rank of Hamiltonian X masks
through `word_masks` and provides an explicit validator that rejects an operator the
caller declares spin conserving when `r_X > 2(N - 1) = n_qubits - 2`. The elimination routine is the same one that
validates the fermion-encoding matrices; the phase does not keep a second binary
algebra implementation that can drift.

`tests/test_pauli_structure.py` applies the invariant to the native fermion
construction, all four
committed FCIDUMPs, Hubbard, extended-Hubbard, Kanamori and Anderson-impurity
lattices, and versioned effective-Hamiltonian JSON ingestion. Their pinned
`(r_X / ceiling)` values are respectively `2/4`; H4 `5/6`, BeH2 `3/6`, LiH `5/6`,
H2O `8/10`; Hubbard `6/6`, extended Hubbard `6/6`, Kanamori `5/6`, Anderson
`4/4`; and the effective dimer `2/2`. A total-parity-preserving but
spin-nonconserving counterexample reaches `3/2` and is refused, so the gate is not
a vacuous range check. Passing is necessary rather than sufficient evidence of spin
conservation; any future test-matrix violation still stops Phase 14 grouping work.
The generic constructors do not enforce a spin-conservation claim. The Phase 14a
compiler invokes the validator only when its caller explicitly supplies a Hamiltonian
through `spin_conserving_jw_hamiltonian=`; it never infers the claim from a generic
word universe. This phase is deterministic and carries no sampled record.

### Phase 14 — QWC plus fully commuting groups

Extend measurement grouping with fully commuting groups and Clifford simultaneous
diagonalization. The metric is not group count alone: report certified leading shot
cost at fixed word universe, allocator, and confidence target, plus diagonalizing
circuit depth and two-qubit gates; connectivity assumptions; covariance-aware
reconstruction; Monte Carlo agreement between predicted and empirical uncertainty.
The multiply-capable-word variance bug remains the gate: lower group count with
inflated or double-counted variance is failure.

The dyadic block-commuting hierarchy already covers both endpoints and their
interior (§6.6); the device card, accuracy-matched shot counts, fidelity term, and
pooled re-measurement that were missing here are Phases R1 and R3, and both have
now shipped them — `protocol_cost.json` carries all four across the `mapping × k`
grid.

**Phase 14a — public compiled-plan boundary (done).**
`clifford_qc.measurement.compile_block_measurement_plan` now freezes a distinct word
universe and block size into one exact-once partition, executable `CompiledSetting`
objects, signed Z-parity readouts, the full setting-by-word compatibility matrix, and
the matched synthesis-resource ledger. Supplied partitions are independently checked
for block-wise commutation and complete assignment; every compiled readable word is
rechecked against the global Clifford tableau. The sampled Clifford is built from the
same reduced circuit whose gates and depths are priced. The compiler remains generic
unless the caller explicitly supplies `spin_conserving_jw_hamiltonian=`, which runs
the Phase 13 gate before grouping.

`tests/test_measurement_planning.py` pins the `k=1` QWC and `k>=n` fully commuting
endpoints, signed Bell readouts, the declared-JW validator hook, invalid supplied
partitions, and the multiply-readable-word failure mode. In the last case pooled
weights sum to one and the covariance prediction agrees with Monte Carlo rather than
duplicating the coefficient across capable settings. Existing exact-shot and protocol-
cost producers now consume the library compiler instead of a benchmark-private copy.
This is a deterministic library/test change: it creates no sampled record.

**Phase 14b — done; preregistered protocol executed once.**
`benchmarks/configs/phase14b_qwc_vs_fc.json` freezes the BeH2/JW five-vector bank,
the 1,814 measured-word universe (identity analytic), the `k=1` QWC and `k=8`
fully commuting endpoints, one outcome-independent coefficient-range Neyman
allocator, familywise `delta=0.05` over both protocols and 17 independent total-
shot endpoints, the 1.6 mHa criterion, all three existing device cards, and fresh
seed namespaces. The primary comparison uses single-assignment covariance-aware
Ritz reconstruction; a crossed assigned/pooled 1,000-replica audit is a blocking
check against inflated or duplicated multiply-readable-word variance. Exact `S`
and `H` reconstruction must agree within `5e-13`, and every card is reported
separately rather than collapsed into a hardware-universal winner.

The result-free declaration merged first as `bb7a76a`; the separate sampled
producer then executed its fixed streams without changing that config. All four
covariance-audit ratios pass (`1.0042`, `1.0114`, `1.0471`, `1.0320`) and exact
signed-readout reconstruction gives zero observed maximum error for both `S` and
`H`. QWC first certifies at `2^29 = 536,870,912` total physical shots, while the
fully commuting endpoint first certifies at `2^24 = 16,777,216`, a `1/32` ratio
that passes the preregistered `<= 1/2` material-reduction rule. At matched
shots, the covariance-aware variance is only 2.22 times lower for fully
commuting; most of the 32-fold certified-endpoint gap comes from the frozen
empirical-Bernstein certificate's per-group union bound over 179 touched QWC
groups versus 7 fully commuting groups. The audit validates the covariance
variance model, not empirical coverage of that radius. Ion-like and
logical-all-to-all card projections also favor the fully commuting endpoint;
the superconducting-like fully commuting plan is inadmissible under that card's
fidelity floor and therefore carries no runtime winner. The sampled state is the
Hartree-Fock computational-basis determinant `|11110000>`, whose stabilizer
variance structure is a special case. This is a positive Q9 answer only for
oracle-sampled measurement of that state on this frozen BeH2/JW bank, not a
state-preparation, hardware, noise, mapping, or cross-instance claim.

### Phase 15 — second-moment bank (Track C, open)

Add a `SecondMomentBank` for `K_ij = <psi|A_i^dagger H^2 A_j|psi>`. Before building
the full bank, add a support/cost preflight for `H^2`; if the estimated word
universe is prohibitive, keep dense or matrix-free residual oracles for validation
and restrict the measured implementation to declared small systems. Full bank
construction additionally requires the Phase 2M memory-bounded storage gate or an
explicit small-system exception; the preflight itself may proceed independently.
Uses: true Ritz residual norms; energy variance and variance extrapolation;
folded-spectrum roots; an independent convergence criterion.

### Phase 16 — time-evolved inputs, split by method

**16A — QSCI input.** Use `PauliLinearOperator.as_linear_operator()` with
matrix-free `scipy.sparse.linalg.expm_multiply`, Krylov propagation, or a validated
Trotter circuit to generate time-evolved sampling states. SciPy is a `research`
extra; when `expm_multiply` receives a `LinearOperator`, supply the analytically
known `traceA` rather than asking SciPy to estimate it from a matrix-free object.

**16B — A-CASE real-time generators.** Do not represent `exp(-iHt)` as an `MV` by
default: Pauli support can become dense. A circuit-native generator requires a
different matrix-element backend and resource model — architecture research. A
short-time polynomial response may be tested only with explicit truncation, norm,
fidelity, energy-error, and conditioning budgets against matrix-free propagation.
This phase owns Q8.

### Phase 17 — mapping validation and breadth

Before using BK or parity in scientific records: transform Hamiltonian, reference,
and generators consistently; verify energy and gradient invariance; compare Pauli
weight, distinct words, grouping, and circuits separately. Lower Pauli weight does
not imply lower `W` or fewer groups — and, more strongly, `W` is *exactly*
invariant across the linear encoding family (§6.2), so the invariance is a check
rather than a measurement. The executable form of this phase is R2 (§5, Phases
R1–R4). A CEO pool and dedicated MORE-ADAPT benchmark are follow-ups after Track A;
they constrain positioning but do not gate the QSCI hybrid experiment.

### Phase 18 — embedding boundary

Keep DMET and projection-based embedding outside the package. Provide a versioned
effective-Hamiltonian schema and a fragment-solver callback returning energy plus
one- and two-particle density matrices. QSCI, selected CI, and the hybrid should
implement the same callback.

### Phase 19 — anticommuting-clique (spin-factor) partitioning — proposed, unexecuted

The contract is §3.6. Two levers share one algebraic fact and sit in two different
compression classes (§6.8), so they are scoped, measured and reported separately —
and they are at different readiness, lever 1 needing a cover and its synthesis
priced, lever 2 needing a primitive the package does not have.

**Lever 1 — a clique cover for a Hamiltonian-energy estimand (Track B).** Phase 14
compares QWC (`k = 1`) against fully commuting (`k ≥ n`), both of which partition
into *commuting* sets. A clique cover partitions into *anticommuting* sets and
reads each one in a single setting through §3.6(i). It is not a strictly better
protocol and it is not on the dyadic block-commuting hierarchy's interior (§6.6):
it buys settings and pays up to `m − 1` Pauli rotations, which is precisely the
trade §6.3 forbids assuming.

**Its estimand is narrower than Phase 14's, and that is a scoping fact, not a
detail.** A rotated clique yields one number — the mean of the single weighted
observable `Σ_i c_i A_i` — and the individual `⟨A_i⟩` are *not* recoverable from
it. Phase 14b is declared on a `1,814`-word universe with a `966`-word Ritz
functional support (`benchmarks/configs/phase14b_qwc_vs_fc.json`), i.e. on
per-word means that many different functionals — every `S_ij` and `H_ij` built
from `supp(A_i† H A_j)` — reconstruct from and re-use. A cover whose coefficients
are frozen into the setting cannot serve that bank: it would need one cover per
functional, which is a different and probably worse protocol.

So lever 1 is scoped to a **fixed linear functional with fixed coefficients**, and
the natural one is the Hamiltonian energy. Calling it a third Phase 14 arm on the
declared bank is not licensed and is not claimed here. Widening it to the
matrix-element bank would first require defining a clique-compatible
matrix-element estimator and a word-coverage contract, and validating both against
the §6.6 pooled/assigned reconstruction; that is a separate deliverable, not an
extension of this one.

**Lever 2 — a candidate further removed qubit in a contextual restriction, and two
things that have to be built first.** `subspace/contextual.py` admits only
mutually commuting stabilizers and projects away everything anticommuting with
them. The method family that R4a's contextual arms belong to (§11, rows 15–16)
also carries one anticommuting clique, reduced by §3.6(i) and then fixed as a
further stabilizer. If that works here the gain is **one more removed qubit per
clique**, not less removed Hamiltonian weight — but "if" is carrying real weight
in that sentence, and the phase may not be written as though the qubit were
already available.

*The transport primitive does not exist.* `Restriction` takes a
`bridges.stim_bridge.CliffordMap`, which sends each Pauli word to a single Pauli
word; §3.6's `V` is a product of Pauli rotations and is generally **not**
Clifford, so it sends a word to a *sum* of words. Neither `Restriction` nor
`compile_contextual_restriction` can carry it. Lever 2 therefore needs a new
non-Clifford transport and projection primitive, with its own correctness
contract and its own cost, and that primitive — not the qubit count — is the
first thing to build and price.

*The contextual admission conditions are not automatic after the rotation.*
`select_contextual_stabilizers` requires more than commutation: the reference
expectation must be within `tol` of `±1`, and that sign must minimise the
candidate word's Hamiltonian coefficient (§5, R4a; `subspace/contextual.py`).
Closure of a clique into an involution establishes none of that about the rotated
`A_1` in the rotated frame. Phase 19 must verify both conditions *after* the
rotation, on the transported Hamiltonian and the transported reference, before
any removed qubit is counted.

**Sizing, from an in-session structural probe — this is not a committed record.**
No producer, config, record or checker exists for the numbers below. They were
computed once against `benchmarks/data/*.FCIDUMP` through the package's own
`gamma`, `rotor`, `qwc_groups`, `_pauli_anticommute` and
`select_contextual_stabilizers`, and they are here to size the phase, not to price
it. They authorize no rung, no arm, and no claim; reproducing them is Phase 19's
first task, not a precondition already met. Note also what the BeH₂ row is and is
not: it is that Hamiltonian's own term set, **not** the frozen `1,814`-word
element bank Phases 14b and R4a are declared on, so it may not be compared
against those records.

| bank | `n` | non-identity terms | clique cover (largest clique / ceiling `2n+1`) | QWC groups | `L1/L2` proxy |
|---|---|---|---|---|---|
| H₄ sto-3g | 8 | 184 | 42 (8 / 17) | 68 | 2.30× |
| LiH cas(4e,4o) | 8 | 192 | 42 (9 / 17) | 44 | 1.31× |
| BeH₂ sto-3g | 8 | 60 | 36 (3 / 17) | 13 | 1.18× |

The proxy is `(Σ_w |h_w|)² / (Σ_cliques ‖h_clique‖₂)²`: a variance-free ratio of
worst-case leading terms, not a certified shot cost, and not comparable to any
number in §6. It is reported because it is the only quantity computable before the
phase exists, and because the setting-count ordering between the two arms
*reverses* across these three banks — 42 against 68 on H₄, 36 against 13 on BeH₂
— which is what makes the comparison worth freezing rather than guessing.

**The lever-2 sizing is discouraging on the bank that matters, and that is
reported here rather than discovered later.** The quantity to read is *not* the
span of the ladder but its largest single-rung **marginal**, because what lever 2
trades is one commuting stabilizer for one clique — i.e. it reaches `k` removed
qubits from `k − 1` commuting stabilizers instead of `k`, and what it saves is the
weight the `k`-th stabilizer would have dropped. Removed Hilbert–Schmidt fraction
across one to five stabilizers, with the per-rung marginals under it:

```text
BeH₂  0.02339  0.02339  0.02371  0.02371  0.02414    marginals ≤ 0.00043
H₄    0.03345  0.05339  0.06993  0.07651  0.08172    marginals ≤ 0.01994
LiH   0.00181  0.00355  0.00373  0.00386  0.00396    marginals ≤ 0.00175
```

So the trade is worth at most `0.043%` of HS weight on BeH₂, `1.99%` on H₄ and
`0.175%` on LiH — and on BeH₂ two of the four rungs save exactly nothing. R4a
stopped on the **accuracy** gate — contextual bias floors of `5.345` and `5.899`
mHa against a `1.6` mHa target — and a lever worth `0.00043` of HS weight on that
bank is not a candidate to move a floor of that size. Lever 2 is therefore scoped
as a compression mechanism to be measured on its own terms and explicitly **not**
as a route to reopening R4a.

**Gate, before any producer is written.** The §3.6 invariants ship as tests first:
`(Σ_i c_i A_i)² = 1`; `V(Σ_i c_i A_i)V† = A_1` for `m ≥ 2` against the dense
reference, with the singleton and `c_1 < 0` sign cases pinned explicitly; the
`2n+1` construction and its `n = 2` maximality; and `Σ_i ⟨A_i⟩² ≤ 1` in exact
arithmetic.

The odd-grade invariant is scoped to the **parity-conserving fermionic** models
and must be written that way. A package-wide version fails on shipped builders,
not hypothetically: `tfim` and `random_ising` carry single-qubit `X_i` field
terms, four of seven at `n = 4` in both, and those are odd. `kitaev_honeycomb`
passes but proves nothing, being a direct spin model with no Jordan–Wigner
transformation. A test that asserts the invariant over every lattice model in the
package would be red on arrival.

A clique arm whose settings look cheaper only because the rotation resources went
unpriced is the same failure mode as Phase 14's multiply-capable-word variance
bug, and it blocks the phase on the same terms — with the §3.6(i) refinement that
those resources are synthesised and measured per setting, not derived from `m`.

**First deliverable** is a result-free preregistration in the Phase 14b shape —
frozen bank, frozen cover, allocator, confidence target, all three device cards,
and the depth accounting — landed before any sampled comparison runs.

### Phases G1–G3 — GA structural preconditioner (G1 done; G2–G3 retired)

The contract is §3.5. These are labelled **G**1–G3, not R1–R3, because Phases
R1–R4 already exist and R1 is shipped; where an external note called these
"R1–R3", read G1–G3.

**G1 is built; G2 and G3 are retired by its result.** G1 ships the project's
standard triple — `clifford_qc/subspace/ga_restriction.py`,
`run_g1_structural_preconditioner.py`,
`reference_results/g1_structural_preconditioner.json`,
`check_g1_structural_preconditioner.py` — plus its `REPRODUCING.md` entry, and it
runs in the deterministic `structural-records` matrix. It samples nothing and
prices nothing; the checker fails a record carrying any cost field.

**G1 — the structural preconditioner.** Build the fermionic/Majorana candidate
pool in `Cl(2n,ℂ)` (§3.5) and implement filters A–E. Output: surviving abstract
operators, before any encoding.

*Deliverables.* A Majorana-generated candidate pool; the five filters; and a
record that reports, per filter and **in the order applied**, how many candidates
entered and survived. The per-filter marginal is the deliverable — an aggregate
"GA removed 60%" is not interpretable, because A and B are already enforced on the
Pauli side today and would otherwise be double-counted as new.

*Gates.* (1) Filters A–D reproduce, on the same instance, exactly the candidate
set the existing Pauli-side symmetry and reference-leakage machinery accepts;
disagreement is a bug in one of the two and blocks the phase. (2) Filter E's
marginal contribution is reported **separately**, since it is the only filter with
no post-encoding analogue (§3.5E). (3) The `Restriction` object of R2/R4 is reused,
not reimplemented, so the congruence rule holds by construction (§3.5).

*The falsifier for the whole G programme.* If A–D reproduce the existing filters
and **E removes nothing**, then §3.5 is a reformulation in nicer language, not a
method, and G2/G3 do not run. That outcome is publishable as a short negative —
"the pre-encoding algebraic structure of this candidate family is already
exhausted by post-encoding symmetry filtering" — and it is cheaper to discover
here than after building the mapping experiment on top of it.

*Result — both gates pass, and the falsifier does not fire.* On BeH₂, H₄ and
Hubbard 2×2 at 8 qubits, over the Hermitian Majorana monomial pool of degree
`0..4` (`2517` candidates) and the rank-≤2 excitation pool built without its
`conserve_sz` argument (`52`):

- **Gate 1 holds with an empty symmetric difference**, both directions, on every
  pool and instance: filters A–D admit exactly the `549` (Majorana) and `26`
  (excitation) candidates the shipped `reference_sector_leakage` accepts. A
  second, independent form holds on the excitation pool — A–D land on exactly
  the set `determinant_excitations(conserve_sz=True)` keeps, which is the pool
  R2b, R3 and every cost record are built on.
- **Gate 2, E's marginal, reported separately and split by pool.** E removes
  `522` of the `549` reaching it on the Majorana pool and **nothing** on the
  excitation pool. All `549` entrants are distinct Pauli words under the
  package's own `scalar_free_key`, so no removal is deduplication — which is the
  measurement §3.5E's claim actually needs.
- **The two pools coincide at the matched cap.** The `27` surviving Majorana
  classes reach exactly the determinants `{identity}` plus the `26` excitations
  reach: same set, nothing on either side the other misses. The chain
  *reconstructs* the excitation pool rather than producing a different or smaller
  one. A degree sweep (`9 → 27 → 35 → 36` classes at caps `2, 4, 6, 8`, against
  the `36`-dimensional sector) shows the match is at the matched cap and nowhere
  else; a wider pool reaching more determinants is the pool being wider.

*Two things the record measured that the design did not anticipate.* First,
§3.5B's reference-conditioned form is load-bearing by `512` candidates: the
global commutant admits `37` where the reference-conditioned test admits `549`,
so a filter B written as `[A,Q]=0` would fail gate 1 outright. Second, the target
character and the restriction arm are **not independent declarations**. Under a
declared `(N=4, S_z=1)` character filter B admits `240` candidates — the
parameter is live — but filter C, carrying stabilizer signs fixed from the
*reference* sector, then annihilates all `240` and the chain returns an empty
admissible pool rather than an error. §7.4's excited-state track needs both moved
together; a G1 that let them drift would close that track at filter C while
§3.5B's requirement at filter B still looked satisfied.

**G2 — retired, on G1's result rather than on a schedule.** The paragraph below
is the design as written. It does not run, and the reason is G1's matched-cap
correspondence: G2 exists to give JW and BK the same *physical operator domain*,
but on this instance family that domain is the pool R2b already maps. QG2's
falsifier — "identical mapping conclusions from both pools" — therefore holds by
construction, and rerunning the mapping producer would reproduce the frozen
record rather than test anything. G3's `raw → GA` arrow is zero for the same
reason: the two pools span the same directions, so the arrow's interval covers
zero before any cost is measured. Both stay written here because the retirement
is a *result about this pool family at this cap*, not a judgment that the design
was wrong — a candidate family the excitation builder does not already exhaust
would put them back in scope, and would need its own declaration.

**G2 — the design as written, retained for the record.** Encode exactly the G1 pool
through JW and BK independently. Verify energies and subspace actions agree;
record the mapping-*dependent* quantities separately: `W`, the three weight
multisets, QWC structure, `G(k)`, `N_1q`/`N_2q`/`D_2q`.

This is a **re-scope of R2, not a duplicate of it.** R2 as written maps the raw
generator family; G2 gives both mappings the same *physical operator domain*,
which is the cleaner experiment because it separates structural compression from
encoding locality instead of confounding them. R2's pre-registered predictions
P1–P6 and its hidden-cost gate (the JW-specific sector layer) transfer unchanged.
The initial R2b experiment runs on the raw pool, as §13 explicitly orders. If G1
later survives its falsifier, G2 reruns the same mapping producer on the
G1-admissible pool; it does not replace or retroactively redefine the raw result.

**G3 — PRD and WISE on the admissible pool, with a cost decomposition.** PRD ranks
only GA-admissible operators; WISE measures the resulting JW/BK representation.
The deliverable is the attribution:

```text
C_HW^raw  →  C_HW^GA  →  C_HW^GA+PRD  →  C_HW^GA+PRD+WISE
```

each measured at a **fixed accuracy target on a declared evidence tier** (§6.1) and
under a named device card (§6.4) — a chain of counts would be exactly the reading
§6 exists to block, and P2 already reports the count reductions (§1.2(1)).

*Gate.* Each arrow reports its own margin with the shot-search uncertainty
propagated. An arrow whose interval covers zero is reported as "no measured
contribution at this instance and target", not dropped. The decomposition is the
result whichever way it comes out: it says where the saving actually comes from,
and a finding that GA contributes nothing once PRD runs is as informative as the
converse.

### Phases R1–R4 — hardware-aware resource accounting

The cost model these phases install is §6; the phases themselves:

**R1 — cost infrastructure, no solver change — done.** Deliverables:
`clifford_qc/measurement/cost.py` (device card loader, per-setting
gate/depth/fidelity accounting, `C_time`, admissibility, break-even surface);
`benchmarks/configs/device_cards/*.json` with at least `logical-alltoall`, one
superconducting-like and one ion-like card; `run_clifford_hierarchy.py` extended to
schema `clifford_qc.clifford_measurement_hierarchy.v3` carrying the §6.5 columns for
both estimators; an extended `check_clifford_hierarchy.py`; a `REPRODUCING.md`
entry. *Gates:* (1) under `logical-alltoall`, every v2 field regenerates digit for
digit — a changed number is a regression in the extension, not a finding; (2) the
pooled arm is reported beside the single-assignment arm on the same bank, and if
coverage `f_w` moves the accuracy-matched cost ordering between `k` rungs, that is
R1's headline result and it lands before any mapping work; (3) no solver, selector,
or certificate code changes.

The shipped schema-v3 hierarchy closes the structural part of this phase: all three
device cards, `N_1q/N_2q/D_1q/D_2q`, timing, routing sensitivity, fidelity,
admissibility, break-even surfaces, and assigned/pooled coverage are recorded while
the frozen v2 projection regenerates exactly. The 1.6 mHa comparison is explicitly
`asymptotic`, using the exact frozen-bank bias plus the covariance-aware first-order
Ritz functional. H₄ is unattainable because its 3.019 mHa subspace bias already
exceeds the target; on BeH₂, pooling reorders the `k` rungs under all three cards.
The exact-tier nonlinear finite-shot search is a separate producer
(`benchmarks/run_exact_shot_search.py`), record
(`reference_results/exact_shot_search.json`), and regenerating checker.  It samples
the actual joint bitstrings after each synthesized Clifford diagonalizer, reruns the
complete measured `(S,H)` reconstruction and `E + 2 sigma` selected-rank sweep, and
compares every replica with the full exact-sector ground energy.  The primary aggregate
is replica RMSE: a search endpoint passes only with zero solver failures and a one-sided
95% nonparametric-bootstrap upper bound at or below 1.6 mHa.  Thirty paired exploratory
replicas locate a persistent crossing on the predeclared geometric grid
`64, 256, 1024, 4096, 16384, 65536` effective shots per setting; an independent block
of 100 paired replicas confirms every grid point through that crossing.  The estimator
arms share the same sampled caches, endpoints are nested, and phase/rung/replica/
bootstrap streams use disjoint `SeedSequence` namespaces.  The producer prices only
the smallest confirmed passing endpoint, requires a confirmed failing endpoint below
it unless the crossing lies below the grid, and abstains on a nonmonotone confirmation.

**A crossing decided on the target is a region, not an integer.**  Each arm records
the distance from 1.6 mHa for both deciding endpoints and flags the crossing when
either lies within ±10%.  Three of BeH₂'s eight arms are flagged: `k=1` assigned
(passing at −4.6%), `k=4` assigned (passing at −8.4%), and `k=4` pooled (passing at
−4.2%).  The band is calibrated against a measured effect rather than chosen —
rebuilding the record under a different NumPy moved the `k=4` assigned upper bound at
4096 shots across the target by 2.2%, changing that arm's reported count from 4096 to
16384 with identical shot histograms, because a few ill-conditioned rank-5 solves land
elsewhere under a different bundled LAPACK.  The flag names the same three arms in
both environments, so it is stable where the count is not.  This is §6.7's `k*`-as-a-
region rule applied one level down, to the shot count feeding each `C(ε)`: a flagged
arm's price carries the width of its crossing, and R3 must not read it as exact.

*And an independent stream says the same thing.*  R3's cost record reprices this
exact BeH₂ bank through the `jw` mapping arm, reaching it by the mapping-axis
transport instead of the DA-CASE producer and drawing from disjoint seed roots.
Six of the eight crossings agree endpoint for endpoint.  The two that differ —
`k=4` under both estimators, where R3 confirms 16384 against R1's 4096 — are two
of the three arms flagged above.  Nothing was rebuilt and no environment changed;
two independent streams simply landed on opposite sides of a crossing decided on
the target, which is what the flag predicts and what an unflagged crossing must
not do.  `check_protocol_cost.py` states it that way round: a disagreement is
tolerated where either record flags the crossing and is a failure where neither
does.

**Where a backend change may and may not reach.** The standard this search is held
to is that swapping BLAS/LAPACK may move floating-point residuals but must not move
the grouping, the Clifford equivalence, the random samples, or the verdict. The
first three hold by construction: grouping is packed GF(2) parity over integer
codes, the diagonalizers are exact `stim` tableaus whose coset representative is
chosen by *integer* gate count, and every replica draws from
`SeedSequence(root, spawn_key=(k, replica))`, so its stream is a function of its own
coordinates rather than of what ran before it. The estimator arms consume the same
drawn batch through nested endpoints, so `single_assignment` and `pooled` differ by
reconstruction and not by luck.

One float comparison survives all of that and becomes a *discrete* choice: the
retained rank, cut on the eigenvalues of an ill-conditioned overlap matrix. That is
the mechanism behind the recorded LAPACK sensitivity, and
`check_exact_shot_search.py` now gates it directly — a deciding endpoint, passing or
failing, whose 100-replica panel does not agree on one retained rank is reported as
**rank-marginal** and its crossing is not a resolved shot count. The gate is live
rather than vacuous: it passes on the committed record, whose eight deciding
endpoints are unanimous, while six non-deciding endpoints at 64–256 shots do split
their rank and are left alone because they decide nothing. With that invariant
enforced, a residual `4096 ↔ 16384` fluctuation is classifiable as Monte Carlo
threshold uncertainty rather than environment sensitivity.

The comparator tier is `exact` (the unavailable-on-hardware oracle); the bootstrap
uncertainty remains `heuristic`.  It is not a finite-sample Ritz-energy certificate or
deployable stopping rule.  H₄ still exits before sampling because its 3.019 mHa exact
subspace bias exceeds the target.  On BeH₂ the independent 100-replica block confirms
assigned/pooled passing endpoints of `4096/1024` shots per setting at `k=1`,
`16384/4096` at `k=2`, `4096/4096` at `k=4`, and `16384/4096` at `k=8`.  Every arm
reports a confirmed bracket — a confirmed failing endpoint directly below the priced
passing one — rather than a loose pilot upper endpoint.

*The `k=4` endpoints moved when the diagonalizers did.* They were `16384/16384`
before the minimal-synthesis correction below and are `4096/4096` after it. This is
not a shot-search regression and not a re-tuning: the search samples real bitstrings
after each synthesized Clifford, so changing the diagonalizer changes the joint
readout structure and therefore the estimator's variance. `k=4` is exactly the rung
whose crossing this section already flags as environment-marginal — the arm that a
different bundled LAPACK had already moved across the target — so it was sitting on
the crossing and a cheaper diagonalizer pushed it over. The flag, not the count, is
the stable quantity, which is the rule this paragraph exists to state. Both `k=4`
arms remain flagged after the change, and no endpoint at `k∈{1,2,8}` moved at all.
Pooling changes the accuracy-cost order under the ion-like and logical-all-to-all
cards; only `k={1,2}` is admissible on the superconducting-like card, and pooling does
not reorder that common set.  The asymptotic ledger was not promoted: in particular,
its very small `k=4` assigned-shot prediction does not survive the nonlinear search.

**R2 — the mapping axis.** Arms: `JW`, `parity`, `parity+2q`, `BK`, `BK+2q`. Held
identical across arms: Hamiltonian and active space, reference determinant,
generator family and its enumeration order, growth budget, accuracy target,
estimator, rank rule, seed.

*Which pool this runs on.* The first R2b record uses the raw generator family. This
is the explicit ordering decision in §13: R1 has cleared the dependency, while G1
is speculative and may stop at its own falsifier. If G1 later survives, G2 applies
the same producer to the G1-admissible pool as a second, separately labelled
comparison. Everything below holds for both records; neither is silently replaced.

*Implementation state.* `clifford_qc/fermion_mapping.py` constructs the five
declared arms as explicit invertible GF(2) occupation-bit maps. JW is the identity,
parity is the prefix-parity network, and BK uses Fenwick-tree rows; every unreduced
arm is a CNOT-only Clifford change of representation. The `+2q` arms use a linear
base-network-plus-fixup construction, complete their base rows with spin-up and
total-parity rows, derive the fixed signs from declared `(N,S_z)`, and delegate
rotate/fix/delete to the R2a `Restriction`. The shared
`assert_mapping_invariants` gate compares the projected `(S,H)` matrices, retained
rank, condition number, Ritz values, reference energy, and word bijection; at small
`n` it also compares the full mapped/fixed-sector spectrum against an independently
extracted dense block. These are implementation checks only: no R2 cost row or QR3
answer is accepted from an arm that fails them. The completed experiment is
`benchmarks/run_mapping_axis.py`, configured by `configs/mapping_axis.json` and
frozen in `reference_results/mapping_axis.json`; `check_mapping_axis.py` rebuilds
the record and independently enforces the numerical invariants, tier,
QWC-partition, QR3-arithmetic, and bias-floor contracts.

*Step 1 — invariance, as checks.* Construct the encoding change as a CNOT network,
verify it is Clifford through `clifford_tableau`, then assert: spectrum on the
sector, reference energy, `W`, `S_H`, `M`, `κ_S`, Ritz values, and the word-multiset
bijection. Failure stops the phase — this is Phase 13's discipline applied to
mappings.

*Registration/deviation note.* The original Step 2 registered `G(k)` and coverage
across protocol rungs. This closing PR executes only the fixed-QWC (`k=1`) mapping
axis and defers those two quantities to R3, where the complete `mapping × k` grid
belongs. This is an explicit post-registration scope change, not evidence for P5;
the record carries `deferred_to_r3` so the successor experiment cannot silently
drop them.

*Step 2 executed here:* weight distributions on the Hamiltonian, generator,
element-operator, and deduplicated word multisets; one fixed QWC grouping per arm;
`N_1q/N_2q/D_1q/D_2q`; fixed-shot projections; and an explicitly asymptotic
single-assignment `C_time(ε)`. P5 remains untested until R3.

*Pre-registered predictions and falsifiers.*

- **P1.** `W_JW = W_BK`, and `κ_S`, `M`, Ritz values identical to solver tolerance
  for the non-reduced arms. *Falsifier:* any difference — which indicts the
  implementation, not the hypothesis.
- **P2.** `w̄` on the Hamiltonian multiset falls under BK relative to JW, most
  visibly at larger `n`. *Falsifier:* no reduction at `n = 8, 12`, which would mean
  the asymptotic argument has no purchase at the sizes this project runs — itself a
  publishable negative for a chemistry-scale claim.
- **P3.** At QWC (`k = 1`) the mapping's effect appears in **single-qubit gate count
  and group count, not depth**: the rotation is one gate per non-identity axis
  whatever the weight, and the frozen record carries `gate_counts_per_sweep`
  (H₄: 5 184 `H`, 2 592 `S_DAG` at `k = 1`, so `N_1q = 7 776`) as the place it
  shows. *Falsifier:* a material `D_1q` difference at `k = 1` beyond the two
  layers a `Y` rotation needs, meaning the synthesizer is not emitting a minimal
  rotation layer.

  *Restated after a synthesis defect.* This prediction was originally written
  against `14 608 H` and `15 552 S`, and its falsifier said "not emitting a
  single rotation layer". Those figures were an artifact: stim's
  `from_stabilizers(...).inverse()` returns an arbitrary member of the coset of
  Cliffords that diagonalize a block, and `to_circuit("elimination")` then
  expands it unoptimized — nine gates for a one-qubit `Y` rotation that two
  realise. P3 would therefore have been falsified by tooling rather than by the
  encoding. `clifford_qc/measurement/block_synthesis.py` now picks the cheapest
  `Z`-preserving coset representative per output qubit and reduces maximal
  one-qubit runs to shortest words over a declared `{H, S, S_DAG}` set. The
  correction is not confined to the rotation layer — it lowers `CX` at every
  `k > 1` as well (H₄ `k = 4`: 3 688 → 3 277) — so both the v3 records and their
  frozen v2 projections were regenerated, and no cost number published before
  that regeneration is comparable with one published after it. Settings counts
  are unchanged at every rung (H₄ `913/647/238/64`), because the grouping rule
  never moved.
- **P4.** The weight advantage attenuates from the Hamiltonian multiset to the
  element-operator universe, because the latter is built from products `A_i†HA_j`
  and JW's Z-strings cancel structurally in products. *Falsifier:* equal ratios on
  both multisets — in which case Hamiltonian weight is a sufficient proxy and the
  extra bookkeeping can be dropped.
- **P5.** `G(k=n)` agrees between mappings within tie-break noise, and the mapping's
  `C_time` gap closes monotonically as `k → n`. *Falsifier:* a gap at `k = n` beyond
  the §6.2 bound, meaning either the coloring is not comparing isomorphic graphs or
  diagonalizer synthesis is the dominant effect.
- **P6.** The `+2q` arms beat their unreduced parents on every cost column, and by
  more than the `JW → BK` difference at fixed `n`. *Falsifier:* reduction worth less
  than the encoding change, which would invert the plan's advice on where to spend
  effort.

*The claim this phase may make.* Not "BK is shallower". Either "the mapping changes
measurement cost by `X%` at protocol `k` on device card `D`, while leaving the
subspace and `W` provably unchanged", or "it does not, at the sizes measured".

*Completed fixed-QWC result.* QR2 passes for every arm on H₄, BeH₂, equilibrium
H₂O CAS(8e,6o), and the open 2×2 Hubbard model. H₄, BeH₂, and Hubbard use the same
established largest-degree greedy; their `JW/parity/parity+2q/BK/BK+2q` setting
counts are `913/533/351/615/403`, `353/41/27/41/27`, and
`1406/798/457/907/478`. H₂O alone needs the scalable full-basis-seeded first-fit
cover and records `24334/17118/9908/18108/8759`; those counts are descriptive upper
bounds and are excluded from the cross-instance QWC verdict.

The corrected like-for-like QR3 result is negative: on matched-greedy QWC settings,
the maximum mapping spread is `13.074×` versus a minimum instance spread of
`3.983×`; on algorithm-independent mean word weight the corresponding factors are
`1.571×` and `1.489×`. Mapping spread is therefore not smaller on either independent
metric. The three fixed-shot device-card rows are not counted as corroboration:
QWC has `N_2q=D_2q=0`, so those times are derived projections of setting count and
one-qubit rotations. Accuracy-matched QR3 still abstains because only BeH₂ clears
the 1.6 mHa exact bias floor in all five arms. **That is no longer the case.** The
`h4_converged` bank clears the same floor at `0.766 mHa` on all five arms, so this
record's `qr3.accuracy_matched` now reads `eligible_for_cross_instance_comparison`
over `[h4_converged, beh2]`. Its prices remain asymptotic; R1's nonlinear
exact-oracle search was not approximated for this record, and at that tier the
second instance is deferred on resolution rather than accuracy (§5, Phase R3).

**R3 — the protocol axis and `k*`.** The `mapping × k` grid under the R1 cost model,
not a new protocol; `k*` as defined in §6.7. *Gate:* margins reported; regions, not
integers, wherever the margin sits inside the shot-search uncertainty.

*Implementation state.* The protocol axis is a library primitive:
`clifford_qc/measurement/block_commuting.py` owns the dyadic block-commuting
compatibility rule and its largest-conflict-degree greedy, and both R1 producers
(`run_clifford_hierarchy.py`, `run_exact_shot_search.py`) consume it rather than
carrying private copies. `tests/test_block_commuting.py` pins the packed
predicate against a letter oracle and pins the two endpoints the family
interpolates — `k = 1` is qubit-wise commutation, `k >= n` is full commutation —
so the claim that the hierarchy spans both protocols is tested rather than
asserted. The frozen hierarchy and shot-search records regenerate digit for
digit across the extraction, which is what makes it a refactor. The mapping arms
of R2b still group at fixed QWC; joining the two axes is R3's own work.

*The two axes are now joined, in two records.* `protocol_axis.json` carries the
structural half — `G(k)`, coverage, synthesis resources at uniform shots — and
`protocol_cost.json` carries the accuracy-matched half, running R1's exact-tier
search once per `(arm, k)` cell and pricing each confirmed crossing as an
interval rather than a point. Splitting them is deliberate: they sit at
different evidence tiers (`structural` against `exact`), they cover different
system sets (both against BeH₂ only), and they differ by two orders of magnitude
in cost, so folding the cheap structural grid into the expensive search would
make it unrunnable for the question it actually answers. What ties them together
is a gate rather than a convention — every cell of the cost record reproduces
the structural grid's setting count, and the producer refuses to write a record
where it does not.

*The gate, discharged.* Margins are reported on both sides of every crossing and
on every `k*`; and regions rather than integers is not a fallback here but the
uniform outcome — all thirty `k*` determinations are regions. The one thing this
phase does **not** deliver is a cross-instance accuracy-matched mapping verdict:
QR3 needs two priced instances, so the record abstains explicitly rather than
reporting BeH₂'s mapping spread as though it answered the question.

*Why the second instance is still missing, now that a qualifying bank exists.*
The obstacle was read as H₄'s bias floor, and that reading was incomplete. The
budget-8 bank misses the target because A-CASE was stopped at eight additions
while still lowering; carrying the same greedy on the same Hartree–Fock
reference to its own threshold reaches `M=15`, `W=7927` and `0.766 mHa`, which
clears `1.6 mHa` with a factor of two to spare and costs almost exactly the same
to measure — `913/533/351/615/405` settings at `k=1` against the frozen bank's
`913/533/351/615/403`. That bank ships as `h4_converged`, and at the
*asymptotic* tier it does what it was built for: `mapping_axis.json` now records
`eligible_for_cross_instance_comparison` where it recorded
`insufficient_eligible_instances`.

At the **exact** tier it is **right-censored** by a second gate, and one this
plan had not separated from the first. The nonlinear shot search resolves a crossing only
inside `SEARCH_ENDPOINTS`, and this bank's crossings do not land there. Its word
universe is `7926` against BeH₂'s `1814` on the full-width arms and `2047`
against `511` on the `+2q` arms; four times the words reconstructed from the
same shots is four times the pencil variance, which moves the confirmed
crossings from BeH₂'s `4096–16384` up to `16384–65536` — against a grid whose
last point *is* `65536`. A reduced-replica scoping probe left 12 of 20
single-assignment cells unresolved and put three of the eight that did confirm
on the final grid point. Extrapolated to the headline replica counts that
attempt costs about a day of four-core time and still returns mostly unpriced
cells.

So `protocol_cost.json` carries an explicit `cost_layer_scope`: the structural
grid's three banks are partitioned into the ones it prices and the ones it
defers, each deferral with its reason, and `check_protocol_cost.py` fails a
record where a system is merely absent. **The correction this forces is worth
stating plainly: clearing the bias floor is necessary for a price and is not
sufficient.** A bank must also be resolvable at the declared endpoint grid, and
that is a condition on `W`, which §6.2 says the linear encoding family cannot
move. Lifting it means endpoints above `65536`, which `exact_shot_search.json`
shares — a change to R1 and R3 together, not a scope change to R3.

**R3S — the screen that tests whether that change is necessary.** The deferral
above rests on a premise it never examines: that a wider grid is the only route
to a second priced instance. The QR3b record already carries the instrument that
checks it — a word-universe ceiling, and the admission
`screen_would_have_rejected_before_probe: true`, meaning the gate was evaluated
*after* the probe had spent all forty of its cells.
`run_priceability_screen.py` evaluates it first, over every declared candidate,
in deterministic double-precision arithmetic — dense eigensolves, reproducible
run to run on fixed BLAS threading, but floating-point and compared to tolerance
by the checker rather than exact.

The premise does not survive. The frozen selection rule runs the A-CASE greedy
to its own predicted-lowering threshold; that rule is accuracy-maximizing while
the exact-tier price is resolution-limited, and on LiH the two conflict badly.
The greedy spends four orders of magnitude of bias headroom — `0.370` mHa at
`M = 2` down to `0.0002` mHa at `M = 13` — to buy `5.4×` the word universe,
`1439 → 7740`. Stopping instead at the smallest prefix clearing the target with
a declared margin puts LiH at `M = 2`, bias `0.370` mHa, `W = 1439`: *below*
BeH₂'s `1814`, the one bank this repository has ever priced inside the grid. So
QR3b rejected a stopping rule and not an instance, and a second priceable
candidate is reachable with `SEARCH_ENDPOINTS` untouched.

*The margin has a reason, and the reason has a limit.* The shot search's pass
rule bounds replica RMSE, and RMSE combines bank bias with sampling scatter in
quadrature, so a bank sitting at the target has no allowance left for shot noise
at any endpoint. At margin `3` the bias takes `0.533` mHa, which is `(1/3)² =
11.1%` of the *MSE* budget, and the statistical *RMSE* allowance falls from
`1.600` to `sqrt(1.6² − 0.533²) = 1.5085` mHa — a `5.7%` reduction. Those are two
different fractions and the smaller one is not a share of a shot budget consumed.
What the argument does is motivate having a margin; what it does not do is pick
`3` out of `2` or `5`.

*So the margin factor is labelled declared, not preregistered.* This rule and the
first result it produces enter the repository in the same commit, which makes the
LiH admission exploratory evidence for the rule rather than a test of it, and the
record says so in `margin_factor_status`. Two things limit what that costs. Every
candidate carries a `margin_sensitivity` range re-derived from its own walked
rows: LiH is admitted for every margin from `1` to about `4.33`, and the three
rejected candidates stay rejected at every margin at or above `1`, so no verdict
in this record turns on the number. And the rule is frozen from this commit — the
preregistered use is the next candidate screened under it.

*Two things the screen is not.* It is not a price, and the ceiling is not a
necessary condition: it is an operational admission threshold calibrated on a
single priced bank, while coefficient magnitudes, grouping, estimator variance
and pencil conditioning all bear on whether a bank resolves — a rank-2 pencil may
condition differently from BeH₂'s rank-5. Candidates here are admitted or
rejected *under this declared screen*, and deciding the rest is what the 2+2
probe is for; this phase authorizes one rather than replacing it. And the margin
rule is not licence to pick a prefix that is cheap to measure, which is the
cost-direction selection QR3b's own gate forbids: it is a function of the
accuracy target alone, evaluated on the source-side bias the linear encoding
family leaves invariant, and the record carries the per-arm bias agreement at the
chosen prefix so the checker confirms the choice was encoding-blind instead of
taking the declaration on trust.

*Result on the other four candidates.* `h4_converged` is rejected under both
rules and stays deferred: its full-width arms pass the ceiling at `M = 3` and its
bias never reaches the margin anywhere in the frozen ordering — `0.766` mHa at
`M = 15` against an admissible `0.533`. H₂O CAS(8e,6o) and the 2×2 Hubbard model
fail the same ceiling, at `143116` and `5536` intrinsic words. BeH₂ is admitted,
and admitted at `M = 3` rather than `M = 5` — the same bank at `1223` words
instead of `1814`, which is a cheaper repricing of the instance already priced,
not a new one. The binding-arm rule is what carries `h4_converged`: its two `+2q`
arms sit at `2047`, one word under the gate, so a screen reading any single
reduced arm would have admitted a bank the frozen grid has already failed to
resolve.

**R4 — contextual subspace as comparator, then preconditioner.** Arms, at matched
accuracy target and matched candidate family: full QSE, CS-QSE, A-CASE, CS + A-CASE.

**R4a was preregistered without an outcome, then executed once.**
`benchmarks/configs/r4a_contextual_interaction.json` freezes the BeH₂/JW bank,
the four ordered arms, the reference-conditioned stabilizer constructor, its
canonical coefficient/code tie-break, the fixed-qubit ladder, the structural
bias/word-universe gate, and the conditional exact-tier sampling protocol.
`benchmarks/check_r4a_preregistration.py` validates that declaration without
selecting a contextual rung or running the structural or sampled experiment.
That commit authorized one later structural execution only. The separately
committed result in `benchmarks/reference_results/r4a_contextual_screen.json`
now records that execution, and `benchmarks/check_r4a_contextual_screen.py`
regenerates and gates it.

**The structural result is negative, so R4a stops without sampling.** No rung in
the frozen one-through-seven-qubit ladder clears both gates for all four arms.
Full QSE clears the `0.533` mHa bias margin at `0.0033` mHa but exceeds the
`2048`-word ceiling with `14350` words. Standalone ACASE clears both gates at
`0.0695` mHa and `1223` words. The contextual arms clear the word ceiling from
rung two onward but miss even the original `1.6` mHa accuracy target: their
smallest observed bias floors are `5.345` mHa for CS-QSE and `5.899` mHa for
CS+ACASE. Therefore `selected_contextual_rung` is null, sampled execution stays
unauthorized, and QR5 remains undetermined. Full QSE's ceiling failure is an
operational screen rejection, not evidence that its finite-cost estimate is
infinite.

*Report the bias floor, not only the compression.* The contextual restriction is an
approximation whose error is not variationally controlled by the restricted solve; a
rung where the restriction's own floor exceeds the accuracy target cannot be
compared to an unrestricted arm on cost, because it never reaches the target. Every
CS arm reports the restricted-space exact energy against the same exact reference
used elsewhere on that rung, and the qubits removed. An arm that cannot reach `ε`
reports `C(ε) = ∞` and its floor.

*The interaction question, made measurable.* The test compares the joint gain against
the two **standalone** gains, each measured against the same full-QSE baseline `C₀`:

```
r_CS    = C₀ / C_CS            (contextual restriction alone)
r_A     = C₀ / C_ACASE         (adaptive selection alone)
r_joint = C₀ / C_CS+ACASE      (both)

Δ = log r_joint − log r_CS − log r_A
```

with the shot-search uncertainty propagated into `Δ`. The four arms this phase already
runs supply all three ratios, so nothing extra is measured — but the marginals must be
the standalone ones. Chaining the *conditional* gain instead
(`r_joint = r_CS · C_CS/C_CS+ACASE`) is an algebraic identity that telescopes, so its
"deviation from the product" is zero by construction and classifies nothing. The
conditional gain `r_joint/r_CS` is still worth reporting — it is A-CASE's marginal
value *given* CS — and the test is equivalently whether it differs from the standalone
`r_A`.

Four pre-registered outcomes, in decreasing order of interest:

| outcome | signature | what follows |
|---|---|---|
| **complementary** | `Δ > 0` beyond uncertainty (super-multiplicative) | the two compressions attack different structure; CS-preconditioned A-CASE is worth building |
| **multiplicative** | `Δ ≈ 0` | the gains compose but do not reinforce; build it only if the engineering is cheap |
| **redundant** | `r_joint ≈ max(r_CS, r_A)` | CS removes what A-CASE would have pruned; report it and do not build the preconditioner |
| **antagonistic** | `r_joint < max(r_CS, r_A)` | restriction removes directions adaptive selection needed; the interesting negative, and it belongs in the paper |

The redundancy boundary is stated on the ratios rather than on `Δ` because `Δ ≈ 0` and
`r_joint ≈ max` are different statements whenever one marginal is near 1, and it is the
second that means "one mechanism did all the work".

*Gate before any solver change.* Complementarity must appear in `C(ε)`, not only in
`M` or `W`. A drop in `M` that leaves the accuracy-matched cost flat is exactly the
reading §6 exists to block — the ladder has already produced one such case (§7).

*Every arm passes the screen before it is sampled.* R3S made the failure mode
concrete: the exact-tier price is resolution-limited, and the full-QSE baseline
`C₀` is the widest of the four arms, so it is the one most likely to censor at
the frozen grid — while `Δ` needs all three ratios finite. So R4a declares, per
arm and before any shot is spent: the stopping rule (the margin rule R3S froze,
so that "matched candidate family" means matched *rule*, not matched `M`), the
`(bias, binding W)` pair from `run_priceability_screen.py`, and the arm's
admission under the declared ceiling. A censored arm does not report
`C(ε) = ∞` — that label is reserved for a bias floor above the target — it
reports `right_censored` with its last confirmed failing endpoint, and `Δ` is
then recorded as **undetermined**, with `r_CS` and `r_A` given as one-sided
bounds where the censoring direction allows. That fifth outcome is declared here
so it cannot later be reclassified as one of the four in the table.

**One primitive serves R2 and R4.** BK/parity change-of-encoding, `Z₂` tapering, and
contextual-subspace restriction are the same two-step object: a Clifford rotation,
then fixing a set of commuting stabilizer qubits to `±1`.

| operation | Clifford part | fixing part |
|---|---|---|
| JW → BK / parity | CNOT network of the encoding change | none |
| BK/parity two-qubit reduction | same | fix the two symmetry qubits to `±1` |
| `Z₂` symmetry tapering | Clifford mapping each symmetry generator to a single `Z` | fix those qubits to the reference's eigenvalue |
| contextual-subspace restriction | Clifford rotations of the noncontextual stabilizers | fix the stabilizer qubits |

So both phases share `clifford_qc/subspace/restriction.py` — **shipped** — exposing a
`Restriction` carrying `(clifford, fixed_qubits, signs)` that transports Hamiltonian,
reference, generator pool, and observables together through
`Restriction.transport`, returning a `RestrictedProblem` that also reports per-generator
sector leakage. R4a's shipped `subspace/contextual.py` adds the frozen deterministic
selection and compilation boundary, while `project_contextual_problem` records the
non-identity Hamiltonian Hilbert–Schmidt fraction removed by an artificial contextual
stabilizer, invariant to scalar energy shifts;
it does not weaken exact `Restriction.transport`. `CliffordMap.conjugate`
(`bridges/stim_bridge.py`) provides the word
action; the module adds the fixing step and the bookkeeping that keeps the four objects
consistent. The three oracle checks are in `tests/test_restriction.py`:
`⟨HF_B|H_B|HF_B⟩ = ⟨HF_JW|H_JW|HF_JW⟩`, spectrum equality against the dense sector
block, and word-multiset bijection under a pure encoding change.

Two contracts the implementation fixes, both of which are the silent-failure mode this
class of code has:

- **A state is checked into its sector, never rescaled into it.** The input is first
  required to be Hermitian with unit trace. For an in-sector density multivector the
  `W` and `W·Z_q` terms merge under the fix, so the restricted trace returns to 1 on
  its own; `Restriction.state` therefore asserts that rather than normalizing,
  because a shortfall is evidence the declared signs are wrong.
- **Commuting is required only where it is meant.** The Hamiltonian and the symmetry
  generators transport with `require_commuting=True` and raise on any anticommuting
  term; candidate generators transport without it and are projected term-wise, which
  is the §3.5C rule that an operator moving the state to an orthogonal sector vanishes.
  A generator annihilated this way is reported through `annihilated_indices`, not
  dropped silently.

`restricted_sector_operators` transports `(N, S_z)` through the same restriction, which
is the first of the two choices the hidden-cost gate below demands.

**The hidden cost these phases must price.** The package identifies the
computational basis with the occupation-number basis in several places that are
correct **only under Jordan–Wigner**: `fermion.py` `total_number_op`/`total_sz_op`
build `n_j = (I − Z_j)/2` directly and `subspace/symmetry.py` `sector_operators`
consumes them, so `sector_leakage`, `reference_sector_leakage`,
`infer_reference_sector`, and `subspace_sector_certificate` are JW-specific;
`backends/sector_statevector.py` groups by X-mask over occupation strings; and
`subspace/qsci.py` and `subspace/selected_ci.py` read bitstrings as determinants.
Under BK or parity these are wrong by construction, and wrong in the quiet way —
they return numbers. Phase 7's Kitaev incident (§7) records what that failure mode
costs. R2 is therefore gated on one of two choices, declared per arm in the record:
either the sector layer is transported through the `Restriction` (the number
operator's image is still diagonal, so this is mechanical but must be tested), or
the mapping arm runs with the sector filter **off** and says so. No mapping arm runs
with a JW sector filter silently applied to non-JW operators.

---

## 6. Resource accounting (equal partner to basis size)

A 20-dimensional basis is not compact if its projected entries carry millions of
words. The compactness question is not "is `M` small?" but: **does the adaptive
basis stay small while its projected operator bank stays measurably smaller than
competing QSE/Krylov constructions — at a cost a device would actually pay?**

The completed PRD suite sharpens that question: Davidson already supplies the
strongest exact compact basis in the matched study, while complete-bank QWC
measurement fails to convert that compactness into finite-shot energy accuracy at
up to one million aggregate shots. Resource accounting therefore evaluates the
WISE measurement layer around a frozen basis; it is not evidence that the basis can
be selected on hardware or that the resulting energy estimate is certified.

### 6.1 Cost is only meaningful at a fixed accuracy, on a declared evidence tier

Phase 4R settles the first half on this codebase: at 104 000 setting-shots on the
frozen four-qubit TFIM bank, changing nothing but the estimator and the rank rule
moved the RMSE from `1786` to `4.86` mHa. A cost quoted at fixed shots is therefore a
statement about the estimator, not about the hardware, and it can be moved by two
orders of magnitude without touching a circuit. So `N_g` is never an input: it is the
output of a shot-to-target search, and two protocols are compared at equal accuracy or
not at all.

The second half is which accuracy statement the target is made against, and here the
plan has to obey its own certificate contract. **There is no finite-sample certificate
on the Ritz energy in this codebase.** Phase 4C certifies a candidate's residual
coupling with a frozen Ritz pair, conditional on the construction batch, and says
explicitly that this is not a bound on the energy; Phase 4R's `solve_selected_rank` is
a selection rule whose returned value is not a variational bound; and R1 forbids
touching the solver or the certificate. A cost defined through a *certified* energy
interval would therefore be `∞` for every arm, and R1–R4 would produce no comparisons
at all.

`C(ε)` is consequently defined on a **declared evidence tier**, using the labels the
package already carries (`exact`, `asymptotic`, `heuristic`, `finite_sample`), and the
tier is a required field of every cost row:

| tier | `ε` measured as | available today | used by |
|---|---|---|---|
| `exact` (default for R1–R4) | replica RMSE of the nonlinear estimate's absolute error against the rung's exact reference, which §7.3 supplies for every rung but HCl | yes | the accuracy-matched shot search in R1–R4 |
| `asymptotic` | half-width of the delta-method Ritz interval at declared nominal coverage | yes, uncertified | reported beside the `exact` tier as the estimator's own view |
| `structural` | no accuracy claim: counts, biases against an exact reference, and synthesis resources in deterministic double-precision arithmetic, nothing sampled | yes | the R3 structural layer (`protocol_axis.json`) and the R3S screen (`priceability_screen.json`); a structural record may authorize or withhold a sampled run and may not price one. The QR3b and R3b probes sample, so they are `exact`-tier evidence with a `scope_decision_only` role: they resolve or fail to resolve a crossing without pricing it |
| `finite_sample` | half-width of a finite-sample Ritz-energy certificate | **no** | nothing, until such a certificate exists |

The default tier is an **oracle** target: it uses the exact answer, which a device run
would not have. That is legitimate for comparing *protocols and mappings* on frozen
benchmark banks, which is all R1–R4 do — the oracle enters identically in every arm, so
it cannot favour one — and it is illegitimate as a claim about what a hardware run
would cost. Every row says which tier produced it, and no row mixes tiers.

Promoting the cost model to a certified tier is a *prerequisite*, not a knob: it needs
a finite-sample Ritz-energy certificate, which is the anytime-valid confidence-sequence
work Phase 4R names and Phase 15's second moments would support. Until that lands, no
certified `C(ε)` may be published, and `k*` (§6.7) is an oracle-accuracy statement.

### 6.2 What the fermion mapping can and cannot move

For the linear (encoding-matrix) family — Jordan–Wigner, parity, Bravyi–Kitaev,
segment codes — two encodings differ by an invertible `GF(2)` change of basis on
occupation vectors, realized on qubits by a **CNOT network**. A CNOT network is
Clifford, so `H_B = U H_JW U†` with `U` Clifford, and conjugation by a Clifford is an
algebra automorphism mapping Pauli words bijectively to Pauli words up to sign:

| quantity | under `U · U†` | why |
|---|---|---|
| spectrum, Ritz values, `M`, rank, `κ_S` | **invariant** | same operator, similarity transform |
| `W`, `S_A`, `S_H`, `nnz` | **invariant** | word-to-word bijection, products map to products |
| per-word variance `1 - μ_w²` | **invariant** | `μ_w` invariant up to the conjugation sign |
| full-commutation conflict graph (`k = n`) | **isomorphic** | commutation is preserved |
| Pauli weight distribution `w̄, w₅₀, w₉₀, w_max` | **variant** | weight is not Clifford-invariant |
| QWC compatibility, `G(k)` for `k < n` | **variant** | qubit-wise commutation is basis-dependent |
| single-qubit rotation count, CX count, two-qubit depth | **variant** | synthesis depends on the tableau |

So `W_JW = W_BK` **exactly**, and a measured difference in `W`, `M`, `κ_S`, or energy
between mappings is a bug in the transformation, the reference, or the generator
pool — not a finding. Two riders, both of which go in the record:

- **The greedy coloring is not canonical.** At `k = n` the conflict graph is
  isomorphic, so the chromatic number is invariant, but the largest-degree greedy
  partition can differ by tie-breaking. The check compares degree-sequence and
  component invariants of the conflict graph, and reports `|G_JW − G_BK|` at `k = n`
  as tie noise with its own tolerance.
- **Diagonalizer cost is not invariant even where the grouping is.** If `V_g`
  diagonalizes a group, `V_g U†` diagonalizes its image, so
  `cost_BK ≤ cost_JW + cost(U)` — a bound, not an equality, because synthesis starts
  from the image tableau rather than composing. A measured `cost_BK` far above the
  bound means the synthesizer, not the mapping, is what is being measured.

**Qubit reduction is a different operation.** The literature's BK advantage is largely
the two-qubit reduction, which is conjugation **plus fixing** stabilizer qubits and
deleting them. Deletion is where `W` can genuinely fall (distinct words collide on
fewer qubits), where `n` falls, and where every downstream cost falls with it. Hence
the separate `+2q` arms in R2: comparing JW at `n` against reduced BK at `n−2` and
attributing the difference to "the mapping" would publish a confound.

### 6.3 Fewer settings is not automatically cheaper

**Pooling coverage.** Under the pooled estimator a word is read by every compatible
setting, so at a fixed *total* budget `N_total = G·N_g` the effective shots on word
`w` are `f_w · N_total` with coverage fraction `f_w = m_w / G` (`m_w` = settings that
record `w`). Coarser protocols raise `W/G` — the frozen H₄ record goes `8.07 → 115.17`
from `k = 1` to `k = 8` — and `f_w` moves with it in a direction no one has measured,
because the frozen hierarchy predates the pooled estimator. The protocol trade must
be re-measured with `pooling='shots'` before any `k*` is claimed.

**Fidelity.** A depth-versus-time model with no error model monotonically prefers the
deepest protocol, because it only ever removes state preparations. Depth actually
enters through infidelity, and protocols whose damping exceeds a declared floor are
**inadmissible**, not merely expensive.

### 6.4 The cost model

**Device card — a declared, versioned artifact.** No scalar cost may be printed
without one. Schema `clifford_qc.device_card.v1`, stored beside the benchmark configs
and stamped into every record that consumes it:

```json
{
  "schema": "clifford_qc.device_card.v1",
  "name": "logical-alltoall",
  "t_prep_us": 0.0, "t_1q_us": 0.05, "t_2q_us": 0.3,
  "t_readout_us": 1.0, "t_reset_us": 1.0,
  "eps_1q": 0.0, "eps_2q": 0.0, "eps_readout": 0.0,
  "connectivity": "all-to-all",
  "routing": false
}
```

The all-zero-error, all-to-all card reproduces the current logical model exactly,
which is what makes the extension checkable (R1 gate 1).

**Time cost.** For each setting `g` the synthesized diagonalizer supplies `N_1q,g`,
`N_2q,g`, `D_1q,g`, `D_2q,g`; the readout covers all `n` qubits regardless of group:

```
C_time(ε) = Σ_g N_g(ε) · [ t_prep + t_1q·D_1q,g + t_2q·D_2q,g + t_ro + t_reset ]
```

*The A-CASE-specific term.* A-CASE measures every setting on **one** reference state,
so `t_prep` factors out: `C_time = N_total·t_prep + Σ_g N_g·(measurement + readout +
reset)`. On the current pipeline the reference is a Hartree–Fock determinant — a
computational basis state, prep depth zero up to `X` gates — and under any linear
encoding it stays one, so the prep term is very nearly free. That inverts the usual
VQE accounting in which preparation dominates: the currency the frozen table reports
(`state_preparations_at_uniform_shots`, `7.304 M → 0.512 M` on H₄) is the right
currency only on a device where reset and readout are cheap relative to preparation.
On a superconducting card where `t_ro + t_reset` dominates, `C_time` tracks total
executions and the deep-protocol advantage is near its maximum; on a trapped-ion card
with slow gates and fast state prep, near its minimum.

**Fidelity and admissibility.** With a depolarizing surrogate,
`F_g = (1-eps_1q)^{N_1q,g}·(1-eps_2q)^{N_2q,g}·(1-eps_ro)^n`. An unbiased estimator
built on damped readings inflates variance by `F_g^{-2}`, so `N_g(ε) ∝ F_g^{-2}` and
the record reports effective as well as raw shots. A setting is inadmissible when
`F_g` falls below the declared floor (default `0.5`, recorded per run); runs report
the admissible `k` set before reporting `k*`. This is deliberately a surrogate, not a
noise simulation: it exists so the cost model cannot recommend a protocol a device
could not execute, and claiming a calibrated error prediction from it would be exactly
the overreach §14 forbids.

**What may be printed.** Default output is a **break-even surface** — the admissible
region and the `k*` boundary over the `(t_2q/(t_ro+t_reset), ε_2q)` plane with the
accuracy target fixed. The frozen table's `c_CX/c_prep` column is its one-parameter
version and is kept. A scalar `C_time(ε)` may be printed **only** under a named device
card, with the card's name and hash in the row; a cost with no card is a schema error,
not a default.

### 6.5 The metric ledger

Every run records `M`; `W = |⋃_ij supp(O_ij^H) ∪ supp(O_ij^S)|`;
`S_A = max_i |supp(A_i)|` and `S_H = max_ij |supp(A_i†HA_j)|`; `r_S` retained rank at
threshold `τ`; `κ_S` retained condition number; bank build time and peak memory;
reused vs. newly introduced words per accepted generator; QWC group count. Added by
this section, with the axis on which each is measured:

| metric | measured on |
|---|---|
| `w̄, w₅₀, w₉₀, w_max` | three distinct multisets, reported separately: (a) Hamiltonian words, (b) the A-CASE element-operator universe `⋃ supp(A_i†HA_j)`, (c) per-setting support. (b) is what A-CASE actually pays for, and it is *not* predicted by (a) — see R2-P4 |
| `G(k)`, `W/G`, coverage `f_w` (mean, min) | per protocol rung |
| `N_1q`, `N_2q`, `D_1q`, `D_2q` (mean, max) | per setting, summed per sweep |
| `F_g`, `N_eff`, admissibility | per setting, under the device card |
| `N_g(ε)`, `C_time(ε)`, `ε`, evidence tier, coverage, abstention | per arm, at the accuracy target (§6.1); the tier is required, never defaulted |
| encoding, reduction, taper qubits, `n_eff` | per mapping arm |

Each projected observable's word universe, and the words it adds beyond what `(S, H)`
already require, enter this accounting too (§8).

### 6.6 What the measurement layer already reports

`benchmarks/run_clifford_hierarchy.py` implements the dyadic block-commuting
hierarchy: for `n = 2^L` qubits and block size `k = 2^ℓ`, partition the register into
contiguous blocks and call two words `k`-compatible when their restrictions commute
inside every block. The endpoints are QWC at `k = 1` and full Pauli commutation at
`k = n`; each group is diagonalized by block-local stabilizer Clifford circuits
synthesized with stim, and every word — not only an independent stabilizer basis — is
verified to map to a `Z`-only word. The hierarchy leaves the subspace and `W`
unchanged while trading fewer settings for more logical CX gates and depth.

Frozen (`reference_results/clifford_hierarchy_{h4,beh2}.json`, schema
`clifford_qc.clifford_measurement_hierarchy.v3`, 8000 shots per setting; the
schema-v2 controls these columns are gated against sit beside them with the
`_v2.json` suffix):

| system | `k` | `G` | `W/G` | `N_CX` | mean `D_CX` | max `D_CX` | preps (10⁶) | `c_CX/c_prep` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H₄ (`M=9`, `W=7 371`) | 1 (QWC) | 913 | 8.07 | 0 | 0.00 | 0 | 7.304 | — |
| | 2 | 647 | 11.39 | 3185 | 2.55 | 4 | 5.176 | 0.084 |
| | 4 | 238 | 30.97 | 3688 | 8.90 | 17 | 1.904 | 0.183 |
| | 8 (full) | 64 | 115.17 | 2006 | 25.72 | 42 | 0.512 | 0.423 |
| BeH₂ (`M=5`, `W=1 815`) | 1 (QWC) | 353 | 5.14 | 0 | 0.00 | 0 | 2.824 | — |
| | 2 | 41 | 44.27 | 168 | 1.78 | 2 | 0.328 | 1.857 |
| | 4 | 26 | 69.81 | 234 | 5.12 | 10 | 0.208 | 1.397 |
| | 8 (full) | 14 | 129.64 | 332 | 19.00 | 27 | 0.112 | 1.021 |

This is a tunable logical trade, not a preferred block size, and the instance spread
is the reason: the `k = 2` break-even ratio is `0.084` on H₄ and `1.857` on BeH₂ — 22×
under an identical logical model, before any device enters. Connectivity, routing,
device noise, and error mitigation are excluded.

**Two word-universe conventions are in the tree, and they differ by one.**
`mapping_axis.json` counts the identity word (`word_universe_after`: BeH₂
full-width `1815`); `protocol_axis.json`, the QR3b probe and the R3S screen do
not (`1814`). Every gate calibrated on a word count — the `2048` ceiling above
all — was calibrated in the second convention, and the identity is the one word
a shot budget never buys, so the second is canonical for gates. A producer that
compares counts across the two records is off by exactly one on every arm and
learns nothing from it. Until `mapping_axis` is regenerated under the canonical
convention, every new consumer states which convention it counts in, as
`priceability_screen.json` does through `word_universe_convention`.

### 6.7 `k*`, defined

```
k*(ε, device card, instance) = argmin_{k admissible} C_time(ε; k)
```

— the smallest-cost *admissible* block size at a fixed accuracy on a named device card
and a declared evidence tier (§6.1, `exact` by default), reported with the argmin's
margin over its neighbours. It is not a
function of `W` alone, it is not an instance-independent constant, and it is reported
as a region over the break-even plane whenever the margin is within the estimator's
own uncertainty.

*How the region is computed, once there is a measured `C_time`.* The uncertainty
that matters is the shot search's own resolution. A confirmed crossing brackets
the shot count in `(confirmed_fail, confirmed_pass]`, so the rung's cost lies in
`(C_time(confirmed_fail), C_time(confirmed_pass)]`, and a crossing R1 flags
environment-marginal widens that interval by one grid step on the side that could
move — a marginal pass could fail in another environment and push the count up, a
marginal failure could pass and pull it down. `k*` is then every admissible rung
whose interval reaches the smallest upper bound, and the point argmin of the
reported counts is recorded beside it. The point is always inside its own region,
since `C_time(pass) ≤ C_time(upper)` rung by rung. `run_protocol_cost.py` builds
this and `check_protocol_cost.py` re-derives it; on the BeH₂ grid every one of the
thirty determinations comes out a region rather than a single rung, so nothing in
this project currently publishes a `k*` integer.

### 6.8 Three compression classes stay separate

In every table: **Hilbert-space compression** (qubits, sector, restriction);
**operator-pool compression** (`M`, candidates evaluated); **measurement/hardware
compression** (`W`, `G`, weight, gates, depth, `C_time`). Contextual-subspace methods
attack the first two; word reuse, pooling, and grouping attack the third. A table that
sums them into one number cannot show which mechanism paid.

The GA structural preconditioner (§3.5) attacks the first two as well, which is
precisely why Phase G3's deliverable is the decomposition

```text
C_HW^raw  →  C_HW^GA  →  C_HW^GA+PRD  →  C_HW^GA+PRD+WISE
```

rather than a single ratio: GA, PRD, and WISE act on different classes, and only a
per-arrow margin at fixed accuracy shows which one paid. Reporting the chain in
counts instead of accuracy-matched cost would reproduce exactly the error §6.1
forbids — and P2 already reports the count reductions on this bank (§1.2(1)).

The anticommuting cliques of §3.6 are the sharpest case of why the classes stay
separate, because there one algebraic fact — `(Σ_i c_i A_i)² = 1` — is claimed in
two of them. In the **first** class a clique would become one more fixed
stabilizer, so one more qubit leaves the register; in the **third** it becomes one
measurement setting of a fixed linear functional, at up to `m − 1` Pauli rotations
of synthesised cost. The two are at different readiness: the third is available
once a cover and its synthesis are priced, while the first is conditional on a
non-Clifford transport primitive the package does not have and on contextual
admission conditions that must be rechecked after the rotation (§5, Phase 19).
They are not the same gain, they do not add, and a table reporting "clique
partitioning saved X" without naming the class has already made the error this
subsection exists to block.

---

## 7. Validation ladder and benchmark inventory

### 7.1 The ladder (Phase 7 — done)

`benchmarks/run_acase_ladder.py` and `summarize_ladder.py`: H₂ → LiH → H₄ (equilibrium
and stretched) → stretched H₂O CAS(4e,4o) → H₂O CAS(8e,6o) → 2×2/2×3 Hubbard → one
Kitaev cluster with link correlations. Comparison set: the reference determinant,
exact diagonalization, plain QSE, fixed Krylov, generator-coordinate-style fixed
subspaces, ADAPT-VQE (exact and finite-shot), A-CASE (exact and finite-shot
certified). Every row carries the §6 metrics, shots, circuits, and abstentions beside
the energy, plus the `evidence` label that says what kind of number it is.

Three conventions decide what the table means:

- *The error column is measured against the reference's own symmetry sector.* A-CASE
  never leaves the sector its reference lives in, and a grand-canonical Hubbard
  cluster's global minimum sits at a different filling — scoring against the global
  ground state would charge every method for a particle-number difference none of them
  can cross.
- *The generator-coordinate arm takes an even stride through the candidate family, not
  its prefix.* The excitation family lists singles first and every single is
  Brillouin-dead on a Hartree–Fock reference, so a prefix of eight reproduces the
  reference energy to machine precision. That is a fact about the ordering, not about
  non-adaptive subspaces. With the stride, the arm clears HF on H₄ (−2.1307 against
  −2.1243) and is still beaten.
- *Wide generators fall back to the cyclic contraction.* Above `max_tracked_support`
  the row is solved through `Tr((HA_j)(ρA_i†))` and marks its support columns `n/t`
  rather than reporting guessed ones.

*A configuration error the ladder caught, which would have published as a method
failure.* `sector_leakage` measures `||[A,N]||/||A||` and the same for `S_z` — the
*fermionic* symmetries — and it takes a generator alone, so it cannot know that the
Kitaev cluster is one qubit per site with no Jordan–Wigner transformation behind it
and no particle number to conserve. Applied there, the §4.2 sector filter rejected all
72 candidates (the whole pool leaks at 0.94 or above) and A-CASE reported a basis of
size one sitting at the reference energy, `−4.0` against the cluster's `−4.9624`. Read
off the table that is "A-CASE cannot grow on a frustrated spin cluster." It is nothing
of the kind: with the filter correctly not applied, A-CASE reaches the exact energy to
`7e-13` at `M = 6`. The tolerance is now dropped for models with no fermionic sector
and the row records that it was dropped, so the failure cannot recur silently.

That recovered row is worth reading carefully rather than banking as a win, because it
is exactly the reading §6 exists to block. A-CASE's selected basis is
`I, H¹, H², H³, H⁷, H⁸` — six of the nine Krylov powers sitting in its own candidate
pool, with every `pauli_orbit` and `commutator_response` candidate passed over. So it
did not find a more compact *kind* of basis; it pruned the Krylov basis by a third. And
the resources do not follow the basis size down: the word universe is identical
(`W = 140`, `S_H = 76`, the same words either way) and the conditioning is marginally
*worse* (`κ_S = 3.8e+04` against Krylov's `3.3e+04`).

*The measurement layer, not the shot budget, was what confined certified growth to four
qubits.* The QWC partition is greedy and quadratic in the word universe, and that was
the obvious suspect — but it is memoized on the word set, and the certified loop's
universe is the same index set at every growth step, so it was computed **once**, not
per step, and it was never the binding cost (15 846 words, 1 689 groups: ten seconds of
a batch that took hours). The binding cost was `computational_probabilities`: 11.4 s per
eight-qubit readout, called once per QWC group per batch. A Z-basis readout only sees
the *diagonal* Pauli content of the state, so with `|b⟩⟨b| = Π_j (I + s_j Z_j)/2`,

    p(b) = Σ_z ρ_z (−1)^popcount(z ∧ b),

the Walsh–Hadamard transform of the diagonal coefficients — `O(2^n n)` for all outcomes
at once, against `O(4^n)` word products for building each projector and multiplying it
out. With that, plus holding the shot-independent part of each group's sampling plan
instead of rebuilding it every batch, a warm batch over those 1 689 groups is 0.19 s.
The partition was separately made output-preserving-but-fast: 15 846 words, 77 s →
2.3 s. The lesson is the one §6 keeps making in a different register: the quantity that
looked expensive by inspection was not the quantity that was expensive, and only
measurement distinguished them.

*Certified growth at eight qubits* (`acase_certified_n8`, 32 000 shots per group,
`max_size = 4`):

| rung | M reached | error | shots | outcome |
|---|---|---|---|---|
| h4_equilibrium | 2 | +3.66e-02 | 216 M | abstained |
| h4_stretched | 5 | +4.09e-02 | 461 M | budget reached |
| h2o_cas4e4o_stretched | 4 | +1.03e-01 | 432 M | abstained |
| hubbard_2x2 | 5 | +1.28e+00 | 442 M | budget reached |
| kitaev_2x2 | 2 | +1.20e-01 | 151 M | abstained |

This is the first Q2 evidence above four qubits. Two rungs grow to the full budget
without a single uncertified step; three stop by abstention rather than by growing on a
decision the shots do not support. At 4 000 shots per group — the four-qubit budget —
H₄ abstains immediately, so the certified arm is *shot-limited* at eight qubits rather
than structurally blocked. Every certified energy here sits well above chemical
accuracy, so this establishes that the certificate keeps working at `n = 8`, not that
certified A-CASE is accurate there.

*Measured* (`benchmarks/reference_results/acase_ladder.jsonl`, 72 runs; error in Hartree
against the reference's own sector, `—` where the config does not run that arm):

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

### 7.2 What the ladder does and does not support

- **Q1, against blind selection: supported.** At equal `M = 9` and drawing from the
  *same* candidate family, adaptive selection beats the strided generator-coordinate
  subspace on every rung — in error (by 1.5× on the 2×3 Hubbard up to 17× on
  equilibrium H₄) and in word universe simultaneously (7 371 against 13 646 on H₄;
  5 358 against 27 870 on the 2×3 Hubbard, a 5.2× saving). The `W` column is what makes
  it a resource claim rather than a basis-size claim.
- **Q1, against fixed Krylov: not supported.** Krylov wins the energy on every fermionic
  rung where it ran, at the same `M` and by up to five orders of magnitude (1.0e-08
  against 3.0e-03 on equilibrium H₄). The compactness claim does not survive that
  comparison and should not be advertised as if it did. What Krylov pays is conditioning
  — `κ_S` from 3.3e+04 to 6.6e+10, against A-CASE's `1.0` on every fermionic rung — and
  at eight qubits its generators are too wide to bank at all. The honest summary is a
  trade, not a win: Krylov buys accuracy with an overlap matrix no finite-shot run could
  invert, and A-CASE buys a conditioned, measurable subspace at a worse energy.
- **A-CASE does not beat ADAPT-VQE.** ADAPT is better on stretched H₄ (7.4e-03 against
  3.9e-02), stretched water (1.2e-02 against 5.5e-02), and the 2×2 Hubbard (4.0e-01
  against 8.6e-01), and ties at equilibrium H₄. This reproduces the Phase 3 finding
  rather than overturning it. The one place ADAPT collapses is the Kitaev cluster, where
  its exact gradient is below threshold at the reference and it selects *zero* operators:
  the reference is a stationary point, and a first-order selection rule has nothing to
  see. A-CASE's 2×2 generalized lowering is not a gradient and does grow there.
- **Q4: the span half is answered, the compactness half only on the smaller cluster.**
  With levels 0–3 the singles-and-doubles family saturates 0.86 Ha above the 2×2 sector
  ground energy and 2.2 Ha above the 2×3, and no arm but Krylov comes close. Level 4
  removes the span barrier outright. From the committed record, both arms grown
  adaptively at the same budget:

  | rung | family | M | error | `κ_S` | `W` | level-4 picked | stop |
  |---|---|---|---|---|---|---|---|
  | hubbard_2x2 | levels 0–3 | 23 | +2.55e-01 | 1.00 | 13 665 | 0 | saturated |
  | hubbard_2x2 | levels 0–4 | 26 | **−1.07e-14** | 1.00 | 15 191 | 5 | budget |
  | hubbard_2x3 | levels 0–3 | 26 | +1.09e+00 | 1.00 | 85 264 | 0 | budget |
  | hubbard_2x3 | levels 0–4 | 26 | +1.09e+00 | 1.00 | 85 264 | 0 | budget |

  On the 2×2 that is the sector ground state at machine precision with `M = 26` against a
  **36**-state sector, at `κ_S = 1` and 11 % more words than the saturated levels-0-3
  basis — a compact, well-conditioned, measurable basis rather than a re-derivation of
  full CI. It is the only arm on that rung to reach chemical accuracy at all; fixed
  Krylov gets to `+2.6e-03` and misses, at `κ_S = 9.7e+09`. The five generators selected
  are `afm*E(2<-0)`, `afm*E(1<-3)`, `afm*E(1,2<-0,3)`, `afm_flipped*E(6<-4)` and
  `afm_flipped*E(5<-7)` — the antiferromagnet and its spin-flipped partner, each dressed
  by an excitation. The levels-0-3 arm stops on its own at `M = 23` with predicted
  lowering below threshold: it is not budget-limited, it is out of directions. The 2×3
  cluster is the honest negative: identical rows, zero level-4 generators selected.
- **Q2: supported as far as the ladder reaches.** Certified growth runs at eight qubits
  on five rungs; no run grows on a step the shots do not certify, and abstention rather
  than silent growth is what stops three of them. Coverage itself is calibrated in the
  Paper A experiments, not here.
- **Q3 has a concrete instance in the record.** The certified H₂ row reports
  `−1.13821296` against a reference of `−1.13727017` — 0.94 mHa *below* the exact energy.
  The variational bound does not survive thresholding a noisy overlap matrix, which is
  why the summarizer refuses to rank on energy without the evidence label.

### 7.3 Benchmark inventory — what each rung can support

| rung | `n` (JW) | exact reference | frozen artifact | role |
|---|---:|---|---|---|
| H₄ `r = 0.9` | 8 | yes | `matched_h4.json`, `clifford_hierarchy_h4.json` | primary; the frozen bank both cost axes reuse |
| BeH₂ CAS(4e,4o) | 8 | yes | `clifford_hierarchy_beh2.json` | second instance; the 22× break-even spread; the one bank priced inside the frozen grid |
| LiH CAS(4e,4o) | 8 | yes | `qr3b_instance_preflight.json`, `priceability_screen.json`, `r3b_margin_stop_probe.json` | chemically independent second instance; rejected at its intrinsic stop (`W = 7740`); the `M = 2` bank (`W = 1439`) resolves 29/40 cells in the target-environment 2+2 redraw (30/40 historically), and its separate 30+100 cost run prices 38/40 cells, making LiH the second exact-tier priced instance; QR3 remains indeterminate after R3d (§13, 13b–d) |
| H₂O CAS(4e,4o) stretched | 8 | yes | ladder | strong-correlation control |
| H₂O CAS(8e,6o) | 12 | yes | ladder | size stress for `W` and grouping |
| Hubbard 2×2 / 2×3 | 8 / 12 | yes | ladder | strongly correlated control, non-molecular weight profile |
| Kitaev 2×2 | 8 | yes | ladder | **excluded from the mapping axis** — one qubit per site, no fermionic encoding behind it |
| TFIM `n = 4` | 4 | yes | `finite_shot_rethink.json` | where the accuracy-matched cost search is calibrated first |
| HCl | 20 | **no** | — | resource-only unless given a declared active space |

**HCl.** STO-3G HCl is ten spatial orbitals, so twenty qubits under JW — past this
project's exact-reference reach, and the ladder already runs without ADAPT/Krylov arms
at `n = 12`. It enters on one of two terms, declared in the row: with an explicit active
space small enough for an exact reference (a full rung), or **resource-only** — weights,
`W`, `G`, gates, depth, no error column, no accuracy-matched cost, and an evidence label
saying the row carries no accuracy claim. It is worth running on the second term because
the contextual-subspace literature reports its largest operator-pool compression there,
and a resource-only row can still falsify a compression claim. It is not worth running on
terms that let a reader mistake it for an accuracy result.

### 7.4 Excited states — deferred, with the reason

A-CASE's projected generalized eigensolver already returns multiple Ritz roots
(`acase_states`, `roots: 3`), so root-aware or state-balanced selection is a natural
extension and contextual-subspace methods are a relevant comparator. Two things must land
first.

*The metric changes.* The word universe is shared across roots — one bank, one measurement
campaign, `R` roots — so the right figure is the **amortized** `C(ε)` per root, and that is
where A-CASE should be structurally ahead of running `R` separate variational
optimizations. Report `C(ε)/R` beside the per-root accuracy, never the total alone.

*The certificate does not cover it.* The current machinery certifies a ground-state Ritz
interval; interior roots need interval statements that the delta-method scoring in
`SharedMeasurement.solve_selected_rank` does not supply, and inventing one is theory work,
not benchmark work. Until then, excited-state rows carry heuristic evidence labels and no
certified cost. This track is scheduled after R1–R4 and is explicitly not on the critical
path.

---

## 8. Projected observables

Material observables are conceptually `PauliSum` expectations of Ritz states, but the Ritz
state is never stored natively. For every observable `Q`, build
`Q_sub[i,j] = ⟨ψ|A_i†QA_j|ψ⟩` through the same bank machinery and evaluate
`⟨Q⟩_k = (c_k†Q_sub c_k)/(c_k†Sc_k)` and transition elements between Ritz roots. A
Hermitian `Q` is mirrored from its upper triangle like `(S, H)`; a non-Hermitian one is not
(`Q_sub[j,i]` is then an independent element), and `expectation` refuses it rather than
quietly returning the real part. The extra word and measurement cost of each projected
observable enters the §6 accounting.

---

## 9. Paper A — confidence-certified, measurement-efficient ADAPT-VQE

The predecessor programme, and the machinery A-CASE builds on. Its software and
data are complete, and its manuscript is written and drift-checked (§9.6.1) but
**not submitted** — it is the one manuscript in the repository still unpublished
(§1.2). It answers a different question from §1's:

> **Can algebraic symmetry, shared Pauli-word structure, and stabilizer
> information make ADAPT-VQE operator selection statistically reliable with
> materially fewer measurements and fewer non-Clifford operations?**

**Naming collisions this consolidation resolves.** (1) The contingent "Paper
B — stabilizer-seeded residual ADAPT" of the old Paper A plan is **retired**: its
go/no-go returned NO-GO (§9.7) and it became a negative-result section of Paper A.
(2) Paper A's phases are relabelled **A0–A5** here, so they cannot be confused with
the Phases 0–19 of §5. (3) "Paper B" is now a **manuscript line, not a manuscript**:
it named the A-CASE/QSCI work of §1–§8, and that line has already produced two
public preprints, P1 and P2 (§1.2). Where the text below says "Paper B", read *the
unwritten successor in that line* — the QSCI and classical-selected-CI confrontation
at accuracy-matched cost (§12). The label is left in place because Phase 12, §7, and
the committed records use it, and renaming it in prose while the records keep it
would be worse than the ambiguity.

### 9.1 Positioning, novelty, and what not to claim

No published work duplicates the full combination (density-multivector formulation
+ exact adjoint gradients + odd-Y pool restriction + finite-shot rank resolution),
but five developments constrain the novelty claims:

| Work | Constraint |
|---|---|
| Magoulas & Evangelista, PRA 113 (2026); Evangelista & Magoulas, PRA 111, 042825 (2025) | Fermionic Clifford transformations exist; do **not** claim the first Clifford treatment of Pauli–Majorana–Dirac structure. Cite in the Clifford/JW sections. |
| Majland et al. (FAST-VQE), PRA 108, 052422 (2023); Long et al., PRA 109, 042413 (2024) | Measurement-efficient ADAPT selection is an active area; do **not** claim generic "measurement-efficient ADAPT-VQE". Cite in the finite-shot sections and use as baselines. |
| Cheng et al., PRA 111, 062413 (2025); Robin, PRA 112, 052408 (2025) | Clifford-point initialization and stabilizer/residual splitting exist; frame the stabilizer work as combining them with the operator-centric representation. |
| Scriva et al., PRA 109, 032408 (2024) | Shot noise can dominate outer-loop cost; resource accounting must cover the full outer loop. Cite in limitations. |
| Yoo, Bae & Kim, PRA 111, 032615 (2025) | Symmetry-preserving ansätze are known; position odd-Y restriction as an *algebraic pool* symmetry with an exactness theorem, not as the first symmetry-aware pool. |

**Defensible novelty statement.** *We introduce an operator-centric ADAPT-VQE
framework in which commutator-gradient observables are represented collectively in
a sparse Clifford-algebra basis. This enables exact algebraic pool reduction,
global reuse of shared Pauli measurements, sequential confidence-certified
operator selection, and stabilizer-aware initialization, all within one versioned
Pauli-rotor intermediate representation.*

Avoid claiming: first Clifford formulation of QC; first Clifford fermions; first
measurement-efficient ADAPT; first symmetry-restricted pool; or a "fundamentally
faster" representation than matrices.

**Differentiator vs FAST-VQE.** FAST-VQE ranks by sampled determinant populations
(a proxy). This work estimates the *exact* commutator selection observable
`G_j = Tr[ρ·(−i/2)[H,P_j]]` with (a) cumulative shot reuse, (b) global shared-word
measurement reuse, (c) simultaneous confidence bounds with an explicit
wrong-selection probability `δ`, (d) an explicit ambiguity outcome instead of
silently picking the empirical max, and (e) algebraic (odd-Y / real-sector) pool
reduction.

### 9.2 Hypotheses H1–H5

Each has a falsifiable test; all comparisons are at matched empirical
wrong-selection rate.

- **H1 — Symmetry restriction.** For Hamiltonians and states real in the
  computational basis (antiunitary-real sector), restricting the pool to odd-Y
  Pauli words yields the *same exact* ADAPT trajectory as the unrestricted pool
  while shrinking candidate and word counts. *Test:* exact trajectories agree
  operator-for-operator on TFIM/XXZ; extend the TFIM observation to a proof for
  the antiunitary-real sector.
- **H2 — Shared-word measurement reuse.** The candidate observables `G_j` overlap
  heavily in Pauli words; measuring the union `𝒲 = ⋃_j supp(G_j)` once per round
  and reconstructing every `ĝ_j` from the shared cache costs strictly fewer
  distinct measurements than per-candidate estimation. *Test:* unique-word counts
  and shot totals against the per-candidate baseline at equal selection accuracy.
  (The most immediate practical win, and the ancestor of the A-CASE bank.)
- **H3 — Sequential selection guarantee.** A best-arm identification rule — select
  `ĵ` only when its simultaneous lower bound exceeds every other upper bound —
  controls `Pr(wrong operator) ≤ δ`. *Test:* empirical wrong-selection rate ≤ δ
  within sampling error over ≥100 seeds (calibration is the headline validation).
- **H4 — Adaptive allocation beats uniform doubling.** Spending new shots on words
  that dominate the variance of *unresolved pairwise gaps* resolves selection with
  fewer total shots than doubling everything. *Test:* median shots-to-resolution,
  uniform vs variance-weighted vs successive elimination.
- **H5 — Layering preserves selection quality at lower depth.** Adding a commuting
  layer of sufficiently strong candidates after the top selection reduces depth
  without degrading the energy trajectory. *Test:* energy vs two-qubit depth
  against single-operator ADAPT.

### 9.3 Statistical method

- **Global commutator bank.** All selection observables in one sparse
  candidate-by-word coefficient matrix `C = (c_jw)`, with `ĝ_j = Σ_w c_jw μ̂_w` and
  `μ̂_w = (N_w⁺ − N_w⁻)/N_w`. No per-candidate accumulators owning duplicate
  measurements.
- **Confidence bounds, two tiers.** *Tier 1 (robust default):* Jeffreys pseudocount
  `p̃_w = (N_w⁺ + ½)/(N_w + 1)`, variance propagated through the linear estimator
  `Var(ĝ_j) = Σ_w |c_jw|² Var(μ̂_w)`, Šidák/Bonferroni correction across
  candidates. *Tier 2 (publication-grade):* empirical-Bernstein / confidence-
  sequence bounds valid under adaptive stopping.
- **Selection outcomes.** The selector returns one of `resolved_best`,
  `resolved_near_optimal`, `below_threshold`, `budget_exhausted_ambiguous`. Never
  silently pick the empirical max while ambiguity remains.
- **Allocation policies (interchangeable):** `uniform_fixed` (baseline),
  `uniform_doubling`, `variance_proportional`, `successive_elimination` (drop
  candidates whose upper bound falls below the best lower bound), `pairwise_gap`.
  Expected winner: successive elimination plus the shared-word cache.

### 9.4 Benchmarks and baselines

*Stage 1 — spin systems:* open/periodic TFIM (`n = 4…12`, `h/J` sweep through
criticality), random-field Ising (30 disorder seeds), XXZ chain, LMG (stabilizer
path). The `n = 4` result is a golden regression test, not the evidence.

*Stage 2 — chemistry* (OpenFermion bridge): H₂, linear H₄, LiH and BeH₂ active
spaces. FAST-VQE determinant-population selection is only meaningful here, so the
FAST-inspired baseline lives in stage 2.

*Baselines for every headline result:* exact-gradient ADAPT; fixed-shot ADAPT;
cumulative-doubling ADAPT; shared-word cumulative ADAPT; confidence-certified
(successive-elimination) ADAPT; random selection; subpool exploration; layered
ADAPT; FAST-inspired selection (chemistry); fixed-depth HVA/VQE.

### 9.5 Resource accounting and statistical protocol

**Report separately, never just total shots:** `N_shots` (individual
measurements), `N_circuits` (distinct measurement circuits), `N_words` (unique
Pauli expectations), `D_2q` (compiled two-qubit depth), `N_non-Clifford`
(non-Clifford rotation count), `S_max` (peak active Pauli support), plus optimizer
evaluations and classical preprocessing time. (This is the accounting
`benchmarks/summarize.py` implements; §6 is its A-CASE-side successor, and the
hardware-aware extension of §6.4 applies to both.)

**Statistical protocol:** ≥30 seeds exploratory, ≥100 seeds for headline
finite-shot comparisons; predeclare seeds, initialization, budgets, tolerances,
`δ`, and near-optimality `ε`. Report wrong-selection probability, top-k
probability, selection regret `R_t = |g*_t| − |g_selected,t|`, median/IQR shots,
95% CIs, ambiguity and failure rates. Headline check: empirical
`Pr(wrong selection) ≤ δ` within sampling uncertainty.

### 9.6 Phase status A0–A5, and what each measured

All 18 backlog items are implemented; the remaining work is the manuscript.

- **A0 — consolidation (done).** IR schema versioning with a backward reader;
  gate-by-gate execution (`Program.state()` evolves `ρ` op-by-op instead of forming
  the whole unitary, so Pauli support stays as sparse as the circuit allows);
  Clifford-angle rotor recognition (`θ ∈ πℤ/2`) and stim lowering, validated
  against MV conjugation. *Exit:* all tests pass; old JSON still loads.
- **A1 — exact research layer (done).** Backend protocol and `ExactMVBackend`
  (gate-by-gate density-multivector evolution, exact expectations, exact adjoint
  gradients, per-gate support tracking), `FiniteShotBackend` (seeded binomial
  sampling, cumulative counts), `DenseStatevectorBackend` (independent reference,
  never presented as the native representation), `StimBackend` (stabilizer
  expectations, discrete Clifford-point search at large `n`); model builders;
  optimizer adapters; fixed-depth HVA VQE; exact ADAPT; odd-Y/real-sector pools.
- **A2 — measurement and confidence layer (done).** Shot data structures, global
  commutator bank, shared word cache, uniform/variance allocation, simultaneous
  confidence intervals, successive elimination, explicit ambiguity outcomes, cost
  accounting. *Exit:* confidence calibration validated on synthetic Pauli means and
  small TFIM instances.
- **A3 — scaling and layering (done).** Commutation graph and layer construction
  (`algorithms/layering.py`), subpool exploration with dead-subpool redraw, QWC
  grouping with joint-distribution sampling, a random-selection baseline, and the
  config-driven benchmark matrix. *Findings*, from the 100-seed `n=4` matrix (TFIM
  `h ∈ {0.5, 1, 1.5}`, periodic TFIM, random-field Ising) and the 20-seed `n=6`
  exploratory matrix: confidence-gated selection beats random selection by 2–4
  orders of magnitude in median energy error at equal operator budget across every
  family; QWC grouping cuts circuits 3.3–4.0×; at `n=6` noisy selection matches
  exact-selection final error (random Ising: 4.1e-3 vs 4.5e-3, near-optimality
  0.96); XXZ with the two-local odd-Y pool hits an ADAPT gradient plateau at ~1e-1
  error — **the pool, not the selector, is the bottleneck**, and it needs
  higher-weight or repeat-enabled pools. A known finding pinned by a regression
  test: alpha-layering without operator repeats can stall at symmetric stationary
  points. Cost: noisy runs are ~3.5 s at `n=4`, ~5 min at `n=6`, and infeasible at
  `n≥8` on a laptop-class core — the `n = 8–12` headline sweeps need dedicated
  hardware (configs are committed and shardable via `--shard i/k`).
- **A4 — stabilizer initialization (done; go/no-go decided).** See §9.7.
- **A5 — chemistry (done; the manuscript is written, see §9.6.1).**
  `models/chemistry.py` with PySCF-computed H₂, LiH(2e,2o), BeH₂(4e,3o) and
  H₄-chain models (HF/FCI cross-checked to machine precision), the JW
  singles/doubles odd-Y `excitation_pool`, the FAST-inspired determinant-population
  selector, the four-arm chemistry benchmark, and `REPRODUCING.md`.

  *Findings, quoted from the manuscript rather than from the superseded plan
  drafts* (`paper/manuscript.tex` §"Molecular systems and the proxy–gradient
  boundary"; the figures and tables regenerate from the committed records):

  - **The proxy failure is intrinsic, not statistical.** Evaluated at `N → ∞` on
    *exact* computational-basis populations, the proxy's top pick carries
    `8.15e-7` of the maximum gradient on H₄ at 0.9 Å (rank 81 of 160) and
    `9.43e-7` on LiH (rank 9 of 12), while picking the true argmax on H₂ and
    BeH₂. No number of shots removes that bias.
  - **And it is regime-confined.** Sweeping H₄ over 0.7–2.0 Å, the proxy's
    gradient fraction stays below `1.5e-5` from 0.7 to 1.5 Å but recovers the
    true argmax at the dissociated 2.0 Å geometry, where a single excitation
    dominates. The misalignment lives wherever several excitations compete.
  - **H₄ is the discriminating case.** Hartree–Fock is `56.0569 mHa` above FCI;
    the proxy recovers less than half of that correlation energy and plateaus at
    **33.89 mHa**, identically on all three seeds, worse than the random arm
    (3.55, 14.26, 18.30 mHa; median 14.26). Exact commutator-gradient selection
    reaches chemical accuracy after nine operators and, at its twelve-operator
    budget, gives `−2.179923798653 Ha` — **0.392816 mHa** above
    `E_FCI = −2.180316614324 Ha`, recovering 99.2993 % of the HF-to-FCI
    correlation energy in 141 optimizer evaluations.
  - **Cost, where the comparison is fair.** At equilibrium the proxy is ~34×
    cheaper than confidence-guided gradient selection on BeH₂ (one operator,
    4 096 shots against `1.39e5` to chemical accuracy) and ~11–13× cheaper on H₂
    and LiH. On correlated H₄ it is far cheaper still and selects a
    numerically negligible-gradient direction.

  *Two scoping caveats the manuscript states and this plan must not drop.* All
  arms share the word-level qubit-ADAPT pool, so the result bounds the
  *mechanism* — a population proxy against the commutator gradient — and is not
  an indictment of any specific FAST implementation or of its natural
  excitation-operator pool. And the exact-gradient H₄ trajectory is the
  exact-gradient quality ceiling obtained with **zero measurement shots**: it is
  *not* a certified finite-shot H₄ trajectory, and no full strict finite-shot
  trajectory on the eight-qubit chain has been run, because grouped selection
  over 160 candidates is beyond the present reference implementation's
  classical-simulation budget. The selector calibration is a separate per-selection
  experiment on well-posed spin instances.

  So the trade-off Paper A reports is: the proxy is an order of magnitude cheaper
  where determinant structure carries the signal, and intrinsically misaligned
  where correlation is essential, whereas gradient selection pays more per step
  and does not silently fail.

#### 9.6.1 Manuscript status

Both manuscripts exist in-tree, complete, with figures and tables regenerated from
the committed records and a checker that verifies they have not drifted:

| manuscript | source | scope | checker |
|---|---|---|---|
| Paper A | `paper/manuscript.tex`, `paper/README.md` | the measurement layer: commutator bank, shared-word caching and QWC grouping, confidence-certified selection, allocation policies, the 100-seed spin study, the proxy-versus-gradient boundary, and the stabilizer-seeding negative | `paper/check_manuscript.py` |
| DA-CASE | `paper_acase/manuscript.tex` (dated 9 August 2026) | "DA-CASE: reusable measurements for adaptive quantum subspaces" — the **Dyadic** Adaptive Clifford-Algebra Subspace Eigensolver: one reference, reused measurements, and the dyadic block-commuting hierarchy of §6.6 | `paper_acase/check_manuscript.py` |

**A naming distinction this plan owes the reader.** The method described in §1–§8
is **A-CASE**; **DA-CASE** is the manuscript's name for the dyadic-measurement
variant, and the dyadic hierarchy is its contribution, not a separate track. Where
this plan says "the A-CASE manuscript" it means the DA-CASE paper. §6.6's frozen
hierarchy tables are that manuscript's data.

This plan does not record submission or archival state for either manuscript — that
is not derivable from the tree, and asserting it here is how a status line goes
stale. What is derivable, and is the claim made above, is that the sources are
complete and regenerate from committed data.

### 9.7 Stabilizer-seeded residual ADAPT — the NO-GO

The design: decompose `H = H_stab + λV` with `H_stab` admitting an efficiently
preparable stabilizer ground state; prepare it as a Clifford `Program`; run odd-Y
ADAPT restricted to non-Clifford residual rotations; count the non-Clifford
operations needed to reach target accuracy. Two initialization variants —
**(A)** discrete Clifford-point search over HVA angles `θ ∈ {0, π/2, π, 3π/2}` with
a stabilizer backend, and **(B)** Hamiltonian stabilizer approximation, a mutually
commuting subset maximizing `Σ_{w∈S} |h_w|` with a consistent eigenspace.

**Verdict: NO-GO for a standalone paper — stabilizer initialization becomes a
negative-result section of Paper A.** Evidence from the three-arm experiment on 9
model instances plus an `n=24` stabilizer-only demo
(`reference_results/stabilizer_seeding.jsonl`): (a) Variant A is *vacuous for this
ansatz class* — the TFIM HVA's optimal Clifford point **is** the `|+…+⟩` reference
on every family tested, exhaustively verified at depth ≤ 3; (b) Variant B's
scaffold, despite starting up to 2 J per bond lower in energy (−23 vs −12 at
`n=24, h=0.5`), leads residual ADAPT into gradient plateaus — final error
1.8e-2–3.3e-2 against ≤ 1e-4 for the baseline on `h=1.0` TFIM (open and periodic)
and 4 of 5 disorder realizations, with early stopping on vanishing gradients. The
one partial positive: at `h=0.5` the scaffold reaches 1e-2 relative error in 6
operators against the baseline's 8.

Interpretation for the manuscript: **symmetry alignment of the reference with the
ground sector, not raw scaffold energy, governs the non-Clifford correction cost.**
The section should present the alignment criterion and the `h=0.5`
early-convergence trade-off.

### 9.8 Go/no-go criteria

**Paper A proceeds** when, at the same empirical wrong-selection rate and across ≥3
benchmark families, the method achieves at least one of: fewer shots; fewer
distinct measurement circuits; materially fewer ambiguous selections; or better
final energy at a fixed measurement budget.

**The stabilizer-seeding paper proceeded separately** only on ≥2× fewer
non-Clifford rotations, or ≥2× fewer optimizer evaluations, or materially higher
success probability at fixed shot/depth budget, on more than one model class. It
did not meet them (§9.7), so it folds into Paper A. (These are the criteria
`benchmarks/run_stabilizer_seeding.py` reports against.)

### 9.9 Non-goals

No generic quantum SDK: no device plugins, transpilers, or vendor runtimes beyond
the existing validated bridges. No new dependencies in the core import path (numpy
only). No claims outside the novelty statement of §9.1.

---

## 10. Falsifiable questions

Paper A's hypotheses H1–H5 (§9.2) are the ADAPT-side questions; those below are
A-CASE's.

**Core (Q1–Q4).**

- **Q1 (compactness).** Does adaptive selection reach chemical accuracy with materially
  smaller `M` *and* `W` than fixed QSE/Krylov at equal generator budget? *Falsifier:* no gap
  on the Phase 7 ladder. *Status:* supported against blind selection, not supported against
  fixed Krylov (§7.2).
- **Q2 (certification).** Do finite-shot Ritz intervals achieve nominal coverage while
  abstention prevents uncertified growth? *Falsifier:* coverage collapse or near-total
  abstention at realistic budgets. *Status:* supported as far as the ladder reaches.
- **Q3 (bound survival).** Under what measurable conditions does the variational upper bound
  survive noisy PSD repair? *Deliverable:* a bias bound in terms of `τ_S`, shot covariance,
  and conditioning — or a documented counterexample family. *Status:* counterexample in the
  record.
- **Q4 (materials reach).** Do competing-order stabilizer configurations plus response
  directions compactly represent low-energy states of the Hubbard/Kitaev clusters?
  *Falsifier:* basis growth tracking sector dimension. *Status:* span half answered,
  compactness half answered on the 2×2 and open on the 2×3.

**Track A and beyond (Q5–Q13).**

- **Q5 — QSCI dominance:** at equal `M`, does QSCI match or beat A-CASE on fermionic
  chemistry while spending zero measured words on projected matrices, once preparation,
  sampling, and classical costs are also reported?
- **Q6 — hybrid versus classical closure:** does the dressed hybrid reach a Pareto point
  unavailable to bare QSCI, selected CI, and A-CASE, and is its span different from or more
  compact than determinant excitation closure?
- **Q7 — overlap selection:** does a QSCI/selected-CI target expose useful candidates that
  lowering-only growth misses on `hubbard_2x3`?
- **Q8 — real-time family:** can a real-time or controlled short-time family recover
  fixed-Krylov accuracy at a `kappa(S)` and propagation error budget a finite-shot
  calculation could survive?
- **Q9 — Clifford grouping:** does fully commuting grouping reduce certified leading shot
  cost with covariance and circuit overhead accounted for? **Answered yes on the
  preregistered BeH2/JW bank**: the certified endpoints are `2^24` versus `2^29`
  total physical shots, all four covariance gates pass, and card-specific runtime
  reporting preserves one inadmissible fully commuting hardware scenario. The
  covariance-aware variance advantage itself is 2.22-fold; most of the endpoint
  gap is the frozen per-group union-bound penalty. The sampled state is the
  Hartree-Fock determinant `|11110000>`; the result is not instance-independent
  or hardware evidence.
- **Q10 — parity ceiling:** does `r_X <= 2(N - 1)` hold across every declared
  spin-conserving Jordan–Wigner Hamiltonian construction path? *Status:* yes on the
  current native, FCIDUMP, fermionic-lattice, and effective-ingestion matrix; the
  test matrix blocks Phase 14 on a violation; the Phase 14a compiler now calls the
  validator whenever its caller explicitly declares a spin-conserving JW source.
- **Q11 — packet gain:** do Haar-stage policies survive ordering ablations and improve the
  final Pareto frontier rather than one finite instance only?
- **Q12 — spin sampled subspaces:** is computational-basis sampled diagonalization
  noncompact on Kitaev, rather than nonexistent?
- **Q13 — mapping invariance:** do JW, BK, and parity reproduce exact energies and equivalent
  fermionic gradients under consistent transforms?
- **Q14 — anticommuting-clique grouping:** on a **fixed-coefficient
  Hamiltonian-energy estimand**, does a clique cover (§3.6, Phase 19) reduce
  certified leading shot cost against QWC and fully commuting covers of the same
  estimand, *after* each setting's Pauli rotations are synthesised and priced
  through the same three device cards? The estimand is part of the question, not a
  simplification of it: a clique setting returns one weighted mean and cannot
  supply the per-word means Phase 14b's `1,814`-word bank reconstructs, so this is
  not a re-run of Q9 on a third rung. *Falsifier:* the rotation resources cancel or
  reverse the setting-count advantage on every card, in which case the
  anticommuting axis is a curiosity and §3.6 keeps only its variance floor and its
  ceiling — itself a reportable result. *Status:* unmeasured. Phase 19 carries a
  structural sizing probe with no committed record, and the ceiling caps any
  affirmative answer in advance: no clique cover can save more than a factor
  `2n+1` against term-by-term measurement.

**Resource accounting (QR1–QR6).**

- **QR1 (accounting).** Does the accuracy-matched cost `C(ε)` ever reorder the protocol rungs
  relative to the settings-count ordering? *Falsifier:* identical ordering on every rung and
  card, in which case settings count was an adequate proxy and this machinery is overhead —
  record it and say so. **Answered yes on the R3 grid** (`protocol_cost.json`): the two
  orderings differ in 28 of 30 arm × card × estimator combinations, and the two that agree
  are the `superconducting-like` single-assignment `parity` and `bk` arms, where an
  inadmissible `k = 8` leaves only three rungs to order. The mechanism is visible in the
  `jw` column — `k = 1` buys 353 cheap settings against 41 at `k = 2`, and the
  shot-to-target crossing moves the other way. So the falsifier does not fire and the
  count is not an adequate proxy on this instance.
- **QR2 (mapping invariance).** Do the §6.2 invariants hold exactly across mappings on every
  rung? *Falsifier:* any violation, which halts R2 as an implementation defect.
- **QR3 (mapping cost).** Is there a device card and protocol rung at which the mapping
  changes `C(ε)` by more than the instance-to-instance spread already present between H₄ and
  BeH₂? *Falsifier:* the mapping effect is smaller than the instance effect everywhere —
  which would demote fermion mapping from an optimization dimension to a footnote, itself a
  useful result. **The original R3 record abstains at the accuracy-matched tier.** It records the mapping
  spread in `C(ε)` — `3.87×` among the three full-width arms, and `4.00×` between the two
  `+2q` arms at `k = 6` under pooling, the largest at equal measured width — but the question
  weighs that against an *instance* spread, and there is still one priced instance. The reason
  changed, though, and the new one is the more interesting half. A second H₄ bank that clears
  the bias floor now ships (`h4_converged`, `0.766 mHa`), and it makes QR3 eligible at the
  *asymptotic* tier — `mapping_axis.json` records
  `eligible_for_cross_instance_comparison`. At the *exact* tier it is blocked by a
  second condition this question had folded into the first: a crossing is resolvable only
  inside `SEARCH_ENDPOINTS`, and this bank's `W = 7926` against BeH₂'s `1814` puts its
  crossings at `16384–65536` on a grid ending at `65536`. So the accuracy-matched
  abstention now rests on **resolution**, not accuracy, and `protocol_cost.json` says which
  through `cost_layer_scope`. `check_protocol_cost.py` fails any record that upgrades the
  verdict, and equally any record that drops a structural system without deferring it.
  R3S then showed part of the barrier is a property of the frozen stopping rule: LiH
  at the margin-rule prefix sits at `W = 1439`, under the one priced bank. The 13b
  probe tested that and **did not price it** — the target-environment redraw resolves
  29 of 40 cells (30 historically) against a gate needing 40. The half that held is
  the one `W` governs: `not_bracketed_within_search_grid`
  went from 15 cells to none, so the crossings are inside the grid now. The half that
  did not is the probe's own confirmation power.

  **R3c then answered that, and QR3 is no longer abstaining.** The preregistered
  `30+100` run on the same forty cells prices thirty-eight of them with confirmed
  finite intervals, so the exact tier has its second instance and the comparison
  this question needs is now *available*. Re-derived over the 88 mapping and 99
  instance cells both banks have admissible and priced, it returns
  `indeterminate_at_this_shot_grid`. The point estimates order emphatically —
  widest mapping spread `17.52×`, narrowest instance spread `1.02×`, which if
  read as numbers would answer the question affirmatively and retire the
  falsifier — but a cost here is the bracket
  `(C(confirmed_fail), C(confirmed_pass)]` the geometric grid licenses, and those
  brackets overlap: `[1.36, 70.08]` against `[1.00, 4.08]`. §6.7's rule is that
  an overlap is a finding about resolution, not a mapping result, so the record
  reports the overlap rather than the ratio. The reason for not answering has
  therefore moved once more — from accuracy, to grid resolution, to whether a
  `2+2` scope probe can confirm a crossing it has already located, and now to the
  width of the intervals the two effects are compared across. What would separate
  them is finer resolution *between* existing endpoints, not endpoints above
  `65536`; the grid's ceiling was not the binding constraint on either bank here,
  and `r3c_lih_full_cost.json` records `wider_grid_licensed_by_this_record:
  false`. `protocol_cost.json`'s own abstention is untouched and stays true of
  the record that carries it.
- **QR4 (pooling × protocol).** Does the coverage fraction `f_w` change the `k*` chosen under
  the pooled estimator relative to the single-assignment one? *Falsifier:* identical `k*`
  under both, which retires the concern. **The falsifier fires on the R3 grid**
  (`protocol_cost.json`): the two estimators' `k*` regions overlap on all fifteen arm × card
  pairs, so pooling never moves `k*`. It does move the point argmin on five of them, which is
  why the question is answered on regions — a relocated point inside a shared region is not
  evidence that coverage relocated `k*`, and reporting it as one would be the same error
  §6.7 exists to block one level up.
- **QR5 (compression interaction).** Is the CS × A-CASE cost ratio multiplicative,
  sub-multiplicative, or antagonistic? *Falsifier for the preconditioner plan:* anything but
  complementary.
- **QR6 (weight propagation).** Does the Hamiltonian-level weight advantage survive into the
  element-operator universe? *Falsifier:* equal ratios on both multisets.

**Track G — structural restriction before encoding (QG1–QG3, §3.5).**

- **QG1 (independent content).** Does pre-encoding algebraic restriction remove
  candidates that the package's existing post-encoding symmetry and reference-leakage
  filters keep? *Falsifier:* filters A–D agree with the existing filters, as required
  by G1's gate, **and** the action-equivalence quotient E removes nothing — in which
  case §3.5 is a reformulation and Track G stops. This is the question that decides
  whether the track exists. **Answered, and the falsifier does not fire — but the
  content is pool-dependent, which is the more useful half.** A–D reproduce the
  existing accept set exactly on every declared pool and instance (the gate, not a
  finding). E removes `522` of `549` on the Majorana pool §3.5 specifies, over
  entrants that are all distinct Pauli words, and `0` on the excitation pool the
  mapping and cost records are built on. At the matched degree cap the surviving
  Majorana classes reach exactly the determinants the excitation pool reaches, so
  what E removes is redundancy that the excitation builder never creates. The
  answer is therefore that §3.5 has independent *derivational* content — a
  quotient no existing code path performs — and no independent *selective*
  content on the family the project actually measures. The checker re-derives
  this verdict from the measured fields, so it cannot be written by hand, and it
  fails any record carrying a cost field.
- **QG2 (encoding cleanliness).** Does giving JW and BK the same GA-admissible
  operator domain change the measured mapping comparison relative to running both on
  the raw pool? *Falsifier:* identical mapping conclusions from both pools, meaning
  the confound G2 exists to remove was never material at these sizes.
  **Closed without running G2, on G1's matched-cap correspondence.** The
  GA-admissible domain *is* the raw pool on this instance family, so the two
  pools are the same object and the falsifier holds by construction. That is a
  weaker discharge than a measurement and is labelled as one: it says the
  confound is absent here, not that it would be absent on a candidate family the
  excitation builder does not already exhaust.
- **QG3 (attribution).** In the `raw → GA → GA+PRD → GA+PRD+WISE` decomposition at
  fixed accuracy under a named card, does the GA arrow carry a margin excluding zero?
  *Falsifier:* it does not — GA is then an architectural convenience with no measured
  resource contribution, which is a reportable result and not a reason to suppress
  the decomposition. **Not run, and its answer is forced by QG2's.** Identical
  pools span identical directions, so the `raw → GA` arrow is exactly zero
  before any cost is measured — the falsifier's own outcome, reached
  structurally rather than by spending the decomposition. GA is an
  architectural convenience on this family, which is reportable and is
  reported.

---

## 11. Literature index and citation discipline

| # | arXiv | Theme | Integrated disposition |
|---|---|---|---|
| 1 | 2302.11320 | QSCI | Mandatory sampled-subspace baseline; Track A. |
| 2 | 2411.00468 | ext-SQD excited states | Direct chemistry/excited-state comparator; Track A. |
| 3 | 2407.08696 | CEO-ADAPT-VQE | Constrains Paper A framing; pool benchmark deferred until after Track A. |
| 4 | 2409.03747 | Oscillator-qubit | Qumode layer declined; retain symmetry-post-selection accounting only. |
| 5 | 2301.10196 | Overlap-ADAPT-VQE | Motivates overlap-targeted A-CASE selection; Phase 11. |
| 6 | 2412.13839 | Time-evolved QSCI | Time-evolved QSCI input; Phase 16A. |
| 7 | 2302.03052 | Projection-based embedding | Interface boundary only; Phase 18. |
| 8 | 2606.30551 | Generative-ML QSCI | Withdrawn; do not cite. Use paper 14 instead. |
| 9 | 2501.14968 | Measurement review | Fully commuting grouping context; Track B. |
| 10 | 2305.04783 | Folded-spectrum VQE | Second moments, variance, folded spectrum; Phase 15. |
| 11 | 2606.05968 | BK symmetry trap | Not evidence for the headline claim; mapping regression motivation only. |
| 12 | 2409.11210 | MORE-ADAPT-VQE | Existing multi-root capability deserves a later benchmark; deferred. |
| 13 | 2311.01393 | FLDC barren plateaus | Positioning only; build nothing. |
| 14 | 2607.20585 | ML-compact QSCI subspaces | Compactness comparison structure; unrefereed benchmark claims require reproduction. |
| 15 | 2011.10027 | Contextual subspace VQE | Kirby, Tranter & Love. The method family R4a's contextual arms sit in; `subspace/contextual.py` implements its commuting half only. Cite for the noncontextual/contextual split. |
| 16 | 2207.03451 | Unitary partitioning × CS-VQE | Ralli, Weaving, Tranter, Kirby, Love & Coveney. Owns the anticommuting-clique half of CS-VQE, which is Phase 19's lever 2. Reusable; never announceable. |
| 17 | 1907.09040 | Unitary partitioning | Izmaylov, Yen, Lang & Verteletskyi. The measurement-side clique reduction behind Phase 19's lever 1. |
| 18 | 2609.10078 | Unbiased questions, spin factors, hyperbit | Hance. Source of §3.6's framing — unbiased ⟺ anticommuting, maximal sets odd (`2q+1`), and the Majorana/parity-superselection reading that supplies §3.6's no-go. A foundations paper: cite for those algebraic statements only, never as chemistry-method evidence. |

**This project's own public preprints** — cite as prior art, never as forthcoming
(§1.2): **P1** [2608.00560](https://arxiv.org/abs/2608.00560), A-CASE;
**P2** [2608.08739](https://arxiv.org/abs/2608.08739), DA-CASE.

**GA structural-preconditioner sources** (§3.5). These entered through an external
design note; each row records what the source actually supports, because two of the
four were cited there for a slightly stronger claim than their abstracts carry.

| # | arXiv | Work | Integrated disposition |
|---|---|---|---|
| P-BK | 1208.5986 | Seeley, Richard & Love, *The Bravyi–Kitaev transformation for quantum computation of electronic structure* | The `O(log n)`-vs-`O(n)` support argument behind the encoding-locality axis. Cite for BK's asymptotic operator support; it is **not** evidence about `W`, which §6.2 shows is encoding-invariant. |
| P-ENC | 2602.07151 | Chien, Chiew, Harrison, Necaise, Wang, Mudassar, McLauchlan, Henderson, Scuseria, Strelchuk & Whitfield, *Putting fermions onto a digital quantum computer* | Review of fermion-to-qubit encodings; the reference for §3.5's framing of JW/BK as two representations of one fermionic algebra. Treat as a review: attribute the Majorana→Pauli formulation to it only after checking the text, which the design note did not do. |
| P-IDEM | 1705.06600 | Ul Haq & Kauffman, *Iterants, Idempotents and Clifford algebra in Quantum Theory* | Supports §3.5C: projections/idempotents are native Clifford structures, not bolted-on matrix constructions. It does **not** establish the minimal-left-ideal treatment §3.5D uses — that needs a separate source or an in-repo derivation before any manuscript leans on it. |
| P-GAGATE | 2606.12480 | Amraoui & Toffano, *Geometric Algebra Quantum Gate Decomposition* | Pauli group identified with blades up to global phase; Clifford operators as products of π/4 Pauli rotors. Confirms §3.3's existing "blades up to phase" reading and is the natural source if the measurement-grouping idea of §3.5's out-of-scope note is ever pursued. Not used for structural restriction. |
| 15 | 2607.16869 | Correlation rank and Clifford-accessible measurement | Test the explicit invariant first; benchmark claims remain unverified. |

**Actionable hygiene.**

- **arXiv:2606.30551** was withdrawn by arXiv administrators because the submitter did not
  have the rights to agree to the licence at submission. It must not enter either
  bibliography. For related RBM/configuration-recovery content, cite **arXiv:2607.20585**
  and label its benchmark evidence as unrefereed until reproduced.
- **arXiv:2606.05968** is not evidence for an intrinsic Bravyi–Kitaev symmetry trap or
  one-cycle FCI convergence. Its stated fixed-UCCSD derivative and ADAPT commutator gradient
  are the same derivative for the same anti-Hermitian generator, while its claimed FCI states
  retain large commutator gradients. Use it only to motivate mapping-consistency regression
  tests.
- **arXiv:2607.16869** supplies an explicit structural claim that can be tested
  independently. Separate a reproduced parity/X-rank invariant from its unreproduced
  shot-reduction benchmarks.

The mapping regression suite motivated by paper 11 must check exact energy invariance across
mappings; correctly encoded reference states; equality of finite-difference, analytic, and
commutator gradients; and separate measurement of Pauli weight, distinct-word count, grouping
cost, and compiled circuits (R2, §5).

**Deferred or positioning-only.** MORE-ADAPT-VQE: multi-root A-CASE already exists, but a
dedicated comparison waits for the QSCI/ext-SQD ladder. FLDC barren plateaus: positioning for
finite-depth ADAPT circuits only — A-CASE solves a generalized eigenproblem and has no
variational trainability landscape. Circuit-native real-time A-CASE: Q8 and Phase 16B, not an
incremental generator.

---

## 12. Manuscript positioning

**Publication state.** Two manuscripts are out (§1.2): P1 (A-CASE, 2608.00560) and
P2 (DA-CASE, 2608.08739). Paper A (§9) is written, drift-checked, and unsubmitted.
Everything else in this section describes work that has not been written up.

**The rule that now governs every positioning statement below.** A new manuscript
must state what it adds *on top of P1 and P2*, not merely what it adds on top of
the external literature. The three claims most at risk of accidental
self-duplication are word-bank reduction, setting-count reduction, and
covariance-aware allocation — all already reported in P2 (§1.2(1)–(2)).

1. **A successor to P1/P2 must confront QSCI and classical selected CI at
   accuracy-matched cost.** A QSCI comparison alone is
   insufficient once the hybrid dresses determinants.
2. **Paper A's novelty is certification, not generic measurement efficiency.** CEO-ADAPT
   constrains the pool-design claim; a benchmark can follow Track A.
3. **Ancilla-free A-CASE is a resource-boundary statement, not a zero-T-count statement.**
4. **Configuration Haar packets are an opt-in coarse basis with a measured trade-off**, not a
   universal wavelet advantage.
5. **Orbital basis is a recorded parameter.** The repository contains a negative result
   against standardizing wavelet orbitals.
6. **BK-trap claims are excluded** unless independent mapping and gradient invariants
   reproduce them.
7. **FLDC is positioning only.**
8. **Mapping-invariant claims must be labelled as such.** The subspace-level resource claims
   are invariant by construction across the linear encoding family (§6.2); only the
   measurement-compilation layer is mapping-dependent, and only it may carry an empirical
   mapping claim.
9. **Geometric algebra is a structural language, not a compression result.** §3.5 may
   be presented as an architecture — admissibility decided before encoding, so that
   the encoding comparison is clean and PRD ranks only surviving candidates. It may
   **not** be presented as finding a smaller Hilbert space, and its filters A–D
   largely re-express restrictions the package already applies after encoding. The
   only part that can carry an independent quantitative claim is the
   action-equivalence quotient (§3.5E), and only against the per-filter marginals
   Phase G1 is required to report.

---

## 13. Execution order

Track A must not wait for Tracks B or C. Within each track the order is dependency-forced;
between tracks, R1 is cheap and its result can reorder Track B's protocol conclusions, so it
comes early.

**Immediate order, after the R3 environment migration.** Step 13b did not price:
it left the exact tier with one priced instance, and the binding constraint moved
from grid fit to the `2+2` probe's confirmation power. R3c therefore froze the
headline 30+100 protocol — the target instrument, not another resized scope
probe — in a result-free commit, and then ran it: thirty-eight of the same forty
cells price, and the exact tier has its second instance. Restoring the record gates also exposed that
PR #73's two records were stamped on the older 3.11 / NumPy 2.4.6 stack while
the other ten use 3.12 / 2.5.2. That split is now repaired by a genuine rebuild,
with the old and new record identities connected by
`migrations/r3_environment_3_12.json`: R3S keeps the same verdicts, while R3b
moves from 30/40 to 29/40 resolved cells without changing its rejection, zero
grid-fit failures, or one ceiling cell. So (1) R3c has been executed exactly
once under its frozen 3.12 / 2.5.2 environment, and its producer, record and
checker have landed; (2) Phase 13's deterministic X-rank invariant now ships
and Q10 passes on the current construction matrix; (3) Phase 14a's public compiled
measurement-plan boundary now ships without a sampled record, and Phase 14b's frozen
QWC-versus-fully-commuting comparison has since executed once against it, which
closes Track B's step 9 and answers Q9 on that bank. R4a's
exact-tier blocker is lifted on its own terms — R3c supplied the
second priced instance — and its gate is restated below, because "QR5 answered"
was circular: R4a's four arms are what measure QR5. (4) G1 has since been built
and measured against that opening, and its result retires G2 and G3 rather than
scheduling them: filters A–D reproduce the existing post-encoding filters
exactly, filter E has content on the Majorana pool §3.5 specifies and none on the
excitation pool the mapping records use, and at the matched degree cap the two
pools reach the identical determinant set. Track G therefore has no open step
either. **R4a's declared structural execution has now run, and it stops.** No
contextual rung admits all four arms under the frozen bias and word-universe
gates: the full-QSE control misses only the word ceiling, standalone ACASE passes,
and both contextual arms miss the accuracy gate. The derived selection is null,
sampled execution remains unauthorized, R4b is unscheduled, and QR5 is
undetermined. **The remaining declared QR3 resolution experiment, R3d, has now
run and remains indeterminate.** Its seven midpoint cells narrow four of five
parent intervals; the BeH2 midpoint is environment-marginal and therefore
non-informative. The resulting mapping-support minimum (`2.1906x`) does not
strictly exceed the instance-support maximum (`3.9998x`). That failed positive
separation is not a negative result: the targeted design did not refine all 187
cells entering the global extrema and cannot reselect them after the draw.
Paper A's submission stays schedulable and blocks nothing.

**Paper A — the one item outside the tracks.** Its software, data, go/no-go decisions,
and manuscript are complete (§9.6, §9.6.1) but it is not submitted. Nothing below
depends on it, and it depends on nothing below, so submission is schedulable at any
time; it may now cite P1/P2 as companion work.

**Housekeeping — done.** `paper_acase/` remains the P2 source tree. P1's sixteen
published text/data/table sources from `67ea0dd` are restored byte-for-byte under
`paper_a_case_subspaces/`; `SOURCE_SNAPSHOT.json` records every historical blob,
and `check_snapshot.py` plus the restored `check_manuscript.py` gate provenance and
manuscript structure. The four PDF figures are generated outputs and may carry
different PDF metadata under a newer Matplotlib, while their historical blob ids,
scientific inputs, and staleness checks remain recorded.

**Track A — Paper B critical path.** Steps 1–8 are built and have been read
(§5, Phases 8–12). Their result led to the completed PRD programme: at matched
budget, preconditioned residual Davidson is the accuracy engine, while A-CASE and
selected CI are controls rather than the minimum-error method. The order is retained
because it is also the dependency order for re-running or extending any of it.

1. QSCI contracts and exact sampled-subspace restriction (Phase 8A–8D).
2. QSCI on the ladder, including a raw spin-system arm (8E).
3. Excitation-closure and selected-CI controls (Phase 9).
4. QSCI configuration generators and dressed families (10A–10B).
5. Span/principal-angle diagnostics (Phase 9 diagnostic, gating 10D).
6. The Haar packet tier as a staged hybrid arm (10C).
7. Overlap-targeted scoring and ordering ablations (Phase 11).
8. The complete Track A ladder; reposition Paper B (Phase 12).

**PRD/WISE architecture — completed.** The orthogonal-residual regression,
Davidson shift sweep, packet-cost preflight, matched A-CASE study, and frozen exact
and finite-shot suites are complete (Phase PRD). Their positive exact and negative
finite-shot results are the premise for the measurement-first resource work below,
not another open accuracy phase.

**Track B — measurement.**

9. Explicit X-rank invariant (Phase 13), the compiled-plan boundary (Phase 14a),
   and the frozen QWC-versus-fully-commuting comparison (Phase 14b) are all done.
   Step 9 is closed, and every Track B step declared before Phase 19 is closed
   with it; Q9 is answered on the preregistered BeH2/JW bank and nowhere wider.
9b. Phase 19's clique cover is a **later proposal, unscheduled and not built** —
   added after step 9 closed, so it does not reopen it. It depends on nothing
   above, its algebra is already in the package (§3.6), and it opens with a
   result-free preregistration and the §3.6 invariant tests rather than with a
   run. Note what it is scoped to before it is counted as measurement work: a
   fixed-coefficient Hamiltonian-energy estimand, not Phase 14b's matrix-element
   word bank (§5, Phase 19, lever 1).

**Resource accounting** (interleaves with Track B; R1 first).

10. R1 — cost model on the frozen banks. **Done:** structural, asymptotic, and
    exact-oracle nonlinear shot-search layers ship; v2 regenerates exactly under
    `logical-alltoall`, and QR1/QR4 are answered on H₄ and BeH₂.
11. R2a — restriction primitive and invariance checks. **Shipped**
    (`clifford_qc/subspace/restriction.py`, `tests/test_restriction.py`): the
    `Restriction`/`RestrictedProblem` transport, the three oracle checks, and
    `restricted_sector_operators`. The R2b mapping foundation now consumes this
    primitive. The completed R2b record applies it across every declared mapping
    rung. *Gate:* QR2 passes on every rung; no cost numbers are published from a run
    whose invariants failed.
12. R2b — mapping measurements on H₄, BeH₂, H₂O CAS(8e,6o), Hubbard.
    **Done:** the raw-pool producer, pinned configuration and H₂O input, stamped
    record, regenerating checker, and four-system QWC/device ledger ship. QR2 passes;
    corrected like-for-like QR3 is negative on both independent metrics, while the
    accuracy-matched comparison abstains because only BeH₂ clears its bias floor.
13. R3 — the explicitly deferred `mapping × k`, coverage, and `k*` regions under
    three device cards. **Structural layer shipped**
    (`run_protocol_axis.py`, `reference_results/protocol_axis.json`,
    `check_protocol_axis.py`): `G(k)` and coverage are recorded on H₄, the
    converged H₄ bank and BeH₂ across `k ∈ {1,2,4,8}` for all five mapping
    arms, discharging R2b's `deferred_to_r3`. The `k=1` column reproduces every
    frozen R2b setting count, which is the condition that makes the grid an
    extension of that record.
    **Accuracy-matched layer shipped** (`run_protocol_cost.py`,
    `reference_results/protocol_cost.json`, `check_protocol_cost.py`): R1's
    exact-tier nonlinear shot search runs once per `(arm, k)` cell, and each
    confirmed crossing becomes a `C_time(ε)` *interval* — `(C(fail), C(pass)]`,
    widened one grid step on any side R1 flagged environment-marginal — from
    which `k*` is read as the set of rungs reaching the smallest upper bound.
    BeH₂ only, and the cost layer now declares its own scope rather than
    inheriting the structural one: `cost_layer_scope` partitions the structural
    grid's three banks into what it prices and what it defers, with a reason on
    each deferral, and the checker fails a record where a system is merely
    absent. `h4` is unpriced because its 3.019 mHa bank bias exceeds the target
    on every arm; `h4_converged` is right-censored at the frozen `65536` search ceiling, while
    `further_search: deferred` records the separate decision not to extend
    `SEARCH_ENDPOINTS`. Every priced cell reproduces the
    structural grid's setting count, which is what makes this the cost layer of
    that grid.

    *Result.* **All thirty `k*` determinations — three cards × two estimators ×
    five arms — are regions, not integers.** The endpoint grid never separates
    the rungs on this instance, which is §6.7's rule firing rather than an
    evasion of it: `logical-alltoall` admits the whole ladder on full-width
    arms, `ion-like` narrows to `k ∈ {1,2}`, and `superconducting-like` loses
    `k = 8` on the three full-width arms to its fidelity floor (minimum setting
    fidelity `0.299`–`0.357` against `0.5`) while the `+2q` arms clear `k = 6`
    at `0.513`. The point argmins move around inside those regions — `k = 4`
    under `logical-alltoall` single assignment, `k = 1` or `2` elsewhere — which
    is exactly the integer §6.7 forbids publishing.

    *The defect this phase recorded is now fixed.* Both `run_protocol_axis.py`
    and `run_mapping_axis.py` took their exact sector reference from
    `ground_state(..., k=1)`, whose `method='auto'` selects ARPACK and returns a
    different last bit in every process — a `5e-14` Ha spread, measured, which
    lands as `1e-10` mHa on a residue of order `1`. Both now take
    `method='dense'`, as `run_protocol_cost.py` already did. The regeneration
    that the second priced instance required made the change free, which is why
    it happened here rather than in a phase of its own: rebuilding both records
    moved exactly the two energy-difference fields and nothing else, and their
    `exact_sector_energy` values now agree with each other bit for bit instead
    of each carrying its own draw.

    One expectation recorded here did **not** survive the fix. It said the same
    change would let both older records tighten their `error_millihartree`
    tolerance; it does not. ARPACK was one contributor to the drift, and
    `check_mapping_axis.py` documents another the fix does not touch — a
    measured `1.5e-10` mHa spread between `OMP_NUM_THREADS=1` and `=8`, which
    the default `atol=1e-10` sits directly on top of. The `(1e-10, 1e-8)`
    tolerance therefore stays, now with the right reason attached to it.

    *Result on P5.* Its first clause survives on the full-width arms and its
    monotonicity clause does not survive at all. The `JW/parity/BK` spread in
    `G(k)` falls from `1.713×` at `k=1` to `1.048×` at `k=8` on H₄, and from
    `8.610×` to `1.071×` on BeH₂ — `G(k=n)` does agree between mappings within
    tie-break noise. But the closing is not monotone: BeH₂ touches `1.000×` at
    `k=2` and `k=4` before rising to `1.071×`, and on H₄ the two `+2q` arms
    narrow to `1.054×` at `k=4` and then *widen* to `1.181×` at `k=6`, the
    largest spread of any rung except `k=1`. A reduced arm measures a narrower
    register, so its `k=n` is a different problem; the record compares only arms
    of equal measured width, and `tests/test_protocol_axis.py` pins the
    counter-example so it cannot be refactored away.
13a. R3S — the priceability screen, before spending another probe. **Done**
    (`run_priceability_screen.py`, `reference_results/priceability_screen.json`,
    `check_priceability_screen.py`): every declared candidate is walked along its
    frozen greedy ordering and gated on two structural quantities — bias against
    the accuracy target with a declared margin, and the binding word universe
    against the ceiling QR3b calibrated. Nothing is sampled, and the margin
    factor is labelled `declared_here_not_preregistered` because it arrives with
    its first result; each candidate's `margin_sensitivity` records the range of
    margins over which its verdict is unchanged. The walk's early exit rests on
    two monotonicities the checker re-derives per candidate rather than
    assuming: bias non-increasing along the prefix (a Ritz value cannot rise
    as the span grows) and `W` non-decreasing (a longer prefix adds
    matrix-element pairs and removes none). *Result:* LiH is admissible at
    `M = 2`, `W = 1439`, under BeH₂'s `1814`; `h4_converged`, H₂O and Hubbard are
    rejected on the ceiling — all under this declared screen, which ranks
    candidates for probe spending rather than establishing which instances are
    priceable. *Gate:* the screen authorizes or withholds a probe and prices
    nothing — `check_priceability_screen.py` fails a record carrying any cost
    field. *Next:* step 13b.
13b. R3b — the LiH `margin_stop` probe. **Done, mixed.**
    *Preregistration first, in its own commit*
    (`benchmarks/configs/r3b_margin_stop_probe.json`,
    `check_r3b_preregistration.py`): the bank, gates and seed roots landed before
    any sampling, which is what lets this record call its rule preregistered
    where R3S could not — the margin rule itself keeps R3S's
    `declared_here_not_preregistered` label, and this is its first preregistered
    *use*. The checker re-runs that rule over the frozen QR3b ordering and
    requires it to select exactly the declared prefix, so a prefix chosen for
    cheapness fails before a shot is spent.
    *The run* (`run_r3b_margin_stop_probe.py`,
    `reference_results/r3b_margin_stop_probe.json`,
    `check_r3b_margin_stop_probe.py`): the frozen QR3b `2+2` scoping probe on
    LiH CAS(4e,4o) at `M = 2`, labels `I` and `E(6,7<-2,3)`, bias `0.370` mHa,
    binding `W = 1439`, under QR3b's grid, estimators and block sizes unchanged,
    so the two records differ in the bank and nothing else. Three minutes on four
    workers against QR3b's 150 CI-minutes.

    *Result: the bank is rejected and no full run is authorized.* The record
    originally resolved 30 of 40 cells under Python 3.11 / NumPy 2.4.6. Its
    genuine target-environment redraw resolves 29 of 40 under Python 3.12 /
    NumPy 2.5.2; the preregistered gate needs all 40 to resolve strictly before
    `65536`. Both draws therefore leave the exact tier with one priced instance.

    *What changed is the failure mode, and that is the finding.*
    `not_bracketed_within_search_grid`, the failure the word-universe ceiling is
    a proxy for, accounted for 15 of QR3b's 22 unresolved cells and **none** of
    R3b's; the passing endpoints fell about two grid steps, modal `65536 →
    16384`. The historical draw had nine unconfirmed exploratory crossings and
    one nonmonotone confirmation; the target-environment redraw has eleven
    unconfirmed exploratory crossings. Both have one cell bracketing only at
    the last grid point. Those are the probe's replica count and its headroom,
    not `W`. So the ceiling is corroborated on
    what it actually predicts, and the binding constraint has moved from the grid
    to the instrument. The record's verdict is
    `corroborated_on_grid_fit_headroom_marginal`, and the split behind it is
    labelled `post_hoc_diagnostic_not_preregistered`: the preregistration
    declared a corroborate/falsify binary and the drawn cells showed it conflates
    two mechanisms. The preregistered gate, its inputs and its rejection are
    untouched by that relabelling.

    *Not licensed by this result.* A probe re-sized after seeing which cells
    failed — that is the move preregistration exists to prevent, and re-sizing
    needs its own declaration. A wider `SEARCH_ENDPOINTS` — the grid was never
    the binding constraint here. And no `30+100` run on this bank is authorized
    *by R3b*. R3c below is a new, result-free declaration motivated by the
    explicitly post-hoc failure-mode split; it does not turn R3b's rejection into
    a pass or upgrade that probe's evidence.
13c. R3c — the LiH `margin_stop` headline cost run. **Done; second instance priced, QR3 indeterminate.**
    `benchmarks/configs/r3c_lih_full_cost.json` and
    `check_r3c_preregistration.py` freeze the exact R3b bank, all five mapping
    arms, `k ∈ {1,2,4,8}`, both estimators, the unchanged `64…65536` grid,
    30 exploratory plus 100 confirmatory replicas, the zero-failure and one-sided
    95% bootstrap-RMSE rule, fresh disjoint seed roots, and the three existing
    device-card hashes. The execution environment is also frozen at Python 3.12,
    NumPy 2.5.2, SciPy 1.18.0 and Stim 1.16.0, because seed roots do not define
    the same stream across NumPy releases. The claim boundary is tenseless: this
    config carries no sampled result, and a later record is limited to those
    frozen choices.

    This is not a re-sized scope probe. R3b's 2+2 instrument answered its own
    preregistered gate negatively and remains immutable; its labelled post-hoc
    diagnosis motivates testing the target 30+100 instrument directly. That
    config independently authorized the one full run discharged below while
    preserving the earlier `full_run_authorized: false` finding as a fact about R3b.

    *Gate, discharged.* The producer, sampled record and result checker landed
    after the preregistration and the run was executed once under the frozen
    stack. `run_r3c_lih_full_cost.py` refuses before a shot is drawn: it re-runs
    the result-free checker over the config it is about to consume, binds that
    config's bytes to the digest the migration manifest froze, refuses a stack
    other than the frozen one, refuses a pilot record that has stopped rejecting
    the bank, and refuses a first priced instance drawn by another instrument.
    Thirty-eight of forty cells supply a confirmed finite interval and two —
    `parity` k=4 and `bk` k=2, both single-assignment — stay right-censored on
    unconfirmed exploratory crossings; censoring is not infinite cost and the
    record licenses no wider grid. `check_r3c_lih_full_cost.py` imports
    `check_protocol_cost`'s per-cell rules rather than restating them, so the
    crossing, bracket, rank-stability and `k*` contracts that gate BeH2 gate
    these cells too, and it runs in the `sampled-records` matrix. Each cell either supplies a confirmed finite interval or remains
    right-censored at `65536`; censoring is not infinite cost and does not
    license a wider grid. QR3 is re-derived only if the record supplies a second
    priced instance.
13d. R3d — targeted within-grid QR3 resolution. **Done, indeterminate.**
    `benchmarks/configs/r3d_qr3_refinement.json` and
    `check_r3d_preregistration.py` bind the immutable R3c and BeH2 cost records,
    their exact sampler implementations, execution stack, and device cards.
    They freeze the five cells supporting R3c's widest mapping and narrowest
    instance point estimates, and every geometric midpoint strictly inside each
    inherited conservative interval: seven new endpoint cells total, with no
    endpoint above `65536`. Each receives the target 30 exploratory plus 100
    confirmatory replicas under fresh, disjoint streams.

    The support selection is explicitly post-R3c but prospective with respect to
    every R3d draw. Parent endpoint decisions are read-only. A non-marginal R3d
    failure may raise an inherited lower endpoint and a non-marginal pass may
    lower its upper endpoint; an endpoint in the frozen 10% environment-marginal
    band is reported but cannot tighten either side. Missing evidence,
    nonmonotonicity, or an invalid interval forces
    `indeterminate_after_refinement`.

    The decisive readout is asymmetric. R3d reports a positive QR3 direction on
    the preregistered supports only when the refined mapping-spread lower bound is
    strictly greater than the refined instance-spread upper bound. Every other
    outcome remains indeterminate. Because the design targets point-support cells
    rather than all 187 cells entering the global extrema, it cannot establish a
    negative QR3 direction or reselect new extrema after the draw.

    The one authorized 30+100 execution reports every frozen midpoint. LiH/JW
    `k=4` passes at `32768`, narrowing to `(16384,32768]`; LiH parity and BK
    `k=4` fail at `2048` and pass at `8192`, narrowing both to `(2048,8192]`;
    LiH parity `k=2` fails at `2048`, narrowing to `(2048,4096]`. BeH2 parity
    `k=2` returns a `1.702140` mHa UCB at `8192`, inside the frozen 10% marginal
    band, so its `(4096,16384]` interval stays unchanged. The refined mapping
    minimum is `2.1906417x`, below the instance maximum `3.9997993x`; therefore
    the licensed readout is `indeterminate_after_refinement` and no negative
    or mapping-wide QR3 claim follows. The regenerating checker and manual
    sampled CI row cover the sealed record.
14. R4a — contextual-subspace comparator arms with bias floors. **Done at the
    structural gate, negative.** The one authorized structural execution finds no
    rung where all four arms clear both gates. Full QSE has `0.0033` mHa bias but
    `14350` nonidentity words; standalone ACASE passes at `0.0695` mHa and `1223`
    words; CS-QSE and CS+ACASE reach best bias floors of `5.345` and `5.899` mHa.
    `selected_contextual_rung` is therefore null. No sampled execution, finite-cost
    ratios, or interaction `Δ` is authorized, and QR5 remains undetermined. The
    full-QSE word-ceiling failure is an operational screen rejection, not an
    infinite-cost claim.
15. R4b — CS-preconditioned A-CASE, built only on a complementary QR5. This is the
    step QR5 genuinely gates, and it already said so.

**Track G — structural restriction before encoding** (§3.5). Independent of Track A;
it shares the `Restriction` primitive with R2a, so it starts no earlier than step 11.

16. G1 — the preconditioner and its per-filter marginals. **Done.** Both gates
    pass: A–D reproduce the existing Pauli-side filters with an empty symmetric
    difference on every pool and instance, and E's marginal is reported
    separately. The stop condition was written for the case where E removes
    nothing; what happened is narrower and needed the second pool to see. E
    removes `522` of `549` on the Majorana pool §3.5 specifies and `0` on the
    excitation pool the mapping records use, and at the matched cap the two
    pools reach the identical determinant set — so the chain reconstructs R2b's
    pool rather than restricting it.
17. G2 — **retired, not scheduled.** It exists to give JW and BK the same
    physical operator domain, and step 16 measured that domain to be the pool
    R2b already maps. Rerunning the mapping producer on it would reproduce the
    frozen record, so QG2's falsifier holds by construction. Recorded as the
    weaker discharge it is: the confound is absent on this family, which is not
    a claim about a family the excitation builder does not exhaust.
18. G3 — **retired with G2**, and for the same reason: identical pools span
    identical directions, so the `raw → GA` arrow is zero before any cost is
    measured. That is QG3's own falsifier reached structurally instead of by
    spending the decomposition, and it is reported rather than suppressed.

Two orderings were defensible and the choice was deliberate. G1 before R2b gives the
cleaner mapping experiment (§5, Phase G2) at the cost of delaying R2. R2b first
gets the mapping result out on the raw pool and treats the G pool as a later
refinement. **The plan took the second**, because R1's exact-tier gate was already
the binding constraint on R2 and G1's own falsifier might retire the G track
entirely; spending the mapping experiment's schedule on an unbuilt preconditioner
would have been betting the near-term result on the more speculative branch.

*That ordering is now settled by its own outcome, and it was the right one.* G1
did not retire the track on the falsifier as written, but it retired G2 and G3 on
something the first ordering could not have shown: the two pools coincide. Had G1
run first, the "cleaner mapping experiment" it was supposed to buy would have been
the same experiment R2b already ran, and the schedule spent reaching it would have
bought a duplicate record.

**Track C — longer horizon.**

19. Second moments, time-evolved inputs, mapping breadth, and embedding (Phases 15–18) after
    the successor-manuscript result is known.
20. CEO and MORE-ADAPT benchmarks after the critical comparison is stable.
21. The excited-state track, after the certificate question of §7.4 has an answer. Note
    that §3.5B's transformation-character parameter is what keeps this track reachable
    from Track G; a G1 that hard-codes `[A,Q]=0` would close it.

Each phase ships the project's standard triple: a `run_*.py` producer, a stamped
`reference_results/*.json` record with an explicit `schema` string, and a `check_*.py` that
regenerates and compares it — plus its `REPRODUCING.md` entry and its evidence labels.

**Preregistration is a property of the commit history, not of the word.** A gate,
threshold, margin, or stopping rule is preregistered with respect to a result only
if the commit that declares it precedes the commit that first reports a result
under it. R3S could not meet that — its config and its first verdicts entered
together — and its record carries `margin_factor_status:
declared_here_not_preregistered`, with the checker refusing an upgrade. That is the
template: a declared-not-preregistered parameter is labelled as such in the record
and the docs, ships with a sensitivity range showing whether any verdict turns on
it, and is frozen from that commit so the *next* use is the preregistered one.
Producers that consume a preregistration read it from the config that owns it
rather than redeclaring it, as `candidate_specs()` does.

**A claim boundary bounds the artifact it sits in, so a record does not inherit a
preregistration's.** Most producers here copy `config["claim_boundary"]` into the
record and should: the config's sentence is a statement about the phase, true of
both files. A preregistration-only config is the exception. Its boundary says no
sampling has been performed — true of the commit that landed it, false of the
record that ran the probe — and R3b's record inherited it, so a file reporting a
rejection over forty sampled cells opened by denying that any cell was sampled.
The fix is not to edit the config, which is the one thing landing it first exists
to prevent: the record states its own boundary and quotes the config's under
`preregistration.config_claim_boundary_at_landing`, where it stays true of what it
describes, and `check_r3b_margin_stop_probe.py` fails a record that inherits a
boundary asserting nothing was sampled while `scoping_probe.executed` is true.
The forward fix is R3c's preregistration, which states its boundary tenselessly
from the start — "this config carries no sampled result" and "a later record may
report" rather than "no sampling has been performed" — so its record can inherit
the boundary without contradicting the sampling it reports.

**The record gates run in three cost-aware tiers.** The `test` job and the
deterministic `structural-records` matrix run on every pull request, push to
`main`, and manual dispatch. The latter covers `check_mapping_axis`,
`check_protocol_axis`, `check_priceability_screen`, the R3 environment-migration
lineage gate, and both result-free preregistration checkers. The replica-drawing
`sampled-records` matrix runs only
on explicit dispatch because it costs roughly ten runner-hours. A PR that changes
a sampled producer, config or record names the dispatch run against its exact
head before merge; a result-free structural preregistration does not spend that
matrix. This restores automatic claim and lineage protection without silently
turning every rebase into a full benchmark campaign.

Each value gate installs from its own named record stamp. That is not a majority
resolution: the manual `environment-consistency` job still requires every record
to agree, and it is green only because both PR #73 records were genuinely rebuilt
under the common stack. `check_r3_environment_migration.py` separately preserves
their historical identities and the R3c evidence boundary.

---

## 14. What this plan does not claim

**On Paper A.** Its claim discipline is separate and stated where it belongs: the defensible
novelty statement and the five things not to claim are §9.1, and its non-goals are §9.9.
Stabilizer seeding is reported as a negative result (§9.7), not as a deferred success.

**On method.** No claim to the first subspace/QSE method, the first quantum Krylov or
non-orthogonal eigensolver, a speedup over dense linear algebra at small `n`, DFT
replacement, or an exponential-complexity escape via geometric algebra. The Clifford
representation is an algebraic backend that makes the measurement-sharing and certification
layers natural; the physics guarantees (variational bounds, Ritz theory) come from the same
eigensolver principles as always.

**On the GA structural preconditioner (§3.5).** It does not find a smaller Hilbert
space, and no wording suggesting that it does may survive review. It is a
pre-encoding language for restrictions, most of which the package already imposes
after encoding; its filters are asserted to *agree* with those, and that agreement
is a gate rather than a discovery. Grade is not excitation rank (§3.4, §3.5A).
Working in a Majorana generating set does not change the algebra — it is the same
`Cl(2n,ℂ)`, not a smaller one (§3.5). No claim to originate Majorana
representations of fermionic algebras, contextual-subspace projection, symmetry
tapering, or idempotent/ideal formulations of quantum mechanics; §11's P-BK,
P-ENC, P-IDEM, P-GAGATE rows record who does own them and, in two cases, what they
do *not* establish. Phase G1 has now reported the per-filter marginals, and they
bound the section rather than releasing it. Filters A–D reproduce the existing
post-encoding accept set exactly, which §5 makes a *gate* — agreement is the
condition for proceeding, not a discovery, and it may not be reported as one.
Filter E's marginal is real but pool-dependent: it removes `522` of `549`
candidates on the wide Majorana pool and nothing at all on the excitation pool
every mapping and cost record is built on, and at the matched degree cap the two
pools reach the identical set of determinants. So the defensible statement is
that the pre-encoding chain *reconstructs* the pool the package already builds.
Wording that reports E's Majorana-pool marginal as a reduction of the pool R2b
uses, or as any resource quantity at all, is the specific error this paragraph
exists to block.

**On anticommuting cliques and the hyperbit (§3.6).** The `m`-ball is the state
space of a set of unbiased questions. It is a perfectly good state space; what is
degenerate is its *image* under this package's physics. Every parity-commuting
state — every physical state of a fermionic problem — maps to the single centre
point, and no fermionic Hamiltonian the package builds contains such a question at
all, so the ball carries one point of physical content and no structure. That is
the claim, and the loose version of it ("the ball is empty") is wrong twice over:
the centre *is* reachable, and the package's non-fermionic spin builders do carry
odd-grade terms, so the statement is scoped to the parity-conserving fermionic
sector and nowhere else. No wording that treats the hyperbit, the `m`-ball, or a
spin factor as a state space of this package's systems, a representation, or a
compression of anything computed here may survive review.

What §3.6 does claim is three operator facts — closure under unit combination for
`m ≥ 2`, an exact floor on the *intrinsic* variance, and a `2n+1` ceiling — and
the ceiling is a *limit* on the measurement lever, not evidence for it. The
variance floor is not a bound on any estimator's sampling variance and may not be
used to gate finite-shot data against `1`. No claim to originate unitary partitioning, contextual-subspace projection,
anticommuting-set measurement reduction, or the maximality of anticommuting sets;
§11 rows 15–18 record who owns each. Phase 19's sizing numbers are an in-session
probe with no committed producer, config, record or checker, and may not be cited
as a measurement or carried into any table in §6.

**On the priceability screen (R3S).** It is not a price, and it does not establish
which instances are priceable. Its ceiling is an operational admission threshold
calibrated on one priced bank, not a demonstrated necessary condition; coefficient
magnitudes, grouping, estimator variance and pencil conditioning all bear on whether
a bank resolves, so an admission is not a demonstration that a bank will resolve and
a rejection is not a demonstration that it cannot. Its margin factor is declared, not
preregistered, and the record says so. Its quantities are deterministic
double-precision compared to tolerance, not exact arithmetic. No wording that
upgrades any of these may survive review.

**On this project's own prior work.** P1 and P2 (§1.2) are public arXiv preprints. Their results
— exact-arithmetic subspace compactness, the `7371 → 2240` word bank, the
`913 → 64` setting reduction, the 68.9% covariance-aware variance reduction — are
not available for re-announcement, and a successor manuscript that reports them as
new would be self-plagiarism regardless of intent.

**On prior art.** No claim to originate QSCI, selected CI, overlap-guided adaptation, Haar
transforms, folded-spectrum methods, real-time Krylov, fully commuting measurement, the BK
mapping, CEO operators, MORE-ADAPT, or embedding. The possible contributions are narrower: a
resource-honest comparison of sampled determinant and measured operator subspaces; a tested
hybrid whose distinction from classical determinant closure is made explicit; a
support-pruned multiresolution staging policy built on the existing virtual-configuration
transform; and certified measurement and convergence diagnostics around those methods.

**On hardware.** No device-runtime prediction: the cost model is a logical model plus a
declared card, with no routing, no crosstalk, no mitigation, and a depolarizing surrogate
standing in for a noise simulation. No mapping, protocol, or restriction is preferable in
general — every `k*` is a function of an instance, an accuracy target, and a card. Lower
Pauli weight does not imply lower `W`, fewer settings, or lower cost; §6.2 says the first
implication is false by construction. And hardware-aware accounting is not assumed to favour
A-CASE: the honest form of the target statement is that the subspace-level resource claims
are *mapping-invariant by construction*, and the open question is whether the
measurement-compilation layer, which is not invariant, preserves or erodes them.

**On failure.** Failure remains acceptable. If QSCI or classical selected CI dominates
chemistry and the hybrid span collapses to ordinary excitation closure without a compactness
advantage, Paper B must narrow its claim rather than hide the comparison.
