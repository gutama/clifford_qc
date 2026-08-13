# Hardware-aware resource accounting for A-CASE

Fermion mapping, measurement protocol, and structural restriction as three
separate axes — with the cost quoted at matched certified accuracy.

This document revises and schedules a ten-point improvement proposal that came
out of a discussion of CS-QSE, Jordan–Wigner versus Bravyi–Kitaev, and
measurement depth. It does not open a new track. It is the executable form of
`ACASE_RESEARCH_PLAN.md` §6 (resource accounting) and it numbers its work
**R1–R4** inside the roadmap phases that already own the material:
`LITERATURE_ROADMAP.md` Phase 13 (structural invariant), Phase 14 (QWC plus
fully commuting groups), and Phase 17 (mapping validation).

The proposal's stated goal — move resource accounting from "how many Pauli
words" to hardware-aware measurement cost, and test whether structural
compression still pays once adaptive selection has run — is kept intact. What
changes is the order, four of the definitions, and the estimate of how much of
it is new work. §12 maps every source point to its disposition.

---

## 1. What the repository already has

Three of the ten proposed items are extraction jobs on shipped code, not new
experiments. Buying them again would produce a second, disagreeing cost model.

| capability | where | what it already reports |
|---|---|---|
| QWC grouping, memoized on the word set | `measurement/grouping.py` | settings `G`, per-group shared basis |
| Dyadic block-commuting hierarchy `k ∈ {1,2,4,8}` with stim-synthesized block-local Clifford diagonalizers | `benchmarks/run_clifford_hierarchy.py` | `G`, `W/G`, logical CX per sweep, mean/max two-qubit depth, `H`/`S` counts per sweep, state preparations at uniform shots, break-even ratio `c_CX/c_prep` |
| Frozen hierarchy records for two eight-qubit systems | `benchmarks/reference_results/clifford_hierarchy_{h4,beh2}.json`, schema `clifford_qc.clifford_measurement_hierarchy.v2` | the table above, plus the `Z`-only diagonalization invariant |
| Cross-setting shot pooling and rank-by-confidence | `measurement/cache.py`, `measurement/session.py`, `FINITE_SHOT_RETHINK.md` | words read by 3.72 settings on average (up to 23); median error `4.48 → 2.56` mHa, RMSE `1786 → 11.70` mHa at *identical* shots |
| Word-universe and support ledger | `subspace/projection.py` `resources` | `M`, `W`, `S_A`, `S_H`, operator products, assemble time |
| Clifford conjugation of Pauli words, `P ↦ (phase, P')` | `bridges/stim_bridge.py` `CliffordMap` | exactly the primitive the mapping axis needs |
| Reference-aware sector diagnostics | `subspace/symmetry.py` | `(N, S_z)` leakage — **hard-coded to the Jordan–Wigner form**, see §4.3 |

So R1 is mostly *promotion*: lift the cost model out of one benchmark script
into a library module with a declared device card, and add the two things that
are genuinely missing — accuracy matching and a fidelity term.

---

## 2. Four corrections to make before anything is measured

### 2.1 Cost is only meaningful at a fixed certified accuracy

The proposal's shot-weighted cost holds shots fixed and compares circuits. The
finite-shot rethink already refutes that framing on this codebase: at
104,000 setting-shots on the frozen four-qubit TFIM bank, changing nothing but
the estimator (single-assignment → pooled) and the rank rule moved the RMSE
from `1786` to `4.86` mHa. A cost figure quoted at fixed shots is therefore a
figure about the estimator, not about the hardware, and it can be moved by two
orders of magnitude without touching a circuit.

Every cost in this plan is `C(ε)`: the cost of reaching a **certified**
interval of half-width `ε` on the target Ritz value, at declared coverage, with
the estimator and rank rule named in the record. Where certification abstains,
the row reports `C(ε) = ∞` with the abstention reason rather than a number
obtained by dropping the certificate.

Consequence for the protocol comparison: `N_g` is not an input, it is the
output of a shot-to-target search. Two protocols are compared by
`(C(ε), ε, coverage, abstention rate)`, never by settings alone.

### 2.2 Half of the JW/BK comparison is a theorem

For the linear (encoding-matrix) family of fermion-to-qubit maps — Jordan–Wigner,
parity, Bravyi–Kitaev, and segment codes — two encodings differ by an invertible
`GF(2)` change of basis on occupation vectors, realized on qubits by a **CNOT
network**. A CNOT network is Clifford, so the two Hamiltonians are related by
`H_B = U H_JW U†` with `U` Clifford, and conjugation by a Clifford is an algebra
automorphism that maps Pauli words bijectively to Pauli words up to sign.

Everything A-CASE's ledger counts at the *operator* level is therefore invariant
by construction, not by experiment:

| quantity | under `U · U†` | why |
|---|---|---|
| spectrum, Ritz values, `M`, rank, `κ_S` | **invariant** | same operator, similarity transform |
| `W` (element-operator word universe), `S_A`, `S_H`, `nnz` | **invariant** | word-to-word bijection, products map to products |
| per-word variance `1 - μ_w²` | **invariant** | `μ_w` invariant up to the conjugation sign |
| full-commutation conflict graph (`k = n`) | **isomorphic** | commutation is preserved |
| Pauli weight distribution `w̄, w₅₀, w₉₀, w_max` | **variant** | weight is not Clifford-invariant |
| QWC compatibility, `G(k)` for `k < n` | **variant** | qubit-wise commutation is basis-dependent |
| single-qubit rotation count, CX count, two-qubit depth | **variant** | synthesis depends on the tableau |

This upgrades the source proposal's hypothesis `W_JW ≈ W_BK` to `W_JW = W_BK`
**exactly**, and it changes what the benchmark is for. A measured difference in
`W`, `M`, `κ_S`, or energy between mappings is not a finding — it is a bug in
the transformation, the reference, or the generator pool, and the run must stop.
The empirical question narrows to the quantities in the bottom half of the
table, all of which live in the measurement-compilation layer.

Two riders, both of which must be in the record:

- **The greedy coloring is not canonical.** At `k = n` the conflict graph is
  isomorphic, so the *chromatic number* is invariant, but the largest-degree
  greedy partition can differ by tie-breaking. The check compares
  degree-sequence and component invariants of the conflict graph, and reports
  `|G_JW − G_BK|` at `k = n` as tie noise with its own tolerance — not as a
  protocol difference.
- **Diagonalizer cost is not invariant even where the grouping is.** If `V_g`
  diagonalizes a group, `V_g U†` diagonalizes its image, so
  `cost_BK ≤ cost_JW + cost(U)` — a bound, not an equality, because synthesis
  starts from the image tableau rather than composing. Report the bound
  alongside the measured cost; a measured `cost_BK` far above the bound means
  the synthesizer, not the mapping, is what is being measured.

### 2.3 Qubit reduction is where the real gain is, and it is a different operation

The literature's Bravyi–Kitaev advantage is largely the two-qubit reduction, and
that is *not* the Clifford conjugation above: it is conjugation **plus fixing**
stabilizer qubits to `±1` eigenvalues and deleting them. Deletion is where `W`
can genuinely fall (distinct words collide on fewer qubits), where `n` falls,
and where every downstream cost falls with it.

So the mapping axis has three arms, not two: `JW`, `BK` (no reduction), `BK+2q`
(reduced). Comparing `JW` at `n` against `BK+2q` at `n−2` and attributing the
difference to "the mapping" is the single most likely way for this study to
publish a confound. Parity is included as a fourth arm because it makes the same
reduction available with a different weight profile, which separates "reduction
helps" from "BK's tree structure helps".

### 2.4 Fewer settings is not automatically cheaper, in two independent ways

**Pooling coverage.** Under the pooled estimator a word is read by every
compatible setting, so at a fixed *total* budget `N_total = G·N_g` the effective
shots on word `w` are `f_w · N_total` with the coverage fraction
`f_w = m_w / G` (`m_w` = settings that record `w`). Coarser protocols raise
`W/G` — the frozen H₄ record goes `8.07 → 115.17` from `k = 1` to `k = 8` — and
`f_w` moves with it in a direction no one has measured, because the frozen
hierarchy was produced under the single-assignment estimator. The protocol trade
must be re-measured with `pooling='shots'` before any `k*` is claimed; the
existing table prices circuits correctly and prices variance under an estimator
the project has since superseded.

**Fidelity.** A depth-versus-time model with no error model monotonically
prefers the deepest protocol, because it only ever removes state preparations.
Depth actually enters through infidelity: a diagonalizer with `N_2q` entangling
gates at error `ε_2q` returns readings whose contrast is damped, and an unbiased
estimator pays for that with an effective-shot penalty (§3.3). Protocols whose
damping exceeds a declared floor are **inadmissible**, not merely expensive, and
the report must show the admissible region rather than an unconditional winner.

---

## 3. The cost model

### 3.1 Device card — a declared, versioned artifact

No scalar cost may be printed without one. Schema `clifford_qc.device_card.v1`,
stored beside the benchmark configs and stamped into every record that consumes
it:

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

The all-zero-error, all-to-all card reproduces the current logical model
exactly, which is what makes the extension checkable: R1 must regenerate the
frozen `clifford_measurement_hierarchy.v2` numbers under it, digit for digit.

### 3.2 Time cost

For each setting `g` the synthesized diagonalizer supplies `N_1q,g`, `N_2q,g`,
`D_1q,g`, `D_2q,g`; the readout covers all `n` qubits regardless of group:

```
C_time(ε) = Σ_g N_g(ε) · [ t_prep + t_1q·D_1q,g + t_2q·D_2q,g + t_ro + t_reset ]
```

`N_g(ε)` comes from the shot-to-target search of §2.1, so the protocol that
needs more shots per setting pays for it here.

**The A-CASE-specific term, and why the existing "preparations" currency is
conditional.** A-CASE measures every setting on *one* reference state, so
`t_prep` is a constant that factors out of the sum: `C_time = N_total·t_prep +
Σ_g N_g·(measurement + readout + reset)`. On the current pipeline the reference
is a Hartree–Fock determinant — a computational basis state, prep depth zero up
to `X` gates — and under any linear encoding it stays a computational basis
state, so the prep term is very nearly free. That inverts the usual VQE
accounting in which preparation dominates: for A-CASE the currency the frozen
table reports (`state_preparations_at_uniform_shots`, `7.304 M → 0.512 M` on H₄)
is the *right* currency only on a device where reset and readout are cheap
relative to preparation. On a superconducting card where `t_ro + t_reset`
dominates, `C_time` tracks total executions rather than preparation depth, and
the deep-protocol advantage is close to its maximum. On a trapped-ion card with
slow gates and fast state prep, it is close to its minimum. This is precisely
why §3.4 refuses to print one number.

### 3.3 Fidelity and admissibility

Per setting, with a depolarizing surrogate:

```
F_g = (1-eps_1q)^{N_1q,g} · (1-eps_2q)^{N_2q,g} · (1-eps_ro)^n
```

An unbiased estimator built on damped readings inflates variance by `F_g^{-2}`,
so the accuracy-matched shot count carries `N_g(ε) ∝ F_g^{-2}`, and the record
reports the **effective** shots as well as the raw ones. A setting is
inadmissible when `F_g` falls below the declared floor (default `0.5`, recorded
per run), because below it the mitigation assumption, not the shot count, is
doing the work. Runs report the admissible `k` set before reporting `k*`.

This is deliberately a *surrogate*, not a noise simulation. It exists so that
the cost model cannot recommend a protocol that a device could not execute;
claiming a calibrated error prediction from it would be exactly the
overreach `ACASE_RESEARCH_PLAN.md` §9 forbids.

### 3.4 What may be printed

- **Default output: a break-even surface.** The frozen table's
  `c_CX/c_prep` column is the one-parameter version of this and is kept:
  H₄ breaks even at `0.423`, BeH₂ at `1.857` for `k = 2`. The generalization is
  the admissible region and the `k*` boundary over the
  `(t_2q/(t_ro+t_reset), ε_2q)` plane, with the accuracy target fixed.
- **A scalar `C_time(ε)` only under a named device card**, with the card's name
  and hash in the row. A cost with no card is a schema error, not a default.

### 3.5 The metric ledger

`ACASE_RESEARCH_PLAN.md` §6 keeps `M`, `W`, `S_A`, `S_H`, `r_S`, `κ_S`, build
time, word reuse, and group count. This plan adds, and requires the axis on
which each is measured:

| metric | measured on |
|---|---|
| `w̄, w₅₀, w₉₀, w_max` | three distinct multisets, reported separately: (a) Hamiltonian words, (b) the A-CASE element-operator universe `⋃ supp(A_i†HA_j)`, (c) per-setting support. The source proposal conflates them; (b) is the one A-CASE actually pays for, and it is *not* predicted by (a) — Z-string products cancel structurally under JW, so the mapping's weight advantage at (a) may not survive to (b). Pre-registered as an open direction in R2-P4. |
| `G(k)`, `W/G`, coverage `f_w` (mean, min) | per protocol rung |
| `N_1q`, `N_2q`, `D_1q`, `D_2q` (mean, max) | per setting, summed per sweep |
| `F_g`, `N_eff`, admissibility | per setting, under the device card |
| `N_g(ε)`, `C_time(ε)`, `ε`, coverage, abstention | per arm, at the accuracy target |
| encoding, reduction, taper qubits, `n_eff` | per mapping arm |

---

## 4. One primitive: Clifford rotation plus stabilizer fixing

### 4.1 The observation

Three of the source proposal's separate work items are the same object:

| operation | Clifford part | fixing part |
|---|---|---|
| JW → BK / parity | CNOT network of the encoding change | none |
| BK/parity two-qubit reduction | same | fix the two symmetry qubits to `±1` |
| `Z₂` symmetry tapering | Clifford mapping each symmetry generator to a single `Z` | fix those qubits to the reference's eigenvalue |
| Contextual-subspace restriction | Clifford rotations of the noncontextual stabilizers | fix the stabilizer qubits |

So R2 and R4 share one implementation: `clifford_qc/subspace/restriction.py`
exposing a `Restriction` that carries `(clifford_program, fixed_qubits, signs)`
and transports Hamiltonian, reference, generator pool, and observables together.
`CliffordMap.conjugate` already provides the word action; what is missing is the
fixing step and the bookkeeping that keeps the four objects consistent.

Building it once means the invariance checks of §2.2 are written once and every
consumer inherits them, and it means CS-QSE arrives as a configuration of an
already-tested primitive instead of a second transformation stack.

### 4.2 Risks this primitive carries

Sign conventions on fixed qubits, phase accumulation in `CliffordMap.conjugate`,
and the reference state's image are the three places this class of code goes
wrong silently. Each gets an explicit test *before* any benchmark consumes it:
`⟨HF_B|H_B|HF_B⟩ = ⟨HF_JW|H_JW|HF_JW⟩`, spectrum equality on the sector, and
word-multiset bijection.

### 4.3 The hidden cost the source proposal does not price

The package identifies the computational basis with the occupation-number basis
in several places that are correct **only under Jordan–Wigner**:

- `fermion.py` `total_number_op` / `total_sz_op` build `n_j = (I − Z_j)/2`
  directly, and `subspace/symmetry.py` `sector_operators` consumes them — so
  `sector_leakage`, `reference_sector_leakage`, `infer_reference_sector`, and
  `subspace_sector_certificate` are JW-specific.
- `backends/sector_statevector.py` groups by X-mask over occupation strings.
- `subspace/qsci.py` and `subspace/selected_ci.py` read bitstrings as
  determinants.

Under BK or parity these are wrong by construction, and wrong in the quiet way:
they return numbers. `ACASE_RESEARCH_PLAN.md` §5 already records what that
failure mode costs — the Kitaev rung where a misapplied sector filter rejected
all 72 candidates and would have published as "A-CASE cannot grow on a
frustrated spin cluster".

R2 is therefore gated on one of two choices, declared per arm in the record:
either the sector layer is transported through the `Restriction` (the number
operator's image is still diagonal, so this is mechanical but must be tested),
or the mapping arm runs with the sector filter **off** and says so. No mapping
arm runs with a JW sector filter silently applied to non-JW operators.

---

## 5. Phase R1 — cost infrastructure, no solver change

**Deliverables.** `clifford_qc/measurement/cost.py` (device card loader, per-
setting gate/depth/fidelity accounting, `C_time`, admissibility, break-even
surface); `benchmarks/configs/device_cards/*.json` with at least
`logical-alltoall`, one superconducting-like and one ion-like card;
`run_clifford_hierarchy.py` extended to schema
`clifford_qc.clifford_measurement_hierarchy.v3` carrying the §3.5 columns for
both estimators; `benchmarks/check_clifford_hierarchy.py` extended; a
`REPRODUCING.md` entry.

**Gates.**

1. Under `logical-alltoall`, every v2 field regenerates digit for digit. A
   changed number here is a regression in the extension, not a finding.
2. The pooled arm is reported beside the single-assignment arm on the same
   bank. If coverage `f_w` moves the accuracy-matched cost ordering between `k`
   rungs, that is R1's headline result and it lands before any mapping work.
3. No solver, selector, or certificate code changes in R1.

---

## 6. Phase R2 — the mapping axis

**Arms.** `JW`, `parity`, `parity+2q`, `BK`, `BK+2q`. Held identical across
arms: Hamiltonian and active space, reference determinant, generator family and
its enumeration order, growth budget, accuracy target, estimator, rank rule,
seed.

**Step 1 — invariance, as checks.** Construct the encoding change as a CNOT
network, verify it is Clifford through `clifford_tableau`, then assert:
spectrum on the sector, reference energy, `W`, `S_H`, `M`, `κ_S`, Ritz values,
and the word-multiset bijection. Failure stops the phase. This is the Phase 13
discipline — structural invariant first — applied to mappings.

**Step 2 — measure only the variant quantities.** Weight distributions on all
three multisets, `G(k)` across the protocol rungs, `N_1q`/`N_2q`/`D_2q`,
coverage, and `C_time(ε)` under each device card.

**Pre-registered predictions and their falsifiers.**

- **P1.** `W_JW = W_BK` and `κ_S`, `M`, Ritz values identical to solver
  tolerance for the non-reduced arms. *Falsifier:* any difference — and it
  indicts the implementation, not the hypothesis.
- **P2.** `w̄` on the Hamiltonian multiset falls under BK relative to JW, most
  visibly at larger `n`. *Falsifier:* no reduction at `n = 8, 12`, which would
  mean the asymptotic argument has no purchase at the sizes this project runs —
  itself a publishable negative for a chemistry-scale claim.
- **P3.** At QWC (`k = 1`) the mapping's effect appears in **single-qubit gate
  count and group count, not depth**: the basis rotation is one layer whatever
  the weight, and the frozen record already carries `gate_counts_per_sweep`
  (H₄: 14 608 `H`, 15 552 `S` at `k = 1`) as the place it shows. *Falsifier:*
  a material `D_1q` difference at `k = 1`, which would mean the synthesizer is
  not emitting a single rotation layer.
- **P4.** The weight advantage attenuates from multiset (a) to multiset (b):
  the element-operator universe is built from products `A_i†HA_j`, and JW's
  Z-strings cancel structurally in products. *Falsifier:* the ratio
  `w̄_BK/w̄_JW` is the same on (a) and (b) — in which case Hamiltonian weight is
  a sufficient proxy and the extra bookkeeping can be dropped.
- **P5.** `G(k=n)` agrees between mappings within tie-break noise, and the
  mapping's `C_time` gap closes monotonically as `k → n`. *Falsifier:* a gap at
  `k = n` beyond the §2.2 bound, which means either the coloring is not
  comparing isomorphic graphs or the diagonalizer synthesis is the dominant
  effect.
- **P6.** The `+2q` arms beat their unreduced parents on every cost column, and
  by more than the `JW → BK` difference at fixed `n`. *Falsifier:* reduction
  worth less than the encoding change, which would invert the plan's advice on
  where to spend effort.

**The claim this phase is allowed to make.** Not "BK is shallower". Either
"the mapping changes measurement cost by `X%` at protocol `k` on device card
`D`, while leaving the subspace and `W` provably unchanged", or "it does not,
at the sizes measured".

---

## 7. Phase R3 — the protocol axis, and `k*` defined

The protocol axis already exists as the dyadic hierarchy `k ∈ {1,2,4,8}`; R3 is
the `mapping × k` grid under the R1 cost model, not a new protocol. The source
proposal's "two regimes" (QWC and general commuting) are its endpoints, and
running the interior rungs is what makes the "BK is always shallower" claim
falsifiable rather than rhetorical.

**`k*` is defined as**

```
k*(ε, device card, instance) = argmin_{k admissible} C_time(ε; k)
```

— the smallest-cost *admissible* block size at a fixed certified accuracy on a
named device card, reported with the argmin's margin over its neighbours. It is
not a function of `W` alone, it is not an instance-independent constant, and it
is reported as a region over the break-even plane whenever the margin is within
the estimator's own uncertainty. The frozen records already show why a scalar
would mislead: at `k = 2` the break-even ratio is `0.084` on H₄ and `1.857` on
BeH₂ — a 22× instance difference under an identical logical model, before any
device enters.

---

## 8. Phase R4 — CS-QSE as comparator, then and only then as preconditioner

**Arms, at matched accuracy target and matched candidate family:** full QSE,
CS-QSE, A-CASE, CS + A-CASE.

**Report the bias floor, not only the compression.** The contextual restriction
is an approximation whose error is not variationally controlled by the restricted
solve; a rung where the restriction's own floor exceeds the accuracy target
cannot be compared to an unrestricted arm on cost, because it never reaches the
target. Every CS arm reports (i) the restricted-space exact energy against the
same exact reference used elsewhere on that rung, and (ii) the qubits removed.
An arm that cannot reach `ε` reports `C(ε) = ∞` and its floor.

**The interaction question, made measurable.** "Does A-CASE still compress after
CS?" becomes a multiplicativity test. With `C₀` the full-QSE cost,

```
C₀/C_CS+ACASE  =  (C₀/C_CS) · (C_CS/C_CS+ACASE)
```

and the interaction is the deviation of the observed joint ratio from the
product of the marginals, computed in logs with the shot-search uncertainty
propagated. Three pre-registered outcomes:

- **Complementary** (joint ≥ product, within uncertainty) — the two compressions
  attack different structure, and CS-preconditioned A-CASE is worth building.
- **Redundant** (joint ≈ larger marginal) — CS removes what A-CASE would have
  pruned; report it and do not build the preconditioner.
- **Antagonistic** (joint < larger marginal) — restriction removes directions
  adaptive selection needed; this is the interesting negative and it belongs in
  the paper.

**Gate before any solver change.** Complementarity must appear in `C(ε)`, not
only in `M` or `W`. A drop in `M` that leaves the accuracy-matched cost flat is
exactly the reading §6 of the main plan exists to block — the ladder has already
produced one such case, where A-CASE cut `M` from 9 to 6 with `W = 140`
unchanged and `κ_S` marginally worse.

The three compression classes stay separated in every table, as the source
proposal asks: Hilbert-space compression (qubits, sector, restriction),
operator-pool compression (`M`, candidates evaluated), and
measurement/hardware compression (`W`, `G`, weight, gates, depth, `C_time`).
CS-QSE attacks the first two; word reuse, pooling, and grouping attack the
third; a table that sums them into one number cannot show which mechanism paid.

---

## 9. Benchmark inventory — what each rung can support

| rung | `n` (JW) | exact reference | frozen artifact | role in this plan |
|---|---:|---|---|---|
| H₄ `r = 0.9` | 8 | yes | `matched_h4.json`, `clifford_hierarchy_h4.json` | primary; the frozen bank both axes reuse |
| BeH₂ CAS(4e,4o) | 8 | yes | `clifford_hierarchy_beh2.json` | second instance; already shows the 22× break-even spread |
| H₂O CAS(4e,4o) stretched | 8 | yes | ladder | strong correlation control |
| H₂O CAS(8e,6o) | 12 | yes (ladder) | ladder | size stress for `W` and grouping |
| Hubbard 2×2 / 2×3 | 8 / 12 | yes | ladder | strongly correlated control, non-molecular weight profile |
| Kitaev 2×2 | 8 | yes | ladder | **excluded from the mapping axis** — one qubit per site, no fermionic encoding behind it |
| TFIM `n = 4` | 4 | yes | `finite_shot_rethink.json` | the accuracy-matched cost search is calibrated here first |
| HCl | 20 | **no** | — | see below |

**HCl.** STO-3G HCl is ten spatial orbitals, so twenty qubits under JW — past
this project's exact-reference reach, and the ladder already runs without
ADAPT/Krylov arms at `n = 12`. It enters on one of two terms, declared in the
row: with an explicit active space small enough for an exact reference (then it
is a full rung), or **resource-only** — weights, `W`, `G`, gates, depth, no
error column, no accuracy-matched cost, and an evidence label that says the row
carries no accuracy claim. It is worth running on the second term because the
CS-QSE literature reports its largest operator-pool compression there, and a
resource-only row can still falsify a compression claim. It is not worth
running on terms that let a reader mistake it for an accuracy result.

---

## 10. Excited states — deferred, with the reason

A-CASE's projected generalized eigensolver already returns multiple Ritz roots
(`acase_states`, `roots: 3` in `benchmarks/configs/acase_ladder.json`), so
root-aware or state-balanced selection is a natural extension and CS-QSE is a
relevant comparator there. Two things must land first.

**The metric changes.** The word universe is shared across roots — one bank,
one measurement campaign, `R` roots — so the right figure is the *amortized*
`C(ε)` per root, and that is where A-CASE should be structurally ahead of
running `R` separate variational optimizations. Report `C(ε)/R` beside the
per-root accuracy, never the total alone.

**The certificate does not cover it.** The current finite-shot machinery
certifies a ground-state Ritz interval; interior roots need interval statements
that the delta-method scoring in `SharedMeasurement.solve_selected_rank` does
not supply, and
inventing one is theory work, not benchmark work. Until then, excited-state rows
carry heuristic evidence labels and no certified cost. This phase is scheduled
after R1–R4 and is explicitly not on the critical path.

---

## 11. Falsifiable questions

- **QR1 (accounting).** Does the accuracy-matched cost `C(ε)` ever reorder the
  protocol rungs relative to the settings-count ordering? *Falsifier:* the
  ordering is identical on every rung and card, in which case settings count was
  an adequate proxy and this plan's machinery is overhead — record it and say so.
- **QR2 (mapping invariance).** Do the §2.2 invariants hold exactly across
  mappings on every rung? *Falsifier:* any violation, which halts R2 as an
  implementation defect.
- **QR3 (mapping cost).** Is there a device card and protocol rung at which the
  mapping changes `C(ε)` by more than the instance-to-instance spread already
  present between H₄ and BeH₂? *Falsifier:* the mapping effect is smaller than
  the instance effect everywhere — which would demote fermion mapping from an
  optimization dimension to a footnote, and that is a useful result.
- **QR4 (pooling × protocol).** Does the coverage fraction `f_w` change the
  `k*` chosen under the pooled estimator relative to the single-assignment one?
  *Falsifier:* identical `k*` under both, which retires the concern.
- **QR5 (compression interaction).** Is the CS × A-CASE cost ratio
  multiplicative, sub-multiplicative, or antagonistic? *Falsifier for the
  preconditioner plan:* anything but complementary.
- **QR6 (weight propagation).** Does the Hamiltonian-level weight advantage
  survive into the element-operator universe? *Falsifier:* equal ratios on both
  multisets.

---

## 12. Disposition of the source ten points

| # | source point | disposition |
|---|---|---|
| 1 | extend metrics; scalar `C_meas = Σ_g N_g(D_prep + D_meas,g + D_readout)` | **kept, redefined.** Metrics adopted in full (§3.5). The scalar is replaced: depths are converted to time through a declared device card, fidelity and admissibility are added, and `N_g` is solved for at a fixed certified accuracy rather than assumed (§2.1, §3). |
| 2 | JW vs BK on an identical problem | **kept, restructured.** Half becomes invariance checks (§2.2); the hypothesis sharpens from `W_JW ≈ W_BK` to exact equality; a third and fourth arm are added so qubit reduction is not confounded with the encoding (§2.3, §6). |
| 3 | separate QWC and general-commuting regimes | **already shipped, extended.** The dyadic hierarchy `k ∈ {1,2,4,8}` covers both endpoints and their interior; R3 adds the mapping cross and the cost model (§7). |
| 4 | mapping as its own optimization dimension | **kept, with the hidden cost priced.** The JW-specific sector and occupation-basis layers must be transported or explicitly disabled per arm (§4.3). |
| 5 | CS-QSE as comparator, then preconditioner | **kept, gated.** Comparator arms and the multiplicativity test as specified; the preconditioner is gated on complementarity in `C(ε)`, not in `M` or `W` (§8). |
| 6 | keep three compression classes distinct | **kept as written** (§8). |
| 7 | broaden molecules (H₂O, HCl, H₄, Hubbard) | **kept, scoped.** HCl is twenty qubits under JW and enters resource-only unless a declared active space gives it an exact reference (§9). |
| 8 | excited-state track after ground state | **kept, deferred with a stated blocker** — the certificate does not cover interior roots (§10). |
| 9 | `k*` from compiled cost, not word count | **kept, defined.** `k*` is the argmin over admissible block sizes at fixed certified accuracy on a named card, reported as a region with a margin (§7). |
| 10 | implementation order | **revised.** Cost infrastructure is promotion of shipped code, not new construction, and the pooled-estimator re-measurement of the existing hierarchy comes *before* the mapping work, since it can reorder the protocol rungs the mapping study would otherwise be measured against (§13). |

---

## 13. Execution order and go/no-go gates

1. **R1 — cost model on the frozen banks.** Device card, `measurement/cost.py`,
   hierarchy record v3, both estimators, no solver change.
   *Gate:* v2 regenerates exactly under `logical-alltoall`; QR1 and QR4 answered
   on H₄ and BeH₂.
2. **R2a — restriction primitive and invariance checks.** `restriction.py`,
   encoding networks, the §4.2 tests, the §4.3 decision recorded per arm.
   *Gate:* QR2 passes on every rung. No cost numbers published from a run whose
   invariants failed.
3. **R2b — mapping measurements** on H₄, BeH₂, H₂O CAS(8e,6o), Hubbard.
   *Gate:* QR3 answered with the instance spread as the comparison scale.
4. **R3 — `mapping × k` grid and `k*` regions** under three device cards.
   *Gate:* margins reported; regions, not integers, wherever the margin is
   inside the shot-search uncertainty.
5. **R4a — CS comparator arms** with bias floors.
   *Gate:* QR5 answered.
6. **R4b — CS-preconditioned A-CASE**, built only on a complementary QR5.
7. **Excited-state track**, after the certificate question of §10 has an answer.

Each phase ships the project's standard triple: a `run_*.py` producer, a
stamped `reference_results/*.json` record with an explicit `schema` string, and
a `check_*.py` that regenerates and compares it, plus its `REPRODUCING.md`
entry and its evidence labels.

---

## 14. What this plan does not claim

It does not claim a device-runtime prediction: the cost model is a logical model
plus a declared card, with no routing, no crosstalk, no mitigation, and a
depolarizing surrogate standing in for a noise simulation. It does not claim
that any mapping, protocol, or restriction is preferable in general — every
`k*` in it is a function of an instance, an accuracy target, and a card. It does
not claim that lower Pauli weight implies lower `W`, fewer settings, or lower
cost; the invariance argument of §2.2 says the first implication is false by
construction. And it does not claim that hardware-aware accounting will favour
A-CASE: the honest form of the target statement is that the subspace-level
resource claims are *mapping-invariant by construction*, and the open question
is whether the measurement-compilation layer, which is not invariant, preserves
or erodes them.
