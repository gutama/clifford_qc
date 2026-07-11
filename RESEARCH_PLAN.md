# `clifford_qc` research plan — confidence-certified, measurement-efficient ADAPT-VQE

This plan turns `clifford_qc` from a validated operator-centric Cl(2n,C)
engine into a research platform that answers one focused question:

> **Can algebraic symmetry, shared Pauli-word structure, and stabilizer
> information make ADAPT-VQE operator selection statistically reliable with
> materially fewer measurements and fewer non-Clifford operations?**

It supersedes the free-form notes in `next_research.pdf`; `simple_plan.md`
(phases 1–5, all complete) remains the record of the core/bridge work this
builds on.

---

## 1. Positioning against 2023–2026 PRA literature

No published work duplicates the full combination here (density-multivector
formulation + exact adjoint gradients + odd-Y pool restriction + finite-shot
rank resolution), but three developments constrain the novelty claims:

| Work | Constraint on us |
|---|---|
| Magoulas & Evangelista, PRA 113 (2026); Evangelista & Magoulas, PRA 111, 042825 (2025) | Fermionic Clifford transformations exist; do **not** claim the first Clifford treatment of Pauli–Majorana–Dirac structure. Cite in the Clifford/JW sections. |
| Majland et al. (FAST-VQE), PRA 108, 052422 (2023); Long et al., PRA 109, 042413 (2024) | Measurement-efficient ADAPT selection is an active area; do **not** claim generic "measurement-efficient ADAPT-VQE". Cite in the finite-shot sections and use as baselines. |
| Cheng et al., PRA 111, 062413 (2025); Robin, PRA 112, 052408 (2025) | Clifford-point initialization and stabilizer/residual splitting exist; frame Paper B as combining them with the operator-centric representation. |
| Scriva et al., PRA 109, 032408 (2024) | Shot noise can dominate outer-loop cost; resource accounting must cover the full outer loop. Cite in limitations. |
| Yoo, Bae & Kim, PRA 111, 032615 (2025) | Symmetry-preserving ansätze are known; position odd-Y restriction as an *algebraic pool* symmetry with an exactness theorem, not as the first symmetry-aware pool. |

**Defensible novelty statement.** *We introduce an operator-centric
ADAPT-VQE framework in which commutator-gradient observables are represented
collectively in a sparse Clifford-algebra basis. This enables exact algebraic
pool reduction, global reuse of shared Pauli measurements, sequential
confidence-certified operator selection, and stabilizer-aware initialization,
all within one versioned Pauli-rotor intermediate representation.*

Avoid claiming: first Clifford formulation of QC; first Clifford fermions;
first measurement-efficient ADAPT; first symmetry-restricted pool; or a
"fundamentally faster" representation than matrices.

**Differentiator vs FAST-VQE**: FAST-VQE ranks by sampled determinant
populations (a proxy). We estimate the *exact* commutator selection
observable G_j = Tr[ρ·(−i/2)[H,P_j]] with (a) cumulative shot reuse,
(b) global shared-word measurement reuse, (c) simultaneous confidence bounds
with an explicit wrong-selection probability δ, (d) an explicit ambiguity
outcome instead of silently picking the empirical max, and (e) algebraic
(odd-Y / real-sector) pool reduction.

## 2. Outputs

1. **Paper A (priority): statistically certified, measurement-efficient
   ADAPT-VQE.** Builds directly on the existing finite-shot ADAPT code.
2. **Paper B (contingent): Clifford/stabilizer-seeded residual ADAPT-VQE.**
   Proceeds only if the go/no-go gate in §9 passes; otherwise it becomes a
   section or negative-result appendix of Paper A.
3. **Software: `clifford_qc` v0.2 → v1.0** as the reproducible
   implementation, with versioned IR and benchmark artifacts.

## 3. Research hypotheses (Paper A)

Each hypothesis has a falsifiable test; all comparisons are at matched
empirical wrong-selection rate.

- **H1 — Symmetry restriction.** For Hamiltonians and states that are real
  in the computational basis (antiunitary-real sector), restricting the pool
  to odd-Y Pauli words yields the *same exact* ADAPT trajectory as the
  unrestricted pool while shrinking candidate and word counts.
  *Test:* exact trajectories agree operator-for-operator on TFIM/XXZ;
  extend the TFIM observation to a proof for the antiunitary-real sector.
- **H2 — Shared-word measurement reuse.** The candidate observables
  G_j overlap heavily in Pauli words; measuring the union 𝒲 = ⋃_j supp(G_j)
  once per round and reconstructing every ĝ_j from the shared cache costs
  strictly fewer distinct measurements than per-candidate estimation.
  *Test:* unique-word counts and shot totals vs the per-candidate baseline
  at equal selection accuracy. (Most immediate practical win.)
- **H3 — Sequential selection guarantee.** A best-arm identification rule —
  select ĵ only when its simultaneous lower bound exceeds every other upper
  bound — controls Pr(wrong operator) ≤ δ for user-chosen δ.
  *Test:* empirical wrong-selection rate ≤ δ within sampling error over
  ≥100 seeds (calibration is the headline validation).
- **H4 — Adaptive allocation beats uniform doubling.** Spending new shots on
  words that dominate the variance of *unresolved pairwise gaps* resolves
  selection with fewer total shots than doubling everything.
  *Test:* median shots-to-resolution, uniform vs variance-weighted vs
  successive elimination.
- **H5 — Layering preserves selection quality at lower depth.** Adding a
  commuting layer of sufficiently strong candidates after the top selection
  reduces depth without degrading the energy trajectory.
  *Test:* energy vs two-qubit depth against single-operator ADAPT.

## 4. Statistical method

- **Global commutator bank.** Represent all selection observables in one
  sparse candidate-by-word coefficient matrix C = (c_jw), with
  ĝ_j = Σ_w c_jw μ̂_w and μ̂_w = (N_w⁺ − N_w⁻)/N_w. No per-candidate
  accumulators owning duplicate measurements.
- **Confidence bounds, two tiers.**
  *Tier 1 (robust default):* Jeffreys pseudocount p̃_w = (N_w⁺ + ½)/(N_w + 1),
  variance propagated through the linear estimator
  Var(ĝ_j) = Σ_w |c_jw|² Var(μ̂_w), Šidák/Bonferroni correction across
  candidates.
  *Tier 2 (publication-grade):* empirical-Bernstein / confidence-sequence
  bounds valid under adaptive stopping (shots chosen from past data).
- **Selection outcomes.** The selector returns one of
  `resolved_best`, `resolved_near_optimal`, `below_threshold`,
  `budget_exhausted_ambiguous`. Never silently pick the empirical max while
  ambiguity remains.
- **Allocation policies (interchangeable):** `uniform_fixed` (baseline),
  `uniform_doubling` (current algorithm), `variance_proportional`,
  `successive_elimination` (drop candidates whose upper bound falls below
  the best lower bound), `pairwise_gap`. Expected winner:
  successive elimination + shared-word cache.

## 5. Software architecture

Keep the numpy-only core inviolate. New layers are subpackages that import
nothing beyond numpy (SciPy optional, with a pure-Python Adam fallback):

```
clifford_qc/
  models/        # spin.py: tfim, xxz, random_ising (+ lmg later); Model dataclass
  backends/      # protocol.py, exact_mv.py (gate-by-gate MV), finite_shot.py,
                 # dense_statevector.py (reference), stabilizer.py (stim, extra)
  measurement/   # bank.py (commutator bank), cache.py (cumulative word counts),
                 # confidence.py, allocation.py, grouping.py (QWC, later)
  algorithms/    # optimize.py, vqe.py, adapt.py, pools.py,
                 # layering.py + initialization.py (later)
benchmarks/      # configs/, run_benchmark.py, summarize.py, schemas/, reference_results/
paper/           # manuscript.tex, figures/, tables/, bibliography.bib
```

Extras in `pyproject.toml`:
`research = [scipy>=1.11, pandas>=2.0, matplotlib>=3.8, networkx>=3.0]`,
`chemistry = [openfermion>=1.6, pyscf>=2.4]`.

### Backend protocol

```python
class Backend(Protocol):
    def state(self, program, values, initial_state=None) -> MV: ...
    def expectation(self, program, observable, values, initial_state=None) -> float: ...
    def sample_paulis(self, program, words, shots, values, seed) -> MeasurementBatch: ...
```

- `ExactMVBackend` — gate-by-gate density-multivector evolution (never the
  full program unitary), exact expectations, exact adjoint gradients
  (reusing `ir.adjoint_gradient`), per-gate support-size tracking.
- `FiniteShotBackend` — seeded binomial sampling of single-word outcomes
  from exact expectations; cumulative counts; deterministic experiments;
  later swappable for a PennyLane device.
- `DenseStatevectorBackend` — independent numerical reference via
  `matrix.to_matrix`; never presented as the native representation.
- `StimBackend` (extra) — named Cliffords *and Clifford-angle Pauli rotors*
  (θ ∈ πℤ/2), stabilizer expectations, discrete Clifford-point search at
  large n.

### IR upgrades required first

1. **Schema versioning**: `Program.to_dict()` writes
   `{"schema": "clifford_qc/program", "schema_version": 1, ...}`;
   `from_dict` accepts version-0 (headerless) dicts forever.
2. **Gate-by-gate execution**: `Program.state()` evolves ρ op-by-op instead
   of forming the whole unitary, so Pauli support stays as sparse as the
   circuit allows. `unitary()` remains for bridges/tests.
3. **Clifford-angle rotors**: `is_clifford_only()` (and the stim bridge)
   recognize rotors with bound angles in πℤ/2 as Clifford operations and
   lower them to stabilizer gates, validated against MV conjugation.

## 6. Core API

```python
result = run_vqe(model=tfim(n=8), ansatz=hva(depth=3), backend=ExactMVBackend(),
                 optimizer=LBFGSB(), x0=...)                      # -> VQEResult

result = run_adapt(model=model, pool=odd_y_pool(model),
                   backend=FiniteShotBackend(seed=7),
                   selector=ConfidenceSelector(delta=0.05),
                   allocator=SuccessiveElimination(),
                   max_operators=30)                              # -> AdaptResult
```

`VQEResult`: energy, parameters, gradient norm, evaluations, iterations,
support peak, wall seconds, metadata. `AdaptResult` carries one
`SelectionRecord` per step — including rejected/ambiguous steps — with
estimate, lower/upper bounds, exact gradient (when available), shots added,
cumulative shots, unique words measured, circuits executed, active
candidates, and status.

## 7. Benchmarks and baselines

**Stage 1 — spin systems**: open/periodic TFIM (n = 4…12, h/J sweep through
criticality), random-field Ising (30 disorder seeds), XXZ chain, LMG
(stabilizer path). The current n = 4 result becomes a golden regression
test, not the evidence.

**Stage 2 — chemistry** (via the existing OpenFermion bridge): H₂, linear
H₄, LiH and BeH₂ active spaces. FAST-VQE determinant-population selection is
only meaningful here, so the FAST-inspired baseline lives in stage 2.

**Baselines for every headline result**: exact-gradient ADAPT; fixed-shot
ADAPT; current cumulative-doubling ADAPT; shared-word cumulative ADAPT;
confidence-certified (successive-elimination) ADAPT; random selection;
subpool exploration; layered ADAPT; FAST-inspired selection (chemistry);
fixed-depth HVA/VQE.

**Resource accounting** — report separately, never just total shots:
N_shots (individual measurements), N_circuits (distinct measurement
circuits), N_words (unique Pauli expectations), D_2q (compiled two-qubit
depth), N_non-Clifford (non-Clifford rotation count), S_max (peak active
Pauli support), plus optimizer evaluations and classical preprocessing time.

**Statistical protocol**: ≥30 seeds exploratory, ≥100 seeds for headline
finite-shot comparisons; predeclare seeds, initialization, budgets,
tolerances, δ and near-optimality ε. Report wrong-selection probability,
top-k probability, selection regret R_t = |g*_t| − |g_selected,t|,
median/IQR shots, 95% CIs, ambiguity and failure rates. Headline check:
empirical Pr(wrong selection) ≤ δ within sampling uncertainty.

## 8. Paper B — stabilizer-seeded residual ADAPT (contingent)

Decompose H = H_stab + λV with H_stab admitting an efficiently preparable
stabilizer ground state; prepare it as a Clifford `Program`; run odd-Y ADAPT
restricted to non-Clifford residual rotations; count the non-Clifford
operations needed to reach target accuracy.

Initialization variants: **(A)** discrete Clifford-point search over HVA
angles θ ∈ {0, π/2, π, 3π/2} with a stabilizer backend (annealing / beam /
local search); **(B)** Hamiltonian stabilizer approximation — a mutually
commuting subset maximizing Σ_{w∈S} |h_w| with a consistent eigenspace.

Claim shape: *a classically optimized stabilizer scaffold reduces the
non-Clifford correction needed by ADAPT-VQE* — not "Clifford states solve
the problem". Metrics: non-Clifford rotor count, two-qubit depth, optimizer
evaluations, selection shots, energy error, ground-space overlap, support
growth, classical preprocessing cost.

## 9. Go/no-go criteria

**Paper A proceeds** when, at the same empirical wrong-selection rate and
across ≥3 benchmark families, the method achieves at least one of: fewer
shots; fewer distinct measurement circuits; materially fewer ambiguous
selections; or better final energy at a fixed measurement budget.

**Paper B proceeds separately** when stabilizer seeding gives ≥2× fewer
non-Clifford rotations, or ≥2× fewer optimizer evaluations, or materially
higher success probability at fixed shot/depth budget, on more than one
model class. Otherwise it folds into Paper A.

## 10. Schedule and backlog

Phases (each with an exit criterion; calendar per the six-month schedule,
Phase 0 starting 13 July 2026):

- **Phase 0 — consolidation** *(done in this branch)*: tag v0.1.0 baseline;
  IR schema versioning with a backward reader; gate-by-gate execution;
  Clifford-angle rotor recognition + stim lowering.
  *Exit:* all existing tests pass; old JSON still loads.
- **Phase 1 — exact research layer** *(done in this branch)*: backend
  protocol; `ExactMVBackend`; model builders; optimizer adapters (SciPy
  L-BFGS-B + pure-Python Adam fallback); fixed-depth HVA VQE; exact ADAPT;
  odd-Y/real-sector pools; support diagnostics.
  *Exit:* packaged VQE and exact ADAPT reproduce the standalone
  `clifford_qc_paper_code` results to numerical tolerance.
- **Phase 2 — measurement & confidence layer** *(core done in this
  branch)*: shot data structures; global commutator bank; shared word
  cache; uniform/variance allocation; simultaneous confidence intervals;
  successive elimination; explicit ambiguity outcomes; cost accounting.
  *Exit:* confidence calibration validated on synthetic Pauli means and
  small TFIM instances.
- **Phase 3 — scaling & layering** *(machinery done in this branch)*:
  commutation graph + layer construction (`algorithms/layering.py`),
  subpool exploration with dead-subpool redraw, QWC measurement grouping
  with joint-distribution sampling (`measurement/grouping.py`), a random
  selection baseline, and the config-driven benchmark matrix
  (`benchmarks/run_benchmark.py` + `summarize.py`, 30-seed reference
  results committed under `benchmarks/reference_results/`). Known finding:
  alpha-layering without operator repeats can stall at symmetric
  stationary points (pinned by a regression test) — quantify in the
  ablations. *Remaining:* scale the sweep to n = 6–12 and 100 seeds for
  the headline tables. *Exit:* Paper A tables and ablations.
- **Phase 4 — stabilizer initialization**: discrete Clifford-point search,
  stabilizer Hamiltonian approximation, residual ADAPT, large-n
  stabilizer-only benchmarks. *Exit:* Paper B go/no-go decision.
- **Phase 5 — chemistry & manuscript**: PySCF/OpenFermion molecules,
  FAST-inspired baseline, figures/tables, v0.3.0 release, submit Paper A.

Ordered backlog (issue → acceptance test):

| # | Issue | Acceptance test |
|---|---|---|
| 1 | feat(ir): schema/version header | old and new JSON both load |
| 2 | perf(ir): gate-by-gate state evolution | agrees with full-unitary execution |
| 3 | feat(backends): Backend protocol + ExactMV/FiniteShot | backends pass common tests |
| 4 | feat(models): TFIM, XXZ, random Ising (LMG later) | exact Hamiltonian golden vectors |
| 5 | feat(algorithms): fixed-depth VQE | reproduces current n=4 TFIM results |
| 6 | feat(algorithms): exact ADAPT-VQE | reproduces six-operator convergence |
| 7 | feat(pools): real-sector / odd-Y pools | restricted and unrestricted exact trajectories agree |
| 8 | feat(measurement): cumulative word cache | repeated allocations preserve old counts |
| 9 | feat(measurement): global commutator bank | scores agree with direct commutators |
| 10 | feat(measurement): simultaneous confidence bounds | synthetic coverage ≥ nominal |
| 11 | feat(selection): successive elimination | fixed-seed regressions; calibration ≤ δ |
| 12 | feat(measurement): QWC grouping | grouped and ungrouped expectations agree |
| 13 | feat(adapt): layered + subpool selection | exact energy non-increasing |
| 14 | feat(stim): Clifford-angle rotors | stim and MV conjugation agree |
| 15 | feat(initialization): Clifford-point search | discrete search reproducible by seed |
| 16 | bench: spin experiment matrix | one-command JSONL/CSV generation |
| 17 | bench: chemistry baselines | reference energies and mappings recorded |
| 18 | docs: reproducibility artifact | clean environment reproduces figures |

Items 1–14 and 16 are implemented (1–11 and 14 merged in PR #2; 12, 13 and
16 on this branch); 15 and 17–18 are the remaining Phase 4–5 work.

## 11. Non-goals

- No generic quantum SDK: no device plugins, transpilers, or vendor
  runtimes beyond the existing validated bridges.
- No new dependencies in the core import path (numpy only).
- No claims outside the novelty statement in §1.
- Paper B work does not start until Paper A's Phase 2 exit criterion holds.
