# Geometric commutator subspaces and resource-aware information acquisition

Two structural extensions to the A-CASE operator-response framework are
implemented and stress-tested here against their strongest non-geometric
comparators:

1. **Hierarchical commutator subspaces** ([`clifford_qc/subspace/commutators.py`](../clifford_qc/subspace/commutators.py)): recursive Lie-algebraic hierarchies 𝒜₀ = {A₁, A₂, …}, 𝒜₁ = {[H, Aᵢ]}, 𝒜₂ = {[H, [H, Aᵢ]]}, … and demand-driven [`CommutatorFrontier`](../clifford_qc/subspace/commutators.py) expansion tested against global Krylov subspaces, explicit excitation/Pauli pools, and classical selected-CI controls.
2. **Resource-aware information acquisition** ([`clifford_qc/subspace/information.py`](../clifford_qc/subspace/information.py)): candidate selection via

$$A^\star = \arg\max_A \frac{\mathbb{E}[\Delta E_A]}{C_{\mathrm{incremental}}(A) + \epsilon}$$

and an oracle-free closed-loop controller [`acquire_to_accuracy`](../clifford_qc/subspace/information.py) that allocates finite shots between basis expansion and Neyman-optimal shot refinement against a physical error proxy without inspecting the exact ground energy.

Every figure below comes directly from [`benchmarks/run_geometric_information_study.py`](../benchmarks/run_geometric_information_study.py) and its machine-readable output [`benchmarks/geometric_information_study.json`](../benchmarks/geometric_information_study.json) (total wall time 201.86 s).

| Experiment | Main finding | Boundary / negative result |
|---|---|---|
| **A1–A2** Spin commutators (`tfim6`, `xxz6`) | Lazy dynamic expansion cuts commutators by 48–84% and QWC groups by up to 69% vs `Static_d<=3`; hits −0.00 mHa on correlated `xxz6` at depth 2 | Global `Krylov6` dominates on fixed-budget energy (0.24–0.26 mHa at `M = 7` vs 177–1 715 mHa for local orbit seeds at `M ≤ 8`) |
| **A3** Hubbard & chemistry (`2×1`, `2×2`, `H₄`, `LiH`) | Demand-driven `[H, S]` heals Brillouin stalls on Hubbard `2×1` (0.000 mHa in 4 commutators) and `2×2` (1 274 → 1 057 mHa in 10 commutators) | Depth-1 singles commutators lie inside `span(S, D)` (residual ≤ 2.38 × 10⁻¹⁵), pay ~7 355–7 359 Pauli words on `H₄`/`LiH`, and lose to `SD_Pool`, `Krylov2`, and classical `IterSCI` |
| **A1** Fermionic truncation | Exact commutators have 0.000 operator `(N, S_z)` leakage across depths 1–2 | Generic Pauli tail clipping breaks `(N, S_z)` symmetry immediately on `H₄` (`0.1388` at `l2_keep=0.90`; `0.2318` at `top_k=16`) |
| **B1** Cost-ratio Pareto sweeps | `QWCCost` / `TimeCost` on `tfim5` cuts QWC groups 69 → 64 (`309 → 305` words, `213.1 → 197.6 µs/shot`) with identical 11.395 mHa error | Heavy `word_weight ≥ 0.2` or `Myopia_eps1e-3` starves Krylov directions early, degrading `tfim5` 4.7× (53.962 mHa) |
| **B2** 7D delta-method MC (`25` seeds) | Predicted lowering SE matches finite-shot MC dispersion within 1.23–1.34× (`orth` SE within 1.11–1.35×) | Pure collinear eigenstate controls (`I + 0.05 X₀` on `|+⟩⊗⁴`) are identically deterministic (`0.0`); stochastic near-duplicates are exercised on `|0⟩⊗⁴` in unit tests |
| **B3** Closed-loop `acquire_to_accuracy` | Meets its guarded candidate-family proxy on 20/20 runs; beats `UniformFullRegistry` on 13/20 and `UniformBasisOnly` on 2/20 | Statistical precision and candidate lowering do not bound true ground-state error; `tfim4_sc_se` median true error is 22.36 mHa against a 20 mHa proxy target |

---

## 1. Hierarchical commutator growth, closure, and symmetry (**Part A1**)

`MV` prunes amplitudes with `|c| <= 1e-12` **absolutely** upon construction,
so naive nested multiplication `H * A - A * H` corrupts or erases rescaled
hierarchies. [`commutator_mv`](../clifford_qc/subspace/commutators.py) accumulates packed anticommuting Pauli products in raw dictionaries, checks zero relative to `relative_zero_tol * ||H||₂ ||A||₂`, applies Hermiticity-preserving phases (`phase_mode="hermitian"`), and normalizes before wrapping in `MV`.

**Lie closure check.** On the 1-qubit 𝖘𝖑(2) instance `H = X + 0.5 Z` seeded by
`𝒜₀ = {Z}`, [`nested_commutator_hierarchy`](../clifford_qc/subspace/commutators.py) retains one novel generator at depths `0, 1, 2` (max support `1 → 1 → 2`, dimensionless growth ratio `1.000 → 1.789 → 2.000`) and closes by collinear deduplication at `closure_depth = 3` (`closure_fraction = 1.0`). A generator cap or support/sector filter can leave branches uncomputed; an empty frontier then does not establish closure. Tail-truncated hierarchies also remain uncertified.

**Non-integrable spin chains grow explosively on `xxz6` and slowly on `tfim6`.**
Starting from the Pauli terms of `H` (`11` seeds on `tfim6`, `E₀ = −7.296230 Ha`; `15` seeds on `xxz6`, `E₀ = −9.974309 Ha`), neither algebra closes by depth 3, yet their operator scrambling and subspace spans diverge sharply:

| System | Depth `k` | Kept | Max `S_A` | Total `S_A` | Mean growth ratio | Det rank / Err (mHa) | Corr rank / Err (mHa) | `W` (`corr`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tfim6` | 0 | 11 | 1 | 11 | 1.000 | 6 / 296.23 | 12 / 168.65 | 232 |
| `tfim6` | 1 | 11 | 2 | 20 | 0.807 | 6 / 296.23 | 15 / 164.01 | 688 |
| `tfim6` | 2 | 11 | 6 | 58 | 1.387 | 10 / 180.76 | 16 / 163.85 | 1 069 |
| `tfim6` | 3 | 11 | 8 | 70 | 1.629 | 10 / 180.76 | 16 / 163.85 | 1 285 |
| `xxz6` | 0 | 15 | 1 | 15 | 1.000 | 6 / 1 770.95 | 16 / 1 567.26 | 376 |
| `xxz6` | 1 | 15 | 4 | 48 | 0.912 | 10 / 1 770.95 | 29 / 201.12 | 952 |
| `xxz6` | 2 | 15 | 23 | 243 | 1.715 | 16 / 1 140.63 | 32 / **−0.00** | 1 024 |
| `xxz6` | 3 | 15 | 64 | 660 | 2.156 | 20 / 1 140.63 | 32 / **−0.00** | 1 024 |

Two structural effects appear cleanly in [`hierarchy_span_diagnostics`](../clifford_qc/subspace/commutators.py):

* **Even–odd parity on product states (`tfim6 det`).** From `|0⟩⊗⁶`, Hermitianized depth-1 commutators `[H, P]` are pure imaginary-signed single-qubit/two-qubit flips that leave the real reference subspace decoupled at depth 1 (rank stays `6`, gain `0.00 mHa`); even double commutators at depth 2 add four new directions and drop the error by `115.47 mHa`.
* **Reference sensitivity on `xxz6`.** On a single Néel determinant (`010101`), even all `61` cumulative generators through depth 3 plateau at `1 140.63 mHa` (rank `20`), whereas on a shallow Trotter state (`Δt = 0.35`, `6` steps), depth-2 commutators saturate the `32`-state magnetization-and-parity block and recover the exact ground state (`error = −0.00 mHa`, `W = 1 024`).

**Truncated commutator support vs symmetry preservation.** On `xxz(4)` (`E₀ = −6.464102 Ha`, `W = 64`), capping generator support from `15` down to `8` (`support_policy="truncate"`, `truncation_l2_keep=0.98`, minimum retained `ℓ₂` weight `0.8542`) keeps the full depth-2 span exact (`0.000 mHa`) on both references and keeps greedy A-CASE (`M = 6`) unchanged at `0.0095 mHa` on the correlated state (`33` QWC groups), while greedy A-CASE from the bare Néel determinant shifts from `0.000` to `15.473 mHa` because tail clipping reorders intermediate 2×2 lowering scores. On Jordan-Wigner fermionic systems, however, naive Pauli clipping cannot be used safely:

| System | Mode | Nodes | Max support | Max operator leakage | Max ref leakage |
|---|---|---:|---:|---:|---:|
| `hubbard(2)_d1..2` | Exact | 6 | — | **0.0000** | — |
| `h4_d1` | Exact | 8 | — | **0.0000** | — |
| `h4_d1` | `truncation_l2_keep=0.90` | 8 | 49 | 0.1388 | 0.0025 |
| `h4_d1` | `truncation_top_k=16` | 8 | 16 | 0.2318 | 0.0272 |
| `lih_d1` | Exact | 16 | — | **0.0000** | — |

Exact nested commutators preserve `(N, S_z)` identically by Jacobi identity (`sector_leakage = 0.0`), whereas dropping low-magnitude Pauli tails breaks electron-number and spin symmetry by up to `0.232` in Hilbert-Schmidt norm on `H₄` (`sector_leakage` vs `reference_sector_leakage` in [`clifford_qc/subspace/symmetry.py`](../clifford_qc/subspace/symmetry.py)).

---

## 2. Spin chains: lazy frontier vs static hierarchy vs Krylov (**Part A2**)

At fixed basis budget `M = 8` (`1 + 7` added generators), [`run_dynamic_commutator_acase`](../clifford_qc/subspace/commutators.py) starts from the depth-0 Pauli orbit and expands `[H, A]` lazily on demand, avoiding upfront materialization of `Static_d<=3`:

| System (`ref`) | Arm | `M` | Error (mHa) | `W` | `G` (QWC) | Commutators eval'd | Max `S_A` | Depths used |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `tfim6` (`det`) | `Orbit_d0` | 6 | 296.23 | 122 | 15 | 0 | 1 | `d0` |
| `tfim6` (`det`) | `Static_d<=3` | 8 | 183.88 | 976 | 209 | 33 | 8 | static pool |
| `tfim6` (`det`) | `Dyn_on_select_or_stall` | 8 | 218.07 | 370 | 65 | **17** | 6 | `{'0': 5, '2': 2}` |
| `tfim6` (`det`) | `Krylov6` | 7 | **0.24** | 879 | 130 | 0 | 285 | fixed `H^k` |
| `tfim6` (`corr`) | `Static_d<=3` | 8 | 203.29 | 462 | 89 | 33 | 8 | static pool |
| `tfim6` (`corr`) | `Dyn_on_select_or_stall` | 8 | **177.31** | **180** | **29** | **7** | 1 | `{'0': 7}` |
| `tfim6` (`corr`) | `Krylov6` | 7 | **0.25** | 879 | 130 | 0 | 285 | fixed `H^k` |
| `xxz6` (`det`) | `Static_d<=3` | 8 | 1 184.10 | 1 024 | 333 | 45 | 23 | static pool |
| `xxz6` (`det`) | `Dyn_on_select_or_stall` | 8 | 1 715.46 | 766 | 217 | **9** | 10 | `{'0': 5, '2': 2}` |
| `xxz6` (`det`) | `Krylov6` | 7 | **0.26** | 544 | 183 | 0 | 544 | fixed `H^k` |
| `xxz6` (`corr`) | `Static_d<=3` | 8 | 1 553.71 | 310 | 115 | 45 | 4 | static pool |
| `xxz6` (`corr`) | `Dyn_on_select_or_stall` | 8 | **1 434.20** | **300** | **112** | **7** | 2 | `{'0': 5, '1': 2}` |
| `xxz6` (`corr`) | `Krylov6` | 7 | **0.26** | 544 | 183 | 0 | 544 | fixed `H^k` |

**What works.** Relative to enumerating the depth-3 static hierarchy, `Dyn_on_select_or_stall` cuts commutator evaluations by `48%–84%` (`33 → 17` on `tfim6 det`, `33 → 7` on `tfim6 corr`, `45 → 9` and `45 → 7` on `xxz6`) and shrinks the measurement universe (`W` down `61%–62%` and QWC settings down `67%–69%` on `tfim6`). On both correlated references, the lazy frontier even beats `Static_d<=3` on fixed-`M` energy (`177.31` vs `203.29 mHa` on `tfim6`; `1 434.20` vs `1 553.71 mHa` on `xxz6`), because exposing all 33–45 depth-2/3 operators up front induces greedy rank-1 steps toward high-support directions that underperform over a 7-step horizon.

**Boundary / negative result against Krylov.** Global `Krylov6` reaches `0.24–0.26 mHa` in `M = 7` across all four cases (`W = 879`, `G = 130` on `tfim6`; `W = 544`, `G = 183` on `xxz6`). On translationally invariant spin chains, powers `H^k` form symmetry-adapted, globally connected multi-site superpositions in a single vector per order `k`, whereas one-bond Pauli-orbit seeds fracture those superpositions across `11`–`15` local bond operators that exhaust the `M = 8` basis budget before covering every link cleanly. Locally seeded commutator trees should therefore complement global Krylov directions rather than replace them when global translationally symmetric superpositions dominate and `M` is small.

---

## 3. Fermionic lattices and chemistry vs excitation pools and selected CI (**Part A3**)

Here the question is sharper: when only rank-1 single excitations `S` are placed in the initial pool (`M ≤ 4` on `hubbard_2x1`, `M ≤ 9` on `hubbard_2x2`/`H₄`/`LiH`), can nested commutators `[H, S]` recover correlation that `Singles` miss by Brillouin's theorem without outgrowing `SD_Pool`, and do they step outside the determinant closure of `(S, D)` ([`span_comparison`](../clifford_qc/subspace/selected_ci.py#L993-L1023))?

| System | Arm / Control | `M` (rank) | Error (mHa) | `W` / `nnz` | `G` (QWC) | Max `S_A` | Depth counts / work |
|---|---|---:|---:|---:|---:|---:|---|
| `hubbard_2x1` (`det`) | `Singles` | 3 (3) | 378.937 | 55 | 25 | 2 | pool exhausted |
| `hubbard_2x1` (`det`) | `DynComm(S)_d2` | 4 (4) | **0.000** | 62 | 25 | 8 | `{'0': 2, '2': 1}`, 4 comms |
| `hubbard_2x1` (`det`) | `SD_Pool` / `Krylov3` / `IterSCI` | 4 (3–4) | **0.000** | 60 / 24 / `10 nnz` | 25 / 13 / — | 8 / 24 / — | — |
| `hubbard_2x2` (`det`) | `Singles` / `DynComm(S)_d1` | 5 (5) | 1 274.32 | 637 | 259 | 2 | stalled below `min_lowering` |
| `hubbard_2x2` (`det`) | `DynComm(S)_d2_stall` | 9 (9) | **1 057.35** | 5 040 | 948 | 34 | `{'0': 7, '2': 1}`, 10 comms |
| `hubbard_2x2` (`det`) | `SD_Pool` | 9 (9) | **863.31** | 5 537 | 1 406 | 8 | `d` pool explicit |
| `hubbard_2x2` (`det`) | `Krylov3` | 4 (4) | **668.97** | 3 752 | 809 | 615 | `κ = 15 483` |
| `hubbard_2x2` (`det`) | `IterSCI` | 8 | **908.77** | `26 nnz` | — | — | `224` scored, 1.9 ms |
| `h4_sto3g_r0.9` | `Singles` (RHF) | 1 (1) | 56.057 | 185 | 68 | 1 | Brillouin stall |
| `h4_sto3g_r0.9` | `DynComm(S)_d1` | 9 (9) | 55.329 | 7 359 | 869 | 116 | `{'0': 4, '1': 4}`, 4 comms |
| `h4_sto3g_r0.9` | `SD_Pool` | 9 (9) | **3.019** | 7 371 | 913 | 8 | explicit doubles |
| `h4_sto3g_r0.9` | `Krylov2` | 3 (3) | **1.722** | 4 224 | 441 | 1 775 | `κ = 4 735` |
| `h4_sto3g_r0.9` | `IterSCI` | 8 | **4.953** | `58 nnz` | — | — | `224` scored, 4.0 ms |
| `lih_cas4e4o` | `Singles` (RHF) | 1 (1) | 1.196 | 193 | 42 | 1 | Brillouin stall |
| `lih_cas4e4o` | `DynComm(S)_d1` | 8 (8) | 0.547 | 7 355 | 917 | 128 | `{'0': 4, '1': 3}`, 8 comms |
| `lih_cas4e4o` | `DynComm(S)_d2_top32` | 7 (7) | 0.849 | 7 091 | 917 | 32 | `top_k=32`, 12 comms |
| `lih_cas4e4o` | `SD_Pool` | 9 (9) | **0.0005** | 7 697 | 961 | 8 | explicit doubles |
| `lih_cas4e4o` | `Krylov2` | 3 (3) | **0.110** | 4 224 | 477 | 1 783 | `κ = 7.17 × 10⁶` |
| `lih_cas4e4o` | `IterSCI` | 8 | **0.016** | `48 nnz` | — | — | `224` scored, 3.6 ms |

Notice how cleanly [`span_comparison`](../clifford_qc/subspace/selected_ci.py#L993-L1023) explains both outcomes:

1. **Where commutators help:** When only singles are given, depth-2 commutators exit the singles determinant manifold (`span_vs_S` `contained_in_closure = False`, `containment_residual = 1.000` on `hubbard_2x1`) and inject double-excitation support on demand — reaching exact ground energy (`0.000 mHa` in 4 commutators) on `hubbard_2x1`, and rescuing `hubbard_2x2` when given enough child budget (`max_children_total = 20`, dropping error by `216.97 mHa` at depth 2 with `10` commutators vs stopping completely at depth 1 under an `8`-child cap).
2. **Where they lose cleanly to direct doubles, Krylov, and SCI:** Every single- and double-commutator basis here remains strictly inside the fixed `(S, D)` determinant closure (`containment_residual ≤ 4.35 × 10⁻¹⁶` on `hubbard_2x2`, `2.21 × 10⁻¹⁵` on `H₄`, `2.38 × 10⁻¹⁵` on `LiH`). Because `[H, Sᵢ]` superposes many weak two-body couplings into one dense multivector (`S_A = 116–128` on `H₄`/`LiH`), `DynComm(S)_d1` ends up paying nearly the entire `SD_Pool` Pauli measurement universe (`W = 7 355–7 359`, `G = 869–917`) while recovering far less correlation than selective rank-2 excitations (`3.019` / `0.0005 mHa`), `Krylov2` (`1.722` / `0.110 mHa`), or classical `IterSCI` (`4.953` / `0.016 mHa` in ~3.6 ms). Thus commutator generation does not beat explicit doubles or sparse selected CI on molecular ground states.

---

## 4. Resource-aware Pareto sweeps and myopia boundary (**Part B1**)

[`estimate_incremental_cost`](../clifford_qc/subspace/information.py) prices every candidate by the incremental observables `ΔW`, genuinely new QWC settings `ΔG` (after first-fit compatibility checks against existing QWC bases), hardware circuit duration `ΔT` under a declared `DeviceCard` ([`superconducting-like.json`](../benchmarks/configs/device_cards/superconducting-like.json), [`ion-like.json`](../benchmarks/configs/device_cards/ion-like.json), [`logical-alltoall.json`](../benchmarks/configs/device_cards/logical-alltoall.json)), and classical Pauli pair products / terms (`classical_products`, `classical_terms`), populated regardless of cost weights.

| System | Arm | `M` | Final error (mHa) | `W` | `G` (QWC) | SC time (`µs`/shot) | Classical products | Selected sequence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `tfim4` | `EnergyOnly` | 5 | 1.176 | 90 | 28 | 86.3 | 4 411 | `H^3, H^2, H^1, IZZI` |
| `tfim4` | `QWCCost_SC_0.5` / `TimeCost_*` / `Classical` | 5 | 1.176 | 90 | 28 | 86.3 | **4 140** | `H^1, H^2, H^3, IZZI` |
| `tfim4` | `Word_0.2` / `Word_1.0` / `Myopia_eps1e-3` | 6 | 1.176 | 120 | 38 | 117.3 | 3 261 | `ZZII, IZZI, IIZZ, H^2, H^3` |
| `tfim5` | `EnergyOnly` | 6 | 11.395 | 309 | 69 | 213.1 | 18 667 | `H^3, H^1, H^2, ZZIII, IIIZZ` |
| `tfim5` | `QWCCost_SC_0.5` / `2.0` / `TimeCost_*` | 6 | **11.395** | **305** | **64** | **197.6** | **17 399** | `H^1, H^2, H^3, IZZII, IIZZI` |
| `tfim5` | `Word_0.2` / `Word_1.0` / `Myopia_eps1e-3` | 6 | 53.962 | 277 | 55 | 169.7 | 3 157 | `ZZIII, IZZII, …, H^2` |
| `xxz4` | `EnergyOnly` / `Word_0.05` | 5 | 0.000 | **56** | **29** | **89.4** | 10 315 | `H^2, H^1, H^3, IXXI` |
| `xxz4` | `QWCCost_SC_0.5` / `TimeCost_*` | 6 | 0.000 | 64 | 33 | 101.8 | **8 707** | `XXII, IIYY/IYYI, …, H^3, H^2` |
| `hub1_corr` | All 11 arms | 3 | −0.000 | 24 | 13 | 40.1 | 1 151–1 179 | `H^2, H^1` (`H^1, H^2` on `Classical`) |

Two practical lessons follow directly from the sweep:

1. **Zero-accuracy-loss Pareto pruning (`tfim5`).** Moderate QWC or hardware-time weights (`QWCCost_SC_0.5`, `QWCCost_SC_2.0`, `TimeCost_SC`, `TimeCost_Ion`, `TimeCost_Logical`) match `EnergyOnly`'s exact `11.395 mHa` energy while swapping out boundary bond operators for QWC-compatible interior bonds (`IZZII, IIZZI`), reducing QWC settings from `69` to `64` (`−7.2%`), words from `309` to `305`, SC shot-round time from `213.1` to `197.6 µs`, and classical pair products from `18 667` to `17 399`.
2. **Small-`ϵ` / heavy-`ΔW` myopia (`tfim5`, `tfim4`).** When `word_weight >= 0.2` or `epsilon = 1e-3` (`Myopia_eps1e-3`), dividing by `C_incremental + ϵ` prioritizes cheap single-Pauli words (`ZZIII`, `IZZII`) at step 1 instead of global `H^k` directions, costing an extra basis slot on `tfim4` (`W = 120` vs `90`) and increasing final error on `tfim5` by `4.7×` (`53.962` vs `11.395 mHa`). Normalization `ϵ` should stay comparable to the Hamiltonian lowering scale (e.g., `epsilon = 1.0`).

---

## 5. Seven-dimensional delta-method uncertainty validation (**Part B2**)

In finite-shot acquisition, plugging noisy matrix elements directly into the
nonlinear 2×2 generalized eigenvalue ratio amplifies shot variance on near-linearly dependent directions. [`expected_lowering_estimate`](../clifford_qc/subspace/information.py) (via [`_expected_lowering_details`](../clifford_qc/subspace/information.py)) propagates the joint grouped covariance from [`GroupedWordCache`](../clifford_qc/measurement/cache.py) through one scalarized `WordFunctional` over `(Re s_a, Im s_a, Re h_a, Im h_a, s_aa, h_aa, E_m)` for `σ_{ΔE}`, alongside linearized functionals for full-subspace orthogonal fraction `orth_f` and residual coupling `coup_f`. The derivative includes the measured basis eigenvectors, their normalization, and the retained-subspace projector. Rank changes and unresolved root degeneracies keep uncertainty unresolved. Across `25` independent [`FiniteShotBackend`](../clifford_qc/backends/finite_shot.py) seeds at `3 000` shots/group on `tfim4` (`|++++⟩` reference):

| Candidate | Exact `ΔE` (mHa) | MC `σ_{ΔE}` (mHa) | Predicted median `σ̂_{ΔE}` (IQR, mHa) | Ratio (`σ̂/σ`) | MC `σ_{orth}` | Predicted `σ̂_{orth}` (ratio) | Zeroed / Accepted |
|---|---:|---:|---:|---:|---:|---:|---|
| `H^2` | 710.63 | 20.42 | 25.04 (`[24.82, 25.14]`) | 1.23× | 0.0049 | 0.0066 (1.35×) | 0.00 / 1.00 |
| `H` | 645.75 | 15.12 | 20.24 (`[19.03, 21.46]`) | 1.34× | 0.0060 | 0.0067 (1.11×) | 0.00 / 1.00 |
| `I + 0.05 X₀` | 0.00 | 0.00 | 0.00 (`[0.00, 0.00]`) | — | 0.0000 | 0.0000 (—) | **1.00 / 0.00** |

This B2 experiment uses the identity-only basis and therefore does not test noisy Ritz-vector response. A separate 200-seed two-qubit regression with a measured two-dimensional basis gives empirical lowering SE `0.00548` and median predicted SE `0.00568` (the frozen-vector calculation gave `0.000142`). Full-estimator finite differences also check real, complex, and truncated pencils in [`tests/test_information_regressions.py`](../tests/test_information_regressions.py).

The propagated delta-method standard errors are mildly conservative by `1.11–1.35×` across both lowering and orthogonality with tight IQRs (`±0.2` to `±1.2 mHa`). On `tfim(4)`'s reference `|+⟩⊗⁴`, `X₀|+⟩ = +|+⟩`, so `I + 0.05 X₀` is an exact collinear deterministic control (`orth = 0`, deterministic `X` measurement variance `0`, zeroed in 25/25 runs); non-deterministic stochastic near-duplicates such as `I + 0.01 X₀` on `|0000⟩` and `X₀ + 0.08 Z₁` on a Trotterized state are verified in [`tests/test_information_selection.py`](../tests/test_information_selection.py).

---

## 6. Closed-loop acquisition against an error proxy (**Part B3**)

[`acquire_to_accuracy`](../clifford_qc/subspace/information.py) inspects no exact oracle ground energy (`exact_ground_energy` and `target_error` are absent from its signature). It compares marginal RMSE-proxy reduction from expansion against covariance-aware Neyman refinement. Refinement targets the Ritz energy and unresolved candidate functionals, including the measured basis response. Every screening batch is included in the returned solve and interval.

The stopping proxy is `hypot(stat, max_candidate(lowering + lowering_std))`, with unresolved uncertainty retained. Candidate screening is evaluated at a basis limit when its words are already measured; unavailable candidate data keeps bias unknown. Shot and execution-time caps apply before each batch. An initial pilot that cannot fund one shot per required setting raises `ValueError` before sampling.

Across five seeds (`11, 12, 13, 17, 23`, `max_total_shots = 60 000`, `max_steps = 80`), acquisition is compared against uniform allocation on the discovered final basis. `UniformFullRegistry` (`UF`) uses every discovered QWC setting; `UniformBasisOnly` (`UB`) uses only final-basis settings and pays no screening cost. Each uniform comparator receives the acquisition shot budget and leaves the integer-division remainder unused.

| Scenario (`device`, metric) | Target (mHa) | Proxy met | Med shots (`time`) | Med `acq` stat / true (mHa) | Med `UF` stat / true (mHa) | Med `UB` stat / true (mHa) | Wins vs `UF` / `UB` (stat) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tfim4_sc_se` (`superconducting`, `std_error`) | 20.0 | 5/5 | 5 340 (`16.42 ms`) | 8.91 / 22.36 | 10.55 / 20.87 | 7.51 / 5.48 | 3/5 / 0/5 |
| `tfim4_ion_se` (`ion-like`, `std_error`) | 20.0 | 5/5 | 5 340 (`1.15 s`) | 11.51 / 1.39 | 12.81 / 13.71 | 10.17 / 5.48 | 4/5 / 1/5 |
| `tfim4_sc_ci` (`superconducting`, `ci_halfwidth`) | 40.0 | 5/5 | 17 460 (`53.39 ms`) | 30.10 / 13.82 | 26.44 / 4.60 | 19.98 / 11.04 | 3/5 / 1/5 |
| `xxz4_sc_se` (`superconducting`, `std_error`) | 20.0 | 5/5 | 2 580 (`7.96 ms`) | 9.92 / 10.40 | 12.11 / 12.88 | 6.54 / 7.81 | 3/5 / 0/5 |
| **Total (20 runs)** | — | **20/20** | — | — | — | — | **13/20** / **2/20** |

These results replace the pre-fix 19/20, 12/20, and 14/20 claims. Including the measured Ritz-vector response changes candidate gates, selected bases, and shot allocation. Acquisition often improves statistical precision relative to the full-registry baseline, while the final-basis-only baseline usually has lower statistical error because it pays no screening cost.

`converged` records this candidate-family proxy, not a true-ground-error certificate. For example, `tfim4_sc_se` meets the 20 mHa proxy in all five runs but has median true error 22.36 mHa. Its returned interval is an asymptotic single-look diagnostic; this adaptive schedule has no claimed anytime coverage or finite-sample guarantee. The study is exploratory, and its JSON records the dirty-tree state and implementation SHA-256 digests alongside the base revision and runtime versions.

---

## 7. Reproducing and verifying

All implementations rely strictly on pure Python (`>= 3.10`) and NumPy. Frozen project baselines (`PHASE_STATUS.json`, `benchmarks/reference_results/*`, `benchmarks/provenance/*`, `benchmarks/data/*`, `docs/restart/*`) remain untouched.

```bash
python3 -u benchmarks/run_geometric_information_study.py
python3 -m pytest tests/test_commutator_hierarchy.py tests/test_information_selection.py tests/test_information_regressions.py
python3 -m clifford_qc.verify
```
