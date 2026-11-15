# Measuring many Pauli observables on one state: grouping, measurement circuits, shadow-style estimators, shot allocation and uncertainty (state of the art, November 2026)

Scope: methods for estimating a large, fixed set of Pauli words on one prepared state, with an eye on clifford_qc's A-CASE/DA-CASE bank (2,240–7,371 words on 8 qubits; QWC 913 → dyadic k=2: 647 → k=4: 238 → k=8: 64 settings for the 7,371-word H4 bank, per `benchmarks/GROUPING_REDUCTION.md`). Results older than 2022 are marked **[background]**. "Sim" means numerical simulation and "HW" means a hardware experiment. Unless stated otherwise, papers report Hamiltonian-energy benchmarks on molecular Hamiltonians, not general banks.

---

## 1. Grouping: how do the partitioning heuristics compare in setting count and in estimator variance, and what objective should be minimized?

### Takeaway
Since 2021 the field has agreed that **the number of groups is the wrong objective**. The right one is estimator variance under optimal shot allocation, which Crawford et al.'s R / R̂ ratio summarizes. Coefficient-ordered **sorted insertion (SI)** beats largest-first, DSATUR, independent-set and similar colourings on that metric, even when it produces more groups. In 2023–2026 the gains come from (i) letting words sit in several groups (overlapping, coefficient splitting, "repacking"), which cuts variance by 1.2–2.35× in 2026 work and up to 5× in covariance-informed work; (ii) variance-aware vertex orderings; and (iii) intermediate compatibility relations between QWC and full commutation. The "dyadic block-commuting" hierarchy in clifford_qc is essentially the published **k-commutativity** of DalFavero et al. (arXiv 2312.11840, v3 Sept 2025), which already includes a variable-block-size generalization and open code.

### Cited Findings
**Metric and objective**
- **[background]** Crawford et al. (Quantum 5, 385, 2021) define the state-dependent ratio R = (Σ_i Σ_j |a_ij| √Var[P_ij] / Σ_i √Var[O_i])² and the state-independent R̂ = (Σ_i Σ_j |a_ij| / Σ_i √(Σ_j a_ij²))². Both equal M_ungrouped / M_grouped, the ratio of shots needed without and with grouping when shots are allocated optimally across groups. Larger is better. — [ar5iv of arXiv 1908.06942](https://ar5iv.labs.arxiv.org/html/1908.06942)
- **[background]** Crawford Table 3 gives R̂ for SI vs NetworkX Largest-First / DSATUR (the commutation relation used for this table was not confirmed in this pass). H2O: 10.66 vs 2.98 / 3.18. NH3: 15.31 vs 6.50 / 2.92. CH4: 20.27 vs 5.84 / 4.90. LiH: 23.97 vs 23.87 / 10.47. SI is best or joint best in all but 2 of the 6 molecules. The paper reports a "10 to 60 fold" saving from SI across H2 to H2Se. — [ar5iv 1908.06942](https://ar5iv.labs.arxiv.org/html/1908.06942); [Quantum 5, 385](https://quantum-journal.org/papers/q-2021-01-20-385/)
- Yen, Ganeshram & Izmaylov (npj QI 2023) conclude that "the number of fragments is not a good proxy for the total number of measurements". Largest-first (LF) minimizes group count but gives higher variance than SI. For example, exact-state FC variance for H2O (14 qubits, 1,086 Paulis) is LF 43.4 vs SI 7.59, and for NH3 (16 qubits, 3,609 Paulis) LF 78.7 vs SI 18.8. They recommend benchmarking against greedy SI rather than against group-count minimizers. — [PMC11041696](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)
- The 2025 Toronto review (Patel, Jayakumar, Yen, Izmaylov, arXiv 2501.14968) says "the appropriate metric … is the variance of the estimator". It reports SI giving a 2–4× cost reduction vs fragment-count-minimizing grouping, and coefficient splitting a further reduction of up to 3× vs SI. — [arXiv 2501.14968v2](https://arxiv.org/html/2501.14968v2)
- DalFavero et al.'s Appendix B (C2, 10 qubits) shows that random insertion gives about the same number of groups as SI but a much lower and noisier R̂. Group count and R̂ therefore decouple. — [arXiv 2312.11840v3](https://arxiv.org/html/2312.11840v3)
- In Jørgensen et al. (arXiv 2603.20029, Mar 2026), variance-aware vertex orderings (LDVF / LVF) beat largest-degree-first (LDF) "in almost all cases" under ℓ² inverse-probability weighting. Example variances in Ha² for one-shot Horvitz–Thompson estimation (Table 6): H2O QWC is LDF 921, LDVF 450, LVF 590. NH3 FC is LDF 625, LDVF 315, LVF 470. The ℓ² sampling weights alone cut NH3 QWC variance from 96,700 (uniform) to 732. — [arXiv 2603.20029v1](https://arxiv.org/html/2603.20029v1)

**Intermediate compatibility relations (QWC ↔ FC)**
- **k-commutativity** (DalFavero, Sarkar, Rowland, Camps, Sawaya, LaRose; arXiv 2312.11840, v3 19 Sep 2025, no journal listed):
  - Definition: two strings k-commute if their restrictions to every contiguous block of k qubits commute. k=1 is QWC and k=n is full commutation. If k does not divide n, pad with identities. The paper generalizes to variable block sizes (k₁,…,k_m) with Σk_i = n.
  - Grouping uses SI. The metric is R̂ plus group counts, not shots.
  - Test set: HamLib molecules with 10–16 qubits (156–3,637 terms) and N2 at 32 qubits.
  - R̂ is not monotone in k. For N2, R̂(n/2) > R̂(3n/4), and for O2 they are about equal.
  - For an H chain with n=64 at k=32, R̂ ≈ 200 vs no grouping.
  - For 1D unary Bose–Hubbard, the best k is 4 at every size from 16 to 88 qubits, with R̂max/R̂min only 1.07–1.66.
  - Circuits are diagonalized in ⌊n/k⌋ parallel blocks. Depth is O(1) at k=1 and O(n²/log n) at k=n. Theorem 4 gives a lower bound for {CNOT, H, S}.
  - There is no noise model or break-even analysis. The authors argue qualitatively that noisy devices with large budgets favour small k.
  - Code: github.com/rmlarose/kcommute. — [arXiv 2312.11840v3](https://arxiv.org/html/2312.11840v3)
- QED-C benchmark framework (Niu, Kökcü, Johri, …, Camps, Lubinski; arXiv 2504.09813, Apr 2025): k-commuting groups, custom Clifford measurement circuits and weighted shot distribution on HamLib instances. It reports a "27.1% error reduction through Pauli grouping methods" plus "an additional 37.6%" from optimized shot distribution. The abstract does not state the baseline or the system sizes. — [arXiv 2504.09813](https://arxiv.org/abs/2504.09813)
- **GALIC** (Burns et al., PNNL; arXiv 2409.00576; PNNL lists it as a 2025 publication):
  - Acceptance rule: a fully commuting group is accepted only if its worst-case CNOT estimate, ½·N_AC(N_AC−1)·(3(D_max−1)+1), with N_AC the number of anticommuting qubits and D_max their maximum device distance, stays below log(1−ε_tar)/log(1−p), where p is the median two-qubit error and ε_tar the target relative bias. QWC groups always pass. Grouping uses SI. The objective is (Σ_ℓ √Var O_ℓ)², with n_ℓ ∝ √Var O_ℓ.
  - Sim (density-matrix NWQ-Sim, 5 molecules up to 14 qubits, IBM Sherbrooke/Torino heavy-hex and IonQ Aria/Forte models): about 20% average variance reduction vs QWC, while FC averages 3.4× lower variance than QWC.
  - Table 5 group counts. Torino H4: QWC 49, GALIC 42, HEC 39, FC 10. Forte H6: QWC 191, GALIC 126, FC 32.
  - Bias cases above 1 kcal/mol (out of 20): QWC 0, GALIC 5, HEC 15, FC 20.
  - Gate cost: GALIC uses about 30% fewer entangling gates than HEC on average, at about 4% more groups. — [arXiv 2409.00576v2](https://arxiv.org/html/2409.00576v2); [PNNL](https://www.pnnl.gov/publications/galic-hybrid-multi-qubitwise-pauli-grouping-quantum-computing-measurement)
- **Hardware-tailored (HT) diagonalization** (Miller, Fischer, et al., npj QI 10, 122, Nov 2024):
  - Groups are restricted to sets whose diagonalizing Clifford is native to the coupling graph, with no SWAPs. CZ depth is at most the maximum degree of the graph: 3 on heavy-hex, 4 on a square lattice.
  - 8-qubit H chain (184 terms): 10 HT settings vs 35 TPB/QWC. R̂_HT/R̂_TPB ≈ 1.76 (theory 1.87), and the ratio is 1.3–3.5 across three Hamiltonian classes.
  - Linear-connectivity constant-depth guess (n/2 CZ): n=20 (7.2k terms) needs 19 TPB circuits vs 191 HT-diagonalizable; n=120 (9.6M terms) needs 32 TPB circuits vs 7,141.
  - Grouping is a modified SI. Runtime is O(M² f(n)), empirically about n^9.4–9.8 for H chains.
  - C++ code: github.com/Mc-Zen/HT-Grouper. — [arXiv 2203.03646v3](https://arxiv.org/html/2203.03646v3); [DOAJ record](https://doaj.org/article/1825a941cc414038944bf0d50fd4866c)
- **k-NoCliD** (Sawaya, Camps, DalFavero, Tubman, Rotskoff, LaRose; arXiv 2408.11898, v3 Jun 2025) drops commutativity altogether. It groups arbitrary terms into k-local tensor-product blocks diagonalized by non-Clifford unitaries. It measures in "far fewer bases in most cases", often with large shot reductions even at k=2, for vibrational, Fermi–Hubbard and Bose–Hubbard models, at greater depth. The abstract gives no numbers. — [arXiv 2408.11898](https://arxiv.org/abs/2408.11898)
- Dense operators: Reggio, Butt, Lytle & Draper (PRA 110, 022606, 2024) partition the full Pauli set on any number of qubits into the minimal number of commuting families in time linear in the set size, with diagonalizing circuits. They report a speedup over QWC close to the theoretical (3/2)^m for m=2–6 qubits, on IBM hardware. Python code and a Qiskit integration are provided. — [arXiv 2305.11847](https://arxiv.org/abs/2305.11847)

**Overlapping groups, 2026 state of the art**
- Rowland, Sarkar, Sawaya, Tubman & LaRose ("Overlapped groupings for quantum energy estimation", arXiv 2604.07156, Apr 2026):
  - They prove the maximal variance reduction from overlapping is of order m (the number of disjoint groups) relative to the best SI grouping, i.e. "linear in the number of Hamiltonian terms". This is asymptotic, under a state-independent zero-covariance model.
  - **Post-hoc repacking** reuses existing bases and bitstrings: any term whose conjugate by a group's diagonalizer is Z/I-only is added to that group. **Ad-hoc repacking** inserts terms by c_i²/μ_i before circuits are built and usually wins.
  - Estimator: the shot-weighted average w_ij = M_j/ΣM_k, which is minimum-variance only under zero covariance.
  - Results: up to about 2.35× variance reduction vs SI on the largest system, a metaphosphate model with 575×10³ terms at about 44 qubits. The reduction grows roughly linearly with size, and repacking never increases optimal variance (Theorem 6). — [arXiv 2604.07156v1](https://arxiv.org/html/2604.07156v1)
- Yen et al. 2023, iterative measurement allocation (IMA) and iterative coefficient splitting (ICS) vs SI. Exact-state FC variances: H2O 7.59 → 1.50 (about 5×) and NH3 18.8 → 3.32. QWC: NH3 97.0 → 44.8. The authors summarize "up to forty and eighty percent reduction" vs the best non-overlapping method, and FC beating QWC "by up to a factor of seven". Coefficient splitting raises the number of variables sharply: NH3 QWC goes from 1,359 to 61,137 variables. — [PMC11041696](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)
- VGMC (Zhao, Yao, Sun, Li; arXiv 2611.14986, 11 Nov 2026) adds every term measurable in each existing basis, jointly optimizes estimator coefficients and shots, then greedily adds new QWC groups by benefit against shots withdrawn. It reports an 18.19–23.38% variance reduction vs "state-of-the-art measurement design methods" with exact covariances, and says the gains persist with approximate covariances. Baselines and molecules are not given in the abstract. — [arXiv 2611.14986](https://arxiv.org/abs/2611.14986)
- **[background]** Wu, Sun, Huang & Yuan, overlapped grouping measurement (OGM; Quantum 7, 896, 2023, arXiv 2021): a unified framework in which importance sampling, compatibility grouping and classical shadows are special cases. The optimal scheme partitions observables into overlapping compatible groups. Tested on molecules up to 16 qubits. — [arXiv 2105.13091](https://www.arxiv.org/abs/2105.13091); [DOAJ](https://doaj.org/article/e05d0b6c80d249f79ba26fcf046e3a58)

**Scale of colouring**
- Picasso (Ferdous et al., PNNL/NCSU; arXiv 2401.06713) is palette-based randomized colouring for unitary partitioning (anticommutation graphs). It handles molecules with more than 2 million Pauli strings (trillion-edge graphs) in under 15 minutes on one A100. That is 49.5× more vertices and 2,478× more edges than prior parallel codes, with up to 68× memory savings. CPU and GPU code are in the NTUDDSNLab/Picasso repository. — [arXiv 2401.06713](https://www.arxiv.org/abs/2401.06713); [PNNL](https://www.pnnl.gov/publications/picasso-memory-efficient-graph-coloring-using-palettes-applications-quantum-computing)
- Moudghalya, Kockum & Gaikwad (arXiv 2609.02633, Sep 2026) benchmark graph-colouring heuristics for commuting partitions with Clifford synthesis and information-gain prioritization of settings. Systems are only up to 5 qubits or 3 qutrits. Code: github.com/ssm8015/QST_GT. — [arXiv 2609.02633](https://arxiv.org/abs/2609.02633)

### Inferences
- clifford_qc's dyadic block-commuting relation coincides with DalFavero et al.'s k-commutativity on contiguous blocks. Their variable-block generalization (k₁,…,k_m) and freedom of qubit order are exactly the "which qubits share a block" choice that the repo's GROUPING_REDUCTION note flags as unpriced. The repo should cite this prior art, cross-check setting counts and R̂ against `rmlarose/kcommute` (Apache-2.0), and treat qubit-to-block assignment as an optimization variable, which neither paper optimizes.
- The repo's regrouping probe measures interior-rung quality by setting count (the 2.2–2.6× loss). The literature says that is the wrong yardstick. The probe should report a bank-level R or R̂ and the projected-matrix variance under the repo's own estimator (pooled or overlapped). SI ordered by an *effective* per-word weight (Section 3) should be added as a baseline. DSATUR and RLF are known to lose to SI on R̂ even when they win on count.
- The repo's "pooled" estimator, where a word is read from every compatible setting, is Rowland et al.'s post-hoc repacking. Their ad-hoc repacking (inserting by c²/μ before synthesis) and Yen-style ICS with covariance-optimal weights should both beat a shot-weighted pool when covariances are non-zero. Published gains are about 1.2–2.35× for repacking and up to about 5× for ICS with covariance knowledge.

### Gaps
- No source compares k-commuting groupings with overlapping estimators and covariance-aware allocation in one study. None reports k-commuting results for a non-Hamiltonian "bank" of words with several weight vectors.
- Exact R̂ values per k for molecules in DalFavero et al. appear only in figures.
- VGMC's baselines and molecules could not be extracted from the abstract.

---

## 2. Shadow-family and adaptive estimators: which give the lowest total shots for molecular-like word sets, and do any target a general union of words?

### Takeaway
For chemistry-like weighted Pauli sets at fixed state, well-weighted **deterministic** methods are at least as good as randomized shadows. That covers SI or ICS grouping with covariance-informed allocation, ShadowGrouping, decision-diagram derandomization, and 2026 overlapped or IPW grouping. The best covariance-informed grouping beats derandomized shadows and OGM by about 4–6× in variance on NH3 (Yen et al.). Several methods accept an arbitrary list of weighted Pauli words rather than one Hamiltonian, so they apply directly to a union bank: derandomization, ShadowGrouping, derandomized shallow shadows (DSS) and dual-frame IC-POVMs. Shallow-circuit shadows (DSS) are the main new direction that uses entangling readout and still targets a given word list.

### Cited Findings
- **[background]** Derandomized classical shadows (Huang, Kueng & Preskill, PRL 127, 030503, 2021) target any list of L Pauli observables. About log L copies suffice for low-weight observables, and the deterministic scheme is "guaranteed to perform at least as well as the randomized one". The gain is larger when some observables have high weight. — [arXiv 2103.07510](https://arxiv.org/abs/2103.07510)
- ShadowGrouping (Gresch & Kliesch; Nat. Commun., DOI 10.1038/s41467-024-54859-x, per the TUHH repository):
  - It builds single-qubit (QWC, or optionally general-commuting) settings greedily from a tail-bound-derived weight that depends on each term's coefficient and on how many compatible settings it already has. Each pass costs O(M log M).
  - Sim tests: H2 (8 qubits, 6-31G), LiH 12, BeH2 14, H2O 14, NH3 16, under JW, BK and parity mappings.
  - About 10× better than AEQuO on NH3 (JW). OGM is competitive only on small molecules. Derandomization and Adaptive Paulis stop improving as the budget grows.
  - No hardware. The bound applies to any weighted Pauli sum. — [arXiv 2301.03385v3](https://arxiv.org/html/2301.03385v3); [TUHH record](https://tore.tuhh.de/handle/11420/53433)
- CSHOREBench (Dutt, Kirby, Raymond, Hadfield, Sheldon, Chuang, Mezzacapo; arXiv 2312.07497) covers up to 16 qubits and counts classical cost too. On IBM hardware, decision diagrams (DD) used more than 80% fewer measurements than classical shadows and about 57% fewer than LBCS. In simulation, DD and derandomization were the most preferable, and DD needed fewer measurements and less classical runtime than derandomization. — [arXiv 2312.07497](https://arxiv.org/abs/2312.07497)
- Yen et al. 2023 (QWC, CISD-chosen parameters, exact-state variance). NH3: OGM 305, Derand 251, SI 97.0, ICS 53.8. H2O: OGM 148, Derand 114, SI 48.6, ICS 15.9. For random states (Table 5), the gap shrinks: NH3 Derand 865 vs ICS (QWC) 391 vs ICS (FC) 246. — [PMC11041696](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)
- Jørgensen et al. (Mar 2026) present a unified framework covering grouping with inverse-probability weighting, shadow variants, ShadowGrouping, derandomization and OGM. "Simple grouping algorithms" approach or exceed state-of-the-art accuracy on the standard H2–NH3 benchmark. Adding Clifford measurement circuits to otherwise Clifford-free methods (notably ShadowGrouping) cuts cost "by up to a factor of two". For NH3 the lowest cost came from a deterministic ShadowGrouping variant with Cliffordization and ℓ² weighting. Average CNOTs per LDF Clifford circuit range from 4.0 (H2) to 76.0 (NH3) before TKET simplification. — [arXiv 2603.20029v1](https://arxiv.org/html/2603.20029v1); [abs](https://arxiv.org/abs/2603.20029)
- Derandomized shallow shadows (DSS; Van Kirk, Kokail, …, Lukin; arXiv 2412.18973):
  - It outputs shallow brickwork circuits (d two-qubit layers) that approximately minimize sample complexity for a **given set of Pauli strings**, using tensor networks to keep classical cost polynomial.
  - Sim: at d=1, DSS gives the lowest error at M=1000 vs LBCS, derandomization, shallow shadows and OGM on H2 (4 and 8 qubits).
  - Sim, 12-qubit Hubbard H² verification: 1,236 experiments at d=0, about 30% fewer with depth. A 100-site H² has about 2.4×10⁵ Pauli terms.
  - Code: github.com/katherinevankirk/dss (MIT). — [arXiv 2412.18973v1](https://arxiv.org/html/2412.18973v1); [code](https://github.com/katherinevankirk/dss)
- For asymmetric expectation values (MC-PQE, up to 12 qubits, Sim), Baid & Filip (arXiv 2608.30612, Aug–Sep 2026) find that FC Pauli grouping with tailored allocation reduces standard error 5–10× at fixed shots. They also find that conventional grouped measurement outperformed the classical-shadow techniques they tried. — [arXiv 2608.30612](https://arxiv.org/abs/2608.30612)
- Informationally complete (IC) POVMs with optimized dual frames (Fischer, Dao, Tavernelli, Tacchino; PRA 2024) cut shadow-estimator variance purely in post-processing, with no extra quantum cost. Duals can be "individually tailored to different observables estimated from the same data". Malmi et al. (arXiv 2401.18049) show that optimized duals can reach zero variance where canonical-dual shadows have exponential overhead. Open-source implementation: qiskit-community/povm-toolbox (Apache-2.0). — [arXiv 2401.18071v2](https://arxiv.org/html/2401.18071v2); [arXiv 2401.18049](https://arxiv.org/abs/2401.18049); [povm-toolbox](https://github.com/qiskit-community/povm-toolbox)
- Two-copy Bell-basis measurement with Bayesian error estimation and adaptive allocation (Simon, Shi, Nation, Jena, Dellantonio; Quantum 10, 2229, 7 Oct 2026) can beat the state-of-the-art protocol when there are many non-commuting Paulis. It needs two copies, and the paper gives only toy-sized examples. — [Quantum 10, 2229](https://quantum-journal.org/papers/q-2026-10-07-2229/)
- Fermionic and matchgate shadows (Wan, Huggins, Lee, Babbush, arXiv 2207.13723) process each sample in polynomial time and handle local fermionic observables and Gaussian overlaps. A 2025 adaptive-depth variant notes that local Pauli shadows have poor sample complexity for Majorana strings, which are nonlocal in the Pauli basis. I found no head-to-head energy benchmark vs Pauli grouping. — [arXiv 2207.13723](https://arxiv.org/pdf/2207.13723); [arXiv 2501.09574](https://web3.arxiv.org/pdf/2501.09574)

### Inferences
- For a fixed, known bank on one state, the strongest evidence favours deterministic grouping with overlap and covariance-aware allocation over randomized shadows (Yen 2023; Baid & Filip 2026; Jørgensen 2026). Shadows only help where the word set is unknown in advance or very high weight. Since clifford_qc already has the bank, a shadow layer is unlikely to beat a pooled, overlapped, Neyman-allocated QWC/k-commuting scheme. The exception worth testing is DSS with d=1–2, because it directly searches entangling readouts for a given word list and has MIT-licensed code.
- Derandomization, ShadowGrouping and DSS all take a word list plus per-word weights. A clifford_qc bank with effective weights from the Ritz-value sensitivity (Section 3) could be passed to them unchanged as an external baseline.
- With 804 of 7,371 words at full weight (the repo's QWC floor), QWC-only shadows inherit the same floor. Only entangling readouts (k ≥ 2, HT circuits or DSS) can go below it.

### Gaps
- I found no published benchmark of ShadowGrouping, DSS or derandomization on a union-of-matrix-element bank (several weight vectors over one word set).
- I could not verify the official ShadowGrouping repository URL or its licence. GitHub search returned only an unofficial "Shadowgrouping code" repository.
- Decision-diagram code availability was not established.

---

## 3. Many observables with arbitrary weights and nonlinear targets (Ritz values): joint grouping and shot optimisation

### Takeaway
The closest published work is on **Krylov / subspace matrix elements**. Lee, Choi, Huh & Izmaylov (Digital Discovery 2025) measure ⟨φ0|H|φk⟩ and S_0k with SI fragments plus a symmetry "shift" and coefficient splitting across circuits that share terms. They cut Mε² by 12–700× vs SI, but they optimize the *sum of matrix-element variances*, not the eigenvalue. Oumarou et al. (2026) allocate shots by |∂E₀/∂moment|, computed by autodiff through the regularized generalized eigenproblem, which is a first-order (delta-method) allocation. I found no paper that jointly optimizes grouping *and* shots for a nonlinear Ritz-value objective, or that uses formal optimal-experimental-design criteria (A-, D- or E-optimality) for Pauli banks.

### Cited Findings
- Lee et al. ("Efficient strategies for reducing sampling error in quantum Krylov subspace diagonalization", Digital Discovery 4, 954–969, 2025; arXiv 2409.02504):
  - Matrix elements H_0k and S_0k are measured either by Hadamard-test LCU (anticommuting groups) or by Clifford-diagonalizable fragments with an extended swap test (commuting groups, SI).
  - **Shift**: minimize ‖H − T(τ)‖₁ subject to T|φ0⟩ = t|φ0⟩, then add back t·S_0k. This removes number-operator-type terms that annihilate the reference. Closed-form parameters: τ_q^(1) = 2h_qq and τ^(2)_qrs = 4g_rsqq − 2g_rqsq.
  - **ICS across circuits**: minimize ΣVar/m with covariances from a CISD proxy state.
  - Mε² in Ha², SI → Shift+ICS, fragment (FH) variant: H2 2.18 → 0.13, H4 34.66 → 0.80, LiH 50.19 → 0.37, BeH2 151.65 → 1.46, H2O 2,284.67 → 3.25. LCU variant: H2O 7,265 → 14.5.
  - With thresholding of S, H2O energies (10⁸ shots) reach chemical accuracy only with these techniques.
  - A Python package is released (URL not extracted). — [arXiv 2409.02504v4](https://arxiv.org/html/2409.02504v4); [abs](https://arxiv.org/abs/2409.02504)
- Oumarou, Ollitrault, Polla & Gogolin (arXiv 2603.15552, Mar 2026):
  - Allocation rule for QKSD: M_k = ⌊|g_k|/Σ|g_j| · M⌋, where g_k = ∂E₀/∂⟨T_k(H)⟩ is obtained by JAX autodiff through the regularized GEVP. In simulation it is evaluated at noiseless moments; on hardware they propose a pilot-then-allocate scheme.
  - "The number of samples required for QKSD is largely dominated by the need to reduce the noise on the eigenvalues of the overlap matrix". They use M_guess = 1/(sε)², with s the smallest retained S-eigenvalue.
  - Systems: Fe2S2 (30e, 20o), Fe4S4 (54e, 36o), Co(salophen), naphthalene, all MPS simulations. No allocation-vs-uniform numbers are given. — [arXiv 2603.15552v1](https://arxiv.org/html/2603.15552v1)
- Kirby (Quantum 8, 1457, 2024) analyzes quantum Krylov with generic errors. The energy error upper bound is linear in the input error rate η. With overlap-eigenvalue threshold ε = O(√η), both upper and lower bounds are O(√η). — [arXiv 2401.01246](https://arxiv.org/abs/2401.01246)
- Delta-corrected shot reduction (DCSR; Ghozaly et al., arXiv 2611.10524, 7 Nov 2026) uses the multivariate delta method to correct finite-pilot bias in the empirical reduction factor of variance-preserved shot reduction (VPSR). It reports faster convergence than VPSR at equal or lower cumulative shots, with numbers only in the body. — [arXiv 2611.10524](https://arxiv.org/abs/2611.10524)
- QSCI (Kanno et al., arXiv 2302.11320) derives shot allocation for several observables that minimizes total shots subject to a bound on the *sum* of variances. It notes that this differs from bounding the maximum per-observable variance. — [arXiv 2302.11320](https://arxiv.org/pdf/2302.11320)
- In ADAPT-VQE gradient measurement (arXiv 2306.03227), a Lagrangian minimizes the summed variances of gradients measured together. At the variance upper bound, shots ∝ |weight| are optimal. — [arXiv 2306.03227](https://arxiv.org/pdf/2306.03227)
- The 2025 review states that matrix elements and overlaps can be written as expectation values and measured with the same partitioning tools (Hadamard or extended swap tests). It cites excited-state partitioning with up to two orders of magnitude lower cost than greedy-only grouping, and a BLISS shift giving 1–2 orders of magnitude improvement for matrix elements of small molecules. — [arXiv 2501.14968v2](https://arxiv.org/html/2501.14968v2)
- MC-PQE asymmetric expectation values (⟨Φ_i|H|Ψ⟩-type, up to 12 qubits): FC grouping with tailored allocation gives 5–10× lower standard error. — [arXiv 2608.30612](https://arxiv.org/abs/2608.30612)

### Inferences
- For clifford_qc, the natural objective is first-order (delta-method) propagation to the Ritz value. For a generalized eigenpair (λ, c) of (H, S), δλ ≈ c†(δH − λ δS)c / c†Sc (standard first-order perturbation theory). Since every H_ij and S_ij is linear in the bank, ⟨P_w⟩ → λ has gradient g_w = Σ_ij c_i* c_j (∂H_ij/∂⟨P_w⟩ − λ ∂S_ij/∂⟨P_w⟩) / c†Sc.
- Then Var(λ̂) ≈ gᵀ Σ g, where Σ is the bank's grouped covariance, which the repo already estimates. Neyman allocation over settings should use per-setting variance of gᵀP̂ restricted to that setting, not raw word variance. This is what Oumarou et al. do with autodiff, and it plugs straight into the repo's covariance-aware allocator.
- For several Ritz values, or a Ritz value plus residuals, use a weighted sum of gᵀΣg (A-optimal) or minimize the max (E/minimax). Kanno et al. caution that these differ.
- The effective weights g_w also supply the coefficients for SI ordering, ICS splitting and R̂-style grouping metrics. That turns the bank, which has no Hamiltonian of its own, into a weighted Pauli sum that every Hamiltonian-oriented method in Sections 1–2 can consume.
- The Lee et al. shift idea carries over to A-CASE when A_j|φ0⟩ is an eigenvector of a cheap symmetry operator: subtract the symmetry part from H inside each matrix element to shrink the ℓ¹ norm before grouping.
- Because thresholding of S dominates the shot budget (Oumarou; Lee; Kirby), the allocation should also weight words by their contribution to S's small retained eigenvalues, not only by their contribution to H.

### Gaps
- I found no paper on joint grouping-and-allocation for a nonlinear Ritz-value target. None uses formal optimal-design criteria for Pauli-bank measurement, and none validates delta-method variance for Ritz values against Monte Carlo at realistic conditioning.
- Lee et al.'s package URL and licence could not be extracted.

---

## 4. Fully commuting vs QWC in practice, and the best synthesis results for diagonalizing commuting Pauli sets

### Takeaway
In noiseless or simple-noise simulation, FC beats QWC by 2–7× in variance (Yen; GALIC 3.4× mean). On real or realistic noisy devices, FC's extra two-qubit gates produce **bias**, not just variance. GALIC reports FC exceeding 1 kcal/mol bias in 20 of 20 cases, and on IBM Kyoto FC plateaued at about 2 kcal/mol. HT readout on ibm_washington beat QWC only below about 10⁴ shots. The emerging consensus is noise-aware hybrids (GALIC, k-commuting, HT/connectivity-native). Synthesis bounds for diagonalizing r independent commuting Paulis on n qubits are about n·r − r(r+1)/2 CNOTs (Crawford; Murairi & Cervia), improved to n⌊r/2⌋ − ⌊r/2⌋² with optimization at O(n log r) depth. HT circuits need no SWAPs, with CZ depth ≤ graph degree (3 on heavy-hex).

### Cited Findings
- **[background]** Crawford et al.: rotating k commuting n-qubit Paulis to Z needs at most kn − k(k+1)/2, or O(kn/log k), two-qubit gates. — [Quantum 5, 385](https://quantum-journal.org/papers/q-2021-01-20-385/)
- Murairi & Cervia ("Reducing circuit depth with qubitwise diagonalization", arXiv 2306.00170, Dec 2023):
  - Bounds: CNOT ≤ n·r − r(r+1)/2 unoptimized and ≤ n⌊r/2⌋ − ⌊r/2⌋² with per-stage null-space search (practical for n ≲ 30), at depth O(n log r).
  - STO-3G averages per commuting set: LiH (12 qubits, 25 sets) 7.4 CNOT, depth 11.1. NH3 (16 qubits, 80 sets) 21.4 CNOT, depth 22.5.
  - 8-qubit H4 (BK) on linear connectivity: their largest-set circuit uses 5 CNOT + 11 SWAP, vs the cited HT-reference circuit's 18 CNOT + 21 CZ + 66 SWAP. — [arXiv 2306.00170v3](https://arxiv.org/html/2306.00170v3)
- The 2025 review notes that a general Clifford needs O(N²/log N) gates. It says QWC avoids SWAP overhead entirely (each SWAP is 3 CNOT), and that FC trades lower variance for higher bias under noise. — [arXiv 2501.14968v2](https://arxiv.org/html/2501.14968v2)
- HT diagonalization:
  - GC circuits can need up to n(n−1)/2 two-qubit gates plus O(n²) SWAPs on limited connectivity, while HT circuits need no SWAPs.
  - HW on ibm_washington, 8-qubit H chain, 184 terms, readout mitigation: HT error < TPB error for budgets below 10⁴ shots. The asymptotic bias is slightly worse for HT (0.0299 vs 0.0284 Ha). — [arXiv 2203.03646v3](https://arxiv.org/html/2203.03646v3)
- GALIC on HW (IBM Kyoto, H4 Hartree–Fock state): FC variance 2.1× below QWC, GALIC 1.2×, HEC 1.27×. FC bias plateaued at about 2 kcal/mol vs about 1.15 kcal/mol for the others. A regression over device models shows the error-rate term affects device-aware variance more than 10× as strongly as connectivity (abstract: ">10×"). The separate "13.6×" figure is HEC's 90th-percentile bias relative to GALIC's. — [arXiv 2409.00576v2](https://arxiv.org/html/2409.00576v2)
- **[background-adjacent, 2022]** Bansingh, Yen, Johnson & Izmaylov (arXiv 2205.07113): with a simple model of errors added by extra rotation gates, non-local (FC) schemes still need fewer measurements than local ones for molecular Hamiltonians. The abstract gives no thresholds. — [arXiv 2205.07113](https://arxiv.org/abs/2205.07113)
- Yen et al. 2023 likewise state that FC needs fewer measurements than QWC even after CNOT infidelity is accounted for. They report no depths. — [PMC11041696](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)
- Maslov & Yang (attribution from the search index): a CNOT-CZ stage can be implemented in depth 5n on linear nearest neighbour (LNN), giving any Clifford in depth ≤ 7n−4 on LNN. — [arXiv 2210.16195](https://web3.arxiv.org/pdf/2210.16195); [npj QI PDF](https://www.nature.com/articles/s41534-023-00760-2.pdf)
- QED-C (2025) uses k-commuting groups precisely because they "enable shorter Clifford diagonalization circuits". — [arXiv 2504.09813](https://arxiv.org/abs/2504.09813)

### Inferences
- clifford_qc's FC rung uses 1,886 logical CX over 64 settings, about 29.5 CX per setting on n=8. That is above the *unoptimized* Crawford / Murairi–Cervia bound for a maximal set (r=8): 8·8 − 36 = 28. It is roughly 2× the optimized bound of 8·4 − 16 = 16. Graph-state form (diagonalizer = local Cliffords + CZ graph, at most n(n−1)/2 = 28 CZ) on all-to-all hardware can be edge-coloured into ≤ 7 CZ layers (K8 has chromatic index 7), vs the reported mean two-qubit depth of 24.
- So the Stim tableau elimination looks far from optimal for *diagonalization*. It likely synthesizes more of the Clifford than needed, since only the X-part must be cleared and the Z-side and Pauli frame are free. The repo's opt-in graph-state path should be benchmarked against these bounds and made the default if it wins.
- For realistic noise, the literature favours a noise-gated compatibility test (GALIC-style CNOT budget per group from p and ε_tar) or connectivity-native groups (HT) over a fixed k. The repo's block hierarchy could add an acceptance test of the form "logical CX after routing ≤ budget", evaluated per group.

### Gaps
- I found no 2023–2026 study that measures the FC/QWC break-even on hardware for a bank of thousands of words, or for a non-Hamiltonian objective.
- Bansingh et al.'s numeric thresholds were not extracted.
- No 2024–2026 paper was found that gives provably optimal CZ count or depth for diagonalizing commuting Pauli sets on heavy-hex beyond HT's degree bound.

---

## 5. Fermionic-structure alternatives (basis-rotation / double factorization, fluid fragments, ghost Paulis, Cartan, THC) and applicability to A_i†HA_j

### Takeaway
Fermionic fragment methods are among the best for molecular **H**. Low-rank and double-factorized basis rotations, fluid fermionic fragments (F3, more than 10× fewer measurements), ghost Paulis (several-fold) and the 2026 Bincoletto–Kottmann protocol (30–80%) all exploit the two-body form of H. They have been extended to Krylov and MC-PQE matrix elements *between states*, via the symmetry shift plus ICS. I found none designed for operator *products* A_i†HA_j, whose fermionic degree grows with the A's. For a Pauli bank produced by products, the transferable ideas are the coefficient-shifting family (ghost Paulis, symmetry shifts) rather than orbital-rotation fragments.

### Cited Findings
- **[background]** Low-rank factorization measurement (Huggins et al., npj QI 2021) factorizes the two-electron tensor. Each fragment is measured after an orbital rotation (N choose 2 Givens rotations, depth N) that maps it to number operators. It requires a linear-depth pre-measurement circuit but avoids nonlocal Jordan–Wigner strings. — [npj QI s41534-020-00341-7](https://link.springer.com/10.1038/s41534-020-00341-7)
- Fluid fermionic fragments (Choi, Loaiza & Izmaylov, Quantum 7, 889, 2023) move parts of two-electron fragments into a one-electron fragment using occupation-number idempotency. This leaves ⟨H⟩ unchanged and cuts measurements by more than an order of magnitude on several molecules. — [Quantum 7, 889](https://quantum-journal.org/papers/q-2023-01-03-889/)
- Ghost Pauli products (Choi, Yen & Izmaylov, JCTC 2022) add Pauli products, compatible with several fragments, whose coefficients sum to zero across fragments. They report a several-fold reduction vs other recent techniques. — [arXiv 2208.06563](https://arxiv.org/abs/2208.06563v1)
- Bincoletto & Kottmann (Digital Discovery 5, 1257–1268, 2026; CC-BY) iteratively measure easily accessible operator groups in different fermionic bases and report a 30–80% reduction in measurements and measuring-circuit gate depth vs state-of-the-art methods. — [RSC landing page](https://pubs.rsc.org/en/Content/ArticleLanding/2026/DD/D5DD00251F)
- The 2025 review: low-rank (SVD or Cholesky) fragments are much cheaper to obtain than full-rank fragments, which need O(N⁵)-per-evaluation nonlinear optimization. Symmetry-incorporated fragments give "considerably worse" statistics. Fermionic excitation grouping yields O(N³) fragments. — [arXiv 2501.14968v2](https://arxiv.org/html/2501.14968v2)
- Symmetry shift for matrix elements: number-operator terms that annihilate |φ0⟩ cut the ℓ¹ norm by 74–96% across H2…H2O (Lee et al. Table 1). — [arXiv 2409.02504v4](https://arxiv.org/html/2409.02504v4)

### Inferences
- If A-CASE's A_i are Pauli or Clifford operators, A_i†HA_j is a generic Pauli sum. Orbital-rotation fragments no longer apply, and Pauli-level tools (grouping, ICS, ghost Paulis) are the right layer.
- If some A_i are fermionic excitations, A_i†HA_j becomes a higher-body fermionic operator. Double factorization does not directly cover it. Symmetry shifts (N, S_z) still help whenever A_j|φ0⟩ stays in a symmetry sector.
- Ghost-Pauli or coefficient-splitting freedom is natural in a bank: a word that feeds several matrix elements can be "split" across settings at no bias.

### Gaps
- I found no paper applying THC to VQE-style measurement, and none applying double factorization, Cartan grouping or F3 to operator products A_i†HA_j.
- The Cartan subalgebra approach (Yen & Izmaylov, PRX Quantum 2021) was not re-checked in this pass.

---

## 6. Shot allocation and adaptivity: variance-optimal, covariance-aware, iterative, Bayesian, readout-aware

### Takeaway
Optimal static allocation is the Rubin/Neyman rule: shots ∝ |c|·σ per term, or ∝ √Var(O_ℓ) per group. Covariance-aware iterative allocation and coefficient splitting add 15–80% on top of SI in noiseless sims (Yen 2023). Reported 2025–2026 gains from allocation alone are 37.6% error reduction (QED-C) and 18–23% from joint construction plus allocation (VGMC). Adaptive schemes with rigorous stopping exist: AEQuO, ROGS (convex), and empirical-Bernstein stopping, which is up to 10× tighter than Hoeffding guarantees. Bayesian (Dirichlet/Beta) per-term posteriors are standard. I found no published readout-error-aware *allocation*.

### Cited Findings
- **[background]** Rubin et al. (2018) proved that, at fixed total budget, variance is minimized by shots ∝ |coefficient| × √(Pauli variance). — [VQE-to-SQD review, arXiv 2509.21555](https://arxiv.org/pdf/2509.21555); [HT paper](https://arxiv.org/html/2203.03646v3)
- Yen et al. 2023, exact covariances:
  - Allocation-only methods vs SI, FC: IMA (iterative) and GMA (gradient) give NH3 18.8 → 13.6 → 9.35 and H2O 7.59 → 5.88 → 4.27.
  - With covariances estimated from 1,000 simulated shots per fragment, *including that cost*, the results match exact covariances (Table 6). Random states, ε = 10⁻³ Ha, in millions of shots: NH3 FC SI 324 vs ICS 249. — [PMC11041696](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)
- AEQuO (Shlosberg et al., Quantum 7, 906, 2023) monitors the running mean and error and chooses the next measurement adaptively. It has a greedy bucket-filling variant for small cases and an ML variant for larger ones, and works on an edge-weighted covariance graph. ShadowGrouping later reported about 10× better accuracy than AEQuO on NH3. — [Quantum 7, 906](https://quantum-journal.org/papers/q-2023-01-26-906/); [arXiv 2301.03385v3](https://arxiv.org/html/2301.03385v3)
- ROGS (Li, Lin & Beach, Quantum 9, 1694, 2025): QWC minimum-clique-cover groups, then convex optimization of a Hoeffding-type confidence bound to allocate shots. The bound prioritizes total-H accuracy rather than per-observable guarantees and can exceed 1. They report the lowest error at a given budget with significantly fewer circuits. — [Quantum 9, 1694](https://quantum-journal.org/papers/q-2025-04-07-1694/)
- Empirical Bernstein stopping (EBS) plus grouping (Gresch, Tepe & Kliesch, JCTC 21(15), 7352–7359, 2025) stops sampling using the state's empirical variance. It "consistently improves upon elementary readout guarantees up to one order of magnitude" vs the Hoeffding-based estimator. — [arXiv 2502.01730](https://arxiv.org/abs/2502.01730)
- QED-C framework: weighted shot distribution gives a further 37.6% error reduction after grouping. — [arXiv 2504.09813](https://arxiv.org/abs/2504.09813)
- Rowland et al. 2026: for overlapped groups under zero covariance, the minimum-variance weights are shot-proportional, and the allocation is M_j ∝ √(Σ_i c_i²) in the state-independent model. — [arXiv 2604.07156v1](https://arxiv.org/html/2604.07156v1)
- Bayesian estimation: a Dirichlet(Beta) prior per Pauli term updates by ±1 counts, which gives Laplace-smoothing-like estimators. — [CSHOREBench, arXiv 2312.07497](https://arxiv.org/pdf/2312.07497)
- AQUIRE (arXiv 2605.00682, 2026, qudits) folds hardware noise into a Bayesian adaptive estimator and claims the fewest measurements for a target accuracy. Not independently verified. — [arXiv 2605.00682](https://arxiv.org/html/2605.00682)
- Readout: linear-inversion readout mitigation removes bias but amplifies finite-shot fluctuations. Within a grouped configuration, one corrected distribution serves every compatible Pauli. I found no allocation scheme that uses confusion matrices. — [arXiv 2608.18548](https://arxiv.org/pdf/2608.18548)
- DCSR (Nov 2026) corrects finite-pilot bias in VPSR-type adaptive reduction with the multivariate delta method. — [arXiv 2611.10524](https://arxiv.org/abs/2611.10524)

### Inferences
- clifford_qc's 68.9% variance cut from Neyman allocation on a four-qubit test sits at the high end of published allocation-only gains (about 25–45% in Yen's tables and 37.6% in QED-C). The next increments should come from (a) target-aware weights (Section 3), (b) overlap with covariance-optimal coefficient splitting, and (c) sequential allocation with an EBS or confidence-sequence stopping rule.
- Pilot cost matters. Yen shows 1,000 shots per group is enough for covariance estimates at 16 qubits. DCSR shows small pilots bias the reduction factor, so the repo's pilot size should be reported with any savings figure.
- Readout-aware allocation is an open niche. Mitigation inflates per-setting variance by a computable factor, so a Neyman allocator could include it directly.

### Gaps
- I found no head-to-head of Bayesian vs frequentist adaptive allocation on the same chemistry benchmark.
- No readout-error-aware allocation paper was found.
- AQUIRE's claims are unverified.

---

## 7. Uncertainty quantification for quantities assembled from grouped measurements

### Takeaway
Practice uses three tools: shot- or record-level nonparametric bootstrap, which preserves within-setting covariances; analytic delta-method covariances; and rigorous tail or confidence bounds (Hoeffding, empirical Bernstein, confidence sequences). Rigorous bounds are validated by proof. Bootstrap intervals for nonlinear eigenvalue quantities are used widely, but I found **no published coverage study** for grouped-measurement bootstraps or for Ritz values near ill-conditioned S. For subspace methods, the dominant UQ issue is the thresholding of S, which mixes statistical error with projection bias.

### Cited Findings
- A joint nonparametric bootstrap resamples whole circuit records (2,000 replicates) to keep empirical covariances among Paulis read from the same record. The paper notes covariance terms are non-negligible. It describes the method but reports no coverage study. — [arXiv 2607.24937](https://arxiv.org/pdf/2607.24937)
- Fewer groups can introduce larger covariances and increase total shots for a target precision (generator-coordinate methods paper). — [arXiv 2212.09205](https://arxiv.org/pdf/2212.09205)
- Rigorous guarantees: ShadowGrouping is built on tail bounds for the grouped estimator, and EBS adds an empirical-variance tail bound with adaptive stopping. — [arXiv 2301.03385v3](https://arxiv.org/html/2301.03385v3); [arXiv 2502.01730](https://arxiv.org/abs/2502.01730)
- Closed-form empirical-Bernstein *confidence sequences*, which are time-uniform and valid under optional stopping, are asymptotically tighter than earlier closed-form confidence sequences (Chugg & Ramdas, Dec 2025, non-quantum). — [arXiv 2512.21300](https://arxiv.org/abs/2512.21300v1)
- ROGS's bound is explicitly not a per-observable guarantee and can exceed 1. — [Quantum 9, 1694](https://quantum-journal.org/papers/q-2025-04-07-1694/)
- Subspace UQ: thresholding S at ε_th = Õ(n/√M_S) trades statistical error for projection bias (Lee et al.). Discarding S-eigenvalues "that cannot be reliably distinguished from statistical noise" sets the shot count (Oumarou et al.). The error bound is O(√η) with ε = O(√η) (Kirby). — [arXiv 2409.02504v4](https://arxiv.org/html/2409.02504v4); [arXiv 2603.15552v1](https://arxiv.org/html/2603.15552v1); [arXiv 2401.01246](https://arxiv.org/abs/2401.01246)
- Confidence intervals for convex functions of informationally incomplete measurement data have been developed (arXiv 2311.06094). Not examined in detail here. — [arXiv 2311.06094](https://arxiv.org/html/2311.06094v2)

### Inferences
- clifford_qc's grouped bootstrap, which resamples shots within each setting, is the right construction for preserving covariance. Pooled or overlapped estimators reuse each shot for many words, so the bootstrap must resample at the shot level within each setting, never per word.
- The repo should run its own coverage study: repeated finite-shot simulations, checking nominal 68% and 95% coverage of bootstrap and delta-method intervals for Ritz values, stratified by smallest retained S-eigenvalue. The literature does not supply this.
- For adaptive allocation with stopping, an empirical-Bernstein confidence sequence per setting, combined over settings through the effective weights g, gives a valid stopping rule. Naive bootstrap after adaptive stopping does not.

### Gaps
- No validated coverage study for bootstrap or delta-method intervals on generalized-eigenvalue outputs from grouped Pauli data was found.
- No confidence-sequence method tailored to correlated multi-word settings was found.

---

## 8. Tooling: open-source implementations, licences and maturity

### Takeaway
Mainstream SDKs provide only disjoint QWC/FC colouring. PennyLane has LF, RLF, DSATUR and GIS, and Qiskit has greedy group_commuting, with each word in exactly one group. The state-of-the-art pieces are research codes: k-commutativity (Apache-2.0), HT-Grouper (C++), DSS (MIT), OGM, Picasso (GPU colouring), povm-toolbox (Apache-2.0) and symmer (MIT). For ICS, ShadowGrouping, GALIC and Lee et al.'s shift+ICS, code is promised or released but I could not verify licences.

### Cited Findings
- PennyLane (Apache-2.0): `qml.pauli.group_observables` solves minimum clique cover by graph colouring, with methods 'lf', 'rlf', 'dsatur' and 'gis'. The last two need rustworkx ≥ 0.15. The docs disagree on whether the default is 'rlf' or 'lf'. Also provided: `optimize_measurements` and `estimate_shots`. — [PennyLane docs](https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli.group_observables.html); [source](https://docs.pennylane.ai/en/stable/_modules/pennylane/pauli/grouping/group_observables.html); [repo](https://github.com/PennyLaneAI/pennylane)
- Qiskit: `SparsePauliOp/PauliList.group_commuting(qubit_wise=…)` has existed since 0.21. Qiskit's circuit-cutting add-on notes it puts each observable in at most one set. — [Qiskit API](https://docs.quantum.ibm.com/api/qiskit/0.38/qiskit.quantum_info.SparsePauliOp); [qiskit-addon-cutting source](https://qiskit.github.io/qiskit-addon-cutting/_modules/qiskit_addon_cutting/utils/observable_grouping.html)
- k-commutativity code: rmlarose/kcommute, Apache-2.0, last updated Sep 2025. — [GitHub](https://github.com/rmlarose/kcommute); [paper](https://arxiv.org/html/2312.11840v3)
- HT-Grouper (C++, Miller et al.): github.com/Mc-Zen/HT-Grouper. Licence not verified. — [arXiv 2203.03646v3](https://arxiv.org/html/2203.03646v3)
- DSS: katherinevankirk/dss, MIT, updated Jun 2026. — [GitHub](https://github.com/katherinevankirk/dss)
- OGM reference code: GillianOoO/Overlapped-grouping-measurement. No licence detected by GitHub. — [GitHub](https://github.com/GillianOoO/Overlapped-grouping-measurement)
- Picasso GPU/CPU colouring: NTUDDSNLab/Picasso. Licence not verified. — [arXiv 2401.06713](https://www.arxiv.org/abs/2401.06713)
- IC-POVM and dual-frame estimators: qiskit-community/povm-toolbox, Apache-2.0. — [GitHub](https://github.com/qiskit-community/povm-toolbox)
- symmer (unitary partitioning and qubit subspace; qmatter-labs): MIT, active as of Nov 2026. — [GitHub](https://github.com/qmatter-labs/symmer)
- Tequila (tequilahub; Kottmann is a co-author of the 2026 fermionic measurement protocol): MIT, active Nov 2026. Whether that protocol ships in Tequila was not verified. — [GitHub](https://github.com/tequilahub/tequila)
- OpenFermion: Apache-2.0, active. — [GitHub](https://github.com/quantumlib/OpenFermion)
- Microsoft QDK/Chemistry: MIT, has a `pauli_commutation` utility module. — [docs](https://microsoft.github.io/qdk-chemistry/api/api_autogen/qdk_chemistry.utils.pauli_commutation.html); [GitHub](https://github.com/microsoft/qdk-chemistry)
- QED-C application-oriented benchmarks integrate k-commuting grouping, Clifford measurement circuits and weighted shots on HamLib. — [arXiv 2504.09813](https://arxiv.org/abs/2504.09813)
- Graph-colouring QST settings code: github.com/ssm8015/QST_GT. Licence not stated. — [arXiv 2609.02633](https://arxiv.org/abs/2609.02633)
- Lee et al. (shift + ICS for Krylov matrix elements): released as a Python package with reproduction data. URL and licence not extracted. — [arXiv 2409.02504v4](https://arxiv.org/html/2409.02504v4)

### Inferences
- The cheapest high-value external baselines for clifford_qc are kcommute (direct comparison of the dyadic hierarchy, Apache-2.0), DSS (MIT; shallow entangling readouts for a given word list), povm-toolbox (Apache-2.0; post-hoc dual-frame variance reduction) and HT-Grouper (connectivity-native FC groups). All four accept or can be adapted to an arbitrary weighted word list.
- No mainstream SDK implements overlapping or coefficient-split estimators or target-aware allocation. clifford_qc's pooled estimator plus covariance-aware Neyman allocation is already ahead of PennyLane and Qiskit defaults. Its remaining gap is in the objective (variance, not count) and in circuit synthesis.

### Gaps
- Licences for HT-Grouper, Picasso, the official ShadowGrouping code, GALIC's data or code and Lee et al.'s package were not verified. GitHub API access to those repositories was unavailable from this session.
- Maturity (tests, releases) was assessed only by last-update dates.
