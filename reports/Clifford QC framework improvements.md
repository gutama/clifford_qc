# Stabilise the pencil before scaling clifford_qc

clifford_qc's most valuable next improvements are statistical and engineering upgrades aimed at the failure modes its own records document. New subspace variants are not among them. The repository shows that A-CASE loses at matched budget: it brings 0 of 12 systems within 1.6 mHa at M=7, against 9 for Davidson, and CCSD is more accurate on 7 of 7 molecular geometries. Finite-shot projected energies also fail badly: one million shots still leave a median error of 196.80 mHa on stretched H₂O, and 65–95% of seeds violate the variational bound. The live question is therefore whether the noisy (H, S) pencil can be estimated reliably. Work from 2024–2026 offers fixes for exactly that which need no extra shots: nearest-correlation projection plus pencil rotation (up to ~100× fewer samples), error-budgeted ratio minimisation demonstrated on 80 qubits, and shot allocation aimed at the Ritz value. All three can be tested by replaying the preregistered, seeded 180-cell PRD campaign.

Next, the Pauli-word bank should shrink at its source. Filtering the generator pool by point group and spin is exact and cut UCCSD parameter counts by 50–81% in published work. By contrast, no Majorana-to-Pauli encoding can change the distinct-word count W at all.

The pure-Python dictionary product spends about half of a 14-qubit solve building pairs. A NumPy structure-of-arrays kernel ran 3–41× faster in this session with no new dependency.

The measurement layer has two problems. It reports setting counts where the field optimises variance, and its dyadic hierarchy duplicates published k-commutativity without citing it. It also already contains a graph-state readout compiler that cuts two-qubit gates by 75.6% on the H₄ fully commuting rung, but keeps it opt-in.

Credibility now depends on four things the project lacks:
- an execution path through Qiskit primitives;
- community benchmark instances with strong classical baselines;
- a PyPI release;
- a rendered documentation site.

Finally, any bank on a determinant reference can be evaluated exactly on a classical computer. A quantum case for A-CASE therefore has to be built on correlated references that classical methods cannot handle.

## Seven changes ranked by evidence and cost

The project's own ledger already says where the problem is. All 20 numbered phases are implemented, but the scientific record is mostly bounded or negative results. The last declared campaign closed negative on 1 November 2026, and the stated forward claim is "whether that compactness survives finite-shot energy accuracy" ([PLAN.md](../PLAN.md); [PHASE_STATUS.json](../PHASE_STATUS.json)). The ranking below puts first the changes that answer that question with the least new machinery. It then lists the changes that let outsiders compare and check the results.

| Rank | Change | Strongest evidence | Effort | What to report |
|---|---|---|---|---|
| 1 | Pencil estimator ladder (PSD projection, rotation, constrained ratio, posteriors) plus an interval-coverage study | Up to ~100× fewer samples (preprint); ratio estimator worked on 80 qubits; no published coverage study exists | Low: classical post-processing, replays seeded campaigns | Median, 90th-percentile error and violation rate on the 180 PRD cells; empirical coverage |
| 2 | Shot allocation and grouping weighted by Ritz-value sensitivity | Overlap-matrix noise dominates Krylov shot budgets; symmetry shift plus splitting gives 20–500× | Low–medium | Var(λ̂) per shot against the current Neyman allocator |
| 3 | Shrink the bank at source: point-group/spin pool filtering, word-cost-aware scoring, symmetry shifts | Parameter cuts of 50–81%; H₄ selection cache of 14,401–15,803 words | Medium | W and total selection-cache shots at matched error |
| 4 | Make graph-state readout the default; cite k-commutativity; report R̂ and variance, not counts | 1,886 → 460 two-qubit gates on H₄ k=8, already built | Low | R̂ per rung, CZ per setting, projected variance |
| 5 | NumPy structure-of-arrays (SoA) kernel, global collate, retire the memo; ffsim sector backend | 3–41× measured in this session; pair building ≈52% of a 14-qubit solve | Medium | Committed throughput benchmark; 14-qubit wall time |
| 6 | Qiskit-primitives execution bridge; HamLib and "chemically decisive" instances; hardened baselines | No device path exists; Davidson stopped on the wrong root | Medium | One Open Plan run; shots to 1.6 mHa against ADAPT-GCIM and selected CI |
| 7 | PyPI release, Sphinx docs, narrowed API, evidence repository split | No index release; "poorly documented" is the top adoption barrier at 50.8% | Low–medium | Release, docs site, JOSS clock started |

Items 1–3 should come before any new compactness campaign, because the project's own rule is that new claims must be "about accuracy-matched cost … not about counts" ([PLAN.md](../PLAN.md)). Item 5 changes no science and can run in parallel. Several tempting directions should be dropped:
- New encodings cannot lower W (see the next section but one).
- GPU kernels pay off only near 10⁷–10⁸ output terms.
- Randomised classical shadows lose to covariance-aware grouping when the bank is known in advance.
- A PennyLane device plugin buys an operator library little.

Many literature figures below come from preprints or from library authors benchmarking their own code on their own hardware. Each is marked accordingly, and none substitutes for a matched clifford_qc measurement.

## Finite-shot pencil fixes cost zero extra shots

Every finite-shot failure in the record traces to estimating S and H from noisy expectations. The preregistered PRD campaign used frozen packet Davidson at M=7 across 180 cells. One million shots still left median errors of 6.35 mHa on stretched H₄ and 196.80 mHa on stretched H₂O. **Variational-violation rates were 65% and 95%**, and chemical accuracy came in at most 5% of seeds ([PLAN.md](../PLAN.md)). Phase 4R found that "the tail, not the median, is the error". The median was about 4.5 mHa, but the RMSE was **1,786 mHa because 4 of 200 replicas landed Hartrees away**. Thresholding removed the tail only by doubling the median and adding about +10 mHa of bias ([PLAN.md](../PLAN.md); [finite_shot.tex](../paper_dacase/tables/finite_shot.tex)). Raising κ_S by four orders of magnitude cut successful bootstrap replicas from 200/200 to 137/200 ([DA-CASE manuscript](../paper_dacase/manuscript.tex)). The A-CASE manuscript concedes that regularising a measured S "does not by itself restore a variational bound" and that it supplies no response certificate ([A-CASE manuscript](../paper_acase/manuscript.tex)).

Theory has caught up with this failure mode, and none of it depends on time evolution:
- Kirby proved that Krylov-type ground-energy error grows **linearly** in the matrix-element error. He also bounded how far noise can push an estimate below E₀ ([arXiv:2401.01246](https://arxiv.org/abs/2401.01246)).
- Lee, Lee and Huh tied Gaussian shot noise to eigenvalue error non-asymptotically and recommended discarding ill-conditioned directions at a truncation point ([arXiv:2307.16279](https://arxiv.org/abs/2307.16279v3)).
- Epperly, Lin and Nakatsukasa recommend thresholding S at roughly the noise level, with an AutoThresholding check when that level is unknown ([arXiv:2110.07492](https://arxiv.org/abs/2110.07492)).

All three results concern a perturbed (H, S) pencil, so they apply to a Pauli-bank pencil unchanged.

Four recent post-processing steps should become one estimator ladder.
1. **Project and rotate.** Project the sampled S onto the nearest unit-diagonal positive-semidefinite matrix. Then rotate the pencil (H_θ = H cos θ − S sin θ, S_θ = S cos θ + H sin θ), which preserves eigenvectors, before thresholding, choosing θ to minimise batch variance. De Vriendt et al. report **up to ~100× fewer samples** with this heuristic, and on Fe(III)-NTA it cut the error from 0.89 to 0.38 mHa **[preprint]** ([arXiv:2602.11985](https://arxiv.org/html/2602.11985)).
2. **Constrained ratio.** Replace the hard threshold with Fischer et al.'s program: minimise c†Hc / c†Sc subject to a propagated standard error below a budget. It never inverts S. It explicitly includes the covariance from reusing shots across matrix entries. It recovered ground energies within statistical error on **80 qubits**, where SVD truncation had to drop up to 6 of 22 singular values to stay variational ([Phys. Rev. Lett. 137, 160602; arXiv:2510.25640](https://arxiv.org/html/2510.25640v1)).
3. **Posteriors.** Report posteriors instead of point estimates. Hicks and Lee's Bayesian trimmed sampling weights (H, S) draws by likelihoods that encode physical constraints, and it "performs far better than standard regularization methods" ([arXiv:2209.02083](https://arxiv.org/abs/2209.02083v1)).
4. **Partitioning.** Partitioned QSE splits one ill-conditioned problem into several better-conditioned ones built from the same data ([Quantum 9, 1726](https://quantum-journal.org/papers/q-2025-05-05-1726/)).

All four reuse data the bank already holds. The cleanest test is to re-run the seeded 180-cell PRD campaign and report median error, 90th-percentile error and violation rate for each step. One caution applies. Published thresholding rules assume independent noise on each matrix entry, but a shared-shot bank correlates entries strongly. Only the Fischer-style estimator carries covariance explicitly, so it alone transfers without recalibration.

The uncertainty layer has a gap that the literature has not filled. Joint record-level bootstraps that preserve within-setting covariance are standard, but **no published study checks their coverage for generalized-eigenvalue outputs** ([arXiv:2607.24937](https://arxiv.org/pdf/2607.24937)). clifford_qc is well placed to supply one. It would repeat seeded finite-shot simulations, check the nominal 68% and 95% coverage of bootstrap and delta-method intervals for Ritz values, and stratify the results by the smallest retained S eigenvalue. Sequential allocation needs a stopping rule that stays valid. Empirical-Bernstein stopping already tightens Hoeffding-style readout guarantees by up to an order of magnitude ([JCTC 21, 7352; arXiv:2502.01730](https://arxiv.org/abs/2502.01730)). Closed-form empirical-Bernstein confidence sequences remain valid under optional stopping ([arXiv:2512.21300](https://arxiv.org/abs/2512.21300v1)). Paper A proposed anytime-valid sequences but states, "We have not built or validated any of these" ([Paper A manuscript](../paper/manuscript.tex)).

Adaptive selection also needs a score that is aware of noise. ADAPT-GCIM found that **finite-shot errors in S matter more than errors in H** ([arXiv:2312.07691](https://arxiv.org/html/2312.07691v3)). Shadow subspace expansion needed about 10⁸ snapshots before it reliably beat direct estimation near the ground state ([Phys. Rev. A 111, 022423; arXiv:2406.11533](https://arxiv.org/html/2406.11533)). For near-dependent candidates, therefore, 2×2 scores are dominated by S noise. The natural fix is to require ΔE to exceed its bootstrap spread, or to score ΔE − kσ(ΔE); no source tests this yet.

That same Boyd–Koczor–Cai paper ranks Pauli candidates by the energy lowering of a 2×2 {I, G_i} expansion, which is functionally A-CASE's overlap-aware candidate score. It scaled this to 3,000–10,000 basis operators in simulation. Separately, Fischer et al. ran shadow QSE on up to 80 qubits. Neither work appears in the A-CASE or DA-CASE bibliographies ([A-CASE references.bib](../paper_acase/references.bib); [DA-CASE references.bib](../paper_dacase/references.bib)). Both manuscripts should cite and benchmark against them, and should present Pauli-sum generators, symmetry-leakage rejection, the PRD residual engine and the second-moment bank as A-CASE's distinctive elements.

Finally, the manuscript's own suggestion of block or state-averaged growth has published support. QDavidson reached chemical accuracy for benzene S₀, T₁ and T₂ in 11 iterations only after seeding with several initial determinants. It accepts a new vector only when the component orthogonal to the current span exceeds a tolerance ([arXiv:2204.10741](https://arxiv.org/abs/2204.10741)). Seeding A-CASE with generators that create the leading excited determinants would keep every matrix element a single-reference Pauli expectation.

## Symmetry and selection, not encodings, shrink the bank

No fermion-to-qubit encoding can reduce the quantity that dominates A-CASE cost. Any encoding that maps the 2n Majoranas to pairwise-anticommuting, algebraically independent Pauli strings induces an isomorphism of Cl(2n, C), and a Clifford unitary implements it. **The number of distinct words W in A_i†A_j and A_i†HA_j is therefore identical in every such encoding.** Three results establish this:
- Chiew, Harrison and Strelchuk showed that every product-preserving ternary-tree transformation is equivalent to a linear encoding ([arXiv:2412.07578](https://arxiv.org/abs/2412.07578)).
- Fermihedral imposes algebraic independence explicitly in its SAT formulation ([arXiv:2403.17794](https://arxiv.org/html/2403.17794v2)).
- PLAN.md §6.2 already derives W_JW = W_BK ([PLAN.md](../PLAN.md)).

Together they extend the repository's invariance result from the linear family to HATT, Bonsai, treespilation and Clifford-optimised encodings.

Those encodings still help circuits. HATT cuts CNOT counts by **21–26%** against JW, BK and balanced ternary trees ([arXiv:2409.02010](https://arxiv.org/html/2409.02010v2)), and treespilation cuts them by up to 74% ([npj QI 2025; arXiv:2403.03992](https://arxiv.org/abs/2403.03992)). An optional ternary-tree or HATT Clifford frame is worth adding to reduce rotor and readout depth and to improve QWC compatibility; ferrmion ships both ([ferrmion docs](https://ferrmion.readthedocs.io/en/main/encode.html)). It should never back a W claim.

W changes only when the operator set or the qubit count changes. clifford_qc already tapers Z₂ symmetries and the spin-parity qubits through one rotate-and-fix primitive ([restriction.py](../clifford_qc/subspace/restriction.py)). That can lower W by at most a factor of 2^k for k removed qubits. What is missing is point-group and spin adaptation: the package's exchange pools "do not impose total-spin or point-group symmetry" ([exchange.py](../clifford_qc/algorithms/exchange.py)).

Picozzi and Tennyson's symmetry-adapted encodings show the size of that gap. Every excitation that breaks a Boolean point-group symmetry projects to zero, and **UCCSD parameter counts fell from 92 to 34 for LiH, 140 to 48 for H₂O and 204 to 38 for BeH₂**. The authors note that the same gain is available in Jordan–Wigner "by only including excitations that preserve point-group symmetries" ([Quantum Sci. Technol. 8, 035026](https://doi.org/10.1088/2058-9565/acd86c)). Element pairs scale quadratically with pool size, so a pool cut to a fraction f cuts A_i†HA_j pairs to about f². The effect on W will be smaller, because W counts the union of words across pairs, and it must be measured.

Spin compatibility matters for correctness as well as cost. On an FeS quintet, a 35-operator spin-tailored pool beat a 261-operator generalised pool, while the closed-shell SD pool did not converge **[preprint]** ([arXiv:2601.10813](https://arxiv.org/html/2601.10813v1)).

The selection cache, not the retained bank, is the cost to attack. On H₄, word-resolution A-CASE evaluates a **14,401-word selection cache in 1,672 QWC settings** ([DA-CASE manuscript](../paper_dacase/manuscript.tex)). PRD word resolution compressed the retained bank by a median 79.8% but the full cache by only 36.5% ([PLAN.md](../PLAN.md)). Four levers follow:
- **Word-cost-aware scoring.** Score each candidate by energy lowering per new Pauli word it adds. Candidates with similar ΔE can differ greatly in how many words A_i†HA_j introduces. This is an inference that no source tests.
- **Aggressive truncation of generator Pauli sums.** In variational subspace expansion, keeping the top 10% of operators by coefficient still reached chemical accuracy **[preprint]** ([arXiv:2608.16362](https://arxiv.org/html/2608.16362v1)).
- **Symmetry shifts.** Lee et al. subtract number-operator terms that annihilate the reference. This cuts the ℓ¹ norm by 74–96% from H₂ to H₂O and, combined with coefficient splitting, reduces Krylov matrix-element sampling cost by 20–500× ([Digital Discovery 4, 954; arXiv:2409.02504](https://arxiv.org/abs/2409.02504)). The same identity holds exactly for A-CASE. If K annihilates the N-electron sector and every A_j conserves N, then ⟨ψ|A_i†(H−K)A_j|ψ⟩ = ⟨ψ|A_i†HA_j|ψ⟩. A BLISS-type K ([arXiv:2304.13772](https://arxiv.org/abs/2304.13772)) can therefore be chosen to minimise bank variance without bias. This is untested in a subspace setting.
- **Cache reuse.** Freezing and reusing large selection caches across related Hamiltonians is already a stated DA-CASE next step and belongs in the same campaign.

The geometric-algebra framing should stop being presented as a cost lever. The package should exploit specific structural facts instead. Outside work on GA itself, such as Quantum Register Algebra and a GA Jordan–Wigner transformation, reports no performance data ([arXiv:2208.02608](https://arxiv.org/abs/2208.02608); [Entropy 26, 410](https://www.mdpi.com/1099-4300/26/5/410)). The repository's own G1 study found that its structural filters reproduce existing post-encoding filters, and it concluded that nothing "has been shown to reduce any cost" ([PLAN.md](../PLAN.md)).

What has delivered elsewhere is narrower:
- Majorana Propagation truncates by monomial length, which is the Clifford grade, and built adaptive ansätze for 28–52-mode active spaces **[preprint]** ([arXiv:2503.18939](https://arxiv.org/abs/2503.18939)).
- Frustration-graph tests detect free-fermion-solvable Hamiltonians even when no Jordan–Wigner map applies ([CMP 2021; arXiv:2012.07857](https://arxiv.org/abs/2012.07857)).
- Gaussian states admit Wick/Pfaffian evaluation.

The multivector representation already makes the corresponding features computable: grade filtration of the bank, a free-fermion check before spending shots, and a fast path for Gaussian references.

## Measurement planning should chase variance, and it already owns a better compiler

The dyadic block-commuting hierarchy is published prior art. DalFavero et al. define **k-commutativity**: two strings k-commute when their restrictions to every contiguous block of k qubits commute. This interpolates from QWC at k=1 to full commutation at k=n. The paper generalises to variable block sizes and releases Apache-2.0 code ([arXiv:2312.11840](https://arxiv.org/html/2312.11840v3); [kcommute](https://github.com/rmlarose/kcommute)), and the QED-C benchmark framework already uses k-commuting groups ([arXiv:2504.09813](https://arxiv.org/abs/2504.09813)). The DA-CASE bibliography does not cite it ([references.bib](../paper_dacase/references.bib)). Three steps would turn a priority risk into a contribution:
- cite the paper;
- cross-check setting counts against kcommute on the 7,371-word bank;
- treat the assignment of qubits to blocks as an optimisation variable, which neither work optimises.

DalFavero et al. also find that the grouping benefit R̂ is not monotone in k, so the coarsest rung is not automatically best.

The objective needs to change from settings to variance. Since Crawford et al., the accepted figure of merit has been the shot ratio R̂ under optimal allocation. On it, coefficient-ordered sorted insertion beats largest-first and DSATUR colourings even when sorted insertion produces more groups ([Quantum 5, 385](https://quantum-journal.org/papers/q-2021-01-20-385/)). For H₂O at 14 qubits, largest-first grouping gave a fully commuting variance of **43.4 against 7.59** for sorted insertion ([npj QI 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)). The headline reduction from 913 QWC to 64 fully commuting settings for the 7,371-word H₄ bank is a count ([clifford_hierarchy.tex](../paper_dacase/tables/clifford_hierarchy.tex)). Every rung should also report R̂ and the projected-matrix variance under the package's own estimator.

The pooled estimator, which reads each word from every compatible setting, is what Rowland et al. call post-hoc repacking. Their ad-hoc repacking inserts terms by c²/μ before the readout circuits are synthesised. It never increases optimal variance and reached **about 2.35× below sorted insertion** on a 575,000-term, 44-qubit Hamiltonian **[preprint]** ([arXiv:2604.07156](https://arxiv.org/html/2604.07156v1)). Yen et al.'s iterative coefficient splitting cut exact-state H₂O variance from 7.59 to 1.50 ([npj QI 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)).

A bank has no Hamiltonian of its own, which is why weighting by the target quantity is the publishable opening. To first order a generalized eigenvalue moves as δλ ≈ c†(δH − λδS)c / c†Sc. Every S_ij and H_ij is linear in the bank, so each word w gets an effective weight g_w, and Var(λ̂) ≈ gᵀΣg, where Σ is the grouped covariance the package already estimates. Oumarou et al. allocate Krylov shots in proportion to |∂E₀/∂moment|, computed by autodiff through the regularised eigenproblem. They conclude that the budget is "largely dominated by the need to reduce the noise on the eigenvalues of the overlap matrix" **[preprint]** ([arXiv:2603.15552](https://arxiv.org/html/2603.15552v1)). With g_w in hand, sorted insertion, coefficient splitting, ShadowGrouping, derandomisation and derandomised shallow shadows can all consume the bank as an ordinary weighted Pauli sum.

**No published work jointly optimises grouping and shot allocation for a nonlinear Ritz-value target, or validates delta-method variance against Monte Carlo at realistic conditioning.** This is therefore both an engineering improvement and a paper. It also gives a hypothesis to test. The package's 68.9% covariance-allocation variance cut moved the median error only from 4.49 to 4.44 mHa on its single four-qubit test bank ([DA-CASE manuscript](../paper_dacase/manuscript.tex)). A likely reason is that variance was saved on words that barely move the Ritz value. Published gains from allocation alone are about 23–50% in Yen et al.'s exact-covariance tables and 37.6% in QED-C ([arXiv:2504.09813](https://arxiv.org/abs/2504.09813)), so the raw allocation is already strong. Further gains will come from aiming it at the target.

On circuits, the package already owns a better compiler than its default. Tableau elimination spends **1,886 logical CX** on the 64 fully commuting H₄ settings, about 29.5 per setting, with a mean two-qubit depth of 24.16 ([GRAPH_READOUTS.md](../docs/restart/GRAPH_READOUTS.md)). The Crawford / Murairi–Cervia bounds are n·r − r(r+1)/2 = 28 unoptimised and about 16 optimised for r=8 ([arXiv:2306.00170](https://arxiv.org/html/2306.00170v3)). The repository's opt-in graph-state readout compiles the same settings with **460 CZ, 7.2 per setting, at mean depth 4.17 and maximum 7**, a 75.6% gate cut. It uses fewer two-qubit gates on five of six tested rungs and ties on the sixth, yet elimination remains the default ([GRAPH_READOUTS.md](../docs/restart/GRAPH_READOUTS.md); [restart README](../docs/restart/README.md)). Switching the default is a small code change. Its effect on frozen evidence should be handled by versioning the records, not by keeping the slower compiler.

Fully commuting readout also needs a noise gate. In density-matrix simulation of IBM and IonQ device models, GALIC found that **fully commuting groups exceeded 1 kcal/mol bias in 20 of 20 cases**. On IBM Kyoto hardware, fully commuting bias plateaued near 2 kcal/mol ([arXiv:2409.00576](https://arxiv.org/html/2409.00576v2)). The package's own superconducting-like device card already flags BeH₂ fully commuting settings at mean fidelity 0.452, with 12 of 14 inadmissible ([numbers.tex](../paper_architecture/tables/numbers.tex)). A GALIC-style acceptance test per group would turn that diagnostic into a planner rule: accept a group only if its logical CX count after routing stays below log(1−ε)/log(1−p), where p is the two-qubit error rate and ε the tolerated relative bias.

Randomised classical shadows are not the next step for a bank that is known in advance. With exact-state variances on NH₃, overlapped grouping measurement gave 305, derandomised shadows 251, sorted insertion 97.0 and coefficient splitting **53.8** ([npj QI 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11041696/)). A 2026 study found that conventional grouped measurement beat the shadow techniques it tried **[preprint]** ([arXiv:2608.30612](https://arxiv.org/abs/2608.30612)). Two exceptions deserve a baseline run:
- derandomised shallow shadows, which search one- or two-layer entangling readouts for a given word list and come with MIT-licensed code ([arXiv:2412.18973](https://arxiv.org/html/2412.18973v1));
- dual-frame IC-POVM estimators, which cut variance purely in post-processing and are implemented in the Apache-2.0 povm-toolbox ([arXiv:2401.18071](https://arxiv.org/html/2401.18071v2)).

Readout-error-aware allocation is an open niche. Linear-inversion mitigation inflates per-setting variance by a computable factor ([arXiv:2608.18548](https://arxiv.org/pdf/2608.18548)), but no published allocator uses it.

## A NumPy batch kernel buys 3–41× without a new dependency

The algebra kernel is correct but makes one Python call per word pair. MV stores a `dict[int, complex]` keyed by a Python integer with 2 bits per qubit. `MV.__mul__` is a double loop that calls a memoised bit-twiddling product for each pair ([multivector.py](../clifford_qc/multivector.py); [pauli_kernel.py](../clifford_qc/pauli_kernel.py)). The word-product formula matches Stim, PauliArray and PauLIB. The container and the per-pair call are the problem.

The recorded costs follow from that design:
- **Memory.** Each stored coefficient costs 85–96 bytes, against the 24 bytes the package's own packed model assumes ([bank_storage_ledger.json](../benchmarks/reference_results/bank_storage_ledger.json)).
- **Pair building.** On the 14-qubit stretched-H₂O M=31 cell, **444.9 s of an 850.4 s solve went to building the first pairs** ([bank_policy_matrix.json](../benchmarks/reference_results/bank_policy_matrix.json)).
- **Packing.** Packing and streaming cut peak memory from 9.87 to 2.63 GiB but lengthened wall time from 14.4 to 26.2 minutes ([PLAN.md](../PLAN.md)). Rows are still multiplied through MV before they are packed ([packed.py](../clifford_qc/subspace/packed.py)).
- **The memo.** By the code's own comment, the process-wide memo costs about 250 MiB at 12 qubits and pays off only above a 70% hit rate ([pauli_kernel.py](../clifford_qc/pauli_kernel.py)).
- **No throughput record.** No committed record measures kernel throughput at all.

The fast libraries of 2025–2026 converge on one design:
- x and z bit-planes packed into 64-bit words, with a small phase exponent;
- sums stored as structure-of-arrays, meaning a code array plus a coefficient array;
- merging by sorting or bucketing arrays, not by inserting into a hash map one product at a time.

Qiskit's `SparsePauliOp` and Rust `SparseObservable`, sparse Pauli dynamics, cuPauliProp, paulistrings-rs and PauLIB all follow it ([PauLIB, arXiv:2605.25974](https://arxiv.org/html/2605.25974); [Qiskit source](https://github.com/Qiskit/qiskit/blob/main/qiskit/quantum_info/operators/symplectic/sparse_pauli_op.py)).

PauLIB's outer product of two 200-term sums on 500 qubits ran **49× faster than PauliEngine and 131× faster than Qiskit**. Its authors attribute the PauliEngine gap to "one string per call", not to a slower word kernel. Hand-written intrinsics added nothing over compiler auto-vectorisation, and 72 threads gave only 3.2× ([arXiv:2605.25974](https://arxiv.org/html/2605.25974)). At small sizes, low overhead wins: pure-NumPy PauliArray beats C++ PauliEngine on small inputs ([arXiv:2601.02233](https://arxiv.org/html/2601.02233)). clifford_qc's operands, hundreds to thousands of terms at 8–14 qubits, sit squarely in that regime, so batching matters more than compiling. All cross-library figures here are self-reported, each on its authors' own hardware.

A NumPy rewrite of the existing phase formula confirms the gap locally. The rewrite broadcasts XOR over code arrays, computes phases with `np.bitwise_count`, and merges with a dense `np.bincount` at n ≤ 12 or with `np.unique`. On random 8- and 12-qubit sums of 300–3,000 terms it reproduced `MV.__mul__` term for term. Across two runs it was **5–41× faster than the current product with a cold memo and 3–18× faster with a warm one**. At 9 million pairs the warm memo was no faster than the cold one, because the memo thrashed. These figures come from a benchmark run in this session, not from a repository record ([benchmark script](../research_notes/Clifford%20QC%20framework%20improvements/bench_kernel.py)).

The dictionary loop ran at roughly 0.3–0.6 million pairs per second and the batched path at roughly 10–30 million. That is still a fifth to a tenth of PauLIB's multithreaded 123–145 million on different hardware. The test words were random. Real A_i†HA_j products are chemistry-structured, should produce more duplicate outputs, and should therefore favour batched merging further.

The kernel plan follows directly.
1. **SoA storage.** Store each MV internally as sorted structure-of-arrays: uint16, uint32 or uint64 codes chosen from n, plus complex128 coefficients. Keep the Python-integer path as the reference oracle the package already maintains.
2. **Global collate.** Build element universes with one merge over all pair products, instead of hundreds of per-pair dictionaries.
3. **Commutator pre-filter.** Skip pairs whose symplectic parity shows they commute, as PauliArray and PauLIB do ([arXiv:2405.19287](https://arxiv.org/html/2405.19287)). About half of random pairs commute.
4. **Sorted intersections.** Turn expectation pairings into sorted-array intersections.
5. **Retire the memo.**

`np.bitwise_count` was available on NumPy 2.5.3 in this session. The package already carries a fallback popcount for NumPy builds without it ([selected_ci.py](../clifford_qc/subspace/selected_ci.py)). SPEC 0's Q4 2026 schedule drops NumPy 2.2 regardless ([SPEC 0](https://scientific-python.org/specs/spec-0000/)), so a NumPy 2 floor would be conventional.

The packed store also uses int64 codes without an explicit guard, which implies a 31-qubit ceiling. Multiword codes, as in sparse Pauli dynamics, PauLIB and cuPauliProp, remove it.

Only after this should an optional compiled backend be considered. PauliEngine is MIT-licensed and ships wheels for the main platforms ([GitHub](https://github.com/tequilahub/pauliengine)). paulistrings-rs is MIT/Apache-2.0 but only at version 0.2.0 ([docs.rs](https://docs.rs/paulistrings)). PauliArray (CC-BY-SA-4.0) and sparse Pauli dynamics (GPL-3.0) must not be vendored into Apache-2.0 code. GPU paths pay off only near 10⁷–10⁸ output terms; PauLIB's H100 peaks at 27.4 G terms/s at 10⁸. They are not justified at current sizes.

A committed throughput benchmark should land together with the new kernel. PauliEngine's methodology is a good model: it states the hardware, the repetitions and the sweeps, and it shows the cases where a competitor wins.

For the classical arms that limit scale, peers bypass the Pauli representation entirely. ffsim stores particle- and spin-sector vectors of length C(N,N_α)·C(N,N_β) in PySCF's FCI ordering, behind Rust kernels ([ffsim docs](https://qiskit.github.io/ffsim/explanations/state-vectors-and-gates.html)). qiskit-addon-sqd wires in external selected-CI solvers through extras ([PyPI](https://pypi.org/pypi/qiskit-addon-sqd/json)). clifford_qc's sector-vector work stopped at a 16-qubit, 4,900-determinant Hubbard lattice, and its own blueprint projects 13.0 GiB just for V and HV at 12 sites ([blueprint](../docs/restart/blueprint_2026-10-24.md)). The low-effort route past that wall is to delegate sector actions to ffsim or pyscf.fci behind an extra, and keep the MV path for validation at 12 qubits or fewer.

## Quantum value begins where classical evaluation of the bank ends

One structural fact should reshape every claim about quantum cost. Under any linear encoding, a Hartree–Fock determinant is a computational-basis state. Every Pauli word therefore has expectation 0 or ±1 on it, and the whole bank can be evaluated exactly in O(W·n) bit operations. For a Gaussian reference, each Majorana monomial expectation is a Pfaffian of a small covariance submatrix, and free-fermion states are classically simulable ([Terhal & DiVincenzo](https://arxiv.org/abs/quant-ph/0108010)). **A-CASE on a determinant reference is therefore a selected-CI problem that needs no quantum device.**

The repository's own evidence agrees. Determinant reference mode is now the CLI default, and a compatibility test evaluates a 4,096-qubit determinant reference analytically ([pipeline.py](../clifford_qc/pipeline.py); [COMPATIBILITY.md](../docs/restart/COMPATIBILITY.md)). CCSD beat A-CASE on 7 of 7 geometries in milliseconds against minutes ([molecular/report.md](../molecular/report.md)).

The 2025–2026 critiques of sample-based diagonalisation point the same way:
- In same-active-space comparisons, quantum-sampled determinant sets were less compact than classical selected CI ([JCTC 21, 6811; arXiv:2501.07231](https://arxiv.org/abs/2501.07231)).
- For delocalised lattice ground states, even optimal selection needs exponentially many configurations **[preprint]** ([arXiv:2605.02494](https://arxiv.org/abs/2605.02494)).

Both match the repository's negative results for time-evolved QSCI and sample-fed dressing.

The device earns its place only for references that classical methods cannot evaluate: deep ADAPT/UCC states, time-evolved states, or MPS-prepared states such as the 540-CZ, 80-qubit reference of Fischer et al. ([arXiv:2510.25640](https://arxiv.org/html/2510.25640v1)). For such references the measured objects are k-RDMs, with k ≤ 4 for single-excitation generators and k ≤ 6 for doubles. Fermionic measurement makes the number of settings independent of both W and the encoding. O(N²) circuits suffice for the full 2-RDM, against O(N⁴) for Pauli schemes ([PRX 10, 031064](https://arxiv.org/abs/1908.05628)), and fermionic shadows estimate all k-RDM elements with a near-optimal number of state preparations ([PRL 2021; arXiv:2010.16094](https://arxiv.org/abs/2010.16094)). At 8 qubits this barely matters. At 20–30 qubits it should become decisive, though that remains an inference.

No route from a clifford_qc plan to any device exists today. `FiniteShotBackend` turns exact expectations into seeded binomial counts and is described as "swappable later" ([finite_shot.py](../clifford_qc/backends/finite_shot.py)). Noise exists only as single-qubit Kraus channels on density multivectors ([channels.py](../clifford_qc/channels.py)). Device cards are illustrative accounting scenarios ([README.md](../README.md)).

The cheapest credible path is a bridge to Qiskit primitives. It would emit one EstimatorV2 PUB per commuting family, because commuting observables placed in different PUBs are measured separately, or emit SamplerV2 circuits. It would then map `BitArray` shot data back into the existing covariance and bootstrap machinery ([IBM primitive I/O](https://quantum.cloud.ibm.com/docs/en/guides/primitive-input-output)). Three routes then cover testing and hardware:
- **Local rehearsal.** Qiskit Runtime local mode supplies fake backends that apply device noise automatically ([IBM local testing](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode)).
- **Free hardware.** IBM's Open Plan gives **10 minutes per 28-day window** on the Heron r2 device ibm_kingston ([IBM blog](https://www.ibm.com/quantum/blog/open-plan-updates)).
- **Paid fallback.** On Amazon Braket, a 20-setting, 80,000-shot plan costs about $40 on Rigetti Cepheus, $122 on IQM Garnet and about $6,400 on IonQ Forte at published prices ([Braket pricing](https://aws.amazon.com/braket/pricing/)).

Error mitigation should be passed through to the hardware stack, not reimplemented: TREX or M3 readout correction plus zero-noise extrapolation at roughly 3× overhead ([IBM mitigation guide](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques)). The package can add mitigation that its own structure already supports: sector post-selection on sampled bitstrings, and Clifford-data-regression training sets built with its Stim machinery ([arXiv:2107.13470](https://arxiv.org/pdf/2107.13470)).

The bar for a first hardware result is modest. The symmer group's N₂ run on a 5-qubit contextual subspace reported errors of 47 meV to 1.2 eV. The authors called this explicitly "not algorithmically accurate" and kept their data in a separate repository ([npj QI 11, 25](https://pmc.ncbi.nlm.nih.gov/articles/PMC11821530/)).

Comparability with the field requires named instances and strong baselines. Peers now report on 50–100-qubit hardware: SQD at 77 qubits ([Sci. Adv. 11, eadu9991](https://arxiv.org/abs/2405.05068)), Krylov diagonalisation on 56 sites ([Nat. Commun. 16, 5014](https://arxiv.org/html/2407.14431v3)) and shadow QSE on 80 qubits. By contrast, **all of clifford_qc's sampled and priced evidence is at eight qubits** ([DA-CASE manuscript](../paper_dacase/manuscript.tex)).

The cheapest step is to re-run the headline comparisons on openly released Hamiltonians:
- **The "chemically decisive" set.** It covers N₂, an FeS quintet, reduced-space [2Fe-2S] and U₂, and comes with Jordan–Wigner files and ADAPT-GCIM errors against CASCI **[preprint]** ([arXiv:2601.10813](https://arxiv.org/html/2601.10813v1)).
- **HamLib.** Instances should be cited by version, key and encoding and read through an h5py loader ([Quantum 8, 1559](https://quantum-journal.org/papers/q-2024-12-11-1559/)).

Each comparison should report:
- total shots to reach 1.6 and 0.16 mHa, including the selection cache, with median and 90th-percentile seed outcomes;
- classical post-processing time and memory;
- a sample-free classical selected-CI baseline at the same determinant count, with the orbital gauge fixed ([arXiv:2608.05314](https://arxiv.org/abs/2608.05314)).

The comparisons should also follow McGeoch's rules on reporting runtimes and tuning time and avoiding cherry-picking ([arXiv:2411.08860](https://arxiv.org/html/2411.08860v1)).

The internal comparators need hardening too:
- **Davidson.** The Davidson arm has no restarts or root tracking. On the 2×4 Hubbard lattice it **stopped at residual 9.2×10⁻⁹ on an excited singlet whose ground-state overlap was 1.5×10⁻¹²** ([PLAN.md](../PLAN.md)).
- **Selected CI.** The selected-CI control is greedy, not CIPSI or heat-bath CI ([compactness manuscript](../paper_compactness/manuscript.tex)).
- **PRIMME.** The blueprint's PRIMME adapter was never built ([blueprint](../docs/restart/blueprint_2026-10-24.md)).

## Distribution and docs are the cheapest credibility gains

Outside researchers cannot currently install, navigate or cite a stable clifford_qc. It is distributed from source only, with a frozen 0.3.0 version string and a single git tag ([CHANGELOG.md](../CHANGELOG.md)). It has no rendered docs and no notebooks, and it keeps a 6,221-line PLAN.md and about 149 MB of benchmark records inside the library repository ([numbers.tex](../paper_architecture/tables/numbers.tex); [README.md](../README.md)). By import count it exports 122 top-level names and 180 from `clifford_qc.subspace` alone ([clifford_qc/\_\_init\_\_.py](../clifford_qc/__init__.py)).

Keeping the research archive inside the product has a hidden engineering cost. Two model modules are exempt from mypy because frozen evidence hashes them byte for byte ([pyproject.toml](../pyproject.toml)). Evidence binding is now blocking ordinary refactoring.

Every peer toolkit with measurable adoption is on PyPI, and adoption tracks how well a package plugs into a larger ecosystem rather than polish. **ffsim draws 38,060 downloads a month at version 0.0.86. Standalone Pauli libraries such as symmer and PauliEngine draw 306 and 158** ([ffsim](https://pypistats.org/packages/ffsim); [symmer](https://pypistats.org/packages/symmer); [pauliengine](https://pypistats.org/packages/pauliengine)). "Poorly documented" is the most-cited reason developers skip a tool they want, at 50.8% ([Unitary Foundation survey](https://unitaryfoundation.github.io/survey-2025)).

The fixes are one-time costs:
- **Release.** Publish to PyPI through Trusted Publishing, which needs no long-lived token ([PyPI docs](https://docs.pypi.org/trusted-publishers/)). Use version numbers derived from git tags and move developer tools into dependency groups ([Scientific Python guide](https://learn.scientific-python.org/development/guides/packaging-simple/)).
- **Docs.** Build docs with Sphinx, MyST and sphinx-gallery from the existing example scripts. Organise them into tutorials, how-to guides, explanations and an API reference, as ffsim does ([ffsim docs](https://qiskit.github.io/ffsim/)). MkDocs-Material's critical-fix window closed on 5 November 2026 ([issue #8523](https://github.com/squidfunk/mkdocs-material/issues/8523)), which argues against building on it.
- **API.** Narrow the public surface to one supported path per task, as the restart blueprint already proposed ([blueprint](../docs/restart/blueprint_2026-10-24.md)). Put a newcomer layer above the evidence apparatus: build model, plan measurements, estimate with uncertainty, export.
- **Evidence repository.** Move records and paper trees to a companion evidence repository archived on Zenodo with its own DOI. Add `codemeta.json` alongside `CITATION.cff` ([Zenodo GitHub docs](https://help.zenodo.org/docs/github/)).
- **Python floor.** Raise it to 3.11. SPEC 0 already recommends this ([SPEC 0](https://scientific-python.org/specs/spec-0000/)), and PennyLane, Cirq, Qiskit Aer, Braket and ffsim all require it ([pennylane PyPI](https://pypi.org/pypi/pennylane/json); [cirq-core PyPI](https://pypi.org/pypi/cirq-core/json)).
- **Community files.** Add `CONTRIBUTING.md`, a code of conduct and issue forms. pyOpenSci requires them and JOSS reviewers expect them ([pyOpenSci guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html); [JOSS criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)).

Venue choice has timing constraints. **JOSS now requires more than six months of public history** with releases and public issues or pull requests. It also requires a research-impact statement and an explicit AI-usage disclosure ([JOSS blog, 5 January 2026](https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future)). In the shallow local clone, 61 of 217 commits are attributed to an AI agent (`git log --format=%an`), so the disclosure should cite the checker scripts, frozen records and CI gates as the evidence of verification. SciPost Physics Codebases makes benchmark tests and one detailed example application mandatory, and it issues separate DOIs for the user guide and the code release ([SciPost Codebases](https://scipost.org/SciPostPhysCodeb/about)). It fits the architecture paper better.

Before any submission, fix the document drift the self-audit found. The DA-CASE preprint calls the repository private while the README calls it public ([DA-CASE manuscript](../paper_dacase/manuscript.tex); [README.md](../README.md)). PLAN.md still quotes superseded Phase 4R figures ([PLAN.md](../PLAN.md)).

Preregistration is almost absent in quantum computing, and the NQI algorithms-workshop report proposes registered reports for quantum-advantage claims ([arXiv:2508.13973](https://arxiv.org/pdf/2508.13973)). clifford_qc's preregistered campaigns already match that recommendation. A short "evidence policy" docs page plus timestamped Zenodo deposits would let readers verify the "declared before run" claim without reading a 340 KB ledger.

## Conclusion

This research changes how clifford_qc's contribution should be understood. Its compactness in exact arithmetic is real but classically reproducible, and on determinant references it is classically trivial. The defensible contribution is therefore not a better basis but trustworthy finite-shot estimation of projected quantities: pencil estimators that carry covariance, measurement design aimed at the Ritz value, and interval coverage that has been validated. That is exactly where the 2024–2026 literature is thin. There is no published joint grouping-and-allocation method for Ritz values, no coverage study for eigenvalue intervals built from grouped measurements, and no thresholding theory for shared-shot correlated noise. The project's evidence culture is unusual in a field that barely preregisters, and once it is made legible it becomes a competitive asset rather than overhead.

For sequencing, scale should come last. A vectorised kernel and ffsim sector backends will push the classical arms past 14 qubits, but an exact 20-qubit A-CASE run would prove nothing new. The decisive experiment is a seeded replay of the 180-cell PRD campaign showing that the improved estimator turns 65–95% violation rates into calibrated intervals. One honest hardware run on a correlated reference through Qiskit primitives should follow. If the pencil cannot be made reliable at eight qubits, neither kernel speed nor measurement compression will rescue the finite-shot claim. If it can, the same machinery serves every subspace method that estimates (H, S) from expectations, an audience far larger than A-CASE.
