# Fermion-to-qubit encodings, symmetry reduction and Clifford/GA structure as cost levers (state of the art, Nov 2026)

Scope note for the report writer. "W" means the number of distinct Pauli words in the products A_i†A_j and A_i†HA_j (7,371 for 8-qubit H4 in clifford_qc). Each finding says which quantity it improves: Pauli weight, term count / W, measurement groups or settings, shots/variance, CNOT count or depth, qubit count, 1-norm. Preprints are marked **[preprint]**. Results from before 2020 are marked **[background]**. Repository facts come from local files and are labelled with their paths.

## Q1. Modern encodings: what do ternary-tree, Bonsai, HATT, Fermihedral, treespilation, compact/local and Clifford-optimised encodings report, and which are implemented in open-source tools?

### Takeaway
Every modern encoding reports savings in **Pauli weight, CNOT count, depth, SWAP overhead or entanglement**, typically 5–40% on average and up to 74% CNOT in favourable cases. **None reports a reduction in the number of distinct Pauli terms**, and the compact/local encodings for lattices **add** qubits. Open-source support is good: ferrmion (JW/BK/parity/JKMN ternary tree, HATT and others), the Fermihedral code, Qiskit Nature's TernaryTreeMapper, OpenFermion binary codes and BK-tree, and yaf2q.

### Cited Findings
- **Ternary tree (Jiang–Kalev–Mruczkiewicz–Neven, Quantum 4, 276, 2020).** Each Majorana maps to a Pauli string of weight ⌈log₃(2n+1)⌉. This is optimal: no mapping can have average Majorana weight below log₃(2n). The same paper shows that with the TT mapping, all k-fermion RDMs can be learned to precision ε by repeating a single circuit about (2n+1)^k ε⁻² times. So the measurement saving here comes from low weight under randomized single-qubit measurement. — [Jiang et al. arXiv:1910.10746](https://arxiv.org/abs/1910.10746)
- **Bonsai (Miller, Zimborás, Knecht, Maniscalco, García-Pérez, PRX Quantum 4, 030314, 2023).** Grows ternary-tree mappings from the device connectivity graph. On IBM heavy-hex it gives O(√N) Pauli weight (a quadratic improvement over JW's O(N)) and **no SWAPs for single-excitation operations**. It also gives a recipe that keeps Fock states as computational-basis states. Improves weight and SWAP count. — [Bonsai arXiv:2212.09731](https://arxiv.org/abs/2212.09731)
- **Treespilation (Miller, Glos, Zimborás; npj Quantum Information 2025).** Extends Bonsai to optimise the mapping for a specific state-preparation circuit (ADAPT-VQE). Reports **up to 74% CNOT reduction** on full connectivity, with similar reductions on IBM Eagle and Google Sycamore connectivity. It also improves CNOT and parameter efficiency of QEB- and qubit-ADAPT. Improves CNOT count, not term count. — [arXiv:2403.03992](https://arxiv.org/abs/2403.03992); [npj QI version](https://www.nature.com/articles/s41534-025-01170-2)
- **HATT (Liu, Li, Shi et al., arXiv:2409.02010; the arXiv v2 is an HPCA 2025 submission, and its acceptance status could not be confirmed from the PDF).** Builds a Hamiltonian-adaptive ternary tree bottom-up in O(N³), designed so that Pauli multiplications cancel, and preserves the vacuum. Benchmarks: H₂ (4), LiH frozen core (6), LiH (12), H₂O (14), CH₄ (18), O₂ (20), NaF (28), CO₂ (30) qubits in STO-3G, some 6-31G; Fermi–Hubbard 2×2 up to 4×5 (40 qubits); collective neutrino models up to 42 qubits. Average reductions on electronic structure: Pauli weight 14.77% / 13.83% / 11.77%, CNOT 25.84% / 24.35% / 21.37%, depth 19.33% / 18.91% / 20.58% versus JW / BK / balanced TT. Fermi–Hubbard: weight 20.90% / 6.48% / 4.77%, CNOT 22.90% / 12.11% / 16.58% versus the same three. Against SAT-optimal Fermihedral: H₂ total weight 32 for both; LiH-fc HATT 188 versus Fermihedral (approximate) 193; Hubbard 2×2 Fermihedral 56 versus HATT 76. **The paper reports Pauli weight, CNOT and depth only. It does not report term counts.** On an IonQ Forte 1 run HATT gave the smallest variance. — [HATT arXiv HTML v2](https://arxiv.org/html/2409.02010v2)
- **Fermihedral (Liu, Che, Zhou, Shi, Li, ASPLOS '24).** Poses the encoding search as SAT, with constraints for anticommutation, vacuum and **algebraic independence**, and tightens a weight bound until the solver times out. Hamiltonian-independent per-Majorana weight: −11.16% versus BK for 1–8 modes and −17.36% for 9–19 modes. Hamiltonian-dependent total weight versus BK (full SAT, small cases): average −37.26% (electronic structure at 6 modes −29.43%, Hubbard at 8 modes −27.85%, SYK at 5 modes −51.52%). Circuits versus BK: H₂ CNOT 26→21 and depth 39→33; 3×1 Hubbard CNOT 56→31 and depth 76→26; 2×2 Hubbard CNOT 53→31 and depth 73→26. Scalability: full SAT reaches only 8 modes, and relaxed versions reach 12 (electronic structure) to 18 (Hubbard) modes. MIT-licensed code. — [Fermihedral arXiv:2403.17794](https://arxiv.org/html/2403.17794v2); [code](https://github.com/acasta-yhliu/fermihedral)
- **Mode-ordering optimisation (Chiew & Strelchuk, Quantum 7, 1145, 2023).** Average-Pauli-weight minimisation over enumeration patterns is NP-complete in general. On square lattices the Mitchison–Durbin enumeration gives 13.9% shorter average Pauli weight. Two ancilla qubits give 37.9%. Cellular arrangements give an n^{1/4} improvement. — [arXiv:2110.12792](https://arxiv.org/abs/2110.12792)
- **Clifford-circuit-parametrised mappings (Yu, Liu, Sugiura, Van Voorhis, Zeytinoğlu, JCTC 21, 9430, 2025).** Simulated annealing over Clifford circuits minimises average Pauli weight. Reports 15–40% improvement on intermediate-complexity Hamiltonians, >40% on 6×6 nearest-neighbour hopping and >20% on 6×6 Hubbard. Some Hamiltonians admit mappings better than any ternary tree. Improves weight only. — [arXiv:2502.11933](https://arxiv.org/abs/2502.11933)
- **Entanglement-reducing mappings (Parella-Dilmé et al., PRX Quantum 5, 030333, 2024).** Mappings guided by the dominant excitations lower qubit-space entanglement. Hardware-efficient (RY) VQE needs fewer entangling layers for LiH, H₂, (H₂)₂, stretched H₄ and the benzene π system, and DMRG on N₂ improves. Improves ansatz depth and MPS bond dimension, not term count. — [arXiv:2311.07409](https://arxiv.org/abs/2311.07409)
- **Sierpinski-triangle transform (Harrison, Chiew, Necaise, Projansky, Strelchuk, Whitfield, 2024) [preprint].** A Fenwick-like data structure that matches the ternary tree's operator locality while keeping Fock states as computational-basis states. — [arXiv:2409.04348](https://arxiv.org/abs/2409.04348)
- **Compact local encodings for lattices.** Derby–Klassen (PRB 104, 035118, 2021): about 1.5 − 2/L qubits per mode on an L×L lattice, maximum hopping weight 3 and Coulomb weight 2, against weight 4 / 6 for earlier local encodings. Part 2 handles degree-≤4 tilings with <1.67 qubits/mode and weight ≤3, and cubic lattices with about 2.5 qubits/mode and weight ≤4. A "super-compact" variant reaches 1.25 qubits/mode. These encodings **add** qubits in order to obtain O(1) weight on lattices. — [arXiv:2003.06939](https://arxiv.org/abs/2003.06939); [Part 2 arXiv:2101.10735](https://arxiv.org/abs/2101.10735); [Error Correction Zoo](https://errorcorrectionzoo.org/c/super_compact)
- **2D equivalence (Chen & Xu, PRX Quantum 4, 010326, 2023).** Clifford finite-depth generalised local unitaries connect all such 2D mappings, up to re-pairing of Majoranas. The paper uses this to improve qubit-to-fermion ratios. — [PRX Quantum 4, 010326](https://dx.doi.org/10.1103/PRXQuantum.4.010326)
- **Communication-aware co-design (COSMA, IEEE QCE 2026) [preprint].** Joint optimisation of mapping, Pauli scheduling and qubit allocation for modular hardware. Up to 2.5× (median 1.7×) less inter-core communication on molecular benchmarks. — [arXiv:2607.09381](https://arxiv.org/abs/2607.09381)
- **Review (Chien, Chiew, Harrison, Necaise, Wang, Mudassar, McLauchlan, Henderson, Scuseria, Strelchuk, Whitfield, Feb 2026) [preprint; listed as "to appear in Nature Physics Reviews"].** Reviews fermion-to-qubit encodings and argues against the belief that fermions in more than one dimension are inherently harder to simulate. — [arXiv:2602.07151](https://arxiv.org/abs/2602.07151)
- **Software, as verified in 2026 documentation:**
  - **ferrmion** (Python with a Rust core) implements JW, BK, parity and JKMN as ternary trees, arbitrary mode/qubit labelling (for example N² JW variants), **HATT**, a reduced-entanglement tree, Huffman trees (vacuum-preserving only) and numerical Pauli-weight minimisation. — [ferrmion docs](https://ferrmion.readthedocs.io/en/main/encode.html); [HATT notebook](https://ferrmion.readthedocs.io/en/main/notebooks/hatt.html)
  - **Qiskit Nature** (now under qiskit-community) ships a `TernaryTreeMapper`, which maps FermionicOp → MajoranaOp → SparsePauliOp. — [source](https://qiskit-community.github.io/qiskit-nature/_modules/qiskit_nature/second_q/mappers/ternary_tree_mapper.html)
  - **OpenFermion** has `binary_codes` (JW, BK and parity codes plus a generic `binary_code_transform`), `bravyi_kitaev_tree`, `symmetry_conserving_bravyi_kitaev` (two-qubit removal) and `taper_off_qubits`. — [OpenFermion transforms](https://quantumai.google/reference/python/openfermion/transforms); [SCBK](https://quantumai.google/reference/python/openfermion/transforms/symmetry_conserving_bravyi_kitaev)
  - **yaf2q** covers JW, parity, BK and ternary tree. — [yaf2q](https://samn33.github.io/yaf2q/introduction.html)
  - **Fermihedral**: MIT-licensed code on GitHub (see the Fermihedral item above).
  - **HATT**: the paper names no repository of its own, but HATT is available through ferrmion.

### Inferences
- Every result above improves a **qubit-basis-dependent** quantity: weight, qubit-wise-commuting (QWC) compatibility, CNOT, depth, SWAPs or entanglement. For clifford_qc's dominant cost W, none of them is a lever (see Q2). They would lower CNOT/depth for any circuit the framework compiles (rotors exp(−iθP/2), diagonalisers) and would raise QWC-group merging. They would not change full-commutation (GC) grouping.
- Measurement can still benefit from lower weight in two places: randomized single-qubit (Pauli-shadow) measurements, where cost grows like 3^weight, and QWC grouping. The Jiang et al. (2n+1)^k RDM-learning result is an example.
- The cheapest addition with real value to clifford_qc is a ternary-tree / HATT encoding as **one more Clifford frame**. Its benefit would appear only in the weight, QWC and depth columns of the mapping table in PLAN.md §6.2.

### Gaps
- None of the encoding papers reports W-type counts (distinct words in products of operators). Their silence is consistent with invariance (Q2) but is not an explicit statement.
- I could not confirm whether tket/pytket ships ternary-tree mappings (Quantinuum's InQuanto is proprietary), nor whether PennyLane qchem has a ternary-tree mapping. Searches returned nothing.
- I could not determine whether HATT was accepted at HPCA 2025; one search snippet said yes, but the PDF is marked as a submission.

## Q2. Does any encoding change the NUMBER of distinct Pauli words in products of fermionic operators, or only their weight?

### Takeaway
For every encoding that sends the 2n Majoranas to pairwise-anticommuting, algebraically independent Pauli strings, the number of distinct words is **exactly invariant**. That class includes JW, parity, BK, BK-tree, all ternary trees, HATT, Bonsai, treespilation, Fermihedral and the Clifford-optimised encodings. Such encodings differ by a Clifford unitary, which maps Majorana monomials one-to-one onto Pauli words. W changes only under operations that **leave this class**:
- Qubit removal (tapering, symmetry-adapted encodings, two-qubit reduction): W can fall by at most a factor 2^k for k removed qubits.
- Qubit-efficient / compact-subspace encodings (log₂ C(N,K)): Majoranas are no longer single Pauli strings, and term counts can grow.
- Ancilla-based local encodings: representatives are fixed only modulo stabilizers.
- Changing the operator itself, at the fermionic level (factorisation, symmetry shifts).

### Cited Findings
- "A Clifford circuit maps a single Pauli string to a single Pauli string, preserving the total number of terms." This is used as the basis for searching mappings over Clifford circuits; the Clifford ansatz reproduces all optimal ternary-tree mappings for single creation operators. — [Yu et al. arXiv:2502.11933](https://arxiv.org/pdf/2502.11933) (statement as quoted in a search excerpt; the abstract page alone did not contain it)
- **Ternary trees are linear encodings in disguise (Chiew, Harrison, Strelchuk, Dec 2024) [preprint].**
  - Two Pauli-based mappings are *equivalent* (Def. 3.4) when they are related by a Clifford C with Γ'_i = CΓ_iC† built from qubit permutations, local basis changes, braids within a Majorana pair, Majorana sign changes and mode relabelling.
  - Theorem 1: a Clifford that maps computational-basis states to computational-basis states acts as an affine map |f⟩ → |G(f⊕b)⟩.
  - Corollary 4.7: every classical encoding of the Fock basis that represents Majoranas by Pauli strings is equivalent to a linear encoding, up to Majorana signs.
  - Main result: every product-preserving ternary-tree transformation is equivalent to a linear encoding of the Fock basis.
  - [arXiv:2412.07578](https://arxiv.org/abs/2412.07578) ([HTML](https://arxiv.org/html/2412.07578))
- Encodings given by invertible binary matrices on Fock states correspond to Clifford operators. — [Sierpinski transform arXiv:2409.04348](https://arxiv.org/pdf/2409.04348) (search excerpt)
- Fermihedral's SAT encoding imposes an explicit **algebraic-independence** constraint on the 2n Majorana strings, in addition to anticommutation. This is the condition under which the Majorana monomials map injectively to Pauli strings. The paper notes that Pauli strings form an orthonormal basis, so each Hamiltonian has a unique decomposition, and that Majorana multiplication can cancel Pauli factors. That cancellation changes **weight**, not the number of terms. — [Fermihedral arXiv:2403.17794](https://arxiv.org/html/2403.17794v2)
- When observables are measured through fermionic shadows, the number of measurements "is independent of the fermionic encoding". — [arXiv:2406.03204](https://arxiv.org/pdf/2406.03204) [preprint]
- **Caveat for Majorana-level Cliffords.** Clifford transformations acting on Majorana strings can send even-length strings to odd-length ones, so naive "term preservation" needs care when fermion parity is not conserved. — [Clifford transformations for fermionic quantum systems, arXiv:2510.23923](https://arxiv.org/pdf/2510.23923)
- **Qubit-efficient encodings leave the Pauli-string class.** Harrison et al. reduce K fermions in N modes to ⌈log₂ C(N,K)⌉ qubits via a permutation of the computational basis. Circuit depth grows with the encoded Pauli weight, and a permutation that is Clifford **can remove at most one qubit**. Shee et al.'s qubit-efficient encoding uses O(m log₂ N) qubits for m particles (shown for H₂ in 6-31G and LiH in STO-3G). — [Harrison et al. arXiv:2211.04501](https://arxiv.org/abs/2211.04501); [Shee et al. arXiv:2110.04112](https://arxiv.org/abs/2110.04112)
- **Repository (local).** clifford_qc's own PLAN.md §6.2 already derives W_JW = W_BK exactly for the linear family. It classifies spectrum, W, S_A, S_H, nnz and per-word variance as invariant; the full-commutation conflict graph as isomorphic; and weight, QWC groups G(k<n), single-qubit rotation count and CX count as variant. It notes that "deletion is where W can genuinely fall". — `/home/user/clifford_qc/PLAN.md` lines ~4359–4400

### Inferences
- **Proof sketch** (standard algebra, not taken from a single source). Suppose an encoding maps γ_1..γ_2n to Hermitian Pauli strings that pairwise anticommute and are algebraically independent (no product of a nonempty proper subset is ∝ I). Then the 2^{2n} = 4^n Majorana monomials map to 4^n distinct Pauli strings up to phase. That is a bijection between the monomial basis of Cl(2n,C) and the Pauli basis of M(2^n,C). The induced algebra isomorphism is inner (Skolem–Noether), and the implementing unitary maps Paulis to Paulis, so it is Clifford. Any operator therefore has the same number of nonzero basis coefficients in every such encoding. This is exactly the repository's W invariance, and it now provably covers ternary trees, HATT, Bonsai, treespilation, Fermihedral (whose SAT encoding imposes algebraic independence explicitly) and Clifford-optimised encodings, not only the linear JW/parity/BK family. Chiew et al. 2024 also shows that ternary trees are linear encodings up to equivalence.
- **What does change** under these encodings:
  - Weight distribution.
  - QWC compatibility and the number of QWC groups G(k) for k<n.
  - Pauli-shadow variance, which grows like 3^weight.
  - CNOT/depth of rotors and of measurement diagonalisers, and SWAP overhead.
  - Qubit-space entanglement.
  What does **not** change: W, GC (full-commutation) groups up to tie-breaking, spectrum and per-word variances. The report should never credit a W reduction to an encoding.
- **Where W can actually move:**
  - **Qubit deletion by fixing k commuting stabilizers** (tapering, symmetry-adapted encodings, two-qubit reduction). A word W and its partner W·S merge, so the reduced count is at least W/2^k, and at most 4^{n−k} words can exist. For H4 at 8 qubits with the two spin-parity qubits removed, W ≥ ⌈7,371/4⌉ = 1,843. The full factor is reached only if every word's partner is present.
  - **Non-Clifford compact encodings** (log₂ C(N,K), segment / qubit-efficient). Each Majorana becomes a sum of Pauli strings, so term counts can grow. No quantitative source was found (see Gaps).
  - **Representation changes at the fermionic level** (Q5) and **symmetry-filtered generator pools** (Q3).
- **Ancilla-based local encodings** (Verstraete–Cirac, Bravyi–Kitaev superfast, Derby–Klassen): logical operators are defined only modulo the code stabilizers. In principle a word count could be lowered by choosing stabilizer-equivalent representatives, at the price of more physical qubits. No source quantifies this for molecular operator products.

### Gaps
- I found no paper that states and proves "W (distinct words in operator products) is invariant across all Majorana-to-Pauli-string encodings" as a named theorem. The argument above is standard Clifford-algebra reasoning, supported by the Clifford-equivalence results of Chiew et al. and Yu et al.
- No quantitative term-count data was found for log₂ C(N,K), segment or qubit-efficient encodings applied to molecular Hamiltonians.

## Q3. Qubit tapering and symmetry: Z2 tapering, point-group and non-abelian symmetries, symmetry-adapted encodings, number-conserving encodings and spin adaptation. How many qubits and terms do they remove, and do they work with excitation-generator subspaces?

### Takeaway
Z2/Boolean symmetries remove **2–5 qubits** for standard small molecules in STO-3G: two from spin parities plus up to three from abelian point-group generators. The removal is exact in the chosen sector. Point-group symmetry also **zeroes out excitation operators that break the symmetry**: in one study the UCCSD parameter count fell by 50–81%. The authors note that this pool reduction could be had in JW simply by filtering the pool. That filtering is the larger and cheaper lever for a subspace method whose cost scales with the number of generator products. Non-abelian symmetries (total spin S², C3v and similar) cannot be tapered as Z2 symmetries. They act through spin- or symmetry-adapted bases and pools, and qubit savings from them need non-Clifford encodings.

### Cited Findings
- **Bravyi–Gambetta–Mezzacapo–Temme "Tapering off qubits" (2017) [background].** Removes qubits by exploiting Z2 symmetries such as particle-number and spin parities; it is the basis of OpenFermion's `symmetry_conserving_bravyi_kitaev`, which removes two qubits. — [arXiv:1701.08213](https://arxiv.org/abs/1701.08213); [OpenFermion SCBK](https://quantumai.google/reference/python/openfermion/transforms/symmetry_conserving_bravyi_kitaev)
- **Setia et al. (JCTC 2020).** Point-group operators written in second quantisation and tapered via the largest abelian subgroup. All molecules are STO-3G. Qubits as extracted (original → Bravyi Z2 → point-group method):

  | Molecule | Original | Bravyi Z2 | Point-group method |
  |---|---|---|---|
  | H₂O (C2v) | 14 | 11 | 10 |
  | NH₃ (C3v) | 16 | 14 | 13 |
  | C₂H₄ (D2h) | 28 | 25 | 23 |
  | BF₃ (D3h) | 40 | 37 | 35 |
  | LiH | 12 | 8 | 8 |
  | CO₂ | 30 | 26 | 25 |
  | C₂H₂ (D∞h) | 24 | 20 | 19 |
  | BeH₂ (D∞h) | 14 | 10 | 9 |

  The "after" values were computed by subtracting the paper's tapered-qubit counts; verify against Table 1. In C3v the rotation and reflection classes do not commute, so only one qubit is removed for NH₃. The paper does not report term counts. — [arXiv:1910.14644](https://ar5iv.labs.arxiv.org/html/1910.14644)
- **Picozzi & Tennyson, symmetry-adapted encodings (QST 8, 035026, 2023).**
  - The Boolean point-group symmetries are read straight from the character table. Each is a Z-string in JW.
  - The encoding is a Clifford, given by an explicit tableau, followed by projection. It removes k = 2–5 qubits and "acts simply as a relabelling of the Pauli terms of the JW Hamiltonian (followed by a simplification of like terms)", with eigenvalues exact in the target sector. The "simplification of like terms" is the merging that can lower W.
  - JW → SAE qubits: H₃⁺ (STO-3G) 6→3, H₂ (double-zeta) 8→5, LiH (STO-3G) 12→8, BeH₂ 14→9, H₂O 14→10.
  - UCCSD parameters (spin already accounted for): 8→4, 15→7, 92→34, 204→38, 140→48, a reduction of up to about 80% for BeH₂.
  - For a fermionic excitation operator, either all of its Pauli terms commute with all Boolean symmetries or they all anticommute with at least one. Symmetry-breaking excitations are therefore projected to zero.
  - The authors attribute the faster convergence to the smaller parameter count and pool, not to the qubit count, and state that "the same improvement … could be obtained in JW or other encoding by only including excitations that preserve point-group symmetries".
  - CNOT count and depth were consistently lower for SAE (no numbers in the text). The open-source **QuantumSymmetry** package works with OpenFermion and Qiskit.
  - [Picozzi & Tennyson, QST 2023](https://doi.org/10.1088/2058-9565/acd86c) ([PDF](https://discovery.ucl.ac.uk/10172278/1/Picozzi_2023_Quantum_Sci._Technol._8_035026.pdf))
- A 2024 follow-up gives a shorter derivation of the affine symmetry-adapted encoding and corrects a small point in the original proof. Later preprints extend the approach to CAS methods and periodic crystals. — [arXiv:2407.03452](https://arxiv.org/abs/2407.03452)
- **Contextual subspace (CS-VQE).**
  - Kirby–Tranter–Love (Quantum 2021): on tapered small-molecule Hamiltonians, the qubit count needed for chemical accuracy can drop by more than half in some cases, and at least one qubit is saved in nearly all cases. — [Quantum 5, 456](https://quantum-journal.org/papers/q-2021-05-14-456/)
  - Weaving et al. (JCTC 2023) recast it in a stabilizer framework (tapering and CS share one rotate-and-fix structure) and warn that STO-3G gives chemical *precision*, not accuracy. — [JCTC 2023 PDF](https://discovery.ucl.ac.uk/10163800/1/acs.jctc.2c00910.pdf)
  - Hardware runs: HCl ground state on 27-qubit IBM Falcon devices with tapering plus CS ([arXiv:2303.00445](https://arxiv.org/abs/2303.00445)), with "just 3 qubits" sufficient for HCl according to a later paper; and the N₂ dissociation curve in STO-3G on superconducting hardware ([arXiv:2312.04392](https://arxiv.org/abs/2312.04392)).
  - CS is approximate: it trades qubits for an energy bias.
- **Number-conserving compact encodings** reach ⌈log₂ C(N,K)⌉ qubits but are non-Clifford beyond one qubit of saving (see Q2). — [arXiv:2211.04501](https://arxiv.org/abs/2211.04501)
- **Spin adaptation.**
  - Spin-adapted (CSF) qubit encodings for spin systems truncate intermediate total-spin quantum numbers. The truncation is approximate in principle. [preprint] — [arXiv:2412.14797](https://arxiv.org/abs/2412.14797)
  - The "Quantum Paldus Transform" block-diagonalises spin-free fermionic Hamiltonians and prepares CSFs. [preprint] — [arXiv:2506.09151](https://arxiv.org/abs/2506.09151)
  - A qutrit spatial-orbital encoding reports that the CASCI(8,6) singlet space has 105 CSFs, of which its six-qutrit encoding keeps 90. [preprint; very recent, treat with caution] — [arXiv:2608.15577](https://arxiv.org/abs/2608.15577)
- **Repository (local).**
  - clifford_qc **already implements** Z2 symmetry tapering, the two-qubit spin-parity reduction and contextual-subspace restriction as one "Clifford rotate then fix stabilizer qubits" primitive. The primitive transports the Hamiltonian, reference, generators and observables together; words W and W·Z_q merge under the fix. — `/home/user/clifford_qc/clifford_qc/subspace/restriction.py`, `/home/user/clifford_qc/ARCHITECTURE.md` ("Reduction semantics")
  - Its exchange/generator pools "do not impose total-spin or point-group symmetry". — `/home/user/clifford_qc/clifford_qc/algorithms/exchange.py` line 8
  - The task brief's statement that the framework lacks qubit tapering is therefore inaccurate. **Point-group symmetry and spin adaptation are the missing pieces.**

### Inferences
- **Pool filtering is the largest exact lever.** For A-CASE / DA-CASE, cost is driven by the number of generator pairs (i,j) and the words they generate. Filtering the pool by point-group irrep (keep A_i whose irrep product with the reference irrep matches the target) is exact, cheap, encoding-independent and needs no qubit change. Picozzi's numbers suggest pool reductions of 50–81% for small STO-3G molecules. Pair counts scale quadratically, so a pool cut to a fraction f shrinks the number of A_i†HA_j pairs to about f², and the union of their words W is expected to fall with it (not measured; W is a union, so the fall is sublinear in pair count).
- **Point-group tapering on top of spin parities** adds 0–3 more removed qubits. W can fall by at most 2 per extra removed qubit, and the restriction primitive in clifford_qc can already host these extra stabilizers. A highly symmetric H4 (D2h/D4h) could plausibly reach 4–5 removed qubits of 8. That is unverified: no source gives H4 counts.
- **Total-spin (S²) symmetry is non-abelian.** It gives no Z2 qubit removal beyond the S_z parity. It helps by (i) spin-adapting the generator pool (singlet-coupled excitation pairs), which cuts pool size and removes spin-contaminated subspace directions, and (ii) leakage tests. Qubit savings from it require CSF/Paldus encodings, which are non-Clifford and leave the sparse-Pauli representation.
- **Compatibility with excitation-generator subspaces is good** for Z2 tapering and SAE. Symmetry-preserving generators commute with the stabilizers and transport exactly, while symmetry-breaking ones vanish. CS restriction is approximate and its bias must be reported, as the repository's ARCHITECTURE table already says.

### Gaps
- No source gives tapered qubit counts or term/word counts for H4 (chain, square or rectangular) or for operator products A†HA after tapering.
- I did not find a 2023–2026 paper that tapers genuinely non-abelian point groups beyond abelian subgroups, other than via symmetry-adapted bases. The "non-abelian tapering" literature was not located in this session.
- Spin-adapted encodings for *molecular* Hamiltonians with qubit counts were not found. The located works cover spin systems, block-diagonalisation or qutrits.

## Q4. Working directly in the Majorana/fermionic algebra: fermionic and matchgate shadows, Majorana-basis measurement, Gaussian references and Wick's theorem. Could a subspace on a Slater/Gaussian reference be evaluated classically, and what would remain for the device?

### Takeaway
Fermionic-algebra measurement schemes cut the number of measurement **settings and shots** in a way that does not depend on the encoding or on W:
- O(N²) circuits suffice for the full 2-RDM, against O(N⁴) for Pauli schemes.
- Fermionic shadows estimate all k-RDM elements with about C(n,k)k^{3/2} log n/ε² state preparations.
- Matchgate shadows give Gaussian overlaps in cubic time.

A subspace built from polynomial fermionic generators on a **Slater-determinant or Gaussian reference is entirely classical**. Every Majorana-monomial expectation is a Pfaffian or determinant (Wick), and for a determinant in its own orbital basis it is simply ±1 or 0. The quantum device earns its place only when the reference is non-Gaussian and hard to simulate classically. Even then, Majorana-length truncation (Majorana Propagation) and Gaussian-extent methods push the classical frontier to 28–52 modes for typical circuits.

### Cited Findings
- **Zhao–Rubin–Miyake (PRL 2021).** Estimating all k-RDM elements to additive ε needs O(C(n,k) k^{3/2} log n / ε²) state preparations, optimal up to the log. Settings come from a discrete group of fermionic Gaussian unitaries with linear depth. The paper claims substantially better constants than earlier deterministic schemes for k ≥ 2. — [arXiv:2010.16094](https://arxiv.org/abs/2010.16094)
- **Bonet-Monroig–Babbush–O'Brien (PRX 10, 031064, 2020).** The fermionic 2-RDM can be sampled with **O(N²) unique circuits**, which is asymptotically optimal, against O(N⁴) for earlier Pauli-based schemes. Each circuit is O(N)-depth on a line. Lower bound: Ω(N^k ε⁻²) preparations for a k-RDM. Qubit k-RDMs need O(3^k log^{k−1} N) circuits. — [arXiv:1908.05628](https://arxiv.org/abs/1908.05628)
- **Matchgate shadows (Wan, Huggins, Lee, Babbush, CMP 2023).** The first three moments of Haar-random matchgate circuits equal those of the discrete set of **Clifford matchgates**. Overlaps of an arbitrary state with any fermionic Gaussian state come out of the shadow in cubic time, with variance O(√n log n). Motivated by QC-AFQMC. — [arXiv:2207.13723](https://arxiv.org/abs/2207.13723); noise follow-up [arXiv:2404.18303](https://arxiv.org/abs/2404.18303)
- **Ternary-tree RDM learning.** All k-fermion RDMs to precision ε from about (2n+1)^k ε⁻² repetitions of a single circuit. — [arXiv:1910.10746](https://arxiv.org/abs/1910.10746)
- **Higher-order RDMs via cumulants.** For quantum subspace methods on 1D/2D Fermi–Hubbard, approximating higher RDMs by cumulant expansion is a biased estimate, but it is more measurement-efficient and suppresses hardware noise better than unbiased optimal measurement. [preprint] — [arXiv:2312.17452](https://ar5iv.labs.arxiv.org/html/2312.17452)
- **Fermionic grouping vs qubit grouping.**
  - Li et al. (PRA 2025) partition the O(N⁴) terms into O(N²) commuting fermionic groups (best earlier grouping: O(N³)), and a parallel-evolution scheme cuts evolution depth by a factor N. — [arXiv:2505.08206](https://arxiv.org/abs/2505.08206)
  - Bincoletto & Kottmann (Digital Discovery 2025) use a hard-core-boson approximation that gives three commuting groups plus a residual measured in rotated orbital bases. They report 30–80% fewer measurements and lower measurement-circuit depth, and the method is approximate. — [RSC Digital Discovery](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00251f)
  - A *negative* result: the Izmaylov group's "fluid fermionic fragments" work (Quantum 2023) found that even its best fermionic-space repartitioning had higher measurement cost M(ε) than the qubit-space methods. — [Quantum 2023](https://quantum-journal.org/papers/q-2023-01-03-889/pdf/)
- **Majorana joint measurements (McNulty, Calegari, Oszmaniec, 2024).** Binary measurements of products of anticommuting Majoranas, with a joint-measurement scheme of asymptotically optimal noise built from a few fermionic Gaussian unitaries. [preprint; the arXiv ID was inferred from a search listing and not opened] — [arXiv:2402.19349](https://arxiv.org/abs/2402.19349)
- **Majorana Propagation (Miller, García-Pérez et al., Algorithmiq; Mar–Dec 2025) [preprint].** Heisenberg-picture evolution of observables in the **Majorana-monomial basis**, truncated by monomial **length**, on the argument that long monomials rarely contribute for unstructured circuits. Used to build adaptive fermionic ansätze for active spaces of 28, 40 and 52 modes. The authors claim it is orders of magnitude faster and more accurate than state-of-the-art tensor-network circuit simulators. — [arXiv:2503.18939](https://arxiv.org/abs/2503.18939)
- **Classical simulation beyond Gaussian states (Dias & König, Quantum 2024).** Fermionic linear optics on non-Gaussian initial states runs in time polynomial in the number of fermions, the accuracy and measures of non-Gaussianity, via the fermionic Gaussian extent. This is the fermionic analogue of stabilizer-rank simulation. — [Quantum 8, 1350](https://quantum-journal.org/papers/q-2024-05-21-1350/)
- **Free fermions are classically simulable [background].** Non-interacting fermion (matchgate) circuits and Gaussian states can be simulated classically. — [Terhal & DiVincenzo, quant-ph/0108010](https://arxiv.org/abs/quant-ph/0108010)

### Inferences
- **Classical evaluation of the whole bank on a determinant reference.** Under JW, BK or any linear encoding, the HF determinant is a computational-basis state. Every Pauli word then has ⟨Φ|P|Φ⟩ ∈ {0, ±1}, nonzero only when the word has no X/Y component in that frame. The whole bank of W words can be evaluated exactly in O(W·n) bit operations.
  - More generally, A_j|Φ⟩ for excitation generators is a small combination of determinants. The A-CASE generalized eigenproblem on an HF reference is therefore a **selected-CI problem that needs no quantum device**.
  - For a Gaussian reference that is not a determinant in the working basis (an orbital-rotated or BCS-like state), each Majorana monomial of degree d has expectation Pf(M_S), the Pfaffian of a d×d submatrix of the covariance matrix: O(d³) with d ≤ 12 for A†HA with double-excitation generators.
  - This argument is encoding-independent.
- **What remains for the device** is a correlated, classically hard reference: for example a deep ADAPT/UCC state, time-evolved Krylov states e^{−iHt}|Φ⟩, or hardware-efficient states. For such a reference the measured object is the set of reference expectation values of the W words. These are linear functionals of the reference's k-RDMs, with k = (deg A_i + deg H + deg A_j)/2 in Majorana degree: k ≤ 4 for single-excitation generators and k ≤ 6 for doubles.
  - Fermionic measurement settings (Gaussian rotations, fermionic shadows) cover many words that do not QWC in any encoding. The number of settings then scales as O(n^k) or C(n,k) and is **independent of W and of the encoding**.
  - For 8-qubit H4 this asymptotic gain is small: C(8,4) = 70, and full tomography is only 4^8. It becomes decisive at n ≳ 20–30 qubits.
  - Fermionic shadows estimate many monomials at once from the same circuits. clifford_qc's pooled estimator could ingest such data unchanged if it were indexed by Majorana monomial.
- **Majorana degree is the Clifford-algebra grade.** It is encoding-invariant, while Pauli weight is not. Grade-based truncation or prioritisation of the operator bank (as in Majorana Propagation) is a concrete, measurable way for clifford_qc's multivector representation to save work. With JW-realised generators, the grade of a stored Pauli word follows from the inverse GF(2) map at negligible cost.
- **Cumulant (approximate) reconstruction** of 3- to 6-RDMs from lower RDMs would cut the measured set to roughly the 2-RDM: O(N²) circuits by Bonet-Monroig. This introduces a bias that must be reported, in the same spirit as the CS bias.

### Gaps
- No paper was found that applies fermionic shadows or Gaussian-rotation measurement schedules specifically to A†HA products in a generator-based subspace method (QSE / A-CASE), with measured setting counts.
- I did not verify whether Majorana Propagation has been published in a refereed venue, nor any independent replication.

## Q5. Low-rank and factorised Hamiltonians (double factorisation, compressed DF, THC, Cholesky, symmetry shifts): reported term and measurement reductions, and relevance to A†HA

### Takeaway
Factorisations change **the operator's representation**, not the encoding. They reduce measurement groups for ⟨H⟩ from O(N³–N⁴) to O(N) basis-rotation groups. Compressed DF roughly halves term counts (one PennyLane demo), and symmetry shifts (BLISS) cut 1-norms by 2–3×. They help directly with ⟨H⟩ and with subspace methods whose matrix elements are H between prepared states (QFD/Krylov). They **do not** factor the mixed products A_i†HA_j unless one first contracts through RDMs.

### Cited Findings
- **Basis-rotation grouping (Huggins et al., npj QI 2021).** Double factorisation; each factor's one-body basis rotation U_ℓ lets all number and density expectation values in that basis be sampled at once. L = O(N) rotations suffice for arbitrary-basis chemistry, and L = 1 in plane-wave or dual bases. This is a **cubic reduction in term groupings** over earlier methods, with added noise resilience. — [arXiv:1907.13117](https://arxiv.org/abs/1907.13117)
- **Compressed DF in a subspace method (Cohn, Motta, Parrish, PRX Quantum 2, 040352, 2021).** Quantum filter diagonalisation (a non-orthogonal subspace method) with a compressed DF Hamiltonian for time propagation and a converged DF for measurement. C-DF gives "substantial additional compression" over explicit DF at fixed accuracy, with shorter circuits and fewer measurements, shown on simulators, noise models and hardware. — [arXiv:2104.08957](https://arxiv.org/abs/2104.08957)
- **Regularised compressed DF (Oumarou et al., Quantum 2024)** gives compact molecular Hamiltonian representations. — [Quantum 8, 1371](https://quantum-journal.org/papers/q-2024-06-13-1371/)
- **PennyLane.**
  - `qml.qchem.factorize(..., cholesky=True, compressed=True, regularization="L2")` implements DF/CDF (JAX/Optax required).
  - The March 2025 demo chains DF → BLISS → compression and reports that "the number of terms decreased almost by half".
  - [factorize docs](https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.factorize.html); [CDF demo](https://www.pennylane.ai/demos/tutorial_how_to_build_compressed_double_factorized_hamiltonians)
- **BLISS (Loaiza & Izmaylov, 2023).** Subtracts an operator that annihilates the target symmetry sector (particle number), leaving that sector's spectrum unchanged. This cuts the LCU 1-norm by 2–3× compared with a Pauli-product LCU. A linear-programming version (Patel et al., 2024) handles active spaces up to 76 orbitals and reduces both spectral range and Pauli or fermionic LCU 1-norms. — [arXiv:2304.13772](https://arxiv.org/abs/2304.13772); [arXiv:2409.18277](https://arxiv.org/abs/2409.18277)
- **Measurement grouping with non-local transformations (Yen, Ganeshram, Izmaylov, npj QI 2023).** Greedy compatible grouping, non-local transformations before measurement, and Pauli products shared across several groups (with covariance estimates) give a severalfold reduction in measurements over earlier state of the art for model molecules. — [arXiv:2201.01471](https://arxiv.org/abs/2201.01471)

### Inferences
- **For A-CASE**, DF of H alone does not give a common eigenbasis for A_i†U_ℓ†D_ℓU_ℓA_j, so it does not directly reduce W or the groups for the mixed products. There are two compatible routes:
  - **RDM contraction.** Measure the reference's k-RDM with Gaussian-rotation or fermionic-shadow settings, then contract classically with h, g and the generator coefficients. DF and THC then only speed up the classical contraction.
  - **Generator states.** Prepare A_j|ψ⟩ (or e^{iθA_j}|ψ⟩) as states and measure H between them, QFD-style, with DF basis-rotation groups: O(N) groups per matrix element.
- **BLISS-type shifts are exactly compatible with number-conserving generator subspaces.** If K annihilates the N_e-electron sector and every A_j conserves N, then ⟨ψ|A_i†(H−K)A_j|ψ⟩ = ⟨ψ|A_i†HA_j|ψ⟩. The free parameters of K could be chosen to minimise the *variance*, or possibly the word count, of the A†HA bank. This is untested; I found no source applying BLISS to subspace matrix elements.
- **THC and DF Toffoli-count results** concern fault-tolerant QPE (block encodings). They are not measurement reductions for a near-term subspace method and should not be credited as such.

### Gaps
- No source reports the number of distinct Pauli words of A†HA (or of H) after DF/CDF truncation in a form comparable to W.
- Tensor hypercontraction measurement-cost data for near-term estimation was not retrieved; its main published metrics concern block-encoding cost.

## Q6. Geometric/Clifford algebra in quantum computing (2020–2026): what has the GA / Clifford-algebra / Majorana viewpoint concretely delivered, and what is only reformulation?

### Takeaway
"GA proper" work (Quantum Register Algebra, GA Jordan–Wigner, GA online calculators, real-GA formalisms) is **reformulation and software notation**. I found no benchmark showing a GA-based simulator beating state-vector or stabilizer simulators. By contrast, the **Clifford-algebra / Majorana structure** (anticommutation graphs, grades, Gaussian/matchgate groups, Clifford equivalence of encodings) has delivered measurable results:
- Classical solvability tests (free fermions in disguise).
- Optimal anticommuting sets.
- Measurement schemes (matchgate and fermionic shadows, unitary partitioning).
- Encoding search spaces (Clifford-circuit optimisation, equivalence classes).
- Truncated classical simulation (Majorana Propagation).

### Cited Findings
- **Quantum Register Algebra (Hrdina, Hildenbrand, Návrat, Steinmetz, Alves, Lavor, Vašík, Eryganov, Quantum Inf. Process. 22, 2023).** Presented as "the mathematical language for quantum computing", linked to Dirac notation and implemented in GAALOP. It reports no speedups. A predecessor "online calculator" paper (AACA 2022) and a 2024 LNCS version exist. — [arXiv:2208.02608](https://arxiv.org/abs/2208.02608)
- **GA Jordan–Wigner transformation (Veyrac & Toffano, Entropy 26, 410, 2024).** A conceptual reformulation with no performance data. — [Entropy 2024](https://www.mdpi.com/1099-4300/26/5/410)
- A 2025 preprint, revised April 2026, develops a real formalism for quantum information aimed at classical simulation, and a separate 2025 complex-Clifford-algebra paper treats the geometric product as a unified primitive for qubits and gates. Search excerpts from both show no runtime benchmarks. A search for head-to-head GA-simulator benchmarks against state-vector simulators returned none. — [arXiv:2512.07902](https://arxiv.org/html/2512.07902v2) [preprint; I saw only excerpts]; [state-vector benchmark review, SciPost Phys. Core 7, 075](https://www.scipost.org/SciPostPhysCore.7.4.075)
- **GA software.**
  - **kingdon** (Roelfs, 2025): MIT licence; symbolic optimisation, common-subexpression elimination and JIT code generation per operation and input type; exploits **input sparsity** (only provably nonzero blades are computed) and has a `graded` mode. It is input-agnostic (NumPy, PyTorch, SymPy). The authors contrast it with `clifford` (numeric only) and `galgebra` (symbolic only).
  - None of these targets the 2^n-dimensional Cl(2n,C) at the qubit counts used in quantum chemistry.
  - [kingdon arXiv:2503.10451](https://arxiv.org/abs/2503.10451)
- **Free fermions behind the disguise (Elman, Chapman, Flammia, CMP 2021).** If a spin Hamiltonian's **frustration graph** (its anticommutation graph) is (even-hole, claw)-free, an explicit free-fermion solution exists for any couplings, even when no Jordan–Wigner transformation applies. Claw-free graphs alone give a hierarchy of commuting conserved charges. Recognition takes polynomial time. — [arXiv:2012.07857](https://arxiv.org/abs/2012.07857)
- **Quasi-Clifford algebras to Pauli algebras (Huber, Aug 2025) [preprint].** Generators with α_i² = k_i and α_jα_i = (−1)^{χ_ij} α_iα_j map to Pauli algebras. This gives a Wedderburn decomposition and block-diagonalisation of matrix groups with quasi-Clifford structure, with applications to symmetry reduction of SDPs and to constructing maximal anticommuting subsets. Jordan–Wigner is recovered for Majoranas. No quantitative savings are reported. — [arXiv:2508.01470](https://arxiv.org/abs/2508.01470)
- **Clifford equivalence of encodings** (Chiew–Harrison–Strelchuk 2024; Chen–Xu PRX Quantum 2023) and **Clifford-circuit search for encodings** (Yu et al., JCTC 2025: 15–40% lower weight) — see Q1–Q2. — [arXiv:2412.07578](https://arxiv.org/abs/2412.07578); [arXiv:2502.11933](https://arxiv.org/abs/2502.11933)
- **Matchgate shadows** rely on the fact that Clifford matchgates match the Haar moments of all matchgates up to third order (Q4). — [arXiv:2207.13723](https://arxiv.org/abs/2207.13723)
- **Majorana Propagation** truncates by Majorana monomial length, i.e. Clifford-algebra grade (Q4). — [arXiv:2503.18939](https://arxiv.org/abs/2503.18939)
- **Maximally strong uncertainty relations for anticommuting observables** are tied to Clifford-algebra generators, with a JW construction on n qubits [background]. — [Wehner & Winter survey, arXiv:0907.3704](https://arxiv.org/abs/0907.3704)

### Inferences
- The repository's own conclusion holds against the outside literature. The GA framing itself (rotors, blades, the geometric product) has nowhere been shown to reduce operator or measurement cost. The **savings come from specific structural facts**:
  - Anticommutation graphs: free-fermion detection, unitary partitioning, the 2n+1 bound.
  - Grade: Majorana-length truncation.
  - The Gaussian/matchgate subgroup: fermionic measurement settings, classical Wick evaluation.
  - Clifford equivalence of encodings: W invariance and encoding search.
  clifford_qc already stores its operators as multivectors with these facts readily computable.
- **Concrete GA-flavoured features that would deliver measurable gains** in this framework:
  1. Grade (Majorana-degree) filtration of the word bank, for truncation and prioritisation.
  2. Gaussian (grade-2 generated) reference detection with Pfaffian/Wick classical evaluation (Q4).
  3. Frustration-graph tests to detect free-fermion-solvable or noncontextual sub-Hamiltonians before spending device shots.
  4. Kingdon-style sparsity-aware code generation for products, which is a constant-factor classical speedup only.

### Gaps
- No refereed benchmark of a GA-based quantum simulator against state-vector, stabilizer or Pauli-propagation simulators was found.
- I found no source that treats anticommuting Pauli sets explicitly as **spin factors (Jordan algebras)** with a measurement or cost consequence. The search returned only general joint-measurability and uncertainty literature.

## Q7. Anticommuting structure: unitary partitioning, maximal anticommuting sets (2n+1), anticommuting LCU and simulation. What does 2023–2026 work say about practical value?

### Takeaway
A set of mutually anticommuting Paulis has at most 2n+1 elements. Any normalised real combination of them squares to the identity. This lets an anticommuting group be measured as one observable after a rotation (unitary partitioning), and lets its exponential be implemented exactly. The practical results are modest:
- Sequence-of-rotations unitary partitioning beat LCU in small-molecule tests.
- It combines with CS-VQE.
- The variance gain over commuting grouping is case-dependent.
- 2026 work points to covariance-aware overlapped groupings as the larger lever.

clifford_qc already implements an anticommuting clique cover.

### Cited Findings
- **Maximum anticommuting sets.** Necessary and sufficient conditions for maximality, an efficient generator of maximum-size anticommuting sets, and counts of maximal (anti)commuting sets. Maximal anticommuting sets have odd size, and the maximum is 2n+1 on n qubits. — [Sarkar & van den Berg arXiv:1909.08123](https://arxiv.org/abs/1909.08123)
- **Unitary partitioning implementation (Ralli, Love, Tranter, Coveney, PRR 3, 033195, 2021).** Groups whose weighted sum is unitary are realised by a sequence of rotations or by LCU. For H₂ and LiH, the **sequence of rotations gives the greatest benefit**, while LCU's probabilistic nature reduces its effectiveness. The reduction lowers the standard error of the mean. Both versions worked under noise, including a 5-qubit IBM run claimed as the first LCU on hardware. — [arXiv:2012.02765](https://arxiv.org/abs/2012.02765)
- **Unitary partitioning plus contextual subspace (Ralli, Weaving, Tranter, Kirby, Love, Coveney, PRR 2023).** Unitary partitioning further reduces the measurements needed for the CS-VQE correction on tapered molecular Hamiltonians. — [arXiv:2207.03451](https://arxiv.org/abs/2207.03451)
- **Grouping heuristics.** Sorted insertion (ordering terms by decreasing coefficient) is reported to outperform conventional greedy colourings for variance-aware grouping. — [Crawford et al. arXiv:1908.06942](https://arxiv.org/abs/1908.06942) (as summarised in search excerpts)
- **Overlapped groupings (2026) [preprint].** Variance depends on shot allocation and on the covariance of jointly measured observables. Relaxing disjointness (a term measured in several groups) gives further variance reduction, with deterministic algorithms. — [arXiv:2604.07156](https://arxiv.org/abs/2604.07156)
- **Repository (local).** `clifford_qc/measurement/cliques.py` is titled "Deterministic unitary partitioning of a fixed real Pauli functional" and implements `anticommuting_clique_cover`. PLAN.md §6.3 notes that a pooled estimator already reads each word from every compatible setting, i.e. an overlapped estimator. — `/home/user/clifford_qc/clifford_qc/measurement/cliques.py`; `/home/user/clifford_qc/PLAN.md` §6.3

### Inferences
- **Algebra** (standard; this is the Clifford-algebra content of unitary partitioning). If P_1..P_m mutually anticommute and Σc_i² = 1 with real c_i, then (Σc_iP_i)² = I. So exp(−iθΣc_iP_i) = cos θ·I − i sin θ·Σc_iP_i, and a product of m−1 Pauli rotations maps Σc_iP_i to a single Pauli word. A group's measurement weight √(Σc_i²) is never larger than Σ|c_i|.
  - This is exactly "a unit vector in the grade-1 subspace of a Clifford algebra squares to 1".
  - It is the one place where GA-style rotor algebra yields an exact circuit identity with measurable value, through fewer measured groups and exact exponentiation of anticommuting clusters.
  - The cost is extra two-qubit gates for the rotation sequence, which do depend on the encoding.
- **For W-dominated subspace costs**, unitary partitioning reduces measured **groups and shots**, not W. With m ≤ 2n+1 terms per group, an anticommuting cover needs at least W/(2n+1) groups: at least about 434 for W = 7,371 at n = 8. Commuting (GC) groups can be much larger (up to 2^n words, all products of n independent commuting generators), so on raw group count commuting grouping usually wins at large n. The relevant comparison is variance per shot, which depends on the reference state and on covariances; it is not settled in the literature for subspace products.
- **Linear combinations of anticommuting Paulis in Hamiltonian simulation.** The exact-exponentiation identity can collapse an anticommuting cluster's Trotter step into one multi-qubit rotation conjugated by Cliffords. I found no 2023–2026 source quantifying Trotter-error or gate savings from this for molecular Hamiltonians (see Gaps).

### Gaps
- I found no head-to-head, shot-budget-matched comparison of anticommuting (unitary-partitioning) versus commuting grouping on molecular Hamiltonians from 2023–2026.
- No 2023–2026 source was found that quantifies Hamiltonian-simulation savings from exponentiating anticommuting clusters exactly (the identity is folklore). Searches returned only commuting-cluster diagonalisation work ([Quantum 4, 322 (2020)](https://quantum-journal.org/papers/q-2020-09-12-322/); [npj QI 2023](https://www.nature.com/articles/s41534-023-00697-6.pdf)).
