# Pauli-operator algebra kernels: state of the art (Nov 2026) and what it implies for clifford_qc

Scope note for the report writer: these notes cover outside libraries and papers, plus one local microbenchmark run in a scratch directory (no repository file was changed apart from this note). Every library claim below is self-reported by that library's authors unless marked otherwise. "Pairs" means term-pairs in an outer product of two Pauli sums. Dates: PauliArray (2024), Stim (2021) and Picasso (2024) are older than 2025. All other primary sources are from 2025–2026.

## 1. How leading libraries encode Pauli strings and sparse Pauli sums

### Takeaway
Libraries now converge on one design: each Pauli string is a pair of bit-planes (x|z, or an equivalent GF(2)-linear variant) packed into 64-bit words, with a small integer phase exponent mod 4. A sum is stored structure-of-arrays (SoA): a code array plus a coefficient array, usually kept sorted or bucketed. Libraries that store strings as Python objects or character strings (Cirq, PennyLane, OpenFermion), or that key a hash map by string, are the slow end of every comparison. clifford_qc already uses a correct bitwise word kernel (2 bits per qubit in an arbitrary-precision Python int), but stores sums as a Python `dict[int, complex]` and multiplies them with a per-pair Python double loop.

### Cited Findings

**clifford_qc baseline (local repository, read-only skim)**
- `MV` stores a sparse map from packed Pauli-word codes to complex coefficients. Each qubit uses two bits (`0=I, 1=X, 2=Y, 3=Z`). — [ARCHITECTURE.md](/home/user/clifford_qc/ARCHITECTURE.md)
- The word product is `c = a ^ b` on the packed codes, with the phase exponent `e = (pop(xa&za) + pop(xb&zb) − pop(xc&zc) + 2·pop(za&xb)) mod 4`. Here x and z are unpacked from the 2-bit lanes with masks (`z = (c>>1)&lo`, `x = (c&lo)^z`). The lane map is GF(2)-linear, so XOR of codes equals XOR of the (x,z) planes. — [clifford_qc/pauli_kernel.py](/home/user/clifford_qc/clifford_qc/pauli_kernel.py)
- Word products are memoised process-wide with `functools.lru_cache` (default 1,000,000 entries). The source comment says a full table at 12 qubits is "roughly 250 MiB for products", that a miss costs "about twice an unmemoized call", and that the table "pays for itself only above a ~70% hit rate". — [clifford_qc/pauli_kernel.py](/home/user/clifford_qc/clifford_qc/pauli_kernel.py)
- `MV.__mul__` is a Python double loop over `self.terms.items()` × `other.terms.items()`. It calls `_word_mul_unchecked` per pair and accumulates with `out[m] = out.get(m, 0j) + ca*cb*ph`. `multivector.py` has no dedicated commutator that pre-filters anticommuting pairs. — [clifford_qc/multivector.py](/home/user/clifford_qc/clifford_qc/multivector.py)
- Packaging constraints: `license = "Apache-2.0"`, `requires-python = ">=3.10"`, `dependencies = ["numpy>=1.23"]`. — [pyproject.toml](/home/user/clifford_qc/pyproject.toml)

**Survey of encodings**

| Library | String encoding | Sum container | Core language / binding | Licence | Source |
|---|---|---|---|---|---|
| Stim (PauliString, Tableau) | Binary symplectic bits in cache-friendly, 256-bit-SIMD-aligned words; stabilizer tableau, tracking the inverse tableau | No general Pauli-sum algebra (stabilizer formalism only, per PauLIB's survey) | C++ with Python bindings | not checked this session | [Stim paper](https://arxiv.org/abs/2103.02202); [PauLIB survey](https://arxiv.org/html/2605.25974) |
| Qiskit `SparsePauliOp` / `PauliList` | NumPy boolean arrays `x`, `z` of shape (M, n) plus an integer group phase | Parallel arrays (paulis, coeffs); `simplify` packs bits with `np.packbits` into a uint16-combined key and dedups with Rust `unordered_unique` | Python/NumPy with Rust (`qiskit._accelerate`) helpers | Apache-2.0 (Qiskit) | [sparse_pauli_op.py](https://github.com/Qiskit/qiskit/blob/main/qiskit/quantum_info/operators/symplectic/sparse_pauli_op.py) |
| Qiskit `SparseObservable` (Rust core) | Per-term list of non-identity letters only: `bit_terms` (uint8 codes; the low 2 bits are the symplectic Pauli, the upper 2 bits mark projectors), `indices` (uint32, sorted per term) | CSR-like: `coeffs`, `bit_terms`, `indices`, `boundaries` (uintp, length t+1). Extended alphabet includes the projectors 0, 1, +, −, r, l | Rust with a Python API | Apache-2.0 | [SparseObservable docs](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparseObservable) |
| PauliEngine | Binary symplectic (x,y) variant: x_i set for X or Z, y_i set for Y or Z; (0,0)=I, (1,0)=X, (0,1)=Y, (1,1)=Z | The paper describes a "memory-efficient symbolic dictionary"; the README says complex-coefficient Hamiltonians store terms "as flat arrays" | Header-only C++20 template library, nanobind frontend, SymEngine coefficients | MIT | [paper](https://arxiv.org/html/2601.02233); [GitHub](https://github.com/tequilahub/pauliengine) |
| PauliArray | Two boolean vectors x, z per string, stored as NumPy `bool` arrays with a hidden qubit axis. Not bit-packed: the authors say a dedicated bit-array backend "could prove beneficial" | NumPy arrays with broadcasting; operators are weighted arrays; `simplify` merges duplicates | Pure NumPy | **CC-BY-SA-4.0** | [paper](https://arxiv.org/html/2405.19287); [GitHub](https://github.com/algolab-quantique/pauliarray) |
| PauliStrings.jl | Pair of binary integers (v, w); products by XOR; sign `(−1)^pop(v1∧w2)` | Paper describes coefficients keyed by (v,w) and merged on insertion; `OperatorTS1D` stores translation-symmetric operators with an N-fold memory saving | Julia (Python port PauliStrings.py is separate and less mature) | MIT | [paper](https://arxiv.org/html/2410.09654v3); [SciPost Codeb. 54](https://scipost.org/SciPostPhysCodeb.54); [GitHub](https://github.com/nicolasloizeau/PauliStrings.jl) |
| PauliPropagation.jl | One unsigned integer per string, read as a base-4 numeral (2 bits/qubit). The integer width is chosen from the qubit count, and wider types are generated and dispatched on demand | `PauliSum` (PauLIB describes it as hash-map storage); since v0.7 also an array-based `VectorPauliSum` | Julia | not checked | [paper](https://arxiv.org/html/2505.21606v1); [search summary of the paper](https://arxiv.org/pdf/2505.21606) |
| PauLIB (2026) | Bit-packed struct: ⌈n/64⌉ X words + ⌈n/64⌉ Z words + 1 flags byte for phase (16W+8 bytes, no heap) | SoA: each word index of the X and Z planes, plus flags and coeffs, in its own contiguous array. **Sorted flat arrays, explicitly not hash maps** | C++ (OpenMP, CUDA, MPI prototype); a Python interface is mentioned without detail | "open source"; licence not named in the paper | [PauLIB](https://arxiv.org/html/2605.25974) |
| NVIDIA cuPauliProp (cuQuantum ≥ 25.11) | Packed uint64: `[X_ints…, Z_ints…]` per term | `(max_terms, 2·num_packed_ints)` uint64 array + float64 coeff array, with `is_sorted` / `is_unique` flags; operator application has `make_sorted` and `keep_duplicates` options | CUDA C library; experimental Python `cuquantum.pauliprop.experimental`; `pip install cupauliprop-cu13` | NVIDIA proprietary SDK terms (licence text not shown on the pages read) | [docs](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/); [blog](https://developer.nvidia.com/blog/advanced-large-scale-quantum-simulation-techniques-in-cuquantum-sdk-v25-11/) |
| Sparse Pauli dynamics (SPD; Begušić, Gray, Chan) | 2n bits (z…, x…) packed into uint64 arrays of length 2⌈n/64⌉, plus integer phase q | Three arrays (Pauli rows N×n64, phases, complex coeffs). **Kept sorted; lookup by binary search** in O(n64·log N) | Python ("our Python implementation") | **GPL-3.0** | [paper](https://arxiv.org/html/2308.05077v3); [GitHub](https://github.com/tbegusic/spd) |
| paulistrings-rs (`paulistrings` crate 0.2.0, 3 Nov 2026) | x and z parts of W×64-bit words each (up to 64·W qubits) | SoA columns **partitioned into buckets by a GF(2)-linear hash** h(v)=H·v (≤31 rows); 2^p partitions for threads, NUMA, MPI and GPU | Rust (rayon; optional MPI and CUDA via cudarc) with a Python package | **MIT OR Apache-2.0** | [docs.rs](https://docs.rs/paulistrings) |
| Qiskit `pauli-prop` (addon, 2025) | not detailed | Sparse Pauli operator; truncation by absolute tolerance and/or term count; docs say much time goes to "sorting the operators and performing Pauli multiplication" | Rust-accelerated Python; deps numpy, qiskit, qiskit-aer | not checked | [PyPI](https://pypi.org/project/pauli-prop/); [docs](https://qiskit.github.io/pauli-prop) |
| fast-pauli (Qognitive) | not stated on README | Focus on applying Pauli strings/operators to batches of states | C++20 + OpenMP, Python bindings via scikit-build-core; `pip install fast_pauli` | BSD-2-Clause | [GitHub](https://github.com/qognitive/fast-pauli) |
| Zixy (Quantinuum, 2026) | not verified | not verified | Rust with PyO3 bindings; PyPI 0.3.0 | not verified | [PyPI 0.3.0 (search snippet only)](https://pypi.org/project/zixy/0.3.0/) |
| PennyLane `PauliWord` / `PauliSentence` | Python dict-based words (PauLIB: "Python objects or character strings") | `PauliSentence` (dict of `PauliWord` → coeff) | Python | Apache-2.0 | [PennyLane docs](https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli.PauliSentence.html); [PauLIB survey](https://arxiv.org/html/2605.25974) |
| OpenFermion `QubitOperator` | Tuple multi-index keys | Python dictionary mapping multi-index → coefficient (PauliEngine mirrors this API) | Python | Apache-2.0 | [PauliEngine paper](https://arxiv.org/html/2601.02233) |
| Cirq `PauliSum` | Python objects (per PauLIB) | — | Python | Apache-2.0 | [PauLIB survey](https://arxiv.org/html/2605.25974) |
| QuantumClifford.jl | Bit-packed Pauli operators in pure Julia | Stabilizer tableaux | Julia | not checked | [docs v0.8.9](https://docs.juliahub.com/QuantumClifford/BsGZO/0.8.9) |
| paulie (QPauLie) | DLA classification from Pauli-string generators | — | Python ≥ 3.12; v0.0.2; citation dated 2026 | not checked | [PyPI](https://pypi.org/project/paulie/) |

- Qiskit's RFC 0021 says `SparseObservable` was designed primarily to fix the scaling problems of `SparsePauliOp`. It was optimised first for minimal memory and only second for mathematical manipulation. — [Qiskit RFC 0021 (via search snippet)](https://github.com/Qiskit/RFCs/blob/master/0021-sparse-observable.md)
- `SparseObservable` memory: a sum of single-qubit Z terms is linear in n (`SparsePauliOp` is quadratic). The all-zeros projector is 1 stored term, versus 2^n terms for `SparsePauliOp`. `+=` appends in amortised linear time, and `+` and `-` do not simplify automatically. — [SparseObservable docs](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparseObservable)

### Inferences
- clifford_qc's interleaved 2-bit-lane code is equivalent to the x|z planes up to a GF(2)-linear relabelling, and its word-level phase formula matches the Stim/PauliArray/PauLIB family. The algebra is not the problem. The container (Python dict of boxed ints) and the call granularity (one Python call per pair) are the problem. This is exactly the per-string-call cost that PauLIB identifies as PauliEngine's main handicap (see §2 and §7).
- clifford_qc's codes fit in machine integers for any published workload: n ≤ 8 fits uint16 and n ≤ 32 fits uint64 (2 bits/qubit). That makes a NumPy SoA layout (`codes: uint16|uint32|uint64[t]`, `coeffs: complex128[t]`) a drop-in internal representation. It follows PauliPropagation.jl's "choose the integer width from n" idea. Beyond 32 qubits, use two or more uint64 words per term (SPD, PauLIB and cuPauliProp all do this).
- Qiskit's projector alphabet is directly relevant to clifford_qc's "states as Pauli multivectors". A computational-basis or stabilizer-like reference density operator has 2^n Pauli terms. A projector-letter alphabet stores |0…0⟩⟨0…0| as one term.

### Gaps
- I did not verify the internals of QuTiP/qutip-qip, PennyLane Catalyst/Lightning Pauli paths, symmer, Qulacs, or Yao/YaoBlocks this session. My background knowledge is unverified here: symmer uses NumPy boolean symplectic matrices, Qulacs a C++ `GeneralQuantumOperator` of per-qubit index lists, and Yao block trees. Treat these as unconfirmed.
- Zixy's data layout, licence and benchmarks could not be read: the PyPI page failed to render and the guessed GitHub URL returned 404.
- The two PauliEngine descriptions of its sum container differ ("dictionary" in the paper, "flat arrays" in the README). The exact container, hash map or sorted, is not documented.
- PauliPropagation.jl's licence and the exact `VectorPauliSum` merge algorithm were not confirmed: that part of the paper was not read.

## 2. Reported benchmark numbers and the design choices behind them

### Takeaway
Reported gaps between well-engineered kernels and Python-object libraries are 1–2 orders of magnitude. Between SIMD/SoA batch kernels and per-string C++ calls, they reach tens of times. The decisive factors are batch (array-at-a-time) APIs, SoA bit-packed layout, and sort-based or array-based merging. Hand-written SIMD intrinsics and many threads add little: PauLIB found intrinsics gave no significant gain and saw only about 3× from 72 threads. On this machine, a plain vectorised NumPy rewrite of clifford_qc's sum product is 3–31× faster than the current dict loop, measured below.

### Cited Findings

**PauLIB (arXiv:2605.25974, May 2026)**: kernels on an NVIDIA GH200 (Grace 72-core Arm CPU, H100 96 GB); GCC 13.2 `-O3 -march=native -ffast-math` — [PauLIB](https://arxiv.org/html/2605.25974)
- Outer product of two sums at 500 qubits, N=200 terms each (40,000 outputs): 0.27 ms. That is **49× faster than PauliEngine and 131× faster than Qiskit**. At N=25: 3.4× vs PauliEngine and 11× vs Qiskit. The Fig. 3 range is 33–49× vs PauliEngine and 11–131× vs Qiskit.
- Memory at 500 qubits is about 137 B/term. RSS for 10^4/10^5/10^6 terms: PauLIB 9/17/135 MB vs Qiskit 83/162/1,051 MB (7.8–9.5× less).
- GPU outer product: H100 peaks at **27.4 G terms/s** at 10^8 output terms. At 10,000×10,000 terms: GPU 3.65 ms vs CPU 684 ms (187×). The CPU curve is flat over 10^5–10^8 outputs, which the paper attributes to memory-bandwidth saturation.
- Thread scaling (3,000×3,000 = 9×10^6 pairs): Grace 2.7× at 8 threads and 3.2× at 72 threads; M4 Pro 1.5× at 8 threads. CPU throughput is **123–145 M pairs/s** at 8+ threads.
- Hand-written intrinsics gave no significant speedup over compiler auto-vectorisation with OpenMP SIMD pragmas. Vector lanes run over terms, not qubits.
- Truncated Pauli propagation, 40 qubits, 10 layers, 1,190 gates, M4 Pro: 0.20 s at trim M=2^14 and 0.62 s at M=2^18.
- On a small propagation workload versus PauliPropagation.jl (same M4 Pro), PauLIB is **1.5–2.1× slower** at matched cutoffs (e.g., 0.44 ms vs 0.21 ms at cutoff 1e-2). The authors attribute this to parallel overhead on tiny inputs.
- Caveat (my reading of the reported numbers): the abstract claims "orders-of-magnitude" gains, but the largest measured CPU gain over a baseline is 131×.

**PauliEngine (arXiv:2601.02233, 2026)** — [paper](https://arxiv.org/html/2601.02233)
- Hamiltonian multiplication at Pauli length 500 (size sweep) and 500 terms (length sweep), on an Intel i9-11900KF with 32 GB: PauliEngine and PauliArray are both 1–2 orders of magnitude faster than OpenFermion. PauliArray is faster on small instances (roughly up to 250 qubits, or small Hamiltonians). PauliArray's memory grows sharply, and its runtime rises steeply once RAM saturates. PauliEngine scales quadratically in Hamiltonian size and near-linearly in string length.
- DLA closure and structure constants vs PennyLane on an Apple M4 with 24 GB: 1–2 orders of magnitude faster. The structure-constant speedup grows from about 44× at N=3 to about 160× at N=10.

**PauliArray (arXiv:2405.19287, 2024; older source)**, Apple M2 MacBook, 100 repetitions — [paper](https://arxiv.org/html/2405.19287)
- Jordan–Wigner mapping of molecular Hamiltonians, from LiH (12 spin-orbitals, 631 strings) to C2H4 (28, 8,919 strings): about **6× faster than Qiskit and about 3× faster than OpenFermion**.
- Commutators with excitation operators: about 2× faster than Qiskit, and far fewer intermediate strings, because non-commuting pairs are selected before composing.

**PauliStrings.jl (SciPost Phys. Codebases 54, March 2025)**: single-threaded ("there is currently no parallelization implemented and only 1 CPU is used") — [paper](https://arxiv.org/html/2410.09654v3)
- XX model, N=40, 40 Lanczos steps with trim M=2^24: 46 min. At n=30, tensor networks were about 40× slower.
- XXZ, N=30, M=2^18: about 5 GB and 1 day to reach t≈10, versus TEBD at 9 GB and 4 days to reach t≈1. Lanczos runs were capped at 40 GB of memory.

**PauliPropagation.jl (arXiv:2505.21606, May 2025)** — [paper](https://arxiv.org/html/2505.21606v1)
- 6×6 tilted-field Ising, single CPU thread: 35 s to 5 min, producing 1.2–9.3 million unique Pauli strings. A tighter cutoff took 26 min and produced about 50 million strings. Merging-BFS "can fill 1 TB of memory in a few hours on a single thread". Thread scaling flattens or degrades with many threads, and GPU support is early-stage.

**NVIDIA cuPauliProp (cuQuantum 25.11)**: π/4-rotation 127-qubit IBM utility circuit — [NVIDIA blog](https://developer.nvidia.com/blog/advanced-large-scale-quantum-simulation-techniques-in-cuquantum-sdk-v25-11/)
- DGX B200 vs single-threaded Qiskit PauliProp on a Xeon Platinum 8570: **55×–177×** across coefficient cutoffs 1e-4, 5e-5 and 2.5e-5. The blog's "multiple orders of magnitude" overstates this range. The same post reports cuStabilizer + Stim at up to 1,060.7× over CPU Stim for surface-code frame sampling (distance 30, 10^6 shots).

**Stim (Quantum 5, 497, 2021; older source)** — [Stim](https://arxiv.org/abs/2103.02202)
- A distance-100 surface-code circuit (about 20,000 qubits, 8M gates, 1M measurements) was analysed in about 15 s, then sampled at about 1 kHz. Stim credits its speed to tracking the inverse tableau (deterministic measurement goes from quadratic to linear) plus a cache-friendly layout and 256-bit SIMD.

**QuantumClifford.jl**: in-place multiplication of 1M-qubit Pauli operators takes 14 μs, versus 16 μs for Stim. The authors call the difference inconsequential and note QuantumClifford.jl is pure Julia. — [QuantumClifford.jl docs v0.8.9](https://docs.juliahub.com/QuantumClifford/BsGZO/0.8.9)

**Local microbenchmark (this session, scratchpad only; not a library claim)**
- Setup: Intel Xeon @ 2.10 GHz, 4 vCPUs, Python 3.13.18, NumPy 2.5.3, single-threaded NumPy. Random distinct words with complex Gaussian coefficients (not chemistry-structured). Every vectorised result matched `MV.__mul__` term-for-term. The vectorised kernel uses the repo's own phase formula: `codes_a[:,None] ^ codes_b[None,:]`, `np.bitwise_count` popcounts, then either a dense `np.bincount` over all 4^n codes or `np.unique` + `np.bincount`. Script: /tmp/claude-0/-home-user-clifford-qc/d48c2e9d-7273-52fc-8535-9900e920c354/scratchpad/bench_kernel.py

| n | t×t | output words | `MV*` cold cache | `MV*` warm cache | NumPy products | + dense bincount | + `np.unique` |
|---|---|---|---|---|---|---|---|
| 8 | 300×300 | 48,890 | 0.116 s | 0.058 s | 0.0097 s | 0.0025 s | 0.0097 s |
| 8 | 1000×1000 | 65,536 | 1.723 s | 0.876 s | 0.055 s | 0.009 s | 0.061 s |
| 8 | 3000×3000 | 65,536 | 15.7 s | 17.9 s | 0.451 s | 0.101 s | 0.999 s |
| 12 | 1000×1000 | 970,900 | 2.13 s | 1.46 s | 0.045 s | 0.385 s | 0.080 s |
| 12 | 3000×3000 | 6,962,734 | 24.6 s | 24.5 s | 0.320 s | 0.481 s | 1.110 s |

- Speedups of the vectorised path over the current `MV.__mul__`: **5–31×** (products + dense accumulation, vs cold cache) and **3–17×** (products + sort-unique, vs warm cache). For the 9M-pair cases, the warm `lru_cache` was **no faster or slower than cold**. 9M distinct pairs exceed the 1M-entry memo, so the memo thrashes. The current dict loop runs at about 0.4–0.6 M pairs/s. The NumPy path runs at about 11–28 M pairs/s single-threaded, roughly a fifth to a tenth of PauLIB's multithreaded 123–145 M pairs/s on different hardware.

### Inferences
- Rough throughput ladder for a pair-product kernel, assembled from different hardware (so only orders of magnitude are meaningful): clifford_qc dict loop ~10^5.5–10^6 pairs/s; vectorised NumPy ~10^7; SIMD/OpenMP C++ (PauLIB) ~10^8; H100 GPU (PauLIB) ~10^10 terms/s. Each step is roughly one order of magnitude.
- For clifford_qc's word-universe sizes (7k–16k words at n=8, out of 4^8 = 65,536 possible words), the universe is 11–24% dense. A dense code-indexed accumulator (`np.bincount(codes, weights, minlength=4**n)`) is the cheapest dedup at n ≤ ~12 (4^12 ≈ 16.8M bins ≈ 134 MB per float64 plane). Sort/unique or hash dedup becomes necessary above that.
- The `lru_cache` memo is a liability once vectorised: it costs up to ~250 MiB at 12 qubits per the repo's own comment, and it thrashes when pairs exceed its size.
- Small-input regimes favour low-overhead implementations: PauliArray beats PauliEngine on small inputs, and PauliPropagation.jl beats PauLIB on small propagation. clifford_qc's typical operands, hundreds to a few thousand terms, are in that regime. A NumPy batch kernel without threading overhead is a good match.

### Gaps
- There is no independent, multi-library benchmark on common hardware. All cross-library numbers come from one library's authors, each on their own hardware.
- No published numbers were found for Qiskit `SparseObservable.compose`, PennyLane Lightning/Catalyst Pauli-sum arithmetic, symmer, Qulacs or Yao.
- fast-pauli's benchmark figures were not reachable: the README links to them without numbers.
- The local benchmark used random words. Chemistry-structured A_i†HA_j products will have more duplicate outputs, which favours the dense/unique accumulation even more, but this was not measured.

## 3. Algorithms for fast products of large Pauli sums and commutator-heavy workloads

### Takeaway
The standard toolkit has five parts:
- (i) XOR of x and z planes, with a mod-4 phase from a few popcounts;
- (ii) the symplectic inner product parity to decide (anti)commutation before doing any work;
- (iii) generate all pair products in one batched outer product, then merge once (a sort + adjacent-sum "collate", a code-indexed bincount, or bucketed hashing) rather than inserting into a hash map per product;
- (iv) keep sums in canonical sorted order so lookups are binary searches and merges are linear;
- (v) for propagation-style workloads, truncate by coefficient, weight or top-K after each merge.

### Cited Findings
- Phase via popcounts: PauliEngine computes `i^((|F+| − |F−|) mod 4)` from two Boolean masks, "one subtraction and two popcounts". For a commutator it reuses the same product: even τ means commute (result 0), odd τ means `[P,P′] = 2·P·P′`, with no second multiplication. — [PauliEngine](https://arxiv.org/html/2601.02233)
- PauliArray's composition phase is `(−i)^f` with `f = 2·x1·z2 + z1·x1 + z2·x2 − z3·x3 (mod 4)`. Commutation is tested by `z1·x2 + x1·z2 ≡ 0 (mod 2)`. For operator commutators it first builds the full commutation matrix c_ij by broadcasting, then composes only the non-commuting pairs. — [PauliArray](https://arxiv.org/html/2405.19287)
- PauliStrings.jl uses the product sign `(−1)^pop(v1∧w2)`. Its commutator phase is the difference of two popcount signs, and the README says `commutator(H1,H2)` is "much faster than" `H1*H2-H2*H1`. — [paper](https://arxiv.org/html/2410.09654v3); [GitHub](https://github.com/nicolasloizeau/PauliStrings.jl)
- PennyLane documents the same point: computing a commutator directly is faster than `P @ O − O @ P`. Its `structure_constants(..., matrix=True)` route is "sometimes faster … for sums of many Pauli words". — [PennyLane structure_constants](https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli.structure_constants.html)
- Qiskit `SparsePauliOp.compose` is fully broadcast:
  - `x3 = np.logical_xor(x1[:, None], x2).reshape(-1, n)`
  - `phase = np.add.outer(ph1, ph2) + 2*np.logical_and(z1[:, None], x2).sum(axis=1)`
  - `coeffs = np.multiply.outer(c1, c2).ravel()`

  `simplify` then packs bits (`np.packbits(x)·256 + np.packbits(z)`), dedups with the Rust `unordered_unique`, and sums with `np.add.at(coeffs, inverses, nz_coeffs)`. — [sparse_pauli_op.py](https://github.com/Qiskit/qiskit/blob/main/qiskit/quantum_info/operators/symplectic/sparse_pauli_op.py)
- PauLIB's merge ("collate") sorts an index array keyed by the packed words, then linearly sums adjacent duplicates. Above a size threshold it runs a chunked parallel sort plus stable pairwise merges, and the output is bit-identical across thread counts. Commutators are generated only for anticommuting pairs (GPU: one thread per pair, an atomic output counter, and on-device compaction). Truncation options: coefficient magnitude, weight (popcount), and top-M via `nth_element`, with "protected" strings. — [PauLIB](https://arxiv.org/html/2605.25974)
- SPD keeps the Pauli list sorted (binary search, O(n64·log N) per lookup). For each Pauli rotation it:
  1. finds Paulis anticommuting with σ;
  2. forms σ·P in a new array;
  3. looks up which already exist;
  4. updates the existing coefficients;
  5. deletes entries with |a|<δ and inserts new ones with |a|≥δ.

  Clifford parts of angles (kπ/2) are pushed through as Cliffords, leaving |θ′|<π/4. — [SPD](https://arxiv.org/html/2308.05077v3)
- SPD v0.2 adds "sparse Majorana dynamics", which exploits fermionic structure, and accompanies PRX Quantum 6, 020302 (2025). — [spd GitHub](https://github.com/tbegusic/spd)
- paulistrings-rs uses a GF(2)-linear hash `h(v)=H·v` of the (x,z) key to bucket and partition terms. Because the hash is linear, h(a⊕b) = h(a)⊕h(b), so the bucket of a product can be computed from the operand buckets. It accumulates unsorted terms with a `BuildAccumulator`. Truncation: thresholds, weight cutoffs, top-N, collapse sampling. — [paulistrings crate docs](https://docs.rs/paulistrings)
- PauliPropagation.jl merges identical strings before propagating them further ("merging-BFS"), which saves memory and time versus depth-first. Truncation options: `min_abs_coeff` (default 1e-10), `max_weight`, frequency, small-angle (sine) and path-weight, plus custom truncation functions. — [PauliPropagation.jl paper](https://arxiv.org/html/2505.21606v1)
- cuPauliProp exposes the same design as API flags: input `is_sorted` and `is_unique`, and output `make_sorted` and `keep_duplicates`. Output buffers must be preallocated because a Pauli rotation can at most double the term count. — [NVIDIA blog](https://developer.nvidia.com/blog/advanced-large-scale-quantum-simulation-techniques-in-cuquantum-sdk-v25-11/)
- Theory for truncation budgets: Shao, Cheng and Liu (arXiv:2510.22311, v3 June 2026) prove that Pauli-propagation truncation error is governed by the operator stabilizer Rényi entropy, and give a Top-K budget for a target accuracy. For the 1D Heisenberg model at J_z=0, they show the number of nonzero coefficients grows at most quadratically in the number of Trotter steps. — [arXiv:2510.22311](https://arxiv.org/abs/2510.22311)
- Exploiting structure: PauliStrings.jl's `OperatorTS1D` stores only strings starting on site 1 for translation-invariant operators (N× memory saving). — [PauliStrings.jl paper](https://arxiv.org/html/2410.09654v3)

### Inferences
- For clifford_qc's A_i†A_j and A_i†HA_j universes, the best-matched algorithm is a single global collate. Concatenate the code arrays of all pair products (chunked to bound memory), then run one `np.unique` (or a dense bincount at n ≤ 12) to obtain the union of words and the summed coefficients in one pass. This avoids building and merging hundreds of per-pair dicts.
- For commutator-heavy paths ([H, A], DLA-style closures), pre-filter pairs with the symplectic parity, `popcount(xa&zb ^ za&xb) & 1`, before forming products, as PauliArray and PauLIB do. For random pairs about half commute, so this roughly halves product work and removes cancellation noise.
- If the A_i are fermionic excitation operators (a few Majorana monomials each), products of Majorana monomials are also XOR on 2n-bit masks with a sign from a popcount-based parity. Working in the Majorana basis before the JW map could keep the expansion sparser. This is suggested by SPD's Majorana variant, but I have no measured gain for chemistry universes.
- Keeping each `MV`'s term arrays sorted by code turns `expectation(rho, O)` (a Pauli-basis trace pairing) into a sorted-array intersection (`np.intersect1d` or `np.searchsorted`) instead of Python dict lookups.

### Gaps
- No source quantified the speedup of sort-merge versus hashing in Python/NumPy specifically. PauLIB argues qualitatively (cache misses, SIMD, threaded merges), and the local run gives only sort versus dense-bincount, not hashing.
- No 2025–2026 source specific to fast Majorana-basis products for chemistry word universes was found in this session.

## 4. Options for accelerating a NumPy-first package without losing pure-Python installability

### Takeaway
Three tiers are available.
1. **Pure NumPy**: SoA integer codes, batched XOR/popcount, and sort-, unique- or bincount-based merge. This alone gives about 3–31× on clifford_qc's workloads in the local test, needs no new dependency, and needs NumPy ≥ 2 for `np.bitwise_count` (or a fallback).
2. **Optional compiled backend**: Rust/PyO3 is now the dominant choice in this ecosystem (Qiskit, pauli-prop, Zixy, paulistrings-rs, TenCirPauli, propaq). PauliEngine uses C++/nanobind and fast-pauli uses C++/OpenMP. Each adds wheel-building and toolchain cost.
3. **GPU (CuPy/JAX/CUDA)**: pays off only at about 10^5–10^8 output terms (PauLIB, cuPauliProp), far above clifford_qc's current sizes.

Reusing a permissively licensed backend (PauliEngine MIT, paulistrings-rs MIT/Apache-2.0, fast-pauli BSD-2, Qiskit Apache-2.0) as an optional extra is cheaper than writing one. PauliArray (CC-BY-SA-4.0) and SPD (GPL-3.0) should not be vendored into Apache-2.0 code.

### Cited Findings
- PauliArray shows a pure-NumPy library can beat Qiskit (about 6× on JW mapping) and OpenFermion (about 3×), and beats PauliEngine on small inputs, even without bit-packing (boolean arrays). Its install is `pip install .` or flit, with NumPy as the only named dependency. — [PauliArray paper](https://arxiv.org/html/2405.19287); [GitHub](https://github.com/algolab-quantique/pauliarray); [PauliEngine paper](https://arxiv.org/html/2601.02233)
- PauliArray's memory "grows sharply" and its runtime rises steeply once RAM saturates on large inputs. This is the cost of materialising broadcast outer products as unpacked booleans. — [PauliEngine paper](https://arxiv.org/html/2601.02233)
- PauliEngine packaging: prebuilt wheels for Linux (x86_64/aarch64), macOS arm64 and Windows (x86_64/arm64), Python 3.10–3.13. Building from source requires Conan 2, CMake ≥ 3.20, nanobind and a C++20 compiler (MSVC 19.3+, GCC 11+, Clang 14+). Without SymEngine, "the runtime cost of certain numeric paths increases". Licence MIT. — [PauliEngine GitHub](https://github.com/tequilahub/pauliengine)
- fast-pauli packaging: `pip install fast_pauli` binaries. Source builds need CMake ≥ 3.25, Ninja ≥ 1.11, scikit-build-core, a C++20 compiler with OpenMP, and LLVM/libc++ on macOS. The repo has 542 commits, 15 stars and 20 open issues. Licence BSD-2-Clause. — [fast-pauli GitHub](https://github.com/qognitive/fast-pauli)
- Qiskit uses Rust helpers called from NumPy code (e.g., `unordered_unique` for dedup, and Rayon/SIMD `fast_sum` for expectation values). — [sparse_pauli_op.py](https://github.com/Qiskit/qiskit/blob/main/qiskit/quantum_info/operators/symplectic/sparse_pauli_op.py); [pauli_exp_val.rs](https://github.com/Qiskit/qiskit/blob/main/crates/accelerate/src/pauli_exp_val.rs)
- Rust/PyO3 newcomers in 2025–2026:
  - Qiskit `pauli-prop`: Rust-accelerated, pip-installable, 0.2.0 — [PyPI](https://pypi.org/project/pauli-prop/)
  - Quantinuum Zixy: Rust with PyO3 bindings, 0.3.0 — [PyPI snippet](https://pypi.org/project/zixy/0.3.0/)
  - `paulistrings` crate 0.2.0 (3 Nov 2026): rayon, optional MPI and CUDA, Python package, MIT OR Apache-2.0 — [docs.rs](https://docs.rs/paulistrings)
  - TenCirPauli: Rust-native Pauli algebra for TensorCircuit-NG — [search summary of PyPI](https://pypi.org/project/tencirpauli/)
  - propaq: parallel Rust backend for Heisenberg-picture propagation — [search summary of PyPI](https://pypi.org/project/propaq)
- Parallelism returns are modest. PauLIB: 2.7× at 8 threads and 3.2× at 72 threads on Grace; 1.5× at 8 threads on M4 Pro; CPU throughput is memory-bandwidth-bound. PauliPropagation.jl reports diminishing returns or degradation with many threads. PauliStrings.jl is single-threaded. — [PauLIB](https://arxiv.org/html/2605.25974); [PauliPropagation.jl](https://arxiv.org/html/2505.21606v1); [PauliStrings.jl](https://arxiv.org/html/2410.09654v3)
- SIMD: PauLIB found hand-written intrinsics gave no significant speedup over compiler auto-vectorisation with OpenMP SIMD annotations. — [PauLIB](https://arxiv.org/html/2605.25974)
- GPU thresholds: PauLIB's H100 reaches 27.4 G terms/s only at 10^8 output terms; at 10^4×10^4 it is 187× faster than the CPU. cuPauliProp ships as CUDA-13 wheels (`cupauliprop-cu13`) with an experimental Python API. — [PauLIB](https://arxiv.org/html/2605.25974); [NVIDIA blog](https://developer.nvidia.com/blog/advanced-large-scale-quantum-simulation-techniques-in-cuquantum-sdk-v25-11/)
- Locally, `np.bitwise_count` is present in NumPy 2.5.3, and the full product kernel (XOR, 4 popcounts, phase lookup, outer coefficient product) ran at about 11–28 M pairs/s single-threaded. clifford_qc's floor is `numpy>=1.23`. — local run (see §2); [pyproject.toml](/home/user/clifford_qc/pyproject.toml)
- Licences of candidate dependencies:
  - Compatible with Apache-2.0 for optional use or vendoring: PauliEngine MIT; PauliStrings.jl MIT; paulistrings-rs MIT/Apache-2.0; fast-pauli BSD-2; Qiskit Apache-2.0.
  - PauliArray is CC-BY-SA-4.0, a non-software ShareAlike licence.
  - SPD is GPL-3.0.
  - PauLIB's licence is not named.

  — sources as cited in §1 table

### Inferences
Ranked kernel-improvement options for a two-author package (my synthesis):
1. **SoA NumPy core** (highest value per effort). Store `codes` as uint16/uint32/uint64 by n plus `coeffs` as complex128, kept sorted. Do sum products by chunked broadcast XOR + popcount phase, with dedup by dense `bincount` (n ≤ ~12) or `np.unique`/lexsort (larger n). Measured 3–31× locally, with no new dependency.
2. **Global collate for word universes and commutator pre-filtering** (§3). Cheap to add on top of (1).
3. **Drop or shrink the `lru_cache` word memo** once (1) exists. It saves up to about 250 MiB and avoids thrashing above 1M pairs.
4. **Optional Numba JIT** of the same kernels. This keeps pure-Python installability through a fallback, but I found no 2025–2026 Pauli-library source reporting Numba packaging or maintenance experience (see Gaps).
5. **Optional adapter to an existing compiled backend.** PauliEngine (MIT; wheels for the main platforms; OpenFermion-like API) is the most direct candidate. paulistrings-rs (MIT/Apache-2.0) is the newest Rust option but very young (0.2.0, Nov 2026). Writing one's own Rust/PyO3 extension is the highest-maintenance option, because of the per-platform wheel matrix.
6. **GPU (CuPy/JAX/cuPauliProp).** Not justified at 8–12 qubits and 10^4 words. Revisit only when output sizes reach about 10^7–10^8 terms.

For n > 32, multiword codes need a two-key sort (`np.lexsort`) or a void-dtype view for `np.unique`. The 2-bits-per-qubit Python-int path can remain as a reference oracle, which the repo already keeps as `word_mul_reference`.

### Gaps
- I found no source documenting the maintenance and packaging cost of Numba, JAX, CuPy or Array-API backends in a Pauli-algebra library. Such a source should cover CI matrix, wheel size, cold-start JIT latency, and Python-version lag. Recommendations about these remain unsourced judgment.
- The NumPy version that introduced `np.bitwise_count` was not verified this session; it is present in 2.5.3. Keeping `numpy>=1.23` support would need a fallback popcount, such as a byte lookup table.
- PauLIB's Python interface and licence are undocumented in the paper, so its suitability as a dependency is unknown.

## 5. Computing expectation values of many Pauli words on a state

### Takeaway
Methods and practical ceilings:
- **Statevector**: one O(2^n) bitmask pass per Pauli word (`index ^ x_mask` pairing plus a `popcount(index & z_mask)` sign). Qiskit threads this only from 19 qubits upward.
- **Stabilizer/tableau states**: scale to about 10^4 qubits (Stim) but only for Clifford states.
- **Pauli-basis (Heisenberg) propagation**: ⟨0|O|0⟩ is the sum of coefficients of Z-only words. This reaches 127 qubits for near-Clifford or low-depth circuits (SPD, cuPauliProp, PauliPropagation.jl), and 30–50 qubits for long-time many-body dynamics (PauliStrings.jl), subject to truncation.
- **Pauli-basis density operators** (clifford_qc's representation): ⟨O⟩ is a sparse dot product of coefficient vectors, so it is cheap once both are sorted arrays.

### Cited Findings
- Qiskit's Rust `expval_pauli_*`:
  - Diagonal words: sum of `|ψ_i|²` with sign `(i & z_mask).count_ones() & 1`.
  - Words with X: loop over 2^(n−1) pairs `index_1 = index_0 ^ x_mask`, building `index_0` by inserting a zero bit at the highest X position.
  - Density-matrix variants read the diagonal, or the entries `(i, i ^ x_mask)`.
  - `PARALLEL_THRESHOLD = 19` qubits for Rayon; below that, a SIMD `fast_sum` reduction (crate `pulp`).
  - Complexity O(2^n) or O(2^(n−1)) per Pauli.

  — [pauli_exp_val.rs](https://github.com/Qiskit/qiskit/blob/main/crates/accelerate/src/pauli_exp_val.rs)
- SPD computes ⟨0|O|0⟩ as the sum of coefficients of Paulis whose x-bits are all zero, and the operator Frobenius norm as the 2-norm of the coefficients. It simulated 127-qubit kicked-Ising circuits in about 10 s per point on one laptop core, up to 6 h for the most expensive converged points on 4 cores. — [SPD](https://arxiv.org/html/2308.05077v3)
- PauliPropagation.jl: `overlapwithzero(psum)` gives the all-zero-state expectation, and the paper notes overlaps with stabilizer states are efficiently computable. — [PauliPropagation.jl](https://arxiv.org/html/2505.21606v1)
- cuPauliProp computes expectation values as the trace with the zero state, after Heisenberg-picture application (gates in reverse, `adjoint=True`), on the 127-qubit IBM circuit. — [NVIDIA blog](https://developer.nvidia.com/blog/advanced-large-scale-quantum-simulation-techniques-in-cuquantum-sdk-v25-11/)
- PauliStrings.jl's limits are stated as "the inefficiency of representing non-trivial pure states (or other low-rank operators)" and the need for dissipation to probe long times. Its largest demonstrations are N=50 (XX with translation symmetry), N=40 (XX) and N=30 (XXZ). — [PauliStrings.jl](https://arxiv.org/html/2410.09654v3); [SciPost search summary](https://scipost.org/SciPostPhysCodeb.54)
- PauLIB's g-sim (Lie-algebraic simulation) for TFIM handles n=64 (dim g = 8,128): closure 39.5 ms, setup 50.2 ms, and a 200-layer circuit in 2.2 ms on an M4 Pro. It was validated against dense statevector simulation for n ≤ 4. — [PauLIB](https://arxiv.org/html/2605.25974)
- Stabilizer states: Stim handles about 20,000 qubits for surface-code circuits. QuantumClifford.jl reports a dense 1000-qubit Pauli measurement at about 70 μs (per an archived mirror quoted in the search summary; version and hardware unstated). — [Stim](https://arxiv.org/abs/2103.02202); [QuantumClifford.jl docs (search summary)](https://docs.juliahub.com/QuantumClifford/BsGZO/0.8.9)
- PauliArray provides expectation values and covariances for arrays of operators computed from per-string values, plus simultaneous diagonalisation of commuting sets by Clifford conjugation. — [PauliArray](https://arxiv.org/html/2405.19287)
- `SparseObservable.to_matrix()` scales as O(4^n), quoted as about 4 KB at 8 qubits and about 69 GB at 16 qubits. — [SparseObservable docs](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparseObservable)

### Inferences
- **Many words on one statevector**: all words with the same x-mask share the same pairing `i ↔ i ^ x_mask`. Their expectations over all z-masks are the Walsh–Hadamard transform of `conj(ψ[i ^ x]) ψ[i]`. Grouping the word universe by x-mask therefore costs one O(2^n) gather plus one O(n·2^n) fast WHT per distinct x-mask, instead of O(2^n) per word. At n=8 even the full 4^8 set is 256 WHTs of length 256. (This is standard Walsh–Hadamard reasoning; no 2025–2026 source verifying it was fetched this session.)
- For clifford_qc's Pauli-basis density operators, `expectation(rho, O)` is `Σ_w conj?(ρ_w)·O_w·2^n`-type pairing over shared words. With sorted code arrays this is an O(t log t) or O(t) intersection.
- Ceilings relevant to clifford_qc: a dense statevector or a dense code-indexed accumulator is comfortable to about 20–25 qubits (2^n amplitudes) or about 12–13 qubits (4^n Pauli coefficients). Beyond about 13 qubits a dense Pauli accumulator stops fitting, and sorted sparse arrays become mandatory.

### Gaps
- MPS-based Pauli-sum expectation values (e.g., quimb, ITensor MPO contraction) were not researched this session, so no ceiling numbers are given for them.
- PennyLane Lightning's and Qulacs' batched Pauli-sum expectation kernels were not inspected.
- No 2025–2026 primary source was fetched for the x-mask-grouped WHT batching trick.

## 6. Fast commutation-graph construction and grouping at scale

### Takeaway
At scale the bottleneck is the O(N²) pair check, not the colouring. The state of the art avoids storing the full graph. Picasso (IPDPS 2024) builds only small random-palette conflict subgraphs, uses popcount-based anticommutation checks, and coloured graphs of over 10^12 edges (2.1M Pauli strings, 24 qubits) in under 800 s on an A100. clifford_qc's grouping sizes (≤ 16k words) are 2–3 orders of magnitude smaller, so a chunked, bit-parallel NumPy anticommutation matrix plus greedy colouring is enough.

### Cited Findings
- Picasso:
  - Encodes Pauli letters as inverse one-hot 3-bit codes (X=110, Y=101, Z=011, I=000) and decides anticommutation by bitwise AND + popcount parity, reporting 1.4–2.0× CPU speedups from this encoding.
  - On the GPU, one thread checks each of the |V|(|V|−1)/2 pairs.
  - Each iteration assigns random colour lists of size α·log|V| from a palette 𝒫. Only vertices whose lists collide form the conflict graph, which is coloured by a dynamic greedy list-colouring.
  - Datasets: 18 hydrogen-chain molecules (H4–H10; STO-3G/6-31G/6-311G), 12–24 qubits, 8,721 to 2,101,361 Pauli terms, about 19M to about 1.12T complement edges.
  - Performance: largest graph in under 800 s on the GPU. The conflict-graph step is about 60× faster on GPU than CPU (geometric mean), and the whole run about 16×.
  - Memory on H4_2D_6-311G: ColPack 140.23 GB vs Picasso 2.06 GB (normal) or 57.12 GB (aggressive).
  - Colour quality on H6_3D_sto3g: Picasso 1,425.4 (normal) or 880.6 (aggressive) colours, vs ColPack 901–2,479 depending on ordering.
  - Hardware: 64-core AMD EPYC 7742, 512 GB RAM, A100 40 GB.

  — [Picasso arXiv:2401.06713v2](https://arxiv.org/html/2401.06713v2); [PNNL summary](https://www.pnnl.gov/publications/picasso-memory-efficient-graph-coloring-using-palettes-applications-quantum-computing)
- Picasso reports up to 68× memory savings, 49.5× more Pauli strings and 2,478× more edges than prior parallel approaches. Code: NTUDDSNLab/Picasso (CPU and CUDA ≥ 11); the licence is not stated in the paper. — [Picasso search summary](https://www.arxiv.org/abs/2401.06713)
- Qiskit `SparsePauliOp.group_commuting` delegates to `PauliList._commuting_groups`. `noncommutation_graph` returns a rustworkx `PyGraph`, with colouring done there. — [sparse_pauli_op.py](https://github.com/Qiskit/qiskit/blob/main/qiskit/quantum_info/operators/symplectic/sparse_pauli_op.py)
- PauliArray provides commuting-set partitioning by a minimum-clique-cover heuristic, plus simultaneous diagonalisation via Clifford conjugation. — [PauliArray](https://arxiv.org/html/2405.19287)
- For dense operators, which may involve all 4^n strings, Butt et al. give a partition of the full Pauli set into the minimal number of commuting families. The method scales linearly in the number of strings. — [arXiv:2311.08551 (search snippet)](https://arxiv.org/pdf/2311.08551)

### Inferences
- At clifford_qc scale (N ≤ 16k words at n=8), the full anticommutation matrix has 2.56×10^8 entries: 256 MB as bool, 32 MB bit-packed. It is computed in chunks as `popcount((xa[:,None] & zb) ^ (za[:,None] & xb)) & 1` on uint16 codes with `np.bitwise_count`. Greedy or DSATUR colouring can be done via rustworkx (Apache-2.0, already used by Qiskit) or NetworkX. Picasso-style palette methods matter only above about 10^6 words.
- Because the n=8 universe is 11–24% of all 4^8 words, the Butt et al. full-set partition (which is linear in the number of strings) may give near-optimal commuting families without any graph colouring. This was not tested.
- The Stim-based Clifford diagonalisation step that clifford_qc already uses is per group, not per pair, so it is unlikely to be the bottleneck at these sizes (not measured).

### Gaps
- I found no 2025–2026 follow-up to Picasso, such as a distributed or multi-GPU version, in this session. Other GPU colouring libraries (Kokkos-EB, ECL-GC-R) appear only as Picasso baselines.
- No benchmark was found for qubit-wise-commuting versus fully-commuting grouping throughput in Python libraries at 10^4–10^5 words.

## 7. PauliEngine in particular, and how others compare with it

### Takeaway
PauliEngine (Müller, Bärligea, Knapp, Kottmann; arXiv:2601.02233; SE 2026 workshop, DOI 10.18420/se2026-ws_20) is an MIT-licensed, header-only C++20 library with nanobind bindings. It uses binary-symplectic bitwise arithmetic, popcount phases and SymEngine symbolic coefficients behind an OpenFermion-like API, and ships wheels for the major platforms. It beats OpenFermion and PennyLane by 1–2 orders of magnitude. Pure-NumPy PauliArray beats it on small inputs, and the SIMD/SoA batch library PauLIB beats it by 33–49× on 500-qubit outer products, attributing the gap to PauliEngine's one-string-per-call design. For clifford_qc, the main lesson is "batch the pairs", not "switch to C++".

### Cited Findings
- Design:
  - (x,y) symplectic bits; product = XOR; phase `i^((|F+|−|F−|) mod 4)` from two popcounts.
  - Commutator from a single product, using the parity of τ.
  - Constant-time dictionary lookup for DLA closure.
  - C++ backend, nanobind (named in the acknowledgments), SymEngine for symbolic coefficients with differentiation and substitution.
  - The paper does not discuss multithreading, SIMD or a 64-qubit word limit.

  — [PauliEngine paper](https://arxiv.org/html/2601.02233)
- README:
  - MIT licence; "Functional core with a large test suite. The Python API surface is stable."
  - `pip install pauliengine` wheels for Linux x86_64/aarch64, macOS arm64 and Windows x86_64/arm64, Python 3.10–3.13.
  - Operations: multiplication, commutators, addition, powers, dagger/conjugate/transpose, simplify, split, map_qubits, to_matrix, Lie closure, OpenFermion conversion, Hermiticity checks, measurement counting.
  - Claimed complexity "O(n/64)" for multiplication, commutators and hashing.
  - Expectation values and Clifford conjugation are not mentioned.

  — [PauliEngine GitHub](https://github.com/tequilahub/pauliengine)
- Reported results:
  - 1–2 orders of magnitude over OpenFermion on Hamiltonian products (i9-11900KF).
  - About 44× to about 160× over PennyLane on structure constants (N=3→10, Apple M4).
  - PauliArray is faster for small strings or Hamiltonians (up to about 250 qubits), but PauliEngine wins once PauliArray's memory saturates.
  - Qiskit was omitted "because PauliArray was previously benchmarked against it".
  - The conclusion's claim that PauliEngine "consistently outperforms" PauliArray conflicts with its own Fig. 5, which shows PauliArray faster on small instances.

  — [PauliEngine paper](https://arxiv.org/html/2601.02233)
- Independent comparison, from a competing library's authors: at 500 qubits PauLIB's outer product is 3.4× (N=25) to 49× (N=200) faster than PauliEngine. PauLIB says PauliEngine has "the same bitwise arithmetic, but one string per call, so it pays call and cache-miss costs per string", and that this, not a faster word kernel, is the main difference. — [PauLIB](https://arxiv.org/html/2605.25974)
- clifford_qc's architecture paper already cites PauliEngine as an operator-level comparator. It explicitly makes no speed claim between the packages, and notes that using PauliEngine as a backend would need an adapter covering qubit ordering, phase conventions, coefficient and tolerance semantics, and conversion cost. — [paper_architecture/manuscript.tex](/home/user/clifford_qc/paper_architecture/manuscript.tex)

### Inferences
- An optional PauliEngine backend would likely beat clifford_qc's current dict loop on large products, given 1–2 orders over OpenFermion-style dicts. But PauliArray-style NumPy batching already beats PauliEngine at clifford_qc's sizes (≤ 12 qubits, 10^2–10^4 terms), and the local NumPy test gives 3–31× with no dependency. So a NumPy batch kernel should come first, and a PauliEngine adapter is a secondary, optional path, mainly attractive for its symbolic coefficients.
- Conversion cost matters for an adapter. clifford_qc's interleaved I=0, X=1, Y=2, Z=3 lanes map to PauliEngine's (x,y) planes with a few masks and shifts per word. Doing that per call in Python would erode the gain unless done as a batched NumPy transform.

### Gaps
- No exact crossover sizes between PauliEngine and PauliArray are given beyond the "about 250 qubits" regime, and there are no absolute timings in the text (figures only).
- The PauliEngine release history and changelog were not visible: GitHub Releases is empty.
- No source benchmarks PauliEngine against Qiskit `SparseObservable`, PauliStrings.jl, or Rust newcomers (Zixy, paulistrings-rs).
