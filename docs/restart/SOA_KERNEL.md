# Structure-of-Arrays (SoA) Pauli algebra kernel (Rank 5a)

This implements the Rank 5a vectorized Pauli-operator kernel and global bank
collation step of the [pencil improvement sequence](PENCIL_ESTIMATORS.md#validation-and-remaining-sequence).
It is opt-in and preserves every frozen `MV` / `pauli_kernel` / `GlobalWordTable`
byte-identity and storage contract unchanged.

## Design and invariants

All primitives live in [`clifford_qc/soa_kernel.py`](../../clifford_qc/soa_kernel.py)
with NumPy as the only required dependency:

1. **Width-selected and multiword codes.** `PauliSoA` stores canonically sorted,
   deduplicated codes (`uint16` for $n \le 8$, `uint32` for $9 \le n \le 16$,
   `uint64` for $17 \le n \le 32$, and 2-D little-endian `uint64` limbs of
   shape `(t, ceil(n / 32))` for $n > 32$) alongside `complex128` coefficients.
   Coefficients follow `MV`'s absolute pruning cutoff (`|c| > 1e-12`) and
   canonicalize `-0.0` to `+0.0`.
2. **Batched XOR + popcount products without process memo.** `soa_mul` computes
   chunked outer products using the exact 2-bit-lane phase formula of
   `word_mul_reference`, using `np.bitwise_count` on NumPy $\ge 2.0$ with a
   256-byte lookup fallback for older NumPy builds. Accumulation switches
   automatically between dense `np.bincount` (for small $n \le 12$ at high pair
   counts) and sorted `np.add.reduceat` (for sparse, large-$n$, and multiword
   operands). The process-wide `lru_cache` memo in `pauli_kernel.py` is never
   populated.
3. **Symplectic commutator pre-filter.** `soa_commutator` evaluates
   `popcount((xa & zb) ^ (za & xb)) & 1` first (using limb-wise XOR before a
   single popcount for multiword $n > 32$) and multiplies only anticommuting
   pairs (`[P, Q] = 2 P Q`).
4. **Sorted-array intersections & determinant evaluation.** `soa_trace_pairing`
   and `soa_hs_product` intersect sorted code arrays with `np.searchsorted`;
   `soa_determinant_expectation` evaluates computational-basis expectations in
   $O(\mathrm{nnz})$ vectorized bit operations and accepts `DeterminantReference`,
   `q0`-leftmost bitstrings, or integer masks.
5. **Global pair collation & 31-qubit ceiling removal.** `build_soa_bank`
   precomputes $H A_j$ once per generator, builds $A_i^\dagger A_j$ and
   $A_i^\dagger H A_j$ directly in SoA, and collates all pair products in one
   pass into a multiword-capable `SoAWordTable` + CSR-packed `GlobalSoABank`.
   `GlobalWordTable` in `clifford_qc/subspace/packed.py` now explicitly raises
   `OverflowError` for codes beyond signed `int64` ($n > 31$) and points
   callers to `SoAWordTable`.
6. **Optional CUDA GPU backends.** `backend="cupy"` uses fused CUDA
   `ElementwiseKernel` routines with hardware `__popcll` (supporting both
   single-word $n \le 32$ and multiword $n > 32$ via `cp.lexsort`);
   `backend="torch"` uses CUDA `int64` SWAR popcount + `scatter_add_` for
   $n \le 31$. Both are imported lazily only when requested and remain optional.

## Exploratory CPU throughput measurements

Measured single-threaded (`OMP_NUM_THREADS=1`, Python 3.12.16, NumPy 2.5.5,
`bitwise_count=True`, 2-core x86\_64 VM) via
`python benchmarks/benchmark_soa_kernel.py --quick --repeats 3 --warmup 1`:

| Workload | Legacy `MV` / Bank | NumPy `PauliSoA` | Speedup | Memo entries (Legacy / SoA) |
| --- | ---: | ---: | ---: | ---: |
| Random $n=8, t=300$ (90k pairs) | 0.385 s cold / 0.163 s warm | 0.0040 s (dense) | 95.8× / 40.5× | 90,000 / 0 |
| Random $n=8, t=1000$ (1M pairs) | 3.184 s cold / 1.514 s warm | 0.0314 s (dense) | 101.3× / 48.2× | 1,000,000 / 0 |
| Random $n=12, t=1000$ (1M pairs) | 7.502 s cold / 6.496 s warm | 0.2798 s (sort) | 26.8× / 23.2× | 1,000,000 / 0 |
| $\text{H}_4$ 8q $H^2$ (185 terms) | 0.187 s cold / 0.069 s warm | 0.0026 s | 72.0× / 26.6× | 34,225 / 0 |
| $\text{LiH}$ 8q $H^2$ (193 terms) | 0.118 s cold / 0.034 s warm | 0.0014 s | 82.9× / 23.6× | 37,249 / 0 |
| $\text{H}_2\text{O}$ 12q $H^2$ (551 terms) | 1.444 s cold / 0.858 s warm | 0.0569 s | 25.4× / 15.1× | 303,601 / 0 |
| $\text{H}_4$ 8q bank ($m=10, W=15{,}502$) | 2.895 s (`MatrixElementBank`) | 0.1571 s (`build_soa_bank`, 20.3 B/coeff) | 18.4× | 403,257 / 0 |
| $\text{LiH}$ 8q bank ($m=10, W=15{,}578$) | 2.880 s (`MatrixElementBank`) | 0.1342 s (`build_soa_bank`, 20.3 B/coeff) | 21.5× | 450,367 / 0 |
| $\text{BeH}_2$ 8q bank ($m=10, W=13{,}226$) | 0.951 s (`MatrixElementBank`) | 0.0489 s (`build_soa_bank`, 20.7 B/coeff) | 19.4× | 160,033 / 0 |
| $\text{H}_2\text{O}$ 12q bank ($m=6, W=43{,}044$) | 1.069 s (`MatrixElementBank`) | 0.1029 s (`build_soa_bank`, 22.1 B/coeff) | 10.4× | 121,305 / 0 |
| Commutator $\text{H}_2\text{O}$ 12q `[H, pool8]` | 0.0059 s (naive SoA `AB-BA`) | 0.0009 s (pre-filter) | 6.30× | 0 / 0 |
| Multiword $n=40, t=400$ (160k pairs) | 1.354 s (`MV` cold) | 0.1328 s | 10.2× | — / 0 |
| Multiword $n=64, t=400$ (160k pairs) | 1.381 s (`MV` cold) | 0.1676 s | 8.2× | — / 0 |

Every cell passed term-for-term oracle parity (`max_abs_diff < 1e-8`) and exact
`(S, H)` matrix agreement (`max_diff < 1e-10`) against `MV` and
`MatrixElementBank`. These timings are exploratory machine measurements, not
hardware-independent scientific claims.

## Kaggle Tesla T4 GPU parity and CPU-vs-GPU crossover

Verified on Kaggle (`2× NVIDIA Tesla T4`, 15,360 MiB VRAM, CUDA 13.0, Python
3.13.16, NumPy 2.1.3, CuPy 14.0.1, PyTorch 2.11.0+cu130, 4 CPU cores) at commit
`5136e89`:

- **Test suite:** `pytest --noconftest -v tests/test_soa_kernel.py` — **83/83
  passed** (including `test_gpu_backends_match_numpy_when_available` covering
  CuPy single-word dense/sort, CuPy multiword `n=40`/`n=64`, and PyTorch CUDA
  single-word dense/sort against NumPy and `MV`).
- **Full sweep (`python benchmarks/benchmark_soa_kernel.py --backends all --gpu-sweep --repeats 5 --warmup 2`):**

| Workload | Legacy `MV` / Bank | NumPy CPU | CuPy T4 GPU | PyTorch T4 GPU | Best GPU vs. NumPy / `MV` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random $n=8, t=300$ (90k pairs) | 0.202 s cold | **0.0031 s** (29.0 Mpairs/s) | 0.0042 s | 0.0046 s | 0.7× (CPU wins) / 48.2× |
| Random $n=8, t=1000$ (1M pairs) | 2.050 s cold | 0.0359 s (27.8 Mpairs/s) | **0.0068 s** (147.3 Mpairs/s) | 0.0094 s | **5.3×** / **301.9×** |
| Random $n=8, t=3000$ (9M pairs) | 18.066 s cold | 0.2336 s (38.5 Mpairs/s) | **0.0311 s** (289.7 Mpairs/s) | 0.0420 s | **7.5×** / **581.5×** |
| Random $n=12, t=1000$ (1M pairs) | 2.986 s cold | 0.1385 s (sort) | **0.0302 s** (33.2 Mpairs/s) | 0.0329 s | **4.6×** / **99.0×** |
| Random $n=12, t=3000$ (9M pairs) | 29.015 s cold | 1.5993 s (sort) | **0.2551 s** (35.3 Mpairs/s) | 0.3323 s | **6.3×** / **113.7×** |
| Random $n=12, t=6000$ (36M pairs) | skipped | 10.791 s (dense) | **0.7841 s** (45.9 Mpairs/s) | 1.0254 s | **13.8×** / — |
| Random $n=12, t=10000$ (100M pairs) | skipped | 27.675 s (dense) | **1.1969 s** (83.6 Mpairs/s) | 1.5685 s | **23.1×** / — |
| Random $n=16, t=2000$ (4M pairs) | 15.063 s cold | 0.7252 s (sort) | **0.1338 s** (29.9 Mpairs/s) | 0.1487 s | **5.4×** / **112.6×** |
| Random $n=16, t=4000$ (16M pairs) | skipped | 3.3750 s (sort) | **0.7968 s** (20.1 Mpairs/s) | 0.9982 s | **4.2×** / — |
| Random $n=16, t=6000$ (36M pairs) | skipped | 7.2534 s (sort) | **1.7984 s** (20.0 Mpairs/s) | 2.2475 s | **4.0×** / — |
| $\text{H}_4$ / $\text{LiH}$ / $\text{BeH}_2$ 8q $H^2$ | 0.050 / 0.053 / 0.007 s | **0.0014 / 0.0013 / 0.0005 s** | 0.0033 / 0.0034 / 0.0042 s | 0.0038 / 0.0036 / 0.0041 s | CPU wins (<40k pairs) |
| $\text{H}_2\text{O}$ 12q $H^2$ (304k pairs) | 0.540 s cold | 0.0159 s (19.1 Mpairs/s) | **0.0059 s** (51.5 Mpairs/s) | **0.0059 s** (51.8 Mpairs/s) | **2.7×** / **92.1×** |
| Spin-adapted banks ($m=10$, $\text{H}_4$–$\text{H}_2\text{O}$) | 0.317–1.307 s | **0.033–0.081 s** (9.6–16.1×) | 0.498–0.543 s | 0.507–0.543 s | CPU wins (many small pairs) |
| Commutator $n=16, t=1500$ (2.25M pairs) | — | 0.2220 s (5.20× pre-filter) | **0.0363 s** (15.11× pre-filter) | 0.0422 s (12.86×) | **6.1×** vs. NumPy comm |
| Multiword $n=40, t=800$ (640k pairs) | skipped | 0.3322 s (1.9 Mpairs/s) | **0.0318 s** (20.1 Mpairs/s) | — | **10.4×** / — |
| Multiword $n=64, t=800$ (640k pairs) | skipped | 0.3300 s (1.9 Mpairs/s) | **0.0307 s** (20.8 Mpairs/s) | — | **10.7×** / — |
| Multiword $n=64, t=1500$ (2.25M pairs) | skipped | 1.6646 s (1.4 Mpairs/s) | **0.1419 s** (15.9 Mpairs/s) | — | **11.7×** / — |

**Empirical CPU-vs-GPU crossover:** On Tesla T4, per-call host-to-device and
kernel launch overhead is ~3–4 ms, so pure NumPy CPU wins on small chemistry
products and per-pair bank assembly (`< 10^5` pairs per call, where NumPy
already achieves 9.6–41.6× over legacy `MV`). Above ~`2.5 × 10^5` pairs per call
(matching the `backend="auto"` threshold of `250,000` pairs), the fused CuPy
`__popcll` kernel pulls decisively ahead—reaching 2.7× over NumPy on $\text{H}_2\text{O}$
12q $H^2$ (92× vs. `MV`), 10–11.7× on 40q/64q multiword products, and 23.1× on
100M-pair 12q dense outer products (289.7 Mpairs/s peak on 8q dense, 581.5× vs.
legacy `MV`).

