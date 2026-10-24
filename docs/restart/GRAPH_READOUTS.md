# Opt-in graph readouts and signed stabilizer references

The restart now has a NumPy-only commuting-readout compiler and a pure
stabilizer reference evaluator. These are support for Steps 2 and 5. The
independent pilot/evaluation ratio experiment and its sampled consumer remain
unfinished. Historical evidence, digests, ancestry, default elimination
circuits and negative campaign conclusions are unchanged.

## Use the compiler

```python
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.planning import compile_block_measurement_plan

codes = [PauliWord.from_label(label).code for label in ("XX", "YY", "ZZ")]
plan = compile_block_measurement_plan(2, codes, 2, method="graph_state")
assert plan.method == "graph_state"
print(plan.synthesis.gate_counts)
print(plan.synthesis.logical_two_qubit_per_sweep)
```

`groups=` can supply a frozen partition; it is checked for complete exact-once
assignment and block-wise commutation. Omitting `method` retains
`method="elimination"`, including its optional Stim requirement and historical
resource fields. `synthesize_block_settings(..., method="graph_state")` exposes
the graph resource ledger directly. Both ledgers provide
`logical_two_qubit_per_sweep`, `two_qubit_per_setting` and
`two_qubit_depth_per_setting`; inspect `gate_counts` to distinguish CX from CZ.
No graph count is mislabeled as a CX count.

The graph plan uses the existing `CompiledMeasurementSampler` and
`GroupedWordCache`. It has immutable native H/S/SDG/CZ circuits and signed
readouts, and prices those same circuits. Graph and elimination cache keys
are distinct; graph keys include the actual circuit fingerprint. Core graph
compilation and execution do not import Stim. A fresh subprocess test blocks
all Stim imports, compiles a plan and samples its joint readouts. Exact test
sampling still uses the existing small-system density simulator; the new
compiler does not make that simulator scalable or supply a hardware backend.

## Algebra and scheduling contract

The object is a commuting Pauli subgroup on a block of `k <= 32` qubits, with
rank `r <= k`, represented by an isotropic GF(2) matrix `[X | Z]`. Pauli-code
lane zero is qubit zero and the leftmost occupation character. Row operations
track the unsigned subgroup for synthesis; exact signed Pauli conjugation
separately determines every decoded observable.

After X-row reduction and local pre-Hadamards that complete X rank, write
the rows in pivot/nonpivot order as `[I, A | B, C]`. Isotropicity gives
`B + A C^T = (B + A C^T)^T`, where addition is XOR. Therefore

```math
R = \begin{pmatrix}B + AC^T & C\\ C^T & 0\end{pmatrix},
\qquad [I,A]R=[B,C].
```

Diagonal entries of R prescribe local S gates and off-diagonal entries
prescribe CZs. They cancel Z components on active X columns; a final H on
those columns produces Z-only readouts. Initially Z-only spectators need no
final H, so their remaining Z parity need not be cleared or entangled.
Qubit-wise commuting families take a local-only fast path; a single YY word
must not acquire a gratuitous CZ. No CZ joins two nonpivot vertices.

This directly proves subgroup diagonalization. The theorem that every pure
stabilizer **state** is local-Clifford equivalent to a graph state does not
assert that every diagonalizing **unitary** belongs to that state orbit.

The CZ interaction graph is finite, simple and undirected. Mutually commuting
CZs are one algebraic stage, which can require several physical layers.
The implementation restricts a round-robin factorization of the complete graph
to the required edges: at most `k-1` disjoint two-qubit layers for even k, or
`k` for odd k using a dummy vertex. A triangle needs three layers, so a general
`k-1` claim for odd k would be false. This is a schedulable bound under
all-to-all connectivity, not optimal coloring or a hardware routing claim.

An eight-word unsplit JW double excitation on modes `{0,3,4,7}` is tested
explicitly. It compiles to a three-edge star with no entanglers on the
intermediate Z string. Its adjacency matrix has GF(2) rank **2**, and its
three incident CZs require **3** ordinary CZ layers. X-mask rank one does
not imply adjacency rank one or a depth-two implementation in this gate model.

## Rebuilt finite-bank comparison

The [machine-readable audit](graph_readout_comparison.json) reconstructs the
committed H4 (7,371 words) and BeH2 (1,815 words) banks and identical dyadic
partitions. Before comparing, it reproduces each frozen v2 elimination
setting count, total CX count and sum of two-qubit critical depths exactly.
It checks every assigned graph readout with signed conjugation. Independent
MV-unitary tests check the native phase maps, and a nonstabilizer control
compares assigned joint probability distributions between both compilers.

| System, k | Settings | Elimination CX | Graph CZ | Gate reduction | Elimination depth mean / max | Graph depth mean / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H4, 2 | 647 | 2,803 | 904 | 67.7% | 2.25 / 4 | 0.78 / 1 |
| H4, 4 | 238 | 3,277 | 1,057 | 67.7% | 7.91 / 14 | 2.21 / 3 |
| H4, 8 | 64 | 1,886 | 460 | 75.6% | 24.16 / 35 | 4.17 / 7 |
| BEH2, 2 | 41 | 112 | 112 | 0.0% | 0.98 / 1 | 0.98 / 1 |
| BEH2, 4 | 26 | 193 | 118 | 38.9% | 4.46 / 10 | 2.08 / 3 |
| BEH2, 8 | 14 | 296 | 118 | 60.1% | 17.86 / 23 | 5.50 / 7 |

Depth here is the logical two-qubit critical depth, ignoring intervening
one-qubit gates. Both backends use one shared scheduling helper, applied per
block. Independent blocks run in parallel: each setting takes the maximum
one-qubit and two-qubit layer counts across blocks, preserving the historical
elimination convention. Local cancellation preserves the surviving gates'
stage order, so it cannot split a disjoint CZ matching by delaying local gates.
The JSON separately reports mean scheduled one-/two-qubit layers for the
duration surrogate. These six rows are finite exact
compilation results, not a general optimality theorem or a sampled experiment.
The producer records source/package identities and frozen input digests;
`--check` compares bank/card digests, compiled counts, readout/compatibility summaries and costs
strictly. Package/source fingerprints are advisory execution provenance: a
mismatch prints a warning, including when only the NumPy version or unrelated
package code changes. They do not make a successful numerical replay fail.
This is an explicit audit replay, not a CI guard tying live code to a
historical live-source digest.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/compare_graph_readout.py \
  --check docs/restart/graph_readout_comparison.json
```

The existing illustrative superconducting-like card admits 66 additional
H4 k=4 settings, 61 H4 k=8 settings and 12 BeH2 k=8 settings. Its other rows
are already fully admitted. The ion-like and logical-alltoall cards already
admit every elimination setting, so they have no newly admitted settings.
None of the cards loses an admitted setting in this audit. These calculations
apply the same generic two-qubit error/time and routing multipliers to CX and
CZ. They are scenario surrogates, not calibrated native-gate or routed-device
measurements.

Assigned joint observables are preserved, but incidental pooled readouts can
change with the diagonalizer. H4 k=2 gains 825 and loses 669 readable
setting/word pairs; H4 k=4 gains 121 and loses 333. The other four rows retain
their compatibility matrices. Gate savings alone therefore establish neither
shot savings nor an end-to-end advantage at matched accuracy. A new sampling
comparison must use each compiler's actual compatibility and covariance.

## Signed stabilizer reference contract

```python
from clifford_qc.reference_states import StabilizerReference

reference = StabilizerReference(2, [("XX", 1), ("ZZ", 1)])
assert reference.word_expectation(PauliWord.from_label("YY").code) == -1
determinant = StabilizerReference.from_determinant("01")
```

The constructor requires `n` independent commuting Hermitian generators with
integer eigenvalues ±1. It accepts labels, packed codes or PauliWord objects,
canonicalizes the signed row space and owns immutable rows. Its physical
identity is independent of generator basis and includes the declared spin
ordering. Both constructors default to `spin_ordering="interleaved"`, matching
`DeterminantReference`; explicitly choose `"qubits"` for that convention.
Dependent, inconsistent, noncommuting and underconstrained inputs
are rejected. Determinants are a special case. Signed subgroup membership
returns exact Pauli expectations in `{0, ±1}` without `2**n` density expansion.
`ExactProjectedProvider` consumes this exact-expectation capability directly.

`storage()` reports a separate ideal `n*(2*n+1)`-bit tableau model and the
measured canonical Python tuple/int footprint. The latter excludes metadata
and neither quantity is process RSS. Construction and expectation evaluation
use uncached packed products, so they do not grow or evict the MV product LRU.
This implementation supplies neither amplitudes, a preparation circuit,
sector-action capability nor a hardware sampler. Analytic expectation support
does not by itself integrate stabilizer warm starts into the molecular solver.

## Fermionic symmetry boundary

Stabilizer references carry no implicit particle-number or spin promise.
Tests use interleaved `(alpha, beta)` spin-orbital modes and verify moments of
N, Sz and S², including variance rather than only a mean.

| Tested state | Exact moments | Consequence |
| --- | --- | --- |
| Two-mode `(10 - 01)/sqrt(2)` across alpha/beta | N=1, Sz mean 0, Sz²=1/4, S²=3/4 | A qubit Bell singlet across orbital modes is a one-electron spin superposition, not a physical spin singlet |
| Product of same-spin mode-Bell pairs on four modes | N=2, Sz=0 with zero variance; S²=1, S⁴=2 | Number and spin-projection locking alone does not establish total-spin purity |
| Four-mode `(1001 - 0110)/sqrt(2)` | N=2, Sz=0, S²=0, all corresponding variances zero | A physical singlet stabilizer reference exists beyond a product of two-mode Bell pairs |
| Canonical two-vertex graph state | Var(N)=1/2, ZZ parity mean 0 | Canonical graph preparation does not generally stay in a JW sector |

The four-mode singlet is specified by signed generators
`ZZII=-1, IIZZ=-1, ZIZI=-1, XXXX=-1`. It is a GHZ-type state with two
determinants; a product of two nontrivial two-mode Bell pairs has four.
Thus an “only pair-locked dimers conserve (N,Sz)” restriction is too strong,
and arbitrary mode-Bell pairs or inter-pair CZs cannot be advertised as
spin-pure valence-bond preparations. Clifford-angle single-word excitation
rotors likewise require checks on the chosen reference and total spin.
No state-preparation advantage or post-selection guarantee is adopted here.

## Primary references and validation

Van den Nest, Dehaene and De Moor's graph-state LC construction is
[PRA 69, 022316 (2004)](https://doi.org/10.1103/PhysRevA.69.022316),
[arXiv:quant-ph/0308151](https://arxiv.org/abs/quant-ph/0308151).
Their [PRA 70, 034302](https://doi.org/10.1103/PhysRevA.70.034302) is the
distinct efficient LC-equivalence recognition paper.
[arXiv:2302.06537](https://arxiv.org/abs/2302.06537) is by Willers Yang and
Patrick Rall, *Harnessing the Power of Long-Range Entanglement for Clifford
Circuit Synthesis*. Its GHZ-bus injection model does not establish ordinary
CZ-layer depths for this implementation.

```bash
python -m pytest tests/test_graph_state_readout.py tests/test_stabilizer_reference.py \
  tests/test_graph_state_review.py
```

The focused suite covers ranks zero through full rank, dependent subgroup
members, every small-span word, signed phases, 32-qubit packing boundaries,
odd/even matching schedules, Z spectators, QWC fast paths, independent joint
probabilities, cache isolation, reference ownership and large-register
evaluation without density allocation. Existing default/compiler, cost,
provider and sector-action regressions remain separate checks.
