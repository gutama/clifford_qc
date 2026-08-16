# clifford_qc

`clifford_qc` is an operator-centric quantum computing package built on the
complexified Clifford algebra

\[
Cl(2n,\mathbb C) \cong M(2^n,\mathbb C).
\]

The core representation is a sparse multivector/operator (`MV`) in the
Pauli-word basis. States, gates, observables, channels, Jordan-Wigner
Clifford generators, and fermionic creation/annihilation operators all live in
the same algebra, so workflows can move between circuit, operator, density
matrix, and fermionic views without changing representations.

This is not a speedup claim over dense matrices. The goal is structural: one
small algebraic engine with exact sparse Pauli semantics, dense matrix
conversion for validation and small systems, and optional bridges to external
quantum tooling.

## Development access status

`clifford_qc` is currently a private, pre-release research framework. The
operator IR, measurement contracts, subspace APIs, and backend boundaries are
still being consolidated into a stable foundation, so no public package or
public-repository availability is claimed at this stage. Collaborators and
scientific reviewers should work from an explicitly identified frozen source
snapshot rather than assume that the moving development branch is an archival
release.

The benchmark inputs, raw records, generators, semantic drift gates, and tests
remain versioned together. A tagged archival release with a persistent
identifier is intended once the framework interfaces and evidence contracts
are stable enough for external reuse.

## Install

For collaborators with access to a checked-out development snapshot:

```bash
pip install -e .[test]
pytest
```

Install optional research and chemistry dependencies only when reproducing the
corresponding paper artifacts:

```bash
pip install -e .[test,research,chemistry]
```

Optional ecosystem bridges are installed independently:

```bash
pip install -e .[stim]
pip install -e .[openfermion]
pip install -e .[pytket]
pip install -e .[pennylane]
pip install -e .[pyzx]
pip install -e .[test,bridges]
```

The core package depends only on `numpy`. Bridge modules import their
third-party dependencies only when those modules are imported.

## Quick Start

```python
from clifford_qc import *

rho = bell_density()

print(rho.to_labels())
print("negativity:", negativity(rho, {1}))
print("probabilities:", computational_probabilities(rho))
```

Build and evolve states directly:

```python
from clifford_qc import CNOT, H, X, Z, evolve, expectation, ket_density

rho0 = ket_density(2, "00")
U = CNOT(2, 0, 1) * H(2, 0)
rho = evolve(rho0, U)

print(expectation(rho, X(2, 0) * X(2, 1)).real)
print(expectation(rho, Z(2, 0) * Z(2, 1)).real)
```

Use the dense matrix bridge when a small-system reference is useful:

```python
import numpy as np
from clifford_qc import P, to_matrix

A = P("XI") + 0.25 * P("ZZ")
assert np.allclose(to_matrix(A * A), to_matrix(A) @ to_matrix(A))
```

## What Is Included

`clifford_qc` covers:

- sparse Pauli-word operators through `MV`
- Pauli helpers: `I`, `X`, `Y`, `Z`, `P`, commutators, anticommutators, tensor products
- Clifford/Jordan-Wigner generators: `gamma`, pseudoscalar, blades, grades
- fermionic operators: `c_op`, `cdag_op`, `number_op`
- gates and unitaries: `H`, `S`, `T`, `RX`, `RY`, `RZ`, `rotor`, `CNOT`, `CZ`, `SWAP`, `TOFFOLI`
- density operators, evolution, measurement, probabilities, partial trace, partial transpose
- Kraus channels: depolarizing, dephasing, amplitude damping
- diagnostics: fidelity, entropy, negativity, trace checks
- dense matrix conversion and exact small-system ground states, a sparse
  reference tier (`sparse.py`) with sector-restricted diagonalization for `n`
  past dense `eigh`, and a sector-restricted statevector backend
  (`backends/sector_statevector.py`) that stores `C(n,k)` amplitudes and never
  builds a matrix at all
- a Pauli-rotor intermediate representation with exact gate-by-gate execution, versioned JSON serialization, gradients, and QASM3 export
- optional bridges for Stim, OpenFermion, pytket, PennyLane, and PyZX
- materials clusters (`models/lattice.py`): Hubbard, extended Hubbard,
  Kanamori, Anderson impurity, Kitaev honeycomb — built from the package's own
  Jordan-Wigner operators — with observables (`models/observables.py`) for
  occupations, double occupancy, spin correlations, and structure factors
- orbital bases (`models/orbital.py`): the single-particle rotation
  `b_p = sum_i W_pi a_i` as a named argument rather than an unstated default —
  site, momentum, Daubechies wavelet, and non-interacting natural orbitals,
  each emittable as a rotor circuit (`orbital_rotation_program`), since a
  Jordan-Wigner Givens rotation is exactly two commuting rotors. The spectrum is
  invariant; the cost is not — on the half-filled 8-site Hubbard ring the
  Pauli-word count spans 41–3833 across these bases and the subspace reaching
  1.6 mHa spans 906–4310 determinants, so a word count quoted without its basis
  is not reproducible (`benchmarks/run_orbital_basis.py`)
- a versioned effective-Hamiltonian boundary (`models/effective.py`) that reads
  a spin-independent Wannier one-body matrix plus onsite embedding interactions
  from JSON, validates orbital/spin/sector conventions, and emits the same
  `Model` used by the native lattice builders
- a research layer for VQE/ADAPT-VQE: model builders (`models/`), execution
  backends (`backends/`), a finite-shot measurement/confidence stack
  (`measurement/`), and packaged algorithms (`algorithms/`) — see
  `PLAN.md` §9
- competing-order configurations and compound generators (`subspace/`, §4.2
  level 4): a stabilizer configuration enters as the operator `VR†` that
  carries the reference onto it, so it costs one Pauli word between
  determinants and no second state is ever prepared
- an experimental configuration-space Haar tier
  (`subspace.configuration_haar_packets`): deterministic non-dyadic tree
  details over a caller-declared configuration order, pruned by generator
  support and used only for early A-CASE growth before handing off to level 4.
  On the committed `2x2` Hubbard check this staged policy reaches the exact
  energy with `W=9869` versus `14762` for cost-aware level 4 alone, while
  recording its larger element support and overlap condition number rather
  than hiding that tradeoff (`benchmarks/run_configuration_packets.py`)
- A-CASE (`subspace/`): Rayleigh-Ritz in an operator-generated subspace whose
  basis states `A_i|psi>` are never prepared — every projected matrix element
  is an expectation on one reference state — with a cached matrix-element bank,
  adaptive basis growth, projected observables (expectations and transitions
  without materializing a Ritz state), and finite-shot layers whose intervals
  are labelled `asymptotic`, `heuristic`, or `finite_sample` and never
  conflated; see `PLAN.md`
- QSCI/SQD as a first-class comparison (`subspace/qsci.py`), its classical
  controls (`subspace/selected_ci.py`: excitation closure, one-step and
  budget-matched selected CI, span/principal-angle containment diagnostics), and
  the hybrid that dresses sampled determinants with operator-response directions
  (`subspace/hybrid.py`). QSCI's projected matrix costs zero measured Pauli
  words because it is built classically, so its rows carry sampling yield,
  duplicate fraction, retained probability, preparation cost, matrix nonzeros,
  build/solve time, and memory instead — `W=0` alone is not a resource verdict
- cross-setting shot pooling and rank selection (`measurement/cache.py`,
  `measurement/session.py`): a QWC setting's histogram records every word
  supported inside its basis, not only the one the partition assigned there, so
  `pooling='shots'` reads each word from all of them under inverse-variance
  weights, and `solve_selected_rank` picks the retained rank by
  `E + gamma*sigma` rather than by an overlap-mode cutoff. Both are off by
  default, since the committed records predate them
- a validation ladder (`benchmarks/run_acase_ladder.py`) running H2 through
  H2O CAS(8e,6o), Hubbard clusters, and a Kitaev cluster against the reference
  determinant, sector-exact diagonalization, QSE, fixed Krylov,
  generator-coordinate subspaces, QSCI, and ADAPT-VQE — every row carrying the
  resource metrics, shots, and abstentions beside the energy, and an evidence
  label saying what kind of number it is — plus the integrated Paper B ladder
  (`benchmarks/run_phase12_paper_b.py`), which normalizes every arm onto one
  schema and computes Pareto frontiers only inside a single evidence category,
  so an oracle-sampled arm cannot dominate an implementable one by having no
  state-preparation cost to report
- a dyadic block-commuting measurement hierarchy
  (`benchmarks/run_clifford_hierarchy.py`): block size `k` interpolates from
  qubit-wise commuting (`k=1`) to fully commuting (`k=n`), each group
  diagonalized by a stim-synthesized block-local Clifford circuit, trading
  settings for entangling depth — 913 settings to 64 on the H4 bank, at 2006
  logical CX and two-qubit depth 42. The trade is instance-dependent and is
  reported as a break-even ratio, not as a preferred block size

### Smallest end-to-end correlated-materials showcase

The bundled two-site Wannier-Hubbard record is synthetic and canonical, not a
claimed DFT calculation.  It exercises the real software boundary an upstream
DFT/Wannier/embedding workflow would use:

```text
one-body Wannier matrix + onsite U (JSON)
    -> Jordan-Wigner effective many-body Hamiltonian
    -> A-CASE
    -> energy, projected state coefficients, correlations, Lehmann response
```

Run it from the repository root:

```bash
python examples/acase_effective_model.py
```

The four-qubit `N=2, Sz=0` Hubbard dimer is small enough for an independent
sector-exact oracle but already has a correlated singlet ground state,
suppressed double occupancy, antiferromagnetic spin correlation, and a
nontrivial staggered-spin response.  The schema and example record are in
`examples/data/wannier_hubbard_dimer.json`; complex hopping entries use
`[real, imag]`. All Hamiltonian values are interpreted in the declared
`energy_unit`; the loader labels but does not convert units.

### Real FCIDUMP active-space benchmark

`clifford_qc.models.fcidump_model` is a strict NumPy-only adapter for real,
restricted FCIDUMP records. It restores packed chemist-notation integrals,
constructs the interleaved-spin Jordan–Wigner Hamiltonian with the package's
own fermion operators, validates `NELEC/MS2`, and records the source SHA-256.
Unrestricted and complex extensions are rejected explicitly.

The committed linear-H4 STO-3G CAS(4e,4o) fixture is independently reproducible:

```bash
python benchmarks/run_fcidump_h4.py
```

It maps to 8 qubits, 185 Pauli terms, and a 36-determinant `(N=4, Sz=0)`
sector. The sector oracle agrees with the external PySCF determinant-space FCI
energy to `3.1e-15 Ha`. At the predeclared adaptive budget A-CASE uses `M=9`
and `W=7,371`, with `3.019 mHa` error; the complete singles/doubles coordinate
space uses `M=27` and reaches `0.766 mHa` error. The latter support is
deliberately untracked and is not presented as a measurement-resource result.

### Finite-shot response uncertainty

`ResponseMeasurement` measures `S`, `H`, and a Hermitian projected observable
from one shared QWC cache. `bootstrap_response` resamples the grouped joint
histograms and reruns overlap thresholding, the generalized eigensolve,
transition weights, susceptibility, and optional Lorentzian broadening:

```bash
python examples/acase_finite_shot_response.py
```

Every resulting interval is labelled `heuristic`; `certified` is always
`False`. Root-resolved intervals require isolated Ritz roots, rank-changing
replicas are reported, and broadened-spectrum intervals are pointwise rather
than simultaneous. This is finite-shot uncertainty, not a finite-sample
coverage certificate.

## System Architecture & Methodological Framework

```text
            +-------------------------------------------------------+
            | DFT / Wannier downfolding / chemistry FCIDUMP records |
            +-------------------------------------------------------+
                                     |
                                     v
            +-------------------------------------------------------+
            |     clifford_qc.models (effective, FCIDUMP, lattice)   |
            +-------------------------------------------------------+
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
+---------------------+  +-------------------------+  +----------------------+
|       A-CASE        |  |       QSCI / SQD        |  |      ADAPT-VQE       |
| Rayleigh-Ritz in an |  | Sampled determinant     |  | Confidence-certified |
| operator-response   |  | subspace; the projected |  | selection on the     |
| subspace; no basis  |  | matrix is built         |  | odd-Y pool from      |
| state is prepared   |  | classically (W = 0)     |  | shared word caches   |
+---------------------+  +-------------------------+  +----------------------+
        |                            |                            |
        +-------------+--------------+                            |
                      v                                           |
     +---------------------------------------+                    |
     | Hybrid: sampled determinants dressed  |                    |
     | by operator response, scored against  |                    |
     | classical selected-CI controls        |                    |
     +---------------------------------------+                    |
                      |                                           |
                      +---------------------+---------------------+
                                            |
                                            v
            +-------------------------------------------------------+
            | Sector statevector / SciPy-sparse oracles & validation |
            +-------------------------------------------------------+
```

`clifford_qc` is structured around six core engineering and theoretical pillars:

1. **Unified Multivector Representation (`MV`):** States ($\rho$), unitary gates ($U$), observables ($O$), Kraus channels, Jordan-Wigner Clifford generators ($\gamma_j$), and CAR creation/annihilation operators ($c_j, c_j^\dagger$) all exist as sparse multivectors in $Cl(2n, \mathbb{C}) \cong M(2^n, \mathbb{C})$. Qubit Pauli letters are packed into 2 bits per qubit ($0=I, 1=X, 2=Y, 3=Z$), enabling fast binary-symplectic multiplication via bitwise `XOR`, `AND`, and `popcount` (mod 4).
2. **Three Exact Scalar Pairings:**
   - `scalar_product`: Bilinear inner product with reversion sign $\langle A \widetilde{B}\rangle_0$.
   - `hs_product`: Hilbert-Schmidt sesquilinear inner product $\frac{1}{2^n}\operatorname{Tr}(A^\dagger B)$.
   - `trace_pairing`: Bilinear trace pairing $\frac{1}{2^n}\operatorname{Tr}(A B)$ without conjugation or reversion (used for non-Hermitian operator subspace matrices like $A^\dagger H A$).
3. **A-CASE Subspace Eigensolver:** Operates via Rayleigh-Ritz projection in an adaptively grown operator-response subspace basis $\{A_i |\psi_0\rangle\}$. Matrix elements $H_{ij} = \langle \psi_0| A_i^\dagger H A_j |\psi_0\rangle$ and overlaps $S_{ij} = \langle \psi_0| A_i^\dagger A_j |\psi_0\rangle$ are calculated as expectation values on a single reference state $|\psi_0\rangle$ without ever preparing basis states $A_i|\psi_0\rangle$ on hardware.
4. **Statistically Certified ADAPT-VQE:** Evaluates candidate selection gradients $G_j = \operatorname{Tr}\left[\rho \cdot \left(-\frac{i}{2}\right)[H, P_j]\right]$ using odd-Y algebraic pool reduction for antiunitary-real Hamiltonians, shared QWC measurement caches, and Šidák/Bonferroni confidence bounds.
5. **Sector-Restricted Statevector & Matrix-Free Tier:** `backends/sector_statevector.py` tracks statevector amplitudes directly in $C(n,k)$ particle/spin symmetry sectors without building dense $2^n \times 2^n$ matrices or full statevectors.
6. **Sampled Determinant Subspaces and Their Controls:** `subspace/qsci.py` diagonalizes the Hamiltonian restricted to sampled computational-basis configurations, taken as a submatrix of the same compiled operator action the matvecs use rather than a second Slater-Condon implementation. `subspace/selected_ci.py` supplies the classical controls — excitation closure, one-step and budget-matched selected CI, and the containment diagnostic $\lVert (I - Q_D Q_D^\dagger) Q_A \rVert$ with $\operatorname{rank}(A) \le \operatorname{rank}(D)$ — so a hybrid gain cannot be claimed against an absent comparator.

## Core Conventions

- Pauli labels are strings over `I`, `X`, `Y`, `Z`.
- Qubit `0` is the leftmost character in labels such as `"XIZ"`.
- Word codes use two bits per qubit: `0=I`, `1=X`, `2=Y`, `3=Z`.
- The matrix backend uses the same ordering, so `to_matrix(P("XIZ"))` equals
  `kron(X, I, Z)`.
- Trace normalization is

\[
\operatorname{Tr}(A)=2^n \langle A\rangle_0.
\]

- Jordan-Wigner Clifford generators are

\[
\gamma_{2j}=Z_0\cdots Z_{j-1}X_j,
\qquad
\gamma_{2j+1}=Z_0\cdots Z_{j-1}Y_j.
\]

- Rotor convention is

\[
R_P(\theta)=\exp(-i\theta P/2)
=\cos(\theta/2)-i\sin(\theta/2)P,
\qquad P^2=1.
\]

See `CONVENTIONS.md` for the detailed convention notes.

## Pauli-Rotor IR

The native interchange layer is a small Pauli-rotor IR. It represents programs
as named Clifford gates, Pauli-word rotors `exp(-i theta P / 2)`, ordered
parameters, and measurement tasks. Every operation lowers exactly to `MV`.

```python
from clifford_qc import Program, Parameter, PauliSum, adjoint_gradient, to_qasm3

theta = Parameter("theta")
prog = (
    Program(2, parameters=[theta])
    .clifford("H", 0)
    .clifford("CX", 0, 1)
    .rotor("ZZ", theta)
    .measure_expectation({"XX": 1.0})
    .measure_z(0, 1)
)

print(prog.run([0.4]))
print(to_qasm3(prog, [0.4]))

obs = PauliSum.from_labels({"XX": 1.0})
print(adjoint_gradient(prog, obs, [0.4]))
```

IR objects include:

| Object | Role |
|---|---|
| `PauliWord` | single Pauli word with label/code/support helpers |
| `PauliSum` | sparse Pauli-word observable type |
| `Parameter`, `ParameterGroup` | ordered parameter binding and gradient order |
| `Rotor` | `exp(-i theta P / 2)` for a Pauli word |
| `NamedClifford` | `X`, `Y`, `Z`, `H`, `S`, `SDG`, `CX`, `CZ`, `SWAP` |
| `MeasurementTask` | expectation values or computational-basis probabilities |
| `Program` | operations, measurements, exact execution, JSON serialization |

OpenQASM 3 export is an export pass, not the native format. Pauli rotors lower
to basis changes, a CX parity ladder, `rz(theta)`, and uncompute. Expectation
measurements are evaluated by `clifford_qc`; only `sample_z` tasks become QASM
measure statements.

## Optional Bridges

Bridge modules live under `clifford_qc.bridges`:

- `stim_bridge`: Clifford-only `Program -> stim.Circuit` and tableau-backed
  Pauli conjugation maps for large-n Clifford validation.
- `openfermion_bridge`: lossless `QubitOperator <-> MV/PauliSum` conversion
  and one-way `FermionOperator -> MV` through the package's Jordan-Wigner
  operators.
- `pytket_bridge`: `Program -> pytket.Circuit` using `PauliExpBox` for rotors,
  plus supported round-trips back to the IR.
- `pennylane_bridge`: `Program -> PennyLane` operations and observables for
  expectation and gradient comparison.
- `pyzx_bridge`: `Program <-> ZX-calculus` circuits for phase-gadget optimization,
  diagram simplification, and global-phase invariant verification.

The QASM3 exporter is part of the core package and needs no extra dependency.

## Examples

Run the included examples from the repository root:

```bash
PYTHONPATH=. python examples/bell_chsh.py
PYTHONPATH=. python examples/grover_2q.py
PYTHONPATH=. python examples/fermion_car.py
PYTHONPATH=. python examples/noisy_channel.py
PYTHONPATH=. python examples/tfim_exact.py
PYTHONPATH=. python examples/acase_effective_model.py
PYTHONPATH=. python examples/acase_premise_check.py
PYTHONPATH=. python examples/acase_adaptive.py
PYTHONPATH=. python examples/acase_finite_shot.py
PYTHONPATH=. python examples/acase_finite_shot_response.py
PYTHONPATH=. python examples/acase_materials.py
PYTHONPATH=. python examples/acase_sector_backend.py
```

They cover Bell/CHSH diagnostics, a two-qubit Grover step, fermionic CAR
checks, noisy channels, a small transverse-field Ising Hamiltonian, the
A-CASE subspace invariants with their resource accounting, A-CASE adaptive
growth against the fixed QSE/Krylov/ADAPT-VQE baselines at matched operator
budget, and the finite-shot layers (shared grouped measurement, a
delta-method-versus-Monte-Carlo uncertainty study, and certified growth with
abstention), and the materials layer (Hubbard and Kitaev clusters, projected
observables, excited states by state-averaged growth), and the
sector-restricted exact tier (C(n,k) amplitudes instead of 2^n, matrix-free
Lanczos, 24 qubits without a matrix).

## Tests And Validation

Core tests:

```bash
pip install -e .[test]
pytest
```

Full bridge conformance suite:

```bash
pip install -e .[test,bridges]
pytest
```

Dependency-light smoke check:

```bash
PYTHONPATH=. python -m clifford_qc.verify
```

The test suite covers algebraic invariants, property-based checks,
dense-matrix homomorphism and round-trips, density/channel behavior, IR JSON
golden vectors, QASM3 lowering semantics, exact gradients, and optional bridge
conformance where dependencies are installed.

Published numbers have a second gate. Every committed benchmark record has a
producer (`benchmarks/run_*.py`), a stamped JSON/JSONL artifact under
`benchmarks/reference_results/`, and a checker (`benchmarks/check_*.py`) that
regenerates it and compares field by field — discrete fields exactly, floats to
tolerance. `benchmarks/check_docs.py` closes the same loop on the prose: it
verifies that every command in `REPRODUCING.md` names a script that exists,
that every long flag is one the script accepts, that the predeclared parameter
table matches the constants in the code, and that no committed record goes
undocumented. Documentation drift is a reproduction failure, not a cosmetic one.

## Package Layout

```text
clifford_qc/
  multivector.py   # sparse MV, word product, label/word-code utilities
  pauli.py         # I, X, Y, Z, P, commutators, tensor product
  clifford.py      # gamma, pseudoscalar, blade masks, grades
  fermion.py       # c, cdag, number operators
  gates.py         # H/S/T, rotors, controlled gates, Trotter/expm helpers
  states.py        # densities, evolution, measurement, traces
  channels.py      # Kraus channels
  matrix.py        # dense matrix bridge for validation/small n
  diagnostics.py   # entropy, negativity, fidelity, trace diagnostics
  ir.py            # Pauli-rotor IR, serialization, gradients
  qasm3.py         # OpenQASM 3 export pass for IR programs
  verify.py        # dependency-light smoke suite
  bridges/         # optional Stim/OpenFermion/pytket/PennyLane/PyZX bridges
  sparse.py        # sparse Pauli reference tier: eigsh, (N,Sz) sectors
  models/          # TFIM, XXZ, random-Ising; Hubbard/Kanamori/Anderson/
                   # Kitaev; versioned effective-Hamiltonian ingestion;
                   # orbital bases + Givens rotor networks;
                   # material observables; NumPy-only FCIDUMP; chemistry
  backends/        # Backend protocol: exact MV, dense reference, finite-shot,
                   # sector-restricted statevector + matrix-free Lanczos
  measurement/     # commutator bank, shared word cache with cross-setting
                   # pooling, confidence, allocation policies, QWC and dyadic
                   # block-commuting grouping, device-card cost model,
                   # shared grouped-measurement sessions and rank selection
  algorithms/      # optimizers, pools (odd-Y), fixed-depth VQE, ADAPT-VQE
                   # (exact / finite-shot / layered / subpool / random)
  subspace/        # A-CASE: generator families, the normalized/thresholded
                   # generalized eigenproblem, cached matrix-element bank
                   # with projected observables, adaptive growth, finite-shot
                   # layers (shared grouped measurement, asymptotic Ritz
                   # uncertainty, whole-pipeline response bootstrap,
                   # sample-split growth certificate), reference-aware sector
                   # certificates, Lehmann response, support-pruned
                   # configuration Haar tier, dense cross-check;
                   # QSCI/SQD, classical selected-CI controls, the
                   # QSCI x A-CASE hybrid, and coarse-to-fine packet selection
```

## Project Notes

- The project is currently alpha (`0.3.0`).
- `PLAN.md` is the single research plan — one document, consolidating the four
  former roadmaps plus newly integrated hardware-aware resource accounting. It is
  the place to look for what is built, what is not, and what would falsify each
  claim:

  | § | contents |
  |---|---|
  | 1–4 | scope, repository facts, the algebra contract and standing invariants, the method |
  | 5 | the phase ledger: Phases 0–7 and 8–12 shipped, 13–18 open, R1–R4 for hardware-aware costing |
  | 6 | resource accounting — word universe, supports, conditioning, and the device-card cost model |
  | 7 | the validation ladder, what it does and does not support, and the benchmark inventory |
  | 9 | Paper A: confidence-certified, measurement-efficient ADAPT-VQE (phases A0–A5, all complete; the manuscript's findings are quoted there from `paper/manuscript.tex`) |
  | 10–11 | falsifiable questions Q1–Q13 and QR1–QR6; the fifteen-paper literature index |
  | 14 | what the plan does not claim |

  It records the comparisons the project expects to lose — A-CASE does not beat
  fixed Krylov on energy, or ADAPT-VQE on stretched geometries — and confines
  wavelets to the implemented configuration-space Haar staging experiment.
- `REPRODUCING.md` is the reproduction contract: the command, record, and
  drift check for every committed benchmark.
- `MIGRATION.md` maps the old single-file API onto this package.
- `simple_plan.md` records the implemented roadmap and bridge validation
  criteria.
- `paper/` holds the unsubmitted Paper A manuscript (REVTeX),
  `paper_a_case_subspaces/` the restored source snapshot for the first public
  A-CASE preprint, and `paper_acase/` the later DA-CASE manuscript — "DA-CASE:
  reusable measurements for adaptive quantum subspaces", the dyadic-measurement
  variant of the method this README calls A-CASE. Each public snapshot has its
  own drift checker and provenance boundary.
- **Public preprints.** Two papers from this repository are on arXiv:
  [arXiv:2608.00560](https://arxiv.org/abs/2608.00560), *Adaptive
  operator-generated subspaces for effective many-body Hamiltonians* (A-CASE), and
  [arXiv:2608.08739](https://arxiv.org/abs/2608.08739), *DA-CASE: reusable
  measurements for adaptive quantum subspaces*. Both are Utama & Dipojono; see
  `CITATION.cff`. The A-CASE sources from commit `67ea0dd` are restored under
  `paper_a_case_subspaces/`; `SOURCE_SNAPSHOT.json` pins their historical blobs.
- License: Apache-2.0.
