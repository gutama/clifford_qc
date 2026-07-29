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

## Install

```bash
pip install clifford-qc
```

For local development:

```bash
pip install -e .[test]
pytest
```

Optional ecosystem bridges are installed independently:

```bash
pip install -e .[stim]
pip install -e .[openfermion]
pip install -e .[pytket]
pip install -e .[pennylane]
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
- dense matrix conversion and exact small-system ground states
- a Pauli-rotor intermediate representation with exact gate-by-gate execution, versioned JSON serialization, gradients, and QASM3 export
- optional bridges for Stim, OpenFermion, pytket, and PennyLane
- a research layer for VQE/ADAPT-VQE: model builders (`models/`), execution
  backends (`backends/`), a finite-shot measurement/confidence stack
  (`measurement/`), and packaged algorithms (`algorithms/`) — see
  `RESEARCH_PLAN.md`
- A-CASE (`subspace/`): Rayleigh-Ritz in an operator-generated subspace whose
  basis states `A_i|psi>` are never prepared — every projected matrix element
  is an expectation on one reference state — with a cached matrix-element bank
  for incremental growth and projected observables (expectations and
  transitions without materializing a Ritz state); see `ACASE_RESEARCH_PLAN.md`

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

The QASM3 exporter is part of the core package and needs no extra dependency.

## Examples

Run the included examples from the repository root:

```bash
PYTHONPATH=. python examples/bell_chsh.py
PYTHONPATH=. python examples/grover_2q.py
PYTHONPATH=. python examples/fermion_car.py
PYTHONPATH=. python examples/noisy_channel.py
PYTHONPATH=. python examples/tfim_exact.py
PYTHONPATH=. python examples/acase_premise_check.py
```

They cover Bell/CHSH diagnostics, a two-qubit Grover step, fermionic CAR
checks, noisy channels, a small transverse-field Ising Hamiltonian, and the
A-CASE subspace invariants with their resource accounting.

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
  bridges/         # optional Stim/OpenFermion/pytket/PennyLane bridges
  models/          # TFIM, XXZ, random-Ising benchmark models
  backends/        # Backend protocol: exact MV, dense reference, finite-shot
  measurement/     # commutator bank, shared word cache, confidence,
                   # allocation policies, QWC measurement grouping
  algorithms/      # optimizers, pools (odd-Y), fixed-depth VQE, ADAPT-VQE
                   # (exact / finite-shot / layered / subpool / random)
  subspace/        # A-CASE: generator families, the normalized/thresholded
                   # generalized eigenproblem, cached matrix-element bank
                   # with projected observables, dense cross-check
```

## Project Notes

- The project is currently alpha (`0.1.0`).
- `MIGRATION.md` maps the old single-file API onto this package.
- `simple_plan.md` records the implemented roadmap and bridge validation
  criteria.
- `RESEARCH_PLAN.md` is the Paper A roadmap (confidence-certified,
  measurement-efficient ADAPT-VQE).
- `ACASE_RESEARCH_PLAN.md` is the active roadmap (A-CASE: adaptive
  Clifford-algebra subspace eigensolver); Phases 1-2 ship in `subspace/`.
- `paper/` holds the Paper A manuscript (REVTeX) with figures regenerated
  from the committed benchmark data.
- License: Apache-2.0.
