# clifford_qc v0.1

`clifford_qc` is a polished operator-centric quantum Clifford engine for
quantum computing in the complexified Clifford algebra

\[
Cl(2n,\mathbb C) \cong M(2^n,\mathbb C).
\]

It stores multivectors sparsely in the Pauli-word basis. States, gates,
observables, channels, Jordan-Wigner Clifford generators, and fermionic Witt
operators all live in the same algebra.

## Core conventions

- Word code: two bits per qubit, `0=I`, `1=X`, `2=Y`, `3=Z`.
- Qubit `0` is the leftmost character in labels such as `"XIZ"`.
- Matrix backend uses the same ordering, so `to_matrix(P("XIZ"))` equals
  `kron(X, I, Z)`.
- Trace normalization:

\[
\operatorname{Tr}(A)=2^n \langle A\rangle_0.
\]

- Jordan-Wigner Clifford generators:

\[
\gamma_{2j}=Z_0\cdots Z_{j-1}X_j,
\qquad
\gamma_{2j+1}=Z_0\cdots Z_{j-1}Y_j.
\]

- Rotor convention:

\[
R_P(\theta)=\exp(-i\theta P/2)
=\cos(\theta/2)-i\sin(\theta/2)P,
\qquad P^2=1.
\]

## Package layout

```text
clifford_qc/
  multivector.py   # sparse MV, word product, label/word-code utilities
  pauli.py         # I, X, Y, Z, P, commutators, tensor product
  clifford.py      # gamma, pseudoscalar, blade masks, grades
  fermion.py       # c, c†, number operators
  gates.py         # H/S/T, rotors, CNOT/CZ/SWAP/Toffoli, Trotter/expm
  states.py        # density operators, evolution, measurement, partial traces
  channels.py      # Kraus channels
  matrix.py        # dense matrix bridge for validation/small n
  diagnostics.py   # entropy, negativity, fidelity, trace diagnostics
  verify.py        # dependency-light smoke suite (pytest is canonical)
  ir.py            # Pauli-rotor IR: PauliWord/PauliSum/Rotor/Program, gradients
  qasm3.py         # OpenQASM 3 export pass for IR programs
  bridges/         # optional: stim, openfermion, pytket, pennylane
```

## Quick start

```python
from clifford_qc import *

bell = bell_density()
print(bell.to_labels())
print(negativity(bell, {1}))
```

## Pauli-rotor IR and bridges

Circuits are expressed in a small IR (rotors `exp(-iθP/2)` + named
Cliffords + measurement tasks) that lowers exactly to `MV` and exports to
the wider ecosystem:

```python
from clifford_qc import Program, Parameter, PauliSum, adjoint_gradient, to_qasm3

theta = Parameter("theta")
prog = (Program(2, parameters=[theta])
        .clifford("H", 0).clifford("CX", 0, 1)
        .rotor("ZZ", theta)
        .measure_expectation({"XX": 1.0}))

prog.run([0.4])                 # exact expectation values via MV evolution
print(to_qasm3(prog))           # OpenQASM 3 export (rz + CX parity ladder)
obs = PauliSum.from_labels({"XX": 1.0})
adjoint_gradient(prog, obs, [0.4])  # exact gradients, matches PennyLane
```

Optional bridges (each an extra: `pip install clifford-qc[bridges]`):
`bridges.stim_bridge` (Clifford tableau validation at large n),
`bridges.openfermion_bridge` (`QubitOperator`/`FermionOperator ↔ MV`),
`bridges.pytket_bridge` (`Rotor → PauliExpBox`, round-trips),
`bridges.pennylane_bridge` (`Rotor → qml.PauliRot`, gradient comparison).
See `simple_plan.md` for the roadmap and validation criteria.

## Tests

```bash
pip install -e .[test]   # add ,bridges for the ecosystem conformance tests
pytest
```

Run the quick verification entry point:

```bash
PYTHONPATH=. python -m clifford_qc.verify
```

Run examples:

```bash
PYTHONPATH=. python examples/bell_chsh.py
PYTHONPATH=. python examples/grover_2q.py
PYTHONPATH=. python examples/fermion_car.py
PYTHONPATH=. python examples/noisy_channel.py
PYTHONPATH=. python examples/tfim_exact.py
```

## Scope

This is not a speedup claim over dense matrices. The goal is structural:
one algebra for states, gates, observables, channels, fermions, Clifford
structure, and sparse Pauli-word semantics. Dense matrix conversion is kept as
a validation and small-system bridge.
