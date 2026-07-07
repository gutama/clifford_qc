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
  verify.py        # full invariant suite
```

## Quick start

```python
from clifford_qc import *

bell = bell_density()
print(bell.to_labels())
print(negativity(bell, {1}))
```

Run the verification suite:

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
