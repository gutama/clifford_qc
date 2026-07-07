# `clifford_qc` conventions

## Algebra

The engine represents the complexified Clifford algebra

\[
Cl(2n,\mathbb C) \cong M(2^n,\mathbb C)
\]

using the Pauli-word image of the Jordan-Wigner map. Every object is an `MV`:
state densities, gates, observables, channels, Clifford generators, and
fermionic operators.

## Pauli-word labels and qubit ordering

- Labels are strings over `I`, `X`, `Y`, `Z`.
- Qubit `0` is the leftmost label character.
- `P("XIZ")` means `X_0 I_1 Z_2`.
- `to_matrix(P("XIZ")) == kron(X, I, Z)`.

## Word-code layout

Each qubit uses two bits:

```text
0 = I, 1 = X, 2 = Y, 3 = Z
```

The code of a label is

```text
sum(letter_code[label[j]] << (2*j) for j in range(n))
```

## Trace normalization

For a multivector/operator `A`,

\[
\operatorname{Tr}(A)=2^n\langle A\rangle_0.
\]

In code:

```python
A.trace() == (2**A.n) * A.scalar_part()
```

## Jordan-Wigner Clifford generators

\[
\gamma_{2j}=Z_0\cdots Z_{j-1}X_j,
\qquad
\gamma_{2j+1}=Z_0\cdots Z_{j-1}Y_j.
\]

The engine can recover Clifford grade from a Pauli word by inverse JW decoding:

```python
P("IXI").grades()
```

## Fermions

The Witt/CAR layer is

\[
c_j=\frac12(\gamma_{2j}+i\gamma_{2j+1}),
\qquad
c_j^\dagger=\frac12(\gamma_{2j}-i\gamma_{2j+1}).
\]

It verifies

\[
\{c_i,c_j^\dagger\}=\delta_{ij},\quad
\{c_i,c_j\}=0,
\quad
c_j^\dagger c_j=\frac12(1-Z_j).
\]

## Rotor convention

For a Hermitian Pauli-word generator `P` with `P*P = I`,

\[
R_P(\theta)=\exp(-i\theta P/2)
=\cos(\theta/2)-i\sin(\theta/2)P.
\]

Code:

```python
R = rotor(P("ZZ"), theta)
rho2 = evolve(rho, R)
```

## Dense matrix bridge

The matrix backend is for validation and small-system exact targets, not the
primary representation. It obeys:

```python
to_matrix(A * B) == to_matrix(A) @ to_matrix(B)
from_matrix(to_matrix(A)).is_close(A)
```
