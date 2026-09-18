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

## The three pairings

`MV` carries three scalar pairings. They collapse to a sum over the shared
Pauli-word support and differ only in the factor in front of `a_w b_w`:

| Method | Word coordinates | Character |
|---|---|---|
| `scalar_product` | `sum_w a_w b_w (-1)^{k_w(k_w-1)/2}` | bilinear, reversion sign (`<A ~B>_0`) |
| `hs_product` | `sum_w conj(a_w) b_w` | sesquilinear (`Tr(A' B)/2^n`) |
| `trace_pairing` | `sum_w a_w b_w` | bilinear, no reversion, no conjugation (`Tr(A B)/2^n`) |

They are not interchangeable, and the difference hides in easy test cases: a
Hermitian operand has real coefficients, which masks the conjugation, and an
operand of Clifford grade `0, 1 mod 4` masks the reversion sign. Projected
subspace matrix elements need `trace_pairing`, because `A' H A` is not
Hermitian.

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

## Model metadata

A `Model` states the conventions its Pauli sum is read under, and
`models/metadata.py` enforces that in `Model.__post_init__`. Every model
declares:

```python
metadata["metadata_schema"] == "clifford_qc.model_metadata.v1"
metadata["kind"] in MODEL_KINDS   # fermionic_lattice, fermionic_orbital_basis,
                                  # anderson_impurity, molecular, spin_lattice
```

A fermionic `kind` additionally declares `spin_convention` (`"interleaved"` or
`"blocked"`), `spin_orbitals`, `n_spatial_orbitals`, `n_electrons`, and `sz`.
A spin `kind` declares none of those and is refused if it tries: a spin
Hamiltonian has no spin-orbital ordering, and a convention nothing honours is
worse than an absent one. Build the dict with `model_metadata()`, which takes
`spin_convention` as a required keyword: a default there would only move the
silent default from the readers to the one writer, where the validator can no
longer see the omission.

`n` and the Pauli-word count are **not** required — the `Model` already carries
them, and a second copy can drift. They are instead *checked* when present.

Read `spin_convention` with `models.metadata.spin_convention(model)`, which
raises rather than defaulting. There is no defensible default: under the wrong
ordering the two-qubit reduction of the 2-site Hubbard dimer returns
`-4.000000000000` instead of `-4.828427124746`, with no exception and a
plausible one-word Hamiltonian, because `_spin_parity_rows` tapers a different
row. `metadata_schema` is distinct from `schema`, which `models/effective.py`
uses for the schema of the input payload a model was downfolded from.

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
