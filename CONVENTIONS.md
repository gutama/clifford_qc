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

## Second moments, residuals and folded roots

A subspace state is `|Psi> = sum_j c_j A_j|psi>`. A matrix-element bank pairs
three blocks with the reference `psi`, all in the same way:

```text
S_ij = <psi|A_i' A_j|psi>        overlap
H_ij = <psi|A_i' H A_j|psi>      projected Hamiltonian
K_ij = <psi|A_i' H^2 A_j|psi>    second moment (SecondMomentBank)
```

`K`'s rows are the bank's projected-observable rows for `H * H`, built as
`A_i.dagger() * (H2 * A_j)`. With `n = c'Sc` and `c'` the conjugate transpose:

```text
<H>      = c'Hc / n
<H^2>    = c'Kc / n
sigma^2  = <H^2> - 2E<H> + E^2  = ||(H - E)Psi||^2 / <Psi|Psi>
mu(w)    = c'F(w)c / n          = ||(H - w)Psi||^2 / <Psi|Psi>,   F(w) = K - 2wH + w^2 S
```

`RitzResidual.variance` is `sigma^2` at the root's reported energy `E`. At the
Rayleigh quotient `E = <H>`, which `energy_variance` and `FoldedRoot.variance`
use, it is the energy variance `<H^2> - <H>^2`. The package reserves "residual
norm" for `sigma` alone. `mu(w)` is the folded value. When `E = <H>`,
`mu(w) = sigma^2 + (E - w)^2`, so `sigma^2 <= mu(w)` for every target.

**Resolution floors.** Each quantity is a difference of terms of order
`<H^2>`, so it is reported signed and never clipped. It counts as resolved only
above `RESOLUTION = 1e-12` times its cancellation scale, the magnitude the
difference cancels per unit norm, with `|.|` taken elementwise:

```text
scale_sigma2 = |c|'|K||c| / n                         sigma^2 > RESOLUTION * scale_sigma2
scale_mu     = |c|'(|K| + 2|w||H| + w^2|S|)|c| / n    mu(w)   > RESOLUTION * scale_mu
```

`RitzResidual.resolved` and `FoldedRoot.variance_resolved` test the first,
`FoldedRoot.resolved` the second.

The floor is a numerical convention for rounding, not a confidence bound. An
unresolved value stays unresolved: a rounded zero is not a converged residual.
`residual_norm` is `sqrt(max(sigma^2, 0))`.

**Inclusion intervals.** `||(H - w)Psi||^2 / <Psi|Psi>` is a weighted mean of
`(E_k - w)^2` over the eigenvalues `E_k` the state touches. So for every folded
root, ground or interior, some eigenvalue of `H` lies in each of

```text
target interval     [w - b_t, w + b_t],  b_t = sqrt(max(mu, 0)      + RESOLUTION * scale_mu)
Weinstein interval  [E - b_W, E + b_W],  b_W = sqrt(max(sigma^2, 0) + RESOLUTION * scale_sigma2)
```

with `E = <H>`. The floors keep a rounding-level value from producing a falsely
tight interval. Neither interval says which eigenvalue it holds. Ground-state
claims need an independently established spectrum, supplied to convergence
reporting as a `GroundStateReference`.

**Counting clusters.** Overlapping intervals may hold the same eigenvalue, so
per-root intervals do not count eigenvalues. `ritz_clusters` does. For Ritz
roots `I` of one solve, let `Q` hold their states orthonormally and
`Theta = Q'HQ`. Then `R = HQ - Q Theta` has `R'R = Q'KQ - Theta^2`, and Kahan's
theorem (Parlett, Thm 11.5.1) pairs the `|I|` Ritz values one-to-one with
distinct eigenvalues of `H` within `||R||_2`. Every root starts alone; clusters
whose hulls meet, endpoints included, are merged and `r_I` recomputed until
the hulls are disjoint:

```text
r_I = sqrt(max(lambda_max(Q'KQ - Theta^2), 0) + RESOLUTION * || |Q|'|K||Q| ||_2)
[min_I theta - r_I, max_I theta + r_I]  holds at least |I| eigenvalues, with multiplicity
```

A one-root cluster is that root's Weinstein interval, to rounding. Disjoint
hulls add, so a degenerate level the span reaches is counted with its
multiplicity, which no per-root interval can do. A hull may hold more
eigenvalues than roots, and a cluster of roots `0..m-1` need not hold the `m`
lowest. The count needs a pure reference. A mixed `rho` pairs `A_j rho^(1/2)`,
on which `H` repeats every eigenvalue, so it is refused, as is a nonzero
`overlap_ridge`.

**Constant shifts.** Under `H -> H + a`, the blocks become `H + aS` and
`K + 2aH + a^2 S`. Shifting the target with them, `w -> w + a`, leaves `F`, and
so `sigma^2` and `mu`, unchanged; energies, targets and intervals move by `a`.
The cancellation scales do not stay fixed. A large constant offset, such as a
molecule's nuclear-repulsion and core energy, adds about `a^2|S|` to `|K|` and
raises the floor with it.

**Solves.** A folded solve uses `solve_projected` on `(S, F(w))`, so it retains
the same span a Ritz solve with the same thresholds would. A nonzero
`overlap_ridge` is refused, because it changes the metric and its vectors
would no longer solve `F c = mu S c`.

**Exact only.** All three blocks are exact pairings with the reference. No
finite-shot estimator of `K` exists. Under uniform allocation, measuring the
residual to the energy's standard error costs more than ten times the energy's
shots on every bank priced: Q18-S1 (uniform QWC, five banks) and Q18-S4 (fully
commuting settings, the four 8-qubit banks) both read `NONE`. Before constructing
`H^2`, `SecondMomentBank` applies a product-free coefficient-storage preflight to
its approved indices, including banks outside the five Phase 15 preflight bases.
The finite `SecondMomentStoragePolicy` budget bounds coefficient occurrences;
it does not bound peak RSS or certify finite-shot affordability.

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

Grade-dependent operations (`grade`, `grades`, `reverse`, `wedge`,
`lcontract`, `rcontract`, `scalar_product`, and `is_blade`) always use this fixed JW generator frame:
on parity- or BK-encoded operators they describe the JW pre-image of the stored
Pauli words, not the transported fermionic structure. For a fermionic
grade/blade interpretation, use the operator's JW image before encoding or
register reduction; `MV` does not track or transport the encoding's generator
frame. Operator products, adjoints, and trace pairings remain valid in each
encoding.

`blade_mask(n, code)` and `mask_to_code(n, mask)` are inverse subset/word maps.
They encode which generators occur, not the phase of their ordered product.
`blade_from_mask` retains that phase by multiplying Majoranas in ascending order.

## Exterior products and contractions

For homogeneous grades `r` and `s`, the named contractions are

\[
A_r\mathbin{\lrcorner} B_s=\langle\widetilde{A_r} B_s\rangle_{s-r}
\quad(r\leq s),\qquad
A_r\mathbin{\llcorner} B_s=\langle A_r\widetilde{B_s}\rangle_{r-s}
\quad(r\geq s).
\]

Other grade pairs give zero. `A.lcontract(B)` and
`A.rcontract(B)` extend these definitions bilinearly to mixed
grades, including scalars and complex coefficients. Reversion changes grade
signs without conjugating coefficients. For example, an orthonormal bivector
contracts with itself to `+1`, whereas its unreversed square is `-1`.
These conventions are distinct from the full geometric product and adjoint.
The named methods make reversion explicit; shift aliases are omitted because
they can be confused with unreversed grade-projected contractions. For example,
with orthonormal vectors `a`, `b` and `C = a ^ b`,
`(a ^ b).lcontract(C) == b.lcontract(a.lcontract(C))`; exchanging the nested
order changes the sign. Translate formulas using the definitions above.

`is_blade(tol)` uses the full Plücker condition on homogeneous elements,
including odd grades. Only contractions obtained by deleting one generator
from an active support mask can be nonzero. At grade `k` with `t` stored terms
there are at most `k*t` candidate masks; contraction/wedge support can still
grow. Zero is accepted and mixed-grade nonzero elements are rejected. The
existing constructor pruning (`TOL`) and absolute decision tolerance remain.

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

## Density matrices

Fragment solvers (`subspace/fragment.py`, Phase 18) return spin-orbital
density matrices indexed by the model's qubit order. Occupied is `|1>`, and
an occupation word carries qubit `p` at bit `n-1-p`, as the sector backend
does:

```text
rdm1[p, q]       = <c+_p c_q>
rdm2[p, q, r, s] = <c+_p c+_q c_r c_s>
```

`spatial_rdms` spin-sums them into the chemist order the FCIDUMP reader
assembles, `D1[p, q] = sum_s <c+_ps c_qs>` and
`D2[p, q, r, s] = sum_(s,t) <c+_ps c+_rt c_st c_qs>`. Then
`E = E_core + sum h_pq D1_pq + 1/2 sum (pq|rs) D2_pqrs` with no reindexing.
The spin sum reads the model's declared `spin_convention` and never assumes
one.

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

Time evolution uses the same sign, `|psi(t)> = exp(-iHt)|psi(0)>`, with
`hbar = 1` and `t` in inverse units of the model's energy. A Trotter circuit
from `subspace.time_evolution.trotter_program` is a list of these rotors whose
unitary equals `gates.trotter_unitary` / `gates.trotter2_unitary`.

A product of rotors is generally not a single `R_P(theta)`. The Givens
sequence of a compiled anticommuting clique (`measurement/cliques.py`) has
generators `i A_1 A_k`, so each factor is `exp(theta A_1 A_k / 2)`. That is a
simple rotor of the clique's own `Cl(m,0)` frame. Their product is a `Spin(m)`
rotor that need not be simple (the exponential of one 2-blade) once `m >= 4`.
`tests/test_clique_reference_obstruction.py` pins an `m = 4` double rotation.

## Dense matrix bridge

The matrix backend is for validation and small-system exact targets, not the
primary representation. It obeys:

```python
to_matrix(A * B) == to_matrix(A) @ to_matrix(B)
from_matrix(to_matrix(A)).is_close(A)
```
