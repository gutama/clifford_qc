# Phase 16B, second preregistration: a control that can lose

The first decision experiment (`reference_results/phase16b_feasibility.json`,
merged in #109) returned CONDITIONAL, and its own evidence says the verdict was
not about real-time Krylov. On every TFIM instance the incumbent never reached
the accuracy target **in exact arithmetic**, so its cost was censored and the
frozen rule correctly routed the instance to UNDETERMINED. The real-time arms
reached the target there against nothing.

This is the second preregistration. It is a new declaration, not an edit: the
v1 config and record stay exactly as committed, and nothing here is permitted
to change how v1 is read.

## 1. Why the v1 control could not reach the target

Not a tuning miss. A structural property of the pool the v1 config froze.

**The Hamming-ball result.** For a computational-basis reference `|b>` and a
Pauli word `P`, `P|b> = (phase) |b XOR x(P)>`, where `x(P)` is the word's X
support — `Z` letters contribute phase only. So

```text
span{ P|b> : weight(P) <= k }  =  span{ |b'> : hamming(b', b) <= k }
```

An A-CASE pool of weight-≤`k` Pauli words on a product-state reference spans
exactly the Hamming ball of radius `k`, and nothing else. Its dimension is
combinatorial in `k`, not adaptive. Measured on `tfim(4,J=1,h=1)` with the
`|0000>` reference, against the predicted ball size:

| pool | predicted ball | basis reached | error |
|---|---:|---:|---:|
| weight ≤ 2 | 11 | 11 | 3.780e-1 |
| weight ≤ 3 | 15 | 15 | 2.391e-1 |
| weight ≤ 4 | 16 | 16 | 8.882e-16 |

Exact agreement at every `k`. The target is reached only at `k = 4`, where the
ball is the whole 16-dimensional space — that is full diagonalization wearing an
A-CASE costume, not a control worth pricing against. Raising the v1 cap would
not have helped: growth stalls when the ball is exhausted, not when the cap
binds, which is why `max_size` 10 through 24 all return the same basis.

The v1 pool (`pauli_orbit` over the Hamiltonian's own words) and every stronger
Pauli-family variant tested — commutator response, compound products, all
weight ≤ 2, the full response hierarchy — sit inside this same ceiling.

**The lesson is about regime, not strength.** A-CASE is built for a reference
whose correlated space excitation operators reach efficiently. A product state
under a Pauli pool is not that. The fix is to run the incumbent where it is
designed to work, not to hand it a bigger Pauli pool.

## 2. What changes

**Molecular instances with the excitation pool.** `excitation_multivectors`
singles and doubles on an RHF determinant — A-CASE's design regime. On
`h4_chain` (8 qubits, 256-dimensional, reference support on 12 distinct
energies) the incumbent reaches 7.66e-4 with a basis of 15 and a 7 927-word
universe. Both arms reach the target, so the comparison is uncensored and the
word universe is large enough for the cost question to be real.

**H2/STO-3G is demoted to a diagnostic.** Its singlet sector holds two
configurations at any geometry, so reference support is 2 and `m = 2` spans the
answer exactly. v1 declared that risk in advance and it duly produced a PASS
that meant nothing. A trivial instance cannot be a required decision instance.

**An admission criterion, enforced before execution.** For every required
instance the gate recomputes, in exact arithmetic, that the incumbent *and* at
least one exact-propagation real-time arm reach the target within their declared
caps. An instance failing admission cannot be preregistered as required. This is
the check whose absence produced v1's censored verdict.

This criterion reads only **zero-noise reachability**. It never inspects a noisy
comparison, a shot count, or a ratio — those are the result. The line is that a
control must be *able* to reach the target before it is worth asking what it
costs to.

**It biases toward the incumbent, deliberately.** Admission keeps instances
where A-CASE works and drops instances where it does not. That direction is
conservative: it makes the candidate's job harder. Selecting instances where the
*candidate* wins would be the bias to avoid, and nothing here does that.

**Per-arm basis grids, matched on accuracy.** v1 applied one basis-size grid to
every arm, which priced arms at equal basis size rather than equal accuracy.
Each arm now uses the grid its own construction needs and the comparison is
total modelled shots at the same target.

## 3. Admissibility, measured before freezing

`h4_chain` at four spacings, target 1.6e-3 Ha, A-CASE capped at 16:

| instance | support | A-CASE error | words | real-time `m*` | components | admissible |
|---|---:|---:|---:|---:|---:|:--|
| `h4_chain(0.75)` | 12 | 3.057e-4 | 7 927 | 3 | 9 | yes |
| `h4_chain(0.9)` | 12 | 7.659e-4 | 7 927 | 4 | 13 | yes |
| `h4_chain(1.0)` | 11 | 1.356e-3 | 7 927 | 4 | 13 | yes |
| `h4_chain(1.1)` | 12 | 2.329e-3 | 7 927 | 4 | 13 | **no** |

The required set is **every geometry that admits**: `h4_chain(0.75)`,
`h4_chain(0.9)` and `h4_chain(1.0)`. `h4_chain(1.0)` admits by only a small
margin, and it is included anyway. Dropping an admissible instance because its
margin looks thin would be selecting on the outcome the experiment has not
measured yet, which is the failure this document exists to avoid. If it fails
under noise, that is a result.

`h4_chain(1.1)` and `h4_chain(1.5)` are diagnostics carrying a declared
expectation of censoring: the singles-and-doubles pool saturates at a basis of
15 at every cap from 16 to 26, which is the ordinary static-correlation limit of
SD at stretched geometry, not a defect of the run.

**Scope limit, stated plainly.** Three geometries of one molecule is a narrow
required set. It is what the admission criterion leaves once trivial and
censoring instances are excluded, and a GO earned on it would be evidence about
H4 under this cost model — not about chemistry, and not about scaling.

Because the required set is no longer a pair, the combination table generalizes:
all required instances PASS gives GO, all FAIL gives NO_GO, anything else is
CONDITIONAL. The v1 two-instance table is the special case.

## 4. What does not change

The cost model, budget grid, allocation contract, noise model, regularizers,
evidence vocabulary, decision ladder, and combination table carry over from v1
unchanged, so the two experiments remain comparable where they overlap. The
prespecified follow-up is still exactly one estimator-variance refinement.

No hardware, compiled-circuit, scaling, or quantum-advantage claim follows from
this design, and any record produced under it must carry
`quantum_advantage_claim: false`.

## 5. Order

The config and its gate are committed first and pass before any producer runs,
as in v1. `check_phase16b_v2_preregistration.py` verifies that ordering from git
history, and additionally enforces the admission criterion in Section 2.
