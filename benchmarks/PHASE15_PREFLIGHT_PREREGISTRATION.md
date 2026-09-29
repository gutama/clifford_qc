# Phase 15 preregistration: the H² support/cost preflight

Phase 15 adds a `SecondMomentBank` for `K_ij = ⟨ψ|A_i† H² A_j|ψ⟩`. It supplies
true Ritz residual norms, energy variances and an independent convergence
criterion, none of which the projected `(S, H)` pair can supply (PLAN.md §4.4).
PLAN.md §5 asks for a support/cost preflight before the bank is built. If the
word universe is prohibitive, residual norms come from dense or matrix-free
oracles, and the measured bank is confined to declared small systems. This
declaration fixes, before any second-moment row exists, what "prohibitive"
means, on which banks it is read, and what each outcome licenses. Its config is
`benchmarks/configs/phase15_h2_preflight.json`, and its gate is
`benchmarks/check_phase15_preregistration.py`.

## 1. Question (Q16)

On each of the five frozen mapping-axis banks, add the second-moment rows over
the retained block. Then:

1. does the measured word universe stay within **ten times** the `(S, H)`
   universe the bank already measures; and
2. do the bank's resident coefficients, first and second moments together,
   stay within **96,107,619**, the largest count a committed run has held?

## 2. What a row is

The definition is operational, because a support count means nothing until the
product that produces it is fixed.

- `H2 = H * H`, formed once by the multivector product from the bank's own
  Hamiltonian, identity term included.
- `K_ij = A_i.dagger() * (H2 * A_j)` for `i ≤ j`. This mirrors the product
  order `MatrixElementBank._build_pair` uses for `A_i† (H A_j)`.
- A row's support is the keys of that multivector after the product's own
  pruning, `|c| > 1e-12`.
- Only the retained block counts: `M(M + 1)/2` rows over the frozen basis. A
  residual norm reads the Ritz vector's block, so rejected-candidate rows are
  outside the question.

`U_SH` is the `(S, H)` bank's word universe over the same block, and `T_SH`
its coefficient count over both row kinds. `U_K` is the union of the K rows'
supports, and `T_K` their coefficient count.

## 3. Banks

The banks are `h4`, `h4_converged`, `beh2`, `h2o_cas8e6o` and `hubbard_2x2`,
each at the `selected_labels` that `benchmarks/configs/mapping_axis.json`
freezes. They are built through the same `_build_model` and
`_selected_generators` that the mapping-axis record and both Phase 2M records
use. They are the banks the project already prices storage on, so a
second-moment price sits beside a committed first-moment one on the same
block. The gate rebuilds every `(S, H)` bank and requires `U_SH`, `T_SH`, the
basis size and the block to equal the committed Phase 2M-A ledger.

| bank | qubits | M | pairs | `\|H\|` | `\|U_SH\|` | `T_SH` |
|---|---|---|---|---|---|---|
| h4 | 8 | 9 | 45 | 185 | 7,371 | 52,794 |
| h4_converged | 8 | 15 | 120 | 185 | 7,927 | 129,954 |
| beh2 | 8 | 5 | 15 | 61 | 1,815 | 6,846 |
| h2o_cas8e6o | 12 | 9 | 45 | 551 | 143,117 | 351,080 |
| hubbard_2x2 | 8 | 9 | 45 | 21 | 5,537 | 8,334 |

The committed 14-qubit molecular banks, stretched BeH₂ at `M = 31` among
them, record no frozen basis, and rebuilding one means rerunning adaptive
growth. They are not declared here. A scaling preflight on them would be a
declaration of its own.

## 4. The two clauses and why these thresholds

**Words: `|U_SH ∪ U_K| ≤ 10 |U_SH|`.** A decade is the unit this repository
reads as material, as in Phase 16B's 10× modelled-shot rule. Second moments
within a decade of what the bank already measures extend an existing
measurement; beyond that they are a different measurement. The comparison is
exact, on integers.

**Storage: `T_SH + T_K ≤ 96,107,619`.** That is the stretched-BeH₂ `M = 31`
bank's coefficient count, recovered exactly as `cached_operator_bytes / 24` in
the Phase 2M-A ledger. It completed at 10.75 GiB peak RSS on the object
backend, and it is the largest count in the ledger. The gate checks all four
facts against the committed record. A bank at or below the anchor is one the
package has demonstrably carried. Above it, full construction waits for the
Phase 2M storage gate, as PLAN.md already requires.

## 5. The symmetry ceiling, and which banks can decide

Every declared Hamiltonian and generator conserves `N↑` and `N↓`. So each one
commutes with the parity `Z`-string over each spin block, and so does every
product of them, `K` rows included. A Pauli word commutes with a block's
`Z`-string exactly when it has an even number of `X`/`Y` letters on that
block. Under the interleaved convention, with up spins on even qubits, `4ⁿ/4`
words remain: 16,384 on 8 qubits and 4,194,304 on 12. The gate checks this
premise word by word on `H`, on every generator and on all of `U_SH`, with no
violation.

That ceiling decides, from committed numbers alone, which banks a clause can
bind:

- **Words.** A bank can fail only if `4ⁿ/4 > 10 |U_SH|`. On the four 8-qubit
  banks the ceiling is 2.1–9.0 times `U_SH`, so none can fail.
- **Storage.** A bank can fail only if `T_SH + (pairs)(4ⁿ/4)` exceeds the
  anchor. No 8-qubit bank reaches even 2.1 million.

H₂O CAS(8e,6o) can fail either clause: its ceiling is 29 times `U_SH`, and its
block could hold 189 million coefficients. **It is the one decisive bank.** The
four 8-qubit banks are eligible by the size of their sector, and their counts
are reported rather than tested. The reachable verdicts are therefore FULL
and RESTRICTED. NONE is kept for totality but cannot occur on this
population, and the gate requires the config to say so.

## 6. Pruning

The arithmetic is pure-Python IEEE floating point in a fixed order, so the
counts reproduce exactly on any platform. They are still conventional in one
place. A word whose coefficient is rounding residue from an exact cancellation
can sit just above the `1e-12` threshold. Both clauses are therefore evaluated
twice: at the declared threshold, and again after removing every coefficient of
magnitude at most `1e-9` from `H2` and from each row. The strict threshold is
three decades above the product's own, and below every declared Hamiltonian's
smallest coefficient (`2.05e-4` on H₂O), so it cannot remove a Hamiltonian
word. Strict pruning only removes words, so it can only turn a failing clause
into a passing one. A bank whose status differs between the two thresholds is
CONVENTION_SENSITIVE and is not eligible.

## 7. Decision rule

| bank status | condition |
|---|---|
| ELIGIBLE | both clauses pass at both thresholds |
| WORD_PROHIBITIVE | the word clause fails and storage passes, at both |
| STORAGE_PROHIBITIVE | storage fails and the word clause passes, at both |
| BOTH_PROHIBITIVE | both fail at both |
| CONVENTION_SENSITIVE | a clause fails at `1e-12` and passes at `1e-9` |

A failed deterministic check makes that bank INVALID. The checks are:

- the rebuilt `(S, H)` baseline equals the ledger;
- every word of `H2` and of each `K` row lies in the sector;
- each row, paired with the reference, reproduces `⟨ψ|A_i†H²A_j|ψ⟩` from dense
  statevectors within `1e-8 (1 + |value|)`;
- the block of those pairings is positive semidefinite, since it is the Gram
  matrix of `H A_j|ψ⟩`;
- no strict count exceeds its declared-threshold count;
- the record carries clean provenance.

Every bank ELIGIBLE gives **FULL**, and no bank ELIGIBLE gives **NONE**.
Anything else is **RESTRICTED**, and any INVALID bank gives **INVALID**.

| verdict | what follows |
|---|---|
| FULL | Phase 15 builds `SecondMomentBank` for the retained block and validates it on all five banks against a dense residual oracle. A bank outside the five still needs the Phase 2M storage gate, or its own preflight. |
| RESTRICTED | The bank is built and validated only on the eligible banks, which become Phase 15's small-system exception by name. Residual norms elsewhere come from dense or matrix-free oracles, and extending the measured bank needs a new declaration. |
| NONE | No measured bank; dense and matrix-free residual oracles only. Unreachable here. |
| INVALID | No verdict. The record names the failed check. |

## 8. What this can and cannot conclude

The counts are exact and structural. They price the words a measured
second-moment bank must estimate and the coefficients it must hold. They do not
price the grouping of the new universe: a greedy partition is quadratic in a
universe that can reach millions of words. Nor do they price shots, estimator
variance, circuits, wall time or hardware. Wall time and peak RSS are reported
as diagnostics that no clause reads. FULL does not say a measured residual is
affordable in shots. It says the bank is not ruled out by what it must measure
or hold. Nothing here bounds a bank outside the five.

## 9. Disclosure

Before freezing, only `(S, H)` quantities were computed on the declared
banks. Those are the numbers in `measured_before_freezing`, which the committed
ledger already carried, together with the sector premise. No `H²` product and no
second-moment row was formed on any declared bank. The row and sector code is
exercised in tests on the Hubbard dimer, which is not declared.

## 10. Revisions and order

**Revision 0** is this declaration. The producer
(`run_phase15_h2_preflight.py`) and the result checker
(`check_phase15_h2_preflight.py`) come after it, and the producer must re-run
this gate and refuse to count if it fails. Once a record exists, the gate
requires the config's last change to strictly precede the record's first
commit. It keeps the inputs bound by SHA-256 for good. It binds the
multivector product and the Pauli kernel only until the record exists; after
that the record's provenance names the commit that produced it. A revision
after this one must state that no second-moment count existed when it was
made. No follow-up is permitted: another threshold, bank set or row definition
would be a new declaration.
