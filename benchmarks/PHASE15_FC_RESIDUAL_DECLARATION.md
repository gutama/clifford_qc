# Q18-S4: the measured residual's cost under fully commuting grouping

## Status and disclosure

This is a separate, post-hoc declaration. Q18-S1 read `NONE` on all five
licensed banks: under single-assignment QWC with uniform shots, a residual norm
costs 17 to 184,000 times its energy's shots, against a bar of ten. Q18 and
Q18-S1 both excluded fully commuting grouping from cost, because it "would
change both sides". Q18-S4 prices that endpoint. Every Q18 and Q18-S1 ratio was
seen first, so declaring the config before its execution protects identity and
reproducibility; it does not make the inference preregistered. Its config is
`configs/phase15_fc_residual.json`, and its gate is
`check_phase15_fc_residual_declaration.py`.

## Question

On each of the four 8-qubit licensed banks, does estimating the ground Ritz
root's residual norm `σ` from shared word means under single-assignment fully
commuting settings, to the same standard error as its energy measured the same
way, cost at most ten times the energy's shots?

## What is unchanged

The estimand, the estimator, its linearization (the Ritz-vector response
included), Q18-S1's four-step Richardson derivative validator, single
assignment, uniform shots, the statistic

```text
R = (G_U · V_σ²) / (4 σ² · G_SH · V_E)
```

the tenfold bar, the bank ladder and the combination rule are Q18-S1's, word for
word. The gate refuses any change to them. Only the partition changes, and with
it both sides' setting counts `G` and one-shot variances `V`.

## The partitions

Both sides use fully commuting settings, so the ratio compares like with like.
Each protocol partitions the measured universe listed in ascending code order,
identity excluded, as Q18 lists it:

- **Declared: `fc_greedy`.** `block_commuting_partition(n, codes, n)`, the
  package's shipped colouring, with one block spanning the register. Words go
  largest conflict degree first, ties by code, into the first setting they
  commute with.
- **Alternative: `fc_isotropic_cover`.** `isotropic_cover(n, codes, n)`, the
  regrouping module's strongest construction at block size `n`. Each setting
  grows as an isotropic subspace. Every generator is the compatible word that
  brings the most unassigned words into its span, and the setting takes every
  unassigned word its span contains.

Q18 declared two QWC constructions so that no status rests on one heuristic;
these are two constructions of the fully commuting rule for the same reason.
Both are deterministic, and neither claims a minimum setting count.

**Variance.** A fully commuting setting is read through a Clifford
diagonalizer, so each one-shot outcome is an eigenvalue of `Σ_{w∈g} f_w P_w`.
Its variance is `Var_ψ` of that sum, the formula Q18 applies to a QWC setting,
computed by the same `run_mapping_axis._functional_variance`. Every setting
variance must also equal `⟨O²⟩ − ⟨O⟩²` from matrix-free action on the
reference statevector, within `10⁻⁹` of `max(1, ⟨O²⟩, ⟨O⟩²)`. The checker
recomputes each one a third way, from its own Pauli action.

## Banks

H₄, converged H₄, BeH₂ and Hubbard 2×2, each at the frozen basis Q18 and Q18-S1
use. Their first moments are frozen, together with the fully commuting setting
counts of each `(S, H)` universe:

| bank | raw `\|U_SH\|` | raw `\|U\|` | `σ` | `G_SH`, greedy | `G_SH`, cover | `G_SH`, QWC (Q18-S1) |
|---|---|---|---|---|---|---|
| h4 | 7,371 | 8,192 | 0.0756 | 64 | 55 | 913 |
| h4_converged | 7,927 | 8,192 | 0.0498 | 65 | 55 | 913 |
| beh2 | 1,815 | 2,048 | 0.00234 | 14 | 11 | 353 |
| hubbard_2x2 | 5,537 | 13,940 | 1.52 *t* | 146 | 112 | 1,406 |

**H₂O CAS(8e,6o) is outside this successor.** Both partitioners build a
`W × W` conflict-mask table. At H₂O's 868,149 measured words that is about
`7.5 × 10¹¹` pairs, and the package ships no scalable fully commuting cover.
Building one would be a new heuristic, not this question. H₂O's Q18-S1 reading
under QWC, `R = 1,579`, stands.

## Decision rule and consequences

The ladder and combination rule are Q18-S1's. `R ≤ 10` under both protocols is
AFFORDABLE, `R > 10` under both is PROHIBITIVE, and a split is
GROUPING_SENSITIVE. A failed deterministic check makes a bank INVALID, and any
INVALID bank makes the verdict INVALID. Every bank AFFORDABLE gives FULL, no bank
AFFORDABLE gives NONE, and anything else is RESTRICTED. All three verdicts are
reachable: no frozen number settles a bank in advance.

| verdict | what follows |
|---|---|
| FULL | Phase 15 builds the finite-shot second-moment estimator on the four banks with fully commuting settings. Its sampled validation is a new preregistration that freezes seeds, budgets and coverage targets. This licenses building the estimator, not any sampled claim, and covers the banks' determinant references only. |
| RESTRICTED | The estimator is built and validated only on the AFFORDABLE banks, which become its scope by name. Extending it needs a new declaration. |
| NONE | No finite-shot second-moment estimator under uniform single-assignment fully commuting settings on these four banks. With Q18-S1's NONE, the uniform single-assignment estimator closes negative at both grouping endpoints the package ships, on these banks. Exact residuals, convergence reports and folded roots remain. |
| INVALID | No verdict. The record names the failed check. |

## Deterministic checks

A bank is INVALID unless all of these hold:

- every frozen number recomputes, and every premise flag holds;
- the raw combined universe has the committed size and SHA-256 and contains the
  identity;
- both protocols' `(S, H)` partitions have the frozen setting counts;
- both functionals reproduce Q18-S1's record: the same word counts, and norms,
  `σ²` and ground energy within `10⁻⁹` relative;
- each partition assigns every measured word to exactly one setting, and every
  setting is pairwise commuting;
- `σ²` at the exact means equals the validation record's, and `⟨ψ|G|ψ⟩` is zero,
  each within `10⁻⁹ (1 + κ)`;
- Q18-S1's Richardson rule passes on both derivative directions;
- no setting variance is negative beyond rounding, and every one agrees with the
  statevector;
- the record carries clean provenance from a declared environment.

## What this can and cannot conclude

`R` stays a first-order, asymptotic ratio read from exact reference variances.
Under fully commuting settings it prices shots, and nothing else:

- **Not the diagonalizers.** Fully commuting settings need entangling Clifford
  circuits. Their two-qubit gates, depth and fidelity, and so device time, are
  not priced. Both sides need them, but in different numbers.
- **Not a correlated reference.** Every bank's reference is a determinant, a
  stabilizer state. A fully commuting setting whose words act on it nearly as a
  scalar has a variance near zero. On the undeclared dimer, that is why the two
  protocols disagree by four orders of magnitude. A prepared correlated state
  would change both sides.
- **Not span reading or multiple assignment.** A setting can read its group's
  GF(2) span, but every word is assigned once.
- **Not** the plug-in bias of `σ²`, finite-sample intervals, bootstrap
  coverage, another allocation or threshold, H₂O, or any bank outside the five.

## Disclosure

Seen before this declaration:
- every Q18 and Q18-S1 record value, including the QWC ratios and Q18-S1's
  oracle allocation diagnostics;
- the Q18-S2 and Q18-S3 records;
- Phase 14b's energy-side result on BeH₂: under fully commuting grouping its
  covariance-aware variance was 2.22 times lower at matched shots.

Before freezing, the declared banks saw first moments only. That means each
`(S, H)` solve, and both fully commuting partitions of each `(S, H)` universe,
which give the frozen `G_SH` counts. No second-moment row, residual functional,
combined-universe partition or setting variance was formed on a declared bank.

The fully commuting pricing was developed on the undeclared Hubbard dimer
(`M = 3`). There the greedy gives `R = 0.0019` and the cover `R = 37`, and every
setting variance matches a dense statevector computation to `10⁻¹⁴`. Both
partitioners were timed on random 8-qubit word sets of 2,047, 8,191 and 13,939
words, taking at most 3.1 s each.

## Execution

The declaration, gate, helpers, producer, independent checker and tests land
together. The producer re-runs the gate and refuses unless the declaration is
committed, no record exists, the tree is clean and the environment is a
declared one. Run it once, commit the record alone, and then update the ledger.
No follow-up is permitted: another threshold, protocol, allocation, reference
or bank set would be a new declaration.
