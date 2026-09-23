# Phase 16B, third preregistration: pricing the incumbent with grouping

The v2 experiment returned GO under a cost model that charges every Pauli word
its own shots. That model's largest exclusion was measurement grouping, and the
exclusion is symmetric in the rules but asymmetric in effect: the incumbent's
words collapse into far fewer settings, while each real-time component needs
its own circuit. The merged v2 description says so, and reports a post-hoc
sensitivity in which `h4_chain_100` falls to 0.87 and the verdict would become
CONDITIONAL.

Post-hoc arithmetic on a frozen record is not an experiment. This is the third
declaration, and it asks the question properly: **priced with grouping, does the
real-time family still reduce modelled shots by at least 10x against A-CASE?**

The v1 and v2 configs and records stay exactly as committed. Nothing here
changes how either is read.

## 1. What the v2 model left out, measured

Group counts on this experiment's own banks, not borrowed from another record:

| partition | A-CASE word universe (`h4_chain`, basis 15) | reduction | H (184 traceless terms) | reduction |
|---|---:|---:|---:|---:|
| ungrouped | 7 927 | 1.00x | 184 | 1.00x |
| QWC (`k = 1`) | 913 | 8.68x | 68 | 2.71x |
| block-commuting `k = 2` | 596 | 13.30x | — | — |
| block-commuting `k = 4` | 223 | 35.55x | — | — |
| fully commuting (`k = 8`) | 65 | 121.95x | 9 | 20.44x |

The v2 post-hoc sensitivity borrowed 7 371 -> 913 -> 64 from
`benchmarks/GROUPING_REDUCTION.md`, a different bank on the same molecule. The
measured values above are close to it, so that estimate was well founded — but
it was an estimate, and this experiment replaces it with the bank's own numbers.

## 2. Covariance is carried, not assumed away

Grouping makes correlation material. Words sharing a setting are read from the
same shots, and every pencil entry is a linear combination of word expectations,
so their covariance enters the entry's variance. Assuming independence inside a
group would be an unlabelled approximation of exactly the kind this programme
exists to refuse.

It does not have to be assumed. The reference is an RHF determinant, and on all
three required instances it is exactly the computational basis state
`|11110000>` (verified, purity 1.0, one nonzero amplitude). Measuring it in any
product basis therefore factorises over qubits: a `Z` lane is deterministic, an
`X` or `Y` lane is a fair coin, and lanes are independent. Group readouts are
exactly samplable at negligible cost, and every word in the group is a parity of
the same sampled bitstring.

So this experiment **simulates the readouts** rather than propagating a variance
bound: per setting, draw its shot budget of bitstrings from the exact product
distribution in that setting's shared basis, and estimate every word in the
group as a parity mean of those draws. The resulting word estimates carry the
true within-group covariance for this reference.

The precondition is checked, not assumed: the gate refuses any required instance
whose reference is not a computational basis state, because the sampling
argument above fails for a superposition and the honest treatment would then
need the full joint distribution.

## 3. Setting model per arm

What one "setting" costs is where a grouping comparison is won or lost, so each
arm's model is declared here rather than derived in the producer.

| arm | settings for one evaluation | grouping available |
|---|---|---|
| `acase_control` | one per group of the retained bank's word universe | yes — the whole point |
| `power_krylov_control` | one per group of the union of `supp(H^p)` | yes |
| `rt_unitary` | `2m` — real and imaginary parts of `c(1..m)` | **no** |
| `rt_hermitian` | `2(m-1)` for `c`, plus `m * G_H` for `d` | partial, on `d` only |
| `rt_trotter` | `2(m-1)` for `S`, plus the full Hermitian pencil priced as `d` | partial |

`rt_unitary` gets no grouping and that is not an oversight: each `(lag, part)`
is a distinct controlled-evolution circuit with its own ancilla basis, so two
lags cannot be read from one setting. `rt_hermitian`'s `d(k)` inserts a Pauli
term of `H` into the Hadamard test, and terms that commute under the active
scheme can share a setting at fixed lag — so `d` is charged `m * G_H` rather
than `m * |H|`. That is a deliberate concession to the candidate, declared in
advance, and it is the only grouping any real-time arm receives.

Allocation stays uniform per setting, as in v2. A total budget `N` splits
equally across an arm's settings; a bounded ancilla observable then has standard
error at most `sqrt(settings / N)`, and a sampled group's words inherit whatever
precision their `N / settings` shots give them.

## 4. What is frozen from v2, and what is new

Carried over unchanged so the two experiments remain comparable: the target
(1.6e-3 Ha), the three required `h4_chain` instances and their diagnostics, the
admission criterion, per-arm basis grids, the budget decades, 200 replicas, the
status ladder, the combination rule, and the prespecified follow-up.

New: the grouping schemes (`ungrouped`, `qwc`, `fully_commuting`), the setting
model in Section 3, and simulated group readouts in place of a variance bound
for every arm whose estimands are Pauli expectations on the reference. The
`ungrouped` scheme is retained deliberately — it should reproduce v2's ordering,
and a disagreement there is a defect in this producer, not a finding.

## 5. What this can and cannot conclude

A GO here means the real-time family keeps a 10x modelled-shot advantage after
the incumbent is given its grouping. A NO-GO or CONDITIONAL means the v2 GO was
an artifact of a cost model that overcharged the incumbent — which the v2
description already flags as possible, and which would be the more useful
finding.

Still excluded for every arm alike, and still not claimed: gate depth, state
preparation, circuit compilation, hardware throughput, and the ancilla overhead
of a Hadamard test beyond its shot count. Three geometries of one molecule
remains a narrow base. No hardware, scaling, or quantum-advantage claim follows.

## 6. Order

Config and gate commit first and pass before any producer runs, as in v1 and v2.
`check_phase16b_v3_preregistration.py` verifies that ordering from git history
and re-derives admission, and adds two checks no earlier gate needed: the
Section 2 basis-state precondition on every required instance, and the whole of
the Section 1 table, recomputed from the instances rather than read back. The
second matters because `G_H` is priced — `rt_hermitian` charges `d` at
`m * G_H` — so a group count is a term in the comparison, not a footnote.

The gate was tested against 28 deliberate mutations of the frozen config, among
them a superposition reference promoted to required, `ungrouped` promoted to a
primary scheme, covariance downgraded from simulated readouts to a bound, an
inadmissible instance promoted to required, the exact ground energy leaked into
the incumbent's stopping rule, and each declared group count moved in the
direction that would flatter one side. All 28 were rejected.
