# Where the H₄ and BeH₂ setting counts actually go

The dyadic block-commuting hierarchy (`run_clifford_hierarchy.py`, §6.6 of
`PLAN.md`) reports one reduction per instance: the number of measurement
settings falls from the qubit-wise-commuting rung to the fully-commuting one,
and logical CX gates are paid for it.

| | `W` | `G(k=1)` | `G(k=2)` | `G(k=4)` | `G(k=8)` | `qwc_to_full` |
|---|---:|---:|---:|---:|---:|---:|
| H₄ (`M=9`) | 7 371 | 913 | 647 | 238 | 64 | 0.9299 |
| BeH₂ (`M=5`) | 1 815 | 353 | 41 | 26 | 14 | 0.9603 |

Those are the frozen numbers, and `benchmarks/probe_grouping_refinement.py`
reproduces them before it varies anything. The question this note answers is
what limits them — the compatibility *rule*, or the two choices the producer
makes around it.

Both, it turns out, but not where the record's framing suggests. The rule is
close to exhausted at `k = 1` and nowhere near it in the interior, and the
interior loss is split between the colouring and a choice nobody priced: which
qubits share a block.

## 1. The QWC rung is nearly done; stop optimizing it

At `k = 1` a setting is one basis choice per qubit, so a word with a
non-identity letter on **every** qubit is readable by exactly one setting — its
own letters — and two distinct full-weight words name two distinct settings.
The count of full-weight words is therefore a hard floor on `G(k=1)`
(`regrouping.qwc_forcing_bound`):

| | full-weight words | `G(k=1)` shipped | best found | gap to floor |
|---|---:|---:|---:|---:|
| H₄ | 804 | 913 | 881 | +9.6 % |
| BeH₂ | 304 | 353 | 353 | +16.1 % |

Every strategy below — DSATUR, iterated greedy, span recovery, an exhaustive
greedy set cover over all `3⁸ = 6561` tensor-product bases — lands in the same
band. The exhaustive TPB cover is *worse* than the shipped greedy on H₄ (1 011
settings), which is the useful negative result: at QWC the bank's own weight
distribution, not the grouping heuristic, sets the count. Work spent here buys
at most ~10 %, and only for H₄.

## 2. The interior rungs are colouring-limited

Two measurements say the shipped colouring leaves room.

**The tail.** The greedy processes words in descending conflict degree and
drops each into the first group that fits, which produces a long tail of tiny
settings. At `k = 1` BeH₂'s median group holds **one** word and 189 of its 353
settings are singletons; H₄ has 164 singletons.

**The span slack.** A setting's diagonalizer sends its group to `Z`-only words,
and the block-restricted symplectic form is bilinear and alternating, so the
whole GF(2) span of the group is block-commuting and diagonal too. A setting
therefore *reads* its group's span — and the shipped partitions assign it far
less than that:

| | `k=1` | `k=2` | `k=4` | `k=8` |
|---|---:|---:|---:|---:|
| H₄ words read / words assigned | 2.55× | 2.61× | 2.62× | 2.03× |
| BeH₂ words read / words assigned | 2.09× | 1.96× | 1.86× | 1.82× |

Charging each setting for everything it reads and covering the bank with those
spans (`regrouping.span_recover`) is a strict reduction at no synthesis cost.
Re-colouring in group order (`regrouping.iterated_greedy`, Culberson) is the
bigger lever and is monotone by construction — a first fit that sees each
current group consecutively can never need more groups. `regrouping.dsatur_partition`
re-decides the order after every placement instead of fixing it once, and wins
at the ends of the ladder where the fixed order carries least information.

## 3. The interior rungs are frame-limited, and nobody priced that

The rule compares restrictions to **contiguous** `k`-qubit blocks. Which qubits
those are is fixed by the register's labelling, not by the operator content —
and the record's own logical model is *all-to-all logical Clifford circuits, no
routing*, under which a relabelling is free. So the block assignment is a free
parameter that the frozen rungs happen to fix at the identity.

Sweeping all 105 pairings (`k=2`) and all 35 splits (`k=4`):

| instance | rung | contiguous | best frame | worst frame | best assignment |
|---|---:|---:|---:|---:|---|
| H₄ | `k=2` | 647 | **456** | 807 | `{0,1}{2,6}{3,4}{5,7}` |
| H₄ | `k=4` | 238 | **110** | 378 | `{0,2,4,6}{1,3,5,7}` |
| BeH₂ | `k=2` | 41 | 41 | 241 | contiguous |
| BeH₂ | `k=4` | 26 | 26 | 68 | contiguous |

H₄'s best `k=4` frame is exactly the **spin split**: under the interleaved
convention (`2p` = α, `2p+1` = β, `CONVENTIONS.md`) `{0,2,4,6}` is every α spin
orbital and `{1,3,5,7}` every β one. And it is not a settings-for-gates trade —
it is strictly better on every column the record prices:

| H₄ `k=4` | settings | CX/sweep | mean CX/setting | max `D_CX` |
|---|---:|---:|---:|---:|
| contiguous | 238 | 3 277 | 13.77 | 14 |
| spin-split | **105** | **1 135** | **10.81** | **9** |

The instance spread has a mechanical explanation. BeH₂'s retained basis is four
pure *spatial-pair* excitations — `E(4,5<-0,1)`, `E(6,7<-0,1)`, `E(4,5<-2,3)`,
`E(6,7<-2,3)` — and each pair `(2p, 2p+1)` is contiguous, so contiguous blocks
already match the generator family. H₄'s basis also carries `E(4,7<-0,3)`,
`E(5,6<-1,2)`, `E(5,6<-0,3)` and `E(4,7<-1,2)`, whose α and β halves move
independently across spatial orbitals; those factor by spin, not by spatial
pair. The best frame is the one that puts each generator's support inside one
block, and it differs by instance — which is why it has to be searched, not
assumed. BeH₂'s own best `k=4` frame, once refinement is applied, is
`{0,1,4,5}{2,3,6,7}` — spatial orbitals `{0,2}` and `{1,3}`, i.e. the source and
target of two of its four excitations.

This is the finding with consequences beyond the setting count: the `22×`
break-even spread between H₄ and BeH₂ at `k = 2` that §6.6 reports "under an
identical logical model, before any device enters" is measured with H₄ in a bad
frame and BeH₂ in its best one. The spread is real, but part of it prices the
labelling rather than the instance.

## 4. What the alternatives are worth

`reduce_settings` runs the shipped greedy, DSATUR and an isotropic cover as
seeds, refines each with iterated greedy and span recovery, and searches the
block assignments. Everything below is re-synthesized by
`synthesize_block_settings`, so it passes the same `Z`-only invariant and the
CX columns are the record's own quantities.

<!-- RESULTS TABLE -->

## 5. What was tried and did not pay

* **Exhaustive tensor-product-basis set cover** at `k = 1` (all `3⁸` settings,
  greedy max coverage): 353 on BeH₂ (tie) and 1 011 on H₄ (worse). §1 explains
  why.
* **The isotropic cover on its own** at narrow blocks: 1 494 settings on H₄ at
  `k = 1` against 913. Growing a subspace one generator at a time is the wrong
  move when a single generator already fixes the basis on every qubit it
  touches. It is the *best* seed at `k = n`, where each generator costs only one
  of `n` dimensions, which is why it is kept as a seed rather than dropped.
* **Tighter lower bounds.** `ceil(W / 2ⁿ)` is 29 for H₄ and 8 for BeH₂ at every
  block size — a Lagrangian per block multiplies out to dimension `n` whatever
  the blocks are, so the packing floor does not move along the ladder. Refining
  it to `ceil(W / max words any one setting reads)` gives 34/30 for H₄ at
  `k=1,4` and 10/8 for BeH₂, still far under what any construction reaches. A
  linear-programming floor would say more, and nothing here computes one: no
  number in this note is a minimum.

## 6. Reproducing

```bash
pip install -e '.[stim,research]'
python benchmarks/probe_grouping_refinement.py                  # both instances
python benchmarks/probe_grouping_refinement.py --system h4 --block-size 4
```

The probe freezes no record and gates nothing; the frozen hierarchy record is
untouched, `block_commuting_partition` still returns its committed colouring,
and `benchmarks/check_clifford_hierarchy.py` still regenerates the committed
numbers digit for digit.
