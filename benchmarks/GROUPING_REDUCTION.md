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
makes around it: one constructive colouring, laid out over contiguous blocks.

Both, it turns out, but not where the record's framing suggests. The rule is
close to exhausted at `k = 1` and nowhere near it in the interior, and the
interior loss splits between the colouring and a choice nobody priced: which
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

Every strategy below — DSATUR, iterated greedy, span recovery, and an
exhaustive greedy set cover over all `3⁸ = 6561` tensor-product bases — lands in
the same band. The exhaustive TPB cover is *worse* than the shipped greedy on
H₄ (1 011 settings), which is the useful negative result: at QWC the bank's own
weight distribution, not the grouping heuristic, sets the count. Work spent
here buys at most ~10 %, and only for H₄.

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
current group consecutively can never need more groups.
`regrouping.dsatur_partition` re-decides the order after every placement
instead of fixing it once, and wins at the ends of the ladder where the fixed
order carries least information.

## 3. The interior rungs are frame-limited, and nobody priced that

The rule compares restrictions to **contiguous** `k`-qubit blocks. Which qubits
those are is fixed by the register's labelling, not by the operator content —
and the record's own logical model is *all-to-all logical Clifford circuits, no
routing*, under which a relabelling is free. So the block assignment is a free
parameter that the frozen rungs happen to fix at the identity.

Sweeping every block assignment with the shipped greedy
(`regrouping.search_block_order`: 105 pairings at `k=2`, 35 splits at `k=4`):

| instance | rung | frames | contiguous | best | worst | best assignment |
|---|---:|---:|---:|---:|---:|---|
| H₄ | `k=2` | 105 | 647 | **460** | 802 | `{0,1}{2,6}{3,4}{5,7}` |
| H₄ | `k=4` | 35 | 238 | **105** | 384 | `{0,2,4,6}{1,3,5,7}` |
| BeH₂ | `k=2` | 105 | 41 | 41 | 241 | contiguous |
| BeH₂ | `k=4` | 35 | 26 | 26 (3-way tie) | 56 | contiguous |

H₄'s best `k=4` frame under the greedy is exactly the **spin split**: under the
interleaved convention (`2p` = α, `2p+1` = β, `CONVENTIONS.md`) `{0,2,4,6}` is
every α spin orbital and `{1,3,5,7}` every β one. And the frame alone — before
any refinement — is not a settings-for-gates trade. It is strictly better on
every column the record prices:

| H₄ `k=4`, shipped greedy | settings | CX/sweep | mean CX/setting | max `D_CX` |
|---|---:|---:|---:|---:|
| contiguous | 238 | 3 277 | 13.77 | 14 |
| spin-split | **105** | **1 135** | **10.81** | **9** |

The instance spread has a mechanical explanation. BeH₂'s retained basis is four
pure *spatial-pair* excitations — `E(4,5<-0,1)`, `E(6,7<-0,1)`, `E(4,5<-2,3)`,
`E(6,7<-2,3)` — and each pair `(2p, 2p+1)` is contiguous, so contiguous blocks
already match the generator family and no relabelling helps. H₄'s basis also
carries `E(4,7<-0,3)`, `E(5,6<-1,2)`, `E(5,6<-0,3)` and `E(4,7<-1,2)`, whose α
and β halves move independently across spatial orbitals; those have no
contiguous home, and the spin split is where they find one.

**The greedy ranking is a proxy, not the answer.** Refinement reorders it: at
H₄ `k=4` the greedy-best spin split (105) refines to 96, while the
greedy-*third* frame `{0,1,4,5}{2,3,6,7}` (116) refines to **93** and wins. That
is why `reduce_settings` refines a shortlist rather than the single best-ranked
frame, and why the shortlist is a real parameter rather than a formality.

This is also the finding with consequences beyond the setting count. The frozen
records give a `k = 2` break-even ratio `(G₁ − G_k)/N_CX` of 0.0949 on H₄ and
2.7857 on BeH₂ — a 29× spread "under an identical logical model, before any
device enters". But H₄ is measured in a poor frame and BeH₂ in its best one.
Refined, the same ratios are 0.3474 and 2.7857, and the spread falls to **8.0×**.
The spread is real; part of it prices the labelling rather than the instance.

## 4. What the alternatives are worth

`reduce_settings` runs the shipped greedy, DSATUR and an isotropic cover as
seeds, refines each with iterated greedy and span recovery, and searches the
block assignments. Everything below is re-synthesized by
`synthesize_block_settings`, so it passes the same `Z`-only invariant and the
CX columns are the record's own quantities.

| | `k` | `G` frozen → best | | `N_CX` frozen → best | max `D_CX` | winning frame |
|---|---:|---:|---:|---:|---:|---|
| H₄ | 1 | 913 → **881** | 1.04× | 0 → 0 | 0 → 0 | contiguous |
| H₄ | 2 | 647 → **295** | 2.19× | 2 803 → 1 687 | 4 → 4 | `{0,1}{2,6}{3,4}{5,7}` |
| H₄ | 4 | 238 → **93** | 2.56× | 3 277 → 1 095 | 14 → 12 | `{0,1,4,5}{2,3,6,7}` |
| H₄ | 8 | 64 → **55** | 1.16× | 1 886 → 1 836 | 35 → 39 | contiguous |
| BeH₂ | 1 | 353 → 353 | 1.00× | 0 → 0 | 0 → 0 | contiguous |
| BeH₂ | 2 | 41 → 41 | 1.00× | 112 → 112 | 1 → 1 | contiguous |
| BeH₂ | 4 | 26 → **21** | 1.24× | 193 → 150 | 10 → 9 | `{0,1,4,5}{2,3,6,7}` |
| BeH₂ | 8 | 14 → **9** | 1.56× | 296 → 213 | 23 → 28 | contiguous |

Six of the eight rungs improve, and five of those six take fewer settings *and*
fewer gates *and* no more depth — the interior of the ladder was not sitting on
a trade. The exceptions are the two `k = 8` rungs, which are genuine trades:
BeH₂ buys 36 % fewer settings for 22 % more two-qubit depth, H₄ 14 % for 11 %.

BeH₂'s `k = 8` rung ends at 9 settings against a packing floor of 8 — 1 815
words at up to 256 per setting — so that rung is essentially closed. H₄'s ends
at 55 against a floor of 29, which is where the remaining headroom is.

Restated in the record's own headline, `preparation_reduction_fraction` moves
from 0.9299 to **0.9376** on H₄ and from 0.9603 to **0.9745** on BeH₂ — modest,
because that statistic divides by the QWC rung, which §1 shows is nearly fixed.
The per-rung counts above are what actually moved.

## 5. What was tried and did not pay

* **Exhaustive tensor-product-basis set cover** at `k = 1` (all `3⁸` settings,
  greedy max coverage): 353 on BeH₂ (tie) and 1 011 on H₄ (worse). §1 explains
  why.
* **The isotropic cover on its own** at narrow blocks: 1 492 settings on H₄ at
  `k = 1` against 913, and refining it costs more than every other seed
  together. Growing a subspace one generator at a time is the wrong move when a
  single generator already fixes the basis on every qubit it touches. It is the
  *best* seed at `k = n`, where each generator costs only one of `n` dimensions,
  which is why it is kept as a seed behind a cost guard rather than dropped.
* **Drawing isotropic-cover generators from the whole bank** rather than only
  from unassigned words: within two settings either way on all eight rungs. Not
  worth the extra scan.
* **Tighter lower bounds.** `ceil(W / 2ⁿ)` is 29 for H₄ and 8 for BeH₂ at every
  block size — a Lagrangian per block multiplies out to dimension `n` whatever
  the blocks are, so the packing floor does not move along the ladder. Refining
  it to `ceil(W / max words any one setting reads)` gives 34 and 30 for H₄ at
  `k=1,4` and 10 and 8 for BeH₂, still far under what any construction reaches.
  A linear-programming floor would say more, and nothing here computes one: no
  number in this note is a minimum.

## 6. Reproducing

```bash
pip install -e '.[stim,research]'
python benchmarks/probe_grouping_refinement.py                  # both instances
python benchmarks/probe_grouping_refinement.py --system h4 --block-size 4
```

H₄'s `k = 2` rung is the slow one: ranking 105 frames costs about two and a
half minutes before any refinement starts.

The probe freezes no record and gates nothing; the frozen hierarchy record is
untouched, `block_commuting_partition` still returns its committed colouring,
and `benchmarks/check_clifford_hierarchy.py` still regenerates the committed
numbers digit for digit.
