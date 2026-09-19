"""Lowering the setting count of a block-commuting partition.

:mod:`clifford_qc.measurement.block_commuting` owns the compatibility rule and
one constructive colouring of it: largest conflict degree first, then first
fit. That colouring is an upper bound on the setting count, and on the two
frozen hierarchy instances it is a loose one. This module keeps the rule fixed
and attacks the bound, so every partition it returns is measurable by exactly
the block-local diagonalizers :mod:`clifford_qc.measurement.block_synthesis`
already synthesizes, and the ``k`` axis keeps its meaning.

Three levers, in increasing order of what they assume:

**Re-colouring.** :func:`dsatur_partition` re-decides the order after every
placement instead of fixing it once, and :func:`iterated_greedy` re-runs first fit in an order that
presents the current groups consecutively. Culberson's argument gives the
monotonicity for free -- such an order can never need more groups than the
colouring it was read from -- so the pass is safe to iterate and the result is
never worse than its seed.

**Span closure.** A setting's diagonalizer sends every member of its group to a
``Z``-only word, and the block-restricted symplectic form is bilinear and
alternating, so the whole GF(2) span of the group is block-commuting and lands
in the computational basis too. A setting therefore *reads* its group's span,
not just the words the colouring inserted -- on the frozen banks, two to two
and a half times as many. :func:`span_recover` covers the bank with those spans
and drops the settings nothing needs, which is a strict count reduction with no
new synthesis.

**Frame choice.** The rule is stated over *contiguous* blocks, so which qubits
share a block is fixed by the register's labelling rather than by the operator
content. Under the declared all-to-all logical model a relabelling is free, and
:func:`search_block_order` searches the block assignments directly: on H4 at
``k = 4`` the spin-split frame costs less than half the settings the contiguous
frame does, and fewer CX gates with it.

Nothing here claims a minimum. Exact colouring and exact set cover are both
NP-hard; these are better constructions of the same upper bound, and the
counting bound ``ceil(W / 2**n)`` is the only floor any of them may be measured
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np

from .block_commuting import (
    block_commuting_partition,
    conflict_masks,
    first_fit_partition,
    packed_conflicts,
)

# The span routines index a dense lookup over the whole ``4**n`` code space.
# At the hierarchy's ``n = 8`` that is 65536 entries; the guard is stated in
# qubits rather than bytes so the refusal names the thing the caller controls.
MAX_SPAN_QUBITS = 12


@dataclass(frozen=True)
class Regrouping:
    """A partition, the frame it is read in, and how it was built.

    ``order`` is a qubit relabelling: position ``j`` of every code in ``codes``
    carries the letter the *input* bank put on qubit ``order[j]``. A caller that
    prices this partition must synthesize against ``codes``, not against the
    bank it passed in, and must undo ``order`` before reporting per-qubit
    quantities. ``groups`` indexes ``codes``, which is the same indexing as the
    input bank: the relabelling permutes letters, never words.
    """

    block_size: int
    order: tuple[int, ...]
    codes: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    provenance: str

    @property
    def n_settings(self) -> int:
        return len(self.groups)


def permuted_codes(n: int, codes: Sequence[int], order: Sequence[int]) -> list[int]:
    """Relabel qubits so position ``j`` carries the letter of ``order[j]``."""
    order = tuple(order)
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of the register")
    out = []
    for code in codes:
        moved = 0
        for j, q in enumerate(order):
            moved |= ((code >> (2 * q)) & 3) << (2 * j)
        out.append(moved)
    return out


def block_orders(n: int, block_size: int) -> Iterator[tuple[int, ...]]:
    """Every distinct block assignment, as a qubit order for contiguous blocks.

    One representative per *set partition* of the register into blocks: the
    words within a block and the order of the blocks do not change which pairs
    the rule compares, so enumerating permutations instead would repeat each
    assignment ``(k!)**(n/k) * (n/k)!`` times. At ``block_size`` 1 or at least
    ``n`` the rule does not see the labelling at all, and only the identity is
    yielded.
    """
    if block_size < 1:
        raise ValueError("block size must be at least 1")
    identity = tuple(range(n))
    if block_size == 1 or block_size >= n:
        yield identity
        return
    if n % block_size:
        raise ValueError(
            "block-order search needs a block size dividing the register; "
            f"got {block_size} for {n} qubits")

    def partitions(items: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], ...]]:
        if not items:
            yield ()
            return
        head, rest = items[0], items[1:]
        for others in combinations(rest, block_size - 1):
            block = (head,) + others
            remaining = tuple(i for i in rest if i not in others)
            for tail in partitions(remaining):
                yield (block,) + tail

    for blocks in partitions(identity):
        yield tuple(q for block in blocks for q in block)


def iterated_greedy(
    n: int,
    codes: Sequence[int],
    block_size: int,
    groups: Sequence[Sequence[int]],
    *,
    rounds: int = 64,
    seed: int = 0,
    conflicts: Sequence[int] | np.ndarray | None = None,
) -> list[list[int]]:
    """Re-colour repeatedly in group order; never returns more groups.

    Each round concatenates the current groups into a vertex order and re-runs
    first fit. Because every group stays contiguous in that order, first fit
    cannot need more groups than it was given, so the sequence is monotone and
    the only question a round settles is whether it is strictly better. Four
    schedules alternate -- smallest group first, largest first, reversed, and a
    seeded shuffle -- because a fixed one stalls on its own fixed point.
    """
    codes = list(codes)
    if not codes:
        return []
    if conflicts is None:
        conflicts, _ = conflict_masks(n, codes, block_size)
    # Packed once: every round scans the open groups for each word, so the
    # lane form is what keeps a many-round refinement affordable at H4's width.
    if not isinstance(conflicts, np.ndarray):
        conflicts = packed_conflicts(conflicts, len(codes))
    best = [list(group) for group in groups]
    rng = np.random.default_rng(seed)
    for step in range(rounds):
        blocks = [list(group) for group in best]
        schedule = step % 4
        if schedule == 0:
            blocks.sort(key=len)
        elif schedule == 1:
            blocks.sort(key=len, reverse=True)
        elif schedule == 2:
            blocks.reverse()
        else:
            rng.shuffle(blocks)
        trial = first_fit_partition(
            conflicts, [i for block in blocks for i in block])
        if len(trial) > len(best):
            raise AssertionError(
                "iterated greedy grew the setting count, which a group-ordered "
                "first fit cannot do")
        if len(trial) < len(best):
            best = trial
    return best


def dsatur_partition(
    n: int,
    codes: Sequence[int],
    block_size: int,
    *,
    conflicts: Sequence[int] | None = None,
    degrees: np.ndarray | None = None,
) -> list[list[int]]:
    """Colour by saturation degree: most-constrained word next.

    The shipped greedy fixes its order once, from the conflict degrees of the
    whole bank; DSATUR re-decides after every placement, taking the word whose
    settings are most nearly exhausted and breaking ties on conflict degree and
    then on the code. It is the better seed at the two ends of the ladder --
    where the shipped order is least informative -- and the worse one in the
    middle, which is why :func:`reduce_settings` runs both rather than choosing.

    The saturation sets are tracked as colour bitmasks over the packed conflict
    planes, so nothing here materializes the ``W**2`` conflict matrix.
    """
    codes = list(codes)
    width = len(codes)
    if not width:
        return []
    if conflicts is None or degrees is None:
        conflicts, degrees = conflict_masks(n, codes, block_size)
    degrees = np.asarray(degrees, dtype=np.int64)

    # Work in code order so the first argmax winner is also the tie-break
    # winner, which keeps the colouring a function of the bank alone.
    rank = np.argsort(np.asarray(codes, dtype=np.int64), kind="stable")
    ranked_degrees = degrees[rank]
    scale = int(ranked_degrees.max()) + 1
    saturation = np.zeros(width, dtype=np.int64)
    live = np.ones(width, dtype=bool)
    # ``forbidden[c, v]``: colour c already sits on a neighbour of v. A boolean
    # plane per colour rather than a bitmask per word, because the saturation
    # update touches every neighbour of the word just coloured -- thousands of
    # them on these banks -- and that has to be one array operation, not a loop.
    capacity = 8
    forbidden = np.zeros((capacity, width), dtype=bool)
    groups: list[list[int]] = []
    for _ in range(width):
        score = np.where(live, saturation * scale + ranked_degrees, -1)
        position = int(np.argmax(score))
        vertex = int(rank[position])
        open_colours = len(groups)
        column = forbidden[:open_colours, position]
        if open_colours and not bool(column.all()):
            colour = int(np.argmin(column))
        else:
            colour = open_colours
            if colour == capacity:
                capacity *= 2
                grown = np.zeros((capacity, width), dtype=bool)
                grown[:open_colours] = forbidden[:open_colours]
                forbidden = grown
            groups.append([])
        groups[colour].append(vertex)
        live[position] = False
        neighbours = _unpack(conflicts[vertex], width)[rank] & live
        newly = neighbours & ~forbidden[colour]
        forbidden[colour] |= newly
        saturation += newly
    return groups


def _code_slots(n: int, codes: Sequence[int]) -> dict[int, list[int]]:
    """Bank positions of each distinct code, so duplicates stay addressable."""
    slots: dict[int, list[int]] = {}
    for index, code in enumerate(codes):
        slots.setdefault(int(code), []).append(index)
    return slots


def group_span(codes: Sequence[int]) -> list[int]:
    """Every GF(2) combination of ``codes``.

    Phase-free Pauli multiplication *is* XOR on these packed codes: the low bit
    of a letter is ``x ^ z`` and the high bit is ``z``, and both compose by XOR
    under the product. The span of a block-commuting group is block-commuting
    because the block-restricted symplectic form is bilinear and alternating,
    which is what makes this set readable by the group's own diagonalizer.
    """
    span = [0]
    seen = {0}
    for code in codes:
        code = int(code)
        if code in seen:
            continue
        span += [element ^ code for element in span]
        seen = set(span)
    return span


def span_recover(
    n: int,
    codes: Sequence[int],
    block_size: int,
    groups: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Cover the bank with the groups' spans and drop what nothing needs.

    Greedy max coverage over the settings already synthesized: at each step the
    surviving setting that reads the most still-unassigned words takes them.
    Every returned group is a subset of the span of one input group, so it is
    block-commuting, and the result is a partition of the same bank into at
    most as many settings.
    """
    if n > MAX_SPAN_QUBITS:
        raise ValueError(
            f"span recovery is declared for at most {MAX_SPAN_QUBITS} qubits; "
            f"got {n}")
    codes = list(codes)
    if not codes:
        return []
    slots = _code_slots(n, codes)
    reads: list[list[int]] = []
    for group in groups:
        members: list[int] = []
        for element in group_span([codes[i] for i in group]):
            members.extend(slots.get(element, ()))
        reads.append(members)

    unassigned = np.ones(len(codes), dtype=bool)
    chosen: list[list[int]] = []
    while unassigned.any():
        best_index, best_gain, best_members = -1, 0, []
        for index, members in enumerate(reads):
            taken = [i for i in members if unassigned[i]]
            if len(taken) > best_gain:
                best_index, best_gain, best_members = index, len(taken), taken
        if best_index < 0:
            raise AssertionError(
                "the spans of a partition must cover the bank it partitions")
        unassigned[best_members] = False
        reads[best_index] = []
        chosen.append(sorted(best_members))
    if len(chosen) > len(list(groups)):
        raise AssertionError("span recovery grew the setting count")
    return chosen


def isotropic_cover(
    n: int, codes: Sequence[int], block_size: int
) -> list[list[int]]:
    """Build settings directly as spans that cover the most unassigned words.

    A setting is a block-isotropic subspace, and what it reads is that
    subspace, so the object to maximize while growing one is the span's
    coverage of the bank -- not the number of words a colouring happens to
    insert. Each generator is the still-compatible word whose addition brings
    the most new bank words into the span. This is the one strategy here that
    ignores the colouring entirely, and on the frozen banks it is the strongest
    at ``k = n`` and the weakest at ``k = 1``, where a single generator already
    fixes the basis on every qubit it touches.
    """
    if n > MAX_SPAN_QUBITS:
        raise ValueError(
            f"the isotropic cover is declared for at most {MAX_SPAN_QUBITS} "
            f"qubits; got {n}")
    codes = list(codes)
    if not codes:
        return []
    conflicts, _ = conflict_masks(n, codes, block_size)
    slots = _code_slots(n, codes)
    array = np.asarray(codes, dtype=np.int64)
    open_word = np.zeros(1 << (2 * n), dtype=bool)
    open_word[array] = True

    unassigned = np.ones(len(codes), dtype=bool)
    groups: list[list[int]] = []
    while unassigned.any():
        span = np.zeros(1, dtype=np.int64)
        allowed = unassigned.copy()
        blocked = 0
        while len(span) < 1 << n:
            pool = np.nonzero(allowed)[0]
            if pool.size == 0:
                break
            candidates = array[pool]
            fresh = ~np.isin(candidates, span)
            pool, candidates = pool[fresh], candidates[fresh]
            if pool.size == 0:
                break
            gain = open_word[candidates[:, None] ^ span[None, :]].sum(axis=1)
            best = int(np.argmax(gain))
            if gain[best] == 0:
                break
            pick = int(candidates[best])
            span = np.concatenate([span, span ^ np.int64(pick)])
            blocked |= conflicts[int(pool[best])]
            allowed &= ~_unpack(blocked, len(codes))
        members: list[int] = []
        for element in span.tolist():
            members.extend(i for i in slots.get(int(element), ()) if unassigned[i])
        if not members:
            raise AssertionError("the isotropic cover made no progress")
        unassigned[members] = False
        open_word[array[members]] = False
        groups.append(sorted(members))
    return groups


def _unpack(mask: int, width: int) -> np.ndarray:
    """A conflict bitmask as a boolean array over bank positions."""
    raw = mask.to_bytes((width + 7) // 8, "little")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8),
                         count=width, bitorder="little").astype(bool)


def search_block_order(
    n: int,
    codes: Sequence[int],
    block_size: int,
    *,
    limit: int | None = None,
) -> list[tuple[tuple[int, ...], int]]:
    """Rank block assignments by the shipped greedy's setting count.

    Returns ``(order, settings)`` ascending, ties broken by the order itself so
    the ranking is a function of the bank alone. The greedy is the ranking
    signal rather than the final answer because it is the cheap one: refining
    every assignment costs as much as refining the winner many times over, and
    on both frozen instances the assignment the greedy prefers is the one that
    survives refinement.
    """
    ranked: list[tuple[tuple[int, ...], int]] = []
    for index, order in enumerate(block_orders(n, block_size)):
        if limit is not None and index >= limit:
            break
        moved = permuted_codes(n, codes, order)
        ranked.append((order, len(block_commuting_partition(n, moved, block_size))))
    ranked.sort(key=lambda row: (row[1], row[0]))
    return ranked


def reduce_settings(
    n: int,
    codes: Sequence[int],
    block_size: int,
    *,
    rounds: int = 64,
    seed: int = 0,
    search_frames: bool = True,
    shortlist: int = 3,
) -> Regrouping:
    """The best partition these strategies find, with the frame that carries it.

    Runs, per shortlisted block assignment: the shipped greedy, then iterated
    greedy, then span recovery, then iterated greedy again, and -- because it
    wins at the wide end -- an isotropic cover refined the same way. The best
    setting count wins; ties keep the earlier frame, so the identity frame is
    preferred whenever a relabelling buys nothing.
    """
    codes = list(codes)
    if not codes:
        return Regrouping(block_size, tuple(range(n)), (), (), "empty bank")
    frames = [tuple(range(n))]
    if search_frames:
        ranked = search_block_order(n, codes, block_size)
        frames = [order for order, _ in ranked[:max(1, shortlist)]]
        if tuple(range(n)) not in frames:
            frames.append(tuple(range(n)))

    best: Regrouping | None = None
    for order in frames:
        moved = permuted_codes(n, codes, order)
        conflicts, degrees = conflict_masks(n, moved, block_size)
        seeds = {
            "greedy": block_commuting_partition(n, moved, block_size),
            "DSATUR": dsatur_partition(n, moved, block_size,
                                       conflicts=conflicts, degrees=degrees),
        }
        planes = packed_conflicts(conflicts, len(moved))
        if n <= MAX_SPAN_QUBITS:
            seeds["isotropic cover"] = isotropic_cover(n, moved, block_size)
        for name, groups in seeds.items():
            refined = iterated_greedy(n, moved, block_size, groups,
                                      rounds=rounds, seed=seed,
                                      conflicts=planes)
            if n <= MAX_SPAN_QUBITS:
                recovered = span_recover(n, moved, block_size, refined)
                if len(recovered) < len(refined):
                    refined = iterated_greedy(n, moved, block_size, recovered,
                                              rounds=rounds, seed=seed,
                                              conflicts=planes)
            candidate = Regrouping(
                block_size=block_size,
                order=order,
                codes=tuple(moved),
                groups=tuple(tuple(group) for group in refined),
                provenance=(
                    f"{name} seed, iterated greedy ({rounds} rounds, seed "
                    f"{seed}) with span recovery, frame {order}"),
            )
            if best is None or candidate.n_settings < best.n_settings:
                best = candidate
    assert best is not None
    return best


def qwc_forcing_bound(n: int, codes: Sequence[int]) -> int:
    """A floor for the ``k = 1`` rung: one setting per full-weight word.

    At QWC a setting is a choice of one basis per qubit, so a word carrying a
    non-identity letter on *every* qubit is read by exactly one setting -- its
    own letters -- and two distinct full-weight words name two distinct
    settings. The count of full-weight words in the bank is therefore a lower
    bound on the QWC setting count, and a far stronger one than
    :func:`counting_bound`: on both frozen hierarchy banks it lands within a
    sixth of what the shipped greedy already achieves, which is the evidence
    that the QWC rung is close to done and the interior rungs are not.

    Stated for ``block_size = 1`` only. A ``k``-qubit block's maximal isotropic
    subspace can hold several full-weight restrictions -- three of them at
    ``k = 2`` -- so the same counting argument at a wider block gives a floor
    weaker by that factor, and this function does not pretend to compute it.
    """
    full = {int(code) for code in codes
            if all((int(code) >> (2 * q)) & 3 for q in range(n))}
    return len(full)


def counting_bound(n: int, codes: Sequence[int]) -> int:
    """The floor every strategy here is measured against.

    A setting reads at most the ``2**n`` elements of one maximal isotropic
    subspace, at *every* block size -- a Lagrangian per block multiplies out to
    dimension ``n`` whatever the blocks are -- so no partition of ``W`` distinct
    words can use fewer than ``ceil(W / 2**n)`` settings. It is a weak bound and
    it is the only one stated here: exact set cover is NP-hard and nothing in
    this module may be reported as a minimum.
    """
    distinct = len(set(int(code) for code in codes))
    if not distinct:
        return 0
    return -(-distinct // (1 << n))
