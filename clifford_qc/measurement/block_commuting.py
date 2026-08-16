"""Dyadic block-commuting measurement grouping.

Two Pauli words share a measurement setting under this rule when their
restrictions to every contiguous ``k``-qubit block commute. The rule is a
one-parameter family spanning the two endpoints the measurement literature
usually treats as separate protocols:

``k = 1``
    Each block is a single qubit, and two single-qubit Paulis commute exactly
    when they are equal or one is identity -- the qubit-wise-commuting rule of
    :mod:`clifford_qc.measurement.grouping`.
``k >= n``
    One block covers the register, and the rule is ordinary commutation, i.e.
    fully commuting groups.

Interior ``k`` interpolates: more words share a setting than at QWC, and the
diagonalizing circuit stays block-local, so its two-qubit gates never cross a
block boundary. That locality is the point -- it is what makes the entangling
cost of the setting a function of ``k`` rather than of the register width, and
it is why the hierarchy is priced as a protocol axis in its own right.

The compatibility test is packed. From the ``x``/``z`` masks of two words,
``(x_a & z_b) ^ (z_a & x_b)`` carries one bit per qubit marking a local
anticommutation contribution; the pair commutes on a block iff that bit plane
has even parity when restricted to the block's mask. The greedy partition
computes one such bit plane per word against all words at once, then places
words into groups by big-integer mask test.

This module owns the compatibility rule alone. Synthesizing the block-local
Clifford diagonalizer for a group, and costing it, is the caller's job --
:mod:`clifford_qc.measurement.cost` prices the result.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..ir import PauliWord


def block_ranges(n: int, block_size: int) -> tuple[tuple[int, int], ...]:
    """The ``(start, size)`` pairs of the contiguous blocks covering ``n``.

    The final block is truncated when ``block_size`` does not divide ``n``, so
    the blocks always partition the register exactly.
    """
    if n < 0:
        raise ValueError("qubit count must be non-negative")
    if block_size < 1:
        raise ValueError("block size must be at least 1")
    return tuple(
        (start, min(block_size, n - start)) for start in range(0, n, block_size)
    )


def _xz(n: int, code: int) -> tuple[int, int]:
    """Split a packed Pauli code into its ``x`` and ``z`` qubit masks."""
    x = z = 0
    for q in range(n):
        letter = (code >> (2 * q)) & 3
        if letter in (1, 2):
            x |= 1 << q
        if letter in (2, 3):
            z |= 1 << q
    return x, z


def _parity_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized uint64 parity, compatible with the declared NumPy >=1.23."""
    work = np.array(values, dtype=np.uint64, copy=True)
    work ^= work >> 32
    work ^= work >> 16
    work ^= work >> 8
    work ^= work >> 4
    work ^= work >> 2
    work ^= work >> 1
    return (work & np.uint64(1)).astype(bool)


def block_wise_commute(a: PauliWord, b: PauliWord, block_size: int) -> bool:
    """Whether ``a`` and ``b`` commute inside every contiguous block.

    At ``block_size = 1`` this is
    :func:`~clifford_qc.measurement.grouping.qubit_wise_commute`; at
    ``block_size >= a.n`` it is ordinary commutation.
    """
    if a.n != b.n:
        raise ValueError("words act on different qubit counts")
    if block_size < 1:
        raise ValueError("block size must be at least 1")
    xa, za = _xz(a.n, a.code)
    xb, zb = _xz(b.n, b.code)
    cross = (xa & zb) ^ (za & xb)
    for start, size in block_ranges(a.n, block_size):
        mask = ((1 << size) - 1) << start
        if bin(cross & mask).count("1") % 2:
            return False
    return True


def block_commuting_partition(
    n: int, codes: Sequence[int], block_size: int
) -> list[list[int]]:
    """Largest-conflict-degree greedy partition of ``codes`` into settings.

    Returns groups of *indices into* ``codes``, in the order the greedy opened
    them. Indices are the contract rather than words, because callers need to
    map each word back to the setting that reads it; duplicate codes are
    therefore kept and each occurrence is assigned.

    Words are processed in descending conflict degree, ties broken by code, and
    placed into the first group they do not conflict with. The group count is a
    constructive upper bound on the chromatic number of the incompatibility
    graph, not a proof of the minimum setting count -- exact coloring is
    NP-hard, and no result derived from this partition may claim a minimum.
    """
    if block_size < 1:
        raise ValueError("block size must be at least 1")
    codes = list(codes)
    width = len(codes)
    if not width:
        return []
    if n and max(codes) >= 1 << (2 * n):
        raise ValueError("a code carries letters beyond the declared qubit count")
    if n > 64:
        raise ValueError("block-commuting partition requires at most 64 qubits")

    xs = np.asarray([_xz(n, code)[0] for code in codes], dtype=np.uint64)
    zs = np.asarray([_xz(n, code)[1] for code in codes], dtype=np.uint64)
    block_masks = [
        np.uint64(((1 << size) - 1) << start)
        for start, size in block_ranges(n, block_size)
    ]

    degrees = np.empty(width, dtype=np.int32)
    conflicts: list[int] = []
    for i in range(width):
        # One bit per qubit marks a local anticommutation contribution.
        cross = (xs[i] & zs) ^ (zs[i] & xs)
        bad = np.zeros(width, dtype=bool)
        for mask in block_masks:
            bad |= _parity_u64(cross & mask)
        degrees[i] = int(bad.sum())
        packed = np.packbits(bad, bitorder="little")
        conflicts.append(int.from_bytes(packed.tobytes(), "little"))

    order = sorted(range(width), key=lambda i: (-int(degrees[i]), codes[i]))
    groups: list[list[int]] = []
    group_masks: list[int] = []
    for i in order:
        conflict = conflicts[i]
        for group_index, member_mask in enumerate(group_masks):
            if conflict & member_mask == 0:
                groups[group_index].append(i)
                group_masks[group_index] = member_mask | (1 << i)
                break
        else:
            groups.append([i])
            group_masks.append(1 << i)

    # Independent grouping invariant: no member conflicts with its group mask.
    for members, member_mask in zip(groups, group_masks):
        for i in members:
            if conflicts[i] & member_mask:
                raise AssertionError("block-commuting partition invariant failed")
    return groups


def block_commuting_groups(
    words: Sequence[PauliWord], block_size: int
) -> list[list[PauliWord]]:
    """Partition ``words`` into groups commuting inside every ``k``-qubit block.

    The word-level counterpart of :func:`block_commuting_partition`, mirroring
    :func:`~clifford_qc.measurement.grouping.qwc_groups`: a measurement
    partition is over distinct observables, so duplicate inputs are collapsed
    before grouping rather than being assigned twice.
    """
    original = list(words)
    if not original:
        return []
    n = original[0].n
    if any(word.n != n for word in original):
        raise ValueError("words act on different qubit counts")
    unique = list({word.code: word for word in original}.values())
    codes = [word.code for word in unique]
    partition = block_commuting_partition(n, codes, block_size)
    return [[PauliWord(n, codes[index]) for index in group] for group in partition]
