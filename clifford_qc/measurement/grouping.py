"""Qubit-wise-commuting (QWC) measurement grouping.

Two Pauli words qubit-wise commute when at every qubit their letters are
equal or at least one is identity. All words of a QWC group are diagonal
in one shared per-qubit measurement basis, so the whole group costs a
single measurement circuit: every shot yields one outcome for *every* word
in the group. Grouping therefore divides N_circuits and multiplies the
information per shot; the per-word marginal statistics are unchanged
(outcomes across words become correlated, which per-word variance
estimates are insensitive to).

The compatibility test is packed: from a word's 2-bit-per-qubit code we read
the "non-identity lane" bit plane, and two words are QWC-incompatible iff
some lane is non-identity in both *and* their letters differ there -- a
single ``(nz_a & nz_b & nz_{a^b}) != 0`` test, no per-qubit loop. The greedy
partition precomputes an incompatibility bitset per word so placement is a
big-integer mask test, and the finished partition is memoized on the word
set so a repeated candidate set (across ADAPT steps / after elimination)
reuses the grouping instead of recomputing it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from ..ir import PauliWord
from ..multivector import _lane_mask


def _nonident_lanes(n: int, code: int) -> int:
    """Bit plane (one bit per lane) marking qubits where ``code`` is not I."""
    lo = _lane_mask(n)
    return (code & lo) | ((code >> 1) & lo)


def _qwc_codes(n: int, a: int, b: int) -> bool:
    """QWC test on packed codes: compatible unless some lane is non-identity
    in both words with differing letters (``a^b`` non-identity there)."""
    nz_a = _nonident_lanes(n, a)
    nz_b = _nonident_lanes(n, b)
    nz_d = _nonident_lanes(n, a ^ b)
    return (nz_a & nz_b & nz_d) == 0


def qubit_wise_commute(a: PauliWord, b: PauliWord) -> bool:
    if a.n != b.n:
        raise ValueError("words act on different qubit counts")
    return _qwc_codes(a.n, a.code, b.code)


@lru_cache(maxsize=8192)
def _qwc_partition(n: int, codes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Greedy largest-degree-first QWC partition of ``codes`` (in the given
    order). Returns groups as tuples of codes. Memoized on ``(n, codes)``.

    Identical in output to a plain letter-by-letter greedy: an incompatibility
    bitset per word replaces the inner ``all(qwc(...))`` scan, and conflict
    degree drives the same stable ordering.
    """
    w = len(codes)
    incompat = [0] * w
    for i in range(w):
        ci = codes[i]
        for j in range(i + 1, w):
            if not _qwc_codes(n, ci, codes[j]):
                incompat[i] |= 1 << j
                incompat[j] |= 1 << i
    order = sorted(range(w), key=lambda i: -incompat[i].bit_count())  # stable
    group_masks: list[int] = []
    group_members: list[list[int]] = []
    for i in order:
        for g, mask in enumerate(group_masks):
            if mask & incompat[i] == 0:  # i commutes with every current member
                group_masks[g] = mask | (1 << i)
                group_members[g].append(i)
                break
        else:
            group_masks.append(1 << i)
            group_members.append([i])
    return tuple(tuple(codes[i] for i in members) for members in group_members)


def qwc_groups(words: Sequence[PauliWord]) -> list[list[PauliWord]]:
    """Greedy largest-degree-first partition of ``words`` into QWC groups.

    Graph coloring on the QWC-incompatibility graph: words are processed in
    descending conflict degree and placed into the first compatible group.
    Not optimal (that is NP-hard) but standard and effective. The partition
    is cached on the exact word set, so repeated candidate sets reuse it.
    """
    words = list(words)
    if not words:
        return []
    n = words[0].n
    if any(word.n != n for word in words):
        raise ValueError("words act on different qubit counts")
    partition = _qwc_partition(n, tuple(word.code for word in words))
    return [[PauliWord(n, code) for code in group] for group in partition]


def shared_basis(group: Sequence[PauliWord]) -> dict[int, str]:
    """The per-qubit measurement basis of a QWC group: qubit -> X/Y/Z.

    Qubits untouched by every word in the group are omitted.
    """
    basis: dict[int, str] = {}
    for w in group:
        for j in w.support():
            letter = w.letter(j)
            if basis.setdefault(j, letter) != letter:
                raise ValueError("group is not qubit-wise commuting")
    return basis
