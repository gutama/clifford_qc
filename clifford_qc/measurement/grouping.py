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
from ..pauli_kernel import pauli_lane_mask


def _nonident_lanes(n: int, code: int) -> int:
    """Bit plane (one bit per lane) marking qubits where ``code`` is not I."""
    lo = pauli_lane_mask(n)
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


def _conflict_degrees(n: int, codes: tuple[int, ...]) -> list[int]:
    """How many words each word is QWC-*incompatible* with.

    The all-pairs count is irreducible here: compatibility is agreement of two
    partial letter assignments on the intersection of their supports, and
    counting agreeing pairs of partial functions has no known sub-quadratic
    form. What it does not have to be is ``O(w^2)`` *interpreted* work. At
    ``w = 13646`` -- an eight-qubit A-CASE element universe at ``M = 9`` -- the
    Python double loop is 1.9e8 iterations and takes about a hundred seconds,
    which is what confined certified growth to four qubits. One vectorized row
    per word is the same arithmetic at numpy speed.

    Falls back to the interpreted loop when a packed code will not fit a signed
    64-bit lane (``n > 31``), rather than letting numpy wrap.
    """
    w = len(codes)
    if w < 2:
        return [0] * w
    if 2 * n > 63:
        degrees = [0] * w
        for i in range(w):
            ci = codes[i]
            for j in range(i + 1, w):
                if not _qwc_codes(n, ci, codes[j]):
                    degrees[i] += 1
                    degrees[j] += 1
        return degrees

    import numpy as np

    lo = np.int64(pauli_lane_mask(n))
    packed = np.fromiter(codes, dtype=np.int64, count=w)
    nz = (packed & lo) | ((packed >> np.int64(1)) & lo)
    degrees = np.zeros(w, dtype=np.int64)
    # Row blocks, not one w x w matrix: the full boolean matrix at w = 13646 is
    # 186 MB, and blocking is the same arithmetic in a few MB.
    block = max(1, (1 << 22) // w)
    for start in range(0, w, block):
        stop = min(start + block, w)
        diff = packed[start:stop, None] ^ packed[None, :]
        nz_d = (diff & lo) | ((diff >> np.int64(1)) & lo)
        conflict = (nz[start:stop, None] & nz[None, :] & nz_d) != 0
        degrees[start:stop] = conflict.sum(axis=1)
    return degrees.tolist()


def _place(n: int, codes: Sequence[int], order: Sequence[int],
           bases: list[int], members: list[list[int]]) -> None:
    """First-fit placement of ``order`` into the groups held in ``bases``.

    ``bases`` carries one *merged* code per group, and testing a word against it
    is exactly as strong as scanning every member. Within a group all pairs
    qubit-wise commute, so at each lane every non-identity member agrees and the
    merged code holds that agreed letter; a word conflicts with the merged code
    iff it conflicts with some member. Both directions are immediate, so this
    is the same predicate as the per-word incompatibility bitset it replaces --
    and it costs two machine-word operations per group instead of an ``O(w)``-bit
    mask test, which is what lets the ``O(w^2)`` bitset build go away.

    Mutates ``bases`` and ``members`` so a partition can be *extended*: pass the
    state of an existing partition and only the new indices in ``order``.
    """
    if len(order) * max(len(bases), 1) > _VECTOR_PLACEMENT_WORK and 2 * n <= 63:
        _place_vectorized(n, codes, order, bases, members)
        return
    for i in order:
        code = codes[i]
        for g, basis in enumerate(bases):
            if _qwc_codes(n, code, basis):
                bases[g] = basis | code
                members[g].append(i)
                break
        else:
            bases.append(code)
            members.append([i])


# Above this much scanning work (words x groups) the interpreted first-fit loop
# costs more than numpy's per-call overhead. Purely a performance switch: both
# paths implement the same first-fit and a test pins them to the same partition.
_VECTOR_PLACEMENT_WORK = 200_000


def _place_vectorized(n: int, codes: Sequence[int], order: Sequence[int],
                      bases: list[int], members: list[list[int]]) -> None:
    """``_place`` with the group scan vectorized; same first-fit, same output.

    Once the groups number in the thousands, the placement scan is the whole
    cost -- 13646 words against 2357 groups is 3.2e7 interpreted iterations.
    Testing a word against every current basis at once and taking the *first*
    compatible index (``argmax`` on the boolean, which is exactly first-fit)
    keeps the semantics and moves the loop into numpy.
    """
    import numpy as np

    lo = np.int64(pauli_lane_mask(n))
    one = np.int64(1)
    size = max(len(bases) + len(order), 1)
    basis_arr = np.zeros(size, dtype=np.int64)
    nz_arr = np.zeros(size, dtype=np.int64)
    count = len(bases)
    if count:
        basis_arr[:count] = np.fromiter(bases, dtype=np.int64, count=count)
        nz_arr[:count] = (basis_arr[:count] & lo) | ((basis_arr[:count] >> one) & lo)

    for i in order:
        code = np.int64(codes[i])
        nz_code = (code & lo) | ((code >> one) & lo)
        live_nz = nz_arr[:count]
        diff = basis_arr[:count] ^ code
        nz_d = (diff & lo) | ((diff >> one) & lo)
        free = (nz_code & live_nz & nz_d) == 0
        if count and free.any():
            g = int(np.argmax(free))  # first compatible group, i.e. first fit
            basis_arr[g] |= code
            nz_arr[g] = (basis_arr[g] & lo) | ((basis_arr[g] >> one) & lo)
            members[g].append(i)
        else:
            basis_arr[count] = code
            nz_arr[count] = nz_code
            members.append([i])
            count += 1
    bases[:] = [int(value) for value in basis_arr[:count]]


@lru_cache(maxsize=8192)
def _qwc_partition(n: int, codes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Greedy largest-degree-first QWC partition of ``codes`` (in the given
    order). Returns groups as tuples of codes. Memoized on ``(n, codes)``.

    Identical in output to a plain letter-by-letter greedy, and to every earlier
    version of this function: same stable descending-degree ordering, same
    placement predicate (see ``_place``). Only the cost changed, which matters
    because committed records carry group counts.
    """
    degrees = _conflict_degrees(n, codes)
    order = sorted(range(len(codes)), key=lambda i: -degrees[i])  # stable
    bases: list[int] = []
    members: list[list[int]] = []
    _place(n, codes, order, bases, members)
    return tuple(tuple(codes[i] for i in group) for group in members)


def qwc_groups(words: Sequence[PauliWord]) -> list[list[PauliWord]]:
    """Greedy largest-degree-first partition of ``words`` into QWC groups.

    Graph coloring on the finite simple QWC-incompatibility graph (vertices
    are distinct Pauli words; an edge joins a non-QWC pair): words are
    processed in descending conflict degree and placed into the first
    compatible group. The returned group count is therefore a constructive
    upper bound on the chromatic number, not a proof of the minimum circuit
    count (exact coloring is NP-hard). The partition is cached on the exact
    word set, so repeated candidate sets reuse it.
    """
    # A measurement partition is over distinct observables.  Duplicate inputs
    # would otherwise manufacture multiple assignments for the same word.
    words = list({word.code: word for word in words}.values())
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
