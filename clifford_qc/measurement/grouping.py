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


def _first_compatible_full_bases(
    n: int,
    partial_codes,
    full_bases,
    *,
    work_items: int = 1 << 22,
):
    """Vectorized first compatible full basis, or ``-1`` when none exists."""
    import numpy as np

    partial = np.asarray(partial_codes, dtype=np.int64)
    bases = np.asarray(full_bases, dtype=np.int64)
    selected = np.full(len(partial), -1, dtype=np.int64)
    if not len(partial) or not len(bases):
        return selected
    lo = np.int64(pauli_lane_mask(n))
    block = max(1, int(work_items) // len(bases))
    for start in range(0, len(partial), block):
        stop = min(start + block, len(partial))
        values = partial[start:stop, None]
        nonidentity = (values & lo) | ((values >> np.int64(1)) & lo)
        difference = values ^ bases[None, :]
        difference_nonidentity = (
            (difference & lo) | ((difference >> np.int64(1)) & lo)
        )
        compatible = (nonidentity & difference_nonidentity) == 0
        found = compatible.any(axis=1)
        first = np.argmax(compatible, axis=1)
        selected[start:stop] = np.where(found, bases[first], -1)
    return selected


def qwc_basis_cover(
    words: Sequence[PauliWord],
    *,
    max_candidate_bases: int = 8192,
    refinement_passes: int = 2,
) -> list[list[PauliWord]]:
    """Scalable deterministic QWC cover for very wide word universes.

    The exact largest-conflict-degree greedy used by :func:`qwc_groups` needs
    all pairwise conflict degrees.  That contract is valuable for frozen small
    records but becomes quadratic at the 143k-word H2O R2b rung.  This routine
    constructs a different, explicitly labelled upper bound:

    1. use observed full-support words as candidate product-measurement bases;
    2. assign each partial word to the lexicographically first compatible
       candidate, or complete its identity lanes with the most frequent local
       non-identity letter;
    3. run a bounded number of deterministic refinement passes using the
       largest current groups as additional candidate bases.

    Every returned group is checked through the same packed QWC predicate.
    The result is a constructive circuit cover, not a chromatic-number claim,
    and must not be compared numerically with the older degree-greedy counts
    without naming the changed heuristic.
    """
    if isinstance(max_candidate_bases, bool) or not isinstance(max_candidate_bases, int):
        raise TypeError("max_candidate_bases must be an integer")
    if max_candidate_bases < 1:
        raise ValueError("max_candidate_bases must be positive")
    if isinstance(refinement_passes, bool) or not isinstance(refinement_passes, int):
        raise TypeError("refinement_passes must be an integer")
    if refinement_passes < 0:
        raise ValueError("refinement_passes must be non-negative")

    original = list(words)
    if not original:
        return []
    n = original[0].n
    if any(word.n != n for word in original):
        raise ValueError("words act on different qubit counts")
    unique = list({word.code: word for word in original}.values())
    if 2 * n > 63:
        # The packed NumPy implementation is intentionally bounded to int64.
        # Keep a valid dependency-light fallback rather than wrapping codes.
        return qwc_groups(unique)

    import numpy as np

    codes = np.asarray(sorted(word.code for word in unique), dtype=np.int64)
    lo = np.int64(pauli_lane_mask(n))
    nonidentity = (codes & lo) | ((codes >> np.int64(1)) & lo)
    full = np.sort(codes[nonidentity == lo])
    if len(full) > max_candidate_bases:
        full = full[:max_candidate_bases]
    assignments = _first_compatible_full_bases(n, codes, full)

    completion = codes.copy()
    for qubit in range(n):
        letters = (codes >> np.int64(2 * qubit)) & np.int64(3)
        counts = np.bincount(letters, minlength=4)
        # Identities do not define a measurement basis.  Ties among X/Y/Z are
        # resolved by their packed order because np.argmax returns the first.
        fill = int(np.argmax(counts[1:]) + 1)
        completion |= np.where(
            letters == 0, np.int64(fill << (2 * qubit)), np.int64(0)
        )
    assignments = np.where(assignments < 0, completion, assignments)

    for _ in range(refinement_passes):
        keys, inverse = np.unique(assignments, return_inverse=True)
        sizes = np.bincount(inverse)
        actual = np.zeros(len(keys), dtype=np.int64)
        for index, group_index in enumerate(inverse):
            actual[group_index] |= codes[index]
        order = sorted(
            range(len(keys)), key=lambda index: (-int(sizes[index]), int(keys[index]))
        )[:max_candidate_bases]
        candidates = keys[np.asarray(order, dtype=np.int64)]
        reassigned = _first_compatible_full_bases(n, actual, candidates)
        next_keys = np.where(reassigned < 0, keys, reassigned)
        updated = next_keys[inverse]
        if np.array_equal(updated, assignments):
            break
        assignments = updated

    keys, inverse = np.unique(assignments, return_inverse=True)
    groups: list[list[PauliWord]] = [[] for _ in range(len(keys))]
    for code, group_index in zip(codes, inverse):
        groups[int(group_index)].append(PauliWord(n, int(code)))
    for group in groups:
        basis = 0
        for word in group:
            if not _qwc_codes(n, word.code, basis):
                raise AssertionError("basis-cover grouping produced a QWC conflict")
            basis |= word.code
    return groups


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
