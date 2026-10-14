"""Phase 2M-B -- packed CSR/SoA storage for the matrix-element bank's rows.

Phase 2M-A measured what the current representation costs: one retained nonzero
coefficient occupies ``85.24``--``96.19`` bytes as a ``dict[int, complex]``
entry, against the ``24`` a packed row would need. This module is the packed
row, and ``benchmarks/reference_results/bank_storage_ledger.json`` is the
baseline it has to beat.

**Generation order is part of the contract, not an implementation detail.**
``MV.trace_pairing`` iterates the *smaller* operand's dict in insertion order,
and on the frozen banks 54--84% of rows are smaller than the reference, so the
row drives the summation for most matrix entries. Re-inserting a row's words in
sorted order therefore changes where the floating-point additions land: measured
across the five frozen mapping-axis banks it moves 4--17% of ``S``/``H`` entries
by up to ``3.1e-13``. ``MatrixElementBank._build_pair`` promises those entries
reproduce ``solver.projected_matrices`` "to the last bit rather than merely to a
tolerance", so a packed row that sorted its words would quietly break that
promise on every committed record.

Each row is therefore stored twice over: ``indices`` ascending, so a probe is a
binary search rather than a scan, and ``generation`` giving the positions that
replay the order the geometric product actually emitted. Four bytes of index,
four of permutation and sixteen of coefficient is the same ``24`` bytes
``MV.memory_estimate`` has always modelled -- the permutation is not overhead
added to that model, it is what lets the model be reached without changing an
answer.

Two coefficient layouts are implemented because the plan asks for both to be
benchmarked: ``interleaved`` keeps one ``complex128`` array, ``soa`` keeps
separate ``float64`` real and imaginary arrays. They store the same sixteen
bytes per coefficient and differ only in access pattern.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np

from ..multivector import MV

# Coefficient layouts. ``interleaved`` is one complex128 array; ``soa`` splits
# it into real and imaginary float64 arrays. Same bytes, different access.
LAYOUTS = ("interleaved", "soa")

# Index width. The committed banks reach a word universe of 1.17e6, so a signed
# 32-bit index has three decades of headroom; the store refuses to grow past it
# rather than silently wrapping.
INDEX_DTYPE = np.int32
MAX_WORDS = 2 ** 31 - 1

# Bytes one packed coefficient occupies: an index, a permutation entry and the
# coefficient itself. This is the model ``MV.memory_estimate`` has always
# reported -- eight bytes of index beside sixteen of complex -- reached here as
# 4 + 4 + 16 rather than 8 + 16.
PACKED_BYTES_PER_COEFFICIENT = 24


class GlobalWordTable:
    """The one canonical code-to-index map every packed row shares, on numpy.

    A Pauli word that appears in forty rows is one entry here and forty
    four-byte indices there. The table is append-only: indices are handed out in
    first-seen order and never move, because a row already packed holds them.

    **It holds no Python objects, and that is the point.** The obvious
    implementation -- a ``list`` of codes beside a ``dict`` mapping code to
    index -- measured at ``101.4`` bytes per word: ``8.5`` of list pointers,
    ``36.9`` of dict slots, ``28.0`` of boxed code integers and ``28.0`` of
    boxed *index* integers, since every assigned index above CPython's
    small-integer cache is its own object. Three of those four are avoidable,
    and at the low-reuse end of the ladder they were enough to make the packed
    representation cost *more* than the dictionaries it replaced.

    What replaces them is an int64 array of codes in assignment order and an
    open-addressed int32 slot array holding ``index + 1``, empty being zero.
    Eight bytes of code and eight of slot -- the table runs at half load -- is
    ``16`` bytes per word against ``101.4``.

    The boxed code integers are not counted here any more, and that is a
    statement about where they live rather than a saving: they are the same
    objects ``MatrixElementBank._universe`` holds, which both storage backends
    keep, so :meth:`MatrixElementBank.measured_storage_bytes` charges them once
    at the bank level on either side. Dropping them from this table's own total
    without charging them there would credit packing with an allocation it never
    removed.
    """

    __slots__ = ("_codes", "_count", "_slots", "_shift")

    # Knuth's multiplicative constant, 2**64 / phi. Pauli word codes are dense
    # small integers whose low bits carry the leading qubits' letters, so the
    # identity hash a dict would use clusters badly under a power-of-two mask;
    # multiplying and reading the *high* bits spreads them.
    _MIX = 0x9E3779B97F4A7C15
    _MASK64 = (1 << 64) - 1

    def __init__(self, capacity: int = 1024) -> None:
        shift = max(4, (capacity - 1).bit_length())
        self._codes = np.empty(capacity, dtype=np.int64)
        self._count = 0
        self._slots = np.zeros(1 << shift, dtype=INDEX_DTYPE)
        self._shift = shift

    def __len__(self) -> int:
        return self._count

    def _probe(self, code: int) -> int:
        """First slot for ``code``: its entry, or the empty one it would take."""
        slots = self._slots
        codes = self._codes
        mask = len(slots) - 1
        position = ((code * self._MIX) & self._MASK64) >> (64 - self._shift)
        while True:
            occupant = slots[position]
            if occupant == 0 or codes[occupant - 1] == code:
                return position
            position = (position + 1) & mask

    def _grow(self) -> None:
        """Double the slot array and rehash; assigned indices do not move."""
        self._shift += 1
        self._slots = np.zeros(1 << self._shift, dtype=INDEX_DTYPE)
        for index in range(self._count):
            self._slots[self._probe(int(self._codes[index]))] = index + 1

    def intern_many(self, codes: Sequence[int]) -> list[int]:
        """Intern a whole row's codes, resolving the common case in numpy.

        The scalar probe is a Python loop where a ``dict`` lookup used to be C,
        and it runs once per coefficient occurrence -- measured at ``922`` ns a
        word against a dict's ``138``. Most of those calls are *hits*: a bank
        with coefficient reuse of fifty sees each word about fifty times and
        assigns it once. So the first probe is computed for the whole row at
        once, and only the entries it does not resolve -- an empty slot, or a
        collision landing on some other word -- fall back to the scalar path.

        Resolving hits against the slot array as it stands at entry is safe
        even though the fallback may grow the table underneath: assigned indices
        never move, so an index read before a rehash is still that word's index
        after one.
        """
        if not codes:
            return []
        wanted = np.array(codes, dtype=np.int64)
        positions = ((wanted.astype(np.uint64) * np.uint64(self._MIX))
                     >> np.uint64(64 - self._shift)).astype(np.intp)
        occupants = self._slots[positions]
        # -1 marks "the first probe did not settle this one": either the slot
        # was empty or it holds a different word. Both go to the scalar path.
        out = np.full(len(wanted), -1, dtype=np.int64)
        occupied = np.flatnonzero(occupants)
        if occupied.size:
            indices = occupants[occupied].astype(np.intp) - 1
            hit = self._codes[indices] == wanted[occupied]
            out[occupied[hit]] = indices[hit]
        resolved = out.tolist()
        if (out < 0).any():
            for position in np.flatnonzero(out < 0):
                resolved[position] = self.intern(int(wanted[position]))
        return resolved

    def intern(self, code: int) -> int:
        """Return this word's index, assigning the next one if it is new."""
        position = self._probe(code)
        occupant = self._slots[position]
        if occupant != 0:
            return int(occupant) - 1
        if self._count >= MAX_WORDS:
            raise OverflowError(
                f"the packed word table is limited to {MAX_WORDS} words by its "
                f"{np.dtype(INDEX_DTYPE).name} index")
        if self._count == len(self._codes):
            grown = np.empty(2 * len(self._codes), dtype=np.int64)
            grown[:self._count] = self._codes[:self._count]
            self._codes = grown
        assigned = self._count
        self._codes[assigned] = code
        self._count += 1
        # Half load. Open addressing with linear probing degrades sharply past
        # about two thirds, and the slot array is four bytes a word, so buying
        # the headroom is cheaper than paying the probe chains.
        if 2 * self._count > len(self._slots):
            self._grow()
        else:
            self._slots[position] = assigned + 1
        return assigned

    def lookup(self, code: int) -> int | None:
        """This word's index, or ``None`` when the table has never seen it."""
        occupant = self._slots[self._probe(code)]
        return int(occupant) - 1 if occupant != 0 else None

    def code(self, index: int) -> int:
        return int(self._codes[index])

    def codes(self, indices: Iterable[int]) -> list[int]:
        return [int(self._codes[i]) for i in indices]

    def nbytes(self) -> int:
        """The table's own arrays. No boxed object is counted, or held.

        Excludes the boxed code integers deliberately: this table no longer
        references them, and they remain resident through the bank's own word
        universe under either storage backend, so the bank charges them once on
        each side rather than this table charging them on one.
        """
        return self._codes.nbytes + self._slots.nbytes

    def live_bytes(self) -> int:
        """Bytes the live entries occupy, excluding growth slack."""
        return (self._count * self._codes.dtype.itemsize
                + len(self._slots) * self._slots.dtype.itemsize)

    def bytes_per_word(self) -> float:
        return (self.live_bytes() / self._count) if self._count else 0.0


class _Buffer:
    """An append-only numpy buffer that grows geometrically.

    CSR wants one contiguous array per functional, but rows arrive one at a
    time during adaptive growth, so the array has to grow. Doubling keeps the
    amortized append cost constant and the slack bounded by the live length,
    and :meth:`view` hands out a zero-copy window rather than a slice copy.
    """

    __slots__ = ("_array", "_length")

    def __init__(self, dtype, capacity: int = 1024) -> None:
        self._array = np.empty(capacity, dtype=dtype)
        self._length = 0

    def __len__(self) -> int:
        return self._length

    def extend(self, values: np.ndarray) -> tuple[int, int]:
        """Append ``values`` and return the ``(start, stop)`` they occupy."""
        start = self._length
        stop = start + len(values)
        if stop > len(self._array):
            capacity = max(stop, 2 * len(self._array))
            grown = np.empty(capacity, dtype=self._array.dtype)
            grown[:self._length] = self._array[:self._length]
            self._array = grown
        self._array[start:stop] = values
        self._length = stop
        return start, stop

    def view(self, start: int, stop: int) -> np.ndarray:
        return self._array[start:stop]

    def nbytes(self) -> int:
        """Bytes actually occupied by live entries, excluding growth slack."""
        return self._length * self._array.dtype.itemsize

    def reserved_bytes(self) -> int:
        """Bytes the buffer holds, including the slack doubling leaves."""
        return self._array.nbytes


class PackedRowStore:
    """CSR rows over one shared word table, in the order the product emitted.

    One store holds one functional family -- every ``A_i' A_j`` row, or every
    ``A_i' H A_j`` row -- keyed by the pair that owns it. Rows are immutable
    once added, which is what lets the buffers be shared.
    """

    def __init__(self, n: int, table: GlobalWordTable, *,
                 layout: str = "interleaved", capacity: int = 1024) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        self.n = n
        self.layout = layout
        self._table = table
        self._rows: dict[tuple[int, int], tuple[int, int]] = {}
        self._indices = _Buffer(INDEX_DTYPE, capacity)
        self._generation = _Buffer(INDEX_DTYPE, capacity)
        if layout == "interleaved":
            self._data = _Buffer(np.complex128, capacity)
            self._real = self._imag = None
        else:
            self._data = None
            self._real = _Buffer(np.float64, capacity)
            self._imag = _Buffer(np.float64, capacity)

    # ----------------------------------------------------------- structure

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._rows

    def keys(self) -> Iterator[tuple[int, int]]:
        return iter(self._rows)

    def row_length(self, key: tuple[int, int]) -> int:
        start, stop = self._rows[key]
        return stop - start

    def total_coefficients(self) -> int:
        return len(self._indices)

    # --------------------------------------------------------------- write

    def add(self, key: tuple[int, int], operator: MV) -> None:
        """Pack one operator row, preserving the order its terms were emitted in.

        The ascending ``indices`` make a probe a binary search; ``generation``
        records where each emitted term landed among them, so replaying the row
        reproduces the dict iteration order ``trace_pairing`` consumed before
        this store existed.
        """
        if key in self._rows:
            raise KeyError(f"row {key} is already packed; rows are immutable")
        if operator.n != self.n:
            raise ValueError("operator lives in a different algebra")
        terms = list(operator.terms.items())
        interned = self._table.intern_many([code for code, _ in terms])
        emitted = [(index, coefficient)
                   for index, (_, coefficient) in zip(interned, terms)]
        order = sorted(range(len(emitted)), key=lambda p: emitted[p][0])
        # ``generation[p]`` is the ascending slot the p-th emitted term landed
        # in, so walking p in order replays the emission sequence.
        generation = np.empty(len(emitted), dtype=INDEX_DTYPE)
        for ascending_slot, emitted_position in enumerate(order):
            generation[emitted_position] = ascending_slot
        indices = np.fromiter((emitted[p][0] for p in order),
                              dtype=INDEX_DTYPE, count=len(order))
        start, stop = self._indices.extend(indices)
        self._generation.extend(generation)
        values = [emitted[p][1] for p in order]
        if self.layout == "interleaved":
            self._data.extend(np.fromiter(values, dtype=np.complex128,
                                          count=len(values)))
        else:
            self._real.extend(np.fromiter((v.real for v in values),
                                          dtype=np.float64, count=len(values)))
            self._imag.extend(np.fromiter((v.imag for v in values),
                                          dtype=np.float64, count=len(values)))
        self._rows[key] = (start, stop)

    # ---------------------------------------------------------------- read

    def _coefficient(self, position: int) -> complex:
        """One coefficient as a Python ``complex``.

        Converted rather than handed over as a numpy scalar deliberately. The
        arithmetic below has to agree with ``MV.trace_pairing`` to the last bit,
        and that means running in the same type the dict held; a numpy scalar
        would be free to contract or reassociate its own way.
        """
        if self.layout == "interleaved":
            return complex(self._data.view(position, position + 1)[0])
        return complex(float(self._real.view(position, position + 1)[0]),
                       float(self._imag.view(position, position + 1)[0]))

    def iter_terms(self, key: tuple[int, int]) -> Iterator[tuple[int, complex]]:
        """Replay generation order without allocating a list of boxed tuples."""
        start, stop = self._rows[key]
        generation = self._generation.view(start, stop)
        indices = self._indices.view(start, stop)
        for slot in generation:
            yield (self._table.code(int(indices[slot])),
                   self._coefficient(start + int(slot)))

    def terms(self, key: tuple[int, int]) -> list[tuple[int, complex]]:
        """Compatibility list of coefficients in their original emission order."""
        return list(self.iter_terms(key))

    def word_indices(self, key: tuple[int, int]) -> np.ndarray:
        """The row's global word indices, ascending. Zero-copy."""
        start, stop = self._rows[key]
        return self._indices.view(start, stop)

    def materialize(self, key: tuple[int, int]) -> MV:
        """Rebuild the row as an ``MV`` with its original insertion order.

        The order matters even here: an ``MV`` handed back to a caller may be
        paired, and ``trace_pairing`` would then iterate this dict.
        """
        return MV(self.n, dict(self.iter_terms(key)))

    def trace_pairing(self, key: tuple[int, int], other: MV) -> complex:
        """``Tr(row * other) / 2^n``, summed in ``MV.trace_pairing``'s order.

        That method picks the operand with fewer terms and iterates *its* dict,
        so this reproduces both branches rather than always walking the row:
        the sum is over the same addends in the same sequence, which is what
        makes the packed backend bit-identical rather than merely close.
        """
        start, stop = self._rows[key]
        length = stop - start
        right = other.terms
        total = 0j
        if length <= len(right):
            for code, coefficient in self.iter_terms(key):
                partner = right.get(code)
                if partner is None:
                    continue
                total += coefficient * partner
            return total
        indices = self._indices.view(start, stop)
        for code, coefficient in right.items():
            index = self._table.lookup(code)
            if index is None:
                continue
            position = int(np.searchsorted(indices, index))
            if position >= length or int(indices[position]) != index:
                continue
            total += coefficient * self._coefficient(start + position)
        return total

    # ------------------------------------------------------------ resources

    def nbytes(self) -> int:
        """Live bytes in the packed buffers, excluding growth slack."""
        parts = [self._indices.nbytes(), self._generation.nbytes()]
        if self.layout == "interleaved":
            parts.append(self._data.nbytes())
        else:
            parts += [self._real.nbytes(), self._imag.nbytes()]
        return sum(parts)

    def reserved_bytes(self) -> int:
        """Bytes the buffers hold, slack from geometric growth included.

        Reported beside :meth:`nbytes` because the slack is real resident
        memory: a store that has just doubled holds up to twice what it uses,
        and a reduction quoted from live bytes alone would not survive contact
        with the process that has to run.
        """
        parts = [self._indices.reserved_bytes(), self._generation.reserved_bytes()]
        if self.layout == "interleaved":
            parts.append(self._data.reserved_bytes())
        else:
            parts += [self._real.reserved_bytes(), self._imag.reserved_bytes()]
        return sum(parts)

    def bytes_per_coefficient(self) -> float:
        total = self.total_coefficients()
        return (self.nbytes() / total) if total else 0.0


class SegmentedPackedRowStore:
    """Independently owned, tightly sized rows that can actually be released.

    The append-only CSR store remains preferable for retain_all. A streaming
    bank instead pays small per-row metadata to avoid retaining dead capacity.
    The shared intern table has a separate, cumulative lifetime.
    """

    def __init__(self, n, table, *, layout="interleaved"):
        self.n, self._table, self.layout = n, table, layout
        self._segments = {}

    def __len__(self):
        return len(self._segments)

    def __contains__(self, key):
        return key in self._segments

    def keys(self):
        return iter(self._segments)

    def add(self, key, operator):
        if key in self._segments:
            raise KeyError(f"row {key} is already packed")
        segment = PackedRowStore(self.n, self._table, layout=self.layout,
                                 capacity=operator.nnz())
        segment.add(key, operator)
        self._segments[key] = segment

    def remove(self, key):
        del self._segments[key]

    def row_length(self, key):
        return self._segments[key].row_length(key)

    def total_coefficients(self):
        return sum(segment.total_coefficients() for segment in self._segments.values())

    def iter_terms(self, key):
        return self._segments[key].iter_terms(key)

    def terms(self, key):
        return list(self.iter_terms(key))

    def word_indices(self, key):
        return self._segments[key].word_indices(key)

    def materialize(self, key):
        return self._segments[key].materialize(key)

    def trace_pairing(self, key, other):
        return self._segments[key].trace_pairing(key, other)

    def nbytes(self):
        return sum(segment.nbytes() for segment in self._segments.values())

    def reserved_bytes(self):
        return sum(segment.reserved_bytes() for segment in self._segments.values())
