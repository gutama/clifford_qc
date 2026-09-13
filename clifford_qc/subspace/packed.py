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

from typing import Iterable, Iterator

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
    """The one canonical code-to-index map every packed row shares.

    A Pauli word that appears in forty rows is one entry here and forty
    four-byte indices there, where the object store holds one shared boxed
    integer and forty dict slots pointing at it. The table is append-only:
    indices are handed out in first-seen order and never move, because a row
    already packed holds them.
    """

    __slots__ = ("_codes", "_index")

    def __init__(self) -> None:
        self._codes: list[int] = []
        self._index: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._codes)

    def intern(self, code: int) -> int:
        """Return this word's index, assigning the next one if it is new."""
        existing = self._index.get(code)
        if existing is not None:
            return existing
        if len(self._codes) >= MAX_WORDS:
            raise OverflowError(
                f"the packed word table is limited to {MAX_WORDS} words by its "
                f"{np.dtype(INDEX_DTYPE).name} index")
        assigned = len(self._codes)
        self._codes.append(code)
        self._index[code] = assigned
        return assigned

    def lookup(self, code: int) -> int | None:
        """This word's index, or ``None`` when the table has never seen it."""
        return self._index.get(code)

    def code(self, index: int) -> int:
        return self._codes[index]

    def codes(self, indices: Iterable[int]) -> list[int]:
        return [self._codes[i] for i in indices]

    def nbytes(self) -> int:
        """What the table itself costs, containers and boxed objects together.

        Reported separately from the rows because it scales differently: the
        table is paid once per distinct *word*, the rows once per *occurrence*,
        and on the committed molecular banks those differ by a factor of sixty
        to a hundred and fifty. That ratio -- coefficient reuse -- is what
        decides how much of the per-coefficient packing survives into the total.

        Four populations, and the fourth is the one easy to miss: the mapping's
        boxed *values*. Every assigned index above CPython's small-integer cache
        is its own object, and omitting them understated this table by 1.42x in
        the direction that flatters the packed backend. The boxed *keys* are
        counted once here and are the same objects the bank's own ``_universe``
        holds, which both backends pay for, so charging them here keeps the two
        measurements on the same footing rather than crediting packing with an
        allocation it never removed.
        """
        import sys
        total = sys.getsizeof(self._codes) + sys.getsizeof(self._index)
        seen: set[int] = set()
        for boxed in (*self._codes, *self._index.values()):
            if id(boxed) in seen:
                continue
            seen.add(id(boxed))
            total += sys.getsizeof(boxed)
        return total

    def bytes_per_word(self) -> float:
        return (self.nbytes() / len(self._codes)) if self._codes else 0.0


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
                 layout: str = "interleaved") -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        self.n = n
        self.layout = layout
        self._table = table
        self._rows: dict[tuple[int, int], tuple[int, int]] = {}
        self._indices = _Buffer(INDEX_DTYPE)
        self._generation = _Buffer(INDEX_DTYPE)
        if layout == "interleaved":
            self._data = _Buffer(np.complex128)
            self._real = self._imag = None
        else:
            self._data = None
            self._real = _Buffer(np.float64)
            self._imag = _Buffer(np.float64)

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
        emitted = [(self._table.intern(code), coefficient)
                   for code, coefficient in operator.terms.items()]
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

    def terms(self, key: tuple[int, int]) -> list[tuple[int, complex]]:
        """``(code, coefficient)`` in emission order -- the dict's own order."""
        start, stop = self._rows[key]
        generation = self._generation.view(start, stop)
        indices = self._indices.view(start, stop)
        return [(self._table.code(int(indices[slot])),
                 self._coefficient(start + int(slot)))
                for slot in generation]

    def word_indices(self, key: tuple[int, int]) -> np.ndarray:
        """The row's global word indices, ascending. Zero-copy."""
        start, stop = self._rows[key]
        return self._indices.view(start, stop)

    def materialize(self, key: tuple[int, int]) -> MV:
        """Rebuild the row as an ``MV`` with its original insertion order.

        The order matters even here: an ``MV`` handed back to a caller may be
        paired, and ``trace_pairing`` would then iterate this dict.
        """
        return MV(self.n, dict(self.terms(key)))

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
            for code, coefficient in self.terms(key):
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
