"""Bounded exact-selection coefficient rows with persistent scalar/support history."""
from __future__ import annotations

from collections import OrderedDict
import sys
import time

import numpy as np

from .packed import SegmentedPackedRowStore
from .projection import MatrixElementBank


class StreamingMatrixElementBank(MatrixElementBank):
    """Keep a selected block and at most ``frontier_pairs`` additional S/H pairs.

    Scalars and compact supports survive row eviction, preserving exact
    rescoring and its word-cost objective. The bound concerns coefficient rows,
    not the cumulative support history, intern table, reference, generator
    products, or explicitly requested observables; resources names each scope.
    Both object and packed rows are supported. Packed segments own their arrays
    independently so eviction releases capacity, not just dictionary keys.
    """

    def __init__(self, *args, frontier_pairs=32, **kwargs):
        if isinstance(frontier_pairs, bool) or int(frontier_pairs) != frontier_pairs or frontier_pairs < 1:
            raise ValueError("frontier_pairs must be a positive integer")
        self.frontier_pairs = int(frontier_pairs)
        self._supports = {}
        self._recent = OrderedDict()
        self._retained = frozenset()
        self._evicted_rows = self._recomputed_rows = 0
        self._physical_builds = self._processed_coefficients = 0
        self._cumulative_coefficients = 0
        super().__init__(*args, **kwargs)
        if self._storage == "packed":
            self._packed = {kind: SegmentedPackedRowStore(
                self.n, self._word_table, layout=self._layout)
                for kind in ("overlap", "element")}

    def retain_basis(self, indices):
        """Declare which block is exempt from the frontier row limit."""
        self._retained = frozenset(self.resolve(indices))
        self._evict()
        # A selected pair may have left the frontier before it was chosen.
        # Cached scalar entries alone would let solve() skip rebuilding it.
        order = sorted(self._retained)
        for offset, i in enumerate(order):
            for j in order[offset:]:
                key = (i, j)
                if key in self._supports and not self._has_pair(key):
                    self._ensure_pair(key)

    def _is_retained(self, key):
        return key[0] in self._retained and key[1] in self._retained

    def _evict(self):
        frontier = [key for key in self._recent if not self._is_retained(key)]
        for key in frontier[:-self.frontier_pairs]:
            if self._storage == "packed":
                for store in self._packed.values():
                    store.remove(key)
            else:
                del self._overlap_ops[key]
                del self._element_ops[key]
            del self._recent[key]
            self._evicted_rows += 2

    def _build_pair(self, i, j, *, evaluate=False):
        key = (i, j)
        started = time.perf_counter()
        overlap = self._adjoints[i] * self._generators[j].mv
        element = self._adjoints[i] * self._h_acted[j]
        self._seconds += time.perf_counter() - started
        count = overlap.nnz() + element.nnz()
        self._physical_builds += 1
        self._processed_coefficients += count
        if key not in self._supports:
            # Use arbitrary-width Python integers only beyond int64 word codes.
            dtype = np.int64 if self.n <= 31 else object
            self._supports[key] = tuple(np.array(list(op.terms), dtype=dtype)
                                        for op in (overlap, element))
            for op in (overlap, element):
                self._account(j, op)
            self._pairs_built += 1
            self._cumulative_coefficients += count
        else:
            self._recomputed_rows += 2
        if evaluate and key not in self._entries:
            self._entries[key] = (self._scale * overlap.trace_pairing(self._rho),
                                  self._scale * element.trace_pairing(self._rho))
        if self._storage == "packed":
            self._packed["overlap"].add(key, overlap)
            self._packed["element"].add(key, element)
        else:
            self._overlap_ops[key], self._element_ops[key] = overlap, element
        self._recent[key] = None
        self._evict()
        # One pair of product temporaries exists during construction in addition
        # to this persistent-row bound; it is reported explicitly below.
        self._peak_resident_rows = max(self._peak_resident_rows, self._resident_row_count())

    def _ensure_pair(self, key, *, evaluate=False):
        super()._ensure_pair(key, evaluate=evaluate)
        self._recent.move_to_end(key)

    def _pair_keys(self):
        return list(self._supports)

    def _row_codes(self, kind, key):
        return (int(code) for code in self._supports[key][kind == "element"])

    def _row_nnz(self, kind, key):
        return len(self._supports[key][kind == "element"])

    def update_pair_support(self, destination, i, j):
        key = self._canonical_pair(i, j)
        if key not in self._supports:
            self._ensure_pair(key)
        for codes in self._supports[key]:
            destination.update(int(code) for code in codes)

    def measured_storage_bytes(self):
        result = super().measured_storage_bytes()
        result["storage_policy"] = "stream_recompute"
        return result

    def resources(self, indices=None):
        result = super().resources(indices)
        order = set(self.resolve(indices))
        resident_words = set()
        for key in self._recent:
            for codes in self._supports[key]:
                resident_words.update(int(code) for code in codes)
        for record in self._observables.values():
            resident_words.update(record["universe"])
        resident = len(resident_words)
        kept = sum(i in order and j in order for i, j in self._recent)
        history_bytes = sum(codes.nbytes for pair in self._supports.values() for codes in pair)
        if self.n > 31:
            history_bytes += sum(sys.getsizeof(code) for pair in self._supports.values()
                                 for codes in pair for code in codes)
        result.update({
            "storage_policy": "stream_recompute",
            "resident_word_universe": resident,
            "coefficient_reuse": result["coefficient_occurrences"] / resident if resident else 0.0,
            "retained_block_pairs": kept,
            "selection_pairs": self._pairs_built - sum(
                i in order and j in order for i, j in self._supports),
            "retained_pair_fraction": kept / self._pairs_built if self._pairs_built else 0.0,
            "evicted_rows": self._evicted_rows,
            "recomputed_rows": self._recomputed_rows,
            "physical_pair_builds": self._physical_builds,
            "operator_products": 2 * self._physical_builds + len(self),
            "cumulative_coefficient_occurrences": self._cumulative_coefficients,
            "processed_coefficient_occurrences": self._processed_coefficients,
            "cumulative_word_universe": len(self._universe),
            "support_history_payload_bytes": history_bytes,
            "support_history_container_bytes": sys.getsizeof(self._supports) + sum(
                sys.getsizeof(key) + sys.getsizeof(pair)
                + sum(sys.getsizeof(codes) - codes.nbytes for codes in pair)
                for key, pair in self._supports.items()),
            "scalar_cache_shallow_bytes": sys.getsizeof(self._entries) + sum(
                sys.getsizeof(key) + sys.getsizeof(values) + sum(map(sys.getsizeof, values))
                for key, values in self._entries.items()),
            "hamiltonian_action_payload_bytes": sum(op.memory_estimate() for op in self._h_acted),
            "word_history_set_bytes": sys.getsizeof(self._universe),
            "word_table_reserved_bytes": self._word_table.nbytes() if self._word_table else 0,
            "frontier_pair_limit": self.frontier_pairs,
            "resident_sh_rows": 2 * len(self._recent),
            "resident_sh_row_limit": 2 * (len(self._retained) * (len(self._retained) + 1) // 2
                                           + self.frontier_pairs),
            "product_temporary_row_limit": 2,
            "observable_rows_outside_frontier_bound": sum(
                len(record["operators"]) for record in self._observables.values()),
        })
        return result
