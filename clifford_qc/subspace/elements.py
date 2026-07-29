"""A-CASE Phase 2: the exact matrix-element bank.

Phase 1 assembles a subspace by recomputing every element operator from
scratch. Adaptive growth (Phase 3) adds one row and column at a time and
re-solves, so that assembly would recompute ``A_i'A_j`` and ``A_i'HA_j`` many
times over -- making a sound method look uncompetitive because of a
deliberately temporary implementation. This module caches them.

The bank is the ``CommutatorBank`` architecture generalized from
candidate-by-word to matrix-entry-by-word:

- every element operator is stored as its sparse Pauli-word coefficient map,
  which is the object Phase 4 must reconstruct from measured word means, not
  merely an intermediate to a scalar;
- one global word universe ``W = |union_ij supp(O^S_ij) union supp(O^H_ij)|``
  is tracked as it grows, together with how many words each accepted generator
  *reuses* versus *introduces* -- the §6 metric that decides whether a basis is
  compact in the sense that matters for measurement;
- Hermitian pairs share work: ``S_ji = conj(S_ij)`` exactly, so only the upper
  triangle is ever computed, and the invariant is a property of the layout.

Everything here is exact. Finite-shot reconstruction is Phase 4, and the
separation is deliberate: the bank's coefficient maps are precisely the
sufficient statistics that layer will need, so it can be added without
disturbing this one.

The projected-observable route of §8 lives here too. A Ritz state is never
stored, so an observable ``Q`` is answered by building ``Q_sub`` through the
same machinery and contracting it with the Ritz coefficients -- and its extra
words enter the same accounting.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np

from ..ir import PauliWord
from ..multivector import MV
from .generators import Generator, as_generators
from .solver import SubspaceResult, _as_mv, _check_reference, solve_projected


def _identity_key(mv: MV) -> tuple:
    """Exact identity of a generator: its Pauli-word coefficient map.

    Deliberately exact rather than tolerant. Deduplication is a cache
    optimization, so a missed near-duplicate costs a recomputation and nothing
    else, while a tolerant key that merged two *different* directions would
    silently change the subspace.
    """
    return tuple(sorted(mv.terms.items()))


class MatrixElementBank:
    """Cached exact ``(S, H)`` assembly over a growing generator set.

    Generators are added incrementally and identified by a canonical integer
    id; adding the same operator twice returns the id it already has, and
    reusing a label for a different operator is an error, because labels are
    what records and result objects report.

    Pair products are computed lazily on first use, so a candidate that is
    scored and rejected costs only its own row -- and a candidate that is never
    scored costs nothing.
    """

    def __init__(self, rho: MV, hamiltonian, generators: Iterable = ()):
        self._rho = rho
        self._H = _as_mv(hamiltonian)
        if not self._H.is_hermitian():
            raise ValueError("Hamiltonian must be Hermitian")
        if rho.n != self._H.n:
            raise ValueError("reference and Hamiltonian disagree on n")
        self._purity = _check_reference(rho)
        self._scale = float(2 ** rho.n)

        self._generators: list[Generator] = []
        self._adjoints: list[MV] = []
        self._h_acted: list[MV] = []
        self._by_label: dict[str, int] = {}
        self._by_terms: dict[tuple, int] = {}

        self._overlap_ops: dict[tuple[int, int], MV] = {}
        self._element_ops: dict[tuple[int, int], MV] = {}
        self._entries: dict[tuple[int, int], tuple[complex, complex]] = {}
        self._universe: set[int] = set()
        self._new_words: list[int] = []
        self._reused_words: list[int] = []

        self._observables: dict[tuple, dict] = {}
        self._pairs_built = 0
        self._cache_hits = 0
        self._seconds = 0.0
        self.extend(generators)

    # ------------------------------------------------------------- structure

    @property
    def n(self) -> int:
        return self._rho.n

    def __len__(self) -> int:
        return len(self._generators)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(g.label for g in self._generators)

    def generator(self, index: int) -> Generator:
        return self._generators[index]

    def index_of(self, label: str) -> int:
        return self._by_label[label]

    def add(self, generator) -> int:
        """Register one generator and return its canonical id."""
        gen = as_generators([generator])[0]
        if gen.n != self.n:
            raise ValueError("generator lives in a different algebra")
        key = _identity_key(gen.mv)
        existing = self._by_terms.get(key)
        if existing is not None:
            return existing
        if gen.label in self._by_label:
            raise ValueError(
                f"label {gen.label!r} is already bound to a different generator")
        index = len(self._generators)
        self._generators.append(gen)
        self._adjoints.append(gen.mv.dagger())
        started = time.perf_counter()
        self._h_acted.append(self._H * gen.mv)
        self._seconds += time.perf_counter() - started
        self._by_label[gen.label] = index
        self._by_terms[key] = index
        self._new_words.append(0)
        self._reused_words.append(0)
        return index

    def extend(self, generators: Iterable) -> list[int]:
        return [self.add(g) for g in generators]

    def _resolve(self, indices: Sequence[int] | None) -> tuple[int, ...]:
        if indices is None:
            return tuple(range(len(self._generators)))
        out = tuple(int(i) for i in indices)
        for i in out:
            if not (0 <= i < len(self._generators)):
                raise IndexError(f"generator index {i} out of range")
        if len(set(out)) != len(out):
            raise ValueError("repeated generator index in subset")
        return out

    # ------------------------------------------------------- element cache

    def _account(self, owner: int, operator: MV) -> None:
        """Attribute an element operator's words to the newer generator's row."""
        for code in operator.terms:
            if code in self._universe:
                self._reused_words[owner] += 1
            else:
                self._universe.add(code)
                self._new_words[owner] += 1

    def _build_pair(self, i: int, j: int) -> None:
        """Compute and cache the ``i <= j`` element operators of one pair.

        The product order matches ``solver.projected_matrices`` exactly --
        ``A_i'`` from ``dagger()``, then the right factor, then the pairing
        against ``rho`` -- so the bank reproduces the Phase-1 matrices to the
        last bit rather than merely to a tolerance. Floating-point addition is
        not associative; a "mathematically equivalent" rearrangement here would
        make committed records irreproducible across the two routes.
        """
        started = time.perf_counter()
        overlap_op = self._adjoints[i] * self._generators[j].mv
        element_op = self._adjoints[i] * self._h_acted[j]
        self._seconds += time.perf_counter() - started
        self._overlap_ops[(i, j)] = overlap_op
        self._element_ops[(i, j)] = element_op
        self._account(j, overlap_op)
        self._account(j, element_op)
        self._pairs_built += 1

    def _pair(self, i: int, j: int) -> tuple[MV, MV]:
        key = (i, j) if i <= j else (j, i)
        if key not in self._overlap_ops:
            self._build_pair(*key)
        else:
            self._cache_hits += 1
        return self._overlap_ops[key], self._element_ops[key]

    def overlap_operator(self, i: int, j: int) -> MV:
        """``O^S_ij = A_i' A_j`` (stored for ``i <= j``; the transpose is its adjoint)."""
        operator = self._pair(i, j)[0]
        return operator if i <= j else operator.dagger()

    def element_operator(self, i: int, j: int) -> MV:
        """``O^H_ij = A_i' H A_j`` (stored for ``i <= j``)."""
        operator = self._pair(i, j)[1]
        return operator if i <= j else operator.dagger()

    def entry(self, i: int, j: int) -> tuple[complex, complex]:
        """``(S_ij, H_ij)`` -- the conjugate of the stored pair when ``i > j``."""
        key = (i, j) if i <= j else (j, i)
        value = self._entries.get(key)
        if value is None:
            overlap_op, element_op = self._pair(i, j)
            value = (self._scale * overlap_op.trace_pairing(self._rho),
                     self._scale * element_op.trace_pairing(self._rho))
            self._entries[key] = value
        else:
            self._cache_hits += 1
        s, h = value
        return (s, h) if i <= j else (s.conjugate(), h.conjugate())

    def matrices(self, indices: Sequence[int] | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        """``(S, H)`` over a generator subset, Hermitian by construction."""
        order = self._resolve(indices)
        m = len(order)
        S = np.zeros((m, m), dtype=complex)
        Hm = np.zeros((m, m), dtype=complex)
        for b, j in enumerate(order):
            for a, i in enumerate(order[:b + 1]):
                s, h = self.entry(i, j)
                S[a, b], Hm[a, b] = s, h
        for b in range(m):
            S[b, b] = S[b, b].real
            Hm[b, b] = Hm[b, b].real
            for a in range(b):
                S[b, a] = S[a, b].conjugate()
                Hm[b, a] = Hm[a, b].conjugate()
        return S, Hm

    # --------------------------------------------------- projected observables

    def project_observable(self, observable, indices: Sequence[int] | None = None,
                           *, label: str | None = None) -> np.ndarray:
        """``Q_sub[i, j] = <psi|A_i' Q A_j|psi>`` (§8), through the same pairing.

        A Hermitian ``Q`` is built from its upper triangle and mirrored, as
        ``(S, H)`` are. A non-Hermitian one is not: ``Q_sub[j, i]`` is then an
        independent element, not a conjugate, so the full ``M^2`` is computed.

        The observable's own word universe is tracked separately and reported
        by :meth:`resources` -- an observable that doubles the words to be
        measured is not free just because the energy was already paid for.

        ``label`` names the observable in that report; the first name
        registered for a given operator wins, since the operator, not the
        name, is its identity.
        """
        Q = _as_mv(observable)
        if Q.n != self.n:
            raise ValueError("observable lives in a different algebra")
        order = self._resolve(indices)
        key = _identity_key(Q)
        record = self._observables.setdefault(
            key, {"label": label or f"Q{len(self._observables)}", "universe": set(),
                  "acted": {}, "operators": {}, "hermitian": Q.is_hermitian()})
        hermitian = record["hermitian"]
        acted, cache = record["acted"], record["operators"]

        m = len(order)
        out = np.zeros((m, m), dtype=complex)
        started = time.perf_counter()
        for b, j in enumerate(order):
            if j not in acted:  # Q A_j once per column, as H A_j is
                acted[j] = Q * self._generators[j].mv
            for a, i in enumerate(order):
                if hermitian and a > b:
                    continue
                if (i, j) not in cache:
                    cache[(i, j)] = self._adjoints[i] * acted[j]
                    record["universe"].update(cache[(i, j)].terms)
                out[a, b] = self._scale * cache[(i, j)].trace_pairing(self._rho)
        self._seconds += time.perf_counter() - started
        if hermitian:
            for b in range(m):
                out[b, b] = out[b, b].real
                for a in range(b):
                    out[b, a] = out[a, b].conjugate()
        return out

    # ------------------------------------------------------------- reporting

    def word_set(self, indices: Sequence[int] | None = None) -> frozenset[int]:
        """Word codes the cached elements of a generator subset carry.

        The whole cache when ``indices`` is ``None``; otherwise the union over
        pairs with *both* endpoints in the subset, which is what a measurement
        plan for that subspace would actually have to cover. Adaptive growth
        uses the difference against a candidate's row to price it.
        """
        if indices is None:
            return frozenset(self._universe)
        wanted = set(self._resolve(indices))
        out: set[int] = set()
        for (i, j), operator in self._overlap_ops.items():
            if i in wanted and j in wanted:
                out.update(operator.terms)
                out.update(self._element_ops[(i, j)].terms)
        return frozenset(out)

    def words(self, indices: Sequence[int] | None = None) -> tuple[PauliWord, ...]:
        """The word universe as sorted ``PauliWord``s -- the measurement plan's input."""
        return tuple(PauliWord(self.n, code) for code in sorted(self.word_set(indices)))

    def qwc_group_count(self) -> int:
        """Circuits one exhaustive measurement of the universe would cost.

        Not part of :meth:`resources`: the greedy partition is quadratic in the
        universe size, which reaches five figures on an 8-qubit molecule, so
        this is opt-in rather than a hidden cost of every build.
        """
        from ..measurement.grouping import qwc_groups
        return len(qwc_groups(self.words()))

    def resources(self, indices: Sequence[int] | None = None) -> dict[str, Any]:
        """The §6 accounting for the current cache state."""
        order = self._resolve(indices)
        wanted = set(order)
        keys = [key for key in self._overlap_ops
                if key[0] in wanted and key[1] in wanted]
        overlap_ops = [self._overlap_ops[key] for key in keys]
        element_ops = [self._element_ops[key] for key in keys]
        # A proper subset owns a smaller universe than the cache does, and a
        # record that quoted the cache's would overstate the subset's
        # measurement cost -- which matters most during adaptive growth, where
        # the cache also holds every rejected candidate's row.
        universe = (len(self._universe) if len(order) == len(self._generators)
                    else len(self.word_set(order)))
        cached = (list(self._overlap_ops.values()) + list(self._element_ops.values())
                  + [op for record in self._observables.values()
                     for op in record["operators"].values()])
        observables = {
            record["label"]: {
                "words": len(record["universe"]),
                # words this observable adds to what (S, H) already require
                "additional_words": len(record["universe"] - self._universe),
            }
            for record in self._observables.values()
        }
        return {
            "basis_size": len(order),
            "registered_generators": len(self._generators),
            "support_tracked": True,
            "word_universe": universe,
            "max_generator_support": max((self._generators[i].support() for i in order),
                                         default=0),
            "max_overlap_element_support": max((op.nnz() for op in overlap_ops), default=0),
            "max_hamiltonian_element_support": max((op.nnz() for op in element_ops), default=0),
            "hamiltonian_support": self._H.nnz(),
            "reference_purity": self._purity,
            "pairs_built": self._pairs_built,
            "element_cache_hits": self._cache_hits,
            "operator_products": 2 * self._pairs_built + len(self._generators),
            "new_words_per_generator": tuple(
                (self._generators[i].label, self._new_words[i], self._reused_words[i])
                for i in order),
            "cached_operator_bytes": sum(op.memory_estimate() for op in cached),
            "assemble_seconds": self._seconds,
            "projected_observables": observables,
        }

    def solve(self, indices: Sequence[int] | None = None, **kwargs) -> SubspaceResult:
        """Solve the §4.1 generalized eigenproblem over a generator subset.

        The result carries the bank and the subset it was solved on, which is
        what lets ``result.expectation(Q)`` and ``result.transition(Q, i, j)``
        answer observable questions without ever forming a Ritz state.
        """
        order = self._resolve(indices)
        S, Hm = self.matrices(order)
        result = solve_projected(S, Hm, [self._generators[i].label for i in order],
                                 resources=self.resources(order), **kwargs)
        return result.with_bank(self, order)
