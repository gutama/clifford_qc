"""Projected-operator construction and cached matrix-element banks."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ..ir import PauliWord
from ..multivector import MV
from .contracts import as_multivector, check_reference
from .generator_core import Generator, as_generators
from .linalg import (
    DEFAULT_MAX_CONDITION,
    DEFAULT_NORM_FLOOR,
    DEFAULT_TAU_S,
    SubspaceResult,
    solve_projected,
)


@dataclass(frozen=True)
class ProjectedProblem:
    """One generalized eigenproblem plus its resource ledger."""

    overlap: np.ndarray
    hamiltonian: np.ndarray
    labels: tuple[str, ...]
    resources: dict[str, Any]

    def solve(self, **kwargs) -> SubspaceResult:
        return solve_projected(
            self.overlap,
            self.hamiltonian,
            self.labels,
            resources=self.resources,
            **kwargs,
        )


def projected_matrices(rho: MV, hamiltonian, generators: Sequence,
                       *, track_support: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Assemble ``(S, H)`` exactly, plus the §6 resource metrics.

    Two evaluation routes, identical in exact arithmetic:

    ``track_support=True`` forms each element operator ``O^S_ij = A_i' A_j``
    and ``O^H_ij = A_i' H A_j`` and pairs it with ``rho``. Forming them is what
    makes the word universe ``W`` and the support metrics ``S_H`` observable,
    and those operators are precisely what the Phase-2 bank caches and the
    Phase-4 measurement layer must reconstruct -- so this is the route that
    prices the method.

    ``track_support=False`` contracts cyclically instead:
    ``Tr(A_i' H A_j rho) = Tr((H A_j)(rho A_i'))``, which needs ``2M`` operator
    products and ``M^2`` sparse pairings rather than ``M^2`` triple products.
    It is much cheaper on wide generators, and correspondingly blind: no
    element operator is ever formed, so ``W`` and ``S_H`` are reported as
    ``None``. Use it to explore, not to make a compactness claim.
    """
    gens = as_generators(generators)
    H = as_multivector(hamiltonian)
    if not H.is_hermitian():
        raise ValueError("Hamiltonian must be Hermitian")
    if H.n != gens[0].n or rho.n != H.n:
        raise ValueError("reference, Hamiltonian, and generators disagree on n")
    purity = check_reference(rho)

    started = time.perf_counter()
    m = len(gens)
    scale = float(2 ** rho.n)
    S = np.zeros((m, m), dtype=complex)
    Hm = np.zeros((m, m), dtype=complex)
    words: set[int] = set()
    max_overlap_support = 0
    max_element_support = 0
    products = 0

    h_acted = [H * g.mv for g in gens]  # H A_j, shared down each column
    products += m
    if track_support:
        adjoints = [g.mv.dagger() for g in gens]
        for j in range(m):
            for i in range(j + 1):
                overlap_op = adjoints[i] * gens[j].mv
                element_op = adjoints[i] * h_acted[j]
                products += 2
                if overlap_op.nnz() > max_overlap_support:
                    max_overlap_support = overlap_op.nnz()
                if element_op.nnz() > max_element_support:
                    max_element_support = element_op.nnz()
                words.update(overlap_op.terms)
                words.update(element_op.terms)
                S[i, j] = scale * overlap_op.trace_pairing(rho)
                Hm[i, j] = scale * element_op.trace_pairing(rho)
    else:
        # rho A_i' once per row; every element is then one sparse pairing.
        weighted = [rho * g.mv.dagger() for g in gens]
        products += m
        for j in range(m):
            for i in range(j + 1):
                S[i, j] = scale * gens[j].mv.trace_pairing(weighted[i])
                Hm[i, j] = scale * h_acted[j].trace_pairing(weighted[i])

    # Hermiticity is structural: the lower triangle is the conjugate of the
    # upper one by definition of the matrix elements, not by symmetrization.
    for j in range(m):
        S[j, j] = S[j, j].real
        Hm[j, j] = Hm[j, j].real
        for i in range(j):
            S[j, i] = S[i, j].conjugate()
            Hm[j, i] = Hm[i, j].conjugate()

    resources = {
        "basis_size": m,
        "support_tracked": bool(track_support),
        "word_universe": len(words) if track_support else None,
        "max_generator_support": max(g.support() for g in gens),
        "max_overlap_element_support": max_overlap_support if track_support else None,
        "max_hamiltonian_element_support": max_element_support if track_support else None,
        "hamiltonian_support": H.nnz(),
        "operator_products": products,
        "reference_purity": purity,
        "assemble_seconds": time.perf_counter() - started,
    }
    return S, Hm, resources


def solve_subspace(rho: MV, hamiltonian, generators: Sequence, *,
                   tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
                   max_condition: float = DEFAULT_MAX_CONDITION,
                   norm_floor: float = DEFAULT_NORM_FLOOR,
                   track_support: bool = True) -> SubspaceResult:
    """Assemble and solve one fixed-basis A-CASE subspace.

    The Ritz values obey ``E_sub >= E_0`` and are monotone non-increasing under
    nested basis growth in exact arithmetic; both are checked in the test
    suite rather than asserted here, since a violation is evidence about the
    numerics and should surface as a failing invariant, not an exception.
    """
    gens = as_generators(generators)
    S, Hm, resources = projected_matrices(rho, hamiltonian, gens,
                                          track_support=track_support)
    return solve_projected(S, Hm, [g.label for g in gens], tau_s=tau_s,
                           rel_tau=rel_tau, max_condition=max_condition,
                           norm_floor=norm_floor, resources=resources)


# The Phase 2M-A baseline policy: every materialized operator row is kept for
# the life of the bank. Phase 2M-C introduces ``stream_recompute`` and
# ``disk_backed_csr`` beside it; until then the eviction, recomputation and
# spill counters below are structurally zero rather than merely unobserved, and
# the ledger says which of the two it is by naming the policy that produced it.
STORAGE_POLICY = "retain_all"


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
        self._H = as_multivector(hamiltonian)
        if not self._H.is_hermitian():
            raise ValueError("Hamiltonian must be Hermitian")
        if rho.n != self._H.n:
            raise ValueError("reference and Hamiltonian disagree on n")
        self._purity = check_reference(rho)
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
        # Candidate-sector diagnostics depend only on a fixed reference and a
        # generator, not on the growing Ritz basis.  Adaptive scoring revisits
        # every live candidate at each step, so retaining this cache avoids
        # rebuilding the same reference-conditioned projector contraction M
        # times on the fine-grained word pool.
        self._sector_leakage_cache: dict[tuple, tuple[dict[str, float], float]] = {}
        self._reference_sector_context: dict[tuple, tuple[Any, np.ndarray] | None] = {}

        self._observables: dict[tuple, dict] = {}
        self._pairs_built = 0
        self._cache_hits = 0
        self._seconds = 0.0
        # Counted rather than derived. Under ``retain_all`` the resident row
        # count never falls, so the peak is the current value and this counter
        # is redundant -- but it is the quantity Phase 2M-C's streaming and
        # disk-backed policies have to report against, and a field that starts
        # out derived from "nothing is ever freed" would quietly keep answering
        # for a policy that frees things.
        self._peak_resident_rows = 0
        self.extend(generators)

    # ------------------------------------------------------------- structure

    @property
    def n(self) -> int:
        return self._rho.n

    @property
    def reference(self) -> MV:
        """The single reference state every matrix element is an expectation on."""
        return self._rho

    @property
    def hamiltonian(self) -> MV:
        return self._H

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

    def resolve(self, indices: Sequence[int] | None) -> tuple[int, ...]:
        """Validate and canonicalize a generator subset."""
        if indices is None:
            return tuple(range(len(self._generators)))
        out = tuple(int(i) for i in indices)
        for i in out:
            if not (0 <= i < len(self._generators)):
                raise IndexError(f"generator index {i} out of range")
        if len(set(out)) != len(out):
            raise ValueError("repeated generator index in subset")
        return out

    # Compatibility for the pre-refactor implementation.
    _resolve = resolve

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
        self._peak_resident_rows = max(self._peak_resident_rows,
                                       self._resident_row_count())

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

    def observable_operator(self, observable, i: int, j: int, *,
                            label: str | None = None) -> MV:
        """Return ``A_i' Q A_j`` and register its measurement words.

        This is the operator-valued counterpart of :meth:`project_observable`.
        Exact observables pair it immediately with ``rho``; finite-shot response
        reconstruction keeps its coefficient map and estimates the same pairing
        from the shared word cache.  For Hermitian ``Q`` only the upper triangle
        is stored and the lower triangle is its adjoint, exactly as for ``S`` and
        ``H``.
        """
        Q = as_multivector(observable)
        if Q.n != self.n:
            raise ValueError("observable lives in a different algebra")
        for index in (i, j):
            if not 0 <= index < len(self._generators):
                raise IndexError(f"generator index {index} out of range")
        key = _identity_key(Q)
        record = self._observables.setdefault(
            key, {"label": label or f"Q{len(self._observables)}", "universe": set(),
                  "acted": {}, "operators": {}, "hermitian": Q.is_hermitian()})
        if record["hermitian"] and i > j:
            return self.observable_operator(Q, j, i, label=label).dagger()

        acted, cache = record["acted"], record["operators"]
        started = time.perf_counter()
        if j not in acted:
            acted[j] = Q * self._generators[j].mv
        if (i, j) not in cache:
            cache[(i, j)] = self._adjoints[i] * acted[j]
            record["universe"].update(cache[(i, j)].terms)
            self._peak_resident_rows = max(self._peak_resident_rows,
                                           self._resident_row_count())
        operator = cache[(i, j)]
        self._seconds += time.perf_counter() - started
        return operator

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
        Q = as_multivector(observable)
        if Q.n != self.n:
            raise ValueError("observable lives in a different algebra")
        order = self._resolve(indices)
        hermitian = Q.is_hermitian()

        m = len(order)
        out = np.zeros((m, m), dtype=complex)
        for b, j in enumerate(order):
            for a, i in enumerate(order):
                if hermitian and a > b:
                    continue
                operator = self.observable_operator(Q, i, j, label=label)
                out[a, b] = self._scale * operator.trace_pairing(self._rho)
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

    # ------------------------------------------------------- storage ledger

    def _resident_operators(self) -> list[MV]:
        """Every operator row the bank is holding, in one canonical order.

        The overlap and Hamiltonian rows of every pair ever built -- rejected candidates
        included under ``retain_all`` -- plus projected-observable rows. This is the set
        ``cached_operator_bytes`` has always priced; naming it once keeps byte estimates,
        coefficient counts and the measured walk on the same population.
        """
        return (list(self._overlap_ops.values()) + list(self._element_ops.values())
                + [op for record in self._observables.values()
                   for op in record["operators"].values()])

    def _resident_row_count(self) -> int:
        return (len(self._overlap_ops) + len(self._element_ops)
                + sum(len(record["operators"]) for record in self._observables.values()))

    def measured_storage_bytes(self) -> dict[str, Any]:
        """What the resident rows actually cost this interpreter, against the packed model.

        ``cached_operator_bytes`` prices Phase 2M-B's target representation and
        is exactly ``24 * coefficient_occurrences``; this walks the live objects
        and reports what the ``dict[int, complex]`` in front of that costs now.
        The quotient is the measured packing headroom 2M-B's go/no-go uses; PLAN.md
        section 5 records its earlier working figure as a hypothesis for this reason.

        Boxed keys and coefficients are deduplicated by identity. The word-code
        integers are shared across rows because ``_word_mul_unchecked`` is
        memoized, so a per-row sum counts one allocation many times and
        overstates what resident memory would actually recover.

        Not part of :meth:`resources`, like :meth:`qwc_group_count`: it is linear in the
        coefficient count with an identity set beside it, reaches tens of millions of
        entries on committed molecular banks, and ``resources`` runs every adaptive step.
        """
        rows = self._resident_operators()
        seen: set[int] = set()
        container_bytes = 0
        boxed_bytes = 0
        shared_boxes = 0
        occurrences = 0
        for row in rows:
            row_bytes, boxes = row.boxed_storage()
            container_bytes += row_bytes
            occurrences += row.nnz()
            for box in boxes:
                if id(box) in seen:
                    shared_boxes += 1
                    continue
                seen.add(id(box))
                boxed_bytes += sys.getsizeof(box)
        measured = container_bytes + boxed_bytes
        # Through ``memory_estimate`` rather than a literal, so the packed model
        # this quotient is taken against cannot drift away from the one
        # ``cached_operator_bytes`` reports and the committed records were
        # back-filled through.
        packed = sum(row.memory_estimate() for row in rows)
        return {
            "storage_policy": STORAGE_POLICY,
            "resident_operator_rows": len(rows),
            "coefficient_occurrences": occurrences,
            "packed_operator_bytes": packed,
            "measured_operator_bytes": measured,
            "measured_container_bytes": container_bytes,
            "measured_boxed_bytes": boxed_bytes,
            "shared_boxed_slots": shared_boxes,
            "distinct_boxed_objects": len(seen),
            "measured_bytes_per_coefficient": (measured / occurrences
                                               if occurrences else 0.0),
            "packing_headroom": (measured / packed) if packed else 0.0,
        }

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
        # A proper subset owns a smaller universe than the cache; quoting the cache's
        # would overstate measurement cost, especially with rejected candidate rows.
        universe = (len(self._universe) if len(order) == len(self._generators)
                    else len(self.word_set(order)))
        cached = self._resident_operators()
        occurrences = sum(op.nnz() for op in cached)
        resident_universe = len(self._universe)
        if self._observables:
            resident_universe = len(set(self._universe).union(
                *(record["universe"] for record in self._observables.values())))
        # The retained block is what a converged solve keeps; every other built
        # pair is frontier or rejected-candidate storage. Counting the pairs
        # actually materialized rather than the M(M+1)/2 the block would need
        # keeps the fraction below one on a bank whose block is still filling.
        retained_pairs = len(keys)
        complete_block_pairs = len(order) * (len(order) + 1) // 2
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
            # Phase 2M-A's storage ledger. Counts below describe the whole resident
            # cache; ``word_universe`` remains the selected subspace's measurement cost.
            # ``cached_operator_bytes`` is exactly 24 * ``coefficient_occurrences``.
            "coefficient_occurrences": occurrences,
            "resident_word_universe": resident_universe,
            "coefficient_reuse": occurrences / resident_universe if resident_universe else 0.0,
            "coefficient_occurrences_per_selected_element_word": (
                occurrences / universe if universe else 0.0),
            "storage_policy": STORAGE_POLICY,
            "resident_operator_rows": len(cached),
            "peak_resident_operator_rows": self._peak_resident_rows,
            "retained_block_pairs": retained_pairs,
            "complete_block_pairs": complete_block_pairs,
            "selection_pairs": self._pairs_built - retained_pairs,
            "retained_pair_fraction": (retained_pairs / self._pairs_built
                                       if self._pairs_built else 0.0),
            # Structurally zero under ``retain_all``: the bank has no eviction
            # path, so these are the baseline Phase 2M-C's policies report
            # against rather than quantities that went unobserved.
            "evicted_rows": 0,
            "recomputed_rows": 0,
            "spill_bytes": 0,
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
