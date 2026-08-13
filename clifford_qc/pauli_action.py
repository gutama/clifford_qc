"""Matrix-free action of packed Pauli/Clifford operators on statevectors.

For a Pauli word

``W = i**n_Y X**x Z**z``

the computational-basis action is a signed permutation,

``W|b> = i**n_Y (-1)**popcount(z & b) |b xor x>``.

That identity is the scaling bridge from the sparse geometric-algebra
representation to statevector numerics: applying a ``K``-term operator costs
``O(K 2^n)`` work and ``O(2^n + K)`` storage, without constructing a
``2^n x 2^n`` matrix.  Dense conversion lives in :mod:`dense_reference` and is
used only as an oracle or interop boundary.
"""

from __future__ import annotations

import numpy as np

from .ir import PauliSum
from .multivector import MV, validate_word_code

_PHASE4 = (1 + 0j, 1j, -1 + 0j, -1j)


def _as_mv(operator) -> MV:
    mv = operator.to_mv() if isinstance(operator, PauliSum) else operator
    if not isinstance(mv, MV):
        raise TypeError(f"expected MV or PauliSum, got {type(operator).__name__}")
    return mv


def word_masks(n: int, code: int) -> tuple[int, int, int]:
    """Return ``(x_mask, z_mask, y_count)`` in computational-basis bit order.

    Qubit ``j`` is bit ``n-1-j`` of a basis index, matching the Kronecker order
    of :func:`clifford_qc.dense_reference.code_to_matrix`.
    """
    validate_word_code(n, code)
    x_mask = z_mask = y_count = 0
    for j in range(n):
        letter = (code >> (2 * j)) & 3
        bit = 1 << (n - 1 - j)
        if letter in (1, 2):
            x_mask |= bit
        if letter in (2, 3):
            z_mask |= bit
        if letter == 2:
            y_count += 1
    return x_mask, z_mask, y_count


def parity(values: np.ndarray, mask: int) -> np.ndarray:
    """Vectorized parity of ``popcount(values & mask)`` for int64 indices."""
    folded = values & mask
    for shift in (32, 16, 8, 4, 2, 1):
        folded = folded ^ (folded >> shift)
    return folded & 1


class PauliLinearOperator:
    """Compile an ``MV`` or ``PauliSum`` to matrix-free full-space action.

    Terms are grouped by their X mask so each distinct permutation of the
    statevector is gathered once per matvec.  Z/Y phases are evaluated on the
    fly rather than cached as ``K`` arrays of length ``2^n``; this keeps
    persistent storage linear in the state size plus the packed term list.
    """

    def __init__(self, operator):
        self.mv = _as_mv(operator)
        self.n = self.mv.n
        self.dimension = 2 ** self.n
        self.shape = (self.dimension, self.dimension)
        self.dtype = np.dtype(complex)
        self._index = np.arange(self.dimension, dtype=np.int64)
        groups: dict[int, list[tuple[int, complex]]] = {}
        for code, coeff in self.mv.terms.items():
            x_mask, z_mask, y_count = word_masks(self.n, code)
            phase = complex(coeff) * _PHASE4[y_count & 3]
            groups.setdefault(x_mask, []).append((z_mask, phase))
        self._groups = tuple((x_mask, tuple(entries))
                             for x_mask, entries in sorted(groups.items()))
        self.words = len(self.mv.terms)
        self.groups = len(self._groups)

    def _vector(self, psi) -> np.ndarray:
        vector = np.asarray(psi, dtype=complex).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError(f"state has {vector.size} amplitudes, expected "
                             f"{self.dimension} for n={self.n}")
        return vector

    def matvec(self, psi) -> np.ndarray:
        """Return ``H|psi>`` without materializing ``H``."""
        vector = self._vector(psi)
        out = np.zeros(self.dimension, dtype=complex)
        for x_mask, entries in self._groups:
            # For target t, the unique source is b=t xor x.  The Z phase is a
            # function of that source basis word, not the target.
            source = self._index ^ x_mask
            gathered = vector[source]
            for z_mask, phase in entries:
                signs = 1.0 - 2.0 * parity(source, z_mask)
                out += phase * signs * gathered
        return out

    def matmat(self, vectors) -> np.ndarray:
        """Return ``H @ vectors`` for a stack of statevector columns."""
        matrix = np.asarray(vectors, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != self.dimension:
            raise ValueError(f"vectors must have shape ({self.dimension}, m)")
        out = np.zeros_like(matrix, dtype=complex)
        for x_mask, entries in self._groups:
            source = self._index ^ x_mask
            gathered = matrix[source, :]
            for z_mask, phase in entries:
                signs = 1.0 - 2.0 * parity(source, z_mask)
                out += phase * signs[:, None] * gathered
        return out

    def expectation(self, psi) -> complex:
        """Normalized expectation ``<psi|H|psi>/<psi|psi>``."""
        vector = self._vector(psi)
        norm = float(np.vdot(vector, vector).real)
        if norm <= 0.0:
            raise ValueError("cannot take the expectation of a zero state")
        return complex(np.vdot(vector, self.matvec(vector))) / norm

    def as_linear_operator(self):
        """SciPy ``LinearOperator`` view, still without a stored matrix."""
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(self.shape, matvec=self.matvec, matmat=self.matmat,
                              dtype=complex)

    def restrict(self, indices) -> np.ndarray:
        """Exact ``H[I, I]`` on a declared set of full-space basis words.

        This is the sampled-subspace projection of ``PLAN.md``
        Phase 8B for models with no particle-number sector -- the spin arm.  It is a
        row/column restriction of the same compiled action the matvec uses, not
        a second Hamiltonian builder: whatever the word list means, the
        restricted matrix inherits it.

        ``indices`` are computational-basis words in the bit order of
        :func:`word_masks`.  Order is the caller's; the returned matrix is in
        that order, and permuting it similarity-transforms the matrix, so no
        eigenvalue moves.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("cannot restrict to an empty index set")
        if indices.min() < 0 or indices.max() >= self.dimension:
            raise ValueError(f"indices must lie in [0, {self.dimension})")
        if np.unique(indices).size != indices.size:
            raise ValueError("restriction indices must be distinct")
        # Membership is resolved by binary search on a sorted copy rather than
        # by a 2^n lookup table.  The table would be the obvious way to write
        # this and costs O(2^n) scratch on top of the caller's O(M^2) result;
        # `order` carries the caller's ordering back through the sort, so the
        # returned matrix is still in the order they asked for.
        order = np.argsort(indices)
        ascending = indices[order]
        rows_all = np.arange(indices.size, dtype=np.int64)
        out = np.zeros((indices.size, indices.size), dtype=complex)
        for x_mask, entries in self._groups:
            # Row `t` of the restricted matrix draws from the single source
            # `t xor x`; keep the pair only when both ends were sampled.
            sources = indices ^ x_mask
            slot = np.searchsorted(ascending, sources)
            safe = np.where(slot < indices.size, slot, 0)
            keep = ascending[safe] == sources
            if not keep.any():
                continue
            rows = rows_all[keep]
            live_sources, live_columns = sources[keep], order[safe[keep]]
            for z_mask, phase in entries:
                signs = 1.0 - 2.0 * parity(live_sources, z_mask)
                np.add.at(out, (rows, live_columns), phase * signs)
        return out

    def memory_estimate(self) -> dict[str, int]:
        """Persistent numeric storage, excluding Python-container overhead."""
        return {
            "index_bytes": int(self._index.nbytes),
            "term_bytes": 32 * self.words,
            "state_bytes": 16 * self.dimension,
            "dense_operator_bytes": 16 * self.dimension * self.dimension,
            "groups": self.groups,
            "words": self.words,
        }

    def ground_state(self, k: int = 1, *, method: str = "auto",
                     tol: float = 1e-9, maxiter: int | None = None,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Lowest ``k`` Hermitian eigenpairs with matrix-free Krylov action.

        ``'eigsh'`` feeds this object's matvec directly to ARPACK.  The
        Hamiltonian is shifted by its Pauli 1-norm so the ground state is a
        largest-magnitude extremal eigenpair, which is robust on highly
        degenerate spectra.  ``'lanczos'`` uses the package's NumPy-only
        fully-reorthogonalized implementation.  ``'auto'`` prefers ARPACK when
        available and otherwise selects that NumPy implementation.
        """
        if not self.mv.is_hermitian(1e-9):
            raise ValueError("ground_state requires a Hermitian Hamiltonian")
        if not isinstance(k, int) or not (1 <= k <= self.dimension):
            raise ValueError(f"k must be in [1, {self.dimension}]")
        if method == "auto":
            try:
                import scipy.sparse.linalg  # noqa: F401
            except ImportError:  # pragma: no cover - scipy is a research extra
                method = "lanczos"
            else:
                method = "eigsh" if k < self.dimension - 1 else "lanczos"
        if method == "lanczos":
            from .backends.sector_statevector import lanczos_ground

            iterations = 300 if maxiter is None else int(maxiter)
            return lanczos_ground(self.matvec, self.dimension, k=k,
                                  max_iter=iterations, tol=tol, seed=seed)
        if method == "eigsh":
            if k >= self.dimension - 1:
                raise ValueError("eigsh requires k < dimension - 1; use lanczos")
            from scipy.sparse.linalg import LinearOperator, eigsh

            shift = float(sum(abs(coeff) for coeff in self.mv.terms.values()))
            shifted = LinearOperator(
                self.shape, dtype=complex,
                matvec=lambda vector: self.matvec(vector) - shift * vector,
            )
            values, vectors = eigsh(shifted, k=k, which="LM", tol=tol,
                                    maxiter=maxiter)
            order = np.argsort(values)
            return values[order].real + shift, vectors[:, order]
        raise ValueError("method must be 'auto', 'eigsh', or 'lanczos'")


def apply_pauli_sum(operator, psi) -> np.ndarray:
    """One-shot matrix-free ``operator @ psi`` for an ``MV`` or ``PauliSum``."""
    return PauliLinearOperator(operator).matvec(psi)


def matrix_free_ground(operator, k: int = 1, **kwargs
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Lowest eigenpairs through :class:`PauliLinearOperator`."""
    return PauliLinearOperator(operator).ground_state(k=k, **kwargs)


__all__ = [
    "word_masks", "parity", "PauliLinearOperator", "apply_pauli_sum",
    "matrix_free_ground",
]
