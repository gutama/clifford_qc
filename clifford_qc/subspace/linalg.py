"""Deterministic linear algebra for projected subspace problems."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from ..multivector import MV
from ..selection import TIE_ATOL, TIE_RTOL, canonical_argmax

# Absolute floor on an overlap eigenvalue of the *normalized* S (unit diagonal,
# so its eigenvalues live in [0, M] and an absolute threshold is meaningful).
DEFAULT_TAU_S = 1e-10
# Largest condition number the retained subspace may carry.
DEFAULT_MAX_CONDITION = 1e12
# Floor on ||A_i|psi>||: below it the generator annihilates the reference and
# owns a zero row/column, which normalization cannot divide by.
DEFAULT_NORM_FLOOR = 1e-14


@dataclass(frozen=True)
class SubspaceResult:
    """Outcome of one exact fixed-basis A-CASE solve.

    ``coefficients`` is ``(M, k)``: column ``k`` holds the Ritz vector of
    ``energies[k]`` in the *original* generator coordinates, so the Ritz state
    is ``sum_i coefficients[i, k] A_i|psi>`` and rows of dropped generators are
    exactly zero. Columns are normalized to ``c' S c = 1``.

    ``overlap_eigenvalues`` are those of the normalized overlap matrix, largest
    first, *before* truncation; ``effective_rank`` and ``condition_number``
    describe what survived it.

    ``bank`` and ``indices`` are set when the solve came from a
    ``MatrixElementBank`` (Phase 2), and are what the projected-observable
    methods need. A result from :func:`solve_subspace` has no bank, so it
    answers energies and coefficients but not observables.
    """

    energies: tuple[float, ...]
    coefficients: np.ndarray
    basis_labels: tuple[str, ...]
    overlap_eigenvalues: tuple[float, ...]
    condition_number: float
    effective_rank: int
    resources: dict[str, Any] = field(default_factory=dict)
    bank: Any = None
    indices: tuple[int, ...] = ()

    @property
    def ground_energy(self) -> float:
        return self.energies[0]

    def ritz_vector(self, k: int = 0) -> np.ndarray:
        return self.coefficients[:, k]

    def with_bank(self, bank: Any, indices: Sequence[int]) -> "SubspaceResult":
        return replace(self, bank=bank, indices=tuple(indices))

    def _projected(self, observable) -> tuple[np.ndarray, np.ndarray]:
        if self.bank is None:
            raise ValueError(
                "projected observables need a MatrixElementBank; solve through "
                "MatrixElementBank.solve() rather than solve_subspace()")
        return (self.bank.project_observable(observable, self.indices),
                self.bank.matrices(self.indices)[0])

    def expectation(self, observable, k: int = 0) -> float:
        """``<Q>_k = (c_k' Q_sub c_k) / (c_k' S c_k)`` for Ritz root ``k`` (§8).

        The Ritz state is never formed: ``Q_sub`` comes from the same element
        machinery as ``(S, H)``. ``Q`` must be Hermitian, since only then is
        this an expectation at all -- for a general operator use
        :meth:`transition` with ``i == j``.
        """
        Q = observable if isinstance(observable, MV) else observable.to_mv()
        if not Q.is_hermitian():
            raise ValueError("expectation needs a Hermitian observable; "
                             "use transition(Q, k, k) for a general operator")
        Q_sub, S = self._projected(observable)
        c = self.coefficients[:, k]
        return float((c.conj() @ Q_sub @ c).real / (c.conj() @ S @ c).real)

    def transition(self, observable, i: int, j: int) -> complex:
        """``<Psi_i|Q|Psi_j>`` between Ritz roots, normalized in the ``S`` metric."""
        Q_sub, S = self._projected(observable)
        ci, cj = self.coefficients[:, i], self.coefficients[:, j]
        norm = math.sqrt((ci.conj() @ S @ ci).real * (cj.conj() @ S @ cj).real)
        return complex(ci.conj() @ Q_sub @ cj) / norm


def degenerate_blocks(values: np.ndarray, rtol: float, atol: float) -> list[tuple[int, int]]:
    """Half-open index ranges of ascending values that are equal to tolerance.

    The tolerance is anchored to the first eigenvalue in a candidate block.
    Comparing only neighbours allows a long chain of individually-close but
    collectively-distinct values to collapse into one false degeneracy.
    """
    if values.size == 0:
        return []
    blocks = []
    start = 0
    for i in range(1, values.size):
        anchor, current = float(values[start]), float(values[i])
        tol = atol + rtol * max(abs(anchor), abs(current))
        if current - anchor > tol:
            blocks.append((start, i))
            start = i
    blocks.append((start, values.size))
    return blocks


def fix_phase(v: np.ndarray) -> np.ndarray:
    """Pin an eigenvector's arbitrary global phase: largest component real positive.

    The pivot is chosen with the tolerant argmax the deterministic ADAPT
    selection uses, so two components that are equal by symmetry do not hand
    the phase to whichever one floating-point noise happened to favour.
    """
    magnitudes = np.abs(v)
    pivot = canonical_argmax(range(v.size), lambda k: float(magnitudes[k]),
                             rtol=TIE_RTOL, atol=TIE_ATOL)
    lead = v[pivot]
    if abs(lead) <= 0.0:
        return v
    return v * (np.conjugate(lead) / abs(lead))


def canonical_block(block: np.ndarray) -> np.ndarray:
    """Orthonormal basis of a degenerate eigenspace that ignores how it arrived.

    Any two eigenbases of the same degenerate eigenvalue differ by a unitary
    mixing, which LAPACK picks arbitrarily. The spectral projector
    ``P = B B'`` does not: it depends only on the matrix and the degeneracy
    grouping. So the basis is rebuilt from ``P`` by column-pivoted
    Gram-Schmidt over the canonical (index-ordered) coordinates -- pivoting for
    numerical stability, with the tolerant argmax breaking the ties that
    symmetry makes common, so the result is a function of ``P`` alone.
    """
    dim = block.shape[1]
    if dim <= 1:
        return block
    projector = block @ block.conj().T
    residual = projector.copy()
    chosen: list[np.ndarray] = []
    for _ in range(dim):
        norms = np.linalg.norm(residual, axis=0)
        pivot = canonical_argmax(range(norms.size), lambda k: float(norms[k]),
                                 rtol=TIE_RTOL, atol=TIE_ATOL)
        if norms[pivot] <= 1e-12:  # pragma: no cover - rank(P) == dim by construction
            raise np.linalg.LinAlgError("degenerate eigenspace lost rank")
        u = residual[:, pivot] / norms[pivot]
        chosen.append(u)
        residual = residual - np.outer(u, u.conj() @ residual)
    return np.column_stack(chosen)


def canonical_eigh(matrix: np.ndarray, *, rtol: float = TIE_RTOL,
                   atol: float = TIE_ATOL) -> tuple[np.ndarray, np.ndarray]:
    """``numpy.linalg.eigh`` with a reproducible eigenbasis.

    Eigenvalues ascending, degenerate eigenspaces re-canonicalized from their
    spectral projector, and every eigenvector's phase pinned. On a
    non-degenerate spectrum this is ``eigh`` plus the phase convention.
    """
    values, vectors = np.linalg.eigh(matrix)
    for start, stop in degenerate_blocks(values, rtol, atol):
        if stop - start > 1:
            candidate = canonical_block(vectors[:, start:stop])
            # A tolerance-grouped block is only safe to rotate if it is an
            # eigenspace to numerical precision.  Otherwise canonicalisation
            # mixes distinct near-null modes and destroys S-whitening.
            residual = matrix @ candidate - candidate * values[None, start:stop]
            scale = max(float(np.linalg.norm(matrix, ord=np.inf)), 1.0)
            limit = max(10.0 * atol,
                        100.0 * np.finfo(float).eps * scale)
            if float(np.linalg.norm(residual, ord=np.inf)) <= limit:
                vectors[:, start:stop] = candidate
    for k in range(vectors.shape[1]):
        vectors[:, k] = fix_phase(vectors[:, k])
    return values, vectors


def solve_projected(S: np.ndarray, Hm: np.ndarray, labels: Sequence[str] | None = None,
                    *, tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
                    max_condition: float = DEFAULT_MAX_CONDITION,
                    overlap_noise_floor: float = 0.0,
                    norm_floor: float = DEFAULT_NORM_FLOOR,
                    resources: dict | None = None) -> SubspaceResult:
    """The normalized, thresholded, deterministic generalized eigenproblem (§4.1.3).

    ``S`` and ``H`` are taken as Hermitian by construction; a caller that
    assembles them any other way is checked here rather than silently
    symmetrized. The retained subspace is decided on the *normalized* overlap
    matrix by four rules at once -- absolute floor ``tau_s``, relative floor
    ``rel_tau``, a cap on the retained condition number, and an optional
    ``overlap_noise_floor`` supplied by a measurement layer -- because the
    near-singular regime is generic here, not exceptional.  The numerical
    solver does not estimate that statistical floor itself; keeping the two
    layers separate prevents exact solves from acquiring a hidden shot model.
    """
    S = np.asarray(S, dtype=complex)
    Hm = np.asarray(Hm, dtype=complex)
    m = S.shape[0]
    if S.shape != (m, m) or Hm.shape != (m, m):
        raise ValueError("S and H must be square and the same size")
    for name, matrix in (("overlap", S), ("Hamiltonian", Hm)):
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0.0):
            raise ValueError(f"projected {name} matrix is not Hermitian")
    labels = tuple(labels) if labels is not None else tuple(f"A{i}" for i in range(m))
    if len(labels) != m:
        raise ValueError("labels and matrix size disagree")
    if overlap_noise_floor < 0.0 or not np.isfinite(overlap_noise_floor):
        raise ValueError("overlap_noise_floor must be nonnegative and finite")

    diagonal = np.clip(S.diagonal().real, 0.0, None)
    norms = np.sqrt(diagonal)
    live = norms > norm_floor
    if not live.any():
        raise ValueError("every generator annihilates the reference state")
    dropped = tuple(label for label, keep in zip(labels, live) if not keep)

    scaling = np.where(live, norms, 1.0)
    S_bar = (S / scaling[:, None]) / scaling[None, :]
    H_bar = (Hm / scaling[:, None]) / scaling[None, :]
    index = np.flatnonzero(live)
    S_bar = S_bar[np.ix_(index, index)]
    H_bar = H_bar[np.ix_(index, index)]

    overlap_values, overlap_vectors = canonical_eigh(S_bar)
    largest = float(overlap_values[-1])
    cutoff = max(tau_s, rel_tau * largest, largest / max_condition,
                 overlap_noise_floor)
    keep = overlap_values > cutoff
    if not keep.any():
        raise ValueError(f"overlap threshold {cutoff:g} retained no direction; "
                         f"largest overlap eigenvalue is {largest:g}")
    kept_values = overlap_values[keep]
    # X maps retained overlap modes to an S-orthonormal frame: X' S_bar X = 1.
    X = overlap_vectors[:, keep] / np.sqrt(kept_values)
    whitening_residual = float(np.linalg.norm(
        X.conj().T @ S_bar @ X - np.eye(kept_values.size), ord=np.inf))
    condition = float(kept_values[-1] / kept_values[0])
    whitening_limit = max(
        1e-10, 100.0 * np.finfo(float).eps * max(condition, 1.0))
    if whitening_residual > whitening_limit:
        raise np.linalg.LinAlgError(
            "overlap whitening failed: "
            f"||X^* S X - I||_inf={whitening_residual:.3g} exceeds "
            f"{whitening_limit:.3g}")
    H_tilde = X.conj().T @ H_bar @ X
    H_tilde = 0.5 * (H_tilde + H_tilde.conj().T)
    energies, ritz = canonical_eigh(H_tilde)

    coefficients = np.zeros((m, ritz.shape[1]), dtype=complex)
    # Back to the original generator coordinates: the normalized frame carries
    # A_i / scaling_i, so a normalized coefficient divides by the same factor.
    coefficients[index, :] = (X @ ritz) / scaling[index, None]

    # Rank at the numerical floor, against rank at the policy threshold: the
    # gap between them is how much of the basis the conditioning rule discards
    # beyond what round-off already destroyed.
    numerical_rank = int((overlap_values > 1e-14 * largest).sum())
    # Exactly zero for an exact assembly; nonzero once (S, H) are estimated,
    # where dropping those modes is a PSD repair that does *not* automatically
    # preserve the variational bound (plan §4, Q3).
    negative_modes = int((overlap_values < -1e-12 * max(largest, 1.0)).sum())
    out = dict(resources or {})
    out.update({
        "basis_size": m,
        "live_generators": int(live.sum()),
        "dropped_generators": dropped,
        "rank_before_truncation": numerical_rank,
        "retained_rank": int(keep.sum()),
        "retained_condition_number": condition,
        "whitening_residual_inf": whitening_residual,
        "overlap_threshold": float(cutoff),
        "overlap_noise_floor": float(overlap_noise_floor),
        "overlap_eigenvalue_max": largest,
        "overlap_eigenvalue_min": float(overlap_values[0]),
        "overlap_negative_modes": negative_modes,
    })
    return SubspaceResult(
        energies=tuple(float(e) for e in energies),
        coefficients=coefficients,
        basis_labels=labels,
        overlap_eigenvalues=tuple(float(v) for v in overlap_values[::-1]),
        condition_number=condition,
        effective_rank=int(keep.sum()),
        resources=out,
    )



# Compatibility aliases for callers that used the former solver internals.
_degenerate_blocks = degenerate_blocks
_fix_phase = fix_phase
_canonical_block = canonical_block
