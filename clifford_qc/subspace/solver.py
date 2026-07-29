"""A-CASE Phase 1: exact fixed-basis Rayleigh-Ritz in an operator-response subspace.

Given a reference state ``rho = |psi><psi|``, a Hamiltonian ``H``, and
generators ``{A_i}``, the method diagonalizes ``H`` in the span of the virtual
states ``A_i|psi>``. Nothing in that span is ever prepared: both projected
matrices are expectations on the *single* reference state,

    S_ij = <psi|A_i' A_j|psi> = Tr(A_i' A_j rho),
    H_ij = <psi|A_i' H A_j|psi> = Tr(A_i' H A_j rho),

and in the Pauli-word coordinates each is the bilinear pairing
``Tr(O rho) = 2^n sum_w o_w r_w`` -- one global word universe, every word
reusable across many entries. That sharing, not the use of Clifford algebra,
is the computational proposition; Phase 2 turns it into a cached bank and
Phase 4 replaces the exact pairing with a measured one.

Three things in the exact-arithmetic solve are load-bearing rather than
cosmetic:

*Normalization before thresholding.* Generators arrive on wildly different
scales (``H^k`` rows dwarf ``P_j`` rows). Thresholding the raw overlap matrix
would make the retained subspace depend on that arbitrary scaling, so ``S`` is
conjugated to unit diagonal first: with ``D_ii = sqrt(S_ii)``,
``S_bar = D^-1 S D^-1`` is invariant (up to a diagonal phase, which leaves its
spectrum alone) under ``A_i -> c_i A_i``.

*Structural Hermiticity.* ``S_ji = conj(S_ij)`` holds exactly by construction,
so only the upper triangle is computed and mirrored. The invariant is enforced
by the data layout, not by a numerical symmetrization that would hide a bug.

*Deterministic degenerate eigenspaces.* LAPACK's basis inside a degenerate
eigenspace is arbitrary and moves with the reduction order, the thread count,
and the BLAS vendor -- the same failure mode the deterministic ADAPT selection
on main was written for. Every eigenbasis here is re-canonicalized from its
spectral projector, which depends only on the matrix and the degeneracy
grouping, and every eigenvector's phase is pinned.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from ..algorithms.adapt import TIE_ATOL, TIE_RTOL, canonical_argmax
from ..ir import PauliSum
from ..multivector import MV
from .generators import as_generators

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


def _as_mv(operator) -> MV:
    if isinstance(operator, MV):
        return operator
    if isinstance(operator, PauliSum):
        return operator.to_mv()
    raise TypeError(f"expected MV or PauliSum, got {type(operator).__name__}")


def _check_reference(rho: MV) -> float:
    if not rho.is_hermitian():
        raise ValueError("reference state must be Hermitian")
    trace = rho.trace()
    if abs(trace - 1.0) > 1e-9:
        raise ValueError(f"reference state must have unit trace, got {trace}")
    # Tr(rho^2) = 2^n sum_w r_w^2 for Hermitian rho: the trace pairing again,
    # and O(nnz) where forming rho*rho is O(nnz^2).
    return float((2 ** rho.n) * rho.trace_pairing(rho).real)


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
    H = _as_mv(hamiltonian)
    if not H.is_hermitian():
        raise ValueError("Hamiltonian must be Hermitian")
    if H.n != gens[0].n or rho.n != H.n:
        raise ValueError("reference, Hamiltonian, and generators disagree on n")
    purity = _check_reference(rho)

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


def _degenerate_blocks(values: np.ndarray, rtol: float, atol: float) -> list[tuple[int, int]]:
    """Half-open index ranges of ascending values that are equal to tolerance.

    The tolerance is relative to the *neighbouring pair*, not to the spectrum's
    largest value. An overlap spectrum here routinely runs from ``O(M)`` down
    to ``1e-16``, and a tolerance scaled to its top would call two eigenvalues
    at 1e-10 and 3e-8 degenerate -- merging them into one canonical block whose
    shared eigenvalue is wrong for both, in exactly the near-null directions
    the thresholding rule is deciding on.
    """
    if values.size == 0:
        return []
    blocks = []
    start = 0
    for i in range(1, values.size):
        previous, current = float(values[i - 1]), float(values[i])
        tol = atol + rtol * max(abs(previous), abs(current))
        if current - previous > tol:
            blocks.append((start, i))
            start = i
    blocks.append((start, values.size))
    return blocks


def _fix_phase(v: np.ndarray) -> np.ndarray:
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


def _canonical_block(block: np.ndarray) -> np.ndarray:
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
    for start, stop in _degenerate_blocks(values, rtol, atol):
        if stop - start > 1:
            vectors[:, start:stop] = _canonical_block(vectors[:, start:stop])
    for k in range(vectors.shape[1]):
        vectors[:, k] = _fix_phase(vectors[:, k])
    return values, vectors


def solve_projected(S: np.ndarray, Hm: np.ndarray, labels: Sequence[str] | None = None,
                    *, tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
                    max_condition: float = DEFAULT_MAX_CONDITION,
                    norm_floor: float = DEFAULT_NORM_FLOOR,
                    resources: dict | None = None) -> SubspaceResult:
    """The normalized, thresholded, deterministic generalized eigenproblem (§4.1.3).

    ``S`` and ``H`` are taken as Hermitian by construction; a caller that
    assembles them any other way is checked here rather than silently
    symmetrized. The retained subspace is decided on the *normalized* overlap
    matrix by three rules at once -- absolute floor ``tau_s``, relative floor
    ``rel_tau``, and a cap on the retained condition number -- because the
    near-singular regime is generic here, not exceptional.
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
    cutoff = max(tau_s, rel_tau * largest, largest / max_condition)
    keep = overlap_values > cutoff
    if not keep.any():
        raise ValueError(f"overlap threshold {cutoff:g} retained no direction; "
                         f"largest overlap eigenvalue is {largest:g}")
    kept_values = overlap_values[keep]
    # X maps retained overlap modes to an S-orthonormal frame: X' S_bar X = 1.
    X = overlap_vectors[:, keep] / np.sqrt(kept_values)
    H_tilde = X.conj().T @ H_bar @ X
    energies, ritz = canonical_eigh(H_tilde)

    coefficients = np.zeros((m, ritz.shape[1]), dtype=complex)
    # Back to the original generator coordinates: the normalized frame carries
    # A_i / scaling_i, so a normalized coefficient divides by the same factor.
    coefficients[index, :] = (X @ ritz) / scaling[index, None]

    condition = float(kept_values[-1] / kept_values[0])
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
        "overlap_threshold": float(cutoff),
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
