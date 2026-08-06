r"""Classical selected-CI controls for the sampled-subspace arm.

Phase 9 of ``LITERATURE_ROADMAP.md``, and the roadmap calls it mandatory for a
reason: a hybrid that dresses sampled determinants with excitation operators
may reach its accuracy by *spanning more determinants*, which is what ordinary
selected CI does classically and cheaply.  Without a control that expands the
same determinant space by classical means, a hybrid gain and a re-derivation of
selected CI are indistinguishable from the energy alone.

Controls over one sampled determinant set ``D``:

1. ``qsci`` -- diagonalize ``span(D)`` and nothing else;
2. ``family_closure`` -- add every determinant the *declared generator family*
   actually reaches, which is §9's comparator;
3. ``excitation_closure`` -- add every determinant reachable by re-deriving
   singles and doubles from each determinant's own occupancy; a legitimate but
   strictly larger classical control, not a substitute for (2);
4. ``selected_ci`` -- add determinants by a declared classical score, one pass;
5. ``budget_matched`` -- the same score, stopped at a declared determinant count
   or measured matrix-nonzero budget.

Controls 2-5 are strictly classical.  Every determinant they add is one a
laptop could have found, so any advantage the hybrid claims has to survive
them.

Keeping (2) and (3) apart matters more than it looks.  A fixed pool built
relative to the reference determinant annihilates many sampled determinants and
moves different electrons in the rest, so its reach is far smaller: on the 2x2
Hubbard sector three sampled determinants reach 15 under the pool and all 36
under the per-determinant rule.  Feed the larger one to the span diagnostic and
containment becomes nearly tautological -- everything is inside a comparator
that is the whole sector.

The span diagnostic answers the sharper question.  Operator-generated states
``A_mu|D_k>`` live *somewhere*; if that somewhere is inside the determinant
closure of the same ``D_k``, the operator form is a representation and a
measurement-cost choice, not a richer variational space.  Containment is
decided by :func:`containment_residual`, which is directional; principal angles
are reported alongside but cannot decide it on their own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..backends.sector_statevector import SectorOperator
from ..pauli_action import PauliLinearOperator

__all__ = [
    "ControlResult",
    "SpanComparison",
    "containment_residual",
    "excitation_closure",
    "family_closure",
    "principal_angles",
    "run_control",
    "score_candidates",
    "span_comparison",
]

_SCORES = ("epstein_nesbet", "first_order", "coupling")


@dataclass(frozen=True)
class ControlResult:
    """One control arm, with the classical work it took to build it."""

    name: str
    determinants: np.ndarray
    energy: float
    variance: float
    matrix_nonzeros: int
    matrix_bytes: int
    # Candidate determinants *scored*, not retained. A control that reaches a
    # good energy by scoring the whole sector has not found a cheap route to
    # it, and reporting only the retained count would hide that.
    selection_work: int
    build_seconds: float
    solve_seconds: float
    exact_energy: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def determinant_count(self) -> int:
        return int(self.determinants.size)

    @property
    def error(self) -> float | None:
        if self.exact_energy is None:
            return None
        return float(self.energy - self.exact_energy)

    def to_record(self) -> dict:
        record = {
            "control": self.name,
            "determinant_count": self.determinant_count,
            "energy": float(self.energy),
            "variance": float(self.variance),
            "matrix_nonzeros": int(self.matrix_nonzeros),
            "matrix_bytes": int(self.matrix_bytes),
            "selection_work": int(self.selection_work),
            "build_seconds": float(self.build_seconds),
            "solve_seconds": float(self.solve_seconds),
        }
        if self.exact_energy is not None:
            record["exact_energy"] = float(self.exact_energy)
            record["error"] = float(self.error)
        record.update(self.metadata)
        return record


@dataclass(frozen=True)
class SpanComparison:
    """Whether an operator-generated basis leaves a determinant closure."""

    operator_rank: int
    determinant_rank: int
    angles: np.ndarray
    contained: bool
    max_angle: float
    containment_residual: float
    operator_columns: int
    determinant_columns: int

    def to_record(self) -> dict:
        return {
            "operator_rank": int(self.operator_rank),
            "determinant_rank": int(self.determinant_rank),
            "operator_columns": int(self.operator_columns),
            "determinant_columns": int(self.determinant_columns),
            "max_principal_angle": float(self.max_angle),
            "containment_residual": float(self.containment_residual),
            "contained_in_closure": bool(self.contained),
            "principal_angles": [float(a) for a in self.angles],
        }

    @property
    def verdict(self) -> str:
        """The §9 reading, so a caller cannot pick a flattering one.

        Three outcomes, not four.  §9 also names "a smaller operator-generated
        basis spanning a much larger determinant closure" as a compactness
        result, but *span* is not where that can be decided: a determinant
        closure's columns are distinct basis states and so are independent, and
        no set of fewer vectors spans a space of larger dimension.  A smaller
        operator rank therefore always means a proper subspace, and compactness
        has to be argued on **energy at matched size** against the controls in
        :func:`run_control` -- which is exactly why §9 demands those controls
        rather than this diagnostic alone.
        """
        if not self.contained:
            return ("outside closure: requires an algebraic explanation and an "
                    "independent check")
        if self.operator_rank < self.determinant_rank:
            return ("proper subspace: the operator family spans strictly less "
                    "than the determinant closure, so any compactness claim "
                    "must come from energy at matched size, not from span")
        return ("representation only: equal spans, so the operator form is a "
                "measurement-cost choice, not a richer variational space")


def _spin(index: int) -> int:
    """Interleaved convention: even spin-orbitals up, odd down."""
    return index % 2


def _occupied_bits(word: int, n: int) -> list[int]:
    return [j for j in range(n) if (word >> (n - 1 - j)) & 1]


def excitation_closure(words, *, n: int, max_rank: int = 2,
                       conserve_sz: bool = True) -> np.ndarray:
    """All spin-conserving singles and doubles applied to *each* determinant.

    A CISD-per-determinant closure, and deliberately **not** the closure of a
    fixed generator pool.  The distinction is the whole point and it is easy to
    lose: this routine re-derives excitations relative to every sampled
    determinant's own occupancy, whereas a family such as
    :func:`~clifford_qc.subspace.fermionic_generators.determinant_excitations`
    is built once relative to the *reference* determinant and then applied
    unchanged.  The two differ badly -- three sampled determinants of the 2x2
    Hubbard sector reach 15 determinants under that fixed pool and all 36 under
    this rule.

    Use :func:`family_closure` for the §9 comparator, which asks for the
    determinants reached by "the same singles/doubles used for operator
    dressing".  This function is a legitimate but *stronger* classical control
    -- larger space, better energy, harder for a hybrid to beat -- and it must
    not be substituted for the family closure in the span diagnostic, where an
    oversized comparator makes containment nearly tautological.

    Mirrors the symmetry rule of ``determinant_excitations``: a single needs
    matching spins, and a double needs the same number of down spins on each
    side.  The weaker parity test admits ``Delta S_z = +-2`` doubles.
    """
    if max_rank not in (1, 2):
        raise ValueError("max_rank must be 1 or 2")
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    if words.size == 0:
        raise ValueError("cannot close an empty determinant set")
    reached: set[int] = set(words.tolist())
    for word in words.tolist():
        occupied = _occupied_bits(word, n)
        virtual = [j for j in range(n) if j not in set(occupied)]
        for i in occupied:
            for a in virtual:
                if conserve_sz and _spin(i) != _spin(a):
                    continue
                reached.add((word & ~(1 << (n - 1 - i))) | (1 << (n - 1 - a)))
        if max_rank < 2:
            continue
        for i, j in combinations(occupied, 2):
            for a, b in combinations(virtual, 2):
                if conserve_sz and (_spin(i) + _spin(j)) != (_spin(a) + _spin(b)):
                    continue
                moved = word & ~(1 << (n - 1 - i)) & ~(1 << (n - 1 - j))
                reached.add(moved | (1 << (n - 1 - a)) | (1 << (n - 1 - b)))
    return np.array(sorted(reached), dtype=np.int64)


def family_closure(words, generators, *, backend=None, n: int | None = None,
                   tol: float = 1e-12) -> np.ndarray:
    """Determinants a declared generator family actually reaches (§9 control 2).

    The roadmap asks for "every unique determinant reached by the same
    singles/doubles used for operator dressing", which is a property of the
    *family*, not of a symmetry rule re-derived per determinant.  A fixed pool
    built relative to the reference determinant annihilates many sampled
    determinants and moves different electrons in the rest, so its reach is
    generally far smaller than :func:`excitation_closure` -- and the span
    diagnostic is only meaningful against this one.

    Computed by applying each generator to each determinant and collecting
    whatever acquires weight, so it inherits the generators' algebra instead of
    re-encoding it.  ``backend`` selects sector mode (``words`` are occupation
    words) and its absence selects full space (``words`` are basis indices,
    ``n`` required).
    """
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    if words.size == 0:
        raise ValueError("cannot close an empty determinant set")
    generators = list(generators)
    if not generators:
        raise ValueError("a family closure needs at least one generator")

    if backend is not None:
        positions = np.searchsorted(backend.basis, words)
        safe = np.where(positions < backend.basis.size, positions, 0)
        if not np.array_equal(backend.basis[safe], words):
            raise ValueError("determinants are not in this sector's basis")
        dimension, labels = backend.dimension, backend.basis
        compile_one = lambda mv: backend.operator(mv, validate_sector=False)
        seeds = safe
    else:
        if n is None:
            raise ValueError("full-space closure needs the qubit count n")
        dimension, labels = 2 ** n, None
        compile_one = PauliLinearOperator
        seeds = words

    reached = set(words.tolist())
    for generator in generators:
        compiled = compile_one(getattr(generator, "mv", generator))
        for index in seeds:
            probe = np.zeros(dimension, dtype=complex)
            probe[index] = 1.0
            hit = np.flatnonzero(np.abs(compiled.matvec(probe)) > tol)
            reached.update((labels[hit] if labels is not None else hit).tolist())
    return np.array(sorted(reached), dtype=np.int64)


def _operator_space(operator):
    """``(dimension, index_to_word)`` for either operator flavour."""
    if isinstance(operator, SectorOperator):
        return operator.dimension, operator.backend.basis
    if isinstance(operator, PauliLinearOperator):
        return operator.dimension, None
    raise TypeError("operator must be a SectorOperator or PauliLinearOperator")


def _embed(indices: np.ndarray, coefficients: np.ndarray,
           dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=complex)
    vector[indices] = coefficients
    return vector


def _diagonal(operator, dimension: int) -> np.ndarray:
    """``H_dd`` for every basis element, by one pass of unit vectors.

    Costs ``dimension`` matvecs, which is why callers cache it.  Written this
    way rather than by reaching into the compiled passes so it stays correct for
    both operator flavours without duplicating their internals.
    """
    diagonal = np.empty(dimension, dtype=float)
    probe = np.zeros(dimension, dtype=complex)
    for index in range(dimension):
        probe[index] = 1.0
        diagonal[index] = float(operator.matvec(probe)[index].real)
        probe[index] = 0.0
    return diagonal


def _solve(operator, indices: np.ndarray):
    """``(energy, ritz vector, matrix)`` on a declared index set."""
    matrix = operator.restrict(indices)
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    return float(values[0]), vectors[:, 0], matrix


def _variance(operator, indices: np.ndarray, coefficients: np.ndarray,
              energy: float, dimension: int) -> float:
    """``<H^2> - <H>^2`` on the Ritz vector, via one matvec.

    The honest convergence signal for a truncated space: a determinant set can
    give a good energy and a large variance, and §9 asks for both because the
    second is what says whether the first is converged or lucky.
    """
    psi = _embed(indices, coefficients, dimension)
    norm = float(np.vdot(psi, psi).real)
    if norm <= 0.0:
        raise ValueError("cannot take the variance of a zero state")
    applied = operator.matvec(psi / np.sqrt(norm))
    return float(np.vdot(applied, applied).real - energy ** 2)


def score_candidates(operator, indices, coefficients, *, energy: float,
                     score: str = "epstein_nesbet",
                     diagonal: np.ndarray | None = None):
    """Rank determinants outside the current set by their classical importance.

    One matvec gives every coupling at once: ``w = H|psi>`` has ``w[d]`` equal
    to ``sum_k H_dk c_k``, which is the numerator of first-order perturbation
    theory for every candidate simultaneously.  Enumerating candidate-by-
    candidate would recompute the same product.

    Scores, all declared rather than implicit:

    ``epstein_nesbet``
        ``|w_d|^2 / (E - H_dd)`` -- the second-order energy lowering, the
        CIPSI/Epstein-Nesbet criterion.
    ``first_order``
        ``|w_d / (E - H_dd)|`` -- the first-order wavefunction coefficient.
    ``coupling``
        ``|w_d|`` -- coupling magnitude alone, the HCI-flavoured screen that
        ignores the denominator.

    Returns ``(candidates, scores, work)`` sorted by descending score, where
    ``work`` counts every candidate scored.
    """
    if score not in _SCORES:
        raise ValueError(f"score must be one of {_SCORES}, got {score!r}")
    dimension, _ = _operator_space(operator)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    applied = operator.matvec(_embed(indices, coefficients, dimension))

    outside = np.ones(dimension, dtype=bool)
    outside[indices] = False
    candidates = np.flatnonzero(outside)
    if candidates.size == 0:
        return candidates, np.zeros(0), 0
    numerator = np.abs(applied[candidates])
    if score == "coupling":
        values = numerator
    else:
        if diagonal is None:
            diagonal = _diagonal(operator, dimension)
        gap = energy - diagonal[candidates]
        # A candidate degenerate with the current Ritz value has no perturbative
        # denominator. Rather than clamp it to a large finite score -- which
        # would promote it for being singular -- it is scored by coupling alone.
        safe = np.abs(gap) > 1e-12
        values = np.where(safe, numerator, 0.0)
        if score == "epstein_nesbet":
            values = np.where(safe, numerator ** 2 / np.abs(np.where(safe, gap, 1.0)),
                              numerator)
        else:
            values = np.where(safe, numerator / np.abs(np.where(safe, gap, 1.0)),
                              numerator)
    order = np.argsort(-values)
    return candidates[order], values[order], int(candidates.size)


def run_control(operator, sampled, *, name: str, kind: str, n: int | None = None,
                exact_energy: float | None = None,
                max_determinants: int | None = None,
                max_nonzeros: int | None = None,
                score: str = "epstein_nesbet",
                diagonal: np.ndarray | None = None,
                max_rank: int = 2, conserve_sz: bool = True,
                generators=None) -> ControlResult:
    """One Phase 9 control over a sampled determinant set.

    ``kind`` is one of ``qsci``, ``family_closure``, ``excitation_closure``,
    ``selected_ci``, or ``budget_matched``.  ``family_closure`` requires
    ``generators`` -- it measures a declared family's reach, and there is no
    such thing without the family.  ``sampled`` are indices into the operator's own space
    -- sector positions for a :class:`SectorOperator`, computational-basis words
    for a :class:`PauliLinearOperator` -- exactly as the QSCI arm returns them.

    ``budget_matched`` requires ``max_determinants`` or ``max_nonzeros``: a
    budget-matched control with no declared budget is just the unbudgeted one
    under a different name, and silently accepting that would put a mislabelled
    row in the comparison.
    """
    dimension, basis = _operator_space(operator)
    sampled = np.asarray(sampled, dtype=np.int64).reshape(-1)
    if sampled.size == 0:
        raise ValueError("cannot build a control from an empty sampled set")
    start = time.perf_counter()
    work = 0
    metadata: dict = {"kind": kind}

    if kind == "qsci":
        indices = np.unique(sampled)
    elif kind == "family_closure":
        if generators is None:
            raise ValueError("family_closure needs the generator family whose "
                             "reach it is meant to measure; without it there is "
                             "no declared family to close over")
        base = np.unique(sampled)
        if basis is None:
            if n is None:
                raise ValueError("full-space closure needs the qubit count n")
            reached = family_closure(base, generators, n=n)
            indices = np.unique(reached)
        else:
            reached = family_closure(basis[base], generators,
                                     backend=operator.backend)
            positions = np.searchsorted(basis, reached)
            safe = np.where(positions < basis.size, positions, 0)
            if not np.array_equal(basis[safe], reached):
                raise ValueError("the generator family left the sector; its "
                                 "reach and the sector disagree")
            indices = np.unique(safe)
        work = int(len(list(generators)) * base.size)
        metadata["closure_added"] = int(indices.size - base.size)
        metadata["generators"] = int(len(list(generators)))
    elif kind == "excitation_closure":
        if basis is None:
            if n is None:
                raise ValueError("full-space closure needs the qubit count n")
            closed = excitation_closure(sampled, n=n, max_rank=max_rank,
                                        conserve_sz=conserve_sz)
            indices = np.unique(closed)
        else:
            if n is None:
                n = operator.n
            closed = excitation_closure(basis[sampled], n=n, max_rank=max_rank,
                                        conserve_sz=conserve_sz)
            # The closure of a sector-conserving excitation stays in the sector,
            # so every reached word must be present; a miss is a symmetry bug,
            # not a determinant to drop.
            positions = np.searchsorted(basis, closed)
            safe = np.where(positions < basis.size, positions, 0)
            if not np.array_equal(basis[safe], closed):
                raise ValueError("excitation closure left the sector; the "
                                 "symmetry rule and the sector disagree")
            indices = np.unique(safe)
        work = int(indices.size)
        metadata["closure_added"] = int(indices.size - np.unique(sampled).size)
    elif kind in ("selected_ci", "budget_matched"):
        if kind == "budget_matched" and max_determinants is None \
                and max_nonzeros is None:
            raise ValueError("budget_matched needs max_determinants or "
                             "max_nonzeros; without a declared budget it is "
                             "the unbudgeted control under another name")
        base = np.unique(sampled)
        seed_energy, seed_vector, _ = _solve(operator, base)
        candidates, values, work = score_candidates(
            operator, base, seed_vector, energy=seed_energy, score=score,
            diagonal=diagonal)
        # One priority order over seed *and* candidates. The seed is ranked by
        # Ritz weight so that a budget below the seed size truncates the least
        # important determinants rather than silently overrunning -- a
        # budget-matched control that exceeds its budget is not matched.
        seed_order = base[np.argsort(-np.abs(seed_vector))]
        ranked = np.concatenate([seed_order, candidates])
        limit = ranked.size if max_determinants is None else int(max_determinants)
        limit = max(1, min(limit, ranked.size))
        if max_nonzeros is not None:
            # Actual Hamiltonian nonzeros, not a floor(sqrt(budget)) proxy for
            # them. Adding a determinant adds a row and a column and changes no
            # existing entry, so nonzeros are monotone in the prefix length and
            # the largest admissible prefix is a binary search.
            low, high = 1, limit
            while low < high:
                middle = (low + high + 1) // 2
                if np.count_nonzero(
                        operator.restrict(np.unique(ranked[:middle]))) <= max_nonzeros:
                    low = middle
                else:
                    high = middle - 1
            limit = low
            metadata["nonzero_budget"] = int(max_nonzeros)
        indices = np.unique(ranked[:limit])
        metadata.update({"score": score,
                         "selected_added": int(max(0, limit - base.size)),
                         "seed_truncated": int(max(0, base.size - limit)),
                         "candidates_scored": int(work)})
    else:
        raise ValueError("kind must be qsci, family_closure, "
                         "excitation_closure, selected_ci, or budget_matched")

    # Three phases, three clocks. Folding the eigensolve into `build_seconds`
    # and then naming the variance matvec `solve_seconds` would put the
    # diagonalization cost under the wrong heading in every comparison the
    # controls exist to support.
    matrix = operator.restrict(indices)
    build_seconds = time.perf_counter() - start

    solve_start = time.perf_counter()
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    energy, vector = float(values[0]), vectors[:, 0]
    solve_seconds = time.perf_counter() - solve_start

    variance_start = time.perf_counter()
    variance = _variance(operator, indices, vector, energy, dimension)
    metadata["variance_seconds"] = time.perf_counter() - variance_start

    return ControlResult(
        name=name, determinants=indices, energy=energy, variance=variance,
        matrix_nonzeros=int(np.count_nonzero(matrix)),
        matrix_bytes=int(matrix.nbytes), selection_work=work,
        build_seconds=build_seconds, solve_seconds=solve_seconds,
        exact_energy=exact_energy, metadata=metadata)


def _orthonormal(matrix, tol: float) -> np.ndarray:
    """Rank-revealing orthonormal basis of a column span, by SVD.

    Not by the diagonal of an unpivoted QR.  That test is not rank revealing
    and fails on ordinary inputs: for columns ``[e1, e1, e2]`` the R diagonal is
    ``[1, 0, 0]``, so a diagonal filter keeps one column for a span of rank two
    -- and a span diagnostic that silently under-counts the operator basis will
    report containment that is not there.
    """
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("each subspace needs at least one column")
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > tol * max(1.0, float(singular.max()))
    if not keep.any():
        raise ValueError("subspace collapsed to rank zero at this tolerance")
    return left[:, keep]


def containment_residual(operator_basis, determinant_basis, *,
                         tol: float = 1e-10) -> float:
    """``max_{a in span(A), |a|=1} dist(a, span(D))`` -- the containment measure.

    The quantity the §9 go/no-go actually needs, and the one principal angles
    cannot supply on their own: there are only ``min(rank A, rank D)`` of them,
    so an operator span of rank 2 sharing a single direction with a rank-1
    closure yields the single angle ``[0]`` and reads as contained while a whole
    direction sits outside.

    Computed as the largest singular value of ``(I - Q_D Q_D^H) Q_A``, which is
    the sine of the largest principal angle *of A into D* and is 1 whenever
    ``rank A > rank D``.  Directional by construction, where principal angles
    are symmetric.
    """
    q_operator = _orthonormal(operator_basis, tol)
    q_determinant = _orthonormal(determinant_basis, tol)
    if q_operator.shape[0] != q_determinant.shape[0]:
        raise ValueError("subspaces live in different ambient dimensions")
    residual = q_operator - q_determinant @ (q_determinant.conj().T @ q_operator)
    return float(np.linalg.svd(residual, compute_uv=False).max())


def principal_angles(first: np.ndarray, second: np.ndarray, *,
                     tol: float = 1e-10) -> np.ndarray:
    """Principal angles between two column spans, in radians and ascending.

    NumPy only.  SciPy has ``subspace_angles``, but it is an optional extra
    here and a diagnostic that decides a go/no-go should not be the thing that
    makes it mandatory.

    Small angles come from a **sine** formula, not from ``arccos`` of the
    singular values of ``Q1^H Q2``.  That cosine route is the textbook one and
    it is unusable for exactly the question §9 asks: ``arccos`` has infinite
    derivative at 1, so a singular value one ulp below it yields an angle near
    ``sqrt(2 * eps) ~ 2e-8`` rather than 0.  Two genuinely identical spans then
    report ``2e-8``, which leaves any containment threshold sitting a factor of
    a few above pure round-off.  Taking the singular values of the residual
    ``(I - Q1 Q1^H) Q2`` gives ``sin(theta)`` directly, which is accurate near
    zero; the Knyazev-Argentati hybrid below switches to the cosine route past
    ``pi/4`` where the roles reverse.
    """
    q_first, q_second = _orthonormal(first, tol), _orthonormal(second, tol)
    if q_first.shape[0] != q_second.shape[0]:
        raise ValueError("subspaces live in different ambient dimensions")

    overlap = q_first.conj().T @ q_second
    cosines = np.clip(np.linalg.svd(overlap, compute_uv=False), 0.0, 1.0)
    count = cosines.size  # min(k1, k2) principal angles

    residual = q_second - q_first @ overlap
    sines = np.clip(np.linalg.svd(residual, compute_uv=False), 0.0, 1.0)
    # `sines` is descending over k2 values; when the second span is wider its
    # extra directions are fully orthogonal and contribute sines of 1 at the
    # front. Reversing and truncating pairs the smallest sines with the
    # largest cosines, so both arrays index the same ascending angles.
    sines = sines[::-1][:count]

    return np.where(cosines > np.sqrt(0.5), np.arcsin(sines),
                    np.arccos(cosines))


def span_comparison(operator_basis: np.ndarray, determinant_basis: np.ndarray,
                    *, tol: float = 1e-10) -> SpanComparison:
    """Is the operator-generated span inside the determinant closure (§9)?

    ``operator_basis`` holds the dressed states ``A_mu|D_k>`` as columns;
    ``determinant_basis`` holds the closure's determinants as columns in the
    same ambient space.  Containment is decided on the largest principal angle
    of the operator span *into* the closure, which is the quantity that is zero
    exactly when every operator direction already lives there.
    """
    operator_basis = np.asarray(operator_basis, dtype=complex)
    determinant_basis = np.asarray(determinant_basis, dtype=complex)
    angles = principal_angles(operator_basis, determinant_basis, tol=tol)
    # Ranks from the same rank-revealing basis the residual uses, so the three
    # numbers on the record cannot disagree with each other.
    operator_rank = int(_orthonormal(operator_basis, tol).shape[1])
    determinant_rank = int(_orthonormal(determinant_basis, tol).shape[1])
    residual = containment_residual(operator_basis, determinant_basis, tol=tol)
    # Containment is decided on the directional residual, never on the angle
    # list: a rank-2 operator span sharing one direction with a rank-1 closure
    # produces exactly one angle, [0], and would otherwise read as contained.
    # The rank guard is redundant against the residual and kept because it
    # fails loudly for the same reason rather than silently.
    contained = bool(residual < 1e-7 and operator_rank <= determinant_rank)
    max_angle = float(angles.max()) if angles.size else 0.0
    return SpanComparison(
        operator_rank=operator_rank, determinant_rank=determinant_rank,
        angles=angles, contained=contained, max_angle=max_angle,
        containment_residual=residual,
        operator_columns=int(operator_basis.shape[1]),
        determinant_columns=int(determinant_basis.shape[1]))
