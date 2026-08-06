r"""Classical selected-CI controls for the sampled-subspace arm.

Phase 9 of ``LITERATURE_ROADMAP.md``, and the roadmap calls it mandatory for a
reason: a hybrid that dresses sampled determinants with excitation operators
may reach its accuracy by *spanning more determinants*, which is what ordinary
selected CI does classically and cheaply.  Without a control that expands the
same determinant space by classical means, a hybrid gain and a re-derivation of
selected CI are indistinguishable from the energy alone.

Four controls over one sampled determinant set ``D``:

1. ``qsci`` -- diagonalize ``span(D)`` and nothing else;
2. ``excitation_closure`` -- add every determinant reachable from ``D`` by the
   same singles and doubles the operator dressing uses;
3. ``selected_ci`` -- add determinants by a declared classical score, one pass;
4. ``budget_matched`` -- the same score, stopped at a declared determinant count
   or matrix-nonzero budget.

Controls 2-4 are strictly classical.  Every determinant they add is one a
laptop could have found, so any advantage the hybrid claims has to survive
them.

The span diagnostic answers the sharper question.  Operator-generated states
``A_mu|D_k>`` live *somewhere*; if that somewhere is inside the determinant
closure of the same ``D_k``, the operator form is a representation and a
measurement-cost choice, not a richer variational space.  Principal angles say
which case holds, and a rank comparison says whether the operator basis is at
least more compact.
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
    "excitation_closure",
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
    operator_columns: int
    determinant_columns: int

    def to_record(self) -> dict:
        return {
            "operator_rank": int(self.operator_rank),
            "determinant_rank": int(self.determinant_rank),
            "operator_columns": int(self.operator_columns),
            "determinant_columns": int(self.determinant_columns),
            "max_principal_angle": float(self.max_angle),
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
    """Every determinant reachable from ``words`` by singles and doubles.

    The classical control for operator dressing: the dressed family applies the
    same singles and doubles as *operators*, so the determinants they can reach
    are exactly this set.  Any direction the dressed family produces beyond this
    span is the thing §9 asks to have explained.

    Mirrors the symmetry rule of
    :func:`~clifford_qc.subspace.fermionic_generators.determinant_excitations`:
    a single needs matching spins, and a double needs the same number of down
    spins on each side.  The weaker parity test admits ``Delta S_z = +-2``
    doubles, so the closure would then be larger than the operators justify --
    and an oversized control makes the hybrid look better, which is the wrong
    direction for a control to err in.
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
                max_rank: int = 2, conserve_sz: bool = True) -> ControlResult:
    """One Phase 9 control over a sampled determinant set.

    ``kind`` is one of ``qsci``, ``excitation_closure``, ``selected_ci``, or
    ``budget_matched``.  ``sampled`` are indices into the operator's own space
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
        _, seed_vector, _ = _solve(operator, base)
        seed_energy = float(np.linalg.eigvalsh(operator.restrict(base))[0])
        candidates, values, work = score_candidates(
            operator, base, seed_vector, energy=seed_energy, score=score,
            diagonal=diagonal)
        budget = max_determinants
        if budget is None and max_nonzeros is not None:
            # Nonzeros are bounded by M^2, so the largest M that can satisfy a
            # nonzero budget is its square root. Reported as the derived figure
            # it is, rather than presented as an independent budget.
            budget = max(base.size, int(np.floor(np.sqrt(max_nonzeros))))
            metadata["budget_from_nonzeros"] = int(max_nonzeros)
        if budget is None:
            budget = base.size + int((values > 0).sum())
        take = max(0, int(budget) - int(base.size))
        indices = np.unique(np.concatenate([base, candidates[:take]]))
        metadata.update({"score": score, "selected_added": int(take),
                         "candidates_scored": int(work)})
    else:
        raise ValueError("kind must be qsci, excitation_closure, selected_ci, "
                         "or budget_matched")

    energy, vector, matrix = _solve(operator, indices)
    build_seconds = time.perf_counter() - start
    solve_start = time.perf_counter()
    variance = _variance(operator, indices, vector, energy, dimension)
    solve_seconds = time.perf_counter() - solve_start

    if max_nonzeros is not None:
        metadata["nonzero_budget"] = int(max_nonzeros)
    return ControlResult(
        name=name, determinants=indices, energy=energy, variance=variance,
        matrix_nonzeros=int(np.count_nonzero(matrix)),
        matrix_bytes=int(matrix.nbytes), selection_work=work,
        build_seconds=build_seconds, solve_seconds=solve_seconds,
        exact_energy=exact_energy, metadata=metadata)


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
    def orthonormal(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("each subspace needs at least one column")
        q, r = np.linalg.qr(matrix)
        keep = np.abs(np.diag(r)) > tol * max(1.0, float(np.abs(r).max()))
        if not keep.any():
            raise ValueError("subspace collapsed to rank zero at this tolerance")
        return q[:, keep]

    q_first, q_second = orthonormal(first), orthonormal(second)
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
    operator_rank = int(np.linalg.matrix_rank(operator_basis, tol=tol))
    determinant_rank = int(np.linalg.matrix_rank(determinant_basis, tol=tol))
    # Containment asks about the operator span's own directions, so the test
    # runs over its first `operator_rank` angles: extra angles only exist
    # because the closure is larger, and counting them would report every
    # genuinely contained operator basis as escaping.
    relevant = angles[:operator_rank] if operator_rank else angles
    max_angle = float(relevant.max()) if relevant.size else 0.0
    return SpanComparison(
        operator_rank=operator_rank, determinant_rank=determinant_rank,
        angles=angles, contained=bool(max_angle < 1e-7), max_angle=max_angle,
        operator_columns=int(operator_basis.shape[1]),
        determinant_columns=int(determinant_basis.shape[1]))
