r"""Classical selected-CI controls for the sampled-subspace arm.

Phase 9 of ``PLAN.md``, and the roadmap calls it mandatory for a
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
   or measured matrix-nonzero budget;
6. ``matched_selected_ci`` -- the same score at a declared budget, ranked from
   the reference determinant instead of the sample, so the row is what a
   laptop reaches without ever seeing the quantum draw;
7. ``random`` -- the floor: the same number of determinants drawn uniformly
   from the arm's own symmetry sector, scoring nothing.

Controls 2-6 are strictly classical.  Every determinant they add is one a
laptop could have found, so any advantage the hybrid claims has to survive
them.  (7) is the other end of the same comparison: a subspace that does not
beat chance at its own size is evidence about how many configurations the arm
found, not about which.

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

import hashlib
import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..backends.sector_statevector import SectorOperator
from ..pauli_action import PauliLinearOperator
from ..selection import TIE_ATOL, TIE_RTOL

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
    # Ground-state Ritz coefficients in ``determinants`` order.  Kept out of
    # ``to_record`` because Phase 11 consumes the vector in-memory as an
    # overlap target; benchmark JSON should not grow with the subspace.
    coefficients: np.ndarray | None = None

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


def _popcount(words: np.ndarray) -> np.ndarray:
    """Set bits per word, without assuming ``np.bitwise_count``.

    Added in NumPy 2.0; the package supports older runtimes, and the fallback
    is exact rather than approximate, so the two agree bit for bit.
    """
    words = np.asarray(words, dtype=np.int64)
    counter = getattr(np, "bitwise_count", None)
    if counter is not None:
        return counter(words).astype(np.int64)
    counts = np.zeros(words.shape, dtype=np.int64)
    remaining = words.copy()
    while np.any(remaining):
        counts += (remaining & 1).astype(np.int64)
        remaining >>= 1
    return counts


def _up_mask(n: int) -> int:
    """Bits of the spin-up orbitals, under the convention of :func:`_spin`.

    Orbital ``j`` occupies bit ``n - 1 - j``, and even ``j`` is spin up, so the
    mask is read off the same rule the closures use rather than re-derived.
    The down mask is its complement within the word and is never needed: a word
    with a known electron count and a known up count has its down count too.
    """
    return sum(1 << (n - 1 - j) for j in range(n) if _spin(j) == 0)


def _sector_pool(words: np.ndarray, *, n: int,
                 conserve_sz: bool = True) -> np.ndarray:
    """Full-space words sharing the sampled set's symmetry sector.

    The uniform control's pool.  A draw over the whole ``2**n`` Fock space is
    not matched to an arm that lives in one particle-number sector: most of its
    determinants land where the Hamiltonian has no coupling to the arm's
    subspace at all, so the resulting "chance floor" is the energy of a
    multi-sector junk subspace and any margin over it measures symmetry
    conservation rather than configuration selection.

    A sampled set that already spans sectors has no matched pool, and guessing
    one would put a mislabelled row in the comparison -- the same reason
    :func:`run_control`'s closure kinds refuse a reach that leaves the sector.
    """
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    up_mask = _up_mask(n)
    electrons = np.unique(_popcount(words))
    if electrons.size != 1:
        raise ValueError(
            "the sampled set spans several particle-number sectors "
            f"({electrons.tolist()}); a uniform control matched to it has no "
            "single sector to draw from")
    space = np.arange(2 ** n, dtype=np.int64)
    keep = _popcount(space) == electrons[0]
    if conserve_sz:
        up = np.unique(_popcount(words & up_mask))
        if up.size != 1:
            raise ValueError(
                f"the sampled set spans several S_z sectors (n_up in "
                f"{up.tolist()}); a uniform control matched to it has no "
                "single sector to draw from")
        keep &= (_popcount(space & up_mask) == up[0])
    return space[keep]


def _seed_label(seed) -> int | str:
    """A JSON-safe stamp of whatever ``default_rng`` accepted.

    ``int(seed)`` is wrong here: ``default_rng`` also takes a ``SeedSequence``,
    a sequence of ints, or a ``BitGenerator``, and coercing one of those raises
    *after* the draw has already succeeded -- a crash in the record-keeping
    rather than in the control.
    """
    try:
        return int(seed)
    except (TypeError, ValueError):
        return repr(seed)


def _index_digest(indices: np.ndarray) -> str:
    """Stable digest of a determinant set, for records that omit the set."""
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(
        ",".join(str(int(value)) for value in ordered).encode("utf-8")
    ).hexdigest()


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


def _amplitude_values(operator, indices: np.ndarray, coefficients: np.ndarray,
                      energy: float, diagonal) -> np.ndarray:
    """First-order amplitude estimate for every determinant outside ``indices``.

    Returned in the same order :func:`score_candidates` returns its candidates
    under ``score="first_order"``, so the two can be zipped.  Kept separate
    because ``score_candidates`` sorts by score and discards the mapping back to
    the unsorted candidate order that a joint seed/candidate ranking needs.
    """
    dimension, _ = _operator_space(operator)
    applied = operator.matvec(_embed(indices, coefficients, dimension))
    outside = np.ones(dimension, dtype=bool)
    outside[indices] = False
    candidates = np.flatnonzero(outside)
    if candidates.size == 0:
        return candidates, np.zeros(0)
    if diagonal is None:
        diagonal = _diagonal(operator, dimension)
    gap = energy - diagonal[candidates]
    numerator = np.abs(applied[candidates])
    safe = np.abs(gap) > 1e-12
    # Degenerate denominators are scored by coupling alone, exactly as
    # score_candidates does, rather than clamped to something large.
    values = np.where(safe, numerator / np.abs(np.where(safe, gap, 1.0)),
                      numerator)
    return candidates, values


def _reference_anchor(operator, base: np.ndarray, seed_vector: np.ndarray, *,
                      n: int | None = None) -> np.ndarray:
    """The single determinant a matched-budget selection starts from.

    A selection needs an anchor that is *not* itself chosen by the criterion,
    or the first pick is arbitrary.  The lowest-diagonal determinant of the
    operator's own space is the natural one -- it is the Hartree-Fock-like
    reference of that sector, is sample-independent, and is what a classical
    selected-CI calculation would start from.  Returned as a length-1 array so
    callers can concatenate it ahead of a ranking.
    """
    dimension, _ = _operator_space(operator)
    diagonal = _diagonal(operator, dimension)
    return np.asarray([int(np.argmin(diagonal))], dtype=np.int64)


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
                generators=None, seed: int | None = None) -> ControlResult:
    """One Phase 9 control over a sampled determinant set.

    ``kind`` is one of ``qsci``, ``family_closure``, ``excitation_closure``,
    ``selected_ci``, ``budget_matched``, ``matched_selected_ci``,
    ``iterated_selected_ci``, or ``random``.  ``iterated_selected_ci`` is the
    CIPSI-style sample-independent comparator: from the reference it adds one
    determinant per iteration, by the declared score against the re-solved
    Ritz vector, up to ``max_determinants``.  ``family_closure`` requires
    ``generators`` -- it measures a declared family's reach, and there is no
    such thing without the family.  ``sampled`` are indices into the operator's own space
    -- sector positions for a :class:`SectorOperator`, computational-basis words
    for a :class:`PauliLinearOperator` -- exactly as the QSCI arm returns them.

    ``budget_matched`` requires ``max_determinants`` or ``max_nonzeros``: a
    budget-matched control with no declared budget is just the unbudgeted one
    under a different name, and silently accepting that would put a mislabelled
    row in the comparison.

    ``random`` is the floor of the family: ``M`` determinants drawn uniformly
    from the arm's own symmetry sector, ignoring the sampled set's contents and
    matching only its size.  ``matched_selected_ci`` answers "does the quantum
    sample beat *smart* classical selection on this budget"; ``random`` answers
    "does it beat *chance* on this budget", which is the weaker question the arm
    has to pass before the stronger one means anything.  It needs ``seed``,
    since a control whose draw cannot be reproduced is not a control, and over a
    :class:`PauliLinearOperator` it needs ``n`` as well -- the draw is held to
    the sampled set's particle-number (and, under ``conserve_sz``, ``S_z``)
    sector, because a null drawn from the whole Fock space measures symmetry
    conservation rather than which configurations the arm found.  ``seed``
    belongs to that kind alone and is refused elsewhere.
    """
    if seed is not None and kind != "random":
        # Silently dropping it would mislabel the row: a caller that means
        # kind="random" but writes another kind, or a driver threading one
        # experiment seed through every control, would get a control that
        # reports a seed it never used -- the failure budget_matched and
        # matched_selected_ci already refuse for their own missing arguments.
        raise ValueError(f"seed is only used by the random control, not by "
                         f"{kind!r}; a seed that reaches a kind which cannot "
                         "draw with it is a mislabelled row, not a no-op")
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
    elif kind == "matched_selected_ci":
        # The sample-independent matched-budget comparator: what a purely
        # classical selected CI reaches on the same determinant budget without
        # ever seeing the quantum sample.  `budget_matched` answers "best M
        # determinants available to a method that has seen the sample"; this
        # answers "best M a laptop finds on its own".  Reporting both is what
        # separates a hybrid advantage from a sampling advantage.
        if max_determinants is None:
            raise ValueError(
                "matched_selected_ci needs max_determinants: an unbudgeted "
                "matched comparator is the unbudgeted selected_ci control")
        anchor = _reference_anchor(operator, np.unique(sampled), None, n=n)
        anchor_energy, anchor_vector, _ = _solve(operator, anchor)
        ranked, _, work = score_candidates(
            operator, anchor, anchor_vector, energy=anchor_energy, score=score,
            diagonal=diagonal)
        limit = max(1, int(max_determinants))
        indices = np.unique(np.concatenate([anchor, ranked])[:limit])
        metadata.update({
            "score": score,
            "anchor_determinant": int(anchor[0]),
            "selected_added": int(indices.size - 1),
            "candidates_scored": int(work),
            "sample_independent": True,
            "budget_semantics": (
                "reference determinant plus the highest-scoring determinants "
                "under one criterion, chosen without reference to the sampled "
                "set; the quantum sample informs neither the pool nor the order"),
        })

    elif kind == "iterated_selected_ci":
        # The stronger sample-independent comparator: a CIPSI-style loop that
        # re-solves after every addition, where `matched_selected_ci` ranks
        # once against the reference and can therefore only see what couples to
        # it. One determinant per iteration, taken greedily by the declared
        # score against the current Ritz vector, so the set for budget M is a
        # prefix of the set for M + 1 and a caller can compute one trajectory
        # for every budget. Ties -- spin partners score identically -- go to
        # the lowest index, the rule `canonical_argmax` applies elsewhere.
        if max_determinants is None:
            raise ValueError(
                "iterated_selected_ci needs max_determinants: the loop has no "
                "natural stopping size, and an unbudgeted one is full CI")
        limit = max(1, int(max_determinants))
        if diagonal is None:
            diagonal = _diagonal(operator, dimension)
        current = _reference_anchor(operator, np.unique(sampled), None, n=n)
        anchor = int(current[0])
        iterations = 0
        while current.size < limit:
            energy_now, vector_now, _ = _solve(operator, current)
            ranked, values, scored = score_candidates(
                operator, current, vector_now, energy=energy_now, score=score,
                diagonal=diagonal)
            work += scored
            if ranked.size == 0 or values[0] <= 0.0:
                break  # nothing outside couples to the current space
            best = float(values[0])
            tied = ranked[values >= best - (TIE_ATOL + TIE_RTOL * abs(best))]
            current = np.unique(np.append(current, int(tied.min())))
            iterations += 1
        indices = current
        metadata.update({
            "score": score,
            "anchor_determinant": anchor,
            "iterations": int(iterations),
            "selected_added": int(indices.size - 1),
            "candidates_scored": int(work),
            "closed_before_budget": bool(indices.size < limit),
            "sample_independent": True,
            "budget_semantics": (
                "reference determinant, then one determinant per iteration by "
                "the declared score against the re-solved Ritz vector, until the "
                "budget; the quantum sample informs neither the pool nor the order"),
        })

    elif kind == "random":
        # The null arm. It scores nothing and consults nothing: the sampled set
        # contributes its *size* and not one of its determinants, so any
        # advantage an arm shows over this row is attributable to which
        # configurations it found rather than how many.
        if seed is None:
            raise ValueError("the random control needs an explicit seed; an "
                             "unreproducible draw is not a control")
        base = np.unique(sampled)
        # The pool the draw is uniform *over*. A SectorOperator is already the
        # sector, so its whole space is the pool. A PauliLinearOperator is the
        # full 2^n Fock space, and drawing from that would put most of the
        # control's determinants in particle-number sectors the Hamiltonian
        # never couples to the arm's subspace -- the row would then measure
        # symmetry conservation rather than which configurations the arm found,
        # and every margin over it would be overstated. The other full-space
        # kinds guard the same boundary by refusing a closure that leaves the
        # sector; here the boundary has to be imposed on the pool instead,
        # because a uniform draw has nothing to leave.
        if basis is None:
            if n is None:
                raise ValueError("a full-space random control needs the qubit "
                                 "count n; without it the draw cannot be held "
                                 "to the sampled set's symmetry sector")
            pool = _sector_pool(base, n=n, conserve_sz=conserve_sz)
            sector_semantics = ("the sampled set's particle-number"
                                + (" and S_z" if conserve_sz else "")
                                + " sector of the full space")
        else:
            pool = np.arange(dimension, dtype=np.int64)
            sector_semantics = "the operator's sector"
        budget = (base.size if max_determinants is None
                  else max(1, int(max_determinants)))
        if budget > pool.size:
            raise ValueError(
                f"cannot draw {budget} distinct determinants from a space of "
                f"{pool.size}; a control larger than the space it samples is "
                "the full space under another name")
        rng = np.random.default_rng(seed)
        # Drawn order, not sorted order: a nonzero budget trims a *prefix*
        # below, and trimming a prefix of sorted indices would bias the control
        # toward the low end of the space instead of shrinking the draw.
        drawn = pool[rng.choice(pool.size, size=budget, replace=False)]
        if max_nonzeros is not None:
            # Same rule budget_matched uses: nonzeros are monotone in the
            # prefix length, so the largest admissible prefix is a binary
            # search. A null that silently ignores the budget it is supposed to
            # be matched to is the one arm that must not.
            low, high = 1, drawn.size
            while low < high:
                middle = (low + high + 1) // 2
                if np.count_nonzero(
                        operator.restrict(np.unique(drawn[:middle]))) <= max_nonzeros:
                    low = middle
                else:
                    high = middle - 1
            drawn = drawn[:low]
            metadata["nonzero_budget"] = int(max_nonzeros)
        indices = np.unique(drawn)
        metadata.update({
            "seed": _seed_label(seed),
            # Deliberately *not* ``sample_independent``: that flag is read as
            # "does not move with the draw" (see summarize_m7_replication.py),
            # and this arm moves with its own seed. It is the sample's
            # *contents* it never consults.
            "sample_contents_independent": True,
            "draw_dependent": True,
            "drawn_from": int(pool.size),
            # A seed alone does not pin the draw: numpy guarantees no
            # cross-version bit stream for ``Generator`` (see
            # ``clifford_qc.reproducibility.sampling_stream_mismatch``), and
            # ``choice(replace=False)`` in particular switches algorithms on a
            # heuristic. The digest lets a reader rebuilding this record detect
            # a changed draw instead of silently comparing a different control.
            "draw_sha256": _index_digest(indices),
            "overlap_with_sample": int(np.intersect1d(indices, base).size),
            "budget_semantics": (
                "determinants drawn uniformly without replacement from "
                f"{sector_semantics}; the quantum sample sets the count and "
                "nothing else"),
        })

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
        # One priority order over seed *and* candidates -- but the two are not
        # commensurable as scored above: seed determinants sit *inside* the
        # subspace and carry Ritz weights, while candidates sit outside it and
        # carry second-order lowerings.  Concatenating seed-then-candidates put
        # every seed determinant ahead of every candidate, so a budget below the
        # seed size admitted none of them and the control silently became a
        # truncation of the sample rather than a selection over it.  On
        # hubbard_2x3 that meant 68 sampled determinants, a budget of 7, and
        # `selected_added=0` out of 332 scored candidates.
        #
        # Rescoring against the reference determinant alone puts seed and
        # candidate determinants on one scale: every determinant except the
        # reference is outside that base, so all of them compete under the same
        # criterion.  Below the seed size this is now a genuine selection; at or
        # above it the prefix still contains the whole seed, so the unbudgeted
        # `selected_ci` behaviour is unchanged.
        rescored = False
        limit_request = (None if max_determinants is None
                         else max(1, int(max_determinants)))
        if limit_request is not None and limit_request < base.size:
            # Rank seed and outside determinants by one commensurable quantity:
            # the
            # estimated squared amplitude in the target wavefunction.  A seed
            # determinant already has that amplitude -- its Ritz weight
            # |c_d|^2.  A candidate's is the first-order estimate
            # |w_d / (E - H_dd)|^2 from the same seed solve.  Both are squared
            # coefficients of the same (unnormalized) vector, so comparing them
            # is meaningful in a way that comparing a Ritz weight against a
            # second-order *energy* lowering is not.
            #
            # This keeps the arm sample-dependent, which is the whole point of
            # it: the seed solve supplies both the vector and the energy
            # denominator.  Scoring against the reference determinant instead
            # would make the row identical to ``matched_selected_ci`` and
            # collapse two arms into one.
            outside, amplitudes = _amplitude_values(
                operator, base, seed_vector, seed_energy, diagonal)
            pool = np.concatenate([base, outside])
            weights = np.concatenate([
                np.abs(seed_vector) ** 2, np.abs(amplitudes) ** 2])
            ranked = pool[np.argsort(-weights, kind="stable")]
            rescored = True
        else:
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
        admitted = int(np.setdiff1d(indices, base).size)
        metadata.update({"score": score,
                         "selected_added": admitted,
                         "seed_truncated": int(max(0, base.size - limit)),
                         "candidates_scored": int(work),
                         "unified_amplitude_ranking": rescored})
        if rescored:
            metadata["budget_semantics"] = (
                "budget below the sampled seed size: seed and outside "
                "determinants ranked together by estimated squared amplitude "
                "(Ritz weight for the seed, first-order estimate outside) so "
                "both compete under one criterion. Seed determinants winning "
                "every slot is then a result about the sample, not an artifact "
                "of ranking them first")
    else:
        raise ValueError("kind must be qsci, family_closure, "
                         "excitation_closure, selected_ci, budget_matched, "
                         "matched_selected_ci, iterated_selected_ci, or random")

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
        exact_energy=exact_energy, metadata=metadata, coefficients=vector.copy())


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
