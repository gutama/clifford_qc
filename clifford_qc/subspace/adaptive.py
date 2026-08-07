"""A-CASE Phase 3: exact adaptive basis growth.

One generator is added at a time, chosen by how much it can lower the current
Ritz value. The scoring is the §4.3 rule, and two parts of it are load-bearing:

*The overlap block is mandatory.* The predicted lowering comes from the
**generalized** 2x2 problem in ``span{Psi_m, chi_a}`` -- ``(E_m, h_a; h_a*,
h_aa)`` against ``(1, s_a; s_a*, s_aa)``. Dropping it -- assuming the candidate
is orthonormal to the current Ritz vector -- does not merely lose accuracy, it
invents lowering out of scaling: a candidate that *is* the current Ritz state
times 3.5 scores over a Hartree of predicted gain under the identity overlap
block, and exactly zero under the real one.

*Conditioning is a rejection criterion, not a post-hoc diagnostic.* A
candidate whose component orthogonal to the *retained subspace* falls below the
floor is rejected before it can enter the basis and wreck the overlap spectrum.
Measured against the whole subspace, not against the Ritz vector alone -- the
weaker test admits a candidate that duplicates some other basis direction, and
the thresholded solve then discards it after it has been paid for. The
near-singular regime is generic here, not exceptional.

Everything is exact. The scores are the quantities Phase 4 must estimate from
shared finite-shot measurements, so they are computed here through the same
bank entries that layer will reconstruct -- and a candidate's row stays cached
after a rejection, so scoring it again at the next step costs nothing.

Naming discipline (§4.4): what the selector uses is the *residual coupling*
``<chi_a|(H - E_m)|Psi_m>``, a projected quantity. It is never called a
residual norm. The true Ritz residual needs the second-moment matrix
``<psi|A_i' H^2 A_j|psi>``, which the projected ``(H, S)`` pair cannot supply;
``reference.dense_residual_norm`` computes it densely for small validation
runs, and nothing else claims to.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..selection import TIE_ATOL, TIE_RTOL, canonical_argmax
from ..multivector import MV
from .elements import MatrixElementBank
from .generator_core import Generator, as_generators
from .generators import identity_generator
from .contracts import as_multivector
from .solver import (DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S,
                     SubspaceResult)
from .symmetry import sector_leakage

# Floor on a candidate's S-orthogonal fraction. Below it the candidate is
# (numerically) already in the span and adds conditioning damage, not a
# direction.
DEFAULT_MIN_ORTHOGONALITY = 1e-8
# Floor on the predicted lowering: below it, growth has nothing left to buy.
DEFAULT_MIN_LOWERING = 1e-10
# Relative commutator norm above which a generator counts as leaving the sector.
DEFAULT_LEAKAGE_TOL = 1e-9
# Squared orthogonal norm below which the 2x2 deflation stops being computable
# in double precision (see _two_by_two_lowering). Four orders below the
# orthogonality floor, so live scoring never reaches it.
_GAP_FLOOR = 1e-12


@dataclass(frozen=True)
class CandidateScore:
    """One candidate weighed against the current Ritz pair (or several of them).

    With more than one root the reported ``residual_coupling`` and
    ``predicted_lowering`` are the aggregate the selector ranks on, and
    ``per_root`` keeps the individual lowerings so a record shows which root
    wanted the generator.
    """

    index: int
    label: str
    residual_coupling: float
    predicted_lowering: float
    orthogonal_fraction: float
    new_words: int
    score: float
    rejected: str | None = None
    leakage: dict[str, float] | None = None
    per_root: tuple[float, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.rejected is None


@dataclass(frozen=True)
class OverlapTarget:
    """A classical target vector expressed in virtual A-CASE directions.

    ``coefficients`` multiply ``generators`` in the same order.  The target is
    deliberately *not* part of :class:`CandidateScore`: energy lowering and
    target overlap are different scientific questions and remain separately
    testable.  Normalization is evaluated with the matrix-element bank's
    overlap metric when the target is scored, so neither the generators nor
    their coefficients need a special normalization convention.
    """

    generators: tuple[Generator, ...]
    coefficients: tuple[complex, ...]
    label: str = "target"

    def __post_init__(self):
        if not self.generators:
            raise ValueError("an overlap target needs at least one generator")
        if len(self.generators) != len(self.coefficients):
            raise ValueError("target generators and coefficients have different lengths")
        if not any(abs(value) > 0.0 for value in self.coefficients):
            raise ValueError("an overlap target cannot be the zero vector")

    @classmethod
    def from_coefficients(cls, generators: Sequence, coefficients, *,
                          label: str = "target") -> "OverlapTarget":
        gens = tuple(as_generators(generators))
        values = np.asarray(coefficients, dtype=complex).reshape(-1)
        return cls(gens, tuple(complex(value) for value in values), label)

    @classmethod
    def from_qsci(cls, generators: Sequence, result, *, root: int = 0,
                  label: str | None = None) -> "OverlapTarget":
        """Build a target from a QSCI Ritz vector aligned with ``generators``."""
        vectors = getattr(result, "eigenvectors", None)
        if vectors is None:
            raise ValueError("the QSCI result does not carry Ritz eigenvectors")
        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or not 0 <= root < vectors.shape[1]:
            raise IndexError("QSCI target root is outside the retained Ritz vectors")
        return cls.from_coefficients(
            generators, vectors[:, root],
            label=label or f"qsci_root_{root}",
        )

    @classmethod
    def from_selected_ci(cls, generators: Sequence, result, *,
                         label: str = "selected_ci") -> "OverlapTarget":
        """Build a target from a selected-CI control's retained Ritz vector."""
        coefficients = getattr(result, "coefficients", None)
        if coefficients is None:
            raise ValueError("the selected-CI result does not carry Ritz coefficients")
        return cls.from_coefficients(generators, coefficients, label=label)


@dataclass(frozen=True)
class TargetOverlapScore:
    """Incremental target weight captured by one S-orthogonal direction."""

    index: int
    label: str
    overlap_gain: float
    orthogonal_fraction: float
    score: float
    rejected: str | None = None
    leakage: dict[str, float] | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected is None


@dataclass(frozen=True)
class GrowthRecord:
    """What one accepted (or refused) growth step did."""

    step: int
    selected_label: str | None
    energy: float
    predicted_lowering: float
    actual_lowering: float
    residual_coupling: float
    orthogonal_fraction: float
    basis_size: int
    effective_rank: int
    condition_number: float
    candidates_scored: int
    rejected_conditioning: int
    rejected_sector: int
    new_words: int
    word_universe: int
    leakage: dict[str, float] | None = None
    # State-averaged / block growth: the tracked roots' energies at this step,
    # and the per-root predicted lowerings of the accepted generator.
    root_energies: tuple[float, ...] = ()
    per_root_lowering: tuple[float, ...] = ()
    selection_criterion: str = "lowering"
    selection_score: float = 0.0
    target_overlap_gain: float | None = None


@dataclass(frozen=True)
class AdaptiveResult:
    """Outcome of an adaptive run: the final solve plus how it was reached."""

    labels: tuple[str, ...]
    energy: float
    energy_history: tuple[float, ...]
    records: tuple[GrowthRecord, ...]
    result: SubspaceResult
    bank: MatrixElementBank
    indices: tuple[int, ...]
    stopped_reason: str
    exact_ground_energy: float | None = None
    relative_error: float | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    # ``energy`` and ``energy_history`` track the growth *objective*: the ground
    # Ritz value for a single root, the average of the tracked roots for a
    # state-averaged run. ``root_energies`` is the final per-root spectrum.
    root_energies: tuple[float, ...] = ()

    @property
    def basis_size(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class ACASEConfig:
    """Serializable controls for exact adaptive subspace growth.

    The matrix-element bank and generator sets stay runtime inputs; numerical
    thresholds and stopping policy live here so an A-CASE run has one explicit
    configuration boundary instead of a long list of independent tolerances.
    """

    max_size: int = 10
    roots: int = 1
    aggregation: str = "mean"
    min_lowering: float = DEFAULT_MIN_LOWERING
    min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY
    gamma: float = 0.0
    criterion: str = "lowering"
    min_target_overlap: float = 1e-12
    leakage_tol: float | None = None
    exact_ground_energy: float | None = None
    target_error: float | None = None
    tau_s: float = DEFAULT_TAU_S
    rel_tau: float = 0.0
    max_condition: float = DEFAULT_MAX_CONDITION

    def __post_init__(self):
        if self.max_size < 0:
            raise ValueError("max_size must be nonnegative")
        if self.roots < 1:
            raise ValueError("roots must be at least 1")
        _aggregate([0.0], self.aggregation)
        if self.min_lowering < 0.0:
            raise ValueError("min_lowering must be nonnegative")
        if self.min_orthogonality < 0.0:
            raise ValueError("min_orthogonality must be nonnegative")
        if self.gamma < 0.0:
            raise ValueError("gamma must be nonnegative")
        if self.criterion not in ("lowering", "target_overlap"):
            raise ValueError("criterion must be 'lowering' or 'target_overlap'")
        if self.min_target_overlap < 0.0:
            raise ValueError("min_target_overlap must be nonnegative")
        if self.leakage_tol is not None and self.leakage_tol < 0.0:
            raise ValueError("leakage_tol must be nonnegative")
        if self.target_error is not None and self.target_error < 0.0:
            raise ValueError("target_error must be nonnegative")
        if self.tau_s < 0.0 or self.rel_tau < 0.0:
            raise ValueError("overlap thresholds must be nonnegative")
        if self.max_condition <= 0.0:
            raise ValueError("max_condition must be positive")


@dataclass(frozen=True)
class ACASEState:
    """Immutable orchestration state between A-CASE growth decisions."""

    basis: tuple[int, ...]
    pool: tuple[int, ...]
    result: SubspaceResult
    energy: float
    energy_history: tuple[float, ...]
    records: tuple[GrowthRecord, ...]
    root_energies: tuple[float, ...]
    stopped_reason: str = "basis budget reached"


@dataclass(frozen=True)
class ACASEStepOutcome:
    """Numerical/scoring effects for one A-CASE growth decision."""

    selected_index: int | None = None
    result: SubspaceResult | None = None
    energy: float | None = None
    record: GrowthRecord | None = None
    root_energies: tuple[float, ...] | None = None
    stopped_reason: str | None = None


def acase_step(state: ACASEState, outcome: ACASEStepOutcome) -> ACASEState:
    """Pure state transition for an already-scored A-CASE growth step."""
    basis = state.basis
    if outcome.selected_index is not None:
        basis += (int(outcome.selected_index),)
    energy = state.energy if outcome.energy is None else float(outcome.energy)
    history = state.energy_history
    if outcome.energy is not None:
        history += (energy,)
    return ACASEState(
        basis=basis,
        pool=state.pool,
        result=state.result if outcome.result is None else outcome.result,
        energy=energy,
        energy_history=history,
        records=(state.records if outcome.record is None
                 else state.records + (outcome.record,)),
        root_energies=(state.root_energies if outcome.root_energies is None
                       else tuple(float(value) for value in outcome.root_energies)),
        stopped_reason=(state.stopped_reason if outcome.stopped_reason is None
                        else outcome.stopped_reason),
    )


def _two_by_two_lowering(energy: float, s_a: complex, h_a: complex,
                         s_aa: float, h_aa: float) -> float:
    """Lowest eigenvalue drop of the generalized 2x2 problem of §4.3.

    Solved in closed form on the *normalized* pair (candidate scaled to unit
    norm), so the answer does not depend on how the candidate happens to be
    scaled, and the overlap block is carried explicitly.

    Below ``_GAP_FLOOR`` the deflation is a genuine 0/0: the deflated diagonal
    is a difference of same-size terms divided by a vanishing gap, so double
    precision returns noise -- and the noise is *not* small. An exactly
    parallel candidate lands at a deflated diagonal of -6 instead of -4 and
    would be credited with two Hartree of lowering it cannot deliver. Zero is
    the right answer there, and the caller's orthogonality rejection (four
    orders above this floor) is what keeps live scoring away from the cliff.
    """
    if s_aa <= 0.0:
        return 0.0
    root = math.sqrt(s_aa)
    sigma = s_a / root           # <Psi_m|chi_hat>
    h = h_a / root               # <Psi_m|H|chi_hat>
    h_bb = h_aa / s_aa           # <chi_hat|H|chi_hat>
    gap = 1.0 - abs(sigma) ** 2  # squared norm of the orthogonal component
    if gap <= _GAP_FLOOR:
        return 0.0
    # Deflate to the orthonormal pair {Psi_m, (chi_hat - sigma Psi_m)/sqrt(gap)}:
    # the generalized problem becomes an ordinary 2x2 Hermitian one.
    root_gap = math.sqrt(gap)
    off = (h - sigma * energy) / root_gap
    diag = (h_bb - 2.0 * (np.conjugate(sigma) * h).real
            + abs(sigma) ** 2 * energy) / gap
    half_trace = 0.5 * (energy + diag)
    half_diff = 0.5 * (energy - diag)
    lowest = half_trace - math.sqrt(half_diff ** 2 + abs(off) ** 2)
    return max(0.0, energy - lowest)


def _aggregate(values: Sequence[float], aggregation: str) -> float:
    """Combine per-root scores into the one number the selector ranks on.

    ``mean`` is state-averaged adaptation: the basis is grown for the average of
    the tracked roots, which is what an excited-state calculation wants when the
    roots are to be described together. ``max`` is block adaptation: whichever
    root gains most decides, which chases the worst-described root instead of
    the average. Both are in the plan's §5 language, and they differ in kind --
    an average will never spend a generator on a root that only one state needs.
    """
    if not values:
        return 0.0
    if aggregation == "mean":
        return float(sum(values) / len(values))
    if aggregation == "max":
        return float(max(values))
    raise ValueError("aggregation must be 'mean' or 'max'")


def score_candidate(bank: MatrixElementBank, basis: Sequence[int],
                    result: SubspaceResult, candidate: int, *,
                    roots: Sequence[int] = (0,), aggregation: str = "mean",
                    basis_words: frozenset[int] | None = None,
                    min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
                    gamma: float = 0.0, leakage_tol: float | None = None
                    ) -> CandidateScore:
    """Weigh one candidate against the Ritz pairs ``roots`` of ``result``.

    Builds the candidate's row of the bank -- ``M`` pair products -- which is
    the honest cost of exact selection and precisely what Phase 4 replaces with
    a shared measurement. The row is cached either way, so a rejected candidate
    is free to re-score at the next step.

    ``orthogonal_fraction`` is measured against the *whole retained subspace*,
    not against the Ritz vector alone. The distinction is not academic: a
    candidate that duplicates some other basis direction while sitting at a
    healthy angle to the current Ritz vector would pass the weaker test, enter
    the basis, and make ``S`` exactly singular -- the thresholded solve would
    then silently throw the direction away after it had been paid for.
    """
    label = bank.generator(candidate).label
    leakage = None
    if leakage_tol is not None:
        leakage = sector_leakage(bank.generator(candidate))
        worst = max(leakage.values())
        if worst > leakage_tol:
            return CandidateScore(candidate, label, 0.0, 0.0, 0.0, 0, 0.0,
                                  rejected=f"sector leakage {worst:.2e}",
                                  leakage=leakage)

    s_aa, h_aa = bank.entry(candidate, candidate)
    s_aa, h_aa = s_aa.real, h_aa.real
    if s_aa <= DEFAULT_NORM_FLOOR ** 2:
        return CandidateScore(candidate, label, 0.0, 0.0, 0.0, 0, 0.0,
                              rejected="annihilates the reference", leakage=leakage)

    tracked = tuple(r for r in roots if r < len(result.energies))
    if not tracked:
        tracked = (0,)
    column = np.array([bank.entry(i, candidate) for i in basis], dtype=complex)
    s_column, h_column = column[:, 0], column[:, 1]
    # The Ritz vectors are S-orthonormal, so they are an orthonormal basis of
    # the retained subspace and the projection is a plain sum of squares. It is
    # the same for every root, which is why the conditioning rejection below is
    # root-independent.
    projections = result.coefficients.conj().T @ s_column
    couplings, blocks = [], []
    for r in tracked:
        energy_r = result.energies[r]
        s_a = complex(projections[r])
        h_a = complex(result.coefficients[:, r].conj() @ h_column)
        couplings.append(abs(h_a - energy_r * s_a) / math.sqrt(s_aa))
        blocks.append((energy_r, s_a, h_a))

    orthogonal_fraction = max(0.0, 1.0 - float(np.sum(np.abs(projections) ** 2)) / s_aa)
    # Scale-free residual coupling: the overlap of the candidate's orthogonal
    # component with the residual (H - E)|Psi_m>.
    coupling = _aggregate(couplings, aggregation)

    if basis_words is None:
        basis_words = bank.word_set(basis)
    row_words: set[int] = set()
    for i in tuple(basis) + (candidate,):
        row_words.update(bank.overlap_operator(i, candidate).terms)
        row_words.update(bank.element_operator(i, candidate).terms)
    new_words = len(row_words - basis_words)

    if orthogonal_fraction < min_orthogonality:
        return CandidateScore(candidate, label, coupling, 0.0, orthogonal_fraction,
                              new_words, 0.0,
                              rejected=f"orthogonal fraction {orthogonal_fraction:.2e}",
                              leakage=leakage)

    per_root = tuple(_two_by_two_lowering(energy_r, s_a, h_a, s_aa, h_aa)
                     for energy_r, s_a, h_a in blocks)
    lowering = _aggregate(per_root, aggregation)
    score = lowering if gamma == 0.0 else lowering / (1.0 + new_words) ** gamma
    return CandidateScore(candidate, label, coupling, lowering, orthogonal_fraction,
                          new_words, score, leakage=leakage, per_root=per_root)


def select_candidate(scores: Sequence[CandidateScore]) -> CandidateScore | None:
    """Highest-scoring live candidate, with the deterministic tie rule.

    Symmetric Hamiltonians produce exactly tied candidates in numbers, and a
    strict ``max`` would hand the step to whichever one a reduction-order
    perturbation of ~1e-15 happened to favour -- the failure the deterministic
    ADAPT selection on main was written for. The same tolerant argmax is used
    here, so the tied set is the symmetry class and the lowest bank id wins.
    """
    live = [s for s in scores if s.accepted]
    if not live:
        return None
    best = canonical_argmax(range(len(live)), lambda k: live[k].score,
                            rtol=TIE_RTOL, atol=TIE_ATOL)
    return live[best]


def score_target_overlap(bank: MatrixElementBank, basis: Sequence[int],
                         result: SubspaceResult, candidate: int,
                         target: OverlapTarget, *,
                         min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
                         leakage_tol: float | None = None) -> TargetOverlapScore:
    """Score the new target weight captured by a candidate direction (§11A).

    The candidate is first deflated against the *entire* retained Ritz space,
    exactly as in :func:`score_candidate`.  If ``q=(I-P_B)|chi>`` and ``|T>``
    is the normalized classical target, the score is

    ``|<T|q>|^2 / <q|q>``.

    It is therefore invariant to rescaling either the candidate or the target,
    vanishes for directions already in the retained span, and measures the
    incremental squared projection onto the target rather than energy
    lowering.  All overlaps are assembled from the same virtual-generator bank;
    no target state is prepared on a quantum backend.
    """
    label = bank.generator(candidate).label
    leakage = None
    if leakage_tol is not None:
        leakage = sector_leakage(bank.generator(candidate))
        worst = max(leakage.values())
        if worst > leakage_tol:
            return TargetOverlapScore(
                candidate, label, 0.0, 0.0, 0.0,
                rejected=f"sector leakage {worst:.2e}", leakage=leakage)

    s_aa = float(bank.entry(candidate, candidate)[0].real)
    if s_aa <= DEFAULT_NORM_FLOOR ** 2:
        return TargetOverlapScore(
            candidate, label, 0.0, 0.0, 0.0,
            rejected="annihilates the reference", leakage=leakage)

    s_column = np.array([bank.entry(i, candidate)[0] for i in basis],
                        dtype=complex)
    projections = result.coefficients.conj().T @ s_column
    q_norm = max(0.0, s_aa - float(np.sum(np.abs(projections) ** 2)))
    orthogonal_fraction = q_norm / s_aa
    if orthogonal_fraction < min_orthogonality or q_norm <= _GAP_FLOOR * s_aa:
        return TargetOverlapScore(
            candidate, label, 0.0, orthogonal_fraction, 0.0,
            rejected=f"orthogonal fraction {orthogonal_fraction:.2e}",
            leakage=leakage)

    target_indices = tuple(bank.extend(target.generators))
    coefficients = np.asarray(target.coefficients, dtype=complex)
    target_overlap = np.array([
        [bank.entry(i, j)[0] for j in target_indices] for i in target_indices
    ], dtype=complex)
    target_norm = float((coefficients.conj() @ target_overlap @ coefficients).real)
    if target_norm <= DEFAULT_NORM_FLOOR ** 2:
        raise ValueError("overlap target has zero norm on this reference state")
    coefficients = coefficients / math.sqrt(target_norm)

    target_candidate = sum(
        np.conjugate(coefficient) * bank.entry(index, candidate)[0]
        for index, coefficient in zip(target_indices, coefficients)
    )
    target_basis = np.array([
        [bank.entry(i, j)[0] for j in basis] for i in target_indices
    ], dtype=complex)
    target_ritz = coefficients.conj() @ target_basis @ result.coefficients
    target_q = target_candidate - np.dot(target_ritz, projections)
    gain = max(0.0, float(abs(target_q) ** 2 / q_norm))
    # Roundoff may put a normalized projection a few ulps above one.
    gain = min(1.0, gain)
    return TargetOverlapScore(candidate, label, gain, orthogonal_fraction, gain,
                              leakage=leakage)


def select_target_candidate(scores: Sequence[TargetOverlapScore]
                            ) -> TargetOverlapScore | None:
    """Deterministic counterpart of :func:`select_candidate` for §11A."""
    live = [score for score in scores if score.accepted]
    if not live:
        return None
    best = canonical_argmax(range(len(live)), lambda k: live[k].score,
                            rtol=TIE_RTOL, atol=TIE_ATOL)
    return live[best]


def _acase_objective(result: SubspaceResult, config: ACASEConfig
                     ) -> tuple[float, tuple[float, ...]]:
    tracked = tuple(range(config.roots))
    energies = tuple(result.energies[root] for root in tracked
                     if root < len(result.energies))
    return _aggregate(energies, config.aggregation), energies


def _evaluate_acase_step(bank: MatrixElementBank, state: ACASEState,
                         config: ACASEConfig, step: int,
                         target: OverlapTarget | None = None) -> ACASEStepOutcome:
    """Score/solve one growth decision without owning workflow state."""
    remaining = [index for index in state.pool if index not in state.basis]
    if not remaining:
        return ACASEStepOutcome(stopped_reason="candidate pool exhausted")

    tracked = tuple(range(config.roots))
    basis_words = bank.word_set(state.basis)
    target_gain = None
    if config.criterion == "lowering":
        scores = [
            score_candidate(
                bank, state.basis, state.result, index, roots=tracked,
                aggregation=config.aggregation, basis_words=basis_words,
                min_orthogonality=config.min_orthogonality, gamma=config.gamma,
                leakage_tol=config.leakage_tol,
            )
            for index in remaining
        ]
        best = select_candidate(scores)
        if best is None:
            return ACASEStepOutcome(
                stopped_reason="every candidate rejected (conditioning or sector)")
        if best.predicted_lowering < config.min_lowering:
            return ACASEStepOutcome(stopped_reason="predicted lowering below threshold")
        selection_score = best.score
        rejected_conditioning = sum(
            1 for score in scores
            if score.rejected and score.rejected.startswith("orthogonal"))
        rejected_sector = sum(
            1 for score in scores
            if score.rejected and score.rejected.startswith("sector"))
    else:
        if target is None:
            raise ValueError("target_overlap selection needs an OverlapTarget")
        target_scores = [
            score_target_overlap(
                bank, state.basis, state.result, index, target,
                min_orthogonality=config.min_orthogonality,
                leakage_tol=config.leakage_tol,
            )
            for index in remaining
        ]
        targeted = select_target_candidate(target_scores)
        if targeted is None:
            return ACASEStepOutcome(
                stopped_reason="every candidate rejected (conditioning or sector)")
        if targeted.score < config.min_target_overlap:
            return ACASEStepOutcome(stopped_reason="target overlap below threshold")
        # Energy lowering is still recorded as a diagnostic, but it did not
        # choose this direction.  Keeping the two score objects separate is the
        # §11A boundary that lets both criteria be tested independently.
        best = score_candidate(
            bank, state.basis, state.result, targeted.index, roots=tracked,
            aggregation=config.aggregation, basis_words=basis_words,
            min_orthogonality=config.min_orthogonality, gamma=config.gamma,
            leakage_tol=config.leakage_tol,
        )
        if not best.accepted:
            raise RuntimeError("target and lowering scorers disagree on candidate viability")
        selection_score = targeted.score
        target_gain = targeted.overlap_gain
        rejected_conditioning = sum(
            1 for score in target_scores
            if score.rejected and score.rejected.startswith("orthogonal"))
        rejected_sector = sum(
            1 for score in target_scores
            if score.rejected and score.rejected.startswith("sector"))

    new_basis = state.basis + (best.index,)
    solved = bank.solve(
        new_basis, tau_s=config.tau_s, rel_tau=config.rel_tau,
        max_condition=config.max_condition,
    )
    value, root_energies = _acase_objective(solved, config)
    record = GrowthRecord(
        step=step, selected_label=best.label, energy=value,
        predicted_lowering=best.predicted_lowering,
        actual_lowering=state.energy - value,
        residual_coupling=best.residual_coupling,
        orthogonal_fraction=best.orthogonal_fraction,
        basis_size=len(new_basis), effective_rank=solved.effective_rank,
        condition_number=solved.condition_number,
        candidates_scored=len(remaining),
        rejected_conditioning=rejected_conditioning,
        rejected_sector=rejected_sector,
        new_words=best.new_words,
        word_universe=bank.resources(new_basis)["word_universe"],
        leakage=best.leakage, root_energies=root_energies,
        per_root_lowering=best.per_root,
        selection_criterion=config.criterion, selection_score=selection_score,
        target_overlap_gain=target_gain,
    )

    stopped_reason = None
    if config.exact_ground_energy is not None and config.target_error is not None:
        error = abs(value - config.exact_ground_energy)
        if error <= config.target_error:
            stopped_reason = (
                f"target error reached ({error * 1000.0:.4f} mHa <= "
                f"{config.target_error * 1000.0:.4f} mHa)"
            )
    return ACASEStepOutcome(
        selected_index=best.index, result=solved, energy=value, record=record,
        root_energies=root_energies, stopped_reason=stopped_reason,
    )


def run_acase(rho: MV, hamiltonian, candidates: Sequence, *,
              initial: Sequence | None = None,
              bank: MatrixElementBank | None = None,
              max_size: int = 10,
              roots: int = 1, aggregation: str = "mean",
              min_lowering: float = DEFAULT_MIN_LOWERING,
              min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
              gamma: float = 0.0,
              criterion: str = "lowering",
              min_target_overlap: float = 1e-12,
              target: OverlapTarget | None = None,
              leakage_tol: float | None = None,
              exact_ground_energy: float | None = None,
              target_error: float | None = None,
              tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
              max_condition: float = DEFAULT_MAX_CONDITION,
              config: ACASEConfig | None = None) -> AdaptiveResult:
    """Grow a subspace one generator at a time.

    The legacy keyword surface remains stable. The config object is its
    structured alternative; runtime objects (reference state, generators and
    optional matrix-element bank) remain explicit dependencies.

    Candidate scoring and selection stay separate mathematical functions.
    Numerical work for a growth decision is isolated in the private evaluator,
    while acase_step is the pure state transition. This keeps A-CASE's Ritz
    semantics independent of ADAPT-VQE's optimizer even though both workflows
    expose a similar orchestration boundary.
    """
    legacy_config = ACASEConfig(
        max_size=max_size, roots=roots, aggregation=aggregation,
        min_lowering=min_lowering, min_orthogonality=min_orthogonality,
        gamma=gamma, criterion=criterion, min_target_overlap=min_target_overlap,
        leakage_tol=leakage_tol,
        exact_ground_energy=exact_ground_energy, target_error=target_error,
        tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition,
    )
    if config is None:
        config = legacy_config
    elif legacy_config != ACASEConfig():
        raise ValueError(
            "pass either ACASEConfig or non-default legacy workflow keywords, not both")
    if config.criterion == "target_overlap" and target is None:
        raise ValueError("target_overlap selection needs an OverlapTarget")
    if target is not None and config.criterion != "target_overlap":
        raise ValueError("an OverlapTarget is only used with criterion='target_overlap'")

    started = time.perf_counter()
    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    initial_gens = (as_generators(initial) if initial is not None
                    else [identity_generator(bank.n)])
    basis = tuple(bank.extend(initial_gens))
    pool: list[int] = []
    for index in bank.extend(as_generators(candidates)):
        if index not in basis and index not in pool:
            pool.append(index)
    if not pool:
        raise ValueError("no candidate generators outside the initial basis")

    solved = bank.solve(
        basis, tau_s=config.tau_s, rel_tau=config.rel_tau,
        max_condition=config.max_condition,
    )
    value, root_energies = _acase_objective(solved, config)
    state = ACASEState(
        basis=basis, pool=tuple(pool), result=solved, energy=value,
        energy_history=(value,), records=(), root_energies=root_energies,
    )

    if config.exact_ground_energy is not None and config.target_error is not None:
        error = abs(value - config.exact_ground_energy)
        if error <= config.target_error:
            state = acase_step(
                state,
                ACASEStepOutcome(
                    stopped_reason=(
                        f"target error reached ({error * 1000.0:.4f} mHa <= "
                        f"{config.target_error * 1000.0:.4f} mHa)"
                    )
                ),
            )

    if state.stopped_reason == "basis budget reached":
        for step in range(1, config.max_size + 1):
            outcome = _evaluate_acase_step(bank, state, config, step, target)
            state = acase_step(state, outcome)
            if outcome.stopped_reason is not None:
                break

    relative = None
    if config.exact_ground_energy is not None:
        relative = (
            abs(state.result.ground_energy - config.exact_ground_energy)
            / max(abs(config.exact_ground_energy), 1e-12)
        )
    resources = dict(state.result.resources)
    resources.update({
        "candidate_pool_size": len(state.pool),
        "growth_seconds": time.perf_counter() - started,
        "registered_generators": len(bank),
        "roots": config.roots,
        "aggregation": config.aggregation,
        "selection_criterion": config.criterion,
        "overlap_target": None if target is None else target.label,
    })
    return AdaptiveResult(
        labels=tuple(bank.generator(index).label for index in state.basis),
        energy=state.energy, energy_history=state.energy_history,
        records=state.records, result=state.result, bank=bank,
        indices=state.basis, stopped_reason=state.stopped_reason,
        exact_ground_energy=config.exact_ground_energy,
        relative_error=relative, resources=resources,
        root_energies=state.root_energies,
    )
