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
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from ..algorithms.adapt import TIE_ATOL, TIE_RTOL, canonical_argmax
from ..multivector import MV
from .elements import MatrixElementBank
from .generators import Generator, as_generators, identity_generator
from .solver import (DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S,
                     SubspaceResult, _as_mv)

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


@lru_cache(maxsize=32)
def _sector_operators(n: int) -> tuple[MV, MV]:
    from ..fermion import total_number_op, total_sz_op
    return total_number_op(n), total_sz_op(n)


def sector_leakage(generator) -> dict[str, float]:
    """Relative operator-level leakage out of the ``(N, S_z)`` sectors.

    ``||[A, N]||_HS / ||A||_HS`` and the same for ``S_z``: zero exactly when
    the generator commutes with the symmetry, and scale-free, so it can be
    compared across generators of very different norms. Operator-level rather
    than state-level on purpose -- a generator that commutes with ``N`` cannot
    take *any* reference out of its particle-number sector, which is a stronger
    statement than one state's sector weight.
    """
    A = generator.mv if isinstance(generator, Generator) else _as_mv(generator)
    norm = A.norm_hs()
    if norm <= 0.0:
        return {"particle_number": 0.0, "sz": 0.0}
    N, Sz = _sector_operators(A.n)
    return {"particle_number": (A * N - N * A).norm_hs() / norm,
            "sz": (A * Sz - Sz * A).norm_hs() / norm}


@dataclass(frozen=True)
class CandidateScore:
    """One candidate weighed against the current Ritz pair."""

    index: int
    label: str
    residual_coupling: float
    predicted_lowering: float
    orthogonal_fraction: float
    new_words: int
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

    @property
    def basis_size(self) -> int:
        return len(self.labels)


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


def score_candidate(bank: MatrixElementBank, basis: Sequence[int],
                    result: SubspaceResult, candidate: int, *, root: int = 0,
                    basis_words: frozenset[int] | None = None,
                    min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
                    gamma: float = 0.0, leakage_tol: float | None = None
                    ) -> CandidateScore:
    """Weigh one candidate against the Ritz pair ``root`` of ``result``.

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

    energy = result.energies[root]
    column = np.array([bank.entry(i, candidate) for i in basis], dtype=complex)
    s_column, h_column = column[:, 0], column[:, 1]
    # The Ritz vectors are S-orthonormal, so they are an orthonormal basis of
    # the retained subspace and the projection is a plain sum of squares.
    projections = result.coefficients.conj().T @ s_column
    s_a = complex(projections[root])
    h_a = complex(result.coefficients[:, root].conj() @ h_column)

    orthogonal_fraction = max(0.0, 1.0 - float(np.sum(np.abs(projections) ** 2)) / s_aa)
    # Scale-free residual coupling: the overlap of the candidate's orthogonal
    # component with the residual (H - E)|Psi_m>.
    coupling = abs(h_a - energy * s_a) / math.sqrt(s_aa)

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

    lowering = _two_by_two_lowering(energy, s_a, h_a, s_aa, h_aa)
    score = lowering if gamma == 0.0 else lowering / (1.0 + new_words) ** gamma
    return CandidateScore(candidate, label, coupling, lowering, orthogonal_fraction,
                          new_words, score, leakage=leakage)


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


def run_acase(rho: MV, hamiltonian, candidates: Sequence, *,
              initial: Sequence | None = None,
              bank: MatrixElementBank | None = None,
              max_size: int = 10,
              min_lowering: float = DEFAULT_MIN_LOWERING,
              min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
              gamma: float = 0.0,
              leakage_tol: float | None = None,
              exact_ground_energy: float | None = None,
              tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
              max_condition: float = DEFAULT_MAX_CONDITION) -> AdaptiveResult:
    """Grow a subspace one certified-by-construction generator at a time.

    ``initial`` defaults to the identity generator alone, so the run starts
    from the reference state itself; passing an ADAPT-VQE state as ``rho``
    (see :func:`adapt_warm_start`) or a richer ``initial`` set warm-starts it.

    ``leakage_tol`` turns on the §4.2 sector discipline: candidates whose
    relative commutator norm with ``N`` or ``S_z`` exceeds it are rejected, and
    every accepted generator's leakage is recorded either way.

    ``gamma`` prices measurement cost into the acceptance score,
    ``lowering / (1 + new_words)^gamma``. The default of 0 selects on predicted
    lowering alone, which is the baseline the cost-aware variants are measured
    against.
    """
    started = time.perf_counter()
    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    initial_gens = (as_generators(initial) if initial is not None
                    else [identity_generator(bank.n)])
    basis = bank.extend(initial_gens)
    pool: list[int] = []
    for index in bank.extend(as_generators(candidates)):
        # extend() collapses duplicates onto one id, so the pool can repeat
        if index not in basis and index not in pool:
            pool.append(index)
    if not pool:
        raise ValueError("no candidate generators outside the initial basis")

    solver_kwargs = dict(tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition)
    result = bank.solve(basis, **solver_kwargs)
    history = [result.ground_energy]
    records: list[GrowthRecord] = []
    stopped_reason = "basis budget reached"

    for step in range(1, max_size + 1):
        remaining = [i for i in pool if i not in basis]
        if not remaining:
            stopped_reason = "candidate pool exhausted"
            break
        basis_words = bank.word_set(basis)
        scores = [score_candidate(bank, basis, result, i, basis_words=basis_words,
                                  min_orthogonality=min_orthogonality, gamma=gamma,
                                  leakage_tol=leakage_tol)
                  for i in remaining]
        best = select_candidate(scores)
        rejected_conditioning = sum(
            1 for s in scores if s.rejected and s.rejected.startswith("orthogonal"))
        rejected_sector = sum(
            1 for s in scores if s.rejected and s.rejected.startswith("sector"))
        if best is None:
            stopped_reason = ("every candidate rejected (conditioning or sector)")
            break
        if best.predicted_lowering < min_lowering:
            stopped_reason = "predicted lowering below threshold"
            break

        previous = result.ground_energy
        basis.append(best.index)
        result = bank.solve(basis, **solver_kwargs)
        history.append(result.ground_energy)
        records.append(GrowthRecord(
            step=step, selected_label=best.label, energy=result.ground_energy,
            predicted_lowering=best.predicted_lowering,
            actual_lowering=previous - result.ground_energy,
            residual_coupling=best.residual_coupling,
            orthogonal_fraction=best.orthogonal_fraction,
            basis_size=len(basis), effective_rank=result.effective_rank,
            condition_number=result.condition_number,
            candidates_scored=len(scores),
            rejected_conditioning=rejected_conditioning,
            rejected_sector=rejected_sector,
            new_words=best.new_words,
            word_universe=bank.resources(basis)["word_universe"],
            leakage=best.leakage))

    relative = None
    if exact_ground_energy is not None:
        relative = (abs(result.ground_energy - exact_ground_energy)
                    / max(abs(exact_ground_energy), 1e-12))
    resources = dict(result.resources)
    resources.update({
        "candidate_pool_size": len(pool),
        "growth_seconds": time.perf_counter() - started,
        "registered_generators": len(bank),
    })
    return AdaptiveResult(
        labels=tuple(bank.generator(i).label for i in basis),
        energy=result.ground_energy, energy_history=tuple(history),
        records=tuple(records), result=result, bank=bank, indices=tuple(basis),
        stopped_reason=stopped_reason, exact_ground_energy=exact_ground_energy,
        relative_error=relative, resources=resources)


def adapt_warm_start(model, pool, *, max_operators: int = 4, **kwargs):
    """Run exact ADAPT-VQE and return ``(rho, AdaptResult)`` for use as a reference.

    The Paper A machinery unchanged; A-CASE then treats the optimized ADAPT
    state as its single reference, so the subspace is grown around a state that
    already carries some correlation rather than around a bare determinant.
    """
    from ..algorithms.adapt import _ansatz_program, run_adapt
    from ..backends.exact_mv import ExactMVBackend

    result = run_adapt(model, pool, max_operators=max_operators, **kwargs)
    by_label = {op.label: op for op in pool}
    # Rebuild in the order ADAPT selected them; the ansatz is ordered, so
    # filtering the pool by membership would silently reorder the rotors.
    chosen = [by_label[label] for label in result.labels]
    program = _ansatz_program(model, chosen)
    return ExactMVBackend().state(program, result.parameters), result
