"""Sample-split finite-shot certification for adaptive subspace growth."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..measurement.cache import GroupedWordCache
from ..measurement.functionals import coupling_functional
from ..measurement.session import SharedMeasurement
from ..multivector import MV
from ..selection import EvidenceLevel, TIE_ATOL, TIE_RTOL, canonical_argmax
from .generators import as_generators, identity_generator
from .linalg import (
    DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S, SubspaceResult,
)
from .projection import MatrixElementBank
from .symmetry import sector_leakage

ASYMPTOTIC = EvidenceLevel.ASYMPTOTIC.value
FINITE_SAMPLE = EvidenceLevel.FINITE_SAMPLE.value

@dataclass(frozen=True)
class CouplingBound:
    """Bound on ``|<chi_a|(H - E)|Psi_m>|`` from a certification batch."""

    index: int
    label: str
    estimate: float
    lower: float
    upper: float
    evidence: str
    rejected: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected is None


@dataclass(frozen=True)
class CertifiedGrowthRecord:
    """One sample-split growth decision, including a refusal to grow."""

    step: int
    selected_label: str | None
    energy: float
    construction_shots: int
    certification_shots: int
    circuits: int
    coupling: CouplingBound | None
    threshold: float
    candidates_scored: int
    certified: bool
    resolution: str
    evidence: str
    abstained: bool
    reason: str | None
    basis_size: int
    effective_rank: int
    condition_number: float
    overlap_negative_modes: int
    word_universe: int


@dataclass(frozen=True)
class CertifiedResult:
    labels: tuple[str, ...]
    energy: float
    energy_history: tuple[float, ...]
    records: tuple[CertifiedGrowthRecord, ...]
    result: SubspaceResult
    bank: MatrixElementBank
    indices: tuple[int, ...]
    stopped_reason: str
    total_shots: int
    total_circuits: int
    abstentions: int
    exact_ground_energy: float | None = None
    resources: dict[str, Any] = field(default_factory=dict)

    @property
    def basis_size(self) -> int:
        return len(self.labels)


def certify_couplings(bank: MatrixElementBank, indices: Sequence[int],
                      coefficients: np.ndarray, energy: float,
                      candidates: Sequence[int], cache: GroupedWordCache, *,
                      norms: dict[int, float], delta: float = 0.05,
                      bound: str = "eb", threshold: float = 0.0) -> list[CouplingBound]:
    """Simultaneous bounds on every candidate's residual coupling modulus.

    ``|r| = sqrt(Re^2 + Im^2)`` is not linear, so the interval is built as a
    rectangle: each of the two real functionals gets its own two-sided radius,
    and the modulus bounds follow from the corner of the rectangle nearest to
    (and farthest from) the origin. The simultaneous family is therefore
    ``2 * len(candidates)`` events, union-bounded -- the price of a nonlinear
    functional, paid explicitly rather than assumed away.

    ``norms`` supplies ``||A_a|psi>||`` per candidate, and it must come from the
    *construction* batch, not from this one and not from exact arithmetic. The
    coupling has to be normalized or the ranking would depend on how a
    candidate happens to be scaled; but dividing by a quantity estimated from
    the same shots would make the statistic a ratio of correlated estimates and
    void the certificate. Frozen by the independent batch, the norm is a
    constant and the numerator stays linear.

    With ``bound='eb'`` the radii are finite-sample valid at fixed shot counts,
    so the resulting statements are certificates *conditional on* the
    construction batch, which fixed ``(c, E)`` and these norms.
    """
    family = max(1, 2 * len(candidates))
    evidence = FINITE_SAMPLE if bound == "eb" else ASYMPTOTIC
    out: list[CouplingBound] = []
    for candidate in candidates:
        label = bank.generator(candidate).label
        real, imag = coupling_functional(bank, indices, coefficients, energy, candidate)
        norm = norms.get(candidate, 0.0)
        if not norm > DEFAULT_NORM_FLOOR:
            out.append(CouplingBound(candidate, label, 0.0, 0.0, 0.0, evidence,
                                     rejected="norm not resolved above the floor"))
            continue
        parts = []
        for functional in (real, imag):
            value = functional.estimate(cache)
            radius = functional.radius(cache, delta, family, bound=bound)
            parts.append((value, radius))
        estimate = math.hypot(parts[0][0], parts[1][0]) / norm
        lower = math.hypot(*(max(0.0, abs(v) - r) for v, r in parts)) / norm
        upper = math.hypot(*(abs(v) + r for v, r in parts)) / norm
        rejected = None if lower > threshold else f"coupling not certified above {threshold:g}"
        out.append(CouplingBound(candidate, label, estimate, lower, upper, evidence,
                                 rejected=rejected))
    return out


def run_certified_acase(rho: MV, hamiltonian, candidates: Sequence, backend, *,
                        initial: Sequence | None = None,
                        bank: MatrixElementBank | None = None,
                        max_size: int = 6,
                        construction_shots: int = 4000,
                        certification_shots: int = 4000,
                        delta: float = 0.05, threshold: float = 1e-2,
                        bound: str = "eb",
                        leakage_tol: float | None = None,
                        exact_ground_energy: float | None = None,
                        tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
                        max_condition: float = DEFAULT_MAX_CONDITION
                        ) -> CertifiedResult:
    """Finite-shot adaptive growth with a sample-split growth certificate (4C).

    Per step: a construction batch estimates ``(S, H)`` over the current
    subspace's word universe and fixes the thresholded subspace and its Ritz
    pair ``(c, E)``; an independent certification batch estimates every
    candidate's residual coupling with those held constant; simultaneous
    empirical-Bernstein bounds then either certify a candidate above
    ``threshold`` -- and it is added -- or the run **abstains** and stops.

    What is certified is exactly this and no more: on the ``1 - delta`` event,
    conditional on the construction batch, the accepted candidate's residual
    coupling with the frozen Ritz pair exceeds ``threshold``. It is not a
    statement that the candidate is the best available (``resolution`` records
    separately whether the leader also cleared every rival's upper bound), and
    it is emphatically not a bound on the energy: the reported energies come
    from noisy matrices whose variational bound is not established.

    Each step spends a fresh pair of batches. Reusing earlier shots would need
    a confidence-set argument over a data-dependent subspace, which the plan
    leaves to a later stage.

    Two deliberate omissions. Candidates are ranked by residual coupling, not
    by the generalized 2x2 lowering the exact path uses: the lowering is a
    nonlinear function of ``s_aa``, ``h_aa`` and a square root, so it admits
    only an asymptotic treatment and cannot be the certified gate. And there is
    no conditioning rejection here -- the orthogonal fraction is itself a
    nonlinear statistic of the estimates, so screening on it would need its own
    interval rather than a reused exact-path threshold.
    """
    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    initial_gens = (as_generators(initial) if initial is not None
                    else [identity_generator(bank.n)])
    basis = bank.extend(initial_gens)
    pool: list[int] = []
    for index in bank.extend(as_generators(candidates)):
        if index not in basis and index not in pool:
            pool.append(index)
    if not pool:
        raise ValueError("no candidate generators outside the initial basis")

    solver_kwargs = dict(tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition)
    records: list[CertifiedGrowthRecord] = []
    history: list[float] = []
    total_shots = total_circuits = abstentions = 0
    stopped_reason = "basis budget reached"
    result = None
    solved_basis: tuple[int, ...] | None = None

    for step in range(1, max_size + 1):
        remaining = [i for i in pool if i not in basis]
        if leakage_tol is not None:
            remaining = [i for i in remaining
                         if max(sector_leakage(bank.generator(i)).values()) <= leakage_tol]
        if not remaining:
            stopped_reason = "candidate pool exhausted"
            break

        # One grouping over the whole universe -- basis block and candidate rows
        # alike -- so both batches read the same circuits and the sub-block is
        # estimated from the same shots as everything else.
        full = SharedMeasurement(bank, tuple(basis) + tuple(remaining))
        basis_view = SharedMeasurement(bank, basis, groups=full.groups)

        # --- construction batch: the subspace, the Ritz pair, the candidate norms
        construction_cache = full.measure(backend, construction_shots)
        result = basis_view.solve(construction_cache, **solver_kwargs)
        solved_basis = tuple(basis)
        history.append(result.ground_energy)
        total_shots += construction_cache.total_shots
        total_circuits += construction_cache.total_circuits
        coefficients = result.ritz_vector(0)
        energy = result.ground_energy
        norms = {i: math.sqrt(max(0.0, full.diagonal_functional(i)
                                  .estimate(construction_cache)))
                 for i in remaining}

        # --- certification batch: independent shots, everything above constant
        certification_cache = full.measure(backend, certification_shots)
        total_shots += certification_cache.total_shots
        total_circuits += certification_cache.total_circuits

        bounds = certify_couplings(bank, basis, coefficients, energy, remaining,
                                   certification_cache, norms=norms, delta=delta,
                                   bound=bound, threshold=threshold)
        live = [b for b in bounds if b.accepted]
        best = None
        if live:
            pick = canonical_argmax(range(len(live)), lambda k: live[k].estimate,
                                    rtol=TIE_RTOL, atol=TIE_ATOL)
            best = live[pick]

        resolution = "none"
        if best is not None:
            rivals = max((b.upper for b in bounds if b.index != best.index), default=0.0)
            resolution = "best" if best.lower > rivals else "above_threshold"

        record = CertifiedGrowthRecord(
            step=step, selected_label=best.label if best else None,
            energy=energy, construction_shots=construction_cache.total_shots,
            certification_shots=certification_cache.total_shots,
            circuits=(construction_cache.total_circuits
                      + certification_cache.total_circuits),
            coupling=best, threshold=threshold, candidates_scored=len(bounds),
            certified=bool(best is not None and bound == "eb"),
            resolution=resolution,
            evidence=FINITE_SAMPLE if bound == "eb" else ASYMPTOTIC,
            abstained=best is None,
            reason=None if best else "no candidate certified above threshold",
            basis_size=len(basis), effective_rank=result.effective_rank,
            condition_number=result.condition_number,
            overlap_negative_modes=result.resources.get("overlap_negative_modes", 0),
            word_universe=result.resources.get("word_universe", 0))
        records.append(record)

        if best is None:
            abstentions += 1
            stopped_reason = "abstained: no candidate certified above threshold"
            break
        basis.append(best.index)
    if result is None or solved_basis != tuple(basis):
        # Either nothing was ever solved (a filter emptied the pool at step 1),
        # or the budget ran out with the last accepted generator never solved
        # for. Either way one more construction batch, so the reported energy
        # describes the basis that was actually grown.
        final = SharedMeasurement(bank, basis)
        cache = final.measure(backend, construction_shots)
        result = final.solve(cache, **solver_kwargs)
        solved_basis = tuple(basis)
        history.append(result.ground_energy)
        total_shots += cache.total_shots
        total_circuits += cache.total_circuits

    resources = dict(result.resources)
    resources.update({
        "candidate_pool_size": len(pool),
        "delta": delta,
        "bound": bound,
        "certification_scope": "per_step_conditional_on_construction_batch",
        "shot_accounting": "construction_plus_certification_batches",
        "variational_bound": "not_established_under_noise",
    })
    return CertifiedResult(
        labels=tuple(bank.generator(i).label for i in basis),
        energy=result.ground_energy, energy_history=tuple(history),
        records=tuple(records), result=result, bank=bank, indices=tuple(basis),
        stopped_reason=stopped_reason, total_shots=total_shots,
        total_circuits=total_circuits, abstentions=abstentions,
        exact_ground_energy=exact_ground_energy, resources=resources)

__all__ = [
    "CouplingBound", "CertifiedGrowthRecord", "CertifiedResult",
    "certify_couplings", "run_certified_acase",
]
