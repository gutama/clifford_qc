"""Predeclared, target-aware shot acquisition for a measured Ritz value.

Uniform shots spend the same effort on every measurement setting. Most of them
barely move the quantity that is wanted. :func:`acquire_ritz` instead measures a
uniform pilot, solves, and spends each further round where the current Ritz
value is most sensitive: the round's budget is split across settings by Neyman
allocation on the Ritz functional ``c^*(H - E S)c``, the rule of
:func:`~clifford_qc.measurement.allocation.variance_optimal_group_plan`.
:func:`neyman_group_plan` applies it with rounding that does not depend on the
last bits of a float, because here the shot counts feed back into what is
sampled next.

Stopping semantics are the part that needs care. A rule of the form "stop when
the estimated error is small enough" looks at the data to decide when to stop,
and an interval reported at such a time is not the interval of a fixed-size
experiment. Here:

- the pilot size, the round budgets and the target are fixed before any shot;
- the rule is evaluated only at those predeclared looks, never between them;
- the reported interval splits its error budget over every look at which the
  run could have stopped (Bonferroni), so it is valid over the whole schedule
  to the extent the Gaussian delta method is valid at each look.

That makes the interval *schedule-valid and asymptotic*: not anytime-valid, and
not a finite-sample certificate. The naive single-look interval is returned
beside it so the two can be compared. All data are pooled, pilot included, and
the allocation depends on earlier outcomes, so the estimate carries the bias
any adaptive design has.

Control variates are opt-in (``control_variates``). A declared family of
sector terms ``K_k`` that annihilate every bank state (see
:mod:`~clifford_qc.subspace.symmetry_shifts`) gives zero-mean corrections
``g_k = B† K_k B`` read from the same shots. Each look then reports the
corrected Ritz value, stops on its standard error and, under targeting, spends
the next round by Neyman allocation on the corrected functional. The family is
certified before any shot; the correction coefficients are refitted at every
look from the data so far, which adds an ``O(1/N)`` bias and a slightly
optimistic standard error, so the interval stays asymptotic.

A bootstrap interval at the stop is opt-in (``bootstrap_replicates``). The
stopping rule is unchanged; once it has stopped, the data at that look are
resampled by setting and re-solved, and ``bootstrap_interval`` reports the
estimate plus or minus the same ``z`` times the bootstrap spread
``(q84 - q16) / 2``. The delta method linearizes the Ritz value at a fixed
retained eigenspace, so it cannot see a stop at which the solve's truncation is
unstable; the bootstrap re-solves and can. On the four banks of the declared
target-acquisition study, on fresh seeds and under criteria fixed before the
run, it covered in at least 99% of runs in every cell, where the delta interval
covered 84.5% to 93.5% on ``tfim6_mixed`` (``docs/restart/BOOTSTRAP_STOP.md``).
It is heuristic and often wide: a spread many times ``std_error`` marks a stop
whose truncation the data do not settle.
With control variates, each replica also refits the certified family at its
own solved Ritz vector, so the bootstrap describes the corrected estimator.
The previous coverage study did not include this combination.

For new fixed-bank precision work use :func:`acquire_ritz_precision`, which
requires rank confirmation by default and adds a bootstrap-width guard only
with an explicit width target and resampling budget. :func:`acquire_ritz`
retains its historical defaults for declared comparisons and replay.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from ..measurement.cache import GroupedWordCache
from ..measurement.confidence import simultaneous_z_radius
from ..measurement.functionals import WordFunctional, ritz_functional
from ..measurement.grouping import shared_basis
from ..measurement.session import SharedMeasurement
from .coverage import delta_method_applies
from .linalg import SubspaceResult
from .rank_stopping import RankStopPolicy, rank_margin
from .symmetry_shifts import (ControlVariateEstimate, ShiftTerm, certify_bank_family,
                              corrected_ritz_functional)
from .uncertainty import (ASYMPTOTIC, Interval, bootstrap_ritz_samples,
                          bootstrap_spread_interval, ritz_uncertainty)

ALLOCATIONS = ("uniform", "ritz_targeted")
_SOLVE_FAILURES = (ValueError, np.linalg.LinAlgError)
# Fractions of a shot closer than this are a tie. It is far above the rounding
# of any weight and far below one shot, so it changes no allocation that the
# data actually decide.
TIE_TOLERANCE = 1e-6


class AcquisitionError(ValueError):
    """A look could not be solved, so the schedule ended without an estimate.

    The shots are spent whether or not an estimate came out of them. ``cache``
    holds every outcome measured so far and ``looks`` the ledger up to the look
    that failed, so a caller can count the failure against its real cost.
    """

    def __init__(self, message: str, *, cache: GroupedWordCache,
                 looks: tuple[dict[str, Any], ...]):
        super().__init__(message)
        self.cache = cache
        self.looks = looks

    @property
    def shots(self) -> int:
        return int(self.cache.total_shots)


@dataclass(frozen=True)
class RitzAcquisition:
    """The outcome of one predeclared acquisition schedule.

    ``interval`` is the schedule-valid interval; ``naive_interval`` ignores that
    the run could have stopped at other looks. They coincide when no stopping
    rule was declared. ``looks`` is the ledger: one entry per look with the
    shots spent so far, the energy, its estimated standard error, the retained
    rank, the smallest retained overlap eigenvalue of the solved metric and
    whether the run ended there. ``stopped`` is ``"target_met"``,
    ``"schedule_exhausted"`` or ``"deterministic"``. ``schedule`` records what
    was declared, with ``interval_family`` the number of looks ``interval``
    pays for.

    With control variates, ``energy`` and ``std_error`` in each look are the
    corrected values the stopping rule saw, ``raw_energy`` and
    ``raw_std_error`` the uncorrected Ritz value, ``control_variates_used`` how
    many terms were readable, and both intervals are centred on the corrected
    value. ``control_variate`` holds the correction at the last look;
    ``result`` is always the uncorrected solve.

    With a bootstrap, ``bootstrap_interval`` is the schedule-valid interval
    with the bootstrap spread in place of the delta-method standard error, and
    the last look records ``bootstrap_spread``. If a resampled solve fails, the
    interval is ``None`` and the last look records ``bootstrap_failure``
    instead; the acquisition itself stands.
    """

    result: SubspaceResult
    cache: GroupedWordCache
    interval: Interval
    naive_interval: Interval
    looks: tuple[dict[str, Any], ...]
    stopped: str
    allocation: str
    schedule: dict[str, Any]
    control_variate: ControlVariateEstimate | None = None
    bootstrap_interval: Interval | None = None

    @property
    def shots(self) -> int:
        """Every physical shot spent, pilot included."""
        return int(self.cache.total_shots)

    @property
    def std_error(self) -> float:
        return float(self.looks[-1]["std_error"])

    @property
    def energy(self) -> float:
        """The reported estimate: corrected when control variates were used."""
        return float(self.looks[-1]["energy"])


def uniform_group_plan(groups: Sequence[Sequence], budget: int) -> dict[int, int]:
    """Split a physical-shot budget equally over measurement groups.

    The remainder goes one shot each to the first groups in order, so the plan
    spends exactly ``budget``. Word-keyed, like every allocation plan here.
    """
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Integral) or budget < 1:
        raise ValueError("budget must be a positive integer")
    groups = [tuple(group) for group in groups if group]
    if not groups:
        raise ValueError("no measurement groups to allocate to")
    base, extra = divmod(int(budget), len(groups))
    plan: dict[int, int] = {}
    for index, group in enumerate(groups):
        shots = base + (1 if index < extra else 0)
        if shots:
            plan.update({word.code: shots for word in group})
    return plan


def neyman_group_plan(groups: Sequence[Sequence], cache: GroupedWordCache,
                      functional: WordFunctional, budget: int) -> dict[int, int]:
    """Split a physical-shot budget by Neyman allocation, reproducibly.

    Setting ``g`` receives shots in proportion to the square root of the
    per-shot variance of ``functional`` within it, estimated from ``cache``;
    settings the functional does not read receive none. This is
    :func:`~clifford_qc.measurement.allocation.variance_optimal_group_plan` for
    one target at equal circuit costs, and agrees with it whenever no two
    settings tie.

    The difference is how ties are rounded. Settings related by a symmetry of
    the bank have weights that are equal in exact arithmetic and differ in
    floating point by whatever the eigensolver's last bits were. A
    largest-remainder rule that orders by the raw floats then hands the spare
    shot to a different setting on a different CPU, and an adaptive schedule
    turns that one shot into a different sample stream. Remainders within
    ``TIE_TOLERANCE`` of each other are therefore treated as equal and broken by
    group order. An empty plan means the functional is deterministic.
    """
    budget = _positive_int(budget, "budget")
    if functional.is_deterministic:
        return {}
    statistics = cache.candidate_group_statistics(dict(functional.coefficients))
    if statistics is None:
        raise ValueError("a setting the functional reads has no shots; measure a pilot first")
    touched = []
    for group in groups:
        words = tuple(group)
        if not words:
            continue
        key = tuple(sorted(shared_basis(words).items()))
        if key in statistics:
            touched.append((words, math.sqrt(max(0.0, float(statistics[key][1])))))
    if not touched:
        return {}
    total = sum(weight for _, weight in touched)
    weights = ([weight / total for _, weight in touched] if total > 0.0
               else [1.0 / len(touched)] * len(touched))
    raw = [budget * weight for weight in weights]
    shots = [int(math.floor(value + TIE_TOLERANCE)) for value in raw]
    remainders = [round((value - whole) / TIE_TOLERANCE) for value, whole in zip(raw, shots)]
    order = sorted(range(len(raw)), key=lambda index: (-remainders[index], index))
    for index in order[:budget - sum(shots)]:
        shots[index] += 1
    plan: dict[int, int] = {}
    for (words, _), count in zip(touched, shots):
        if count:
            plan.update({word.code: count for word in words})
    return plan


def _positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def acquire_ritz(shared: SharedMeasurement, backend, *, pilot_shots_per_group: int,
                 round_budgets: Sequence[int], allocation: str = "ritz_targeted",
                 target_std_error: float | None = None,
                 solver_kwargs: dict | None = None, root: int = 0,
                 delta: float = 0.05,
                 control_variates: Sequence[ShiftTerm] | None = None,
                 bootstrap_replicates: int | None = None,
                 bootstrap_seed: int = 0,
                 rank_stop: RankStopPolicy | None = None,
                 max_bootstrap_half_width: float | None = None) -> RitzAcquisition:
    """Measure a pilot, then spend predeclared rounds toward the Ritz value.

    ``pilot_shots_per_group`` shots go uniformly to every setting. Each entry of
    ``round_budgets`` is then one round's physical-shot budget, allocated
    ``"uniform"`` or ``"ritz_targeted"``. Before each round, and after the last,
    the run solves and estimates the Ritz value's standard error from all data
    so far: a *look*. With ``target_std_error`` the run stops at the first look
    whose estimate is at or below it; without it every round is spent.

    The targeted allocation is :func:`neyman_group_plan` on the current Ritz
    functional. It does not discount shots a setting already has, so it is the
    Neyman rule per round, not for the cumulative total, and it buys nothing
    for a direction the current solve has truncated away.

    ``control_variates`` is a family of sector terms (for instance
    :func:`~clifford_qc.subspace.symmetry_shifts.sector_shift_family`). It is
    certified on the bank before any shot and refused if a term fails to
    annihilate every bank state. At each look the correction coefficients are
    fitted to the data so far; the corrected value is what the look reports,
    what the stopping rule tests and, under targeting, what the next round is
    allocated to. Terms with a word no setting reads are left out and counted.

    ``bootstrap_replicates`` adds :attr:`RitzAcquisition.bootstrap_interval`:
    after the run ends, the last look's data are resampled that many times with
    ``bootstrap_seed`` (:func:`~clifford_qc.subspace.uncertainty.bootstrap_ritz_samples`,
    bound to the point result's solver policy) and reported by
    :func:`~clifford_qc.subspace.uncertainty.bootstrap_spread_interval` with the
    same ``delta`` and look count as ``interval``. It costs that many extra
    solves and no shot, uses its own random stream, and changes nothing the run
    measured or reported otherwise. The fresh-seed check used 500. With
    control variates, each replica refits the family using its own resampled
    data and Ritz vector, and the reported interval is centred on the
    corrected point estimate. Coverage of the combined policy is unvalidated.

    Optional ``rank_stop`` requires full declared bank rank and positive
    statistical cutoff margins at consecutive looks. An unresolved rank
    redirects part of targeted rounds to the weakest overlap margin.
    ``max_bootstrap_half_width`` adds a schedule-adjusted bootstrap half-width
    guard at eligible looks. Both need a declared SE target; the latter also
    requires bootstrap replicates. Exhaustion returns diagnostics, not a
    successful precision stop. Both guards remain heuristic.

    Only estimators the delta method accepts are supported, because both the
    allocation target and the interval are its Ritz functional. A look that
    cannot be solved raises :class:`AcquisitionError`, which carries the cache
    and the ledger so the shots already spent are not hidden.
    """
    pilot = _positive_int(pilot_shots_per_group, "pilot_shots_per_group")
    budgets = tuple(_positive_int(budget, "round budget") for budget in round_budgets)
    if not budgets:
        raise ValueError("round_budgets must name at least one round")
    if allocation not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {ALLOCATIONS}")
    if target_std_error is not None:
        if isinstance(target_std_error, (bool, np.bool_)) \
                or not isinstance(target_std_error, Real) \
                or not math.isfinite(float(target_std_error)) or target_std_error <= 0:
            raise ValueError("target_std_error must be a positive finite real scalar")
        target_std_error = float(target_std_error)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if isinstance(root, (bool, np.bool_)) or not isinstance(root, Integral) or root < 0:
        raise ValueError("root must be a nonnegative integer")
    kwargs = dict(solver_kwargs or {})
    if not delta_method_applies(kwargs):
        raise ValueError("target-aware acquisition needs an estimator the delta method "
                         "accepts: no overlap repair, rotation or constrained ratio")
    if bootstrap_replicates is not None:
        bootstrap_replicates = _positive_int(bootstrap_replicates, "bootstrap_replicates")
        if bootstrap_replicates < 2:
            raise ValueError("bootstrap_replicates must be at least two")
    if isinstance(bootstrap_seed, (bool, np.bool_)) or not isinstance(bootstrap_seed, Integral) \
            or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if rank_stop is not None:
        if not isinstance(rank_stop, RankStopPolicy):
            raise ValueError('rank_stop must be a RankStopPolicy')
        if target_std_error is None or rank_stop.consecutive_looks > len(budgets)+1:
            raise ValueError('rank_stop needs a target and enough predeclared looks')
        if kwargs.get('calibrate_overlap', False):
            raise ValueError('rank_stop requires fixed overlap cutoffs')
    if max_bootstrap_half_width is not None:
        if (isinstance(max_bootstrap_half_width, (bool, np.bool_)) or
                not isinstance(max_bootstrap_half_width, Real) or
                not math.isfinite(float(max_bootstrap_half_width)) or max_bootstrap_half_width <= 0):
            raise ValueError('max_bootstrap_half_width must be positive and finite')
        if bootstrap_replicates is None or target_std_error is None:
            raise ValueError('bootstrap width guard needs a target and bootstrap_replicates')
        max_bootstrap_half_width = float(max_bootstrap_half_width)
    terms: tuple[ShiftTerm, ...] | None = None
    if control_variates is not None:
        terms = tuple(control_variates)
        certify_bank_family(shared.bank, terms)
    # Every look at which the run could report: after the pilot and each round.
    family = len(budgets) + 1 if target_std_error is not None else 1

    cache = shared.measure(backend, pilot)
    looks: list[dict[str, Any]] = []
    stopped = "schedule_exhausted"
    point = None
    correction: ControlVariateEstimate | None = None
    consecutive = 0
    bootstrap = None

    def bootstrap_at_look(look, point):
        # Historical terminal-only runs keep their original seed. Width
        # assessments get a deterministic independent seed per look.
        seed = int(bootstrap_seed)
        if max_bootstrap_half_width is not None:
            seed = int(np.random.SeedSequence([seed, look['look']]).generate_state(1, np.uint64)[0])
            look['bootstrap_seed'] = seed
            look['bootstrap_attempted_replicates'] = bootstrap_replicates
        try:
            draws = bootstrap_ritz_samples(shared, cache, root=root, replicates=bootstrap_replicates,
                                           seed=seed, result=point, control_variates=terms)
            result = bootstrap_spread_interval(look['energy'], draws, delta=delta, family=family)
        except _SOLVE_FAILURES as error:
            look['bootstrap_failure'] = f"{type(error).__name__}: {error}"
            return None
        look['bootstrap_spread'] = result.std_error
        return result

    for index in range(len(budgets) + 1):
        try:
            point = shared.solve(cache, **kwargs)
        except _SOLVE_FAILURES as error:
            raise AcquisitionError(
                f"look {index} could not be solved after {cache.total_shots} shots: "
                f"{type(error).__name__}: {error}",
                cache=cache, looks=tuple(looks)) from error
        if root >= len(point.energies):
            raise AcquisitionError(
                f"look {index} retained {len(point.energies)} root(s) after "
                f"{cache.total_shots} shots, fewer than root {root} needs",
                cache=cache, looks=tuple(looks))
        if terms is None:
            functional = ritz_functional(shared.bank, point.indices,
                                         point.ritz_vector(root), point.energies[root])
            sigma = math.sqrt(max(0.0, functional.variance(cache)))
            energy = float(point.energies[root])
        else:
            correction, functional = corrected_ritz_functional(shared, cache, point, terms,
                                                               root=root)
            sigma, energy = correction.std_error, correction.energy
        retained = point.resources.get("smallest_retained_overlap_eigenvalue")
        look = {"look": index, "shots": int(cache.total_shots),
                "energy": energy, "std_error": sigma,
                "retained_rank": int(point.effective_rank),
                "smallest_retained_overlap": None if retained is None else float(retained),
                "stopped": False}
        if correction is not None:
            look.update(raw_energy=correction.raw_energy,
                        raw_std_error=correction.raw_std_error,
                        control_variates_used=len(correction.used))
        rank_ok = True
        rank_functional = None
        if rank_stop is not None:
            try:
                diagnostic, rank_functional = rank_margin(shared, cache, point, rank_stop, family)
            except _SOLVE_FAILURES as error:
                raise AcquisitionError(f'look {index} rank margin failed: {error}', cache=cache,
                                       looks=tuple(looks+[look])) from error
            consecutive = consecutive+1 if diagnostic['resolved'] else 0
            rank_ok = consecutive >= rank_stop.consecutive_looks
            look['rank_guard'] = {**diagnostic, 'consecutive_resolved': consecutive, 'passed': rank_ok}
        se_ok = target_std_error is not None and sigma <= target_std_error
        bootstrap = None
        width_ok = max_bootstrap_half_width is None
        if max_bootstrap_half_width is not None and se_ok and rank_ok:
            bootstrap = bootstrap_at_look(look, point)
            width_ok = bootstrap is not None and (bootstrap.upper-bootstrap.lower)/2 <= max_bootstrap_half_width
            look['bootstrap_width_passed'] = width_ok
        looks.append(look)
        if se_ok and rank_ok and width_ok:
            stopped = "target_met"
            break
        if index == len(budgets):
            break
        if allocation == "uniform":
            plan = uniform_group_plan(shared.groups, budgets[index])
        elif rank_stop is not None and not look['rank_guard']['resolved']:
            # Half of each unresolved round targets overlap, the other half
            # energy; all overlap if the energy SE already qualifies.
            budget = budgets[index]
            rank_budget = budget if se_ok else (budget+1)//2
            energy_budget = budget-rank_budget
            rank_plan = neyman_group_plan(shared.groups, cache, rank_functional, rank_budget) \
                if rank_functional is not None else {}
            if not rank_plan:
                rank_plan = uniform_group_plan(shared.groups, rank_budget)
            plan = dict(rank_plan)
            if energy_budget:
                energy_plan = neyman_group_plan(shared.groups, cache, functional, energy_budget)
                if not energy_plan:
                    energy_plan = uniform_group_plan(shared.groups, energy_budget)
                for code, shots in energy_plan.items():
                    plan[code] = plan.get(code, 0)+shots
            look['next_rank_budget'] = rank_budget
        else:
            plan = neyman_group_plan(shared.groups, cache, functional, budgets[index])
        if not plan and (rank_stop is not None or max_bootstrap_half_width is not None):
            plan = uniform_group_plan(shared.groups, budgets[index])
        if not plan:
            # A deterministic Ritz functional: no shot can change the estimate.
            stopped = "deterministic"
            break
        shared.measure_plan(backend, plan, cache)
    assert point is not None
    looks[-1]["stopped"] = True
    schedule = {"pilot_shots_per_group": pilot, "round_budgets": list(budgets),
                "target_std_error": target_std_error, "delta": float(delta),
                "interval_family": family, "solver_kwargs": kwargs}
    if rank_stop is not None:
        schedule['rank_stop'] = asdict(rank_stop)
    if max_bootstrap_half_width is not None:
        schedule['max_bootstrap_half_width'] = max_bootstrap_half_width
    if terms is None:
        interval = ritz_uncertainty(shared, cache, point, root=root, delta=delta,
                                    family=family)
        naive = ritz_uncertainty(shared, cache, point, root=root, delta=delta)
    else:
        assert correction is not None
        schedule["control_variates"] = [term.label for term in terms]
        interval = _corrected_interval(correction, delta, family)
        naive = _corrected_interval(correction, delta, 1)
    if bootstrap_replicates is not None:
        schedule["bootstrap"] = {"replicates": bootstrap_replicates, "seed": int(bootstrap_seed)}
        if terms is not None:
            schedule["bootstrap"]["estimator"] = "refitted_control_variates"
        if 'bootstrap_seed' not in looks[-1]:
            bootstrap = bootstrap_at_look(looks[-1], point)
    return RitzAcquisition(
        result=point, cache=cache, interval=interval, naive_interval=naive,
        looks=tuple(looks), stopped=stopped, allocation=allocation, schedule=schedule,
        control_variate=correction, bootstrap_interval=bootstrap)


def _corrected_interval(correction: ControlVariateEstimate, delta: float,
                        family: int) -> Interval:
    """The delta-method interval of :func:`ritz_uncertainty`, on the corrected value."""
    radius = simultaneous_z_radius(correction.std_error ** 2, delta, family)
    return Interval(estimate=correction.energy, lower=correction.energy - radius,
                    upper=correction.energy + radius, delta=delta, evidence=ASYMPTOTIC,
                    std_error=correction.std_error)


def acquire_ritz_precision(shared: SharedMeasurement, backend, *,
                           pilot_shots_per_group: int,
                           round_budgets: Sequence[int],
                           target_std_error: float,
                           allocation: str = "ritz_targeted",
                           solver_kwargs: dict | None = None, root: int = 0,
                           delta: float = 0.05,
                           control_variates: Sequence[ShiftTerm] | None = None,
                           rank_stop: RankStopPolicy = RankStopPolicy(),
                           max_bootstrap_half_width: float | None = None,
                           bootstrap_replicates: int | None = None,
                           bootstrap_seed: int = 0) -> RitzAcquisition:
    """Preferred precision controller for a declared independent Ritz bank.

    By default, require the SE target *and* full retained rank with positive
    statistical cutoff margins at two consecutive looks. Ritz-targeted rounds
    redirect shots toward an unresolved overlap margin. A stable truncated
    bank is not a successful stop. If rank cannot be resolved within the
    declared budget, return a diagnostic estimate with ``schedule_exhausted``.

    For tighter uncertainty control, explicitly set ``max_bootstrap_half_width``
    (in the same units as the energy) and ``bootstrap_replicates``. Bootstrap
    width is then checked only at eligible rank/SE looks; all three guards
    must pass. Neither bootstrap resampling nor its extra solves are incurred
    by default. Supplying replicates alone requests the terminal diagnostic
    without a width constraint. The caller should predeclare the width and
    computational budget, rather than select them after inspecting the data.

    This is the rank-aware policy validated on engineered stretched molecular
    banks, not a chemical-accuracy guarantee or finite-sample confidence
    sequence. The combined width/rank policy is more conservative but its
    generalization has not been established. Use :func:`acquire_ritz` for
    explicit SE-only/fixed-budget comparisons and legacy record replay.
    """
    if not isinstance(rank_stop, RankStopPolicy):
        raise ValueError('precision acquisition requires a RankStopPolicy; use acquire_ritz for SE-only stopping')
    return acquire_ritz(
        shared, backend, pilot_shots_per_group=pilot_shots_per_group,
        round_budgets=round_budgets, allocation=allocation,
        target_std_error=target_std_error, solver_kwargs=solver_kwargs, root=root,
        delta=delta, control_variates=control_variates, rank_stop=rank_stop,
        max_bootstrap_half_width=max_bootstrap_half_width,
        bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed)


__all__ = ["ALLOCATIONS", "AcquisitionError", "RitzAcquisition", "RankStopPolicy", "TIE_TOLERANCE",
           "acquire_ritz", "acquire_ritz_precision", "neyman_group_plan", "uniform_group_plan"]
