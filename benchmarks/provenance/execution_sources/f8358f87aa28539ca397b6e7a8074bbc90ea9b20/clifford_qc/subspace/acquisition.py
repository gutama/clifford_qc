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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from ..measurement.cache import GroupedWordCache
from ..measurement.functionals import WordFunctional, ritz_functional
from ..measurement.grouping import shared_basis
from ..measurement.session import SharedMeasurement
from .coverage import delta_method_applies
from .linalg import SubspaceResult
from .uncertainty import Interval, ritz_uncertainty

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
    """

    result: SubspaceResult
    cache: GroupedWordCache
    interval: Interval
    naive_interval: Interval
    looks: tuple[dict[str, Any], ...]
    stopped: str
    allocation: str
    schedule: dict[str, Any]

    @property
    def shots(self) -> int:
        """Every physical shot spent, pilot included."""
        return int(self.cache.total_shots)

    @property
    def std_error(self) -> float:
        return float(self.looks[-1]["std_error"])


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
                 delta: float = 0.05) -> RitzAcquisition:
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
    # Every look at which the run could report: after the pilot and each round.
    family = len(budgets) + 1 if target_std_error is not None else 1

    cache = shared.measure(backend, pilot)
    looks: list[dict[str, Any]] = []
    stopped = "schedule_exhausted"
    point = None
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
        functional = ritz_functional(shared.bank, point.indices,
                                     point.ritz_vector(root), point.energies[root])
        sigma = math.sqrt(max(0.0, functional.variance(cache)))
        retained = point.resources.get("smallest_retained_overlap_eigenvalue")
        look = {"look": index, "shots": int(cache.total_shots),
                "energy": float(point.energies[root]), "std_error": sigma,
                "retained_rank": int(point.effective_rank),
                "smallest_retained_overlap": None if retained is None else float(retained),
                "stopped": False}
        looks.append(look)
        if target_std_error is not None and sigma <= target_std_error:
            stopped = "target_met"
            break
        if index == len(budgets):
            break
        if allocation == "uniform":
            plan = uniform_group_plan(shared.groups, budgets[index])
        else:
            plan = neyman_group_plan(shared.groups, cache, functional, budgets[index])
        if not plan:
            # A deterministic Ritz functional: no shot can change the estimate.
            stopped = "deterministic"
            break
        shared.measure_plan(backend, plan, cache)
    assert point is not None
    looks[-1]["stopped"] = True
    return RitzAcquisition(
        result=point, cache=cache,
        interval=ritz_uncertainty(shared, cache, point, root=root, delta=delta,
                                  family=family),
        naive_interval=ritz_uncertainty(shared, cache, point, root=root, delta=delta),
        looks=tuple(looks), stopped=stopped, allocation=allocation,
        schedule={"pilot_shots_per_group": pilot, "round_budgets": list(budgets),
                  "target_std_error": target_std_error, "delta": float(delta),
                  "interval_family": family, "solver_kwargs": kwargs})


__all__ = ["ALLOCATIONS", "AcquisitionError", "RitzAcquisition", "TIE_TOLERANCE",
           "acquire_ritz", "neyman_group_plan", "uniform_group_plan"]
