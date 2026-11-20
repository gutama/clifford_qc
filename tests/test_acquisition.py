"""Target-aware acquisition: the schedule, the stopping rule and its accounting.

The claims worth testing here are bookkeeping claims. Every shot is counted, a
stopping rule only ever truncates the run it would otherwise have continued,
and the interval reported at a stopped look pays for every look at which the
run could have stopped.
"""

import math
from statistics import NormalDist

import numpy as np
import pytest

from benchmarks import run_interval_coverage as coverage_producer
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.subspace import acquisition
from clifford_qc.measurement.allocation import variance_optimal_group_plan
from clifford_qc.measurement.functionals import WordFunctional, ritz_functional
from clifford_qc.measurement.grouping import shared_basis
from clifford_qc.subspace.acquisition import (
    AcquisitionError, acquire_ritz, neyman_group_plan, uniform_group_plan,
)
from clifford_qc.subspace.uncertainty import (
    HEURISTIC, bootstrap_ritz_samples, bootstrap_spread, bootstrap_spread_interval,
)

ORTHOGONAL = {"key": "tfim4_orthogonal", "role": "development",
              "model": {"family": "tfim", "n": 4, "J": 1.0, "h": 1.0},
              "pool_words": 3, "krylov_powers": 0}


@pytest.fixture(scope="module")
def system():
    return coverage_producer.build_system(ORTHOGONAL)


def run(shared, seed, **kwargs):
    kwargs.setdefault("pilot_shots_per_group", 100)
    kwargs.setdefault("round_budgets", [1600, 1600, 3200])
    return acquire_ritz(shared, FiniteShotBackend(seed=seed), **kwargs)


# --------------------------------------------------------------- uniform plan

def test_uniform_plan_spends_the_budget_and_is_constant_within_groups(system):
    shared, _ = system
    groups = len(shared.groups)
    plan = uniform_group_plan(shared.groups, 5 * groups + 3)
    per_group = [{plan[word.code] for word in group} for group in shared.groups]
    assert all(len(counts) == 1 for counts in per_group)
    spent = [next(iter(counts)) for counts in per_group]
    assert sum(spent) == 5 * groups + 3
    assert spent == [6] * 3 + [5] * (groups - 3)


def test_uniform_plan_smaller_than_the_group_count_leaves_groups_unmeasured(system):
    shared, _ = system
    plan = uniform_group_plan(shared.groups, 2)
    measured = [group for group in shared.groups if group[0].code in plan]
    assert len(measured) == 2
    assert set(plan.values()) == {1}


@pytest.mark.parametrize("budget", [0, -1, 2.0, True])
def test_uniform_plan_rejects_invalid_budgets(system, budget):
    with pytest.raises(ValueError):
        uniform_group_plan(system[0].groups, budget)


def test_uniform_plan_rejects_an_empty_grouping():
    with pytest.raises(ValueError):
        uniform_group_plan([], 10)


# ---------------------------------------------------------------- Neyman plan

def pilot_functional(shared, seed, shots=100):
    cache = shared.measure(FiniteShotBackend(seed=seed), shots)
    point = shared.solve(cache)
    return cache, ritz_functional(shared.bank, point.indices, point.ritz_vector(0),
                                  point.energies[0])


def group_shots(shared, plan):
    return [plan.get(group[0].code, 0) for group in shared.groups]


def test_neyman_plan_spends_the_budget_in_proportion_to_root_variance(system):
    shared, _ = system
    cache, functional = pilot_functional(shared, 41)
    plan = neyman_group_plan(shared.groups, cache, functional, 100_000)
    for group in shared.groups:
        assert len({plan.get(word.code, 0) for word in group}) == 1
    shots = group_shots(shared, plan)
    assert sum(shots) == 100_000
    statistics = cache.candidate_group_statistics(dict(functional.coefficients))
    roots = [math.sqrt(statistics[key][1]) for key in statistics]
    assert sorted(shots) == pytest.approx(
        sorted(100_000 * root / sum(roots) for root in roots), abs=1.0)


def test_neyman_plan_is_the_existing_rule_when_nothing_ties(system):
    shared, _ = system
    agreed = 0
    for seed in range(60, 72):
        cache, functional = pilot_functional(shared, seed)
        ours = neyman_group_plan(shared.groups, cache, functional, 100_003)
        theirs = variance_optimal_group_plan(shared.groups, cache, [functional], 100_003)
        difference = [a - b for a, b in zip(group_shots(shared, ours),
                                            group_shots(shared, theirs))]
        # At most the spare shots of a tie move, and they move between settings.
        assert sum(difference) == 0 and max(map(abs, difference)) <= 1
        agreed += not any(difference)
    assert agreed >= 6


class TiedStatistics:
    """A cache stand-in whose last two settings tie up to a relative ``jitter``."""

    def __init__(self, keys, jitter):
        variances = [4.0, 1.0, 0.25] + [0.01] * (len(keys) - 3)
        variances[-1] *= 1.0 + jitter
        self.rows = {key: (100, variance, 2.0) for key, variance in zip(keys, variances)}

    def candidate_group_statistics(self, coefficients):
        return self.rows


def test_neyman_plan_breaks_a_tie_by_group_order_not_by_the_last_bit(system):
    shared, _ = system
    _, functional = pilot_functional(shared, 41)
    keys = [tuple(sorted(shared_basis(tuple(group)).items())) for group in shared.groups]
    # Root weights 2, 1, 0.5 and five tied at 0.1: of 976 shots the tied
    # settings are each owed 24.4, and exactly two spare shots are left.
    plans = [group_shots(shared, neyman_group_plan(
        shared.groups, TiedStatistics(keys, jitter), functional, 976))
        for jitter in (0.0, 1e-13, -1e-13)]
    assert plans[0] == plans[1] == plans[2] == [488, 244, 122, 25, 25, 24, 24, 24]


def test_a_targeted_schedule_does_not_depend_on_the_last_bits_of_the_ritz_vector(
        system, monkeypatch):
    """Symmetric settings tie; the eigensolver's rounding must not pick the winner.

    Two settings of this bank have weights that are equal in exact arithmetic.
    Ordering their remainders by raw floats gave a different sample stream
    under a different BLAS kernel in about one run in ten. The jitter here
    stands in for that kernel difference.
    """
    shared, _ = system
    groups = len(shared.groups)
    budgets = [groups * step for step in (41, 59, 83, 117, 166, 234, 331, 469)]
    exact = acquisition.ritz_functional

    def jittered(seed):
        noise = np.random.default_rng(seed)

        def functional(*args):
            found = exact(*args)
            return WordFunctional({
                code: value * (1.0 + 1e-13 * noise.standard_normal())
                for code, value in found.coefficients.items()}, found.constant)
        return functional

    def spent(seed, jitter):
        monkeypatch.setattr(acquisition, "ritz_functional",
                            jittered(jitter) if jitter else exact)
        result = acquire_ritz(shared, FiniteShotBackend(seed=seed),
                              pilot_shots_per_group=100, round_budgets=budgets)
        return [result.cache.shots(group[0].code) for group in shared.groups]

    def legacy(groups, cache, functional, budget):
        return variance_optimal_group_plan(groups, cache, [functional], budget)

    seeds = range(9000, 9060)
    assert all(spent(seed, 0) == spent(seed, 1) == spent(seed, 2) for seed in seeds)
    monkeypatch.setattr(acquisition, "neyman_group_plan", legacy)
    assert any(spent(seed, 0) != spent(seed, 1) for seed in seeds)


def test_neyman_plan_of_a_deterministic_functional_is_empty(system):
    shared, _ = system
    cache, _ = pilot_functional(shared, 43)
    assert neyman_group_plan(shared.groups, cache, WordFunctional({}, 1.0), 100) == {}


def test_neyman_plan_needs_a_pilot_and_a_positive_integer_budget(system):
    shared, _ = system
    cache, functional = pilot_functional(shared, 47)
    with pytest.raises(ValueError, match="pilot"):
        neyman_group_plan(shared.groups, shared.new_cache(), functional, 100)
    for budget in (0, -3, 1.5, True):
        with pytest.raises(ValueError):
            neyman_group_plan(shared.groups, cache, functional, budget)


def test_neyman_plan_with_no_variance_anywhere_falls_back_to_equal_shares(system):
    shared, _ = system
    cache, functional = pilot_functional(shared, 53)
    flat = {key: (count, 0.0, width)
            for key, (count, _, width) in cache.candidate_group_statistics(
                dict(functional.coefficients)).items()}

    class Flat:
        def candidate_group_statistics(self, coefficients):
            return flat

    shots = group_shots(shared, neyman_group_plan(shared.groups, Flat(), functional, 803))
    touched = [count for count in shots if count]
    assert sum(shots) == 803 and max(touched) - min(touched) <= 1
    assert touched == sorted(touched, reverse=True)


# ------------------------------------------------------------------- schedule

@pytest.mark.parametrize("allocation", acquisition.ALLOCATIONS)
def test_a_full_schedule_spends_exactly_the_declared_shots(system, allocation):
    shared, _ = system
    groups = len(shared.groups)
    result = run(shared, 3, allocation=allocation)
    assert result.stopped == "schedule_exhausted"
    assert result.allocation == allocation
    assert [look["shots"] for look in result.looks] == [
        100 * groups, 100 * groups + 1600, 100 * groups + 3200, 100 * groups + 6400]
    assert result.shots == result.looks[-1]["shots"] == result.cache.total_shots
    assert [look["stopped"] for look in result.looks] == [False, False, False, True]
    assert [look["look"] for look in result.looks] == [0, 1, 2, 3]
    assert result.std_error == result.looks[-1]["std_error"] == result.interval.std_error
    assert result.result.ground_energy == result.looks[-1]["energy"]
    assert all(look["retained_rank"] >= 1 for look in result.looks)
    assert all(0.0 < look["smallest_retained_overlap"] <= 1.5 for look in result.looks)


def test_without_a_stopping_rule_the_interval_is_the_single_look_interval(system):
    shared, _ = system
    result = run(shared, 5)
    assert result.schedule["interval_family"] == 1
    assert result.schedule["target_std_error"] is None
    assert (result.interval.lower, result.interval.upper) == (
        result.naive_interval.lower, result.naive_interval.upper)
    # The radius uses each group's sample variance and the reported standard
    # error its population variance, so they agree to a factor near N/(N-1).
    radius = (result.interval.upper - result.interval.lower) / 2
    assert radius == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 2) * result.std_error,
                                   rel=0.02)
    assert radius > NormalDist().inv_cdf(1 - 0.05 / 2) * result.std_error


def test_both_allocations_read_the_same_pilot(system):
    shared, _ = system
    uniform = run(shared, 7, allocation="uniform")
    targeted = run(shared, 7, allocation="ritz_targeted")
    assert uniform.looks[0] == {**targeted.looks[0]}
    assert uniform.looks[1]["energy"] != targeted.looks[1]["energy"]
    assert uniform.shots == targeted.shots


def test_targeted_rounds_put_more_shots_where_the_ritz_value_is_sensitive(system):
    shared, _ = system
    uniform = run(shared, 7, allocation="uniform")
    targeted = run(shared, 7, allocation="ritz_targeted")

    def per_group(result):
        return [result.cache.shots(group[0].code) for group in shared.groups]

    assert len(set(per_group(uniform))) == 1
    assert len(set(per_group(targeted))) > 1
    assert sum(per_group(targeted)) == sum(per_group(uniform)) == targeted.shots
    # No setting is ever measured below its pilot.
    assert min(per_group(targeted)) >= 100


# ------------------------------------------------------------------- stopping

def test_a_stopping_rule_only_truncates_the_full_schedule(system):
    shared, _ = system
    full = run(shared, 11)
    sigmas = [look["std_error"] for look in full.looks]
    assert sigmas[0] > sigmas[2] > sigmas[3]
    target = (sigmas[1] + sigmas[2]) / 2 if sigmas[1] > sigmas[2] else sigmas[2] * 1.0001
    stop_at = next(i for i, sigma in enumerate(sigmas) if sigma <= target)
    assert 0 < stop_at < 3

    stopped = run(shared, 11, target_std_error=target)
    assert stopped.stopped == "target_met"
    assert len(stopped.looks) == stop_at + 1
    for early, late in zip(stopped.looks, full.looks):
        assert {**early, "stopped": None} == {**late, "stopped": None}
    assert stopped.looks[-1]["stopped"] is True
    assert stopped.shots == full.looks[stop_at]["shots"] < full.shots
    assert stopped.std_error <= target


def test_the_stopped_interval_pays_for_every_look_it_could_have_stopped_at(system):
    shared, _ = system
    stopped = run(shared, 11, target_std_error=1.0)
    looks = len(stopped.schedule["round_budgets"]) + 1
    assert stopped.schedule["interval_family"] == looks == 4
    naive = (stopped.naive_interval.upper - stopped.naive_interval.lower) / 2
    valid = (stopped.interval.upper - stopped.interval.lower) / 2
    assert valid / naive == pytest.approx(
        NormalDist().inv_cdf(1 - 0.05 / (2 * looks)) / NormalDist().inv_cdf(1 - 0.05 / 2),
        rel=1e-9)
    assert valid > naive
    assert stopped.interval.estimate == stopped.naive_interval.estimate
    assert not stopped.interval.certified


def test_a_target_met_by_the_pilot_spends_only_the_pilot(system):
    shared, _ = system
    stopped = run(shared, 13, target_std_error=10.0)
    assert stopped.stopped == "target_met"
    assert len(stopped.looks) == 1
    assert stopped.shots == 100 * len(shared.groups)


def test_an_unreachable_target_exhausts_the_schedule_and_still_pays_for_its_looks(system):
    shared, _ = system
    full = run(shared, 17)
    stopped = run(shared, 17, target_std_error=1e-9)
    assert stopped.stopped == "schedule_exhausted"
    assert stopped.shots == full.shots
    assert stopped.result.ground_energy == full.result.ground_energy
    assert stopped.schedule["interval_family"] == 4
    assert (stopped.interval.upper - stopped.interval.lower) > (
        full.interval.upper - full.interval.lower)
    assert (stopped.naive_interval.lower, stopped.naive_interval.upper) == (
        full.interval.lower, full.interval.upper)


def test_a_target_met_only_at_the_last_look_is_reported_as_met(system):
    shared, _ = system
    full = run(shared, 17)
    sigmas = [look["std_error"] for look in full.looks]
    assert sigmas[-1] < min(sigmas[:-1])
    stopped = run(shared, 17, target_std_error=sigmas[-1])
    assert stopped.stopped == "target_met"
    assert stopped.shots == full.shots and len(stopped.looks) == len(full.looks)


def test_the_stopping_look_is_the_first_at_or_below_the_target(system):
    shared, _ = system
    full = run(shared, 19)
    exact = full.looks[2]["std_error"]
    stopped = run(shared, 19, target_std_error=exact)
    assert stopped.looks[-1]["std_error"] <= exact
    assert all(look["std_error"] > exact for look in stopped.looks[:-1])


def test_a_deterministic_functional_ends_the_schedule_without_spending(system, monkeypatch):
    shared, _ = system
    monkeypatch.setattr(acquisition, "neyman_group_plan", lambda *args, **kwargs: {})
    result = run(shared, 23)
    assert result.stopped == "deterministic"
    assert result.shots == 100 * len(shared.groups)
    assert len(result.looks) == 1 and result.looks[0]["stopped"] is True


# ------------------------------------------------------------------- failures

def test_a_look_that_cannot_be_solved_reports_the_shots_already_spent(system, monkeypatch):
    shared, _ = system
    solve = type(shared).solve
    calls = {"count": 0}

    def failing(self, cache, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise np.linalg.LinAlgError("no retained mode")
        return solve(self, cache, **kwargs)

    monkeypatch.setattr(type(shared), "solve", failing)
    with pytest.raises(AcquisitionError, match="look 2 could not be solved") as caught:
        run(shared, 29)
    error = caught.value
    assert error.shots == 100 * len(shared.groups) + 3200
    assert len(error.looks) == 2
    assert error.cache.total_shots == error.shots
    assert isinstance(error.__cause__, np.linalg.LinAlgError)


def test_a_root_the_solve_did_not_retain_is_a_failure_with_its_cost(system):
    shared, _ = system
    excited = run(shared, 37, root=1)
    ground = run(shared, 37, root=0)
    assert excited.looks[0]["energy"] > ground.looks[0]["energy"]
    with pytest.raises(AcquisitionError, match="fewer than root 9 needs") as caught:
        run(shared, 37, root=9)
    assert caught.value.shots == 100 * len(shared.groups) and caught.value.looks == ()


@pytest.mark.parametrize("kwargs", [
    {"pilot_shots_per_group": 0}, {"pilot_shots_per_group": 2.5},
    {"pilot_shots_per_group": True}, {"round_budgets": []},
    {"round_budgets": [100, 0]}, {"round_budgets": [100.0]},
    {"allocation": "neyman"}, {"target_std_error": 0.0},
    {"target_std_error": -1.0}, {"target_std_error": math.inf},
    {"target_std_error": math.nan}, {"target_std_error": True},
    {"target_std_error": "small"}, {"delta": 0.0}, {"delta": 1.0},
    {"root": -1}, {"root": 0.0}, {"root": True},
    {"solver_kwargs": {"overlap_repair": "nearest_correlation"}},
    {"solver_kwargs": {"pencil_angle": 0.1}},
    {"solver_kwargs": {"ratio_error_budget": 0.01}},
])
def test_invalid_schedules_are_rejected_before_any_shot(system, kwargs):
    shared, _ = system

    class NoShots:
        def sample_grouped_from_state(self, *args, **kwargs):
            raise AssertionError("a rejected schedule must not sample")

    arguments = {"pilot_shots_per_group": 100, "round_budgets": [800], **kwargs}
    with pytest.raises(ValueError):
        acquire_ritz(shared, NoShots(), **arguments)


def test_solver_policy_is_passed_to_every_look(system):
    shared, _ = system
    result = run(shared, 31, solver_kwargs={"calibrate_overlap": True})
    assert result.schedule["solver_kwargs"] == {"calibrate_overlap": True}
    assert result.result.estimator_parameters["calibrate_overlap"] is True


# ------------------------------------------------------------------ behaviour

def test_targeting_lowers_the_error_at_matched_shots_on_a_well_conditioned_bank(system):
    shared, design = system
    errors = {allocation: np.array([
        run(shared, 4000 + seed, allocation=allocation,
            round_budgets=[4800, 9600]).result.ground_energy - design["target"]
        for seed in range(48)]) for allocation in acquisition.ALLOCATIONS}
    mse = {allocation: float(np.mean(values ** 2)) for allocation, values in errors.items()}
    assert mse["ritz_targeted"] < 0.75 * mse["uniform"]


def test_the_estimated_standard_error_tracks_the_spread_under_targeting(system):
    shared, _ = system
    runs = [run(shared, 5000 + seed, round_budgets=[4800, 9600]) for seed in range(48)]
    spread = float(np.std([r.result.ground_energy for r in runs], ddof=1))
    claimed = float(np.sqrt(np.mean([r.std_error ** 2 for r in runs])))
    assert claimed == pytest.approx(spread, rel=0.3)


# ---------------------------------------------------------- control variates

TRI = ('&FCI NORB=3, NELEC=2, MS2=0, ORBSYM=1,1,1, ISYM=1 /\n'
       '0.6 1 1 1 1\n0.5 2 2 2 2\n0.4 3 3 3 3\n0.1 2 1 2 1\n0.1 3 1 3 1\n'
       '0.12 3 2 3 2\n0.3 2 2 1 1\n0.3 3 3 1 1\n0.3 3 3 2 2\n0.05 2 1 1 1\n0.04 3 2 2 2\n'
       '-0.9 1 1 0 0\n-1.0 2 2 0 0\n0.3 3 3 0 0\n0.05 3 2 0 0\n0.07 2 1 0 0\n0.0 0 0 0 0\n')


@pytest.fixture(scope="module")
def molecular(tmp_path_factory):
    """A correlated two-electron singlet in three orbitals, with spin-adapted doubles."""
    from clifford_qc.dense_reference import density_from_statevector
    from clifford_qc.measurement.session import SharedMeasurement
    from clifford_qc.models import fcidump_model
    from clifford_qc.subspace import MatrixElementBank, identity_generator
    from clifford_qc.subspace.symmetry_pools import spin_adapted_excitations
    from clifford_qc.subspace.symmetry_shifts import sector_shift_family

    path = tmp_path_factory.mktemp("acquisition") / "tri.FCIDUMP"
    path.write_text(TRI)
    hamiltonian = fcidump_model(path).hamiltonian.to_mv()
    rng = np.random.default_rng(5)
    psi = np.zeros(64, complex)
    for a in range(3):
        for b in range(3):
            bits = ["0"] * 6
            bits[2 * a] = bits[2 * b + 1] = "1"
            psi[int("".join(bits), 2)] += rng.normal() * (1.0 if a == b else 0.3)
    rho = density_from_statevector(psi / np.linalg.norm(psi))
    generators = [identity_generator(6)] + spin_adapted_excitations(3, (0,))
    shared = SharedMeasurement(MatrixElementBank(rho, hamiltonian, generators))
    return shared, sector_shift_family(3, 2, sz=0.0)


def corrected(shared, seed, family, **kwargs):
    kwargs.setdefault("pilot_shots_per_group", 100)
    kwargs.setdefault("round_budgets", [100 * len(shared.groups)] * 2)
    return acquire_ritz(shared, FiniteShotBackend(seed=seed), control_variates=family,
                        **kwargs)


@pytest.mark.parametrize("bad", ["leaky", "empty", "not_terms"])
def test_a_control_variate_family_is_certified_before_any_shot(molecular, bad):
    from clifford_qc.subspace.symmetry_shifts import sector_shift_family

    shared, family = molecular
    # Three electrons is the wrong sector: N - 3 does not annihilate the bank.
    families = {"leaky": sector_shift_family(3, 3), "empty": [],
                "not_terms": [term.operator for term in family]}

    class NoShots:
        def sample_grouped_from_state(self, *args, **kwargs):
            raise AssertionError("a refused family must not sample")

    with pytest.raises(ValueError):
        acquire_ritz(shared, NoShots(), pilot_shots_per_group=100, round_budgets=[800],
                     control_variates=families[bad])


def test_a_corrected_run_reports_the_corrected_value_and_its_interval(molecular):
    from clifford_qc.subspace.symmetry_shifts import (corrected_ritz_functional,
                                                      ritz_control_variate, shift_pair_table)

    shared, family = molecular
    result = corrected(shared, 61, family, target_std_error=1e-9)
    assert result.stopped == "schedule_exhausted"
    table = shift_pair_table(shared.bank, shared.indices, family)
    final, _ = corrected_ritz_functional(shared, result.cache, result.result, family,
                                         table=table)
    assert result.control_variate == final
    # The run builds its correction from the pair table; the operator products
    # of ritz_control_variate give the same correction to rounding.
    direct = ritz_control_variate(shared, result.cache, result.result, family)
    assert direct.used == final.used and direct.unreadable == final.unreadable
    assert direct.energy == pytest.approx(final.energy, abs=1e-12)
    assert direct.std_error == pytest.approx(final.std_error, rel=1e-9)
    assert result.energy == result.looks[-1]["energy"] == final.energy
    assert result.std_error == final.std_error
    assert result.looks[-1]["raw_energy"] == result.result.ground_energy == final.raw_energy
    assert result.looks[-1]["control_variates_used"] == len(final.used) > 0
    assert result.schedule["control_variates"] == [term.label for term in family]
    looks = len(result.looks)
    for interval, family_size in [(result.interval, looks), (result.naive_interval, 1)]:
        z = NormalDist().inv_cdf(1 - 0.05 / (2 * family_size))
        assert interval.estimate == final.energy
        assert interval.std_error == final.std_error
        assert interval.upper - interval.estimate == pytest.approx(z * final.std_error)
        assert interval.estimate - interval.lower == pytest.approx(z * final.std_error)
    # A fitted correction can only lower the in-sample variance: xi = 0 is allowed.
    for look in result.looks:
        assert look["std_error"] <= look["raw_std_error"] * (1 + 1e-9)


def test_an_uncorrected_run_is_unchanged(molecular):
    shared, _ = molecular
    plain = corrected(shared, 61, None)
    assert plain.control_variate is None
    assert "control_variates" not in plain.schedule
    assert all("raw_energy" not in look for look in plain.looks)
    assert plain.energy == plain.result.ground_energy


def test_under_uniform_allocation_the_correction_changes_no_shot(molecular):
    from clifford_qc.measurement.session import grouped_cache_fingerprint

    shared, family = molecular
    plain = corrected(shared, 62, None, allocation="uniform")
    fixed = corrected(shared, 62, family, allocation="uniform")
    assert grouped_cache_fingerprint(fixed.cache) == grouped_cache_fingerprint(plain.cache)
    assert [look["raw_energy"] for look in fixed.looks] == \
        [look["energy"] for look in plain.looks]
    assert [look["raw_std_error"] for look in fixed.looks] == \
        pytest.approx([look["std_error"] for look in plain.looks], rel=1e-9)


def test_targeted_rounds_are_allocated_to_the_corrected_functional(molecular):
    from clifford_qc.measurement.session import grouped_cache_fingerprint
    from clifford_qc.subspace.symmetry_shifts import corrected_ritz_functional, shift_pair_table

    shared, family = molecular
    budget = 100 * len(shared.groups)
    result = corrected(shared, 63, family, round_budgets=[budget])
    backend = FiniteShotBackend(seed=63)
    cache = shared.measure(backend, 100)
    _, functional = corrected_ritz_functional(
        shared, cache, shared.solve(cache), family,
        table=shift_pair_table(shared.bank, shared.indices, family))
    plan = neyman_group_plan(shared.groups, cache, functional, budget)
    point = shared.solve(cache)
    raw = ritz_functional(shared.bank, point.indices, point.ritz_vector(0),
                          point.energies[0])
    assert plan != neyman_group_plan(shared.groups, cache, raw, budget)
    shared.measure_plan(backend, plan, cache)
    assert grouped_cache_fingerprint(result.cache) == grouped_cache_fingerprint(cache)


def test_a_corrected_run_stops_on_the_corrected_standard_error(molecular):
    shared, family = molecular
    pilot = corrected(shared, 64, family, round_budgets=[1])
    first = pilot.looks[0]
    assert first["std_error"] < first["raw_std_error"]
    target = 0.5 * (first["std_error"] + first["raw_std_error"])
    stopped = corrected(shared, 64, family, target_std_error=target)
    plain = corrected(shared, 64, None, target_std_error=target)
    assert stopped.stopped == "target_met" and len(stopped.looks) == 1
    assert stopped.shots == 100 * len(shared.groups)
    assert len(plain.looks) > 1


# ------------------------------------------------- bootstrap interval at the stop

def test_the_bootstrap_spread_is_the_central_range_and_ignores_the_tails():
    rng = np.random.default_rng(3)
    draws = rng.normal(2.0, 0.5, 20000)
    assert bootstrap_spread(draws) == pytest.approx(0.5, rel=0.03)
    # Five percent of draws far out in one tail move the standard deviation by
    # orders of magnitude and the central range only by a few percent.
    shifted = draws.copy()
    shifted[:1000] = 1e3
    assert np.std(shifted) > 100 * np.std(draws)
    assert bootstrap_spread(shifted) == pytest.approx(0.5, rel=0.1)
    low, high = np.quantile(shifted, [0.16, 0.84])
    assert bootstrap_spread(shifted) == (high - low) / 2


@pytest.mark.parametrize("draws", [[1.0], [], [1.0, math.nan], [1.0, math.inf], [[1.0, 2.0]]])
def test_the_bootstrap_spread_needs_two_finite_draws(draws):
    with pytest.raises(ValueError):
        bootstrap_spread(draws)


def test_the_bootstrap_spread_interval_uses_the_delta_method_quantile():
    draws = np.linspace(-1.0, 1.0, 101)
    interval = bootstrap_spread_interval(5.0, draws, delta=0.1, family=4)
    z = NormalDist().inv_cdf(1.0 - 0.1 / 8.0)
    assert interval.estimate == 5.0 and interval.std_error == bootstrap_spread(draws)
    assert interval.upper - 5.0 == pytest.approx(z * interval.std_error, rel=1e-12)
    assert 5.0 - interval.lower == pytest.approx(z * interval.std_error, rel=1e-12)
    assert interval.evidence == HEURISTIC and not interval.certified
    with pytest.raises(ValueError):
        bootstrap_spread_interval(5.0, draws, delta=1.0)


def test_without_a_bootstrap_nothing_about_it_is_reported(system):
    found = run(system[0], 61, target_std_error=0.02)
    assert found.bootstrap_interval is None and "bootstrap" not in found.schedule
    assert not any("bootstrap_spread" in look or "bootstrap_failure" in look
                   for look in found.looks)


def test_a_bootstrap_changes_nothing_the_run_measured(system):
    shared, _ = system
    plain = run(shared, 61, target_std_error=0.02)
    booted = run(shared, 61, target_std_error=0.02, bootstrap_replicates=20, bootstrap_seed=4)
    assert booted.shots == plain.shots and booted.stopped == plain.stopped
    assert booted.interval == plain.interval and booted.naive_interval == plain.naive_interval
    assert list(booted.result.energies) == list(plain.result.energies)
    stripped = [{k: v for k, v in look.items() if k != "bootstrap_spread"}
                for look in booted.looks]
    assert stripped == list(plain.looks)
    assert booted.schedule == {**plain.schedule, "bootstrap": {"replicates": 20, "seed": 4}}


@pytest.mark.parametrize("target", [0.02, None])
def test_the_bootstrap_interval_resamples_the_last_look_and_pays_for_every_look(system,
                                                                                 target):
    shared, _ = system
    found = run(shared, 67, target_std_error=target, bootstrap_replicates=40,
                bootstrap_seed=11)
    draws = bootstrap_ritz_samples(shared, found.cache, result=found.result,
                                   replicates=40, seed=11)
    family = 4 if target is not None else 1
    assert found.schedule["interval_family"] == family
    assert found.looks[-1]["bootstrap_spread"] == bootstrap_spread(draws)
    assert found.bootstrap_interval == bootstrap_spread_interval(
        found.result.energies[0], draws, delta=0.05, family=family)
    # The quantile is the delta-method interval's: it pays for the same looks.
    z = NormalDist().inv_cdf(1.0 - 0.05 / (2 * family))
    assert found.bootstrap_interval.upper - found.energy == pytest.approx(
        z * found.looks[-1]["bootstrap_spread"], rel=1e-12)


def test_a_failed_bootstrap_is_recorded_and_the_acquisition_stands(system, monkeypatch):
    shared, _ = system
    plain = run(shared, 71, target_std_error=0.02)

    def failing(*args, **kwargs):
        raise ValueError("bootstrap replica 3 retained 0 root(s); root 0 is unavailable")

    monkeypatch.setattr(acquisition, "bootstrap_ritz_samples", failing)
    found = run(shared, 71, target_std_error=0.02, bootstrap_replicates=10)
    assert found.bootstrap_interval is None
    assert found.looks[-1]["bootstrap_failure"] == (
        "ValueError: bootstrap replica 3 retained 0 root(s); root 0 is unavailable")
    assert "bootstrap_spread" not in found.looks[-1]
    assert found.interval == plain.interval and found.shots == plain.shots


@pytest.mark.parametrize("kwargs", [
    {"bootstrap_replicates": 1}, {"bootstrap_replicates": 0},
    {"bootstrap_replicates": 2.5}, {"bootstrap_replicates": True},
    {"bootstrap_replicates": 10, "bootstrap_seed": -1},
    {"bootstrap_replicates": 10, "bootstrap_seed": 1.5},
    {"bootstrap_replicates": 10, "bootstrap_seed": True},
])
def test_invalid_bootstrap_arguments_are_rejected_before_any_shot(system, kwargs):
    class NoShots:
        def sample_grouped_from_state(self, *args, **kwargs):
            raise AssertionError("a rejected bootstrap must not sample")

    with pytest.raises(ValueError):
        acquire_ritz(system[0], NoShots(), pilot_shots_per_group=100, round_budgets=[800],
                     **kwargs)


def test_a_corrected_run_bootstraps_the_corrected_value(molecular):
    from clifford_qc.subspace.symmetry_shifts import corrected_ritz_functional, shift_pair_table

    shared, family = molecular
    table = shift_pair_table(shared.bank, shared.indices, family)
    plain = corrected(shared, 83, family)
    found = corrected(shared, 83, family, bootstrap_replicates=24, bootstrap_seed=6)
    # The bootstrap changes nothing the run measured or reported.
    assert found.shots == plain.shots and found.interval == plain.interval
    assert [{k: v for k, v in look.items() if k != "bootstrap_spread"} for look in found.looks] \
        == list(plain.looks)

    def refit(replica, solved):
        return corrected_ritz_functional(shared, replica, solved, family, table=table)[0].energy

    draws = bootstrap_ritz_samples(shared, found.cache, result=found.result, replicates=24,
                                   seed=6, statistic=refit)
    assert found.looks[-1]["bootstrap_spread"] == bootstrap_spread(draws)
    assert found.bootstrap_interval == bootstrap_spread_interval(
        found.control_variate.energy, draws, delta=0.05, family=1)
    assert found.bootstrap_interval.estimate == found.energy != found.result.energies[0]
    # Refitting on every resample removes most of the spread the correction removes.
    raw = bootstrap_ritz_samples(shared, found.cache, result=found.result, replicates=24, seed=6)
    assert bootstrap_spread(draws) < 0.7 * bootstrap_spread(raw)


def test_a_statistic_reads_the_same_resamples_as_the_ritz_value(system):
    shared, _ = system
    found = run(shared, 89)
    plain = bootstrap_ritz_samples(shared, found.cache, result=found.result, replicates=12,
                                   seed=3)
    same = bootstrap_ritz_samples(shared, found.cache, result=found.result, replicates=12,
                                  seed=3, statistic=lambda replica, solved: solved.energies[0])
    assert list(same) == list(plain)
    seen = []
    bootstrap_ritz_samples(shared, found.cache, result=found.result, replicates=3, seed=3,
                           statistic=lambda replica, solved: seen.append(
                               (replica.total_shots, solved.estimator_parameters)) or 0.0)
    assert seen == [(found.shots, found.result.estimator_parameters)] * 3
def test_the_bootstrap_spread_interval_refuses_a_non_finite_estimate_and_never_overflows():
    for estimate in (math.nan, math.inf):
        with pytest.raises(ValueError, match="estimate must be finite"):
            bootstrap_spread_interval(estimate, [0.0, 1.0, 2.0])
    # The spread squared would overflow; the radius must not.
    interval = bootstrap_spread_interval(0.0, [-2e200, 0.0, 2e200])
    assert math.isfinite(interval.upper) and interval.upper > 1e200


def test_a_bootstrap_of_an_excited_root_resamples_that_root(system):
    shared, _ = system
    found = run(shared, 37, root=1, bootstrap_replicates=30, bootstrap_seed=5)
    draws = bootstrap_ritz_samples(shared, found.cache, result=found.result, root=1,
                                   replicates=30, seed=5)
    assert found.looks[-1]["bootstrap_spread"] == bootstrap_spread(draws)
    assert found.bootstrap_interval.estimate == found.result.energies[1] == found.energy
    ground = bootstrap_ritz_samples(shared, found.cache, result=found.result, root=0,
                                    replicates=30, seed=5)
    assert bootstrap_spread(ground) != bootstrap_spread(draws)
