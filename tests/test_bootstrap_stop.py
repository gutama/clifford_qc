"""The bootstrap-at-the-stop records: their structure, their verdict and their document."""

import copy
import gzip
import json

import numpy as np
import pytest

from benchmarks import check_bootstrap_stop as checker
from benchmarks import run_bootstrap_stop as producer
from benchmarks.run_interval_coverage import build_system, derived_seed
from benchmarks.run_target_acquisition import load_declaration, schedule
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.reproducibility import compare_json_records
from clifford_qc.subspace.acquisition import acquire_ritz


@pytest.fixture(scope="module")
def declaration():
    return load_declaration()


@pytest.fixture(scope="module")
def fresh():
    return checker.load(producer.RESULTS)


@pytest.fixture(scope="module")
def screen():
    return checker.load(producer.SCREEN)


@pytest.fixture(scope="module")
def live_row(declaration):
    """Replica 0 of the first bank, uniform allocation, regenerated here."""
    return producer.fresh_row(declaration, 0, 0, 0)


def test_the_committed_records_pass_every_structural_check(fresh, screen, declaration):
    assert checker.fresh_problems(fresh, declaration) == []
    assert checker.screen_problems(screen, declaration) == []


def test_the_criteria_are_the_bytes_fixed_before_the_fresh_run(fresh):
    assert producer.sha256(producer.CRITERIA) == checker.CRITERIA_SHA256
    assert fresh["criteria_sha256"] == checker.CRITERIA_SHA256


def test_the_study_document_embeds_the_report_and_its_verdict(fresh, screen, declaration):
    document = checker.DOCUMENT.read_text()
    assert checker.report(fresh, screen, declaration) in document
    verdict = "PASS" if checker.criterion(checker.fresh_cells(fresh, declaration)) else "FAIL"
    assert checker.stated_verdict(document) == verdict == "PASS"


# --------------------------------------------------------------- damaged records

def first_row(record):
    return record["rows"][0]


@pytest.mark.parametrize("damage, message", [
    (lambda row: row.update(seed=row["seed"] + 1), "wrong seed"),
    (lambda row: row.update(target=row["target"] + 1e-6), "wrong seed, exact rank or target"),
    (lambda row: row["A"]["0.02"].update(robust=row["A"]["0.02"]["robust"] * 1.01),
     "spread is not half its q84 - q16"),
    (lambda row: row["A"]["0.02"]["quantiles"].pop(), "spread is not half its q84 - q16"),
    (lambda row: row["A"]["0.02"].update(sigma=1.0), "stopped above the target"),
    (lambda row: row["A"]["0.02"].update(shots=row["A"]["0.02"]["shots"] + 1),
     "schedule's shots at its look"),
    (lambda row: row["A"]["0.02"].update(look=99), "has no look in the schedule"),
    (lambda row: row["B"]["0.01"].update(look=row["A"]["0.01"]["look"] - 1),
     "stopped before arm A"),
    (lambda row: row["A"]["0.01"].update(stopped="schedule_exhausted"),
     "exhausted the schedule before its last look"),
])
def test_a_damaged_fresh_row_is_reported(fresh, declaration, damage, message):
    record = copy.deepcopy(fresh)
    damage(first_row(record))
    assert any(message in problem for problem in checker.fresh_problems(record, declaration))


def test_a_missing_or_repeated_fresh_row_is_reported(fresh, declaration):
    record = copy.deepcopy(fresh)
    record["rows"].append(record["rows"].pop(0))
    assert checker.fresh_problems(record, declaration) == []
    repeated = copy.deepcopy(fresh)
    repeated["rows"][0] = copy.deepcopy(repeated["rows"][1])
    problems = checker.fresh_problems(repeated, declaration)
    assert any("repeated" in p for p in problems) and any("missing" in p for p in problems)


@pytest.mark.parametrize("damage, message", [
    (lambda row: row.update(estimate=row["estimate"] + 1e-9), "differs from the declared record"),
    (lambda row: row.update(sigma=row["sigma"] * 1.001), "differs from the declared record"),
    (lambda row: row.update(stopped="schedule_exhausted"), "differs from the declared record"),
    (lambda row: row.update(error=row["error"] + 1e-6), "wrong error or exact rank"),
    (lambda row: row["quantiles"].reverse(), "does not keep its quantiles"),
    (lambda row: row.update(draws=499), "does not keep its quantiles"),
])
def test_a_damaged_screen_row_is_reported(screen, declaration, damage, message):
    record = copy.deepcopy(screen)
    damage(next(row for row in record["rows"] if row["outcome"] == "ok"))
    assert any(message in problem for problem in checker.screen_problems(record, declaration))


def test_an_empty_screen_is_not_the_declared_screen(screen, declaration):
    record = {**copy.deepcopy(screen), "replicas": 0, "rows": []}
    problems = checker.screen_problems(record, declaration)
    assert "the screen is not the declared replicas and draws" in problems
    assert sum("missing" in problem for problem in problems) == len(screen["rows"])


def test_a_cell_resolved_below_the_criterion_fails_it(fresh, declaration):
    cells = checker.fresh_cells(fresh, declaration)
    assert checker.criterion(cells)
    damaged = copy.deepcopy(cells)
    key = next(key for key in damaged if key[0] == "tfim4_orthogonal")
    damaged[key]["A"] = [True] * 180 + [False] * 20
    assert not checker.criterion(damaged)
    # On the held-out bank the observed rate itself must reach 0.95.
    six = next(key for key in cells if key[0] == "tfim6_mixed")
    damaged = copy.deepcopy(cells)
    damaged[six]["A"] = [True] * 189 + [False] * 11
    assert checker.wilson(189, 200)[1] >= 0.95 and not checker.criterion(damaged)


def test_a_failed_arm_is_a_valid_row_and_counts_as_a_miss(fresh, declaration):
    record = copy.deepcopy(fresh)
    row = first_row(record)
    arm = row["A"]["0.01"]
    failure = {"look": arm["look"], "shots": arm["shots"], "failed": "LinAlgError: lost rank"}
    row["A"]["0.01"] = row["B"]["0.01"] = failure
    assert checker.fresh_problems(record, declaration) == []
    key = (row["system"], row["allocation"], 0.01)
    before = checker.fresh_cells(fresh, declaration)[key]
    after = checker.fresh_cells(record, declaration)[key]
    assert sum(after["A"]) == sum(before["A"]) - 1 and len(after["A"]) == len(before["A"])
    assert sum(after["delta"]) == sum(before["delta"]) - 1
    assert "1 stops with a solve failure" in checker.fresh_report(record, declaration)


# ------------------------------------------------------ producer and acquire_ritz

def test_the_producer_is_acquire_ritz_with_a_bootstrap_at_the_stop(live_row, declaration):
    """In this environment, arm A is the stop and the interval ``acquire_ritz`` reports."""
    shared, _ = build_system(declaration["systems"][0])
    pilot, budgets = schedule(declaration, len(shared.groups))
    z = producer.schedule_z(declaration)
    for t, goal in enumerate(declaration["target_std_errors"]):
        arm = live_row["A"][repr(goal)]
        found = acquire_ritz(
            shared, FiniteShotBackend(seed=live_row["seed"]), pilot_shots_per_group=pilot,
            round_budgets=budgets, allocation=declaration["allocations"][0],
            target_std_error=goal, solver_kwargs=dict(declaration["solver_kwargs"]),
            delta=declaration["delta"], bootstrap_replicates=producer.BOOTSTRAP_REPLICATES,
            bootstrap_seed=derived_seed(producer.FRESH_ROOT, producer.BOOTSTRAP_STREAM, 0, t,
                                        arm["look"]))
        assert len(found.looks) - 1 == arm["look"] and found.shots == arm["shots"]
        assert found.energy == arm["estimate"] and found.std_error == arm["sigma"]
        assert found.looks[-1]["bootstrap_spread"] == pytest.approx(arm["robust"], rel=1e-12)
        assert found.bootstrap_interval.upper - found.energy == pytest.approx(
            z * arm["robust"], rel=1e-12)


def test_the_producer_reproduces_the_committed_row(fresh, live_row):
    """Only under the NumPy and SciPy that drew the record are its values comparable."""
    blockers = checker.replay_blockers(fresh)
    if blockers:
        pytest.skip("; ".join(blockers))
    stored = next(row for row in fresh["rows"]
                  if (row["system"], row["allocation"], row["replica"])
                  == (live_row["system"], live_row["allocation"], 0))
    assert compare_json_records(stored, json.loads(json.dumps(live_row)), "row",
                                atol=checker.REPLAY_TOLERANCE,
                                rtol=checker.REPLAY_TOLERANCE) == []


def test_replay_is_skipped_under_other_dependency_versions(fresh, monkeypatch, capsys,
                                                           tmp_path):
    other = copy.deepcopy(fresh)
    other["provenance"]["dependencies"]["numpy"] = "0.0.1"
    assert checker.replay_blockers(other)
    path = tmp_path / "other.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(other, handle)
    monkeypatch.setattr(checker, "replay_problems",
                        lambda *args: pytest.fail("a blocked replay must not run"))
    assert checker.main(["--results", str(path), "--replay", "1"]) == 0
    assert "NOTE replay unavailable: numpy" in capsys.readouterr().out
    assert checker.main(["--results", str(path), "--replay", "1", "--require-replay"]) == 1


def fake_bootstrap(spread):
    calls = []

    def draws(shared, cache, *, result, replicates, seed):
        calls.append(seed)
        return result.energies[0] + spread * np.linspace(-1.0, 1.0, replicates)
    return draws, calls


def test_one_bootstrap_per_look_and_target_however_many_arms_read_it(declaration, monkeypatch):
    # A spread above every target: arm B never confirms and exhausts the schedule,
    # so its last look is read by the stop and again by the exhaustion.
    draws, calls = fake_bootstrap(10.0)
    monkeypatch.setattr(producer, "bootstrap_ritz_samples", draws)
    row = producer.fresh_row(declaration, 0, 0, 0, replicates=20)
    assert len(calls) == len(set(calls))
    for goal in declaration["target_std_errors"]:
        assert row["B"][repr(goal)]["stopped"] == "schedule_exhausted"
    _, _, cumulative = checker.system_shape(declaration, row["system"])
    assert checker._arm_problems(row, declaration, "row", cumulative) == []


def test_a_failed_bootstrap_is_recorded_on_its_arms(declaration, monkeypatch):
    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError("replica lost its rank")

    monkeypatch.setattr(producer, "bootstrap_ritz_samples", failing)
    row = producer.fresh_row(declaration, 0, 0, 0, replicates=20)
    for goal in declaration["target_std_errors"]:
        a, b = row["A"][repr(goal)], row["B"][repr(goal)]
        assert a["boot_failed"] == "LinAlgError: replica lost its rank" and "robust" not in a
        assert b["stopped"] == "schedule_exhausted" and "boot_failed" in b
    _, _, cumulative = checker.system_shape(declaration, row["system"])
    assert checker._arm_problems(row, declaration, "row", cumulative) == []
    record = {"rows": [row]}
    assert not any(checker.fresh_cells(record, declaration)[key]["A"][0]
                   for key in checker.fresh_cells(record, declaration))


def test_a_later_solve_failure_keeps_the_arms_that_had_already_stopped(declaration, live_row,
                                                                        monkeypatch):
    draws, _ = fake_bootstrap(1e-6)
    monkeypatch.setattr(producer, "bootstrap_ritz_samples", draws)
    stops = {goal: live_row["A"][repr(goal)]["look"] for goal in declaration["target_std_errors"]}
    early, late = min(stops.values()), max(stops.values())
    assert early < late
    # build_system caches its systems, so this is the instance fresh_row uses.
    # Patch the instance: another test's monkeypatch may already have left an
    # instance-level solve there, which a class-level patch would not reach.
    shared = build_system(declaration["systems"][0])[0]
    solve = shared.solve
    seen = {"looks": 0}

    def failing(cache, **kwargs):
        if seen["looks"] == late:
            raise np.linalg.LinAlgError("no retained mode")
        seen["looks"] += 1
        return solve(cache, **kwargs)

    monkeypatch.setattr(shared, "solve", failing)
    row = producer.fresh_row(declaration, 0, 0, 0, replicates=20)
    for goal, look in stops.items():
        a = row["A"][repr(goal)]
        if look == early:
            assert a["stopped"] == "target_met" and a["look"] == early
        else:
            assert a == row["B"][repr(goal)] == {
                "look": late, "shots": a["shots"], "failed": "LinAlgError: no retained mode"}
    _, _, cumulative = checker.system_shape(declaration, row["system"])
    assert checker._arm_problems(row, declaration, "row", cumulative) == []


@pytest.mark.parametrize("mode", ["fresh", "screen"])
def test_a_record_the_producer_writes_passes_the_checker(fresh, screen, declaration, mode):
    committed = fresh if mode == "fresh" else screen
    record = producer.build_record(mode, committed["rows"], declaration=declaration,
                                   replicas=producer.REPLICAS,
                                   replicates=producer.BOOTSTRAP_REPLICATES,
                                   stamp=producer.provenance(0.0), seconds=1.0)
    problems = (checker.fresh_problems if mode == "fresh" else checker.screen_problems)(
        record, declaration)
    assert problems == []
    assert ("criteria_sha256" in record) == (mode == "fresh")
    assert ("quantile_levels" in record) == (mode == "screen")


@pytest.mark.parametrize("target", ["results", "screen"])
def test_the_producer_refuses_to_overwrite_a_committed_record(target):
    path = producer.RESULTS if target == "results" else producer.SCREEN
    with pytest.raises(SystemExit):
        producer.main(["--output", str(path), "--replicas", "1"])
