"""The bootstrap-at-the-stop records: their structure, their verdict and their document."""

import copy
import json

import pytest

from benchmarks import check_bootstrap_stop as checker
from benchmarks import run_bootstrap_stop as producer
from benchmarks.run_interval_coverage import build_system, derived_seed
from benchmarks.run_target_acquisition import load_declaration, schedule
from clifford_qc.backends.finite_shot import FiniteShotBackend
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


def first_row(record):
    return next(row for row in record["rows"] if "failed" not in row)


@pytest.mark.parametrize("damage, message", [
    (lambda row: row.update(seed=row["seed"] + 1), "wrong seed"),
    (lambda row: row["A"]["0.02"].update(robust=row["A"]["0.02"]["robust"] * 1.01),
     "spread is not its quantiles'"),
    (lambda row: row["A"]["0.02"].update(sigma=1.0), "stopped above the target"),
    (lambda row: row["B"]["0.01"].update(look=-1), "stopped before arm A"),
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


def test_a_screen_row_that_is_not_the_declared_run_is_reported(screen, declaration):
    record = copy.deepcopy(screen)
    row = next(row for row in record["rows"] if row["outcome"] == "ok")
    row["estimate"] += 1e-9
    assert any("differs from the declared record" in problem
               for problem in checker.screen_problems(record, declaration))


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


def test_the_producer_and_acquire_ritz_reproduce_a_committed_row(fresh, declaration):
    spec = declaration["systems"][0]
    stored = next(row for row in fresh["rows"]
                  if (row["system"], row["allocation"], row["replica"])
                  == (spec["key"], declaration["allocations"][0], 0))
    row = json.loads(json.dumps(producer.fresh_row(declaration, 0, 0, 0)))
    assert row == stored
    shared, _ = build_system(spec)
    pilot, budgets = schedule(declaration, len(shared.groups))
    for t, goal in enumerate(declaration["target_std_errors"]):
        arm = stored["A"][repr(goal)]
        found = acquire_ritz(
            shared, FiniteShotBackend(seed=stored["seed"]), pilot_shots_per_group=pilot,
            round_budgets=budgets, allocation=declaration["allocations"][0],
            target_std_error=goal, solver_kwargs=dict(declaration["solver_kwargs"]),
            delta=declaration["delta"], bootstrap_replicates=fresh["bootstrap_replicates"],
            bootstrap_seed=derived_seed(fresh["seed_root"], producer.BOOTSTRAP_STREAM, 0, t,
                                        arm["look"]))
        assert found.looks[-1]["bootstrap_spread"] == arm["robust"]
        assert found.energy == arm["estimate"] and len(found.looks) - 1 == arm["look"]


@pytest.mark.parametrize("target", ["results", "screen"])
def test_the_producer_refuses_to_overwrite_a_committed_record(target):
    path = producer.RESULTS if target == "results" else producer.SCREEN
    with pytest.raises(SystemExit):
        producer.main(["--output", str(path), "--replicas", "1"])
