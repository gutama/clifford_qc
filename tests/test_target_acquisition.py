"""The declared target-acquisition study: its declaration, producer and checker.

The library module is tested in ``test_acquisition.py``. These tests are about
the study built on it: that both allocations are compared on the same pilot,
that cost and failures are counted honestly, and that the checker notices a
record which no longer follows from its declaration or its schedule.
"""

import copy
import hashlib
import json
import types

import numpy as np
import pytest

from benchmarks import check_target_acquisition as checker
from benchmarks import run_target_acquisition as producer
from benchmarks.run_interval_coverage import write_record
from clifford_qc.subspace.acquisition import AcquisitionError

ORTHOGONAL = {"key": "tfim4_orthogonal", "role": "development",
              "model": {"family": "tfim", "n": 4, "J": 1.0, "h": 1.0},
              "pool_words": 3, "krylov_powers": 0}
CUMULATIVE = [50, 100, 200, 400]
GOALS = [0.3, 0.15]
REPLICAS = 6
GROUPS = 8


def content():
    return {
        "schema": producer.DECLARATION_SCHEMA, "declared": "test", "question": "test",
        "target": "exact_subspace_ground_energy", "seed_root": 13, "replicas": REPLICAS,
        "delta": 0.05, "summary_confidence": 0.95, "solver_kwargs": {},
        "allocations": ["uniform", "ritz_targeted"],
        "cumulative_shots_per_group": CUMULATIVE, "target_std_errors": GOALS,
        "systems": [copy.deepcopy(ORTHOGONAL)], "comparison": {"bootstrap_replicates": 50},
        "spot_replay_replicas": 1,
    }


@pytest.fixture
def declaration(tmp_path):
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(content()))
    return path


@pytest.fixture(scope="module")
def rows():
    return producer.run(content())


@pytest.fixture
def record(declaration, rows):
    built = producer.build_record(declaration, copy.deepcopy(rows),
                                  declaration_commit="0" * 40)
    return json.loads(json.dumps(built, allow_nan=False))


# ------------------------------------------------------- declaration, producer

def test_the_committed_declaration_is_valid_and_complete():
    declaration = producer.load_declaration()
    roles = [system["role"] for system in declaration["systems"]]
    assert roles == ["development", "development", "held_out", "held_out"]
    for system in declaration["systems"]:
        _, design = producer.build_system(system)
        assert design["exact_rank"] >= 1 and np.isfinite(design["target"])
    cumulative = declaration["cumulative_shots_per_group"]
    assert len(cumulative) == 19 and (cumulative[0], cumulative[-1]) == (100, 51200)
    # Each look is a factor near sqrt(2) above the last.
    assert all(1.40 < b / a < 1.43 for a, b in zip(cumulative, cumulative[1:]))
    pilot, budgets = producer.schedule(declaration, 52)
    assert pilot == 100 and len(budgets) == 18
    assert 52 * pilot + sum(budgets) == 52 * 51200
    assert declaration["target_std_errors"] == [0.02, 0.01]
    assert declaration["solver_kwargs"] == {} and declaration["replicas"] == 400
    assert len(producer.expected_tasks(declaration)) == 4 * 400


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d.update(schema="other"), "schema"),
    (lambda d: d.update(target="true_ground_energy"), "target"),
    (lambda d: d["systems"][0]["model"].update(family="hubbard"), "model family"),
    (lambda d: d["systems"][0].update(role="test"), "development or held_out"),
    (lambda d: d["systems"].append(dict(d["systems"][0])), "distinct"),
    (lambda d: d.update(allocations=["ritz_targeted", "uniform"]), "allocations"),
    (lambda d: d.update(cumulative_shots_per_group=[100]), "increasing"),
    (lambda d: d.update(cumulative_shots_per_group=[100, 100]), "increasing"),
    (lambda d: d.update(cumulative_shots_per_group=[100, 200.0]), "increasing"),
    (lambda d: d.update(target_std_errors=[]), "target_std_errors"),
    (lambda d: d.update(target_std_errors=[0.1, 0.1]), "target_std_errors"),
    (lambda d: d.update(target_std_errors=[1]), "target_std_errors"),
    (lambda d: d.update(solver_kwargs={"overlap_repair": "nearest_correlation"}),
     "delta method"),
    (lambda d: d.update(replicas=1), "replicas"),
    (lambda d: d.update(delta=5), "delta"),
    (lambda d: d["comparison"].update(bootstrap_replicates=1), "bootstrap_replicates"),
    (lambda d: d.update(spot_replay_replicas=REPLICAS + 1), "spot_replay_replicas"),
])
def test_invalid_declarations_are_rejected(declaration, mutation, message):
    changed = content()
    mutation(changed)
    declaration.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=message):
        producer.load_declaration(declaration)


def test_seeds_are_distinct_across_systems_replicas_and_bootstraps():
    declared = content()
    seeds = {producer.measurement_seed(declared, system, replica)
             for system in range(3) for replica in range(5)}
    seeds |= {producer.comparison_seed(declared, system, slot)
              for system in range(3) for slot in range(3)}
    assert len(seeds) == 15 + 9


def test_the_schedule_turns_cumulative_shots_into_round_budgets():
    assert producer.schedule(content(), GROUPS) == (50, [400, 800, 1600])
    assert producer.target_key(0.02) == "0.02"


def test_the_run_is_complete_paired_and_independent_of_worker_count(rows):
    assert len(rows) == 2 * REPLICAS
    by_key = {(row["allocation"], row["replica"]): row for row in rows}
    for replica in range(REPLICAS):
        uniform, targeted = by_key[("uniform", replica)], by_key[("ritz_targeted", replica)]
        assert uniform["measurement_seed"] == targeted["measurement_seed"]
        # Same seed, same pilot: the first look is identical, later ones are not.
        assert uniform["fixed"]["ledger"]["energies"][0] == \
            targeted["fixed"]["ledger"]["energies"][0]
        assert uniform["fixed"]["ledger"]["energies"][1] != \
            targeted["fixed"]["ledger"]["energies"][1]
    assert len({row["measurement_seed"] for row in rows}) == REPLICAS
    assert producer.run(content(), jobs=2) == rows


def test_every_run_spends_its_schedule_and_a_stopped_run_is_a_prefix(rows):
    stopped_early = 0
    for row in rows:
        fixed = row["fixed"]
        assert fixed["outcome"] == "ok" and fixed["stopped"] == "schedule_exhausted"
        assert fixed["shots"] == CUMULATIVE[-1] * GROUPS and fixed["looks_taken"] == 4
        assert fixed["interval"] == fixed["naive_interval"]
        assert set(row["stopping"]) == {"0.3", "0.15"}
        for goal in GOALS:
            run = row["stopping"][producer.target_key(goal)]
            taken = run["looks_taken"]
            assert "ledger" not in run
            assert run["shots"] == CUMULATIVE[taken - 1] * GROUPS
            assert run["estimate"] == fixed["ledger"]["energies"][taken - 1]
            assert run["std_error"] == fixed["ledger"]["std_errors"][taken - 1]
            width = run["interval"]["upper"] - run["interval"]["lower"]
            naive = run["naive_interval"]["upper"] - run["naive_interval"]["lower"]
            assert width > naive
            stopped_early += taken < 4
    assert stopped_early > 0


def test_summaries_count_every_replica_and_carry_the_declared_labels(record):
    [system] = record["summaries"]
    assert system["system"] == "tfim4_orthogonal" and system["role"] == "development"
    for name in ("uniform", "ritz_targeted"):
        found = system["allocations"][name]
        fixed = found["fixed"]
        assert fixed["replicas"] == REPLICAS and fixed["solve_failures"] == 0
        assert [look["shots"] for look in fixed["looks"]] == [GROUPS * c for c in CUMULATIVE]
        assert all(look["errors"]["n"] == REPLICAS for look in fixed["looks"])
        assert fixed["final_coverage"]["n"] == REPLICAS
        assert fixed["final_coverage"]["nominal"] == 0.95
        for key in ("0.3", "0.15"):
            stopping = found["stopping"][key]
            assert sum(stopping["stopped"].values()) == REPLICAS
            assert stopping["mean_shots"] == stopping["mean_shots_per_group"] * GROUPS
            assert CUMULATIVE[0] * GROUPS <= stopping["mean_shots"] <= CUMULATIVE[-1] * GROUPS
            assert stopping["coverage"]["schedule_valid"]["covered"] >= \
                stopping["coverage"]["naive"]["covered"]
    # A tighter target costs at least as many shots.
    for name in ("uniform", "ritz_targeted"):
        stopping = system["allocations"][name]["stopping"]
        assert stopping["0.15"]["mean_shots"] >= stopping["0.3"]["mean_shots"]
    comparison = system["comparison"]
    assert comparison["matched_budget"]["shots"] == CUMULATIVE[-1] * GROUPS
    assert comparison["matched_budget"]["label"] in producer.EFFICIENCY_LABELS
    for key in ("0.3", "0.15"):
        found = comparison["stopping"][key]
        assert found["verdict"] in producer.VERDICTS
        assert found["shots"]["pairs"] == found["matched_error"]["pairs"] == REPLICAS
    headline = record["headline"]
    assert sum(headline["matched_budget"].values()) == 1
    assert all(sum(tally.values()) == 1 for tally in headline["stopping"].values())
    assert sum(headline["coverage"]["fixed_final"].values()) == 2
    assert sum(headline["coverage"]["schedule_valid"].values()) == 4
    assert sum(headline["coverage"]["naive"].values()) == 4
    assert headline["solve_failures"] == 0


def test_the_matched_error_figure_is_the_work_normalized_mse_ratio(record, rows):
    [system] = record["summaries"]
    target = record["systems"]["tfim4_orthogonal"]["target"]
    key = "0.15"
    work = {}
    for name in ("uniform", "ritz_targeted"):
        runs = [row["stopping"][key] for row in rows if row["allocation"] == name]
        shots = np.mean([run["shots"] for run in runs])
        mse = np.mean([(run["estimate"] - target) ** 2 for run in runs])
        work[name] = shots * mse
        errors = system["allocations"][name]["stopping"][key]["errors"]
        assert errors["rmse"] == pytest.approx(np.sqrt(mse))
    found = system["comparison"]["stopping"][key]
    assert found["matched_error"]["ratio"] == pytest.approx(
        work["uniform"] / work["ritz_targeted"])
    uniform, targeted = (system["allocations"][name]["stopping"][key]["mean_shots"]
                         for name in ("uniform", "ritz_targeted"))
    assert found["shots"]["ratio"] == pytest.approx(uniform / targeted)


def test_ratio_intervals_are_paired_seeded_and_undefined_when_they_should_be():
    numerator, denominator = np.array([4.0, 6.0, 8.0, 6.0]), np.array([2.0, 3.0, 4.0, 3.0])
    settings = {"seed": 5, "replicates": 200, "confidence": 0.95}
    exact = producer._ratio_interval([numerator], [denominator], **settings)
    # Perfectly paired: every resample has the same ratio.
    assert exact == {"pairs": 4, "ratio": 2.0, "interval": [2.0, 2.0]}
    noisy = np.array([4.0, 7.0, 7.0, 6.0])
    first = producer._ratio_interval([noisy], [denominator], **settings)
    assert first == producer._ratio_interval([noisy], [denominator], **settings)
    assert first["interval"][0] < first["ratio"] == 2.0 < first["interval"][1]
    assert first != producer._ratio_interval([noisy], [denominator], **{**settings, "seed": 6})
    product = producer._ratio_interval([noisy, numerator], [denominator, denominator],
                                       **settings)
    assert product["ratio"] == pytest.approx(
        noisy.mean() * numerator.mean() / denominator.mean() ** 2)
    undefined = {"pairs": 4, "ratio": None, "interval": None}
    assert producer._ratio_interval([numerator], [np.zeros(4)], **settings) == undefined
    assert producer._ratio_interval([numerator[:1]], [denominator[:1]], **settings) == {
        "pairs": 1, "ratio": None, "interval": None}
    # A resample can hit a zero denominator even when the sample does not.
    sparse = producer._ratio_interval([numerator], [np.array([0.0, 0.0, 0.0, 1.0])], **settings)
    assert sparse["ratio"] == 24.0 and sparse["interval"] is None


@pytest.mark.parametrize("interval, label", [
    ([1.2, 3.0], "targeted_better"), ([0.2, 0.9], "uniform_better"),
    ([0.9, 1.1], "indistinguishable"), ([1.0, 1.0], "indistinguishable"),
    (None, "indistinguishable"),
])
def test_a_ratio_is_labelled_only_when_its_interval_excludes_one(interval, label):
    assert producer._direction({"interval": interval}) == label


def test_a_failed_look_is_an_outcome_that_keeps_its_cost(declaration, monkeypatch):
    real = producer.acquire_ritz
    loaded = producer.load_declaration(declaration)
    pilot = producer.replica_rows(loaded, 0, 2)[0]["fixed"]["ledger"]["energies"][0]

    def failing(shared, backend, **kwargs):
        found = real(shared, backend, **kwargs)
        if kwargs["allocation"] == "ritz_targeted" and found.looks[0]["energy"] == pilot \
                and len(found.looks) > 2:
            raise AcquisitionError(
                "look 2 could not be solved",
                cache=types.SimpleNamespace(total_shots=found.looks[2]["shots"]),
                looks=found.looks[:2])
        return found

    monkeypatch.setattr(producer, "acquire_ritz", failing)
    rows = producer.run(loaded)
    failed = next(row for row in rows
                  if row["allocation"] == "ritz_targeted" and row["replica"] == 2)
    fixed = failed["fixed"]
    assert fixed["outcome"] == "solve_failed" and fixed["estimate"] is None
    assert fixed["solve_error"] == "look 2 could not be solved"
    assert fixed["shots"] == CUMULATIVE[2] * GROUPS and fixed["looks_taken"] == 2
    assert len(fixed["ledger"]["energies"]) == 2

    record = json.loads(json.dumps(producer.build_record(
        declaration, rows, declaration_commit="0" * 40), allow_nan=False))
    assert checker.structural_problems(record, declaration) == []
    [system] = record["summaries"]
    summary = system["allocations"]["ritz_targeted"]["fixed"]
    assert summary["solve_failures"] == 1
    assert summary["looks"][1]["errors"]["n"] == REPLICAS
    assert summary["looks"][2]["errors"]["n"] == REPLICAS - 1
    # The failure stays in the coverage denominator as a miss.
    assert summary["final_coverage"]["n"] == REPLICAS
    assert summary["final_coverage"]["covered"] <= REPLICAS - 1
    assert system["comparison"]["matched_budget"]["pairs"] == REPLICAS - 1
    assert record["headline"]["solve_failures"] >= 1
    for key in ("0.3", "0.15"):
        run = failed["stopping"][key]
        stopping = system["allocations"]["ritz_targeted"]["stopping"][key]
        assert stopping["solve_failures"] == (run["outcome"] == "solve_failed")
        # Cost counts every replica; the matched-error figure only resolved pairs.
        comparison = system["comparison"]["stopping"][key]
        assert comparison["shots"]["pairs"] == REPLICAS
        assert comparison["matched_error"]["pairs"] == REPLICAS - stopping["solve_failures"]


def test_the_record_is_json_safe_and_written_deterministically(record, tmp_path):
    first, second = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    write_record(record, first)
    write_record(record, second)
    assert first.read_bytes() == second.read_bytes()
    assert checker.read_record(first) == record
    assert record["implementation"].keys() == set(producer.IMPLEMENTATION)
    assert all((producer.ROOT / relative).is_file() for relative in producer.IMPLEMENTATION)


# -------------------------------------------------------------------- checker

def test_a_faithful_record_passes_structure_and_replay(declaration, record):
    assert checker.structural_problems(record, declaration) == []
    problems, compared = checker.replay_problems(record, replicas=1)
    assert problems == [] and compared == 2
    problems, compared = checker.replay_problems(record, replicas=None)
    assert problems == [] and compared == len(record["rows"])


def test_summary_roundoff_is_allowed_but_material_changes_are_rejected(declaration, record):
    interval = record["summaries"][0]["comparison"]["matched_budget"]["interval"]
    # Supported NumPy versions differ in the last bits of this recomputation.
    interval[1] *= 1.0 + 5e-12
    assert checker.structural_problems(record, declaration) == []
    interval[1] += 1e-6
    assert any("comparison.matched_budget.interval[1]" in problem
               for problem in checker.structural_problems(record, declaration))


def _stopped_early(record):
    """A stopped run that ended before the last look, and its row."""
    for row in record["rows"]:
        for run in row["stopping"].values():
            if 1 < run["looks_taken"] < len(CUMULATIVE):
                return row, run
    raise AssertionError("no run stopped between the pilot and the last look")


def _edit_stopped(change):
    def mutation(record):
        change(*_stopped_early(record))
    return mutation


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r["rows"].pop(), "missing"),
    (lambda r: r["rows"].append(copy.deepcopy(r["rows"][0])), "duplicate"),
    (lambda r: r["rows"][0].update(replica=99), "undeclared"),
    (lambda r: r["rows"][0].update(measurement_seed=1), "declared seed"),
    (lambda r: r["declaration"].update(seed_root=12), "embed"),
    (lambda r: r.update(declaration_sha256="0" * 64), "digest"),
    (lambda r: r["systems"]["tfim4_orthogonal"].update(target=0.0), "systems"),
    (lambda r: r["headline"]["coverage"]["naive"].update(under=99), "headline"),
    (lambda r: r["summaries"][0]["comparison"]["matched_budget"].update(label="x"),
     "summaries"),
    (lambda r: r.update(schema="other"), "schema"),
    (lambda r: r["rows"][0]["fixed"].update(shots=1), "the schedule has"),
    (lambda r: r["rows"][0]["fixed"].update(looks_taken=9), "undeclared number of looks"),
    (lambda r: r["rows"][0]["fixed"].update(outcome="maybe"), "unknown outcome"),
    (lambda r: r["rows"][0]["fixed"].update(stopped="target_met"), "target"),
    (lambda r: r["rows"][0]["fixed"]["ledger"]["energies"].pop(), "one entry per look"),
    (lambda r: r["rows"][0]["fixed"].update(estimate=r["rows"][0]["fixed"]["estimate"] + 1e-6),
     "ledger's last look"),
    (lambda r: r["rows"][0]["fixed"]["interval"].update(
        covered=not r["rows"][0]["fixed"]["interval"]["covered"]), "coverage flag"),
    (lambda r: r["rows"][0]["fixed"]["interval"].update(lower=1e6), "contain its estimate"),
    (lambda r: r["rows"][0]["stopping"].pop("0.3"), "one stopped run per declared target"),
    (_edit_stopped(lambda row, run: run.update(estimate=run["estimate"] + 1e-6)), "prefix"),
    (_edit_stopped(lambda row, run: run.update(stopped="schedule_exhausted")),
     "before its last look"),
    (_edit_stopped(lambda row, run: run["interval"].update(
        lower=run["naive_interval"]["lower"], upper=run["naive_interval"]["upper"])),
     "not widened for 4 look"),
    (_edit_stopped(lambda row, run: run.update(std_error=1.0)),
     "standard error does not meet"),
    (_edit_stopped(lambda row, run: row["fixed"]["ledger"]["std_errors"].__setitem__(0, 1e-9)),
     "ran past the first look"),
])
def test_structural_tampering_is_detected(declaration, record, mutation, message):
    mutation(record)
    problems = checker.structural_problems(record, declaration)
    assert any(message in problem for problem in problems), problems


def test_a_run_recorded_as_exhausted_must_not_have_met_its_target(declaration, record):
    row = next(row for row in record["rows"]
               if row["stopping"]["0.15"]["looks_taken"] == len(CUMULATIVE))
    run = row["stopping"]["0.15"]
    run["stopped"] = "target_met" if run["stopped"] == "schedule_exhausted" \
        else "schedule_exhausted"
    assert any("target" in problem
               for problem in checker.structural_problems(record, declaration))


def test_a_failure_must_be_consistent_with_the_full_schedule(declaration, record):
    run = record["rows"][0]["stopping"]["0.3"]
    run.update({"outcome": "solve_failed", "solve_error": "invented", "looks_taken": 1,
                "shots": CUMULATIVE[1] * GROUPS, "estimate": None, "std_error": None,
                "stopped": None, "interval": None, "naive_interval": None})
    assert any("failed where the full schedule did not" in problem
               for problem in checker.structural_problems(record, declaration))
    run["shots"] = 7
    assert any("shots the schedule does not contain" in problem
               for problem in checker.structural_problems(record, declaration))


def test_replay_detects_a_changed_estimate_but_not_rounding(record):
    record["rows"][0]["fixed"]["estimate"] += 1e-9
    assert checker.replay_problems(record, replicas=1)[0] == []
    record["rows"][0]["fixed"]["estimate"] += 1e-3
    assert any("estimate" in problem for problem in checker.replay_problems(record, replicas=1)[0])


def test_an_ill_conditioned_look_limits_the_replay_to_what_preceded_it(record):
    row = record["rows"][1]
    assert row["allocation"] == "ritz_targeted"
    ledger = row["fixed"]["ledger"]
    assert checker.comparable_looks(row) == len(CUMULATIVE)
    ledger["smallest_retained_overlaps"][2] = 1e-9
    assert checker.comparable_looks(row) == 2
    # From the ill-conditioned look on, floats are not comparable across machines.
    ledger["energies"][2] += 1.0
    ledger["energies"][3] += 1.0
    row["fixed"]["estimate"] += 1.0
    for run in row["stopping"].values():
        if run["looks_taken"] > 2:
            run["estimate"] += 1.0
    assert checker.replay_problems(record, replicas=1)[0] == []
    # Its retained rank is discrete and still has to replay.
    ledger["retained_ranks"][2] += 1
    assert any("retained_ranks" in problem
               for problem in checker.replay_problems(record, replicas=1)[0])
    ledger["retained_ranks"][2] -= 1
    # Everything before it is compared as usual.
    ledger["energies"][1] += 1e-3
    assert any("energies" in problem
               for problem in checker.replay_problems(record, replicas=1)[0])


def test_a_stopped_run_that_ended_before_the_ill_conditioned_look_still_replays(record):
    row = next(row for row in record["rows"] if row["replica"] == 0
               and any(run["looks_taken"] <= 2 for run in row["stopping"].values()))
    row["fixed"]["ledger"]["smallest_retained_overlaps"][2] = 1e-9
    run = next(run for run in row["stopping"].values() if run["looks_taken"] <= 2)
    run["estimate"] += 1e-3
    assert any("estimate" in problem
               for problem in checker.replay_problems(record, replicas=1)[0])


def test_replay_is_blocked_by_an_evolved_source(record):
    assert not any(blocker.startswith("EVOLVED") for blocker in checker.replay_blockers(record))
    record["implementation"]["clifford_qc/subspace/acquisition.py"] = "0" * 64
    assert "EVOLVED clifford_qc/subspace/acquisition.py" in checker.replay_blockers(record)


def test_an_edited_bound_source_replays_only_with_its_packaged_snapshot(record, tmp_path,
                                                                           monkeypatch):
    relative = "clifford_qc/subspace/acquisition.py"
    monkeypatch.setattr(checker, "EXECUTION_SOURCES", tmp_path)
    record["implementation"][relative] = "f" * 64
    assert checker.source_drift(record) == ([], [relative])
    assert f"EVOLVED {relative}" in checker.replay_blockers(record)
    snapshot = tmp_path / record["declaration_commit"] / relative
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"the bytes the record was produced with")
    record["implementation"][relative] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert checker.source_drift(record) == ([relative], [])
    assert not any(blocker.startswith("EVOLVED") for blocker in checker.replay_blockers(record))
    # A snapshot that is not the recorded bytes explains nothing.
    snapshot.write_bytes(b"some other bytes")
    assert checker.source_drift(record) == ([], [relative])
    # Nor does a declaration commit that is not a commit name.
    record["declaration_commit"] = "../" + record["declaration_commit"][3:]
    assert checker.source_drift(record) == ([], [relative])


def test_history_check_reports_an_unknown_declaration_commit(record):
    problems = checker.history_problems(record, producer.DECLARATION, producer.RESULTS)
    assert problems == [f"declaration commit {'0' * 40!r} is not in this repository's history"]


def test_the_report_is_generated_from_the_record(record):
    text = checker.report(record)
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "`tfim4_orthogonal` | development | 0.3 |" in text
    assert "Stopped at 0.15, schedule-valid" in text
    assert "nominal 95% level" in text
    assert "Solve failures over all runs: 0." in text
    for verdict in producer.VERDICTS:
        assert f"`{verdict}`" in text
    # One row per look in the per-system table.
    assert sum(line.startswith("| 50 |") or line.startswith("| 100 |")
               or line.startswith("| 200 |") or line.startswith("| 400 |")
               for line in text.splitlines()) == len(CUMULATIVE)
    changed = copy.deepcopy(record)
    changed["headline"]["solve_failures"] = 3
    assert checker.report(changed) != text


def test_the_checker_command_line_reports_and_fails_closed(declaration, record, tmp_path,
                                                           capsys):
    path = tmp_path / "record.json.gz"
    write_record(record, path)
    arguments = ["--declaration", str(declaration), "--results", str(path)]
    assert checker.main([*arguments, "--no-replay"]) == 0
    assert "replay: not requested" in capsys.readouterr().out
    assert checker.main([*arguments, "--report"]) == 0
    assert capsys.readouterr().out == checker.report(record)
    assert checker.main([*arguments, "--require-replay"]) == 0
    assert "replay: 2 row(s) regenerated" in capsys.readouterr().out
    # A record whose sources have moved on is still structurally checked, and
    # says why it was not replayed; requiring the replay turns that into a failure.
    evolved = copy.deepcopy(record)
    evolved["implementation"]["clifford_qc/subspace/acquisition.py"] = "0" * 64
    stale = tmp_path / "stale.json.gz"
    write_record(evolved, stale)
    stale_arguments = ["--declaration", str(declaration), "--results", str(stale)]
    assert checker.main(stale_arguments) == 0
    output = capsys.readouterr().out
    assert "NOTE replay unavailable: EVOLVED clifford_qc/subspace/acquisition.py" in output
    assert "replay: skipped" in output
    assert checker.main([*stale_arguments, "--require-replay"]) == 1
    assert "replay was required" in capsys.readouterr().out
    record["rows"].pop()
    broken = tmp_path / "broken.json.gz"
    write_record(record, broken)
    assert checker.main(["--declaration", str(declaration), "--results", str(broken),
                         "--no-replay"]) == 1
    assert "target-acquisition problem(s)" in capsys.readouterr().out


# ------------------------------------------------------- the committed record

@pytest.fixture(scope="module")
def committed():
    return checker.read_record(producer.RESULTS)


def test_the_committed_record_matches_its_declaration(committed):
    assert checker.structural_problems(committed, producer.DECLARATION) == []
    declaration = producer.load_declaration()
    expected = len(declaration["systems"]) * declaration["replicas"] * 2
    assert len(committed["rows"]) == expected == 3200
    assert committed["implementation"].keys() == set(producer.IMPLEMENTATION)
    # The held-out systems were sampled adaptively only under the declared root.
    assert committed["declaration"]["seed_root"] == 20261117
    # The record was produced from a clean checkout of the declaration commit.
    provenance = committed["provenance"]
    assert provenance["git_dirty"] is False
    assert provenance["git_sha"] == committed["declaration_commit"]


def test_the_committed_record_reports_what_the_document_says(committed):
    headline = committed["headline"]
    assert headline["solve_failures"] == 0
    # Targeting was never dearer at matched error, and cheaper in five of eight cells.
    verdicts = [found["verdict"] for system in committed["summaries"]
                for found in system["comparison"]["stopping"].values()]
    assert verdicts.count("fewer_shots_at_matched_error") == 5
    assert verdicts.count("no_resolved_difference") == 3
    assert "more_shots_at_matched_error" not in verdicts
    # The stopped interval over-covers on the four-qubit banks and fails on the
    # held-out six-qubit bank.
    labels = {system["system"]: [stopping["coverage"]["schedule_valid"]["label"]
                                 for found in system["allocations"].values()
                                 for stopping in found["stopping"].values()]
              for system in committed["summaries"]}
    assert all(labels[key] == ["over"] * 4
               for key in ("tfim4_orthogonal", "tfim4_mixed", "xxz4_mixed"))
    assert sorted(labels["tfim6_mixed"]) == ["consistent", "under", "under", "under"]
    naive = [stopping["coverage"]["naive"]["label"]
             for system in committed["summaries"] if system["system"] == "tfim6_mixed"
             for found in system["allocations"].values()
             for stopping in found["stopping"].values()]
    assert naive == ["under"] * 4


def test_only_rows_with_an_ill_conditioned_look_are_replayed_in_part(committed):
    partial = [(row["system"], row["allocation"], row["replica"])
               for row in committed["rows"]
               if checker.comparable_looks(row) < len(row["fixed"]["ledger"]["energies"])]
    assert partial == [("tfim6_mixed", "ritz_targeted", 239)]


def test_every_bound_source_edited_since_the_run_is_packaged(committed):
    # An edit to a bound source without its execution-time bytes would turn the
    # gate's replay off without failing it. Packaging the snapshot keeps it on.
    packaged, unexplained = checker.source_drift(committed)
    assert unexplained == []
    assert set(packaged) <= set(producer.IMPLEMENTATION)


def test_the_study_document_embeds_the_generated_report(committed):
    document = (producer.ROOT / "docs" / "restart" / "TARGET_ACQUISITION.md").read_text()
    assert checker.report(committed) in document
    assert committed["declaration_commit"] in document


def test_the_declaration_predates_the_committed_record(committed, environment_skip):
    import subprocess

    commit = committed["declaration_commit"]
    present = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                             cwd=producer.ROOT, capture_output=True).returncode == 0
    if not present:
        environment_skip("the declaration commit is not in this checkout's history")
    assert checker.history_problems(committed, producer.DECLARATION, producer.RESULTS) == []
