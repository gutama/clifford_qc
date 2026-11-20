"""The declared molecular acquisition study: declaration, producer and checker.

The producer's look loop mirrors ``acquire_ritz`` so that each arm's stop can
be bootstrapped from that look's data. The claim worth testing is that it is
``acquire_ritz``: on a short development schedule every stop, estimate,
interval and bootstrap must be reproduced by ``acquire_ritz`` itself. The rest
are bookkeeping claims: completeness, seeds, stops at the first met look, and
failures counted rather than dropped.
"""

import copy
import json

import numpy as np
import pytest

from benchmarks import check_molecular_acquisition as checker
from benchmarks import run_molecular_acquisition as producer
from clifford_qc.backends.finite_shot import FiniteShotBackend

# Development only: a throwaway seed root and a short schedule on `tri`.
DEVELOPMENT_ROOT = 424242


@pytest.fixture(scope="module")
def declaration():
    found = copy.deepcopy(producer.load_declaration())
    found["seed_root"] = DEVELOPMENT_ROOT
    found["cumulative_shots_per_group"] = [100, 200, 400, 800]
    found["target_std_errors"] = [0.02, 0.01]
    found["systems"] = [dict(found["systems"][0], replicas=3)]
    found["bootstrap"]["plans"] = [{"systems": ["tri"], "replicas": 2, "draws": 12,
                                    "arms": ["uniform", "uniform_cv", "targeted",
                                             "targeted_cv"]}]
    found["criteria"]["systems"] = ["tri"]
    found["comparison"]["bootstrap_replicates"] = 200
    return found


@pytest.fixture(scope="module")
def system(declaration):
    built = producer.build_system(declaration["systems"][0])
    producer.prepare(built)
    return built


@pytest.fixture(scope="module")
def rows(declaration, system):
    return [json.loads(json.dumps(producer.replica_row(declaration, system, 0, replica)))
            for replica in range(3)]


@pytest.fixture(scope="module")
def record(system, rows):
    return {"systems": {"tri": system["design"]}, "rows": rows}


# ------------------------------------------------------------------- declaration

def test_the_committed_declaration_is_valid_and_names_its_files():
    declared = producer.load_declaration()
    assert [s["key"] for s in declared["systems"]] == ["tri", "h4", "lih", "beh2"]
    assert {s["role"] for s in declared["systems"] if s["key"] in ("lih", "beh2")} == {"held_out"}
    for spec in declared["systems"]:
        if "path" in spec["fcidump"]:
            assert producer.sha256(producer.ROOT / spec["fcidump"]["path"]) \
                == spec["fcidump"]["sha256"]


@pytest.mark.parametrize("damage", [
    lambda d: d.update(schema="other"),
    lambda d: d["systems"].append(dict(d["systems"][0])),
    lambda d: d["systems"][0].update(role="peeked"),
    lambda d: d["systems"][0].update(electrons=3),
    lambda d: d["arms"].append({"key": "odd", "run": "targeted", "corrected": True}),
    lambda d: d.update(target_std_errors=[0.01, 0.01]),
    lambda d: d.update(cumulative_shots_per_group=[100, 100]),
    lambda d: d["bootstrap"]["plans"][1].update(replicas=500),
    lambda d: d["comparison"]["pairs"].append(["uniform", "nobody"]),
    lambda d: d["criteria"].update(reference="uniform_cv"),
    lambda d: d.update(solver_kwargs={"pencil_angle": 0.1}),
])
def test_a_malformed_declaration_is_refused(tmp_path, damage):
    declared = copy.deepcopy(producer.load_declaration())
    damage(declared)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declared))
    with pytest.raises((ValueError, KeyError)):
        producer.load_declaration(path)


def test_the_bank_and_family_are_built_as_declared(system):
    design = system["design"]
    assert design == {**design, "qubits": 6, "generators": 6, "exact_rank": 6, "settings": 169,
                      "family_terms": 10}
    assert design["target"] == pytest.approx(system["shared"].bank.solve().ground_energy)


# -------------------------------------------------------------- producer is acquire_ritz

def test_acquire_ritz_reproduces_every_stop_interval_and_bootstrap(declaration, system, rows):
    for row in rows:
        assert checker.replay_row(declaration, system, 0, row) == []
    booted = [row["arms"][arm]["stopping"][key]["bootstrap"] for row in rows[:2]
              for arm in row["arms"] for key in row["arms"][arm]["stopping"]]
    assert all(entry is not None and "spread" in entry for entry in booted)
    assert all(stop["bootstrap"] is None for arm in rows[2]["arms"].values()
               for stop in arm["stopping"].values())


def test_a_fresh_backend_draws_the_rows_a_prepared_one_does(declaration, system, rows):
    fresh = producer.replica_row(declaration, system, 0, 1,
                                 backend=FiniteShotBackend(seed=0))
    assert json.loads(json.dumps(fresh)) == rows[1]


def test_uniform_and_uniform_cv_share_one_run(rows):
    for row in rows:
        ledger = row["runs"]["uniform"]["ledger"]
        assert all("energy" in entry and "raw_energy" in entry for entry in ledger)
        assert "energy" not in row["runs"]["targeted"]["ledger"][0]


def test_the_committed_rows_pass_the_structural_checks(declaration, record):
    assert checker.row_problems(record, declaration) == []


@pytest.mark.parametrize("damage, message", [
    (lambda row: row.update(measurement_seed=row["measurement_seed"] + 1), "wrong seed"),
    (lambda row: row["runs"]["targeted"]["ledger"][1].update(shots=1), "not the schedule's"),
    (lambda row: row["arms"]["targeted"]["stopping"]["0.02"].update(
        look=row["arms"]["targeted"]["stopping"]["0.02"]["look"] + 1), "first met look"),
    (lambda row: row["arms"]["targeted_cv"]["stopping"]["0.01"]["interval"].update(
        upper=row["arms"]["targeted_cv"]["stopping"]["0.01"]["interval"]["upper"] + 1e-3),
     "interval is not"),
    (lambda row: row["arms"]["uniform_cv"]["stopping"]["0.02"]["bootstrap"].update(seed=1),
     "planned draws and seed"),
])
def test_a_damaged_row_is_reported(declaration, record, damage, message):
    damaged = copy.deepcopy(record)
    damage(damaged["rows"][0])
    assert any(message in problem for problem in checker.row_problems(damaged, declaration))


def test_a_missing_or_unplanned_bootstrap_is_reported(declaration, record):
    damaged = copy.deepcopy(record)
    stop = damaged["rows"][2]["arms"]["targeted"]["stopping"]["0.01"]
    stop["bootstrap"] = {"draws": 12, "seed": 1, "spread": 0.0,
                         "interval": stop["interval"]}
    assert any("does not ask for" in p for p in checker.row_problems(damaged, declaration))
    missing = copy.deepcopy(record)
    missing["rows"].pop()
    assert any("is missing" in p for p in checker.row_problems(missing, declaration))


# ---------------------------------------------------------------------- failures

def test_a_failed_look_fails_only_the_arms_still_running(declaration, system, rows, monkeypatch):
    # Fail the targeted run at its last look: arms that stopped earlier keep
    # their stops, the others fail there with the shots already spent.
    reference = rows[0]
    last = len(declaration["cumulative_shots_per_group"]) - 1
    shared = system["shared"]
    solve = shared.solve
    calls = {"targeted": 0, "active": False}

    def failing(cache, **kwargs):
        if calls["active"]:
            if calls["targeted"] == last:
                raise np.linalg.LinAlgError("no retained mode")
            calls["targeted"] += 1
        return solve(cache, **kwargs)

    monkeypatch.setattr(shared, "solve", failing)
    run = next(run for run in declaration["runs"] if run["key"] == "targeted")
    arms = [arm for arm in declaration["arms"] if arm["run"] == "targeted"]
    monkeypatch.setattr(producer, "bootstrap_plan", lambda *args: None)
    calls["active"] = True
    found = producer.run_schedule(declaration, system, system_index=0, replica=0, run=run,
                                  arms=arms, backend=system["backend"])
    found = json.loads(json.dumps(found))
    assert found["failure"]["look"] == last and len(found["ledger"]) == last
    for key, stop in found["stops"]["targeted"].items():
        expected = reference["arms"]["targeted"]["stopping"][key]
        if expected["look"] < last:
            assert stop["look"] == expected["look"] and stop["outcome"] == "ok"
        else:
            assert stop == {"outcome": "solve_failed", **found["failure"]}
    assert found["fixed"]["targeted"]["outcome"] == "solve_failed"


def test_a_failed_arm_counts_as_a_miss_and_as_cost(declaration, system, rows):
    damaged = copy.deepcopy(rows)
    stop = damaged[0]["arms"]["targeted_cv"]["stopping"]["0.01"]
    failure = {"outcome": "solve_failed", "look": stop["look"], "shots": stop["shots"],
               "error": "LinAlgError: no retained mode"}
    damaged[0]["arms"]["targeted_cv"]["stopping"]["0.01"] = failure
    designs = {"tri": system["design"]}
    before = producer.summarize(declaration, rows, designs)["summaries"][0]
    after = producer.summarize(declaration, damaged, designs)["summaries"][0]
    cell, was = (summary["arms"]["targeted_cv"]["stopping"]["0.01"] for summary in (after, before))
    assert cell["solve_failures"] == 1 and cell["coverage"]["schedule_valid"]["n"] == 3
    assert cell["coverage"]["schedule_valid"]["covered"] \
        == was["coverage"]["schedule_valid"]["covered"] - stop["interval"]["covered"]
    assert cell["bootstrap"]["coverage"]["n"] == 2
    assert cell["mean_shots"] == was["mean_shots"]


# ------------------------------------------------------------- summaries, criteria

def test_criteria_follow_the_declared_cells(declaration, system, rows):
    summary = producer.summarize(declaration, rows, {"tri": system["design"]})
    cells = summary["criteria"]["cells"]
    assert [cell["target"] for cell in cells] == ["0.02", "0.01"]
    assert summary["criteria"]["C1"] == all(
        cell["verdict"] == "fewer_shots_at_matched_error" for cell in cells)
    forced = copy.deepcopy(summary["summaries"])
    for cell in forced[0]["comparisons"]["uniform/targeted_cv"]["stopping"].values():
        cell["verdict"] = "fewer_shots_at_matched_error"
    for arm in forced[0]["arms"]["targeted_cv"]["stopping"].values():
        arm["coverage"]["schedule_valid"]["label"] = "consistent"
        arm["bootstrap"]["coverage"]["label"] = "under"
    found = producer.criteria(declaration, forced)
    assert found["C1"] and found["C2"] and not found["C3"]


def test_the_summary_is_what_the_checker_recomputes(declaration, system, rows):
    designs = {"tri": system["design"]}
    record = {"rows": rows, "systems": designs, **producer.summarize(declaration, rows, designs)}
    assert checker.summary_problems(record, declaration) == []
    record["summaries"][0]["arms"]["uniform"]["stopping"]["0.01"]["mean_shots"] += 1
    assert checker.summary_problems(record, declaration)


def test_the_producer_refuses_to_overwrite_a_record(tmp_path, capsys):
    output = tmp_path / "record.json.gz"
    output.write_bytes(b"")
    assert producer.main(["--output", str(output)]) == 1
    assert "never overwritten" in capsys.readouterr().out


def test_errors_at_every_look_read_each_arms_own_ledger(declaration, system, rows):
    record = {"declaration": declaration, "systems": {"tri": system["design"]}, "rows": rows}
    found = checker.look_errors(record, "tri")
    assert len(found) == len(declaration["cumulative_shots_per_group"])
    # Every run reads the same pilot, so the first look is shared by the arms
    # that report the same functional.
    first = found[0]["arms"]
    assert first["uniform"] == first["targeted"]
    assert first["uniform_cv"] == first["targeted_cv"]
    target = system["design"]["target"]
    last = len(found) - 1
    errors = [row["runs"]["targeted_cv"]["ledger"][last]["energy"] - target for row in rows]
    assert found[last]["arms"]["targeted_cv"]["rmse"] \
        == pytest.approx(float(np.sqrt(np.mean(np.square(errors)))))
    short = copy.deepcopy(record)
    short["rows"][0]["runs"]["targeted"]["ledger"].pop()
    assert checker.look_errors(short, "tri")[last]["arms"]["targeted"]["n"] == 2


# ------------------------------------------------------------ the committed record

DECLARATION_COMMIT = "324fefd05d402774e5c3342284007f5e962167e9"


@pytest.fixture(scope="module")
def committed():
    return checker.read_record(producer.RESULTS)


def _cells(committed, systems=("lih", "beh2")):
    by_key = {entry["system"]: entry for entry in committed["summaries"]}
    return [(system, key, by_key[system]) for system in systems
            for key in ("0.01", "0.005")]


def test_the_committed_record_matches_its_declaration(committed):
    declaration = producer.load_declaration()
    assert checker.declaration_problems(committed, producer.DECLARATION) == []
    assert checker.row_problems(committed, declaration) == []
    assert checker.summary_problems(committed, declaration) == []
    assert len(committed["rows"]) == sum(s["replicas"] for s in declaration["systems"]) == 700
    assert committed["declaration"]["seed_root"] == 20261120
    assert committed["declaration_commit"] == DECLARATION_COMMIT
    assert committed["implementation"].keys() == set(producer.IMPLEMENTATION)
    assert committed["provenance"]["git_dirty"] is False
    # The eight-qubit banks take minutes to build and are checked by the gate.
    tri = producer.build_system(declaration["systems"][0])
    assert checker.design_problems(committed, {"systems": declaration["systems"][:1]},
                                   {"tri": tri}) == []


def test_the_committed_record_reports_what_the_document_says(committed):
    criteria = committed["criteria"]
    assert criteria["C1"] and criteria["C2"] and criteria["C3"]
    for entry in committed["summaries"]:
        for arm in entry["arms"].values():
            for cell in arm["stopping"].values():
                assert cell["solve_failures"] == 0
                assert cell["stopped"]["schedule_exhausted"] == 0
                assert cell["bootstrap"] is None or cell["bootstrap"]["failures"] == 0
    for system, key, entry in _cells(committed):
        uniform = entry["arms"]["uniform"]["stopping"][key]
        corrected = entry["arms"]["targeted_cv"]["stopping"][key]
        raw = entry["arms"]["targeted"]["stopping"][key]
        # Fewer shots and a lower error at the stop, in every held-out cell, so
        # the direction of C1 needs no model of how error falls with shots.
        shots = entry["comparisons"]["uniform/targeted_cv"]["stopping"][key]["shot_ratio"]
        assert 3.3 < shots < 9.8
        assert corrected["errors"]["rmse"] < uniform["errors"]["rmse"] / 1.4
        # Both intervals cover every replica; the corrected value's reported
        # error is honest and its bootstrap is wider than it needs to be.
        assert corrected["coverage"]["schedule_valid"]["rate"] == 1.0
        assert corrected["bootstrap"]["coverage"]["rate"] == 1.0
        assert corrected["errors"]["rmse"] <= corrected["rms_std_error"]
        assert 1.4 < corrected["bootstrap"]["median_spread_over_std_error"] < 1.8
        # The raw Ritz value's reported error is optimistic under targeting:
        # an unadjusted 1.96 sigma interval under-covers there.
        assert raw["errors"]["rmse"] > 1.4 * raw["rms_std_error"]
        assert raw["coverage"]["naive"]["label"] == "under"
        # Targeting with and without the correction stops at nearly the same
        # shots; the correction's gain is error at the stop.
        same = entry["comparisons"]["targeted/targeted_cv"]["stopping"][key]["shot_ratio"]
        assert 0.95 < same < 1.1


def test_error_falls_faster_than_one_over_root_shots_across_the_schedule(committed):
    # The declared caveat on matched error, measured: over the 128-fold
    # schedule the raw uniform error falls far more than the 11.3-fold that
    # one over root shots predicts.
    for system in ("lih", "beh2"):
        found = checker.look_errors(committed, system)
        first, last = found[0]["arms"]["uniform"], found[-1]["arms"]["uniform"]
        assert last["mean_shots_per_setting"] / first["mean_shots_per_setting"] \
            == pytest.approx(128.0, rel=1e-3)
        assert first["rmse"] / last["rmse"] > 50


def test_the_study_document_embeds_the_generated_report(committed):
    document = checker.DOCUMENT.read_text()
    assert checker.report(committed) in document
    assert committed["declaration_commit"] in document


def test_the_declaration_predates_the_committed_record(committed, environment_skip):
    import subprocess

    commit = committed["declaration_commit"]
    history = subprocess.run(["git", "rev-parse", "--is-shallow-repository"],
                             cwd=producer.ROOT, capture_output=True, text=True).stdout.strip()
    present = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                             cwd=producer.ROOT, capture_output=True).returncode == 0
    if history != "false" or not present:
        environment_skip("the declaration commit's history is not in this checkout")
    assert checker.history_problems(committed, producer.DECLARATION, producer.RESULTS) == []
