"""The interval-coverage instrument, its declared study, producer and checker.

Coverage is a counting exercise, so most of these tests are about counting
honestly: a failure is never dropped from a denominator, both bootstrap
constructions come from one set of draws, and the checker notices a record that
no longer matches its declaration.
"""

import copy
import hashlib
import json

import numpy as np
import pytest

from benchmarks import check_interval_coverage as checker
from benchmarks import run_interval_coverage as producer
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.subspace import coverage
from clifford_qc.subspace.coverage import (
    interval_replica, level_key, summarize_coverage, wilson_interval,
)
from clifford_qc.subspace.uncertainty import bootstrap_ritz, bootstrap_ritz_samples

ORTHOGONAL = {"key": "tfim4_orthogonal", "role": "development",
              "model": {"family": "tfim", "n": 4, "J": 1.0, "h": 1.0},
              "pool_words": 3, "krylov_powers": 0}


@pytest.fixture(scope="module")
def system():
    return producer.build_system(ORTHOGONAL)


@pytest.fixture(scope="module")
def excited_measurement():
    spec = {"key": "tfim4_mixed", "role": "development",
            "model": {"family": "tfim", "n": 4, "J": 1.0, "h": 1.0},
            "pool_words": 6, "krylov_powers": 2}
    shared, _ = producer.build_system(spec)
    cache = shared.measure(FiniteShotBackend(seed=1), 10)
    point = shared.solve(cache, calibrate_overlap=True)
    assert len(point.energies) > 4
    return shared, cache, point


@pytest.fixture
def declaration(tmp_path):
    content = {
        "schema": producer.DECLARATION_SCHEMA, "declared": "test",
        "question": "test", "target": "exact_subspace_ground_energy",
        "seed_root": 11, "levels": [0.68, 0.95],
        "methods": list(coverage.METHODS), "strata_edges": [0.001, 0.01, 0.1],
        "summary_confidence": 0.95, "systems": [ORTHOGONAL], "shots_per_group": [200, 400],
        "arms": [
            {"key": "threshold", "systems": ["tfim4_orthogonal"], "replicas": 3,
             "bootstrap_replicates": 8, "solver_kwargs": {}},
            {"key": "repair", "systems": ["tfim4_orthogonal"], "replicas": 2,
             "bootstrap_replicates": 8,
             "solver_kwargs": {"overlap_repair": "nearest_correlation"}},
        ],
        "classification": {}, "spot_replay_replicas": 1,
    }
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(content))
    return path


@pytest.fixture
def record(declaration):
    loaded = producer.load_declaration(declaration)
    rows = producer.run(loaded)
    return producer.build_record(declaration, rows, declaration_commit="0" * 40)


# ---------------------------------------------------------------- instrument

def test_wilson_interval_matches_known_values_and_stays_in_range():
    assert wilson_interval(95, 100) == pytest.approx((0.8882495, 0.9784563), abs=1e-6)
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.2775328, abs=1e-6))
    assert wilson_interval(10, 10) == (pytest.approx(0.7224672, abs=1e-6), 1.0)
    narrow, wide = wilson_interval(50, 100, 0.68), wilson_interval(50, 100, 0.95)
    assert wide[0] < narrow[0] < 0.5 < narrow[1] < wide[1]


@pytest.mark.parametrize("args", [(1, 0), (-1, 5), (6, 5), (1, 5, 0.0), (1, 5, 1.0)])
def test_wilson_interval_rejects_invalid_counts(args):
    with pytest.raises(ValueError):
        wilson_interval(*args)


def test_bootstrap_samples_are_the_draws_behind_the_bootstrap_interval(system):
    shared, _ = system
    cache = shared.measure(FiniteShotBackend(seed=5), 300)
    point = shared.solve(cache)
    samples = bootstrap_ritz_samples(shared, cache, result=point, replicates=25, seed=9)
    interval = bootstrap_ritz(shared, cache, result=point, replicates=25, seed=9, delta=0.1)
    lower, upper = np.quantile(samples, [0.05, 0.95])
    assert (interval.lower, interval.upper) == (lower, upper)
    assert interval.std_error == pytest.approx(np.std(samples, ddof=1))
    assert samples.shape == (25,)
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_ritz_samples(shared, cache, result=point, replicates=1)


@pytest.mark.parametrize("bootstrap", [bootstrap_ritz_samples, bootstrap_ritz])
def test_bootstrap_reports_a_replica_that_loses_the_requested_root(excited_measurement,
                                                                 bootstrap):
    shared, cache, point = excited_measurement
    with pytest.raises(ValueError, match="bootstrap replica .*root 4 is unavailable"):
        bootstrap(shared, cache, result=point, root=4, replicates=10, seed=5)


def test_bootstrap_root_loss_keeps_the_measurement_in_the_coverage_denominator(
        excited_measurement):
    shared, cache, point = excited_measurement
    row = interval_replica(shared, cache, target=shared.bank.solve().energies[4],
                           solver_kwargs={"calibrate_overlap": True}, root=4,
                           bootstrap_replicates=10, bootstrap_seed=5)
    assert row["outcome"] == "ok" and row["estimate"] == point.energies[4]
    assert row["shots"] == cache.total_shots
    assert row["intervals"]["delta_normal"]["status"] == "ok"
    summary = summarize_coverage([row])
    for method in ("bootstrap_percentile", "bootstrap_basic"):
        interval = row["intervals"][method]
        assert interval["status"] == "failed"
        assert "root 4 is unavailable" in interval["error"]
        assert interval["levels"] == {} and interval["std_error"] is None
        found = summary["methods"][method]
        assert found["interval_failures"] == 1
        for level in found["levels"].values():
            assert level["all_replicas"]["n"] == 1
            assert level["all_replicas"]["covered"] == 0
            assert level["successful_intervals"]["n"] == 0


def test_a_root_missing_from_the_point_solve_is_a_failed_replica(excited_measurement):
    shared, cache, point = excited_measurement
    root = len(point.energies)
    row = interval_replica(shared, cache, target=shared.bank.solve().energies[root],
                           solver_kwargs={"calibrate_overlap": True}, root=root)
    assert row["outcome"] == "solve_failed"
    assert f"root {root} is unavailable" in row["solve_error"]
    assert row["estimate"] is None and row["intervals"] == {}
    assert row["shots"] == cache.total_shots
    summary = summarize_coverage([row])
    assert summary["replicas"] == summary["solve_failures"] == 1


def test_one_replica_reports_every_interval_against_the_target(system):
    shared, design = system
    cache = shared.measure(FiniteShotBackend(seed=6), 500)
    row = interval_replica(shared, cache, target=design["target"],
                           bootstrap_replicates=20, bootstrap_seed=3)
    json.dumps(row, allow_nan=False)
    point = shared.solve(cache)
    assert row["outcome"] == "ok" and row["estimate"] == point.ground_energy
    assert row["error"] == pytest.approx(point.ground_energy - design["target"])
    assert row["shots"] == cache.total_shots
    assert row["stratification_source"] == "smallest_retained_overlap_eigenvalue"
    samples = bootstrap_ritz_samples(shared, cache, result=point, replicates=20, seed=3)
    for level in (0.68, 0.95):
        key = level_key(level)
        low, high = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2])
        percentile = row["intervals"]["bootstrap_percentile"]["levels"][key]
        basic = row["intervals"]["bootstrap_basic"]["levels"][key]
        assert (percentile["lower"], percentile["upper"]) == (low, high)
        # The basic interval is the percentile interval reflected about the estimate.
        assert basic["lower"] == pytest.approx(2 * row["estimate"] - high, abs=1e-12)
        assert basic["upper"] == pytest.approx(2 * row["estimate"] - low, abs=1e-12)
        for method in coverage.METHODS:
            entry = row["intervals"][method]["levels"][key]
            assert entry["covered"] == (entry["lower"] <= design["target"] <= entry["upper"])
    wide = row["intervals"]["delta_normal"]["levels"][level_key(0.95)]
    narrow = row["intervals"]["delta_normal"]["levels"][level_key(0.68)]
    assert wide["lower"] < narrow["lower"] < narrow["upper"] < wide["upper"]


def test_the_delta_method_is_not_applicable_to_a_repaired_estimator(system):
    shared, design = system
    cache = shared.measure(FiniteShotBackend(seed=7), 300)
    row = interval_replica(shared, cache, target=design["target"], bootstrap_replicates=6,
                           solver_kwargs={"overlap_repair": "nearest_correlation"})
    assert row["intervals"]["delta_normal"]["status"] == "not_applicable"
    assert row["intervals"]["bootstrap_basic"]["status"] == "ok"
    assert not coverage.delta_method_applies({"ratio_error_budget": 0.1})
    assert not coverage.delta_method_applies({"pencil_angle": 0.1})
    assert coverage.delta_method_applies({"calibrate_overlap": True})


def test_a_failed_solve_is_a_row_not_an_exception(system, monkeypatch):
    shared, design = system
    cache = shared.measure(FiniteShotBackend(seed=8), 300)

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("no usable direction")

    monkeypatch.setattr(shared, "solve", fail)
    row = interval_replica(shared, cache, target=design["target"])
    assert row["outcome"] == "solve_failed"
    assert row["solve_error"] == "LinAlgError: no usable direction"
    assert row["intervals"] == {} and row["estimate"] is None


def test_a_failed_bootstrap_fails_both_bootstrap_intervals_only(system, monkeypatch):
    shared, design = system
    cache = shared.measure(FiniteShotBackend(seed=9), 300)

    def fail(*args, **kwargs):
        raise ValueError("replica 3 retained no direction")

    monkeypatch.setattr(coverage, "bootstrap_ritz_samples", fail)
    row = interval_replica(shared, cache, target=design["target"])
    assert row["outcome"] == "ok" and row["intervals"]["delta_normal"]["status"] == "ok"
    for method in ("bootstrap_percentile", "bootstrap_basic"):
        assert row["intervals"][method]["status"] == "failed"
        assert row["intervals"][method]["error"] == "ValueError: replica 3 retained no direction"


@pytest.mark.parametrize("kwargs", [
    {"levels": ()}, {"levels": (0.5, 0.5)}, {"levels": (1.0,)}, {"methods": ("jackknife",)},
    {"target": float("nan")},
])
def test_invalid_replica_requests_are_rejected(system, kwargs):
    shared, design = system
    cache = shared.measure(FiniteShotBackend(seed=10), 100)
    with pytest.raises(ValueError):
        interval_replica(shared, cache, **{"target": design["target"], **kwargs})


def test_a_ratio_estimator_is_stratified_by_the_measured_overlap(system):
    pytest.importorskip("scipy")
    shared, design = system
    cache = shared.measure(FiniteShotBackend(seed=12), 2000)
    row = interval_replica(shared, cache, target=design["target"], bootstrap_replicates=3,
                           solver_kwargs={"ratio_error_budget": 0.1})
    assert row["outcome"] == "ok"
    assert row["stratification_source"] == "measured_overlap_eigenvalue_min"
    assert row["intervals"]["delta_normal"]["status"] == "not_applicable"


# ------------------------------------------------------------------ counting

def _row(*, covered, value=0.5, low=-1.0, high=1.0, status="ok"):
    levels = {level_key(0.95): {"lower": low, "upper": high, "covered": covered}}
    interval = ({"status": status, "error": None, "std_error": 0.1, "levels": levels}
                if status == "ok" else
                {"status": status, "error": "ValueError: boom", "std_error": None, "levels": {}})
    return {"outcome": "ok", "solve_error": None, "estimate": 0.25, "error": 0.25,
            "shots": 10, "retained_rank": 2, "stratification_value": value,
            "stratification_source": "smallest_retained_overlap_eigenvalue",
            "intervals": {"delta_normal": interval}}


def _failed():
    return {"outcome": "solve_failed", "solve_error": "ValueError: dead", "estimate": None,
            "error": None, "shots": 10, "retained_rank": None, "stratification_value": None,
            "stratification_source": None, "intervals": {}}


def test_failures_stay_in_the_all_replica_denominator():
    rows = ([_row(covered=True)] * 6 + [_row(covered=False, low=0.5)] * 1
            + [_row(covered=False, high=-0.5)] * 1
            + [_row(covered=False, status="failed")] * 1 + [_failed()] * 1)
    summary = summarize_coverage(rows, levels=(0.95,), methods=("delta_normal",))
    assert summary["replicas"] == 10 and summary["solve_failures"] == 1
    assert summary["solve_failure_messages"] == {"ValueError: dead": 1}
    method = summary["methods"]["delta_normal"]
    assert method["interval_failures"] == 1
    assert method["interval_failure_messages"] == {"ValueError: boom": 1}
    level = method["levels"][level_key(0.95)]
    assert level["all_replicas"]["n"] == 10 and level["all_replicas"]["covered"] == 6
    assert level["all_replicas"]["rate"] == 0.6
    assert level["successful_intervals"]["n"] == 8
    assert level["successful_intervals"]["rate"] == 0.75
    assert level["all_replicas"]["wilson"] == list(wilson_interval(6, 10))
    # The target is estimate - error = 0: once below an interval, once above one.
    assert level["target_below_interval"] == 1 and level["target_above_interval"] == 1
    assert level["median_width"] == 2.0
    assert summary["estimate"]["n"] == 9 and summary["estimate"]["below_target"] == 0


def test_strata_use_fixed_half_open_edges_and_exclude_failed_solves():
    rows = [_row(covered=True, value=v) for v in (1e-4, 1e-3, 5e-3, 1e-2, 0.5)] + [_failed()]
    summary = summarize_coverage(rows, levels=(0.95,), methods=("delta_normal",),
                                 strata_edges=(1e-3, 1e-2))
    assert [(s["lower"], s["upper"], s["n"]) for s in summary["strata"]] == [
        (None, 1e-3, 1), (1e-3, 1e-2, 2), (1e-2, None, 2)]
    assert summary["strata"][1]["methods"]["delta_normal"]["levels"][level_key(0.95)][
        "all_replicas"]["n"] == 2
    assert summary["unstratified_solved"] == 0


def test_a_method_no_estimator_row_supports_is_reported_as_not_applicable():
    row = _row(covered=True)
    row["intervals"]["delta_normal"] = {"status": "not_applicable", "error": None,
                                        "std_error": None, "levels": {}}
    summary = summarize_coverage([row, _failed()], levels=(0.95,), methods=("delta_normal",))
    assert summary["methods"]["delta_normal"] == {"applicable": False}


@pytest.mark.parametrize("kwargs", [{"strata_edges": (0.1, 0.1)}, {"strata_edges": (1.0, 0.1)},
                                    {"strata_edges": (float("inf"),)}, {"rows": []}])
def test_invalid_summaries_are_rejected(kwargs):
    arguments = {"rows": [_row(covered=True)], "levels": (0.95,),
                 "methods": ("delta_normal",), **kwargs}
    with pytest.raises(ValueError):
        summarize_coverage(arguments.pop("rows"), **arguments)


# ------------------------------------------------------- declaration, producer

def test_the_committed_declaration_is_valid_and_complete():
    declaration = producer.load_declaration()
    grid = producer.cells(declaration)
    assert [cell["cell"] for cell in grid] == list(range(6))
    roles = {system["key"]: system["role"] for system in declaration["systems"]}
    assert set(roles.values()) == {"development", "held_out"}
    for system in declaration["systems"]:
        _, design = producer.build_system(system)
        assert design["exact_rank"] >= 1 and np.isfinite(design["target"])
    ratio = next(arm for arm in declaration["arms"] if arm["key"] == "constrained_ratio")
    assert producer.arm_kwargs(ratio, 2000) == {"ratio_error_budget": 0.005}
    assert producer.arm_kwargs(ratio, 20000) == {"ratio_error_budget": 0.0016}
    assert declaration["levels"] == [0.68, 0.95]


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d.update(schema="other"), "schema"),
    (lambda d: d.update(target="true_ground_energy"), "target"),
    (lambda d: d.update(shots_per_group=[200, 200]), "shots_per_group"),
    (lambda d: d["arms"][0].update(systems=["unknown"]), "undeclared system"),
    (lambda d: d["arms"][0].update(bootstrap_replicates=1), "bootstrap replicates"),
    (lambda d: d["arms"][0].update(ratio_error_budget_by_shots={"200": 0.1}), "one budget"),
    (lambda d: d["systems"][0]["model"].update(family="hubbard"), "model family"),
    (lambda d: d.update(methods=["jackknife"]), "interval method"),
])
def test_invalid_declarations_are_rejected(declaration, mutation, message):
    content = json.loads(declaration.read_text())
    mutation(content)
    declaration.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=message):
        producer.load_declaration(declaration)


def test_seeds_are_distinct_across_cells_replicas_and_arms(declaration):
    loaded = producer.load_declaration(declaration)
    seeds = {producer.measurement_seed(loaded, cell, replica)
             for cell in range(2) for replica in range(3)}
    seeds |= {producer.bootstrap_seed(loaded, cell, replica, arm)
              for cell in range(2) for replica in range(3) for arm in range(2)}
    assert len(seeds) == 6 + 12


def test_the_run_is_complete_paired_and_independent_of_worker_count(declaration, record):
    loaded = producer.load_declaration(declaration)
    rows = record["rows"]
    assert len(rows) == 2 * (3 + 2)
    by_key = {(row["cell"], row["arm"], row["replica"]): row for row in rows}
    # Arms on one replica read the same measurement.
    assert by_key[(0, "threshold", 1)]["measurement_seed"] == \
        by_key[(0, "repair", 1)]["measurement_seed"]
    assert by_key[(0, "threshold", 1)]["bootstrap_seed"] != \
        by_key[(0, "repair", 1)]["bootstrap_seed"]
    assert (0, "repair", 2) not in by_key
    assert producer.run(loaded, jobs=2) == rows


def test_summaries_carry_the_declared_labels_and_headline(record):
    assert [(cell["cell"], cell["arm"]) for cell in record["cells"]] == [
        (0, "threshold"), (0, "repair"), (1, "threshold"), (1, "repair")]
    first = record["cells"][0]
    assert set(first["labels"]) == set(coverage.METHODS)
    assert "delta_normal" not in record["cells"][1]["labels"]
    tally = record["headline"]["threshold"]["delta_normal"]
    assert tally["under"] + tally["consistent"] + tally["over"] == 2 * 2
    assert producer.classify({"nominal": 0.95, "all_replicas": {"wilson": [0.5, 0.9]}}) == "under"
    assert producer.classify({"nominal": 0.68, "all_replicas": {"wilson": [0.7, 0.9]}}) == "over"
    assert producer.classify({"nominal": 0.95, "all_replicas": {"wilson": [0.9, 0.99]}}) \
        == "consistent"


def test_the_record_is_written_deterministically_and_never_overwritten(record, tmp_path):
    first, second = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    producer.write_record(record, first)
    producer.write_record(record, second)
    assert first.read_bytes() == second.read_bytes()
    assert producer.read_record(first) == json.loads(json.dumps(record))
    with pytest.raises(FileExistsError):
        producer.write_record(record, first)


# -------------------------------------------------------------------- checker

def test_a_faithful_record_passes_structure_and_spot_replay(declaration, record):
    assert checker.structural_problems(record, declaration) == []
    problems, compared = checker.replay_problems(record, replicas=1)
    assert problems == [] and compared == 2 * 2
    problems, compared = checker.replay_problems(record, replicas=None)
    assert problems == [] and compared == len(record["rows"])


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r["rows"].pop(), "missing"),
    (lambda r: r["rows"].append(copy.deepcopy(r["rows"][0])), "duplicate"),
    (lambda r: r["rows"][0].update(replica=99), "undeclared"),
    (lambda r: r["rows"][0].update(measurement_seed=1), "declared seeds"),
    (lambda r: r["rows"][0].update(shots=1), "shots"),
    (lambda r: r["declaration"].update(seed_root=12), "embed"),
    (lambda r: r.update(declaration_sha256="0" * 64), "digest"),
    (lambda r: r["systems"]["tfim4_orthogonal"].update(target=0.0), "systems"),
    (lambda r: r["headline"]["threshold"]["delta_normal"].update(under=99), "headline"),
    (lambda r: r.update(schema="other"), "schema"),
])
def test_structural_tampering_is_detected(declaration, record, mutation, message):
    mutation(record)
    assert any(message in problem for problem in checker.structural_problems(record, declaration))


def test_a_changed_coverage_flag_no_longer_matches_its_summary(declaration, record):
    entry = record["rows"][0]["intervals"]["delta_normal"]["levels"][level_key(0.95)]
    entry["covered"] = not entry["covered"]
    assert any(problem.startswith("cells") or problem.startswith("headline")
               for problem in checker.structural_problems(record, declaration))


def test_replay_detects_a_changed_estimate_but_not_rounding(record):
    record["rows"][0]["estimate"] += 1e-9
    assert checker.replay_problems(record, replicas=1)[0] == []
    record["rows"][0]["estimate"] += 1e-3
    assert any("estimate" in problem for problem in checker.replay_problems(record, replicas=1)[0])


def test_an_ill_conditioned_row_replays_its_discrete_fields_only(record):
    row = record["rows"][0]
    row["stratification_value"] = 1e-9
    row["estimate"] += 1.0
    assert checker.replay_problems(record, replicas=1)[0] == []
    row["retained_rank"] += 1
    assert any("retained_rank" in problem
               for problem in checker.replay_problems(record, replicas=1)[0])


def test_only_a_searching_arm_that_failed_is_marked_machine_dependent():
    record = {
        "declaration": {"arms": [
            {"key": "threshold"},
            {"key": "ratio", "ratio_error_budget_by_shots": {"200": 0.1}}]},
        "rows": [
            {"cell": 0, "arm": "threshold", "outcome": "solve_failed"},
            {"cell": 0, "arm": "ratio", "outcome": "ok"},
            {"cell": 1, "arm": "ratio", "outcome": "ok"},
            {"cell": 1, "arm": "ratio", "outcome": "solve_failed"},
            {"cell": 2, "arm": "ratio", "outcome": "ok"}]}
    # A thresholded solve that fails is still replayable; a search that never
    # failed in a cell is too.
    assert checker.search_unstable(record) == {(1, "ratio")}


def test_machine_dependent_rows_replay_their_seeds_and_shots_only(record, monkeypatch):
    monkeypatch.setattr(checker, "search_unstable", lambda record: {(0, "threshold")})
    row = record["rows"][0]
    assert (row["cell"], row["arm"]) == (0, "threshold")
    row["estimate"] += 1.0
    row["outcome"] = "solve_failed"
    assert checker.replay_problems(record, replicas=1)[0] == []
    row["bootstrap_seed"] += 1
    assert any("bootstrap_seed" in problem
               for problem in checker.replay_problems(record, replicas=1)[0])
    row["bootstrap_seed"] -= 1
    row["shots"] += 1
    assert any("shots" in problem for problem in checker.replay_problems(record, replicas=1)[0])
    # Every other cell and arm is still compared in full.
    other = next(r for r in record["rows"] if (r["cell"], r["arm"]) == (0, "repair"))
    other["estimate"] += 1.0
    assert any("estimate" in problem for problem in checker.replay_problems(record, replicas=1)[0])


def test_replay_is_blocked_by_an_evolved_source_or_another_sampling_stream(record):
    record["implementation"]["clifford_qc/subspace/coverage.py"] = "0" * 64
    blockers = checker.replay_blockers(record)
    assert "EVOLVED clifford_qc/subspace/coverage.py" in blockers


def test_an_edited_bound_source_replays_only_with_its_packaged_snapshot(record, tmp_path,
                                                                           monkeypatch):
    relative = "clifford_qc/subspace/coverage.py"
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


def test_history_check_reports_an_unknown_declaration_commit(declaration, record, tmp_path):
    results = producer.ROOT / "docs" / "restart" / "interval_coverage_results.json.gz"
    problems = checker.history_problems(record, producer.DECLARATION, results)
    assert problems == [f"declaration commit {'0' * 40!r} is not in this repository's history"]


# ------------------------------------------------------- the committed record

@pytest.fixture(scope="module")
def committed():
    return producer.read_record(producer.RESULTS)


def test_the_committed_record_matches_its_declaration(committed):
    assert checker.structural_problems(committed, producer.DECLARATION) == []
    declaration = producer.load_declaration()
    expected = sum(arm["replicas"] * len(arm["systems"]) * len(declaration["shots_per_group"])
                   for arm in declaration["arms"])
    assert len(committed["rows"]) == expected == 4000
    assert committed["implementation"].keys() == set(producer.IMPLEMENTATION)
    # The held-out system was sampled only under the declared seed root.
    assert committed["declaration"]["seed_root"] == 20261116


def test_the_committed_record_marks_the_unattainable_search_as_machine_dependent(committed):
    # The constrained ratio failed solves only on the held-out bank, whose two
    # cells are the only rows not replayed in full.
    assert checker.search_unstable(committed) == {(4, "constrained_ratio"),
                                                  (5, "constrained_ratio")}
    cells = {cell["cell"]: cell["system"] for cell in producer.cells(committed["declaration"])}
    assert {cells[4], cells[5]} == {"xxz4_mixed"}


def test_every_bound_source_edited_since_the_run_is_packaged(committed):
    # An edit to a bound source without its execution-time bytes would turn the
    # gate's replay off without failing it. Packaging the snapshot keeps it on.
    packaged, unexplained = checker.source_drift(committed)
    assert unexplained == []
    assert set(packaged) <= set(producer.IMPLEMENTATION)


def test_the_study_document_embeds_the_generated_report(committed):
    document = (producer.ROOT / "docs" / "restart" / "INTERVAL_COVERAGE.md").read_text()
    assert checker.report(committed) in document


def test_the_declaration_predates_the_committed_record(committed, environment_skip):
    import subprocess

    commit = committed["declaration_commit"]
    present = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                             cwd=producer.ROOT, capture_output=True).returncode == 0
    if not present:
        environment_skip("the declaration commit is not in this checkout's history")
    assert checker.history_problems(committed, producer.DECLARATION, producer.RESULTS) == []
