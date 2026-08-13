"""Statistical contract checks for the R1 exact-tier search."""

from benchmarks.run_exact_shot_search import (
    _confirm_result,
    _exploratory_bracket,
    _summary,
)


def _row(endpoint, passes):
    return {"effective_shots_per_setting": endpoint, "passes_target": passes}


def test_exploratory_bracket_requires_a_persistent_crossing():
    rows = [
        _row(64, False), _row(256, True), _row(1024, False),
        _row(4096, True), _row(16384, True),
    ]
    assert _exploratory_bracket(rows) == (1024, 4096)


def test_confirmation_reports_a_region_not_a_false_exact_integer():
    exploration = [_row(64, False), _row(256, True), _row(1024, True)]
    confirmation = [_row(64, False), _row(256, True)]
    assert _confirm_result(exploration, confirmation) == {
        "status": "confirmed_bracket",
        "confirmed_failing_effective_shots_per_setting": 64,
        "confirmed_passing_effective_shots_per_setting": 256,
    }


def test_solver_failure_forces_an_endpoint_to_fail():
    rows = [
        {"failure": None, "energy": -1.0, "rank": 2},
        {"failure": "ValueError"},
    ]
    summary = _summary(rows, -1.0, bootstrap_seed=7)
    assert not summary["zero_failure_gate"]
    assert not summary["passes_target"]
    assert summary["failures"] == {"ValueError": 1}
