"""Unit tests for the warm-start replication driver.

The driver exists to decide whether a single-system, three-point trend is real,
so these tests concentrate on the machinery that could manufacture a false
answer: the rank statistic, the positive control that ties the replication to
the committed record, and the invariant gate.
"""

import json

import pytest

from benchmarks import run_warm_start_replication as ws


# ------------------------------------------------------------ rank statistic


def test_kendall_tau_is_minus_one_on_a_perfect_inversion():
    """The committed H4 shape: reference improves, subspace degrades."""
    adapt_errors = [27.09, 13.78, 6.79]
    acase_errors = [0.342, 0.612, 0.768]
    assert ws.kendall_tau(adapt_errors, acase_errors) == pytest.approx(-1.0)


def test_kendall_tau_is_plus_one_when_a_better_reference_helps():
    assert ws.kendall_tau([27.0, 13.0, 6.0],
                          [0.9, 0.5, 0.1]) == pytest.approx(1.0)


def test_kendall_tau_reports_no_trend_on_a_flat_ladder():
    """A flat downstream error must not be tie-broken into a spurious trend."""
    assert ws.kendall_tau([27.0, 13.0, 6.0], [0.5, 0.5, 0.5]) is None


def test_kendall_tau_corrects_for_partial_ties():
    tau = ws.kendall_tau([1.0, 2.0, 3.0], [1.0, 1.0, 2.0])
    assert tau is not None
    assert 0.0 < tau < 1.0            # tie-corrected, so short of a clean +1


def test_kendall_tau_needs_two_points():
    assert ws.kendall_tau([1.0], [2.0]) is None


def test_kendall_tau_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        ws.kendall_tau([1.0, 2.0], [1.0])


# --------------------------------------------------------------- the analysis


def _warm(operators, adapt_error, acase_error, capture=0.99,
          condition=1.05, rank=9):
    return {"requested_operators": operators, "adapt_operators": operators,
            "adapt_error_millihartree": adapt_error,
            "error_millihartree": acase_error, "subspace_capture": capture,
            "condition_number": condition, "effective_rank": rank,
            "ground_energy": -1.0, "reference_capture": 0.5}


def test_the_anomaly_is_reported_when_a_better_reference_costs_accuracy():
    analysis = ws.analyse_anomaly(
        {"error_millihartree": 3.0},
        [_warm(2, 27.0, 0.34, capture=0.9997),
         _warm(4, 13.8, 0.61, capture=0.9995),
         _warm(6, 6.8, 0.77, capture=0.9995)])

    assert analysis["replicated"] is True
    assert analysis["tau_adapt_vs_acase"] == pytest.approx(-1.0)
    assert analysis["best_adapt_operators"] == 6
    assert analysis["best_acase_operators"] == 2
    assert analysis["best_reference_is_best_subspace"] is False
    assert analysis["warm_rows_beating_cold"] == 3


def test_no_anomaly_is_reported_when_a_better_reference_helps():
    analysis = ws.analyse_anomaly(
        {"error_millihartree": 3.0},
        [_warm(2, 27.0, 0.9), _warm(4, 13.8, 0.5), _warm(6, 6.8, 0.1)])

    assert analysis["replicated"] is False
    assert analysis["tau_adapt_vs_acase"] == pytest.approx(1.0)
    assert analysis["best_reference_is_best_subspace"] is True


def test_the_capture_correlation_separates_a_span_problem():
    """Falling capture is what makes the anomaly a span problem."""
    analysis = ws.analyse_anomaly(
        {"error_millihartree": 3.0},
        [_warm(2, 27.0, 0.34, capture=0.9997),
         _warm(4, 13.8, 0.61, capture=0.9995),
         _warm(6, 6.8, 0.77, capture=0.9993)])
    # Better reference (lower adapt error) goes with lower capture.
    assert analysis["tau_adapt_vs_capture"] == pytest.approx(-1.0)


def test_the_committed_rungs_are_scored_separately():
    """The denser ladder must not hide what the committed three points say."""
    analysis = ws.analyse_anomaly(
        {"error_millihartree": 3.0},
        [_warm(1, 40.0, 0.20), _warm(2, 27.0, 0.34),
         _warm(4, 13.8, 0.61), _warm(6, 6.8, 0.77), _warm(8, 3.0, 0.10)])
    assert analysis["tau_adapt_vs_acase_committed_rungs"] == pytest.approx(-1.0)
    # The full ladder is not a clean inversion, so the two disagree -- which is
    # exactly the comparison the denser ladder exists to make.
    assert analysis["tau_adapt_vs_acase"] > -1.0


def test_the_analysis_abstains_without_enough_rows():
    analysis = ws.analyse_anomaly({"error_millihartree": 3.0},
                                  [{"adapt_operators": 2, "failed": "boom"}])
    assert analysis["replicated"] is None
    assert "fewer than two" in analysis["reason"]


def test_failed_rows_are_excluded_from_the_statistic():
    analysis = ws.analyse_anomaly(
        {"error_millihartree": 3.0},
        [_warm(2, 27.0, 0.34), {"adapt_operators": 4, "failed": "boom"},
         _warm(6, 6.8, 0.77)])
    assert analysis["tau_adapt_vs_acase"] == pytest.approx(-1.0)
    assert len(analysis["condition_numbers"]) == 2


# ------------------------------------------------------------ positive control


def _control_record(cold=3.018781343635979,
                    warm=((2, 0.3423008937817329), (4, 0.6124), (6, 0.7684))):
    return {"cold": {"error_millihartree": cold},
            "warm": [{"adapt_operators": k, "error_millihartree": value}
                     for k, value in warm]}


def test_the_control_passes_when_the_committed_numbers_reproduce(monkeypatch,
                                                                 tmp_path):
    committed = tmp_path / "warm_start_h4.json"
    committed.write_text(json.dumps(_control_record()), encoding="utf-8")
    monkeypatch.setattr(ws, "CONTROL_RECORD", committed)

    control = ws.check_control(_control_record())
    assert control["passed"] is True
    assert control["problems"] == []
    assert set(control["compared"]) == {"2", "4", "6"}


def test_the_control_fails_on_a_drifted_warm_row(monkeypatch, tmp_path):
    committed = tmp_path / "warm_start_h4.json"
    committed.write_text(json.dumps(_control_record()), encoding="utf-8")
    monkeypatch.setattr(ws, "CONTROL_RECORD", committed)

    drifted = _control_record(warm=((2, 0.9), (4, 0.6124), (6, 0.7684)))
    control = ws.check_control(drifted)
    assert control["passed"] is False
    assert any("k=2" in problem for problem in control["problems"])


def test_the_control_fails_on_a_drifted_cold_row(monkeypatch, tmp_path):
    committed = tmp_path / "warm_start_h4.json"
    committed.write_text(json.dumps(_control_record()), encoding="utf-8")
    monkeypatch.setattr(ws, "CONTROL_RECORD", committed)

    control = ws.check_control(_control_record(cold=9.9))
    assert control["passed"] is False
    assert any("cold row" in problem for problem in control["problems"])


def test_the_control_fails_when_a_committed_rung_is_missing(monkeypatch,
                                                            tmp_path):
    committed = tmp_path / "warm_start_h4.json"
    committed.write_text(json.dumps(_control_record()), encoding="utf-8")
    monkeypatch.setattr(ws, "CONTROL_RECORD", committed)

    partial = _control_record(warm=((2, 0.3423008937817329), (4, 0.6124)))
    control = ws.check_control(partial)
    assert control["passed"] is False
    assert any("k=6" in problem for problem in control["problems"])


def test_a_missing_committed_record_is_reported_not_silently_passed(monkeypatch,
                                                                    tmp_path):
    monkeypatch.setattr(ws, "CONTROL_RECORD", tmp_path / "absent.json")
    control = ws.check_control(_control_record())
    assert control["available"] is False
    assert "not present" in control["note"]


# ------------------------------------------------------------------ invariants


def _document(**overrides):
    row = {"ground_energy": -1.0, "subspace_capture": 0.99,
           "reference_capture": 0.5, "adapt_operators": 2}
    document = {
        "systems": [{"system": "toy", "exact_energy": -1.0,
                     "cold": dict(row, adapt_operators=0),
                     "warm": [dict(row)]}],
        "control": {"available": True, "passed": True, "problems": []},
    }
    document.update(overrides)
    return document


def test_the_gate_accepts_a_well_formed_document():
    ws.check_invariants(_document())


def test_a_row_below_the_variational_bound_is_rejected():
    document = _document()
    document["systems"][0]["warm"][0]["ground_energy"] = -1.5
    with pytest.raises(AssertionError, match="variational bound"):
        ws.check_invariants(document)


def test_a_span_capturing_less_than_its_own_reference_is_rejected():
    """The span contains the reference direction, so this cannot happen."""
    document = _document()
    document["systems"][0]["warm"][0].update({"subspace_capture": 0.2,
                                              "reference_capture": 0.9})
    with pytest.raises(AssertionError, match="captures less"):
        ws.check_invariants(document)


def test_an_out_of_range_capture_is_rejected():
    document = _document()
    document["systems"][0]["warm"][0]["subspace_capture"] = 1.5
    with pytest.raises(AssertionError, match="out-of-range subspace capture"):
        ws.check_invariants(document)


def test_a_failed_positive_control_blocks_the_record():
    document = _document()
    document["control"] = {"available": True, "passed": False,
                           "problems": ["k=2: expected 0.34, got 0.9"]}
    with pytest.raises(AssertionError, match="positive control failed"):
        ws.check_invariants(document)


def test_failed_warm_rows_do_not_trip_the_gate():
    document = _document()
    document["systems"][0]["warm"] = [{"adapt_operators": 4, "failed": "boom"}]
    ws.check_invariants(document)


def test_a_memory_ceiling_is_reported_when_requested():
    limit = ws.apply_memory_limit(0.0)
    assert limit["applied"] is False


def test_a_failed_rung_keeps_its_requested_operator_count():
    """A rung the exact backend cannot carry must stay identifiable."""
    analysis = ws.analyse_anomaly(
        {"error_millihartree": 3.0},
        [_warm(2, 27.0, 0.34), _warm(6, 6.8, 0.77),
         {"requested_operators": 8, "adapt_operators": 8,
          "failed": "MemoryError: "}])
    assert analysis["replicated"] is True
    assert len(analysis["effective_ranks"]) == 2


def test_an_unknown_system_is_refused():
    with pytest.raises(ValueError, match="unknown primary system"):
        ws.run_system("not_a_system")
