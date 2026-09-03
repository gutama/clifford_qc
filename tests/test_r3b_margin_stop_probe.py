"""Contracts for the R3b margin-stop probe record.

The probe rejected the bank, and the interesting question is *why*. Its gate
fails on two unrelated things -- a crossing that does not fit inside the frozen
grid, and a crossing the confirmatory replicas do not reproduce -- and only the
first implicates the word-universe ceiling R3S admitted this bank on. Most of
what follows pins that distinction, because collapsing it is how an underpowered
probe would come to read as a refuted screen.
"""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.check_r3b_margin_stop_probe import (
    contract_problems,
    qr3b_untouched_problems,
)
from benchmarks.check_r3b_preregistration import load_config
from benchmarks.run_r3b_margin_stop_probe import (
    QR3B_RECORD,
    REFERENCE,
    screen_prediction,
)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def qr3b() -> dict:
    return json.loads(QR3B_RECORD.read_text(encoding="utf-8"))


def test_the_committed_record_satisfies_its_contracts(record):
    assert contract_problems(record) == []
    assert qr3b_untouched_problems() == []


def test_the_bank_is_the_preregistered_one(record):
    config = load_config()
    selection = record["selection"]
    assert selection["labels"] == config["candidate"]["selected_labels"]
    assert selection["basis_size"] == 2
    assert selection["source_word_universe"] == 1439
    assert selection["stopping_reason"] == "smallest prefix clearing the accuracy margin"
    # Shorter than the bank QR3b probed, which is the point of the phase.
    assert selection["basis_size"] < selection["intrinsic_stop_basis_size"] == 13


def test_the_probe_rejects_the_bank_and_authorizes_nothing(record):
    """The preregistered gate is not relaxed by a near miss."""
    decision = record["decision"]
    assert decision["status"] == "rejected_unresolved_at_frozen_grid"
    assert decision["resolution_gate_passes"] is False
    assert decision["eligible_for_full_run"] is False
    assert decision["full_run_authorized"] is False
    assert decision["qr3b_verdict"] == "not_evaluated_by_this_probe"


def test_the_bias_and_word_universe_gates_both_passed(record):
    """So the rejection isolates resolution, as QR3b's did."""
    decision = record["decision"]
    assert decision["bias_gate_passes"] is True
    assert decision["word_universe_gate_passes"] is True
    assert decision["maximum_word_universe"] == 1439 <= decision["word_universe_ceiling"]


def test_the_grid_fit_failure_mode_is_gone(record, qr3b):
    """The finding. QR3b failed on grid fit; this bank does not fail on it at all.

    ``not_bracketed_within_search_grid`` means the crossing lies above the
    frozen grid -- the failure the word-universe ceiling is a proxy for. It
    accounts for 15 of QR3b's 22 unresolved cells and none of R3b's.
    """
    def statuses(rec):
        out = {}
        for row in rec["decision"]["unresolved_coordinates"]:
            out[row["status"]] = out.get(row["status"], 0) + 1
        return out

    assert statuses(qr3b).get("not_bracketed_within_search_grid") == 15
    assert statuses(record).get("not_bracketed_within_search_grid") is None
    assert record["decision"]["cells_at_or_beyond_ceiling"] == 1
    assert qr3b["decision"]["cells_at_or_beyond_ceiling"] == 9
    assert record["decision"]["resolved_cells"] > qr3b["decision"]["resolved_cells"]


def test_the_verdict_separates_grid_fit_from_confirmation_power(record):
    prediction = record["screen_prediction"]
    assert prediction["verdict"] == "corroborated_on_grid_fit_headroom_marginal"
    assert prediction["grid_fit_failures"] == 0
    assert prediction["unclassified_failures"] == 0
    assert prediction["confirmation_failures"] == record["decision"]["unresolved_cells"]
    assert prediction["grid_fit_holds"] is True
    # One cell bracketed only at the last grid point: short of a pass, and not
    # the failure mode the ceiling predicts.
    assert prediction["cells_at_or_beyond_ceiling"] == 1
    # And it does not read as a licence to relax the gate.
    assert "not relaxed here" in prediction["what_follows"]
    assert "new preregistration" in prediction["what_follows"]


def test_the_failure_mode_split_is_labelled_post_hoc(record):
    """It was written after seeing the cells, and says so.

    The preregistration declared a corroborate/falsify binary. Refining that
    binary after the draw is legitimate only if it is labelled, and only if it
    leaves the preregistered decision alone -- which the test above checks by
    pinning the rejection.
    """
    prediction = record["screen_prediction"]
    assert prediction["classification_status"] == "post_hoc_diagnostic_not_preregistered"
    assert "after seeing them" in prediction["classification_status_basis"]
    assert "unchanged" in prediction["classification_status_basis"]


def test_a_marginal_headroom_cell_does_not_read_as_falsification(record):
    """The distinction the corrected taxonomy exists to make.

    A cell that resolves at the last grid point *did* bracket a crossing inside
    the grid, which is what the ceiling predicts. Folding it into falsification
    would let one marginal cell outweigh fifteen bracketing failures going to
    zero.
    """
    prediction = record["screen_prediction"]
    assert prediction["cells_at_or_beyond_ceiling"] > 0
    assert prediction["grid_fit_holds"] is True
    assert prediction["verdict"] != "falsified_on_its_first_admission"


def test_a_grid_fit_failure_would_falsify_the_ceiling(record):
    """The verdict that the committed evidence does not support, kept reachable.

    If a cell had failed to bracket a crossing, the ceiling would have admitted
    a bank whose crossings do not fit the grid, and that is the falsification.
    """
    broken = copy.deepcopy(record)
    broken["decision"]["unresolved_coordinates"][0]["status"] = (
        "not_bracketed_within_search_grid"
    )
    prediction = screen_prediction(broken["decision"], broken["selection"])
    assert prediction["verdict"] == "falsified_on_its_first_admission"
    assert prediction["grid_fit_holds"] is False


def test_a_cell_resolving_at_the_ceiling_does_not_break_grid_fit(record):
    """Bracketing at the last point is still bracketing."""
    marginal = copy.deepcopy(record)
    marginal["decision"]["unresolved_coordinates"] = []
    marginal["decision"]["cells_at_or_beyond_ceiling"] = 3
    prediction = screen_prediction(marginal["decision"], marginal["selection"])
    assert prediction["grid_fit_holds"] is True
    assert prediction["verdict"] == "corroborated_on_grid_fit_headroom_marginal"


def test_no_marginal_cell_would_read_as_underpowered_only(record):
    clean = copy.deepcopy(record)
    clean["decision"]["cells_at_or_beyond_ceiling"] = 0
    prediction = screen_prediction(clean["decision"], clean["selection"])
    assert prediction["verdict"] == "corroborated_on_grid_fit_probe_underpowered"


def test_full_resolution_would_corroborate_outright(record):
    clean = copy.deepcopy(record)
    clean["decision"]["unresolved_coordinates"] = []
    clean["decision"]["cells_at_or_beyond_ceiling"] = 0
    clean["decision"]["resolution_gate_passes"] = True
    prediction = screen_prediction(clean["decision"], clean["selection"])
    assert prediction["verdict"] == "corroborated"


def test_an_unknown_failure_status_is_not_silently_absorbed(record):
    """A status neither bucket recognises must not read as grid fit holding."""
    broken = copy.deepcopy(record)
    broken["decision"]["unresolved_coordinates"][0]["status"] = "something_new"
    prediction = screen_prediction(broken["decision"], broken["selection"])
    assert prediction["unclassified_failures"] == 1
    assert prediction["grid_fit_holds"] is False
    assert prediction["verdict"] == "falsified_on_its_first_admission"


def test_the_screen_prediction_must_rederive(record):
    broken = copy.deepcopy(record)
    broken["screen_prediction"]["verdict"] = "corroborated"
    assert any("does not follow" in p for p in contract_problems(broken))


def test_a_hand_written_decision_is_caught(record):
    broken = copy.deepcopy(record)
    broken["decision"]["resolution_gate_passes"] = True
    assert any("does not rederive" in p for p in contract_problems(broken))


def test_an_authorized_full_run_is_caught(record):
    broken = copy.deepcopy(record)
    broken["decision"]["full_run_authorized"] = True
    assert any("silently authorizes" in p for p in contract_problems(broken))


def test_qr3b_hindsight_field_may_not_reappear(record):
    """QR3b recorded that the screen should have run first. Here it did."""
    assert "screen_would_have_rejected_before_probe" not in record["decision"]
    broken = copy.deepcopy(record)
    broken["decision"]["screen_would_have_rejected_before_probe"] = False
    assert any("hindsight field" in p for p in contract_problems(broken))


def test_the_protocol_is_qr3b_s_so_the_two_records_compare(record, qr3b):
    mine, theirs = record["protocol"], qr3b["protocol"]
    for field in ("block_sizes", "estimators", "exploratory_replicas",
                  "confirmatory_replicas",
                  "search_endpoints_effective_shots_per_setting"):
        assert mine[field] == theirs[field], field
    assert not set(mine["seed_roots"].values()) & set(theirs["seed_roots"].values())


def test_the_record_carries_no_cost_derivative(record):
    assert contract_problems(record) == []
    for forbidden in ("k_star", "cost_bracket", "device_costs", "C_time"):
        assert forbidden not in json.dumps(record)
