"""Contracts for the Phase 2M-B packed-storage record.

The quiet failures here are all ways of reporting a reduction that is not one.
A byte ratio taken from a bank whose matrix elements moved measures a different
bank. A ratio graded on the five convenient banks -- reuse 1.5 to 2.5, against
the 35 to 155 of the banks that actually ran out of memory -- understates the
gate by more than a factor of two. A net-loss list that stopped tracking the
measurement, in either direction, would describe a different phase. A packed
side that skipped an allocation the object side is charged for would report a
reduction that is an accounting artifact. And a verdict written by hand would
turn one graded clause of a three-clause go/no-go into "Phase 2M passes".

These tests read the committed record rather than rebuilding it -- the producer
prices ten banks under three backends and takes minutes -- and pin the contract
functions the checker uses, plus the failures each is there to catch.
"""

from __future__ import annotations

import copy
import json

import pytest

from clifford_qc.subspace.packed import LAYOUTS, PACKED_BYTES_PER_COEFFICIENT
from clifford_qc.subspace.projection import STORAGE_BACKENDS

from benchmarks.check_packed_bank_storage import (
    INTERPRETER_DEPENDENT_FIELDS,
    INTERPRETER_RTOL,
    byte_problems,
    committed_reuse_problems,
    contract_problems,
    equivalence_problems,
    interpreter_field_problems,
    layout_problems,
    model_problems,
    verdict_problems,
)
from benchmarks.run_packed_bank_storage import (
    CONFIG,
    GO_NO_GO_THRESHOLD,
    GRADED_CLAUSE,
    REFERENCE,
    load_config,
)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _broken(record: dict, mutate) -> dict:
    copied = copy.deepcopy(record)
    mutate(copied)
    return copied


# --- The declaration

def test_config_and_producer_agree():
    config = load_config()
    assert config["packed_bytes_per_coefficient"] == PACKED_BYTES_PER_COEFFICIENT
    assert tuple(config["layouts"]) == LAYOUTS
    assert config["go_no_go"]["threshold"] == GO_NO_GO_THRESHOLD
    assert config["go_no_go"]["graded_clause"] == GRADED_CLAUSE
    assert config["equivalence_gate"]["required"] is True
    assert config["claim_boundary"].strip()


def test_config_refuses_an_unasserted_equivalence_gate(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["equivalence_gate"]["required"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="equivalence gate"):
        load_config(path)


def test_config_refuses_a_softened_threshold(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["go_no_go"]["threshold"] = 1.5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="grades against"):
        load_config(path)


# --- The record

def test_committed_record_passes_every_contract(record):
    assert contract_problems(record) == []


def test_the_object_backend_is_still_the_default(record):
    """2M-A's baseline prices it, so 2M-B may not move it before being graded."""
    assert record["default_storage_backend"] == "object"
    assert tuple(record["storage_backends"]) == STORAGE_BACKENDS


def test_equivalence_holds_on_every_priced_bank(record):
    for row in record["banks"]:
        verdict = row["equivalence"]
        assert verdict["overlap_matrix_bitwise_equal"], row["bank"]
        assert verdict["hamiltonian_matrix_bitwise_equal"], row["bank"]
        assert verdict["basis_labels_equal"], row["bank"]
        assert verdict["mismatched_fields"] == [], row["bank"]


def test_packed_rows_hit_the_model_on_every_bank(record):
    for row in record["banks"]:
        for layout, priced in row["layouts"].items():
            assert priced["row_bytes"] == (
                PACKED_BYTES_PER_COEFFICIENT * row["coefficient_occurrences"]), (
                f"{row['bank']}/{layout}")


def test_both_layouts_hold_identical_bytes(record):
    assert record["layout_comparison"]["byte_identical_on_every_bank"]
    assert record["layout_comparison"]["banks_where_bytes_differ"] == []


def test_the_priced_set_spans_coefficient_reuse(record):
    """The whole finding is that the reduction is a function of reuse."""
    model = record["reduction_model"]
    required = float(load_config()["banks"]["reuse_span_requirement"])
    assert model["reuse_span_factor"] >= required
    assert model["measured_reuse_range"][0] < 2.0
    assert model["measured_reuse_range"][1] > 40.0


def test_the_threshold_is_separated(record):
    """The lowest reuse that clears the gate is above the highest that fails."""
    model = record["reduction_model"]
    assert model["lowest_reuse_clearing_threshold"] is not None
    assert model["highest_reuse_failing_threshold"] is not None
    assert (model["lowest_reuse_clearing_threshold"]
            > model["highest_reuse_failing_threshold"])


def test_net_loss_banks_are_reported_exactly(record):
    """A bank where packing costs more must be named, and one that does not must not.

    The list was non-empty before the word table moved off Python objects --
    ``hubbard_2x2`` at reuse 1.51 measured 0.94x, worse than what it replaced.
    It is empty now. The contract is that the list tracks the measurements
    either way: a record carrying only the wins would be describing a different
    phase, and one inventing a loss would be describing a different measurement.
    """
    reported = record["reduction_model"]["banks_where_packing_costs_more"]
    measured = [row["bank"] for row in record["banks"]
                if row["layouts"]["interleaved"]["reduction"] < 1.0]
    assert reported == measured


def test_no_bank_is_a_net_loss_after_the_table_moved_off_python_objects(record):
    """What shrinking the word table bought at the bottom of the ladder."""
    for row in record["banks"]:
        for layout, priced in row["layouts"].items():
            assert priced["reduction"] > 1.0, f"{row['bank']}/{layout}"


def test_the_shared_word_code_integers_are_charged_on_the_packed_side(record):
    """They are resident under either backend, so both must pay for them once."""
    for row in record["banks"]:
        for layout, priced in row["layouts"].items():
            assert priced["shared_word_code_bytes"] > 0, f"{row['bank']}/{layout}"
            assert (priced["row_bytes"] + priced["word_table_bytes"]
                    + priced["shared_word_code_bytes"]) == priced["total_bytes"]


def test_the_committed_banks_all_exceed_the_threshold(record):
    model = record["reduction_model"]
    floor = min(record["committed_bank_reuse"]["ratio_range"])
    assert floor >= model["lowest_reuse_clearing_threshold"]


def test_the_committed_rows_are_quoted_with_their_caveat(record):
    """They carry selected-subspace W, so they bound reuse rather than state it."""
    committed = record["committed_bank_reuse"]
    assert committed["source_record"] == "bank_storage_ledger.json"
    assert "upper bound" in committed["caveat"] or "no greater" in committed["caveat"]


def test_exactly_one_go_no_go_clause_is_graded(record):
    verdict = record["go_no_go"]
    assert verdict["graded_clause"] == GRADED_CLAUSE
    assert len(verdict["ungraded_clauses"]) == 2
    assert all(reason.strip() for reason in verdict["ungraded_clauses"].values())
    assert verdict["what_this_does_not_establish"].strip()


def test_the_verdict_does_not_claim_phase_2m_passes(record):
    assert record["go_no_go"]["outcome"] == (
        "reached_above_a_measured_reuse_threshold_committed_banks_exceed_it")
    assert "Not that Phase 2M passes" in record["go_no_go"][
        "what_this_does_not_establish"]


# --- The checker catches each failure

@pytest.mark.parametrize("name,mutate,gate", [
    ("a silently failed equivalence",
     lambda r: r["banks"][0]["equivalence"].update(overlap_matrix_bitwise_equal=False),
     equivalence_problems),
    ("a diverged structural field",
     lambda r: r["banks"][0]["equivalence"].update(mismatched_fields=["word_universe"]),
     equivalence_problems),
    ("rows plus table that miss the total",
     lambda r: r["banks"][0]["layouts"]["interleaved"].update(word_table_bytes=1),
     byte_problems),
    ("a reduction that is not the quotient",
     lambda r: r["banks"][0]["layouts"]["interleaved"].update(reduction=99.0),
     byte_problems),
    ("reserved bytes below live bytes",
     lambda r: r["banks"][0]["layouts"]["interleaved"].update(reserved_bytes=1),
     byte_problems),
    ("a faked layout byte-equality verdict",
     lambda r: r["layout_comparison"].update(byte_identical_on_every_bank=False),
     layout_problems),
    # Inventing a loss rather than hiding one: the measured list is empty now
    # that the word table is numpy-backed, so clearing it would be a no-op and
    # would test nothing. The contract is that the list tracks the measurement
    # in both directions.
    ("an invented net-loss bank",
     lambda r: r["reduction_model"].update(
         banks_where_packing_costs_more=["h4_M53_full"]),
     model_problems),
    ("an understated reuse span",
     lambda r: r["reduction_model"].update(reuse_span_factor=1.0),
     model_problems),
    ("a hand-moved threshold",
     lambda r: r["reduction_model"].update(lowest_reuse_clearing_threshold=1.0),
     model_problems),
    ("a hand-written verdict",
     lambda r: r["go_no_go"].update(outcome="passes"),
     verdict_problems),
    ("a record grading all three clauses",
     lambda r: r["go_no_go"].update(ungraded_clauses={}),
     verdict_problems),
    ("committed rows quoted without their caveat",
     lambda r: r["committed_bank_reuse"].pop("caveat"),
     committed_reuse_problems),
])
def test_checker_catches(record, name, mutate, gate):
    assert gate(_broken(record, mutate)) != [], name


def test_checker_catches_a_hand_written_threshold_flag(record):
    """The flag follows from the measured reduction, not from the author."""
    broken = _broken(record, lambda r: r["banks"][4]["layouts"]["interleaved"].update(
        clears_threshold=True))
    assert any("not writable by hand" in problem for problem in byte_problems(broken))


def test_checker_catches_a_flipped_default_backend(record):
    assert contract_problems(
        _broken(record, lambda r: r.update(default_storage_backend="packed"))) != []


def test_checker_catches_a_cost_field(record):
    assert contract_problems(
        _broken(record, lambda r: r["go_no_go"].update(k_star=4))) != []


def test_checker_catches_an_upgraded_evidence_tier(record):
    assert contract_problems(
        _broken(record, lambda r: r.update(evidence_tier="exact"))) != []


def test_checker_survives_a_malformed_record(record):
    problems = contract_problems(_broken(record, lambda r: r.update(banks="x")))
    assert any("malformed record" in problem for problem in problems)


# --- The interpreter tolerance

def test_measured_bytes_are_excluded_from_the_exact_comparison():
    for field in ("object_bytes", "total_bytes", "reduction",
                  "word_table_bytes", "reuse_span_factor"):
        assert field in INTERPRETER_DEPENDENT_FIELDS


def test_interpreter_tolerance_admits_drift_and_rejects_a_regression(record):
    inside = _broken(record, lambda r: r["banks"][0].update(
        object_bytes=int(r["banks"][0]["object_bytes"] * (1 + INTERPRETER_RTOL / 4))))
    assert interpreter_field_problems(record, inside) == []
    outside = _broken(record, lambda r: r["banks"][0].update(
        object_bytes=int(r["banks"][0]["object_bytes"] * (1 + 10 * INTERPRETER_RTOL))))
    assert interpreter_field_problems(record, outside) != []
