"""Contracts for the R3S priceability screen.

The screen decides where a sampling probe is spent, so its failure mode is a
verdict that does not follow from its own declared gates. These tests pin the
preregistration, the two monotonicities the walk's early exit rests on, and the
checker's ability to catch a record whose verdict was written rather than
derived.
"""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.check_priceability_screen import (
    calibration_problems,
    contract_problems,
    omission_problems,
)
from benchmarks.run_priceability_screen import (
    CONFIG,
    REFERENCE,
    candidate_specs,
    load_config,
)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _candidate(record: dict, key: str) -> dict:
    return next(c for c in record["candidates"] if c["candidate"] == key)


def _write(tmp_path, config: dict):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_config_declares_every_gate_with_its_basis():
    config = load_config()
    gates = config["acceptance_gates"]
    assert gates["accuracy_target_millihartree"] == 1.6
    assert gates["margin_factor"] >= 1.0
    assert gates["margin_factor_basis"]
    assert gates["word_universe_ceiling_basis"]
    assert gates["selection_may_not_use_mapping_cost_direction"] is True
    assert gates["screen_authorizes_a_probe_not_a_price"] is True
    assert config["selection_rule"]["minimum_prefix_size"] >= 2
    assert config["selection_rule"]["minimum_prefix_size_basis"]


def test_the_selection_rule_is_a_function_of_the_accuracy_target_alone():
    """The QR3b gate this screen inherits forbids selecting on measurement cost.

    A rule that could see a word count or a mapping arm would be choosing the
    prefix that is cheap to measure, which is the thing that gate exists to
    prevent. The record's per-arm bias agreement is the evidence; this pins the
    declaration that goes with it.
    """
    rule = load_config()["selection_rule"]
    assert "accuracy target alone" in rule["statement"]
    assert "invariant" in rule["mapping_neutrality"]
    assert rule["ordering_provenance"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"acceptance_gates": {"selection_may_not_use_mapping_cost_direction": False}},
        {"acceptance_gates": {"screen_authorizes_a_probe_not_a_price": False}},
        {"acceptance_gates": {"margin_factor": 0.5}},
        {"acceptance_gates": {"margin_factor_basis": ""}},
        {"acceptance_gates": {"word_universe_ceiling_basis": ""}},
        {"selection_rule": {"minimum_prefix_size": 1}},
        {"selection_rule": {"minimum_prefix_size_basis": ""}},
        {"candidates": []},
    ],
)
def test_a_relaxed_preregistration_is_refused_at_load(tmp_path, mutation):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for section, values in mutation.items():
        if isinstance(values, dict):
            config[section].update(values)
        else:
            config[section] = values
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, config))


def test_candidate_specs_come_from_the_preregistrations_that_own_them():
    specs = candidate_specs()
    # LiH keeps the QR3b digests; the R2b banks keep their mapping_axis labels.
    assert specs["lih_cas4e4o"]["source_sha256"]
    assert specs["beh2"]["selected_labels"][0] == "I"


def test_committed_record_satisfies_its_contracts(record):
    assert contract_problems(record) == []
    assert omission_problems(record) == []
    assert calibration_problems(record) == []


def test_the_screen_prices_nothing(record):
    assert record["evidence_tier"] == "structural"
    assert "price C(epsilon)" in record["claim_boundary"]


def test_derived_gates_are_the_declared_arithmetic(record):
    gates = record["acceptance_gates"]
    derived = record["derived_gates"]
    assert derived["admissible_bias_millihartree"] == pytest.approx(
        gates["accuracy_target_millihartree"] / gates["margin_factor"]
    )
    # What the target leaves for shot noise once the bank takes its share.
    assert derived["statistical_allowance_millihartree"] == pytest.approx(1.5085, abs=1e-4)


def test_lih_is_admissible_below_the_single_calibration_point(record):
    """The finding the screen exists to make cheap.

    LiH clears the margin at a two-generator prefix whose binding word universe
    is under BeH2's -- the one bank this repository has priced inside the frozen
    grid -- while the same instance at the greedy's own stopping point sits four
    times above the ceiling.
    """
    lih = _candidate(record, "lih_cas4e4o")
    beh2 = _candidate(record, "beh2")
    stop = lih["margin_stop"]

    assert lih["verdict"]["admissible_for_a_probe"] is True
    assert stop["basis_size"] == 2
    assert stop["bias_millihartree"] < record["derived_gates"]["admissible_bias_millihartree"]
    assert stop["binding_word_universe"] <= beh2["intrinsic_stop"]["binding_word_universe"]
    # It is the stopping rule, not the instance, that the QR3b probe rejected.
    assert lih["intrinsic_stop"]["within_ceiling"] is False
    assert lih["verdict"]["rule_change_is_what_admits_it"] is True


def test_h4_converged_is_rejected_under_both_rules(record):
    """The frozen deferral was right for H4, and stays right under the new rule."""
    h4 = _candidate(record, "h4_converged")
    assert h4["verdict"]["admissible_for_a_probe"] is False
    assert h4["intrinsic_stop"]["within_ceiling"] is False
    # Its best bias over the whole frozen ordering still misses the margin, so
    # this is not a case a longer walk would have rescued.
    assert h4["intrinsic_stop"]["bias_millihartree"] > (
        record["derived_gates"]["admissible_bias_millihartree"]
    )


def test_beh2_stops_earlier_under_the_margin_rule_than_the_greedy_did(record):
    beh2 = _candidate(record, "beh2")
    assert beh2["verdict"]["admissible_for_a_probe"] is True
    assert beh2["margin_stop"]["basis_size"] < beh2["intrinsic_stop"]["basis_size"]
    # BeH2 was already priceable, so the rule change is not what admits it.
    assert beh2["verdict"]["rule_change_is_what_admits_it"] is False


def test_every_walk_is_monotone_in_both_gated_quantities(record):
    for candidate in record["candidates"]:
        walk = candidate["prefix_walk"]
        biases = [row["bias_millihartree"] for row in walk]
        words = [row["source_word_universe"] for row in walk]
        assert biases == sorted(biases, reverse=True), candidate["candidate"]
        assert words == sorted(words), candidate["candidate"]


def test_a_hand_written_verdict_is_caught(record):
    broken = copy.deepcopy(record)
    _candidate(broken, "h4_converged")["verdict"]["admissible_for_a_probe"] = True
    assert any("does not follow" in p for p in contract_problems(broken))


def test_a_non_smallest_margin_stop_is_caught(record):
    broken = copy.deepcopy(record)
    beh2 = _candidate(broken, "beh2")
    beh2["prefix_walk"][0]["clears_margin"] = True
    beh2["prefix_walk"][0]["bias_millihartree"] = 0.0
    assert any("is not the smallest one" in p for p in contract_problems(broken))


def test_a_binding_word_universe_that_is_not_the_maximum_is_caught(record):
    broken = copy.deepcopy(record)
    stop = _candidate(broken, "lih_cas4e4o")["margin_stop"]
    stop["binding_word_universe"] = min(a["word_universe"] for a in stop["arms"])
    assert any("is not the maximum over arms" in p for p in contract_problems(broken))


def test_a_prefix_chosen_on_one_arm_is_caught(record):
    """If the arms disagree on the bias, the prefix cannot have been arm-blind."""
    broken = copy.deepcopy(record)
    _candidate(broken, "lih_cas4e4o")["margin_stop"]["arms"][0][
        "bias_millihartree"
    ] += 0.25
    assert any("not encoding-blind" in p for p in contract_problems(broken))


def test_a_rising_bias_walk_is_caught(record):
    broken = copy.deepcopy(record)
    walk = _candidate(broken, "h4_converged")["prefix_walk"]
    walk[-1]["bias_millihartree"] = walk[0]["bias_millihartree"] + 1.0
    assert any("bias rose along the greedy prefix" in p for p in contract_problems(broken))


def test_a_shrinking_word_universe_walk_is_caught(record):
    broken = copy.deepcopy(record)
    walk = _candidate(broken, "h4_converged")["prefix_walk"]
    walk[-1]["source_word_universe"] = 1
    walk[-1]["within_ceiling"] = True
    assert any("word universe fell" in p for p in contract_problems(broken))


def test_a_leaked_cost_field_is_caught(record):
    broken = copy.deepcopy(record)
    _candidate(broken, "beh2")["margin_stop"]["k_star"] = [1, 2]
    assert any("cost-record field" in p for p in contract_problems(broken))


def test_the_h4_omission_is_verified_against_the_frozen_labels(record):
    """The omission is a claim about two label lists, so it is checked as one."""
    assert "h4" in record["omitted_candidates"]
    specs = candidate_specs()
    short = specs["h4"]["selected_labels"]
    assert specs["h4_converged"]["selected_labels"][: len(short)] == short


def test_the_calibration_point_binds_the_ceiling_to_the_r2b_record(record):
    broken = copy.deepcopy(record)
    _candidate(broken, "beh2")["intrinsic_stop"]["arms"][0]["word_universe"] = 4096
    assert any("the ceiling is calibrated on" in p for p in calibration_problems(broken))


def test_the_word_universe_convention_matches_the_record_the_ceiling_came_from(record):
    """Two conventions live in the tree and they differ by one.

    ``mapping_axis`` counts the identity word, ``protocol_axis`` and the QR3b
    probe do not, and the 2048 ceiling was calibrated in the second. Counting in
    the first would inflate every candidate by one against a gate that never
    included it.
    """
    assert record["word_universe_convention"] == "non_identity_words_only"
    gates = record["acceptance_gates"]
    assert gates["word_universe_convention"] == "non_identity_words_only"
    assert gates["word_universe_convention_basis"]

    frozen = json.loads(
        (REFERENCE.parent / "protocol_axis.json").read_text(encoding="utf-8")
    )
    beh2_frozen = {
        arm["mapping"]: arm["word_universe"]
        for system in frozen["systems"]
        if system["system"] == "beh2"
        for arm in system["arms"]
    }
    assert beh2_frozen["jw"] == 1814
    for arm in _candidate(record, "beh2")["intrinsic_stop"]["arms"]:
        assert arm["word_universe"] == beh2_frozen[arm["mapping"]]


def test_the_binding_arm_rule_is_what_rejects_h4_converged(record):
    """Its reduced arms fit under the ceiling; its full-width arms do not.

    h4_converged is the case that makes the max-over-arms rule load-bearing
    rather than bookkeeping: at 2047 the two ``+2q`` arms sit one word under the
    gate, so a screen reading any single reduced arm would have admitted a bank
    the frozen grid has already failed to resolve.
    """
    intrinsic = _candidate(record, "h4_converged")["intrinsic_stop"]
    ceiling = record["acceptance_gates"]["word_universe_ceiling"]
    by_arm = {arm["mapping"]: arm["word_universe"] for arm in intrinsic["arms"]}

    assert by_arm["parity+2q"] < ceiling
    assert by_arm["bk+2q"] < ceiling
    assert by_arm["jw"] > ceiling
    assert intrinsic["binding_word_universe"] == max(by_arm.values())
    assert intrinsic["within_ceiling"] is False


def test_the_walk_and_the_verdict_gate_on_the_same_word_count(record):
    """The early exit reads the source-side count; the verdict reads the binding one.

    They coincide because a pure encoding preserves the word universe and a
    ``+2q`` arm can only shrink it. If that stopped holding, a candidate could be
    rejected early on a count no arm actually carries, so the checker ties them
    together rather than leaving the equality to the encoding contract.
    """
    for candidate in record["candidates"]:
        stop = candidate["margin_stop"]
        if stop is None:
            continue
        assert stop["binding_word_universe"] == candidate["prefix_walk"][-1][
            "source_word_universe"
        ], candidate["candidate"]

    broken = copy.deepcopy(record)
    stop = _candidate(broken, "beh2")["margin_stop"]
    for arm in stop["arms"]:
        arm["word_universe"] = 7
    stop["binding_word_universe"] = 7
    stop["binding_arms"] = sorted(a["mapping"] for a in stop["arms"])
    assert any("must be the same quantity" in p for p in contract_problems(broken))
