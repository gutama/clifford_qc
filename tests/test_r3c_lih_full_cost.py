"""Contracts for the R3c LiH full-cost producer, before its record exists.

The producer lands ahead of the run it draws, so there is no committed record to
compare against yet and these tests hold the two things that do not need one:
that every gate standing between the preregistration and forty priced cells
actually refuses what it claims to refuse, and that the pricing and QR3 layers
read the frozen rules out of the config rather than restating them.

The BeH2 system in ``protocol_cost.json`` stands in for the sampled evidence.
It is the one bank this repository has priced at the exact tier, drawn by the
same producer over the same grid at the same 30+100 replicas, so it has exactly
the shape R3c's own system record will have -- and its 40 finite intervals and
its 6 cells inadmissible on one card exercise both sides of the pricing table.
"""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.check_r3c_preregistration import load_config
from benchmarks.run_mapping_axis import load_device_cards
import benchmarks.run_r3c_lih_full_cost as producer
from benchmarks.run_r3c_lih_full_cost import (
    FINITE_INTERVAL,
    FIRST_PRICED_INSTANCE,
    NO_CONFIRMED_CROSSING,
    OPEN_ABOVE,
    OPEN_BELOW,
    OPEN_BOTH,
    PROTOCOL_COST_RECORD,
    R3B_RECORD,
    SCHEMA,
    _classify,
    bank_problems,
    build_record,
    environment_problems,
    instrument_problems,
    preregistration_problems,
    pricing_decision,
    qr3_reference_problems,
    qr3_second_instance,
    r3b_untouched_problems,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def cards():
    return load_device_cards()


@pytest.fixture(scope="module")
def priced_system() -> dict:
    """The frozen BeH2 cost cells, as a stand-in for a run that resolves."""
    return json.loads(PROTOCOL_COST_RECORD.read_text(encoding="utf-8"))["systems"][
        FIRST_PRICED_INSTANCE
    ]


@pytest.fixture(scope="module")
def structural_record() -> dict:
    """The producer's structural half: every gate, no replicas."""
    return build_record(run=False)


def _censor(system: dict) -> dict:
    """The same cells with no confirmed crossing anywhere in the grid."""
    censored = copy.deepcopy(system)
    for arm in censored["arms"]:
        for rung in arm["rungs"]:
            for payload in rung["estimators"].values():
                payload["shot_to_target"] = {
                    "status": "not_bracketed_within_search_grid",
                    "confirmed_failing_effective_shots_per_setting": None,
                    "confirmed_passing_effective_shots_per_setting": None,
                }
                payload["cost_bracket"] = {
                    "status": "not_bracketed_within_search_grid",
                    "priced": False,
                    "unbounded_below": True,
                    "unbounded_above": True,
                    "widened_sides": [],
                    "cards": {},
                }
    return censored


# --- the gates that stand between the preregistration and the shots ----------


def test_the_committed_preregistration_passes_every_gate(config, cards):
    assert preregistration_problems(config) == []
    assert r3b_untouched_problems() == []
    assert qr3_reference_problems(config, cards) == []


def test_a_preregistration_edited_after_landing_is_refused(config):
    """``static_problems`` validates whatever it is handed; the digest does not."""
    original = producer.FROZEN_CONFIG_SHA256["r3c_lih_full_cost"]
    producer.FROZEN_CONFIG_SHA256["r3c_lih_full_cost"] = "0" * 64
    try:
        problems = preregistration_problems(config)
    finally:
        producer.FROZEN_CONFIG_SHA256["r3c_lih_full_cost"] = original
    assert any("edited after the fact" in problem for problem in problems)


def test_a_drifted_protocol_is_refused_by_the_result_free_checker(config):
    edited = copy.deepcopy(config)
    edited["protocol"]["exploratory_replicas"] = 10
    assert any(
        "exploratory_replicas" in problem for problem in preregistration_problems(edited)
    )


def test_the_running_stack_is_the_one_the_seed_roots_name(config):
    """Stricter than the stamp guard, and about the sampling stream."""
    assert environment_problems(config) == []
    drifted = copy.deepcopy(config)
    drifted["protocol"]["execution_environment"]["numpy"] = "2.4.6"
    problems = environment_problems(drifted)
    assert len(problems) == 1
    assert "different stream" in problems[0]


def test_a_pilot_that_authorized_its_own_successor_is_refused(tmp_path, monkeypatch):
    """R3c's authorization is its config's, never a promotion of R3b's rejection."""
    moved = json.loads(R3B_RECORD.read_text(encoding="utf-8"))
    moved["decision"]["full_run_authorized"] = True
    path = tmp_path / "r3b_margin_stop_probe.json"
    path.write_text(json.dumps(moved), encoding="utf-8")
    monkeypatch.setattr(producer, "R3B_RECORD", path)
    assert any(
        "may not be sourced from the pilot" in problem
        for problem in r3b_untouched_problems()
    )


def test_a_pilot_that_stopped_rejecting_the_bank_is_refused(tmp_path, monkeypatch):
    moved = json.loads(R3B_RECORD.read_text(encoding="utf-8"))
    moved["decision"]["status"] = "eligible_for_separately_authorized_full_run"
    path = tmp_path / "r3b_margin_stop_probe.json"
    path.write_text(json.dumps(moved), encoding="utf-8")
    monkeypatch.setattr(producer, "R3B_RECORD", path)
    assert r3b_untouched_problems() == ["R3b's recorded rejection changed; R3c may not reopen it"]


@pytest.mark.parametrize("field", ["exploratory_replicas", "confirmatory_replicas"])
def test_the_first_priced_instance_must_share_the_replica_counts(config, cards, field):
    """Two costs are comparable only if one instrument measured both."""
    edited = copy.deepcopy(config)
    edited["protocol"][field] = 2
    problems = qr3_reference_problems(edited, cards)
    assert any("not the same instrument" in problem for problem in problems)


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("estimators", ["pooled"], "different estimators"),
        ("search_endpoints_effective_shots_per_setting", [64], "different endpoint grid"),
        ("exploratory_seed", 132813000, "aliases"),
    ],
)
def test_a_first_instance_drawn_by_another_instrument_is_refused(
    config, cards, tmp_path, monkeypatch, field, value, expected
):
    """The grid and the estimator pair are the sampler's own constants.

    So they are checked against the frozen record rather than against the
    config, which the preregistration checker separately binds to the same
    constants. Drifting the record is what exercises them.
    """
    moved = json.loads(PROTOCOL_COST_RECORD.read_text(encoding="utf-8"))
    moved["protocol"][field] = value
    path = tmp_path / "protocol_cost.json"
    path.write_text(json.dumps(moved), encoding="utf-8")
    monkeypatch.setattr(producer, "PROTOCOL_COST_RECORD", path)
    assert any(expected in problem for problem in qr3_reference_problems(config, cards))


def test_a_first_instance_that_no_longer_prices_is_refused(config, cards, tmp_path, monkeypatch):
    moved = json.loads(PROTOCOL_COST_RECORD.read_text(encoding="utf-8"))
    moved["systems"]["beh2"]["status"] = "bias_floor_exceeds_target"
    path = tmp_path / "protocol_cost.json"
    path.write_text(json.dumps(moved), encoding="utf-8")
    monkeypatch.setattr(producer, "PROTOCOL_COST_RECORD", path)
    assert any(
        "no longer prices BeH2" in problem
        for problem in qr3_reference_problems(config, cards)
    )


def test_seed_roots_that_alias_the_frozen_stream_are_refused(config, cards):
    edited = copy.deepcopy(config)
    frozen = json.loads(PROTOCOL_COST_RECORD.read_text(encoding="utf-8"))["protocol"]
    edited["protocol"]["seed_roots"]["exploratory"] = frozen["exploratory_seed"]
    assert any("aliases" in problem for problem in qr3_reference_problems(edited, cards))


def test_device_cards_must_be_the_three_the_price_was_taken_under(config, cards):
    assert any(
        "different device cards" in problem
        for problem in qr3_reference_problems(config, cards[:2])
    )


def test_a_drifted_arm_is_caught_before_the_search_starts(config):
    """The margin rule is a source-side statement; these are the measured ones."""
    edited = copy.deepcopy(config)
    edited["candidate"]["expected_arm_settings"]["jw"] = 350
    edited["candidate"]["expected_arm_word_universe"]["bk"] = 7
    mapping = {
        "arms": [
            {"mapping": name, "measurement": {"settings": settings}}
            for name, settings in config["candidate"]["expected_arm_settings"].items()
        ]
    }
    spec = dict(
        config["candidate"], expected_selected_energy=0.0, selected_energy_tolerance=1.0
    )
    problems = bank_problems(edited, mapping, spec)
    assert any("349 settings against the declared 350" in p for p in problems)
    assert any("W=1439 against the declared 7" in p for p in problems)


def test_the_declared_bank_passes_its_own_arm_gates(config, structural_record):
    mapping = structural_record["mapping_preflight"]
    spec = dict(
        config["candidate"],
        expected_selected_energy=(
            structural_record["selection"]["exact_sector_energy"]
            + structural_record["selection"]["bias_millihartree"] * 1e-3
        ),
        selected_energy_tolerance=1.0,
    )
    assert bank_problems(config, mapping, spec) == []


def test_cells_drawn_at_another_replica_count_are_refused(config, priced_system):
    assert instrument_problems(priced_system, config) == []
    short = copy.deepcopy(priced_system)
    short["arms"][0]["rungs"][0]["estimators"]["pooled"]["confirmation"][0][
        "replicas"
    ] = 99
    assert any("not the declared 100" in p for p in instrument_problems(short, config))


# --- the pricing rule --------------------------------------------------------


@pytest.mark.parametrize(
    "bracket, expected",
    [
        ({"priced": False}, NO_CONFIRMED_CROSSING),
        ({"priced": True, "unbounded_below": False, "unbounded_above": False}, FINITE_INTERVAL),
        ({"priced": True, "unbounded_below": True, "unbounded_above": False}, OPEN_BELOW),
        ({"priced": True, "unbounded_below": False, "unbounded_above": True}, OPEN_ABOVE),
        ({"priced": True, "unbounded_below": True, "unbounded_above": True}, OPEN_BOTH),
    ],
)
def test_every_bracket_shape_has_its_own_name(bracket, expected):
    """A finite interval needs both endpoints; the three ways short of it differ."""
    assert _classify(bracket) == expected


def test_a_resolving_draw_prices_the_second_instance(config, priced_system):
    pricing = pricing_decision(priced_system, config)
    assert pricing["status"] == "priced_second_instance"
    assert pricing["evaluated_cells"] == 40
    assert pricing["cells_with_a_finite_interval"] == 40
    assert pricing["every_cell_priced"] is True
    assert pricing["right_censored_cells"] == 0
    assert pricing["confirmatory_solves_attempted"] == 18400
    assert pricing["confirmatory_solve_failures"] == 0
    assert pricing["cells_open_above_at_the_grid_ceiling"] == 0
    assert pricing["cells_open_below_without_a_confirmed_failure"] == 0
    # Admissibility is per card, so a priced cell is not priced everywhere.
    assert pricing["priced_cells_by_card"] == {
        "ion-like": 40,
        "logical-alltoall": 40,
        "superconducting-like": 34,
    }
    assert pricing["wider_grid_licensed_by_this_record"] is False


def test_a_censored_draw_prices_nothing_and_does_not_widen_the_grid(config, priced_system):
    pricing = pricing_decision(_censor(priced_system), config)
    assert pricing["status"] == "right_censored_at_frozen_grid"
    assert pricing["cells_with_a_finite_interval"] == 0
    assert pricing["right_censored_cells"] == 40
    assert pricing["right_censored_by_search_status"] == {
        "not_bracketed_within_search_grid": 40
    }
    assert pricing["grid_ceiling_effective_shots_per_setting"] == 65536
    assert pricing["wider_grid_licensed_by_this_record"] is False


def test_a_cell_priced_on_no_card_is_not_a_priced_instance(config, priced_system):
    """The readout is about logical times, which only exist on an admissible card."""
    unusable = copy.deepcopy(priced_system)
    for arm in unusable["arms"]:
        for rung in arm["rungs"]:
            for payload in rung["estimators"].values():
                for entry in payload["cost_bracket"]["cards"].values():
                    entry["admissible"] = False
    pricing = pricing_decision(unusable, config)
    assert pricing["cells_with_a_finite_interval"] == 40
    assert pricing["cells_with_a_finite_interval_admissible_on_some_card"] == 0
    assert pricing["status"] == "right_censored_at_frozen_grid"


def test_the_censoring_tabulation_is_not_a_gate(config, priced_system):
    """R3b's failure-mode split was labelled post-hoc; reporting it is not a test."""
    pricing = pricing_decision(_censor(priced_system), config)
    assert "post_hoc_diagnostic_not_preregistered" in (
        pricing["right_censored_by_search_status_note"]
    )
    assert pricing["crossing_policy"] == config["acceptance_rule"]["crossing_policy"]
    assert pricing["censoring_policy"] == config["acceptance_rule"]["censoring_policy"]


# --- QR3 ---------------------------------------------------------------------


def test_qr3_is_re_derived_only_on_a_second_priced_instance(config, cards, priced_system):
    censored = _censor(priced_system)
    pricing = pricing_decision(censored, config)
    qr3 = qr3_second_instance(dict(censored, system="lih"), cards, config, pricing)
    assert qr3["status"] == "not_re_derived"
    assert qr3["priced_instances"] == [FIRST_PRICED_INSTANCE]
    assert "comparison" not in qr3


def test_qr3_compares_the_two_instances_when_the_second_prices(config, cards, priced_system):
    pricing = pricing_decision(priced_system, config)
    qr3 = qr3_second_instance(
        dict(priced_system, system="lih"), cards, config, pricing
    )
    assert qr3["status"] == "re_derived"
    comparison = qr3["comparison"]
    assert comparison["status"] == "compared"
    assert comparison["priced_instances"] == [FIRST_PRICED_INSTANCE, "lih"]
    assert comparison["verdict"] in {
        "mapping_spread_smaller_than_instance_spread",
        "mapping_spread_not_smaller_than_instance_spread",
        "indeterminate_at_this_shot_grid",
    }


def test_qr3_reads_the_frozen_abstention_and_leaves_it_standing(config, cards, priced_system):
    """``protocol_cost.json`` keeps its own verdict; the combination lives here."""
    before = PROTOCOL_COST_RECORD.read_bytes()
    pricing = pricing_decision(priced_system, config)
    qr3 = qr3_second_instance(dict(priced_system, system="lih"), cards, config, pricing)
    assert qr3["first_instance"]["verdict_in_that_record"] == "abstains"
    assert qr3["first_instance"]["left_unchanged"] is True
    assert PROTOCOL_COST_RECORD.read_bytes() == before


# --- the record the producer assembles ---------------------------------------


def test_the_structural_half_draws_nothing(structural_record):
    assert structural_record["schema"] == SCHEMA
    assert structural_record["full_cost_run"]["executed"] is False
    assert structural_record["full_cost_run"]["is_a_cost_record"] is False
    assert structural_record["pricing"] == {"status": "run_not_executed"}
    assert structural_record["qr3_second_instance"]["status"] == "not_re_derived"


def test_the_structural_half_rederives_the_preregistered_bank(config, structural_record):
    selection = structural_record["selection"]
    candidate = config["candidate"]
    assert selection["labels"] == candidate["selected_labels"]
    assert selection["basis_size"] == candidate["expected_basis_size"] == 2
    assert selection["source_word_universe"] == candidate["expected_binding_word_universe"]
    assert selection["stopping_reason"] == "smallest prefix clearing the accuracy margin"
    # The bank R3S admitted, not the one the greedy's own rule stops at.
    assert selection["basis_size"] < selection["intrinsic_stop_basis_size"] == 13
    assert (
        abs(selection["bias_millihartree"] - candidate["expected_bias_millihartree"])
        <= candidate["bias_tolerance_millihartree"]
    )


def test_the_record_binds_the_config_it_consumed(config, structural_record):
    assert structural_record["config_file_sha256"] == (
        producer.FROZEN_CONFIG_SHA256["r3c_lih_full_cost"]
    )
    prereg = structural_record["preregistration"]
    assert prereg["config"] == "benchmarks/configs/r3c_lih_full_cost.json"
    assert prereg["landed_before_any_sampling"] is True
    assert prereg["authorized_by_this_config"] is True


def test_the_record_inherits_the_configs_tenseless_boundary(config, structural_record):
    """The thing R3c's wording was designed for, and R3b's could not do."""
    assert structural_record["claim_boundary"] == config["claim_boundary"]
    assert structural_record["preregistration"]["claim_boundary_inherited_from_config"]
    lowered = structural_record["claim_boundary"].lower()
    for phrase in ("no sampling has been performed", "has not been sampled", "not run yet"):
        assert phrase not in lowered


def test_the_structural_half_carries_no_exact_tier_price(structural_record):
    """The asymptotic preflight may not put a second C_time notion in a cost record."""

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)

    keys = set(walk(structural_record["mapping_preflight"]))
    keys |= set(walk(structural_record["structural_protocol_preflight"]))
    assert not [key for key in keys if key.startswith("C_time")]
    assert "cost_bracket" not in keys
    assert "device_costs" not in keys
