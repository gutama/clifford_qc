"""The exact-tier QR3 comparison, tested on synthetic costs.

The real record costs hours to produce, so the arithmetic that turns it into a
verdict is exercised here instead: intervals in, spread brackets out, and the
three-valued verdict that follows. Every case is a shape the real grid can
present -- a separated comparison, an overlapping one, a cell one instance lost
to a fidelity floor -- written small enough that the expected answer is
checkable by hand.
"""

import pytest

from benchmarks.run_protocol_cost import (
    _cost_interval,
    _instance_cost_spread,
    _spread_bracket,
    _spread_extremum,
)


class _Card:
    def __init__(self, name):
        self.name = name


CARDS = [_Card("logical-alltoall")]
ESTIMATOR_KEYS = ("single_assignment", "pooled")


def _rung(block_size, cost, *, admissible=True, priced=True, width=0.0):
    """One rung whose cost is ``cost`` with a symmetric ``width`` bracket."""
    cards = {
        card.name: {
            "admissible": admissible,
            "C_time_lower_us": cost * (1.0 - width),
            "C_time_epsilon_us": cost,
            "C_time_upper_us": cost * (1.0 + width),
        }
        for card in CARDS
    }
    bracket = {"priced": priced, "cards": cards}
    return {
        "block_size": block_size,
        "estimators": {key: {"cost_bracket": bracket} for key in ESTIMATOR_KEYS},
    }


def _system(costs, *, status="searched", measured_qubits=8, **rung_kwargs):
    """``costs`` maps a mapping name to its cost at block size 1."""
    return {
        "status": status,
        "arms": [
            {
                "mapping": mapping,
                "measured_qubits": measured_qubits,
                "rungs": [_rung(1, cost, **rung_kwargs)],
            }
            for mapping, cost in costs.items()
        ],
    }


def test_spread_bracket_is_the_widest_and_narrowest_ratio_the_intervals_allow():
    # Two costs, each free in its own interval.
    bracket = _spread_bracket([(90.0, 100.0, 110.0), (180.0, 200.0, 220.0)])
    assert bracket["point"] == pytest.approx(2.0)
    assert bracket["maximum_possible"] == pytest.approx(220.0 / 90.0)
    assert bracket["minimum_possible"] == pytest.approx(180.0 / 110.0)


def test_spread_bracket_clamps_at_one_when_the_intervals_overlap():
    # Intervals sharing a point can all be equal, so the spread can be 1.
    bracket = _spread_bracket([(90.0, 100.0, 150.0), (120.0, 200.0, 260.0)])
    assert bracket["minimum_possible"] == 1.0
    assert bracket["point"] == pytest.approx(2.0)


def test_one_priced_instance_abstains():
    systems = {
        "a": _system({"jw": 100.0, "bk": 110.0}),
        "b": _system({"jw": 1.0, "bk": 1.1}, status="bias_floor_exceeds_target"),
    }
    payload = _instance_cost_spread(systems, CARDS, ["a"])
    assert payload["status"] == "abstains"
    assert payload["priced_instances"] == ["a"]
    assert "verdict" not in payload


def test_a_separated_comparison_returns_the_directional_verdict():
    # Mapping moves cost by 1.1x within each instance; changing instance moves
    # it by 100x. Brackets are exact, so the two never overlap.
    systems = {
        "a": _system({"jw": 100.0, "bk": 110.0}),
        "b": _system({"jw": 10_000.0, "bk": 11_000.0}),
    }
    payload = _instance_cost_spread(systems, CARDS, ["a", "b"])
    assert payload["status"] == "compared"
    assert payload["verdict"] == "mapping_spread_smaller_than_instance_spread"
    assert payload["widest_mapping_spread"]["point"] == pytest.approx(1.1)
    assert payload["narrowest_instance_spread"]["point"] == pytest.approx(100.0)


def test_the_reverse_separation_also_returns_a_directional_verdict():
    systems = {
        "a": _system({"jw": 100.0, "bk": 10_000.0}),
        "b": _system({"jw": 110.0, "bk": 11_000.0}),
    }
    payload = _instance_cost_spread(systems, CARDS, ["a", "b"])
    assert payload["verdict"] == "mapping_spread_not_smaller_than_instance_spread"


def test_overlapping_brackets_are_reported_as_indeterminate():
    # Same point ratios as the separated case would give, but each cost carries
    # a bracket wide enough that the two effects are not resolved apart.
    systems = {
        "a": _system({"jw": 100.0, "bk": 150.0}, width=0.6),
        "b": _system({"jw": 200.0, "bk": 300.0}, width=0.6),
    }
    payload = _instance_cost_spread(systems, CARDS, ["a", "b"])
    assert payload["verdict"] == "indeterminate_at_this_shot_grid"



def test_cell_extrema_are_enveloped_before_the_directional_verdict():
    # The point-max mapping cell and point-min instance cell look separated,
    # but different cells support the wider uncertainty endpoints. Selecting
    # the point witnesses first would falsely report a directional result.
    mapping_cells = [
        {"cell": "point-max", "minimum_possible": 1.9, "point": 2.0,
         "maximum_possible": 2.1},
        {"cell": "wide", "minimum_possible": 1.0, "point": 1.9,
         "maximum_possible": 10.0},
    ]
    instance_cells = [
        {"cell": "point-min", "minimum_possible": 2.9, "point": 3.0,
         "maximum_possible": 3.1},
        {"cell": "wide", "minimum_possible": 1.0, "point": 4.0,
         "maximum_possible": 5.0},
    ]
    mapping = _spread_extremum(mapping_cells, extremum="maximum")
    instance = _spread_extremum(instance_cells, extremum="minimum")

    assert mapping["point"] == 2.0
    assert mapping["maximum_possible"] == 10.0
    assert instance["point"] == 3.0
    assert instance["minimum_possible"] == 1.0
    assert mapping["supporting_cells"]["point"]["cell"] == "point-max"
    assert mapping["supporting_cells"]["maximum_possible"]["cell"] == "wide"
    assert not (
        mapping["maximum_possible"] < instance["minimum_possible"]
        or mapping["minimum_possible"] > instance["maximum_possible"]
    )


def test_right_censored_cost_brackets_are_not_coerced_to_finite_ratios():
    rung = {
        "estimators": {
            "single_assignment": {
                "cost_bracket": {
                    "priced": True,
                    "unbounded_below": False,
                    "unbounded_above": True,
                    "cards": {
                        "logical-alltoall": {
                            "admissible": True,
                            "C_time_lower_us": 90.0,
                            "C_time_epsilon_us": 100.0,
                            "C_time_upper_us": None,
                        }
                    },
                }
            }
        }
    }
    assert _cost_interval(rung, "logical-alltoall", "single_assignment") is None


def test_a_cell_only_one_instance_can_run_is_excluded():
    # 'b' lost its bk arm to a fidelity floor. Counting it would report the
    # missing rung as a cost difference.
    systems = {
        "a": _system({"jw": 100.0, "bk": 110.0}),
        "b": {
            "status": "searched",
            "arms": [
                {"mapping": "jw", "measured_qubits": 8,
                 "rungs": [_rung(1, 10_000.0)]},
                {"mapping": "bk", "measured_qubits": 8,
                 "rungs": [_rung(1, 11_000.0, admissible=False)]},
            ],
        },
    }
    payload = _instance_cost_spread(systems, CARDS, ["a", "b"])
    # Two estimators x one shared (card, arm, k) cell.
    assert payload["compared_cells"]["instance"] == 2
    assert all(
        cell["mapping"] == "jw"
        for cell in [payload["narrowest_instance_spread"]]
    )


def test_an_unpriced_bracket_never_enters_the_comparison():
    systems = {
        "a": _system({"jw": 100.0, "bk": 110.0}),
        "b": _system({"jw": 10_000.0, "bk": 11_000.0}, priced=False),
    }
    payload = _instance_cost_spread(systems, CARDS, ["a", "b"])
    assert payload["status"] == "abstains"
    assert "nothing to compare like for like" in payload["reason"]


# --- the cost layer's declared scope -----------------------------------------


def _scoped_record(priced, deferred, *, structural=None, probe_is_record=False):
    """A record carrying only the fields the scope contract reads."""
    return {
        "systems": {key: {"status": "searched"} for key in priced},
        "structural_reference": {
            "systems_in_structural_grid": (
                structural if structural is not None
                else list(priced) + [item["system"] for item in deferred]
            ),
        },
        "cost_layer_scope": {
            "systems": list(priced),
            "deferred": [
                {
                    "status": "right_censored",
                    "search_ceiling_effective_shots_per_setting": 65536,
                    "further_search": "deferred",
                    **item,
                    "scoping_probe": {
                        "is_a_record": probe_is_record,
                        "exploratory_replicas": 2,
                        "confirmatory_replicas": 2,
                        "single_assignment_cells_unresolved": 1,
                        "single_assignment_cells_total": 1,
                    },
                }
                for item in deferred
            ],
        },
    }


def test_a_scope_that_partitions_the_structural_grid_passes():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(["beh2"], [{"system": "h4c", "reason": "grid ceiling"}])
    assert _scope_problems(record) == []


def test_a_system_that_is_neither_priced_nor_deferred_is_a_failure():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(
        ["beh2"], [{"system": "h4c", "reason": "grid ceiling"}],
        structural=["beh2", "h4c", "h4"],
    )
    assert any("does not partition" in problem for problem in _scope_problems(record))


def test_a_deferral_without_a_reason_is_a_failure():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(["beh2"], [{"system": "h4c", "reason": ""}])
    assert any("deferred with no reason" in problem for problem in _scope_problems(record))


def test_a_scoping_probe_must_disclaim_record_status():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(
        ["beh2"], [{"system": "h4c", "reason": "grid ceiling"}],
        probe_is_record=True,
    )
    assert any("disclaim record status" in problem for problem in _scope_problems(record))


def test_a_probe_at_headline_replicas_is_not_a_probe():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(["beh2"], [{"system": "h4c", "reason": "grid ceiling"}])
    probe = record["cost_layer_scope"]["deferred"][0]["scoping_probe"]
    probe["confirmatory_replicas"] = 100
    assert any(
        "that is a record, not a probe" in problem
        for problem in _scope_problems(record)
    )


def test_the_record_must_carry_exactly_the_systems_its_scope_evaluates():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(["beh2"], [{"system": "h4c", "reason": "grid ceiling"}])
    record["systems"]["h4c"] = {"status": "searched"}
    assert any(
        "not the ones its scope evaluates" in problem
        for problem in _scope_problems(record)
    )


def test_a_subset_config_is_one_the_producer_accepts():
    """The contract is round-tripped, not asserted on the intermediate shape.

    An earlier version of this test checked the two system lists separately and
    passed while the config it described was one ``_cost_layer`` rejected on
    sight. Asking the consumer is the only version of this test that means
    anything.
    """
    from benchmarks.check_protocol_cost import _subset_config
    from benchmarks.run_protocol_cost import _cost_layer

    config = _subset_config(("beh2",))
    layer = _cost_layer(config)
    assert config["systems"] == ["beh2"]
    assert layer["systems"] == ["beh2"]
    # A subset is its own complete partition: nothing priced elsewhere, and
    # nothing left deferred for it to have to explain.
    assert layer["deferred"] == []


def test_the_full_config_is_also_one_the_producer_accepts():
    from benchmarks.run_protocol_axis import CONFIG, load_config
    from benchmarks.run_protocol_cost import _cost_layer

    layer = _cost_layer(load_config(CONFIG))
    assert layer["systems"] == ["h4", "beh2"]
    assert [item["system"] for item in layer["deferred"]] == ["h4_converged"]


def test_the_checker_subset_filter_rejects_an_unknown_system():
    import pytest as _pytest

    from benchmarks.check_protocol_cost import _subset_config

    with _pytest.raises(ValueError, match="unknown systems"):
        _subset_config(("nope",))


def test_the_checker_subset_filter_rejects_a_deferred_system_by_name():
    """A deferred system has no subtree to rebuild, so naming one is a mistake."""
    import pytest as _pytest

    from benchmarks.check_protocol_cost import _subset_config

    with _pytest.raises(ValueError, match="deferred, not evaluated"):
        _subset_config(("h4_converged",))



def test_a_deferred_system_must_name_its_evidence_status():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(
        ["beh2"], [{"system": "h4c", "reason": "grid ceiling", "status": ""}]
    )
    assert any("no evidence status" in problem for problem in _scope_problems(record))


def test_right_censoring_pins_the_frozen_ceiling_and_search_decision():
    from benchmarks.check_protocol_cost import _scope_problems

    record = _scoped_record(
        ["beh2"], [{"system": "h4c", "reason": "grid ceiling"}]
    )
    item = record["cost_layer_scope"]["deferred"][0]
    item["search_ceiling_effective_shots_per_setting"] = 131072
    item["further_search"] = "extended"
    problems = _scope_problems(record)
    assert any("frozen ceiling 65536" in problem for problem in problems)
    assert any("further_search=deferred" in problem for problem in problems)


def test_scope_problems_reports_a_nameless_deferral_instead_of_raising():
    from benchmarks.check_protocol_cost import _scope_problems

    record = {
        "cost_layer_scope": {"systems": ["beh2"], "deferred": [{"reason": "x"}]},
        "structural_reference": {"systems_in_structural_grid": ["beh2", "h4c"]},
        "systems": {"beh2": {}},
    }
    problems = _scope_problems(record)
    assert any("names no system" in problem for problem in problems)


def test_contract_problems_reports_malformed_records_instead_of_raising():
    from benchmarks.check_protocol_cost import contract_problems

    problems = contract_problems({"systems": None, "protocol": None})
    assert problems
    assert all(isinstance(problem, str) for problem in problems)
