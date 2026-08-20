"""The exact-tier QR3 comparison, tested on synthetic costs.

The real record costs hours to produce, so the arithmetic that turns it into a
verdict is exercised here instead: intervals in, spread brackets out, and the
three-valued verdict that follows. Every case is a shape the real grid can
present -- a separated comparison, an overlapping one, a cell one instance lost
to a fidelity floor -- written small enough that the expected answer is
checkable by hand.
"""

import pytest

from benchmarks.run_protocol_cost import _instance_cost_spread, _spread_bracket


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
