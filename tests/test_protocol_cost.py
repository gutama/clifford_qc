"""Contracts for the R3 accuracy-matched ``C(epsilon)`` and ``k*`` record.

The unit tests here exercise the two pieces of reasoning this layer adds on top
of R1's search: turning a bracketed crossing into a cost *interval*, and turning
a set of intervals into a ``k*`` region rather than an integer. Both are pure
functions of a record fragment, so they are tested on fixtures that state the
case being made instead of on whatever the committed record happens to contain.
"""

import copy
import json

import pytest

pytest.importorskip("stim")

from clifford_qc.measurement.cost import SettingResources  # noqa: E402

from benchmarks.check_protocol_cost import contract_problems  # noqa: E402
from benchmarks.run_protocol_cost import (  # noqa: E402
    REFERENCE,
    _cost_bracket,
    _k_star,
    _pooling_verdict,
    _settings_ordering_verdict,
)


@pytest.fixture(scope="module")
def record():
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cards():
    from benchmarks.run_mapping_axis import load_device_cards

    return load_device_cards()


RESOURCES = [SettingResources(n_1q=4, n_2q=0, d_1q=1, d_2q=0)] * 3


def _search(passing, failing, marginal=()):
    return {
        "status": "confirmed_bracket",
        "confirmed_passing_effective_shots_per_setting": passing,
        "confirmed_failing_effective_shots_per_setting": failing,
        "environment_marginal_endpoints": list(marginal),
        "crossing_is_environment_marginal": bool(marginal),
    }


# ---------------------------------------------------------------------------
# The interval a crossing licenses


def test_a_resolved_crossing_brackets_between_its_two_deciding_endpoints(cards):
    bracket = _cost_bracket(cards, RESOURCES, 8, _search(4096, 1024))
    assert bracket["lower_effective_shots_per_setting"] == 1024
    assert bracket["point_effective_shots_per_setting"] == 4096
    assert bracket["upper_effective_shots_per_setting"] == 4096
    assert bracket["widened_sides"] == []
    assert not bracket["unbounded_below"]
    entry = bracket["cards"]["logical-alltoall"]
    assert entry["C_time_lower_us"] < entry["C_time_epsilon_us"]
    assert entry["C_time_upper_us"] == entry["C_time_epsilon_us"]


def test_a_marginal_passing_side_widens_upward_only(cards):
    """The deciding pass could fail elsewhere, pushing the count one step up."""
    bracket = _cost_bracket(cards, RESOURCES, 8, _search(4096, 1024, ("passing",)))
    assert bracket["upper_effective_shots_per_setting"] == 16384
    assert bracket["lower_effective_shots_per_setting"] == 1024
    assert bracket["widened_sides"] == ["passing"]
    entry = bracket["cards"]["logical-alltoall"]
    assert entry["C_time_upper_us"] > entry["C_time_epsilon_us"]


def test_a_marginal_failing_side_widens_downward_only(cards):
    """The deciding failure could pass elsewhere, pulling the count one step down."""
    bracket = _cost_bracket(cards, RESOURCES, 8, _search(16384, 4096, ("failing",)))
    assert bracket["lower_effective_shots_per_setting"] == 1024
    assert bracket["upper_effective_shots_per_setting"] == 16384
    assert bracket["widened_sides"] == ["failing"]


def test_a_crossing_below_the_grid_is_unbounded_below_not_pinned_to_64(cards):
    """No confirmed failure means no lower bound was measured, so none is claimed."""
    bracket = _cost_bracket(cards, RESOURCES, 8, _search(64, None))
    assert bracket["lower_effective_shots_per_setting"] is None
    assert bracket["unbounded_below"] is True
    assert bracket["cards"]["logical-alltoall"]["C_time_lower_us"] == 0.0


def test_an_unconfirmed_crossing_is_not_priced(cards):
    bracket = _cost_bracket(
        cards, RESOURCES, 8,
        {"status": "nonmonotone_confirmation",
         "confirmed_passing_effective_shots_per_setting": None,
         "confirmed_failing_effective_shots_per_setting": None},
    )
    assert bracket["priced"] is False
    assert bracket["cards"] == {}


# ---------------------------------------------------------------------------
# k* as a region


def _arm(cells, estimator="single_assignment"):
    """``cells`` maps ``block_size -> (settings, lower, point, upper)``."""
    return {
        "rungs": [
            {
                "block_size": block_size,
                "settings": settings,
                "estimators": {
                    estimator: {
                        "cost_bracket": {
                            "priced": True,
                            "cards": {
                                "card": {
                                    "admissible": True,
                                    "C_time_lower_us": lower,
                                    "C_time_epsilon_us": point,
                                    "C_time_upper_us": upper,
                                }
                            },
                        }
                    }
                },
            }
            for block_size, (settings, lower, point, upper) in sorted(cells.items())
        ]
    }


def test_separated_intervals_resolve_k_star_to_one_rung():
    arm = _arm({1: (100, 80.0, 100.0, 100.0), 2: (50, 800.0, 1000.0, 1000.0)})
    entry = _k_star(arm, "card", "single_assignment")
    assert entry["k_star_region"] == [1]
    assert entry["k_star_point"] == 1
    assert entry["resolved"] is True
    assert entry["status"] == "resolved"


def test_overlapping_intervals_report_a_region_not_the_cheaper_point():
    """Two rungs the search does not separate are a region, however they sort."""
    arm = _arm({1: (100, 80.0, 100.0, 100.0), 2: (50, 90.0, 110.0, 110.0)})
    entry = _k_star(arm, "card", "single_assignment")
    assert entry["k_star_region"] == [1, 2]
    assert entry["k_star_point"] == 1
    assert entry["resolved"] is False
    assert entry["status"] == "region"


def test_the_point_argmin_always_lies_inside_its_own_region():
    """``C_point <= C_upper`` elementwise makes this a theorem, so it is pinned."""
    for cheaper_upper in (100.0, 400.0):
        arm = _arm({
            1: (100, 10.0, 100.0, cheaper_upper),
            2: (50, 20.0, 200.0, 200.0),
            4: (20, 30.0, 300.0, 300.0),
        })
        entry = _k_star(arm, "card", "single_assignment")
        assert entry["k_star_point"] in entry["k_star_region"]


def test_an_unbounded_rung_cannot_shrink_another_rungs_region():
    """A rung whose passing side is marginal at the grid top bounds nothing."""
    arm = _arm({1: (100, 10.0, 100.0, None), 2: (50, 20.0, 200.0, None)})
    entry = _k_star(arm, "card", "single_assignment")
    assert entry["status"] == "no_bounded_rung"
    assert entry["k_star_region"] == [1, 2]
    assert entry["resolved"] is False


def test_unpriced_and_inadmissible_rungs_are_named_not_dropped():
    arm = _arm({1: (100, 80.0, 100.0, 100.0)})
    arm["rungs"].append({
        "block_size": 2, "settings": 50,
        "estimators": {"single_assignment": {"cost_bracket": {
            "priced": True,
            "cards": {"card": {"admissible": False, "C_time_lower_us": None,
                               "C_time_epsilon_us": None, "C_time_upper_us": None}},
        }}},
    })
    arm["rungs"].append({
        "block_size": 4, "settings": 20,
        "estimators": {"single_assignment": {"cost_bracket": {
            "priced": False, "cards": {}}}},
    })
    entry = _k_star(arm, "card", "single_assignment")
    assert entry["priced_block_sizes"] == [1]
    assert entry["inadmissible_block_sizes"] == [2]
    assert entry["unresolved_block_sizes"] == [4]
    assert entry["k_star_region"] == [1]


# ---------------------------------------------------------------------------
# The QR verdicts


def test_qr1_reports_a_reordering_only_when_the_orderings_differ():
    agreeing = _settings_ordering_verdict({"costs": [
        {"block_size": 1, "settings": 100, "C_time_epsilon_us": 500.0},
        {"block_size": 2, "settings": 50, "C_time_epsilon_us": 200.0},
    ]})
    assert agreeing["reorders_settings_ordering"] is False

    reordered = _settings_ordering_verdict({"costs": [
        {"block_size": 1, "settings": 100, "C_time_epsilon_us": 200.0},
        {"block_size": 2, "settings": 50, "C_time_epsilon_us": 500.0},
    ]})
    assert reordered["reorders_settings_ordering"] is True
    assert reordered["by_accuracy_matched_cost"] == [1, 2]
    assert reordered["by_setting_count"] == [2, 1]


def test_qr4_is_answered_on_the_regions_not_on_the_point_argmins():
    """A moved point inside a shared region has not moved ``k*``."""
    verdict = _pooling_verdict(
        {"k_star_region": [1, 2], "k_star_point": 1},
        {"k_star_region": [2, 4], "k_star_point": 2},
    )
    assert verdict["pooling_moves_k_star"] is False
    assert verdict["pooling_moves_k_star_point"] is True

    moved = _pooling_verdict(
        {"k_star_region": [1], "k_star_point": 1},
        {"k_star_region": [8], "k_star_point": 8},
    )
    assert moved["pooling_moves_k_star"] is True
    assert moved["regions_disjoint"] is True


# ---------------------------------------------------------------------------
# The committed record, and the gate that guards it


def test_committed_record_passes_its_contract(record):
    assert contract_problems(record) == []


def test_h4_is_recorded_as_unpriced_rather_than_omitted(record):
    """"Unattainable at this target" is the measurement, so it is in the record."""
    h4 = record["systems"]["h4"]
    assert h4["status"] == "bias_floor_exceeds_target"
    assert h4["max_exact_subspace_bias_millihartree"] > record[
        "accuracy_target_millihartree"
    ]
    assert "k_star" not in h4
    for arm in h4["arms"]:
        for rung in arm["rungs"]:
            for cell in rung["estimators"].values():
                assert cell["device_costs"] == {}
                assert cell["cost_bracket"]["priced"] is False


def test_the_jw_arm_reproduces_r1s_frozen_bank_bias_bit_for_bit(record):
    """The two construction paths reach the same bank, to the last bit.

    R1 grows the BeH2 bank with the DA-CASE producer; this record reaches it
    through the mapping-axis selection transported by ``jw``. Both take their
    exact reference from the dense eigensolve, so the agreement is exact rather
    than tolerant -- and a drift here would mean the mapping transport moved the
    subspace, which is the R2b invariant, not a rounding question.
    """
    r1 = json.loads(
        (REFERENCE.parent / "exact_shot_search.json").read_text(encoding="utf-8")
    )
    jw = next(
        arm for arm in record["systems"]["beh2"]["arms"] if arm["mapping"] == "jw"
    )
    assert jw["exact_subspace_bias_millihartree"] == r1["systems"]["beh2"][
        "exact_subspace_bias_millihartree"
    ]
    assert record["systems"]["beh2"]["exact_ground_energy"] == r1["systems"]["beh2"][
        "exact_ground_energy"
    ]


def test_every_cell_reproduces_the_structural_grids_setting_count(record):
    """The condition that makes this the cost layer of that grid."""
    from benchmarks.run_protocol_cost import structural_settings

    frozen = structural_settings()
    for key, system in record["systems"].items():
        for arm in system["arms"]:
            for rung in arm["rungs"]:
                assert rung["settings"] == frozen[
                    (key, arm["mapping"], rung["block_size"])
                ], (key, arm["mapping"], rung["block_size"])


def test_contract_catches_a_cell_that_repartitioned_the_grid(record):
    broken = copy.deepcopy(record)
    broken["systems"]["beh2"]["arms"][0]["rungs"][0]["settings"] += 1
    assert any("structural R3 grid froze" in p for p in contract_problems(broken))


def test_contract_catches_a_region_narrowed_to_its_point(record):
    """Reporting the argmin where the intervals overlap is the error §6.7 forbids."""
    broken = copy.deepcopy(record)
    found = False
    for payload in broken["systems"]["beh2"]["k_star"].values():
        for estimator in ("single_assignment", "pooled"):
            for entry in payload[estimator]["by_arm"].values():
                if len(entry["k_star_region"]) > 1:
                    entry["k_star_region"] = [entry["k_star_point"]]
                    entry["resolved"] = True
                    entry["status"] = "resolved"
                    found = True
    assert found, "expected at least one unresolved region in the committed record"
    assert any("is not the set of rungs" in p for p in contract_problems(broken))


def test_contract_catches_an_interval_widened_without_a_marginal_flag(record):
    broken = copy.deepcopy(record)
    cell = broken["systems"]["beh2"]["arms"][0]["rungs"][0]["estimators"]["pooled"]
    cell["cost_bracket"]["widened_sides"] = ["passing"]
    problems = contract_problems(broken)
    assert any("do not match" in p for p in problems)


def test_contract_catches_an_overstated_qr3_verdict(record):
    """Only BeH2 is priced, so the cross-instance comparison must abstain."""
    broken = copy.deepcopy(record)
    broken["qr3_accuracy_matched"]["status"] = "answered"
    problems = contract_problems(broken)
    assert any("abstains if and only if" in p for p in problems)


def test_contract_recomputes_the_qr3_payload_rather_than_reading_it(record):
    """A verdict is a conclusion, so the checker re-derives it from the costs."""
    broken = copy.deepcopy(record)
    broken["qr3_accuracy_matched"]["reason"] = "because I said so"
    assert any("QR3:" in p for p in contract_problems(broken))


def test_contract_catches_a_system_dropped_without_a_deferral(record):
    """A structural system must be priced or deferred, never merely absent."""
    broken = copy.deepcopy(record)
    broken["cost_layer_scope"]["deferred"] = []
    assert any("does not partition" in p for p in contract_problems(broken))


def test_contract_catches_a_deferral_with_no_reason(record):
    broken = copy.deepcopy(record)
    broken["cost_layer_scope"]["deferred"][0]["reason"] = ""
    assert any("deferred with no reason" in p for p in contract_problems(broken))


def test_contract_catches_a_scoping_probe_promoted_to_a_record(record):
    """A reduced-replica probe may never be quoted as a cost."""
    broken = copy.deepcopy(record)
    broken["cost_layer_scope"]["deferred"][0]["scoping_probe"]["is_a_record"] = True
    assert any("disclaim record status" in p for p in contract_problems(broken))


def test_contract_catches_a_misquoted_r1_crossing(record):
    broken = copy.deepcopy(record)
    broken["systems"]["beh2"]["r1_cross_check"]["rows"][0]["r1_confirmed_passing"] = 7
    assert any("the frozen record says" in p for p in contract_problems(broken))


def test_the_jw_arm_reprices_r1_under_an_independent_stream(record):
    """Corroboration, not identity: the two records draw disjoint seeds."""
    cross = record["systems"]["beh2"]["r1_cross_check"]
    assert cross["status"] == "compared"
    assert cross["compared_crossings"] == 8
    assert cross["disagreements_all_environment_marginal"] is True


def test_the_cross_check_quotes_r1s_own_marginal_flags(record):
    """R1 flags three of its eight BeH2 crossings; the comparison must carry them."""
    from benchmarks.run_protocol_cost import r1_crossings

    pinned = r1_crossings()
    assert sum(row["environment_marginal"] for row in pinned.values()) == 3
    for row in record["systems"]["beh2"]["r1_cross_check"]["rows"]:
        entry = pinned[(row["block_size"], row["estimator"])]
        assert row["r1_environment_marginal"] == entry["environment_marginal"]
        assert row["r1_confirmed_passing"] == entry["confirmed_passing"]


def _cross_row(record, marginal_in_r1):
    """A cross-check row R1 does, or does not, call environment-marginal."""
    return next(
        row for row in record["systems"]["beh2"]["r1_cross_check"]["rows"]
        if row["r1_environment_marginal"] is marginal_in_r1
    )


def test_an_unflagged_disagreement_with_r1_is_rejected(record):
    """A crossing neither record calls marginal must reprice to the same count."""
    broken = copy.deepcopy(record)
    row = _cross_row(broken, False)
    row["r3_environment_marginal"] = False
    row["r3_confirmed_passing"] = 7
    row["agrees"] = False
    problems = contract_problems(broken)
    assert any("neither record calls environment-marginal" in p for p in problems)


def test_a_disagreement_r1_itself_flagged_is_tolerated(record):
    """The flag is symmetric: R1 saying "not resolved" is enough to excuse it.

    Three of R1's eight BeH2 crossings were decided on the target, and the
    record that says so is the reason an independent stream landing elsewhere
    on those three is expected rather than alarming.
    """
    broken = copy.deepcopy(record)
    row = _cross_row(broken, True)
    row["r3_environment_marginal"] = False
    row["r3_confirmed_passing"] = 7
    row["agrees"] = False
    broken["systems"]["beh2"]["r1_cross_check"][
        "disagreements_all_environment_marginal"
    ] = True
    problems = contract_problems(broken)
    assert not any("environment-marginal" in p for p in problems)
