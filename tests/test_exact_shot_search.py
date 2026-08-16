"""Statistical contract checks for the R1 exact-tier search."""

import copy
import json

import numpy as np
import pytest

pytest.importorskip("stim")

from benchmarks.check_exact_shot_search import contract_problems
from benchmarks.run_exact_shot_search import (
    REFERENCE,
    _confirmation_endpoints,
    _confirm_result,
    _exploratory_bracket,
    _ordering_verdict,
    _seed_sequence,
    _summary,
    _system_status,
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
        # These fixtures carry no bootstrap bound, so the margin is
        # unknown rather than comfortable.
        "environment_marginal_endpoints": [],
        "crossing_is_environment_marginal": False,
        "passing_target_margin_fraction": None,
        "failing_target_margin_fraction": None,
    }


def test_confirmation_extends_downward_and_prices_the_smallest_pass():
    exploration = [
        _row(64, False), _row(256, False), _row(1024, False),
        _row(4096, False), _row(16384, True),
    ]
    assert _confirmation_endpoints(exploration) == (64, 256, 1024, 4096, 16384)
    confirmation = [
        _row(64, False), _row(256, False), _row(1024, True),
        _row(4096, True), _row(16384, True),
    ]
    assert _confirm_result(exploration, confirmation) == {
        "status": "confirmed_bracket",
        "confirmed_failing_effective_shots_per_setting": 256,
        "confirmed_passing_effective_shots_per_setting": 1024,
        # These fixtures carry no bootstrap bound, so the margin is
        # unknown rather than comfortable.
        "environment_marginal_endpoints": [],
        "crossing_is_environment_marginal": False,
        "passing_target_margin_fraction": None,
        "failing_target_margin_fraction": None,
    }


def test_confirmation_refuses_to_price_a_nonmonotone_crossing():
    exploration = [_row(64, False), _row(256, True), _row(1024, True)]
    confirmation = [_row(64, True), _row(256, False), _row(1024, True)]
    assert _confirm_result(exploration, confirmation) == {
        "status": "nonmonotone_confirmation",
        "confirmed_failing_effective_shots_per_setting": None,
        "confirmed_passing_effective_shots_per_setting": None,
    }


def test_seed_namespaces_do_not_overlap_across_phases_or_rungs():
    streams = [
        np.random.default_rng(_seed_sequence(root, block_size, replica)).bytes(64)
        for root in (260_813_000, 260_913_000)
        for block_size in (1, 2, 4, 8)
        for replica in range(4)
    ]
    assert len(streams) == len(set(streams))


def test_system_status_is_derived_from_the_bias_floor():
    assert _system_status(1.599) == "searched"
    assert _system_status(1.600) == "bias_floor_exceeds_target"


def test_disjoint_admissible_sets_use_the_shared_ordering_semantics():
    verdict = _ordering_verdict({
        "single_assignment": [{"block_size": 1}],
        "pooled": [{"block_size": 2}],
    })
    assert verdict == {
        "compared_block_sizes": [],
        "pooling_reorders_protocols": False,
    }


def test_record_gate_rejects_nonminimal_pricing_and_a_false_failing_endpoint():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    broken = copy.deepcopy(record)
    arm = broken["systems"]["beh2"]["rows"][0]["estimators"]["pooled"]
    low, high = 64, 256
    arm["confirmation"] = [
        _row(low, True),
        _row(high, True),
    ]
    arm["shot_to_target"]["confirmed_passing_effective_shots_per_setting"] = high
    arm["shot_to_target"]["confirmed_failing_effective_shots_per_setting"] = low
    problems = contract_problems(broken)
    assert any("smallest confirmed pass" in problem for problem in problems)
    assert any("reported failing endpoint did not fail" in problem for problem in problems)


def test_record_gate_rejects_a_marginal_listing_that_is_not_a_list_of_labels():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    # Each of these satisfies `label in marginal` for at least one label while
    # meaning something else: a bare string matches by substring, a mapping by
    # key, and an empty string is falsy enough to agree with a cleared flag.
    for malformed in ("passing", "", {"passing": True}, ["passing", "sideways"]):
        broken = copy.deepcopy(record)
        arm = broken["systems"]["beh2"]["rows"][0]["estimators"]["pooled"]
        arm["shot_to_target"]["environment_marginal_endpoints"] = malformed
        problems = contract_problems(broken)
        assert any("are not a list of" in problem for problem in problems), malformed


def test_solver_failure_forces_an_endpoint_to_fail():
    rows = [
        {"failure": None, "energy": -1.0, "rank": 2},
        {"failure": "ValueError"},
    ]
    summary = _summary(rows, -1.0, bootstrap_seed=7)
    assert not summary["zero_failure_gate"]
    assert not summary["passes_target"]
    assert summary["failures"] == {"ValueError": 1}


def test_crossing_margin_flags_either_side_of_the_target():
    from benchmarks.run_exact_shot_search import (
        ACCURACY_TARGET_MILLIHARTREE as T,
        MARGINAL_TARGET_FRACTION,
        _crossing_margin,
    )

    def endpoint(upper):
        return {"rmse_one_sided_95pct_upper_millihartree": upper}

    # A crossing is only as reproducible as the endpoint deciding it, so the
    # failing side counts too: a comfortable pass over a barely-failing lower
    # endpoint is one environment away from reporting the smaller count.
    by_endpoint = {4096: endpoint(0.2 * T), 1024: endpoint(1.02 * T)}
    margin = _crossing_margin(by_endpoint, 4096, 1024)
    assert margin["environment_marginal_endpoints"] == ["failing"]
    assert margin["crossing_is_environment_marginal"] is True
    assert margin["passing_target_margin_fraction"] == pytest.approx(-0.8)
    assert margin["failing_target_margin_fraction"] == pytest.approx(0.02)

    # Both sides clear of the band is a resolved crossing.
    resolved = _crossing_margin(
        {4096: endpoint(0.2 * T), 1024: endpoint(3.0 * T)}, 4096, 1024
    )
    assert resolved["environment_marginal_endpoints"] == []
    assert resolved["crossing_is_environment_marginal"] is False

    # The band is symmetric: a pass just under the target is equally unresolved.
    just_under = _crossing_margin(
        {4096: endpoint((1.0 - MARGINAL_TARGET_FRACTION / 2) * T)}, 4096, None
    )
    assert just_under["environment_marginal_endpoints"] == ["passing"]
    assert just_under["failing_target_margin_fraction"] is None


# ---------------------------------------------------------------------------
# Rank stability at the deciding endpoints
#
# Everything upstream of the solve is already backend-invariant by
# construction: grouping is packed GF(2) parity, the diagonalizers are exact
# stim tableaus chosen by integer gate count, and each replica draws from
# SeedSequence(root, spawn_key=(k, replica)) so its stream does not depend on
# execution order. The retained rank is the one float comparison that becomes a
# discrete choice, and it is what moved a published endpoint across the target
# under a different bundled LAPACK.


def _arm_with_ranks(passing_histogram, failing_histogram=None):
    arm = {
        "exploration": [],
        "confirmation": [
            {"effective_shots_per_setting": 1024, "passes_target": False,
             "rank_histogram": failing_histogram or {"5": 100}},
            {"effective_shots_per_setting": 4096, "passes_target": True,
             "rank_histogram": passing_histogram},
        ],
        "shot_to_target": {
            "confirmed_passing_effective_shots_per_setting": 4096,
            "confirmed_failing_effective_shots_per_setting": 1024,
        },
    }
    return arm


def _rank_problems(arm):
    from benchmarks.check_exact_shot_search import _rank_stability_problems

    confirmation = {
        item["effective_shots_per_setting"]: item for item in arm["confirmation"]
    }
    return _rank_stability_problems("probe", confirmation, 4096, 1024)


def test_a_unanimous_rank_panel_is_accepted():
    assert _rank_problems(_arm_with_ranks({"5": 100})) == []


def test_a_split_rank_at_the_passing_endpoint_is_rejected():
    problems = _rank_problems(_arm_with_ranks({"4": 3, "5": 97}))
    assert len(problems) == 1
    assert "passing endpoint 4096 is rank-marginal" in problems[0]
    assert "rank 4 x3, rank 5 x97" in problems[0]


def test_a_split_rank_at_the_failing_endpoint_is_rejected():
    """The failing endpoint decides the bracket too, so it is held to the same bar."""
    problems = _rank_problems(
        _arm_with_ranks({"5": 100}, failing_histogram={"4": 1, "5": 99})
    )
    assert len(problems) == 1
    assert "failing endpoint 1024 is rank-marginal" in problems[0]


def test_a_missing_rank_histogram_is_rejected_rather_than_assumed_stable():
    arm = _arm_with_ranks({"5": 100})
    arm["confirmation"][1].pop("rank_histogram")
    problems = _rank_problems(arm)
    assert len(problems) == 1
    assert "recorded no rank histogram" in problems[0]


def test_committed_record_has_rank_stable_deciding_endpoints():
    """The gate is live, not vacuous: it passes on the real record today."""
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert [p for p in contract_problems(record) if "rank-marginal" in p] == []


def test_committed_record_still_splits_rank_away_from_the_crossing():
    """Non-deciding endpoints may split, and do -- so the gate is discriminating.

    If this ever reached zero, the gate would no longer be distinguishing
    deciding from non-deciding endpoints and the test above would pass for the
    wrong reason.
    """
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    mixed = [
        (row["block_size"], estimator, item["effective_shots_per_setting"])
        for row in record["systems"]["beh2"]["rows"]
        for estimator, arm in row["estimators"].items()
        for phase in ("exploration", "confirmation")
        for item in arm[phase]
        if len(item.get("rank_histogram", {})) > 1
    ]
    assert mixed, "expected rank splits away from the crossing"
    assert all(shots <= 256 for _, _, shots in mixed)
