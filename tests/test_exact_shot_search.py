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


def test_solver_failure_forces_an_endpoint_to_fail():
    rows = [
        {"failure": None, "energy": -1.0, "rank": 2},
        {"failure": "ValueError"},
    ]
    summary = _summary(rows, -1.0, bootstrap_seed=7)
    assert not summary["zero_failure_gate"]
    assert not summary["passes_target"]
    assert summary["failures"] == {"ValueError": 1}
