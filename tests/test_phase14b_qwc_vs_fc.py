"""Contracts for the sampled Phase 14b QWC-versus-FC record."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import benchmarks.check_phase14b_qwc_vs_fc as checker
from benchmarks.run_phase14b_qwc_vs_fc import (
    PREREGISTRATION_MERGE_COMMIT,
    REFERENCE,
    _decision,
    build_record,
    coefficient_range_neyman_schedule,
    derived_seed,
)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _plan(*rows: tuple[int, ...]):
    settings = tuple(
        SimpleNamespace(assigned_word_codes=row, key=("setting", index))
        for index, row in enumerate(rows)
    )
    return SimpleNamespace(settings=settings)


def test_coefficient_range_neyman_allocator_preserves_floor_total_and_zero_scores():
    plan = _plan((1,), (2,), (3,))
    schedule = coefficient_range_neyman_schedule(
        plan, {1: 1.0, 2: 2.0}, 16
    )
    assert schedule == (5, 9, 2)
    assert sum(schedule) == 16


def test_allocator_breaks_equal_remainders_by_setting_index():
    plan = _plan((1,), (2,), (3,))
    assert coefficient_range_neyman_schedule(
        plan, {1: 1.0, 2: 1.0}, 9
    ) == (4, 3, 2)


def test_allocator_refuses_impossible_and_zero_score_schedules():
    plan = _plan((1,), (2,))
    with pytest.raises(ValueError, match="smaller than the setting floors"):
        coefficient_range_neyman_schedule(plan, {1: 1.0}, 3)
    with pytest.raises(ValueError, match="touches no compiled setting"):
        coefficient_range_neyman_schedule(plan, {}, 8)


def test_seed_sequence_namespace_is_stable_and_disjoint():
    headline = derived_seed(140_814_000, 0, 0)
    assert headline == derived_seed(140_814_000, 0, 0)
    assert len({
        headline,
        derived_seed(140_814_000, 1, 0),
        derived_seed(140_814_000, 0, 1),
        derived_seed(140_914_000, 0, 0),
    }) == 4


def test_fast_build_record_override_exercises_end_to_end_producer():
    built = build_record(endpoints=(65_536, 131_072), audit_replicas=3)
    assert built["protocol"]["total_physical_shot_endpoints"] == [65_536, 131_072]
    assert {"status", "shot_efficiency_go", "card_specific"} <= set(built["decision"])
    for protocol_index, name in enumerate(("qwc", "fully_commuting")):
        protocol = built["protocols"][name]
        settings = protocol["compiled_plan"]["settings"]
        for endpoint_index, row in enumerate(protocol["headline"]):
            assert len(row["shot_vector"]) == settings
            assert min(row["shot_vector"]) >= 2
            assert sum(row["shot_vector"]) == row["total_physical_shots"]
            assert row["seed"]["spawn_key"] == [protocol_index, endpoint_index]
        first = next(
            (row for row in protocol["headline"] if row["passes_stochastic_radius"]),
            None,
        )
        certified = protocol["certification"]["certified_total_physical_shots"]
        assert certified == (None if first is None else first["total_physical_shots"])


def test_right_censored_qwc_can_establish_shot_efficiency_go():
    protocols = {
        "qwc": {
            "certification": {
                "certified_total_physical_shots": None,
                "right_censored_above": 2**32,
            },
            "device_costs": {},
        },
        "fully_commuting": {
            "certification": {
                "certified_total_physical_shots": 2**31,
                "right_censored_above": None,
            },
            "device_costs": {},
        },
    }
    config = {
        "acceptance": {"material_shot_reduction_fraction": 0.5},
        "protocol": {"device_cards": []},
    }
    decision = _decision(protocols, config, [])
    assert decision["status"] == "go"
    assert decision["shot_efficiency_go"] is True
    assert decision["fully_commuting_to_qwc_certified_shot_ratio"] is None


def test_committed_sampled_record_passes_every_internal_contract(record):
    assert checker.contract_problems(record) == []
    assert record["preregistration"]["merge_commit"] == PREREGISTRATION_MERGE_COMMIT


def _redistribute_headline_shots(row):
    schedule = row["protocols"]["qwc"]["headline"][0]["shot_vector"]
    schedule[0] += 1
    schedule[1] -= 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row["protocol"].__setitem__("primary_estimator", "pooled"),
            "pooled estimator was promoted",
        ),
        (
            lambda row: row["preregistration"].__setitem__("merge_commit", "0" * 40),
            "merged preregistration",
        ),
        (
            lambda row: row["protocols"]["qwc"]["headline"][0]["seed"].__setitem__(
                "derived_seed", 0
            ),
            "sampled seed",
        ),
        (
            _redistribute_headline_shots,
            "shot vector digest",
        ),
        (
            lambda row: row["protocols"]["qwc"]["certification"].__setitem__(
                "status", "tampered"
            ),
            "first passing look",
        ),
        (
            lambda row: row["protocols"]["fully_commuting"]["covariance_audit"]
            ["estimators"]["pooled"].__setitem__("empirical_to_predicted_ratio", 1.3),
            "covariance ratio",
        ),
        (
            lambda row: row["decision"].__setitem__("shot_efficiency_go", None),
            "decision",
        ),
    ],
)
def test_contract_checker_rejects_tampering(record, mutation, message):
    broken = copy.deepcopy(record)
    mutation(broken)
    assert any(message in problem for problem in checker.contract_problems(broken))
