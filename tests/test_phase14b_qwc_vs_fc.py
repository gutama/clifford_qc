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


def test_committed_sampled_record_passes_every_internal_contract(record):
    assert checker.contract_problems(record) == []
    assert record["preregistration"]["merge_commit"] == PREREGISTRATION_MERGE_COMMIT


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
            lambda row: row["protocols"]["qwc"]["headline"][0]["shot_vector"].__setitem__(
                0, 1
            ),
            "shot vector",
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
