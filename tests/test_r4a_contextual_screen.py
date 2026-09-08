"""Contracts for the separately committed R4a structural screen."""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_r4a_contextual_screen as checker
import benchmarks.check_r4a_preregistration as preregistration
from benchmarks.run_r4a_contextual_screen import REFERENCE, _deduplicate_projected
from clifford_qc.multivector import MV
from clifford_qc.pauli import I, Z
from clifford_qc.subspace import Generator


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def test_projected_deduplication_is_scalar_free_and_source_ordered():
    source = [
        Generator("I", I(2)),
        Generator("first", Z(2, 0)),
        Generator("proportional", -2.0 * Z(2, 0)),
        Generator("zero", Z(2, 1)),
    ]
    kept, annihilated, duplicates = _deduplicate_projected(
        source,
        [I(2), Z(2, 0), -2.0 * Z(2, 0), MV(2)],
        tol=1e-9,
    )

    assert [row.label for row in kept] == ["I", "first"]
    assert annihilated == [3]
    assert duplicates == [
        {
            "candidate_index": 2,
            "candidate_label": "proportional",
            "retained_index": 1,
            "retained_label": "first",
        }
    ]


def test_committed_structural_record_satisfies_every_derived_contract(record):
    assert checker.contract_problems(record) == []


def test_every_rung_reports_the_four_frozen_arms_and_fields(record):
    expected_arms = [row[0] for row in preregistration.EXPECTED_ARMS]
    required = preregistration.EXPECTED_STRUCTURAL_FIELDS

    assert [row["fixed_qubits"] for row in record["contextual_rungs"]] == list(
        range(1, 8)
    )
    for rung in record["contextual_rungs"]:
        assert [arm["name"] for arm in rung["arms"]] == expected_arms
        assert all(required <= set(arm) for arm in rung["arms"])


def test_structural_decision_is_the_largest_four_arm_passing_rung(record):
    passing = [
        row["fixed_qubits"]
        for row in record["contextual_rungs"]
        if all(arm["admissible"] for arm in row["arms"])
    ]
    expected = max(passing) if passing else None

    assert record["structural_gate"]["passing_contextual_rungs"] == passing
    assert record["structural_gate"]["selected_contextual_rung"] == expected
    assert record["structural_gate"]["sampled_execution_authorized"] is False


def test_projected_basis_accounting_is_explicit(record):
    for rung in record["contextual_rungs"]:
        for arm in rung["arms"]:
            before = arm["basis_size_before_deduplication"]
            removed = len(arm["annihilated_candidate_indices"])
            duplicate = len(arm["duplicate_projected_candidate_indices"])
            assert arm["basis_size_after_deduplication"] == before - removed - duplicate
            assert 0 not in arm["annihilated_candidate_indices"]
            assert 0 not in arm["duplicate_projected_candidate_indices"]


def test_absolute_bias_and_tampered_gate_are_detected(record):
    broken = copy.deepcopy(record)
    arm = broken["contextual_rungs"][0]["arms"][1]
    arm["bias_millihartree"] = -abs(arm["bias_millihartree"])
    arm["admissible"] = not arm["admissible"]

    problems = checker.contract_problems(broken)
    assert any("bias is not the absolute error" in row for row in problems)
    assert any("admissible flag" in row for row in problems)


def test_sampled_or_cost_fields_are_rejected(record):
    broken = copy.deepcopy(record)
    broken["structural_gate"]["shots"] = 64

    assert any(
        "sampled/cost field" in row for row in checker.contract_problems(broken)
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("annihilated_candidate_indices", [True], "index lies outside"),
        ("word_universe", True, "invalid non-identity word universe"),
        ("settings_by_block_size", {"1": True, "2": 1, "4": 1, "8": 1},
         "setting counts must be non-negative integers"),
        ("retained_overlap_rank", True, "retained overlap rank"),
    ],
)
def test_boolean_values_are_rejected_for_integer_contracts(
    record, field, value, message
):
    broken = copy.deepcopy(record)
    broken["contextual_rungs"][0]["arms"][0][field] = value

    assert any(message in row for row in checker.contract_problems(broken))
