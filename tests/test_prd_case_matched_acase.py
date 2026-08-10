"""Validity-revised manifest and matched-resolution A-CASE producer gates."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from benchmarks import run_prd_case_matched_acase as matched
from benchmarks import run_prd_case_paper_suite as suite


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "benchmarks" / "configs" / "prd_case_paper_suite.json"
V2_CONFIG = ROOT / "benchmarks" / "configs" / "prd_case_paper_suite_v2.json"


def test_frozen_v1_is_unchanged_while_v2_caps_only_invalid_rank_budgets():
    v1 = suite.load_manifest(V1_CONFIG)
    v2 = suite.load_manifest(V2_CONFIG)
    assert v1["schema"] == suite.SCHEMA
    assert v2["schema"] == suite.CONFIRMATORY_SCHEMA
    assert len(suite.build_tasks(v1)) == 61
    assert len(suite.build_tasks(v2)) == 49

    budgets = {system["id"]: system["m_budgets"]
               for system in v2["systems"] if system["default_run"]}
    assert budgets["h4_equilibrium"] == [3, 5, 7, 11]
    assert budgets["hubbard_2x2_u4"] == [3, 5, 7]
    assert budgets["hubbard_2x3_u4"] == [3, 5, 7, 11, 21]
    assert budgets["fcidump_h4_equilibrium"] == [7]


def test_v2_records_the_pilot_without_relabelling_it_confirmatory():
    manifest = suite.load_manifest(V2_CONFIG)
    pilot = manifest["pilot_provenance"]
    assert pilot["pilot_completed_records"] == 49
    assert pilot["pilot_invariant_rejections"] == 12
    assert pilot["selection_rule_declared_before_new_acase_results"] is True
    assert "never an energy" in pilot["selection_rule"]
    assert manifest["manifest_revision"]["evidence_role"].startswith(
        "The v1 preconditioned-expansion results are transparent pilot")


def test_matched_plan_is_one_nested_task_per_system_and_resolution():
    manifest = suite.load_manifest(V2_CONFIG)
    tasks = matched.build_tasks(manifest)
    assert len(tasks) == 26
    assert sum(task["analysis_role"] == "primary" for task in tasks) == 24
    assert sum(task["analysis_role"] == "construction_validation"
               for task in tasks) == 2
    system = [task for task in tasks if task["system"] == "hubbard_2x3_u4"]
    assert {task["method"] for task in system} == set(matched.METHODS)
    assert all(task["m_budgets"] == [3, 5, 7, 11, 21] for task in system)


@pytest.fixture(scope="module")
def plaquette_records():
    manifest = suite.load_manifest(V2_CONFIG)
    tasks = matched.build_tasks(
        manifest, system_ids=("hubbard_2x2_u4",))
    records = {}
    for task in tasks:
        small = copy.deepcopy(task)
        small["m_budgets"] = [3]
        records[task["method"]] = matched.execute_task(small, manifest)
    return manifest, records


def test_two_resolutions_match_the_state_span_but_not_the_word_ledger(
        plaquette_records):
    _, records = plaquette_records
    coarse = records["acase_determinant"]["rows"][0]
    fine = records["acase_word"]["rows"][0]
    assert coarse["energy"] == pytest.approx(fine["energy"], abs=1e-12)
    assert coarse["direction_fingerprints"] == fine["direction_fingerprints"]
    assert coarse["W_total"] != fine["W_total"]
    for row in (coarse, fine):
        assert row["retained_rank"] == row["M"] == 3
        assert row["W_selection_cache"] >= row["W_total"]
        assert row["true_residual"] ** 2 == pytest.approx(row["variance"])

    certificate = records["acase_word"]["sector_certificate"]
    assert certificate["max_sector_leakage"] == 0.0
    assert certificate["min_sector_weight"] == 1.0


def test_resolution_cross_check_is_a_producer_gate(tmp_path, plaquette_records):
    _, records = plaquette_records
    tasks = []
    for method, record in records.items():
        task_id = f"hubbard_2x2_u4__{method}"
        tasks.append({"task_id": task_id, "system": "hubbard_2x2_u4",
                      "method": method})
        suite._atomic_json(tmp_path / f"{task_id}.json", record)
    checks = matched.cross_check_resolution_records(tasks, tmp_path)
    assert checks["checks"] == [{
        "system": "hubbard_2x2_u4",
        "M": 3,
        "energy_gap": pytest.approx(0.0, abs=1e-12),
        "energy_tolerance": 1e-10,
        "direction_fingerprints_match": True,
        "status": "passed",
    }]
