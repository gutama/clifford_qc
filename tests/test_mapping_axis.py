"""Contracts for the R2b raw-pool mapping-axis benchmark."""

import copy
import json

from benchmarks.check_mapping_axis import contract_problems
from benchmarks.run_mapping_axis import REFERENCE, build_system_record, load_config


def test_committed_mapping_axis_record_passes_its_contract():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert contract_problems(record) == []


def test_contract_rejects_pricing_above_the_bias_floor():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    broken = copy.deepcopy(record)
    h4_jw = broken["systems"][0]["arms"][0]["accuracy_matched_cost"]
    h4_jw["status"] = "priced"
    h4_jw["device_costs"] = {"logical-alltoall": {}}
    problems = contract_problems(broken)
    assert any("status disagrees with bias floor" in problem for problem in problems)
    assert any("unattainable target was priced" in problem for problem in problems)


def test_contract_gates_serialized_numerical_invariants():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    broken = copy.deepcopy(record)
    invariant = broken["systems"][0]["arms"][0]["invariants"]
    invariant["max_hamiltonian_matrix_error"] = 1e3
    problems = contract_problems(broken)
    assert any(
        "max_hamiltonian_matrix_error exceeds its stored tolerance" in problem
        for problem in problems
    )


def test_contract_requires_a_reason_for_an_unmaterialized_dense_oracle():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    broken = copy.deepcopy(record)
    invariant = broken["systems"][2]["arms"][0]["invariants"]
    assert invariant["dense_spectrum_checked"] is False
    invariant["dense_spectrum_reason"] = None
    problems = contract_problems(broken)
    assert any(
        "dense-spectrum omission has no declared reason" in problem
        for problem in problems
    )


def test_contract_reports_malformed_fields_instead_of_raising():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    mutations = (
        (
            "target",
            lambda item: item["measurement_contract"].update(
                accuracy_target_millihartree=None
            ),
        ),
        (
            "shots",
            lambda item: item["measurement_contract"].update(
                uniform_raw_shots_per_setting=None
            ),
        ),
        (
            "energy",
            lambda item: item["systems"][0]["arms"][0]["basis"].update(
                ground_energy=None
            ),
        ),
    )
    for label, mutate in mutations:
        broken = copy.deepcopy(record)
        mutate(broken)
        problems = contract_problems(broken)
        assert problems, label
        assert not any("Traceback" in problem for problem in problems)


def test_contract_recomputes_qr3_ratio_verdict():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    broken = copy.deepcopy(record)
    summary = broken["qr3"]["structural"]["qwc_settings_matched_greedy"]
    summary["max_mapping_spread_factor"] = 1.0
    problems = contract_problems(broken)
    assert any(problem.startswith("QR3 structural:") for problem in problems)


def test_jw_rebuild_preserves_the_h4_physical_bank():
    config = load_config()
    spec = next(system for system in config["systems"] if system["key"] == "h4")
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))["systems"][0]
    actual = build_system_record(
        spec,
        arms=["jw"],
        cards=[],
        measurement=config["measurement"],
    )
    assert actual["raw_pool"] == expected["raw_pool"]
    assert actual["selected_domain"] == expected["selected_domain"]
    assert actual["arms"][0]["invariants"]["word_bijection_checked"] is True
    assert actual["arms"][0]["measurement"]["settings"] == (
        expected["arms"][0]["measurement"]["settings"]
    )
