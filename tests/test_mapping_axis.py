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
