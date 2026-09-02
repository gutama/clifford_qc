"""Contracts for the R3b preregistration.

A preregistration's whole value is that it was fixed before the sampling, so
the failure mode worth testing is a config that looks preregistered while
describing a bank chosen after the fact -- or one that quietly relaxes a gate
QR3b froze. Each test below is one of those.
"""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.check_r3b_preregistration import (
    CONFIG,
    _result_key_paths,
    digest_problems,
    gate_problems,
    lineage_problems,
    load_config,
    protocol_problems,
    rederivation_problems,
)
from benchmarks.run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
from benchmarks.run_priceability_screen import REFERENCE as SCREEN_RECORD


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


def _write(tmp_path, config: dict):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_the_committed_preregistration_passes_every_static_gate(config):
    assert digest_problems(config) == []
    assert protocol_problems(config) == []
    assert gate_problems(config) == []
    assert lineage_problems(config) == []
    assert _result_key_paths(config) == []


def test_it_declares_itself_a_preregistration_and_carries_no_results(config):
    assert config["status"] == "preregistration_only_no_run_yet"
    assert config["acceptance_gates"]["full_30_plus_100_run_authorized_by_this_config"] is False
    assert "no sampling has been performed" in config["claim_boundary"].lower()


def test_a_config_claiming_record_status_is_refused(tmp_path, config):
    broken = copy.deepcopy(config)
    broken["status"] = "record"
    with pytest.raises(ValueError, match="preregistration"):
        load_config(_write(tmp_path, broken))


def test_a_file_path_named_provenance_is_not_mistaken_for_a_record_stamp(config):
    """The false positive the first draft of this checker had.

    ``candidate.provenance`` is the committed provenance *file*; a record's
    execution stamp is an object under the same key. Only the second makes a
    config a record.
    """
    assert isinstance(config["candidate"]["provenance"], str)
    assert _result_key_paths(config) == []

    stamped = copy.deepcopy(config)
    stamped["provenance"] = {"schema": "clifford_qc.execution_provenance.v1"}
    assert any("provenance" in p for p in _result_key_paths(stamped))


@pytest.mark.parametrize("key", ["decision", "verdict", "k_star", "scoping_probe"])
def test_an_injected_result_field_is_caught(config, key):
    broken = copy.deepcopy(config)
    broken[key] = {"status": "eligible"}
    assert any(key in p for p in _result_key_paths(broken))


def test_a_drifted_input_digest_is_caught(config):
    for field in ("source_sha256", "provenance_sha256"):
        broken = copy.deepcopy(config)
        broken["candidate"][field] = "0" * 64
        assert any("drifted" in p for p in digest_problems(broken))


def test_a_drifted_lineage_digest_is_caught(config):
    for path in (("admitted_by", "record_sha256"), ("admitted_by", "config_sha256"),
                 ("extends", "config_sha256")):
        broken = copy.deepcopy(config)
        broken["parent_lineage"][path[0]][path[1]] = "0" * 64
        assert any("no longer binds" in p for p in digest_problems(broken))


def test_the_frozen_search_grid_may_not_move(config):
    broken = copy.deepcopy(config)
    broken["protocol"]["search_endpoints_effective_shots_per_setting"] = [
        *SEARCH_ENDPOINTS, 262144,
    ]
    assert any("may not change the frozen" in p for p in protocol_problems(broken))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("block_sizes", [1, 2]),
        ("estimators", ["single_assignment"]),
        ("exploratory_replicas", 30),
        ("confirmatory_replicas", 100),
    ],
)
def test_protocol_drift_from_qr3b_is_caught(config, field, value):
    broken = copy.deepcopy(config)
    broken["protocol"][field] = value
    assert any(field in p for p in protocol_problems(broken))


def test_seed_roots_may_not_alias_the_qr3b_streams(config):
    """Two probes on the same instance must not share replica streams."""
    frozen = json.loads(
        (CONFIG.parent / "qr3b_instance_preflight.json").read_text(encoding="utf-8")
    )["protocol"]["seed_roots"]
    assert not set(config["protocol"]["seed_roots"].values()) & set(frozen.values())

    broken = copy.deepcopy(config)
    broken["protocol"]["seed_roots"]["exploratory"] = frozen["exploratory"]
    assert any("alias" in p for p in protocol_problems(broken))


def test_duplicate_seed_roots_are_caught(config):
    broken = copy.deepcopy(config)
    roots = broken["protocol"]["seed_roots"]
    roots["confirmatory"] = roots["exploratory"]
    assert any("distinct" in p for p in protocol_problems(broken))


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("full_30_plus_100_run_authorized_by_this_config", True, "authorize a full run"),
        ("selection_may_not_use_mapping_cost_direction", False, "cost-direction"),
        ("word_universe_ceiling", 8192, "ceiling differs"),
        ("accuracy_target_millihartree", 3.2, "accuracy target differs"),
        ("all_probe_cells_must_resolve_strictly_before", 262144, "frozen search ceiling"),
        ("selection_must_stop_intrinsically", True, "must say so"),
        ("selection_must_stop_intrinsically_basis", "", "declare its reason"),
    ],
)
def test_a_relaxed_gate_is_caught(config, field, value, needle):
    broken = copy.deepcopy(config)
    broken["acceptance_gates"][field] = value
    assert any(needle in p for p in gate_problems(broken))


def test_the_margin_rule_may_not_be_claimed_as_originating_here(config):
    """R3S declared the rule; this config is only its first preregistered use."""
    assert config["candidate"]["selection_rule_is_preregistered_here"] is False
    assert config["candidate"]["selection_rule_preregistration_note"]

    broken = copy.deepcopy(config)
    broken["candidate"]["selection_rule_is_preregistered_here"] = True
    assert any("may not claim to originate" in p for p in gate_problems(broken))


def test_the_bank_is_a_prefix_of_the_ordering_qr3b_froze(config):
    frozen = json.loads(
        (CONFIG.parent / "qr3b_instance_preflight.json").read_text(encoding="utf-8")
    )["candidate"]["selected_labels"]
    declared = config["candidate"]["selected_labels"]
    assert frozen[: len(declared)] == declared

    broken = copy.deepcopy(config)
    broken["candidate"]["selected_labels"] = ["I", "E(4,5<-2,3)"]
    assert any("not a prefix" in p for p in lineage_problems(broken))


def test_the_declared_bank_matches_the_r3s_record(config):
    stop = next(
        c for c in json.loads(SCREEN_RECORD.read_text(encoding="utf-8"))["candidates"]
        if c["candidate"] == "lih_cas4e4o"
    )["margin_stop"]
    candidate = config["candidate"]
    assert list(stop["labels"]) == list(candidate["selected_labels"])
    assert stop["basis_size"] == candidate["expected_basis_size"] == 2
    assert stop["binding_word_universe"] == candidate["expected_binding_word_universe"] == 1439
    assert stop["binding_word_universe"] < 1814  # BeH2, the one priced bank
    for arm in stop["arms"]:
        assert candidate["expected_arm_word_universe"][arm["mapping"]] == arm["word_universe"]
        assert candidate["expected_arm_settings"][arm["mapping"]] == arm["settings"]


def test_the_committed_bank_reproduces_when_rebuilt(config):
    """The check that stops the preregistration being a wish. Rebuilds the bank."""
    assert rederivation_problems(config) == []


def test_a_hand_picked_prefix_is_caught(config):
    """A bank chosen for cheapness rather than by the rule fails re-derivation.

    The rule selects M=2 on this ordering. Declaring any other prefix -- even a
    longer, more accurate one -- means the declared bank is not the one the
    declared rule picks, which is exactly the cost-direction selection QR3b's
    gate forbids.
    """
    broken = copy.deepcopy(config)
    broken["candidate"]["expected_basis_size"] = 3
    broken["candidate"]["selected_labels"] = ["I", "E(6,7<-2,3)", "E(4,5<-2,3)"]
    problems = rederivation_problems(broken)
    assert any("not the one the rule picks" in p for p in problems)
