"""Contracts for the result-free Phase 14b QWC-versus-FC declaration."""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_phase14b_preregistration as phase14b


@pytest.fixture(scope="module")
def config() -> dict:
    return phase14b.load_config()


def _write(tmp_path, payload: dict):
    path = tmp_path / "phase14b.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_committed_preregistration_passes_every_static_gate(config):
    assert phase14b.static_problems(config) == []


def test_it_is_result_free_and_authorizes_only_a_later_record(tmp_path, config):
    assert config["status"] == phase14b.EXPECTED_STATUS
    assert config["claim_boundary"] == phase14b.EXPECTED_CLAIM_BOUNDARY
    assert phase14b.claim_problems(config) == []

    broken = copy.deepcopy(config)
    broken["winner"] = "fully_commuting"
    with pytest.raises(ValueError, match="result field"):
        phase14b.load_config(_write(tmp_path, broken))


def test_a_config_claiming_record_status_is_refused(tmp_path, config):
    broken = copy.deepcopy(config)
    broken["status"] = "record"
    with pytest.raises(ValueError, match="later commit"):
        phase14b.load_config(_write(tmp_path, broken))


def test_every_parent_input_is_cryptographically_bound(config):
    assert phase14b.lineage_problems(config) == []
    assert set(phase14b.EXPECTED_LINEAGE) == {
        row["path"] for row in config["parent_lineage"]["files"]
    }

    broken = copy.deepcopy(config)
    broken["parent_lineage"]["files"][0]["sha256"] = "0" * 64
    assert any("declared sha256 drifted" in problem
               for problem in phase14b.lineage_problems(broken))


def test_the_fixed_bank_splits_identity_from_1814_measured_words(config):
    system = config["system"]
    assert system["basis_labels"] == phase14b.EXPECTED_LABELS
    assert system["total_pauli_words_including_identity"] == 1815
    assert system["measured_word_universe"] == 1814
    assert system["ritz_functional_measured_support"] == 966
    assert phase14b.system_problems(config) == []


def test_only_the_qwc_and_fully_commuting_endpoints_are_admitted(config):
    endpoints = {row["name"]: row for row in config["comparison"]["endpoints"]}
    assert endpoints == {
        name: {"name": name, **row}
        for name, row in phase14b.EXPECTED_ENDPOINTS.items()
    }
    assert phase14b.comparison_problems(config) == []

    broken = copy.deepcopy(config)
    broken["comparison"]["endpoints"][1]["block_size"] = 4
    assert any("fully_commuting block_size" in problem
               for problem in phase14b.comparison_problems(broken))


def test_both_endpoints_use_one_outcome_independent_allocator(config):
    allocator = config["protocol"]["allocator"]
    assert allocator["name"] == "coefficient_range_neyman"
    assert allocator["outcome_independent"] is True
    assert allocator["same_rule_for_both_endpoints"] is True
    assert allocator["min_shots_per_setting"] == 2

    broken = copy.deepcopy(config)
    broken["protocol"]["allocator"]["outcome_independent"] = False
    assert any("allocator" in problem for problem in phase14b.protocol_problems(broken))


def test_confidence_family_covers_both_protocols_and_all_seventeen_looks(config):
    protocol = config["protocol"]
    confidence = protocol["confidence"]
    assert confidence["delta"] == 0.05
    assert confidence["family"] == 2
    assert confidence["rounds"] == 17
    assert protocol["total_physical_shot_endpoints"] == (
        phase14b.EXPECTED_SHOT_ENDPOINTS
    )
    assert all(right == 2 * left for left, right in zip(
        phase14b.EXPECTED_SHOT_ENDPOINTS,
        phase14b.EXPECTED_SHOT_ENDPOINTS[1:],
    ))


def test_seed_roots_are_fresh_and_namespaced(config):
    roots = set(config["protocol"]["seed_roots"].values())
    assert len(roots) == 2
    historical = phase14b._historical_seed_roots()
    assert 132813000 in historical  # nested R3c seed_roots.exploratory
    assert not roots & historical

    broken = copy.deepcopy(config)
    broken["protocol"]["seed_roots"]["headline"] = 132813000
    assert any("aliases" in problem for problem in phase14b.protocol_problems(broken))


def test_matrix_and_covariance_failure_gates_are_binding(config):
    protocol = config["protocol"]
    assert protocol["matrix_reconstruction"][
        "max_absolute_entry_error_tolerance"
    ] == 5e-13
    assert protocol["covariance_audit"]["replicas"] == 1000
    assert protocol["covariance_audit"]["estimators"] == [
        "single_assignment",
        "pooled",
    ]
    assert protocol["covariance_audit"][
        "acceptable_empirical_to_predicted_interval"
    ] == [0.8, 1.2]
    assert "suppresses the efficiency verdict" in config["acceptance"]["failure_rule"]


def test_cards_retain_their_connectivity_and_nonhardware_boundary(config):
    assert phase14b.card_problems(config) == []
    cards = {row["name"]: row for row in config["protocol"]["device_cards"]}
    assert cards["superconducting-like"]["connectivity"] == "sparse-nearest-neighbor"
    assert cards["superconducting-like"]["routing"] is True
    assert "not state-preparation" in config["reporting"]["evidence_boundary"]


def test_materiality_and_censoring_rules_cannot_move_after_the_draw(config):
    acceptance = config["acceptance"]
    assert acceptance["material_shot_reduction_fraction"] == 0.5
    assert "0.5 times" in acceptance["shot_efficiency_go_rule"]
    assert "right-censored above 4294967296" in acceptance[
        "certified_endpoint_rule"
    ]

    broken = copy.deepcopy(config)
    broken["acceptance"]["material_shot_reduction_fraction"] = 0.25
    assert any("materiality" in problem
               for problem in phase14b.acceptance_problems(broken))
