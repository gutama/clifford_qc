"""Contracts for the result-free R4a contextual-interaction declaration."""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_r4a_preregistration as r4a


@pytest.fixture(scope="module")
def config() -> dict:
    return r4a.load_config()


def _write(tmp_path, payload: dict):
    path = tmp_path / "r4a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_committed_preregistration_passes_every_static_gate(config):
    assert r4a.static_problems(config) == []


def test_it_contains_no_result_and_authorizes_only_a_later_screen(tmp_path, config):
    assert config["status"] == r4a.EXPECTED_STATUS
    assert config["claim_boundary"] == r4a.EXPECTED_CLAIM_BOUNDARY
    assert r4a.claim_problems(config) == []

    broken = copy.deepcopy(config)
    broken["selected_contextual_rung"] = 4
    with pytest.raises(ValueError, match="result field"):
        r4a.load_config(_write(tmp_path, broken))


def test_a_config_claiming_record_status_is_refused(tmp_path, config):
    broken = copy.deepcopy(config)
    broken["status"] = "structural_record"
    with pytest.raises(ValueError, match="must land later"):
        r4a.load_config(_write(tmp_path, broken))


def test_every_parent_input_is_cryptographically_bound(config):
    assert r4a.lineage_problems(config) == []
    assert set(r4a.EXPECTED_LINEAGE) == {
        row["path"] for row in config["parent_lineage"]["files"]
    }

    broken = copy.deepcopy(config)
    broken["parent_lineage"]["files"][0]["sha256"] = "0" * 64
    assert any(
        "declared sha256 drifted" in problem
        for problem in r4a.lineage_problems(broken)
    )


def test_system_and_acase_bank_are_inherited_without_reselection(config):
    system = config["system"]
    bank = system["acase_bank"]
    assert system["mapping"] == "jw"
    assert system["n_qubits"] == 8
    assert bank["basis_labels"] == [
        "I",
        "E(4,5<-0,1)",
        "E(6,7<-0,1)",
    ]
    assert bank["inherited_word_universe"] == 1223
    assert bank["selection_may_not_be_reopened"] is True
    assert r4a.system_problems(config) == []


def test_exactly_four_matched_factorial_arms_are_frozen(config):
    arms = config["comparison"]["arms"]
    assert [
        (
            row["name"],
            row["cost_symbol"],
            row["contextual_restriction"],
            row["adaptive_compression"],
        )
        for row in arms
    ] == r4a.EXPECTED_ARMS
    assert r4a.comparison_problems(config) == []

    broken = copy.deepcopy(config)
    broken["comparison"]["arms"].reverse()
    assert any(
        "four ordered comparator arms" in problem
        for problem in r4a.comparison_problems(broken)
    )


def test_contextual_constructor_and_ladder_are_frozen(config):
    contextual = config["contextual_restriction"]
    assert contextual["fixed_qubit_ladder"] == list(range(1, 8))
    assert contextual["tolerance"] == 1e-9
    assert contextual["tolerance_scope"] == (
        "reference expectation and compiled-image checks only; every represented "
        "non-identity Hamiltonian word remains a selection candidate regardless of "
        "coefficient magnitude"
    )
    assert "packed Pauli-word code" in contextual["term_order"]
    assert "GF(2) symplectic rank" in contextual["independence_rule"]
    assert "largest fixed-qubit count" in contextual["rung_rule"]
    assert r4a.contextual_problems(config) == []


def test_structural_gate_requires_every_arm_to_pass_before_sampling(config):
    screen = config["structural_screen"]
    assert screen["accuracy_target_millihartree"] == 1.6
    assert screen["admissible_bias_millihartree"] == pytest.approx(1.6 / 3.0)
    assert screen["word_universe_ceiling"] == 2048
    assert screen["block_sizes"] == [1, 2, 4, 8]
    assert screen["hamiltonian_removed_hs_fraction_denominator"] == (
        "Hilbert-Schmidt norm of the non-identity Hamiltonian; "
        "identity energy shifts are excluded"
    )
    assert screen["all_arms_must_clear_bias_margin"] is True
    assert screen["all_arms_must_clear_word_universe_ceiling"] is True
    assert r4a.structural_screen_problems(config) == []


def test_sample_protocol_is_frozen_but_not_authorized(config):
    protocol = config["conditional_sampled_protocol"]
    assert protocol["authorized_by_this_config"] is False
    assert protocol["search_endpoints_effective_shots_per_setting"] == (
        r4a.EXPECTED_SHOT_ENDPOINTS
    )
    assert protocol["exploratory_replicas"] == 30
    assert protocol["confirmatory_replicas"] == 100
    assert protocol["bootstrap_replicates"] == 10000
    assert protocol["one_sided_delta"] == 0.05
    assert r4a.sampled_protocol_problems(config) == []


def test_future_seed_roots_are_fresh_and_namespaced(config):
    roots = set(config["conditional_sampled_protocol"]["seed_roots"].values())
    assert roots == set(r4a.EXPECTED_SEED_ROOTS.values())
    historical = r4a._historical_seed_roots()
    assert not roots & historical

    broken = copy.deepcopy(config)
    old_root = next(iter(historical))
    broken["conditional_sampled_protocol"]["seed_roots"]["exploratory"] = old_root
    assert any(
        "aliases" in problem for problem in r4a.sampled_protocol_problems(broken)
    )


def test_future_device_cards_and_execution_stack_are_exact(config):
    protocol = config["conditional_sampled_protocol"]
    assert {
        row["name"]: row["sha256"] for row in protocol["device_cards"]
    } == r4a.EXPECTED_CARD_SHA256
    assert protocol["execution_environment"] == {
        "python_minor": "3.12",
        "numpy": "2.5.2",
        "scipy": "1.18.0",
        "stim": "1.16.0",
    }


def test_structural_and_sampled_results_must_land_separately(config):
    authorization = config["authorization"]
    reporting = config["reporting_contract"]
    assert authorization["exactly_one_later_structural_execution"] is True
    assert authorization["sampled_execution_authorized_by_this_config"] is False
    assert reporting["structural_result_commit_must_be_separate"] is True
    assert reporting["sampled_result_commit_must_be_separate"] is True

    broken = copy.deepcopy(config)
    broken["reporting_contract"]["sampled_result_commit_must_be_separate"] = False
    assert any(
        "sampled_result_commit_must_be_separate" in problem
        for problem in r4a.claim_problems(broken)
    )
