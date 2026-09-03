"""Contracts for the result-free R3c full-cost preregistration."""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_r3c_preregistration as r3c
from benchmarks.run_exact_shot_search import (
    CONFIRMATORY_REPLICAS,
    EXPLORATORY_REPLICAS,
    SEARCH_ENDPOINTS,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return r3c.load_config()


def _write(tmp_path, payload: dict):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_committed_preregistration_passes_every_static_gate(config):
    assert r3c.static_problems(config) == []


def test_the_inherited_bank_still_rebuilds_deterministically():
    inherited = json.loads(r3c.R3B_CONFIG.read_text(encoding="utf-8"))
    assert r3c.r3b_lineage_problems(inherited) == []
    assert r3c.r3b_rederivation_problems(inherited) == []


def test_it_is_result_free_and_uses_a_tenseless_boundary(config):
    assert config["status"] == "preregistration_only_no_run_yet"
    assert config["claim_boundary"] == r3c.EXPECTED_CLAIM_BOUNDARY
    assert r3c.claim_problems(config) == []


def test_a_config_claiming_record_status_is_refused(tmp_path, config):
    broken = copy.deepcopy(config)
    broken["status"] = "record"
    with pytest.raises(ValueError, match="later commit"):
        r3c.load_config(_write(tmp_path, broken))


def test_the_r3b_config_and_record_are_cryptographically_bound(config):
    assert r3c._canonical_sha256(json.loads(r3c.R3B_CONFIG.read_text())) == (
        r3c.R3B_CONFIG_CANONICAL_SHA256
    )
    assert r3c._git_blob_sha1(r3c.R3B_RECORD) == r3c.R3B_RECORD_GIT_BLOB_SHA1

    broken = copy.deepcopy(config)
    broken["parent_lineage"]["pilot"]["record_git_blob_sha1"] = "0" * 40
    assert any("record_git_blob_sha1" in p for p in r3c.lineage_problems(broken))


def test_the_rejected_pilot_finding_is_preserved(config):
    assert r3c.pilot_problems(config) == []

    broken = copy.deepcopy(config)
    broken["parent_lineage"]["pilot"]["preserved_finding"] = ""
    assert any("finding" in p for p in r3c.pilot_problems(broken))


def test_the_candidate_and_mapping_arms_are_exactly_r3bs(config):
    assert r3c.candidate_problems(config) == []

    broken = copy.deepcopy(config)
    broken["candidate"]["selected_labels"].append("E(4,5<-2,3)")
    assert any("candidate differs" in p for p in r3c.candidate_problems(broken))


def test_the_headline_protocol_is_30_plus_100_on_the_frozen_grid(config):
    protocol = config["protocol"]
    assert protocol["exploratory_replicas"] == EXPLORATORY_REPLICAS == 30
    assert protocol["confirmatory_replicas"] == CONFIRMATORY_REPLICAS == 100
    assert protocol["search_endpoints_effective_shots_per_setting"] == list(SEARCH_ENDPOINTS)
    assert protocol["execution_environment"]["numpy"] == "2.5.2"

    broken = copy.deepcopy(config)
    broken["protocol"]["search_endpoints_effective_shots_per_setting"].append(262144)
    assert any("search_endpoints" in p for p in r3c.protocol_problems(broken))


def test_seed_roots_are_distinct_and_disjoint_from_every_prior_stream(config):
    roots = set(config["protocol"]["seed_roots"].values())
    assert len(roots) == 3
    assert not roots & r3c._historical_seed_roots()

    broken = copy.deepcopy(config)
    broken["protocol"]["seed_roots"]["exploratory"] = next(
        iter(r3c._historical_seed_roots())
    )
    assert any("aliases" in p for p in r3c.protocol_problems(broken))


def test_device_card_hashes_are_the_existing_cost_layer_cards(config):
    assert r3c.protocol_problems(config) == []

    broken = copy.deepcopy(config)
    broken["protocol"]["device_cards"][0]["sha256"] = "0" * 64
    assert any("device-card" in p for p in r3c.protocol_problems(broken))

    duplicate = copy.deepcopy(config)
    duplicate["protocol"]["device_cards"].append(
        copy.deepcopy(duplicate["protocol"]["device_cards"][0])
    )
    assert any("duplicate name" in p for p in r3c.protocol_problems(duplicate))

    malformed = copy.deepcopy(config)
    del malformed["protocol"]["device_cards"][0]["sha256"]
    assert any("valid SHA-256" in p for p in r3c.protocol_problems(malformed))


def test_pass_censoring_and_separate_result_commit_are_frozen(config):
    assert r3c.rule_problems(config) == []

    broken = copy.deepcopy(config)
    broken["reporting_contract"]["result_commit_must_be_separate"] = False
    assert any("later commit" in p for p in r3c.rule_problems(broken))


def test_result_fields_and_temporal_claims_are_caught(config):
    result_bearing = copy.deepcopy(config)
    result_bearing["decision"] = {"status": "priced"}
    assert any("$.decision" in p for p in r3c.claim_problems(result_bearing))

    temporal = copy.deepcopy(config)
    temporal["claim_boundary"] = "No sampling has been performed under this config."
    problems = r3c.claim_problems(temporal)
    assert any("temporal" in p for p in problems)
