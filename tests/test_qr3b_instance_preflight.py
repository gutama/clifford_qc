"""Contracts for the lineage-preserving QR3b LiH preflight."""

from __future__ import annotations

import json

import pytest

from benchmarks.check_qr3b_instance_preflight import contract_problems
from benchmarks.run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
from benchmarks.run_qr3b_instance_preflight import (
    REFERENCE,
    derive_selection,
    load_config,
    resolution_decision,
)


def _probe(endpoint, *, bias=0.1):
    arms = []
    for mapping in ("jw", "parity", "parity+2q", "bk", "bk+2q"):
        arms.append(
            {
                "mapping": mapping,
                "exact_subspace_bias_millihartree": bias,
                "rungs": [
                    {
                        "block_size": 1,
                        "estimators": {
                            estimator: {
                                "shot_to_target": {
                                    "status": (
                                        "confirmed_bracket"
                                        if endpoint is not None
                                        else "unresolved_at_grid_ceiling"
                                    ),
                                    "confirmed_passing_effective_shots_per_setting": endpoint,
                                }
                            }
                            for estimator in ESTIMATORS
                        },
                    }
                ],
            }
        )
    return {"arms": arms}


def test_config_freezes_the_parent_lineage_and_search_grid():
    config = load_config()
    assert config["parent_lineage"]["merged_pull_request"] == 67
    assert "right-censored" in config["parent_lineage"]["preserved_result"]
    assert tuple(
        config["protocol"]["search_endpoints_effective_shots_per_setting"]
    ) == SEARCH_ENDPOINTS
    assert config["protocol"]["probe_is_a_cost_record"] is False
    assert config["acceptance_gates"]["full_30_plus_100_run_authorized_by_this_config"] is False


def test_lih_selection_is_rederived_and_stops_intrinsically():
    config = load_config()
    evidence = derive_selection(config)
    assert evidence["budget_is_nonbinding"] is True
    assert evidence["stopping_reason"] == "predicted lowering below threshold"
    assert evidence["labels"] == config["candidate"]["selected_labels"]
    assert evidence["basis_size"] == 13
    assert evidence["bias_millihartree"] == pytest.approx(0.000195923, abs=1e-9)


def test_a_probe_with_one_grid_rung_of_headroom_is_eligible():
    decision = resolution_decision(_probe(16_384), load_config())
    assert decision["status"] == "eligible_for_separately_authorized_full_run"
    assert decision["eligible_for_full_run"] is True
    assert decision["preferred_headroom_gate_passes"] is True
    assert decision["full_run_authorized"] is False
    assert decision["qr3b_verdict"] == "not_evaluated_by_preflight"


def test_a_crossing_at_the_frozen_ceiling_has_no_required_headroom():
    decision = resolution_decision(_probe(65_536), load_config())
    assert decision["status"] == "rejected_no_endpoint_headroom"
    assert decision["eligible_for_full_run"] is False
    assert decision["cells_at_or_beyond_ceiling"] == 10


def test_an_unresolved_probe_is_rejected_without_becoming_a_cost_failure():
    decision = resolution_decision(_probe(None), load_config())
    assert decision["status"] == "rejected_unresolved_at_frozen_grid"
    assert decision["unresolved_cells"] == 10
    assert decision["qr3b_verdict"] == "not_evaluated_by_preflight"


def test_bias_floor_is_a_separate_gate_from_resolution():
    decision = resolution_decision(_probe(4_096, bias=2.0), load_config())
    assert decision["status"] == "rejected_bias_floor"
    assert decision["resolution_gate_passes"] is True
    assert decision["bias_gate_passes"] is False


def test_committed_record_contains_no_cost_or_qr3b_verdict_derivatives():
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert contract_problems(record) == []


def test_custom_bootstrap_root_is_used_by_endpoint_summaries(monkeypatch):
    import benchmarks.run_exact_shot_search as search

    seen = []

    def fake_summary(rows, exact_energy, *, bootstrap_seed):
        seen.append(bootstrap_seed)
        return {"passes_target": False}

    monkeypatch.setattr(search, "_summary", fake_summary)
    raw = {
        estimator: {64: [object()]}
        for estimator in ESTIMATORS
    }
    search._summaries(
        raw,
        -1.0,
        block_size=1,
        seed_namespace=0,
        bootstrap_seed=281_013_000,
    )
    assert len(seen) == len(ESTIMATORS)
    assert len(set(seen)) == len(ESTIMATORS)
