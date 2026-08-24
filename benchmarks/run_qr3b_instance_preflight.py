#!/usr/bin/env python3
"""QR3b independent-instance preflight, not an accuracy-matched cost record.

The frozen R3 result needs two independently priceable physical instances to
compare mapping spread with instance spread at the exact tier.  H4-converged
clears the bias floor but is right-censored by the frozen 65,536-shot grid.
This producer evaluates whether a chemically independent LiH CAS(4e,4o) bank
is suitable for a *separately labelled* QR3b extension.

The 2+2-replica search is deliberately a scope-decision probe.  It may reject
or admit the candidate for a later 30+100 run; it does not price C(epsilon),
answer QR3b, or alter H4's status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import run_acase

try:  # package import in tests versus direct benchmark execution
    from benchmarks.run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from benchmarks.run_mapping_axis import (
        _build_model,
        _canonical_sha256,
        _raw_pool,
        build_system_record as build_mapping_system,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from benchmarks.run_protocol_axis import (
        build_system_record as build_protocol_system,
    )
    from benchmarks.run_protocol_cost import build_record as build_cost_record
except ImportError:  # pragma: no cover
    from run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from run_mapping_axis import (
        _build_model,
        _canonical_sha256,
        _raw_pool,
        build_system_record as build_mapping_system,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from run_protocol_axis import build_system_record as build_protocol_system
    from run_protocol_cost import build_record as build_cost_record


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "qr3b_instance_preflight.json"
REFERENCE = HERE / "reference_results" / "qr3b_instance_preflight.json"
SCHEMA = "clifford_qc.qr3b_instance_preflight.v1"
CONFIG_SCHEMA = "clifford_qc.qr3b_instance_preflight_config.v1"
COST_DERIVATIVE_KEYS = {
    "cost_bracket",
    "device_costs",
    "k_star",
    "qr3_accuracy_matched",
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported QR3b preflight config schema")
    protocol = config.get("protocol", {})
    if tuple(protocol.get("search_endpoints_effective_shots_per_setting", ())) != tuple(
        SEARCH_ENDPOINTS
    ):
        raise ValueError("QR3b may not change the frozen R1/R3 search grid")
    if protocol.get("exploratory_replicas") != 2 or protocol.get(
        "confirmatory_replicas"
    ) != 2:
        raise ValueError("the scope-decision probe is frozen at 2+2 replicas")
    if protocol.get("probe_is_a_cost_record") is not False:
        raise ValueError("the QR3b scoping probe must disclaim cost-record status")
    roots = protocol.get("seed_roots", {})
    if set(roots) != {"exploratory", "confirmatory", "bootstrap"}:
        raise ValueError("the probe must declare all three seed roots")
    if any(not isinstance(value, int) or value < 0 for value in roots.values()):
        raise ValueError("seed roots must be non-negative integers")
    if len(set(roots.values())) != 3:
        raise ValueError("seed roots must be distinct")

    candidate = config.get("candidate", {})
    if candidate.get("key") != "lih_cas4e4o":
        raise ValueError("this preregistration freezes exactly one LiH candidate")
    if candidate.get("unbound_implementation_budget", 0) <= 26:
        raise ValueError("the implementation budget must exceed the complete raw pool")
    source = HERE.parent / candidate["source"]
    provenance = HERE.parent / candidate["provenance"]
    if _file_sha256(source) != candidate.get("source_sha256"):
        raise ValueError("LiH FCIDUMP digest drifted")
    if _file_sha256(provenance) != candidate.get("provenance_sha256"):
        raise ValueError("LiH provenance digest drifted")
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    if payload.get("fcidump_sha256") != candidate["source_sha256"]:
        raise ValueError("LiH provenance does not bind the committed FCIDUMP")

    gates = config.get("acceptance_gates", {})
    if gates.get("source_qubits") != 8:
        raise ValueError("QR3b freezes an eight-qubit source instance")
    if gates.get("all_probe_cells_must_resolve_strictly_before") != SEARCH_ENDPOINTS[-1]:
        raise ValueError("the resolution gate must name the frozen search ceiling")
    if gates.get("full_30_plus_100_run_authorized_by_this_config") is not False:
        raise ValueError("the preflight must not authorize a full run")
    ceiling = gates.get("word_universe_ceiling")
    if not isinstance(ceiling, int) or ceiling <= 0:
        raise ValueError("the candidate screen must declare a word-universe ceiling")
    if not gates.get("word_universe_ceiling_basis"):
        raise ValueError("the word-universe ceiling must declare what calibrates it")
    return config


def derive_selection(config: dict) -> dict:
    spec = config["candidate"]
    model, _ = _build_model(spec)
    sector = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact = float(sector.ground_state(model.hamiltonian, k=1, method="dense")[0][0])
    reference = ExactMVBackend().state(model.reference, ())
    pool = _raw_pool(model)
    run = run_acase(
        reference,
        model.hamiltonian,
        pool,
        max_size=int(spec["unbound_implementation_budget"]),
        exact_ground_energy=exact,
    )
    labels = list(run.result.basis_labels)
    if run.stopped_reason != spec["expected_stopping_reason"]:
        raise ValueError("LiH selection did not stop on its preregistered intrinsic rule")
    if labels != list(spec["selected_labels"]):
        raise ValueError("LiH selected labels drifted from the preregistration")
    energy = float(run.result.ground_energy)
    if abs(energy - float(spec["expected_selected_energy"])) > float(
        spec["selected_energy_tolerance"]
    ):
        raise ValueError("LiH selected energy drifted from the preregistration")
    return {
        "raw_pool_size": len(pool),
        "implementation_budget": int(spec["unbound_implementation_budget"]),
        "budget_is_nonbinding": int(spec["unbound_implementation_budget"]) > len(pool),
        "stopping_reason": run.stopped_reason,
        "basis_size": len(labels),
        "labels": labels,
        "selected_energy": energy,
        "exact_sector_energy": exact,
        "bias_millihartree": abs(energy - exact) * 1e3,
    }


def _probe_config(config: dict) -> dict:
    key = config["candidate"]["key"]
    return {
        "schema": "clifford_qc.protocol_axis_config.v1",
        "systems": [key],
        "mapping_arms": list(config["mapping_arms"]),
        "protocol": {"block_sizes": list(config["protocol"]["block_sizes"])},
        "cost_layer": {"systems": [key], "deferred": []},
    }


def _sanitize_probe(system: dict) -> dict:
    """Retain sampling evidence while removing every cost/verdict derivative."""
    arms = []
    for arm in system["arms"]:
        rungs = []
        for rung in arm["rungs"]:
            rungs.append(
                {
                    "block_size": rung["block_size"],
                    "settings": rung["settings"],
                    "estimators": {
                        estimator: {
                            "exploration": rung["estimators"][estimator]["exploration"],
                            "confirmation": rung["estimators"][estimator]["confirmation"],
                            "shot_to_target": rung["estimators"][estimator][
                                "shot_to_target"
                            ],
                        }
                        for estimator in ESTIMATORS
                    },
                }
            )
        arms.append(
            {
                "mapping": arm["mapping"],
                "measured_qubits": arm["measured_qubits"],
                "word_universe": arm["word_universe"],
                "basis_size": arm["basis_size"],
                "retained_rank": arm["retained_rank"],
                "exact_subspace_bias_millihartree": arm[
                    "exact_subspace_bias_millihartree"
                ],
                "rungs": rungs,
            }
        )
    return {
        "system": system["system"],
        "exact_ground_energy": system["exact_ground_energy"],
        "max_exact_subspace_bias_millihartree": system[
            "max_exact_subspace_bias_millihartree"
        ],
        "arms": arms,
    }


def _without_cost_derivatives(value):
    """Remove pricing/verdict fields inherited from reusable structural producers."""
    if isinstance(value, dict):
        return {
            key: _without_cost_derivatives(item)
            for key, item in value.items()
            if key not in COST_DERIVATIVE_KEYS and not key.startswith("C_time")
        }
    if isinstance(value, list):
        return [_without_cost_derivatives(item) for item in value]
    return value


def resolution_decision(probe: dict, config: dict) -> dict:
    ceiling = int(
        config["acceptance_gates"]["all_probe_cells_must_resolve_strictly_before"]
    )
    headroom = int(config["acceptance_gates"]["preferred_headroom_endpoint"])
    target = float(config["acceptance_gates"]["accuracy_target_millihartree"])
    rows = []
    for arm in probe["arms"]:
        for rung in arm["rungs"]:
            for estimator in ESTIMATORS:
                search = rung["estimators"][estimator]["shot_to_target"]
                rows.append(
                    {
                        "mapping": arm["mapping"],
                        "block_size": rung["block_size"],
                        "estimator": estimator,
                        "status": search.get("status"),
                        "confirmed_passing_effective_shots_per_setting": search.get(
                            "confirmed_passing_effective_shots_per_setting"
                        ),
                    }
                )
    passing = [
        row["confirmed_passing_effective_shots_per_setting"]
        for row in rows
        if row["confirmed_passing_effective_shots_per_setting"] is not None
    ]
    unresolved = [
        row for row in rows
        if row["confirmed_passing_effective_shots_per_setting"] is None
    ]
    at_or_beyond_ceiling = [
        row
        for row in rows
        if row["confirmed_passing_effective_shots_per_setting"] is not None
        and row["confirmed_passing_effective_shots_per_setting"] >= ceiling
    ]
    bias_passes = all(
        float(arm["exact_subspace_bias_millihartree"]) < target
        for arm in probe["arms"]
    )
    # The screen the probe's own outcome argues for. W sets the variance of the
    # reconstructed pencil, so it -- not the bias floor -- is what decides
    # whether a crossing lands inside the frozen grid: this candidate's bias is
    # seventeen times better than the one bank that prices, and it still failed
    # on resolution at four times that bank's W. Reading it costs seconds, and
    # a candidate that fails it should never reach a forty-cell probe.
    ceiling = int(config["acceptance_gates"]["word_universe_ceiling"])
    widest = max(int(arm["word_universe"]) for arm in probe["arms"])
    word_universe_passes = widest <= ceiling
    resolved_before_ceiling = not unresolved and not at_or_beyond_ceiling
    maximum = max(passing, default=None)
    preferred_headroom = (
        maximum is not None and not unresolved and maximum <= headroom
    )
    eligible = bias_passes and resolved_before_ceiling and word_universe_passes
    if not bias_passes:
        status = "rejected_bias_floor"
    elif unresolved:
        status = "rejected_unresolved_at_frozen_grid"
    elif at_or_beyond_ceiling:
        status = "rejected_no_endpoint_headroom"
    else:
        status = "eligible_for_separately_authorized_full_run"
    return {
        "status": status,
        "eligible_for_full_run": eligible,
        "bias_gate_passes": bias_passes,
        "resolution_gate_passes": resolved_before_ceiling,
        "word_universe_gate_passes": word_universe_passes,
        "maximum_word_universe": widest,
        "word_universe_ceiling": ceiling,
        # Deliberately not folded into ``status``: this probe was run before the
        # screen existed, and the forty cells it drew are the evidence actually
        # collected, so the recorded reason stays the one the evidence supports.
        # For the next candidate the screen fires first and no probe is drawn.
        "screen_would_have_rejected_before_probe": not word_universe_passes,
        "preferred_headroom_gate_passes": preferred_headroom,
        "evaluated_cells": len(rows),
        "resolved_cells": len(rows) - len(unresolved),
        "unresolved_cells": len(unresolved),
        "cells_at_or_beyond_ceiling": len(at_or_beyond_ceiling),
        "maximum_confirmed_passing_effective_shots_per_setting": maximum,
        "unresolved_coordinates": unresolved,
        "ceiling_coordinates": at_or_beyond_ceiling,
        "selection_used_mapping_cost_direction": False,
        "full_run_authorized": False,
        "qr3b_verdict": "not_evaluated_by_preflight",
    }


def build_record(*, workers: int = 1, run_probe: bool = True) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = load_config()
    spec = config["candidate"]
    cards = load_device_cards()
    selection = derive_selection(config)
    mapping_base = load_mapping_config()
    measurement = {
        **mapping_base["measurement"],
        "grouping_protocol_by_system": {spec["key"]: "qwc_groups"},
    }
    mapping = build_mapping_system(
        spec,
        arms=config["mapping_arms"],
        cards=cards,
        measurement=measurement,
    )
    structural = build_protocol_system(
        spec,
        arms=config["mapping_arms"],
        block_sizes=config["protocol"]["block_sizes"],
        cards=cards,
        shots=int(mapping_base["measurement"]["uniform_raw_shots_per_setting"]),
        frozen={},
    )
    base = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_role": "scope_decision_only",
        "config_sha256": _canonical_sha256(config),
        "parent_lineage": config["parent_lineage"],
        "estimand": config["estimand"],
        "claim_boundary": config["claim_boundary"],
        "selection": selection,
        "mapping_preflight": _without_cost_derivatives(mapping),
        "structural_protocol_preflight": _without_cost_derivatives(structural),
        "protocol": config["protocol"],
        "acceptance_gates": config["acceptance_gates"],
    }
    if not run_probe:
        return stamp_record(
            {
                **base,
                "scoping_probe": {
                    "executed": False,
                    "is_a_cost_record": False,
                    "reason": "explicitly skipped by caller",
                },
                "decision": {"status": "probe_not_run", "full_run_authorized": False},
            }
        )

    exact = build_cost_record(
        exploratory_replicas=int(config["protocol"]["exploratory_replicas"]),
        confirmatory_replicas=int(config["protocol"]["confirmatory_replicas"]),
        workers=workers,
        config=_probe_config(config),
        cards=cards,
        system_specs={spec["key"]: spec},
        frozen_settings={},
        seed_roots=config["protocol"]["seed_roots"],
        r1_cross_check_enabled=False,
    )
    probe = _sanitize_probe(exact["systems"][spec["key"]])
    return stamp_record(
        {
            **base,
            "scoping_probe": {
                "executed": True,
                "is_a_cost_record": False,
                "exploratory_replicas": config["protocol"]["exploratory_replicas"],
                "confirmatory_replicas": config["protocol"]["confirmatory_replicas"],
                "search_endpoints_effective_shots_per_setting": list(SEARCH_ENDPOINTS),
                "sampling_evidence": probe,
            },
            "decision": resolution_decision(probe, config),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record(workers=args.workers, run_probe=not args.skip_probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(record["decision"]["status"])
    print(args.out)


if __name__ == "__main__":
    main()
