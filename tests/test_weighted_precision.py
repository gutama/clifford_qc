"""Regression tests for the weighted-precision preregistration harness."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks import check_weighted_precision as checker
from benchmarks import run_weighted_precision as producer
from benchmarks.noisy_readout_sim import NoisyReadoutSimulator, ReadoutNoiseModel
from clifford_qc.multivector import MV
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace.sampled import SampledProjectedProvider


def test_declaration_structure_and_seed_uniqueness():
    d = producer.declaration()
    stats = checker.validate_declaration_structure(d)
    assert stats["cases"] == 4
    assert stats["part_a_jobs"] == 832
    assert stats["part_b_jobs"] == 576
    assert stats["total_jobs"] == 1408
    assert stats["unique_seeds"] == 2928


def test_declaration_only_anchors():
    pytest.importorskip("scipy")
    d = producer.declaration()
    systems = checker.anchors(d, full=False)
    assert [s["name"] for s in systems] == [
        "beh2_8q",
        "lih_8q",
        "h4_8q",
        "lih_10q_stretched",
    ]
    assert [s["rank"] for s in systems] == [3, 3, 3, 4]


def test_make_schedule_preserves_matched_total_shots():
    totals = [40000, 100000, 250000, 625000]
    for n_settings in (12, 44, 297, 794):
        pilot_pg, rounds = producer.make_schedule(n_settings, totals)
        assert pilot_pg >= 2
        assert len(rounds) == len(totals) - 1
        cumulative = [pilot_pg * n_settings]
        for b in rounds:
            assert b >= n_settings
            cumulative.append(cumulative[-1] + b)
        # Every look after pilot lands on exact declared total (when totals >> n_settings).
        assert cumulative[1:] == totals[1:]


def test_mini_part_a_single_stage_and_paid_pilot_replan_pass_checker():
    h = MV.from_terms(2, {"ZI": 0.5, "XI": 0.3, "IZ": 0.1, "XX": 0.2, "YY": -0.1})
    ident = MV.scalar(2, 1)
    gens = [
        ("I", ident),
        ("G1", ident + 0.35 * MV.from_terms(2, {"XI": 1.0}) + 0.2 * MV.from_terms(2, {"IZ": 1.0})),
    ]
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    sim = NoisyReadoutSimulator(state, 2, ReadoutNoiseModel.ideal())
    ref = PreparedSampledReference(2, "toy-2q-ideal", sim.draw)
    prov = SampledProjectedProvider(ref, h, gens, block_size=2, grouping="sorted_insertion")

    d = {
        "seed_root": 999111,
        "rank_policy": {"consecutive_looks": 2, "margin_delta": 0.05},
        "target_std_error_Ha": 0.05,
        "bootstrap_half_width_Ha": 0.05,
        "delta": 0.05,
        "device_scenario_us": {
            "preparation_per_qubit": 1.0,
            "one_qubit_layer": 0.05,
            "two_qubit_layer": 0.3,
            "readout_reset": 2.0,
        },
        "part_a_arms": [
            {"key": "si_fc", "block_size": "full", "grouping": "sorted_insertion", "width_guard": False},
            {
                "key": "ritz_si_fc",
                "block_size": "full",
                "grouping": "ritz_replan",
                "pilot_arm": "si_fc",
                "overlap_weight": 0.5,
                "width_guard": False,
            },
        ],
    }
    case = {
        "name": "toy_2q",
        "n": 2,
        "solver": {"tau_s": 0.005, "max_condition": 1e4},
        "bootstrap_replicas": 16,
        "cumulative_total_shots": [400, 1000, 2500, 6250],
        "replan_pilot_total_shots": 120,
    }
    system = {"rank": 2}

    # Single-stage arm
    acq_seed = producer.derived_seed(d["seed_root"], 0, 0, 0, 0, 0, 0)
    boot_seed = producer.derived_seed(d["seed_root"], 0, 0, 0, 0, 0, 1)
    row_single = producer.run_single_attempt(
        prov, case, d, acq_seed=acq_seed, boot_seed=boot_seed
    )
    row_single.update(
        part="A", case=0, arm_index=0, regime_index=0, replica=0,
        acq_seed=acq_seed, boot_seed=boot_seed, pilot_seed=None,
    )
    checker.check_row(row_single, d, case, system)
    assert row_single["pilot_shots"] == 0
    assert row_single["total_shots"] == row_single["acquisition_shots"]
    assert row_single["r_hat_ritz"] is not None and row_single["r_emp_ritz"] is not None

    # Paid-pilot Ritz-replanned arm
    acq_s2 = producer.derived_seed(d["seed_root"], 0, 0, 1, 0, 0, 0)
    boot_s2 = producer.derived_seed(d["seed_root"], 0, 0, 1, 0, 0, 1)
    pilot_s2 = producer.derived_seed(d["seed_root"], 0, 0, 1, 0, 0, 2)
    row_replan = producer.run_single_attempt(
        prov,
        case,
        d,
        acq_seed=acq_s2,
        boot_seed=boot_s2,
        pilot_seed=pilot_s2,
        replan_spec=d["part_a_arms"][1],
    )
    row_replan.update(
        part="A", case=0, arm_index=1, regime_index=0, replica=0,
        acq_seed=acq_s2, boot_seed=boot_s2, pilot_seed=pilot_s2,
    )
    checker.check_row(row_replan, d, case, system)
    assert row_replan["pilot_shots"] > 0
    assert row_replan["total_shots"] == row_replan["pilot_shots"] + row_replan["acquisition_shots"]
    assert row_replan["pilot_fingerprint"] is not None


def test_mini_part_b_noisy_infinite_shot_bias_and_checker():
    h = MV.from_terms(2, {"XX": 0.6, "YY": -0.4, "ZZ": 0.5})
    ident = MV.scalar(2, 1)
    gens = [("I", ident), ("Z0", MV.from_terms(2, {"ZI": 1.0}))]
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)

    ideal_sim = NoisyReadoutSimulator(state, 2, ReadoutNoiseModel.ideal())
    noisy_sim = NoisyReadoutSimulator(state, 2, ReadoutNoiseModel(p_2q=0.05, routing_2q_multiplier=2.0))
    ref_ideal = PreparedSampledReference(2, "toy-b-ideal", ideal_sim.draw)
    ref_noisy = PreparedSampledReference(2, "toy-b-noisy", noisy_sim.draw)

    prov_qwc = SampledProjectedProvider(ref_noisy, h, gens, block_size=1, grouping="sorted_insertion")
    prov_fc = SampledProjectedProvider(ref_noisy, h, gens, block_size=2, grouping="sorted_insertion")
    prov_ideal = SampledProjectedProvider(ref_ideal, h, gens, block_size=2, grouping="sorted_insertion")

    e_exact, _, _ = producer.exact_projected_from_expectations(
        prov_ideal, ideal_sim.exact_word_expectations(prov_ideal.plan.settings), {"tau_s": 0.005, "max_condition": 1e4}
    )
    e_qwc_noisy, _, _ = producer.exact_projected_from_expectations(
        prov_qwc, noisy_sim.exact_word_expectations(prov_qwc.plan.settings), {"tau_s": 0.005, "max_condition": 1e4}
    )
    e_fc_noisy, _, _ = producer.exact_projected_from_expectations(
        prov_fc, noisy_sim.exact_word_expectations(prov_fc.plan.settings), {"tau_s": 0.005, "max_condition": 1e4}
    )
    # QWC has 0 CZ -> zero 2Q-depolarizing bias; FC uses CZ -> nonzero positive bias toward 0.
    assert abs(e_qwc_noisy - e_exact) < 1e-12
    assert abs(e_fc_noisy - e_exact) > 0.05
