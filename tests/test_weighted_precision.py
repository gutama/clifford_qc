"""Regression tests for the weighted-precision preregistration harness."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from benchmarks import check_weighted_precision as checker
from benchmarks import run_weighted_precision as producer
from benchmarks.noisy_readout_sim import NoisyReadoutSimulator, ReadoutNoiseModel
from clifford_qc.measurement import DeviceCard, NoiseGatePolicy, evaluate_plan_variance
from clifford_qc.multivector import MV
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace.sampled import SampledProjectedProvider


def _toy_2q_system():
    h = MV.from_terms(2, {"ZI": 0.5, "XI": 0.3, "IZ": 0.1, "XX": 0.2, "YY": -0.1})
    ident = MV.scalar(2, 1)
    gens = [
        ("I", ident),
        ("G1", ident + 0.35 * MV.from_terms(2, {"XI": 1.0}) + 0.2 * MV.from_terms(2, {"IZ": 1.0})),
    ]
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    return h, gens, state


def _toy_declaration():
    return {
        "seed_root": 999111,
        "device_card": "benchmarks/configs/device_cards/superconducting-like.json",
        "rank_policy": {"consecutive_looks": 2, "margin_delta": 0.05},
        "target_std_error_Ha": 0.05,
        "bootstrap_half_width_Ha": 0.05,
        "delta": 0.05,
        "analysis": {
            "coverage_levels": [0.68, 0.95],
            "shot_ratio_bootstrap_replicates": 400,
            "shot_ratio_bootstrap_seed": 999222,
            "shot_ratio_alpha": 0.05,
            "shot_ratio_holm_bonferroni": True,
        },
        "device_scenario_us": {
            "preparation_per_qubit": 1.0,
            "one_qubit_layer": 0.05,
            "two_qubit_layer": 0.3,
            "readout_reset": 2.0,
        },
        "cases": [
            {
                "name": "toy_2q",
                "kind": "molecule_8q",
                "parts": ["A", "B"],
                "n": 2,
                "solver": {"tau_s": 0.005, "max_condition": 1e4},
                "outer_replicas": 2,
                "bootstrap_replicas": 16,
                "cumulative_total_shots": [400, 1000, 2500, 6250],
                "replan_pilot_total_shots": 120,
            }
        ],
        "part_a_arms": [
            {"key": "si_fc", "block_size": "full", "grouping": "sorted_insertion", "width_guard": False, "cases": ["toy_2q"]},
            {
                "key": "ritz_si_fc",
                "block_size": "full",
                "grouping": "ritz_replan",
                "pilot_arm": "si_fc",
                "overlap_weight": 0.5,
                "width_guard": False,
                "cases": ["toy_2q"],
            },
            {"key": "si_fc_width", "block_size": "full", "grouping": "sorted_insertion", "width_guard": True, "cases": ["toy_2q"]},
            {"key": "ldf_fc", "block_size": "full", "grouping": "largest_degree", "width_guard": False, "cases": ["toy_2q"]},
        ],
        "part_b_noise_regimes": [
            {
                "key": "depol2q_p010_r3",
                "p_1q": 0.0,
                "p_2q": 0.01,
                "routing_2q_multiplier": 3.0,
                "p_readout": 0.0,
                "galic_tight_eps": 0.05,
                "galic_moderate_eps": 0.10,
            },
            {
                "key": "sc_card_full",
                "p_1q": 0.001,
                "p_2q": 0.01,
                "routing_2q_multiplier": 3.0,
                "p_readout": 0.02,
                "galic_tight_eps": 0.05,
                "galic_moderate_eps": 0.10,
            },
        ],
        "part_b_arms": [
            {"key": "si_qwc", "block_size": 1, "grouping": "sorted_insertion", "gate_tier": "none"},
            {"key": "si_fc_unconstrained", "block_size": "full", "grouping": "sorted_insertion", "gate_tier": "none"},
            {"key": "si_fc_galic_tight", "block_size": "full", "grouping": "sorted_insertion", "gate_tier": "tight"},
            {"key": "si_fc_galic_moderate", "block_size": "full", "grouping": "sorted_insertion", "gate_tier": "moderate"},
        ],
        "part_b_outer_replicas": 2,
    }


def test_declaration_structure_and_seed_uniqueness():
    d = producer.declaration()
    stats = checker.validate_declaration_structure(d)
    assert stats["cases"] == 4
    assert stats["part_a_jobs"] == 832
    assert stats["part_b_jobs"] == 576
    assert stats["total_jobs"] == 1408
    assert stats["unique_seeds"] == 2928
    assert d["analysis"]["coverage_levels"] == [0.68, 0.95]
    assert d["analysis"]["shot_ratio_bootstrap_replicates"] == 2000


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


def test_manifest_binds_full_tracked_tree_and_clean_worktree_guard(monkeypatch):
    d = producer.declaration()
    head = producer.git("rev-parse", "HEAD").decode("utf-8").strip()
    m = producer.manifest(head, d, live=False)
    # Must bind directly imported and transitive scientific dependencies across both trees.
    for required in (
        "clifford_qc/subspace/linalg.py",
        "clifford_qc/measurement/weighted_planning.py",
        "clifford_qc/subspace/sampled_acquisition.py",
        "benchmarks/run_cross_domain_ratio.py",
        "benchmarks/run_stretched_molecular_stop.py",
        "benchmarks/run_sampled_ratio.py",
        "benchmarks/noisy_readout_sim.py",
        "benchmarks/run_weighted_precision.py",
        "benchmarks/check_weighted_precision.py",
    ):
        assert required in m

    monkeypatch.setattr(producer, "git", lambda *args: b" M clifford_qc/subspace/linalg.py\n")
    with pytest.raises(ValueError, match="clean Git worktree"):
        producer.require_clean_worktree()
    monkeypatch.setattr(producer, "git", lambda *args: b"?? benchmarks/shadow.py\n")
    with pytest.raises(ValueError, match="clean Git worktree"):
        producer.require_clean_worktree()
    monkeypatch.setattr(producer, "git", lambda *args: b"")
    producer.require_clean_worktree()


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
        assert cumulative[1:] == totals[1:]


def test_mini_part_a_single_stage_paid_pilot_and_width_guard_pass_checker():
    h, gens, state = _toy_2q_system()
    d = _toy_declaration()
    case = d["cases"][0]
    sim = NoisyReadoutSimulator(state, 2, ReadoutNoiseModel.ideal())
    ref = PreparedSampledReference(2, "toy-2q-ideal", sim.draw)
    prov = SampledProjectedProvider(ref, h, gens, block_size=2, grouping="sorted_insertion")
    m = prov.plan_metrics()
    plan_info = {"settings": m.n_settings, "cz_total": m.two_qubit_gates_total, "cz_max": m.two_qubit_gates_max}
    system = {"rank": 2, "part_a_plans": {"si_fc": plan_info, "si_fc_width": plan_info}}

    # Single-stage arm
    acq_seed = producer.derived_seed(d["seed_root"], 0, 0, 0, 0, 0, 0)
    boot_seed = producer.derived_seed(d["seed_root"], 0, 0, 0, 0, 0, 1)
    row_single = producer.run_single_attempt(prov, case, d, acq_seed=acq_seed, boot_seed=boot_seed)
    row_single.update(
        part="A", case=0, case_name="toy_2q", arm="si_fc", arm_index=0,
        regime="ideal", regime_index=0, replica=0,
        acq_seed=acq_seed, boot_seed=boot_seed, pilot_seed=None,
    )
    checker.check_row(row_single, d, case, system, expected_plan=plan_info)
    assert row_single["pilot_shots"] == 0
    assert row_single["total_shots"] == row_single["acquisition_shots"]
    assert row_single["r_hat_ritz"] is not None and row_single["r_emp_ritz"] is not None

    # Paid-pilot Ritz-replanned arm
    acq_s2 = producer.derived_seed(d["seed_root"], 0, 0, 1, 0, 0, 0)
    boot_s2 = producer.derived_seed(d["seed_root"], 0, 0, 1, 0, 0, 1)
    pilot_s2 = producer.derived_seed(d["seed_root"], 0, 0, 1, 0, 0, 2)
    row_replan = producer.run_single_attempt(
        prov, case, d, acq_seed=acq_s2, boot_seed=boot_s2, pilot_seed=pilot_s2, replan_spec=d["part_a_arms"][1]
    )
    row_replan.update(
        part="A", case=0, case_name="toy_2q", arm="ritz_si_fc", arm_index=1,
        regime="ideal", regime_index=0, replica=0,
        acq_seed=acq_s2, boot_seed=boot_s2, pilot_seed=pilot_s2,
    )
    checker.check_row(row_replan, d, case, system)
    assert row_replan["pilot_shots"] > 0
    assert row_replan["total_shots"] == row_replan["pilot_shots"] + row_replan["acquisition_shots"]
    assert row_replan["pilot_fingerprint"] is not None

    # Width-guarded arm
    acq_s3 = producer.derived_seed(d["seed_root"], 0, 0, 2, 0, 0, 0)
    boot_s3 = producer.derived_seed(d["seed_root"], 0, 0, 2, 0, 0, 1)
    row_width = producer.run_single_attempt(
        prov, case, d, acq_seed=acq_s3, boot_seed=boot_s3, width_guard=True
    )
    row_width.update(
        part="A", case=0, case_name="toy_2q", arm="si_fc_width", arm_index=2,
        regime="ideal", regime_index=0, replica=0,
        acq_seed=acq_s3, boot_seed=boot_s3, pilot_seed=None,
    )
    checker.check_row(row_width, d, case, system, expected_plan=plan_info)


def test_check_row_rejects_impossible_and_tampered_rows():
    h, gens, state = _toy_2q_system()
    d = _toy_declaration()
    case = d["cases"][0]
    sim = NoisyReadoutSimulator(state, 2, ReadoutNoiseModel.ideal())
    ref = PreparedSampledReference(2, "toy-2q-neg", sim.draw)
    prov = SampledProjectedProvider(ref, h, gens, block_size=2, grouping="sorted_insertion")
    m = prov.plan_metrics()
    plan_info = {"settings": m.n_settings, "cz_total": m.two_qubit_gates_total, "cz_max": m.two_qubit_gates_max}
    system = {"rank": 2, "part_a_plans": {"si_fc": plan_info, "si_fc_width": plan_info}}

    acq_seed = producer.derived_seed(d["seed_root"], 0, 0, 2, 0, 0, 0)
    boot_seed = producer.derived_seed(d["seed_root"], 0, 0, 2, 0, 0, 1)
    valid_width = producer.run_single_attempt(prov, case, d, acq_seed=acq_seed, boot_seed=boot_seed, width_guard=True)
    valid_width.update(
        part="A", case=0, case_name="toy_2q", arm="si_fc_width", arm_index=2,
        regime="ideal", regime_index=0, replica=0,
        acq_seed=acq_seed, boot_seed=boot_seed, pilot_seed=None,
    )
    checker.check_row(valid_width, d, case, system, expected_plan=plan_info)

    # 1. Zero-shot looks + zero acquisition_shots must be rejected.
    bad_zero = copy.deepcopy(valid_width)
    for lk in bad_zero["looks"]:
        lk["shots"] = 0
    bad_zero["acquisition_shots"] = 0
    bad_zero["total_shots"] = 0
    with pytest.raises(ValueError, match="cumulative shots mismatch"):
        checker.check_row(bad_zero, d, case, system, expected_plan=plan_info)

    # 2. target_met with failed terminal width guard must be rejected.
    bad_width = copy.deepcopy(valid_width)
    bad_width["status"] = "target_met"
    bad_width["looks"][-1]["bootstrap_spread"] = 10.0
    bad_width["looks"][-1]["bootstrap_width_passed"] = False
    with pytest.raises(ValueError, match="stop eligibility"):
        checker.check_row(bad_width, d, case, system, expected_plan=plan_info)

    # 3. Lying bootstrap_width_passed=True when z*spread > half_width must be rejected.
    bad_width_lie = copy.deepcopy(valid_width)
    bad_width_lie["looks"][-1]["bootstrap_spread"] = 10.0
    bad_width_lie["looks"][-1]["bootstrap_width_passed"] = True
    with pytest.raises(ValueError, match="bootstrap_width_passed decision mismatch"):
        checker.check_row(bad_width_lie, d, case, system, expected_plan=plan_info)

    # 4. Tampered schedule must be rejected.
    bad_sched = copy.deepcopy(valid_width)
    bad_sched["schedule"]["pilot_shots_per_group"] += 1
    with pytest.raises(ValueError, match="recorded schedule differs"):
        checker.check_row(bad_sched, d, case, system, expected_plan=plan_info)

    # 5. Mismatched arm label vs arm_index must be rejected.
    bad_label = copy.deepcopy(valid_width)
    bad_label["arm"] = "si_fc"
    with pytest.raises(ValueError, match="arm label mismatch"):
        checker.check_row(bad_label, d, case, system, expected_plan=plan_info)

    # 6. Tampered rank_guard margin arithmetic must be rejected.
    bad_guard = copy.deepcopy(valid_width)
    bad_guard["looks"][0]["rank_guard"]["comparisons"][0]["lower_margin"] += 0.1
    with pytest.raises(ValueError, match="lower_margin arithmetic mismatch"):
        checker.check_row(bad_guard, d, case, system, expected_plan=plan_info)

    # 7. Marking an eligible target_met row as schedule_exhausted must be rejected.
    if valid_width["status"] == "target_met":
        bad_exh = copy.deepcopy(valid_width)
        bad_exh["status"] = "schedule_exhausted"
        with pytest.raises(ValueError, match="stop eligibility"):
            checker.check_row(bad_exh, d, case, system, expected_plan=plan_info)


def test_unconditional_coverage_excludes_exhaustions_and_failures():
    exact = -1.0
    family = 4
    delta = 0.05
    z1 = checker.z_score(1.0 - delta, 1)
    zf = checker.z_score(1.0 - delta, family)

    def _row(status, est, scale, failure=None):
        if failure is not None:
            return {
                "status": "failed",
                "failure": failure,
                "estimate": None,
                "naive_interval": None,
                "delta_interval": None,
                "bootstrap_interval": None,
            }
        return {
            "status": status,
            "failure": None,
            "estimate": est,
            "naive_interval": {"lower": est - z1 * scale, "upper": est + z1 * scale, "std_error": scale},
            "delta_interval": {"lower": est - zf * scale, "upper": est + zf * scale, "std_error": scale},
            "bootstrap_interval": {"lower": est - zf * scale, "upper": est + zf * scale, "std_error": scale},
        }

    # 1 eligible target_met covering row, 1 exhausted row whose interval ALSO contains exact, 1 failed row.
    rows = [
        _row("target_met", -1.0, 0.01),
        _row("schedule_exhausted", -1.0, 0.50),
        _row("failed", None, None, failure="LinAlgError: singular"),
    ]
    cov = checker._coverage_block(rows, exact, family, delta, (0.68, 0.95))
    for kind in ("delta", "bootstrap"):
        for lvl in ("68", "95"):
            block = cov[kind][lvl]
            assert block["unconditional_covered"] == 1
            assert block["unconditional_n"] == 3
            assert block["unconditional_coverage"] == pytest.approx(1.0 / 3.0)
            assert block["conditional_covered"] == 1
            assert block["conditional_n"] == 1
            assert block["conditional_coverage"] == pytest.approx(1.0)
            assert block["median_width_mHa_among_stops"] is not None
            assert block["median_width_mHa_all_nonfailed"] > block["median_width_mHa_among_stops"]


def test_verify_noisy_anchors_recomputes_and_rejects_tampering():
    h = MV.from_terms(2, {"XX": 0.6, "YY": -0.4, "ZZ": 0.5})
    ident = MV.scalar(2, 1)
    gens = [("I", ident), ("Z0", MV.from_terms(2, {"ZI": 1.0}))]
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)

    d = _toy_declaration()
    case = d["cases"][0]
    card = DeviceCard.load(d["device_card"])

    ideal_sim = NoisyReadoutSimulator(state, 2, ReadoutNoiseModel.ideal())
    ideal_ref = PreparedSampledReference(2, "toy-b-ideal", ideal_sim.draw)
    prov_ideal = SampledProjectedProvider(ideal_ref, h, gens, block_size=2, grouping="sorted_insertion")
    exact_e, exact_rank, exact_spec = producer.exact_projected_from_expectations(
        prov_ideal, ideal_sim.exact_word_expectations(prov_ideal.plan.settings), case["solver"]
    )
    systems = [{"case": 0, "name": "toy_2q", "independent_target": exact_e, "rank": exact_rank, "overlap_spectrum": exact_spec}]

    # Build genuine noisy anchors for all 2 regimes x 4 arms = 8 entries.
    anchors = []
    for regime in d["part_b_noise_regimes"]:
        nmodel = ReadoutNoiseModel(
            p_1q=regime["p_1q"], p_2q=regime["p_2q"],
            routing_2q_multiplier=regime["routing_2q_multiplier"], p_readout=regime["p_readout"],
            label=regime["key"],
        )
        nsim = NoisyReadoutSimulator(state, 2, nmodel)
        nref = PreparedSampledReference(2, f"toy-{regime['key']}", nsim.draw)
        for arm in d["part_b_arms"]:
            bs = 2 if arm["block_size"] == "full" else int(arm["block_size"])
            gate = None
            if arm["gate_tier"] == "tight":
                gate = NoiseGatePolicy(two_qubit_error=regime["p_2q"], max_relative_bias=regime["galic_tight_eps"], routing_2q_multiplier=regime["routing_2q_multiplier"], device_card=card)
            elif arm["gate_tier"] == "moderate":
                gate = NoiseGatePolicy(two_qubit_error=regime["p_2q"], max_relative_bias=regime["galic_moderate_eps"], routing_2q_multiplier=regime["routing_2q_multiplier"], device_card=card)
            prov = producer.build_arm_provider(nref, h, gens, 2, block_size=bs, grouping=arm["grouping"], noise_gate=gate)
            bm = evaluate_plan_variance(prov.plan, prov.session.matrix_functionals(), strategy=arm["grouping"], noise_gate=gate)
            ne, nrank, nspec = producer.exact_projected_from_expectations(prov, nsim.exact_word_expectations(prov.plan.settings), case["solver"])
            anchors.append({
                "case": 0, "case_name": "toy_2q", "regime": regime["key"], "arm": arm["key"],
                "settings": bm.n_settings, "r_hat": float(bm.r_hat),
                "cz_total": int(bm.two_qubit_gates_total), "cz_max": int(bm.two_qubit_gates_max),
                "admissible_settings": bm.noise_gate_admissible_settings,
                "noisy_exact_energy_Ha": float(ne), "bias_Ha": float(ne - exact_e),
                "noisy_rank": int(nrank), "noisy_overlap_spectrum": nspec,
                "noise_model": nmodel.as_dict(), "noise_gate": None if gate is None else gate.as_dict(),
            })

    verified = checker.verify_noisy_anchors({"noisy_anchors": anchors}, d, systems, case_inputs_fn=lambda _c: (h, gens, state))
    assert len(verified) == 8
    # Unconstrained FC has strictly larger bias than QWC under depol2q_p010_r3.
    assert abs(verified[(0, "depol2q_p010_r3", "si_qwc")]["bias_Ha"]) < 1e-12
    assert abs(verified[(0, "depol2q_p010_r3", "si_fc_unconstrained")]["bias_Ha"]) > 0.01

    # Negative checks: missing, duplicate, tampered bias, tampered cz_max.
    with pytest.raises(ValueError, match="incomplete noisy anchors"):
        checker.verify_noisy_anchors({"noisy_anchors": anchors[:-1]}, d, systems, case_inputs_fn=lambda _c: (h, gens, state))

    with pytest.raises(ValueError, match="duplicate noisy anchor"):
        checker.verify_noisy_anchors({"noisy_anchors": anchors + [anchors[0]]}, d, systems, case_inputs_fn=lambda _c: (h, gens, state))

    bad_bias = copy.deepcopy(anchors)
    bad_bias[1]["bias_Ha"] = 0.0
    with pytest.raises(ValueError, match="bias_Ha mismatch"):
        checker.verify_noisy_anchors({"noisy_anchors": bad_bias}, d, systems, case_inputs_fn=lambda _c: (h, gens, state))

    bad_cz = copy.deepcopy(anchors)
    bad_cz[1]["cz_max"] += 1
    with pytest.raises(ValueError, match="cz_max mismatch"):
        checker.verify_noisy_anchors({"noisy_anchors": bad_cz}, d, systems, case_inputs_fn=lambda _c: (h, gens, state))


def test_summarize_and_evaluate_criteria_deterministic_end_to_end():
    d = _toy_declaration()
    exact = -1.0
    systems = [{"case": 0, "name": "toy_2q", "independent_target": exact, "rank": 2}]
    zf = checker.z_score(0.95, 4)
    z1 = checker.z_score(0.95, 1)

    def _mk_row(part, arm, ai, reg, ri, rep, shots, est, scale, status="target_met"):
        return {
            "part": part, "case": 0, "case_name": "toy_2q", "arm": arm, "arm_index": ai,
            "regime": reg, "regime_index": ri, "replica": rep,
            "status": status, "failure": None, "pilot_shots": 0, "acquisition_shots": shots, "total_shots": shots,
            "device_seconds": shots * 1e-6, "measured_solves": 2, "bootstrap_solves": 16,
            "estimate": est, "std_error": scale,
            "looks": [{"retained_rank": 2}],
            "naive_interval": {"lower": est - z1 * scale, "upper": est + z1 * scale, "std_error": scale},
            "delta_interval": {"lower": est - zf * scale, "upper": est + zf * scale, "std_error": scale},
            "bootstrap_interval": {"lower": est - zf * scale, "upper": est + zf * scale, "std_error": scale},
            "r_hat_ritz": 10.0, "r_emp_ritz": 8.0,
        }

    rows = []
    for rep in range(2):
        rows.append(_mk_row("A", "si_fc", 0, "ideal", 0, rep, 1000, exact + 0.0005, 0.001))
        rows.append(_mk_row("A", "ritz_si_fc", 1, "ideal", 0, rep, 600, exact - 0.0004, 0.001))
        rows.append(_mk_row("A", "si_fc_width", 2, "ideal", 0, rep, 1200, exact + 0.0003, 0.001))
        rows.append(_mk_row("A", "ldf_fc", 3, "ideal", 0, rep, 2500, exact + 0.0008, 0.001))
        # Part B depol2q_p010_r3: unconstrained biased (>0.05), galic_tight accurate
        rows.append(_mk_row("B", "si_qwc", 0, "depol2q_p010_r3", 0, rep, 2000, exact, 0.001))
        rows.append(_mk_row("B", "si_fc_unconstrained", 1, "depol2q_p010_r3", 0, rep, 1000, exact + 0.08, 0.001))
        rows.append(_mk_row("B", "si_fc_galic_tight", 2, "depol2q_p010_r3", 0, rep, 1500, exact + 0.002, 0.001))
        rows.append(_mk_row("B", "si_fc_galic_moderate", 3, "depol2q_p010_r3", 0, rep, 1200, exact + 0.06, 0.001))
        rows.append(_mk_row("B", "si_qwc", 0, "sc_card_full", 1, rep, 2000, exact + 0.10, 0.001))
        rows.append(_mk_row("B", "si_fc_unconstrained", 1, "sc_card_full", 1, rep, 1000, exact + 0.12, 0.001))
        rows.append(_mk_row("B", "si_fc_galic_tight", 2, "sc_card_full", 1, rep, 1500, exact + 0.10, 0.001))
        rows.append(_mk_row("B", "si_fc_galic_moderate", 3, "sc_card_full", 1, rep, 1200, exact + 0.11, 0.001))

    noisy_anchors = [
        {"case": 0, "regime": "depol2q_p010_r3", "arm": "si_qwc", "bias_Ha": 0.0, "cz_max": 0, "settings": 4},
        {"case": 0, "regime": "depol2q_p010_r3", "arm": "si_fc_unconstrained", "bias_Ha": 0.08, "cz_max": 2, "settings": 2},
        {"case": 0, "regime": "depol2q_p010_r3", "arm": "si_fc_galic_tight", "bias_Ha": 0.002, "cz_max": 1, "settings": 3},
        {"case": 0, "regime": "depol2q_p010_r3", "arm": "si_fc_galic_moderate", "bias_Ha": 0.06, "cz_max": 2, "settings": 2},
        {"case": 0, "regime": "sc_card_full", "arm": "si_qwc", "bias_Ha": 0.10, "cz_max": 0, "settings": 4},
        {"case": 0, "regime": "sc_card_full", "arm": "si_fc_unconstrained", "bias_Ha": 0.12, "cz_max": 2, "settings": 2},
        {"case": 0, "regime": "sc_card_full", "arm": "si_fc_galic_tight", "bias_Ha": 0.10, "cz_max": 1, "settings": 3},
        {"case": 0, "regime": "sc_card_full", "arm": "si_fc_galic_moderate", "bias_Ha": 0.11, "cz_max": 2, "settings": 2},
    ]
    rec = {"rows": rows, "noisy_anchors": noisy_anchors}
    s1 = checker.summarize(rec, d, systems)
    s2 = checker.summarize(rec, d, systems)
    # Deterministic frozen bootstrap seed guarantees identical output across calls.
    assert s1 == s2
    assert s1["criteria"]["P1a_weighted_vs_unweighted"]["all_passed"] is True
    assert s1["criteria"]["P3_galic_noise_gate"]["all_passed"] is True
    assert len(s1["criteria"]["P3_galic_noise_gate"]["sc_card_full_diagnostics"]) == 1
    assert len(s1["criteria"]["P1b_ritz_replan_vs_bank_fc"]["cells"]) == 1
    assert len(s1["criteria"]["width_guard_overhead"]) == 1
