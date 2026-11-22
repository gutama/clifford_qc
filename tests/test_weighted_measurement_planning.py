"""Tests for variance-first, Ritz-weighted, and noise-gated measurement planning."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.run_sampled_ratio import JointShotSimulator
from clifford_qc.backends import ExactMVBackend
from clifford_qc.ir import PauliSum, PauliWord, Program, Rotor
from clifford_qc.measurement import (
    CompiledMeasurementSampler,
    DeviceCard,
    GroupedWordCache,
    NoiseGatePolicy,
    SettingResources,
    block_commuting_partition,
    compare_block_rungs,
    evaluate_plan_variance,
    plan_measurement,
    ritz_sensitivity_weights,
    sorted_insertion_partition,
    word_weights,
)
from clifford_qc.measurement.cost import _routed_ceil
from clifford_qc.measurement.functionals import WordFunctional, ritz_functional
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.multivector import MV
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.states import bell_density
from clifford_qc.subspace import PrecisionAcquisitionConfig
from clifford_qc.subspace.acquisition import neyman_group_plan
from clifford_qc.subspace.rank_stopping import RankStopPolicy, rank_margin
from clifford_qc.subspace.sampled import SampledProjectedProvider
from clifford_qc.workflows import run_workflow

ROOT = Path(__file__).resolve().parents[1]
H4_FCIDUMP = ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"
BEH2_FCIDUMP = ROOT / "benchmarks/data/beh2_sto3g_r1.3264.FCIDUMP"
SUPERCONDUCTING_CARD = ROOT / "benchmarks/configs/device_cards/superconducting-like.json"
LOGICAL_CARD = ROOT / "benchmarks/configs/device_cards/logical-alltoall.json"


def _forbidden(*args, **kwargs):
    pytest.fail("exact state, density, or materialized row touched in sampled planning")


def test_plan_measurement_defaults_to_graph_state_without_stim():
    script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "stim" or name.startswith("stim."):
        raise AssertionError("Stim imported on default plan_measurement path")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import (
    CompiledMeasurementSampler, GroupedWordCache, plan_measurement,
)
from clifford_qc.states import bell_density
codes = [PauliWord.from_label(s).code for s in ("XX", "YY", "ZZ")]
plan = plan_measurement(2, codes, 2, weights={codes[0]: 2.0, codes[1]: 1.0, codes[2]: 0.5})
assert plan.method == "graph_state"
assert plan.synthesis.gate_counts["CZ"] == 1
assert "stim" not in sys.modules
cache = GroupedWordCache(2)
cache.add_batch(CompiledMeasurementSampler(11).sample_from_state(bell_density(), plan.settings, 100))
assert [cache.candidate_estimate({c: 1.0}) for c in codes] == [1.0, -1.0, 1.0]
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_r_hat_and_exact_state_dependent_r_match_closed_form():
    # Prepare an entangled non-stabilizer 2-qubit state.
    program = Program(2)
    program.clifford("H", 0)
    program.clifford("H", 1)
    program.clifford("CZ", 0, 1)
    program.append(Rotor(PauliWord.from_label("XI"), 0.7))
    program.append(Rotor(PauliWord.from_label("IY"), -0.4))
    rho = ExactMVBackend().state(program)

    xx = PauliWord.from_label("XX").code
    yy = PauliWord.from_label("YY").code
    zz = PauliWord.from_label("ZZ").code
    xi = PauliWord.from_label("XI").code
    coeffs = {xx: 1.5, yy: -0.8, zz: 0.6, xi: 0.9}
    functional = WordFunctional(coeffs, constant=0.25)
    codes = [xx, yy, zz, xi]

    # Put {XX, YY, ZZ} in one FC group and {XI} in a second group.
    plan = plan_measurement(2, codes, 2, groups=((0, 1, 2), (3,)))
    metrics = evaluate_plan_variance(plan, functional, reference=rho, strategy="custom")

    # Closed-form Crawford R_hat:
    expected_ungrouped = (1.5 + 0.8 + 0.6 + 0.9) ** 2
    expected_grouped = (math.hypot(1.5, 0.8, 0.6) + 0.9) ** 2
    assert metrics.ungrouped_variance == pytest.approx(expected_ungrouped, rel=1e-12)
    assert metrics.grouped_variance == pytest.approx(expected_grouped, rel=1e-12)
    assert metrics.r_hat == pytest.approx(expected_ungrouped / expected_grouped, rel=1e-12)

    # Closed-form exact state-dependent covariance on {XX, YY, ZZ} + {XI}:
    scale = float(2 ** 2)
    op_g0 = MV(2, {xx: 1.5, yy: -0.8, zz: 0.6})
    op_g1 = MV(2, {xi: 0.9})
    v0 = float(
        scale * (op_g0 * op_g0).trace_pairing(rho).real
        - (scale * op_g0.trace_pairing(rho).real) ** 2
    )
    v1 = float(
        scale * (op_g1 * op_g1).trace_pairing(rho).real
        - (scale * op_g1.trace_pairing(rho).real) ** 2
    )
    expected_sd_grouped = (math.sqrt(v0) + math.sqrt(v1)) ** 2
    expected_sd_ungrouped = sum(
        abs(c)
        * math.sqrt(
            max(0.0, 1.0 - float(scale * MV(2, {code: 1.0}).trace_pairing(rho).real) ** 2)
        )
        for code, c in coeffs.items()
    ) ** 2
    assert metrics.state_dependent_source == "exact_reference_covariance"
    assert metrics.state_dependent_variance == pytest.approx(expected_sd_grouped, rel=1e-12)
    assert metrics.state_dependent_ungrouped_variance == pytest.approx(
        expected_sd_ungrouped, rel=1e-12
    )
    assert metrics.state_dependent_r == pytest.approx(
        expected_sd_ungrouped / expected_sd_grouped, rel=1e-12
    )

    # Ungrouped singleton plan must give R_hat == 1 and R == 1 identically.
    singleton = plan_measurement(2, codes, 2, groups=((0,), (1,), (2,), (3,)))
    single_exact = evaluate_plan_variance(singleton, functional, reference=rho)
    assert single_exact.r_hat == pytest.approx(1.0, rel=1e-12)
    assert single_exact.state_dependent_r == pytest.approx(1.0, rel=1e-12)

    # Cached assigned-pooling singleton plan also gives R == 1 identically.
    cache = GroupedWordCache(2, pooling="assigned")
    cache.add_batch(CompiledMeasurementSampler(7).sample_from_state(rho, singleton.settings, 500))
    single_cached = evaluate_plan_variance(singleton, functional, cache=cache)
    assert single_cached.state_dependent_source == "grouped_cache_assigned"
    assert single_cached.state_dependent_r == pytest.approx(1.0, rel=1e-12)


def test_cached_optimal_variance_matches_monte_carlo_neyman_variance():
    program = Program(3)
    program.clifford("H", 0)
    program.clifford("CZ", 0, 1)
    program.append(Rotor(PauliWord.from_label("IXY"), 0.55))
    program.append(Rotor(PauliWord.from_label("ZIX"), -0.35))
    rho = ExactMVBackend().state(program)

    terms = {"XXI": 1.2, "YYI": -0.7, "ZZI": 0.4, "IIZ": 0.8, "IXZ": -0.5, "ZIX": 0.6}
    op = PauliSum.from_mv(MV.from_terms(3, terms))
    functional = WordFunctional({code: float(c.real) for code, c in op.terms.items()})
    codes = sorted(functional.coefficients)

    plan = plan_measurement(3, codes, 2, weights=functional)
    exact_metrics = evaluate_plan_variance(plan, functional, reference=rho)

    # Pilot cache estimates per-setting covariance and optimal Neyman variance.
    pilot = GroupedWordCache(3, pooling="assigned")
    pilot.add_batch(CompiledMeasurementSampler(101).sample_from_state(rho, plan.settings, 2000))
    cached_metrics = evaluate_plan_variance(plan, functional, cache=pilot)
    assert cached_metrics.state_dependent_variance == pytest.approx(
        exact_metrics.state_dependent_variance, rel=0.15
    )

    # Allocate a fresh Neyman budget using compiled setting keys and verify MC variance.
    groups = [tuple(PauliWord(3, c) for c in s.assigned_word_codes) for s in plan.settings]
    keys = tuple(s.key for s in plan.settings)
    budget = 4000
    shot_plan = neyman_group_plan(groups, pilot, functional, budget, setting_keys=keys)
    counts = {s.key: shot_plan.get(s.assigned_word_codes[0], 0) for s in plan.settings}
    assert sum(counts.values()) == budget

    estimates = []
    for seed in range(250):
        sampler = CompiledMeasurementSampler(500 + seed)
        cache = GroupedWordCache(3, pooling="assigned")
        cache.add_batch(sampler.sample_from_state(rho, plan.settings, counts))
        estimates.append(functional.estimate(cache))

    empirical_var = float(np.var(estimates, ddof=1))
    predicted_var = exact_metrics.state_dependent_variance / budget
    assert empirical_var == pytest.approx(predicted_var, rel=0.25)

    # Shots pooling (post-hoc repacking) reads extra compatible words without
    # inflating state-dependent variance on a multi-reader plan.
    pooled = GroupedWordCache(3, pooling="shots")
    pooled.add_batch(CompiledMeasurementSampler(101).sample_from_state(rho, plan.settings, 2000))
    pooled_metrics = evaluate_plan_variance(plan, functional, cache=pooled)
    assert pooled_metrics.state_dependent_source == "grouped_cache_shots"
    assert pooled_metrics.state_dependent_variance is not None
    assert pooled_metrics.state_dependent_variance > 0.0


def test_sorted_insertion_beats_largest_degree_on_r_hat_and_matches_when_unweighted():
    # Construct a 2-qubit QWC conflict graph where two heavy words (XX, XI)
    # commute with each other, while Z-words have higher conflict degree with
    # X/Y words so largest-degree splits or delays them differently.
    labels = ("XX", "XI", "IX", "YY", "YI", "IY", "ZZ", "ZI", "IZ", "XZ", "ZX", "YZ", "ZY")
    codes = [PauliWord.from_label(lbl).code for lbl in labels]

    # Unweighted sorted_insertion matches block_commuting_partition identically.
    for k in (1, 2):
        assert sorted_insertion_partition(2, codes, k, weights=None) == (
            block_commuting_partition(2, codes, k)
        )

    # Put large weights on XX and XI (which QWC-commute) and tiny weights on others.
    # Make XZ conflict with XX and XI, and put high degree on XZ/ZX/YZ/ZY.
    heavy_weights = {
        PauliWord.from_label("XX").code: 10.0,
        PauliWord.from_label("XI").code: 9.5,
        PauliWord.from_label("XZ").code: 0.1,
        PauliWord.from_label("ZX").code: 0.1,
        PauliWord.from_label("YZ").code: 0.1,
        PauliWord.from_label("ZY").code: 0.1,
        PauliWord.from_label("ZZ").code: 0.1,
        PauliWord.from_label("YY").code: 0.1,
    }
    si_plan = plan_measurement(2, codes, 1, weights=heavy_weights, strategy="sorted_insertion")
    ldf_plan = plan_measurement(2, codes, 1, strategy="largest_degree")
    si_metrics = evaluate_plan_variance(si_plan, heavy_weights, strategy="sorted_insertion")
    ldf_metrics = evaluate_plan_variance(ldf_plan, heavy_weights, strategy="largest_degree")
    assert si_metrics.r_hat >= ldf_metrics.r_hat
    assert si_metrics.grouped_variance <= ldf_metrics.grouped_variance

    # Real 8-qubit H4 molecular Hamiltonian: sorted insertion improves R_hat
    # over unweighted largest-conflict-degree colouring.
    model = fcidump_model(H4_FCIDUMP)
    ham = model.hamiltonian.to_mv()
    ham_codes = sorted(c for c in ham.terms if c != 0)
    for k in (1, 8):
        si = plan_measurement(model.n, ham_codes, k, weights=ham, strategy="sorted_insertion")
        ldf = plan_measurement(model.n, ham_codes, k, strategy="largest_degree")
        si_m = evaluate_plan_variance(si, ham, strategy="sorted_insertion")
        ldf_m = evaluate_plan_variance(ldf, ham, strategy="largest_degree")
        assert si_m.r_hat > ldf_m.r_hat
        assert si_m.grouped_variance < ldf_m.grouped_variance


def test_noise_gate_policy_exact_boundaries_and_rational_routing():
    # Exact boundary (1 - p)^k == 1 - eps must admit k and reject k + 1.
    p = 0.01
    for k in (1, 5, 10):
        eps = 1.0 - (1.0 - p) ** k
        gate = NoiseGatePolicy(two_qubit_error=p, max_relative_bias=eps)
        assert gate.max_routed_two_qubit_gates == k
        assert gate.admits(SettingResources(n_1q=4, n_2q=k, d_1q=2, d_2q=k), 8)
        assert not gate.admits(SettingResources(n_1q=4, n_2q=k + 1, d_1q=2, d_2q=k + 1), 8)

    # Exact rational routing multiplier: ceil(g * 2.5) <= max_routed.
    routed_gate = NoiseGatePolicy(
        two_qubit_error=0.005, max_relative_bias=0.05, routing_2q_multiplier=2.5
    )
    cap = routed_gate.max_routed_two_qubit_gates
    assert cap == 10  # 0.995**10 = 0.95111 >= 0.95 > 0.995**11 = 0.94635
    logical_cap = routed_gate.max_logical_two_qubit_gates
    assert logical_cap == 4
    assert _routed_ceil(logical_cap, 2.5) <= cap
    assert _routed_ceil(logical_cap + 1, 2.5) > cap
    assert routed_gate.admits(SettingResources(2, logical_cap, 1, logical_cap), 8)
    assert not routed_gate.admits(SettingResources(2, logical_cap + 1, 1, logical_cap + 1), 8)

    # Ideal device (p == 0) imposes no two-qubit gate cap.
    logical = DeviceCard.load(LOGICAL_CARD)
    ideal_gate = NoiseGatePolicy.from_device_card(logical, max_relative_bias=0.01)
    assert ideal_gate.max_routed_two_qubit_gates is None
    assert ideal_gate.max_logical_two_qubit_gates is None
    assert ideal_gate.admits(SettingResources(10, 50, 5, 25), 8)

    # Strict budget max_routed == 0 forces QWC (n_2q == 0) even at block_size == n.
    qwc_only = NoiseGatePolicy(two_qubit_error=0.05, max_relative_bias=0.01)
    assert qwc_only.max_routed_two_qubit_gates == 0
    bell_codes = [PauliWord.from_label(s).code for s in ("XX", "YY", "ZZ")]
    plan = plan_measurement(2, bell_codes, 2, noise_gate=qwc_only)
    assert len(plan.settings) == 3
    assert all(res.n_2q == 0 for res in plan.resources)


def test_noise_gated_hybrid_planning_on_molecular_hamiltonian():
    model = fcidump_model(BEH2_FCIDUMP)
    ham = model.hamiltonian.to_mv()
    codes = sorted(c for c in ham.terms if c != 0)
    card = DeviceCard.load(SUPERCONDUCTING_CARD)
    # At eps=0.10 on superconducting-like (eps_2q=0.01, routing=3.0),
    # max_routed=10 -> max_logical=3 CZs, giving a true hybrid QWC/FC regime.
    gate = NoiseGatePolicy.from_device_card(card, max_relative_bias=0.10)

    qwc_plan = plan_measurement(model.n, codes, 1, weights=ham)
    fc_plan = plan_measurement(model.n, codes, model.n, weights=ham)
    gated_plan = plan_measurement(model.n, codes, model.n, weights=ham, noise_gate=gate)

    qwc_m = evaluate_plan_variance(qwc_plan, ham, strategy="sorted_insertion", noise_gate=gate)
    fc_m = evaluate_plan_variance(fc_plan, ham, strategy="sorted_insertion", noise_gate=gate)
    gated_m = evaluate_plan_variance(
        gated_plan, ham, strategy="sorted_insertion", noise_gate=gate
    )

    # Unconstrained FC violates the superconducting gate budget on BeH2.
    assert fc_m.noise_gate_admissible_settings is not None
    assert fc_m.noise_gate_admissible_settings < fc_m.n_settings
    # Gated plan guarantees 100% admissibility while improving R_hat over QWC.
    assert gated_m.noise_gate_admissible_settings == gated_m.n_settings
    assert gate.max_logical_two_qubit_gates is not None
    assert gated_m.two_qubit_gates_max <= gate.max_logical_two_qubit_gates
    assert gated_m.r_hat > qwc_m.r_hat
    assert qwc_m.n_settings > gated_m.n_settings >= fc_m.n_settings

    # Tighter eps=0.05 (max_logical=1 CZ) honestly falls back to QWC on BeH2.
    tight_gate = NoiseGatePolicy.from_device_card(card, max_relative_bias=0.05)
    tight_plan = plan_measurement(model.n, codes, model.n, weights=ham, noise_gate=tight_gate)
    tight_m = evaluate_plan_variance(
        tight_plan, ham, strategy="sorted_insertion", noise_gate=tight_gate
    )
    assert tight_m.noise_gate_admissible_settings == tight_m.n_settings
    assert tight_m.two_qubit_gates_max <= (tight_gate.max_logical_two_qubit_gates or 0)

    # Fidelity-floor enforcement also keeps every merged setting above card.fidelity_floor.
    fidelity_gate = NoiseGatePolicy.from_device_card(
        card, max_relative_bias=0.20, enforce_fidelity_floor=True
    )
    fid_plan = plan_measurement(model.n, codes, model.n, weights=ham, noise_gate=fidelity_gate)
    fid_m = evaluate_plan_variance(
        fid_plan, ham, strategy="sorted_insertion", noise_gate=fidelity_gate
    )
    assert fid_m.noise_gate_admissible_settings == fid_m.n_settings


def test_ritz_weights_and_sampled_provider_integration_are_density_free(monkeypatch):
    h = MV.from_terms(2, {"ZI": 0.5, "XI": 0.3, "IZ": 0.1, "XX": 0.2, "YY": -0.1})
    ident = MV.scalar(2, 1)
    generators = [
        ("I", ident),
        ("mixed", ident + 0.3 * MV.from_terms(2, {"XI": 1.0}) + 0.2 * MV.from_terms(2, {"IZ": 1.0})),
    ]
    state = np.array([1.0, 0.0, 0.0, 0.0], complex)
    sampler = JointShotSimulator(state, 2)
    reference = PreparedSampledReference(2, "weighted-sampled-fixture", sampler.draw)

    # Default provider preserves legacy largest_degree plan and circuit identity.
    baseline = SampledProjectedProvider(reference, h, generators, block_size=2)
    explicit_legacy = SampledProjectedProvider(
        reference, h, generators, block_size=2, grouping="largest_degree"
    )
    assert baseline.circuit_identity == explicit_legacy.circuit_identity

    # Block every exact/density/materialized-row call on the bank and session.
    monkeypatch.setattr(baseline.bank, "matrices", _forbidden)
    monkeypatch.setattr(baseline.bank, "element_operator", _forbidden)
    monkeypatch.setattr(baseline.bank, "overlap_operator", _forbidden)
    monkeypatch.setattr(baseline.session, "exact_matrices", _forbidden)
    monkeypatch.setattr("clifford_qc.states.ket_density", _forbidden)
    for store in baseline.bank._packed.values():
        monkeypatch.setattr(store, "materialize", _forbidden)

    # State-free weighted provider with GALIC noise gate.
    gate = NoiseGatePolicy(two_qubit_error=0.01, max_relative_bias=0.05)
    weighted = SampledProjectedProvider(
        reference, h, generators, block_size=2, grouping="sorted_insertion", noise_gate=gate
    )
    metrics_before = weighted.plan_metrics()
    assert metrics_before.strategy == "sorted_insertion"
    assert metrics_before.noise_gate_admissible_settings == metrics_before.n_settings

    # Run density-free precision acquisition to get a pilot Ritz vector and overlap functional.
    cfg = PrecisionAcquisitionConfig(
        pilot_shots_per_group=400, round_budgets=(1200, 2400), target_std_error=5.0
    )
    pilot_run = baseline.acquire_precision(cfg, seed=23, solve_options={"tau_s": 0.001})
    assert pilot_run.status == "target_met", pilot_run.error
    assert pilot_run.acquisition is not None
    point = pilot_run.acquisition.result
    _, rank_func = rank_margin(
        baseline.session, pilot_run.acquisition.cache, point, RankStopPolicy(), family=3
    )

    # Replan for first-order Ritz + overlap sensitivity without touching exact states.
    weights = ritz_sensitivity_weights(
        baseline.bank,
        baseline.session.indices,
        point.ritz_vector(0),
        point.energies[0],
        overlap_functional=rank_func,
        overlap_weight=0.25,
    )
    assert weights
    replanned = baseline.with_ritz_plan(
        point.ritz_vector(0),
        point.energies[0],
        overlap_functional=rank_func,
        overlap_weight=0.25,
        noise_gate=gate,
    )
    assert replanned is not baseline
    assert replanned.grouping == "sorted_insertion"
    assert replanned.noise_gate == gate

    # Run unified sampled_precision workflow on the Ritz-weighted noise-gated provider.
    wf = run_workflow(
        replanned,
        path="sampled_precision",
        problem="toy-ritz-weighted",
        unit="Ha",
        precision_config=cfg,
        seed=29,
        solve_options={"tau_s": 0.001},
    ).as_dict()
    assert wf["outcome"]["status"] == "target_met"

    # Evaluate cache-based variance metrics on the replanned provider's cache.
    ritz_func = ritz_functional(
        replanned.bank, replanned.session.indices, point.ritz_vector(0), point.energies[0]
    )
    run2 = replanned.acquire_precision(cfg, seed=29, solve_options={"tau_s": 0.001})
    assert run2.acquisition is not None
    post_metrics = replanned.plan_metrics(target=ritz_func, cache=run2.acquisition.cache)
    assert post_metrics.state_dependent_variance is not None
    assert post_metrics.state_dependent_variance > 0.0
    assert post_metrics.noise_gate_admissible_settings == post_metrics.n_settings


def test_compare_block_rungs_reports_variance_and_2q_tradeoff():
    model = fcidump_model(H4_FCIDUMP)
    ham = model.hamiltonian.to_mv()
    codes = sorted(c for c in ham.terms if c != 0)
    rows = compare_block_rungs(model.n, codes, ham)
    assert [r.block_size for r in rows] == [1, 2, 4, 8]
    assert all(r.method == "graph_state" for r in rows)
    assert rows[0].two_qubit_gates_total == 0
    assert rows[-1].two_qubit_gates_total > 0
    assert all(r.r_hat >= 1.0 for r in rows)
    assert all(isinstance(r.as_dict(), dict) for r in rows)


def test_invalid_inputs_rejected_cleanly():
    with pytest.raises(ValueError):
        NoiseGatePolicy(two_qubit_error=-0.01)
    with pytest.raises(ValueError):
        NoiseGatePolicy(two_qubit_error=1.0)
    with pytest.raises(ValueError):
        NoiseGatePolicy(two_qubit_error=0.01, max_relative_bias=0.0)
    with pytest.raises(ValueError):
        NoiseGatePolicy(two_qubit_error=0.01, max_relative_bias=1.0)
    with pytest.raises(ValueError):
        NoiseGatePolicy(two_qubit_error=0.01, routing_2q_multiplier=0.5)
    with pytest.raises(ValueError):
        NoiseGatePolicy(two_qubit_error=0.01, enforce_fidelity_floor=True)
    with pytest.raises(ValueError):
        word_weights({1: 1.0}, aggregation="bogus")
    with pytest.raises(ValueError):
        word_weights({1: float("nan")})
    with pytest.raises(ValueError):
        word_weights({1: True})

    codes = [PauliWord.from_label("XX").code, PauliWord.from_label("ZZ").code]
    with pytest.raises(ValueError):
        plan_measurement(2, codes, 2, strategy="bogus")
    with pytest.raises(ValueError):
        plan_measurement(2, codes, 2, groups=((0, 1),), weights={codes[0]: 1.0})
    with pytest.raises(ValueError):
        plan_measurement(
            2,
            codes,
            2,
            method="elimination",
            noise_gate=NoiseGatePolicy(two_qubit_error=0.01),
        )

    plan = plan_measurement(2, codes, 2)
    with pytest.raises(ValueError):
        evaluate_plan_variance(plan, {PauliWord.from_label("YY").code: 1.0})
    with pytest.raises(ValueError):
        evaluate_plan_variance(
            plan, {codes[0]: 1.0}, cache=GroupedWordCache(2), reference=bell_density()
        )

    h = MV.from_terms(2, {"ZI": 0.5, "XI": 0.3})
    ref = PreparedSampledReference(2, "bad-opts", _forbidden)
    with pytest.raises(ValueError, match="sorted_insertion"):
        SampledProjectedProvider(
            ref, h, [("I", MV.scalar(2, 1))], grouping="largest_degree", grouping_weights=h
        )
    with pytest.raises(ValueError):
        SampledProjectedProvider(ref, h, [("I", MV.scalar(2, 1))], grouping="bogus")
