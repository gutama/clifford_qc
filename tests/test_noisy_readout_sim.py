"""Tests for the gate-level noisy readout simulator (benchmarks/noisy_readout_sim.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.noisy_readout_sim import NoisyReadoutSimulator, ReadoutNoiseModel
from benchmarks.run_sampled_ratio import JointShotSimulator
from clifford_qc.channels import apply_channel, depolarizing
from clifford_qc.gates import CZ, H, S
from clifford_qc.ir import PauliWord
from clifford_qc.matrix import density_from_statevector
from clifford_qc.measurement import DeviceCard, plan_measurement
from clifford_qc.multivector import MV
from clifford_qc.pauli import I, X, Y, Z
from clifford_qc.sampled_reference import PreparedSampledReference


def _mv_two_qubit_depolarizing(n: int, q0: int, q1: int, p: float) -> list[MV]:
    """Explicit 16-Kraus 2-qubit depolarizing channel in MV algebra for validation."""
    paulis_0 = [I(n), X(n, q0), Y(n, q0), Z(n, q0)]
    paulis_1 = [I(n), X(n, q1), Y(n, q1), Z(n, q1)]
    kraus = [math.sqrt(1.0 - p) * I(n)]
    scale = math.sqrt(p / 15.0)
    for i, p0 in enumerate(paulis_0):
        for j, p1 in enumerate(paulis_1):
            if i == 0 and j == 0:
                continue
            kraus.append(scale * (p0 * p1))
    return kraus


def _mv_gate_unitary(n: int, name: str, qubits: tuple[int, ...]) -> MV:
    if name == "H":
        return H(n, qubits[0])
    if name == "S":
        return S(n, qubits[0])
    if name == "SDG":
        return S(n, qubits[0]).dagger()
    if name == "CZ":
        return CZ(n, qubits[0], qubits[1])
    raise ValueError(name)


def _mv_z_probabilities(rho: MV) -> np.ndarray:
    n = rho.n
    probs = np.zeros(1 << n, dtype=float)
    for idx in range(1 << n):
        proj = I(n)
        for q in range(n):
            bit = (idx >> (n - 1 - q)) & 1
            proj = proj * (0.5 * (I(n) + ((-1.0) ** bit) * Z(n, q)))
        probs[idx] = float((2**n) * rho.trace_pairing(proj).real)
    return probs


def test_noiseless_simulator_matches_joint_shot_simulator_exactly():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=1 << 4) + 1j * rng.normal(size=1 << 4)
    state = raw / np.linalg.norm(raw)
    labels = ["XXXX", "YYYY", "ZZZZ", "XXZZ", "ZZXX", "XYXY", "YXYX", "ZIZI"]
    codes = [PauliWord.from_label(lbl).code for lbl in labels]
    plan = plan_measurement(4, codes, 4)

    ideal_ref = JointShotSimulator(state, 4)
    sim = NoisyReadoutSimulator(state, 4, ReadoutNoiseModel.ideal())

    for setting in plan.settings:
        np.testing.assert_allclose(sim.probabilities(setting), ideal_ref.probabilities(setting), atol=1e-14)

    counts = [250] * len(plan.settings)
    b1 = ideal_ref.draw(plan.settings, counts, seed=19, identity="id")
    b2 = sim.draw(plan.settings, counts, seed=19, identity="id")
    assert b1.shots == b2.shots
    assert b1.plus_counts == b2.plus_counts
    for g1, g2 in zip(b1.groups, b2.groups, strict=True):
        assert g1.hist == g2.hist


def test_gate_level_depolarizing_matches_explicit_mv_kraus_channels():
    rng = np.random.default_rng(13)
    raw = rng.normal(size=1 << 3) + 1j * rng.normal(size=1 << 3)
    state = raw / np.linalg.norm(raw)
    labels = ["XXX", "YYZ", "ZYY", "ZZI", "XZX"]
    codes = [PauliWord.from_label(lbl).code for lbl in labels]
    plan = plan_measurement(3, codes, 3)

    p_1q, p_2q, m_route = 0.015, 0.04, 3
    noise = ReadoutNoiseModel(p_1q=p_1q, p_2q=p_2q, routing_2q_multiplier=float(m_route))
    sim = NoisyReadoutSimulator(state, 3, noise)

    rho0 = density_from_statevector(state)
    for setting in plan.settings:
        rho = rho0
        for name, qubits in setting.clifford.gates:
            u = _mv_gate_unitary(3, name, qubits)
            rho = u * rho * u.dagger()
            if name == "CZ":
                k2 = _mv_two_qubit_depolarizing(3, qubits[0], qubits[1], p_2q)
                for _ in range(m_route):
                    rho = apply_channel(rho, k2, check=False)
            else:
                rho = apply_channel(rho, depolarizing(3, qubits[0], p_1q), check=False)
        expected_probs = _mv_z_probabilities(rho)
        np.testing.assert_allclose(sim.probabilities(setting), expected_probs, atol=1e-12)


def test_symmetric_readout_error_attenuates_weight_w_strings_exactly():
    rng = np.random.default_rng(29)
    raw = rng.normal(size=1 << 3) + 1j * rng.normal(size=1 << 3)
    state = raw / np.linalg.norm(raw)
    labels = ["ZII", "IZI", "ZZI", "ZZZ"]
    codes = [PauliWord.from_label(lbl).code for lbl in labels]
    plan = plan_measurement(3, codes, 1)

    p_ro = 0.03
    ideal = NoisyReadoutSimulator(state, 3, ReadoutNoiseModel.ideal())
    noisy = NoisyReadoutSimulator(state, 3, ReadoutNoiseModel(p_readout=p_ro))

    e_ideal = ideal.exact_word_expectations(plan.settings)
    e_noisy = noisy.exact_word_expectations(plan.settings)
    for label, code in zip(labels, codes, strict=True):
        weight = sum(ch != "I" for ch in label)
        assert math.isclose(e_noisy[code], ((1.0 - 2.0 * p_ro) ** weight) * e_ideal[code], abs_tol=1e-13)


def test_single_cz_bell_readout_matches_closed_form_attenuation():
    # On |Phi+> = (|00> + |11>)/sqrt(2), <XX> = <ZZ> = +1, <YY> = -1.
    # Graph-state diagonalizer for {XX, YY, ZZ} uses exactly 1 CZ gate.
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
    codes = [PauliWord.from_label(lbl).code for lbl in ("XX", "YY", "ZZ")]
    plan = plan_measurement(2, codes, 2)
    assert len(plan.settings) == 1
    assert plan.resources[0].n_2q == 1

    p_2q = 0.05
    noise = ReadoutNoiseModel(p_2q=p_2q, routing_2q_multiplier=2.0)
    sim = NoisyReadoutSimulator(state, 2, noise)
    exps = sim.exact_word_expectations(plan.settings)
    # Every non-identity 2Q Pauli attenuates by (1 - 16*p/15)^m under 2Q depolarizing.
    atten = (1.0 - (16.0 / 15.0) * p_2q) ** 2
    assert math.isclose(exps[codes[0]], atten, abs_tol=1e-13)
    assert math.isclose(exps[codes[1]], -atten, abs_tol=1e-13)
    assert math.isclose(exps[codes[2]], atten, abs_tol=1e-13)

    # Check PreparedSampledReference integration and empirical convergence to exact noisy mean.
    ref = PreparedSampledReference(2, "noisy-bell", sim.draw)
    obs = ref.sample(plan.settings, [20000], seed=41)
    emp_xx = (
        2 * obs.batch.plus_counts[codes[0]] - obs.batch.shots[codes[0]]
    ) / obs.batch.shots[codes[0]]
    assert abs(emp_xx - atten) < 0.02


def test_noise_model_validation_and_device_card_factory():
    card = DeviceCard.load("benchmarks/configs/device_cards/superconducting-like.json")
    full = ReadoutNoiseModel.from_device_card(card)
    assert full.p_1q == card.eps_1q
    assert full.p_2q == card.eps_2q
    assert full.routing_2q_multiplier == card.routing_2q_multiplier
    assert full.p_readout == card.eps_readout
    assert not full.is_gate_noiseless

    only_2q = ReadoutNoiseModel.from_device_card(card, include_1q=False, include_readout=False)
    assert only_2q.p_1q == 0.0 and only_2q.p_readout == 0.0 and only_2q.p_2q == card.eps_2q

    with pytest.raises(ValueError, match="p_2q"):
        ReadoutNoiseModel(p_2q=-0.01)
    with pytest.raises(ValueError, match="p_1q"):
        ReadoutNoiseModel(p_1q=0.9)
    with pytest.raises(ValueError, match="p_readout"):
        ReadoutNoiseModel(p_readout=0.6)
    with pytest.raises(ValueError, match="routing_2q_multiplier"):
        ReadoutNoiseModel(routing_2q_multiplier=0.5)
    with pytest.raises(ValueError, match="n <="):
        NoisyReadoutSimulator(np.eye(1, 1 << 11, dtype=complex)[0], 11, ReadoutNoiseModel(p_2q=0.01))
