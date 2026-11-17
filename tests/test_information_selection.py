"""Unit tests for resource-/information-aware scoring and closed-loop acquisition."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import unittest

import numpy as np

from clifford_qc import I, X, Z, ket_density
from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.ir import PauliSum, PauliWord
from clifford_qc.measurement import DeviceCard
from clifford_qc.models.spin import tfim, xxz
from clifford_qc.subspace import (
    Generator,
    IncrementalCostModel,
    MatrixElementBank,
    SharedMeasurement,
    acquire_to_accuracy,
    estimate_incremental_cost,
    expected_lowering_estimate,
    identity_generator,
    krylov_response,
    pauli_orbit,
    qwc_bases_for_words,
    qwc_setting_resources,
    ritz_uncertainty,
    run_acase,
    run_resource_aware_acase,
    score_candidate,
    score_information_candidate,
    trotter_program,
)

DEVICE_CARD_DIR = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "device_cards"
)


def _load_device_card(filename: str) -> DeviceCard:
    return DeviceCard.load(DEVICE_CARD_DIR / filename)


class IncrementalCostAndEquivalenceTests(unittest.TestCase):
    def test_exact_equivalence_to_legacy_score_candidate(self):
        spin = tfim(4, J=1.0, h=1.0)
        H = spin.hamiltonian
        ref = ket_density(4, "0000")
        bank = MatrixElementBank(ref, H)
        basis = tuple(bank.extend([identity_generator(4)]))
        pool = list(bank.extend(pauli_orbit([w for w, _ in H.items()]) + krylov_response(H, 3)))
        solved = bank.solve(basis)
        basis_words = bank.word_set(basis)

        # Case 1: gamma=1.0 <-> word_weight=1.0, epsilon=1.0
        model_g1 = IncrementalCostModel(word_weight=1.0, epsilon=1.0)
        for idx in pool:
            legacy = score_candidate(bank, basis, solved, idx, gamma=1.0, basis_words=basis_words)
            info = score_information_candidate(
                bank, basis, solved, idx, cost_model=model_g1, basis_words=basis_words
            )
            self.assertEqual(legacy.accepted, info.accepted)
            self.assertEqual(legacy.new_words, info.new_words)
            if legacy.accepted:
                self.assertAlmostEqual(legacy.predicted_lowering, info.predicted_lowering, places=12)
                self.assertAlmostEqual(legacy.score, info.score, places=12)

        # Case 2: gamma=0.0 <-> zero resource weights, epsilon=1.0
        model_g0 = IncrementalCostModel(epsilon=1.0)
        for idx in pool:
            legacy = score_candidate(bank, basis, solved, idx, gamma=0.0, basis_words=basis_words)
            info = score_information_candidate(
                bank, basis, solved, idx, cost_model=model_g0, basis_words=basis_words
            )
            if legacy.accepted:
                self.assertAlmostEqual(legacy.score, info.score, places=12)

    def test_qwc_setting_resources_and_device_card_pricing(self):
        # On 3 qubits: I=0, X=1, Y=2, Z=3 in packed base-4 code
        # Code "XYZ" -> q0=X(1), q1=Y(2), q2=Z(3)
        word_xyz = PauliWord.from_label("XYZ")
        res = qwc_setting_resources(3, word_xyz.code)
        # X needs 1 H, Y needs 2 (H, S†), Z needs 0 -> n_1q=3, d_1q=2, n_2q=0
        self.assertEqual(res.n_1q, 3)
        self.assertEqual(res.d_1q, 2)
        self.assertEqual(res.n_2q, 0)
        self.assertEqual(res.d_2q, 0)

        word_zz = PauliWord.from_label("ZZI")
        res_z = qwc_setting_resources(3, word_zz.code)
        self.assertEqual(res_z.n_1q, 0)
        self.assertEqual(res_z.d_1q, 0)

        # DeviceCard pricing sensitivity
        sc_card = _load_device_card("superconducting-like.json")
        ion_card = _load_device_card("ion-like.json")

        # Diagonal/Ising Z Hamiltonian so Z1 only creates Z-basis words compatible with existing ZZZZ group
        H_z = (-1.0 * (Z(4, 0) * Z(4, 1))) + (-0.5 * (Z(4, 2) * Z(4, 3))) + (0.4 * Z(4, 0))
        H_z_ps = PauliSum(4, dict(H_z.terms))
        ref = ket_density(4, "0000")
        bank = MatrixElementBank(ref, H_z_ps)
        basis = tuple(bank.extend([identity_generator(4)]))
        bank.solve(basis)  # populate basis word cache
        c_compatible = bank.add(Generator("Z1", Z(4, 1)))
        c_new = bank.add(Generator("X0Y1", X(4, 0) * PauliWord.from_label("IYII").to_mv()))
        basis_words = bank.word_set(basis)
        qwc_bases = qwc_bases_for_words(4, basis_words)

        for card in (sc_card, ion_card):
            m_plain = IncrementalCostModel(
                setting_weight=1.0,
                time_weight_us=1.0,
                classical_product_weight=1e-4,
                classical_term_weight=1e-4,
                device_card=card,
                shots_per_setting=50.0,
                fidelity_inflate=False,
            )
            m_fid = IncrementalCostModel(
                setting_weight=1.0,
                time_weight_us=1.0,
                device_card=card,
                shots_per_setting=50.0,
                fidelity_inflate=True,
            )
            b_compat = estimate_incremental_cost(
                bank, basis, c_compatible, cost_model=m_plain, basis_words=basis_words, qwc_bases=qwc_bases
            )
            b_new = estimate_incremental_cost(
                bank, basis, c_new, cost_model=m_plain, basis_words=basis_words, qwc_bases=qwc_bases
            )
            b_new_fid = estimate_incremental_cost(
                bank, basis, c_new, cost_model=m_fid, basis_words=basis_words, qwc_bases=qwc_bases
            )
            self.assertEqual(b_compat.incremental_groups, 0)
            self.assertEqual(b_compat.incremental_time_us, 0.0)
            self.assertGreater(b_new.incremental_groups, 0)
            self.assertGreater(b_new.incremental_time_us, 0.0)
            self.assertGreater(b_new.classical_products, 0)
            self.assertGreater(b_new.classical_terms, 0)
            self.assertGreaterEqual(b_new_fid.incremental_time_us, b_new.incremental_time_us)

    def test_resource_aware_acase_reduces_measurement_universe(self):
        spin = tfim(4, J=1.0, h=1.0)
        H = spin.hamiltonian
        ref = ket_density(4, "0000")
        pool = pauli_orbit([w for w, _ in H.items()]) + krylov_response(H, 4)

        res_energy = run_acase(ref, H, pool, max_size=5, gamma=0.0)
        sc_card = _load_device_card("superconducting-like.json")
        model = IncrementalCostModel(
            word_weight=0.05,
            setting_weight=0.2,
            time_weight_us=0.01,
            device_card=sc_card,
            shots_per_setting=10.0,
            epsilon=1.0,
        )
        res_info = run_resource_aware_acase(
            ref, H, pool, cost_model=model, max_size=5
        )
        self.assertEqual(res_info.resources["selection_criterion"], "information_ratio")
        self.assertLessEqual(
            res_info.resources["final_qwc_groups"],
            len(qwc_bases_for_words(4, res_energy.bank.word_set(res_energy.indices))),
        )
        self.assertLessEqual(
            res_info.resources["word_universe"],
            res_energy.resources["word_universe"],
        )

    def test_classical_cost_telemetry_populated_with_zero_weights(self):
        H = tfim(4, J=1.0, h=1.0).hamiltonian
        ref = ket_density(4, "0000")
        bank = MatrixElementBank(ref, H)
        basis = tuple(bank.extend([identity_generator(4)]))
        cand = bank.extend(krylov_response(H, 1))[0]

        zero_weight_model = IncrementalCostModel()
        breakdown = estimate_incremental_cost(
            bank,
            basis,
            cand,
            cost_model=zero_weight_model,
        )
        self.assertGreater(breakdown.classical_products, 0)
        self.assertGreater(breakdown.classical_terms, 0)
        self.assertEqual(breakdown.total_cost, 0.0)


class DeltaMethodUncertaintyAndAcquisitionTests(unittest.TestCase):
    def test_delta_method_lowering_se_matches_empirical_mc_and_gates_noise(self):
        spin = tfim(4, J=1.0, h=1.0)
        H = spin.hamiltonian
        ref = ket_density(4, "0000")
        bank = MatrixElementBank(ref, H)
        basis = list(bank.extend([identity_generator(4)]))
        cand_idx = bank.extend(krylov_response(H, 2))[0]

        sess = SharedMeasurement(bank, basis + [cand_idx], pooling="shots")
        basis_sess = SharedMeasurement(bank, basis, groups=sess.groups, pooling="shots")

        # Measure empirical MC spread across independent finite-shot runs
        shots = 3000
        low_samples = []
        orth_samples = []
        pred_low_stds = []
        pred_orth_stds = []
        for seed in range(25):
            backend = FiniteShotBackend(seed=100 + seed)
            cache = sess.measure(backend, shots)
            res_b = basis_sess.solve(cache)
            low, low_std, orth, orth_std, _ = expected_lowering_estimate(
                bank, basis, res_b, cand_idx, cache=cache
            )
            low_samples.append(low)
            orth_samples.append(orth)
            pred_low_stds.append(low_std)
            pred_orth_stds.append(orth_std)

        emp_low_std = float(np.std(low_samples, ddof=1))
        emp_orth_std = float(np.std(orth_samples, ddof=1))
        med_low_std = float(np.median(pred_low_stds))
        med_orth_std = float(np.median(pred_orth_stds))
        self.assertGreater(med_low_std, 0.0)
        self.assertGreater(med_orth_std, 0.0)
        # Delta-method SE should be on the same order of magnitude as MC std (within factor ~2.8)
        self.assertLess(abs(math.log10(med_low_std / emp_low_std)), 0.45)
        self.assertLess(abs(math.log10(med_orth_std / emp_orth_std)), 0.45)

        # Noise gate 1: risk_aversion zeroes out noise-dominated lowering for I + 0.01 * X_0
        noisy_dup = bank.add(Generator("near_dup", I(4) + 0.01 * X(4, 0)))
        dup_sess = SharedMeasurement(bank, basis + [noisy_dup], pooling="shots")
        tiny_cache = dup_sess.measure(FiniteShotBackend(seed=7), 30)
        res_tiny = SharedMeasurement(bank, basis, groups=dup_sess.groups, pooling="shots").solve(
            tiny_cache
        )
        risk_model = IncrementalCostModel(
            expected_improvement_mode="risk_adjusted",
            risk_aversion=2.0,
            require_stat_orthogonality=False,
            epsilon=1.0,
        )
        sc_risk = score_information_candidate(
            bank, basis, res_tiny, noisy_dup, cost_model=risk_model, cache=tiny_cache,
            min_orthogonality=1e-8,
        )
        self.assertGreater(sc_risk.lowering_std, sc_risk.predicted_lowering)
        self.assertEqual(sc_risk.risk_adjusted_lowering, 0.0)

        # Noise gate 2: genuine uncertain orthogonal component triggers statistical gate
        corr_ref = ExactMVBackend().state(trotter_program(H, 0.25, steps=4), ())
        bank_c = MatrixElementBank(corr_ref, H)
        basis_c = list(bank_c.extend([identity_generator(4), Generator("X0", X(4, 0))]))
        noisy_orth = bank_c.add(Generator("X0_eps", X(4, 0) + 0.08 * Z(4, 1)))
        orth_sess = SharedMeasurement(bank_c, basis_c + [noisy_orth], pooling="shots")
        orth_cache = orth_sess.measure(FiniteShotBackend(seed=7), 30)
        res_orth = SharedMeasurement(bank_c, basis_c, groups=orth_sess.groups, pooling="shots").solve(
            orth_cache
        )
        strict_model = IncrementalCostModel(
            require_stat_orthogonality=True,
            orthogonality_z=2.0,
            epsilon=1.0,
        )
        sc_strict = score_information_candidate(
            bank_c, basis_c, res_orth, noisy_orth, cost_model=strict_model, cache=orth_cache,
            min_orthogonality=1e-8,
        )
        self.assertFalse(sc_strict.accepted)
        self.assertIn("orthogonal fraction unresolved", sc_strict.rejected or "")

        # Verify expected improvement modes
        cache_hi = sess.measure(FiniteShotBackend(seed=9), 1000)
        res_hi = basis_sess.solve(cache_hi)
        scores_by_mode = {}
        for mode in ("point", "risk_adjusted", "shrinkage", "snr_gated"):
            cm = IncrementalCostModel(
                expected_improvement_mode=mode,
                risk_aversion=2.0,
                epsilon=1.0,
            )
            scores_by_mode[mode] = score_information_candidate(
                bank, basis, res_hi, cand_idx, cost_model=cm, cache=cache_hi
            )
        self.assertGreaterEqual(
            scores_by_mode["point"].risk_adjusted_lowering,
            scores_by_mode["risk_adjusted"].risk_adjusted_lowering,
        )
        self.assertGreaterEqual(
            scores_by_mode["point"].risk_adjusted_lowering,
            scores_by_mode["shrinkage"].risk_adjusted_lowering,
        )

    def test_acquire_to_accuracy_edge_case_identity_and_no_oracle_leakage(self):
        # Strictly verify acquire_to_accuracy accepts no oracle ground-energy argument
        sig = inspect.signature(acquire_to_accuracy)
        self.assertNotIn("exact_ground_energy", sig.parameters)
        self.assertNotIn("target_error", sig.parameters)

        # Edge case: Hamiltonian = I (all words deterministic / empty req set {0})
        H_triv = I(2)
        ref = ket_density(2, "00")
        cands = [X(2, 0)]
        acq = acquire_to_accuracy(
            ref,
            H_triv,
            cands,
            FiniteShotBackend(seed=1),
            target_accuracy=0.05,
            pilot_shots=100,
        )
        self.assertTrue(acq.converged)
        self.assertAlmostEqual(acq.energy, 1.0, places=10)
        self.assertEqual(acq.std_error, 0.0)

    def test_acquire_to_accuracy_balances_expand_and_refine_shots(self):
        spin = tfim(4, J=1.0, h=1.0)
        H = spin.hamiltonian
        ref = ket_density(4, "0000")
        cands = krylov_response(H, 4) + pauli_orbit([w for w, _ in H.items()])[:6]
        sc_card = _load_device_card("superconducting-like.json")

        for metric in ("std_error", "ci_halfwidth"):
            with self.subTest(stop_metric=metric):
                cost_model = IncrementalCostModel(
                    setting_weight=1.0,
                    time_weight_us=0.01,
                    device_card=sc_card,
                    shots_per_setting=100.0,
                    risk_aversion=1.0,
                    epsilon=1.0,
                )
                acq = acquire_to_accuracy(
                    ref,
                    H,
                    cands,
                    FiniteShotBackend(seed=17),
                    target_accuracy=0.015,
                    stop_metric=metric,
                    cost_model=cost_model,
                    max_basis_size=6,
                    max_steps=30,
                    max_total_shots=60_000,
                    pilot_shots=60,
                    refine_batch_shots=600,
                )
                self.assertEqual(acq.evidence, "asymptotic")
                self.assertGreaterEqual(acq.resources["expand_actions"], 1)
                self.assertGreaterEqual(acq.resources["refine_actions"], 1)
                self.assertGreater(acq.total_time_us, 0.0)
                self.assertLessEqual(acq.total_shots, 60_000 + 2000)
                if metric == "std_error":
                    self.assertLessEqual(acq.std_error, acq.target_accuracy + 1e-12)
                else:
                    halfw = 0.5 * (acq.ci_interval.upper - acq.ci_interval.lower)
                    self.assertLessEqual(halfw, acq.target_accuracy + 1e-12)
                self.assertTrue(acq.converged)

        # Matched-total-shots comparison against uniform open-loop SharedMeasurement on same final basis
        uniform_sess = SharedMeasurement(acq.bank, acq.indices, pooling="shots")
        per_group_shots = max(1, acq.total_shots // max(1, len(uniform_sess.groups)))
        uniform_cache = uniform_sess.measure(FiniteShotBackend(seed=17), per_group_shots)
        uniform_res = uniform_sess.solve(uniform_cache)
        uniform_iv = ritz_uncertainty(uniform_sess, uniform_cache, uniform_res, root=0)
        self.assertLess(acq.std_error, float(uniform_iv.std_error))

        # Verify tight shot budget cap terminates gracefully without crashing
        acq_capped = acquire_to_accuracy(
            ref,
            H,
            cands,
            FiniteShotBackend(seed=3),
            target_accuracy=1e-4,
            max_total_shots=3500,
            pilot_shots=150,
            refine_batch_shots=400,
        )
        self.assertFalse(acq_capped.converged)
        self.assertIn("budget", acq_capped.stopped_reason)
        self.assertLessEqual(acq_capped.total_shots, 3600)

    def test_orthogonality_corroboration_and_unresolved_lowering_preservation(self):
        H = tfim(4, J=1.0, h=1.0).hamiltonian
        ref = ket_density(4, "0000")
        bank = MatrixElementBank(ref, H)
        basis = list(bank.extend([identity_generator(4)] + krylov_response(H, 2)))
        cand_h3 = bank.extend(krylov_response(H, 3))[2]
        sess = SharedMeasurement(bank, basis + [cand_h3], pooling="shots")
        cache = sess.measure(FiniteShotBackend(seed=11), 200)
        res = SharedMeasurement(bank, basis, groups=sess.groups, pooling="shots").solve(cache)

        # H^3 has small noisy orthogonal fraction (< 2 * orth_std) but strong lowering and coupling SNRs
        sc_z2 = score_information_candidate(
            bank,
            basis,
            res,
            cand_h3,
            cost_model=IncrementalCostModel(
                require_stat_orthogonality=True, orthogonality_z=2.0, epsilon=1.0
            ),
            cache=cache,
            min_orthogonality=1e-8,
        )
        self.assertGreaterEqual(sc_z2.orthogonal_fraction, 0.005)
        self.assertLess(sc_z2.orthogonal_fraction, 2.0 * sc_z2.orthogonal_std)
        self.assertTrue(sc_z2.accepted)

        # Raising z=8.0 fails both orth_frac >= z*orth_std and orth_corroborated,
        # triggering "orthogonal fraction unresolved" while preserving raw predicted_lowering
        sc_z8 = score_information_candidate(
            bank,
            basis,
            res,
            cand_h3,
            cost_model=IncrementalCostModel(
                require_stat_orthogonality=True, orthogonality_z=8.0, epsilon=1.0
            ),
            cache=cache,
            min_orthogonality=1e-8,
        )
        self.assertFalse(sc_z8.accepted)
        self.assertIn("orthogonal fraction unresolved", sc_z8.rejected or "")
        self.assertGreater(sc_z8.predicted_lowering, 0.0)
        self.assertAlmostEqual(sc_z8.predicted_lowering, sc_z2.predicted_lowering, places=12)
        self.assertEqual(sc_z8.risk_adjusted_lowering, 0.0)
        self.assertEqual(sc_z8.score, 0.0)

        # Near-duplicate I + 0.01*X0 and weak X0 + 0.08*Z1 do not leak past orth_corroborated
        dup = bank.add(Generator("I_epsX", I(4) + 0.01 * X(4, 0)))
        dup_sess = SharedMeasurement(bank, basis[:1] + [dup], pooling="shots")
        dup_cache = dup_sess.measure(FiniteShotBackend(seed=7), 30)
        res_dup = SharedMeasurement(bank, basis[:1], groups=dup_sess.groups, pooling="shots").solve(dup_cache)
        sc_dup = score_information_candidate(
            bank,
            basis[:1],
            res_dup,
            dup,
            cost_model=IncrementalCostModel(
                require_stat_orthogonality=True, orthogonality_z=2.0, risk_aversion=2.0, epsilon=1.0
            ),
            cache=dup_cache,
            min_orthogonality=1e-8,
        )
        self.assertLess(sc_dup.orthogonal_fraction, 0.005)
        self.assertGreater(sc_dup.lowering_std, sc_dup.predicted_lowering)
        self.assertEqual(sc_dup.risk_adjusted_lowering, 0.0)
        self.assertEqual(sc_dup.score, 0.0)

        corr_ref = ExactMVBackend().state(trotter_program(H, 0.25, steps=4), ())
        bank_c = MatrixElementBank(corr_ref, H)
        basis_c = list(bank_c.extend([identity_generator(4), Generator("X0", X(4, 0))]))
        noisy_orth = bank_c.add(Generator("X0_eps", X(4, 0) + 0.08 * Z(4, 1)))
        orth_sess = SharedMeasurement(bank_c, basis_c + [noisy_orth], pooling="shots")
        orth_cache = orth_sess.measure(FiniteShotBackend(seed=7), 30)
        res_orth = SharedMeasurement(bank_c, basis_c, groups=orth_sess.groups, pooling="shots").solve(orth_cache)
        sc_orth = score_information_candidate(
            bank_c,
            basis_c,
            res_orth,
            noisy_orth,
            cost_model=IncrementalCostModel(
                require_stat_orthogonality=True, orthogonality_z=2.0, epsilon=1.0
            ),
            cache=orth_cache,
            min_orthogonality=1e-8,
        )
        self.assertFalse(sc_orth.accepted)
        self.assertIn("orthogonal fraction unresolved", sc_orth.rejected or "")

    def test_acquire_to_accuracy_converges_cleanly_across_models_and_devices(self):
        sc_card = _load_device_card("superconducting-like.json")
        ion_card = _load_device_card("ion-like.json")
        cases = [
            ("tfim4_sc_12", tfim(4, J=1.0, h=1.0), sc_card, 12),
            ("tfim4_sc_17", tfim(4, J=1.0, h=1.0), sc_card, 17),
            ("tfim4_ion_12", tfim(4, J=1.0, h=1.0), ion_card, 12),
            ("xxz4_sc_11", xxz(4, J=1.0, delta=1.0), sc_card, 11),
        ]
        for label, mod, card, seed in cases:
            with self.subTest(case=label):
                rho = ExactMVBackend().state(mod.reference, ())
                hw = [w for w, _ in mod.hamiltonian.items() if w.code != 0]
                pool = krylov_response(mod.hamiltonian, 3) + pauli_orbit(hw)[:8]
                cm = IncrementalCostModel(
                    setting_weight=0.1,
                    time_weight_us=0.001 if card.name == "superconducting-like" else 0.0001,
                    device_card=card,
                    shots_per_setting=60.0,
                    risk_aversion=1.0,
                    epsilon=1.0,
                )
                acq = acquire_to_accuracy(
                    rho,
                    mod.hamiltonian,
                    pool,
                    FiniteShotBackend(seed=seed),
                    target_accuracy=0.02,
                    stop_metric="std_error",
                    cost_model=cm,
                    max_basis_size=6,
                    max_steps=80,
                    pilot_shots=60,
                    refine_batch_shots=600,
                    max_total_shots=60_000,
                )
                self.assertTrue(acq.converged, msg=acq.stopped_reason)
                self.assertLessEqual(acq.std_error, 0.02 + 1e-12)
                self.assertIn("target accuracy reached", acq.stopped_reason)
                self.assertGreaterEqual(acq.resources["expand_actions"], 1)
                self.assertLess(acq.total_shots, 60_000)


if __name__ == "__main__":
    unittest.main()
