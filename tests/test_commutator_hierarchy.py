"""Unit tests for hierarchical commutator operators and dynamic frontier expansion."""

from __future__ import annotations

import json
import unittest

import numpy as np

from clifford_qc import X, Y, Z, ket_density, to_matrix
from clifford_qc.backends import ExactMVBackend
from clifford_qc.dense_reference import exact_ground
from clifford_qc.fermion import total_number_op, total_sz_op
from clifford_qc.models.lattice import hubbard
from clifford_qc.models.spin import tfim, xxz
from clifford_qc.subspace import (
    IncrementalCostModel,
    commutator_apply,
    commutator_mv,
    determinant_excitations,
    hierarchy_span_diagnostics,
    nested_commutator_hierarchy,
    occupied_spin_orbitals,
    pauli_orbit,
    run_acase,
    run_dynamic_commutator_acase,
    sector_leakage,
    trotter_program,
)


class CommutatorAlgebraTests(unittest.TestCase):
    def test_commutator_mv_matches_exact_dense_matrix_across_depths(self):
        spin = tfim(3, J=1.0, h=0.7)
        H = spin.hamiltonian.to_mv()
        H_mat = to_matrix(H)

        # Mix Hermitian and non-Hermitian multi-word seeds
        seeds = [
            X(3, 0),
            Z(3, 1) * X(3, 2),
            0.4 * Y(3, 0) + (0.3 - 0.2j) * Z(3, 1),
        ]
        for seed in seeds:
            cur = seed
            for _ in range(3):
                comm_raw, diag = commutator_mv(H, cur, phase_mode="raw", normalize=False)
                dense_expected = H_mat @ to_matrix(cur) - to_matrix(cur) @ H_mat
                if comm_raw is None:
                    self.assertEqual(diag["status"], "vanished")
                    self.assertTrue(np.allclose(dense_expected, 0.0, atol=1e-11))
                    break
                self.assertEqual(diag["status"], "ok")
                self.assertTrue(np.allclose(to_matrix(comm_raw), dense_expected, atol=1e-10))
                # Verify norm consistency: raw_hs_norm == ||[H, A]||_F = 2^(n/2) * coeff_l2
                expected_frob = np.linalg.norm(dense_expected, ord="fro")
                self.assertAlmostEqual(diag["raw_hs_norm"], expected_frob, places=10)
                cur, _ = commutator_mv(H, cur, phase_mode="raw", normalize=True)

    def test_relative_zero_tolerance_immune_to_mv_absolute_pruning(self):
        """Regression test against MV's TOL=1e-12 wiping out tiny seed scales."""
        spin = tfim(3, J=1.0, h=0.8)
        H = spin.hamiltonian.to_mv()
        seed = X(3, 0) + 0.5 * Y(3, 1)

        ref_comm, ref_diag = commutator_mv(H, seed, normalize=True)
        self.assertIsNotNone(ref_comm)

        for scale in (1e-9, 1e9):
            with self.subTest(scale=scale):
                scaled_H = scale * H
                scaled_seed = scale * seed
                got, diag = commutator_mv(scaled_H, scaled_seed, normalize=True)
                self.assertIsNotNone(got)
                self.assertEqual(diag["status"], "ok")
                self.assertTrue(got.is_close(ref_comm, tol=1e-10))

        # Exact zero commutators must vanish cleanly regardless of scale
        self.assertIsNone(commutator_mv(H, H)[0])
        self.assertIsNone(commutator_mv(1e6 * H, Z(3, 0) * Z(3, 1) * Z(3, 2) if False else 1e-6 * H)[0])

    def test_phase_modes_hermiticity_and_span_invariance(self):
        spin = xxz(4, J=1.0, delta=1.1)
        H = spin.hamiltonian
        ref = ket_density(4, "0101")
        herm_seeds = pauli_orbit([w for w, _ in H.items()])[:6]

        diag_by_mode = {}
        for mode in ("raw", "hermitian", "i_commutator"):
            hier = nested_commutator_hierarchy(
                H, herm_seeds, max_depth=2, phase_mode=mode
            )
            if mode in ("hermitian", "i_commutator"):
                for node in hier.all_nodes:
                    self.assertTrue(
                        node.mv.is_hermitian(1e-10),
                        f"{node.label} not Hermitian under {mode}",
                    )
            diag_by_mode[mode] = hierarchy_span_diagnostics(ref, H, hier)

        # Projected eigenvalues, rank, and word universe must agree across phase modes
        for d in range(len(diag_by_mode["raw"])):
            self.assertEqual(
                diag_by_mode["raw"][d]["effective_rank"],
                diag_by_mode["hermitian"][d]["effective_rank"],
            )
            self.assertEqual(
                diag_by_mode["raw"][d]["effective_rank"],
                diag_by_mode["i_commutator"][d]["effective_rank"],
            )
            self.assertAlmostEqual(
                diag_by_mode["raw"][d]["ground_energy"],
                diag_by_mode["hermitian"][d]["ground_energy"],
                places=10,
            )
            self.assertAlmostEqual(
                diag_by_mode["raw"][d]["ground_energy"],
                diag_by_mode["i_commutator"][d]["ground_energy"],
                places=10,
            )
            self.assertEqual(
                diag_by_mode["raw"][d]["word_universe"],
                diag_by_mode["hermitian"][d]["word_universe"],
            )

        # Anti-Hermitian fermionic seeds under 'hermitian' must make depth>=1 strictly Hermitian
        hub = hubbard(2, t=1.0, U=4.0)
        occ = occupied_spin_orbitals(hub)
        exc = determinant_excitations(hub.n, occ)
        anti_hier = nested_commutator_hierarchy(
            hub.hamiltonian, exc, max_depth=2, phase_mode="hermitian"
        )
        for node in anti_hier.all_nodes:
            if node.depth >= 1:
                self.assertTrue(
                    node.mv.is_hermitian(1e-10),
                    f"depth {node.depth} node {node.label} not Hermitian",
                )

    def test_fermionic_symmetry_preservation_across_depths(self):
        hub = hubbard(2, t=1.0, U=4.0)
        H_mv = hub.hamiltonian.to_mv()
        occ = occupied_spin_orbitals(hub)
        exc = determinant_excitations(hub.n, occ)

        # Direct commutators with N and Sz must vanish
        N_op = total_number_op(hub.n)
        Sz_op = total_sz_op(hub.n)
        self.assertIsNone(commutator_mv(H_mv, N_op)[0])
        self.assertIsNone(commutator_mv(H_mv, Sz_op)[0])

        hier = nested_commutator_hierarchy(
            H_mv, exc, max_depth=2, phase_mode="hermitian"
        )
        for node in hier.all_nodes:
            leak = sector_leakage(node.generator)
            self.assertLessEqual(
                max(leak.values()),
                1e-11,
                f"sector leakage on {node.label}: {leak}",
            )

        # Verify summary_table is cleanly JSON-serializable
        table = hier.summary_table()
        encoded = json.dumps(table)
        self.assertTrue(len(json.loads(encoded)) == 3)

    def test_truncation_and_support_policies(self):
        spin = xxz(4, J=1.0, delta=0.9)
        H = spin.hamiltonian
        seeds = [X(4, 0) + Y(4, 1) + Z(4, 2)]

        # Truncate to max_support=3
        hier_trunc = nested_commutator_hierarchy(
            H,
            seeds,
            max_depth=2,
            phase_mode="hermitian",
            max_support=3,
            support_policy="truncate",
        )
        for node in hier_trunc.all_nodes:
            self.assertLessEqual(node.support(), 3)
            self.assertGreater(node.retained_l2_fraction, 0.0)
            self.assertLessEqual(node.retained_l2_fraction, 1.0 + 1e-12)

        # Drop policy should drop anything exceeding tight max_support=2
        hier_drop = nested_commutator_hierarchy(
            H,
            seeds,
            max_depth=2,
            max_support=2,
            support_policy="drop",
        )
        self.assertGreater(hier_drop.summaries[0].dropped_by_filter, 0)

        # Test commutator_apply helper
        gen, diag = commutator_apply(H, seeds[0], label="dH_s0", phase_mode="i_commutator")
        self.assertIsNotNone(gen)
        self.assertEqual(gen.label, "dH_s0")
        self.assertTrue(gen.mv.is_hermitian(1e-10))

    def test_lie_closure_detection_on_single_qubit_algebra(self):
        # On 1 qubit, [X + 0.5 Z, Z] -> Y -> [H, Y] -> combination of X, Z -> finite closure
        H = X(1, 0) + 0.5 * Z(1, 0)
        hier = nested_commutator_hierarchy(
            H, [Z(1, 0)], max_depth=6, phase_mode="hermitian"
        )
        self.assertGreater(len(hier.all_nodes), 1)
        # By depth <= 6, nested commutators repeat / close in sl(2)
        self.assertGreater(sum(s.deduplicated for s in hier.summaries), 0)

        # Even simpler exact closure: [Z, Z] vanishes immediately at depth 1
        hier_abelian = nested_commutator_hierarchy(Z(1, 0), [Z(1, 0)], max_depth=3)
        self.assertTrue(hier_abelian.closure_reached)
        self.assertEqual(hier_abelian.closure_depth, 1)


class DynamicFrontierTests(unittest.TestCase):
    def test_dynamic_frontier_matches_static_with_fewer_commutators_on_spin(self):
        spin = tfim(4, J=1.0, h=1.0)
        H = spin.hamiltonian
        E_gs, _ = exact_ground(H.to_mv())
        ref = ket_density(4, "0000")
        seeds = pauli_orbit([w for w, _ in H.items()])

        static_hier = nested_commutator_hierarchy(
            H, seeds, max_depth=2, phase_mode="hermitian"
        )
        res_static = run_acase(
            ref, H, static_hier.all_generators, max_size=7, min_lowering=1e-10
        )

        # Static depth 2 materializes many commutators upfront
        total_static_comm = sum(s.kept + s.deduplicated + s.vanished for s in static_hier.summaries[1:])

        for mode in ("on_select", "on_stall", "on_select_or_stall", "top_k"):
            with self.subTest(expand_mode=mode):
                res_dyn = run_dynamic_commutator_acase(
                    ref,
                    H,
                    seeds,
                    max_depth=2,
                    initial_depth=0,
                    expand_mode=mode,
                    expand_top_k=2,
                    phase_mode="hermitian",
                    max_size=7,
                    min_lowering=1e-10,
                )
                self.assertGreaterEqual(res_dyn.energy, E_gs - 1e-9)
                self.assertLessEqual(
                    res_dyn.resources["commutators_generated"],
                    total_static_comm,
                )
                self.assertLessEqual(
                    len(res_dyn.bank.word_set()),
                    len(res_static.bank.word_set()),
                )
                self.assertLessEqual(
                    res_dyn.resources["word_universe"],
                    res_static.resources["word_universe"],
                )

        # On TFIM with depth 2, dynamic should match static accuracy closely
        res_oss = run_dynamic_commutator_acase(
            ref,
            H,
            seeds,
            max_depth=2,
            expand_mode="on_select_or_stall",
            phase_mode="hermitian",
            max_size=7,
        )
        self.assertAlmostEqual(res_oss.energy, res_static.energy, places=3)
        self.assertLess(
            len(res_oss.bank.word_set()),
            len(res_static.bank.word_set()),
        )
        self.assertLess(
            res_oss.resources["word_universe"],
            res_static.resources["word_universe"],
        )

    def test_dynamic_frontier_with_correlated_reference_and_cost_model(self):
        spin = xxz(4, J=1.0, delta=1.1)
        H = spin.hamiltonian
        E_gs, _ = exact_ground(H.to_mv())
        # Correlated reference from real-time Trotter evolution of Neel state
        backend = ExactMVBackend()
        ref0 = backend.state(spin.reference, ())
        corr_ref = backend.state(
            trotter_program(H, 0.35, steps=6), (), initial_state=ref0
        )
        seeds = pauli_orbit([w for w, _ in H.items()])

        e_init = float((corr_ref * H.to_mv()).trace().real)
        cost_model = IncrementalCostModel(
            word_weight=0.05,
            setting_weight=0.1,
            epsilon=1.0,
        )
        res = run_dynamic_commutator_acase(
            corr_ref,
            H,
            seeds,
            max_depth=2,
            expand_mode="on_select_or_stall",
            phase_mode="hermitian",
            max_size=6,
            cost_model=cost_model,
        )
        self.assertLess(res.energy, e_init)
        self.assertGreaterEqual(res.energy, E_gs - 1e-9)
        self.assertEqual(res.resources["selection_criterion"], "information_ratio")

    def test_dynamic_frontier_fermionic_sector_and_stagnation_recovery(self):
        hub = hubbard(2, t=1.0, U=4.0)
        H = hub.hamiltonian
        E_gs, _ = exact_ground(H.to_mv())
        ref = ExactMVBackend().state(hub.reference, ())
        # Give ONLY singles as seeds; doubles must emerge dynamically via [H, S]
        occ = occupied_spin_orbitals(hub)
        singles = determinant_excitations(hub.n, occ, max_rank=1)

        res = run_dynamic_commutator_acase(
            ref,
            H,
            singles,
            max_depth=2,
            expand_mode="on_select_or_stall",
            phase_mode="hermitian",
            max_size=6,
            leakage_tol=1e-9,
        )
        # Depth-1 commutators [H, singles] capture doubles couplings on Hubbard dimer
        self.assertGreater(res.resources["commutators_generated"], 0)
        self.assertLess(res.energy, float((ref * H.to_mv()).trace().real))
        self.assertGreaterEqual(res.energy, E_gs - 1e-9)


if __name__ == "__main__":
    unittest.main()
