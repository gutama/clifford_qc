"""Independent dense Kraus controls and falsification of geometric shortcuts."""
import copy
import unittest
from unittest import mock

import numpy as np

import benchmarks.check_observable_noise as independent
import benchmarks.run_observable_noise as producer
from benchmarks.experiment_artifacts import read_json
from benchmarks.run_observable_noise import blocks, dense_noise, inputs, pair, run, verify
from clifford_qc.channels import amplitude_damping, dephasing, depolarizing
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.dense_reference import to_matrix
from clifford_qc.multivector import MV
from clifford_qc.diagnostics import (
    local_noise_adjoint, observable_defect, operator_span_diagnostics,
)


class ObservableNoiseTests(unittest.TestCase):
    def test_sparse_adjoint_matches_independent_dense_kraus_duality(self):
        rng = np.random.default_rng(29031)
        for n in (1, 2, 3):
            v = rng.normal(size=2**n)+1j*rng.normal(size=2**n)
            v /= np.linalg.norm(v)
            rho = np.outer(v, v.conj())
            op = MV(n, {c: rng.normal()+1j*rng.normal() for c in range(4**n)})
            dense = to_matrix(op)
            for kind in ("dephasing", "depolarizing", "amplitude_damping", "z_rotation"):
                for p in (0., .2, 1.):
                    with self.subTest(n=n, kind=kind, p=p):
                        noisy = dense_noise(rho, kind, p)
                        lhs = np.trace(noisy @ dense)
                        rhs = np.vdot(v, to_matrix(local_noise_adjoint(op, kind, p)) @ v)
                        self.assertAlmostEqual(lhs, rhs, places=10)
                        self.assertAlmostEqual(np.trace(noisy), 1., places=12)
                        self.assertGreater(np.linalg.eigvalsh(noisy)[0], -1e-12)

    def test_probability_convention_matches_existing_channels(self):
        op = MV.from_terms(2, {"XY": .3j, "ZZ": -.7, "II": 1., "IZ": .4})
        for kind, factory in (("dephasing", dephasing), ("depolarizing", depolarizing),
                              ("amplitude_damping", amplitude_damping)):
            for p in (0., .3, 1.):
                image = op
                for j in range(2):
                    image = sum((k.dagger()*image*k for k in factory(2, j, p)), MV(2))
                np.testing.assert_allclose(to_matrix(local_noise_adjoint(op, kind, p)),
                                           to_matrix(image), atol=1e-12)

    def test_dephasing_fixed_diagonal_observables(self):
        ops = [MV.from_label(s) for s in ("II", "ZI", "IZ", "ZZ")]
        result = operator_span_diagnostics(ops, lambda o: local_noise_adjoint(o, "dephasing", .4))
        self.assertEqual(result["fixed_dimension"], 4)
        self.assertLess(result["pointwise_defect"], 1e-12)

    def test_closed_span_can_be_strongly_attenuated(self):
        ops = [MV.from_label(s) for s in ("XIII", "XXII", "XXXX")]
        result = operator_span_diagnostics(ops, lambda o: local_noise_adjoint(o, "depolarizing", .3))
        self.assertLess(result["leakage"], 1e-12)
        self.assertEqual(result["fixed_dimension"], 0)
        self.assertGreater(result["pointwise_defect"], .8)
        np.testing.assert_allclose(result["restricted_singular_values"], [.6, .36, .1296])

    def test_rank_loss_not_hidden_by_zero_leakage(self):
        result = operator_span_diagnostics([MV.from_label("X"), MV.from_label("Z")],
                                           lambda o: local_noise_adjoint(o, "depolarizing", .75))
        self.assertEqual(result["image_rank"], 0)
        self.assertLess(result["leakage"], 1e-12)
        self.assertAlmostEqual(result["max_principal_angle_radians"], np.pi/2)

    def test_nonunital_identity_mixing_requires_identity_in_span(self):
        z = MV.from_label("Z")
        image = lambda o: local_noise_adjoint(o, "amplitude_damping", .4)
        isolated = operator_span_diagnostics([z], image)
        closed = operator_span_diagnostics([z, MV.from_label("I")], image)
        self.assertAlmostEqual(isolated["leakage"], .4)
        self.assertLess(closed["leakage"], 1e-12)
        self.assertEqual(closed["fixed_dimension"], 1)
        self.assertGreater(closed["pointwise_defect"], .5)

    def test_coherent_span_rotation_is_not_pointwise_preservation(self):
        image = lambda o: local_noise_adjoint(o, "z_rotation", .7)
        rotated = operator_span_diagnostics([MV.from_label("X"), MV.from_label("Y")], image)
        self.assertLess(rotated["leakage"], 1e-12)
        np.testing.assert_allclose(rotated["restricted_singular_values"], [1., 1.])
        self.assertGreater(rotated["pointwise_defect"], .6)
        self.assertEqual(rotated["fixed_dimension"], 0)

    def test_accidentally_stable_zero_mean_not_a_fixed_observable(self):
        z = MV.from_label("Z")
        v = np.array([1., 1.])/np.sqrt(2)
        image = local_noise_adjoint(z, "depolarizing", .3)
        self.assertAlmostEqual(np.vdot(v, to_matrix(z) @ v), 0.)
        self.assertAlmostEqual(np.vdot(v, to_matrix(image) @ v), 0.)
        self.assertAlmostEqual(observable_defect(z, image)["relative_hs_defect"], .4)

    def test_span_metrics_invariant_under_basis_change_and_rescaling(self):
        x, z = MV.from_label("XI"), MV.from_label("ZI")
        image = lambda o: local_noise_adjoint(o, "amplitude_damping", .2)
        a = operator_span_diagnostics([x, z], image)
        b = operator_span_diagnostics([1e8*(x+2j*z), .01*z, x], image)
        for field in ("rank", "image_rank", "leakage", "pointwise_defect", "fixed_dimension"):
            self.assertAlmostEqual(a[field], b[field], places=10)
        np.testing.assert_allclose(a["restricted_singular_values"], b["restricted_singular_values"])

    def test_large_register_uses_only_declared_support(self):
        ops = [MV.from_label(s+"I"*127) for s in ("I", "X", "Z")]
        result = operator_span_diagnostics(ops, lambda o: local_noise_adjoint(o, "dephasing", .1))
        self.assertEqual(result["support_rows"], 3)
        self.assertEqual(result["fixed_dimension"], 2)

    def test_invalid_input_and_channel_image(self):
        x = MV.from_label("X")
        for p in (float("nan"), float("inf"), -.1, 1.1, True, np.bool_(True), 1j, [0.], "0.2"):
            with self.subTest(p=p), self.assertRaises(ValueError):
                local_noise_adjoint(x, "dephasing", p)
        for ops in ([], [MV(1)], [x, MV.from_label("II")]):
            with self.assertRaises(ValueError):
                operator_span_diagnostics(ops, lambda o: o)
        with self.assertRaises(ValueError):
            operator_span_diagnostics([x], lambda o: MV.from_label("XX"))
        for tol in (True, np.bool_(True), 0., 1., 1j, "bad", float("nan")):
            with self.assertRaises(ValueError):
                operator_span_diagnostics([x], lambda o: o, rtol=tol)

    def test_analytic_global_fidelity_local_observable_separation(self):
        n, _, psi, _, _ = inputs("plus8")
        rho = np.outer(psi, psi.conj())
        noisy = dense_noise(rho, "depolarizing", .1)
        self.assertEqual(n, 8)
        self.assertAlmostEqual(np.vdot(psi, noisy @ psi).real, (1-2*.1/3)**8)
        x = MV.from_label("X"+"I"*7)
        self.assertAlmostEqual(np.trace(noisy @ to_matrix(x)).real, 1-4*.1/3)
        ghz = inputs("ghz8")[2]
        coherent = dense_noise(np.outer(ghz, ghz.conj()), "z_rotation", .4)
        self.assertAlmostEqual(np.vdot(ghz, coherent @ ghz).real, np.cos(8*.4/2)**2)

    def test_fixed_overlap_does_not_ensure_fixed_projected_energy(self):
        record = run({"cases": ["tfim4"], "channels": ["dephasing"], "strengths": [0., .2]})
        row = record["rows"][-1]
        self.assertLess(row["projections"]["diagonal"]["overlap_frobenius_drift"], 1e-12)
        self.assertLess(row["spans"]["diagonal_S"]["pointwise_defect"], 1e-12)
        self.assertGreater(row["projections"]["diagonal"]["absolute_energy_drift"], .1)
        self.assertEqual(verify(record, design=record["design"]), 2)

    def test_replay_rejects_tampered_values_and_missing_outcomes(self):
        record = run({"cases": ["plus4"], "channels": ["dephasing"], "strengths": [.1]})
        for mutate in (lambda r: r["rows"].pop(),
                       lambda r: r["rows"][0].update(fidelity_squared_pure_target=.99),
                       lambda r: r["source_sha256"].update({"clifford_qc/diagnostics.py": "bad"})):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                verify(changed, design=record["design"])
        with self.assertRaises(ValueError):
            verify(record)

    def test_unknown_cases_and_large_registers_are_rejected(self):
        for name in ("hubbard3", "plus10", "tfim6x", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                inputs(name)
        with mock.patch.dict(producer.CASES, {"plus9": ("plus", 9)}):
            with self.assertRaises(ValueError):
                inputs("plus9")

    def test_recorded_duality_detects_reversed_rotation(self):
        correct = producer.local_noise_adjoint

        def reversed_rotation(op, kind, p):
            return correct(op, kind, -p if kind == "z_rotation" else p)

        config = {"cases": ["plus4", "tfim4"], "channels": ["z_rotation"], "strengths": [.4]}
        def worst(record):
            return max(v["duality_defect"] for r in record["rows"] for v in r["observables"].values())
        self.assertLess(worst(run(config)), 1e-12)
        with mock.patch.object(producer, "local_noise_adjoint", reversed_rotation):
            self.assertGreater(worst(run(config)), .1)

    def test_h4_mixed_hop_couples_to_reference(self):
        n, h, psi, _, generators = inputs("h4")
        rho = np.outer(psi, psi.conj())

        def coupling(gens):
            s, hm = blocks(h, gens)
            dense = [pair(rho, [[to_matrix(o) for o in row] for row in ops]) for ops in (s, hm)]
            return [float(np.abs(m[:2, 2]).max()) for m in dense]

        self.assertGreater(min(coupling(generators["mixed"])), 1e-3)
        forbidden = cdag_op(n, 0)*c_op(n, 2) + cdag_op(n, 2)*c_op(n, 0)
        self.assertLess(max(coupling(generators["mixed"][:2] + [forbidden])), 1e-10)

    def test_signed_drift_rank_change_and_entry_bounds_are_recorded(self):
        record = run({"cases": ["plus4"], "channels": ["amplitude_damping"], "strengths": [0., .2]})
        still, damped = record["rows"]
        self.assertFalse(still["projections"]["mixed"]["retained_rank_changed"])
        mixed = damped["projections"]["mixed"]
        self.assertTrue(mixed["retained_rank_changed"])
        self.assertLess(mixed["signed_energy_drift"], 0)
        self.assertEqual(mixed["absolute_energy_drift"], -mixed["signed_energy_drift"])
        for arm in damped["projections"].values():
            for label in ("S", "H"):
                self.assertLessEqual(arm["max_entry_drift"][label],
                                     arm["max_entry_bias_bound_pauli_l1"][label]+1e-12)

    def test_principal_angle_is_accurate_near_zero(self):
        theta = 1e-6
        result = operator_span_diagnostics([MV.from_label("X")],
                                           lambda o: local_noise_adjoint(o, "z_rotation", theta))
        self.assertAlmostEqual(result["max_principal_angle_radians"], theta, delta=1e-15)

    def test_independent_checker_rebuilds_committed_headline_rows(self):
        headline = {("tfim6", "dephasing", .4), ("hubbard2", "dephasing", .4),
                    ("ghz8", "z_rotation", .4), ("h4", "dephasing", .005),
                    ("h4", "dephasing", .4), ("h4", "depolarizing", .2),
                    ("plus8", "depolarizing", .02)}
        problems = independent.check(read_json(independent.RECORD), read_json(independent.DESIGN),
                                     doc_text=independent.DOC.read_text(encoding="utf-8"),
                                     select=lambda *key: key in headline)
        self.assertEqual(problems, [])

    def test_independent_checker_rejects_tampering(self):
        design = {"cases": ["tfim4", "hubbard2"], "channels": ["amplitude_damping"], "strengths": [.2]}
        record = run(design)
        self.assertEqual(independent.check(record, design, pinned=False), [])
        mutations = (
            lambda r: r["rows"][0]["projections"]["mixed"].update(energy=-9.),
            lambda r: r["rows"][1]["spans"]["mixed_S"].update(leakage=0.),
            lambda r: r["rows"][0]["observables"]["Y0"].update(bias_bound_pauli_l1=0.),
            lambda r: r["rows"][1]["projections"]["diagonal"].update(retained_rank_changed=True),
            lambda r: r["rows"][0]["projections"]["diagonal"]["max_entry_drift"].update(S=0.),
            lambda r: r["cases"][1]["ideal_observables"].update(particle_number=3.),
            lambda r: r["rows"].pop(),
            lambda r: r["rows"][0]["observables"].pop("Y0"),
        )
        for j, mutate in enumerate(mutations):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.subTest(mutation=j):
                self.assertNotEqual(independent.check(changed, design, pinned=False), [])
        self.assertNotEqual(independent.check(record, design), [])
        text = independent.DOC.read_text(encoding="utf-8")
        committed = read_json(independent.RECORD)
        self.assertEqual(independent.check_document(committed, text), [])
        self.assertNotEqual(
            independent.check_document(committed, text.replace("0.0784113", "0.0784114")), [])


if __name__ == "__main__":
    unittest.main()
