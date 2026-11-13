"""Cross-path workflow results and portable evidence integrity."""
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from clifford_qc.backends.sector_reference import SectorActionProvider, SectorSpec, SectorStateReference
from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.measurement.compiled import CompiledMeasurementSampler
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.multivector import MV
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.states import ket_density
from clifford_qc.subspace.providers import ExactProjectedProvider
from clifford_qc.subspace.sampled import SampledProjectedProvider
from clifford_qc.subspace.sector_solver import SectorSolveConfig
from clifford_qc.workflows import export_workflow_evidence, run_workflow, verify_workflow_evidence

ROOT = Path(__file__).resolve().parents[1]


def sampled_reference(rho):
    def draw(settings, counts, seed, identity):
        result = CompiledMeasurementSampler(seed).sample_from_state(
            rho, settings, {s.key: count for s, count in zip(settings, counts)})
        return replace(result, state_key=identity)
    return PreparedSampledReference(rho.n, "workflow-fixture", draw)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.h = MV.from_terms(2, {"ZI": 0.7, "IX": -0.4, "XX": 0.2})
        self.gens = [("I", MV.scalar(2, 1)), ("X", MV.from_terms(2, {"IX": 1}))]
        self.exact = ExactProjectedProvider(DeterminantReference("00"), self.h, self.gens)

    def test_exact_matches_independent_dense_projection_and_reports_unknown_costs(self):
        result = run_workflow(self.exact, path="exact_projection", problem="two-qubit", unit="a.u.")
        doc = result.as_dict()
        matrix = to_matrix(self.h)
        projected = matrix[np.ix_([0, 1], [0, 1])]
        np.testing.assert_allclose(doc["outcome"]["energies"], np.linalg.eigvalsh(projected), atol=1e-12)
        self.assertEqual(doc["outcome"]["ground_state_status"], "not_established")
        self.assertIsNone(doc["cost_ledger"]["end_to_end_total"])
        self.assertEqual(doc["cost_ledger"]["stages"]["preparation"]["status"], "not_measured")
        doc["problem"]["label"] = "modified"
        self.assertEqual(result.as_dict()["problem"]["label"], "two-qubit")
        from clifford_qc import workflows
        self.assertEqual(
            set(workflows.__all__),
            {
                "WorkflowResult", "adapt_warm_start", "export_workflow_evidence",
                "run_workflow", "verify_workflow_evidence",
            },
        )
        for symbol in workflows.__all__:
            self.assertTrue(hasattr(workflows, symbol), symbol)

    def test_sampling_preserves_physical_counts_and_refuses_spectral_claims(self):
        p = SampledProjectedProvider(sampled_reference(ket_density(2, "00")), self.h, self.gens)
        result = run_workflow(p, path="sampled_projection", problem="two-qubit", unit="a.u.",
                              shots=10000, seed=17)
        doc = result.as_dict()
        expected = self.exact.solve().ground_energy
        self.assertLess(abs(doc["outcome"]["energies"][0] - expected), 0.06)
        self.assertEqual(doc["cost_ledger"]["measurement"]["shots"], 10000 * len(p.plan.settings))
        self.assertEqual(sum(g["shots"] for g in doc["outcome"]["method_details"]["groups"]),
                         doc["cost_ledger"]["measurement"]["shots"])
        self.assertEqual(doc["outcome"]["uncertainty"]["status"], "entrywise_only")
        self.assertIsNone(doc["outcome"]["uncertainty"]["energy_interval"])
        self.assertEqual(doc["outcome"]["ground_state_status"], "not_established")
        self.assertEqual(doc["cost_ledger"]["stages"]["sampling"]["status"], "measured")

    def test_invalid_request_rejected_before_sampling(self):
        p = SampledProjectedProvider(sampled_reference(ket_density(2, "00")), self.h, self.gens)
        for args in ({"path": "sector_action"}, {"path": "sampled_projection", "shots": 0, "seed": 0},
                     {"path": "sampled_projection", "shots": 10, "seed": 1, "delta": 2}):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                run_workflow(p, problem="x", unit="Ha", **args)

    def test_failed_sampled_solve_preserves_shots_and_observations(self):
        p = SampledProjectedProvider(sampled_reference(ket_density(2, "00")), self.h, self.gens)
        result = run_workflow(p, path="sampled_projection", problem="test", unit="Ha",
                              shots=100, seed=2, solve_options={"tau_s": 2.0}).as_dict()
        self.assertEqual(result["outcome"]["status"], "sampled_projection_failed")
        self.assertEqual(result["outcome"]["energies"], [])
        self.assertEqual(result["cost_ledger"]["measurement"]["shots"], 100 * len(p.plan.settings))
        self.assertEqual(result["cost_ledger"]["stages"]["solve"]["status"], "failed")
        self.assertTrue(result["outcome"]["method_details"]["groups"])

    def test_invalid_projected_options_do_not_draw_samples(self):
        def must_not_draw(*args):
            self.fail("invalid options must be rejected before sampling")

        p = SampledProjectedProvider(PreparedSampledReference(2, "no-draw", must_not_draw),
                                     self.h, self.gens)
        invalid = [
            {"tau_s": -1}, {"tau_s": "1"}, {"tau_s": [0.1]},
            {"tau_s": float("nan")}, {"rel_tau": -1}, {"norm_floor": float("inf")},
            {"max_condition": 0}, {"max_condition": -1},
            {"max_condition": float("nan")}, {"max_condition": True},
            {"overlap_ridge": -1}, {"overlap_noise_floor": [0, -1]},
            {"overlap_noise_floor": [[1]]}, {"overlap_ridge": [float("inf")]},
            {"overlap_ridge": ["0"]}, {"overlap_ridge": []},
            {"overlap_noise_floor": [0, 0, 0]},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_workflow(p, path="sampled_projection", problem="x", unit="Ha",
                             shots=10, seed=1, solve_options=options)

    def test_live_mode_dependent_option_failure_retains_sampling_receipt(self):
        p = SampledProjectedProvider(sampled_reference(ket_density(2, "00")), self.h, self.gens)
        valid = run_workflow(p, path="sampled_projection", problem="x", unit="Ha",
                             shots=100, seed=2,
                             solve_options={"max_condition": float("inf"),
                                            "overlap_ridge": np.array([0., 0.])}).as_dict()
        self.assertEqual(valid["outcome"]["status"], "sampled_projected_solve_completed")
        doc = run_workflow(p, path="sampled_projection", problem="x", unit="Ha",
                           shots=100, seed=2,
                           solve_options={"max_condition": float("inf"),
                                          "overlap_ridge": [0.]}).as_dict()
        self.assertEqual(doc["outcome"]["status"], "sampled_projection_failed")
        self.assertIn("per live generator", doc["outcome"]["method_details"]["error"]["message"])
        self.assertEqual(doc["cost_ledger"]["measurement"]["shots"], 100 * len(p.plan.settings))
        self.assertTrue(doc["outcome"]["method_details"]["groups"])

    def test_h4_exact_and_sector_paths_preserve_different_scopes(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("sector workflow requires SciPy")
        model = fcidump_model(ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP")
        gens = [("I", MV.scalar(model.n, 1))]
        exact = ExactProjectedProvider(DeterminantReference("11110000"), model.hamiltonian, gens)
        a = run_workflow(exact, path="exact_projection", problem="H4", unit="Ha").as_dict()
        sampled = SampledProjectedProvider(sampled_reference(ket_density(8, "11110000")),
                                           model.hamiltonian, gens)
        c = run_workflow(sampled, path="sampled_projection", problem="H4", unit="Ha",
                         shots=2000, seed=17).as_dict()
        self.assertLess(abs(a["outcome"]["energies"][0] - c["outcome"]["energies"][0]), 0.03)
        self.assertEqual(c["problem"]["hamiltonian_identity"], a["problem"]["hamiltonian_identity"])
        spec = SectorSpec(8, 4, 0.)
        backend = SectorStatevectorBackend(8, 4, 0.)
        ref = SectorStateReference(spec, backend.state_from_program(model.reference))
        p = SectorActionProvider(ref, model.hamiltonian)
        b = run_workflow(p, path="sector_action", problem="H4", unit="Ha",
                         sector_config=SectorSolveConfig(ncv=20, recovery_attempts=0)).as_dict()
        # The one-column determinant projection and sector recovery solve different spans.
        self.assertGreaterEqual(a["outcome"]["energies"][0], b["outcome"]["energies"][0] - 1e-10)
        self.assertEqual(a["problem"]["hamiltonian_identity"], b["problem"]["hamiltonian_identity"])
        self.assertEqual(b["outcome"]["residual"]["status"], "reported")
        self.assertEqual(b["outcome"]["ground_state_status"], "not_established")
        self.assertGreater(len(b["outcome"]["method_details"]["attempts"]), 0)

    def test_caller_costs_are_validated_and_do_not_create_a_total(self):
        result = run_workflow(self.exact, path="exact_projection", problem="x", unit="Ha",
                              preparation_cost={"wall_seconds": 1.25, "scope": "external setup"})
        ledger = result.as_dict()["cost_ledger"]
        self.assertEqual(ledger["stages"]["preparation"]["status"], "caller_reported")
        self.assertEqual(ledger["stages"]["preparation"]["wall_seconds"], 1.25)
        self.assertIsNone(ledger["end_to_end_total"])
        with self.assertRaises(ValueError):
            run_workflow(self.exact, path="exact_projection", problem="x", unit="Ha",
                         preparation_cost={"wall_seconds": -1, "scope": "external"})

    def test_archive_roundtrip_tamper_and_no_overwrite(self):
        result = run_workflow(self.exact, path="exact_projection", problem="test", unit="Ha")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_bytes(b"the exact input bytes\n")
            archive = root / "evidence.zip"
            summary = export_workflow_evidence(result, archive, attachments={"input.txt": source})
            manifest = verify_workflow_evidence(archive)
            self.assertEqual(summary["size_bytes"], archive.stat().st_size)
            self.assertIn("attachments/input.txt", manifest["files"])
            with self.assertRaises(FileExistsError):
                export_workflow_evidence(result, archive)
            self.assertEqual(verify_workflow_evidence(archive), manifest)
            with self.assertRaises(ValueError):
                export_workflow_evidence(result, root / "bad.zip", attachments={"../file": source})
            with zipfile.ZipFile(archive) as z:
                payloads = {name: z.read(name) for name in z.namelist()}
            payloads["result.json"] += b" "
            corrupt = root / "corrupt.zip"
            with zipfile.ZipFile(corrupt, "w") as z:
                for name, payload in payloads.items():
                    z.writestr(name, payload)
            with self.assertRaisesRegex(ValueError, "digest or size"):
                verify_workflow_evidence(corrupt)

    def test_archive_checks_remaining_budget_before_opening_next_attachment(self):
        result = run_workflow(self.exact, path="exact_projection", problem="x", unit="Ha")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.write_bytes(b"123")
            second.write_bytes(b"456")
            opened = []
            original_open = Path.open

            def tracked_open(path, *args, **kwargs):
                opened.append(path)
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", tracked_open):
                with self.assertRaisesRegex(ValueError, "remaining export byte budget"):
                    export_workflow_evidence(result, root / "over.zip",
                        attachments={"first": first, "second": second},
                        max_bytes=len(result._payload) + 4)
            self.assertEqual(opened, [first])
            self.assertFalse((root / "over.zip").exists())

    def test_archive_bounds_read_if_attachment_grows_after_stat(self):
        result = run_workflow(self.exact, path="exact_projection", problem="x", unit="Ha")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "growing"
            source.write_bytes(b"1")
            reads = []
            original_open = Path.open

            class GrowingFile(io.BytesIO):
                def read(self, size=-1):
                    reads.append(size)
                    return super().read(size)

            def open_grown(path, *args, **kwargs):
                if path == source:
                    return GrowingFile(b"1234567890")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", open_grown):
                with self.assertRaisesRegex(ValueError, "remaining export byte budget"):
                    export_workflow_evidence(result, root / "over.zip",
                        attachments={"growing": source}, max_bytes=len(result._payload) + 4)
            self.assertEqual(reads, [5])
            self.assertFalse((root / "over.zip").exists())


if __name__ == "__main__":
    unittest.main()
