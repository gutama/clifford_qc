"""Numerical and lifetime regressions for the storage/workflow refactoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.models.spin import tfim
from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.subspace import (
    OverlapTarget, pauli_orbit, commutator_response, identity_generator, run_acase,
)
from clifford_qc.subspace.projection import MatrixElementBank
from clifford_qc.subspace.packed import PackedRowStore, SegmentedPackedRowStore
from clifford_qc.subspace.streaming import StreamingMatrixElementBank
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.measurement.functionals import PackedCoefficients
from clifford_qc.prepared import PreparedProblem, prepare_fcidump
from clifford_qc.pipeline import main as pipeline_main


def fixture(n=4):
    model = tfim(n)
    rho = ExactMVBackend().state(model.reference, ())
    words = [op.word for op in odd_y_filter(local_pool(n))]
    pool = pauli_orbit(words) + commutator_response(model.hamiltonian, words)
    return model, rho, pool


def scientific(result):
    return (result.labels, result.energy_history, result.stopped_reason,
            [(r.selected_label, r.predicted_lowering, r.orthogonal_fraction,
              r.new_words, r.word_universe, r.selection_word_universe,
              r.rejected_conditioning, r.rejected_sector) for r in result.records])


class TestStorageRefactoring(unittest.TestCase):
    def test_target_overlap_selection_survives_eviction(self):
        m, rho, pool = fixture()
        target = OverlapTarget.from_coefficients([pool[1]], [1.0])
        options = dict(criterion='target_overlap', target=target, max_size=3)
        expected = run_acase(rho, m.hamiltonian, pool, **options)
        bank = StreamingMatrixElementBank(rho, m.hamiltonian, storage='packed', frontier_pairs=1)
        actual = run_acase(rho, m.hamiltonian, pool, bank=bank, **options)
        self.assertEqual(scientific(actual), scientific(expected))
        self.assertEqual([r.target_overlap_gain for r in actual.records],
                         [r.target_overlap_gain for r in expected.records])
        self.assertEqual(actual.resources['retained_block_pairs'],
                         actual.resources['complete_block_pairs'])

    def test_final_resources_include_unsuccessful_scoring_pass(self):
        m, rho, pool = fixture()
        for cls in (MatrixElementBank, StreamingMatrixElementBank):
            b = cls(rho, m.hamiltonian)
            result = run_acase(rho, m.hamiltonian, pool, bank=b,
                               max_size=2, min_lowering=1e6)
            self.assertEqual(len(result.records), 0)
            current = b.resources(result.indices)
            for field in ('pairs_built', 'coefficient_occurrences', 'evicted_rows'):
                self.assertEqual(result.resources[field], current[field])
            self.assertGreater(current['pairs_built'], 1)

    def test_packed_functionals_preserve_sampled_matrices_and_covariance(self):
        m, rho, pool = fixture()
        gens = [identity_generator(m.n), *pool[:3]]
        for pooling in ('assigned', 'shots'):
            sessions = [SharedMeasurement(
                StreamingMatrixElementBank(rho, m.hamiltonian, gens,
                                           storage='packed', frontier_pairs=1),
                pooling=pooling, coefficient_storage=storage)
                for storage in ('object', 'packed')]
            caches = [s.measure(FiniteShotBackend(seed=17), 100) for s in sessions]
            self.assertEqual(caches[0].group_states(), caches[1].group_states())
            for a, b in zip(sessions[0].matrices(caches[0]), sessions[1].matrices(caches[1])):
                np.testing.assert_array_equal(a, b)
            left, right = [s.matrix_functionals() for s in sessions]
            for a, b in zip(left, right):
                self.assertEqual(a.estimate(caches[0]), b.estimate(caches[1]))
                self.assertEqual(a.radius(caches[0], .05), b.radius(caches[1], .05))
                for c, d in zip(left, right):
                    self.assertEqual(a.covariance(c, caches[0]), b.covariance(d, caches[1]))

    def test_measurement_never_materializes_packed_rows(self):
        m, rho, pool = fixture()
        b = MatrixElementBank(rho, m.hamiltonian, [identity_generator(m.n), *pool[:3]], storage="packed")
        b.matrices()
        with patch.object(PackedRowStore, "materialize", side_effect=AssertionError("allocation")):
            measured = b.measured_storage_bytes()
        self.assertEqual(measured['packed_operator_bytes'], 24 * measured['coefficient_occurrences'])
        self.assertGreaterEqual(measured['reserved_operator_bytes'], measured['measured_operator_bytes'])

    def test_mixed_observable_bytes_are_counted(self):
        m, rho, _ = fixture(2)
        b = MatrixElementBank(rho, m.hamiltonian, [identity_generator(2)], storage="packed")
        b.matrices()
        before = b.measured_storage_bytes()
        b.project_observable(m.hamiltonian, label="Hcopy")
        after = b.measured_storage_bytes()
        self.assertGreater(after['measured_operator_bytes'], before['measured_operator_bytes'])
        self.assertGreater(after['reserved_operator_bytes'], before['reserved_operator_bytes'])
        self.assertEqual(after['resident_operator_rows'], 3)

    def test_support_reads_preserve_orientation_without_materializing(self):
        m, rho, pool = fixture()
        b = MatrixElementBank(rho, m.hamiltonian, [identity_generator(m.n), *pool[:3]], storage="packed")
        with patch.object(PackedRowStore, "materialize", side_effect=AssertionError("allocation")):
            self.assertEqual(b.pair_support(1, 3), b.pair_support(3, 1))
            b.resources()
        with self.assertRaises(IndexError): b.pair_support(-1, 0)
        with self.assertRaises(IndexError): b.pair_support(0, 100)

    def test_new_exact_pair_is_contracted_before_packing(self):
        m, rho, pool = fixture()
        b = MatrixElementBank(rho, m.hamiltonian, [identity_generator(m.n), *pool[:2]], storage="packed")
        control = MatrixElementBank(rho, m.hamiltonian, b._generators)
        with patch.object(PackedRowStore, "trace_pairing", side_effect=AssertionError("packed conversion")):
            self.assertEqual(b.entry(1, 2), control.entry(1, 2))
            self.assertEqual(b.entry(2, 1), control.entry(2, 1))

    def test_adaptive_variants_match_full_matrices_and_diagnostics(self):
        m, rho, pool = fixture()
        for gamma, roots in [(0.0, 1), (0.3, 1), (0.0, 2)]:
            reference = run_acase(rho, m.hamiltonian, pool, max_size=5, gamma=gamma, roots=roots)
            for storage in ("object", "packed"):
                for cls in (MatrixElementBank, StreamingMatrixElementBank):
                    kwargs = {'frontier_pairs': 1} if cls is StreamingMatrixElementBank else {}
                    bank = cls(rho, m.hamiltonian, storage=storage, **kwargs)
                    with patch.object(PackedRowStore, "materialize", side_effect=AssertionError("scoring conversion")):
                        result = run_acase(rho, m.hamiltonian, pool, bank=bank, max_size=5, gamma=gamma, roots=roots)
                    self.assertEqual(scientific(result), scientific(reference))
                    for actual, expected in zip(bank.matrices(result.indices), reference.bank.matrices(reference.indices)):
                        np.testing.assert_array_equal(actual, expected)
                    if cls is StreamingMatrixElementBank:
                        resources = bank.resources(result.indices)
                        self.assertGreater(resources['evicted_rows'], 0)
                        self.assertLessEqual(resources['resident_sh_rows'], resources['resident_sh_row_limit'])
                        self.assertLess(resources['coefficient_occurrences'], resources['cumulative_coefficient_occurrences'])

    def test_evicted_rows_rebuild_and_packed_capacity_is_released(self):
        m, rho, pool = fixture()
        b = StreamingMatrixElementBank(rho, m.hamiltonian, [identity_generator(m.n), *pool[:5]],
                                       storage="packed", frontier_pairs=1)
        self.assertIsInstance(b._packed['element'], SegmentedPackedRowStore)
        expected = b.element_operator(0, 1)
        for j in range(2, 6): b.entry(0, j)
        self.assertFalse(b._has_pair((0, 1)))
        actual = b.element_operator(0, 1)
        self.assertEqual(list(expected.terms.items()), list(actual.terms.items()))
        self.assertGreater(b.resources()['recomputed_rows'], 0)
        for store in b._packed.values():
            self.assertEqual(store.nbytes(), store.reserved_bytes())
            self.assertLessEqual(len(store), 1)

    def test_structural_session_does_not_evaluate_oracle(self):
        m, rho, pool = fixture()
        gens = [identity_generator(m.n), *pool[:3]]
        control = MatrixElementBank(rho, m.hamiltonian, gens)
        expected = control.matrices()
        for storage in ("object", "packed"):
            bank = StreamingMatrixElementBank(rho, m.hamiltonian, gens, storage=storage, frontier_pairs=1)
            with patch.object(bank, "entry", side_effect=AssertionError("oracle")), patch.object(
                    PackedRowStore, "materialize", side_effect=AssertionError("conversion")):
                session = SharedMeasurement(bank, coefficient_storage="packed")
            self.assertEqual(bank._entries, {})
            for actual, target in zip(session.exact_matrices(), expected):
                np.testing.assert_allclose(actual, target, atol=1e-13, rtol=0)
            # Snapshots survive subsequent eviction and do not alias a row buffer.
            for i in range(len(bank)): bank.entry(i, i)
            for actual, target in zip(session.exact_matrices(), expected):
                np.testing.assert_allclose(actual, target, atol=1e-13, rtol=0)

    def test_packed_coefficient_mapping_order_and_immutability(self):
        values = {12: 1.5, 2: -3.0, 7: 0.25}
        packed = PackedCoefficients(values)
        self.assertEqual(list(packed.items()), list(values.items()))
        self.assertEqual(dict(packed), values)
        self.assertEqual(packed.get(99, 0), 0)
        with self.assertRaises(ValueError): packed._values[0] = 4


class TestPreparedProblem(unittest.TestCase):
    def test_cli_stages_reuse_preparation_and_keep_validation_optional(self):
        source = Path(__file__).resolve().parents[1] / 'benchmarks/data/h4_sto3g_r0.9.FCIDUMP'
        with tempfile.TemporaryDirectory() as directory:
            pipeline_main(['prepare', str(source), '--cache-directory', directory])
            prepared = next(Path(directory).glob('*.json'))
            output = Path(directory) / 'solve.json'
            with patch('clifford_qc.backends.SectorStatevectorBackend',
                       side_effect=AssertionError('hidden reference solve')):
                pipeline_main(['solve', str(prepared), '--output', str(output),
                               '--policy', 'stream_recompute', '--storage', 'packed',
                               '--frontier-pairs', '1', '--max-additions', '1'])
            record = json.loads(output.read_text())
            self.assertFalse(record['stopping_uses_oracle'])
            self.assertEqual(record['resources']['storage_policy'], 'stream_recompute')
            validation = Path(directory) / 'validation.json'
            pipeline_main(['validate', str(prepared), '--output', str(validation),
                           '--method', 'dense'])
            checked = json.loads(validation.read_text())
            self.assertGreater(checked['wall_seconds'], 0)
            self.assertEqual(checked['problem_fingerprint'], record['problem_fingerprint'])
            self.assertLess(max(checked['residual_norms']), 1e-10)
            self.assertLessEqual(checked['energies'][0], record['energy'] + 1e-10)

    def test_roundtrip_preserves_order_and_avoids_mutable_aliases(self):
        m, _, _ = fixture()
        p = PreparedProblem.from_model(m, preparation={'source': 'test'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'problem.json'
            p.save(path)
            loaded = PreparedProblem.load(path)
            self.assertEqual(list(loaded.model().hamiltonian.to_mv().terms.items()),
                             list(m.hamiltonian.to_mv().terms.items()))
            first = loaded.model(); first.metadata['changed'] = True
            self.assertNotIn('changed', loaded.model().metadata)
            data = json.loads(path.read_text());data['problem']['name'] = 'tampered'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): PreparedProblem.load(path)

    def test_cache_key_invalidation_and_hit_skip_builder(self):
        source = Path(__file__).resolve().parents[1] / 'benchmarks/data/h4_sto3g_r0.9.FCIDUMP'
        with tempfile.TemporaryDirectory() as directory:
            first, path, hit = prepare_fcidump(source, directory)
            self.assertFalse(hit)
            with patch('clifford_qc.models.fcidump.fcidump_model', side_effect=AssertionError('rebuild')):
                second, same_path, hit = prepare_fcidump(source, directory)
            self.assertTrue(hit);self.assertEqual(path, same_path);self.assertEqual(first, second)
            _, different, hit = prepare_fcidump(source, directory, integral_tolerance=1e-10)
            self.assertFalse(hit);self.assertNotEqual(path, different)
            with self.assertRaises(ValueError):
                PreparedProblem.load(path, expected_preparation={'wrong': True})


if __name__ == '__main__':
    unittest.main()
