"""Finite-shot second moments and explicitly budgeted real-time generators."""
import unittest
from unittest import mock

import numpy as np

from benchmarks.run_sampled_ratio import JointShotSimulator
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models import tfim
from clifford_qc.multivector import MV
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace import (
    MatrixElementBank, MeasuredMomentProblem, SampledSecondMomentProvider,
    SecondMomentBank, SecondMomentStoragePolicy, truncated_time_family,
    validate_truncated_time_family,
)


def approximation_options(**changes):
    result = dict(degree=18, max_terms=4096, max_product_pairs=2000000,
                  norm_error_budget=1e-6, infidelity_budget=1e-10,
                  energy_error_budget=1e-5, max_condition=1e8, tau_s=1e-8,
                  minimum_retained_rank=2)
    result.update(changes)
    return result


def moment_problem(s, h, k, rs=0., rh=0., rk=0.):
    matrices = [np.atleast_2d(x).astype(complex) for x in (s, h, k)]
    radii = [np.full(matrices[0].shape, r) for r in (rs, rh, rk)]
    return MeasuredMomentProblem(*matrices, *radii,
                                tuple(str(i) for i in range(len(matrices[0]))),
                                .05, {}, 'synthetic-region')


def sampled_fixture(n=2, complex_generators=False):
    h = tfim(n, h=1.2).hamiltonian.to_mv()
    state = np.ones(2**n, complex)/np.sqrt(2**n)
    if complex_generators:
        state = np.array([1., 2j, -.2+.4j, .6j]); state /= np.linalg.norm(state)
        generators = [('I', MV.scalar(2, 1)), ('A', MV.from_terms(2, {'XI': .4j, 'ZY': .7}))]
    else:
        generators = [('I', MV.scalar(n, 1)), ('ZZ', MV.from_label('ZZ'+'I'*(n-2)))]
    reference = PreparedSampledReference(n, 'unit-test-preparation', JointShotSimulator(state, n).draw)
    provider = SampledSecondMomentProvider(reference, h, generators, max_words=256,
               max_settings=128, storage_policy=SecondMomentStoragePolicy(1000000))
    return h, state, generators, reference, provider


class Phase1516CompletionTests(unittest.TestCase):
    def test_sampled_second_moment_matches_dense_complex_gram(self):
        h, state, generators, reference, provider = sampled_fixture(complex_generators=True)
        observations = reference.sample(provider.plan.settings, 200000, seed=8321)
        problem = provider.reconstruct(observations)
        basis = np.column_stack([to_matrix(g) @ state for _, g in generators])
        dense_h = to_matrix(h)
        for estimated, radius, exact in zip(
                (problem.overlap, problem.hamiltonian, problem.second_moment),
                (problem.overlap_radii, problem.hamiltonian_radii, problem.second_moment_radii),
                (basis.conj().T @ basis, basis.conj().T @ dense_h @ basis,
                 (dense_h @ basis).conj().T @ (dense_h @ basis))):
            self.assertTrue(np.all(np.abs(estimated-exact) <= radius+1e-12))
            np.testing.assert_allclose(estimated, estimated.conj().T)
        self.assertEqual(problem.resources['shots'], sum(observations.counts))
        self.assertEqual(problem.resources['circuits'], len(provider.plan.settings))

    def test_inference_never_queries_exact_reference_or_exact_block(self):
        _, _, _, reference, provider = sampled_fixture()
        observations = reference.sample(provider.plan.settings, 1000, seed=81)
        with mock.patch.object(provider.bank, 'matrices', side_effect=AssertionError('exact block')):
            with mock.patch.object(provider.moments, 'matrix', side_effect=AssertionError('exact K')):
                problem = provider.reconstruct(observations)
                self.assertEqual(problem.solve().effective_rank, 2)
        self.assertFalse(hasattr(reference, 'evaluate_terms'))

    def test_nondensity_exact_reference_gets_the_same_second_moments(self):
        h = tfim(2).hamiltonian.to_mv()
        reference = DeterminantReference('00')
        bank = MatrixElementBank(reference, h, [('I', MV.scalar(2, 1)), ('X', MV.from_label('XI'))])
        moments = SecondMomentBank(bank)
        basis = np.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]])
        dense_h = to_matrix(h)
        np.testing.assert_allclose(moments.matrix(), basis.T @ dense_h @ dense_h @ basis)
        self.assertTrue(moments.storage_preflight.matches_inputs(bank))

    def test_storage_word_and_setting_admission_fail_closed(self):
        h, _, generators, reference, _ = sampled_fixture()
        for kwargs in ({'max_words': 1, 'max_settings': 128, 'storage_policy': SecondMomentStoragePolicy(1000000)},
                       {'max_words': 256, 'max_settings': 1, 'storage_policy': SecondMomentStoragePolicy(1000000)},
                       {'max_words': 256, 'max_settings': 128, 'storage_policy': SecondMomentStoragePolicy(1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SampledSecondMomentProvider(reference, h, generators, **kwargs)

    def test_observation_and_plan_mutations_are_rejected(self):
        from dataclasses import replace
        _, _, _, reference, provider = sampled_fixture()
        observation = reference.sample(provider.plan.settings, 100, seed=8)
        for invalid in (replace(observation, reference_identity='other'),
                        replace(observation, settings_identity='other'),
                        replace(observation, counts=(0,)*len(observation.counts))):
            with self.assertRaises(ValueError):
                provider.reconstruct(invalid)
        provider.bank.add(('new', MV.from_label('XI')))
        with self.assertRaises(ValueError):
            provider.reconstruct(observation)

    def test_problem_arrays_are_immutable(self):
        problem = moment_problem(1, .2, 1)
        with self.assertRaises(ValueError):
            problem.second_moment.flags.writeable = True
        with self.assertRaises(TypeError):
            problem.resources['shots'] = 1

    def test_unresolved_normalization_never_certifies(self):
        result = moment_problem(.1, 0, .1, rs=.2).assess([1], residual_tolerance=10)
        self.assertEqual(result['status'], 'normalization_unresolved')
        self.assertIsNone(result['residual_interval'])

    def test_negative_point_variance_is_not_clipped_into_convergence(self):
        result = moment_problem(1, .8, .5, rk=.3).assess([1], residual_tolerance=.01)
        self.assertLess(result['variance'], 0)
        self.assertEqual(result['status'], 'unresolved')
        self.assertGreater(result['residual_interval'][1], .3)

    def test_inconsistent_moments_do_not_return_a_residual_certificate(self):
        result = moment_problem(1, 1, .5, rk=.01).assess([1], residual_tolerance=10)
        self.assertEqual(result['status'], 'inconsistent_moments')
        self.assertIsNone(result['residual_interval'])

    def test_nonzero_residual_is_detected(self):
        result = moment_problem(1, 0, 1).assess([1], residual_tolerance=.1)
        self.assertEqual(result['status'], 'above_tolerance')
        self.assertAlmostEqual(result['residual_interval'][0], 1., places=10)
        self.assertAlmostEqual(result['residual_interval'][1], 1., places=10)

    def test_an_excited_eigenstate_is_not_a_ground_state_certificate(self):
        result = moment_problem(1, 1, 1).assess([1], residual_tolerance=1e-5)
        self.assertEqual(result['status'], 'within_tolerance')
        self.assertFalse(result['ground_state_certificate'])

    def test_direction_rescaling_leaves_normalization_gate_and_intervals_fixed(self):
        p = moment_problem(np.eye(2), np.diag([0., 1.]), np.eye(2), rs=.001, rh=.002, rk=.004)
        a = p.assess([1, 1j], residual_tolerance=.1)
        b = p.assess(np.array([1, 1j])*1e100, residual_tolerance=.1)
        for key in ('status', 'normalization_interval', 'energy_interval', 'variance_interval', 'residual_interval'):
            self.assertEqual(a[key], b[key])

    def test_simultaneous_region_covers_data_selected_complex_directions(self):
        rng = np.random.default_rng(19)
        s = np.array([[1., .1j], [-.1j, 1.]])
        h = np.array([[.2, .3+.1j], [.3-.1j, -.6]])
        # A PSD second moment with strictly positive physical variance margin.
        k = np.eye(2)*2
        p = moment_problem(s, h, k, rs=.001, rh=.002, rk=.003)
        for _ in range(40):
            c = rng.normal(size=2)+1j*rng.normal(size=2)
            n = np.vdot(c, s @ c).real
            energy = np.vdot(c, h @ c).real/n
            variance = np.vdot(c, k @ c).real/n-energy**2
            report = p.assess(c, residual_tolerance=.1)
            self.assertLessEqual(report['energy_interval'][0], energy)
            self.assertGreaterEqual(report['energy_interval'][1], energy)
            self.assertLessEqual(report['variance_interval'][0], variance)
            self.assertGreaterEqual(report['variance_interval'][1], variance)

    def test_folded_roots_report_physical_inclusion_not_ordering(self):
        p = moment_problem(np.eye(2), np.diag([-1., 2.]), np.diag([1., 4.]))
        result = p.folded_roots(-.9, count=1, residual_tolerance=.1)
        self.assertFalse(result['ordered_root_certificate'])
        root = result['roots'][0]
        self.assertAlmostEqual(root['noisy_folded_eigenvalue'], .01)
        interval = root['assessment']['target_eigenvalue_interval']
        self.assertLessEqual(interval[0], -1.+1e-12)
        self.assertGreaterEqual(interval[1], -1.-1e-12)
        with self.assertRaises(ValueError):
            p.folded_roots(0., residual_tolerance=1., overlap_ridge=.1)

    def test_measured_direction_matches_dense_true_variance(self):
        h, state, gens, reference, provider = sampled_fixture(complex_generators=True)
        problem = provider.reconstruct(reference.sample(provider.plan.settings, 100000, seed=183))
        solved = problem.solve()
        c = solved.ritz_vector()
        v = sum(z*to_matrix(g) @ state for z, (_, g) in zip(c, gens)); v /= np.linalg.norm(v)
        dense_h = to_matrix(h)
        energy = np.vdot(v, dense_h @ v).real
        variance = np.linalg.norm((dense_h-energy*np.eye(4)) @ v)**2
        report = problem.assess(c, residual_tolerance=.1)
        self.assertLessEqual(report['variance_interval'][0], variance)
        self.assertGreaterEqual(report['variance_interval'][1], variance)

    def test_more_confidence_means_wider_entry_regions(self):
        _, _, _, reference, provider = sampled_fixture()
        observations = reference.sample(provider.plan.settings, 1000, seed=21)
        narrow, wide = provider.reconstruct(observations, delta=.1), provider.reconstruct(observations, delta=.001)
        self.assertTrue(np.all(wide.second_moment_radii >= narrow.second_moment_radii))

    def test_truncated_family_matches_independent_dense_exponential(self):
        from scipy.linalg import expm
        h = MV.from_terms(2, {'II': .7, 'XI': .4, 'YZ': .3, 'ZZ': -.2})
        family = truncated_time_family(h, [-.2, 0., .2], **approximation_options())
        dense = to_matrix(h)
        for g, t, bound in zip(family.generators, family.times, family.bounds):
            error = np.linalg.norm(to_matrix(g.mv)-expm(-1j*t*dense), 2)
            self.assertLess(error, bound['norm_error_bound']+1e-10)
        np.testing.assert_allclose(to_matrix(family.generators[0].mv),
                                   to_matrix(family.generators[-1].mv).conj().T, atol=1e-12)
        np.testing.assert_allclose(to_matrix(family.generators[1].mv), np.eye(4), atol=0)

    def test_truncated_validation_has_explicit_norm_fidelity_energy_and_rank(self):
        h = tfim(3, h=1.2).hamiltonian
        family = truncated_time_family(h, [0., .1, .2, .3], **approximation_options())
        result = validate_truncated_time_family(family, np.ones(8, complex), max_state_amplitudes=8)
        self.assertGreaterEqual(result['conditioning']['retained_rank'], 2)
        for r in result['actions']:
            self.assertLess(r['action_error'], 1e-6)
            self.assertLess(r['energy_error'], 1e-5)
            self.assertGreater(r['fidelity'], 1-1e-10)

    def test_truncated_operator_constructor_never_allocates_a_density(self):
        h = MV.from_label('Z'+'I'*127)
        with mock.patch('clifford_qc.dense_reference.to_matrix', side_effect=AssertionError('dense')):
            family = truncated_time_family(h, [0., .01], **approximation_options())
        self.assertLessEqual(max(g.mv.nnz() for g in family.generators), 2)
        with self.assertRaises(ValueError):
            validate_truncated_time_family(family, [1], max_state_amplitudes=16)

    def test_truncation_and_work_budgets_reject_unadmitted_families(self):
        h = tfim(3).hamiltonian
        for options in (approximation_options(degree=1), approximation_options(max_terms=1),
                        approximation_options(max_product_pairs=1),
                        approximation_options(norm_error_budget=1e-14)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                truncated_time_family(h, [0., .3], **options)

    def test_time_family_rejects_ill_conditioned_or_collapsed_overlap(self):
        family = truncated_time_family(tfim(2).hamiltonian, [0., .1], **approximation_options(max_condition=100.))
        with self.assertRaises(ValueError):
            family.admit_overlap(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            family.admit_overlap([[1., .99999], [.99999, 1.]])

    def test_time_family_input_ownership_and_invalid_parameters(self):
        h = tfim(2).hamiltonian.to_mv()
        family = truncated_time_family(h, [0., .1], **approximation_options())
        h.terms.clear()
        self.assertTrue(family.hamiltonian.terms)
        with self.assertRaises(TypeError):
            family.generators[0].mv.terms[0] = 2
        for times in ([0., 0.], [float('nan')], [True, .1], []):
            with self.assertRaises(ValueError):
                truncated_time_family(tfim(2).hamiltonian, times, **approximation_options())
        for changes in ({'degree': True}, {'max_terms': 0}, {'max_condition': float('inf')}, {'infidelity_budget': -1}):
            with self.assertRaises(ValueError):
                truncated_time_family(tfim(2).hamiltonian, [0., .1], **approximation_options(**changes))


    def test_independent_histogram_and_radius_reconstruction(self):
        import benchmarks.check_phase15_16_completion as independent
        from benchmarks.run_phase15_16_completion import settings_record
        h, psi, gens, reference, provider = sampled_fixture()
        count = 20000
        obs = reference.sample(provider.plan.settings, count, seed=45)
        native = provider.reconstruct(obs)
        by_key = {g.setting_key: g for g in obs.batch.groups}
        row = {'case': 'tfim2', 'budget': count*len(provider.plan.settings), 'seed': 45,
               'counts': list(obs.counts), 'physical_shots': sum(obs.counts),
               'circuits': len(obs.counts),
               'histograms': [dict(by_key[s.key].hist) for s in provider.plan.settings]}
        operators = independent.entries(to_matrix(h), [to_matrix(g) for _, g in gens])
        matrices, radii = independent.reconstruct(row, settings_record(provider.plan.settings),
                                                   operators, psi, 2, .05)
        for a, b in zip(matrices, (native.overlap, native.hamiltonian, native.second_moment)):
            np.testing.assert_allclose(a, b, atol=1e-12)
        for a, b in zip(radii, (native.overlap_radii, native.hamiltonian_radii, native.second_moment_radii)):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_independent_histogram_checker_rejects_corrupt_counts_and_readouts(self):
        import copy
        import benchmarks.check_phase15_16_completion as independent
        from benchmarks.run_phase15_16_completion import settings_record
        h, psi, gens, reference, provider = sampled_fixture()
        obs = reference.sample(provider.plan.settings, 100, seed=4)
        groups = {g.setting_key: g for g in obs.batch.groups}
        row = {'case': 'tfim2', 'budget': sum(obs.counts), 'seed': 4,
               'counts': list(obs.counts), 'physical_shots': sum(obs.counts),
               'circuits': len(obs.counts),
               'histograms': [dict(groups[s.key].hist) for s in provider.plan.settings]}
        settings = settings_record(provider.plan.settings)
        operators = independent.entries(to_matrix(h), [to_matrix(g) for _, g in gens])
        broken = copy.deepcopy(row)
        bits = next(iter(broken['histograms'][0]))
        broken['histograms'][0][bits] += 1
        with self.assertRaises(ValueError):
            independent.reconstruct(broken, settings, operators, psi, 2, .05)
        broken_settings = copy.deepcopy(settings)
        key = next(iter(broken_settings[0]['readouts']))
        broken_settings[0]['readouts'][key][0] *= -1
        with self.assertRaises(ValueError):
            independent.reconstruct(row, broken_settings, operators, psi, 2, .05)

    def test_independent_interval_arithmetic_matches_the_physical_direction_report(self):
        import benchmarks.check_phase15_16_completion as independent
        p = moment_problem(np.eye(2), np.diag([-1., .7]), np.diag([1., .49]), rs=.01, rh=.02, rk=.03)
        c = np.array([1., 1j])
        config = {'normalization_floor': 1e-6, 'residual_tolerance': .1}
        expected = independent.expected_report([p.overlap, p.hamiltonian, p.second_moment],
                   [p.overlap_radii, p.hamiltonian_radii, p.second_moment_radii], c.copy(), -.9, config)
        actual = p.assess(c, residual_tolerance=.1, target=-.9)
        for key, value in expected.items():
            problems = []
            independent.close(actual[key], value, key, problems)
            self.assertEqual(problems, [])

    def test_catastrophic_cancellation_cannot_manufacture_a_zero_residual(self):
        report = moment_problem(1, 1e12, 1e24).assess([1], residual_tolerance=.01, target=1e12)
        self.assertEqual(report['variance'], 0.)
        self.assertEqual(report['status'], 'unresolved')
        self.assertGreater(report['variance_numerical_allowance'], 1e10)
        self.assertGreater(report['residual_interval'][1], 1e5)
        self.assertGreater(report['folded_numerical_allowance'], 1e10)

    def test_extreme_finite_direction_scales_stay_finite(self):
        p = moment_problem(np.eye(2), np.diag([-1., 1.]), np.eye(2), rk=.001)
        normal = p.assess([1+1j, 1], residual_tolerance=.1)
        for scale in (1e308, 1e-320):
            report = p.assess(np.array([1+1j, 1])*scale, residual_tolerance=.1)
            self.assertTrue(np.isfinite(report['variance']))
            np.testing.assert_allclose(report['variance_interval'], normal['variance_interval'], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
