"""A-CASE Phase 4: the finite-shot layers.

The tests follow the plan's three stages and keep them apart, because the whole
point of the staging is that an asymptotic error bar must never be mistaken for
a certificate:

- *4A* the shared grouped measurement reconstructs ``(S, H)`` from one set of
  shots, and its infinite-shot limit reproduces the exact bank matrices;
- *4B* the delta-method interval is checked against a Monte-Carlo ground truth
  -- accurate when ``S`` is well conditioned, and typical-but-not-tail-safe
  when it is not, which is why it is labelled ``asymptotic``;
- *4C* the sample-split growth certificate accepts only candidates whose
  residual coupling is certifiably above threshold, and abstains otherwise.

One test exists to document a *failure*: the variational bound does not survive
noisy PSD repair. Q3 asks under what conditions it does, and pretending it
holds would be the easiest way to get the rest of the story wrong.
"""

import numpy as np
import pytest

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.matrix import exact_ground
from clifford_qc.measurement.confidence import simultaneous_z_radius
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (
    ASYMPTOTIC, FINITE_SAMPLE, HEURISTIC, Generator, MatrixElementBank,
    SharedMeasurement, WordFunctional, bootstrap_ritz, certify_couplings,
    commutator_response, coupling_functional, identity_generator,
    krylov_response, pauli_orbit, ritz_functional, ritz_uncertainty,
    run_certified_acase, score_candidate,
)

N = 4


@pytest.fixture(scope="module")
def setting():
    model = tfim(N, J=1.0, h=1.0)
    rho = ExactMVBackend().state(model.reference, ())
    words = [op.word for op in odd_y_filter(local_pool(N))]
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    return model, rho, words, E0


@pytest.fixture(scope="module")
def well_conditioned(setting):
    """Mutually orthogonal directions: kappa_S = 1, no conditioning pathology."""
    model, rho, words, E0 = setting
    bank = MatrixElementBank(rho, model.hamiltonian,
                             [identity_generator(N)] + pauli_orbit(words[:3]))
    return bank, SharedMeasurement(bank), bank.solve(), E0


@pytest.fixture(scope="module")
def ill_conditioned(setting):
    """Krylov powers mixed with words: a near-singular overlap matrix."""
    model, rho, words, E0 = setting
    bank = MatrixElementBank(rho, model.hamiltonian,
                             [identity_generator(N)] + pauli_orbit(words[:6])
                             + krylov_response(model.hamiltonian, 2))
    return bank, SharedMeasurement(bank), bank.solve(), E0


# ------------------------------------------- 4A: shared grouped measurement


def test_infinite_shot_limit_reproduces_the_exact_matrices(ill_conditioned):
    """The 4A acceptance: the measured reconstruction is the exact one, with
    ``mu_w`` replaced by estimates -- so with exact means it *is* the exact one."""
    bank, shared, _, _ = ill_conditioned
    S_exact, H_exact = shared.exact_matrices()
    S_bank, H_bank = bank.matrices()
    assert np.abs(S_exact - S_bank).max() < 1e-10
    assert np.abs(H_exact - H_bank).max() < 1e-10


def test_grouping_partitions_the_word_universe(ill_conditioned):
    bank, shared, _, _ = ill_conditioned
    grouped = [w.code for group in shared.groups for w in group]
    assert sorted(grouped) == sorted(w.code for w in shared.words)
    assert len(grouped) == len(set(grouped))  # a partition, not a cover
    # the identity is never measured: <I> = 1 is known
    assert all(w.code != 0 for w in shared.words)


def test_measured_matrices_converge_and_stay_structurally_hermitian(ill_conditioned):
    bank, shared, exact, _ = ill_conditioned
    S_exact, H_exact = shared.exact_matrices()
    errors = []
    for shots in (200, 20000):
        cache = shared.measure(FiniteShotBackend(seed=3), shots)
        S, Hm = shared.matrices(cache)
        assert np.array_equal(S, S.conj().T)  # structural, as in the exact path
        assert np.array_equal(Hm, Hm.conj().T)
        errors.append(max(np.abs(S - S_exact).max(), np.abs(Hm - H_exact).max()))
    assert errors[1] < errors[0] / 5.0  # ~1/sqrt(N) over a 100x budget increase


def test_shot_and_circuit_accounting(ill_conditioned):
    bank, shared, _, _ = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=1), 500)
    result = shared.solve(cache)
    assert result.resources["circuits"] == len(shared.groups)
    assert result.resources["shots"] == 500 * len(shared.groups)
    assert result.resources["measured_words"] == len(shared.words)
    assert result.resources["evidence"] == HEURISTIC  # energies from noisy matrices


def test_a_deterministic_functional_costs_no_shots_and_no_interval(well_conditioned):
    """``S_00`` for the identity generator is exactly 1: no words, no radius."""
    bank, shared, _, _ = well_conditioned
    identity_entry = shared.diagonal_functional(bank.index_of("I"))
    assert identity_entry.is_deterministic
    assert identity_entry.constant == pytest.approx(1.0)
    cache = shared.measure(FiniteShotBackend(seed=0), 100)
    assert identity_entry.estimate(cache) == pytest.approx(1.0, abs=1e-15)
    assert identity_entry.radius(cache, 0.05, 1, bound="eb") == 0.0
    assert identity_entry.variance(cache) == 0.0


def test_word_functional_arithmetic_is_linear(ill_conditioned):
    bank, shared, result, _ = ill_conditioned
    rho = bank.reference
    q = ritz_functional(bank, result.indices, result.ritz_vector(0),
                        result.ground_energy)
    other = shared.diagonal_functional(result.indices[1])
    assert (q + other).exact(rho) == pytest.approx(q.exact(rho) + other.exact(rho),
                                                   abs=1e-9)
    assert (q - q).exact(rho) == pytest.approx(0.0, abs=1e-9)


def test_ritz_functional_is_the_energy_jacobian(ill_conditioned):
    """``sum_w q_w mu_w`` must return the Ritz value itself: ``c'(H - E S)c = 0``
    means the functional evaluates to zero at the exact means, and the same
    coefficients against ``H`` alone give the energy."""
    bank, shared, result, _ = ill_conditioned
    rho = bank.reference
    q = ritz_functional(bank, result.indices, result.ritz_vector(0),
                        result.ground_energy)
    # B'(H - E)B has zero expectation at the solved Ritz pair
    assert q.exact(rho) == pytest.approx(0.0, abs=1e-9)
    # and it is the derivative: rebuild with E = 0 to recover <B' H B> = E
    q0 = ritz_functional(bank, result.indices, result.ritz_vector(0), 0.0)
    assert q0.exact(rho) == pytest.approx(result.ground_energy, abs=1e-9)


# ---------------------------------------------------- 4B: asymptotic uncertainty


def test_variance_agrees_with_the_radius_path(ill_conditioned):
    """Two routes to the same variance: the covariance-vector product and the
    per-group terms the confidence machinery consumes."""
    bank, shared, result, _ = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=4), 4000)
    q = ritz_functional(bank, result.indices, result.ritz_vector(0),
                        result.ground_energy)
    variance = q.variance(cache)
    radius = q.radius(cache, 0.05, 1, bound="normal")
    assert radius == pytest.approx(simultaneous_z_radius(variance, 0.05, 1), rel=1e-9)


def test_covariance_is_bilinear(ill_conditioned):
    """``Var(f+g) = Var f + Var g + 2 Cov(f,g)`` -- the identity a double-counted
    word breaks, and the reason each word is assigned to exactly one group."""
    bank, shared, result, _ = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=5), 4000)
    f = ritz_functional(bank, result.indices, result.ritz_vector(0),
                        result.ground_energy)
    g = shared.diagonal_functional(result.indices[-1])
    assert (f + g).variance(cache) == pytest.approx(
        f.variance(cache) + g.variance(cache) + 2.0 * f.covariance(g, cache),
        rel=1e-9, abs=1e-18)


def test_single_word_variance_uses_one_group_only(ill_conditioned):
    """A word several groups could read must be counted once.

    Its variance is then ``(1 - mu^2)/N`` up to sampling noise; counting it once
    per capable group would inflate that by an integer factor.
    """
    bank, shared, _, _ = ill_conditioned
    cache = shared.measure(FiniteShotBackend(seed=6), 8000)
    code = shared.words[0].code
    functional = WordFunctional({code: 1.0})
    mean = functional.estimate(cache)
    shots = cache.shots(code)
    assert functional.variance(cache) == pytest.approx((1.0 - mean ** 2) / shots,
                                                       rel=0.05)


def test_delta_method_matches_a_monte_carlo_ground_truth(well_conditioned):
    """4B against the truth: with ``kappa_S = 1`` the linearization is accurate.

    The predicted sigma is compared with the actual spread of the measured Ritz
    value over independent measurement seeds -- the only check that can tell a
    correct variance from a plausible-looking one. (It caught a real bug: a word
    that several QWC groups are able to read was being counted once per capable
    group, inflating every variance by that multiplicity.)
    """
    bank, shared, exact, _ = well_conditioned
    energies, sigmas = [], []
    for seed in range(16):
        cache = shared.measure(FiniteShotBackend(seed=200 + seed), 2000)
        result = shared.solve(cache)
        energies.append(result.ground_energy)
        sigmas.append(ritz_uncertainty(shared, cache, result).std_error)
    empirical = float(np.std(energies, ddof=1))
    predicted = float(np.mean(sigmas))
    assert abs(predicted - empirical) <= 0.4 * empirical


def test_intervals_are_typically_in_the_right_ballpark_when_ill_conditioned(
        ill_conditioned):
    """The interval is usable at kappa_S ~ 200, and only usable.

    Typical behaviour is fine -- the median error sits inside a couple of
    estimated sigmas. What the interval does *not* describe is the tail: at
    2000 shots per group roughly one run in forty admits a near-null overlap
    mode and lands whole Hartrees away (``examples/acase_finite_shot.py``
    measures it: MC std 1.85 Ha against a median sigma of 6.6e-3). A mean-and-
    variance description of the error is therefore inadequate here, which is
    the strongest reason in this code base for conditioning-aware growth and
    for never labelling these intervals certified. Asserted here only in the
    robust direction, since a rare event makes a flaky test.
    """
    bank, shared, exact, _ = ill_conditioned
    assert exact.condition_number > 50.0
    errors, sigmas = [], []
    for seed in range(16):
        cache = shared.measure(FiniteShotBackend(seed=400 + seed), 2000)
        result = shared.solve(cache)
        errors.append(abs(result.ground_energy - exact.ground_energy))
        sigmas.append(ritz_uncertainty(shared, cache, result).std_error)
    assert float(np.median(errors)) < 3.0 * float(np.median(sigmas))


def test_uncertainty_labels_never_claim_certification(well_conditioned):
    bank, shared, _, _ = well_conditioned
    cache = shared.measure(FiniteShotBackend(seed=8), 2000)
    result = shared.solve(cache)
    delta_method = ritz_uncertainty(shared, cache, result)
    resampled = bootstrap_ritz(shared, cache, replicates=30, seed=1)
    assert delta_method.evidence == ASYMPTOTIC and not delta_method.certified
    assert resampled.evidence == HEURISTIC and not resampled.certified
    for interval in (delta_method, resampled):
        assert interval.lower <= interval.estimate <= interval.upper


@pytest.mark.parametrize("kwargs, message", [
    ({"replicates": 1}, "at least two"),
    ({"delta": 0.0}, "delta"),
    ({"delta": 1.0}, "delta"),
])
def test_ritz_bootstrap_validates_its_statistical_contract(
        well_conditioned, kwargs, message):
    _, shared, _, _ = well_conditioned
    cache = shared.measure(FiniteShotBackend(seed=8), 100)
    with pytest.raises(ValueError, match=message):
        bootstrap_ritz(shared, cache, **kwargs)


def test_bootstrap_brackets_the_measured_energy(well_conditioned):
    """The resampling cross-check: it re-runs the whole nonlinear pipeline, so
    it sees what the linearization drops, and should agree in scale."""
    bank, shared, _, _ = well_conditioned
    cache = shared.measure(FiniteShotBackend(seed=9), 4000)
    result = shared.solve(cache)
    resampled = bootstrap_ritz(shared, cache, replicates=60, seed=2)
    delta_method = ritz_uncertainty(shared, cache, result)
    assert resampled.lower <= result.ground_energy <= resampled.upper
    ratio = resampled.std_error / delta_method.std_error
    assert 0.4 < ratio < 2.5


# --------------------------------------------------- the bound under noise (Q3)


def test_the_variational_bound_does_not_survive_noise(ill_conditioned):
    """Documented failure, not a bug: thresholding a noisy ``S`` is a PSD repair,
    and nothing establishes that it preserves ``E_sub >= E_0`` (Q3)."""
    bank, shared, _, E0 = ill_conditioned
    below = [shared.solve(shared.measure(FiniteShotBackend(seed=seed), 200)).ground_energy
             for seed in range(8)]
    assert min(below) < E0 - 1e-6
    assert any(shared.solve(shared.measure(FiniteShotBackend(seed=s), 200)
                            ).resources["overlap_negative_modes"] > 0
               for s in range(4))


# ------------------------------------------------- 4C: sample-split certificate


@pytest.fixture(scope="module")
def certified_run(setting):
    model, rho, words, E0 = setting
    candidates = (pauli_orbit(words[:4])
                  + commutator_response(model.hamiltonian, words[:2]))
    return run_certified_acase(rho, model.hamiltonian, candidates,
                               FiniteShotBackend(seed=17), max_size=2,
                               construction_shots=4000, certification_shots=4000,
                               delta=0.05, threshold=0.05, bound="eb",
                               exact_ground_energy=E0), E0


def test_certified_growth_accepts_only_certified_candidates(certified_run):
    result, E0 = certified_run
    assert result.labels[0] == "I"
    accepted = [record for record in result.records if record.selected_label]
    assert accepted
    for record in accepted:
        assert record.certified
        assert record.evidence == FINITE_SAMPLE
        assert record.coupling.lower > record.threshold
        assert record.coupling.lower <= record.coupling.estimate <= record.coupling.upper
        assert record.resolution in ("above_threshold", "best")


def test_sample_splitting_spends_two_independent_batches(certified_run):
    result, _ = certified_run
    for record in result.records:
        assert record.construction_shots > 0
        assert record.certification_shots > 0
    # every record's two batches are accounted for, plus the closing solve when
    # the budget (rather than an abstention) ended the run
    assert result.total_shots >= sum(r.construction_shots + r.certification_shots
                                     for r in result.records)
    assert result.total_circuits >= sum(r.circuits for r in result.records)
    assert result.resources["certification_scope"] == (
        "per_step_conditional_on_construction_batch")
    assert result.resources["variational_bound"] == "not_established_under_noise"


def test_energy_history_tracks_the_growth(certified_run):
    result, E0 = certified_run
    assert len(result.energy_history) == len(result.labels)
    assert result.energy == result.energy_history[-1]


def test_an_unreachable_threshold_abstains_immediately(setting):
    """Abstention is the designed outcome, not an error path."""
    model, rho, words, E0 = setting
    result = run_certified_acase(rho, model.hamiltonian, pauli_orbit(words[:3]),
                                 FiniteShotBackend(seed=19), max_size=3,
                                 construction_shots=500, certification_shots=500,
                                 delta=0.05, threshold=50.0, bound="eb")
    assert result.labels == ("I",)
    assert result.abstentions == 1
    assert result.stopped_reason.startswith("abstained")
    assert result.records[-1].abstained
    assert result.records[-1].coupling is None


def test_normal_bound_is_labelled_asymptotic_not_certified(setting):
    model, rho, words, E0 = setting
    result = run_certified_acase(rho, model.hamiltonian, pauli_orbit(words[:3]),
                                 FiniteShotBackend(seed=21), max_size=1,
                                 construction_shots=2000, certification_shots=2000,
                                 delta=0.05, threshold=0.05, bound="normal")
    record = result.records[0]
    assert record.evidence == ASYMPTOTIC
    assert not record.certified  # resolved, but not a finite-sample certificate
    assert record.selected_label is not None


def test_certified_coupling_converges_to_the_exact_one(setting):
    """4C's statistic is Phase 3's residual coupling, estimated instead of computed."""
    model, rho, words, E0 = setting
    bank = MatrixElementBank(rho, model.hamiltonian, [identity_generator(N)])
    basis = [bank.index_of("I")]
    candidates = bank.extend(pauli_orbit(words[:3]))
    exact = bank.solve(basis)
    shared = SharedMeasurement(bank, tuple(basis) + tuple(candidates))
    cache = shared.measure(FiniteShotBackend(seed=23), 40000)
    norms = {i: np.sqrt(bank.entry(i, i)[0].real) for i in candidates}
    bounds = certify_couplings(bank, basis, exact.ritz_vector(0), exact.ground_energy,
                               candidates, cache, norms=norms, delta=0.05, bound="eb")
    for bound in bounds:
        reference = score_candidate(bank, basis, exact, bound.index).residual_coupling
        assert bound.estimate == pytest.approx(reference, abs=0.05)
        assert bound.lower <= reference <= bound.upper


def test_certified_coupling_is_invariant_under_candidate_rescaling(setting):
    """The norm division is what makes the ranking scale-free -- and it comes
    from the construction batch, so the statistic stays linear."""
    model, rho, words, E0 = setting
    bank = MatrixElementBank(rho, model.hamiltonian, [identity_generator(N)])
    basis = [bank.index_of("I")]
    plain = bank.add(pauli_orbit(words[:1])[0])
    scaled = bank.add(Generator("scaled", 7.5 * pauli_orbit(words[:1])[0].mv))
    exact = bank.solve(basis)
    shared = SharedMeasurement(bank, (basis[0], plain, scaled))
    cache = shared.measure(FiniteShotBackend(seed=25), 20000)
    norms = {i: np.sqrt(bank.entry(i, i)[0].real) for i in (plain, scaled)}
    bounds = {b.index: b for b in certify_couplings(
        bank, basis, exact.ritz_vector(0), exact.ground_energy, [plain, scaled],
        cache, norms=norms, delta=0.05, bound="eb")}
    assert bounds[plain].estimate == pytest.approx(bounds[scaled].estimate, rel=1e-9)
    assert bounds[plain].lower == pytest.approx(bounds[scaled].lower, rel=1e-9)


def test_a_candidate_that_annihilates_the_reference_is_rejected(setting):
    from clifford_qc.pauli import I, Z

    model, rho, words, E0 = setting
    bank = MatrixElementBank(rho, model.hamiltonian, [identity_generator(N)])
    basis = [bank.index_of("I")]
    dead = bank.add(Generator("dead", 0.0 * I(N)))
    exact = bank.solve(basis)
    shared = SharedMeasurement(bank, tuple(basis) + (dead,))
    cache = shared.measure(FiniteShotBackend(seed=27), 500)
    bounds = certify_couplings(bank, basis, exact.ritz_vector(0), exact.ground_energy,
                               [dead], cache, norms={dead: 0.0}, delta=0.05)
    assert bounds[0].rejected and "norm" in bounds[0].rejected
    assert not bounds[0].accepted


def test_run_certified_acase_rejects_an_empty_pool(setting):
    model, rho, words, E0 = setting
    with pytest.raises(ValueError, match="no candidate generators"):
        run_certified_acase(rho, model.hamiltonian, [identity_generator(N)],
                            FiniteShotBackend(seed=0), max_size=1,
                            construction_shots=100, certification_shots=100)
