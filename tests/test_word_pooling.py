"""Cross-setting shot pooling and confidence-selected rank (finite-shot layer).

Two changes to how a projected subspace is reconstructed from the *same*
measurements are pinned here.

Pooling: a QWC partition assigns each word to one group, but a group's shots
record every word its basis can read, so the partition throws away readings it
already paid for.  Pooling recovers them.  The tests check that the recovered
estimator is still unbiased, that its reported variance stays exact (the
covariance machinery has to spread a coefficient over groups, not duplicate
it), and that it beats single-assignment on the same shots.

Rank selection: retaining a mode is a decision about the Ritz value, not about
``S``, so the rule scores ranks by an upper confidence bound on the energy.
"""

import math

import numpy as np
import pytest

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend, FiniteShotBackend
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import GroupedWordCache, qwc_groups
from clifford_qc.measurement.functionals import WordFunctional
from clifford_qc.models import tfim
from clifford_qc.subspace import (
    MatrixElementBank,
    SharedMeasurement,
    identity_generator,
    krylov_response,
    pauli_orbit,
    ritz_functional,
)
from clifford_qc.subspace.linalg import solve_projected


def _bank(n=4):
    model = tfim(n, J=1.0, h=1.0)
    reference = ExactMVBackend().state(model.reference, ())
    words = [operator.word for operator in odd_y_filter(local_pool(n))]
    return MatrixElementBank(
        reference,
        model.hamiltonian,
        [identity_generator(n)] + pauli_orbit(words[:6])
        + krylov_response(model.hamiltonian, 2))


@pytest.fixture(scope="module")
def bank():
    return _bank()


@pytest.fixture(scope="module")
def sessions(bank):
    return (SharedMeasurement(bank), SharedMeasurement(bank, pooling="shots"))


@pytest.fixture(scope="module")
def exact_energy(bank, sessions):
    assigned = sessions[0]
    S, Hm = assigned.exact_matrices()
    labels = [bank.generator(i).label for i in assigned.indices]
    return solve_projected(S, Hm, labels).ground_energy


def test_reader_keys_agree_with_letterwise_scan(sessions):
    """The packed reader test is the letter-by-letter one, on every word."""
    pooled = sessions[1]
    cache = pooled.measure(FiniteShotBackend(0), 8)
    states = {state["basis"]: state for state in cache.group_states()}
    for word in pooled.words:
        support = word.support()
        expected = {key for key, state in states.items()
                    for basis in [dict(state["basis"])]
                    if all(j in basis and basis[j] == word.letter(j) for j in support)}
        assert set(cache.reader_keys(word.code)) == expected
    # The partition is a partition, so pooling can only ever add readings.
    assert all(cache.reader_count(word.code) >= 1 for word in pooled.words)
    assert max(cache.reader_count(word.code) for word in pooled.words) > 1


def test_pooling_weights_are_a_convex_split(sessions):
    """Weights sum to one over readers: a reweighting, never a second copy."""
    pooled = sessions[1]
    cache = pooled.measure(FiniteShotBackend(1), 16)
    for word in pooled.words:
        weights = cache.group_weights(word.code)
        assert set(weights) == set(cache.reader_keys(word.code))
        assert sum(weights.values()) == pytest.approx(1.0)
    # Uniform shots per group make the weights uniform over readers.
    word = max(pooled.words, key=lambda w: cache.reader_count(w.code))
    weights = cache.group_weights(word.code)
    assert all(value == pytest.approx(1.0 / len(weights)) for value in weights.values())


def test_pooled_reconstruction_is_unbiased(bank, sessions):
    """Averaged over seeds, pooled entries sit on the exact matrices."""
    assigned, pooled = sessions
    exact_S, exact_H = assigned.exact_matrices()
    stack_S, stack_H = [], []
    for seed in range(60):
        S, Hm = pooled.matrices(pooled.measure(FiniteShotBackend(300 + seed), 400))
        stack_S.append(S)
        stack_H.append(Hm)
    mean_S = np.mean(stack_S, axis=0)
    mean_H = np.mean(stack_H, axis=0)
    # Bias must be small against the seed-mean standard error, not merely small.
    error_S = np.std(stack_S, axis=0, ddof=1) / math.sqrt(len(stack_S))
    error_H = np.std(stack_H, axis=0, ddof=1) / math.sqrt(len(stack_H))
    assert np.all(np.abs(mean_S - exact_S) <= 4.0 * error_S + 1e-12)
    assert np.all(np.abs(mean_H - exact_H) <= 4.0 * error_H + 1e-12)


def test_pooled_variance_prediction_matches_monte_carlo(bank, sessions):
    """The reported variance stays exact once coefficients are spread.

    Duplicating a coefficient into every capable group instead of splitting it
    would inflate this prediction by the reader count -- the failure the single
    assignment was introduced to avoid, and what the weights replace.
    """
    assigned, pooled = sessions
    exact = bank.solve()
    functional = ritz_functional(bank, exact.indices, exact.ritz_vector(0),
                                 exact.ground_energy)
    samples = {"assigned": [], "pooled": []}
    predicted = {}
    for seed in range(250):
        for name, session in (("assigned", assigned), ("pooled", pooled)):
            cache = session.measure(FiniteShotBackend(9000 + seed), 400)
            samples[name].append(functional.estimate(cache))
            predicted.setdefault(name, functional.variance(cache))
    for name in samples:
        empirical = np.var(samples[name], ddof=1)
        assert predicted[name] == pytest.approx(empirical, rel=0.35)
    # Same shots, same circuits, lower variance: the reader count is the gain.
    assert np.var(samples["pooled"]) < 0.6 * np.var(samples["assigned"])


def test_pooling_improves_projected_energy_on_shared_shots(bank, sessions,
                                                           exact_energy):
    """The end-to-end claim, on the seeds both estimators see identically."""
    assigned, pooled = sessions
    errors = {"assigned": [], "pooled": []}
    for seed in range(40):
        for name, session in (("assigned", assigned), ("pooled", pooled)):
            cache = session.measure(FiniteShotBackend(4000 + seed), 750)
            result = session.solve(cache)
            errors[name].append(abs(result.ground_energy - exact_energy))
    for name in errors:
        errors[name] = np.array(errors[name])
    assert np.median(errors["pooled"]) < np.median(errors["assigned"])
    # The tail is where single assignment actually fails: a near-null overlap
    # mode resolved from one group's shots occasionally lands Hartrees away.
    assert errors["pooled"].max() < errors["assigned"].max()


def test_pooling_is_opt_in(bank, sessions):
    """The default estimator is untouched, so committed records still hold."""
    assigned, pooled = sessions
    assert assigned.pooling == "assigned"
    cache = assigned.measure(FiniteShotBackend(5), 64)
    assert cache.pooling == "assigned"
    for word in assigned.words[:20]:
        weights = cache.group_weights(word.code)
        assert len(weights) == 1
        assert next(iter(weights)) == cache.group_key(word.code)
        assert cache.shots(word.code) == 64
    with pytest.raises(ValueError):
        GroupedWordCache(4, pooling="everything")
    with pytest.raises(ValueError):
        SharedMeasurement(bank, pooling="everything")


def test_pooled_word_mean_is_the_pooled_sample_mean():
    """One word, two settings that both read it: the estimate is the mean over
    all shots, and the shot count reported for it is their sum."""
    n = 2
    rho = ExactMVBackend().state(tfim(n, J=1.0, h=1.0).reference, ())
    xx, x0, xz = (PauliWord.from_label(label) for label in ("XX", "XI", "XZ"))
    # Two distinct settings -- (X, X) and (X, Z) -- both fix X on qubit 0, so
    # both record X0 even though the partition assigns it to the first.
    groups = [[xx, x0], [xz]]
    cache = GroupedWordCache(n, pooling="shots")
    cache.add_batch(FiniteShotBackend(3).sample_grouped_from_state(rho, groups, 500))
    assert cache.reader_count(x0.code) == 2
    assert cache.shots(x0.code) == 1000
    pooled = cache.candidate_estimate({x0.code: 1.0})
    per_group = []
    for state in cache.group_states():
        basis = dict(state["basis"])
        if basis.get(0) != "X":
            continue
        position = state["support"].index(0)
        plus = sum(count for bits, count in state["hist"].items()
                   if bits[position] == "0")
        per_group.append((2.0 * plus - state["shots"]) / state["shots"])
    assert len(per_group) == 2
    assert pooled == pytest.approx(sum(per_group) / 2.0)


def test_selected_rank_beats_the_noise_floor_rule(bank, sessions, exact_energy):
    """Scoring ranks by E + gamma*sigma recovers the true rank more often than
    thresholding overlap modes against their own radii, and carries less bias."""
    assigned = sessions[0]
    exact_rank = bank.solve().effective_rank
    selected, calibrated = [], []
    ranks = {"selected": 0, "calibrated": 0}
    for seed in range(30):
        cache = assigned.measure(FiniteShotBackend(7000 + seed), 2000)
        chosen = assigned.solve_selected_rank(cache, gamma=2.0)
        floored = assigned.solve(cache, calibrate_overlap=True)
        selected.append(chosen.ground_energy - exact_energy)
        calibrated.append(floored.ground_energy - exact_energy)
        ranks["selected"] += chosen.effective_rank == exact_rank
        ranks["calibrated"] += floored.effective_rank == exact_rank
    assert ranks["selected"] > ranks["calibrated"]
    assert np.median(np.abs(selected)) < np.median(np.abs(calibrated))
    assert abs(np.mean(selected)) < abs(np.mean(calibrated))


def test_selected_rank_reports_its_whole_score_sweep(sessions):
    """Every scored rank is recorded, and the returned solve is the argmin."""
    pooled = sessions[1]
    cache = pooled.measure(FiniteShotBackend(11), 500)
    result = pooled.solve_selected_rank(cache, gamma=2.0)
    scores = result.resources["rank_selection_scores"]
    assert scores and all(len(entry) == 4 for entry in scores)
    ranks = [entry[0] for entry in scores]
    assert ranks == sorted(ranks)
    best = min(scores, key=lambda entry: entry[3])
    assert result.effective_rank == best[0]
    assert result.resources["rank_selection_rank"] == best[0]
    assert result.resources["rank_selection_score"] == pytest.approx(best[3])
    # Energies fall with rank (Cauchy interlacing) whatever the noise does.
    energies = [entry[1] for entry in scores]
    assert all(later <= earlier + 1e-9
               for earlier, later in zip(energies, energies[1:]))


def test_zero_gamma_selects_the_full_rank(sessions):
    """Without a variance price the rule degenerates to 'take everything'."""
    pooled = sessions[1]
    cache = pooled.measure(FiniteShotBackend(13), 500)
    greedy = pooled.solve_selected_rank(cache, gamma=0.0)
    scores = greedy.resources["rank_selection_scores"]
    assert greedy.effective_rank == max(entry[0] for entry in scores)
    with pytest.raises(ValueError):
        pooled.solve_selected_rank(cache, gamma=-1.0)


def test_overlap_ridge_keeps_the_whitening_identity_and_damps_modes(bank, sessions):
    """The ridge solves a perturbed metric exactly, rather than the true one
    approximately: X'(S+Delta)X = I holds, and a near-null mode is damped
    instead of dropped.  (It is a worse estimator than truncation -- see
    ``benchmarks/run_finite_shot_rethink.py`` -- but it must be the estimator
    it claims to be.)"""
    assigned = sessions[0]
    cache = assigned.measure(FiniteShotBackend(17), 500)
    S, Hm = assigned.matrices(cache)
    labels = [bank.generator(i).label for i in assigned.indices]
    plain = solve_projected(S, Hm, labels)
    ridged = solve_projected(S, Hm, labels, overlap_ridge=1e-2)
    # The whitening residual check inside solve_projected is against the ridged
    # metric; it raising nothing is the identity holding.
    assert ridged.resources["overlap_ridge"] == pytest.approx(1e-2)
    assert ridged.resources["retained_rank"] >= plain.resources["retained_rank"]
    assert ridged.condition_number < plain.condition_number
    # Reported overlap eigenvalues stay the measured ones, not the shifted ones.
    assert ridged.overlap_eigenvalues == plain.overlap_eigenvalues
    # Zero ridge is exactly the old path.
    assert solve_projected(S, Hm, labels, overlap_ridge=0.0).energies == plain.energies
    with pytest.raises(ValueError):
        solve_projected(S, Hm, labels, overlap_ridge=-1.0)


def test_ridge_and_floor_units_are_per_mode(bank, sessions):
    """A per-mode ridge must have one entry per live generator, like the floor."""
    assigned = sessions[0]
    cache = assigned.measure(FiniteShotBackend(19), 256)
    S, Hm = assigned.matrices(cache)
    labels = [bank.generator(i).label for i in assigned.indices]
    size = S.shape[0]
    solve_projected(S, Hm, labels, overlap_ridge=np.full(size, 1e-3))
    with pytest.raises(ValueError):
        solve_projected(S, Hm, labels, overlap_ridge=np.full(size + 1, 1e-3))


def test_pooling_leaves_deterministic_functionals_alone(sessions):
    """A functional with no words needs no shots under either estimator."""
    _, pooled = sessions
    cache = pooled.measure(FiniteShotBackend(23), 32)
    constant = WordFunctional({}, 1.0)
    assert constant.is_deterministic
    assert constant.estimate(cache) == 1.0
    assert constant.variance(cache) == 0.0
    assert constant.radius(cache, 0.05) == 0.0


def test_pooled_grouped_terms_reject_an_unmeasured_word(sessions):
    """An unreadable word still resolves to 'no statistics', not a silent drop."""
    _, pooled = sessions
    cache = pooled.measure(FiniteShotBackend(29), 32)
    absent = PauliWord.from_label("XYXY").code
    if cache.reader_keys(absent):
        pytest.skip("this universe happens to read the probe word")
    assert cache.candidate_group_statistics({absent: 1.0}) is None
    assert cache.mean_var(absent) == (0.0, float("inf"))
    assert cache.shots(absent) == 0


def test_pooling_survives_a_second_batch(sessions):
    """Cumulative reuse: adding shots re-derives readers and per-group sums."""
    _, pooled = sessions
    backend = FiniteShotBackend(31)
    cache = pooled.measure(backend, 200)
    word = max(pooled.words, key=lambda w: cache.reader_count(w.code))
    first = cache.shots(word.code)
    pooled.measure(backend, 200, cache)
    assert cache.shots(word.code) == 2 * first
    assert sum(cache.group_weights(word.code).values()) == pytest.approx(1.0)
    groups = qwc_groups(list(pooled.words))
    assert cache.num_groups() == len(groups)
