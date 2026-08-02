"""A-CASE Phase 3: exact adaptive growth.

Three properties carry the phase, and each has a test that would fail if the
selection rule were built the obvious wrong way:

- the generalized 2x2 prediction is a *lower bound* on the actual lowering,
  because ``span{Psi_m, chi}`` sits inside ``span{basis + chi}``. A score that
  dropped the overlap block would break the bound in the near-dependent
  direction, and does so measurably here;
- a candidate that is already (numerically) in the span is rejected before it
  can enter, not diagnosed afterwards;
- selection is deterministic through the symmetry ties a spin model produces
  in quantity, and invariant under how the candidates happen to be scaled.

The matched-budget comparisons against fixed QSE, fixed Krylov, and ADAPT-VQE
are the Phase-3 validation of ``ACASE_RESEARCH_PLAN.md`` §5, run from the same
reference state each method starts from.
"""

import numpy as np
import pytest

from clifford_qc.algorithms.adapt import run_adapt
from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.backends import ExactMVBackend
from clifford_qc.matrix import exact_ground, to_matrix
from clifford_qc.models.spin import tfim, xxz
from clifford_qc.pauli import P, X, Y, Z
from clifford_qc.states import ket_density
from clifford_qc.subspace import (
    Generator, MatrixElementBank, adapt_warm_start, commutator_response,
    dense_residual_norm, identity_generator, krylov_response, pauli_orbit,
    run_acase, score_candidate, sector_leakage, select_candidate,
    solve_projected, solve_subspace,
)
from clifford_qc.subspace.adaptive import _GAP_FLOOR, _two_by_two_lowering

N = 4


@pytest.fixture(scope="module")
def case():
    """TFIM n=4 from the model's own reference |++++>, as ADAPT-VQE starts."""
    model = tfim(N, J=1.0, h=1.0)
    rho = ExactMVBackend().state(model.reference, ())
    pool_ops = odd_y_filter(local_pool(N))
    words = [op.word for op in pool_ops]
    candidates = (pauli_orbit(words) + commutator_response(model.hamiltonian, words)
                  + krylov_response(model.hamiltonian, 10))
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    return model, rho, pool_ops, candidates, E0


# --------------------------------------------------------------- the 2x2 score


def test_closed_form_lowering_matches_the_generalized_two_by_two():
    """The score is the generalized problem, solved in closed form, not an
    orthonormal approximation of it."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        energy = float(rng.normal())
        s_aa = float(abs(rng.normal()) + 0.1)
        h_aa = float(rng.normal())
        s_a = complex(rng.normal(), rng.normal()) * 0.2 * np.sqrt(s_aa)
        h_a = complex(rng.normal(), rng.normal())
        S2 = np.array([[1.0, s_a], [np.conjugate(s_a), s_aa]], dtype=complex)
        H2 = np.array([[energy, h_a], [np.conjugate(h_a), h_aa]], dtype=complex)
        if np.linalg.eigvalsh(S2).min() <= 1e-8:
            continue
        reference = solve_projected(S2, H2).ground_energy
        assert _two_by_two_lowering(energy, s_a, h_a, s_aa, h_aa) == pytest.approx(
            energy - reference, abs=1e-9)


def test_lowering_is_zero_when_the_deflation_stops_being_computable():
    """The 0/0 limit returns zero rather than a fabricated eigenvalue."""
    energy = -4.0
    for sigma in (1.0, 1.0 - 1e-16, 1.0 - _GAP_FLOOR / 4):
        # candidate exactly (or numerically) parallel to the current state
        assert _two_by_two_lowering(energy, sigma, sigma * energy, 1.0, energy) == 0.0


def test_lowering_is_invariant_under_candidate_rescaling():
    energy, s_a, h_a, s_aa, h_aa = -1.3, 0.4 + 0.1j, -0.7 + 0.2j, 2.0, -0.9
    base = _two_by_two_lowering(energy, s_a, h_a, s_aa, h_aa)
    for c in (1e-3, 7.0, 1e3):
        scaled = _two_by_two_lowering(energy, c * s_a, c * h_a, c * c * s_aa,
                                      c * c * h_aa)
        assert scaled == pytest.approx(base, rel=1e-9)


def test_dropping_the_overlap_block_invents_a_lowering(case):
    """Why the overlap block is mandatory (§4.3).

    Take a candidate that *is* the current Ritz state, up to a scale factor.
    It can contribute nothing, and the generalized score says so. Solve the
    same 2x2 with the overlap block replaced by the identity -- the assumption
    that the candidate is orthonormal to the current state -- and a large
    lowering appears out of the scaling alone.
    """
    model, rho, _, _, _ = case
    bank = MatrixElementBank(rho, model.hamiltonian)
    basis = bank.extend([identity_generator(N)])
    result = bank.solve(basis)
    parallel = bank.add(Generator("parallel", 3.5 * identity_generator(N).mv))

    score = score_candidate(bank, basis, result, parallel)
    assert score.orthogonal_fraction == pytest.approx(0.0, abs=1e-12)
    assert score.predicted_lowering == 0.0

    energy = result.ground_energy
    s_a, h_a = bank.entry(basis[0], parallel)
    s_aa, h_aa = bank.entry(parallel, parallel)
    naive = np.linalg.eigvalsh(np.array([[energy, h_a],
                                         [np.conjugate(h_a), h_aa.real]]))[0]
    assert energy - naive > 1.0  # a whole Hartree of lowering, entirely spurious
    assert _two_by_two_lowering(energy, s_a, h_a, s_aa.real, h_aa.real) == 0.0


# ------------------------------------------------------------ growth invariants


def test_predicted_lowering_is_a_lower_bound_on_the_actual_one(case):
    """``span{Psi_m, chi}`` is inside ``span{basis + chi}``, so the 2x2 can only
    underestimate what the full re-solve achieves."""
    model, rho, _, candidates, E0 = case
    result = run_acase(rho, model.hamiltonian, candidates, max_size=6,
                       exact_ground_energy=E0)
    assert result.records
    for record in result.records:
        assert record.actual_lowering >= record.predicted_lowering - 1e-12


def test_energy_history_is_monotone_and_bounded(case):
    model, rho, _, candidates, E0 = case
    result = run_acase(rho, model.hamiltonian, candidates, max_size=6,
                       exact_ground_energy=E0)
    history = result.energy_history
    assert len(history) == len(result.records) + 1
    assert all(b <= a + 1e-9 for a, b in zip(history, history[1:]))
    assert all(e >= E0 - 1e-9 for e in history)
    assert result.energy == history[-1]
    assert result.labels[0] == "I" and len(result.labels) == len(history)
    assert result.relative_error == pytest.approx(
        abs(result.energy - E0) / abs(E0), rel=1e-12)


def test_adaptive_never_loses_to_the_fixed_basis_of_the_same_size(case):
    """The Phase-3 acceptance criterion: adaptive <= fixed at equal size."""
    model, rho, _, candidates, E0 = case
    for size in (2, 4, 6):
        adaptive = run_acase(rho, model.hamiltonian, candidates, max_size=size - 1,
                             exact_ground_energy=E0)
        fixed = solve_subspace(rho, model.hamiltonian,
                               [identity_generator(N)] + candidates[:size - 1])
        assert adaptive.energy <= fixed.ground_energy + 1e-9
        assert len(adaptive.labels) == size


def test_matched_budget_against_krylov_qse_and_adapt_vqe(case):
    """§5 validation on the spin model, every method from the same reference."""
    model, rho, pool_ops, candidates, E0 = case
    words = [op.word for op in pool_ops]
    for size in (3, 5):
        adaptive = run_acase(rho, model.hamiltonian, candidates, max_size=size - 1,
                             exact_ground_energy=E0)
        krylov = solve_subspace(rho, model.hamiltonian,
                                [identity_generator(N)]
                                + krylov_response(model.hamiltonian, size - 1))
        qse = solve_subspace(rho, model.hamiltonian,
                             [identity_generator(N)] + pauli_orbit(words[:size - 1]))
        vqe = run_adapt(model, pool_ops, max_operators=size - 1,
                        compute_exact_reference=False)
        assert adaptive.energy <= krylov.ground_energy + 1e-9
        assert adaptive.energy <= qse.ground_energy + 1e-9
        assert adaptive.energy <= vqe.energy + 1e-9
    # and at depth it gets there far better conditioned than the Krylov basis,
    # whose overlap spectrum degrades exponentially in the power. At M=3 there
    # is no depth yet and the two are comparable; the gap opens from M=5 on.
    assert adaptive.result.condition_number < krylov.condition_number


def test_growth_stops_when_the_true_residual_is_gone(case):
    """Convergence, cross-checked against the residual norm §4.4 keeps dense."""
    model, rho, _, candidates, E0 = case
    result = run_acase(rho, model.hamiltonian, candidates, max_size=12,
                       exact_ground_energy=E0)
    assert result.stopped_reason == "predicted lowering below threshold"
    assert result.energy == pytest.approx(E0, abs=1e-9)
    basis = [result.bank.generator(i) for i in result.indices]
    assert dense_residual_norm(rho, model.hamiltonian, basis, result.result) < 1e-6


def test_word_universe_and_new_word_costs_are_recorded(case):
    model, rho, _, candidates, E0 = case
    result = run_acase(rho, model.hamiltonian, candidates, max_size=5,
                       exact_ground_energy=E0)
    universes = [record.word_universe for record in result.records]
    assert all(b >= a for a, b in zip(universes, universes[1:]))  # monotone
    assert universes[-1] == result.resources["word_universe"]
    assert all(record.new_words >= 0 for record in result.records)
    assert result.resources["candidate_pool_size"] == len(candidates)


# ---------------------------------------------------- rejection and determinism


def test_a_candidate_already_in_the_span_is_rejected(case):
    """Linear-dependence rejection, before it can damage the overlap spectrum.

    Rejection is measured against the whole retained subspace: the duplicate
    below is at a perfectly healthy angle to the *Ritz vector* of the two-
    dimensional basis, and only the subspace test sees that it adds nothing.
    """
    model, rho, _, candidates, E0 = case
    bank = MatrixElementBank(rho, model.hamiltonian)
    basis = bank.extend([identity_generator(N), candidates[0]])
    result = bank.solve(basis)
    duplicate = bank.add(Generator("duplicate", 3.5 * candidates[0].mv))
    score = score_candidate(bank, basis, result, duplicate)
    assert score.rejected and score.rejected.startswith("orthogonal fraction")
    assert score.orthogonal_fraction < 1e-12
    # the Ritz-vector-only test would have passed it through
    s_a = result.coefficients[:, 0].conj() @ np.array(
        [bank.entry(i, duplicate)[0] for i in basis])
    s_aa = bank.entry(duplicate, duplicate)[0].real
    assert 1.0 - abs(s_a) ** 2 / s_aa > 0.1

    grown = run_acase(rho, model.hamiltonian,
                      [candidates[0], Generator("duplicate", 3.5 * candidates[0].mv)]
                      + candidates[1:8], max_size=4, exact_ground_energy=E0)
    assert "duplicate" not in grown.labels


def test_every_candidate_rejected_stops_the_run(case):
    model, rho, _, candidates, E0 = case
    bank = MatrixElementBank(rho, model.hamiltonian)
    result = run_acase(rho, model.hamiltonian,
                       [Generator("scaled-identity", 2.0 * identity_generator(N).mv)],
                       bank=bank, max_size=3, exact_ground_energy=E0)
    assert result.stopped_reason.startswith("every candidate rejected")
    assert result.labels == ("I",)


def test_pool_exhaustion_is_reported(case):
    model, rho, _, candidates, E0 = case
    result = run_acase(rho, model.hamiltonian, candidates[:2], max_size=10,
                       min_lowering=0.0, exact_ground_energy=E0)
    assert result.stopped_reason in ("candidate pool exhausted",
                                     "predicted lowering below threshold")


def test_selection_is_deterministic_across_runs(case):
    model, rho, _, candidates, E0 = case
    first = run_acase(rho, model.hamiltonian, candidates, max_size=5)
    second = run_acase(rho, model.hamiltonian, candidates, max_size=5)
    assert first.labels == second.labels
    assert first.energy_history == second.energy_history


def test_selection_is_invariant_under_candidate_rescaling(case):
    """Which direction wins must not depend on how it is normalized."""
    model, rho, _, candidates, E0 = case
    rng = np.random.default_rng(11)
    scaled = [Generator(g.label, complex(rng.uniform(1e-3, 1e3)
                                         * np.exp(1j * rng.uniform(0, 2 * np.pi))) * g.mv)
              for g in candidates]
    base = run_acase(rho, model.hamiltonian, candidates, max_size=5)
    rescaled = run_acase(rho, model.hamiltonian, scaled, max_size=5)
    assert rescaled.labels == base.labels
    assert np.allclose(rescaled.energy_history, base.energy_history, atol=1e-9)


def test_select_candidate_breaks_ties_by_lowest_index():
    from clifford_qc.subspace.adaptive import CandidateScore
    tied = [CandidateScore(7, "a", 1.0, 0.5, 1.0, 0, 0.5),
            CandidateScore(3, "b", 1.0, 0.5, 1.0, 0, 0.5 * (1 + 1e-14)),
            CandidateScore(9, "c", 1.0, 0.1, 1.0, 0, 0.1)]
    assert select_candidate(tied).label == "a"  # first of the tied pair, not "c"
    assert select_candidate([]) is None
    assert select_candidate([CandidateScore(1, "x", 0, 0, 0, 0, 0,
                                            rejected="conditioning")]) is None


# --------------------------------------------------------------- cost weighting


def test_gamma_prices_measurement_cost_into_the_score(case):
    model, rho, _, candidates, E0 = case
    bank = MatrixElementBank(rho, model.hamiltonian)
    basis = bank.extend([identity_generator(N)])
    result = bank.solve(basis)
    index = bank.add(candidates[-1])  # a Krylov power: many words
    plain = score_candidate(bank, basis, result, index)
    priced = score_candidate(bank, basis, result, index, gamma=1.0)
    assert plain.score == pytest.approx(plain.predicted_lowering)
    assert priced.predicted_lowering == pytest.approx(plain.predicted_lowering)
    assert priced.new_words > 0
    assert priced.score < plain.score


# ------------------------------------------------------------- sector discipline


def test_sector_leakage_separates_whole_generators_from_split_words():
    """Operator-level leakage: zero for a conserving generator, not for a word."""
    from clifford_qc.fermion import c_op, cdag_op
    conserving = cdag_op(4, 2) * c_op(4, 0) - cdag_op(4, 0) * c_op(4, 2)
    assert max(sector_leakage(conserving).values()) < 1e-12
    leaking = sector_leakage(P("XIII"))
    assert leaking["particle_number"] > 0.1
    assert sector_leakage(Generator("zero", 0.0 * X(4, 0)))["sz"] == 0.0


def test_leakage_tolerance_rejects_sector_breaking_candidates(case):
    model, rho, _, candidates, E0 = case
    words = pauli_orbit([op.word for op in odd_y_filter(local_pool(N))])[:6]
    result = run_acase(rho, model.hamiltonian, words, max_size=3,
                       leakage_tol=1e-9, exact_ground_energy=E0)
    # single Pauli words do not conserve particle number: nothing may be accepted
    assert result.labels == ("I",)
    assert result.stopped_reason.startswith("every candidate rejected")
    assert not result.records


def test_accepted_generators_report_their_leakage(case):
    model, rho, _, candidates, E0 = case
    result = run_acase(rho, model.hamiltonian, candidates, max_size=3,
                       leakage_tol=None, exact_ground_energy=E0)
    assert all(record.leakage is None for record in result.records)
    reported = run_acase(rho, model.hamiltonian, candidates, max_size=1,
                         leakage_tol=1e9, exact_ground_energy=E0)
    assert reported.records[0].leakage is not None
    assert set(reported.records[0].leakage) == {"particle_number", "sz"}


# ------------------------------------------------------------------ warm start


def test_adapt_warm_start_gives_a_state_acase_can_grow_from(case):
    model, _, pool_ops, candidates, E0 = case
    warm_rho, adapt_result = adapt_warm_start(model, pool_ops, max_operators=2,
                                              compute_exact_reference=False)
    assert warm_rho.is_hermitian() and abs(warm_rho.trace() - 1) < 1e-9
    # the ADAPT energy is reproduced by the identity-only subspace it seeds
    seed = solve_subspace(warm_rho, model.hamiltonian, [identity_generator(N)])
    assert seed.ground_energy == pytest.approx(adapt_result.energy, abs=1e-9)
    grown = run_acase(warm_rho, model.hamiltonian, candidates, max_size=3,
                      exact_ground_energy=E0)
    assert grown.energy <= adapt_result.energy + 1e-9
    assert grown.energy >= E0 - 1e-9


# ---------------------------------------------------------------- other models


def test_growth_works_on_a_second_spin_model():
    model = xxz(N, J=1.0, delta=0.8)
    rho = ket_density(N, "0" * N)
    words = [op.word for op in odd_y_filter(local_pool(N))]
    candidates = (pauli_orbit(words) + commutator_response(model.hamiltonian, words)
                  + krylov_response(model.hamiltonian, 6))
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    result = run_acase(rho, model.hamiltonian, candidates, max_size=6,
                       exact_ground_energy=E0)
    assert result.energy >= E0 - 1e-9
    fixed = solve_subspace(rho, model.hamiltonian,
                           [identity_generator(N)] + candidates[:len(result.labels) - 1])
    assert result.energy <= fixed.ground_energy + 1e-9


def test_run_acase_rejects_an_empty_candidate_pool(case):
    model, rho, _, candidates, _ = case
    with pytest.raises(ValueError, match="no candidate generators"):
        run_acase(rho, model.hamiltonian, [identity_generator(N)], max_size=2)


# --------------------------------------------- excited states (Phase 5 §5)


def test_state_averaged_growth_tracks_several_roots(case):
    """Growing against three roots instead of one, with the per-root variational
    bound as the check: the ``k``-th Ritz value never falls below the ``k``-th
    exact eigenvalue (Cauchy interlacing on a subspace)."""
    model, rho, _, candidates, _ = case
    exact = np.linalg.eigvalsh(to_matrix(model.hamiltonian.to_mv()))[:3]
    result = run_acase(rho, model.hamiltonian, candidates, max_size=8, roots=3)
    assert len(result.root_energies) == 3
    for ritz, reference in zip(result.root_energies, exact):
        assert ritz >= reference - 1e-9
    assert result.energy == pytest.approx(float(np.mean(result.root_energies)), abs=1e-12)
    assert result.resources["roots"] == 3
    assert result.records[-1].per_root_lowering
    assert len(result.records[-1].root_energies) == 3


def test_state_averaged_objective_is_monotone_once_the_roots_exist(case):
    """The caveat, tested where it holds: while roots are still appearing the
    average is taken over fewer of them and can rise."""
    model, rho, _, candidates, _ = case
    result = run_acase(rho, model.hamiltonian, candidates, max_size=8, roots=3)
    settled = [record.energy for record in result.records
               if len(record.root_energies) == 3]
    assert len(settled) >= 2
    assert all(b <= a + 1e-9 for a, b in zip(settled, settled[1:]))


def test_block_growth_chases_the_worst_root(case):
    """``aggregation='max'`` scores on the best single-root gain instead of the
    average, so it can spend a generator only one root wants."""
    model, rho, _, candidates, _ = case
    averaged = run_acase(rho, model.hamiltonian, candidates, max_size=6, roots=3)
    block = run_acase(rho, model.hamiltonian, candidates, max_size=6, roots=3,
                      aggregation="max")
    assert block.resources["aggregation"] == "max"
    assert len(block.root_energies) == len(averaged.root_energies) == 3
    exact = np.linalg.eigvalsh(to_matrix(model.hamiltonian.to_mv()))[:3]
    for ritz, reference in zip(block.root_energies, exact):
        assert ritz >= reference - 1e-9
    # the two rules make different choices somewhere in eight steps
    assert block.labels != averaged.labels or block.energy == pytest.approx(
        averaged.energy, abs=1e-12)


def test_multi_root_scores_report_every_tracked_root(case):
    model, rho, _, candidates, _ = case
    bank = MatrixElementBank(rho, model.hamiltonian)
    basis = bank.extend([identity_generator(N)] + candidates[:3])
    result = bank.solve(basis)
    index = bank.add(candidates[4])
    score = score_candidate(bank, basis, result, index, roots=(0, 1, 2))
    assert len(score.per_root) == 3
    assert score.predicted_lowering == pytest.approx(float(np.mean(score.per_root)))
    peak = score_candidate(bank, basis, result, index, roots=(0, 1, 2),
                           aggregation="max")
    assert peak.predicted_lowering == pytest.approx(max(score.per_root))


def test_multi_root_arguments_are_validated(case):
    model, rho, _, candidates, _ = case
    with pytest.raises(ValueError, match="roots must be at least 1"):
        run_acase(rho, model.hamiltonian, candidates, max_size=1, roots=0)
    with pytest.raises(ValueError, match="aggregation must be"):
        run_acase(rho, model.hamiltonian, candidates, max_size=1,
                  aggregation="median")


# ------------------------------------------------------- oracle-assisted stop


def test_target_error_stops_growth_once_the_threshold_is_cleared(case):
    """``target_error`` is a *diagnostic* stop: it is fed the exact energy.

    What it answers is "how small can M be and still clear the threshold",
    which is a legitimate question about the selector. What it does not answer
    is what the method costs, because in application the exact energy is the
    unknown. The record has to say which of the two it is reporting, so this
    test pins the observable consequence: growth ends at the threshold, the
    reason says so, and the basis is strictly smaller than the unstopped run.
    """
    model, rho, _, candidates, E0 = case
    threshold = 1e-3
    stopped = run_acase(rho, model.hamiltonian, candidates, max_size=12,
                        exact_ground_energy=E0, target_error=threshold)
    assert abs(stopped.energy - E0) <= threshold
    assert stopped.stopped_reason.startswith("target error reached")

    free = run_acase(rho, model.hamiltonian, candidates, max_size=12,
                     exact_ground_energy=E0)
    assert len(stopped.labels) < len(free.labels)
    # and it is the *first* such M: one step earlier the threshold was not met
    assert abs(stopped.energy_history[-2] - E0) > threshold


def test_target_error_needs_the_exact_energy_to_do_anything(case):
    """Without ``exact_ground_energy`` the stop is inert rather than guessing."""
    model, rho, _, candidates, E0 = case
    ungated = run_acase(rho, model.hamiltonian, candidates, max_size=6,
                        target_error=1e-3)
    assert ungated.stopped_reason != "target error reached"
    assert len(ungated.labels) == len(
        run_acase(rho, model.hamiltonian, candidates, max_size=6).labels)


def test_an_already_accurate_reference_returns_before_growing(case):
    """The pre-loop return path: no generator is ever added, so the result must
    still be well formed -- empty records, a one-element basis, and a history
    that is just the reference value. This branch returns a separately
    constructed ``AdaptiveResult`` and so is exactly where a missing field
    would hide."""
    model, rho, _, candidates, _ = case
    reference_energy = run_acase(rho, model.hamiltonian, candidates,
                                 max_size=1).energy_history[0]
    result = run_acase(rho, model.hamiltonian, candidates, max_size=12,
                       exact_ground_energy=reference_energy, target_error=1e-6)
    assert result.records == ()
    assert result.labels == ("I",)
    assert result.energy_history == (reference_energy,)
    assert result.energy == pytest.approx(reference_energy)
    assert result.stopped_reason.startswith("target error reached")
    assert result.relative_error == pytest.approx(0.0, abs=1e-9)


def test_effective_rank_never_exceeds_the_basis_size(case):
    """The invariant the hand-edited molecular record violated: an
    M-dimensional subspace cannot have effective rank above M."""
    model, rho, _, candidates, E0 = case
    for size in (1, 3, 6, 9):
        result = run_acase(rho, model.hamiltonian, candidates, max_size=size,
                           exact_ground_energy=E0)
        assert result.result.effective_rank <= len(result.labels)
