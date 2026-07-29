"""A-CASE Phase 2: the cached matrix-element bank and projected observables.

Two claims are under test. First, the bank is an *optimization only*: it must
reproduce the Phase-1 matrices and energies bit for bit, not merely to a
tolerance, so a record produced through either route is the same record.
Second, it must actually cache -- re-solving builds nothing, and growing the
basis by one generator costs exactly one new row.

The projected-observable API (§8) is checked against the dense Ritz state that
A-CASE never forms: ``B c`` with ``B`` the explicit basis-state matrix.
"""

import numpy as np
import pytest

from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.matrix import exact_ground, to_matrix
from clifford_qc.models.spin import tfim, xxz
from clifford_qc.multivector import MV
from clifford_qc.pauli import I, P, X, Y, Z
from clifford_qc.states import ket_density
from clifford_qc.subspace import (
    Generator, MatrixElementBank, dense_basis, identity_generator,
    krylov_response, pauli_orbit, projected_matrices, response_hierarchy,
    solve_subspace,
)

N = 4


@pytest.fixture(scope="module")
def case():
    model = tfim(N, J=1.0, h=1.0)
    rho = ket_density(N, "0" * N)
    words = [op.word for op in odd_y_filter(local_pool(N))]
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=2)
    return model, rho, gens


def dense_ritz(rho, gens, result, k=0):
    psi = dense_basis(rho, gens) @ result.ritz_vector(k)
    return psi / np.linalg.norm(psi)


# ------------------------------------------------------ agreement with Phase 1


def test_bank_reproduces_the_phase_one_matrices_bit_for_bit(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    S_bank, H_bank = bank.matrices()
    S_direct, H_direct, _ = projected_matrices(rho, model.hamiltonian, gens)
    assert np.array_equal(S_bank, S_direct)
    assert np.array_equal(H_bank, H_direct)


def test_bank_reproduces_the_phase_one_solve_bit_for_bit(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    banked = bank.solve()
    direct = solve_subspace(rho, model.hamiltonian, gens)
    assert banked.energies == direct.energies
    assert np.array_equal(banked.coefficients, direct.coefficients)
    assert banked.basis_labels == direct.basis_labels
    assert banked.effective_rank == direct.effective_rank


def test_bank_matches_phase_one_on_a_prefix_subset(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    for size in (1, 6, 17):
        subset = bank.solve(range(size))
        assert subset.energies == solve_subspace(rho, model.hamiltonian,
                                                 gens[:size]).energies
        assert subset.basis_labels == tuple(g.label for g in gens[:size])


# ------------------------------------------------------------------- caching


def test_registration_builds_no_pair_products(case):
    """Lazy by design: an unscored candidate must cost nothing."""
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    assert len(bank) == len(gens)
    assert bank.resources()["pairs_built"] == 0


def test_resolving_twice_recomputes_nothing(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    bank.solve()
    built = bank.resources()["pairs_built"]
    assert built == len(gens) * (len(gens) + 1) // 2  # upper triangle only
    bank.solve()
    bank.matrices()
    assert bank.resources()["pairs_built"] == built
    assert bank.resources()["element_cache_hits"] > 0


def test_growing_the_basis_costs_exactly_one_row(case):
    """The reason the bank exists: adaptive growth must not re-pay for the block."""
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:-1])
    bank.solve()
    before = bank.resources()["pairs_built"]
    bank.add(gens[-1])
    bank.solve()
    assert bank.resources()["pairs_built"] == before + len(gens)


def test_only_the_upper_triangle_is_stored(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:5])
    bank.matrices()
    assert bank.resources()["pairs_built"] == 15  # 5*6/2, not 25
    for i, j in ((0, 3), (2, 4)):
        assert bank.overlap_operator(j, i).is_close(bank.overlap_operator(i, j).dagger())
        assert bank.element_operator(j, i).is_close(bank.element_operator(i, j).dagger())
        s_ij, h_ij = bank.entry(i, j)
        s_ji, h_ji = bank.entry(j, i)
        assert s_ji == s_ij.conjugate() and h_ji == h_ij.conjugate()


# ------------------------------------------------------- canonical generator ids


def test_the_same_operator_gets_the_same_id(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:4])
    first = bank.index_of(gens[2].label)
    assert bank.add(gens[2]) == first
    assert bank.add(Generator("another name", gens[2].mv)) == first
    assert len(bank) == 4


def test_reusing_a_label_for_a_different_operator_is_rejected(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:4])
    clashing = Generator(gens[1].label, 2.0 * gens[3].mv)
    with pytest.raises(ValueError, match="already bound"):
        bank.add(clashing)


def test_index_and_label_bookkeeping(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:4])
    assert bank.labels == tuple(g.label for g in gens[:4])
    assert bank.generator(1).mv.is_close(gens[1].mv)
    assert bank.n == N
    with pytest.raises(IndexError):
        bank.matrices([0, 9])
    with pytest.raises(ValueError, match="repeated generator index"):
        bank.matrices([0, 1, 1])


# ------------------------------------------------------------- word accounting


def test_word_universe_matches_an_independent_union(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:8])
    bank.matrices()
    H = model.hamiltonian.to_mv()
    universe = set()
    for j, right in enumerate(gens[:8]):
        for left in gens[:j + 1]:
            universe |= set((left.mv.dagger() * right.mv).terms)
            universe |= set((left.mv.dagger() * H * right.mv).terms)
    assert bank.resources()["word_universe"] == len(universe)
    assert {w.code for w in bank.words()} == universe
    assert list(bank.words()) == sorted(bank.words(), key=lambda w: w.code)


def test_new_and_reused_words_are_attributed_per_generator(case):
    """§6: what each accepted generator introduces versus reuses."""
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:8])
    bank.matrices()
    growth = bank.resources()["new_words_per_generator"]
    assert [label for label, _, _ in growth] == [g.label for g in gens[:8]]
    assert sum(new for _, new, _ in growth) == bank.resources()["word_universe"]
    # later generators reuse what earlier ones introduced
    assert sum(reused for _, _, reused in growth) > 0


def test_a_proper_subset_owns_a_smaller_universe(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    bank.matrices()
    subset = bank.resources(range(5))["word_universe"]
    assert 0 < subset < bank.resources()["word_universe"]


def test_qwc_group_count_is_a_partition_of_the_universe(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:5])
    bank.matrices()
    groups = bank.qwc_group_count()
    assert 1 <= groups <= len(bank.words())


# ---------------------------------------------------- projected observables


def test_projected_observable_matches_the_dense_basis_matrix(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:10])
    Q = Z(N, 0) * Z(N, 1)
    Q_sub = bank.project_observable(Q)
    basis = dense_basis(rho, gens[:10])
    assert np.abs(Q_sub - basis.conj().T @ to_matrix(Q) @ basis).max() < 1e-10
    assert np.array_equal(Q_sub, Q_sub.conj().T)  # structural, like S and H


def test_expectation_matches_the_ritz_state_it_never_forms(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    result = bank.solve()
    for Q in (Z(N, 0), Z(N, 0) * Z(N, 1), X(N, 2)):
        dense = to_matrix(Q)
        for k in (0, 1):
            psi = dense_ritz(rho, gens, result, k)
            assert result.expectation(Q, k) == pytest.approx(
                (psi.conj() @ dense @ psi).real, abs=1e-9)


def test_projecting_the_hamiltonian_returns_the_ritz_energies(case):
    """The consistency check the API owes: <H>_k must be E_k."""
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    result = bank.solve()
    for k, energy in enumerate(result.energies):
        assert result.expectation(model.hamiltonian, k) == pytest.approx(energy, abs=1e-9)


def test_transition_elements_match_the_dense_states(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    result = bank.solve()
    Q = X(N, 0)
    dense = to_matrix(Q)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        psi_i, psi_j = dense_ritz(rho, gens, result, i), dense_ritz(rho, gens, result, j)
        assert result.transition(Q, i, j) == pytest.approx(
            complex(psi_i.conj() @ dense @ psi_j), abs=1e-9)
    # diagonal transition is the expectation
    assert result.transition(Q, 0, 0).real == pytest.approx(result.expectation(Q), abs=1e-12)


def test_non_hermitian_observables_take_the_transition_route(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:8])
    result = bank.solve()
    Q = P("XYII") * P("IZZI")  # not Hermitian
    assert not Q.is_hermitian()
    Q_sub = bank.project_observable(Q)
    assert not np.allclose(Q_sub, Q_sub.conj().T)  # full M^2, no mirroring
    basis = dense_basis(rho, gens[:8])
    assert np.abs(Q_sub - basis.conj().T @ to_matrix(Q) @ basis).max() < 1e-10
    with pytest.raises(ValueError, match="Hermitian observable"):
        result.expectation(Q)
    assert isinstance(result.transition(Q, 0, 1), complex)


def test_observable_word_cost_is_recorded(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens[:6])
    bank.matrices()
    bank.project_observable(Y(N, 0) * Y(N, 3), label="YY")
    report = bank.resources()["projected_observables"]
    assert set(report) == {"YY"}
    assert report["YY"]["words"] > 0
    assert report["YY"]["additional_words"] >= 0
    assert report["YY"]["additional_words"] <= report["YY"]["words"]


def test_observables_without_a_bank_say_so():
    model = tfim(3, J=1.0, h=1.0)
    rho = ket_density(3, "000")
    gens = [identity_generator(3)] + krylov_response(model.hamiltonian, 2)
    result = solve_subspace(rho, model.hamiltonian, gens)
    with pytest.raises(ValueError, match="MatrixElementBank"):
        result.expectation(Z(3, 0))
    with pytest.raises(ValueError, match="MatrixElementBank"):
        result.transition(Z(3, 0), 0, 1)


# ----------------------------------------------------------------- validation


def test_bank_rejects_bad_inputs():
    model = tfim(3, J=1.0, h=1.0)
    rho = ket_density(3, "000")
    with pytest.raises(ValueError, match="Hermitian"):
        MatrixElementBank(rho, model.hamiltonian.to_mv() + 1j * P("XII"))
    with pytest.raises(ValueError, match="unit trace"):
        MatrixElementBank(2.0 * rho, model.hamiltonian)
    bank = MatrixElementBank(rho, model.hamiltonian, [identity_generator(3)])
    with pytest.raises(ValueError, match="different algebra"):
        bank.add(identity_generator(4))
    with pytest.raises(ValueError, match="different algebra"):
        bank.project_observable(Z(4, 0))


def test_bank_solve_carries_the_conditioning_report(case):
    model, rho, gens = case
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    result = bank.solve()
    resources = result.resources
    assert resources["retained_rank"] == result.effective_rank
    assert resources["pairs_built"] > 0
    assert resources["cached_operator_bytes"] > 0
    assert resources["max_hamiltonian_element_support"] > 0
    assert resources["reference_purity"] == pytest.approx(1.0, abs=1e-12)


def test_bank_tracks_a_different_model(case):
    """Nothing in the bank is TFIM-specific: same invariants on XXZ."""
    model = xxz(N, J=1.0, delta=0.8)
    rho = ket_density(N, "0" * N)
    words = [op.word for op in odd_y_filter(local_pool(N))]
    gens = [identity_generator(N)] + pauli_orbit(words[:8])
    bank = MatrixElementBank(rho, model.hamiltonian, gens)
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    result = bank.solve()
    assert result.ground_energy >= E0 - 1e-9
    assert result.expectation(model.hamiltonian) == pytest.approx(
        result.ground_energy, abs=1e-9)
