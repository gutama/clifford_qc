"""A-CASE Phase 1: the trace pairing and the exact fixed-basis subspace solve.

The oracle is the dense route in ``subspace/reference.py``: materialize
``|psi>``, materialize every ``A_i|psi>``, and form the Gram and Hamiltonian
matrices with plain linear algebra. The operator route must agree with it
while never preparing a basis state -- that agreement is the Phase-1
correctness statement, and it is what a conjugation or reversion slip in the
pairing would break.

The standing invariants of ``PLAN.md`` §3 are checked here
rather than asserted in the solver: ``E_sub >= E_0``, monotone non-increasing
Ritz values under nested growth, exact Hermiticity and positive
semidefiniteness of ``S``, invariance of the retained subspace under generator
rescaling, and reproduction of the exact ground energy when the span contains
the ground state.
"""

import numpy as np
import pytest

from clifford_qc.algorithms.adapt import TIE_ATOL, TIE_RTOL
from clifford_qc.algorithms.pools import local_pool, odd_y_filter
from clifford_qc.ir import PauliWord
from clifford_qc.matrix import exact_ground, from_matrix, to_matrix
from clifford_qc.models.spin import tfim, xxz
from clifford_qc.multivector import MV
from clifford_qc.pauli import I, P, X, Y, Z
from clifford_qc.states import expectation, ket_density, plus_density
from clifford_qc.subspace import (
    Generator, as_generators, canonical_eigh, commutator_response,
    dense_projected_matrices, dense_subspace, identity_generator,
    krylov_response, pauli_orbit, projected_matrices, pure_statevector,
    response_hierarchy, solve_projected, solve_subspace,
)
from clifford_qc.subspace.solver import _canonical_block, _degenerate_blocks

N = 4


@pytest.fixture(scope="module")
def tfim_case():
    """TFIM n=4 from the |0...0> reference: the plan's premise-check setting."""
    model = tfim(N, J=1.0, h=1.0)
    rho = ket_density(N, "0" * N)
    words = [op.word for op in odd_y_filter(local_pool(N))]
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    return model, rho, words, E0


def random_mv(rng, n=3, terms=8):
    return MV(n, {int(rng.integers(0, 4 ** n)): complex(rng.normal(), rng.normal())
                  for _ in range(terms)})


# ------------------------------------------------------------- trace pairing


def test_trace_pairing_is_the_bilinear_trace_form():
    """``2^n <A, B> == Tr(A B)`` against the dense matrix bridge."""
    rng = np.random.default_rng(0)
    n = 3
    for _ in range(20):
        A, B = random_mv(rng, n), random_mv(rng, n)
        dense = np.trace(to_matrix(A) @ to_matrix(B))
        assert (2 ** n) * A.trace_pairing(B) == pytest.approx(dense, abs=1e-9)


def test_trace_pairing_is_symmetric_and_bilinear():
    rng = np.random.default_rng(1)
    A, B, C = (random_mv(rng) for _ in range(3))
    assert A.trace_pairing(B) == pytest.approx(B.trace_pairing(A), abs=1e-12)
    lhs = A.trace_pairing(2.0 * B + (3.0 + 1j) * C)
    rhs = 2.0 * A.trace_pairing(B) + (3.0 + 1j) * A.trace_pairing(C)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_the_three_pairings_are_genuinely_different():
    """The conjugation and reversion traps the plan's §3 table warns about.

    ``hs_product`` conjugates the left factor and ``scalar_product`` carries
    the reversion sign; on a complex, non-Hermitian operator both differ from
    the trace pairing. A Hermitian test case would hide the first difference
    (real coefficients) and a grade-0/1 one would hide the second.
    """
    n = 2
    # grade 2 (mod 4) word under the JW blade correspondence: reversion flips it
    A = MV.from_terms(n, {"ZI": 1.0 + 2.0j, "XY": 0.5j})
    B = MV.from_terms(n, {"ZI": 3.0 - 1.0j, "XY": 1.0})
    assert A.trace_pairing(B) != pytest.approx(A.hs_product(B), abs=1e-9)
    assert A.trace_pairing(B) != pytest.approx(A.scalar_product(B), abs=1e-9)
    # and the trace pairing is the one that reproduces Tr(A B)
    assert (2 ** n) * A.trace_pairing(B) == pytest.approx(
        np.trace(to_matrix(A) @ to_matrix(B)), abs=1e-9)


def test_expectation_of_a_non_hermitian_operator_is_the_trace_pairing():
    """``Tr(O rho)`` without forming ``O rho``, for the ``O`` A-CASE actually uses."""
    rho = plus_density(3)
    H = tfim(3).hamiltonian.to_mv()
    A_i, A_j = P("XYI"), P("IZY")
    for O in (A_i.dagger() * A_j, A_i.dagger() * H * A_j):
        assert not O.is_hermitian()  # the case that hides a conjugation bug
        assert (2 ** 3) * O.trace_pairing(rho) == pytest.approx(
            expectation(rho, O), abs=1e-12)


def test_purity_via_the_trace_pairing():
    rho = plus_density(3)
    assert (2 ** 3) * rho.trace_pairing(rho).real == pytest.approx(
        (rho * rho).trace().real, abs=1e-12)


# --------------------------------------------------------- matrix assembly


def test_projected_matrices_match_the_dense_route(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=2)
    S, Hm, _ = projected_matrices(rho, model.hamiltonian, gens)
    S_dense, H_dense = dense_projected_matrices(rho, model.hamiltonian, gens)
    assert np.abs(S - S_dense).max() < 1e-10
    assert np.abs(Hm - H_dense).max() < 1e-10


def test_cyclic_contraction_agrees_with_the_element_operator_route(tfim_case):
    """The cheap path forms no element operator; it must still be the same matrix."""
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=2)
    tracked = projected_matrices(rho, model.hamiltonian, gens)
    fast = projected_matrices(rho, model.hamiltonian, gens, track_support=False)
    assert np.abs(tracked[0] - fast[0]).max() < 1e-12
    assert np.abs(tracked[1] - fast[1]).max() < 1e-12
    assert fast[2]["word_universe"] is None  # blind by construction, not by accident
    assert tracked[2]["word_universe"] > 0


def test_overlap_is_exactly_hermitian_and_positive_semidefinite(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=2)
    S, Hm, _ = projected_matrices(rho, model.hamiltonian, gens)
    # structural, not numerical: the lower triangle *is* the conjugate transpose
    assert np.array_equal(S, S.conj().T)
    assert np.array_equal(Hm, Hm.conj().T)
    assert np.linalg.eigvalsh(S).min() > -1e-10


def test_resource_metrics_report_the_word_universe(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words[:6], krylov_order=1)
    _, _, resources = projected_matrices(rho, model.hamiltonian, gens)
    H = model.hamiltonian.to_mv()
    universe, widest = set(), 0
    for j, right in enumerate(gens):
        for left in gens[:j + 1]:
            overlap = left.mv.dagger() * right.mv
            element = left.mv.dagger() * H * right.mv
            universe |= set(overlap.terms) | set(element.terms)
            widest = max(widest, element.nnz())
    assert resources["word_universe"] == len(universe)
    assert resources["max_hamiltonian_element_support"] == widest
    assert resources["max_generator_support"] == max(g.support() for g in gens)
    assert resources["reference_purity"] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------ standing invariants


@pytest.mark.parametrize("model_factory", [
    lambda: tfim(N, J=1.0, h=1.0),
    lambda: tfim(N, J=1.0, h=0.4),
    lambda: xxz(N, J=1.0, delta=0.8),
])
def test_variational_bound_and_nested_monotonicity(model_factory):
    """``E_sub >= E_0``, and adding generators never raises the Ritz value."""
    model = model_factory()
    rho = ket_density(model.n, "0" * model.n)
    words = [op.word for op in odd_y_filter(local_pool(model.n))]
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=3)
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    previous = None
    for size in (1, 1 + len(words), len(gens) - 3, len(gens)):
        result = solve_subspace(rho, model.hamiltonian, gens[:size])
        assert result.ground_energy >= E0 - 1e-9
        if previous is not None:
            assert result.ground_energy <= previous + 1e-9
        previous = result.ground_energy


def test_krylov_growth_reaches_the_exact_ground_energy(tfim_case):
    """A Krylov space deep enough to contain the ground state returns ``E_0``.

    Deep enough, but not arbitrarily accurate: the power basis conditions
    exponentially badly (``kappa_S`` here passes 1e9 by depth 10), so the
    residual error floors out around 1e-8 rather than shrinking to machine
    precision. That floor is the reason the plan treats ``H^k`` as one
    candidate family and conditioning-aware selection as load-bearing.
    """
    model, rho, _, E0 = tfim_case
    gens = [identity_generator(model.n)] + krylov_response(model.hamiltonian, 10)
    result = solve_subspace(rho, model.hamiltonian, gens)
    assert result.ground_energy == pytest.approx(E0, abs=1e-7)
    assert result.ground_energy >= E0 - 1e-9
    assert result.condition_number > 1e6


def test_a_basis_containing_the_ground_state_reproduces_it(tfim_case):
    """Reproduction invariant: the exact ground projector as a generator."""
    model, rho, words, E0 = tfim_case
    _, psi0 = exact_ground(model.hamiltonian.to_mv())
    projector = from_matrix(np.outer(psi0, psi0.conj()), model.n)
    gens = [identity_generator(model.n), Generator("P0", projector)]
    gens += pauli_orbit(words[:4])
    result = solve_subspace(rho, model.hamiltonian, gens)
    assert result.ground_energy == pytest.approx(E0, abs=1e-10)


def test_retained_subspace_is_invariant_under_generator_rescaling(tfim_case):
    """Thresholding raw ``S`` would make this fail: the ``H^k`` rows are huge."""
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=3)
    base = solve_subspace(rho, model.hamiltonian, gens)

    rng = np.random.default_rng(7)
    scaled = [Generator(g.label,
                        complex(rng.uniform(1e-3, 1e3)
                                * np.exp(1j * rng.uniform(0, 2 * np.pi))) * g.mv)
              for g in gens]
    rescaled = solve_subspace(rho, model.hamiltonian, scaled)

    assert rescaled.effective_rank == base.effective_rank
    assert np.allclose(rescaled.energies, base.energies, atol=1e-9)
    assert np.allclose(rescaled.overlap_eigenvalues, base.overlap_eigenvalues, atol=1e-9)
    assert rescaled.condition_number == pytest.approx(base.condition_number, rel=1e-9)


def test_ritz_vectors_are_s_normalized_and_return_their_energy(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=2)
    S, Hm, _ = projected_matrices(rho, model.hamiltonian, gens)
    result = solve_projected(S, Hm, [g.label for g in gens])
    for k, energy in enumerate(result.energies):
        c = result.ritz_vector(k)
        assert (c.conj() @ S @ c).real == pytest.approx(1.0, abs=1e-8)
        assert (c.conj() @ Hm @ c).real == pytest.approx(energy, abs=1e-8)


def test_dense_and_operator_routes_give_the_same_spectrum(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=2)
    operator = solve_subspace(rho, model.hamiltonian, gens)
    dense = dense_subspace(rho, model.hamiltonian, gens)
    assert np.allclose(operator.energies, dense.energies, atol=1e-9)
    assert operator.effective_rank == dense.effective_rank


@pytest.mark.parametrize("order", [2, 4])
def test_ritz_values_match_scipy_generalized_eigh(tfim_case, order):
    """Cross-check against a library solver where ``S`` is positive definite.

    A Krylov basis, not a Pauli-orbit one: distinct words act identically on a
    product-state reference up to a phase, so the level-1 overlap matrix is
    exactly singular and ``scipy.linalg.eigh`` has no generalized problem to
    solve. The thresholded route does -- which is the point of thresholding,
    and why the comparison has to be made where both are defined.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    model, rho, _, _ = tfim_case
    gens = [identity_generator(model.n)] + krylov_response(model.hamiltonian, order)
    S, Hm, _ = projected_matrices(rho, model.hamiltonian, gens)
    result = solve_projected(S, Hm)
    assert result.effective_rank == len(gens)  # nothing truncated: a fair comparison
    reference = scipy_linalg.eigh(Hm, S, eigvals_only=True)
    assert np.allclose(result.energies, reference, atol=1e-8)


# ---------------------------------------------- conditioning and truncation


def test_linearly_dependent_generators_are_truncated(tfim_case):
    """A duplicated direction adds an overlap null mode, not a dimension."""
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words[:5])
    base = solve_subspace(rho, model.hamiltonian, gens)
    duplicated = gens + [Generator("dup", 3.5 * gens[2].mv)]
    grown = solve_subspace(rho, model.hamiltonian, duplicated)
    assert grown.effective_rank == base.effective_rank
    assert grown.ground_energy == pytest.approx(base.ground_energy, abs=1e-9)
    assert grown.resources["basis_size"] == base.resources["basis_size"] + 1


def test_generators_that_annihilate_the_reference_are_dropped(tfim_case):
    """``(1 - Z_0)/2`` kills |0...0>: a zero row normalization cannot divide by."""
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words[:5])
    dead = Generator("dead", 0.5 * (I(model.n) - Z(model.n, 0)))
    result = solve_subspace(rho, model.hamiltonian, gens + [dead])
    assert result.resources["dropped_generators"] == ("dead",)
    assert np.abs(result.coefficients[-1, :]).max() == 0.0
    assert result.ground_energy == pytest.approx(
        solve_subspace(rho, model.hamiltonian, gens).ground_energy, abs=1e-12)


def test_condition_cap_trims_the_retained_subspace(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=4)
    loose = solve_subspace(rho, model.hamiltonian, gens)
    tight = solve_subspace(rho, model.hamiltonian, gens, max_condition=1e2)
    assert tight.effective_rank < loose.effective_rank
    assert tight.condition_number <= 1e2 + 1e-6
    # truncating directions can only raise the Ritz value
    assert tight.ground_energy >= loose.ground_energy - 1e-9


def test_every_direction_rejected_is_an_error(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words[:4])
    with pytest.raises(ValueError, match="retained no direction"):
        solve_subspace(rho, model.hamiltonian, gens, tau_s=1e3)


# ------------------------------------------------------------ determinism


def test_canonical_block_ignores_the_mixing_inside_a_degenerate_eigenspace():
    """The property LAPACK does not have: a basis fixed by the projector alone."""
    rng = np.random.default_rng(3)
    m, dim = 7, 3
    Q, _ = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))
    block = Q[:, :dim]
    mixing, _ = np.linalg.qr(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    canonical = _canonical_block(block)
    assert np.abs(canonical - _canonical_block(block @ mixing)).max() < 1e-10
    assert np.abs(canonical.conj().T @ canonical - np.eye(dim)).max() < 1e-10
    # same subspace, different (canonical) basis for it
    assert np.abs(canonical @ canonical.conj().T
                  - block @ block.conj().T).max() < 1e-10


def test_degeneracy_grouping_is_relative_to_the_neighbouring_pair():
    """Near-null overlap modes must not be swallowed by the top of the spectrum."""
    values = np.array([1e-10, 3e-8, 1.0, 1.0, 4.0])
    assert _degenerate_blocks(values, TIE_RTOL, TIE_ATOL) == [(0, 1), (1, 2), (2, 4), (4, 5)]


def test_degeneracy_grouping_does_not_chain_nearby_values():
    values = np.array([1.0, 1.0 + 0.75e-9, 1.0 + 1.5e-9])
    assert _degenerate_blocks(values, TIE_RTOL, TIE_ATOL) == [(0, 2), (2, 3)]


def test_near_null_modes_remain_whitened():
    angle = 0.37
    Q = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    S = Q @ np.diag([1e-10, 1e-10 + 1.5e-12]) @ Q.T
    Hm = np.diag([0.2, 0.7])
    result = solve_projected(S, Hm, tau_s=1e-12, max_condition=1e14)
    assert result.resources["whitening_residual_inf"] < 1e-8
    assert np.allclose(result.coefficients.conj().T @ S @ result.coefficients,
                       np.eye(2), atol=1e-7)


def test_canonical_eigh_reconstructs_and_pins_phases():
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    spectrum = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 5.0])
    A = Q @ np.diag(spectrum) @ Q.conj().T
    A = 0.5 * (A + A.conj().T)
    values, vectors = canonical_eigh(A)
    assert np.allclose(values, spectrum, atol=1e-10)
    assert np.abs(vectors @ np.diag(values) @ vectors.conj().T - A).max() < 1e-10
    for k in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, k])))
        assert vectors[pivot, k].real > 0
        assert abs(vectors[pivot, k].imag) < 1e-12


def test_repeated_solves_are_bitwise_identical(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words, krylov_order=3)
    first = solve_subspace(rho, model.hamiltonian, gens)
    second = solve_subspace(rho, model.hamiltonian, gens)
    assert first.energies == second.energies
    assert np.array_equal(first.coefficients, second.coefficients)


def test_generator_order_does_not_change_the_spectrum(tfim_case):
    model, rho, words, _ = tfim_case
    gens = response_hierarchy(model.hamiltonian, words[:6], krylov_order=2)
    forward = solve_subspace(rho, model.hamiltonian, gens)
    reversed_ = solve_subspace(rho, model.hamiltonian, list(reversed(gens)))
    assert np.allclose(forward.energies, reversed_.energies, atol=1e-9)
    assert forward.effective_rank == reversed_.effective_rank


# ------------------------------------------------------- inputs and errors


def test_as_generators_coerces_the_usual_suspects(tfim_case):
    model, _, words, _ = tfim_case
    pool_ops = odd_y_filter(local_pool(model.n))[:2]
    gens = as_generators([identity_generator(model.n), words[0], pool_ops[0],
                          model.hamiltonian, ("named", X(model.n, 0)),
                          Y(model.n, 1)])
    assert [g.n for g in gens] == [model.n] * 6
    assert gens[0].label == "I"
    assert gens[1].label == words[0].label
    assert gens[4].label == "named"
    assert gens[5].label == "A5"  # positional fallback


def test_commutator_response_drops_words_that_commute_with_h():
    """A commuting word has no response direction, only an empty row."""
    model = tfim(3, J=1.0, h=1.0)
    words = [op.word for op in odd_y_filter(local_pool(3))]
    gens = commutator_response(model.hamiltonian, words)
    assert gens and all(g.mv.nnz() > 0 for g in gens)
    assert all(g.label.startswith("G[") for g in gens)

    z_word = PauliWord.from_label("ZII", 3)
    ising = tfim(3, J=1.0, h=0.0)  # ZZ only: Z_0 commutes with all of it
    assert commutator_response(ising.hamiltonian, [z_word]) == []
    assert len(commutator_response(ising.hamiltonian, [z_word], drop_zero=False)) == 1
    assert commutator_response(model.hamiltonian, [z_word])  # transverse field revives it


def test_non_hermitian_hamiltonian_is_rejected(tfim_case):
    model, rho, words, _ = tfim_case
    bad = model.hamiltonian.to_mv() + 1j * P("XIII")
    with pytest.raises(ValueError, match="Hermitian"):
        solve_subspace(rho, bad, response_hierarchy(model.hamiltonian, words[:3]))


def test_non_hermitian_projected_matrix_is_rejected():
    S = np.eye(2, dtype=complex)
    Hm = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=complex)
    with pytest.raises(ValueError, match="not Hermitian"):
        solve_projected(S, Hm)


def test_mixed_reference_is_rejected_by_both_routes(tfim_case):
    model, _, words, _ = tfim_case
    mixed = 0.5 * ket_density(model.n, "0" * model.n) + 0.5 * ket_density(model.n, "1" * model.n)
    gens = response_hierarchy(model.hamiltonian, words[:3])
    with pytest.raises(ValueError, match="not a pure state"):
        dense_subspace(mixed, model.hamiltonian, gens)
    with pytest.raises(ValueError, match="not a pure state"):
        pure_statevector(mixed)


def test_unnormalized_reference_is_rejected(tfim_case):
    model, rho, words, _ = tfim_case
    with pytest.raises(ValueError, match="unit trace"):
        solve_subspace(2.0 * rho, model.hamiltonian,
                       response_hierarchy(model.hamiltonian, words[:3]))
