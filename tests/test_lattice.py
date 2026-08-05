"""A-CASE Phase 5: materials models, projected observables, sparse reference.

The models are checked against limits that are known analytically rather than
against themselves: the ``U = 0`` Hubbard chain must reproduce the free-fermion
sum, the ``t = 0`` cluster must reproduce the atomic energy, the free-fermion
double occupancy must be 1/4, and every fermionic Hamiltonian must commute with
particle number and ``S_z``. A construction that gets a hopping sign or a
Jordan-Wigner string wrong fails those; it would pass a self-consistency check.

The sparse reference gets its own scrutiny for a specific reason recorded in
``sparse.py``: asking ARPACK for the smallest algebraic eigenvalue returns the
*wrong* number on a spectrum with a large degenerate ground space, and the
``t = 0`` cluster is exactly that spectrum.
"""

import math

import numpy as np
import pytest

from clifford_qc.fermion import total_number_op, total_sz_op
from clifford_qc.matrix import exact_ground, to_matrix
from clifford_qc.models.lattice import (anderson_impurity, extended_hubbard,
                                        honeycomb_links, hubbard, kanamori,
                                        kitaev_honeycomb, spin_orbital)
from clifford_qc.models.observables import (double_occupancy, link_correlations,
                                            magnetization, occupation,
                                            spin_correlation, structure_factor,
                                            total_spin_squared)
from clifford_qc.models.spin import tfim
from clifford_qc.pauli import P, comm
from clifford_qc.sparse import (sector_indices, sparse_ground,
                                sparse_ground_in_sector, spectral_bound,
                                to_sparse, word_masks)

FERMIONIC = [
    lambda: hubbard(2),
    lambda: hubbard(3),      # odd sites: half filling has S_z = +1/2, not 0
    lambda: hubbard(4),
    lambda: hubbard((2, 2)),
    lambda: extended_hubbard(4, V=1.0),
    lambda: kanamori(2, 2),
    lambda: anderson_impurity(2),
    lambda: anderson_impurity(3),
]


def dense_expectation(observable, psi) -> float:
    return float((psi.conj() @ (to_sparse(observable) @ psi)).real)


# ------------------------------------------------------------ sparse reference


@pytest.mark.parametrize("label", ["XIZ", "YYI", "ZZY", "IXY", "YZXY"])
def test_sparse_words_match_the_dense_bridge(label):
    pytest.importorskip("scipy")
    word = P(label)
    assert np.allclose(to_sparse(word).toarray(), to_matrix(word))


def test_spin_orbital_indexing_is_interleaved():
    """Even spin orbitals up, odd down -- the convention ``total_sz_op`` assumes."""
    assert spin_orbital(0, 0) == 0 and spin_orbital(0, 1) == 1
    assert spin_orbital(3, 0) == 6 and spin_orbital(3, 1) == 7
    assert spin_orbital(1, 0, orbital=1, n_orbitals=2) == 6
    with pytest.raises(ValueError, match="spin must be"):
        spin_orbital(0, 2)
    with pytest.raises(ValueError, match="orbital must be"):
        spin_orbital(0, 0, orbital=3, n_orbitals=2)


def test_word_masks_follow_the_qubit_ordering():
    """Qubit 0 is the most significant index bit, as ``code_to_matrix`` krons."""
    assert word_masks(3, P("XII").terms.popitem()[0])[0] == 0b100
    assert word_masks(3, P("IIX").terms.popitem()[0])[0] == 0b001
    x_mask, z_mask, y_count = word_masks(2, P("YZ").terms.popitem()[0])
    assert (x_mask, z_mask, y_count) == (0b10, 0b11, 1)


def test_sparse_ground_matches_dense_on_spin_models():
    pytest.importorskip("scipy")
    for model in (tfim(4), tfim(6), hubbard(2)):
        values, _ = sparse_ground(model.hamiltonian, k=1)
        assert values[0] == pytest.approx(exact_ground(model.hamiltonian.to_mv())[0],
                                          abs=1e-8)


def test_the_shift_is_what_makes_the_degenerate_case_right():
    """``which='SA'`` returns ``U`` on a matrix whose minimum is zero.

    The ``t = 0``, ``mu = 0`` cluster is diagonal with eigenvalues in
    ``{0, U, 2U, ...}`` and a 256-fold zero eigenspace. ARPACK reports ``U``,
    converged and residual-free -- a residual check cannot catch it, because
    ``U`` genuinely is an eigenvalue. Posing the problem as the extremal
    eigenvalue of a shifted matrix fixes it.
    """
    scipy_sparse = pytest.importorskip("scipy.sparse")
    from scipy.sparse.linalg import eigsh

    model = hubbard(4, t=0.0, U=4.0, mu=0.0)
    matrix = to_sparse(model.hamiltonian)
    naive = eigsh(matrix, k=1, which="SA")[0][0]
    assert naive == pytest.approx(4.0, abs=1e-8)  # the documented failure
    assert sparse_ground(model.hamiltonian, k=1)[0][0] == pytest.approx(0.0, abs=1e-8)
    assert spectral_bound(model.hamiltonian) >= abs(
        exact_ground(model.hamiltonian.to_mv())[0]) - 1e-9


def test_sector_indices_count_the_right_determinants():
    for n, electrons in ((4, 2), (6, 3), (8, 4)):
        assert sector_indices(n, electrons).size == math.comb(n, electrons)
    # N = 2, S_z = 0 on 4 spin orbitals: one up and one down, 2 x 2 choices
    assert sector_indices(4, 2, 0.0).size == 4
    assert sector_indices(4, 2, 1.0).size == 1  # both electrons spin up


def test_sector_restriction_needs_a_conserving_operator():
    pytest.importorskip("scipy")
    model = kitaev_honeycomb(2, 2)
    with pytest.raises(ValueError, match="does not conserve"):
        sparse_ground_in_sector(model.hamiltonian, 4, 0.0)


def test_sector_ground_energy_bounds_the_global_one():
    pytest.importorskip("scipy")
    model = hubbard(4, U=4.0)
    global_value = sparse_ground(model.hamiltonian, k=1)[0][0]
    sector_value = sparse_ground_in_sector(
        model.hamiltonian, model.metadata["n_electrons"], model.metadata["sz"])[0][0]
    assert sector_value >= global_value - 1e-9
    # at the particle-hole symmetric default they coincide: half filling wins
    assert sector_value == pytest.approx(global_value, abs=1e-8)


# ---------------------------------------------------------- model construction


@pytest.mark.parametrize("factory", FERMIONIC)
def test_fermionic_models_are_hermitian_and_conserve_their_symmetries(factory):
    model = factory()
    H = model.hamiltonian.to_mv()
    assert model.hamiltonian.is_hermitian()
    assert H.is_hermitian(1e-12)
    assert comm(H, total_number_op(model.n)).is_zero(1e-10)
    assert comm(H, total_sz_op(model.n)).is_zero(1e-10)


@pytest.mark.parametrize("factory", FERMIONIC)
def test_reference_state_sits_in_the_advertised_sector(factory):
    """The metadata sector has to be the reference's actual sector.

    An odd-site cluster at half filling sits at ``S_z = +1/2``; advertising 0
    would name an empty sector, which is how the Phase-6 backend caught this.
    """
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.states import expectation

    model = factory()
    rho = ExactMVBackend().state(model.reference, ())
    assert expectation(rho, total_number_op(model.n)).real == pytest.approx(
        model.metadata["n_electrons"], abs=1e-9)
    assert expectation(rho, total_sz_op(model.n)).real == pytest.approx(
        model.metadata["sz"], abs=1e-9)


def test_free_fermion_limit_of_the_hubbard_chain():
    """``U = 0``: the open chain's energy is the sum of its filled levels.

    ``eps_k = -2t cos(k pi / (L+1))`` for an open chain of ``L`` sites, doubly
    occupied up to the Fermi level. Nothing about this comes out right if a
    hopping sign or a Jordan-Wigner string is wrong.
    """
    pytest.importorskip("scipy")
    for sites in (2, 4, 6):
        model = hubbard(sites, t=1.0, U=0.0)
        levels = sorted(-2.0 * math.cos(math.pi * k / (sites + 1))
                        for k in range(1, sites + 1))
        expected = 2.0 * sum(level for level in levels if level < 0)
        assert sparse_ground(model.hamiltonian, k=1)[0][0] == pytest.approx(
            expected, abs=1e-8)


def test_atomic_limit_of_the_hubbard_cluster():
    """``t = 0``: energy ``U D - mu N``, so half filling with no double
    occupancy sits at ``-U N / 2`` under the default chemical potential."""
    pytest.importorskip("scipy")
    model = hubbard(4, t=0.0, U=4.0)
    sector = sparse_ground_in_sector(model.hamiltonian, 4, 0.0)[0][0]
    assert sector == pytest.approx(-0.5 * 4.0 * 4, abs=1e-8)


def test_half_filling_is_the_global_minimum_by_default():
    """The reason ``mu`` defaults to ``U/2`` rather than to zero."""
    pytest.importorskip("scipy")
    for factory in (lambda: hubbard(4), lambda: hubbard((2, 2)),
                    lambda: extended_hubbard(4, V=1.0), lambda: kanamori(2, 2)):
        model = factory()
        _, vectors = sparse_ground(model.hamiltonian, k=1)
        electrons = dense_expectation(total_number_op(model.n), vectors[:, 0])
        assert electrons == pytest.approx(model.metadata["n_electrons"], abs=1e-6)
    bare = hubbard(4, U=4.0, mu=0.0)
    _, vectors = sparse_ground(bare.hamiltonian, k=1)
    assert dense_expectation(total_number_op(bare.n), vectors[:, 0]) < 3.5


def test_kanamori_reduces_to_hubbard_at_one_orbital():
    single = kanamori(2, 1, t=1.0, U=4.0, J=0.0)
    reference = hubbard(2, t=1.0, U=4.0)
    assert single.n == reference.n
    assert single.hamiltonian.to_mv().is_close(reference.hamiltonian.to_mv(), 1e-10)


def test_kanamori_atomic_two_electron_multiplets():
    """One two-orbital atom has the rotationally invariant Hund multiplets."""
    model = kanamori(1, 2, t=0.0, U=4.0, J=0.5, mu=0.0)
    indices = sector_indices(model.n, 2)
    block = to_matrix(model.hamiltonian.to_mv())[np.ix_(indices, indices)]
    assert np.linalg.eigvalsh(block) == pytest.approx(
        [2.5, 2.5, 2.5, 3.5, 3.5, 4.5], abs=1e-10)


def test_extended_hubbard_adds_a_neighbour_repulsion():
    plain = hubbard(4, U=4.0)
    extended = extended_hubbard(4, U=4.0, V=1.5)
    difference = extended.hamiltonian.to_mv() - plain.hamiltonian.to_mv()
    assert not difference.is_zero(1e-12)
    assert difference.is_hermitian(1e-12)
    assert extended.metadata["V"] == 1.5
    # V = 0 reproduces the plain model up to the particle-hole constant
    zero = extended_hubbard(4, U=4.0, V=0.0)
    assert zero.hamiltonian.to_mv().is_close(plain.hamiltonian.to_mv(), 1e-10)


def test_odd_site_clusters_advertise_a_half_integer_spin():
    for sites in (3, 5):
        model = hubbard(sites)
        assert model.metadata["n_electrons"] == sites
        assert model.metadata["sz"] == pytest.approx(0.5)
    for sites in (2, 4):
        assert hubbard(sites).metadata["sz"] == pytest.approx(0.0)


def test_anderson_impurity_structure():
    model = anderson_impurity(3, U=4.0, V=0.7)
    assert model.metadata["impurity_site"] == 0
    assert model.metadata["bath_sites"] == [1, 2, 3]
    assert model.metadata["impurity_energy"] == pytest.approx(-2.0)
    # the bath does not talk to itself: no bath-bath hopping bond
    assert all(bond[0] == 0 for bond in model.metadata["bonds"])


def test_honeycomb_links_give_every_bulk_site_one_link_of_each_kind():
    links = honeycomb_links(2, 2, periodic=True)
    kinds = {"x": 0, "y": 0, "z": 0}
    per_site: dict[int, set] = {}
    for kind, i, j in links:
        kinds[kind] += 1
        for site in (i, j):
            per_site.setdefault(site, set()).add(kind)
    assert kinds == {"x": 4, "y": 4, "z": 4}  # one per unit cell per direction
    assert len(per_site) == 8
    assert all(seen == {"x", "y", "z"} for seen in per_site.values())


def test_kitaev_is_a_spin_model_with_one_term_per_link():
    model = kitaev_honeycomb(2, 2, kx=1.0, ky=0.5, kz=2.0)
    assert model.n == 8  # one qubit per site, no Jordan-Wigner doubling
    assert len(model.hamiltonian.terms) == len(model.metadata["links"])
    H = model.hamiltonian.to_mv()
    assert H.is_hermitian(1e-12)
    # a spin model conserves neither fermionic symmetry, and should not pretend to
    assert not comm(H, total_number_op(model.n)).is_zero(1e-10)
    coefficients = {round(abs(c.real), 6) for c in model.hamiltonian.terms.values()}
    assert coefficients == {1.0, 0.5, 2.0}


# ------------------------------------------------------------- observables (§8)


@pytest.fixture(scope="module")
def hubbard_ground():
    pytest.importorskip("scipy")
    model = hubbard(4, U=4.0)
    values, vectors = sparse_ground(model.hamiltonian, k=1)
    return model, values[0], vectors[:, 0]


def test_occupations_sum_to_the_electron_count(hubbard_ground):
    model, _, psi = hubbard_ground
    total = sum(dense_expectation(occupation(model, site), psi)
                for site in range(model.metadata["sites"]))
    assert total == pytest.approx(model.metadata["n_electrons"], abs=1e-6)
    per_spin = sum(dense_expectation(occupation(model, site, spin=spin), psi)
                   for site in range(model.metadata["sites"]) for spin in (0, 1))
    assert per_spin == pytest.approx(total, abs=1e-9)


def test_double_occupancy_is_one_quarter_for_free_fermions():
    """``<n_u n_d> = <n_u><n_d> = 1/4`` at ``U = 0`` and half filling."""
    pytest.importorskip("scipy")
    model = hubbard(4, t=1.0, U=0.0)
    psi = sparse_ground(model.hamiltonian, k=1)[1][:, 0]
    assert dense_expectation(double_occupancy(model), psi) == pytest.approx(0.25, abs=1e-6)


def test_interaction_suppresses_double_occupancy(hubbard_ground):
    model, _, psi = hubbard_ground
    correlated = dense_expectation(double_occupancy(model), psi)
    assert 0.0 <= correlated < 0.25


def test_singlet_ground_state_has_zero_total_spin(hubbard_ground):
    model, _, psi = hubbard_ground
    assert dense_expectation(total_spin_squared(model), psi) == pytest.approx(0.0, abs=1e-6)
    assert dense_expectation(magnetization(model, 0), psi) == pytest.approx(0.0, abs=1e-6)


def test_antiferromagnetic_correlations_and_structure_factor(hubbard_ground):
    model, _, psi = hubbard_ground
    nearest = dense_expectation(spin_correlation(model, 0, 1), psi)
    assert nearest < 0.0  # antiferromagnetic on the neighbouring bond
    factor = dense_expectation(structure_factor(model), psi)
    assert factor > 0.0
    components = sum(dense_expectation(spin_correlation(model, 0, 1, axis=axis), psi)
                     for axis in "xyz")
    assert components == pytest.approx(nearest, abs=1e-9)


def test_kitaev_link_correlations_are_the_bond_operators():
    pytest.importorskip("scipy")
    model = kitaev_honeycomb(2, 2)
    psi = sparse_ground(model.hamiltonian, k=1)[1][:, 0]
    correlations = link_correlations(model)
    assert set(correlations) == {"x", "y", "z"}
    energy = -sum(model.metadata["couplings"][kind]
                  * sum(1 for k, _, _ in model.metadata["links"] if k == kind)
                  * dense_expectation(q, psi)
                  for kind, q in correlations.items())
    assert energy == pytest.approx(sparse_ground(model.hamiltonian, k=1)[0][0], abs=1e-6)


def test_spin_observables_are_rejected_on_the_wrong_model_kind():
    spin_model = kitaev_honeycomb(1, 2)
    with pytest.raises(ValueError, match="fermionic-lattice observable"):
        occupation(spin_model, 0)
    with pytest.raises(ValueError, match="fermionic-lattice observable"):
        double_occupancy(spin_model, 0)
    with pytest.raises(ValueError, match="spin lattice"):
        link_correlations(hubbard(2))
    with pytest.raises(ValueError, match="no lattice metadata"):
        occupation(tfim(2), 0)


def test_projected_observables_on_a_materials_cluster():
    """The §8 route on a Hubbard cluster: no Ritz state is ever formed."""
    pytest.importorskip("scipy")
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.subspace import (MatrixElementBank, dense_basis,
                                      identity_generator, krylov_response)

    model = hubbard(2, U=4.0)
    rho = ExactMVBackend().state(model.reference, ())
    generators = [identity_generator(model.n)] + krylov_response(model.hamiltonian, 3)
    bank = MatrixElementBank(rho, model.hamiltonian, generators)
    result = bank.solve()
    basis = dense_basis(rho, generators)
    psi = basis @ result.ritz_vector(0)
    psi = psi / np.linalg.norm(psi)
    for observable in (occupation(model, 0), double_occupancy(model, 0),
                       spin_correlation(model, 0, 1), total_spin_squared(model)):
        assert result.expectation(observable) == pytest.approx(
            dense_expectation(observable, psi), abs=1e-8)
    # the Krylov space of the reference reaches the sector ground state
    sector = sparse_ground_in_sector(model.hamiltonian, 2, 0.0)[0][0]
    assert result.ground_energy == pytest.approx(sector, abs=1e-8)
    assert result.expectation(total_number_op(model.n)) == pytest.approx(2.0, abs=1e-8)
