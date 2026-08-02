"""Orbital bases: the spectrum is invariant, the cost is not.

The whole point of a basis change is that it must not move the physics, so the
central test here is a negative one -- the ground-state energy in every basis
must equal the site-basis value produced by ``lattice.hubbard``, to solver
precision, in the same ``(N, S_z)`` sector.  A rotation that got the two-body
tensor contraction or a Jordan-Wigner string wrong would still return a
plausible-looking energy; it would not return *that* energy.

Everything else is checked against something known analytically rather than
against the module itself:

- the ``P = 2`` Daubechies taps against their published values, and the
  vanishing-moment sum rules against zero;
- ``W W^T = 1`` for every basis, since a silently non-orthogonal rotation is
  the failure mode that changes the spectrum;
- the two-rotor Givens factorisation against ``expm`` of its generator, for
  adjacent *and* non-adjacent modes, since the Jordan-Wigner string between
  distant modes is where this identity would break if it were going to;
- the emitted rotor program against the matrix it is supposed to implement,
  via the single-particle action, and against ``U H U†`` for the spinful form.

The cost claim that motivates the module -- the word count varies by an order
of magnitude across bases at fixed physics -- is asserted too, weakly, so that
a future change collapsing every basis to the same representation is caught.
"""

import math

import numpy as np
import pytest

from clifford_qc.matrix import to_matrix
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.gates import expm_taylor
from clifford_qc.models.lattice import hubbard
from clifford_qc.models.orbital import (OrbitalBasis, as_basis,
                                        daubechies_filter, givens_network,
                                        givens_rotation, momentum_basis,
                                        natural_orbital_basis,
                                        orbital_rotation_program,
                                        quadrature_mirror, rotate_model,
                                        rotate_one_body,
                                        rotate_onsite_interaction,
                                        single_particle_action, site_basis,
                                        wavelet_basis)
from clifford_qc.sparse import sparse_ground_in_sector

D4_TAPS = [0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551]

BASIS_NAMES = ["site", "momentum", "db1", "db2"]


def ring_hopping(sites: int, t: float = 1.0) -> np.ndarray:
    """Periodic nearest-neighbour hopping, matching ``hubbard(..., periodic=True)``."""
    matrix = np.zeros((sites, sites))
    bonds = [(i, i + 1) for i in range(sites - 1)]
    if sites > 2:
        bonds.append((sites - 1, 0))
    for i, j in bonds:
        matrix[i, j] -= t
        matrix[j, i] -= t
    return matrix


def ground_energy(hamiltonian, n_electrons, sz):
    """Sector ground energy. Needs scipy, which the numpy-only core job lacks."""
    pytest.importorskip("scipy")
    values, _ = sparse_ground_in_sector(hamiltonian, n_electrons=n_electrons, sz=sz)
    return float(np.real(values[0]))


# ------------------------------------------------------------------ filters
@pytest.mark.parametrize("moments", [1, 2, 3, 4])
def test_daubechies_filter_is_orthonormal_with_vanishing_moments(moments):
    h = daubechies_filter(moments)
    assert len(h) == 2 * moments
    assert h.sum() == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert np.linalg.norm(h) == pytest.approx(1.0, abs=1e-12)
    g = quadrature_mirror(h)
    for order in range(moments):
        moment = sum(index ** order * g[index] for index in range(len(g)))
        assert abs(moment) < 1e-9


def test_db2_reproduces_published_taps():
    assert daubechies_filter(2) == pytest.approx(D4_TAPS, abs=1e-9)


# ------------------------------------------------------------------- bases
@pytest.mark.parametrize("name", BASIS_NAMES + ["db3"])
def test_every_basis_is_orthogonal(name):
    basis = as_basis(name, 8)
    residual = np.abs(basis.matrix @ basis.matrix.T - np.eye(8)).max()
    assert residual < 1e-12


def test_non_orthogonal_rotation_is_rejected():
    with pytest.raises(ValueError, match="not orthogonal"):
        OrbitalBasis("bad", np.array([[1.0, 0.0], [0.0, 2.0]]))


def test_momentum_basis_diagonalises_ring_hopping():
    rotated = rotate_one_body(ring_hopping(8), momentum_basis(8))
    off_diagonal = rotated - np.diag(np.diag(rotated))
    assert np.abs(off_diagonal).max() < 1e-12


def test_natural_orbitals_diagonalise_a_disordered_chain():
    rng = np.random.default_rng(11)
    one_body = ring_hopping(8) + np.diag(rng.normal(0.0, 3.0, 8))
    rotated = rotate_one_body(one_body, natural_orbital_basis(one_body))
    assert np.abs(rotated - np.diag(np.diag(rotated))).max() < 1e-10


def test_site_basis_interaction_stays_diagonal():
    tensor = rotate_onsite_interaction(4.0, site_basis(4), n_orbitals=4)
    expected = np.zeros((4, 4, 4, 4))
    for site in range(4):
        expected[site, site, site, site] = 4.0
    assert np.abs(tensor - expected).max() < 1e-12


def test_wavelet_basis_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power-of-two"):
        wavelet_basis(6, 2)


# ------------------------------------------------- the invariance that matters
def test_site_basis_reproduces_the_lattice_builder():
    sites = 4
    reference = hubbard(sites, t=1.0, U=4.0, periodic=True)
    rotated = rotate_model(ring_hopping(sites), 4.0, "site")
    assert rotated.metadata["pauli_words"] == len(reference.hamiltonian.terms)
    assert ground_energy(rotated.hamiltonian, 4, 0.0) == pytest.approx(
        ground_energy(reference.hamiltonian, 4, 0.0), abs=1e-9)


@pytest.mark.parametrize("name", BASIS_NAMES)
def test_ground_state_energy_is_basis_invariant(name):
    sites = 4
    hopping = ring_hopping(sites)
    baseline = ground_energy(hubbard(sites, t=1.0, U=4.0, periodic=True).hamiltonian,
                             4, 0.0)
    model = rotate_model(hopping, 4.0, name)
    assert model.metadata["n_electrons"] == 4
    assert model.metadata["sz"] == pytest.approx(0.0)
    assert ground_energy(model.hamiltonian, 4, 0.0) == pytest.approx(
        baseline, abs=1e-9)


def test_natural_orbital_basis_is_invariant_on_a_disordered_chain():
    rng = np.random.default_rng(4)
    sites = 4
    one_body = ring_hopping(sites) + np.diag(rng.normal(0.0, 2.0, sites))
    baseline = ground_energy(rotate_model(one_body, 4.0, "site").hamiltonian, 4, 0.0)
    rotated = rotate_model(one_body, 4.0, natural_orbital_basis(one_body))
    assert ground_energy(rotated.hamiltonian, 4, 0.0) == pytest.approx(
        baseline, abs=1e-9)


def test_word_count_depends_strongly_on_the_basis():
    """The cost claim the module exists to make: the same physics, ~10x the words."""
    hopping = ring_hopping(4)
    counts = {name: rotate_model(hopping, 4.0, name).metadata["pauli_words"]
              for name in BASIS_NAMES}
    assert counts["site"] == min(counts.values())
    assert max(counts.values()) > 5 * counts["site"]


# ------------------------------------------------------------------ circuits
@pytest.mark.parametrize("modes", [(0, 1), (1, 2), (0, 2), (0, 3), (2, 0)])
def test_givens_rotation_is_exactly_two_rotors(modes):
    p, q = modes
    n, theta = 4, 0.37
    generator = cdag_op(n, p) * c_op(n, q) - cdag_op(n, q) * c_op(n, p)
    exact = to_matrix(expm_taylor(theta * generator, order=40))
    program = givens_rotation(n, p, q, theta)
    assert len(program.ops) == 2
    assert np.abs(to_matrix(program.unitary()) - exact).max() < 1e-10


@pytest.mark.parametrize("name", ["momentum", "db1", "db2"])
@pytest.mark.parametrize("modes", [4, 8])
def test_rotor_program_reproduces_its_rotation(name, modes):
    basis = as_basis(name, modes)
    action = single_particle_action(orbital_rotation_program(basis), modes)
    assert np.abs(action - basis.matrix).max() < 1e-9


def test_site_basis_needs_no_rotations():
    program = orbital_rotation_program(site_basis(8))
    assert program.ops == []


def test_givens_network_triangularises():
    basis = wavelet_basis(8, 2)
    steps, signs = givens_network(basis.matrix)
    assert steps, "a non-trivial rotation must produce rotations"
    assert set(np.abs(signs).round(9)) == {1.0}
    assert all(0 <= mode < 7 for mode, _ in steps)


@pytest.mark.parametrize("name", ["momentum", "db1", "db2"])
def test_spinful_program_conjugates_the_site_hamiltonian(name):
    sites = 4
    hopping = ring_hopping(sites)
    site_hamiltonian = to_matrix(rotate_model(hopping, 4.0, "site").hamiltonian.to_mv())
    rotated = to_matrix(rotate_model(hopping, 4.0, name).hamiltonian.to_mv())
    program = orbital_rotation_program(as_basis(name, sites), spinful=True)
    assert program.n == 2 * sites
    unitary = to_matrix(program.unitary())
    conjugated = unitary @ site_hamiltonian @ unitary.conj().T
    assert np.abs(conjugated - rotated).max() < 1e-9
