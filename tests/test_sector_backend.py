"""A-CASE Phase 6: the sector-restricted statevector backend.

Three things need proving, and the plan names them: the backend agrees with the
dense reference where the dense reference exists, the sector projector is a
projector, and the memory is ``C(n,k)`` rather than ``2^n``.

The subtle one is the matvec. Individual Pauli words of a number-conserving
Hamiltonian do *not* conserve particle number, so most of them map part of the
sector outside it and the backend drops those components term by term. That is
exact rather than approximate -- ``H`` commutes with the sector projector, so
projecting each term equals projecting the sum -- and the test that would catch
a mistake compares the matvec against the same Hamiltonian's sparse matrix
restricted to the sector.
"""

import math

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.backends.sector_statevector import (SectorStatevectorBackend,
                                                    lanczos_ground,
                                                    sector_basis,
                                                    sector_projector)
from clifford_qc.fermion import total_number_op, total_sz_op
from clifford_qc.matrix import exact_ground
from clifford_qc.models.lattice import (anderson_impurity, hubbard, kanamori,
                                        kitaev_honeycomb)
from clifford_qc.pauli import P, comm
from clifford_qc.sparse import sector_indices, sparse_ground_in_sector, to_sparse

MODELS = [
    lambda: hubbard(2),
    lambda: hubbard(4),
    lambda: hubbard((2, 2)),
    lambda: kanamori(2, 2),
    lambda: anderson_impurity(2),
]


def backend_for(model) -> SectorStatevectorBackend:
    return SectorStatevectorBackend(model.n, model.metadata["n_electrons"],
                                    model.metadata["sz"])


# ------------------------------------------------------------------ the basis


@pytest.mark.parametrize("n,electrons", [(4, 2), (6, 3), (8, 4), (12, 6)])
def test_sector_basis_size_without_spin_restriction(n, electrons):
    basis = sector_basis(n, electrons)
    assert basis.size == math.comb(n, electrons)
    assert np.all(np.diff(basis) > 0)  # sorted and unique: searchsorted needs it


@pytest.mark.parametrize("n,electrons", [(4, 2), (8, 4), (12, 6), (16, 8)])
def test_spin_restricted_sector_is_a_product_of_two_binomials(n, electrons):
    basis = sector_basis(n, electrons, 0.0)
    assert basis.size == math.comb(n // 2, electrons // 2) ** 2
    assert basis.size < 2 ** n


def test_basis_construction_does_not_enumerate_the_full_space():
    """``C(40, 2) = 780`` states out of ``2^40``: combinatorial, not filtered.

    ``sparse.sector_indices`` cannot do this -- it builds a ``2^n`` index array
    by design, because it masks an already-dense matrix. The difference is the
    whole point of the backend.
    """
    basis = sector_basis(40, 2)
    assert basis.size == math.comb(40, 2)
    assert int(basis.max()) < 2 ** 40


@pytest.mark.parametrize("n,electrons,sz", [(4, 2, 0.0), (6, 3, 0.5), (8, 4, 0.0),
                                            (8, 4, 1.0), (8, 3, None)])
def test_basis_agrees_with_the_dense_index_enumeration(n, electrons, sz):
    """Two independent constructions of the same sector, same bit convention."""
    assert np.array_equal(sector_basis(n, electrons, sz),
                          sector_indices(n, electrons, sz))


def test_impossible_sectors_are_empty_and_rejected():
    assert sector_basis(4, 2, 2.0).size == 0  # needs 4 spin-up orbitals, has 2
    assert sector_basis(4, 3, 0.0).size == 0  # odd electron count, integer S_z
    with pytest.raises(ValueError, match="n_electrons must be"):
        sector_basis(4, 5)
    with pytest.raises(ValueError, match="integer or half-integer"):
        sector_basis(4, 2, 0.3)
    with pytest.raises(ValueError, match="sector is empty"):
        SectorStatevectorBackend(4, 2, 2.0)


# -------------------------------------------------------------- the projector


def test_sector_projector_is_a_projector():
    for n, electrons, sz in ((4, 2, 0.0), (4, 2, None), (6, 3, 0.5)):
        projector = sector_projector(n, electrons, sz)
        assert (projector * projector).is_close(projector, 1e-10)
        assert projector.is_hermitian(1e-12)
        assert projector.trace().real == pytest.approx(
            sector_basis(n, electrons, sz).size, abs=1e-9)


def test_sector_projector_commutes_with_a_conserving_hamiltonian():
    model = hubbard(2, U=4.0)
    projector = sector_projector(model.n, 2, 0.0)
    assert comm(model.hamiltonian.to_mv(), projector).is_zero(1e-10)
    # and not with a word that moves an electron out of the sector
    assert not comm(P("XIII"), projector).is_zero(1e-10)


# ------------------------------------------------------------------ the matvec


@pytest.mark.parametrize("factory", MODELS)
def test_matvec_matches_the_sparse_matrix_restricted_to_the_sector(factory):
    """The term-wise projection argument, checked against the honest submatrix."""
    pytest.importorskip("scipy")
    model = factory()
    backend = backend_for(model)
    block = to_sparse(model.hamiltonian)[backend.basis][:, backend.basis]
    rng = np.random.default_rng(3)
    psi = rng.normal(size=backend.dimension) + 1j * rng.normal(size=backend.dimension)
    operator = backend.operator(model.hamiltonian)
    assert np.abs(operator.matvec(psi) - block @ psi).max() < 1e-10


def test_precompute_and_lean_paths_agree(monkeypatch):
    model = hubbard(4, U=4.0)
    backend = backend_for(model)
    rng = np.random.default_rng(5)
    psi = rng.normal(size=backend.dimension) + 0j
    eager = backend.operator(model.hamiltonian, precompute=True)
    lean = backend.operator(model.hamiltonian, precompute=False)
    assert np.abs(eager.matvec(psi) - lean.matvec(psi)).max() < 1e-12
    assert eager.memory_estimate()["compiled_bytes"] > 0
    assert lean.memory_estimate()["compiled_bytes"] == 0
    assert eager.groups < eager.words  # the X-mask grouping actually groups


def test_x_mask_grouping_collapses_the_diagonal_terms():
    """Every Z-only word shares the empty X-mask, so they cost one pass together."""
    model = hubbard(4, U=4.0)
    operator = backend_for(model).operator(model.hamiltonian)
    assert operator.words == len(model.hamiltonian.terms)
    assert operator.groups <= operator.words
    diagonal_words = sum(1 for code in model.hamiltonian.terms
                         if all(((code >> (2 * j)) & 3) in (0, 3)
                                for j in range(model.n)))
    assert diagonal_words > 1  # they all fold into a single pass


@pytest.mark.parametrize("factory", MODELS)
def test_expectation_on_the_reference_determinant(factory):
    """The determinant reference maps to one basis word, with the same energy."""
    model = factory()
    backend = backend_for(model)
    psi = backend.state_from_program(model.reference)
    assert np.count_nonzero(psi) == 1
    exact = ExactMVBackend().expectation(model.reference, model.hamiltonian, ())
    assert backend.expectation(model.hamiltonian, psi).real == pytest.approx(
        exact, abs=1e-9)
    assert backend.expectation(total_number_op(model.n), psi).real == pytest.approx(
        model.metadata["n_electrons"], abs=1e-9)
    assert backend.expectation(total_sz_op(model.n), psi).real == pytest.approx(
        model.metadata["sz"], abs=1e-9)


def test_expectation_of_a_non_conserving_observable_uses_in_sector_elements():
    """``<psi|O|psi>`` needs only ``P O P``, so a leaking ``O`` is still fine."""
    pytest.importorskip("scipy")
    model = hubbard(2, U=4.0)
    backend = backend_for(model)
    psi = backend.ground_state(model.hamiltonian, k=1)[1][:, 0]
    leaking = P("XIII") + P("IIZY")  # conserves neither N nor S_z
    dense = backend.to_dense(psi)
    expected = complex(dense.conj() @ (to_sparse(leaking) @ dense))
    assert backend.expectation(leaking, psi) == pytest.approx(expected, abs=1e-10)


# ----------------------------------------------------------- ground states


@pytest.mark.parametrize("factory", MODELS)
def test_ground_energy_matches_the_dense_reference(factory):
    """The plan's acceptance criterion for ``n <= 12``."""
    pytest.importorskip("scipy")
    model = factory()
    backend = backend_for(model)
    reference = sparse_ground_in_sector(model.hamiltonian,
                                        model.metadata["n_electrons"],
                                        model.metadata["sz"])[0][0]
    for method in ("eigsh", "lanczos"):
        value = backend.ground_state(model.hamiltonian, k=1, method=method)[0][0]
        assert value == pytest.approx(reference, abs=1e-8)
    # at the particle-hole symmetric point the sector holds the global minimum,
    # so the dense whole-space diagonalization must agree too
    if model.metadata.get("t") is not None:
        assert reference == pytest.approx(
            exact_ground(model.hamiltonian.to_mv())[0], abs=1e-8)


def test_excited_states_match_the_sector_reference():
    pytest.importorskip("scipy")
    model = hubbard((2, 2), U=4.0)
    backend = backend_for(model)
    values, vectors = backend.ground_state(model.hamiltonian, k=3)
    reference = sparse_ground_in_sector(model.hamiltonian, 4, 0.0, k=3)[0]
    assert np.allclose(values, reference, atol=1e-8)
    assert vectors.shape == (backend.dimension, 3)
    overlaps = vectors.conj().T @ vectors
    assert np.abs(overlaps - np.eye(3)).max() < 1e-8


def test_ground_state_eigenvector_is_an_eigenvector():
    model = hubbard(4, U=4.0)
    backend = backend_for(model)
    value, vector = backend.ground_state(model.hamiltonian, k=1, method="lanczos")
    psi = vector[:, 0]
    residual = backend.operator(model.hamiltonian).matvec(psi) - value[0] * psi
    assert np.linalg.norm(residual) / np.linalg.norm(psi) < 1e-6


def test_lanczos_matches_dense_eigh_on_a_random_hermitian_operator():
    """The numpy-only fallback, checked on its own rather than only through a model.

    Full reorthogonalization is what keeps this honest: the bare three-term
    recurrence starts returning spurious duplicates of converged eigenvalues,
    which on a degenerate spectrum is indistinguishable from a real degeneracy.
    """
    rng = np.random.default_rng(11)
    dimension = 60
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension))
    matrix = matrix + matrix.conj().T
    expected = np.linalg.eigvalsh(matrix)[:3]
    values, vectors = lanczos_ground(lambda v: matrix @ v, dimension, k=3,
                                     max_iter=200)
    assert np.allclose(values, expected, atol=1e-8)
    for k in range(3):
        residual = matrix @ vectors[:, k] - values[k] * vectors[:, k]
        assert np.linalg.norm(residual) < 1e-6


def test_degenerate_spectrum_does_not_fool_the_lanczos_fallback():
    """A diagonal operator with a large zero eigenspace -- the spectrum that
    makes ARPACK's ``which='SA'`` report the wrong minimum."""
    model = hubbard(4, t=0.0, U=4.0, mu=0.0)
    backend = backend_for(model)
    value = backend.ground_state(model.hamiltonian, k=1, method="lanczos")[0][0]
    assert value == pytest.approx(0.0, abs=1e-8)
    assert backend.ground_state(model.hamiltonian, k=1)[0][0] == pytest.approx(
        0.0, abs=1e-8)


# ------------------------------------------------------------------- memory


def test_memory_is_the_sector_not_the_full_space():
    backend = SectorStatevectorBackend(16, 8, 0.0)
    report = backend.memory_estimate()
    assert report["sector_amplitudes"] == math.comb(8, 4) ** 2 == 4900
    assert report["full_space_amplitudes"] == 2 ** 16
    assert report["sector_bytes"] * 13 < report["full_space_bytes"]
    assert backend.basis.nbytes < report["full_space_bytes"] // 8


def test_a_sixteen_qubit_cluster_is_solvable_where_dense_is_not():
    """8 sites, 16 qubits: 4900 amplitudes instead of 65536, and no matrix."""
    pytest.importorskip("scipy")
    model = hubbard(8, t=1.0, U=4.0)
    backend = backend_for(model)
    assert backend.dimension == 4900
    value = backend.ground_state(model.hamiltonian, k=1)[0][0]
    # half filling, particle-hole symmetric: energy per site between the
    # atomic (-U/2) and free-fermion limits
    assert -4.0 * model.metadata["sites"] < value < -1.0 * model.metadata["sites"]
    operator = backend.operator(model.hamiltonian)
    assert operator.groups < operator.words


# ------------------------------------------------------------ dense interface


def test_dense_round_trip_and_leak_detection():
    model = hubbard(2, U=4.0)
    backend = backend_for(model)
    # up on site 0 (qubit 0) and down on site 1 (qubit 3): the sector word
    psi = backend.occupation_state("1001")
    full = backend.to_dense(psi)
    assert full.size == 2 ** model.n
    assert np.allclose(backend.from_dense(full), psi)
    leaking = full.copy()
    leaking[0] += 1.0  # weight on the empty determinant, outside the sector
    with pytest.raises(ValueError, match="outside the sector"):
        backend.from_dense(leaking)


def test_input_validation():
    model = hubbard(2, U=4.0)
    backend = backend_for(model)
    with pytest.raises(KeyError, match="outside the sector"):
        backend.index_of("1111")
    with pytest.raises(ValueError, match="different qubit counts"):
        backend.operator(hubbard(4).hamiltonian)
    with pytest.raises(ValueError, match="sector has"):
        backend.operator(model.hamiltonian).matvec(np.ones(3))
    with pytest.raises(ValueError, match="method must be"):
        backend.ground_state(model.hamiltonian, method="magic")
    with pytest.raises(ValueError, match="only X-gate determinant"):
        backend.state_from_program(kitaev_honeycomb(1, 2).reference.clifford("H", 0))
