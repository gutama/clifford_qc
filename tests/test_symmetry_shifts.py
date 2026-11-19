"""Sector-annihilating shifts: exactness, certification and exact shot variance."""
import numpy as np
import pytest

from clifford_qc.dense_reference import density_from_statevector
from clifford_qc.fermion import cdag_op, total_number_op
from clifford_qc.ir import PauliWord
from clifford_qc.matrix import to_matrix
from clifford_qc.models import fcidump_model
from clifford_qc.multivector import MV
from clifford_qc.subspace import Generator, MatrixElementBank, identity_generator
from clifford_qc.subspace.symmetry_pools import spin_adapted_excitations
from clifford_qc.subspace.symmetry_shifts import (
    fit_sector_shift, grouped_shot_variance, ritz_word_functional, sector_shift_family,
)

TRI = ('&FCI NORB=3, NELEC=2, MS2=0, ORBSYM=1,1,1, ISYM=1 /\n'
       '0.6 1 1 1 1\n0.5 2 2 2 2\n0.4 3 3 3 3\n0.1 2 1 2 1\n0.1 3 1 3 1\n'
       '0.12 3 2 3 2\n0.3 2 2 1 1\n0.3 3 3 1 1\n0.3 3 3 2 2\n0.05 2 1 1 1\n0.04 3 2 2 2\n'
       '-0.9 1 1 0 0\n-1.0 2 2 0 0\n0.3 3 3 0 0\n0.05 3 2 0 0\n0.07 2 1 0 0\n0.0 0 0 0 0\n')


def sector_state(rng, n_spatial=3, off_diagonal=0.3):
    """A correlated two-electron singlet-sector state on interleaved spin orbitals."""
    n = 2 * n_spatial
    psi = np.zeros(1 << n, complex)
    for a in range(n_spatial):
        for b in range(n_spatial):
            bits = ['0'] * n
            bits[2 * a] = bits[2 * b + 1] = '1'
            psi[int(''.join(bits), 2)] += rng.normal() * (1.0 if a == b else off_diagonal)
    return psi / np.linalg.norm(psi)


@pytest.fixture(scope='module')
def problem(tmp_path_factory):
    path = tmp_path_factory.mktemp('shift') / 'tri.FCIDUMP'
    path.write_text(TRI)
    hamiltonian = fcidump_model(path).hamiltonian.to_mv()
    psi = sector_state(np.random.default_rng(5))
    rho = density_from_statevector(psi)
    generators = [identity_generator(6)] + spin_adapted_excitations(3, (0,))
    exact = MatrixElementBank(rho, hamiltonian, generators).solve()
    return hamiltonian, psi, rho, generators, exact


def test_family_terms_are_hermitian_and_annihilate_the_declared_sector():
    rng = np.random.default_rng(1)
    family = sector_shift_family(3, 2, sz=0.0)
    assert [t.label for t in family][:2] == ['N-Ne', '(N-Ne)N']
    assert len(family) == 2 + 6 + 2
    for _ in range(3):
        psi = sector_state(rng)
        for term in family:
            dense = to_matrix(term.operator)
            np.testing.assert_allclose(dense, dense.conj().T, atol=1e-12)
            np.testing.assert_allclose(dense @ psi, 0, atol=1e-12)
    # Outside the sector the number terms act: they are not identically zero.
    three = np.zeros(64, complex)
    three[int('111000', 2)] = 1
    assert np.linalg.norm(to_matrix(family[0].operator) @ three) > 0.5


@pytest.mark.parametrize('objective', ['hamiltonian', 'ritz'])
def test_fitted_shift_leaves_every_bank_element_and_energy_unchanged(problem, objective):
    hamiltonian, _, rho, generators, exact = problem
    fit = fit_sector_shift(hamiltonian, sector_shift_family(3, 2, sz=0.0), objective=objective,
                           reference=rho, generators=generators,
                           ritz_coefficients=exact.ritz_vector(0), energy=exact.ground_energy)
    assert fit.sector_residual < 1e-12 and fit.commutator_residual < 1e-12
    assert fit.after <= fit.before
    assert any(abs(x) > 1e-6 for x in fit.coefficients.values())
    before = MatrixElementBank(rho, hamiltonian, generators)
    after = MatrixElementBank(rho, fit.hamiltonian, generators)
    for (s0, h0), (s1, h1) in [(before.matrices(), after.matrices())]:
        np.testing.assert_allclose(s1, s0, atol=1e-12)
        np.testing.assert_allclose(h1, h0, atol=1e-12)
    assert after.solve().ground_energy == pytest.approx(exact.ground_energy, abs=1e-12)


def test_ritz_shift_lowers_the_exact_grouped_shot_variance(problem):
    hamiltonian, _, rho, generators, exact = problem
    c, energy = exact.ritz_vector(0), exact.ground_energy
    base = grouped_shot_variance(ritz_word_functional(hamiltonian, generators, c, energy), rho)
    fit = fit_sector_shift(hamiltonian, sector_shift_family(3, 2), objective='ritz',
                           reference=rho, generators=generators,
                           ritz_coefficients=c, energy=energy)
    shifted = grouped_shot_variance(ritz_word_functional(fit.hamiltonian, generators, c, energy),
                                    rho)
    assert shifted['settings'] <= base['settings']
    assert base['neyman'] / shifted['neyman'] > 4
    # The objective is the Neyman variance under its fixed design.
    assert fit.after == pytest.approx(shifted['neyman'], rel=1e-9)


def test_grouped_variance_matches_an_independent_dense_computation(problem):
    hamiltonian, psi, rho, generators, exact = problem
    coefficients = ritz_word_functional(hamiltonian, generators, exact.ritz_vector(0),
                                        exact.ground_energy)
    from clifford_qc.measurement.grouping import qwc_groups
    groups = qwc_groups([PauliWord(6, code) for code in sorted(coefficients)])
    sigmas = []
    for group in groups:
        observable = sum(coefficients[w.code] * to_matrix(w.to_mv()) for w in group)
        mean = np.vdot(psi, observable @ psi).real
        sigmas.append(np.sqrt(max(0.0, np.vdot(psi, observable @ (observable @ psi)).real
                                  - mean ** 2)))
    result = grouped_shot_variance(coefficients, rho)
    assert result['settings'] == len(groups)
    assert result['neyman'] == pytest.approx(sum(sigmas) ** 2, rel=1e-10)
    assert result['uniform'] == pytest.approx(len(groups) * sum(s * s for s in sigmas), rel=1e-10)
    assert grouped_shot_variance({}, rho) == {'words': 0, 'settings': 0,
                                              'neyman': 0.0, 'uniform': 0.0}


def test_a_shift_that_could_change_a_bank_element_is_refused(problem):
    hamiltonian, psi, rho, generators, exact = problem
    family = sector_shift_family(3, 2)
    # A reference with weight outside the two-electron sector.
    leaky = psi.copy()
    leaky[int('111000', 2)] = 0.2
    with pytest.raises(ValueError, match='does not annihilate'):
        fit_sector_shift(hamiltonian, family, reference=density_from_statevector(
            leaky / np.linalg.norm(leaky)), generators=generators)
    # A generator that changes particle number.
    creator = cdag_op(6, 4)
    bad = generators + [Generator('create', creator - creator.dagger())]
    with pytest.raises(ValueError, match='does not annihilate'):
        fit_sector_shift(hamiltonian, family, reference=rho, generators=bad)
    # The tolerance is the caller's, and the refusal names both residuals.
    with pytest.raises(ValueError, match='commutator residual'):
        fit_sector_shift(hamiltonian, family, reference=rho, generators=bad, tolerance=1e-3)


def test_invalid_requests_are_rejected(problem):
    hamiltonian, _, rho, generators, exact = problem
    family = sector_shift_family(3, 2)
    with pytest.raises(ValueError, match='objective'):
        fit_sector_shift(hamiltonian, family, objective='l1', reference=rho, generators=generators)
    with pytest.raises(ValueError, match='empty family'):
        fit_sector_shift(hamiltonian, [], reference=rho, generators=generators)
    with pytest.raises(ValueError, match='required to certify'):
        fit_sector_shift(hamiltonian, family)
    with pytest.raises(ValueError, match='ritz_coefficients and energy'):
        fit_sector_shift(hamiltonian, family, objective='ritz', reference=rho,
                         generators=generators)
    with pytest.raises(ValueError, match='one Ritz coefficient'):
        fit_sector_shift(hamiltonian, family, objective='ritz', reference=rho,
                         generators=generators, ritz_coefficients=[1.0], energy=0.0)
    with pytest.raises(ValueError, match='Hermitian'):
        fit_sector_shift(1j * hamiltonian, family, reference=rho, generators=generators)
    with pytest.raises(ValueError, match='disagree on n'):
        fit_sector_shift(MV.scalar(4, 1.0), family, reference=rho, generators=generators)
    with pytest.raises(ValueError, match='exceeds'):
        sector_shift_family(2, 5)
    with pytest.raises(ValueError, match='spin_ordering'):
        sector_shift_family(2, 2, spin_ordering='qubits')
    with pytest.raises(ValueError, match='finite'):
        sector_shift_family(2, 2, sz=float('nan'))
    assert total_number_op(6).n == 6
