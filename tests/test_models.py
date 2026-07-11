"""Model builders: golden Hamiltonians and reference states."""

import pytest

from clifford_qc.models import tfim, xxz, random_ising
from clifford_qc.states import expectation


def test_tfim_terms_and_reference():
    m = tfim(3, J=2.0, h=0.5)
    assert m.hamiltonian.to_labels() == {
        "ZZI": -2.0, "IZZ": -2.0, "XII": -0.5, "IXI": -0.5, "IIX": -0.5}
    # |+++> reference: <ZZ>=0, <X>=1 per site
    rho = m.reference.state()
    assert expectation(rho, m.hamiltonian.to_mv()).real == pytest.approx(-1.5)


def test_tfim_periodic_adds_wraparound_bond_only_above_two_sites():
    assert "ZIZ" in tfim(3, periodic=True).hamiltonian.to_labels()
    labels2 = tfim(2, periodic=True).hamiltonian.to_labels()
    assert labels2["ZZ"] == pytest.approx(-1.0)  # not doubled


def test_xxz_terms_and_neel_reference():
    m = xxz(3, J=1.0, delta=2.0)
    labels = m.hamiltonian.to_labels()
    assert labels["XXI"] == pytest.approx(1.0)
    assert labels["YYI"] == pytest.approx(1.0)
    assert labels["ZZI"] == pytest.approx(2.0)
    # Neel |010>: <ZZ> = -1 per bond, <XX> = <YY> = 0
    rho = m.reference.state()
    assert expectation(rho, m.hamiltonian.to_mv()).real == pytest.approx(-4.0)


def test_random_ising_is_seed_reproducible_and_real():
    a = random_ising(4, seed=7)
    b = random_ising(4, seed=7)
    c = random_ising(4, seed=8)
    assert a.hamiltonian.to_labels() == b.hamiltonian.to_labels()
    assert a.hamiltonian.to_labels() != c.hamiltonian.to_labels()
    assert a.hamiltonian.is_hermitian()
    assert len(a.metadata["bond_couplings"]) == 3


def test_hva_layers_cover_all_terms():
    m = tfim(4)
    layer_codes = {w.code for _, words in m.hva_layers for w in words}
    assert layer_codes == set(m.hamiltonian.terms)
