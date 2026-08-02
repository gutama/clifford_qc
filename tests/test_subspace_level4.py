"""A-CASE §4.2 level 4: compound generators and competing-order configurations.

Levels 0-3 build every basis direction from *one* reference determinant, and on
a half-filled Hubbard cluster that span provably does not contain the ground
state -- the singles-and-doubles subspace saturates a quarter of a Hartree
above it however many directions are added. Level 4 is the plan's answer:
products of generators, and the configurations of the competing orders the
cluster chooses between.

Two things need pinning, and they pull in opposite directions. The
configurations must be cheap and legitimate (one Pauli word each, in the
reference's own symmetry sector), and they must actually enlarge the span --
otherwise they are decoration. The tests below check both, and also record the
selector's blind spot, which is a real limitation rather than a bug.
"""

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.ir import PauliWord
from clifford_qc.models.lattice import bipartition, competing_orders, hubbard
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (compound_response, configuration_generator,
                                  configuration_generators,
                                  configuration_haar_packets,
                                  determinant_excitations, determinant_program,
                                  identity_generator, occupied_spin_orbitals,
                                  pauli_orbit, run_acase, sector_leakage,
                                  solve_subspace, state_sector)


def _words(*labels):
    return [PauliWord.from_label(label) for label in labels]


# --------------------------------------------------------------- compound (4a)


def test_compound_products_are_deduplicated_up_to_a_scalar():
    """``P_i P_j`` and ``P_j P_i`` differ by a sign for anticommuting words, and
    the solver normalizes every generator -- so keeping both would buy a
    duplicate column and a singular overlap matrix."""
    orbit = pauli_orbit(_words("XII", "YII", "ZII"))
    out = compound_response(orbit, orbit)
    labels = [g.label for g in out]
    assert len(labels) == len(set(labels))
    keys = {tuple(sorted(g.mv.terms)) for g in out}
    assert len(keys) == len(out)  # no two survivors span the same direction
    # XY and YX are the same direction up to a sign: only one survives
    assert sum(1 for label in labels if set(label.split("*")) == {"XII", "YII"}) == 1


def test_compound_drops_products_proportional_to_the_identity():
    orbit = pauli_orbit(_words("XII", "IYI"))
    assert all(g.mv.nnz() > 1 or 0 not in g.mv.terms
               for g in compound_response(orbit, orbit))
    # P*P = I for every Pauli word, so the diagonal is pure identity
    assert not compound_response(pauli_orbit(_words("XII")), drop_scalar=True)
    assert compound_response(pauli_orbit(_words("XII")), drop_scalar=False)


def test_compound_budgets_are_declared_not_discovered():
    model = tfim(4, 1.0, 0.7)
    orbit = pauli_orbit(_words("XIII", "IXII", "ZZII", "IZZI"))
    assert len(compound_response(orbit, orbit, max_generators=3)) == 3
    wide = compound_response(orbit, orbit)
    narrow = compound_response(orbit, orbit, max_support=1)
    assert all(g.support() <= 1 for g in narrow)
    assert len(narrow) <= len(wide)
    # deterministic order: left major, right minor
    assert [g.label for g in compound_response(orbit, orbit, max_generators=4)] == \
        [g.label for g in wide][:4]
    assert model.n == 4


def test_compound_rejects_mismatched_algebras():
    with pytest.raises(ValueError, match="different qubit counts"):
        compound_response(pauli_orbit(_words("XI")), pauli_orbit(_words("XII")))


# ---------------------------------------------------- configurations (4b)


def test_configuration_generator_carries_the_reference_onto_the_target():
    """``A = V R'`` must satisfy ``A|psi> = |phi>`` exactly -- checked against
    the dense states A-CASE itself refuses to store."""
    from clifford_qc.matrix import to_matrix
    from clifford_qc.subspace import pure_statevector

    model = hubbard((2, 2))
    target = (0, 3, 5, 6)
    generator = configuration_generator(model.reference, target, label="afm")
    psi = pure_statevector(ExactMVBackend().state(model.reference, ()))
    phi = pure_statevector(ExactMVBackend().state(
        determinant_program(model.n, target), ()))
    moved = to_matrix(generator.mv) @ psi
    assert abs(complex(phi.conj() @ moved)) == pytest.approx(1.0, abs=1e-10)


def test_a_competing_order_costs_one_pauli_word():
    """Determinant to determinant is an X-string, so a whole competing order
    enters the §6 accounting at ``S_A = 1``."""
    model = hubbard((2, 2))
    for generator in configuration_generators(model, competing_orders(model)):
        assert generator.support() == 1


def test_competing_orders_are_emitted_in_the_reference_sector():
    """A configuration in another sector is not a competing order for this
    problem; it is a different problem."""
    model = hubbard((2, 2))
    rho = ExactMVBackend().state(model.reference, ())
    electrons, sz = model.metadata["n_electrons"], model.metadata["sz"]
    orders = competing_orders(model)
    assert {"afm", "afm_flipped", "cdw", "cdw_odd", "stripe"} <= set(orders)
    for name, generator in zip(sorted(orders), configuration_generators(model, orders)):
        sector = state_sector(generator, rho)
        assert sector["particle_number"] == pytest.approx(electrons, abs=1e-9)
        assert sector["sz"] == pytest.approx(sz, abs=1e-9)
        # and sharply so -- a superposition of sectors would show up as variance
        assert sector["particle_number_variance"] == pytest.approx(0.0, abs=1e-12)
        assert sector["sz_variance"] == pytest.approx(0.0, abs=1e-12)


def test_state_sector_and_operator_leakage_disagree_on_configurations():
    """The distinction the plan draws, and the trap it avoids.

    ``sector_leakage`` asks whether the *operator* commutes with ``N``. An
    X-string does not, so a leakage filter rejects every competing-order
    configuration -- while the state it produces has exactly the reference's
    particle number. Configurations must be judged by their sector, not by
    operator-level commutation.
    """
    model = hubbard((2, 2))
    rho = ExactMVBackend().state(model.reference, ())
    generator = configuration_generators(model, competing_orders(model))[0]
    assert max(sector_leakage(generator).values()) > 0.5      # would be rejected
    assert state_sector(generator, rho)["particle_number_variance"] < 1e-12


def test_bipartition_is_the_sublattice_not_the_site_numbering():
    """Neel order is a statement about sublattices. On the 2x2 grid sites 0 and
    3 share a colour, which the site index parity gets wrong."""
    assert bipartition(hubbard((2, 2)).metadata) == ((0, 3), (1, 2))
    assert bipartition(hubbard((2, 3)).metadata) == ((0, 2, 4), (1, 3, 5))


def test_competing_orders_need_a_fermionic_lattice():
    with pytest.raises(ValueError, match="fermionic lattices"):
        competing_orders(tfim(4, 1.0, 0.7))


# ------------------------------------------ configuration-space Haar packets (4c)


def _configuration_leaves(n_qubits=4):
    reference = determinant_program(n_qubits, (0, 1))
    targets = ((0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return reference, [configuration_generator(reference, target, label=f"c{k}")
                       for k, target in enumerate(targets)]


def test_unbalanced_configuration_haar_is_orthogonal_and_preserves_parseval():
    """Five leaves exercise the finite, non-dyadic normalization explicitly."""
    reference, leaves = _configuration_leaves()
    packets = configuration_haar_packets(
        leaves, max_support=None, include_scaling=True)
    assert len(packets) == len(leaves)
    assert packets[0].label == "cfgH[0:5)"
    assert packets[-1].label == "cfgS[0:5)"

    codes = [next(iter(generator.mv.terms)) for generator in leaves]
    transform = np.array([[packet.mv.terms.get(code, 0.0) for code in codes]
                          for packet in packets], dtype=complex)
    assert np.max(np.abs(transform @ transform.conj().T - np.eye(len(leaves)))) \
        < 1e-12

    coefficients = np.arange(1, len(leaves) + 1, dtype=float)
    assert np.vdot(transform @ coefficients, transform @ coefficients).real \
        == pytest.approx(np.vdot(coefficients, coefficients).real, abs=1e-12)

    # The operator coefficient identity transfers to S=I for these distinct
    # determinant configurations, checked on the actual A-CASE reference rho.
    rho = ExactMVBackend().state(reference, ())
    overlap = np.array([
        [(left.mv.dagger() * right.mv * rho).trace()
         for right in packets] for left in packets
    ])
    assert np.max(np.abs(overlap - np.eye(len(packets)))) < 1e-12


def test_configuration_haar_support_cap_prunes_coarse_global_rows():
    reference = determinant_program(5, (0, 1))
    targets = ((0, 2), (0, 3), (0, 4), (1, 2),
               (1, 3), (1, 4), (2, 3), (2, 4))
    leaves = [configuration_generator(reference, target, label=f"c{k}")
              for k, target in enumerate(targets)]
    packets = configuration_haar_packets(
        leaves, max_support=4, include_scaling=True)
    assert len(packets) == 6       # four pair details + two four-leaf details
    assert max(packet.support() for packet in packets) == 4
    assert all("[0:8)" not in packet.label for packet in packets)


def test_configuration_haar_rejects_invalid_budgets_and_duplicate_leaves():
    _, leaves = _configuration_leaves()
    with pytest.raises(ValueError, match="max_support"):
        configuration_haar_packets(leaves, min_support=2, max_support=1)
    with pytest.raises(ValueError, match="distinct directions"):
        configuration_haar_packets(leaves + [leaves[0]])


# ------------------------------------------------- what level 4 buys, and does not


def test_level_four_breaks_the_saturation_levels_0_to_3_cannot():
    """The Q4 result. Every singles-and-doubles direction from one determinant,
    plus the competing-order configurations themselves, still leaves the 2x2
    cluster a quarter Hartree short; the compound family reaches the sector
    ground state exactly.
    """
    pytest.importorskip("scipy")
    from clifford_qc.backends import SectorStatevectorBackend

    model = hubbard((2, 2))
    rho = ExactMVBackend().state(model.reference, ())
    exact = float(SectorStatevectorBackend(
        model.n, model.metadata["n_electrons"],
        model.metadata["sz"]).ground_state(model.hamiltonian, k=1)[0][0])

    identity = identity_generator(model.n)
    excitations = determinant_excitations(model.n, occupied_spin_orbitals(model))
    configurations = configuration_generators(model, competing_orders(model))
    compound = compound_response(configurations, excitations, max_support=64)

    def energy(generators):
        return solve_subspace(rho, model.hamiltonian, generators,
                              track_support=False).ground_energy

    saturated = energy([identity] + excitations)
    with_configurations = energy([identity] + excitations + configurations)
    with_compound = energy([identity] + excitations + configurations + compound)

    assert saturated - exact > 0.2          # levels 0-3 stall well above
    # the configurations alone add nothing once the excitations are saturated:
    # they are determinants H does not connect to the reference at first order
    assert with_configurations == pytest.approx(saturated, abs=1e-9)
    assert with_compound == pytest.approx(exact, abs=1e-8)


def test_the_two_by_two_selector_cannot_see_a_bare_competing_order():
    """A recorded limitation, not a bug.

    A competing-order determinant has *zero* overlap with the reference and
    zero Hamiltonian matrix element to it -- the Hubbard Hamiltonian connects
    determinants differing by one hop, and these differ by a spin flip on two
    sites. The generalized 2x2 score is therefore exactly zero and greedy growth
    never takes one, however useful it would be in combination. Level 4 earns
    its place through the compound products, which do couple at first order.
    """
    pytest.importorskip("scipy")
    model = hubbard((2, 2))
    rho = ExactMVBackend().state(model.reference, ())
    excitations = determinant_excitations(model.n, occupied_spin_orbitals(model))
    configurations = configuration_generators(model, competing_orders(model))

    plain = run_acase(rho, model.hamiltonian, excitations, max_size=9)
    offered = run_acase(rho, model.hamiltonian, excitations + configurations, max_size=9)
    assert list(offered.result.basis_labels) == list(plain.result.basis_labels)
    assert not any(label in dict(competing_orders(model))
                   for label in offered.result.basis_labels)
