"""A-CASE on the molecular targets: H2, H4 (equilibrium and stretched),
LiH(2e,2o).

These are the Phase-1 acceptance targets of ``PLAN.md`` §5. The
generators are the *symmetry-preserving* mode of §4.2 -- whole Jordan-Wigner
images of particle-number- and S_z-conserving excitations, one multivector
each, not the individual words a qubit-ADAPT pool would split them into. The
difference is load-bearing and tested here: split words leave the physical
sector, and a subspace built from them can lower its Ritz value by leaking
into states with the wrong electron count.
"""

import numpy as np
import pytest

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

from clifford_qc.backends import ExactMVBackend
from clifford_qc.matrix import exact_ground
from clifford_qc.models.chemistry import (excitation_multivectors,
                                          excitation_pool, h2, h4_chain, lih)
from clifford_qc.pauli import I, Z, comm
from clifford_qc.subspace import (MatrixElementBank, dense_basis,
                                  fermionic_excitation_generators,
                                  identity_generator, run_acase, solve_subspace)

CHEMICAL_ACCURACY = 1.6e-3  # Hartree


def excitation_generators(model):
    """Level 0 + the symmetry-preserving excitation generators of the model."""
    return [identity_generator(model.n)] + fermionic_excitation_generators(
        model.n, model.metadata["n_electrons"])


def reference_state(model):
    return ExactMVBackend().state(model.reference, ())


def number_operator(n):
    out = 0.0 * I(n)
    for j in range(n):
        out = out + 0.5 * (I(n) - Z(n, j))
    return out


def sz_operator(n):
    """Interleaved JW ordering: even spin-orbitals up, odd down."""
    out = 0.0 * I(n)
    for j in range(n):
        out = out + (0.5 if j % 2 == 0 else -0.5) * 0.5 * (I(n) - Z(n, j))
    return out


def sector_weight(psi, n, n_electrons, sz=0.0):
    """Probability that a state vector carries the target ``(N, S_z)``."""
    weight = 0.0
    for index, amplitude in enumerate(psi):
        bits = format(index, f"0{n}b")
        occupied = [j for j, b in enumerate(bits) if b == "1"]
        if len(occupied) != n_electrons:
            continue
        spin = sum(0.5 if j % 2 == 0 else -0.5 for j in occupied)
        if abs(spin - sz) < 1e-12:
            weight += abs(amplitude) ** 2
    return weight / float(np.vdot(psi, psi).real)


@pytest.fixture(scope="module")
def h2_model():
    return h2()


@pytest.fixture(scope="module")
def lih_model():
    return lih()


@pytest.fixture(scope="module")
def h4_models():
    return h4_chain(0.9), h4_chain(1.8)


# ------------------------------------------------------- reproduction targets


def test_h2_subspace_reproduces_fci(h2_model):
    """Four generators, one of them the double excitation that *is* the correlation."""
    result = solve_subspace(reference_state(h2_model), h2_model.hamiltonian,
                            excitation_generators(h2_model))
    assert len(result.basis_labels) == 4
    assert result.ground_energy == pytest.approx(h2_model.metadata["fci_energy"], abs=1e-9)
    assert result.ground_energy < h2_model.metadata["hf_energy"]
    # excitations out of a determinant are mutually orthogonal: S is the identity
    assert result.condition_number == pytest.approx(1.0, abs=1e-9)
    assert result.effective_rank == 4


def test_lih_active_space_subspace_reproduces_fci(lih_model):
    E0, _ = exact_ground(lih_model.hamiltonian.to_mv())
    result = solve_subspace(reference_state(lih_model), lih_model.hamiltonian,
                            excitation_generators(lih_model))
    assert result.ground_energy == pytest.approx(E0, abs=1e-9)
    assert result.ground_energy < lih_model.metadata["hf_energy"]


def test_h4_equilibrium_reaches_chemical_accuracy(h4_models):
    """Go/no-go, equilibrium leg: 27 generators against a 36-state (N=4, S_z=0) sector.

    ``track_support=False`` because the resource accounting is not what this
    test measures: forming all 378 element operators of an 8-qubit,
    185-term Hamiltonian costs seconds, and the two routes are shown to agree
    exactly in ``test_subspace.py``.
    """
    model, _ = h4_models
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    result = solve_subspace(reference_state(model), model.hamiltonian,
                            excitation_generators(model), track_support=False)
    assert len(result.basis_labels) == 27
    assert result.ground_energy >= E0 - 1e-9
    assert result.ground_energy - E0 < CHEMICAL_ACCURACY


def test_stretched_h4_stays_bounded_and_beats_hartree_fock(h4_models):
    """Stretched leg: the bound holds, chemical accuracy does not arrive.

    At r=1.8 a single-reference determinant is a poor starting point and the
    singles/doubles response layer recovers most, but not all, of the
    correlation energy (~3e-2 Ha short). Recorded rather than hidden: closing
    that gap is what the compound generators and adaptive growth of Phases 3-4
    are for, and a fixed basis is not expected to do it.
    """
    _, model = h4_models
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    result = solve_subspace(reference_state(model), model.hamiltonian,
                            excitation_generators(model), track_support=False)
    hf = model.metadata["hf_energy"]
    assert result.ground_energy >= E0 - 1e-9
    assert result.ground_energy < hf - 0.2  # most of the correlation energy
    assert result.ground_energy - E0 > CHEMICAL_ACCURACY  # but not all of it


@pytest.mark.parametrize("index", [0, 1])
def test_h4_nested_growth_is_monotone(h4_models, index):
    model = h4_models[index]
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    gens = excitation_generators(model)
    rho = reference_state(model)
    previous = None
    for size in (1, 5, 12, len(gens)):
        result = solve_subspace(rho, model.hamiltonian, gens[:size],
                                track_support=False)
        assert result.ground_energy >= E0 - 1e-9
        if previous is not None:
            assert result.ground_energy <= previous + 1e-9
        previous = result.ground_energy


# ------------------------------------------------------------ sector control


def test_excitation_multivectors_conserve_particle_number_and_spin(h4_models):
    model, _ = h4_models
    N, Sz = number_operator(model.n), sz_operator(model.n)
    generators = excitation_multivectors(model.n, model.metadata["n_electrons"])
    assert generators
    for _, image in generators:
        T = image.to_mv()
        assert comm(T, N).is_zero(1e-12)
        assert comm(T, Sz).is_zero(1e-12)


def test_split_words_break_the_symmetry_the_whole_image_keeps(h4_models):
    """Why the chemistry default is the whole image, not the word-level pool."""
    model, _ = h4_models
    N = number_operator(model.n)
    words = excitation_pool(model.n, model.metadata["n_electrons"])
    leaking = [op for op in words if not comm(op.word.to_mv(), N).is_zero(1e-12)]
    assert leaking, "expected individual pool words to leave the particle-number sector"


@pytest.mark.parametrize("name", ["h2", "lih"])
def test_projected_number_operator_counts_the_electrons(name, h2_model, lih_model):
    """A physical observable through the §8 route, on a state never formed.

    ``<N>`` is the sharpest available check that the projected-observable API
    and the subspace agree about which sector the Ritz state occupies: it must
    return the electron count exactly, not on average.
    """
    model = h2_model if name == "h2" else lih_model
    bank = MatrixElementBank(reference_state(model), model.hamiltonian,
                             excitation_generators(model))
    result = bank.solve()
    N = number_operator(model.n)
    for k in range(len(result.energies)):
        assert result.expectation(N, k) == pytest.approx(
            float(model.metadata["n_electrons"]), abs=1e-9)
    assert result.expectation(model.hamiltonian) == pytest.approx(
        result.ground_energy, abs=1e-9)
    assert bank.resources()["projected_observables"]


# ---------------------------------------------------------- adaptive growth


@pytest.mark.parametrize("index", [0, 1])
def test_adaptive_growth_beats_the_fixed_excitation_prefix(h4_models, index):
    """Phase-3 validation on H4: adaptive <= fixed at equal size.

    The fixed prefix is not a straw man, it is the natural ordering -- singles
    before doubles, as ``_excitation_generators`` emits them. On a closed-shell
    determinant the singles contribute almost nothing (Brillouin), so the fixed
    prefix stalls while adaptive selection takes the doubles that matter.
    """
    model = h4_models[index]
    E0, _ = exact_ground(model.hamiltonian.to_mv())
    candidates = excitation_generators(model)[1:]  # drop the identity: it is the seed
    rho = reference_state(model)
    result = run_acase(rho, model.hamiltonian, candidates, max_size=4,
                       exact_ground_energy=E0, leakage_tol=1e-9)
    fixed = solve_subspace(rho, model.hamiltonian,
                           [identity_generator(model.n)] + candidates[:4],
                           track_support=False)
    assert result.energy >= E0 - 1e-9
    assert result.energy <= fixed.ground_energy + 1e-9
    assert len(result.labels) == 5
    for record in result.records:
        assert record.actual_lowering >= record.predicted_lowering - 1e-12


def test_every_selected_generator_conserves_the_sector(h4_models):
    """Leakage reporting (§4.2): the whole-image generators leak nothing, and
    the run says so for each accepted step."""
    model, _ = h4_models
    candidates = excitation_generators(model)[1:]
    result = run_acase(reference_state(model), model.hamiltonian, candidates,
                       max_size=3, leakage_tol=1e-9)
    assert result.records
    for record in result.records:
        assert max(record.leakage.values()) < 1e-12
        assert record.rejected_sector == 0
        assert record.condition_number == pytest.approx(1.0, abs=1e-9)


def test_ritz_state_stays_in_the_reference_sector(h4_models):
    """The subspace cannot buy energy by leaving the physical sector.

    Small-``n`` validation only: the Ritz state is reconstructed densely here
    precisely because A-CASE never stores it (§4.4, §8).
    """
    model, _ = h4_models
    gens = excitation_generators(model)
    rho = reference_state(model)
    result = solve_subspace(rho, model.hamiltonian, gens, track_support=False)
    psi = dense_basis(rho, gens) @ result.ritz_vector(0)
    assert sector_weight(psi, model.n, model.metadata["n_electrons"]) == pytest.approx(
        1.0, abs=1e-9)
