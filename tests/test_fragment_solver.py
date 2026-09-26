"""Phase 18 of ``PLAN.md``: one fragment-solver callback, four implementations.

The callback's contract is an energy plus one- and two-particle density
matrices, so the tests check the matrices three independent ways rather than
trusting one:

1. entry by entry against ``<psi|a+ a|psi>`` built from the package's own
   ``cdag_op``/``c_op`` multivectors, on a random complex state that is not an
   eigenstate and does not conserve particle number;
2. through the exact identities a fixed-``(N, S_z)`` state obeys;
3. by contracting each solver's spatial RDMs with the FCIDUMP (or effective
   Hamiltonian) integrals and recovering that solver's own energy.

The hybrid's RDMs come from a reconstructed Ritz state, so they are also
checked against the projected route, ``SubspaceResult.transition``, which
never forms that state.

Nothing here asserts that one solver beats another. Accuracy between solvers
is Phase 9's question; this boundary only has to report each solver faithfully.
"""

import json

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.models.effective import load_effective_hamiltonian
from clifford_qc.models.fcidump import fcidump_model, read_fcidump
from clifford_qc.models.lattice import hubbard
from clifford_qc.models.spin import tfim
from clifford_qc.subspace import (
    FRAGMENT_SOLUTION_SCHEMA, ExactFragmentSolver, FragmentSolution,
    FragmentSolver, HybridFragmentSolver, QSCIFragmentSolver,
    SelectedCIFragmentSolver, energy_from_spatial_rdms, spatial_rdms,
    spin_orbital_rdms,
)
from clifford_qc.subspace.hybrid import run_hybrid
from clifford_qc.subspace.qsci import exact_ground_state_oracle, sample_state_input

H4 = "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"
LIH = "benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP"
DIMER = "examples/data/wannier_hubbard_dimer.json"
TOL = 1e-10


@pytest.fixture(scope="module")
def h4():
    return fcidump_model(H4), read_fcidump(H4)


@pytest.fixture(scope="module")
def dimer():
    return load_effective_hamiltonian(DIMER)


def _random_state(n, rng, support=None):
    words = np.arange(2 ** n) if support is None else np.sort(
        rng.choice(2 ** n, size=support, replace=False))
    amplitudes = rng.normal(size=words.size) + 1j * rng.normal(size=words.size)
    return words, amplitudes / np.linalg.norm(amplitudes)


def _dense(words, amplitudes, n):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[words] = amplitudes
    return psi


# ------------------------------------------------------------------ RDM kernel

@pytest.mark.parametrize("support", [None, 6])
def test_rdms_match_operator_expectations_entry_by_entry(support):
    """Every entry against the package's own fermion operators, densely."""
    n = 4
    words, amplitudes = _random_state(n, np.random.default_rng(7), support)
    psi = _dense(words, amplitudes, n)
    gamma, two = spin_orbital_rdms(words, amplitudes, n)
    creators = [cdag_op(n, p) for p in range(n)]
    annihilators = [c_op(n, p) for p in range(n)]
    for p in range(n):
        for q in range(n):
            expected = psi.conj() @ to_matrix(creators[p] * annihilators[q]) @ psi
            assert gamma[p, q] == pytest.approx(expected, abs=TOL)
            for r in range(n):
                for s in range(n):
                    operator = (creators[p] * creators[q]
                                * annihilators[r] * annihilators[s])
                    expected = psi.conj() @ to_matrix(operator) @ psi
                    assert two[p, q, r, s] == pytest.approx(expected, abs=TOL)


def test_rdms_ignore_word_order_and_normalization():
    rng = np.random.default_rng(3)
    words, amplitudes = _random_state(5, rng, support=9)
    order = rng.permutation(words.size)
    first = spin_orbital_rdms(words, amplitudes, 5)
    second = spin_orbital_rdms(words[order], 3.0 * amplitudes[order], 5)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left, right, atol=TOL)


@pytest.mark.parametrize("words, amplitudes, message", [
    ([1, 1], [1.0, 1.0], "distinct"),
    ([0, 1], [0.0, 0.0], "zero state"),
    ([0, 16], [1.0, 1.0], "must lie in"),
    ([0, 1], [1.0], "amplitudes"),
    ([0], [np.nan], "finite"),
])
def test_rdm_kernel_rejects_malformed_states(words, amplitudes, message):
    with pytest.raises(ValueError, match=message):
        spin_orbital_rdms(words, amplitudes, 4)


def test_spatial_rdms_do_not_depend_on_the_spin_ordering(dimer):
    """The same state written interleaved and blocked has one spatial RDM."""
    solution = ExactFragmentSolver()(dimer)
    n = solution.n_spin_orbitals
    # interleaved index 2p+s is blocked index s*(n/2)+p
    to_blocked = np.array([(i % 2) * (n // 2) + i // 2 for i in range(n)])
    inverse = np.argsort(to_blocked)
    blocked1 = solution.rdm1[np.ix_(inverse, inverse)]
    blocked2 = solution.rdm2[np.ix_(inverse, inverse, inverse, inverse)]
    expected = solution.spatial_rdms()
    actual = spatial_rdms(blocked1, blocked2, spin_convention="blocked")
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, atol=TOL)
    with pytest.raises(ValueError, match="spin_convention"):
        spatial_rdms(blocked1, blocked2, spin_convention="alternating")


# ------------------------------------------------------------------ solvers

def _solvers():
    return {
        "exact": ExactFragmentSolver(),
        "qsci_oracle": QSCIFragmentSolver(state="exact_ground_oracle",
                                          shots=400, seed=11),
        "qsci_reference": QSCIFragmentSolver(state="reference_determinant",
                                             shots=16),
        "selected_ci": SelectedCIFragmentSolver(),
        "budget_matched": SelectedCIFragmentSolver(kind="budget_matched",
                                                   max_determinants=6),
    }


@pytest.mark.parametrize("key", sorted(_solvers()))
def test_every_solver_recovers_its_energy_from_fcidump_integrals(h4, key):
    """``E_core + h.D1 + 1/2 (pq|rs).D2`` must be the solver's own energy."""
    model, data = h4
    solver = _solvers()[key]
    solution = solver(model)
    dm1, dm2 = solution.spatial_rdms()
    rebuilt = energy_from_spatial_rdms(data.one_body, data.two_body, dm1, dm2,
                                       core_energy=data.core_energy)
    assert rebuilt == pytest.approx(solution.energy, abs=TOL)
    assert solution.state_energy == pytest.approx(solution.energy, abs=TOL)
    assert all(value < TOL for value in solution.diagnostics().values())
    assert isinstance(solver, FragmentSolver)
    json.dumps(solution.to_record(include_tensors=True))
    exact = ExactFragmentSolver()(model).energy
    assert solution.energy >= exact - TOL


def test_hybrid_recovers_its_energy_from_fcidump_integrals():
    model, data = fcidump_model(LIH), read_fcidump(LIH)
    solution = HybridFragmentSolver(state="exact_ground_oracle", shots=64,
                                    seed=1, max_size=3)(model)
    dm1, dm2 = solution.spatial_rdms()
    rebuilt = energy_from_spatial_rdms(data.one_body, data.two_body, dm1, dm2,
                                       core_energy=data.core_energy)
    assert rebuilt == pytest.approx(solution.energy, abs=TOL)
    assert all(value < TOL for value in solution.diagnostics().values())
    assert solution.rdm_route == "reconstructed_ritz_state"
    assert solution.metadata["reconstruction_norm_defect"] < 1e-8
    json.dumps(solution.to_record(include_tensors=True))


def test_hubbard_energy_from_hopping_and_double_occupancy():
    """The lattice form: hopping from ``D1``, ``U`` from ``Gamma[iu,id,id,iu]``."""
    t, U = 1.0, 4.0
    model = hubbard(4, t=t, U=U)
    mu = model.metadata["mu"]
    for solver in (ExactFragmentSolver(),
                   QSCIFragmentSolver(state="exact_ground_oracle", shots=200,
                                      seed=2)):
        solution = solver(model)
        dm1, _ = solution.spatial_rdms()
        two = solution.rdm2
        hopping = sum(-t * (dm1[i, i + 1] + dm1[i + 1, i]) for i in range(3))
        double = sum(two[2 * i, 2 * i + 1, 2 * i + 1, 2 * i] for i in range(4))
        energy = hopping + U * double - mu * np.trace(dm1)
        assert energy.real == pytest.approx(solution.energy, abs=TOL)


def test_effective_model_energy_from_its_payload(dimer):
    one_body = np.asarray(dimer.metadata["one_body"], dtype=float)
    onsite = dimer.metadata["onsite_u"]
    solution = ExactFragmentSolver()(dimer)
    dm1, _ = solution.spatial_rdms()
    double = [solution.rdm2[2 * i, 2 * i + 1, 2 * i + 1, 2 * i]
              for i in range(len(onsite))]
    energy = np.sum(one_body * dm1) + sum(
        u * d for u, d in zip(onsite, double))
    assert all(d.real > 0.0 for d in double)
    assert energy.real == pytest.approx(solution.energy, abs=TOL)
    # (4 - sqrt(32)) / 2 for t = 1, U = 4 at half filling.
    assert solution.energy == pytest.approx(2.0 - np.sqrt(8.0), abs=TOL)


def test_full_sector_qsci_returns_the_exact_rdms(dimer):
    """Sampling every configuration makes QSCI full CI, RDMs included."""
    exact = ExactFragmentSolver()(dimer)
    assert exact.metadata["ground_gap"] > 1e-6  # unique ground state, unique RDMs
    qsci = QSCIFragmentSolver(state="exact_ground_oracle", shots=4096,
                              seed=0)(dimer)
    assert qsci.metadata["solver_record"]["subspace_dimension"] == 4
    np.testing.assert_allclose(qsci.rdm1, exact.rdm1, atol=TOL)
    np.testing.assert_allclose(qsci.rdm2, exact.rdm2, atol=TOL)


def test_hybrid_rdm_matches_the_projected_route(dimer):
    """The reconstructed Ritz state against ``transition``, which never forms it."""
    solver = HybridFragmentSolver(state="exact_ground_oracle", shots=64, seed=4,
                                  max_size=3, arm="bare_configurations")
    solution = solver(dimer)
    backend = SectorStatevectorBackend(dimer.n, 2, 0.0)
    indices, _ = sample_state_input(
        exact_ground_state_oracle(backend, dimer.hamiltonian),
        shots=64, seed=4)
    words = backend.basis[np.unique(indices)]
    rho = ExactMVBackend().state(dimer.reference, ())
    arm = {arm.name: arm for arm in run_hybrid(rho, dimer, words, max_size=3)}[
        "bare_configurations"]
    assert arm.energy == pytest.approx(solution.energy, abs=TOL)
    for p in range(dimer.n):
        for q in range(dimer.n):
            projected = arm.subspace.transition(
                cdag_op(dimer.n, p) * c_op(dimer.n, q), 0, 0)
            assert solution.rdm1[p, q] == pytest.approx(projected, abs=1e-9)


# ------------------------------------------------------------------ contract

def test_input_categories_travel_with_the_solution(h4):
    model, _ = h4
    categories = {key: solver(model).input_category
                  for key, solver in _solvers().items()}
    assert categories == {"exact": "classical", "qsci_oracle": "oracle",
                          "qsci_reference": "implementable",
                          "selected_ci": "classical",
                          "budget_matched": "classical"}
    sampled = SelectedCIFragmentSolver(state="exact_ground_oracle", shots=64)
    assert sampled(model).input_category == "oracle"


def test_record_is_json_and_round_trips_its_tensors(dimer):
    solution = QSCIFragmentSolver(state="exact_ground_oracle", shots=64,
                                  seed=1)(dimer)
    record = json.loads(json.dumps(solution.to_record(include_tensors=True)))
    assert record["schema"] == FRAGMENT_SOLUTION_SCHEMA
    assert record["solver"] == "qsci"
    assert record["n_spin_orbitals"] == 4
    rebuilt = (np.asarray(record["rdm2"]["real"])
               + 1j * np.asarray(record["rdm2"]["imag"]))
    np.testing.assert_array_equal(rebuilt, solution.rdm2)
    assert "rdm1" not in solution.to_record()


def test_solution_arrays_are_read_only_and_shapes_checked(dimer):
    solution = ExactFragmentSolver()(dimer)
    with pytest.raises(ValueError):
        solution.rdm1[0, 0] = 1.0
    fields = dict(solver="x", energy=0.0, state_energy=0.0,
                  rdm1=np.zeros((4, 4)), rdm2=np.zeros((4,) * 4),
                  n_electrons=2, sz=0.0, spin_convention="interleaved",
                  n_spatial_orbitals=2, input_category="classical",
                  rdm_route="sector_eigenvector")
    FragmentSolution(**fields)
    for key, value, message in [
            ("rdm2", np.zeros((4,) * 3), "shape"),
            ("input_category", "measured", "input_category"),
            ("rdm_route", "guessed", "rdm_route"),
            ("spin_convention", "zigzag", "spin_convention"),
            ("energy", float("nan"), "finite")]:
        with pytest.raises(ValueError, match=message):
            FragmentSolution(**{**fields, key: value})


def test_misconfigured_solvers_are_refused(dimer):
    with pytest.raises(ValueError, match="fermionic"):
        ExactFragmentSolver()(tfim(4))
    with pytest.raises(ValueError, match="named sampling states"):
        QSCIFragmentSolver(state="thermal", shots=10)
    with pytest.raises(ValueError, match="positive integer"):
        QSCIFragmentSolver(state="exact_ground_oracle", shots=0)
    with pytest.raises(ValueError, match="both state and shots"):
        SelectedCIFragmentSolver(state="exact_ground_oracle")
    with pytest.raises(ValueError, match="determinant-selecting"):
        SelectedCIFragmentSolver(kind="random")
    with pytest.raises(ValueError, match="unknown hybrid arm"):
        HybridFragmentSolver(state="exact_ground_oracle", shots=8, arm="best")
    # Sampling only the reference leaves no configuration to build packets from.
    packets = HybridFragmentSolver(state="reference_determinant", shots=8,
                                   arm="packets_then_dressed", max_size=2)
    with pytest.raises(ValueError, match="no 'packets_then_dressed' arm"):
        packets(dimer)
