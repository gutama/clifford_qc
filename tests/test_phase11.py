"""Track A Phase 11: overlap targets, multiresolution policy, and sampling stop.

The tests pin the invariants that make the new selectors scientifically
interpretable: overlap scores are scale invariant and S-orthogonal, target and
lowering selection remain independent, packet refinement cannot replace the
complete final pool, all four ordering ablations are declared, and sampling
stability is reported beside an unseen-mass estimate rather than instead of it.
"""

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.pauli import X
from clifford_qc.pauli_action import PauliLinearOperator
from clifford_qc.states import ket_density
from clifford_qc.subspace import (
    ACASEConfig, Generator, MatrixElementBank, OverlapTarget,
    configuration_generator, configuration_ordering, configuration_packet_hierarchy,
    identity_generator, run_acase, run_coarse_to_fine_acase, run_qsci,
    sampling_set_stability, sampling_stop_ready, score_target_overlap,
)
from clifford_qc.subspace.qsci import SamplingRecord, sample_configurations
from clifford_qc.subspace.selected_ci import run_control
from clifford_qc.subspace.fermionic_generators import occupied_spin_orbitals
from clifford_qc.subspace.hybrid import configuration_generators_from_words
from clifford_qc.subspace.qsci import exact_ground_state_oracle, sample_state_input


def _x_generator(n, qubit, label):
    return Generator(label, X(n, qubit))


def _word_generator(n, word):
    mv = identity_generator(n).mv
    for qubit in range(n):
        if (int(word) >> (n - 1 - qubit)) & 1:
            mv = X(n, qubit) * mv
    return Generator(f"word[{int(word)}]", mv)


# --------------------------------------------------------------------- 11A

def test_target_overlap_is_scale_invariant_and_orthogonality_safe():
    n = 2
    rho = ket_density(n, "00")
    hamiltonian = -1.0 * X(n, 0)
    bank = MatrixElementBank(rho, hamiltonian)
    basis = tuple(bank.extend([identity_generator(n)]))
    solved = bank.solve(basis)

    target_generator = _x_generator(n, 1, "target-x1")
    target = OverlapTarget.from_coefficients([target_generator], [7.0j])
    plain = bank.add(_x_generator(n, 1, "x1"))
    scaled = bank.add(Generator("scaled-x1", (3.0 - 4.0j) * X(n, 1)))
    score = score_target_overlap(bank, basis, solved, plain, target)
    rescaled = score_target_overlap(bank, basis, solved, scaled, target)

    assert score.overlap_gain == pytest.approx(1.0, abs=1e-12)
    assert rescaled.overlap_gain == pytest.approx(score.overlap_gain, abs=1e-12)
    assert rescaled.orthogonal_fraction == pytest.approx(
        score.orthogonal_fraction, abs=1e-12)

    duplicate = bank.add(Generator("scaled-I", 9.0 * identity_generator(n).mv))
    rejected = score_target_overlap(bank, basis, solved, duplicate, target)
    assert not rejected.accepted
    assert rejected.rejected.startswith("orthogonal fraction")


def test_target_selector_is_independent_of_lowering_selector():
    n = 2
    rho = ket_density(n, "00")
    hamiltonian = -1.0 * X(n, 0)
    candidates = [_x_generator(n, 0, "x0"), _x_generator(n, 1, "x1")]

    lowering = run_acase(rho, hamiltonian, candidates, max_size=1,
                         min_lowering=0.0)
    assert lowering.records[0].selected_label == "x0"

    target = OverlapTarget.from_coefficients([candidates[1]], [1.0], label="x1")
    targeted = run_acase(
        rho, hamiltonian, candidates, max_size=1,
        criterion="target_overlap", target=target)
    record = targeted.records[0]
    assert record.selected_label == "x1"
    assert record.selection_criterion == "target_overlap"
    assert record.target_overlap_gain == pytest.approx(1.0, abs=1e-12)
    assert record.predicted_lowering == pytest.approx(0.0, abs=1e-12)


def test_qsci_and_selected_ci_expose_vectors_for_overlap_targets():
    n = 2
    hamiltonian = -1.0 * X(n, 0)
    operator = PauliLinearOperator(hamiltonian)
    sampling = SamplingRecord(8, 8, 0, 0, 2, 1.0, "test", 0)
    indices = np.array([0, 2], dtype=np.int64)
    baseline = run_qsci(operator, indices, sampling=sampling, k=2)
    assert baseline.eigenvectors is None
    qsci = run_qsci(
        operator, indices, sampling=sampling, k=2, want_eigenvectors=True)
    restricted = operator.restrict(indices)
    assert qsci.eigenvectors.base is None
    assert np.allclose(qsci.eigenvectors.conj().T @ qsci.eigenvectors,
                       np.eye(2), atol=1e-12)
    assert np.allclose(restricted @ qsci.eigenvectors,
                       qsci.eigenvectors * qsci.eigenvalues, atol=1e-12)
    qsci_target = OverlapTarget.from_qsci(
        [_word_generator(n, word) for word in indices], qsci)
    assert qsci_target.label == "qsci_root_0"

    control = run_control(operator, np.array([0]), name="selected",
                          kind="selected_ci", n=n, max_determinants=2)
    assert control.coefficients is not None
    assert control.coefficients.base is None
    assert np.linalg.norm(control.coefficients) == pytest.approx(1.0, abs=1e-12)
    ci_target = OverlapTarget.from_selected_ci(
        [_word_generator(n, word) for word in control.determinants], control)
    assert len(ci_target.coefficients) == control.determinant_count


def test_hubbard_2x3_overlap_target_exposes_a_lowering_selector_blind_spot():
    """The Phase-11 go/no-go, under a clearly labelled oracle validation input.

    Both A-CASE runs see the same sampled configuration family and receive six
    growth decisions.  The QSCI target comes from exact-ground-state sampling,
    so this test establishes selector capability only; it is not evidence for
    an implementable state-preparation advantage.
    """
    model = hubbard(shape=(2, 3), t=1.0, U=4.0)
    backend = SectorStatevectorBackend(
        model.n, int(model.metadata["n_electrons"]), float(model.metadata["sz"]))
    operator = backend.operator(model.hamiltonian)
    indices, sampling = sample_state_input(
        exact_ground_state_oracle(backend, model.hamiltonian), shots=64, seed=11)
    qsci = run_qsci(
        operator, indices, sampling=sampling, k=1, want_eigenvectors=True)
    words = backend.basis[indices]

    reference = frozenset(occupied_spin_orbitals(model))
    target_generators = []
    for position, word in enumerate(words.tolist()):
        occupied = [j for j in range(model.n)
                    if (int(word) >> (model.n - 1 - j)) & 1]
        if frozenset(occupied) == reference:
            target_generators.append(identity_generator(model.n))
        else:
            target_generators.append(configuration_generator(
                model.reference, occupied, label=f"target[{position}]"))
    target = OverlapTarget.from_qsci(target_generators, qsci)
    candidates = configuration_generators_from_words(model, words)
    rho = ExactMVBackend().state(model.reference, ())

    lowering = run_acase(
        rho, model.hamiltonian, candidates, max_size=6, min_lowering=0.0)
    targeted = run_acase(
        rho, model.hamiltonian, candidates, max_size=6,
        criterion="target_overlap", target=target, min_target_overlap=0.0)

    assert lowering.labels != targeted.labels
    assert targeted.energy < lowering.energy - 0.05
    assert targeted.energy == pytest.approx(-13.039089991675105, abs=1e-9)
    assert qsci.energy == pytest.approx(-13.567439492403267, abs=1e-9)


# ---------------------------------------------------------------- 11B / 11C

def test_all_four_configuration_orderings_are_explicit_and_deterministic():
    words = np.array([12, 9, 6, 3], dtype=np.int64)  # four 2-electron words
    probabilities = np.array([0.20, 0.10, 0.60, 0.10])

    probability = configuration_ordering(
        words, method="probability", probabilities=probabilities)
    assert probability.words[0] == 6

    physics = configuration_ordering(
        words, method="physics", reference_word=12, n=4)
    ranks = [((int(word) ^ 12).bit_count() // 2) for word in physics.words]
    assert ranks == sorted(ranks)
    assert physics.words[0] == 12

    graph = np.zeros((4, 4), dtype=float)
    graph[0, 1] = graph[1, 0] = 0.9
    graph[0, 2] = graph[2, 0] = 0.2
    graph[1, 3] = graph[3, 1] = 0.8
    traversal = configuration_ordering(
        words, method="graph", reference_word=12, graph_matrix=graph)
    assert traversal.words.tolist() == [12, 9, 3, 6]

    random_a = configuration_ordering(words, method="random", seed=17)
    random_b = configuration_ordering(words, method="random", seed=17)
    assert np.array_equal(random_a.words, random_b.words)
    assert sorted(random_a.words.tolist()) == sorted(words.tolist())


def test_packet_hierarchy_refines_then_hands_off_to_complete_pool():
    n = 4
    rho = ket_density(n, "0000")
    coefficients = (1.0, 0.7, 0.2, 0.05)
    hamiltonian = sum(((-weight) * X(n, qubit)
                       for qubit, weight in enumerate(coefficients)),
                      0.0 * X(n, 0))
    configurations = [_x_generator(n, qubit, f"cfg[{qubit}]")
                      for qubit in range(n)]
    nodes = configuration_packet_hierarchy(
        configurations, max_support=4, label_prefix="testMR")
    root = next(node for node in nodes if node.interval == (0, 4))
    assert root.eligible and root.depth == 0
    assert any(node.parent == (0, 4) for node in nodes)

    result = run_coarse_to_fine_acase(
        rho, hamiltonian, configurations, configurations,
        config=ACASEConfig(max_size=4, min_lowering=0.0),
        packet_steps=2, max_packet_support=4,
        competitive_ratio=0.0, max_competitive=2,
        label_prefix="testMR")
    assert result.packet_labels
    assert (0, 4) in result.refined_intervals
    assert result.metadata["final_growth_budget"] == 4 - len(result.packet_labels)
    assert any(label.startswith("cfg[") for label in result.result.labels)
    # The representation remains an ordinary Hilbert-space Rayleigh-Ritz
    # subspace: it cannot beat the exact ground energy -sum |weights|.
    exact = -sum(abs(value) for value in coefficients)
    assert result.result.energy >= exact - 1e-10
    if result.packet_energy_history:
        assert result.result.energy <= result.packet_energy_history[-1] + 1e-10


def test_packet_hierarchy_dedup_uses_bank_index_not_generator_label():
    n = 2
    rho = ket_density(n, "00")
    hamiltonian = -1.0 * X(n, 0) - 0.5 * X(n, 1)
    configurations = [_x_generator(n, qubit, f"cfg[{qubit}]")
                      for qubit in range(n)]
    bank = MatrixElementBank(rho, hamiltonian)
    config = ACASEConfig(max_size=1, min_lowering=0.0)

    first = run_coarse_to_fine_acase(
        rho, hamiltonian, configurations, configurations, bank=bank,
        config=config, packet_steps=1, label_prefix="first")
    second = run_coarse_to_fine_acase(
        rho, hamiltonian, configurations, configurations, bank=bank,
        config=config, packet_steps=1, label_prefix="second")

    assert first.packet_labels
    assert second.packet_labels
    assert second.packet_labels[0].startswith("second")


def test_packet_lowering_gate_is_unpenalized_but_frontier_ranking_is_not():
    n = 2
    rho = ket_density(n, "00")
    hamiltonian = -1.0 * X(n, 0)
    configurations = [_x_generator(n, qubit, f"cfg[{qubit}]")
                      for qubit in range(n)]
    result = run_coarse_to_fine_acase(
        rho, hamiltonian, configurations, configurations,
        config=ACASEConfig(max_size=1, min_lowering=0.3, gamma=2.0),
        packet_steps=1, label_prefix="gamma")

    assert result.packet_labels
    # One live root is scored once by the packet policy and once by run_acase;
    # the resource counter must report both passes.
    assert result.frontiers_scored == 2


def test_target_context_is_prepared_once_per_growth_step(monkeypatch):
    import clifford_qc.subspace.adaptive as adaptive_module

    n = 3
    rho = ket_density(n, "000")
    hamiltonian = -1.0 * X(n, 0)
    candidates = [_x_generator(n, qubit, f"x{qubit}") for qubit in range(n)]
    target = OverlapTarget.from_coefficients([candidates[1]], [1.0])
    original = adaptive_module._prepare_target_overlap_context
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(adaptive_module, "_prepare_target_overlap_context", counted)
    run_acase(
        rho, hamiltonian, candidates, max_size=1,
        criterion="target_overlap", target=target, min_target_overlap=0.0)
    assert calls == 1


# --------------------------------------------------------------------- 11D

def test_sampling_records_duplicate_bootstrap_and_unseen_mass_baselines():
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    _, saturated = sample_configurations(psi, shots=32, seed=4)
    assert saturated.duplicate_fraction == pytest.approx(31 / 32)
    assert saturated.singleton_configurations == 0
    assert saturated.unseen_mass_estimate == 0.0
    assert saturated.bootstrap_set_stability == pytest.approx(1.0)
    assert sampling_stop_ready(
        saturated, max_unseen_mass=0.0, min_set_stability=1.0,
        min_duplicate_fraction=0.9)

    samples = np.array([0, 0, 0, 1, 2, 3], dtype=np.int64)
    stability = sampling_set_stability(samples, replicates=64, seed=9)
    assert 0.0 < stability < 1.0


def test_sampling_bootstrap_can_be_disabled_without_relaxing_stop_rule():
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    _, record = sample_configurations(
        psi, shots=16, seed=4, stability_bootstrap=0)
    assert record.bootstrap_set_stability is None
    assert not sampling_stop_ready(
        record, max_unseen_mass=0.0, min_set_stability=0.0)


def test_good_turing_singletons_are_counted_before_occupancy_repair():
    n = 4
    shots = 32
    seed = 13
    psi = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    probabilities = np.abs(psi) ** 2
    draws = np.random.default_rng(seed).choice(
        probabilities.size, size=shots, p=probabilities)
    _, counts = np.unique(draws, return_counts=True)
    expected_singletons = int(np.count_nonzero(counts == 1))

    _, record = sample_configurations(
        psi, shots=shots, seed=seed, n=n, n_electrons=2,
        recovery="occupancy", stability_bootstrap=0)
    assert record.repaired_shots > 0
    assert record.singleton_configurations == expected_singletons
    assert record.unseen_mass_estimate == pytest.approx(expected_singletons / shots)
