"""Phase 16A of ``PLAN.md``: time-evolved sampling states for QSCI.

What is tested is the input, not a verdict about it. Exact propagation must be
exp(-iHt) to working precision by two independent propagators, and against
dense ``expm``. The Trotter circuit must be the product formula
``gates.trotter2_unitary`` already defines, with its error falling at the
declared order. Each state must carry the evidence category that says what
preparing it would take. Pooled multi-time sampling must account for every
shot. Whether sampling at these times helps QSCI is an experiment's question,
and no test here asserts it.
"""

import warnings

import numpy as np
import pytest

from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.gates import trotter2_unitary, trotter_unitary
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.models.spin import tfim
from clifford_qc.multivector import MV
from clifford_qc.pauli_action import PauliLinearOperator
from clifford_qc.subspace import (
    IMPLEMENTABLE, ORACLE, QSCIFragmentSolver, apply_program, propagate,
    reference_determinant_state, run_qsci, sample_state_input,
    sample_state_inputs, time_evolved_state, trotter_program,
)

# The dense expm oracle, and expm_multiply itself, are the research extra.
expm = pytest.importorskip("scipy.linalg").expm

H4 = "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"
TOL = 1e-10


def _backend(model):
    return SectorStatevectorBackend(model.n, model.metadata["n_electrons"],
                                    model.metadata["sz"])


def _dense(operator):
    return np.column_stack([operator.matvec(column)
                            for column in np.eye(operator.dimension)])


@pytest.fixture(scope="module")
def h4():
    model = fcidump_model(H4)
    backend = _backend(model)
    operator = backend.operator(model.hamiltonian)
    return model, backend, operator, _dense(operator)


# ------------------------------------------------------------------ operators

def test_traces_are_exact_on_the_sector_and_the_full_space(h4):
    model, _, operator, dense = h4
    assert operator.trace() == pytest.approx(np.trace(dense), abs=TOL)
    full = PauliLinearOperator(model.hamiltonian)
    assert full.trace() == pytest.approx(
        np.trace(to_matrix(model.hamiltonian.to_mv())), abs=TOL)


def test_linear_operator_views_carry_the_adjoint():
    """``expm_multiply`` calls ``A.H``; a view without it fails inside SciPy."""
    n = 4
    hop = cdag_op(n, 0) * c_op(n, 2)  # non-Hermitian, conserves N and S_z
    x = np.random.default_rng(1).normal(size=2 ** n) + 0j
    full = PauliLinearOperator(hop)
    np.testing.assert_allclose(full.as_linear_operator().rmatvec(x),
                               to_matrix(hop).conj().T @ x, atol=TOL)
    backend = SectorStatevectorBackend(n, 2, 0.0)
    sector = backend.operator(hop, validate_sector=False)
    y = np.random.default_rng(2).normal(size=backend.dimension) + 0j
    np.testing.assert_allclose(sector.as_linear_operator().rmatvec(y),
                               _dense(sector).conj().T @ y, atol=TOL)
    hermitian = hubbard(2).hamiltonian
    full_hermitian = PauliLinearOperator(hermitian)
    sector_hermitian = backend.operator(hermitian)
    assert full_hermitian.adjoint() is full_hermitian
    assert sector_hermitian.adjoint() is sector_hermitian


# ------------------------------------------------------------------ propagation

@pytest.mark.parametrize("method", ["expm_multiply", "lanczos"])
def test_exact_propagation_matches_dense_expm_in_the_sector(h4, method):
    model, backend, operator, dense = h4
    reference = backend.state_from_program(model.reference)
    times = [0.0, 0.3, 1.0, 5.0, -2.0]
    states = propagate(operator, reference, times, method=method)
    for state, t in zip(states, times):
        np.testing.assert_allclose(state, expm(-1j * t * dense) @ reference,
                                   atol=TOL)
        assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("method", ["expm_multiply", "lanczos"])
def test_exact_propagation_matches_dense_expm_in_the_full_space(method):
    model = tfim(3)
    operator = PauliLinearOperator(model.hamiltonian)
    psi = np.random.default_rng(5).normal(size=8) + 1j
    psi /= np.linalg.norm(psi)
    dense = to_matrix(model.hamiltonian.to_mv())
    states = propagate(operator, psi, [0.7, 3.0], method=method)
    for state, t in zip(states, [0.7, 3.0]):
        np.testing.assert_allclose(state, expm(-1j * t * dense) @ psi, atol=TOL)


def test_expm_multiply_is_handed_the_trace(h4):
    """PLAN §5, 16A: the trace is supplied, never estimated by SciPy."""
    model, backend, operator, _ = h4
    reference = backend.state_from_program(model.reference)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        propagate(operator, reference, [1.5], method="expm_multiply")


def test_propagate_refuses_what_it_cannot_propagate(h4):
    model, backend, operator, _ = h4
    reference = backend.state_from_program(model.reference)
    hop = backend.operator(cdag_op(8, 0) * c_op(8, 2), validate_sector=False)
    with pytest.raises(ValueError, match="Hermitian"):
        propagate(hop, reference, [1.0])
    with pytest.raises(ValueError, match="amplitudes"):
        propagate(operator, reference[:-1], [1.0])
    with pytest.raises(ValueError, match="method"):
        propagate(operator, reference, [1.0], method="rk4")
    with pytest.raises(ValueError, match="at least one time"):
        propagate(operator, reference, [])
    with pytest.raises(ValueError, match="finite"):
        propagate(operator, reference, [np.nan])


# ------------------------------------------------------------------ circuits

@pytest.mark.parametrize("order, reference_formula",
                         [(1, trotter_unitary), (2, trotter2_unitary)])
def test_trotter_program_is_the_gates_product_formula(order, reference_formula):
    """Fusing adjacent rotors must not change the unitary it builds."""
    model = hubbard(2)
    mv = model.hamiltonian.to_mv()
    terms = [(mv.terms[code].real, MV(mv.n, {code: 1.0}))
             for code in sorted(mv.terms) if code != 0]
    program = trotter_program(model.hamiltonian, 0.9, steps=3, order=order)
    expected = to_matrix(reference_formula(terms, 0.9, 3))
    np.testing.assert_allclose(to_matrix(program.unitary()), expected, atol=TOL)
    psi = np.random.default_rng(0).normal(size=16) + 0j
    np.testing.assert_allclose(apply_program(program, psi), expected @ psi,
                               atol=TOL)


@pytest.mark.parametrize("order, minimum_ratio", [(1, 10.0), (2, 100.0)])
def test_trotter_error_falls_at_its_declared_order(order, minimum_ratio):
    """Infidelity scales as ``steps^-2p``: 16x at first order, 256x at second."""
    model = hubbard(4)
    backend = _backend(model)
    coarse, fine = (1.0 - time_evolved_state(
        backend, model, 1.0, method="trotter", trotter_steps=steps,
        trotter_order=order).metadata["fidelity_to_exact"]
        for steps in (8, 32))
    assert coarse / fine > minimum_ratio


# ------------------------------------------------------------------ inputs

def test_exact_state_is_an_oracle_and_conserves_energy(h4):
    model, backend, _, _ = h4
    state = time_evolved_state(backend, model, 1.3)
    assert state.category == ORACLE and state.preparations is None
    np.testing.assert_array_equal(state.basis, backend.basis)
    assert state.metadata["energy_drift"] < TOL
    assert state.metadata["norm_defect"] < 1e-12
    at_zero = time_evolved_state(backend, model, 0.0)
    np.testing.assert_array_equal(at_zero.amplitudes,
                                  backend.state_from_program(model.reference))
    with pytest.raises(ValueError, match="method='trotter' only"):
        time_evolved_state(backend, model, 1.0, trotter_steps=4)


def test_trotter_state_is_implementable_and_validated(h4):
    model, backend, _, _ = h4
    state = time_evolved_state(backend, model, 1.0, method="trotter",
                               trotter_steps=8)
    assert state.category == IMPLEMENTABLE and state.preparations == 1
    meta = state.metadata
    assert meta["state_kind"] == "trotter_circuit"
    assert 0.999 < meta["fidelity_to_exact"] <= 1.0 + 1e-12
    assert meta["rotor_count"] > meta["distinct_rotor_words"] > 0
    with pytest.raises(ValueError, match="use more steps"):
        time_evolved_state(backend, model, 1.0, method="trotter",
                           trotter_steps=1, min_fidelity=1.0 - 1e-12)
    with pytest.raises(ValueError, match="needs trotter_steps"):
        time_evolved_state(backend, model, 1.0, method="trotter")


def test_a_leaking_trotter_state_is_post_selected(h4):
    """Single rotors break particle number; the leak is sampled away, not hidden."""
    model, backend, _, _ = h4
    coarse = time_evolved_state(backend, model, 1.0, method="trotter",
                                trotter_steps=2)
    fine = time_evolved_state(backend, model, 1.0, method="trotter",
                              trotter_steps=32)
    assert coarse.metadata["sector_leakage"] > 1e-9
    assert coarse.post_selection is not None
    assert coarse.amplitudes.size == 2 ** model.n
    assert fine.post_selection is None
    assert fine.amplitudes.size == backend.dimension


def test_spin_model_takes_a_non_determinant_reference():
    model = tfim(4)  # reference |++++>, prepared by Hadamards
    state = time_evolved_state(None, model, 0.7)
    assert state.basis is None
    dense = to_matrix(model.hamiltonian.to_mv())
    np.testing.assert_allclose(
        state.amplitudes, expm(-1j * 0.7 * dense) @ np.full(16, 0.25), atol=TOL)
    circuit = time_evolved_state(None, model, 0.7, method="trotter",
                                 trotter_steps=16)
    assert circuit.metadata["fidelity_to_exact"] > 0.9999


# ------------------------------------------------------------------ pooling

def test_pooling_one_state_is_ordinary_sampling(h4):
    model, backend, _, _ = h4
    state = time_evolved_state(backend, model, 1.0)
    single, single_record = sample_state_input(state, shots=300, seed=9)
    pooled, pooled_record = sample_state_inputs([state], shots=300, seed=9)
    np.testing.assert_array_equal(single, pooled)
    for key in ("raw_shots", "accepted_shots", "unique_configurations",
                "singleton_configurations", "bootstrap_set_stability",
                "input_category"):
        assert getattr(single_record, key) == getattr(pooled_record, key)
    assert pooled_record.retained_probability == pytest.approx(
        single_record.retained_probability, abs=1e-12)


def test_pooled_record_accounts_for_every_shot(h4):
    model, backend, operator, dense = h4
    states = [time_evolved_state(backend, model, t) for t in (0.5, 1.0, 2.0)]
    indices, record = sample_state_inputs(states, shots=[100, 200, 300], seed=3)
    assert record.raw_shots == 600
    assert record.accepted_shots + record.discarded_shots == record.raw_shots
    assert record.unique_configurations == indices.size
    assert 0.0 < record.retained_probability <= 1.0
    assert record.input_category == ORACLE
    assert record.state_preparations is None
    assert record.input_label.startswith("pooled[time_evolved(t=0.5)")
    again, _ = sample_state_inputs(states, shots=[100, 200, 300], seed=3)
    np.testing.assert_array_equal(indices, again)
    result = run_qsci(operator, indices, sampling=record)
    assert result.energy >= np.linalg.eigvalsh(dense)[0] - TOL
    reference, _ = sample_state_input(reference_determinant_state(backend, model),
                                      shots=600, seed=3)
    assert indices.size > reference.size


def test_restricted_and_post_selected_states_pool(h4):
    model, backend, operator, _ = h4
    states = [time_evolved_state(backend, model, 1.0, method="trotter",
                                 trotter_steps=steps) for steps in (2, 32)]
    assert [s.post_selection is None for s in states] == [False, True]
    indices, record = sample_state_inputs(states, shots=200, seed=1)
    assert record.input_category == IMPLEMENTABLE
    assert record.state_preparations == 2
    assert record.state_preparation_executions == 400
    assert indices.max() < backend.dimension
    run_qsci(operator, indices, sampling=record)


def test_pooling_refuses_what_does_not_share_an_index_space(h4):
    model, backend, _, _ = h4
    exact = time_evolved_state(backend, model, 1.0)
    circuit = time_evolved_state(backend, model, 1.0, method="trotter",
                                 trotter_steps=32)
    with pytest.raises(ValueError, match="evidence categories"):
        sample_state_inputs([exact, circuit], shots=10)
    full = time_evolved_state(None, model, 1.0)
    with pytest.raises(ValueError, match="one operator"):
        sample_state_inputs([exact, full], shots=10)
    with pytest.raises(ValueError, match="shot counts"):
        sample_state_inputs([exact, exact], shots=[10])
    with pytest.raises(ValueError, match="positive integer"):
        sample_state_inputs([exact], shots=0)
    with pytest.raises(ValueError, match="at least one state"):
        sample_state_inputs([], shots=10)


def test_fragment_solver_takes_a_time_evolved_state(h4):
    model, _, _, _ = h4
    solver = QSCIFragmentSolver(
        state=lambda backend, m: time_evolved_state(backend, m, 1.0), shots=300)
    solution = solver(model)
    assert solution.input_category == ORACLE
    assert all(value < TOL for value in solution.diagnostics().values())
