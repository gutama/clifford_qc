"""R8: packed Pauli action is the scaling path; dense matrices are the oracle."""

import numpy as np
import pytest

from clifford_qc import MV, P, PauliLinearOperator, Program, apply_pauli_sum
from clifford_qc.dense_reference import exact_ground, to_matrix
from clifford_qc.ir import PauliSum


def test_packed_action_matches_dense_reference():
    rng = np.random.default_rng(8128)
    for n in range(1, 5):
        codes = rng.choice(4 ** n, size=min(12, 4 ** n), replace=False)
        operator = MV(n, {int(code): complex(rng.normal(), rng.normal())
                          for code in codes})
        state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        assert np.allclose(apply_pauli_sum(operator, state),
                           to_matrix(operator) @ state, atol=1e-11)


def test_matmat_and_expectation_match_dense_reference():
    operator = P("XYZ") + (0.3 - 0.2j) * P("IZX") - 0.7 * P("YII")
    compiled = PauliLinearOperator(operator)
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    assert np.allclose(compiled.matmat(vectors), to_matrix(operator) @ vectors,
                       atol=1e-11)
    state = vectors[:, 0]
    expected = np.vdot(state, to_matrix(operator) @ state) / np.vdot(state, state)
    assert compiled.expectation(state) == pytest.approx(expected, abs=1e-11)


def test_matrix_free_lanczos_matches_dense_ground_and_residual():
    hamiltonian = (0.4 * P("XII") - 0.8 * P("IZI") + 0.27 * P("ZZI")
                   + 0.19 * P("IXX") - 0.11 * P("YIY"))
    compiled = PauliLinearOperator(hamiltonian)
    values, vectors = compiled.ground_state(method="lanczos", tol=1e-11,
                                             maxiter=64, seed=3)
    dense_value, _ = exact_ground(hamiltonian)
    residual = compiled.matvec(vectors[:, 0]) - values[0] * vectors[:, 0]
    assert values[0] == pytest.approx(dense_value, abs=1e-10)
    assert np.linalg.norm(residual) < 1e-9


def test_dense_backend_expectation_does_not_materialize_observable(monkeypatch):
    import clifford_qc.backends.dense_statevector as dense_backend

    calls = []
    original = dense_backend.to_matrix

    def counted_to_matrix(operator):
        calls.append(operator)
        return original(operator)

    monkeypatch.setattr(dense_backend, "to_matrix", counted_to_matrix)
    observable = PauliSum.from_labels({"ZI": 0.75, "XX": -0.25})
    value = dense_backend.DenseStatevectorBackend().expectation(Program(2), observable)
    assert value == pytest.approx(0.75, abs=1e-12)
    assert len(calls) == 1  # program unitary only; the observable stays packed


def test_matrix_module_is_only_a_compatibility_facade():
    from clifford_qc import to_matrix as public_to_matrix
    from clifford_qc.dense_reference import to_matrix as reference_to_matrix
    from clifford_qc.matrix import to_matrix as compatibility_to_matrix
    from clifford_qc.pauli_action import word_masks as packed_word_masks
    from clifford_qc.sparse import word_masks as sparse_compatibility_word_masks

    assert public_to_matrix is reference_to_matrix
    assert compatibility_to_matrix is reference_to_matrix
    assert sparse_compatibility_word_masks is packed_word_masks
