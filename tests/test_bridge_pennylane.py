"""PennyLane frontend (Phase 5): expectation agreement and the headline
check — PennyLane parameter-shift gradients versus clifford_qc's exact
adjoint (and parameter-shift) gradients."""

import math

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
from pennylane import numpy as pnp

from clifford_qc import (
    Parameter, PauliSum, Program,
    adjoint_gradient, expectation_value, parameter_shift_gradient,
)
from clifford_qc.bridges.pennylane_bridge import make_qnode, observable_to_pennylane


def ansatz():
    a, b = Parameter("a"), Parameter("b")
    prog = (Program(2, parameters=[a, b]).clifford("H", 0).rotor("YI", a)
            .clifford("CX", 0, 1).rotor("ZZ", b).rotor("XY", 0.25).clifford("SDG", 1))
    obs = PauliSum.from_labels({"ZI": 1.0, "IZ": 0.5, "XX": 0.25})
    return prog, obs


class TestExpectations:
    def test_single_rotor(self):
        prog = Program(1, parameters=["t"]).rotor("Y", Parameter("t"))
        obs = PauliSum.from_labels({"Z": 1.0})
        qnode = make_qnode(prog, obs)
        theta = 0.9
        assert abs(float(qnode(pnp.array([theta]))) - math.cos(theta)) < 1e-9

    def test_ansatz_matches_mv(self):
        prog, obs = ansatz()
        qnode = make_qnode(prog, obs)
        values = [0.4, 1.1]
        assert abs(float(qnode(pnp.array(values))) - expectation_value(prog, obs, values)) < 1e-9

    def test_identity_observable_term(self):
        prog = Program(1).clifford("H", 0)
        obs = PauliSum.from_labels({"I": 2.0, "X": 1.0})
        qnode = make_qnode(prog, obs)
        assert abs(float(qnode(pnp.array([]))) - 3.0) < 1e-9

    def test_non_hermitian_observable_rejected(self):
        with pytest.raises(ValueError):
            observable_to_pennylane(PauliSum.from_labels({"X": 1j}))


class TestGlobalPhase:
    def test_identity_rotor_statevector_matches_mv_exactly(self):
        # qml.GlobalPhase(phi) applies exp(-i phi), so GlobalPhase(theta/2)
        # reproduces the identity rotor exp(-i theta/2) including phase.
        from clifford_qc import to_matrix
        from clifford_qc.bridges.pennylane_bridge import apply_program

        theta = 0.8
        prog = Program(1).rotor("I", theta).rotor("Y", 0.3)
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            apply_program(prog, [])
            return qml.state()

        psi = np.asarray(circuit())
        psi_ref = to_matrix(prog.unitary()) @ np.array([1, 0], complex)
        assert np.allclose(psi, psi_ref, atol=1e-9)  # exact, not just up to phase


class TestGradientComparison:
    def test_pennylane_parameter_shift_vs_our_gradients(self):
        prog, obs = ansatz()
        values = pnp.array([0.4, 1.1], requires_grad=True)
        qnode = make_qnode(prog, obs, diff_method="parameter-shift")
        pl_grad = np.asarray(qml.grad(qnode)(values), dtype=float)
        ours_adjoint = np.array(adjoint_gradient(prog, obs, [0.4, 1.1]))
        ours_shift = np.array(parameter_shift_gradient(prog, obs, [0.4, 1.1]))
        assert np.allclose(pl_grad, ours_adjoint, atol=1e-8)
        assert np.allclose(pl_grad, ours_shift, atol=1e-8)
        assert np.any(np.abs(pl_grad) > 0.05)

    def test_shared_parameter(self):
        t = Parameter("t")
        prog = Program(1, parameters=[t]).rotor("Y", t).rotor("Y", t)
        obs = PauliSum.from_labels({"Z": 1.0})
        qnode = make_qnode(prog, obs, diff_method="parameter-shift")
        values = pnp.array([0.3], requires_grad=True)
        pl_grad = float(qml.grad(qnode)(values)[0])
        assert abs(pl_grad + 2 * math.sin(0.6)) < 1e-9
        assert abs(pl_grad - adjoint_gradient(prog, obs, [0.3])[0]) < 1e-9
