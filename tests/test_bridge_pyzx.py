"""PyZX bridge (Phase 5): rotor -> phase-gadget lowering, unitary agreement
with the MV backend (up to global phase), IR round-trips, ZX-calculus
equivalence checking, and monotone optimization."""

import numpy as np
import pytest

pyzx = pytest.importorskip("pyzx")
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, T, ZPhase

from clifford_qc import Parameter, Program, to_matrix
from clifford_qc.bridges.pyzx_bridge import (
    from_pyzx, optimize_program, resource_counts, to_pyzx, verify_equivalent,
)


def assert_same_unitary_up_to_phase(U, V, atol=1e-6):
    assert U.shape == V.shape
    k = np.argmax(np.abs(V))
    phase = U.flat[k] / V.flat[k]
    assert abs(abs(phase) - 1) < atol
    assert np.allclose(U, phase * V, atol=atol)


class TestUnitaryAgreement:
    @pytest.mark.parametrize("label,theta", [
        ("X", 0.7), ("Y", -1.2), ("Z", 2.3),
        ("XY", 0.4), ("ZZ", 1.0), ("YX", -0.9),
        ("XYZ", 0.61), ("ZIY", 1.7), ("YY", 0.3), ("YZXY", 1.234),
    ])
    def test_single_rotor(self, label, theta):
        prog = Program(len(label)).rotor(label, theta)
        assert_same_unitary_up_to_phase(to_pyzx(prog).to_matrix(),
                                        to_matrix(prog.unitary()))

    def test_named_cliffords(self):
        prog = (Program(3).clifford("H", 0).clifford("S", 1).clifford("SDG", 2)
                .clifford("CX", 0, 1).clifford("CZ", 1, 2).clifford("SWAP", 0, 2)
                .clifford("X", 0).clifford("Y", 1).clifford("Z", 2))
        assert_same_unitary_up_to_phase(to_pyzx(prog).to_matrix(),
                                        to_matrix(prog.unitary()))

    def test_mixed_parameterized_program(self):
        a, b = Parameter("a"), Parameter("b")
        prog = (Program(2, parameters=[a, b]).clifford("H", 0).rotor("YI", a)
                .clifford("CX", 0, 1).rotor("ZZ", b).rotor("XY", 0.25))
        assert_same_unitary_up_to_phase(to_pyzx(prog, [0.4, 1.1]).to_matrix(),
                                        to_matrix(prog.unitary([0.4, 1.1])))

    def test_identity_word_is_global_phase(self):
        # exp(-i theta/2 I) is a global phase PyZX does not carry: dropping
        # it leaves an empty (identity) circuit, equal up to phase.
        prog = Program(1).rotor("I", 0.8)
        assert len(to_pyzx(prog).gates) == 0
        assert verify_equivalent(prog, Program(1))


class TestRoundTrip:
    def test_ir_pyzx_ir(self):
        prog = (Program(3).clifford("H", 0).rotor("XYI", 0.7)
                .clifford("CX", 1, 2).rotor("ZIZ", -0.4).clifford("SDG", 1))
        back = from_pyzx(to_pyzx(prog))
        assert_same_unitary_up_to_phase(to_matrix(back.unitary()),
                                        to_matrix(prog.unitary()))

    def test_clifford_plus_t_ingest(self):
        circ = Circuit(2)
        circ.add_gate(HAD(0)); circ.add_gate(T(0)); circ.add_gate(CNOT(0, 1))
        circ.add_gate(ZPhase(1, __import__("fractions").Fraction(1, 4)))
        prog = from_pyzx(circ)
        assert_same_unitary_up_to_phase(to_pyzx(prog).to_matrix(), circ.to_matrix())

    def test_unsupported_gate_rejected(self):
        # A non-unitary Measurement survives to_basic_gates and has no IR image.
        from pyzx.circuit.gates import Measurement
        circ = Circuit(1)
        circ.gates.append(Measurement(0, 0))
        with pytest.raises(ValueError):
            from_pyzx(circ)


class TestEquivalence:
    def test_equivalent_program_variants(self):
        # H-Z-H == X: two spellings of the same unitary.
        a = Program(1).clifford("H", 0).clifford("Z", 0).clifford("H", 0)
        b = Program(1).clifford("X", 0)
        assert verify_equivalent(a, b)
        assert verify_equivalent(a, b, method="dense")
        assert verify_equivalent(a, b, method="both")

    def test_detects_inequivalence(self):
        a = Program(1).rotor("Z", 0.5)
        b = Program(1).rotor("Z", 0.9)
        assert not verify_equivalent(a, b)
        assert not verify_equivalent(a, b, method="dense")

    def test_double_cnot_is_identity(self):
        prog = Program(2).clifford("CX", 0, 1).clifford("CX", 0, 1)
        assert verify_equivalent(prog, Program(2))

    def test_dense_guard_rejects_large_n(self):
        # The dense path must refuse an exponential blowup rather than OOM.
        a = Program(6).clifford("H", 0)
        b = Program(6).clifford("H", 0)
        with pytest.raises(ValueError, match="max_qubits"):
            verify_equivalent(a, b, method="dense", max_qubits=4)
        # Raising the bound lets the same check run.
        assert verify_equivalent(a, b, method="dense", max_qubits=6)


class TestOptimization:
    def test_optimized_is_equivalent(self):
        prog = (Program(3).clifford("H", 0).rotor("XXI", 0.6)
                .clifford("CX", 0, 1).rotor("IZZ", 1.2).rotor("YIY", -0.3))
        opt = optimize_program(prog, objective="two_qubit_count")
        assert verify_equivalent(prog, opt)

    def test_optimization_is_monotone(self):
        # Redundant Cliffords the ZX pipeline should not make worse.
        prog = (Program(2).clifford("H", 0).clifford("H", 0)
                .clifford("CX", 0, 1).clifford("CX", 0, 1).rotor("ZI", 0.7))
        before = resource_counts(prog)["two_qubit_count"]
        opt = optimize_program(prog, objective="two_qubit_count")
        after = resource_counts(opt)["two_qubit_count"]
        assert after <= before
        assert verify_equivalent(prog, opt)

    def test_resource_counts_fields(self):
        prog = Program(2).clifford("H", 0).clifford("CX", 0, 1).rotor("ZI", 0.7)
        counts = resource_counts(prog)
        assert set(counts) == {"gate_count", "two_qubit_count", "t_count", "two_qubit_depth"}
        assert counts["two_qubit_count"] == 1

    def test_bad_objective_rejected(self):
        with pytest.raises(ValueError):
            optimize_program(Program(1).rotor("Z", 0.3), objective="nonsense")
