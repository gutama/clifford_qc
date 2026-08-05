"""pytket bridge (Phase 4): Rotor -> PauliExpBox lowering, unitary
agreement with the MV backend, IR round-trips, and compilation as a
credibility check."""

import numpy as np
import pytest

pytket = pytest.importorskip("pytket")
from pytket import OpType
from pytket.passes import DecomposeBoxes, FullPeepholeOptimise

from clifford_qc import Parameter, Program, to_matrix
from clifford_qc.bridges.pytket_bridge import program_to_tket, tket_to_program


def assert_same_unitary_up_to_phase(U, V, atol=1e-9):
    k = np.argmax(np.abs(V))
    phase = U.flat[k] / V.flat[k]
    assert abs(abs(phase) - 1) < atol
    assert np.allclose(U, phase * V, atol=atol)


class TestUnitaryAgreement:
    @pytest.mark.parametrize("label,theta", [
        ("X", 0.7), ("Y", -1.2), ("Z", 2.3),
        ("XY", 0.4), ("ZZ", 1.0), ("YX", -0.9),
        ("XYZ", 0.61), ("ZIY", 1.7),
    ])
    def test_single_rotor(self, label, theta):
        prog = Program(len(label)).rotor(label, theta)
        circ = program_to_tket(prog)
        assert_same_unitary_up_to_phase(circ.get_unitary(), to_matrix(prog.unitary()))

    def test_named_cliffords(self):
        prog = (Program(3).clifford("H", 0).clifford("S", 1).clifford("SDG", 2)
                .clifford("CX", 0, 1).clifford("CZ", 1, 2).clifford("SWAP", 0, 2)
                .clifford("X", 0).clifford("Y", 1).clifford("Z", 2))
        circ = program_to_tket(prog)
        assert_same_unitary_up_to_phase(circ.get_unitary(), to_matrix(prog.unitary()))

    def test_mixed_parameterized_program(self):
        a, b = Parameter("a"), Parameter("b")
        prog = (Program(2, parameters=[a, b]).clifford("H", 0).rotor("YI", a)
                .clifford("CX", 0, 1).rotor("ZZ", b).rotor("XY", 0.25))
        circ = program_to_tket(prog, [0.4, 1.1])
        assert_same_unitary_up_to_phase(circ.get_unitary(),
                                        to_matrix(prog.unitary([0.4, 1.1])))

    def test_identity_word_global_phase(self):
        prog = Program(1).rotor("I", 0.8)
        assert_same_unitary_up_to_phase(program_to_tket(prog).get_unitary(),
                                        to_matrix(prog.unitary()))


class TestRoundTrip:
    def test_ir_tket_ir(self):
        prog = (Program(3).clifford("H", 0).rotor("XYI", 0.7)
                .clifford("CX", 1, 2).rotor("ZIZ", -0.4).clifford("SDG", 1))
        back = tket_to_program(program_to_tket(prog))
        assert back.unitary().is_close(prog.unitary(), 1e-9)

    def test_unsupported_gate_rejected(self):
        circ = pytket.Circuit(1)
        circ.add_gate(OpType.T, [0])
        with pytest.raises(ValueError):
            tket_to_program(circ)

    def test_circuit_global_phase_survives_round_trip(self):
        circ = pytket.Circuit(1).H(0)
        circ.add_phase(0.25)
        back = tket_to_program(circ)
        assert np.allclose(to_matrix(back.unitary()), circ.get_unitary(), atol=1e-10)


class TestCompilation:
    def test_compiled_circuit_keeps_semantics(self):
        prog = (Program(3).clifford("H", 0).rotor("XXI", 0.6)
                .clifford("CX", 0, 1).rotor("IZZ", 1.2).rotor("YIY", -0.3))
        circ = program_to_tket(prog)
        DecomposeBoxes().apply(circ)
        FullPeepholeOptimise().apply(circ)
        assert_same_unitary_up_to_phase(circ.get_unitary(), to_matrix(prog.unitary()))
