"""QASM3 export pass: the emitted decomposition must reproduce the rotor
semantics exactly (verified through the shared abstract lowering)."""

import itertools

import pytest

from clifford_qc import Parameter, Program, to_qasm3
from clifford_qc.ir import PauliWord
from clifford_qc.qasm3 import lower_program, lower_rotor, ops_to_mv


class TestRotorLowering:
    @pytest.mark.parametrize("label", ["X", "Y", "Z"])
    def test_single_qubit(self, label):
        prog = Program(1).rotor(label, 0.731)
        assert ops_to_mv(1, lower_program(prog)).is_close(prog.unitary(), 1e-9)

    @pytest.mark.parametrize("pair", ["".join(p) for p in itertools.product("XYZ", repeat=2)])
    def test_two_qubit_words(self, pair):
        prog = Program(2).rotor(pair, -1.234)
        assert ops_to_mv(2, lower_program(prog)).is_close(prog.unitary(), 1e-9)

    @pytest.mark.parametrize("label", ["XYZ", "IYX", "ZIY", "XIIZ", "YXZY"])
    def test_longer_words_with_identity_gaps(self, label):
        prog = Program(len(label)).rotor(label, 0.5)
        assert ops_to_mv(len(label), lower_program(prog)).is_close(prog.unitary(), 1e-9)

    def test_identity_word_is_global_phase(self):
        prog = Program(2).rotor("II", 1.1)
        assert ops_to_mv(2, lower_program(prog)).is_close(prog.unitary(), 1e-12)

    def test_parity_ladder_structure(self):
        ops = lower_rotor(PauliWord.from_label("ZIZ"), 0.3)
        assert ops == [("cx", 0, 2), ("rz", 0.3, 2), ("cx", 0, 2)]


class TestProgramLowering:
    def test_mixed_program(self):
        prog = (Program(3).clifford("H", 0).clifford("CX", 0, 1)
                .rotor("ZZI", 0.9).clifford("SDG", 2).rotor("IXY", 0.4)
                .clifford("SWAP", 0, 2).clifford("CZ", 1, 2))
        assert ops_to_mv(3, lower_program(prog)).is_close(prog.unitary(), 1e-9)


class TestEmission:
    def test_bound_program_text(self):
        prog = Program(2).clifford("H", 0).clifford("CX", 0, 1).measure_z(0, 1)
        text = to_qasm3(prog)
        assert text.splitlines()[0] == "OPENQASM 3.0;"
        assert 'include "stdgates.inc";' in text
        assert "qubit[2] q;" in text
        assert "h q[0];" in text and "cx q[0], q[1];" in text
        assert "bit[2] c;" in text
        assert "c[0] = measure q[0];" in text and "c[1] = measure q[1];" in text

    def test_unbound_parameters_become_inputs(self):
        prog = Program(1, parameters=["theta"]).rotor("Z", Parameter("theta"))
        text = to_qasm3(prog)
        assert "input float[64] theta;" in text
        assert "rz(theta) q[0];" in text

    def test_bound_parameters_are_inlined(self):
        prog = Program(1, parameters=["theta"]).rotor("Z", Parameter("theta"))
        text = to_qasm3(prog, [0.5])
        assert "input" not in text
        assert "rz(0.5) q[0];" in text

    def test_expectation_task_noted(self):
        prog = Program(1).clifford("H", 0).measure_expectation({"Z": 1.0})
        assert "not representable" in to_qasm3(prog)

    @pytest.mark.parametrize("name", ["bad name", "2theta", "__reserved", "input", "rz", "q"])
    def test_invalid_parameter_name_rejected(self, name):
        prog = Program(1, parameters=[name]).rotor("Z", Parameter(name))
        with pytest.raises(ValueError):
            to_qasm3(prog)

    @pytest.mark.parametrize("name", ["Theta", "_theta", "theta_2", "T"])
    def test_valid_qasm_identifiers_accepted(self, name):
        prog = Program(1, parameters=[name]).rotor("Z", Parameter(name))
        text = to_qasm3(prog)
        assert f"input float[64] {name};" in text
        assert f"rz({name}) q[0];" in text
