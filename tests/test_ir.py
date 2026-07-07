"""Tests for the Pauli-rotor IR (Phase 2)."""

import math

import numpy as np
import pytest

from clifford_qc import (
    P, rotor,
    PauliWord, PauliSum, Parameter, ParameterGroup, NamedClifford,
    MeasurementTask, Program, conjugate_pauli_word,
    expectation_value, parameter_shift_gradient, adjoint_gradient,
    H, S, CNOT, SWAP,
)


class TestPauliWord:
    def test_label_roundtrip(self):
        w = PauliWord.from_label("XIZY")
        assert w.label == "XIZY"
        assert w.weight == 3
        assert w.support() == (0, 2, 3)
        assert w.letter(2) == "Z"

    def test_to_mv(self):
        assert PauliWord.from_label("XZ").to_mv(2.0).is_close(2 * P("XZ"))

    def test_validation(self):
        with pytest.raises(ValueError):
            PauliWord(2, 16)


class TestPauliSum:
    def test_roundtrip_mv(self):
        A = P("XI") + 0.5 * P("ZZ") - 2j * P("YX")
        assert PauliSum.from_mv(A).to_mv().is_close(A)

    def test_from_labels(self):
        ps = PauliSum.from_labels({"XX": 1.0, "ZI": -0.5})
        assert ps.to_labels() == {"XX": 1 + 0j, "ZI": -0.5 + 0j}
        assert ps.is_hermitian()
        assert not PauliSum.from_labels({"XX": 1j}).is_hermitian()

    def test_arithmetic(self):
        a = PauliSum.from_labels({"X": 1.0})
        b = PauliSum.from_labels({"Z": 2.0})
        assert (a + 0.5 * b).to_labels() == {"X": 1 + 0j, "Z": 1 + 0j}


class TestParameters:
    def test_bind_sequence_and_mapping(self):
        g = ParameterGroup(["a", "b"])
        assert g.bind([1.0, 2.0]) == {"a": 1.0, "b": 2.0}
        assert g.bind({"a": 1.0, "b": 2.0}) == {"a": 1.0, "b": 2.0}

    def test_bind_errors(self):
        g = ParameterGroup(["a"])
        with pytest.raises(ValueError):
            g.bind([])
        with pytest.raises(ValueError):
            g.bind({"c": 1.0})
        with pytest.raises(ValueError):
            ParameterGroup(["a", "a"])


class TestProgramSemantics:
    def test_unitary_matches_gate_composition(self):
        prog = (Program(2).clifford("H", 0).clifford("CX", 0, 1)
                .rotor("ZZ", 0.7).clifford("SWAP", 0, 1))
        expected = SWAP(2, 0, 1) * rotor(P("ZZ"), 0.7) * CNOT(2, 0, 1) * H(2, 0)
        assert prog.unitary().is_close(expected, 1e-12)

    def test_all_named_cliffords_are_unitary(self):
        for name in ("X", "Y", "Z", "H", "S", "SDG"):
            assert NamedClifford(name, (0,)).to_mv(2).is_unitary()
        for name in ("CX", "CZ", "SWAP"):
            assert NamedClifford(name, (0, 1)).to_mv(2).is_unitary()

    def test_sdg_is_s_dagger(self):
        assert NamedClifford("SDG", (0,)).to_mv(1).is_close(S(1, 0).dagger())

    def test_auto_registers_parameters(self):
        prog = Program(1).rotor("Y", Parameter("t"))
        assert prog.parameters.names == ("t",)
        prog.measure_expectation({"Z": 1.0})
        assert abs(prog.run([0.9])[0] - math.cos(0.9)) < 1e-12

    def test_run_expectation_and_sampling(self):
        prog = (Program(2).clifford("H", 0).clifford("CX", 0, 1)
                .measure_expectation({"XX": 1.0}).measure_z(0, 1))
        exp_xx, probs = prog.run()
        assert abs(exp_xx - 1.0) < 1e-12
        assert abs(probs["00"] - 0.5) < 1e-12 and abs(probs["11"] - 0.5) < 1e-12

    def test_sample_z_partial(self):
        prog = Program(2).clifford("H", 0).clifford("CX", 0, 1).measure_z(1)
        probs, = prog.run()
        assert abs(probs["0"] - 0.5) < 1e-12 and abs(probs["1"] - 0.5) < 1e-12

    def test_sample_z_rejects_unordered_or_duplicate_qubits(self):
        # bitstrings list qubits in ascending order, so permutations and
        # duplicates would be silently ambiguous
        with pytest.raises(ValueError):
            Program(3).measure_z(2, 0)
        with pytest.raises(ValueError):
            Program(3).measure_z(1, 1)

    def test_is_clifford_only(self):
        assert Program(1, [NamedClifford("H", (0,))]).is_clifford_only()
        assert not Program(1).rotor("Z", 0.1).is_clifford_only()

    def test_validation(self):
        with pytest.raises(ValueError):
            Program(1).clifford("CX", 0, 1)
        with pytest.raises(ValueError):
            Program(2).rotor("X", 0.1)  # wrong word length
        with pytest.raises(ValueError):
            NamedClifford("T", (0,))
        with pytest.raises(ValueError):
            MeasurementTask("expectation", observable=PauliSum.from_labels({"X": 1j}))


class TestSerialization:
    def test_json_roundtrip(self):
        prog = (Program(3, parameters=["a"])
                .clifford("H", 0).rotor("ZZI", Parameter("a")).rotor("IXY", 0.4)
                .measure_expectation({"ZII": 1.0, "XXI": 0.25}).measure_z(0, 2))
        clone = Program.from_json(prog.to_json())
        assert clone.parameters.names == prog.parameters.names
        assert clone.unitary([1.2]).is_close(prog.unitary([1.2]), 1e-12)
        r1, r2 = prog.run([1.2]), clone.run([1.2])
        assert abs(r1[0] - r2[0]) < 1e-12
        assert r1[1] == r2[1]


class TestConjugation:
    def test_bell_map(self):
        prog = Program(2).clifford("H", 0).clifford("CX", 0, 1)
        phase, word = conjugate_pauli_word(prog, PauliWord.from_label("XI"))
        assert abs(phase - 1) < 1e-9 and word.label == "ZI"
        phase, word = conjugate_pauli_word(prog, PauliWord.from_label("ZI"))
        assert abs(phase - 1) < 1e-9 and word.label == "XX"

    def test_non_clifford_raises(self):
        prog = Program(1).rotor("X", 0.3)
        with pytest.raises(ValueError):
            conjugate_pauli_word(prog, PauliWord.from_label("Z"))


class TestGradients:
    def _ansatz(self):
        a, b = Parameter("a"), Parameter("b")
        prog = (Program(2, parameters=[a, b])
                .clifford("H", 0).rotor("YI", a).clifford("CX", 0, 1)
                .rotor("ZZ", b).rotor("XI", 0.2))
        obs = PauliSum.from_labels({"ZI": 1.0, "IZ": 0.5, "XX": 0.25})
        return prog, obs

    def test_analytic_single_rotor(self):
        prog = Program(1, parameters=["t"]).rotor("Y", Parameter("t"))
        obs = PauliSum.from_labels({"Z": 1.0})
        theta = 0.9
        assert abs(expectation_value(prog, obs, [theta]) - math.cos(theta)) < 1e-12
        for grad_fn in (parameter_shift_gradient, adjoint_gradient):
            g = grad_fn(prog, obs, [theta])
            assert abs(g[0] + math.sin(theta)) < 1e-10

    def test_methods_agree_with_finite_difference(self):
        prog, obs = self._ansatz()
        values = [0.4, 1.1]
        gs = parameter_shift_gradient(prog, obs, values)
        ga = adjoint_gradient(prog, obs, values)
        assert np.allclose(gs, ga, atol=1e-10)
        eps = 1e-6
        for k in range(2):
            up = list(values); up[k] += eps
            dn = list(values); dn[k] -= eps
            fd = (expectation_value(prog, obs, up) - expectation_value(prog, obs, dn)) / (2 * eps)
            assert abs(gs[k] - fd) < 1e-6
        assert any(abs(g) > 0.05 for g in gs)  # non-trivial gradient

    def test_shared_parameter_accumulates(self):
        t = Parameter("t")
        prog = Program(1, parameters=[t]).rotor("Y", t).rotor("Y", t)
        obs = PauliSum.from_labels({"Z": 1.0})
        theta = 0.3
        # <Z> = cos(2 theta) -> d/dtheta = -2 sin(2 theta)
        for grad_fn in (parameter_shift_gradient, adjoint_gradient):
            g = grad_fn(prog, obs, [theta])
            assert abs(g[0] + 2 * math.sin(2 * theta)) < 1e-10
