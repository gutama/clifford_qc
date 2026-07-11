"""IR schema versioning, gate-by-gate execution, and Clifford-angle rotors."""

import math

import pytest

from clifford_qc import (
    MV, Program, Parameter, clifford_angle_index, evolve, ket_density,
)


def _sample_program():
    prog = Program(3, parameters=["a"])
    prog.clifford("H", 0).clifford("CX", 0, 1)
    prog.rotor("XYI", Parameter("a")).rotor("IZZ", 0.3)
    prog.measure_expectation({"ZII": 1.0})
    return prog


def test_to_dict_carries_schema_header():
    d = _sample_program().to_dict()
    assert d["schema"] == "clifford_qc/program"
    assert d["schema_version"] == 1


def test_versioned_roundtrip():
    prog = _sample_program()
    again = Program.from_dict(prog.to_dict())
    assert again.to_dict() == prog.to_dict()


def test_headerless_version0_still_loads():
    d = _sample_program().to_dict()
    del d["schema"]
    del d["schema_version"]
    prog = Program.from_dict(d)
    assert prog.to_dict()["schema_version"] == 1
    assert prog.run([0.4])[0] == pytest.approx(_sample_program().run([0.4])[0])


def test_unknown_schema_and_newer_version_rejected():
    d = _sample_program().to_dict()
    with pytest.raises(ValueError, match="unknown schema"):
        Program.from_dict({**d, "schema": "someone_else/program"})
    with pytest.raises(ValueError, match="newer than supported"):
        Program.from_dict({**d, "schema_version": 99})


def test_gatewise_state_matches_full_unitary():
    prog = _sample_program()
    values = [0.7]
    rho_gatewise = prog.state(values)
    rho_unitary = evolve(ket_density(3, "000"), prog.unitary(values))
    diff = rho_gatewise + (-1.0) * rho_unitary
    assert max(abs(c) for c in diff.terms.values() or [0.0]) < 1e-12


def test_clifford_angle_index():
    assert clifford_angle_index(0.0) == 0
    assert clifford_angle_index(math.pi / 2) == 1
    assert clifford_angle_index(math.pi) == 2
    assert clifford_angle_index(-math.pi / 2) == 3
    assert clifford_angle_index(2 * math.pi) == 0
    assert clifford_angle_index(5 * math.pi / 2) == 1
    assert clifford_angle_index(0.3) is None


def test_is_clifford_only_recognizes_clifford_angle_rotors():
    prog = Program(2)
    prog.clifford("H", 0)
    prog.rotor("XY", math.pi / 2).rotor("ZZ", math.pi).rotor("XI", -math.pi / 2)
    assert prog.is_clifford_only()
    prog.rotor("ZI", 0.3)
    assert not prog.is_clifford_only()


def test_is_clifford_only_with_parameters():
    prog = Program(2)
    prog.rotor("XY", Parameter("a"))
    assert not prog.is_clifford_only()          # unbound could be anything
    assert prog.is_clifford_only([math.pi])     # bound to a Clifford angle
    assert not prog.is_clifford_only([0.4])
