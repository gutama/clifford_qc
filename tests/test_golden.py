"""Golden conformance vectors: every implementation of the IR must
reproduce these results. The expectation values are re-derived through two
independent paths (sparse MV evolution and the dense matrix backend), and
the QASM3 emission must be byte-identical."""

import json
import pathlib

import numpy as np
import pytest

from clifford_qc import Program, to_qasm3, to_matrix
from clifford_qc.qasm3 import lower_program, ops_to_mv

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden" / "programs.json").read_text())


def golden_cases():
    return [pytest.param(entry, id=entry["name"]) for entry in GOLDEN["programs"]]


def _dense_state(prog, values):
    n = prog.n
    U = to_matrix(prog.unitary(values))
    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1.0
    return U @ psi0


@pytest.mark.parametrize("entry", golden_cases())
def test_golden_roundtrip_and_mv_results(entry):
    prog = Program.from_dict(entry["program"])
    assert prog.to_dict() == entry["program"]
    results = prog.run(entry["values"])
    for got, want in zip(results, entry["results"]):
        if want["type"] == "expectation":
            assert abs(got - want["value"]) < 1e-9
        else:
            for bits, p in want["probabilities"].items():
                assert abs(got[bits] - p) < 1e-9


@pytest.mark.parametrize("entry", golden_cases())
def test_golden_against_dense_backend(entry):
    prog = Program.from_dict(entry["program"])
    psi = _dense_state(prog, entry["values"])
    for task, want in zip(prog.measurements, entry["results"]):
        if want["type"] == "expectation":
            O = to_matrix(task.observable.to_mv())
            assert abs((psi.conj() @ O @ psi).real - want["value"]) < 1e-9
        else:
            probs = np.abs(psi) ** 2
            n = prog.n
            for bits, p in want["probabilities"].items():
                total = 0.0
                for k in range(2 ** n):
                    full = format(k, f"0{n}b")
                    if all(full[q] == bits[i] for i, q in enumerate(task.qubits)):
                        total += probs[k]
                assert abs(total - p) < 1e-9


@pytest.mark.parametrize("entry", golden_cases())
def test_golden_qasm3_emission(entry):
    prog = Program.from_dict(entry["program"])
    assert to_qasm3(prog, entry["values"]) == entry["qasm3"]


@pytest.mark.parametrize("entry", golden_cases())
def test_golden_qasm3_semantics(entry):
    prog = Program.from_dict(entry["program"])
    U = ops_to_mv(prog.n, lower_program(prog, entry["values"]))
    assert U.is_close(prog.unitary(entry["values"]), 1e-9)
