"""PennyLane frontend bridge (Phase 5).

IR programs lower to PennyLane operations (``qml.PauliRot`` shares the
``exp(-i theta P/2)`` convention). No device plugin is provided — the
bridge exists so PennyLane's parameter-shift gradients can be compared
against ``clifford_qc``'s exact adjoint/parameter-shift gradients.
"""

from __future__ import annotations

from functools import reduce

import pennylane as qml

from ..ir import Parameter, PauliSum, Program, Rotor

_QML_GATES = {
    "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ, "H": qml.Hadamard,
    "S": qml.S, "SDG": lambda w: qml.adjoint(qml.S(w)),
    "CX": qml.CNOT, "CZ": qml.CZ, "SWAP": qml.SWAP,
}


def apply_program(program: Program, params) -> None:
    """Queue the program's operations inside a PennyLane context."""
    bindings = dict(zip(program.parameters.names, params)) if len(program.parameters) else {}
    for op in program.ops:
        if isinstance(op, Rotor):
            support = op.word.support()
            # Keep bound tensors intact so PennyLane can differentiate them.
            theta = bindings[op.angle.name] if isinstance(op.angle, Parameter) \
                else float(op.angle)
            if not support:
                qml.GlobalPhase(theta / 2)
                continue
            letters = "".join(op.word.letter(j) for j in support)
            qml.PauliRot(theta, letters, wires=list(support))
        else:
            _QML_GATES[op.name](*op.qubits) if len(op.qubits) == 1 \
                else _QML_GATES[op.name](wires=list(op.qubits))


def observable_to_pennylane(observable: PauliSum):
    if not observable.is_hermitian():
        raise ValueError("PennyLane observables need real coefficients")
    coeffs, ops = [], []
    for word, coeff in observable.items():
        coeffs.append(coeff.real)
        support = word.support()
        if not support:
            ops.append(qml.Identity(0))
        else:
            singles = [_QML_GATES[word.letter(j)](j) for j in support]
            ops.append(reduce(lambda a, b: a @ b, singles))
    return qml.Hamiltonian(coeffs, ops)


def make_qnode(program: Program, observable: PauliSum, *,
               diff_method: str = "parameter-shift", device=None):
    """QNode computing <observable> on |0...0> evolved by the program."""
    dev = device if device is not None else qml.device("default.qubit", wires=program.n)
    H = observable_to_pennylane(observable)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params):
        apply_program(program, params)
        return qml.expval(H)

    return circuit
