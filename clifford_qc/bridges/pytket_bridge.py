"""pytket bridge (Phase 4).

Rotors lower to ``PauliExpBox`` (tket's native Pauli-exponential), named
Cliffords to native gates. tket's ``PauliExpBox(paulis, t)`` implements
``exp(-i (pi/2) t P)``, so tket half-turns relate to our radian angles by
``t = theta / pi``. Both sides use big-endian qubit ordering (qubit 0 most
significant), so unitaries compare entry-for-entry.
"""

from __future__ import annotations

import math

from pytket import Circuit, OpType
from pytket.circuit import PauliExpBox
from pytket.pauli import Pauli

from ..ir import Program, Rotor

TKET_OPTYPES = {"X": OpType.X, "Y": OpType.Y, "Z": OpType.Z, "H": OpType.H,
                "S": OpType.S, "SDG": OpType.Sdg, "CX": OpType.CX,
                "CZ": OpType.CZ, "SWAP": OpType.SWAP}
_OPTYPE_NAMES = {v: k for k, v in TKET_OPTYPES.items()}
TKET_PAULIS = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
_PAULI_LETTERS = {v: k for k, v in TKET_PAULIS.items()}


def program_to_tket(program: Program, values=None) -> Circuit:
    """Lower an IR program to a pytket Circuit (angles must be bindable)."""
    bindings = program.parameters.bind(values) if len(program.parameters) else {}
    circuit = Circuit(program.n)
    for op in program.ops:
        if isinstance(op, Rotor):
            theta = op.resolved_angle(bindings)
            support = op.word.support()
            if not support:
                # exp(-i theta/2) global phase, in tket half-turn units
                circuit.add_phase(-theta / (2 * math.pi))
                continue
            paulis = [TKET_PAULIS[op.word.letter(j)] for j in support]
            circuit.add_pauliexpbox(PauliExpBox(paulis, theta / math.pi), list(support))
        else:
            circuit.add_gate(TKET_OPTYPES[op.name], list(op.qubits))
    return circuit


def tket_to_program(circuit: Circuit) -> Program:
    """Round-trip a tket circuit built from supported gates and PauliExpBoxes
    back into the IR (measurement/classical operations are not supported)."""
    program = Program(circuit.n_qubits)
    global_phase = circuit.phase
    if isinstance(global_phase, str) or hasattr(global_phase, "free_symbols"):
        if getattr(global_phase, "free_symbols", set()):
            raise ValueError("symbolic circuit global phase is not supported")
    if float(global_phase) != 0.0:
        # tket phase p means exp(i*pi*p); an identity rotor has exp(-i*theta/2).
        program.rotor("I" * circuit.n_qubits, -2.0 * math.pi * float(global_phase))
    for cmd in circuit.get_commands():
        optype = cmd.op.type
        qubits = tuple(q.index[0] for q in cmd.qubits)
        if optype == OpType.PauliExpBox:
            letters = [_PAULI_LETTERS[p] for p in cmd.op.get_paulis()]
            phase = cmd.op.get_phase()
            if isinstance(phase, str) or hasattr(phase, "free_symbols"):
                raise ValueError("symbolic PauliExpBox angles are not supported")
            label = ["I"] * circuit.n_qubits
            for q, letter in zip(qubits, letters):
                label[q] = letter
            program.rotor("".join(label), float(phase) * math.pi)
        elif optype in _OPTYPE_NAMES:
            program.clifford(_OPTYPE_NAMES[optype], *qubits)
        else:
            raise ValueError(f"unsupported tket op {optype} for IR round-trip")
    return program
