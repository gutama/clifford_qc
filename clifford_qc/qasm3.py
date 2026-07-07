"""OpenQASM 3 lowering pass for IR programs.

QASM3 is an *export* format, not the native representation. Each rotor
``exp(-i theta P/2)`` lowers to single-qubit basis changes, a CX parity
ladder onto the last support qubit, one ``rz(theta)``, and the uncompute.

The lowering is expressed first as an abstract op list (``lower_program``)
so tests can evaluate exactly the same decomposition back to an ``MV`` and
compare against ``rotor`` — the emitted text and the verified semantics
cannot drift apart.
"""

from __future__ import annotations

import re

from .multivector import MV
from .pauli import I
from . import gates as _gates
from .ir import CLIFFORD_GATES, Parameter, PauliWord, Program, Rotor

_IDENT = re.compile(r"^[a-z][A-Za-z0-9_]*$")

# IR Clifford name -> stdgates.inc name
_STD_NAMES = {"X": "x", "Y": "y", "Z": "z", "H": "h", "S": "s", "SDG": "sdg",
              "CX": "cx", "CZ": "cz", "SWAP": "swap"}

# Basis change A with A† Z A = P, emitted as (pre-ops, post-ops) per letter.
# Y uses A = H·S† (as a matrix), i.e. the circuit applies sdg then h.
_BASIS_PRE = {"X": ("h",), "Y": ("sdg", "h")}
_BASIS_POST = {"X": ("h",), "Y": ("h", "s")}


def lower_rotor(word: PauliWord, theta) -> list[tuple]:
    """Abstract lowering of one rotor: [(gate, *args)] with gates from
    stdgates plus ("rz", theta, qubit) and ("gphase", phase)."""
    support = word.support()
    if not support:
        return [("gphase", _negate_half(theta))]
    ops: list[tuple] = []
    for j in support:
        for g in _BASIS_PRE.get(word.letter(j), ()):
            ops.append((g, j))
    for a, b in zip(support, support[1:]):
        ops.append(("cx", a, b))
    ops.append(("rz", theta, support[-1]))
    for a, b in reversed(list(zip(support, support[1:]))):
        ops.append(("cx", a, b))
    for j in reversed(support):
        for g in _BASIS_POST.get(word.letter(j), ()):
            ops.append((g, j))
    return ops


def _negate_half(theta):
    if isinstance(theta, Parameter):
        return theta  # rendered as -name/2 by the emitter
    return -0.5 * float(theta)


def lower_program(program: Program, values=None) -> list[tuple]:
    """Lower a whole program to the abstract op list (measurements excluded)."""
    bindings = program.parameters.bind(values) if values is not None else None
    ops: list[tuple] = []
    for op in program.ops:
        if isinstance(op, Rotor):
            theta = op.angle
            if bindings is not None or not isinstance(theta, Parameter):
                theta = op.resolved_angle(bindings)
            ops.extend(lower_rotor(op.word, theta))
        else:
            ops.append((_STD_NAMES[op.name], *op.qubits))
    return ops


def ops_to_mv(n: int, ops: list[tuple]) -> MV:
    """Exact MV semantics of an abstract op list (for conformance tests)."""
    import cmath

    by_name = {v: k for k, v in _STD_NAMES.items()}
    U = I(n)
    for op in ops:
        name, args = op[0], op[1:]
        if name == "rz":
            theta, j = args
            G = _gates.RZ(n, j, float(theta))
        elif name == "gphase":
            G = cmath.exp(1j * float(args[0])) * I(n)
        else:
            G = CLIFFORD_GATES[by_name[name]][1](n, *args)
        U = G * U
    return U


def _fmt_angle(theta, *, half_negated: bool = False) -> str:
    if isinstance(theta, Parameter):
        return f"-{theta.name}/2" if half_negated else theta.name
    return repr(float(theta))


def to_qasm3(program: Program, values=None) -> str:
    """Emit OpenQASM 3. Unbound parameters become ``input float[64]`` inputs;
    ``sample_z`` tasks become measure statements; expectation tasks have no
    QASM3 form and are noted in a comment."""
    for name in program.parameters.names:
        if not _IDENT.match(name):
            raise ValueError(f"parameter name {name!r} is not a valid QASM3 identifier")

    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    bound = values is not None or not len(program.parameters)
    if not bound:
        for name in program.parameters.names:
            lines.append(f"input float[64] {name};")
    lines.append(f"qubit[{program.n}] q;")

    measured = [q for t in program.measurements if t.kind == "sample_z" for q in t.qubits]
    if measured:
        lines.append(f"bit[{len(measured)}] c;")

    for op in lower_program(program, values):
        name, args = op[0], op[1:]
        if name == "rz":
            theta, j = args
            lines.append(f"rz({_fmt_angle(theta)}) q[{j}];")
        elif name == "gphase":
            lines.append(f"gphase({_fmt_angle(args[0], half_negated=True)});")
        elif name == "cx":
            lines.append(f"cx q[{args[0]}], q[{args[1]}];")
        elif name in ("cz", "swap"):
            lines.append(f"{name} q[{args[0]}], q[{args[1]}];")
        else:
            lines.append(f"{name} q[{args[0]}];")

    if any(t.kind == "expectation" for t in program.measurements):
        lines.append("// expectation-value tasks are not representable in QASM3")
    for i, q in enumerate(measured):
        lines.append(f"c[{i}] = measure q[{q}];")
    return "\n".join(lines) + "\n"
