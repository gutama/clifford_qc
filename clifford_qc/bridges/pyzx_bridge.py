"""PyZX bridge (Phase 5): IR <-> ZX-calculus circuits.

A native ``Program`` is Clifford gates plus Pauli-word rotors
``exp(-i theta P / 2)``. Each rotor lowers the textbook way -- diagonalize
every non-``Z`` letter with a basis-change Clifford (``H`` for ``X``,
``H S`` for ``Y``), fold the support onto one qubit with a CNOT parity
network, apply a single ``ZPhase`` rotation, then uncompute -- which is
exactly the phase-gadget structure ZX-calculus rewrites efficiently.
PyZX's ``full_reduce`` re-discovers that gadget in the diagram, so we emit
the ladder directly rather than relying on ``PhaseGadget``'s scaling
convention.

Conventions (verified against the exact MV backend):

- Qubit ordering is big-endian on both sides (qubit 0 most significant),
  so unitaries compare entry-for-entry.
- PyZX phases are fractions of ``pi``: a radian angle ``theta`` becomes
  ``Fraction(theta / pi)``. ``ZPhase(p)`` is ``diag(1, e^{i p pi})``, which
  equals ``exp(-i theta Z / 2)`` up to a global phase when ``p = theta/pi``.
  **Everything here is therefore equivalence up to global phase** -- the
  physically meaningful notion, and what PyZX's own ``verify_equality`` and
  ``compare_tensors`` check.

Caveat on arbitrary angles: PyZX represents phases as rationals, so a
general (non-Clifford) rotor angle is rounded to ``Fraction(theta/pi)``
with a bounded denominator (``DEFAULT_MAX_DENOMINATOR``). For angles that
are rational multiples of ``pi`` -- all Clifford angles among them -- this
is exact; for a generic VQE angle it introduces an error below
``pi / DEFAULT_MAX_DENOMINATOR``. ``optimize_program`` re-verifies its
output and never returns a circuit that fails the equivalence check within
tolerance, and never one that is worse on the requested objective.
"""

from __future__ import annotations

from fractions import Fraction
from math import pi
from typing import Union

import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import (
    CNOT, CZ, HAD, NOT, SWAP, S, T, XPhase, YPhase, Z as ZGate, Y as YGate,
    ZPhase,
)

from ..ir import Program, Rotor

# Denominator bound for float-angle -> Fraction(theta/pi) conversion. 2**24
# keeps the angle error below ~1e-7 rad while staying exact on every
# rational-multiple-of-pi angle the IR emits for Clifford operations.
DEFAULT_MAX_DENOMINATOR = 2 ** 24

# IR named Clifford -> factory(*qubits) producing the pyzx gate.
_CLIFFORD_TO_PYZX = {
    "H": lambda q: HAD(q),
    "X": lambda q: NOT(q),
    "Y": lambda q: YGate(q),
    "Z": lambda q: ZGate(q),
    "S": lambda q: S(q),
    "SDG": lambda q: S(q, adjoint=True),
    "CX": lambda c, t: CNOT(c, t),
    "CZ": lambda c, t: CZ(c, t),
    "SWAP": lambda a, b: SWAP(a, b),
}


def _angle_to_phase(theta: float, max_denominator: int) -> Fraction:
    """Radian angle -> PyZX phase (fraction of pi), rounded to a rational."""
    return Fraction(theta / pi).limit_denominator(max_denominator)


def _basis_change(qubit: int, letter: str, dagger: bool) -> list:
    """Clifford(s) rotating ``letter`` to the Z basis (or the inverse).

    ``B P B_dagger = Z`` with ``B = H`` for ``X`` and ``B = H S_dagger`` for
    ``Y``; ``dagger=True`` emits ``B_dagger`` for the uncompute half.
    """
    if letter == "X":
        return [HAD(qubit)]
    if letter == "Y":
        return [HAD(qubit), S(qubit)] if dagger else [S(qubit, adjoint=True), HAD(qubit)]
    return []


def _add_rotor(circuit: Circuit, rotor: Rotor, theta: float, max_denominator: int) -> None:
    support = rotor.word.support()
    if not support:
        # exp(-i theta/2 I): a global phase PyZX does not represent. The
        # bridge is up-to-global-phase, so drop it.
        return
    letters = {j: rotor.word.letter(j) for j in support}
    for j in support:
        for g in _basis_change(j, letters[j], dagger=False):
            circuit.add_gate(g)
    target = support[-1]
    for j in support[:-1]:
        circuit.add_gate(CNOT(j, target))
    circuit.add_gate(ZPhase(target, _angle_to_phase(theta, max_denominator)))
    for j in reversed(support[:-1]):
        circuit.add_gate(CNOT(j, target))
    for j in reversed(support):
        for g in _basis_change(j, letters[j], dagger=True):
            circuit.add_gate(g)


def to_pyzx(program: Program, values=None,
            max_denominator: int = DEFAULT_MAX_DENOMINATOR) -> Circuit:
    """Lower a bound unitary IR program to a ``pyzx.Circuit``.

    Angles must be bindable (pass ``values`` for parameterized programs).
    Measurement tasks are ignored -- only the unitary body is translated.
    """
    bindings = program.parameters.bind(values) if len(program.parameters) else {}
    circuit = Circuit(program.n)
    for op in program.ops:
        if isinstance(op, Rotor):
            _add_rotor(circuit, op, op.resolved_angle(bindings), max_denominator)
        else:
            circuit.add_gate(_CLIFFORD_TO_PYZX[op.name](*op.qubits))
    return circuit


def from_pyzx(circuit: Circuit) -> Program:
    """Round-trip a PyZX circuit back into the IR.

    Clifford gates map to named Cliffords; ``ZPhase``/``XPhase``/``YPhase``
    and ``T`` map to single-letter Pauli rotors (``ZPhase(p)`` -> a ``Z``
    rotor of angle ``p*pi``), so Clifford+T circuits -- including the output
    of ``extract_circuit`` -- ingest cleanly. Non-unitary or unsupported
    gates raise ``ValueError``.
    """
    n = circuit.qubits
    program = Program(n)
    for gate in circuit.to_basic_gates().gates:
        name = gate.name
        if name == "HAD":
            program.clifford("H", gate.target)
        elif name == "NOT":
            program.clifford("X", gate.target)
        elif name == "Y":
            program.clifford("Y", gate.target)
        elif name == "Z":
            program.clifford("Z", gate.target)
        elif name == "S":
            program.clifford("SDG" if getattr(gate, "adjoint", False) else "S", gate.target)
        elif name == "T":
            sign = -1.0 if getattr(gate, "adjoint", False) else 1.0
            program.rotor(_z_label(n, gate.target), sign * pi / 4)
        elif name in ("ZPhase", "XPhase", "YPhase"):
            letter = name[0]
            program.rotor(_single_label(n, gate.target, letter), float(gate.phase) * pi)
        elif name == "CNOT":
            program.clifford("CX", gate.control, gate.target)
        elif name == "CZ":
            program.clifford("CZ", gate.control, gate.target)
        elif name == "SWAP":
            program.clifford("SWAP", gate.control, gate.target)
        else:
            raise ValueError(
                f"unsupported PyZX gate {name!r} for IR round-trip; "
                "supported: HAD, NOT, Y, Z, S, T, ZPhase, XPhase, YPhase, CNOT, CZ, SWAP")
    return program


def _single_label(n: int, qubit: int, letter: str) -> str:
    chars = ["I"] * n
    chars[qubit] = letter
    return "".join(chars)


def _z_label(n: int, qubit: int) -> str:
    return _single_label(n, qubit, "Z")


# ---------------------------------------------------------------------------
# Equivalence and resource accounting


def verify_equivalent(program_a: Union[Program, Circuit],
                      program_b: Union[Program, Circuit],
                      values=None, values_b=None,
                      method: str = "pyzx", atol: float = 1e-7,
                      max_denominator: int = DEFAULT_MAX_DENOMINATOR) -> bool:
    """True when the two circuits are equal up to a global phase.

    ``method="pyzx"`` uses ZX-calculus rewriting (``verify_equality``),
    falling back to a dense tensor comparison when PyZX cannot decide;
    ``method="dense"`` compares full unitaries directly (small ``n`` only);
    ``method="both"`` requires the two independent checks to agree.

    ``Program`` inputs are lowered with ``values`` (and ``values_b`` for the
    second, defaulting to ``values``); ``Circuit`` inputs are used as-is.
    """
    ca = _as_circuit(program_a, values, max_denominator)
    cb = _as_circuit(program_b, values_b if values_b is not None else values, max_denominator)
    if ca.qubits != cb.qubits:
        return False
    if method == "pyzx":
        return _verify_pyzx(ca, cb, atol)
    if method == "dense":
        return _verify_dense(ca, cb, atol)
    if method == "both":
        return _verify_pyzx(ca, cb, atol) and _verify_dense(ca, cb, atol)
    raise ValueError("method must be 'pyzx', 'dense', or 'both'")


def _as_circuit(obj, values, max_denominator) -> Circuit:
    if isinstance(obj, Circuit):
        return obj
    return to_pyzx(obj, values, max_denominator)


def _verify_pyzx(ca: Circuit, cb: Circuit, atol: float) -> bool:
    verdict = ca.verify_equality(cb)  # up to global phase; None if undecided
    if verdict is None:
        return _verify_dense(ca, cb, atol)
    return bool(verdict)


def _verify_dense(ca: Circuit, cb: Circuit, atol: float) -> bool:
    import numpy as np
    U, V = ca.to_matrix(), cb.to_matrix()
    if U.shape != V.shape:
        return False
    k = int(np.argmax(np.abs(V)))
    if abs(V.flat[k]) < atol:
        return bool(np.allclose(U, V, atol=atol))
    phase = U.flat[k] / V.flat[k]
    return abs(abs(phase) - 1.0) < atol and bool(np.allclose(U, phase * V, atol=atol))


def resource_counts(obj: Union[Program, Circuit], values=None,
                    max_denominator: int = DEFAULT_MAX_DENOMINATOR) -> dict:
    """Gate-level metrics for benchmark metadata.

    Reports ``gate_count`` (basic gates), ``two_qubit_count``, ``t_count``,
    and ``two_qubit_depth`` (ASAP layering over the two-qubit gates).
    """
    circuit = _as_circuit(obj, values, max_denominator).to_basic_gates()
    return {
        "gate_count": len(circuit.gates),
        "two_qubit_count": circuit.twoqubitcount(),
        "t_count": zx.tcount(circuit),
        "two_qubit_depth": _two_qubit_depth(circuit),
    }


def _two_qubit_depth(circuit: Circuit) -> int:
    """ASAP-scheduled depth counting only two-qubit gates."""
    frontier = [0] * circuit.qubits
    depth = 0
    for gate in circuit.gates:
        qubits = [q for q in (getattr(gate, "control", None), getattr(gate, "target", None))
                  if q is not None]
        if len(qubits) < 2:
            continue
        layer = max(frontier[q] for q in qubits) + 1
        for q in qubits:
            frontier[q] = layer
        depth = max(depth, layer)
    return depth


def optimize_program(program: Program, values=None,
                     objective: str = "two_qubit_count",
                     max_denominator: int = DEFAULT_MAX_DENOMINATOR,
                     verify: bool = True) -> Program:
    """Return a ZX-optimized, equivalent bound program.

    Runs ``full_reduce`` on the ZX diagram, extracts a circuit, and applies
    ``basic_optimization``. The result is kept only if it does not regress
    on ``objective`` (``"two_qubit_count"``, ``"gate_count"``, or
    ``"t_count"``) -- otherwise the original circuit is returned unchanged,
    so optimization is monotone. With ``verify=True`` the output is checked
    equivalent to the input (up to global phase and angle-rounding
    tolerance); a failure raises ``RuntimeError``.

    ZX extraction favors Clifford+T and structured circuits; for
    arbitrary-angle rotor programs it may not improve the objective, in
    which case the input is returned untouched.
    """
    metrics = {"two_qubit_count": lambda c: c.twoqubitcount(),
               "gate_count": lambda c: len(c.to_basic_gates().gates),
               "t_count": lambda c: zx.tcount(c)}
    if objective not in metrics:
        raise ValueError(f"objective must be one of {sorted(metrics)}")
    score = metrics[objective]

    original = to_pyzx(program, values, max_denominator)
    graph = original.to_graph()
    zx.simplify.full_reduce(graph)
    extracted = zx.extract_circuit(graph.copy()).to_basic_gates()
    extracted = zx.optimize.basic_optimization(extracted).to_basic_gates()

    best = extracted if score(extracted) < score(original) else original
    if verify and not _verify_pyzx(original, best, atol=1e-7):
        raise RuntimeError("ZX optimization produced a non-equivalent circuit")
    return from_pyzx(best)
