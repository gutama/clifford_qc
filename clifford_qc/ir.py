"""Pauli-rotor intermediate representation (IR).

The IR is the native interchange layer of ``clifford_qc``: programs are
sequences of Pauli-word rotors ``exp(-i theta P/2)`` and named Clifford
gates, plus measurement tasks. Everything lowers exactly to an ``MV``
unitary, and the bridges (stim, pytket, PennyLane, QASM3) translate from
this IR rather than from raw multivectors.

Angle/rotor convention matches the core engine:

    R_P(theta) = exp(-i theta P / 2) = cos(theta/2) - i sin(theta/2) P
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Mapping, Sequence, Union

from .multivector import MV, code_to_label, label_to_code, validate_n, validate_qubit
from .pauli import I
from . import gates as _gates
from .states import evolve, expectation, ket_density, computational_probabilities


# ---------------------------------------------------------------------------
# Pauli data


@dataclass(frozen=True)
class PauliWord:
    """A single Pauli word (tensor product of I/X/Y/Z) on n qubits."""

    n: int
    code: int

    def __post_init__(self):
        validate_n(self.n)
        if not isinstance(self.code, int) or not (0 <= self.code < 4 ** self.n):
            raise ValueError(f"word code must be in [0, 4**n), got {self.code!r}")

    @staticmethod
    def from_label(label: str, n: int | None = None) -> "PauliWord":
        if n is None:
            n = len(label)
        return PauliWord(n, label_to_code(label, n))

    @property
    def label(self) -> str:
        return code_to_label(self.n, self.code)

    @property
    def weight(self) -> int:
        """Number of non-identity letters."""
        return sum(1 for j in range(self.n) if (self.code >> (2 * j)) & 3)

    def support(self) -> tuple[int, ...]:
        return tuple(j for j in range(self.n) if (self.code >> (2 * j)) & 3)

    def letter(self, j: int) -> str:
        validate_qubit(self.n, j)
        return "IXYZ"[(self.code >> (2 * j)) & 3]

    def to_mv(self, coeff: complex = 1.0) -> MV:
        return MV(self.n, {self.code: coeff})

    def __str__(self) -> str:
        return self.label


class PauliSum:
    """A complex-linear combination of Pauli words: the IR observable type."""

    __slots__ = ("n", "terms")

    def __init__(self, n: int, terms: Mapping[int, complex] | None = None):
        validate_n(n)
        self.n = n
        self.terms: dict[int, complex] = {}
        if terms:
            for code, coeff in terms.items():
                code = int(code)
                if not (0 <= code < 4 ** n):
                    raise ValueError(f"word code {code} out of range for n={n}")
                c = complex(coeff)
                if c != 0:
                    self.terms[code] = self.terms.get(code, 0j) + c

    @staticmethod
    def from_labels(terms: Mapping[str, complex], n: int | None = None) -> "PauliSum":
        if not terms and n is None:
            raise ValueError("empty PauliSum needs an explicit n")
        if n is None:
            n = len(next(iter(terms)))
        return PauliSum(n, {label_to_code(lbl, n): c for lbl, c in terms.items()})

    @staticmethod
    def from_mv(A: MV) -> "PauliSum":
        return PauliSum(A.n, dict(A.terms))

    def to_mv(self) -> MV:
        return MV(self.n, dict(self.terms))

    def to_labels(self) -> dict[str, complex]:
        return {code_to_label(self.n, k): v for k, v in sorted(self.terms.items())}

    def items(self) -> list[tuple[PauliWord, complex]]:
        return [(PauliWord(self.n, k), v) for k, v in sorted(self.terms.items())]

    def is_hermitian(self, tol: float = 1e-12) -> bool:
        return all(abs(c.imag) < tol for c in self.terms.values())

    def __add__(self, other: "PauliSum") -> "PauliSum":
        if not isinstance(other, PauliSum) or other.n != self.n:
            raise ValueError("can only add PauliSums on the same qubit count")
        out = dict(self.terms)
        for k, v in other.terms.items():
            out[k] = out.get(k, 0j) + v
        return PauliSum(self.n, out)

    def __rmul__(self, c: complex) -> "PauliSum":
        return PauliSum(self.n, {k: complex(c) * v for k, v in self.terms.items()})

    def __repr__(self) -> str:
        return f"PauliSum({self.to_labels()!r})"


# ---------------------------------------------------------------------------
# Parameters


@dataclass(frozen=True)
class Parameter:
    """A named free angle, usable wherever a Rotor angle is expected."""

    name: str

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("parameter name must be a non-empty string")


class ParameterGroup:
    """An ordered set of named parameters with binding helpers.

    Gradients are reported in this group's order, which is what the
    PennyLane bridge compares against parameter-shift gradients.
    """

    __slots__ = ("parameters",)

    def __init__(self, parameters: Iterable[Union[Parameter, str]] = ()):
        params = tuple(p if isinstance(p, Parameter) else Parameter(p) for p in parameters)
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in {names}")
        self.parameters = params

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def bind(self, values: Union[Sequence[float], Mapping[str, float], None]) -> dict[str, float]:
        if values is None:
            values = ()
        if isinstance(values, Mapping):
            unknown = set(values) - set(self.names)
            if unknown:
                raise ValueError(f"unknown parameters {sorted(unknown)}")
            bound = dict(values)
        else:
            values = list(values)
            if len(values) != len(self.parameters):
                raise ValueError(f"expected {len(self.parameters)} values, got {len(values)}")
            bound = dict(zip(self.names, values))
        missing = set(self.names) - set(bound)
        if missing:
            raise ValueError(f"missing values for parameters {sorted(missing)}")
        return {k: float(v) for k, v in bound.items()}


Angle = Union[float, Parameter]


# ---------------------------------------------------------------------------
# Operations


@dataclass(frozen=True)
class Rotor:
    """IR rotor gate exp(-i angle * word / 2)."""

    word: PauliWord
    angle: Angle

    def __post_init__(self):
        if not isinstance(self.word, PauliWord):
            raise TypeError("Rotor.word must be a PauliWord")
        if not isinstance(self.angle, (Real, Parameter)):
            raise TypeError("Rotor.angle must be a real number or Parameter")

    @property
    def n(self) -> int:
        return self.word.n

    def resolved_angle(self, bindings: Mapping[str, float] | None = None) -> float:
        if isinstance(self.angle, Parameter):
            if not bindings or self.angle.name not in bindings:
                raise ValueError(f"unbound parameter {self.angle.name!r}")
            return float(bindings[self.angle.name])
        return float(self.angle)

    def to_mv(self, bindings: Mapping[str, float] | None = None) -> MV:
        return _gates.rotor(self.word.to_mv(), self.resolved_angle(bindings))


def _cx(n: int, c: int, t: int) -> MV:
    return _gates.CNOT(n, c, t)


def _sdg(n: int, j: int) -> MV:
    return _gates.S(n, j).dagger()


# name -> (arity, builder(n, *qubits) -> MV). Clifford gates only; T is not Clifford.
CLIFFORD_GATES = {
    "X": (1, lambda n, j: MV.word(n, "I" * j + "X" + "I" * (n - j - 1))),
    "Y": (1, lambda n, j: MV.word(n, "I" * j + "Y" + "I" * (n - j - 1))),
    "Z": (1, lambda n, j: MV.word(n, "I" * j + "Z" + "I" * (n - j - 1))),
    "H": (1, _gates.H),
    "S": (1, _gates.S),
    "SDG": (1, _sdg),
    "CX": (2, _cx),
    "CZ": (2, _gates.CZ),
    "SWAP": (2, _gates.SWAP),
}


@dataclass(frozen=True)
class NamedClifford:
    """A named Clifford gate on explicit qubits (H, S, SDG, X, Y, Z, CX, CZ, SWAP)."""

    name: str
    qubits: tuple[int, ...]

    def __post_init__(self):
        name = self.name.upper()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        if name not in CLIFFORD_GATES:
            raise ValueError(f"unknown Clifford gate {name!r}; known: {sorted(CLIFFORD_GATES)}")
        arity = CLIFFORD_GATES[name][0]
        if len(self.qubits) != arity:
            raise ValueError(f"{name} expects {arity} qubit(s), got {self.qubits}")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"{name} qubits must be distinct, got {self.qubits}")

    def to_mv(self, n: int) -> MV:
        for q in self.qubits:
            validate_qubit(n, q)
        return CLIFFORD_GATES[self.name][1](n, *self.qubits)


Operation = Union[Rotor, NamedClifford]


# ---------------------------------------------------------------------------
# Measurement tasks


@dataclass(frozen=True)
class MeasurementTask:
    """What to extract at the end of a program.

    ``kind="expectation"`` carries a Hermitian PauliSum observable;
    ``kind="sample_z"`` requests computational-basis probabilities for the
    listed qubits (lowered to measure statements in QASM3).
    """

    kind: str
    observable: PauliSum | None = None
    qubits: tuple[int, ...] = ()

    def __post_init__(self):
        if self.kind == "expectation":
            if self.observable is None:
                raise ValueError("expectation task needs an observable")
            if not self.observable.is_hermitian():
                raise ValueError("expectation observable must have real coefficients")
        elif self.kind == "sample_z":
            object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
            if not self.qubits:
                raise ValueError("sample_z task needs at least one qubit")
        else:
            raise ValueError(f"unknown measurement kind {self.kind!r}")


# ---------------------------------------------------------------------------
# Program


class Program:
    """An IR program: qubit count, operation list, measurements, parameters."""

    def __init__(self, n: int, ops: Iterable[Operation] = (),
                 measurements: Iterable[MeasurementTask] = (),
                 parameters: ParameterGroup | Iterable[Union[Parameter, str]] = ()):
        validate_n(n)
        self.n = n
        self.ops: list[Operation] = []
        self.measurements: list[MeasurementTask] = []
        self.parameters = parameters if isinstance(parameters, ParameterGroup) else ParameterGroup(parameters)
        for op in ops:
            self.append(op)
        for m in measurements:
            self.add_measurement(m)

    # -- construction helpers ------------------------------------------------

    def append(self, op: Operation) -> "Program":
        if isinstance(op, Rotor):
            if op.n != self.n:
                raise ValueError(f"rotor acts on {op.n} qubits, program has {self.n}")
            if isinstance(op.angle, Parameter) and op.angle.name not in self.parameters.names:
                self.parameters = ParameterGroup((*self.parameters, op.angle))
        elif isinstance(op, NamedClifford):
            for q in op.qubits:
                validate_qubit(self.n, q)
        else:
            raise TypeError(f"unsupported operation {op!r}")
        self.ops.append(op)
        return self

    def rotor(self, label: str, angle: Angle) -> "Program":
        return self.append(Rotor(PauliWord.from_label(label, self.n), angle))

    def clifford(self, name: str, *qubits: int) -> "Program":
        return self.append(NamedClifford(name, tuple(qubits)))

    def add_measurement(self, task: MeasurementTask) -> "Program":
        if task.kind == "expectation" and task.observable.n != self.n:
            raise ValueError("observable qubit count differs from program")
        for q in task.qubits:
            validate_qubit(self.n, q)
        self.measurements.append(task)
        return self

    def measure_expectation(self, observable: Union[PauliSum, Mapping[str, complex]]) -> "Program":
        if not isinstance(observable, PauliSum):
            observable = PauliSum.from_labels(observable, self.n)
        return self.add_measurement(MeasurementTask("expectation", observable=observable))

    def measure_z(self, *qubits: int) -> "Program":
        return self.add_measurement(MeasurementTask("sample_z", qubits=qubits))

    # -- semantics -----------------------------------------------------------

    def is_clifford_only(self) -> bool:
        return all(isinstance(op, NamedClifford) for op in self.ops)

    def op_to_mv(self, op: Operation, bindings: Mapping[str, float] | None = None) -> MV:
        return op.to_mv(bindings) if isinstance(op, Rotor) else op.to_mv(self.n)

    def unitary(self, values=None) -> MV:
        """Exact MV unitary of the whole program (ops applied in list order)."""
        bindings = self.parameters.bind(values) if len(self.parameters) else {}
        U = I(self.n)
        for op in self.ops:
            U = self.op_to_mv(op, bindings) * U
        return U

    def state(self, values=None, rho0: MV | None = None) -> MV:
        if rho0 is None:
            rho0 = ket_density(self.n, "0" * self.n)
        return evolve(rho0, self.unitary(values))

    def run(self, values=None, rho0: MV | None = None) -> list:
        """Evaluate all measurement tasks; returns one result per task."""
        rho = self.state(values, rho0)
        out = []
        for task in self.measurements:
            if task.kind == "expectation":
                out.append(expectation(rho, task.observable.to_mv()).real)
            else:
                traced = rho if len(task.qubits) == self.n else _trace_to(rho, task.qubits)
                out.append(computational_probabilities(traced))
        return out

    # -- serialization (golden-vector format) ---------------------------------

    def to_dict(self) -> dict:
        ops = []
        for op in self.ops:
            if isinstance(op, Rotor):
                angle = {"param": op.angle.name} if isinstance(op.angle, Parameter) else float(op.angle)
                ops.append({"type": "rotor", "word": op.word.label, "angle": angle})
            else:
                ops.append({"type": "clifford", "name": op.name, "qubits": list(op.qubits)})
        meas = []
        for task in self.measurements:
            if task.kind == "expectation":
                obs = {lbl: [c.real, c.imag] for lbl, c in task.observable.to_labels().items()}
                meas.append({"type": "expectation", "observable": obs})
            else:
                meas.append({"type": "sample_z", "qubits": list(task.qubits)})
        return {"n": self.n, "parameters": list(self.parameters.names),
                "ops": ops, "measurements": meas}

    @staticmethod
    def from_dict(data: Mapping) -> "Program":
        prog = Program(int(data["n"]), parameters=data.get("parameters", ()))
        for op in data["ops"]:
            if op["type"] == "rotor":
                angle = op["angle"]
                if isinstance(angle, Mapping):
                    angle = Parameter(angle["param"])
                prog.rotor(op["word"], angle)
            elif op["type"] == "clifford":
                prog.clifford(op["name"], *op["qubits"])
            else:
                raise ValueError(f"unknown op type {op['type']!r}")
        for task in data.get("measurements", ()):
            if task["type"] == "expectation":
                obs = {lbl: complex(re, im) for lbl, (re, im) in task["observable"].items()}
                prog.measure_expectation(obs)
            elif task["type"] == "sample_z":
                prog.measure_z(*task["qubits"])
            else:
                raise ValueError(f"unknown measurement type {task['type']!r}")
        return prog

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_json(text: str) -> "Program":
        return Program.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return (f"Program(n={self.n}, ops={len(self.ops)}, "
                f"measurements={len(self.measurements)}, parameters={list(self.parameters.names)})")


def _trace_to(rho: MV, keep: tuple[int, ...]) -> MV:
    from .states import partial_trace
    traced = {j for j in range(rho.n) if j not in keep}
    return partial_trace(rho, traced)


def conjugate_pauli_word(program: Program, word: PauliWord, values=None,
                         tol: float = 1e-9) -> tuple[complex, PauliWord]:
    """Exact MV conjugation U P U† of a single Pauli word.

    Returns ``(phase, word')`` when the image is again a single Pauli word
    (always true for Clifford-only programs); raises otherwise. This is the
    MV-side reference for the stim CliffordMap.
    """
    U = program.unitary(values)
    out = U * word.to_mv() * U.dagger()
    terms = {k: v for k, v in out.terms.items() if abs(v) > tol}
    if len(terms) != 1:
        raise ValueError(f"image has {len(terms)} Pauli words; not a single blade")
    ((code, coeff),) = terms.items()
    return coeff, PauliWord(program.n, code)


# ---------------------------------------------------------------------------
# Gradients


def expectation_value(program: Program, observable: PauliSum, values=None,
                      rho0: MV | None = None) -> float:
    rho = program.state(values, rho0)
    return expectation(rho, observable.to_mv()).real


def parameter_shift_gradient(program: Program, observable: PauliSum, values,
                             rho0: MV | None = None) -> list[float]:
    """Exact parameter-shift gradient d<O>/d(theta_k), one entry per parameter.

    Rotor generators square to one, so the two-point shift rule
    (E(theta+pi/2) - E(theta-pi/2)) / 2 is exact. Parameters shared by
    several rotors accumulate one shift contribution per occurrence.
    """
    bindings = program.parameters.bind(values)
    grads = {name: 0.0 for name in program.parameters.names}
    for k, op in enumerate(program.ops):
        if not (isinstance(op, Rotor) and isinstance(op.angle, Parameter)):
            continue
        theta = bindings[op.angle.name]
        shifted = []
        for shift in (math.pi / 2, -math.pi / 2):
            ops = list(program.ops)
            ops[k] = Rotor(op.word, theta + shift)
            clone = Program(program.n, ops, parameters=program.parameters)
            shifted.append(expectation_value(clone, observable, bindings, rho0))
        grads[op.angle.name] += (shifted[0] - shifted[1]) / 2.0
    return [grads[name] for name in program.parameters.names]


def adjoint_gradient(program: Program, observable: PauliSum, values,
                     rho0: MV | None = None) -> list[float]:
    """Adjoint-mode gradient: one forward state sweep plus one backward
    observable sweep, using d<O>/dtheta_k = -i/2 Tr(O_k [P_k, rho_k]).
    """
    bindings = program.parameters.bind(values)
    if rho0 is None:
        rho0 = ket_density(program.n, "0" * program.n)
    states = [rho0]
    for op in program.ops:
        states.append(evolve(states[-1], program.op_to_mv(op, bindings)))
    grads = {name: 0.0 for name in program.parameters.names}
    Ok = observable.to_mv()
    for k in range(len(program.ops) - 1, -1, -1):
        op = program.ops[k]
        if isinstance(op, Rotor) and isinstance(op.angle, Parameter):
            Pk = op.word.to_mv()
            rho_k = states[k + 1]
            grads[op.angle.name] += (-0.5j * (Ok * (Pk * rho_k - rho_k * Pk)).trace()).real
        Uk = program.op_to_mv(op, bindings)
        Ok = Uk.dagger() * Ok * Uk
    return [grads[name] for name in program.parameters.names]
