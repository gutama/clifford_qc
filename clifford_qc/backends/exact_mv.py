"""Exact density-multivector backend: the native execution path.

Evolution is gate-by-gate (never through the full program unitary) so the
Pauli support of the state stays as sparse as the circuit allows; the
backend records the support size after every gate for the S_max resource
metric.
"""

from __future__ import annotations

from ..multivector import MV
from ..states import evolve, expectation, ket_density
from ..ir import PauliSum, Program, adjoint_gradient


class ExactMVBackend:
    """Gate-by-gate exact evolution with adjoint gradients.

    ``last_support_history`` holds ``|supp(rho_t)|`` after each op of the
    most recent ``state()`` call (index 0 is the initial state).
    """

    def __init__(self, track_support: bool = True):
        self.track_support = track_support
        self.last_support_history: list[int] = []

    def state(self, program: Program, values=None, initial_state: MV | None = None) -> MV:
        rho = initial_state if initial_state is not None else ket_density(program.n, "0" * program.n)
        bindings = program.parameters.bind(values) if len(program.parameters) else {}
        history = [len(rho.terms)]
        for op in program.ops:
            rho = evolve(rho, program.op_to_mv(op, bindings))
            if self.track_support:
                history.append(len(rho.terms))
        if self.track_support:
            self.last_support_history = history
        return rho

    def expectation(self, program: Program, observable: PauliSum, values=None,
                    initial_state: MV | None = None) -> float:
        return expectation(self.state(program, values, initial_state), observable.to_mv()).real

    def gradient(self, program: Program, observable: PauliSum, values,
                 initial_state: MV | None = None) -> list[float]:
        return adjoint_gradient(program, observable, values, rho0=initial_state)

    @property
    def support_peak(self) -> int:
        return max(self.last_support_history, default=0)
