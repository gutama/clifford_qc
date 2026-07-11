"""Fixed-depth VQE on IR programs.

``hva_program`` builds the Hamiltonian-variational ansatz of a model as a
single Program (reference Clifford prep + parameterized rotors, one shared
parameter per term group per layer); ``run_vqe`` minimizes the energy with
exact adjoint gradients through any backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..ir import Parameter, Program, Rotor, adjoint_gradient
from ..backends.exact_mv import ExactMVBackend
from .optimize import OptimizeResult, minimize_energy


@dataclass(frozen=True)
class VQEResult:
    energy: float
    parameters: tuple[float, ...]
    gradient_norm: float
    evaluations: int
    iterations: int
    support_peak: int
    wall_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


def hva_program(model, depth: int) -> Program:
    """HVA ansatz: reference prep, then per layer one shared-angle rotor
    group per Hamiltonian term group (theta order: layer-major, group-minor,
    e.g. TFIM depth-2 -> [zz_1, x_1, zz_2, x_2])."""
    if depth < 0:
        raise ValueError("depth must be non-negative")
    prog = Program(model.n)
    for op in model.reference.ops:
        prog.append(op)
    for layer in range(1, depth + 1):
        for group_name, words in model.hva_layers:
            param = Parameter(f"{group_name}_{layer}")
            for word in words:
                prog.append(Rotor(word, param))
    return prog


def run_vqe(model, program: Program | None = None, *, depth: int = 1,
            backend: ExactMVBackend | None = None, x0: Sequence[float] | None = None,
            method: str = "auto", maxiter: int = 400, gtol: float = 1e-8,
            bound: float | None = None) -> VQEResult:
    """Minimize <H> over the program's parameters with exact adjoint gradients.

    ``program`` defaults to ``hva_program(model, depth)``; ``x0`` defaults to
    zeros (the reference state itself).
    """
    if backend is None:
        backend = ExactMVBackend()
    if program is None:
        program = hva_program(model, depth)
    k = len(program.parameters)
    x0 = tuple(0.0 for _ in range(k)) if x0 is None else tuple(float(v) for v in x0)
    if len(x0) != k:
        raise ValueError(f"expected {k} initial parameters, got {len(x0)}")

    H = model.hamiltonian
    support_peak = 0

    def f_eg(theta):
        nonlocal support_peak
        E = backend.expectation(program, H, theta)
        support_peak = max(support_peak, getattr(backend, "support_peak", 0))
        g = adjoint_gradient(program, H, theta)
        return E, g

    start = time.perf_counter()
    res: OptimizeResult = minimize_energy(f_eg, x0, method=method, maxiter=maxiter,
                                          gtol=gtol, bound=bound)
    wall = time.perf_counter() - start
    return VQEResult(
        energy=res.fun, parameters=res.x, gradient_norm=res.grad_norm,
        evaluations=res.evaluations, iterations=res.iterations,
        support_peak=support_peak, wall_seconds=wall,
        metadata={"model": model.name, "optimizer": res.method,
                  "n_parameters": k, "n_ops": len(program.ops)},
    )
