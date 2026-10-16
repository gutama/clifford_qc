"""Cross-layer orchestration workflows.

This module owns bridges between algorithms and projected-subspace kernels so
neither numerical layer depends on the other's workflow implementation.
"""

from __future__ import annotations

from .backends.exact_mv import ExactMVBackend


def adapt_warm_start(model, pool, *, max_operators: int = 4, **kwargs):
    """Run exact ADAPT-VQE and return ``(rho, AdaptResult)`` for A-CASE.

    A-CASE treats the optimized ADAPT state as its single reference, so the
    subspace grows around a correlated state rather than a bare determinant.
    """
    # Imported at call time so importing either numerical package does not
    # eagerly initialize the other through this orchestration bridge.
    from .algorithms.adapt import ansatz_program, run_adapt

    result = run_adapt(model, pool, max_operators=max_operators, **kwargs)
    by_label = {op.label: op for op in pool}
    # Preserve the selected ansatz order; filtering the pool by membership
    # would silently reorder noncommuting rotors.
    chosen = [by_label[label] for label in result.labels]
    program = ansatz_program(model, chosen)
    return ExactMVBackend().state(program, result.parameters), result


__all__ = ["adapt_warm_start"]
