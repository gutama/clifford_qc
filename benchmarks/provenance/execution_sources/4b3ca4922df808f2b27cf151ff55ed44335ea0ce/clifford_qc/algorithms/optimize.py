"""Optimizer adapters: SciPy L-BFGS-B when available, pure-Python Adam
fallback otherwise, behind one result type that counts evaluations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class OptimizeResult:
    x: tuple[float, ...]
    fun: float
    grad_norm: float
    iterations: int
    evaluations: int
    method: str


def _adam(f_eg, x0, *, maxiter, gtol, lr=0.05, bound=None):
    x = [float(v) for v in x0]
    m = [0.0] * len(x)
    v = [0.0] * len(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    evaluations = 0
    E, g = f_eg(x)
    evaluations += 1
    for it in range(1, maxiter + 1):
        if bound is not None and E < bound - 1e-9:
            raise RuntimeError(f"variational bound violated: {E} < {bound}")
        gnorm = math.sqrt(sum(float(gk) ** 2 for gk in g))
        if gnorm < gtol:
            return OptimizeResult(tuple(x), float(E), gnorm, it - 1, evaluations, "adam")
        for k in range(len(x)):
            m[k] = b1 * m[k] + (1 - b1) * float(g[k])
            v[k] = b2 * v[k] + (1 - b2) * float(g[k]) ** 2
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            x[k] -= lr * mh / (math.sqrt(vh) + eps)
        E, g = f_eg(x)
        evaluations += 1
    gnorm = math.sqrt(sum(float(gk) ** 2 for gk in g))
    return OptimizeResult(tuple(x), float(E), gnorm, maxiter, evaluations, "adam")


def minimize_energy(f_eg: Callable[[Sequence[float]], tuple[float, Sequence[float]]],
                    x0: Sequence[float], *, method: str = "auto", maxiter: int = 400,
                    gtol: float = 1e-8, bound: float | None = None) -> OptimizeResult:
    """Minimize E(x) given a joint (energy, gradient) callable.

    ``bound`` is an optional exact lower bound (e.g. the ground energy);
    dropping below it by more than 1e-9 raises, catching gradient bugs.
    """
    x0 = tuple(float(v) for v in x0)
    if not x0:
        E, _ = f_eg(x0)
        return OptimizeResult(x0, float(E), 0.0, 0, 1, "noop")
    if method not in {"auto", "lbfgsb", "adam"}:
        raise ValueError("method must be 'auto', 'lbfgsb', or 'adam'")

    if method in {"auto", "lbfgsb"}:
        try:
            import numpy as np
            from scipy.optimize import minimize as scipy_minimize

            evaluations = 0

            def fg(x):
                nonlocal evaluations
                evaluations += 1
                E, g = f_eg(tuple(float(v) for v in x))
                if bound is not None and E < bound - 1e-9:
                    raise RuntimeError(f"variational bound violated: {E} < {bound}")
                return float(E), np.asarray(g, dtype=float)

            res = scipy_minimize(fg, np.asarray(x0, dtype=float), jac=True,
                                 method="L-BFGS-B",
                                 options={"maxiter": maxiter, "gtol": gtol,
                                          "ftol": 1e-13, "maxls": 60})
            x = tuple(float(v) for v in res.x)
            E, g = f_eg(x)
            evaluations += 1
            gnorm = math.sqrt(sum(float(gk) ** 2 for gk in g))
            return OptimizeResult(x, float(E), gnorm, int(res.nit), evaluations, "lbfgsb")
        except ImportError:
            if method == "lbfgsb":
                raise
    return _adam(f_eg, x0, maxiter=maxiter, gtol=gtol, bound=bound)
