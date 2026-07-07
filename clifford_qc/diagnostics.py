from __future__ import annotations

import math

from .multivector import MV
from .matrix import to_matrix
from .states import partial_transpose


def negativity(rho: MV, transposed: set[int]) -> float:
    import numpy as np
    ev = np.linalg.eigvalsh(to_matrix(partial_transpose(rho, transposed)))
    return float((sum(abs(x) for x in ev) - 1.0) / 2.0)


def vn_entropy(rho: MV, *, base: float = 2.0, tol: float = 1e-12) -> float:
    import numpy as np
    ev = np.linalg.eigvalsh(to_matrix(rho))
    vals = [max(float(x.real), 0.0) for x in ev if x > tol]
    if not vals:
        return 0.0
    log_base = math.log(base)
    return -sum(p * math.log(p) / log_base for p in vals)


def fidelity_pure(rho: MV, psi) -> float:
    import numpy as np
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    return float((psi.conj() @ to_matrix(rho) @ psi).real)


def trace_cyclicity_error(A: MV, B: MV, C: MV) -> float:
    return abs((A * B * C).trace() - (B * C * A).trace())
