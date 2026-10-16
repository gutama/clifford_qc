"""Small representation contracts shared by subspace layers."""

from __future__ import annotations

from ..ir import PauliSum
from ..multivector import MV


def as_multivector(operator) -> MV:
    if isinstance(operator, MV):
        return operator
    if isinstance(operator, PauliSum):
        return operator.to_mv()
    raise TypeError(
        f"expected MV or PauliSum, got {type(operator).__name__}")


def check_reference(rho: MV) -> float:
    """Validate a reference density and return its exact purity."""
    if not rho.is_hermitian():
        raise ValueError("reference state must be Hermitian")
    trace = rho.trace()
    if abs(trace - 1.0) > 1e-9:
        raise ValueError(f"reference state must have unit trace, got {trace}")
    return float((2 ** rho.n) * rho.trace_pairing(rho).real)
