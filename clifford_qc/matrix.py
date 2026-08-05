"""Backward-compatible facade for the dense reference layer.

Dense matrices are no longer an implementation path.  New internal code should
import :mod:`clifford_qc.dense_reference` explicitly; this module remains so
existing users of ``clifford_qc.matrix`` keep working unchanged.
"""

from .dense_reference import (
    code_to_matrix,
    density_from_statevector,
    exact_ground,
    expm_matrix,
    from_matrix,
    single_pauli_mats,
    to_matrix,
)

__all__ = [
    "single_pauli_mats", "code_to_matrix", "to_matrix", "from_matrix",
    "density_from_statevector", "expm_matrix", "exact_ground",
]
