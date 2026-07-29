"""Benchmark model builders: spin chains, materials clusters, chemistry.

``spin`` and ``lattice`` need nothing beyond numpy. ``chemistry`` needs the
``chemistry`` extra (OpenFermion + PySCF) and is therefore imported by name
rather than re-exported here, so ``from clifford_qc.models import hubbard``
works in a bare install.
"""

from .lattice import (anderson_impurity, extended_hubbard, honeycomb_links,
                      hubbard, kanamori, kitaev_honeycomb, reference_sector,
                      spin_orbital)
from .observables import (double_occupancy, link_correlations, magnetization,
                          occupation, spin_correlation, spin_operators,
                          structure_factor, total_spin_squared)
from .spin import Model, random_ising, tfim, xxz

__all__ = [
    "Model", "tfim", "xxz", "random_ising",
    "hubbard", "extended_hubbard", "kanamori", "anderson_impurity",
    "kitaev_honeycomb", "honeycomb_links", "spin_orbital", "reference_sector",
    "occupation", "double_occupancy", "magnetization", "spin_correlation",
    "spin_operators", "structure_factor", "link_correlations",
    "total_spin_squared",
]
