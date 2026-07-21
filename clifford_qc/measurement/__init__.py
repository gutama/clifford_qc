"""Finite-shot measurement layer: shared word cache, global commutator bank,
simultaneous confidence bounds, and shot-allocation policies."""

from .cache import WordCache, GroupedWordCache
from .bank import CommutatorBank
from .confidence import (simultaneous_z_radius, jeffreys_mean_var,
                         empirical_bernstein_radius, candidate_radius)
from .allocation import UniformFixed, UniformDoubling, VarianceProportional
from .grouping import qubit_wise_commute, qwc_groups, shared_basis

__all__ = ["WordCache", "GroupedWordCache", "CommutatorBank", "simultaneous_z_radius",
           "jeffreys_mean_var", "empirical_bernstein_radius", "candidate_radius",
           "UniformFixed", "UniformDoubling", "VarianceProportional",
           "qubit_wise_commute", "qwc_groups", "shared_basis"]
