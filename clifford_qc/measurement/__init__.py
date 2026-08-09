"""Finite-shot measurement layer: shared word cache, global commutator bank,
simultaneous confidence bounds, and shot-allocation policies."""

from .cache import WordCache, GroupedWordCache
from .bank import CommutatorBank
from .confidence import (simultaneous_z_radius, jeffreys_mean_var,
                         empirical_bernstein_radius, candidate_radius)
from .allocation import (
    GroupVarianceOptimal, UniformFixed, UniformDoubling,
    VarianceProportional, variance_optimal_group_plan,
)
from .grouping import qubit_wise_commute, qwc_groups, shared_basis

__all__ = ["WordCache", "GroupedWordCache", "CommutatorBank", "simultaneous_z_radius",
           "jeffreys_mean_var", "empirical_bernstein_radius", "candidate_radius",
           "UniformFixed", "UniformDoubling", "VarianceProportional",
           "GroupVarianceOptimal", "variance_optimal_group_plan",
           "qubit_wise_commute", "qwc_groups", "shared_basis"]
