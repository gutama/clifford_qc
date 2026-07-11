"""Finite-shot measurement layer: shared word cache, global commutator bank,
simultaneous confidence bounds, and shot-allocation policies."""

from .cache import WordCache
from .bank import CommutatorBank
from .confidence import simultaneous_z_radius, jeffreys_mean_var, empirical_bernstein_radius
from .allocation import UniformFixed, UniformDoubling, VarianceProportional

__all__ = ["WordCache", "CommutatorBank", "simultaneous_z_radius",
           "jeffreys_mean_var", "empirical_bernstein_radius",
           "UniformFixed", "UniformDoubling", "VarianceProportional"]
