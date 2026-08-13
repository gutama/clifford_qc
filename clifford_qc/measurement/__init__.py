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
from .cost import (
    DeviceCard,
    SettingResources,
    break_even_surface,
    cost_schedule,
    estimator_information,
    inflate_shots_for_fidelity,
    setting_duration_us,
    setting_fidelity,
)
from .compiled import CompiledMeasurementSampler, CompiledSetting

__all__ = ["WordCache", "GroupedWordCache", "CommutatorBank", "simultaneous_z_radius",
           "jeffreys_mean_var", "empirical_bernstein_radius", "candidate_radius",
           "UniformFixed", "UniformDoubling", "VarianceProportional",
           "GroupVarianceOptimal", "variance_optimal_group_plan",
           "qubit_wise_commute", "qwc_groups", "shared_basis",
           "DeviceCard", "SettingResources", "break_even_surface",
           "cost_schedule", "estimator_information", "inflate_shots_for_fidelity",
           "setting_duration_us", "setting_fidelity"]
__all__ += ["CompiledSetting", "CompiledMeasurementSampler"]
