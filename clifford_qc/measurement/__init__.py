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
from .grouping import qwc_basis_cover, qubit_wise_commute, qwc_groups, shared_basis
from .block_commuting import (
    block_commuting_groups,
    block_commuting_partition,
    block_ranges,
    block_wise_commute,
)
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
from .planning import CompiledMeasurementPlan, compile_block_measurement_plan

__all__ = ["WordCache", "GroupedWordCache", "CommutatorBank", "simultaneous_z_radius",
           "jeffreys_mean_var", "empirical_bernstein_radius", "candidate_radius",
           "UniformFixed", "UniformDoubling", "VarianceProportional",
           "GroupVarianceOptimal", "variance_optimal_group_plan",
           "qubit_wise_commute", "qwc_groups", "qwc_basis_cover", "shared_basis",
           "DeviceCard", "SettingResources", "break_even_surface",
           "cost_schedule", "estimator_information", "inflate_shots_for_fidelity",
           "setting_duration_us", "setting_fidelity"]
__all__ += ["CompiledSetting", "CompiledMeasurementSampler"]
__all__ += ["CompiledMeasurementPlan", "compile_block_measurement_plan"]
__all__ += ["block_wise_commute", "block_commuting_partition",
            "block_commuting_groups", "block_ranges"]
