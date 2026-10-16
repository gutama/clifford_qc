"""Algorithm layer: optimizers, pools, fixed-depth VQE, and ADAPT-VQE."""

from .optimize import OptimizeResult, minimize_energy
from .exchange import ExchangeOperator, exchange_pools, exchange_energy_gradient, run_exchange_adapt
from .more_adapt import more_energy_gradient, run_more_adapt
from .pools import PoolOperator, local_pool, all_words_pool, odd_y_filter, is_odd_y
from .vqe import VQEResult, hva_program, run_vqe
from .layering import words_commute, build_layer
from .initialization import (
    CLIFFORD_ANGLES, CliffordPointResult, StabilizerApprox, bound_hva_program,
    clifford_point_search, seed_model, stabilizer_ground_program,
    stabilizer_hamiltonian_approximation,
)
from .adapt import (
    SelectionStatus, SelectionRecord, AdaptConfig, AdaptState,
    AdaptStepOutcome, AdaptResult, ConfidenceSelector, RandomSelector,
    FastInspiredSelector, adapt_step, ansatz_program, run_adapt,
)
from .adapt_selectors import (
    AdaptSelectorProtocol, ConfidenceSelectionProtocol,
    PopulationSelectionProtocol, RandomSelectionProtocol,
)

__all__ = [
    "OptimizeResult", "minimize_energy",
    "ExchangeOperator", "exchange_pools", "exchange_energy_gradient", "run_exchange_adapt",
    "more_energy_gradient", "run_more_adapt",
    "PoolOperator", "local_pool", "all_words_pool", "odd_y_filter", "is_odd_y",
    "VQEResult", "hva_program", "run_vqe",
    "words_commute", "build_layer",
    "CLIFFORD_ANGLES", "CliffordPointResult", "StabilizerApprox",
    "bound_hva_program", "clifford_point_search", "seed_model",
    "stabilizer_ground_program", "stabilizer_hamiltonian_approximation",
    "SelectionStatus", "SelectionRecord", "AdaptConfig", "AdaptState",
    "AdaptStepOutcome", "AdaptResult", "adapt_step", "ConfidenceSelector",
    "RandomSelector", "FastInspiredSelector", "AdaptSelectorProtocol",
    "ConfidenceSelectionProtocol", "PopulationSelectionProtocol",
    "RandomSelectionProtocol", "ansatz_program", "run_adapt",
]
