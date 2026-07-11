"""Algorithm layer: optimizers, pools, fixed-depth VQE, and ADAPT-VQE."""

from .optimize import OptimizeResult, minimize_energy
from .pools import PoolOperator, local_pool, all_words_pool, odd_y_filter, is_odd_y
from .vqe import VQEResult, hva_program, run_vqe
from .layering import words_commute, build_layer
from .adapt import (
    SelectionStatus, SelectionRecord, AdaptResult, ConfidenceSelector,
    RandomSelector, run_adapt,
)

__all__ = [
    "OptimizeResult", "minimize_energy",
    "PoolOperator", "local_pool", "all_words_pool", "odd_y_filter", "is_odd_y",
    "VQEResult", "hva_program", "run_vqe",
    "words_commute", "build_layer",
    "SelectionStatus", "SelectionRecord", "AdaptResult", "ConfidenceSelector",
    "RandomSelector", "run_adapt",
]
