"""A-CASE: Rayleigh-Ritz in Clifford-algebra operator-response subspaces.

Phases 1-4 of ``PLAN.md``: exact fixed-basis solves, a cached
matrix-element bank with projected observables, adaptive basis growth, and the
finite-shot layers (shared grouped measurement, asymptotic Ritz uncertainty,
whole-pipeline response bootstrap, and a sample-split growth certificate with
abstention). The basis states
``A_i|psi>`` stay virtual throughout -- every projected matrix element is an
expectation on one reference state, assembled from the bilinear Pauli-word
trace pairing ``MV.trace_pairing``.
"""

from .adaptive import (
    ACASEConfig, ACASEState, ACASEStepOutcome, AdaptiveResult, CandidateScore,
    GrowthRecord, OverlapTarget, TargetOverlapScore, acase_step, run_acase,
    score_candidate, score_target_overlap, sector_leakage, select_candidate,
    select_target_candidate,
)
from ..workflows import adapt_warm_start
from .adapt_gcim import (
    AdaptGCIMIteration, AdaptGCIMResult, adapt_gcim_gradient, run_adapt_gcim,
)
from .elements import MatrixElementBank
from .streaming import StreamingMatrixElementBank
from .measured import (
    ASYMPTOTIC, EXACT, FINITE_SAMPLE, HEURISTIC, CertifiedGrowthRecord,
    CertifiedResult, CouplingBound, Interval, SharedMeasurement, WordFunctional,
    bootstrap_ritz, certify_couplings, coupling_functional, entry_functionals,
    ritz_functional, ritz_uncertainty, run_certified_acase,
)
from .generators import (
    commutator_response, compound_response, identity_generator,
    krylov_response, pauli_orbit, response_hierarchy,
)
from .generator_core import Generator, as_generators
from .configuration import (
    configuration_generator, configuration_generators,
    configuration_haar_packets, determinant_program, state_sector,
)
from .symmetry import (
    infer_reference_sector, project_reference_to_sector,
    reference_sector_leakage, subspace_sector_certificate,
)
from .restriction import (
    RestrictedProblem, Restriction, restricted_sector_operators,
)
from .contextual import (
    ContextualProblem, ContextualRestrictionPlan, ContextualStabilizer,
    ContextualStabilizerSelection, compile_contextual_restriction,
    project_contextual_problem, select_contextual_stabilizers,
)
from .mapping_invariants import (
    MappingInvariantReport, assert_mapping_invariants,
)
from .fermionic_generators import (
    determinant_excitations, fermionic_excitation_generators,
    occupied_spin_orbitals,
)
from .reference import (
    dense_basis, dense_projected_matrices, dense_residual_norm, dense_subspace,
    pure_statevector,
)
from .response import (ResponseLine, broaden_response, lehmann_spectrum,
                       static_susceptibility)
from .measured_response import (
    BootstrapResponse, MeasuredResponseSpectrum, ReplicaOutcome,
    ResponseLineUncertainty,
    ResponseMeasurement, bootstrap_response,
)
from .solver import (
    DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S, SubspaceResult,
    canonical_eigh, projected_matrices, solve_projected, solve_subspace,
)
from .projection import ProjectedProblem
from .hybrid import (
    FamilyReport, HybridArm, configuration_generators_from_words,
    dressed_family, family_projection_report, run_hybrid,
)
from .selected_ci import (
    ControlResult, SpanComparison, containment_residual, excitation_closure,
    family_closure, principal_angles, run_control, score_candidates,
    span_comparison,
)
from .qsci import (
    IMPLEMENTABLE, ORACLE, QSCIResult, SamplingRecord, StateInput,
    adapt_vqe_state, assert_single_evidence_category, exact_ground_state_oracle,
    recover_configurations, reference_determinant_state, run_qsci,
    sample_configurations, sample_state_input, sample_state_inputs,
    sampling_set_stability, sampling_stop_ready,
)
from .time_evolution import (
    apply_program, propagate, time_evolved_state, trotter_program,
)
from .multiresolution import (
    ConfigurationOrdering, HaarPacketNode, MultiresolutionResult,
    configuration_ordering, configuration_packet_hierarchy,
    run_coarse_to_fine_acase,
)
from .fragment import (
    FRAGMENT_SOLUTION_SCHEMA, ExactFragmentSolver, FragmentSolution,
    FragmentSolver, HybridFragmentSolver, QSCIFragmentSolver,
    SelectedCIFragmentSolver, energy_from_spatial_rdms, spatial_rdms,
    spin_orbital_rdms,
)

__all__ = [
    "FRAGMENT_SOLUTION_SCHEMA", "ExactFragmentSolver", "FragmentSolution",
    "FragmentSolver", "HybridFragmentSolver", "QSCIFragmentSolver",
    "SelectedCIFragmentSolver", "energy_from_spatial_rdms", "spatial_rdms",
    "spin_orbital_rdms",
    "FamilyReport", "HybridArm", "configuration_generators_from_words",
    "dressed_family", "family_projection_report", "run_hybrid",
    "ControlResult", "SpanComparison", "containment_residual",
    "excitation_closure", "family_closure", "principal_angles", "run_control",
    "score_candidates", "span_comparison",
    "IMPLEMENTABLE", "ORACLE", "QSCIResult", "SamplingRecord", "StateInput",
    "adapt_vqe_state", "assert_single_evidence_category",
    "exact_ground_state_oracle", "recover_configurations",
    "reference_determinant_state", "run_qsci", "sample_configurations",
    "sample_state_input", "sample_state_inputs", "sampling_set_stability",
    "sampling_stop_ready",
    "apply_program", "propagate", "time_evolved_state", "trotter_program",
    "ACASEConfig", "ACASEState", "ACASEStepOutcome", "AdaptiveResult",
    "CandidateScore", "GrowthRecord", "OverlapTarget", "TargetOverlapScore",
    "adapt_warm_start", "acase_step", "run_acase", "score_candidate",
    "score_target_overlap", "sector_leakage", "select_candidate",
    "select_target_candidate", "infer_reference_sector",
    "reference_sector_leakage", "subspace_sector_certificate",
    "project_reference_to_sector",
    "RestrictedProblem", "Restriction", "restricted_sector_operators",
    "ContextualProblem", "ContextualRestrictionPlan", "ContextualStabilizer",
    "ContextualStabilizerSelection", "compile_contextual_restriction",
    "project_contextual_problem", "select_contextual_stabilizers",
    "MappingInvariantReport", "assert_mapping_invariants",
    "ConfigurationOrdering", "HaarPacketNode", "MultiresolutionResult",
    "configuration_ordering", "configuration_packet_hierarchy",
    "run_coarse_to_fine_acase",
    "AdaptGCIMIteration", "AdaptGCIMResult", "adapt_gcim_gradient",
    "run_adapt_gcim",
    "ASYMPTOTIC", "EXACT", "FINITE_SAMPLE", "HEURISTIC",
    "CertifiedGrowthRecord", "CertifiedResult", "CouplingBound", "Interval",
    "SharedMeasurement", "WordFunctional", "bootstrap_ritz", "certify_couplings",
    "coupling_functional", "entry_functionals", "ritz_functional",
    "ritz_uncertainty", "run_certified_acase",
    "Generator", "MatrixElementBank", "StreamingMatrixElementBank", "as_generators", "commutator_response",
    "compound_response", "configuration_generator", "configuration_generators",
    "configuration_haar_packets", "determinant_excitations", "determinant_program",
    "fermionic_excitation_generators", "identity_generator", "krylov_response",
    "occupied_spin_orbitals", "pauli_orbit", "response_hierarchy", "state_sector",
    "dense_basis", "dense_projected_matrices", "dense_residual_norm",
    "dense_subspace", "pure_statevector",
    "ResponseLine", "broaden_response", "lehmann_spectrum",
    "static_susceptibility",
    "BootstrapResponse", "MeasuredResponseSpectrum", "ReplicaOutcome",
    "ResponseLineUncertainty", "ResponseMeasurement", "bootstrap_response",
    "DEFAULT_MAX_CONDITION", "DEFAULT_NORM_FLOOR", "DEFAULT_TAU_S",
    "SubspaceResult", "ProjectedProblem", "canonical_eigh", "projected_matrices",
    "solve_projected", "solve_subspace",
]
