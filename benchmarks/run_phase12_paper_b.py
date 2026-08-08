"""Phase 12 integrated Paper B ladder.

The Track A end point is a *resource ledger*, not another winner script.  Each
arm is normalized onto the same schema so a Paper B comparison cannot silently
drop an inconvenient cost column.  Pareto frontiers are computed only inside
one evidence category: in particular, an oracle-sampled QSCI hybrid never
dominates an implementable or exact-simulation arm merely because the oracle
has no state-preparation cost to report.

The default sampling input is the exact sector ground state.  That is Phase 8's
validation oracle and is labelled as such on QSCI, selected-CI, and hybrid
rows.  Use ``--sampling-input adapt`` for an implementable ansatz input; its
existing metadata still records that hardware construction cost is not yet
accounted by the exact simulator.

Run from the repository root, for example::

    python -m benchmarks.run_phase12_paper_b --systems hubbard_2x2,hubbard_2x3

The PySCF-built H4 systems require the chemistry extra; the committed FCIDUMP
H4 rung remains dependency-light.  QSCI-aligned H2O and stretched BeH2 are
available to the Phase 11 packet ensemble, but are deliberately not promoted
to this primary ladder until its expensive arms have been demonstrated on
those sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from benchmarks import run_acase_ladder as ladder
from benchmarks import run_phase10_hybrid as phase10
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.reproducibility import execution_provenance, stamp_record
from clifford_qc.subspace import (
    ACASEConfig,
    configuration_generators_from_words,
    configuration_ordering,
    dressed_family,
    identity_generator,
    run_acase,
    run_coarse_to_fine_acase,
    run_control,
)
from clifford_qc.subspace.qsci import (
    IMPLEMENTABLE,
    ORACLE,
    adapt_vqe_state,
    exact_ground_state_oracle,
    reference_determinant_state,
    run_qsci,
    sample_state_input,
)


ROOT = Path(__file__).resolve().parent
# Keep Phase 12's primary ladder at the systems that have actually been taken
# through this expensive driver.  The H2O/BeH2 additions are Phase 11 packet
# benchmarks; promoting them here would make an ordinary Paper B run include
# unvalidated 10/12-qubit fixed-Krylov jobs.
PRIMARY_SYSTEMS = phase10.PRIMARY_SYSTEMS

REQUIRED_ARMS = (
    "reference_state",
    "exact_sector",
    "fixed_qse",
    "fixed_krylov",
    "adapt_vqe",
    "acase",
    "qsci",
    "excitation_closure",
    "selected_ci",
    "budget_selected_ci",
    "matched_selected_ci",
    "qsci_dressed_acase",
    "qsci_haar_dressed_acase",
)

OPTIONAL_MEASURED_ARMS = ("adapt_vqe_finite", "acase_certified")

# Phase 12 says these fields are required.  A field may be ``None`` when it is
# genuinely not defined for an arm, but it may never disappear from the row.
REQUIRED_FIELDS = (
    "method",
    "evidence_category",
    "seed",
    "M",
    "retained_rank",
    "kappa_S",
    "energy",
    "energy_error",
    "absolute_error",
    "variance",
    "true_residual",
    "sampling_shots",
    "unique_configurations",
    "unique_yield",
    "duplicate_rate",
    "discard_rate",
    "recovery_rate",
    "state_preparation_label",
    "state_preparation_category",
    "state_preparations",
    "state_preparation_executions",
    "W",
    "grouping_contexts",
    "certified_shot_cost",
    "matrix_nonzeros",
    "matrix_bytes",
    "build_seconds",
    "solve_seconds",
    "peak_memory_bytes",
    "generator_support",
    "element_support",
    "wall_seconds",
)

PARETO_COST_AXES = (
    "M",
    "W",
    "sampling_shots",
    "state_preparation_executions",
    "certified_shot_cost",
    "matrix_nonzeros",
    "peak_memory_bytes",
    "wall_seconds",
)

# These fields have a meaningful value on every Phase 12 arm.  Other schema
# fields may be ``None`` where the resource is genuinely undefined (for
# example, ADAPT-VQE has no overlap-matrix condition number).
ALWAYS_POPULATED_FIELDS = (
    "method",
    "evidence_category",
    "seed",
    "M",
    "energy",
    "energy_error",
    "absolute_error",
    "sampling_shots",
    "wall_seconds",
)


def _current_rss_bytes() -> int | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


class _PeakRSS:
    """Same Linux RSS metric as QSCI, wrapped around whole benchmark arms."""

    def __init__(self, interval: float = 0.002) -> None:
        self.interval = interval
        self.baseline: int | None = None
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        value = _current_rss_bytes()
        if value is not None:
            self.peak = value if self.peak is None else max(self.peak, value)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.baseline = self.peak = _current_rss_bytes()
        if self.baseline is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.05, 5.0 * self.interval))
        self._sample()

    @property
    def delta(self) -> int | None:
        if self.baseline is None or self.peak is None:
            return None
        return max(0, self.peak - self.baseline)


def _observed(call):
    """Return ``(value, wall_seconds, incremental_peak_rss)`` for one arm."""
    started = time.perf_counter()
    with _PeakRSS() as rss:
        value = call()
    return value, float(time.perf_counter() - started), rss.delta


def _blank_record(method: str, evidence_category: str, seed: int) -> dict:
    row = {key: None for key in REQUIRED_FIELDS}
    row.update({
        "method": method,
        "evidence_category": evidence_category,
        "seed": int(seed),
        "metadata": {},
    })
    return row


def _finish_record(row: dict, exact_energy: float) -> dict:
    missing = [key for key in REQUIRED_FIELDS if key not in row]
    if missing:
        raise AssertionError(f"Phase 12 record dropped required fields: {missing}")
    if row["energy"] is None:
        raise AssertionError(f"{row['method']} did not report an energy")
    row["energy"] = float(row["energy"])
    row["energy_error"] = float(row["energy"] - exact_energy)
    row["absolute_error"] = float(abs(row["energy_error"]))
    unpopulated = [key for key in ALWAYS_POPULATED_FIELDS if row.get(key) is None]
    if row["evidence_category"] == "finite_sample":
        unpopulated.extend(
            key for key in (
                "grouping_contexts", "certified_shot_cost",
                "state_preparations", "state_preparation_executions")
            if row.get(key) is None)
    if row["evidence_category"].endswith("_sampled"):
        unpopulated.extend(
            key for key in (
                "unique_configurations", "unique_yield", "duplicate_rate",
                "discard_rate", "recovery_rate")
            if row.get(key) is None)
    if row["evidence_category"] == "implementable_sampled":
        unpopulated.extend(
            key for key in ("state_preparations", "state_preparation_executions")
            if row.get(key) is None)
    if unpopulated:
        raise AssertionError(
            f"{row['method']} left required resource fields unpopulated: "
            f"{sorted(set(unpopulated))}")
    return row


def _no_sampling_fields() -> dict:
    return {
        "sampling_shots": 0,
        "unique_configurations": None,
        "unique_yield": None,
        "duplicate_rate": None,
        "discard_rate": None,
        "recovery_rate": None,
        "state_preparation_label": "reference_program",
        "state_preparation_category": IMPLEMENTABLE,
        "state_preparations": 1,
        "state_preparation_executions": None,
    }


def _finite_shot_fields(method: str, raw: dict) -> dict:
    """Quantum executions for finite-shot ADAPT/A-CASE measurement arms."""
    total_shots = raw.get("total_shots")
    total_circuits = raw.get("total_circuits")
    if total_shots is None or total_circuits is None:
        raise AssertionError(
            f"finite-shot arm {method!r} did not report shots and circuits")
    operators = raw.get("operators")
    if operators is None:
        preparations = 1
        label = "reference_program"
    else:
        # Each accepted ADAPT operator produces a new ansatz program; the
        # terminal selection pass measures the final one as well.
        preparations = int(operators) + 1
        label = "adaptive_ansatz_programs"
    return {
        # On finite-shot selector arms this is estimator sampling rather than
        # determinant sampling.  ``certified_shot_cost`` carries the same
        # executions under the measurement-resource name used by the roadmap.
        "sampling_shots": int(total_shots),
        "unique_configurations": None,
        "unique_yield": None,
        "duplicate_rate": None,
        "discard_rate": None,
        "recovery_rate": None,
        "state_preparation_label": label,
        "state_preparation_category": IMPLEMENTABLE,
        "state_preparations": preparations,
        "state_preparation_executions": int(total_shots),
    }


def _sampling_fields(sampling, state) -> dict:
    accepted = int(sampling.accepted_shots)
    unique = int(sampling.unique_configurations)
    return {
        "sampling_shots": int(sampling.raw_shots),
        "unique_configurations": unique,
        "unique_yield": 0.0 if accepted == 0 else float(unique / accepted),
        "duplicate_rate": float(sampling.duplicate_fraction),
        "discard_rate": float(sampling.discarded_fraction),
        "recovery_rate": float(sampling.repaired_fraction),
        "state_preparation_label": state.label,
        "state_preparation_category": state.category,
        "state_preparations": sampling.state_preparations,
        "state_preparation_executions": sampling.state_preparation_executions,
    }


def _sampled_evidence(state) -> str:
    if state.category == ORACLE:
        return "oracle_sampled"
    if state.category == IMPLEMENTABLE:
        return "implementable_sampled"
    raise ValueError(f"unknown sampling evidence category {state.category!r}")


def _reference_word(backend: SectorStatevectorBackend, model) -> int:
    vector = backend.state_from_program(model.reference)
    return int(backend.basis[int(np.argmax(np.abs(vector)))])


def _reference_row(model, backend, operator, exact_energy: float, seed: int) -> dict:
    def evaluate():
        state = backend.state_from_program(model.reference)
        applied = operator.matvec(state)
        energy = float(np.vdot(state, applied).real)
        variance = max(0.0, float(np.vdot(applied, applied).real - energy ** 2))
        return energy, variance

    (energy, variance), wall, peak = _observed(evaluate)
    row = _blank_record("reference_state", "exact_simulation", seed)
    row.update(_no_sampling_fields())
    row.update({
        "M": 1,
        "retained_rank": 1,
        "kappa_S": 1.0,
        "energy": energy,
        "variance": variance,
        "true_residual": math.sqrt(variance),
        "peak_memory_bytes": peak,
        "generator_support": 1,
        "wall_seconds": wall,
        "metadata": {"reference": "model.reference"},
    })
    return _finish_record(row, exact_energy)


def _exact_row(backend, exact_energy: float, seed: int, wall: float,
               peak: int | None) -> dict:
    row = _blank_record("exact_sector", "reference", seed)
    row.update(_no_sampling_fields())
    row.update({
        "M": int(backend.dimension),
        "retained_rank": int(backend.dimension),
        "kappa_S": 1.0,
        "energy": exact_energy,
        "variance": 0.0,
        "true_residual": 0.0,
        "solve_seconds": wall,
        "peak_memory_bytes": peak,
        "wall_seconds": wall,
        "metadata": {"role": "exact-sector reference"},
    })
    return _finish_record(row, exact_energy)


def _krylov_depth_metadata(max_size: int, krylov_depth: int) -> dict:
    """Declare whether the Krylov arm ran at the ladder budget or below it.

    The arm's cost is set by the total word count of ``H^k |ref>``, which grows
    geometrically in the qubit count.  Measured on this repository: at depth 6
    the summed generator support is 8,128 words on the 8-qubit hubbard_2x2 and
    274,836 on the 12-qubit hubbard_2x3, and ``projected_matrices`` wall time
    runs close to linear in that sum (~9.3 ms per unit support), so the same
    declared depth is seconds on one system and tens of minutes on the other.

    A capped row is still a Rayleigh-Ritz subspace and still variational, but
    it is *not* the same method as an uncapped one: it spans a shorter Krylov
    space.  Recording the cap is what keeps the ladder honest -- a reader
    comparing ``fixed_krylov`` across systems has to be able to see that the
    arm was narrowed rather than that the system was harder.
    """
    if krylov_depth == max_size:
        return {"krylov_depth": krylov_depth, "krylov_depth_capped": False}
    return {
        "krylov_depth": krylov_depth,
        "krylov_depth_capped": True,
        "krylov_depth_budget": max_size,
        "krylov_depth_semantics": (
            f"Krylov space truncated to H^{krylov_depth} against the ladder's "
            f"max_size={max_size}; this arm spans a strictly smaller subspace "
            "than an uncapped fixed_krylov row and is not directly comparable "
            "to one"),
    }


def _ladder_row(method: str, spec: dict, model, kind: str, context: dict,
                exact_energy: float, seed: int,
                extra_metadata: dict | None = None) -> dict:
    raw, wall, peak = _observed(
        lambda: ladder.run_method(method, spec, model, kind, context))
    evidence = ("finite_sample" if raw.get("evidence") == "finite_sample"
                else "exact_simulation")
    row = _blank_record(method, evidence, seed)
    measured = evidence == "finite_sample"
    row.update(_finite_shot_fields(method, raw) if measured
               else _no_sampling_fields())
    basis_size = raw.get("basis_size")
    operators = raw.get("operators")
    if basis_size is None and operators is not None:
        basis_size = int(operators)
    metadata = {key: value for key, value in raw.items()
                if key not in {"energy", "evidence"}}
    if operators is not None:
        metadata.update({
            "M_semantics": (
                "selected ADAPT ansatz operators; an ADAPT size analogue, "
                "not a Rayleigh-Ritz subspace dimension"),
            "retained_rank_semantics": "not defined for single-state ADAPT-VQE",
            "kappa_S_semantics": "not defined: ADAPT-VQE has no overlap matrix",
        })
    if measured:
        metadata["sampling_shots_semantics"] = (
            "finite-shot estimator executions; not configuration sampling")
    if extra_metadata:
        metadata.update(extra_metadata)
    row.update({
        "M": basis_size,
        "retained_rank": raw.get("retained_rank"),
        "kappa_S": raw.get("condition_number"),
        "energy": raw["energy"],
        "W": raw.get("word_universe"),
        "grouping_contexts": raw.get("total_circuits") if measured else None,
        "certified_shot_cost": raw.get("total_shots") if measured else None,
        "peak_memory_bytes": peak,
        "generator_support": raw.get(
            "max_generator_support", raw.get("support_peak")),
        "element_support": raw.get("max_element_support"),
        "wall_seconds": wall,
        "metadata": metadata,
    })
    return _finish_record(row, exact_energy)


def _qsci_row(result, state, exact_energy: float, seed: int) -> dict:
    row = _blank_record("qsci", _sampled_evidence(state), seed)
    row.update(_sampling_fields(result.sampling, state))
    row.update({
        "M": result.subspace_dimension,
        "retained_rank": result.subspace_dimension,
        "kappa_S": 1.0,
        "energy": result.energy,
        "W": 0,
        "matrix_nonzeros": result.matrix_nonzeros,
        "matrix_bytes": result.matrix_bytes,
        "build_seconds": result.build_seconds,
        "solve_seconds": result.solve_seconds,
        "peak_memory_bytes": result.peak_rss_delta_bytes,
        "generator_support": None,
        "element_support": None,
        "wall_seconds": result.build_seconds + result.solve_seconds,
        "metadata": {
            "hermiticity_residual": result.hermiticity_residual,
            "retained_probability": result.sampling.retained_probability,
            "state_preparation_metadata": dict(state.metadata),
        },
    })
    return _finish_record(row, exact_energy)


def _control_row(method: str, control, sampling, state, exact_energy: float,
                 seed: int, wall: float, peak: int | None) -> dict:
    row = _blank_record(method, _sampled_evidence(state), seed)
    row.update(_sampling_fields(sampling, state))
    variance = max(0.0, float(control.variance))
    row.update({
        "M": control.determinant_count,
        "retained_rank": control.determinant_count,
        "kappa_S": 1.0,
        "energy": control.energy,
        "variance": variance,
        "true_residual": math.sqrt(variance),
        "W": 0,
        "matrix_nonzeros": control.matrix_nonzeros,
        "matrix_bytes": control.matrix_bytes,
        "build_seconds": control.build_seconds,
        "solve_seconds": control.solve_seconds,
        "peak_memory_bytes": peak,
        "wall_seconds": wall,
        "metadata": {
            "selection_work": int(control.selection_work),
            **dict(control.metadata),
            "state_preparation_metadata": dict(state.metadata),
        },
    })
    return _finish_record(row, exact_energy)


def _adaptive_hybrid_row(method: str, result, sampling, state,
                         exact_energy: float, seed: int, wall: float,
                         peak: int | None, metadata: dict,
                         selection_work_addend: int = 0) -> dict:
    solved = result.result
    resources = solved.resources
    row = _blank_record(method, _sampled_evidence(state), seed)
    row.update(_sampling_fields(sampling, state))
    row.update({
        "M": result.basis_size,
        "retained_rank": solved.effective_rank,
        "kappa_S": solved.condition_number,
        "energy": result.energy,
        "W": resources.get("word_universe"),
        "peak_memory_bytes": peak,
        "generator_support": resources.get("max_generator_support"),
        "element_support": resources.get("max_hamiltonian_element_support"),
        "wall_seconds": wall,
        "metadata": {
            "candidate_pool": result.resources.get("candidate_pool_size"),
            "selection_work": int(
                sum(r.candidates_scored for r in result.records)
                + selection_work_addend),
            "stopped_reason": result.stopped_reason,
            "labels": list(result.labels),
            "state_preparation_metadata": dict(state.metadata),
            **metadata,
        },
    })
    return _finish_record(row, exact_energy)


def _pareto_methods(rows: list[dict], cost_axis: str) -> list[str]:
    candidates = [
        row for row in rows
        if row.get("metadata", {}).get("pareto_eligible", True)
        if row.get(cost_axis) is not None
        and np.isfinite(float(row[cost_axis]))
        and np.isfinite(float(row["absolute_error"]))
    ]
    frontier = []
    for row in candidates:
        cost = float(row[cost_axis])
        error = float(row["absolute_error"])
        dominated = any(
            float(other[cost_axis]) <= cost
            and float(other["absolute_error"]) <= error
            and (float(other[cost_axis]) < cost
                 or float(other["absolute_error"]) < error)
            for other in candidates if other is not row
        )
        if not dominated:
            frontier.append(row)
    return [row["method"] for row in sorted(
        frontier, key=lambda item: (float(item[cost_axis]),
                                    float(item["absolute_error"]),
                                    item["method"]))]


def pareto_frontiers(rows: list[dict]) -> dict:
    """Two-dimensional error/cost fronts, partitioned by evidence category."""
    result = {}
    categories = sorted({row["evidence_category"] for row in rows})
    for category in categories:
        compatible = [row for row in rows if row["evidence_category"] == category]
        axes = {axis: _pareto_methods(compatible, axis) for axis in PARETO_COST_AXES}
        result[category] = {axis: methods for axis, methods in axes.items() if methods}
    return result


def _resolve_source_sha(provenance: dict) -> str:
    """Resolve the exact source revision before a Paper B record is written."""
    candidate = os.environ.get("PHASE12_BASE_SHA") or provenance.get("git_sha")
    if not candidate or candidate == "unknown":
        raise SystemExit(
            "no resolvable revision for this Phase 12 record: git reported "
            f"{provenance.get('git_sha')!r} and PHASE12_BASE_SHA is unset")
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet",
         f"{candidate.strip()}^{{commit}}"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    if resolved.returncode != 0 or not resolved.stdout.strip():
        raise SystemExit(
            f"{candidate!r} does not resolve to a commit in this repository")
    return resolved.stdout.strip()


def _sampling_state(which: str, backend, operator, model, pool, *, max_size: int):
    if which == "oracle":
        return exact_ground_state_oracle(backend, model.hamiltonian)
    if which == "reference":
        return reference_determinant_state(backend, model)
    if which == "adapt":
        return adapt_vqe_state(
            backend, model, pool, max_operators=max_size,
            compute_exact_reference=False)
    raise ValueError("sampling_input must be 'oracle', 'reference', or 'adapt'")


def _ordered_hybrid_inputs(model, backend, words, *, max_support: int,
                           max_generators: int, seed: int):
    reference_word = _reference_word(backend, model)
    ordering = configuration_ordering(
        words, method="physics", reference_word=reference_word, n=model.n,
        seed=seed)
    ordered_words = ordering.words
    reference_sampled = bool(np.any(ordered_words == reference_word))
    try:
        configurations = configuration_generators_from_words(
            model, ordered_words, label_prefix="p12cfg")
    except ValueError as exc:
        if "every sampled word was the reference determinant" not in str(exc):
            raise
        configurations = []
    sampled_basis = list(configurations)
    if reference_sampled:
        sampled_basis.append(identity_generator(model.n))
    if not sampled_basis:
        raise RuntimeError("sampled input produced no usable variational direction")
    family = dressed_family(
        sampled_basis, model, kind="excitation", max_support=max_support,
        max_generators=max_generators)
    seed_basis = None if reference_sampled else [configurations[0]]
    return ordering, configurations, family, seed_basis, reference_sampled


def run_system(name: str, *, shots: int = 128, seed: int = 0,
               max_size: int = 6, max_generators: int = 128,
               max_support: int = 64, max_packet_support: int = 16,
               sampling_input: str = "oracle",
               include_measured: bool = False,
               krylov_size: int | None = None) -> dict:
    """Run the complete Phase 12 ladder for one Track A primary system."""
    if shots <= 0 or max_size < 1 or max_generators < 1:
        raise ValueError("shots, max_size, and max_generators must be positive")
    krylov_depth = max_size if krylov_size is None else int(krylov_size)
    if not 1 <= krylov_depth <= max_size:
        raise ValueError(
            f"krylov_size must lie in [1, max_size={max_size}]; a cap above the "
            "ladder budget would make fixed_krylov span more than every other arm")
    if name not in PRIMARY_SYSTEMS:
        raise ValueError(f"unknown Phase 12 primary system {name!r}")

    model, construction = phase10.build_system(name)
    kind = "fermionic_lattice" if name.startswith("hubbard_") else "molecular"
    backend = SectorStatevectorBackend(
        model.n, int(model.metadata["n_electrons"]), float(model.metadata["sz"]))
    operator = backend.operator(model.hamiltonian)
    (exact_solution, exact_wall, exact_peak) = _observed(
        lambda: backend.ground_state(model.hamiltonian, k=1))
    exact_values, _ = exact_solution
    exact_energy = float(exact_values[0])
    rho = ExactMVBackend().state(model.reference, ())
    pool = ladder.word_pool(model, kind)
    candidates = ladder.build_candidates(model, kind)
    context = {
        "rho": rho,
        "reference": exact_energy,
        "sector": f"N={model.metadata['n_electrons']},Sz={model.metadata['sz']}",
        "sector_backend": backend,
        "sampling_operator": operator,
        "observables": {},
        "exact_observables": {},
        "pool": pool,
        "candidates": candidates,
        "candidates_level4": None,
    }

    rows = [
        _reference_row(model, backend, operator, exact_energy, seed),
        _exact_row(backend, exact_energy, seed, exact_wall, exact_peak),
        _ladder_row(
            "fixed_qse", {"kind": "qse", "size": max_size,
                           "selection": "stride", "seed": seed},
            model, kind, context, exact_energy, seed),
        _ladder_row(
            "fixed_krylov", {"kind": "krylov", "size": krylov_depth,
                              "max_tracked_support": 512, "seed": seed},
            model, kind, context, exact_energy, seed,
            extra_metadata=_krylov_depth_metadata(max_size, krylov_depth)),
        _ladder_row(
            "adapt_vqe", {"kind": "adapt_exact", "max_operators": max_size,
                           "seed": seed},
            model, kind, context, exact_energy, seed),
        _ladder_row(
            "acase", {"kind": "acase_exact", "max_size": max_size,
                       "leakage_tol": 1e-9, "seed": seed},
            model, kind, context, exact_energy, seed),
    ]

    state = _sampling_state(
        sampling_input, backend, operator, model, pool, max_size=max_size)
    indices, sampling = sample_state_input(state, shots=shots, seed=seed)
    words = backend.basis[indices]
    qsci = run_qsci(operator, indices, sampling=sampling, exact_energy=exact_energy)
    rows.append(_qsci_row(qsci, state, exact_energy, seed))

    control_specs = (
        ("excitation_closure", "excitation_closure", {}),
        ("selected_ci", "selected_ci", {}),
        ("budget_selected_ci", "budget_matched",
         {"max_determinants": max_size + 1}),
        # The sample-independent twin of the row above. `budget_matched` asks
        # what the best M determinants are for a method that has seen the
        # quantum sample; this asks what a laptop reaches on the same budget
        # having never seen it. Reporting both is what separates a hybrid
        # advantage from a sampling advantage, and it makes the older record's
        # truncation artifact visible as a measured difference.
        ("matched_selected_ci", "matched_selected_ci",
         {"max_determinants": max_size + 1}),
    )
    for method, control_kind, extra in control_specs:
        control, wall, peak = _observed(lambda kind_=control_kind, kw=extra: run_control(
            operator, indices, name=method, kind=kind_, n=model.n,
            exact_energy=exact_energy, **kw))
        rows.append(_control_row(
            method, control, sampling, state, exact_energy, seed, wall, peak))

    ordering, configurations, family, seed_basis, reference_sampled = (
        _ordered_hybrid_inputs(
            model, backend, words, max_support=max_support,
            max_generators=max_generators, seed=seed))
    final_pool = list(configurations) + list(family.generators)
    dressed, dressed_wall, dressed_peak = _observed(lambda: run_acase(
        rho, model.hamiltonian, final_pool, initial=seed_basis,
        max_size=max_size, exact_ground_energy=exact_energy))
    common_hybrid_metadata = {
        **family.to_record(),
        "configuration_order": "physics",
        "ordering_metadata": dict(ordering.metadata),
        "reference_sampled": reference_sampled,
        "sampled_words": [int(word) for word in ordering.words.tolist()],
    }
    rows.append(_adaptive_hybrid_row(
        "qsci_dressed_acase", dressed, sampling, state, exact_energy, seed,
        dressed_wall, dressed_peak,
        {**common_hybrid_metadata, "haar_stage": False}))

    if len(configurations) >= 2:
        packet_steps = max(1, max_size // 3)
        hierarchy, haar_wall, haar_peak = _observed(lambda: run_coarse_to_fine_acase(
            rho, model.hamiltonian, configurations, final_pool,
            initial=seed_basis,
            config=ACASEConfig(
                max_size=max_size, exact_ground_energy=exact_energy),
            packet_steps=packet_steps,
            max_packet_support=max_packet_support,
            label_prefix="p12H"))
        haar_result = hierarchy.result
        haar_metadata = {
            **common_hybrid_metadata,
            "haar_stage": True,
            "haar_object": "support-pruned orthonormal Haar packet hierarchy",
            "packet_budget": packet_steps,
            "packet_labels": list(hierarchy.packet_labels),
            "packet_directions": len(hierarchy.packet_labels),
            "packet_candidates": hierarchy.packet_candidates,
            "refined_intervals": [list(interval)
                                  for interval in hierarchy.refined_intervals],
            "frontiers_scored": hierarchy.frontiers_scored,
            "max_packet_support": max_packet_support,
        }
    else:
        # Fewer than two non-reference samples have no interval to transform.
        # Keep the arm explicit and carry the work of the dressed result it
        # reuses, but exclude this unavailable duplicate from Pareto ranking.
        haar_result, haar_wall, haar_peak = (
            dressed, dressed_wall, dressed_peak)
        haar_metadata = {
            **common_hybrid_metadata,
            "haar_stage": False,
            "haar_unavailable_reason": (
                "fewer than two non-reference sampled configurations"),
            "pareto_eligible": False,
            "packet_directions": 0,
            "packet_candidates": 0,
            "frontiers_scored": 0,
            "max_packet_support": max_packet_support,
        }
    rows.append(_adaptive_hybrid_row(
        "qsci_haar_dressed_acase", haar_result, sampling, state, exact_energy,
        seed, haar_wall, haar_peak, haar_metadata,
        selection_work_addend=(hierarchy.frontiers_scored
                                if len(configurations) >= 2 else 0)))

    if include_measured:
        rows.extend([
            _ladder_row(
                "adapt_vqe_finite",
                {"kind": "adapt_shot", "max_operators": max_size,
                 "base": 256, "max_factor": 16, "seed": seed},
                model, kind, context, exact_energy, seed),
            _ladder_row(
                "acase_certified",
                {"kind": "acase_certified", "max_size": min(max_size, 4),
                 "construction_shots": 2000, "certification_shots": 2000,
                 "delta": 0.05, "threshold": 0.05,
                 "leakage_tol": 1e-9, "seed": seed},
                model, kind, context, exact_energy, seed),
        ])

    present = {row["method"] for row in rows}
    missing_arms = sorted(set(REQUIRED_ARMS) - present)
    if missing_arms:
        raise AssertionError(f"Phase 12 ladder dropped required arms: {missing_arms}")
    for row in rows:
        _finish_record(row, exact_energy)
        if row["evidence_category"] != "finite_sample" \
                and row["energy_error"] < -1e-9:
            raise AssertionError(
                f"{row['method']} violates the variational bound by "
                f"{-row['energy_error']:.3e} Ha")

    return {
        "system_key": name,
        "system": model.name,
        "n_qubits": int(model.n),
        "n_electrons": int(model.metadata["n_electrons"]),
        "sz": float(model.metadata["sz"]),
        "sector_dimension": int(backend.dimension),
        "exact_energy": exact_energy,
        "construction": construction,
        "settings": {
            "shots": int(shots),
            "seed": int(seed),
            "max_growth_steps": int(max_size),
            "max_retained_directions": int(max_size + 1),
            "sampling_input": sampling_input,
            "sampling_evidence": state.category,
            "configuration_order": "physics",
            "max_family_generators": int(max_generators),
            "max_family_support": int(max_support),
            "max_packet_support": int(max_packet_support),
            "include_measured": bool(include_measured),
        },
        "sampling": sampling.to_record(),
        "arms": rows,
        "pareto_frontiers": pareto_frontiers(rows),
        "claim_boundary": (
            "oracle-sampled rows are capability validation only" if state.category == ORACLE
            else "sampling state is implementable in principle; inspect preparation metadata "
                 "for unaccounted construction cost"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(PRIMARY_SYSTEMS))
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=6)
    parser.add_argument(
        "--krylov-size", type=int, default=None,
        help="Krylov depth for the fixed_krylov arm; defaults to --max-size. "
             "Lower it when H^k word growth makes the full depth impractical: "
             "the summed generator support at depth 6 is 8,128 words on the "
             "8-qubit hubbard_2x2 but 274,836 on the 12-qubit hubbard_2x3. A "
             "capped run is recorded as krylov_depth_capped so the narrowed "
             "arm cannot be mistaken for a full-depth one.")
    parser.add_argument("--max-generators", type=int, default=128)
    parser.add_argument("--max-support", type=int, default=64)
    parser.add_argument("--max-packet-support", type=int, default=16)
    parser.add_argument(
        "--sampling-input", choices=("oracle", "reference", "adapt"),
        default="oracle")
    parser.add_argument("--include-measured", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results" / "phase12_paper_b_rerun.json")
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing output file (never enabled by default)")
    args = parser.parse_args(argv)

    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    unknown = sorted(set(systems) - set(PRIMARY_SYSTEMS))
    if unknown:
        raise SystemExit(f"unknown Phase 12 systems: {unknown}")
    if args.out.exists() and not args.force:
        raise SystemExit(
            f"output {args.out} already exists; pass --force to replace it")

    # Revision identity is a precondition, not a postcondition.  Fail before a
    # potentially long multi-system run rather than discarding completed work.
    provenance = execution_provenance()
    source_base_sha = _resolve_source_sha(provenance)

    records = []
    for name in systems:
        record = run_system(
            name, shots=args.shots, seed=args.seed, max_size=args.max_size,
            max_generators=args.max_generators, max_support=args.max_support,
            max_packet_support=args.max_packet_support,
            sampling_input=args.sampling_input,
            include_measured=args.include_measured,
            krylov_size=args.krylov_size)
        records.append(record)
        print(
            f"{name:27s} exact={record['exact_energy']:+.9f} "
            f"sampled={record['sampling']['unique_configurations']:3d}",
            flush=True)
        for row in record["arms"]:
            m = "-" if row["M"] is None else str(row["M"])
            w = "-" if row["W"] is None else str(row["W"])
            print(
                f"  {row['method']:27s} M={m:>3s} "
                f"err={row['energy_error']:+.3e} W={w}", flush=True)

    result = stamp_record({
        "schema": "clifford_qc.phase12_paper_b.v1",
        "source_base_git_sha": source_base_sha,
        "evidence": "partitioned_by_arm; see evidence_category",
        "claim_boundary": (
            "Pareto frontiers never mix evidence categories; oracle sampling is "
            "validation rather than an implementable state-preparation claim"),
        "required_fields": list(REQUIRED_FIELDS),
        "systems": records,
    }, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} systems to {args.out}", flush=True)


if __name__ == "__main__":
    main()
