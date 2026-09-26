r"""QSCI/SQD: sampled determinant subspaces as a first-class method.

Phase 8 of ``PLAN.md``.  A sampled-subspace eigensolver draws
computational-basis configurations from a prepared state, keeps the distinct
ones, and diagonalizes the Hamiltonian restricted to their span.  Its cost
profile is the complement of A-CASE's: the projected matrix needs **zero**
measured Pauli words because it is built classically, and the quantum spend
moves entirely into state preparation and sampling.

Three things this module is deliberate about.

*The restriction is a submatrix, not a second Hamiltonian.*  Both
:meth:`~clifford_qc.backends.sector_statevector.SectorOperator.restrict` and
:meth:`~clifford_qc.pauli_action.PauliLinearOperator.restrict` slice the same
compiled action their matvecs use, so the sampled matrix inherits the operator
tests instead of needing an independent Slater-Condon implementation to be
trusted.  Writing those rules is a later optimization, not a prerequisite.

*``W = 0`` is not a resource verdict.*  :class:`QSCIResult` therefore also
carries sampling yield, duplicate fraction, retained probability, matrix
nonzeros, build and solve time, and matrix bytes -- the costs that replace the
measurement cost rather than vanishing with it.

*Spin models get an arm.*  There is no fermionic recovery rule for a Kitaev
cluster, but raw computational-basis sampled diagonalization is still a
baseline.  The open question is whether that basis is compact, not whether it
exists, so :func:`run_qsci` accepts a full-space operator with no sector.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..backends.sector_statevector import SectorOperator, _spin_sites
from ..pauli_action import PauliLinearOperator
from ..selection import EvidenceLevel

__all__ = [
    "IMPLEMENTABLE",
    "ORACLE",
    "SamplingRecord",
    "QSCIResult",
    "StateInput",
    "adapt_vqe_state",
    "assert_single_evidence_category",
    "exact_ground_state_oracle",
    "reference_determinant_state",
    "sample_configurations",
    "sample_state_input",
    "sample_state_inputs",
    "recover_configurations",
    "run_qsci",
    "sampling_set_stability",
    "sampling_stop_ready",
]

# The §8D evidence split.  An oracle input is one no hardware can prepare --
# it exists to validate the method, and its energy is not a claim about what a
# device would produce.  Mixing the two inside one comparison is the specific
# confusion §8D forbids, so the category travels on every record.
ORACLE = "oracle"
IMPLEMENTABLE = "implementable"
_CATEGORIES = (ORACLE, IMPLEMENTABLE)


def _current_rss_bytes() -> int | None:
    """Current resident set size on Linux, or ``None`` when unavailable.

    The benchmark suite already uses ``/proc/self/statm`` for molecular peak
    RSS because it includes NumPy/BLAS allocations that Python-only allocation
    tracers miss.  Keep the same metric here; portability is represented by a
    missing value rather than by fabricating a cross-platform estimate.
    """
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


class _PeakRSS:
    """Low-overhead RSS sampler covering restriction and eigensolve work."""

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

    def __enter__(self) -> "_PeakRSS":
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


@dataclass(frozen=True)
class SamplingRecord:
    """The §8A sampling contract.

    Every field is a count or a fraction of counts, so a hardware-noise layer
    can populate the same object without changing its meaning: post-selection
    and recovery are later layers *over* this contract, not variants of it.
    """

    raw_shots: int
    accepted_shots: int
    discarded_shots: int
    repaired_shots: int
    unique_configurations: int
    retained_probability: float
    input_label: str
    seed: int | None
    recovery: str = "none"
    input_category: str = "unspecified"
    # Project-wide convention: ``state_preparations`` counts *distinct*
    # preparation circuits/states.  QSCI also needs the execution count because
    # every computational-basis shot destroys the state and therefore requires
    # a fresh preparation.
    state_preparations: int | None = None
    state_preparation_executions: int | None = None
    # For occupancy recovery this is deliberately counted on the pre-repair
    # draws: repair can merge distinct words and would bias Good-Turing unseen
    # mass downward exactly when recovery is doing the most work.
    singleton_configurations: int = 0
    bootstrap_set_stability: float | None = None

    @property
    def duplicate_fraction(self) -> float:
        """Share of accepted shots that landed on an already-seen configuration."""
        if self.accepted_shots <= 0:
            return 0.0
        return 1.0 - self.unique_configurations / self.accepted_shots

    @property
    def discarded_fraction(self) -> float:
        if self.raw_shots <= 0:
            return 0.0
        return self.discarded_shots / self.raw_shots

    @property
    def repaired_fraction(self) -> float:
        if self.raw_shots <= 0:
            return 0.0
        return self.repaired_shots / self.raw_shots

    @property
    def unseen_mass_estimate(self) -> float:
        """Good-Turing first-order unseen-mass estimate ``N_1 / N`` (§11D)."""
        if self.accepted_shots <= 0:
            return 0.0
        return self.singleton_configurations / self.accepted_shots

    def to_record(self) -> dict:
        return {
            "raw_shots": int(self.raw_shots),
            "accepted_shots": int(self.accepted_shots),
            "discarded_shots": int(self.discarded_shots),
            "repaired_shots": int(self.repaired_shots),
            "unique_configurations": int(self.unique_configurations),
            "duplicate_fraction": float(self.duplicate_fraction),
            "discarded_fraction": float(self.discarded_fraction),
            "repaired_fraction": float(self.repaired_fraction),
            "retained_probability": float(self.retained_probability),
            "singleton_configurations": int(self.singleton_configurations),
            "unseen_mass_estimate": float(self.unseen_mass_estimate),
            "bootstrap_set_stability": self.bootstrap_set_stability,
            "input_label": self.input_label,
            "input_category": self.input_category,
            "state_preparations": self.state_preparations,
            "state_preparation_executions": self.state_preparation_executions,
            "recovery": self.recovery,
            "seed": self.seed,
        }


def sampling_set_stability(samples, *, replicates: int = 128,
                           seed: int | None = 0) -> float:
    """Bootstrap stability of the observed configuration set (§11D).

    Each replicate resamples the accepted shots with replacement and reports
    the fraction of the observed unique set recovered.  The mean is a compact
    ``[0, 1]`` stability statistic.  It is deliberately paired with the
    Good-Turing singleton estimate: an empirical bootstrap cannot invent an
    unseen configuration, so stability alone is not an unseen-mass estimate.
    """
    samples = np.asarray(samples, dtype=np.int64).reshape(-1)
    if samples.size == 0:
        raise ValueError("set stability needs at least one accepted sample")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    observed = np.unique(samples)
    rng = np.random.default_rng(seed)
    recovered = 0.0
    for _ in range(int(replicates)):
        draw = rng.choice(samples, size=samples.size, replace=True)
        recovered += np.unique(draw).size / observed.size
    return float(recovered / replicates)


def sampling_stop_ready(record: SamplingRecord, *, max_unseen_mass: float = 0.05,
                        min_set_stability: float = 0.95,
                        min_duplicate_fraction: float | None = None) -> bool:
    """Declared Phase-11D baseline stop using sampling diagnostics.

    Thresholds are caller policy, never hidden tuning constants.  Duplicate
    rate is optional because a compact physical distribution can saturate at a
    very different rate from a broad one; when supplied it is required in
    addition to unseen-mass and bootstrap stability.
    """
    if not 0.0 <= max_unseen_mass <= 1.0:
        raise ValueError("max_unseen_mass must lie in [0, 1]")
    if not 0.0 <= min_set_stability <= 1.0:
        raise ValueError("min_set_stability must lie in [0, 1]")
    if min_duplicate_fraction is not None and not 0.0 <= min_duplicate_fraction <= 1.0:
        raise ValueError("min_duplicate_fraction must lie in [0, 1]")
    if record.bootstrap_set_stability is None:
        return False
    ready = (record.unseen_mass_estimate <= max_unseen_mass
             and record.bootstrap_set_stability >= min_set_stability)
    if min_duplicate_fraction is not None:
        ready = ready and record.duplicate_fraction >= min_duplicate_fraction
    return bool(ready)


@dataclass(frozen=True)
class QSCIResult:
    """Energy plus the resources that replace the measurement cost."""

    energy: float
    eigenvalues: np.ndarray
    indices: np.ndarray
    sampling: SamplingRecord
    evidence: str
    hermiticity_residual: float
    matrix_nonzeros: int
    matrix_bytes: int
    peak_rss_bytes: int | None
    peak_rss_delta_bytes: int | None
    build_seconds: float
    solve_seconds: float
    exact_energy: float | None = None
    metadata: dict = field(default_factory=dict)
    # Retained Ritz vectors in the sampled-configuration basis.  Phase 11 uses
    # these as optional classical overlap targets; they are not a new quantum
    # resource and are intentionally omitted from serialized benchmark rows.
    eigenvectors: np.ndarray | None = None

    @property
    def subspace_dimension(self) -> int:
        """``M`` -- the number of distinct sampled configurations."""
        return int(self.indices.size)

    @property
    def variational_gap(self) -> float | None:
        """``E_QSCI - E_exact``, non-negative up to conditioning when known."""
        if self.exact_energy is None:
            return None
        return float(self.energy - self.exact_energy)

    def to_record(self) -> dict:
        record = {
            "method": "qsci",
            "energy": float(self.energy),
            "subspace_dimension": self.subspace_dimension,
            # The whole point of the arm: the projected matrix is classical.
            "projected_matrix_words": 0,
            "matrix_nonzeros": int(self.matrix_nonzeros),
            "matrix_bytes": int(self.matrix_bytes),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_delta_bytes": self.peak_rss_delta_bytes,
            "hermiticity_residual": float(self.hermiticity_residual),
            "build_seconds": float(self.build_seconds),
            "solve_seconds": float(self.solve_seconds),
            "evidence": self.evidence,
        }
        record.update(self.sampling.to_record())
        if self.exact_energy is not None:
            record["exact_energy"] = float(self.exact_energy)
            record["variational_gap"] = float(self.variational_gap)
        record.update(self.metadata)
        return record


@dataclass(frozen=True)
class StateInput:
    """A declared sampling state, with what it costs and whether it is real (§8D).

    ``basis`` and ``post_selection`` together name one of **three** modes, and
    it is ``post_selection`` -- not ``basis`` alone -- that distinguishes the
    last two:

    ===================  ==================  =================  ==============
    mode                 ``amplitudes``      ``basis``          ``post_selection``
    ===================  ==================  =================  ==============
    full space           ``2^n``             ``None``           ``None``
    sector               sector length       occupation words   ``None``
    post-selected        ``2^n``             occupation words   spec mapping
    ===================  ==================  =================  ==============

    So a post-selected input carries full-space amplitudes *and* a non-``None``
    ``basis``: the amplitudes are what gets sampled, and the basis is what the
    surviving words are mapped back onto.  Use :func:`sample_state_input`, which
    reads both fields and dispatches; passing ``basis=state.basis`` straight to
    :func:`sample_configurations` is wrong for this mode and will fail its
    length check.

    ``preparations`` follows the repository-wide resource convention: it counts
    distinct preparation circuits/states, not repeated executions.  An
    implementable QSCI input normally has one such circuit, but every raw shot
    still executes it once; :func:`sample_state_input` therefore records both
    quantities.  Both are ``None`` for an oracle, which is exactly why an oracle
    row cannot sit on a resource axis beside an implementable one.

    The post-selected mode exists because a qubit-ADAPT ansatz built from
    individual Pauli words does not conserve particle number, so its state
    genuinely carries weight outside the sector.  Sampling it full-space and
    post-selecting is the honest treatment -- it is what a device would face --
    and the discarded fraction lands on the record instead of being defined
    away.
    """

    label: str
    category: str
    amplitudes: np.ndarray
    basis: np.ndarray | None = None
    preparations: int | None = None
    post_selection: dict | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.category not in _CATEGORIES:
            raise ValueError(f"category must be one of {_CATEGORIES}, "
                             f"got {self.category!r}")
        if self.category == ORACLE and self.preparations is not None:
            raise ValueError("an oracle input has no preparation cost to report")
        if self.category == IMPLEMENTABLE and self.preparations is None:
            raise ValueError(f"implementable input {self.label!r} must declare "
                             "its state-preparation count")
        if self.preparations is not None and self.preparations <= 0:
            raise ValueError("state-preparation count must be positive")


def assert_single_evidence_category(records) -> str:
    """Reject a comparison that mixes oracle and implementable inputs (§8D).

    Accepts anything carrying a category: :class:`StateInput`,
    :class:`SamplingRecord`, :class:`QSCIResult`, or a plain record mapping.
    Returns the single category found, so a caller can label the comparison
    with it.

    This exists because the mixing failure is silent otherwise.  An oracle row
    and an implementable row are both real numbers with the same units, and a
    Pareto plot will happily draw a frontier through both -- one that no device
    can reach, presented as if it could.
    """
    categories = set()
    for record in records:
        if isinstance(record, QSCIResult):
            categories.add(record.sampling.input_category)
        elif isinstance(record, (StateInput, SamplingRecord)):
            categories.add(record.category if isinstance(record, StateInput)
                           else record.input_category)
        else:
            categories.add(dict(record).get("input_category", "unspecified"))
    if not categories:
        raise ValueError("no records to check")
    if len(categories) > 1:
        raise ValueError(
            "cannot compare inputs from different evidence categories: "
            f"{sorted(categories)}. An oracle state is a method-validation "
            "input; putting it on a resource axis beside an implementable one "
            "advertises a frontier no device can reach.")
    return categories.pop()


def _to_sampling_basis(backend, dense: np.ndarray, *, tol: float = 1e-9):
    """``(amplitudes, basis, post_selection, leaked)`` for a prepared state.

    ``backend`` is ``None`` for models with no particle-number sector, where the
    full-space vector *is* the sampling state.  Where a sector exists the vector
    is restricted to it when it fits, and otherwise kept full-space with a
    post-selection spec: a leaking state is not an error, it is the ordinary
    situation for an ansatz built from non-conserving words, and discarding its
    out-of-sector shots is what a device would have to do anyway.
    """
    dense = np.asarray(dense, dtype=complex).reshape(-1)
    if backend is None:
        return dense, None, None, 0.0
    restricted = dense[backend.basis]
    # A difference of two norms, so a non-leaking state lands a few ulp below
    # zero. Clamped, because a negative leaked weight is not a small number --
    # it is a meaningless one, and it would travel onto the record as such.
    leaked = max(0.0, float(np.vdot(dense, dense).real
                            - np.vdot(restricted, restricted).real))
    if leaked <= tol:
        return restricted, backend.basis, None, leaked
    return dense, backend.basis, {
        "n": backend.n, "n_electrons": backend.n_electrons, "sz": backend.sz,
        "spin_ordering": backend.spin_ordering}, leaked


def _determinant_amplitudes(program, n: int) -> np.ndarray:
    """One-hot full-space vector of an X-gate-only reference program.

    Read off the gates rather than routed through a density multivector: a
    determinant on ``n`` qubits is one nonzero amplitude, while
    ``to_matrix(rho)`` would sum ``2^n`` Kronecker products into a ``2^n x 2^n``
    matrix to rediscover it.  At twelve qubits that is the difference between a
    dictionary lookup and several minutes.  Mirrors the bit convention of
    :meth:`SectorStatevectorBackend.state_from_program`.
    """
    mask = 0
    for operation in program.ops:
        if getattr(operation, "name", None) != "X":
            raise ValueError("only X-gate determinant references map to a "
                             "single computational basis state")
        for qubit in operation.qubits:
            mask ^= 1 << (n - 1 - qubit)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[mask] = 1.0
    return psi


def reference_determinant_state(backend, model) -> StateInput:
    """The reference determinant -- the cheapest implementable input.

    Sampling it returns exactly one configuration, which is the point: it is
    the floor every other input has to beat, not a competitive arm.
    """
    if backend is None:
        amplitudes, basis = _determinant_amplitudes(model.reference, model.n), None
    else:
        # Raises if the reference is not in this sector, which is the right
        # failure: a reference outside its own sector is a configuration error.
        amplitudes, basis = backend.state_from_program(model.reference), backend.basis
    return StateInput(label="reference_determinant", category=IMPLEMENTABLE,
                      amplitudes=amplitudes, basis=basis, preparations=1,
                      metadata={"state_kind": "computational_determinant"})


def exact_ground_state_oracle(operator_or_backend, hamiltonian=None) -> StateInput:
    """The exact ground state, for method validation only.

    Pass a :class:`~clifford_qc.pauli_action.PauliLinearOperator` for a
    full-space model, or a sector backend together with its Hamiltonian.  The
    returned input is labelled :data:`ORACLE`, so any attempt to place it on a
    resource comparison with an implementable arm fails loudly.
    """
    if isinstance(operator_or_backend, PauliLinearOperator):
        _, vectors = operator_or_backend.ground_state()
        return StateInput(label="exact_ground_oracle", category=ORACLE,
                          amplitudes=vectors[:, 0], basis=None,
                          metadata={"state_kind": "exact_eigenvector"})
    if hamiltonian is None:
        raise ValueError("a sector backend needs its Hamiltonian to solve for")
    backend = operator_or_backend
    _, vectors = backend.ground_state(hamiltonian)
    return StateInput(label="exact_ground_oracle", category=ORACLE,
                      amplitudes=vectors[:, 0], basis=backend.basis,
                      metadata={"state_kind": "exact_eigenvector"})


def adapt_vqe_state(backend, model, pool, *, max_operators: int = 4,
                    **kwargs) -> StateInput:
    """An optimized ADAPT-VQE state as the sampling input.

    The correlated implementable input: it costs a full ADAPT run to build and
    one distinct preparation circuit thereafter.  This helper currently builds
    the ansatz with the exact simulator, so its zero quantum-shot/circuit counts
    are explicitly labelled as *unaccounted hardware construction cost* rather
    than being allowed to read as a free experimental optimization.
    """
    from ..workflows import adapt_warm_start
    from .reference import pure_statevector

    rho, result = adapt_warm_start(model, pool, max_operators=max_operators,
                                   **kwargs)
    amplitudes, basis, post, leaked = _to_sampling_basis(backend,
                                                         pure_statevector(rho))
    return StateInput(
        label="adapt_vqe", category=IMPLEMENTABLE,
        amplitudes=amplitudes, basis=basis, preparations=1,
        post_selection=post,
        metadata={"state_kind": "adapt_vqe_ansatz",
                  "adapt_sector_leakage": leaked,
                  "adapt_operators": len(result.labels),
                  "adapt_labels": list(result.labels),
                  "adapt_energy": float(result.energy),
                  "adapt_construction_cost_accounted": False,
                  "adapt_construction_evidence": "exact_simulation",
                  "adapt_construction_circuits": int(result.total_circuits),
                  "adapt_construction_shots": int(result.total_shots)})


def sample_state_input(state: StateInput, *, shots: int, seed: int | None = 0,
                       **kwargs) -> tuple[np.ndarray, SamplingRecord]:
    """:func:`sample_configurations` with the input's label, mode, and category.

    Returned indices always address the operator the state was built against:
    sector positions where the state has a sector, computational-basis words
    where it does not.  A post-selecting input samples full-space and the
    surviving words are mapped back to sector positions here, so the caller
    never has to know which of the three paths ran.
    """
    if state.post_selection is None:
        indices, record = sample_configurations(
            state.amplitudes, shots=shots, seed=seed, basis=state.basis,
            input_label=state.label, **kwargs)
    else:
        words, record = sample_configurations(
            state.amplitudes, shots=shots, seed=seed, basis=None,
            input_label=state.label, **state.post_selection, **kwargs)
        # Post-selection guarantees membership, so a miss here means the sector
        # spec and the basis disagree -- worth an assertion, not a silent drop.
        # A word above every basis element lands at `basis.size`, so the slots
        # are clipped before indexing: without that the mismatch surfaces as an
        # out-of-bounds IndexError instead of the message written for it.
        slots = np.searchsorted(state.basis, words)
        indices = np.where(slots < state.basis.size, slots, 0)
        if indices.size and not np.array_equal(state.basis[indices], words):
            raise ValueError("post-selected configurations are not in the "
                             "sector basis they were selected against")
    # One destructive computational-basis measurement consumes one preparation
    # execution. ``state.preparations`` is the count of *distinct* preparation
    # circuits, so multiplying by it would mix two different resource axes.
    executions = None if state.category == ORACLE else int(record.raw_shots)
    return indices, replace(
        record, input_category=state.category,
        state_preparations=state.preparations,
        state_preparation_executions=executions)


def sample_state_inputs(states, *, shots, seed: int | None = 0,
                        stability_bootstrap: int | None = 128
                        ) -> tuple[np.ndarray, SamplingRecord]:
    """Sample several inputs and pool the accepted draws into one subspace.

    The multi-time protocol of time-evolved QSCI (Phase 16A): prepare each
    declared state for its share of the shots, keep what post-selection
    accepts, and diagonalize on the union.  ``shots`` is one count for every
    state or one count per state.  All states must share one evidence
    category and address one operator: the same sector basis (restricted and
    post-selected states may mix, since both return its positions) or none,
    because a union of indices means nothing across two index spaces.

    Diagnostics are computed on the pooled accepted draws, so duplicates and
    Good-Turing singletons count across states.  ``retained_probability`` is
    the union's weight under the shot-weighted mixture of the states.
    ``state_preparations`` sums the distinct circuits, and every shot is one
    execution.  Occupancy recovery is not offered here.
    """
    states = list(states)
    if not states:
        raise ValueError("need at least one state to sample")
    for state in states:
        if not isinstance(state, StateInput):
            raise TypeError("every pooled input must be a StateInput")
    category = assert_single_evidence_category(states)
    first = states[0]
    specs = [state.post_selection for state in states
             if state.post_selection is not None]
    for state in states[1:]:
        same_basis = ((state.basis is None and first.basis is None)
                      or (state.basis is not None and first.basis is not None
                          and np.array_equal(state.basis, first.basis)))
        # A restricted sector state and a post-selected one both return sector
        # positions of the same basis, so they pool; two different
        # post-selection specs would not.
        if not same_basis or any(spec != specs[0] for spec in specs):
            raise ValueError("pooled inputs must address one operator: the "
                             "same basis and one post-selection spec")
    counts = ([shots] * len(states) if isinstance(shots, (int, np.integer))
              else list(shots))
    if len(counts) != len(states):
        raise ValueError(f"{len(counts)} shot counts for {len(states)} states")
    if any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) or c <= 0
           for c in counts):
        raise ValueError("every shot count must be a positive integer")

    rng = np.random.default_rng(seed)
    pooled, distributions = [], []
    raw = discarded = 0
    for state, count in zip(states, counts):
        probabilities = _probabilities(state.amplitudes)
        if (state.basis is not None and state.post_selection is None
                and probabilities.size != state.basis.size):
            raise ValueError(f"input {state.label!r}: basis and state have "
                             "different lengths")
        distributions.append(probabilities)
        draws = rng.choice(probabilities.size, size=int(count), p=probabilities)
        raw += int(count)
        if state.post_selection is None:
            pooled.append(draws.astype(np.int64))
            continue
        spec = state.post_selection
        words = draws.astype(np.int64)
        inside = _sector_membership(words, spec["n"], spec["n_electrons"],
                                    spec["sz"], spec["spin_ordering"])
        discarded += int((~inside).sum())
        slots = np.searchsorted(state.basis, words[inside])
        safe = np.where(slots < state.basis.size, slots, 0)
        if not np.array_equal(state.basis[safe], words[inside]):
            raise ValueError("post-selected configurations are not in the "
                             "sector basis they were selected against")
        pooled.append(safe)
    accepted = np.concatenate(pooled)
    if accepted.size == 0:
        raise ValueError("every sampled configuration fell outside the requested "
                         "sector; the states and the sector disagree")
    unique, multiplicity = np.unique(accepted, return_counts=True)
    union_mass = []
    for state, probabilities in zip(states, distributions):
        keys = unique if state.post_selection is None else state.basis[unique]
        union_mass.append(float(probabilities[keys].sum()))
    retained = float(np.dot(counts, union_mass) / sum(counts))
    stability = None
    if stability_bootstrap not in (None, 0):
        stability = sampling_set_stability(
            accepted, replicates=stability_bootstrap,
            seed=None if seed is None else int(seed) ^ 0x5EED)
    implementable = category == IMPLEMENTABLE
    return unique, SamplingRecord(
        raw_shots=raw, accepted_shots=int(accepted.size),
        discarded_shots=discarded, repaired_shots=0,
        unique_configurations=int(unique.size), retained_probability=retained,
        input_label="pooled[" + ",".join(state.label for state in states) + "]",
        seed=seed, recovery="none", input_category=category,
        state_preparations=(sum(int(state.preparations) for state in states)
                            if implementable else None),
        state_preparation_executions=raw if implementable else None,
        singleton_configurations=int(np.count_nonzero(multiplicity == 1)),
        bootstrap_set_stability=stability)


def _probabilities(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    weights = np.abs(psi) ** 2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("cannot sample from a zero state")
    return weights / total


def _sector_membership(words: np.ndarray, n: int, n_electrons: int,
                       sz: float | None, spin_ordering) -> np.ndarray:
    """Boolean mask of words already in the target ``(N, S_z)`` sector."""
    counts = np.zeros(words.size, dtype=np.int64)
    for j in range(n):
        counts += (words >> (n - 1 - j)) & 1
    keep = counts == n_electrons
    if sz is None:
        return keep
    up_sites, _ = _spin_sites(n, spin_ordering)
    up_counts = np.zeros(words.size, dtype=np.int64)
    for j in up_sites:
        up_counts += (words >> (n - 1 - j)) & 1
    # S_z = (n_up - n_down)/2 with n_down = N - n_up, so n_up = N/2 + S_z.
    raw_two_sz = 2.0 * float(sz)
    two_sz = round(raw_two_sz)
    if not np.isclose(raw_two_sz, two_sz, atol=1e-12, rtol=0.0):
        raise ValueError("S_z must be an integer or half-integer")
    if (n_electrons + two_sz) % 2:
        return np.zeros(words.size, dtype=bool)
    return keep & (up_counts == (n_electrons + two_sz) // 2)


def _empirical_occupancy(words: np.ndarray, n: int) -> np.ndarray:
    """Mean orbital occupations estimated from the actual sampling record."""
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    if words.size == 0:
        raise ValueError("cannot estimate occupancy from zero sampled words")
    return np.array([
        float(np.mean((words >> (n - 1 - j)) & 1)) for j in range(n)
    ], dtype=float)


def recover_configurations(words: np.ndarray, occupancy: np.ndarray, *, n: int,
                           n_electrons: int, sz: float | None,
                           spin_ordering="interleaved") -> np.ndarray:
    """Occupancy-guided repair of out-of-sector samples (§8C).

    A declared heuristic in the spirit of SQD configuration recovery, and
    labelled as one: where a sample carries too many electrons of a spin, the
    occupied orbitals with the *lowest* mean occupancy are emptied; where it
    carries too few, the empty orbitals with the *highest* mean occupancy are
    filled.  Ties break on orbital index, so the map is deterministic given
    ``occupancy``.

    The result is guaranteed to sit in the target sector -- that is what makes
    it a repair rather than a second sampling stage -- but a repaired sample is
    no longer a draw from the prepared state's distribution.  Callers must
    report :attr:`SamplingRecord.repaired_fraction` alongside any energy that
    used them.
    """
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    occupancy = np.asarray(occupancy, dtype=float).reshape(-1)
    if occupancy.size != n:
        raise ValueError(f"occupancy must have {n} entries, got {occupancy.size}")
    if sz is None:
        groups = [(list(range(n)), n_electrons)]
    else:
        up_sites, down_sites = _spin_sites(n, spin_ordering)
        raw_two_sz = 2.0 * float(sz)
        two_sz = round(raw_two_sz)
        if not np.isclose(raw_two_sz, two_sz, atol=1e-12, rtol=0.0):
            raise ValueError("S_z must be an integer or half-integer")
        if (n_electrons + two_sz) % 2:
            raise ValueError("particle number and S_z do not define integer counts")
        n_up = (n_electrons + two_sz) // 2
        groups = [(up_sites, n_up), (down_sites, n_electrons - n_up)]
    for sites, target in groups:
        if not 0 <= target <= len(sites):
            raise ValueError("requested sector is empty for this spin ordering")

    repaired = np.empty_like(words)
    for position, word in enumerate(words.tolist()):
        for sites, target in groups:
            occupied = [j for j in sites if (word >> (n - 1 - j)) & 1]
            empty = [j for j in sites if not (word >> (n - 1 - j)) & 1]
            surplus = len(occupied) - target
            if surplus > 0:
                for j in sorted(occupied, key=lambda s: (occupancy[s], s))[:surplus]:
                    word &= ~(1 << (n - 1 - j))
            elif surplus < 0:
                for j in sorted(empty, key=lambda s: (-occupancy[s], s))[:-surplus]:
                    word |= 1 << (n - 1 - j)
        repaired[position] = word
    return repaired


def sample_configurations(psi, *, shots: int, seed: int | None = 0,
                          basis: np.ndarray | None = None,
                          n: int | None = None,
                          n_electrons: int | None = None,
                          sz: float | None = None,
                          spin_ordering="interleaved",
                          recovery: str = "none",
                          input_label: str = "unspecified",
                          stability_bootstrap: int | None = 128,
                          ) -> tuple[np.ndarray, SamplingRecord]:
    """Draw configurations from a state's exact probabilities (§8A).

    Two modes, selected by ``basis``:

    *Sector mode* (``basis`` given, the sector's occupation words): ``psi`` holds
    sector amplitudes and every draw is in-sector by construction, so nothing is
    discarded.  Returned indices index ``basis``.

    *Full-space mode* (``basis`` is ``None``): ``psi`` holds ``2^n`` amplitudes
    and returned indices are computational-basis words.  Supplying
    ``n_electrons`` requests fermionic post-selection; omitting it is the raw
    spin-model arm, which keeps every draw.

    Sampling exact probabilities is the validation layer, not a hardware claim.
    The evidence label a caller attaches must say so.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    if recovery not in ("none", "occupancy"):
        raise ValueError("recovery must be 'none' or 'occupancy'")
    if stability_bootstrap is not None and stability_bootstrap < 0:
        raise ValueError("stability_bootstrap must be nonnegative or None")
    probabilities = _probabilities(psi)
    rng = np.random.default_rng(seed)
    draws = rng.choice(probabilities.size, size=int(shots), p=probabilities)
    stability_seed = None if seed is None else int(seed) ^ 0x5EED

    def diagnostics(accepted):
        _, counts = np.unique(accepted, return_counts=True)
        stability = None
        if stability_bootstrap not in (None, 0):
            stability = sampling_set_stability(
                accepted, replicates=stability_bootstrap, seed=stability_seed)
        return int(np.count_nonzero(counts == 1)), stability

    if basis is not None:
        basis = np.asarray(basis, dtype=np.int64).reshape(-1)
        if basis.size != probabilities.size:
            raise ValueError("basis and state have different lengths")
        if recovery != "none":
            raise ValueError("sector-mode sampling has nothing to recover: "
                             "every draw is already in the sector")
        unique = np.unique(draws)
        singletons, stability = diagnostics(draws)
        return unique, SamplingRecord(
            raw_shots=int(shots), accepted_shots=int(shots), discarded_shots=0,
            repaired_shots=0, unique_configurations=int(unique.size),
            retained_probability=float(probabilities[unique].sum()),
            input_label=input_label, seed=seed,
            singleton_configurations=singletons,
            bootstrap_set_stability=stability)

    if n is None:
        n = int(round(np.log2(probabilities.size)))
    # Checked whether `n` was inferred or supplied. A caller-supplied `n` that
    # disagrees with the state sets the bit width used by post-selection and
    # recovery, so a wrong value does not fail -- it silently keeps the wrong
    # configurations, which is the one failure mode this contract cannot have.
    if 2 ** n != probabilities.size:
        raise ValueError(f"full-space sampling needs a 2^n-length state: "
                         f"n={n} implies {2 ** n} amplitudes, got "
                         f"{probabilities.size}")
    words = draws.astype(np.int64)
    if n_electrons is None:
        # Spin arm: no particle-number sector exists, so nothing is out of it.
        unique = np.unique(words)
        singletons, stability = diagnostics(words)
        return unique, SamplingRecord(
            raw_shots=int(shots), accepted_shots=int(shots), discarded_shots=0,
            repaired_shots=0, unique_configurations=int(unique.size),
            retained_probability=float(probabilities[unique].sum()),
            input_label=input_label, seed=seed, recovery="none",
            singleton_configurations=singletons,
            bootstrap_set_stability=stability)

    inside = _sector_membership(words, n, n_electrons, sz, spin_ordering)
    repaired_shots = 0
    pre_repair_singletons = None
    if recovery == "occupancy" and not inside.all():
        # Every raw draw in this branch will be accepted after repair.  Count
        # Good-Turing singletons before the many-to-one repair map can merge
        # distinct sampled configurations; bootstrap stability still describes
        # the repaired set that is actually diagonalized.
        _, raw_counts = np.unique(words, return_counts=True)
        pre_repair_singletons = int(np.count_nonzero(raw_counts == 1))
        # Recovery is a sampling heuristic, so its occupancy guide must come
        # from the samples too.  Reading it from the exact 2^n probability
        # vector would leak oracle information into a method whose point is to
        # work from computational-basis draws.
        occupancy = _empirical_occupancy(words, n)
        words = words.copy()
        words[~inside] = recover_configurations(
            words[~inside], occupancy, n=n, n_electrons=n_electrons, sz=sz,
            spin_ordering=spin_ordering)
        repaired_shots = int((~inside).sum())
        accepted = words
        discarded_shots = 0
    else:
        accepted = words[inside]
        discarded_shots = int((~inside).sum())
    if accepted.size == 0:
        raise ValueError("every sampled configuration fell outside the requested "
                         "sector; the state and the sector disagree")
    unique = np.unique(accepted)
    singletons, stability = diagnostics(accepted)
    if pre_repair_singletons is not None:
        singletons = pre_repair_singletons
    return unique, SamplingRecord(
        raw_shots=int(shots), accepted_shots=int(accepted.size),
        discarded_shots=discarded_shots, repaired_shots=repaired_shots,
        unique_configurations=int(unique.size),
        retained_probability=float(probabilities[unique].sum()),
        input_label=input_label, seed=seed, recovery=recovery,
        singleton_configurations=singletons,
        bootstrap_set_stability=stability)


def run_qsci(operator, indices, *, sampling: SamplingRecord,
             exact_energy: float | None = None, k: int = 1,
             want_eigenvectors: bool = False,
             evidence: str = EvidenceLevel.FINITE_SAMPLE.value,
             hermiticity_tol: float = 1e-10,
             metadata: dict | None = None) -> QSCIResult:
    """Diagonalize ``H`` restricted to sampled configurations (§8B).

    ``operator`` is a :class:`SectorOperator` (indices index its sector basis)
    or a :class:`PauliLinearOperator` (indices are computational-basis words).
    Hermiticity is *checked* against ``hermiticity_tol`` and the residual is
    reported rather than silently symmetrized away -- invariant 1 of §8B is a
    test, not a repair.  Ritz vectors are opt-in so the default QSCI benchmark
    keeps its historical eigensolve time and peak-memory measurement boundary;
    Phase 11 target construction requests them explicitly.
    """
    if not isinstance(operator, (SectorOperator, PauliLinearOperator)):
        raise TypeError("operator must be a SectorOperator or PauliLinearOperator")
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("cannot solve an empty sampled subspace")
    if not isinstance(k, int) or not 1 <= k <= indices.size:
        raise ValueError(f"k must be in [1, {indices.size}]")

    with _PeakRSS() as rss:
        start = time.perf_counter()
        matrix = operator.restrict(indices)
        build_seconds = time.perf_counter() - start

        residual = (float(np.abs(matrix - matrix.conj().T).max())
                    if matrix.size else 0.0)
        if residual > hermiticity_tol * max(1.0, float(np.abs(matrix).max())):
            raise ValueError(
                f"restricted Hamiltonian is not Hermitian (residual {residual:.3e}); "
                "the operator or the index set is wrong, and symmetrizing would hide it")

        start = time.perf_counter()
        hermitian = 0.5 * (matrix + matrix.conj().T)
        if want_eigenvectors:
            values, vectors = np.linalg.eigh(hermitian)
        else:
            values = np.linalg.eigvalsh(hermitian)
            vectors = None
        solve_seconds = time.perf_counter() - start

    return QSCIResult(
        energy=float(values[0]),
        eigenvalues=values[:k],
        indices=indices,
        sampling=sampling,
        evidence=evidence,
        hermiticity_residual=residual,
        matrix_nonzeros=int(np.count_nonzero(matrix)),
        matrix_bytes=int(matrix.nbytes),
        peak_rss_bytes=rss.peak,
        peak_rss_delta_bytes=rss.delta,
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
        exact_energy=exact_energy,
        metadata=dict(metadata or {}),
        eigenvectors=(None if vectors is None else vectors[:, :k].copy()),
    )
