r"""One fragment-solver callback for an embedding loop outside the package.

Phase 18 of ``PLAN.md`` keeps DMET and projection-based embedding *outside*
``clifford_qc``.  What crosses that boundary is deliberately small.  The outer
loop hands a fragment Hamiltonian in as a :class:`~clifford_qc.models.spin.Model`
-- from :func:`~clifford_qc.models.effective.effective_hamiltonian`, a restricted
FCIDUMP, or a lattice builder -- and receives an energy plus one- and
two-particle density matrices.  QSCI, selected CI and the QSCI x A-CASE hybrid
each implement that one call, so an embedding driver can swap solvers without
knowing which one it holds.  :class:`ExactFragmentSolver` implements it too, as
the sector-FCI reference every other solver is checked against.

Conventions
-----------
Spin-orbital index ``p`` is the model's qubit index under the Jordan--Wigner
convention of :mod:`clifford_qc.fermion`: ``a_p = Z_0 ... Z_{p-1} (X_p + iY_p)/2``,
occupied is ``|1>``, and an occupation word carries qubit ``p`` at bit
``n-1-p`` (the convention of ``pauli_action.word_masks`` and the sector
backend).  Then

.. math::

    \gamma_{pq} = \langle a^\dagger_p a_q \rangle, \qquad
    \Gamma_{pqrs} = \langle a^\dagger_p a^\dagger_q a_r a_s \rangle .

Both are Gram matrices of hole vectors, which is how they are computed:
``gamma[p, q] = <a_p Psi | a_q Psi>`` and
``Gamma[p, q, r, s] = <a_q a_p Psi | a_r a_s Psi>``.  One matrix product each,
and the same code serves a sector eigenvector, a sampled-subspace Ritz vector
and a reconstructed A-CASE Ritz state, because it only ever reads the words
that carry amplitude.

:func:`spatial_rdms` spin-sums these into the chemist-ordered spatial pair a
restricted embedding code consumes,

.. math::

    D_{pq} = \sum_\sigma \langle a^\dagger_{p\sigma} a_{q\sigma} \rangle, \qquad
    D_{pqrs} = \sum_{\sigma\tau}
        \langle a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma} \rangle,

which is exactly the operator order :func:`~clifford_qc.models.fcidump.model_from_fcidump`
assembles, so :func:`energy_from_spatial_rdms` returns
``E_core + sum h_pq D_pq + 1/2 sum (pq|rs) D_pqrs`` with no reindexing.

What is and is not claimed
--------------------------
Every solution carries ``state_energy``: ``<Psi|H|Psi>`` recomputed on the very
state its RDMs were read from.  Agreement with the solver's own ``energy`` is
what ties the matrices to the calculation that produced the number; it is a
consistency check, not an accuracy statement.  Density matrices are computed
from a state vector, so this boundary is a small-fragment interface: the
hybrid's Ritz state is materialized, which the A-CASE measurement route never
does, and ``rdm_route`` says which state each solution used.  A degenerate
ground state has no unique RDMs; the exact solver reports its gap so a caller
can tell.  Solutions keep the evidence category of their input -- an oracle
sampling state stays ``oracle`` -- so an embedding comparison cannot put an
oracle-fed fragment beside an implementable one without saying so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from ..backends import ExactMVBackend
from ..backends.sector_statevector import SectorStatevectorBackend
from ..models.metadata import FERMIONIC_KINDS, SPIN_ORDERINGS, spin_convention
from ..pauli_action import PauliLinearOperator, parity
from .hybrid import run_hybrid
from .qsci import (
    StateInput, exact_ground_state_oracle, reference_determinant_state,
    run_qsci, sample_state_input,
)
from .reference import pure_statevector
from .selected_ci import run_control

__all__ = [
    "CLASSICAL",
    "FRAGMENT_SOLUTION_SCHEMA",
    "ExactFragmentSolver",
    "FragmentSolution",
    "FragmentSolver",
    "HybridFragmentSolver",
    "QSCIFragmentSolver",
    "SelectedCIFragmentSolver",
    "energy_from_spatial_rdms",
    "spatial_rdms",
    "spin_orbital_rdms",
]

FRAGMENT_SOLUTION_SCHEMA = "clifford_qc.fragment_solution.v1"

#: Input category of a solution that consumed no quantum sampling state.
#: ``oracle`` and ``implementable`` are the QSCI vocabulary (§8D) and are
#: carried through unchanged from the sampling record.
CLASSICAL = "classical"
_CATEGORIES = ("oracle", "implementable", CLASSICAL)

#: Which state a solution's density matrices were read from.
_ROUTES = ("sector_eigenvector", "sampled_subspace_ritz_vector",
           "selected_subspace_ritz_vector", "reconstructed_ritz_state")

# Reconstructing an A-CASE Ritz state from its virtual basis must return a unit
# vector; a larger defect means the reference the bank paired against is not
# the model's reference, and the matrices would describe another state.
_RECONSTRUCTION_NORM_TOL = 1e-8


# ------------------------------------------------------------------ RDM kernel

def _below_mask(n: int, mode: int) -> int:
    """Bits of the qubits ``k < mode``: the Jordan--Wigner string of ``a_mode``."""
    return (1 << n) - (1 << (n - mode))


def _annihilate(words: np.ndarray, coefficients: np.ndarray, mode: int,
                n: int) -> tuple[np.ndarray, np.ndarray]:
    """``a_mode`` on a sparse state: the words it survives on, signed."""
    bit = 1 << (n - 1 - mode)
    occupied = (words & bit) != 0
    kept = words[occupied]
    sign = 1 - 2 * parity(kept, _below_mask(n, mode))
    return kept ^ bit, coefficients[occupied] * sign


def _gram(columns: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """``G[a, b] = <v_a|v_b>`` for sparse vectors given as ``(words, values)``.

    Annihilation is injective on distinct words, so each column has distinct
    words and can be scattered into one dense matrix over their union.
    """
    count = len(columns)
    populated = [words for words, _ in columns if words.size]
    if not populated:
        return np.zeros((count, count), dtype=complex)
    union = np.unique(np.concatenate(populated))
    matrix = np.zeros((union.size, count), dtype=complex)
    for column, (words, values) in enumerate(columns):
        matrix[np.searchsorted(union, words), column] = values
    return matrix.conj().T @ matrix


def spin_orbital_rdms(words, amplitudes, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(gamma, Gamma)`` of the state ``sum_b amplitudes[b] |words[b]>``.

    ``gamma[p, q] = <a†_p a_q>`` and ``Gamma[p, q, r, s] = <a†_p a†_q a_r a_s>``
    in the model's spin-orbital (qubit) order; see the module docstring for the
    sign and bit conventions.  The state is normalized here, so a Ritz vector
    need not be.  Words must be distinct: a repeated word is two amplitudes for
    one configuration, and summing them silently would hide the caller's bug.
    """
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 62:
        raise ValueError("n must be an integer in [1, 62]")
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    psi = np.asarray(amplitudes, dtype=complex).reshape(-1)
    if words.shape != psi.shape:
        raise ValueError(f"{words.size} words but {psi.size} amplitudes")
    if words.size and (words.min() < 0 or words.max() >= (1 << n)):
        raise ValueError(f"occupation words must lie in [0, 2^{n})")
    if np.unique(words).size != words.size:
        raise ValueError("occupation words must be distinct")
    if not np.all(np.isfinite(psi)):
        raise ValueError("amplitudes must be finite")
    norm = float(np.vdot(psi, psi).real)
    if norm <= 0.0:
        raise ValueError("cannot take density matrices of a zero state")
    live = psi != 0
    words, psi = words[live], psi[live] / math.sqrt(norm)

    holes = [_annihilate(words, psi, mode, n) for mode in range(n)]
    gamma = _gram(holes)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # phi_(i<j) = a_i a_j |Psi>: annihilate j first, then i.
    two_holes = [_annihilate(*holes[j], i, n) for i, j in pairs]
    gram = _gram(two_holes)
    index = np.zeros((n, n), dtype=np.int64)
    sign = np.zeros((n, n))
    for k, (i, j) in enumerate(pairs):
        index[i, j] = index[j, i] = k
        sign[i, j], sign[j, i] = 1.0, -1.0
    # Gamma[p, q, r, s] = <phi_(q,p)|phi_(r,s)>, with phi_(q,p) = -phi_(p,q).
    left_index, left_sign = index.T, sign.T
    two = (left_sign[:, :, None, None] * sign[None, None, :, :]
           * gram[left_index[:, :, None, None], index[None, None, :, :]])
    return gamma, two


def _spin_orbital_map(spin_convention: str, n_spatial: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Qubit indices of ``(p, up)`` and ``(p, down)`` for each spatial ``p``."""
    spatial = np.arange(n_spatial)
    if spin_convention == "interleaved":
        return 2 * spatial, 2 * spatial + 1
    if spin_convention == "blocked":
        return spatial, spatial + n_spatial
    raise ValueError(f"spin_convention must be one of {SPIN_ORDERINGS}, "
                     f"got {spin_convention!r}")


def spatial_rdms(rdm1, rdm2, *, spin_convention: str
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Spin-summed, chemist-ordered spatial ``(D1, D2)`` from spin-orbital RDMs.

    ``D1[p, q] = sum_s <a†_ps a_qs>`` and
    ``D2[p, q, r, s] = sum_(s,t) <a†_ps a†_rt a_st a_qs>`` -- the order in which
    ``models.fcidump`` assembles ``(pq|rs)``, so :func:`energy_from_spatial_rdms`
    contracts them with FCIDUMP integrals directly.  ``spin_convention`` has no
    default for the reason ``models.metadata`` gives: the same index is a
    different spin orbital under the other ordering.
    """
    rdm1 = np.asarray(rdm1)
    rdm2 = np.asarray(rdm2)
    n = rdm1.shape[0]
    if rdm1.shape != (n, n) or rdm2.shape != (n, n, n, n):
        raise ValueError("expected rdm1 of shape (n, n) and rdm2 of shape (n, n, n, n)")
    if n % 2:
        raise ValueError("spin-orbital RDMs need an even number of spin orbitals")
    up, down = _spin_orbital_map(spin_convention, n // 2)
    dm1 = rdm1[np.ix_(up, up)] + rdm1[np.ix_(down, down)]
    dm2 = np.zeros((n // 2,) * 4, dtype=np.result_type(rdm2, float))
    for first in (up, down):
        for second in (up, down):
            # block[p, r, s, q] = Gamma[p_first, r_second, s_second, q_first]
            block = rdm2[np.ix_(first, second, second, first)]
            dm2 = dm2 + block.transpose(0, 3, 1, 2)
    return dm1, dm2


def energy_from_spatial_rdms(one_body, two_body, dm1, dm2, *,
                             core_energy: float = 0.0) -> float:
    """``E_core + sum h_pq D1_pq + 1/2 sum (pq|rs) D2_pqrs`` (chemist integrals)."""
    one_body = np.asarray(one_body)
    two_body = np.asarray(two_body)
    value = (float(core_energy) + np.einsum("pq,pq->", one_body, dm1)
             + 0.5 * np.einsum("pqrs,pqrs->", two_body, dm2))
    return float(np.real(value))


# ------------------------------------------------------------------ solution

def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=complex, copy=True)
    out.setflags(write=False)
    return out


def _encode(array: np.ndarray) -> dict:
    return {"real": np.real(array).tolist(), "imag": np.imag(array).tolist()}


@dataclass(frozen=True)
class FragmentSolution:
    """What a fragment solver returns across the embedding boundary.

    ``energy`` is the solver's own variational energy; ``state_energy`` is
    ``<Psi|H|Psi>`` on the normalized state ``rdm1`` and ``rdm2`` were read
    from.  Their difference is reported by :meth:`diagnostics` and is what ties
    the density matrices to the number.  The arrays are read-only copies.
    """

    solver: str
    energy: float
    state_energy: float
    rdm1: np.ndarray
    rdm2: np.ndarray
    n_electrons: int
    sz: float
    spin_convention: str
    n_spatial_orbitals: int
    input_category: str
    rdm_route: str
    metadata: dict = field(default_factory=dict)
    schema: str = FRAGMENT_SOLUTION_SCHEMA

    def __post_init__(self):
        if self.schema != FRAGMENT_SOLUTION_SCHEMA:
            raise ValueError(f"schema must be {FRAGMENT_SOLUTION_SCHEMA!r}")
        rdm1, rdm2 = _frozen(self.rdm1), _frozen(self.rdm2)
        n = 2 * int(self.n_spatial_orbitals)
        if rdm1.shape != (n, n) or rdm2.shape != (n, n, n, n):
            raise ValueError(
                f"{self.n_spatial_orbitals} spatial orbitals need rdm1 of shape "
                f"{(n, n)} and rdm2 of shape {(n,) * 4}, got {rdm1.shape} and "
                f"{rdm2.shape}")
        if not (np.all(np.isfinite(rdm1)) and np.all(np.isfinite(rdm2))):
            raise ValueError("density matrices must be finite")
        if not (math.isfinite(self.energy) and math.isfinite(self.state_energy)):
            raise ValueError("energies must be finite")
        if self.spin_convention not in SPIN_ORDERINGS:
            raise ValueError(f"spin_convention must be one of {SPIN_ORDERINGS}")
        if self.input_category not in _CATEGORIES:
            raise ValueError(f"input_category must be one of {_CATEGORIES}")
        if self.rdm_route not in _ROUTES:
            raise ValueError(f"rdm_route must be one of {_ROUTES}")
        object.__setattr__(self, "rdm1", rdm1)
        object.__setattr__(self, "rdm2", rdm2)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_spin_orbitals(self) -> int:
        return int(self.rdm1.shape[0])

    def spatial_rdms(self) -> tuple[np.ndarray, np.ndarray]:
        """Spin-summed chemist-ordered ``(D1, D2)``; see :func:`spatial_rdms`."""
        return spatial_rdms(self.rdm1, self.rdm2,
                            spin_convention=self.spin_convention)

    def diagnostics(self) -> dict:
        """Exact identities a fixed-``(N, S_z)`` state's RDMs must satisfy.

        Each entry is a residual, so zero is the pass value and the caller
        picks the tolerance.  ``energy_consistency`` is ``|energy -
        state_energy|``; the rest are ``tr gamma = N``,
        ``sum_pq Gamma_pqqp = N(N-1)``, the partial trace
        ``sum_q Gamma_pqqr = (N-1) gamma_pr``, the ``S_z`` read off ``gamma``,
        and hermiticity of both matrices.
        """
        gamma, two = self.rdm1, self.rdm2
        electrons = self.n_electrons
        up, down = _spin_orbital_map(self.spin_convention, self.n_spatial_orbitals)
        diagonal = np.real(np.diag(gamma))
        partial = np.einsum("pqqr->pr", two)
        return {
            "energy_consistency": abs(self.energy - self.state_energy),
            "electron_count": abs(complex(np.trace(gamma)) - electrons),
            "pair_count": abs(complex(np.einsum("pqqp->", two))
                              - electrons * (electrons - 1)),
            "partial_trace": float(np.abs(partial - (electrons - 1) * gamma).max()),
            "sz": abs(0.5 * (diagonal[up].sum() - diagonal[down].sum()) - self.sz),
            "rdm1_hermiticity": float(np.abs(gamma - gamma.conj().T).max()),
            "rdm2_hermiticity": float(
                np.abs(two - two.transpose(3, 2, 1, 0).conj()).max()),
        }

    def to_record(self, *, include_tensors: bool = False) -> dict:
        """JSON-ready summary; the tensors only when asked, as real/imag lists."""
        record = {
            "schema": self.schema,
            "solver": self.solver,
            "energy": float(self.energy),
            "state_energy": float(self.state_energy),
            "n_spin_orbitals": self.n_spin_orbitals,
            "n_spatial_orbitals": int(self.n_spatial_orbitals),
            "n_electrons": int(self.n_electrons),
            "sz": float(self.sz),
            "spin_convention": self.spin_convention,
            "input_category": self.input_category,
            "rdm_route": self.rdm_route,
            "rdm1_convention": "rdm1[p, q] = <a+_p a_q>",
            "rdm2_convention": "rdm2[p, q, r, s] = <a+_p a+_q a_r a_s>",
            "diagnostics": {key: float(value)
                            for key, value in self.diagnostics().items()},
            "metadata": dict(self.metadata),
        }
        if include_tensors:
            record["rdm1"] = _encode(self.rdm1)
            record["rdm2"] = _encode(self.rdm2)
        return record


@runtime_checkable
class FragmentSolver(Protocol):
    """The one call an embedding loop makes: a fragment model in, a solution out."""

    name: str

    def __call__(self, model) -> FragmentSolution: ...


# ------------------------------------------------------------------ solvers

@dataclass(frozen=True)
class _Fragment:
    """A fermionic model's declared sector, read once from its metadata."""

    model: Any
    n_electrons: int
    sz: float
    spin_convention: str
    n_spatial_orbitals: int
    backend: SectorStatevectorBackend

    @classmethod
    def of(cls, model) -> "_Fragment":
        metadata = model.metadata
        if metadata.get("kind") not in FERMIONIC_KINDS:
            raise ValueError(
                f"a fragment solver needs a fermionic model; {model.name!r} is "
                f"kind {metadata.get('kind')!r}, which has no particle-number "
                "sector and no density matrices to return")
        convention = spin_convention(model)
        if convention not in SPIN_ORDERINGS:
            raise ValueError("fragment solvers support the named spin orderings "
                             f"{SPIN_ORDERINGS}, not an explicit label sequence")
        spatial = int(metadata["n_spatial_orbitals"])
        if 2 * spatial != model.n:
            raise ValueError(f"{spatial} spatial orbitals do not fill {model.n} qubits")
        electrons, sz = int(metadata["n_electrons"]), float(metadata["sz"])
        backend = SectorStatevectorBackend(model.n, electrons, sz,
                                           spin_ordering=convention)
        return cls(model, electrons, sz, convention, spatial, backend)

    def operator(self):
        return self.backend.operator(self.model.hamiltonian)

    def solution(self, *, solver: str, energy: float, words, amplitudes,
                 operator, input_category: str, rdm_route: str,
                 metadata: Mapping) -> FragmentSolution:
        """Read RDMs and ``<H>`` off one state given on sector words."""
        words = np.asarray(words, dtype=np.int64).reshape(-1)
        amplitudes = np.asarray(amplitudes, dtype=complex).reshape(-1)
        psi = np.zeros(self.backend.dimension, dtype=complex)
        slots = np.searchsorted(self.backend.basis, words)
        inside = slots < self.backend.basis.size
        if not (np.all(inside)
                and np.array_equal(self.backend.basis[slots], words)):
            raise ValueError("the solver's state has configurations outside the "
                             "declared (N, S_z) sector")
        psi[slots] = amplitudes
        psi = psi / np.linalg.norm(psi)
        state_energy = float(operator.expectation(psi).real)
        rdm1, rdm2 = spin_orbital_rdms(words, amplitudes, self.model.n)
        return FragmentSolution(
            solver=solver, energy=float(energy), state_energy=state_energy,
            rdm1=rdm1, rdm2=rdm2, n_electrons=self.n_electrons, sz=self.sz,
            spin_convention=self.spin_convention,
            n_spatial_orbitals=self.n_spatial_orbitals,
            input_category=input_category, rdm_route=rdm_route,
            metadata={"model": self.model.name,
                      "sector_dimension": self.backend.dimension, **metadata})


StateFactory = Callable[[SectorStatevectorBackend, Any], StateInput]

_NAMED_STATES: dict[str, StateFactory] = {
    "exact_ground_oracle":
        lambda backend, model: exact_ground_state_oracle(backend, model.hamiltonian),
    "reference_determinant": reference_determinant_state,
}


def _state_factory(state) -> StateFactory:
    if isinstance(state, str):
        try:
            return _NAMED_STATES[state]
        except KeyError:
            raise ValueError(f"named sampling states are {sorted(_NAMED_STATES)}, "
                             f"got {state!r}") from None
    if callable(state):
        return state
    raise TypeError("state must be a named sampling state or a callable "
                    "(backend, model) -> StateInput")


def _state_label(state) -> str:
    return state if isinstance(state, str) else getattr(state, "__name__", "callable")


def _sample(fragment: _Fragment, state, shots: int, seed):
    """Sector positions sampled from the declared state, plus the record."""
    made = _state_factory(state)(fragment.backend, fragment.model)
    if not isinstance(made, StateInput):
        raise TypeError("a sampling-state factory must return a StateInput")
    return sample_state_input(made, shots=shots, seed=seed)


def _check_shots(shots) -> int:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer")
    return shots


class ExactFragmentSolver:
    """Sector full CI: the reference implementation of the callback.

    ``input_category`` is ``classical``.  This is an ordinary classical fragment
    solver, the one a DMET loop would use at this size, not an oracle
    *sampling* input.  ``ground_gap`` in the metadata says whether the ground
    state, and so its RDMs, is unique.
    """

    def __init__(self, *, method: str = "dense", name: str = "exact_sector"):
        self.method = method
        self.name = name

    def __call__(self, model) -> FragmentSolution:
        fragment = _Fragment.of(model)
        roots = min(2, fragment.backend.dimension)
        values, vectors, operator = fragment.backend.ground_state(
            model.hamiltonian, k=roots, method=self.method, return_operator=True)
        gap = float(values[1] - values[0]) if roots > 1 else None
        return fragment.solution(
            solver=self.name, energy=float(values[0]),
            words=fragment.backend.basis, amplitudes=vectors[:, 0],
            operator=operator, input_category=CLASSICAL,
            rdm_route="sector_eigenvector",
            metadata={"method": self.method, "ground_gap": gap})


class QSCIFragmentSolver:
    """QSCI on configurations sampled from a declared state (Phase 8).

    ``state`` is ``"exact_ground_oracle"``, ``"reference_determinant"`` or a
    callable ``(backend, model) -> StateInput``; its evidence category travels
    to the solution.  The RDMs are those of the sampled-subspace Ritz vector.
    """

    def __init__(self, *, state, shots: int, seed: int | None = 0,
                 name: str = "qsci"):
        _state_factory(state)
        self.state = state
        self.shots = _check_shots(shots)
        self.seed = seed
        self.name = name

    def __call__(self, model) -> FragmentSolution:
        fragment = _Fragment.of(model)
        operator = fragment.operator()
        indices, record = _sample(fragment, self.state, self.shots, self.seed)
        result = run_qsci(operator, indices, sampling=record,
                          want_eigenvectors=True)
        return fragment.solution(
            solver=self.name, energy=result.energy,
            words=fragment.backend.basis[result.indices],
            amplitudes=result.eigenvectors[:, 0], operator=operator,
            input_category=record.input_category,
            rdm_route="sampled_subspace_ritz_vector",
            metadata={"sampling_state": _state_label(self.state),
                      "solver_record": result.to_record()})


class SelectedCIFragmentSolver:
    """A Phase 9 selected-CI control as a fragment solver.

    With ``state=None`` the seed is the reference determinant and nothing
    quantum is consumed, so the category is ``classical``.  With a sampling
    state and ``shots``, the seed is a QSCI sample and the sample's category
    is kept.  ``kind`` and the remaining options are those of
    :func:`~clifford_qc.subspace.selected_ci.run_control`.
    """

    def __init__(self, *, kind: str = "selected_ci", state=None,
                 shots: int | None = None, seed: int | None = 0,
                 max_determinants: int | None = None,
                 score: str = "epstein_nesbet", name: str | None = None):
        if kind not in ("selected_ci", "budget_matched", "matched_selected_ci",
                        "excitation_closure"):
            raise ValueError("kind must be a determinant-selecting control: "
                             "selected_ci, budget_matched, matched_selected_ci "
                             "or excitation_closure")
        if (state is None) != (shots is None):
            raise ValueError("a sampled seed needs both state and shots; a "
                             "classical seed takes neither")
        if state is not None:
            _state_factory(state)
            _check_shots(shots)
        self.kind = kind
        self.state = state
        self.shots = shots
        self.seed = seed
        self.max_determinants = max_determinants
        self.score = score
        self.name = name or f"selected_ci[{kind}]"

    def __call__(self, model) -> FragmentSolution:
        fragment = _Fragment.of(model)
        operator = fragment.operator()
        if self.state is None:
            reference = fragment.backend.state_from_program(model.reference)
            sampled = np.flatnonzero(reference)
            category, seed_label = CLASSICAL, "reference_determinant"
        else:
            sampled, record = _sample(fragment, self.state, self.shots, self.seed)
            category, seed_label = record.input_category, _state_label(self.state)
        result = run_control(operator, sampled, name=self.name, kind=self.kind,
                             max_determinants=self.max_determinants,
                             score=self.score)
        return fragment.solution(
            solver=self.name, energy=result.energy,
            words=fragment.backend.basis[result.determinants],
            amplitudes=result.coefficients, operator=operator,
            input_category=category, rdm_route="selected_subspace_ritz_vector",
            metadata={"seed": seed_label, "solver_record": result.to_record()})


class HybridFragmentSolver:
    """One QSCI x A-CASE arm (Phase 10) as a fragment solver.

    Configurations are sampled as for :class:`QSCIFragmentSolver`, the three
    arms of :func:`~clifford_qc.subspace.hybrid.run_hybrid` are grown over
    them, and ``arm`` picks the one whose Ritz state is returned.  The state is
    rebuilt as ``sum_i c_i A_i|ref>``.  The measured A-CASE route never forms
    it, which is why this solver is limited to fragments whose full register
    fits a state vector.  ``options`` go to ``run_hybrid`` unchanged.
    """

    def __init__(self, *, state, shots: int, seed: int | None = 0,
                 arm: str = "configurations_plus_dressed", max_size: int = 12,
                 name: str | None = None, **options):
        if arm not in ("bare_configurations", "configurations_plus_dressed",
                       "packets_then_dressed"):
            raise ValueError(f"unknown hybrid arm {arm!r}")
        _state_factory(state)
        self.state = state
        self.shots = _check_shots(shots)
        self.seed = seed
        self.arm = arm
        self.max_size = max_size
        self.options = dict(options)
        self.name = name or f"hybrid[{arm}]"

    def __call__(self, model) -> FragmentSolution:
        fragment = _Fragment.of(model)
        operator = fragment.operator()
        indices, record = _sample(fragment, self.state, self.shots, self.seed)
        words = fragment.backend.basis[np.unique(indices)]
        rho = ExactMVBackend().state(model.reference, ())
        arms = {arm.name: arm for arm in run_hybrid(
            rho, model, words, max_size=self.max_size, **self.options)}
        if self.arm not in arms:
            raise ValueError(f"run_hybrid returned no {self.arm!r} arm for this "
                             f"sample (arms: {sorted(arms)}); an empty packet "
                             "family produces no packet arm")
        arm = arms[self.arm]
        subspace = arm.subspace
        try:
            reference = reference_determinant_state(None, model).amplitudes
        except ValueError:
            reference = pure_statevector(rho)
        state = np.zeros(2 ** model.n, dtype=complex)
        for index, coefficient in zip(subspace.indices, subspace.ritz_vector(0)):
            if coefficient != 0:
                generator = subspace.bank.generator(index)
                state += coefficient * PauliLinearOperator(generator.mv).matvec(
                    reference)
        norm = float(np.vdot(state, state).real)
        if abs(norm - 1.0) > _RECONSTRUCTION_NORM_TOL:
            raise ValueError(
                f"the reconstructed Ritz state has squared norm {norm:.12g}, not "
                "1: the bank's reference density and the model reference disagree")
        restricted = fragment.backend.from_dense(state)
        return fragment.solution(
            solver=self.name, energy=arm.energy, words=fragment.backend.basis,
            amplitudes=restricted, operator=operator,
            input_category=record.input_category,
            rdm_route="reconstructed_ritz_state",
            metadata={"sampling_state": _state_label(self.state),
                      "arm": self.arm,
                      "reconstruction_norm_defect": abs(norm - 1.0),
                      "solver_record": arm.to_record()})
