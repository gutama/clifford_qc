r"""QSCI/SQD: sampled determinant subspaces as a first-class method.

Phase 8 of ``LITERATURE_ROADMAP.md``.  A sampled-subspace eigensolver draws
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

import time
from dataclasses import dataclass, field

import numpy as np

from ..backends.sector_statevector import SectorOperator, _spin_sites
from ..pauli_action import PauliLinearOperator
from ..selection import EvidenceLevel

__all__ = [
    "SamplingRecord",
    "QSCIResult",
    "sample_configurations",
    "recover_configurations",
    "run_qsci",
]


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
            "input_label": self.input_label,
            "recovery": self.recovery,
            "seed": self.seed,
        }


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
    build_seconds: float
    solve_seconds: float
    exact_energy: float | None = None
    metadata: dict = field(default_factory=dict)

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
    two_sz = round(2.0 * float(sz))
    if (n_electrons + two_sz) % 2:
        return np.zeros(words.size, dtype=bool)
    return keep & (up_counts == (n_electrons + two_sz) // 2)


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
        two_sz = round(2.0 * float(sz))
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
    probabilities = _probabilities(psi)
    rng = np.random.default_rng(seed)
    draws = rng.choice(probabilities.size, size=int(shots), p=probabilities)

    if basis is not None:
        basis = np.asarray(basis, dtype=np.int64).reshape(-1)
        if basis.size != probabilities.size:
            raise ValueError("basis and state have different lengths")
        if recovery != "none":
            raise ValueError("sector-mode sampling has nothing to recover: "
                             "every draw is already in the sector")
        unique = np.unique(draws)
        return unique, SamplingRecord(
            raw_shots=int(shots), accepted_shots=int(shots), discarded_shots=0,
            repaired_shots=0, unique_configurations=int(unique.size),
            retained_probability=float(probabilities[unique].sum()),
            input_label=input_label, seed=seed)

    if n is None:
        n = int(round(np.log2(probabilities.size)))
        if 2 ** n != probabilities.size:
            raise ValueError("full-space sampling needs a 2^n-length state")
    words = draws.astype(np.int64)
    if n_electrons is None:
        # Spin arm: no particle-number sector exists, so nothing is out of it.
        unique = np.unique(words)
        return unique, SamplingRecord(
            raw_shots=int(shots), accepted_shots=int(shots), discarded_shots=0,
            repaired_shots=0, unique_configurations=int(unique.size),
            retained_probability=float(probabilities[unique].sum()),
            input_label=input_label, seed=seed, recovery="none")

    inside = _sector_membership(words, n, n_electrons, sz, spin_ordering)
    repaired_shots = 0
    if recovery == "occupancy" and not inside.all():
        all_words = np.arange(probabilities.size, dtype=np.int64)
        occupancy = np.array(
            [float(probabilities[((all_words >> (n - 1 - j)) & 1) == 1].sum())
             for j in range(n)], dtype=float)
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
    return unique, SamplingRecord(
        raw_shots=int(shots), accepted_shots=int(accepted.size),
        discarded_shots=discarded_shots, repaired_shots=repaired_shots,
        unique_configurations=int(unique.size),
        retained_probability=float(probabilities[unique].sum()),
        input_label=input_label, seed=seed, recovery=recovery)


def run_qsci(operator, indices, *, sampling: SamplingRecord,
             exact_energy: float | None = None, k: int = 1,
             evidence: str = EvidenceLevel.EXACT.value,
             hermiticity_tol: float = 1e-10,
             metadata: dict | None = None) -> QSCIResult:
    """Diagonalize ``H`` restricted to sampled configurations (§8B).

    ``operator`` is a :class:`SectorOperator` (indices index its sector basis)
    or a :class:`PauliLinearOperator` (indices are computational-basis words).
    Hermiticity is *checked* against ``hermiticity_tol`` and the residual is
    reported rather than silently symmetrized away -- invariant 1 of §8B is a
    test, not a repair.
    """
    if not isinstance(operator, (SectorOperator, PauliLinearOperator)):
        raise TypeError("operator must be a SectorOperator or PauliLinearOperator")
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("cannot solve an empty sampled subspace")
    if not isinstance(k, int) or not 1 <= k <= indices.size:
        raise ValueError(f"k must be in [1, {indices.size}]")

    start = time.perf_counter()
    matrix = operator.restrict(indices)
    build_seconds = time.perf_counter() - start

    residual = float(np.abs(matrix - matrix.conj().T).max()) if matrix.size else 0.0
    if residual > hermiticity_tol * max(1.0, float(np.abs(matrix).max())):
        raise ValueError(
            f"restricted Hamiltonian is not Hermitian (residual {residual:.3e}); "
            "the operator or the index set is wrong, and symmetrizing would hide it")

    start = time.perf_counter()
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
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
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
        exact_energy=exact_energy,
        metadata=dict(metadata or {}),
    )
