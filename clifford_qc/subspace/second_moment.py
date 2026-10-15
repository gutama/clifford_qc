"""Phase 15: the second-moment bank, ``K_ij = <psi|A_i' H^2 A_j|psi>``.

The projected pair ``(S, H)`` cannot tell how far a Ritz state is from an
eigenstate: a solved Ritz pair has zero *projected* residual by construction.
The true residual needs the second moment. For ``|Psi> = sum_j c_j A_j|psi>``
and any energy ``E``,

    ||(H - E)|Psi>||^2 / <Psi|Psi> = (c'Kc - 2E Re c'Hc + E^2 c'Sc) / c'Sc,

and at the Rayleigh quotient that is the energy variance
``<H^2> - <H>^2`` of the Ritz state. PLAN.md section 4.4 keeps the name
"residual norm" for exactly this quantity, and nothing else in the package
offers it outside the dense validation module.

The rows are the bank's own projected-observable rows for ``Q = H * H``:
``A_i.dagger() * (H2 * A_j)``, paired with the reference exactly as ``S`` and
``H`` are. That is the row the Phase 15 preflight priced
(``benchmarks/configs/phase15_h2_preflight.json``), whose verdict, FULL,
licenses building them on the five frozen mapping-axis banks. The word
universe and coefficient count reported here are the preflight's ``U_K`` and
``T_K``. Construction first runs a storage preflight over the requested block.
Every block needs a conservative coefficient bound within the declared storage
budget; all five frozen bases fit that bound. This is not an RSS limit or a
finite-shot affordability assessment.

The values are exact pairings. A finite-shot estimate of ``K`` would share the
measurement machinery of :mod:`.measured_response`, but grouping a universe of
this size is the cost the preflight excluded, and no estimator is offered here.

**Precision.** The variance is a difference of two numbers of order
``<H^2>``, which for a molecule carries the square of its constant energy
offset. Its absolute rounding is a small multiple of machine epsilon times
:attr:`RitzResidual.cancellation_scale`, so a residual norm whose square is not
well above that is not resolved. :attr:`RitzResidual.resolved` says which case
applies, and the variance is reported signed rather than clipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..multivector import MV
from .linalg import SubspaceResult
from .projection import MatrixElementBank
from .second_moment_storage import (
    SecondMomentStoragePolicy, operator_fingerprint, refresh_second_moment_storage,
    require_second_moment_storage,
)

SECOND_MOMENT_LABEL = "H^2"
# Relative size, against the cancelled magnitude, below which a variance is
# rounding rather than signal: about 4500 machine epsilons, generous enough to
# cover the summation depth of a row with several hundred thousand terms.
RESOLUTION = 1e-12


@dataclass(frozen=True)
class RitzResidual:
    """The true residual of one Ritz root, from the second-moment block.

    ``second_moment`` is ``<Psi|H^2|Psi>/<Psi|Psi>`` and ``variance`` is
    ``||(H - E)|Psi>||^2/<Psi|Psi>`` at the root's reported energy. It can be
    slightly negative from rounding, which is why ``residual_norm`` is clipped
    at zero and ``variance`` is not.
    """

    root: int
    energy: float
    rayleigh_energy: float
    second_moment: float
    variance: float
    residual_norm: float
    cancellation_scale: float

    @property
    def resolved(self) -> bool:
        """True when the variance stands above the rounding of its own terms."""
        return self.variance > RESOLUTION * self.cancellation_scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root, "energy": self.energy,
            "rayleigh_energy": self.rayleigh_energy,
            "second_moment": self.second_moment, "variance": self.variance,
            "residual_norm": self.residual_norm,
            "cancellation_scale": self.cancellation_scale, "resolved": self.resolved,
        }


class SecondMomentBank:
    """Second moments over a :class:`MatrixElementBank`'s generators.

    The bank supplies the reference, the generators and the storage; this
    class adds ``H^2`` as one registered observable, so its rows are cached,
    counted and reported with the bank's other rows. Rows are built lazily,
    upper triangle only, on first use. ``indices`` fixes the approved block;
    omitted indices mean the generators registered at construction. Later bank
    growth does not extend that block. ``storage_policy`` declares a finite
    coefficient budget for the preflight, including the ordinary object rows
    used for projected observables on every S/H storage backend.
    """

    def __init__(self, bank: MatrixElementBank, indices: Sequence[int] | None = None,
                 *, storage_policy: SecondMomentStoragePolicy | None = None):
        if not isinstance(bank, MatrixElementBank):
            raise TypeError("SecondMomentBank needs a MatrixElementBank")
        self._storage_preflight = require_second_moment_storage(
            bank, indices, policy=storage_policy)
        self._indices = self._storage_preflight.indices
        self._bank = bank
        self._square = bank.hamiltonian * bank.hamiltonian
        if not self._square.is_hermitian():
            raise ValueError("H squared must be Hermitian for a triangular second-moment block")
        self._square_fingerprint = operator_fingerprint(self._square)

    @property
    def bank(self) -> MatrixElementBank:
        return self._bank

    @property
    def storage_preflight(self):
        """The latest product-free assessment that approved this block."""
        return self._storage_preflight

    @property
    def indices(self) -> tuple[int, ...]:
        """Generator ids covered by this second-moment bank."""
        return self._indices

    def resolve(self, indices: Sequence[int] | None = None) -> tuple[int, ...]:
        """Validate a subset of the approved block before materializing rows."""
        order = self._bank.resolve(self._indices if indices is None else indices)
        if not set(order).issubset(self._indices):
            raise ValueError("second-moment indices exceed the storage-preflight block; "
                             "construct a new SecondMomentBank for that block")
        if not self._storage_preflight.matches_inputs(self._bank):
            raise ValueError("second-moment storage-preflight inputs changed; "
                             "construct a new MatrixElementBank and preflight it")
        if operator_fingerprint(self._square) != self._square_fingerprint:
            raise ValueError("the preflighted Hamiltonian square changed; "
                             "construct a new SecondMomentBank")
        self._storage_preflight = refresh_second_moment_storage(
            self._bank, self._storage_preflight, self._square)
        return order

    @property
    def hamiltonian_square(self) -> MV:
        """``H * H`` by the multivector product, formed once."""
        self.resolve(())
        return self._square

    def row(self, i: int, j: int) -> MV:
        """The operator ``A_i' (H^2 A_j)`` whose reference pairing is ``K_ij``."""
        self.resolve((i,) if i == j else (i, j))
        return self._row(i, j)

    def _row(self, i: int, j: int) -> MV:
        return self._bank.observable_operator(self._square, i, j,
                                              label=SECOND_MOMENT_LABEL)

    def matrix(self, indices: Sequence[int] | None = None) -> np.ndarray:
        """The Hermitian block ``K[a, b] = <psi|A_a' H^2 A_b|psi>``."""
        order = self.resolve(indices)
        return self._bank.project_observable(self._square, order,
                                             label=SECOND_MOMENT_LABEL)

    def _pairs(self, indices: Sequence[int] | None):
        order = self._indices if indices is None else indices
        for b, j in enumerate(order):
            for i in order[:b + 1]:
                yield i, j

    def word_set(self, indices: Sequence[int] | None = None) -> frozenset[int]:
        """Words of the second-moment rows over a block: the preflight's ``U_K``."""
        order = self.resolve(indices)
        words: set[int] = set()
        for i, j in self._pairs(order):
            words.update(self._row(i, j).terms)
        return frozenset(words)

    def resources(self, indices: Sequence[int] | None = None) -> dict[str, Any]:
        """What the block adds to measure and to hold, beside ``(S, H)``."""
        order = self.resolve(indices)
        k_words: set[int] = set()
        second = largest = 0
        for i, j in self._pairs(order):
            operator = self._row(i, j)
            k_words.update(operator.terms)
            second += operator.nnz()
            largest = max(largest, operator.nnz())
            del operator
        sh_words = self._bank.word_set(order)
        # Counted from the block's own rows: the bank's resident total would
        # include these second-moment rows once they exist.
        first = sum(self._bank.overlap_operator(i, j).nnz()
                    + self._bank.element_operator(i, j).nnz()
                    for i, j in self._pairs(order))
        combined = len(k_words | sh_words)
        return {
            "basis_size": len(order),
            "block_pairs": len(order) * (len(order) + 1) // 2,
            "hamiltonian_square_terms": self._square.nnz(),
            "sh_word_universe": len(sh_words),
            "second_moment_word_universe": len(k_words),
            "combined_word_universe": combined,
            "additional_words": combined - len(sh_words),
            "word_ratio": combined / len(sh_words) if sh_words else math.inf,
            "second_moment_coefficient_occurrences": second,
            "largest_row_terms": largest,
            "sh_coefficient_occurrences": first,
            "total_coefficient_occurrences": first + second,
        }

    def moments(self, coefficients, indices: Sequence[int] | None = None,
                ) -> tuple[float, float, float, float]:
        """``(c'Sc, c'Hc, c'Kc, |c|'|K||c|)`` for one coefficient vector."""
        order = self.resolve(indices)
        c = np.asarray(coefficients, dtype=complex).reshape(-1)
        if c.size != len(order):
            raise ValueError(f"{c.size} coefficients for a basis of {len(order)}")
        S, H = self._bank.matrices(order)
        K = self.matrix(order)
        magnitude = np.abs(c)
        return (float((c.conj() @ S @ c).real), float((c.conj() @ H @ c).real),
                float((c.conj() @ K @ c).real), float(magnitude @ np.abs(K) @ magnitude))

    def residual(self, result: SubspaceResult, k: int = 0) -> RitzResidual:
        """The true Ritz residual of root ``k`` of a solve on this bank."""
        if result.bank is not self._bank:
            raise ValueError("the result was not solved on this bank; its "
                             "coefficients index another basis")
        if not 0 <= k < len(result.energies):
            raise IndexError(f"root {k} out of range")
        norm, energy_term, square_term, scale = self.moments(
            result.coefficients[:, k], result.indices)
        if norm <= 0.0:
            raise ValueError("the Ritz vector has no norm on this basis")
        energy = float(result.energies[k])
        rayleigh = energy_term / norm
        second = square_term / norm
        variance = second - 2.0 * energy * rayleigh + energy * energy
        return RitzResidual(
            root=k, energy=energy, rayleigh_energy=rayleigh, second_moment=second,
            variance=variance, residual_norm=math.sqrt(max(variance, 0.0)),
            cancellation_scale=scale / norm)

    def residuals(self, result: SubspaceResult) -> tuple[RitzResidual, ...]:
        return tuple(self.residual(result, k) for k in range(len(result.energies)))

    def energy_variance(self, coefficients, indices: Sequence[int] | None = None) -> float:
        """``<H^2> - <H>^2`` of ``sum_j c_j A_j|psi>``, at its Rayleigh quotient."""
        norm, energy_term, square_term, _ = self.moments(coefficients, indices)
        if norm <= 0.0:
            raise ValueError("the state has no norm on this basis")
        rayleigh = energy_term / norm
        return square_term / norm - rayleigh * rayleigh
