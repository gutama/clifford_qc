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

The original bank supplies exact pairings. The separately scoped
SampledSecondMomentProvider below adds fixed-endpoint finite-shot inference.
Q18-S1 (uniform QWC) and Q18-S4 (fully commuting settings) still read NONE on
their historical banks; the new bounded capability claims no cost saving there.

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


@dataclass(frozen=True)
class MeasuredMomentProblem:
    """A simultaneous entrywise confidence region for fixed S/H/K rows.

    Matrices and absolute complex-entry radii own immutable snapshots. On the
    region's event every coefficient vector is covered, including one selected
    from these same matrices. This does not cover adapting the measured row
    universe or stopping endpoint using those observations.
    """
    overlap: np.ndarray
    hamiltonian: np.ndarray
    second_moment: np.ndarray
    overlap_radii: np.ndarray
    hamiltonian_radii: np.ndarray
    second_moment_radii: np.ndarray
    labels: tuple[str, ...]
    delta: float
    resources: dict
    observation_identity: str

    def __post_init__(self):
        from ..reference_states import immutable_array
        if not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("delta must lie in (0,1)")
        size = len(self.labels)
        if not size:
            raise ValueError("need at least one moment direction")
        for field in ("overlap", "hamiltonian", "second_moment"):
            matrix = np.asarray(getattr(self, field), dtype=complex)
            if (matrix.shape != (size, size) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)):
                raise ValueError("moment matrices must be finite Hermitian blocks")
            object.__setattr__(self, field, immutable_array(matrix))
        for field in ("overlap_radii", "hamiltonian_radii", "second_moment_radii"):
            radius = np.asarray(getattr(self, field), dtype=float)
            if (radius.shape != (size, size) or not np.isfinite(radius).all()
                    or np.any(radius < 0) or not np.allclose(radius, radius.T, atol=0, rtol=0)):
                raise ValueError("entry radii must be finite nonnegative symmetric blocks")
            object.__setattr__(self, field, immutable_array(radius))
        object.__setattr__(self, "labels", tuple(self.labels))
        from types import MappingProxyType
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))

    def solve(self, **kwargs):
        """Noisy Ritz solve; its value alone is not a confidence certificate."""
        from .linalg import solve_projected
        return solve_projected(self.overlap, self.hamiltonian, self.labels,
                               resources=dict(self.resources), **kwargs)

    def assess(self, coefficients, *, residual_tolerance: float,
               normalization_floor: float = 1e-6, target: float | None = None) -> dict:
        """Finite-shot convergence and spectral inclusion for one direction.

        The variance interval encloses the *true* Rayleigh variance, rather
        than a residual at the noisy reported energy. A small residual implies
        proximity to some eigenvalue; ground-root identification is separate.
        """
        for value, name in ((residual_tolerance, "residual_tolerance"),
                            (normalization_floor, "normalization_floor")):
            if not math.isfinite(value) or value < 0:
                raise ValueError(name+" must be finite and nonnegative")
        if normalization_floor == 0:
            raise ValueError("normalization_floor must be positive")
        if target is not None and not math.isfinite(target):
            raise ValueError("target must be finite")
        c = np.asarray(coefficients, dtype=complex)
        if c.shape != (len(self.labels),) or not np.isfinite(c).all() or not np.any(c):
            raise ValueError("coefficients must be a nonzero finite vector matching the block")
        # Direction scale is irrelevant. Normalize it before quadratic forms
        # to avoid dependence of the denominator gate on a caller's rescaling.
        scale = max(np.max(np.abs(c.real)), np.max(np.abs(c.imag)))
        c = c.real/scale+1j*(c.imag/scale)
        c = c / np.linalg.norm(c)
        magnitude = np.abs(c)
        point = [float(np.vdot(c, block @ c).real) for block in
                 (self.overlap, self.hamiltonian, self.second_moment)]
        radii = [float(magnitude @ block @ magnitude) for block in
                 (self.overlap_radii, self.hamiltonian_radii, self.second_moment_radii)]
        n, h, k = point
        rn, rh, rk = radii
        ni, hi, ki = [(v-r, v+r) for v, r in zip(point, radii)]
        raw_energy = h/n if n > 0 else None
        raw_variance = k/n-(h/n)**2 if n > 0 else None
        report = {"status": "normalization_unresolved", "confidence_delta": float(self.delta),
                  "normalization": n, "normalization_interval": ni,
                  "energy": raw_energy, "variance": raw_variance,
                  "energy_interval": None, "variance_interval": None,
                  "residual_interval": None, "residual_tolerance": float(residual_tolerance),
                  "target": target, "folded_value": None, "folded_interval": None,
                  "variance_numerical_allowance": None, "folded_numerical_allowance": None,
                  "target_eigenvalue_interval": None, "energy_eigenvalue_interval": None,
                  "ground_state_certificate": False,
                  "observation_identity": self.observation_identity,
                  "evidence": "simultaneous_fixed_endpoint_iid_bounded_shots"}
        if ni[0] <= normalization_floor:
            return report
        def ratio(numerator):
            corners = [a/b for a in numerator for b in ni]
            return (min(corners), max(corners))
        energy, second = ratio(hi), ratio(ki)
        square_min = 0. if energy[0] <= 0 <= energy[1] else min(x*x for x in energy)
        square_max = max(x*x for x in energy)
        numerical = RESOLUTION*max(abs(second[0]), abs(second[1]), square_max)
        variance = (second[0]-square_max-numerical, second[1]-square_min+numerical)
        report.update(energy_interval=energy, variance_interval=variance,
                      variance_numerical_allowance=numerical,
                      status="inconsistent_moments")
        if variance[1] < 0:
            return report
        residual = (math.sqrt(max(0., variance[0])), math.sqrt(variance[1]))
        report["residual_interval"] = residual
        report["energy_eigenvalue_interval"] = (energy[0]-residual[1], energy[1]+residual[1])
        report["status"] = ("within_tolerance" if residual[1] <= residual_tolerance else
                            "above_tolerance" if residual[0] > residual_tolerance else "unresolved")
        if target is not None:
            w = float(target)
            folded = k-2*w*h+w*w*n
            radius = rk+2*abs(w)*rh+w*w*rn
            numerical_folded = RESOLUTION*(abs(k)+2*abs(w*h)+w*w*abs(n))
            radius += numerical_folded
            interval = ratio((folded-radius, folded+radius))
            report.update(folded_value=folded/n, folded_interval=interval,
                          folded_numerical_allowance=numerical_folded)
            if interval[1] < 0:
                report.update(status="inconsistent_moments", residual_interval=None,
                              energy_eigenvalue_interval=None)
            else:
                bound = math.sqrt(interval[1])
                report["target_eigenvalue_interval"] = (w-bound, w+bound)
        return report

    def folded_roots(self, target: float, *, residual_tolerance: float,
                     normalization_floor: float = 1e-6, count: int | None = None,
                     **solve_kwargs) -> dict:
        """Solve the measured folded pencil and assess each reported direction.

        Intervals refer to each direction's true physical moments and contain
        an eigenvalue of H. They do not certify root ordering, distinctness,
        a full-span exact folded eigenvalue, or a ground state.
        """
        from .linalg import solve_projected
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        if np.any(np.asarray(solve_kwargs.get("overlap_ridge", 0.)) != 0):
            raise ValueError("measured folded roots require the original overlap metric")
        w = float(target)
        folded = self.second_moment-2*w*self.hamiltonian+w*w*self.overlap
        solved = solve_projected(self.overlap, folded, self.labels,
                                 resources=dict(self.resources), **solve_kwargs)
        keep = len(solved.energies) if count is None else count
        if isinstance(keep, bool) or not isinstance(keep, (int, np.integer)) or not 1 <= keep <= len(solved.energies):
            raise ValueError("count must select a nonempty subset of the retained roots")
        roots = [{"root": j, "noisy_folded_eigenvalue": float(solved.energies[j]),
                  "coefficients": [[float(z.real), float(z.imag)] for z in solved.ritz_vector(j)],
                  "assessment": self.assess(solved.ritz_vector(j), target=w,
                      residual_tolerance=residual_tolerance, normalization_floor=normalization_floor)}
                 for j in range(keep)]
        return {"target": w, "effective_rank": solved.effective_rank,
                "condition_number": solved.condition_number, "roots": roots,
                "confidence_delta": float(self.delta), "ordered_root_certificate": False}


class SampledSecondMomentProvider:
    """Bounded coefficient-only S/H/K compilation and grouped reconstruction.

    No exact-state capability is requested. Caps are engineering admission,
    not a promotion of the closed historical H-squared affordability studies.
    """
    capabilities = frozenset({"sampled_projected_blocks", "sampled_second_moments"})

    def __init__(self, reference, hamiltonian, generators, *, max_words: int,
                 max_settings: int, storage_policy: SecondMomentStoragePolicy,
                 storage="packed", block_size=None, method="graph_state"):
        from numbers import Integral
        from types import MappingProxyType
        from ..reference_states import (SampledReference, UnsupportedReferenceOperation,
                                        operator_identity, _digest)
        from ..measurement.functionals import (WordFunctional, entry_functionals,
                                               split_complex_coefficients)
        from ..measurement.session import SharedMeasurement
        from ..measurement.planning import compile_block_measurement_plan
        from ..sampled_reference import settings_identity
        from .generator_core import as_generators
        for value in (max_words, max_settings):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("measurement caps must be positive integers")
        if not isinstance(storage_policy, SecondMomentStoragePolicy):
            raise TypeError("an explicit finite second-moment storage policy is required")
        if (not isinstance(reference, SampledReference)
                or "sample_joint" not in reference.metadata.capabilities):
            raise UnsupportedReferenceOperation("reference lacks joint sampling capability")
        self.reference, self.reference_identity = reference, reference.metadata.identity
        self.bank = MatrixElementBank(reference, hamiltonian, as_generators(generators),
                                      storage=storage, coefficient_only=True)
        # Admission runs before H² or any matrix-entry operator is built.
        self.moments = SecondMomentBank(self.bank, storage_policy=storage_policy)
        self.session = SharedMeasurement(self.bank, coefficient_storage=storage)
        self._labels = self.bank.labels
        self._parts = {}
        for a, i in enumerate(self.session.indices):
            for b, j in enumerate(self.session.indices[a:], a):
                s, h = entry_functionals(self.bank, i, j, storage=storage)
                k = split_complex_coefficients(self.moments.row(i, j).terms, storage=storage)
                def freeze(parts):
                    return tuple(WordFunctional(MappingProxyType(dict(f.coefficients)), f.constant)
                                 for f in parts)
                self._parts[(a, b)] = tuple(freeze(parts) for parts in (s, h, k))
        codes = sorted({code for blocks in self._parts.values() for parts in blocks
                        for functional in parts for code in functional.coefficients})
        if len(codes) > max_words:
            raise ValueError("S/H/K word universe exceeds the declared cap")
        self.plan = compile_block_measurement_plan(self.bank.n, codes,
                         self.bank.n if block_size is None else block_size, method=method)
        if len(self.plan.settings) > max_settings:
            raise ValueError("S/H/K setting count exceeds the declared cap")
        self.hamiltonian_identity = operator_identity(self.bank.hamiltonian)
        self.circuit_identity = settings_identity(self.plan.settings)
        self.cache_identity = _digest(["sampled_second_moments.v1", self.reference_identity,
             self.hamiltonian_identity, [operator_identity(self.bank.generator(i).mv)
                 for i in self.session.indices], self.circuit_identity, storage])
        self.admission = {"max_words": int(max_words), "max_settings": int(max_settings),
                          "measured_words": len(codes), "settings": len(self.plan.settings),
                          "coefficient_preflight": self.moments.storage_preflight.as_dict()}

    def compile_measurement(self):
        self._check_identity()
        return self.plan

    def _check_identity(self):
        from ..sampled_reference import settings_identity
        if (self.bank.labels != self._labels
                or self.reference.metadata.identity != self.reference_identity
                or settings_identity(self.plan.settings) != self.circuit_identity):
            raise ValueError("second-moment provider inputs changed")
        self.moments.resolve()

    def reconstruct(self, observations, *, delta: float = 0.05) -> MeasuredMomentProblem:
        from ..measurement.cache import GroupedWordCache
        from ..measurement.ratio import hoeffding_radius
        from ..sampled_reference import SampledObservations, validate_batch
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must lie in (0,1)")
        self._check_identity()
        if (not isinstance(observations, SampledObservations)
                or observations.reference_identity != self.reference_identity
                or observations.settings_identity != self.circuit_identity):
            raise ValueError("observations do not belong to this second-moment plan")
        validate_batch(observations.batch, self.reference_identity, self.bank.n,
                       self.plan.settings, observations.counts)
        if any(count < 1 for count in observations.counts):
            raise ValueError("every compiled setting needs observations")
        cache = GroupedWordCache(self.bank.n)
        if observations.batch.groups:
            cache.add_batch(observations.batch)
        family = max(1, sum(not f.is_deterministic for blocks in self._parts.values()
                            for parts in blocks for f in parts))
        size = len(self._labels)
        matrices = [np.zeros((size, size), complex) for _ in range(3)]
        radii = [np.zeros((size, size), float) for _ in range(3)]
        for (a, b), blocks in self._parts.items():
            for t, (real, imag) in enumerate(blocks):
                value = complex(real.estimate(cache), imag.estimate(cache))
                matrices[t][a, b] = value.real if a == b else value
                matrices[t][b, a] = matrices[t][a, b].conjugate()
                radius = math.hypot(hoeffding_radius(real, cache, delta/family),
                                    hoeffding_radius(imag, cache, delta/family))
                radii[t][a, b] = radii[t][b, a] = radius
        resources = {"provider": "sampled_second_moments", "shots": cache.total_shots,
                     "circuits": cache.total_circuits, "confidence_delta": float(delta),
                     "simultaneous_real_functionals": family, "spectral_certificate": False,
                     "provider_identity": self.cache_identity, "admission": self.admission}
        return MeasuredMomentProblem(*matrices, *radii, self._labels, float(delta),
                                     resources, observations.identity)
