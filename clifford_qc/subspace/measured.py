"""A-CASE Phase 4: finite-shot layers.

Everything the subspace needs is a linear functional of Pauli-word means. That
is the whole reason the bank stores coefficient maps rather than scalars:

    S_ij = sum_w o^S_ij,w mu_w,     H_ij = sum_w o^H_ij,w mu_w,
    mu_w = <W_w> on the single reference state.

So one shared grouped measurement of the word universe estimates *every* entry
at once, and each derived quantity is another functional of the same means.
:class:`WordFunctional` is that object, and it is the only place shots enter.

What is *not* linear is everything the certification has to talk about. The
retained eigenspace is data-dependent, the Ritz pair ``(c, E)`` is
data-dependent, and residual couplings and 2x2 lowerings are nonlinear
functions of correlated estimates. The Paper A best-arm machinery assumed
linear estimators of fixed observables; it does not transfer unchanged. Hence
the plan's three stages, kept apart in the API so that no asymptotic statement
can be mistaken for a certificate:

*4A -- shared grouped measurement.* :class:`SharedMeasurement` measures the
universe through QWC groups and reconstructs ``(S, H)``. Its
``exact_matrices()`` walks the same reconstruction with exact word means, which
is the infinite-shot limit and must reproduce the Phase-2 matrices.

*4B -- asymptotic uncertainty.* :func:`ritz_uncertainty` linearizes the Ritz
value: to first order ``dE = sum_w q_w dmu_w`` with ``q`` the word coefficients
of ``B'(H - E)B``, ``B = sum_i c_i A_i`` -- one real functional, whose variance
comes from the grouped joint histograms. :func:`bootstrap_ritz` resamples those
histograms instead. Both are labelled ``asymptotic`` / ``heuristic``. Neither
is a certificate.

*4C -- finite-sample growth certificate by sample splitting.*
:func:`run_certified_acase` spends a construction batch on ``(S, H)``, freezes
the thresholded subspace and its Ritz pair, then spends an *independent*
certification batch on the candidates' residual couplings with ``(c, E)``
treated as constants. Those couplings are genuinely linear in the fresh word
means, so empirical-Bernstein bounds apply and the growth decision -- or
abstention -- carries a finite-sample guarantee. Less measurement-efficient
than reusing every shot; it is what can actually be proved.

**Covariance discipline.** No dense covariance over ``(H, S)`` entries is ever
formed (``O(M^2)`` entries, ``O(M^4)`` pairs). The grouped histograms stay the
sufficient statistic and :meth:`WordFunctional.covariance` computes one
covariance-vector product on demand. Comparisons need no covariance object at
all: a difference of linear functionals is a linear functional.

**The variational bound does not survive noise.** Thresholding a noisy ``S`` is
a PSD repair, and nothing here claims it preserves ``E_sub >= E_0``. The tests
demonstrate the violation rather than papering over it; quantifying it in terms
of ``tau_S``, shot covariance, and conditioning is open question Q3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..algorithms.adapt import TIE_ATOL, TIE_RTOL, canonical_argmax
from ..ir import PauliWord
from ..measurement.cache import GroupedWordCache
from ..measurement.confidence import candidate_radius
from ..measurement.grouping import qwc_groups
from ..multivector import MV
from .adaptive import sector_leakage
from .elements import MatrixElementBank
from .generators import as_generators, identity_generator
from .solver import (DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S,
                     SubspaceResult, solve_projected)

# Evidence labels. Only ``finite_sample`` is a certificate; ``asymptotic`` is a
# Gaussian/delta-method approximation and ``heuristic`` a resampling estimate.
EXACT = "exact"
ASYMPTOTIC = "asymptotic"
HEURISTIC = "heuristic"
FINITE_SAMPLE = "finite_sample"

IDENTITY_CODE = 0


@dataclass(frozen=True)
class WordFunctional:
    """``f(mu) = constant + sum_w c_w mu_w``: the only shape shots enter through.

    The identity word is split off into ``constant`` rather than measured.
    ``<I> = 1`` exactly, so measuring it wastes nothing but *reporting* it as
    measured would inflate the empirical-Bernstein range of every group that
    reads it -- a constant offset widens the a-priori range while contributing
    no variance, which loosens the bound for no reason.
    """

    coefficients: dict[int, float]
    constant: float = 0.0

    @property
    def is_deterministic(self) -> bool:
        """True when the functional needs no measurement at all."""
        return not self.coefficients

    def words(self, n: int) -> tuple[PauliWord, ...]:
        return tuple(PauliWord(n, code) for code in sorted(self.coefficients))

    def exact(self, rho: MV) -> float:
        """The infinite-shot value, from ``mu_w = <W_w> = 2^n rho_w``."""
        scale = float(2 ** rho.n)
        total = self.constant
        for code, c in self.coefficients.items():
            coeff = rho.terms.get(code)
            if coeff is not None:
                total += c * scale * coeff.real
        return total

    def estimate(self, cache: GroupedWordCache) -> float:
        if self.is_deterministic:
            return self.constant
        return self.constant + cache.candidate_estimate(self.coefficients)

    def group_terms(self, cache: GroupedWordCache):
        """``(N_g, var_g, range_g)`` per group, or ``None`` if a word is unmeasured."""
        if self.is_deterministic:
            return []
        return cache.candidate_group_terms(self.coefficients)

    def radius(self, cache: GroupedWordCache, delta: float, family: int = 1, *,
               bound: str = "normal", rounds: int = 1,
               method: str = "bonferroni") -> float:
        """Two-sided confidence radius at simultaneous level ``1 - delta``.

        A deterministic functional has radius zero -- ``S_00`` for the identity
        generator is known exactly and should not be charged an interval.
        """
        terms = self.group_terms(cache)
        if terms is None:
            return float("inf")
        if not terms:
            return 0.0
        return candidate_radius(terms, delta, family, bound=bound, rounds=rounds,
                                method=method)

    def variance(self, cache: GroupedWordCache) -> float:
        """Covariance-aware variance of the estimate (within-group correlations
        exact, groups independent)."""
        return self.covariance(self, cache)

    def covariance(self, other: "WordFunctional", cache: GroupedWordCache) -> float:
        """``Cov(f_hat, g_hat)`` from the grouped histograms -- one covariance-vector
        product, no covariance matrix.

        Groups are independent circuits, so the covariance is a sum over the
        groups both functionals touch of the sample covariance of their per-shot
        combined values, divided by that group's shot count. Everything the
        plan's covariance discipline asks for follows from this and linearity:
        the variance of any combination, and the comparison of two candidates
        (whose difference is itself a functional).
        """
        if self.is_deterministic or other.is_deterministic:
            return 0.0
        mine = _bucket_by_group(cache, self.coefficients)
        theirs = _bucket_by_group(cache, other.coefficients)
        total = 0.0
        for group in cache.group_states():
            shots = group["shots"]
            key = group["basis"]
            if shots <= 0 or key not in mine or key not in theirs:
                continue
            left = _positions(cache, mine[key], group)
            right = _positions(cache, theirs[key], group)
            s_left = s_right = s_both = 0.0
            for bits, count in group["hist"].items():
                u = _combined_value(bits, left)
                v = _combined_value(bits, right)
                s_left += u * count
                s_right += v * count
                s_both += u * v * count
            mean_left = s_left / shots
            mean_right = s_right / shots
            total += (s_both / shots - mean_left * mean_right) / shots
        return total

    def __add__(self, other: "WordFunctional") -> "WordFunctional":
        merged = dict(self.coefficients)
        for code, c in other.coefficients.items():
            merged[code] = merged.get(code, 0.0) + c
        return WordFunctional({k: v for k, v in merged.items() if v != 0.0},
                              self.constant + other.constant)

    def __neg__(self) -> "WordFunctional":
        return WordFunctional({k: -v for k, v in self.coefficients.items()},
                              -self.constant)

    def __sub__(self, other: "WordFunctional") -> "WordFunctional":
        return self + (-other)


def _bucket_by_group(cache: GroupedWordCache, coefficients: dict) -> dict[tuple, dict]:
    """Coefficients split by the *one* group the cache reads each word from.

    The single assignment matters: several groups may be able to read the same
    word, and attributing it to each of them would count its contribution once
    per capable group -- inflating a variance by that factor.
    """
    out: dict[tuple, dict] = {}
    for code, c in coefficients.items():
        key = cache.group_key(code)
        if key is None:
            continue
        out.setdefault(key, {})[code] = c
    return out


def _positions(cache: GroupedWordCache, coefficients: dict, group: dict):
    """``[(coefficient, bit positions)]`` for words assigned to this group."""
    support = group["support"]
    out = []
    for code, c in coefficients.items():
        word_support = [j for j in range(cache.n) if (code >> (2 * j)) & 3]
        out.append((c, [support.index(j) for j in word_support]))
    return out


def _combined_value(bits: str, entries) -> float:
    """Per-shot value ``sum_w c_w o_w`` of one outcome bitstring."""
    total = 0.0
    for c, positions in entries:
        parity = sum(bits[p] == "1" for p in positions) % 2
        total += -c if parity else c
    return total


def _split_complex(coefficients: dict[int, complex]) -> tuple[WordFunctional, WordFunctional]:
    """A complex functional as its real and imaginary real-valued parts.

    Element operators are non-Hermitian, so their word coefficients are complex
    while word means are real. Two real functionals of the same measurements is
    the honest decomposition; bounding a modulus then means bounding a rectangle
    and taking the union bound over its two sides.
    """
    real_c: dict[int, float] = {}
    imag_c: dict[int, float] = {}
    real_const = imag_const = 0.0
    for code, value in coefficients.items():
        z = complex(value)
        if code == IDENTITY_CODE:
            real_const += z.real
            imag_const += z.imag
            continue
        if z.real != 0.0:
            real_c[code] = real_c.get(code, 0.0) + z.real
        if z.imag != 0.0:
            imag_c[code] = imag_c.get(code, 0.0) + z.imag
    return (WordFunctional(real_c, real_const), WordFunctional(imag_c, imag_const))


def entry_functionals(bank: MatrixElementBank, i: int, j: int
                      ) -> tuple[tuple[WordFunctional, WordFunctional],
                                 tuple[WordFunctional, WordFunctional]]:
    """``(S_ij, H_ij)`` as (real, imaginary) functional pairs of word means."""
    overlap = bank.overlap_operator(i, j)
    element = bank.element_operator(i, j)
    return _split_complex(overlap.terms), _split_complex(element.terms)


def ritz_functional(bank: MatrixElementBank, indices: Sequence[int],
                    coefficients: np.ndarray, energy: float) -> WordFunctional:
    """Word coefficients of ``B'(H - E)B`` with ``B = sum_i c_i A_i`` (§4B).

    First-order perturbation of a Ritz value with ``c' S c = 1`` gives
    ``dE = c'(dH - E dS)c``, so this single *real* functional is the Jacobian of
    the Ritz value with respect to every word mean at once. Real because
    ``B'(H - E)B`` is Hermitian -- accumulated over the upper triangle and
    mirrored, so no imaginary residue has to be discarded.
    """
    order = tuple(indices)
    out: dict[int, float] = {}
    constant = 0.0

    def add(code: int, value: float) -> None:
        nonlocal constant
        if value == 0.0:
            return
        if code == IDENTITY_CODE:
            constant += value
        else:
            out[code] = out.get(code, 0.0) + value

    for a, i in enumerate(order):
        for b, j in enumerate(order[a:], start=a):
            weight = np.conjugate(coefficients[a]) * coefficients[b]
            if weight == 0:
                continue
            overlap = bank.overlap_operator(i, j).terms
            element = bank.element_operator(i, j).terms
            for code in set(element) | set(overlap):
                value = (complex(element.get(code, 0.0))
                         - energy * complex(overlap.get(code, 0.0)))
                contribution = weight * value
                # i == j is Hermitian on its own; i < j pairs with j > i, whose
                # operator is the adjoint, so the two add to twice the real part.
                add(code, contribution.real if a == b else 2.0 * contribution.real)
    return WordFunctional({k: v for k, v in out.items() if v != 0.0}, constant)


def coupling_functional(bank: MatrixElementBank, indices: Sequence[int],
                        coefficients: np.ndarray, energy: float, candidate: int
                        ) -> tuple[WordFunctional, WordFunctional]:
    """``h_a - E s_a = <Psi_m|(H - E)|chi_a>`` as (real, imaginary) functionals.

    Linear in the word means *given* ``(c, E)``, which is exactly why the
    sample-split certificate can bound it with a finite-sample inequality: the
    construction batch fixes the constants, the certification batch supplies
    fresh, independent means.
    """
    accumulated: dict[int, complex] = {}
    for a, i in enumerate(indices):
        weight = np.conjugate(coefficients[a])
        if weight == 0:
            continue
        overlap = bank.overlap_operator(i, candidate).terms
        element = bank.element_operator(i, candidate).terms
        for code in set(element) | set(overlap):
            value = (complex(element.get(code, 0.0))
                     - energy * complex(overlap.get(code, 0.0)))
            accumulated[code] = accumulated.get(code, 0j) + weight * value
    return _split_complex(accumulated)


@dataclass(frozen=True)
class Interval:
    """An estimate with a two-sided interval and an explicit evidence label."""

    estimate: float
    lower: float
    upper: float
    delta: float
    evidence: str
    std_error: float | None = None

    @property
    def certified(self) -> bool:
        """Only a finite-sample interval is a certificate."""
        return self.evidence == FINITE_SAMPLE


class SharedMeasurement:
    """One QWC-grouped measurement of a bank subspace's whole word universe (4A).

    Every entry of ``(S, H)`` is reconstructed from the same shots, which is the
    measurement-sharing proposition made concrete: a word appearing in many
    element operators is paid for once. The grouping is fixed at construction so
    the same circuits recur every batch, which is what makes each group's
    cumulative histogram a sufficient statistic -- and what the
    finite-sample bounds require.
    """

    def __init__(self, bank: MatrixElementBank, indices: Sequence[int] | None = None,
                 *, groups: Sequence[Sequence[PauliWord]] | None = None):
        self.bank = bank
        self.indices = bank._resolve(indices)
        bank.matrices(self.indices)  # force every pair, so the universe is complete
        self._pairs = {(i, j): entry_functionals(bank, i, j)
                       for a, i in enumerate(self.indices)
                       for j in self.indices[a:]}
        codes = sorted({code
                        for pair in self._pairs.values()
                        for part in pair for f in part
                        for code in f.coefficients})
        self.words = tuple(PauliWord(bank.n, code) for code in codes)
        # A caller may supply a wider grouping -- one covering a superset of
        # these words -- so that a sub-block and the full universe are read out
        # of the *same* circuits and the same cache.
        self.groups = qwc_groups(list(self.words)) if groups is None else [
            list(group) for group in groups]
        covered = {w.code for group in self.groups for w in group}
        if not covered.issuperset(codes):
            raise ValueError("supplied groups do not cover this subspace's words")

    def diagonal_functional(self, index: int) -> WordFunctional:
        """``S_ii`` as a functional -- the candidate norm, when it must be measured."""
        return self._pairs[(index, index)][0][0]

    @property
    def n(self) -> int:
        return self.bank.n

    def new_cache(self) -> GroupedWordCache:
        return GroupedWordCache(self.n)

    def measure(self, backend, shots_per_group: int,
                cache: GroupedWordCache | None = None) -> GroupedWordCache:
        """Sample every group ``shots_per_group`` times into (a fresh) cache.

        A predeclared uniform allocation: the schedule's endpoints are fixed
        before any outcome is seen, which is the condition the
        empirical-Bernstein bounds are valid under.
        """
        if shots_per_group <= 0:
            raise ValueError("shots_per_group must be positive")
        if not self.groups:
            raise ValueError("nothing to measure: the universe is empty")
        cache = self.new_cache() if cache is None else cache
        batch = backend.sample_grouped_from_state(self.bank.reference, self.groups,
                                                 shots_per_group)
        cache.add_batch(batch)
        return cache

    def _assemble(self, evaluate) -> tuple[np.ndarray, np.ndarray]:
        m = len(self.indices)
        S = np.zeros((m, m), dtype=complex)
        Hm = np.zeros((m, m), dtype=complex)
        for a, i in enumerate(self.indices):
            for b, j in enumerate(self.indices[a:], start=a):
                (s_re, s_im), (h_re, h_im) = self._pairs[(i, j)]
                S[a, b] = complex(evaluate(s_re), evaluate(s_im))
                Hm[a, b] = complex(evaluate(h_re), evaluate(h_im))
        # Same structural Hermiticity as the exact path: the lower triangle is
        # the conjugate by definition, never an independent estimate that would
        # have to be symmetrized (and would double the noise on the diagonal).
        for b in range(m):
            S[b, b] = S[b, b].real
            Hm[b, b] = Hm[b, b].real
            for a in range(b):
                S[b, a] = S[a, b].conjugate()
                Hm[b, a] = Hm[a, b].conjugate()
        return S, Hm

    def matrices(self, cache: GroupedWordCache) -> tuple[np.ndarray, np.ndarray]:
        """``(S_hat, H_hat)`` from measured word means."""
        return self._assemble(lambda f: f.estimate(cache))

    def exact_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """The infinite-shot limit of the same reconstruction (4A acceptance)."""
        rho = self.bank.reference
        return self._assemble(lambda f: f.exact(rho))

    def solve(self, cache: GroupedWordCache, *, tau_s: float = DEFAULT_TAU_S,
              rel_tau: float = 0.0, max_condition: float = DEFAULT_MAX_CONDITION,
              norm_floor: float = DEFAULT_NORM_FLOOR) -> SubspaceResult:
        """Thresholded GEP on the measured matrices.

        The result reports the same conditioning metrics as the exact path plus
        ``overlap_negative_modes``: with estimated entries ``S_hat`` is no longer
        positive semidefinite, and thresholding those modes away is a repair
        whose effect on the variational bound is not established (Q3).
        """
        S, Hm = self.matrices(cache)
        resources = dict(self.bank.resources(self.indices))
        resources.update({
            "evidence": HEURISTIC,
            "shots": cache.total_shots,
            "circuits": cache.total_circuits,
            "qwc_groups": len(self.groups),
            "measured_words": len(self.words),
            "shots_per_measured_word": (cache.total_shots / len(self.words)
                                        if self.words else 0.0),
        })
        return solve_projected(S, Hm,
                               [self.bank.generator(i).label for i in self.indices],
                               tau_s=tau_s, rel_tau=rel_tau,
                               max_condition=max_condition, norm_floor=norm_floor,
                               resources=resources).with_bank(self.bank, self.indices)


# --------------------------------------------------------------- 4B: uncertainty


def ritz_uncertainty(shared: SharedMeasurement, cache: GroupedWordCache,
                     result: SubspaceResult, *, root: int = 0, delta: float = 0.05,
                     family: int = 1) -> Interval:
    """Delta-method interval on a measured Ritz value -- ``asymptotic``, not certified.

    Two approximations stack, and both are why the label is what it is: the
    Gaussian radius is asymptotic in the shot count, and the linearization
    ignores the second-order response of the Ritz value (including the fact
    that the retained eigenspace itself moved with the data). It is a useful
    error bar and not a guarantee.
    """
    functional = ritz_functional(shared.bank, result.indices,
                                 result.coefficients[:, root], result.energies[root])
    variance = functional.variance(cache)
    radius = functional.radius(cache, delta, family, bound="normal")
    energy = result.energies[root]
    return Interval(estimate=energy, lower=energy - radius, upper=energy + radius,
                    delta=delta, evidence=ASYMPTOTIC,
                    std_error=math.sqrt(max(0.0, variance)))


def bootstrap_ritz(shared: SharedMeasurement, cache: GroupedWordCache, *,
                   root: int = 0, replicates: int = 200, seed: int = 0,
                   delta: float = 0.05, solver_kwargs: dict | None = None) -> Interval:
    """Grouped bootstrap on the Ritz value -- ``heuristic``, and the cross-check
    the delta method needs.

    Each group's outcome histogram is resampled multinomially at its own shot
    count (groups are independent circuits, so this is the right resampling
    unit), then the whole pipeline -- reconstruct, threshold, diagonalize -- is
    rerun. It therefore captures what the linearization drops: the movement of
    the retained eigenspace itself. It captures nothing about coverage under a
    different true state, which is why it is heuristic.
    """
    from ..backends.protocol import GroupSample, MeasurementBatch

    rng = np.random.default_rng(seed)
    states = cache.group_states()
    samples: list[float] = []
    kwargs = solver_kwargs or {}
    for _ in range(replicates):
        groups = []
        for group in states:
            keys = list(group["hist"])
            counts = np.array([group["hist"][k] for k in keys], dtype=float)
            probs = counts / counts.sum()
            drawn = rng.multinomial(group["shots"], probs)
            groups.append(GroupSample(
                support=group["support"], basis=group["basis"],
                hist={k: int(c) for k, c in zip(keys, drawn) if c},
                shots=group["shots"]))
        replica = shared.new_cache()
        replica.add_batch(MeasurementBatch(n=shared.n, shots={}, plus_counts={},
                                           circuits=len(groups), groups=tuple(groups)))
        samples.append(shared.solve(replica, **kwargs).energies[root])
    values = np.array(samples)
    lower, upper = np.quantile(values, [delta / 2.0, 1.0 - delta / 2.0])
    return Interval(estimate=float(np.mean(values)), lower=float(lower),
                    upper=float(upper), delta=delta, evidence=HEURISTIC,
                    std_error=float(np.std(values, ddof=1)) if replicates > 1 else None)


# ------------------------------------------------- 4C: sample-split certificate


@dataclass(frozen=True)
class CouplingBound:
    """Bound on ``|<chi_a|(H - E)|Psi_m>|`` from a certification batch."""

    index: int
    label: str
    estimate: float
    lower: float
    upper: float
    evidence: str
    rejected: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected is None


@dataclass(frozen=True)
class CertifiedGrowthRecord:
    """One sample-split growth decision, including a refusal to grow."""

    step: int
    selected_label: str | None
    energy: float
    construction_shots: int
    certification_shots: int
    circuits: int
    coupling: CouplingBound | None
    threshold: float
    candidates_scored: int
    certified: bool
    resolution: str
    evidence: str
    abstained: bool
    reason: str | None
    basis_size: int
    effective_rank: int
    condition_number: float
    overlap_negative_modes: int
    word_universe: int


@dataclass(frozen=True)
class CertifiedResult:
    labels: tuple[str, ...]
    energy: float
    energy_history: tuple[float, ...]
    records: tuple[CertifiedGrowthRecord, ...]
    result: SubspaceResult
    bank: MatrixElementBank
    indices: tuple[int, ...]
    stopped_reason: str
    total_shots: int
    total_circuits: int
    abstentions: int
    exact_ground_energy: float | None = None
    resources: dict[str, Any] = field(default_factory=dict)

    @property
    def basis_size(self) -> int:
        return len(self.labels)


def certify_couplings(bank: MatrixElementBank, indices: Sequence[int],
                      coefficients: np.ndarray, energy: float,
                      candidates: Sequence[int], cache: GroupedWordCache, *,
                      norms: dict[int, float], delta: float = 0.05,
                      bound: str = "eb", threshold: float = 0.0) -> list[CouplingBound]:
    """Simultaneous bounds on every candidate's residual coupling modulus.

    ``|r| = sqrt(Re^2 + Im^2)`` is not linear, so the interval is built as a
    rectangle: each of the two real functionals gets its own two-sided radius,
    and the modulus bounds follow from the corner of the rectangle nearest to
    (and farthest from) the origin. The simultaneous family is therefore
    ``2 * len(candidates)`` events, union-bounded -- the price of a nonlinear
    functional, paid explicitly rather than assumed away.

    ``norms`` supplies ``||A_a|psi>||`` per candidate, and it must come from the
    *construction* batch, not from this one and not from exact arithmetic. The
    coupling has to be normalized or the ranking would depend on how a
    candidate happens to be scaled; but dividing by a quantity estimated from
    the same shots would make the statistic a ratio of correlated estimates and
    void the certificate. Frozen by the independent batch, the norm is a
    constant and the numerator stays linear.

    With ``bound='eb'`` the radii are finite-sample valid at fixed shot counts,
    so the resulting statements are certificates *conditional on* the
    construction batch, which fixed ``(c, E)`` and these norms.
    """
    family = max(1, 2 * len(candidates))
    evidence = FINITE_SAMPLE if bound == "eb" else ASYMPTOTIC
    out: list[CouplingBound] = []
    for candidate in candidates:
        label = bank.generator(candidate).label
        real, imag = coupling_functional(bank, indices, coefficients, energy, candidate)
        norm = norms.get(candidate, 0.0)
        if not norm > DEFAULT_NORM_FLOOR:
            out.append(CouplingBound(candidate, label, 0.0, 0.0, 0.0, evidence,
                                     rejected="norm not resolved above the floor"))
            continue
        parts = []
        for functional in (real, imag):
            value = functional.estimate(cache)
            radius = functional.radius(cache, delta, family, bound=bound)
            parts.append((value, radius))
        estimate = math.hypot(parts[0][0], parts[1][0]) / norm
        lower = math.hypot(*(max(0.0, abs(v) - r) for v, r in parts)) / norm
        upper = math.hypot(*(abs(v) + r for v, r in parts)) / norm
        rejected = None if lower > threshold else f"coupling not certified above {threshold:g}"
        out.append(CouplingBound(candidate, label, estimate, lower, upper, evidence,
                                 rejected=rejected))
    return out


def run_certified_acase(rho: MV, hamiltonian, candidates: Sequence, backend, *,
                        initial: Sequence | None = None,
                        bank: MatrixElementBank | None = None,
                        max_size: int = 6,
                        construction_shots: int = 4000,
                        certification_shots: int = 4000,
                        delta: float = 0.05, threshold: float = 1e-2,
                        bound: str = "eb",
                        leakage_tol: float | None = None,
                        exact_ground_energy: float | None = None,
                        tau_s: float = DEFAULT_TAU_S, rel_tau: float = 0.0,
                        max_condition: float = DEFAULT_MAX_CONDITION
                        ) -> CertifiedResult:
    """Finite-shot adaptive growth with a sample-split growth certificate (4C).

    Per step: a construction batch estimates ``(S, H)`` over the current
    subspace's word universe and fixes the thresholded subspace and its Ritz
    pair ``(c, E)``; an independent certification batch estimates every
    candidate's residual coupling with those held constant; simultaneous
    empirical-Bernstein bounds then either certify a candidate above
    ``threshold`` -- and it is added -- or the run **abstains** and stops.

    What is certified is exactly this and no more: on the ``1 - delta`` event,
    conditional on the construction batch, the accepted candidate's residual
    coupling with the frozen Ritz pair exceeds ``threshold``. It is not a
    statement that the candidate is the best available (``resolution`` records
    separately whether the leader also cleared every rival's upper bound), and
    it is emphatically not a bound on the energy: the reported energies come
    from noisy matrices whose variational bound is not established.

    Each step spends a fresh pair of batches. Reusing earlier shots would need
    a confidence-set argument over a data-dependent subspace, which the plan
    leaves to a later stage.

    Two deliberate omissions. Candidates are ranked by residual coupling, not
    by the generalized 2x2 lowering the exact path uses: the lowering is a
    nonlinear function of ``s_aa``, ``h_aa`` and a square root, so it admits
    only an asymptotic treatment and cannot be the certified gate. And there is
    no conditioning rejection here -- the orthogonal fraction is itself a
    nonlinear statistic of the estimates, so screening on it would need its own
    interval rather than a reused exact-path threshold.
    """
    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    initial_gens = (as_generators(initial) if initial is not None
                    else [identity_generator(bank.n)])
    basis = bank.extend(initial_gens)
    pool: list[int] = []
    for index in bank.extend(as_generators(candidates)):
        if index not in basis and index not in pool:
            pool.append(index)
    if not pool:
        raise ValueError("no candidate generators outside the initial basis")

    solver_kwargs = dict(tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition)
    records: list[CertifiedGrowthRecord] = []
    history: list[float] = []
    total_shots = total_circuits = abstentions = 0
    stopped_reason = "basis budget reached"
    result = None

    for step in range(1, max_size + 1):
        remaining = [i for i in pool if i not in basis]
        if leakage_tol is not None:
            remaining = [i for i in remaining
                         if max(sector_leakage(bank.generator(i)).values()) <= leakage_tol]
        if not remaining:
            stopped_reason = "candidate pool exhausted"
            break

        # One grouping over the whole universe -- basis block and candidate rows
        # alike -- so both batches read the same circuits and the sub-block is
        # estimated from the same shots as everything else.
        full = SharedMeasurement(bank, tuple(basis) + tuple(remaining))
        basis_view = SharedMeasurement(bank, basis, groups=full.groups)

        # --- construction batch: the subspace, the Ritz pair, the candidate norms
        construction_cache = full.measure(backend, construction_shots)
        result = basis_view.solve(construction_cache, **solver_kwargs)
        history.append(result.ground_energy)
        total_shots += construction_cache.total_shots
        total_circuits += construction_cache.total_circuits
        coefficients = result.ritz_vector(0)
        energy = result.ground_energy
        norms = {i: math.sqrt(max(0.0, full.diagonal_functional(i)
                                  .estimate(construction_cache)))
                 for i in remaining}

        # --- certification batch: independent shots, everything above constant
        certification_cache = full.measure(backend, certification_shots)
        total_shots += certification_cache.total_shots
        total_circuits += certification_cache.total_circuits

        bounds = certify_couplings(bank, basis, coefficients, energy, remaining,
                                   certification_cache, norms=norms, delta=delta,
                                   bound=bound, threshold=threshold)
        live = [b for b in bounds if b.accepted]
        best = None
        if live:
            pick = canonical_argmax(range(len(live)), lambda k: live[k].estimate,
                                    rtol=TIE_RTOL, atol=TIE_ATOL)
            best = live[pick]

        resolution = "none"
        if best is not None:
            rivals = max((b.upper for b in bounds if b.index != best.index), default=0.0)
            resolution = "best" if best.lower > rivals else "above_threshold"

        record = CertifiedGrowthRecord(
            step=step, selected_label=best.label if best else None,
            energy=energy, construction_shots=construction_cache.total_shots,
            certification_shots=certification_cache.total_shots,
            circuits=(construction_cache.total_circuits
                      + certification_cache.total_circuits),
            coupling=best, threshold=threshold, candidates_scored=len(bounds),
            certified=bool(best is not None and bound == "eb"),
            resolution=resolution,
            evidence=FINITE_SAMPLE if bound == "eb" else ASYMPTOTIC,
            abstained=best is None,
            reason=None if best else "no candidate certified above threshold",
            basis_size=len(basis), effective_rank=result.effective_rank,
            condition_number=result.condition_number,
            overlap_negative_modes=result.resources.get("overlap_negative_modes", 0),
            word_universe=result.resources.get("word_universe", 0))
        records.append(record)

        if best is None:
            abstentions += 1
            stopped_reason = "abstained: no candidate certified above threshold"
            break
        basis.append(best.index)
    if result is None or stopped_reason == "basis budget reached":
        # Either nothing was ever solved (a filter emptied the pool at step 1),
        # or the budget ran out with the last accepted generator never solved
        # for. Either way one more construction batch, so the reported energy
        # describes the basis that was actually grown.
        final = SharedMeasurement(bank, basis)
        cache = final.measure(backend, construction_shots)
        result = final.solve(cache, **solver_kwargs)
        history.append(result.ground_energy)
        total_shots += cache.total_shots
        total_circuits += cache.total_circuits

    resources = dict(result.resources)
    resources.update({
        "candidate_pool_size": len(pool),
        "delta": delta,
        "bound": bound,
        "certification_scope": "per_step_conditional_on_construction_batch",
        "shot_accounting": "construction_plus_certification_batches",
        "variational_bound": "not_established_under_noise",
    })
    return CertifiedResult(
        labels=tuple(bank.generator(i).label for i in basis),
        energy=result.ground_energy, energy_history=tuple(history),
        records=tuple(records), result=result, bank=bank, indices=tuple(basis),
        stopped_reason=stopped_reason, total_shots=total_shots,
        total_circuits=total_circuits, abstentions=abstentions,
        exact_ground_energy=exact_ground_energy, resources=resources)
