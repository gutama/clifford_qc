"""Linear functionals of shared Pauli-word means."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..ir import PauliWord
from ..multivector import MV
from .cache import GroupedWordCache
from .confidence import candidate_radius

if TYPE_CHECKING:
    from ..subspace.projection import MatrixElementBank

IDENTITY_CODE = 0

class PackedCoefficients(Mapping):
    """Immutable, ordered real coefficients independent of bank row lifetime.

    Iteration preserves emission order. A sorted permutation supports lookup
    without a Python dictionary. Larger-than-int64 codes retain Python ints.
    """

    def __init__(self, coefficients):
        pairs = list(coefficients.items())
        codes = [code for code, _ in pairs]
        dtype = np.int64 if all(0 <= code < 2**63 for code in codes) else object
        self._codes = np.array(codes, dtype=dtype)
        self._values = np.array([value for _, value in pairs], dtype=np.float64)
        self._order = np.argsort(self._codes, kind="stable")
        self._sorted = self._codes[self._order]
        for array in (self._codes, self._values, self._order, self._sorted):
            array.flags.writeable = False

    def __len__(self):
        return len(self._codes)

    def __iter__(self):
        return (int(code) for code in self._codes)

    def __getitem__(self, code):
        position = int(np.searchsorted(self._sorted, code))
        if position >= len(self) or self._sorted[position] != code:
            raise KeyError(code)
        return float(self._values[self._order[position]])

    def items(self):
        return ((int(code), float(value)) for code, value in zip(self._codes, self._values))

    def values(self):
        return (float(value) for value in self._values)

    @property
    def nbytes(self):
        return sum(array.nbytes for array in (self._codes, self._values, self._order, self._sorted))


@dataclass(frozen=True)
class WordFunctional:
    """``f(mu) = constant + sum_w c_w mu_w``: the only shape shots enter through.

    The identity word is split off into ``constant`` rather than measured.
    ``<I> = 1`` exactly, so measuring it wastes nothing but *reporting* it as
    measured would inflate the empirical-Bernstein range of every group that
    reads it -- a constant offset widens the a-priori range while contributing
    no variance, which loosens the bound for no reason.
    """

    coefficients: Mapping[int, float]
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
            key = group["key"]
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
    """Coefficients split over the groups the cache reads each word from,
    scaled by the cache's pooling weights.

    The weights are what keeps this honest: several groups may be *able* to read
    the same word, and attributing the full coefficient to each of them would
    count its contribution once per capable group -- inflating a variance by
    that factor.  Weights summing to one over the reading groups spread the
    coefficient instead of duplicating it, which is a reweighting of the same
    estimate rather than a second copy of it.  ``pooling='assigned'`` puts all
    the weight on one group and reproduces the single-assignment behavior.
    """
    out: dict[tuple, dict] = {}
    for code, c in coefficients.items():
        for key, weight in cache.group_weights(code).items():
            row = out.setdefault(key, {})
            row[code] = row.get(code, 0.0) + c * weight
    return out


def _positions(cache: GroupedWordCache, coefficients: dict, group: dict):
    """``[(coefficient, sign, bit positions)]`` for one setting."""
    support = group["support"]
    explicit = group.get("readouts", {})
    out = []
    for code, c in coefficients.items():
        if code in explicit:
            sign, positions = explicit[code]
        elif group.get("setting_key") is not None:
            raise KeyError(
                f"compiled setting {group['setting_key']!r} has no readout "
                f"for word code {code}"
            )
        else:
            sign = 1
            word_support = [j for j in range(cache.n) if (code >> (2 * j)) & 3]
            positions = [support.index(j) for j in word_support]
        out.append((c, sign, positions))
    return out


def _combined_value(bits: str, entries) -> float:
    """Per-shot value ``sum_w c_w o_w`` of one outcome bitstring."""
    total = 0.0
    for c, sign, positions in entries:
        parity = sum(bits[p] == "1" for p in positions) % 2
        total += sign * (-c if parity else c)
    return total


def split_complex_coefficients(coefficients, *, storage="object") -> tuple[WordFunctional, WordFunctional]:
    """A complex functional as its real and imaginary real-valued parts.

    Element operators are non-Hermitian, so their word coefficients are complex
    while word means are real. Two real functionals of the same measurements is
    the honest decomposition; bounding a modulus then means bounding a rectangle
    and taking the union bound over its two sides.
    """
    if storage not in ("object", "packed"):
        raise ValueError("coefficient storage must be object or packed")
    real_c: dict[int, float] = {}
    imag_c: dict[int, float] = {}
    real_const = imag_const = 0.0
    terms = coefficients.items() if isinstance(coefficients, Mapping) else coefficients
    for code, value in terms:
        z = complex(value)
        if code == IDENTITY_CODE:
            real_const += z.real
            imag_const += z.imag
            continue
        if z.real != 0.0:
            real_c[code] = real_c.get(code, 0.0) + z.real
        if z.imag != 0.0:
            imag_c[code] = imag_c.get(code, 0.0) + z.imag
    if storage == "packed":
        real_c, imag_c = PackedCoefficients(real_c), PackedCoefficients(imag_c)
    return (WordFunctional(real_c, real_const), WordFunctional(imag_c, imag_const))


def entry_functionals(bank: MatrixElementBank, i: int, j: int, *, storage="object"
                      ) -> tuple[tuple[WordFunctional, WordFunctional],
                                 tuple[WordFunctional, WordFunctional]]:
    """``(S_ij, H_ij)`` as (real, imaginary) functional pairs of word means."""
    return tuple(split_complex_coefficients(bank.iter_operator_terms(kind, i, j),
                                             storage=storage)
                 for kind in ("overlap", "element"))


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
    return split_complex_coefficients(accumulated)

# Compatibility for callers that used the former module-private helper.
_split_complex = split_complex_coefficients

__all__ = [
    "WordFunctional", "split_complex_coefficients", "entry_functionals",
    "ritz_functional", "coupling_functional",
]
