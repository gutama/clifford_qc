"""Sector-annihilating Hamiltonian shifts for measured subspace banks.

A measured bank estimates ``S_ij = <A_i† A_j>`` and ``H_ij = <A_i† H A_j>``
from the Pauli words of those operator products. Let ``Q`` be a conserved
charge (particle number, or ``S_z``), let the reference lie in its sector
``Q = q``, and let every generator commute with ``Q``. Then every bank state
``A_j|psi>`` lies in that sector, and any ``K = (Q - q) O`` with ``O``
commuting with ``Q`` annihilates it:

    <psi| A_i† (H - K) A_j |psi> = <psi| A_i† H A_j |psi>   for every i, j.

The matrices, the solve and every exact quantity are unchanged; only the Pauli
expansion that is measured changes -- its words, their coefficients and so the
shot variance of every estimate built from them. Choosing ``K`` to cut that
variance is the symmetry shift of Lee et al. (arXiv:2409.02504) and the BLISS
family of Loaiza and Izmaylov (arXiv:2304.13772), applied here to a bank's
Ritz functional rather than to ``H`` alone.

A shift is fitted by linear least squares over a declared family of terms
``K_k = D_k O_k`` and is checked before use: each sector operator ``D_k`` must
commute with its multiplier and with every generator, and must vanish on the
reference (``<D_k^2> = 0``). A family that fails is refused, never fitted.
Nothing here changes a default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..fermion import c_op, cdag_op, total_number_op, total_sz_op
from ..ir import PauliSum, PauliWord
from ..measurement.grouping import qwc_groups
from ..multivector import MV, TOL
from ..pauli_kernel import word_mul
from .generator_core import as_generators, to_mv
from .symmetry_pools import _integer, _spin_pairs

OBJECTIVES = ("hamiltonian", "ritz")


@dataclass(frozen=True)
class ShiftTerm:
    """One family member ``K = sector * multiplier``; ``sector`` is ``Q - q``."""

    label: str
    sector: MV
    multiplier: MV

    @property
    def operator(self) -> MV:
        return self.sector * self.multiplier


@dataclass(frozen=True)
class SectorShift:
    """A fitted shift. ``hamiltonian`` is ``H - sum_k coefficients[k] K_k``.

    ``before`` and ``after`` are the fitted objective at zero shift and at the
    fit. ``sector_residual`` is the largest ``<D_k^2>`` on the reference and
    ``commutator_residual`` the largest coefficient of any checked commutator:
    the evidence that every bank element is unchanged.
    """

    hamiltonian: MV
    coefficients: dict[str, float]
    objective: str
    before: float
    after: float
    sector_residual: float
    commutator_residual: float


def _scalar(n: int, value: float) -> MV:
    return MV.scalar(n, complex(value))


def sector_shift_family(n_spatial: int, n_electrons: int, *, one_body: bool = True,
                        sz: float | None = None,
                        spin_ordering: str = "interleaved") -> list[ShiftTerm]:
    """BLISS-style terms that vanish on the ``N = n_electrons`` sector.

    ``(N - N_e)``, ``(N - N_e) N`` and, with ``one_body``, ``(N - N_e)`` times
    each spin-free ``E_pq + E_qp`` (``p <= q``). With ``sz`` the terms
    ``(S_z - sz)`` and ``(S_z - sz) N`` are added. Every multiplier conserves
    ``N`` (and ``S_z``), so each term is Hermitian.
    """
    count = _integer(n_spatial, "n_spatial", minimum=1)
    electrons = _integer(n_electrons, "n_electrons")
    if electrons > 2 * count:
        raise ValueError("n_electrons exceeds the spin orbitals")
    pairs = _spin_pairs(count, spin_ordering)
    n = 2 * count
    number = total_number_op(n)
    excess = number - _scalar(n, electrons)
    family = [ShiftTerm("N-Ne", excess, _scalar(n, 1.0)),
              ShiftTerm("(N-Ne)N", excess, number)]
    if one_body:
        for p in range(count):
            for q in range(p, count):
                transfer = sum((cdag_op(n, pairs[p][s]) * c_op(n, pairs[q][s]) for s in (0, 1)),
                               MV(n))
                family.append(ShiftTerm(f"(N-Ne)E{p}{q}", excess, transfer + transfer.dagger()))
    if sz is not None:
        if not math.isfinite(float(sz)):
            raise ValueError("sz must be finite")
        spin = total_sz_op(n, spin_ordering=spin_ordering) - _scalar(n, float(sz))
        family += [ShiftTerm("Sz-m", spin, _scalar(n, 1.0)),
                   ShiftTerm("(Sz-m)N", spin, number)]
    return family


def _expectation(rho: MV, code: int) -> float:
    value = rho.terms.get(code)
    return 0.0 if value is None else float((2 ** rho.n) * value.real)


def _largest(operator: MV) -> float:
    return max((abs(v) for v in operator.terms.values()), default=0.0)


def check_sector_family(family: Sequence[ShiftTerm], reference: MV,
                        generators: Iterable) -> tuple[float, float]:
    """``(sector_residual, commutator_residual)`` for a family on a bank.

    The first is the largest ``<D_k^2>`` on ``reference``; the second the
    largest coefficient of ``[D_k, O_k]`` or ``[D_k, A_j]``. Both vanish exactly
    when every term annihilates every bank state.
    """
    pool = [g.mv for g in as_generators(list(generators))]
    sector_residual = commutator_residual = 0.0
    seen: dict[tuple, bool] = {}
    for term in family:
        key = tuple(sorted(term.sector.terms.items()))
        if key not in seen:
            square = term.sector * term.sector
            sector_residual = max(sector_residual, abs(sum(
                value.real * _expectation(reference, code)
                for code, value in square.terms.items())))
            for a in pool:
                commutator_residual = max(commutator_residual,
                                          _largest(term.sector * a - a * term.sector))
            seen[key] = True
        commutator_residual = max(commutator_residual, _largest(
            term.sector * term.multiplier - term.multiplier * term.sector))
    return sector_residual, commutator_residual


def _real_vector(operator: MV, codes: Sequence[int]) -> np.ndarray:
    return np.array([operator.terms.get(code, 0j).real for code in codes])


def _covariance(codes: Sequence[int], reference: MV) -> np.ndarray:
    """Exact one-shot covariance of mutually qubit-wise-commuting words."""
    n = reference.n
    means = np.array([_expectation(reference, code) for code in codes])
    second = np.empty((len(codes), len(codes)))
    for a, x in enumerate(codes):
        for b, y in enumerate(codes):
            phase, product = word_mul(n, x, y)
            second[a, b] = (phase * _expectation(reference, product)).real
    return second - np.outer(means, means)


def _fit_l2(operator: MV, terms: Sequence[MV]) -> tuple[np.ndarray, float, float]:
    """Minimize the squared coefficients of ``H - sum xi_k K_k`` (identity excluded)."""
    codes = sorted({code for op in [operator, *terms] for code in op.terms if code != 0})
    y = _real_vector(operator, codes)
    design = np.column_stack([_real_vector(t, codes) for t in terms])
    xi, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ xi
    before, after = float(y @ y), float(residual @ residual)
    if after > before:  # least squares cannot lose to zero; guard round-off
        return np.zeros_like(xi), before, before
    return xi, before, after


def _fit_grouped(target: MV, basis: Sequence[MV], reference: MV, *,
                 iterations: int = 50) -> tuple[np.ndarray, float, float]:
    """Minimize the Neyman shot variance ``(sum_g sigma_g)^2`` under a fixed design.

    The design is the QWC partition of every word the target or any basis
    operator carries, so it does not move while ``xi`` does. Each ``sigma_g^2``
    is a convex quadratic in ``xi``; ``sum_g sigma_g`` is minimized by
    iteratively reweighted least squares with weights ``1 / sigma_g``.
    """
    n = reference.n
    codes = sorted({code for op in [target, *basis] for code in op.terms if code != 0})
    blocks = []
    for group in qwc_groups([PauliWord(n, code) for code in codes]):
        group_codes = [word.code for word in group]
        values, vectors = np.linalg.eigh(_covariance(group_codes, reference))
        root = vectors * np.sqrt(np.clip(values, 0.0, None))  # C = root @ root.T
        a = _real_vector(target, group_codes)
        v = np.column_stack([_real_vector(t, group_codes) for t in basis])
        blocks.append((root.T @ a, root.T @ v))

    def sigmas(xi: np.ndarray) -> np.ndarray:
        return np.array([float(np.linalg.norm(a - v @ xi)) for a, v in blocks])

    xi = np.zeros(len(basis))
    best = before = float(sigmas(xi).sum() ** 2)
    best_xi = xi
    for _ in range(iterations):
        weights = 1.0 / np.sqrt(np.maximum(sigmas(xi), 1e-12 * max(best, 1e-300) ** 0.5))
        rows = np.vstack([w * v for w, (_, v) in zip(weights, blocks)])
        rhs = np.concatenate([w * a for w, (a, _) in zip(weights, blocks)])
        xi, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
        value = float(sigmas(xi).sum() ** 2)
        if value < best * (1 - 1e-12):
            best, best_xi = value, xi
        else:
            break
    return best_xi, before, best


def fit_sector_shift(hamiltonian, family: Sequence[ShiftTerm], *, objective: str = "hamiltonian",
                     reference: MV | None = None, generators: Iterable | None = None,
                     ritz_coefficients: Sequence[complex] | None = None,
                     energy: float | None = None, tolerance: float = 1e-9) -> SectorShift:
    """Fit ``xi`` in ``H' = H - sum_k xi_k K_k`` to a declared objective.

    ``objective="hamiltonian"`` minimizes ``sum_w h'_w^2`` over the
    non-identity words of ``H'``: the state-free ``l2`` BLISS proxy.
    ``objective="ritz"`` minimizes the Neyman shot variance ``(sum_g
    sigma_g)^2`` of the Ritz functional ``f = B†(H' - E)B``, ``B = sum_i c_i
    A_i``, read from a fixed qubit-wise-commuting design, with ``sigma_g`` the
    exact one-shot standard deviation of setting ``g`` on ``reference``. That
    is the first-order variance of the measured Ritz value. Within-setting
    covariance is essential: measured as one observable, ``f`` and
    ``f - B†KB`` have the same variance, so a shift helps only through how its
    words split across settings, which a per-word objective cannot see. It
    needs ``ritz_coefficients`` and ``energy`` (exact, or from a pilot solve).
    The returned ``before``/``after`` are for that fixed design; regrouping
    the shifted words can change them, so evaluate with
    :func:`grouped_shot_variance`.

    ``reference`` and ``generators`` are required for either objective, because
    the shift is admitted only when :func:`check_sector_family` finds both
    residuals at most ``tolerance``.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if not family:
        raise ValueError("an empty family has nothing to fit")
    if reference is None or generators is None:
        raise ValueError("reference and generators are required to certify the shift")
    generators = list(generators)
    operator = (hamiltonian.to_mv() if isinstance(hamiltonian, PauliSum)
                else to_mv(hamiltonian))
    if not operator.is_hermitian(1e-10):
        raise ValueError("hamiltonian must be Hermitian")
    terms = [term.operator for term in family]
    if any(t.n != operator.n for t in terms) or reference.n != operator.n:
        raise ValueError("family, reference and hamiltonian disagree on n")
    sector_residual, commutator_residual = check_sector_family(family, reference, generators)
    if max(sector_residual, commutator_residual) > tolerance:
        raise ValueError(
            "a shift term does not annihilate every bank state: "
            f"<D^2> residual {sector_residual:.3g}, commutator residual "
            f"{commutator_residual:.3g} (tolerance {tolerance:g})")

    if objective == "hamiltonian":
        xi, before, after = _fit_l2(operator, terms)
    else:
        if ritz_coefficients is None or energy is None:
            raise ValueError("the ritz objective needs ritz_coefficients and energy")
        pool = [g.mv for g in as_generators(generators)]
        coefficients = np.asarray(ritz_coefficients, dtype=complex)
        if coefficients.shape != (len(pool),):
            raise ValueError("one Ritz coefficient per generator is required")
        b = sum((complex(c) * a for c, a in zip(coefficients, pool)), MV(operator.n))
        bd = b.dagger()
        target = bd * (operator - _scalar(operator.n, float(energy))) * b
        xi, before, after = _fit_grouped(target, [bd * t * b for t in terms], reference)
    shifted = operator - sum((float(x) * t for x, t in zip(xi, terms)), MV(operator.n))
    return SectorShift(hamiltonian=shifted,
                       coefficients={term.label: float(x) for term, x in zip(family, xi)},
                       objective=objective, before=before, after=after,
                       sector_residual=float(sector_residual),
                       commutator_residual=float(commutator_residual))


def grouped_shot_variance(coefficients: Mapping[int, float], reference: MV) -> dict[str, float]:
    """Exact per-shot variance of ``sum_w c_w <P_w>`` read from QWC settings.

    Words are partitioned into qubit-wise-commuting settings by
    :func:`~clifford_qc.measurement.grouping.qwc_groups`; each word is read only
    from its own setting. ``sigma_g`` is the exact standard deviation of one
    shot of setting ``g`` on ``reference``, covariances within it included.
    Returned: the word and setting counts, ``neyman`` = ``(sum_g sigma_g)^2``,
    the variance after one total shot at the optimal split, and ``uniform`` =
    ``G sum_g sigma_g^2``, the same at an equal split.
    """
    n = reference.n
    words = [PauliWord(n, code) for code in sorted(coefficients) if code != 0]
    if not words:
        return {"words": 0, "settings": 0, "neyman": 0.0, "uniform": 0.0}
    groups = qwc_groups(words)
    sigmas = []
    for group in groups:
        codes = [word.code for word in group]
        weights = np.array([float(coefficients[code]) for code in codes])
        sigmas.append(math.sqrt(max(0.0, float(weights @ _covariance(codes, reference)
                                                @ weights))))
    total = sum(sigmas)
    return {"words": len(words), "settings": len(groups), "neyman": total * total,
            "uniform": len(groups) * sum(s * s for s in sigmas)}


@dataclass(frozen=True)
class ControlVariateEstimate:
    """A Ritz value corrected by zero-mean sector terms read from the same cache.

    ``energy = raw_energy - sum_k coefficients[k] g_k`` where each ``g_k`` is
    the cache estimate of ``<B† K_k B>``, exactly zero in expectation for the
    solved ``B``. ``std_error`` is the delta-method standard error of the
    corrected functional. ``unreadable`` names terms with a word no measured
    setting can read; they are left out rather than guessed. With
    ``fitted=True`` the coefficients were estimated from this cache, which adds
    an ``O(1/N)`` bias and makes ``std_error`` slightly optimistic.
    """

    energy: float
    std_error: float
    raw_energy: float
    raw_std_error: float
    coefficients: dict[str, float]
    used: tuple[str, ...]
    unreadable: tuple[str, ...]
    fitted: bool


def _reference_density(reference) -> MV:
    if isinstance(reference, MV):
        return reference
    occupied = getattr(reference, "occupied", None)
    if occupied is not None:
        from ..states import ket_density
        filled = set(int(q) for q in occupied)
        return ket_density(reference.n, "".join("1" if q in filled else "0"
                                                for q in range(reference.n)))
    raise ValueError("certifying a shift needs an MV or determinant reference")


def certify_bank_family(bank, family: Sequence[ShiftTerm], *,
                        tolerance: float = 1e-9) -> tuple[float, float]:
    """Refuse a family that does not annihilate every state of ``bank``.

    Each ``D_k`` must vanish on the bank's reference and commute with its
    multiplier and with every generator, within ``tolerance``. Returns the two
    residuals of :func:`check_sector_family` when the family passes.
    """
    family = tuple(family)
    if not family:
        raise ValueError("a shift family must name at least one term")
    if not all(isinstance(term, ShiftTerm) for term in family):
        raise ValueError("a shift family is a sequence of ShiftTerm")
    generators = [bank.generator(i) for i in range(len(bank))]
    sector_residual, commutator_residual = check_sector_family(
        family, _reference_density(bank.reference), generators)
    if max(sector_residual, commutator_residual) > tolerance:
        raise ValueError(
            "a shift term does not annihilate every bank state: "
            f"<D^2> residual {sector_residual:.3g}, commutator residual "
            f"{commutator_residual:.3g} (tolerance {tolerance:g})")
    return sector_residual, commutator_residual


def ritz_control_variate(shared, cache, point, family: Sequence[ShiftTerm], *, root: int = 0,
                         coefficients: Mapping[str, float] | None = None,
                         tolerance: float = 1e-9) -> ControlVariateEstimate:
    """Correct a measured Ritz value with sector terms of known zero mean.

    No shot is spent and the bank is not rebuilt: every ``g_k = B† K_k B`` is
    a fixed linear combination of word means already in ``cache``, with
    ``B = sum_i c_i A_i`` from ``point``. Because ``K_k`` annihilates every
    bank state, ``<g_k> = 0`` exactly for any ``c``, so subtracting ``xi_k
    g_k`` changes the estimate's noise and not its mean -- the symmetry shift
    applied after the fact as a control variate.

    ``coefficients`` fixes ``xi`` (for instance from :func:`fit_sector_shift`
    on prior knowledge). Without it ``xi`` minimizes the corrected variance
    from this cache's grouped covariances, the classic control-variate fit.
    The family is certified on the bank's reference first
    (:func:`certify_bank_family`).
    """
    certify_bank_family(shared.bank, family, tolerance=tolerance)
    return corrected_ritz_functional(shared, cache, point, family, root=root,
                                     coefficients=coefficients)[0]


@dataclass(frozen=True)
class ShiftPairTable:
    """Every ``A_a† K_k A_b`` of one bank and family, as sparse word columns.

    ``B† K_k B = sum_ab conj(c_a) c_b A_a† K_k A_b`` for ``B = sum_a c_a A_a``,
    so once these products are expanded the correction terms of any Ritz
    vector are a sparse matrix-vector product, not three operator products.
    That is what makes refitting the correction on every bootstrap resample
    affordable. Built by :func:`shift_pair_table`; read by
    :func:`corrected_ritz_functional`.
    """

    indices: tuple[int, ...]
    labels: tuple[str, ...]
    # Per term: (codes, word_index, pair_index, values, constant_values), with
    # pairs ordered a * len(indices) + b over all ordered generator pairs.
    terms: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]

    def words(self, position: int, vector) -> tuple[dict[int, float], float]:
        """The real word coefficients and constant of ``B† K B`` for one term.

        A word is kept when its real coefficient exceeds the operator
        tolerance, the rule the operator-product path applies, so a word whose
        real part is rounding is dropped by both.
        """
        codes, word_index, pair_index, values, constants = self.terms[position]
        c = np.asarray(vector, dtype=complex)
        weights = np.outer(np.conjugate(c), c).ravel()
        weighted = values * weights[pair_index]
        real = np.bincount(word_index, weighted.real, minlength=len(codes))
        keep = np.abs(real) > TOL
        constant = complex(constants @ weights)
        return ({int(code): float(value) for code, value in zip(codes[keep], real[keep])},
                constant.real if abs(constant) > TOL else 0.0)


def shift_pair_table(bank, indices: Sequence[int],
                     family: Sequence[ShiftTerm]) -> ShiftPairTable:
    """Expand ``A_a† K_k A_b`` for every ordered generator pair and family term.

    Its cost is ``len(family) * len(indices) ** 2`` operator products, paid
    once for a bank, after which :meth:`ShiftPairTable.words` builds each
    correction term for any Ritz vector. The caller certifies the family.
    """
    indices = tuple(int(i) for i in indices)
    generators = [bank.generator(i).mv for i in indices]
    daggers = [g.dagger() for g in generators]
    width = len(indices)
    terms = []
    for term in family:
        operator = term.operator
        right = [operator * g for g in generators]
        positions: dict[int, int] = {}
        word_index: list[int] = []
        pair_index: list[int] = []
        values: list[complex] = []
        constants = np.zeros(width * width, dtype=complex)
        for a in range(width):
            for b in range(width):
                product = daggers[a] * right[b]
                pair = a * width + b
                for code, value in product.terms.items():
                    if code == 0:
                        constants[pair] = value
                        continue
                    word_index.append(positions.setdefault(code, len(positions)))
                    pair_index.append(pair)
                    values.append(value)
        codes = np.fromiter(positions, dtype=np.int64, count=len(positions))
        terms.append((codes, np.asarray(word_index, dtype=np.int64),
                       np.asarray(pair_index, dtype=np.int64),
                       np.asarray(values, dtype=complex), constants))
    return ShiftPairTable(indices=indices, labels=tuple(term.label for term in family),
                          terms=tuple(terms))


def corrected_ritz_functional(shared, cache, point, family: Sequence[ShiftTerm], *,
                              root: int = 0, coefficients: Mapping[str, float] | None = None,
                              table: ShiftPairTable | None = None):
    """:func:`ritz_control_variate` without certification, plus its functional.

    Returns ``(estimate, functional)``. ``functional`` is the corrected Ritz
    functional ``f - sum_k xi_k g_k`` at the chosen ``xi``: a word functional
    whose variance is ``estimate.std_error ** 2``, so an allocation rule can
    target the corrected estimate directly. The caller must already have
    certified ``family`` on this bank (:func:`certify_bank_family`); nothing
    here checks that ``<g_k> = 0``.

    ``table`` (:func:`shift_pair_table` for this bank, ``point.indices`` and
    ``family``) builds each correction term by a sparse product instead of
    operator products; the terms agree to rounding.
    """
    from ..measurement.functionals import WordFunctional, covariance_matrix, ritz_functional

    bank = shared.bank
    vector = point.ritz_vector(root)
    energy = float(point.energies[root])
    target = ritz_functional(bank, point.indices, vector, energy)
    def readable(codes) -> bool:
        # The test candidate_group_statistics applies, without its moments.
        return all(cache.group_weights(code) for code in codes)

    if not target.is_deterministic and not readable(target.coefficients):
        raise ValueError("the cache does not read every word of the Ritz functional; "
                         "solve the point from the same cache")
    if table is not None:
        if (table.indices != tuple(int(i) for i in point.indices)
                or table.labels != tuple(term.label for term in family)):
            raise ValueError("the pair table was built for another bank order or family")
        expansions = (table.words(position, vector) for position in range(len(family)))
    else:
        b = sum((complex(c) * bank.generator(i).mv for c, i in zip(vector, point.indices)),
                MV(bank.n))
        bd = b.dagger()

        def expand(term):
            operator = bd * term.operator * b
            # Only the real part enters <B† K B>; a word whose real part is
            # rounding (the imaginary part of a non-Hermitian term) is not read.
            return ({code: value.real for code, value in operator.terms.items()
                     if code != 0 and abs(value.real) > TOL},
                    operator.terms.get(0, 0j).real)
        expansions = (expand(term) for term in family)
    used, unreadable, functionals = [], [], []
    for term, (words, constant) in zip(family, expansions):
        if not words:
            continue
        if not readable(words):
            unreadable.append(term.label)
            continue
        functionals.append(WordFunctional(words, constant))
        used.append(term.label)
    if not functionals:
        raw_sigma = math.sqrt(max(0.0, target.variance(cache)))
        return (ControlVariateEstimate(energy, raw_sigma, energy, raw_sigma, {}, (),
                                       tuple(unreadable), coefficients is None), target)
    # One pass over the histograms: index 0 is the target, then each g_k.
    joint = covariance_matrix([target, *functionals], cache)
    target_variance, gf, gg = joint[0, 0], joint[1:, 0], joint[1:, 1:]
    raw_sigma = math.sqrt(max(0.0, float(target_variance)))
    if coefficients is None:
        xi, *_ = np.linalg.lstsq(gg, gf, rcond=None)
    else:
        xi = np.array([float(coefficients.get(label, 0.0)) for label in used])
    variance = target_variance - 2.0 * float(xi @ gf) + float(xi @ gg @ xi)
    correction = float(sum(x * g.estimate(cache) for x, g in zip(xi, functionals)))
    combined = dict(target.coefficients)
    constant = float(target.constant)
    for x, g in zip(xi, functionals):
        constant -= float(x) * g.constant
        for code, value in g.coefficients.items():
            combined[code] = combined.get(code, 0.0) - float(x) * value
    estimate = ControlVariateEstimate(
        energy=energy - correction, std_error=math.sqrt(max(0.0, variance)),
        raw_energy=energy, raw_std_error=raw_sigma,
        coefficients={label: float(x) for label, x in zip(used, xi)},
        used=tuple(used), unreadable=tuple(unreadable), fitted=coefficients is None)
    return estimate, WordFunctional({code: value for code, value in combined.items()
                                     if value != 0.0}, constant)


def ritz_word_functional(hamiltonian, generators: Iterable, ritz_coefficients: Sequence[complex],
                         energy: float) -> dict[int, float]:
    """Non-identity word coefficients of ``B†(H - E)B``, ``B = sum_i c_i A_i``."""
    operator = (hamiltonian.to_mv() if isinstance(hamiltonian, PauliSum)
                else to_mv(hamiltonian))
    pool = [g.mv for g in as_generators(list(generators))]
    b = sum((complex(c) * a for c, a in zip(ritz_coefficients, pool)), MV(operator.n))
    functional = b.dagger() * (operator - _scalar(operator.n, float(energy))) * b
    return {code: value.real for code, value in functional.terms.items() if code != 0}


__all__ = ["OBJECTIVES", "ControlVariateEstimate", "SectorShift", "ShiftPairTable", "ShiftTerm",
           "certify_bank_family", "check_sector_family", "corrected_ritz_functional",
           "fit_sector_shift", "grouped_shot_variance", "ritz_control_variate",
           "ritz_word_functional", "sector_shift_family", "shift_pair_table"]
