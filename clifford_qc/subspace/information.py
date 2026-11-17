"""Resource- and information-aware adaptive selection and closed-loop acquisition.

Implements the generalized selection principle

    A* = argmax_A  E[dE_A] / (C_incremental(A) + eps)

and extends fixed-budget, single-criterion selection into a closed-loop
information-acquisition controller targeting a declared physical accuracy
``target_accuracy`` without oracle ground-energy leakage:

1. **Composable Incremental Cost (C_incremental)**:
   - ``new_words`` (``dW``): Pauli observables not in the current basis universe.
   - ``incremental_groups`` (``dG``): truly new QWC measurement circuits beyond
     first-fit placement into existing QWC bases.
   - ``incremental_time_us`` (``dT``): hardware wall-clock circuit execution cost
     priced through a declared ``DeviceCard`` (optional ``F^-2`` fidelity
     inflation) plus existing-circuit basis widening deltas.
   - ``classical_products`` & ``classical_terms``: sparse Pauli pair-product
     multiplication work (``H A_c`` and ``2|B|+1`` row pairings) and newly
     materialized Pauli coefficient occurrences.

2. **Expected Improvement Numerator (E[dE_A])**:
   - In exact mode, uses the generalized 2x2 Ritz lowering ``_two_by_two_lowering``.
   - In finite-shot mode from ``GroupedWordCache``, computes covariance-aware
     delta-method standard errors for both ``dE_A`` (7D joint covariance over
     ``s_a, h_a, s_aa, h_aa, E_m``, including the response of the measured Ritz
     vectors) and the full-subspace orthogonal fraction
     ``1 - sum_k |<Psi_k|chi_hat>|^2``, rejecting shot-ambiguous near-dependent
     directions before nonlinear 2x2 division amplifies noise, and computing
     risk-adjusted or shrunk expected improvement.

3. **Closed-Loop Information Acquisition (``acquire_to_accuracy``)**:
   - Maintains cumulative ``GroupedWordCache`` histograms with zero discarded
     shots across basis expansions and incremental QWC circuit registration.
   - Balances **subspace expansion** (adding ``A*`` to reduce approximation bias)
     against **covariance-aware Neyman shot refinement** (routing shots to high-
     sensitivity QWC groups of ``ritz_functional`` weighted by ``DeviceCard``
     group timings) by comparing marginal RMSE reduction per unit cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
import time
from typing import Any, Iterable, Mapping, Sequence, cast
import weakref

import numpy as np

from ..ir import PauliWord
from ..measurement.allocation import variance_optimal_group_plan
from ..measurement.cache import GroupedWordCache
from ..measurement.cost import (
    DeviceCard,
    SettingResources,
    setting_duration_us,
    setting_fidelity,
)
from ..measurement.functionals import (
    WordFunctional,
    entry_functionals,
    ritz_functional,
)
from ..measurement.grouping import _nonident_lanes, _qwc_codes, qwc_groups, shared_basis
from ..measurement.session import SharedMeasurement, grouped_cache_fingerprint
from ..multivector import MV
from ..selection import EvidenceLevel, TIE_ATOL, TIE_RTOL, canonical_argmax
from .adaptive import (
    DEFAULT_MAX_CONDITION,
    DEFAULT_MIN_LOWERING,
    DEFAULT_MIN_ORTHOGONALITY,
    DEFAULT_TAU_S,
    _GAP_FLOOR,
    ACASEConfig,
    ACASEState,
    ACASEStepOutcome,
    AdaptiveResult,
    GrowthRecord,
    _acase_objective,
    _aggregate,
    _candidate_sector_leakage,
    _require_matching_bank,
    _two_by_two_lowering,
    acase_step,
)
from .elements import MatrixElementBank
from .generator_core import as_generators
from .generators import identity_generator
from .linalg import DEFAULT_NORM_FLOOR, SubspaceResult, solve_projected
from .symmetry import subspace_sector_certificate
from .uncertainty import Interval, _point_estimator, ritz_uncertainty

_EI_MODES = ("point", "risk_adjusted", "shrinkage", "snr_gated")
_STOP_METRICS = ("std_error", "ci_halfwidth")


def _non_identity_words(n: int, codes: Iterable[int]) -> list[PauliWord]:
    return [PauliWord(n, code) for code in sorted(set(codes)) if code != 0]


def basis_dict_to_code(basis: Mapping[int, str]) -> int:
    """Packed Pauli code of a ``{qubit: letter}`` setting, as the cache computes it."""
    return GroupedWordCache._basis_code({int(qubit): letter for qubit, letter in basis.items()})


def qwc_bases_for_words(n: int, word_codes: Iterable[int]) -> tuple[int, ...]:
    """Build merged packed QWC basis codes for ``word_codes`` (excluding identity)."""
    words = _non_identity_words(n, word_codes)
    if not words:
        return ()
    groups = qwc_groups(words)
    bases: list[int] = []
    for group in groups:
        merged = 0
        for word in group:
            merged |= word.code
        bases.append(merged)
    return tuple(bases)


def qwc_setting_resources(n: int, basis_code: int) -> SettingResources:
    """Clifford measurement setting gate/depth counts for a QWC basis code."""
    n_x = 0
    n_y = 0
    for qubit in range(n):
        letter = (basis_code >> (2 * qubit)) & 3
        if letter == 1:
            n_x += 1
        elif letter == 2:
            n_y += 1
    n_1q = n_x + 2 * n_y
    d_1q = 2 if n_y > 0 else (1 if n_x > 0 else 0)
    return SettingResources(n_1q=n_1q, n_2q=0, d_1q=d_1q, d_2q=0)


def _setting_effective_duration_us(
    card: DeviceCard,
    n: int,
    basis_code: int,
    *,
    fidelity_inflate: bool = True,
) -> float:
    res = qwc_setting_resources(n, basis_code)
    dur = setting_duration_us(card, res)
    if not fidelity_inflate:
        return dur
    fid = setting_fidelity(card, res, n)
    return dur / max(fid * fid, 1e-12)


@dataclass(frozen=True)
class IncrementalCostModel:
    """Composable denominator ``C_incremental(A) + epsilon`` and risk controls."""

    word_weight: float = 0.0
    setting_weight: float = 0.0
    time_weight_us: float = 0.0
    classical_product_weight: float = 0.0
    classical_term_weight: float = 0.0
    epsilon: float = 1e-6
    device_card: DeviceCard | None = None
    shots_per_setting: float = 1.0
    fidelity_inflate: bool = False
    risk_aversion: float = 0.0
    expected_improvement_mode: str = "risk_adjusted"
    require_stat_orthogonality: bool = True
    orthogonality_z: float = 2.0

    def __post_init__(self) -> None:
        for field_name in (
            "word_weight",
            "setting_weight",
            "time_weight_us",
            "classical_product_weight",
            "classical_term_weight",
            "shots_per_setting",
            "risk_aversion",
            "orthogonality_z",
        ):
            val = float(getattr(self, field_name))
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive")
        if self.expected_improvement_mode not in _EI_MODES:
            raise ValueError(
                f"expected_improvement_mode must be one of {_EI_MODES}, "
                f"got {self.expected_improvement_mode!r}"
            )
        if self.device_card is not None and not isinstance(self.device_card, DeviceCard):
            raise TypeError("device_card must be a DeviceCard or None")


@dataclass(frozen=True)
class IncrementalCostBreakdown:
    """Decomposed resource deltas and scalar cost for one candidate."""

    new_words: int
    incremental_groups: int
    widened_groups: int
    incremental_time_us: float
    classical_products: int
    classical_terms: int
    total_cost: float


def estimate_incremental_cost(
    bank: MatrixElementBank,
    basis: Sequence[int],
    candidate: int,
    *,
    cost_model: IncrementalCostModel,
    basis_words: frozenset[int] | None = None,
    qwc_bases: tuple[int, ...] | None = None,
) -> IncrementalCostBreakdown:
    """Price newly revealed words, QWC settings, hardware time, and classical work."""
    n = bank.n
    if basis_words is None:
        basis_words = bank.word_set(basis)

    row_words: set[int] = set()
    delta_terms = 0
    cand_supp = bank.generator(candidate).support()
    # Read cached H * A_c support if materialized; fall back safely if not.
    h_acted = getattr(bank, "_h_acted", None)
    ha_supp = h_acted[candidate].nnz() if h_acted is not None else bank.hamiltonian.nnz() * cand_supp
    classical_products = bank.hamiltonian.nnz() * cand_supp

    for idx in tuple(basis) + (candidate,):
        bank.update_pair_support(row_words, idx, candidate)
        idx_supp = bank.generator(idx).support()
        classical_products += idx_supp * (cand_supp + ha_supp)
        key = bank._canonical_pair(idx, candidate)
        if hasattr(bank, "_row_nnz"):
            delta_terms += bank._row_nnz("overlap", key) + bank._row_nnz("element", key)
        else:
            delta_terms += sum(1 for _ in bank.iter_operator_terms("overlap", idx, candidate))
            delta_terms += sum(1 for _ in bank.iter_operator_terms("element", idx, candidate))

    new_codes = sorted(code for code in (row_words - basis_words) if code != 0)
    new_words = len(row_words - basis_words)

    delta_groups = 0
    widened_groups = 0
    delta_time_us = 0.0
    need_qwc = (
        cost_model.setting_weight > 0.0
        or cost_model.time_weight_us > 0.0
    )
    if need_qwc and new_codes:
        if qwc_bases is None:
            qwc_bases = qwc_bases_for_words(n, basis_words)
        working = list(qwc_bases)
        n_existing = len(working)
        for code in new_codes:
            # A setting that already reads the word measures it for free. Only
            # otherwise widen the first compatible setting, or open a new one.
            if any(_qwc_reads_word(n, code, g_basis) for g_basis in working):
                continue
            for g, g_basis in enumerate(working):
                if _qwc_codes(n, code, g_basis):
                    working[g] = g_basis | code
                    break
            else:
                working.append(code)

        delta_groups = len(working) - n_existing
        widened_groups = sum(
            1 for g in range(n_existing) if working[g] != qwc_bases[g]
        )

        if cost_model.time_weight_us > 0.0:
            card = cost_model.device_card
            if card is None:
                delta_time_us = float(delta_groups) * cost_model.shots_per_setting
            else:
                t_sum = 0.0
                for g in range(n_existing):
                    if working[g] != qwc_bases[g]:
                        before = _setting_effective_duration_us(
                            card, n, qwc_bases[g], fidelity_inflate=cost_model.fidelity_inflate
                        )
                        after = _setting_effective_duration_us(
                            card, n, working[g], fidelity_inflate=cost_model.fidelity_inflate
                        )
                        t_sum += max(0.0, after - before)
                for g in range(n_existing, len(working)):
                    t_sum += _setting_effective_duration_us(
                        card, n, working[g], fidelity_inflate=cost_model.fidelity_inflate
                    )
                delta_time_us = t_sum * cost_model.shots_per_setting

    total_cost = (
        cost_model.word_weight * float(new_words)
        + cost_model.setting_weight * float(delta_groups)
        + cost_model.time_weight_us * float(delta_time_us)
        + cost_model.classical_product_weight * float(classical_products)
        + cost_model.classical_term_weight * float(delta_terms)
    )
    return IncrementalCostBreakdown(
        new_words=new_words,
        incremental_groups=delta_groups,
        widened_groups=widened_groups,
        incremental_time_us=delta_time_us,
        classical_products=classical_products,
        classical_terms=delta_terms,
        total_cost=total_cost,
    )


def _linear_combination_functionals(
    bank: MatrixElementBank,
    basis: Sequence[int],
    coeff_vec: np.ndarray,
    candidate: int,
    *,
    kind_idx: int,
    column: Sequence | None = None,
) -> tuple[WordFunctional, WordFunctional]:
    """Assemble real and imag linear functionals for ``sum_i c_i^* O_{i, candidate}``.

    ``column[a]`` may supply ``entry_functionals(bank, basis[a], candidate)``
    when the caller already built it; it is the same value.
    """
    re_terms: dict[int, float] = {}
    im_terms: dict[int, float] = {}
    re_const = 0.0
    im_const = 0.0

    for a, i in enumerate(basis):
        c = complex(coeff_vec[a])
        if c == 0j:
            continue
        cr, ci = c.real, c.imag
        entry = column[a] if column is not None else entry_functionals(bank, i, candidate)
        f_re, f_im = entry[kind_idx]
        # (cr - i*ci) * (x + i*y) = (cr*x + ci*y) + i*(cr*y - ci*x)
        if cr != 0.0:
            re_const += cr * f_re.constant
            im_const += cr * f_im.constant
            for code, v in f_re.coefficients.items():
                re_terms[code] = re_terms.get(code, 0.0) + cr * v
            for code, v in f_im.coefficients.items():
                im_terms[code] = im_terms.get(code, 0.0) + cr * v
        if ci != 0.0:
            re_const += ci * f_im.constant
            im_const -= ci * f_re.constant
            for code, v in f_im.coefficients.items():
                re_terms[code] = re_terms.get(code, 0.0) + ci * v
            for code, v in f_re.coefficients.items():
                im_terms[code] = im_terms.get(code, 0.0) - ci * v

    re_clean = {k: v for k, v in re_terms.items() if v != 0.0}
    im_clean = {k: v for k, v in im_terms.items() if v != 0.0}
    return WordFunctional(re_clean, re_const), WordFunctional(im_clean, im_const)


def _all_measured(functionals: Sequence[WordFunctional], cache: GroupedWordCache) -> bool:
    for func in functionals:
        if func.is_deterministic:
            continue
        for code in func.coefficients:
            if not cache.group_weights(code):
                return False
    return True


def _combine_linear(
    weights: Sequence[float] | np.ndarray,
    functionals: Sequence[WordFunctional],
) -> WordFunctional:
    """Return ``sum_j w_j f_j`` as a single sparse ``WordFunctional``."""
    terms: dict[int, float] = {}
    const = 0.0
    for w_raw, func in zip(weights, functionals):
        w = float(w_raw)
        if w == 0.0:
            continue
        const += w * func.constant
        if not func.is_deterministic:
            for code, val in func.coefficients.items():
                terms[code] = terms.get(code, 0.0) + w * val
    return WordFunctional({c: v for c, v in terms.items() if v != 0.0}, const)


@dataclass(frozen=True)
class _BasisResponse:
    """Response to independent real coordinates of the measured basis pencil.

    The coordinates are composed with their Pauli functionals afterwards, so
    correlations between repeated entries and measurement groups are retained.
    One response is shared by all candidates scored against the same solve.
    """

    functionals: tuple[WordFunctional, ...]
    coefficients: np.ndarray
    energies: np.ndarray
    projector: np.ndarray
    valid_roots: tuple[bool, ...]
    stable_rank: bool


def _basis_response(bank: MatrixElementBank, basis: Sequence[int],
                    result: SubspaceResult, cache: GroupedWordCache) -> _BasisResponse:
    """Differentiate the measured solve, including normalization and truncation.

    Central differences act on the small Hermitian pencil, not on individual
    Pauli means. They rerun its recorded threshold policy while holding the
    discrete retention decision locally fixed. A rank change or an unresolved
    eigenvalue degeneracy makes the corresponding uncertainty unresolved.
    """
    session = SharedMeasurement(bank, basis, pooling=cache.pooling)
    policy = _point_estimator(session, cache, result)
    if (policy.get("overlap_repair", "none") != "none"
            or policy.get("pencil_angle") is not None
            or policy.get("ratio_error_budget") is not None
            or result.resources.get("overlap_ridge", 0.0) != 0.0):
        raise ValueError("lowering uncertainty requires an unrotated, thresholded pencil; "
                         "use a full-estimator bootstrap for repair, rotation or ridge")
    S, H = session.matrices(cache)
    m, rank = result.coefficients.shape
    C = result.coefficients
    energy = np.asarray(result.energies)
    scales = np.sqrt(np.maximum(np.abs(np.diag(S)), 1e-30))
    h_scale = max(float(np.linalg.norm(H / scales[:, None] / scales[None, :], 2)),
                  float(np.max(np.abs(energy))), 1e-30)
    gaps = np.abs(energy[:, None] - energy[None, :])
    np.fill_diagonal(gaps, np.inf)
    nearest = np.min(gaps, axis=1)
    valid = nearest > 256.0 * np.finfo(float).eps * h_scale
    resolved = nearest[valid & np.isfinite(nearest)]
    relative_step = min(1e-6, 1e-3 * float(np.min(resolved)) / h_scale) \
        if resolved.size else 1e-6
    noise_floor = result.resources.get("overlap_noise_floor_per_mode")
    if noise_floor is None:
        noise_floor = result.resources.get("overlap_noise_floor", 0.0)
    kwargs = {name: policy[name] for name in ("tau_s", "rel_tau", "max_condition", "norm_floor")}
    functionals = []
    d_coefficients = []
    d_energies = []
    d_projector = []
    stable_rank = True
    for a, i in enumerate(basis):
        for b in range(a, m):
            entries = entry_functionals(bank, i, basis[b])
            for kind, matrix in enumerate((S, H)):
                for imaginary in range(1 if a == b else 2):
                    functional = entries[kind][imaginary]
                    if functional.is_deterministic:
                        continue
                    direction = np.zeros_like(matrix)
                    direction[a, b] = 1j if imaginary else 1.0
                    if a != b:
                        direction[b, a] = direction[a, b].conjugate()
                    scale = scales[a] * scales[b] * (h_scale if kind else 1.0)
                    step = relative_step * max(scale, abs(matrix[a, b]), 1e-30)
                    perturbed = []
                    for _ in range(8):
                        try:
                            perturbed = [solve_projected(
                                S + (sign * step * direction if kind == 0 else 0),
                                H + (sign * step * direction if kind == 1 else 0),
                                overlap_noise_floor=noise_floor, **kwargs)
                                for sign in (-1, 1)]
                        except (ValueError, np.linalg.LinAlgError):
                            perturbed = []
                        if perturbed and all(p.effective_rank == rank for p in perturbed):
                            break
                        step *= 0.25
                    else:
                        perturbed = []
                    functionals.append(functional)
                    if not perturbed:
                        stable_rank = False
                        d_coefficients.append(np.zeros_like(C))
                        d_energies.append(np.zeros(rank))
                        d_projector.append(np.zeros((m, m), dtype=complex))
                        continue
                    minus, plus = perturbed
                    aligned = []
                    for point in perturbed:
                        overlap = np.diag(C.conj().T @ S @ point.coefficients)
                        valid &= np.abs(overlap) > 0.5
                        phase = np.ones(rank, dtype=complex)
                        np.divide(overlap.conjugate(), np.abs(overlap), out=phase,
                                  where=np.abs(overlap) > 0.0)
                        aligned.append(point.coefficients * phase[None, :])
                    d_coefficients.append((aligned[1] - aligned[0]) / (2.0 * step))
                    d_energies.append((np.asarray(plus.energies) - minus.energies) / (2.0 * step))
                    d_projector.append((plus.coefficients @ plus.coefficients.conj().T
                                        - minus.coefficients @ minus.coefficients.conj().T)
                                       / (2.0 * step))
    return _BasisResponse(
        tuple(functionals),
        np.asarray(d_coefficients).reshape(-1, m, rank),
        np.asarray(d_energies).reshape(-1, rank),
        np.asarray(d_projector).reshape(-1, m, m),
        tuple(bool(value) for value in valid), stable_rank)


# The last few basis responses, so the public scoring functions do not
# redifferentiate one solve for every candidate and root. Entries hold weak
# references: a cached response never keeps a bank or a solve alive.
_RESPONSE_MEMO: list[tuple[Any, Any, tuple[int, ...], str, str, _BasisResponse]] = []
_RESPONSE_MEMO_SIZE = 8


def _cached_basis_response(bank: MatrixElementBank, basis: Sequence[int],
                           result: SubspaceResult, cache: GroupedWordCache) -> _BasisResponse:
    """:func:`_basis_response`, remembered per (bank, basis, solve, joint outcomes).

    A hit needs the same bank and solve objects, the same basis and pooling,
    and a cache whose outcomes have not changed since; anything else is
    recomputed, so a stale response is never returned.
    """
    key_basis = tuple(int(index) for index in basis)
    fingerprint = grouped_cache_fingerprint(cache)
    for result_ref, bank_ref, entry_basis, pooling, entry_print, response in _RESPONSE_MEMO:
        if (result_ref() is result and bank_ref() is bank and entry_basis == key_basis
                and pooling == cache.pooling and entry_print == fingerprint):
            return response
    response = _basis_response(bank, basis, result, cache)
    _RESPONSE_MEMO.insert(0, (weakref.ref(result), weakref.ref(bank), key_basis,
                              cache.pooling, fingerprint, response))
    del _RESPONSE_MEMO[_RESPONSE_MEMO_SIZE:]
    return response


def _expected_lowering_details(
    bank: MatrixElementBank,
    basis: Sequence[int],
    result: SubspaceResult,
    candidate: int,
    *,
    root: int = 0,
    cache: GroupedWordCache | None = None,
    basis_response: _BasisResponse | None = None,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    WordFunctional | None,
    WordFunctional | None,
]:
    coeffs_all = result.coefficients
    c_r = coeffs_all[:, root]
    energy_r = float(result.energies[root])

    if cache is None:
        s_aa_raw, h_aa_raw = bank.entry(candidate, candidate)
        s_aa = float(s_aa_raw.real)
        h_aa = float(h_aa_raw.real)
        if s_aa <= DEFAULT_NORM_FLOOR ** 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None
        col = np.array([bank.entry(i, candidate) for i in basis], dtype=complex)
        s_col, h_col = col[:, 0], col[:, 1]
        projs = coeffs_all.conj().T @ s_col
        s_a = complex(projs[root])
        h_a = complex(c_r.conj() @ h_col)
        orth = max(0.0, 1.0 - float(np.sum(np.abs(projs) ** 2)) / s_aa)
        coupling = abs(h_a - energy_r * s_a) / math.sqrt(s_aa)
        lowering = _two_by_two_lowering(energy_r, s_a, h_a, s_aa, h_aa)
        return lowering, 0.0, orth, 0.0, coupling, 0.0, None, None

    # The Ritz vectors were estimated from this cache too. Their response is
    # first order for candidate couplings, unlike the stationary Ritz energy.
    response = basis_response if basis_response is not None else _cached_basis_response(
        bank, basis, result, cache)
    # Finite-shot covariance-aware estimation
    (s_aa_f, _), (h_aa_f, _) = entry_functionals(bank, candidate, candidate)
    s_aa = float(s_aa_f.estimate(cache))
    h_aa = float(h_aa_f.estimate(cache))
    if s_aa <= DEFAULT_NORM_FLOOR ** 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None

    # The candidate's column of (S, H) functionals, built once and shared by
    # every projection below.
    column = [entry_functionals(bank, i, candidate) for i in basis]

    # Projections against every retained eigenvector
    k_ret = coeffs_all.shape[1]
    proj_funcs: list[WordFunctional] = []
    proj_vals: list[tuple[float, float]] = []
    for k in range(k_ret):
        pr_f, pi_f = _linear_combination_functionals(
            bank, basis, coeffs_all[:, k], candidate, kind_idx=0, column=column
        )
        proj_funcs.extend((pr_f, pi_f))
        proj_vals.append((float(pr_f.estimate(cache)), float(pi_f.estimate(cache))))

    s_a_re_f, s_a_im_f = proj_funcs[2 * root], proj_funcs[2 * root + 1]
    s_a = complex(*proj_vals[root])
    h_a_re_f, h_a_im_f = _linear_combination_functionals(
        bank, basis, c_r, candidate, kind_idx=1, column=column
    )
    h_a_re = float(h_a_re_f.estimate(cache))
    h_a_im = float(h_a_im_f.estimate(cache))
    h_a = complex(h_a_re, h_a_im)

    s_col = np.array([complex(*[part.estimate(cache) for part in entry[0]])
                      for entry in column])
    h_col = np.array([complex(*[part.estimate(cache) for part in entry[1]])
                      for entry in column])
    ds = np.einsum("ni,i->n", response.coefficients[:, :, root].conj(), s_col)
    dh = np.einsum("ni,i->n", response.coefficients[:, :, root].conj(), h_col)
    s_a_re_f += _combine_linear(ds.real, response.functionals)
    s_a_im_f += _combine_linear(ds.imag, response.functionals)
    h_a_re_f += _combine_linear(dh.real, response.functionals)
    h_a_im_f += _combine_linear(dh.imag, response.functionals)

    proj_sq = sum(u * u + v * v for u, v in proj_vals)
    orth = max(0.0, 1.0 - proj_sq / s_aa)
    coupling = abs(h_a - energy_r * s_a) / math.sqrt(s_aa)

    # Analytical delta-method variance of orthogonal fraction via linearized functional
    orth_funcs = [s_aa_f, *proj_funcs]
    grad_orth = np.zeros(len(orth_funcs), dtype=float)
    grad_orth[0] = proj_sq / (s_aa * s_aa)
    for k, (uk, vk) in enumerate(proj_vals):
        grad_orth[1 + 2 * k] = -2.0 * uk / s_aa
        grad_orth[1 + 2 * k + 1] = -2.0 * vk / s_aa
    orth_f = _combine_linear(grad_orth, orth_funcs)
    dq = np.einsum("i,nij,j->n", s_col.conj(), response.projector, s_col).real
    orth_f += _combine_linear(-dq / s_aa, response.functionals)
    if response.stable_rank and _all_measured([orth_f], cache):
        orth_std = math.sqrt(max(0.0, orth_f.variance(cache)))
    else:
        orth_std = float("inf")

    lowering = _two_by_two_lowering(energy_r, s_a, h_a, s_aa, h_aa)

    # 7D delta-method variance of 2x2 generalized lowering (s_a_r, s_a_i, h_a_r, h_a_i, s_aa, h_aa, E_r)
    e_r_f = _combine_linear(response.energies[:, root], response.functionals)
    low_funcs = [s_a_re_f, s_a_im_f, h_a_re_f, h_a_im_f, s_aa_f, h_aa_f, e_r_f]
    theta = np.array(
        [s_a.real, s_a.imag, h_a.real, h_a.imag, s_aa, h_aa, energy_r],
        dtype=float,
    )

    def _eval_low(t: np.ndarray) -> float:
        return _two_by_two_lowering(
            float(t[6]),
            complex(float(t[0]), float(t[1])),
            complex(float(t[2]), float(t[3])),
            float(t[4]),
            float(t[5]),
        )

    grad_low = np.zeros(7, dtype=float)
    gap_now = 1.0 - (abs(s_a) ** 2) / s_aa
    root_s_aa = math.sqrt(s_aa)
    if gap_now > _GAP_FLOOR:
        scales = (
            max(root_s_aa, abs(float(theta[0]))),
            max(root_s_aa, abs(float(theta[1]))),
            max(root_s_aa, abs(float(theta[2]))),
            max(root_s_aa, abs(float(theta[3]))),
            s_aa,
            max(s_aa, abs(float(theta[5]))),
            max(1e-6, abs(float(theta[6]))),
        )
        for j in range(7):
            h_step = 1e-6 * scales[j]
            if h_step <= 0.0:
                continue
            t_plus = theta.copy()
            t_minus = theta.copy()
            t_plus[j] += h_step
            t_minus[j] -= h_step
            grad_low[j] = (_eval_low(t_plus) - _eval_low(t_minus)) / (2.0 * h_step)
    low_f = _combine_linear(grad_low, low_funcs)

    dr = h_a.real - energy_r * s_a.real
    di = h_a.imag - energy_r * s_a.imag
    dabs = math.hypot(dr, di)
    grad_coup = np.zeros(7, dtype=float)
    if dabs > 0.0 and root_s_aa > 0.0:
        inv = 1.0 / (dabs * root_s_aa)
        grad_coup[0] = -energy_r * dr * inv
        grad_coup[1] = -energy_r * di * inv
        grad_coup[2] = dr * inv
        grad_coup[3] = di * inv
        grad_coup[4] = -0.5 * coupling / s_aa
        grad_coup[6] = -(dr * s_a.real + di * s_a.imag) * inv
    coup_f = _combine_linear(grad_coup, low_funcs)

    if (response.stable_rank and response.valid_roots[root]
            and _all_measured(low_funcs, cache)):
        low_std = math.sqrt(max(0.0, low_f.variance(cache)))
        coup_std = math.sqrt(max(0.0, coup_f.variance(cache)))
    else:
        low_std = float("inf")
        coup_std = float("inf")

    return lowering, low_std, orth, orth_std, coupling, coup_std, orth_f, low_f


def expected_lowering_estimate(
    bank: MatrixElementBank,
    basis: Sequence[int],
    result: SubspaceResult,
    candidate: int,
    *,
    root: int = 0,
    cache: GroupedWordCache | None = None,
) -> tuple[float, float, float, float, float]:
    """Compute ``(lowering, lowering_std, orthogonal_fraction, orth_std, residual_coupling)``.

    When ``cache is None``, evaluates exact values with zero standard error.
    When ``cache`` is supplied, propagates joint grouped covariance via the
    delta method over the 2x2 generalized eigenvalue block and the response of
    the measured basis Ritz values, vectors and retained-space projector.
    Degenerate roots or unstable retention decisions have unresolved uncertainty.
    """
    low, low_std, orth, orth_std, coupling, _, _, _ = _expected_lowering_details(
        bank, basis, result, candidate, root=root, cache=cache
    )
    return low, low_std, orth, orth_std, coupling


def _combine_expected_improvement(
    lowering: float,
    lowering_std: float,
    cost_model: IncrementalCostModel,
) -> float:
    if lowering_std == 0.0:
        return lowering
    if not math.isfinite(lowering_std):
        return 0.0
    mode = cost_model.expected_improvement_mode
    lam = cost_model.risk_aversion
    if mode == "point":
        return lowering
    if mode == "risk_adjusted":
        return max(0.0, lowering - lam * lowering_std)
    if mode == "snr_gated":
        return lowering if lowering > lam * lowering_std else 0.0
    # ``shrinkage``: James-Stein / Wiener SNR attenuation factor
    denom = lowering * lowering
    if denom <= 0.0:
        return 0.0
    factor = max(0.0, 1.0 - ((lam * lowering_std) ** 2) / max(denom, 1e-30))
    return lowering * factor


@dataclass(frozen=True)
class InformationScore:
    """Resource- and uncertainty-aware candidate evaluation."""

    index: int
    label: str
    residual_coupling: float
    predicted_lowering: float
    lowering_std: float
    risk_adjusted_lowering: float
    orthogonal_fraction: float
    orthogonal_std: float
    new_words: int
    incremental_groups: int
    incremental_time_us: float
    classical_products: int
    classical_terms: int
    incremental_cost: float
    score: float
    rejected: str | None = None
    leakage: dict[str, float] | None = None
    per_root: tuple[float, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.rejected is None


def _score_information_candidate_with_details(
    bank: MatrixElementBank,
    basis: Sequence[int],
    result: SubspaceResult,
    candidate: int,
    *,
    cost_model: IncrementalCostModel,
    cache: GroupedWordCache | None = None,
    roots: Sequence[int] = (0,),
    aggregation: str = "mean",
    basis_words: frozenset[int] | None = None,
    qwc_bases: tuple[int, ...] | None = None,
    min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
    leakage_tol: float | None = None,
    leakage_mode: str = "operator",
    sector_target: tuple[int, float] | None = None,
    spin_ordering: str | tuple = "interleaved",
    basis_response: _BasisResponse | None = None,
) -> tuple[InformationScore, WordFunctional | None, WordFunctional | None]:
    label = bank.generator(candidate).label
    leakage = None
    if cache is None:
        s_aa = float(bank.entry(candidate, candidate)[0].real)
    else:
        s_aa_f, _ = entry_functionals(bank, candidate, candidate)[0]
        s_aa = float(s_aa_f.estimate(cache))

    if s_aa <= DEFAULT_NORM_FLOOR ** 2:
        return (
            InformationScore(
                candidate,
                label,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0.0,
                0,
                0,
                0.0,
                0.0,
                rejected="annihilates the reference",
            ),
            None,
            None,
        )

    if leakage_tol is not None:
        leakage, worst = _candidate_sector_leakage(
            bank,
            candidate,
            leakage_tol=leakage_tol,
            leakage_mode=leakage_mode,
            sector_target=sector_target,
            spin_ordering=spin_ordering,
        )
        if worst > leakage_tol:
            return (
                InformationScore(
                    candidate,
                    label,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0.0,
                    0,
                    0,
                    0.0,
                    0.0,
                    rejected=f"sector leakage {worst:.2e}",
                    leakage=leakage,
                ),
                None,
                None,
            )

    tracked = tuple(r for r in roots if r < len(result.energies))
    if not tracked:
        tracked = (0,)
    if cache is not None and basis_response is None:
        basis_response = _cached_basis_response(bank, basis, result, cache)

    per_root_low: list[float] = []
    per_root_std: list[float] = []
    per_root_ei: list[float] = []
    per_root_coup_std: list[float] = []
    couplings: list[float] = []
    orth_frac = 1.0
    orth_std = 0.0
    orth_f_0: WordFunctional | None = None
    low_f_0: WordFunctional | None = None

    for idx_r, r in enumerate(tracked):
        low_r, std_r, ofrac, ostd, coup, cstd, of_r, lf_r = _expected_lowering_details(
            bank, basis, result, candidate, root=r, cache=cache,
            basis_response=basis_response
        )
        if idx_r == 0:
            orth_frac, orth_std = ofrac, ostd
            orth_f_0, low_f_0 = of_r, lf_r
        couplings.append(coup)
        per_root_coup_std.append(cstd)
        per_root_low.append(low_r)
        per_root_std.append(std_r)
        per_root_ei.append(_combine_expected_improvement(low_r, std_r, cost_model))

    coupling = _aggregate(couplings, aggregation)
    lowering = _aggregate(per_root_low, aggregation)
    lowering_std = _aggregate(per_root_std, aggregation)
    ei_value = _aggregate(per_root_ei, aggregation)

    breakdown = estimate_incremental_cost(
        bank,
        basis,
        candidate,
        cost_model=cost_model,
        basis_words=basis_words,
        qwc_bases=qwc_bases,
    )

    if orth_frac < min_orthogonality:
        return (
            InformationScore(
                candidate,
                label,
                coupling,
                0.0,
                lowering_std,
                0.0,
                orth_frac,
                orth_std,
                breakdown.new_words,
                breakdown.incremental_groups,
                breakdown.incremental_time_us,
                breakdown.classical_products,
                breakdown.classical_terms,
                breakdown.total_cost,
                0.0,
                rejected=f"orthogonal fraction {orth_frac:.2e}",
                leakage=leakage,
            ),
            orth_f_0,
            low_f_0,
        )

    z = cost_model.orthogonality_z
    orth_corroborated = (
        orth_frac >= max(min_orthogonality, 0.005)
        and all(
            math.isfinite(sr)
            and sr > 0.0
            and lr >= z * sr
            and cp > 0.0
            and math.isfinite(cs)
            and cp >= z * cs
            for lr, sr, cp, cs in zip(per_root_low, per_root_std, couplings, per_root_coup_std)
        )
    )
    if (
        cache is not None
        and cost_model.require_stat_orthogonality
        and (
            not math.isfinite(orth_std)
            or orth_frac < z * orth_std
        )
        and not orth_corroborated
    ):
        return (
            InformationScore(
                candidate,
                label,
                coupling,
                lowering,
                lowering_std,
                0.0,
                orth_frac,
                orth_std,
                breakdown.new_words,
                breakdown.incremental_groups,
                breakdown.incremental_time_us,
                breakdown.classical_products,
                breakdown.classical_terms,
                breakdown.total_cost,
                0.0,
                rejected=f"orthogonal fraction unresolved ({orth_frac:.2e} vs {orth_std:.2e})",
                leakage=leakage,
            ),
            orth_f_0,
            low_f_0,
        )

    score = ei_value / (breakdown.total_cost + cost_model.epsilon)
    return (
        InformationScore(
            candidate,
            label,
            coupling,
            lowering,
            lowering_std,
            ei_value,
            orth_frac,
            orth_std,
            breakdown.new_words,
            breakdown.incremental_groups,
            breakdown.incremental_time_us,
            breakdown.classical_products,
            breakdown.classical_terms,
            breakdown.total_cost,
            score,
            leakage=leakage,
            per_root=tuple(per_root_low),
        ),
        orth_f_0,
        low_f_0,
    )


def score_information_candidate(
    bank: MatrixElementBank,
    basis: Sequence[int],
    result: SubspaceResult,
    candidate: int,
    *,
    cost_model: IncrementalCostModel,
    cache: GroupedWordCache | None = None,
    roots: Sequence[int] = (0,),
    aggregation: str = "mean",
    basis_words: frozenset[int] | None = None,
    qwc_bases: tuple[int, ...] | None = None,
    min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
    leakage_tol: float | None = None,
    leakage_mode: str = "operator",
    sector_target: tuple[int, float] | None = None,
    spin_ordering: str | tuple = "interleaved",
) -> InformationScore:
    """Evaluate ``E[dE_A] / (C_incremental(A) + eps)`` for one candidate."""
    score, _, _ = _score_information_candidate_with_details(
        bank,
        basis,
        result,
        candidate,
        cost_model=cost_model,
        cache=cache,
        roots=roots,
        aggregation=aggregation,
        basis_words=basis_words,
        qwc_bases=qwc_bases,
        min_orthogonality=min_orthogonality,
        leakage_tol=leakage_tol,
        leakage_mode=leakage_mode,
        sector_target=sector_target,
        spin_ordering=spin_ordering,
    )
    return score


def retained_sector_certificate(
    bank: MatrixElementBank,
    basis: Sequence[int],
    *,
    leakage_tol: float | None,
    leakage_mode: str,
    sector_target: tuple[int, float] | None,
    spin_ordering: str | tuple,
) -> dict[str, Any] | None:
    """The whole-subspace sector check ``run_acase`` ends with, for its variants.

    Candidates that each pass a reference-conditioned leakage test can still
    combine into a span that leaks more. With ``leakage_tol`` set and a mode
    other than ``'operator'``, this certifies the retained span and raises if it
    fails, exactly as ``run_acase`` does; otherwise it returns ``None``.
    """
    if leakage_tol is None or leakage_mode == "operator":
        return None
    certificate = subspace_sector_certificate(
        cast(MV, bank.reference), [bank.generator(index) for index in basis],
        sector_target=sector_target, spin_ordering=spin_ordering)
    if certificate["max_sector_leakage"] > leakage_tol:
        raise ValueError(
            "retained subspace fails the reference-conditioned sector "
            f"certificate: {certificate['max_sector_leakage']:.3g} > {leakage_tol:.3g}")
    return certificate


def select_information_candidate(
    scores: Sequence[InformationScore],
    *,
    min_lowering: float | None = None,
    lowering: str = "predicted_lowering",
) -> InformationScore | None:
    """Deterministic tie-breaking argmax of the information ratio.

    With ``min_lowering``, only accepted candidates whose ``lowering`` field
    reaches it compete. Applying the threshold after the argmax instead would
    let a cheap candidate with negligible lowering win on ratio and end the run
    while a costlier candidate with real lowering was available.
    """
    if lowering not in ("predicted_lowering", "risk_adjusted_lowering"):
        raise ValueError("lowering must be 'predicted_lowering' or 'risk_adjusted_lowering'")
    live = [s for s in scores if s.accepted and (
        min_lowering is None or getattr(s, lowering) >= min_lowering)]
    if not live:
        return None
    best = canonical_argmax(
        range(len(live)),
        lambda k: live[k].score,
        rtol=TIE_RTOL,
        atol=TIE_ATOL,
    )
    return live[best]


def run_resource_aware_acase(
    rho: MV,
    hamiltonian,
    candidates: Sequence,
    *,
    cost_model: IncrementalCostModel,
    initial: Sequence | None = None,
    bank: MatrixElementBank | None = None,
    max_size: int = 10,
    roots: int = 1,
    aggregation: str = "mean",
    min_lowering: float = DEFAULT_MIN_LOWERING,
    min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
    leakage_tol: float | None = None,
    leakage_mode: str = "operator",
    sector_target: tuple[int, float] | None = None,
    spin_ordering: str | tuple = "interleaved",
    exact_ground_energy: float | None = None,
    target_error: float | None = None,
    tau_s: float = DEFAULT_TAU_S,
    rel_tau: float = 0.0,
    max_condition: float = DEFAULT_MAX_CONDITION,
) -> AdaptiveResult:
    """Exact adaptive A-CASE driven by ``E[dE_A] / (C_incremental(A) + eps)``."""
    started = time.perf_counter()
    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    else:
        _require_matching_bank(bank, rho, hamiltonian)

    config = ACASEConfig(
        max_size=max_size,
        roots=roots,
        aggregation=aggregation,
        min_lowering=min_lowering,
        min_orthogonality=min_orthogonality,
        leakage_tol=leakage_tol,
        leakage_mode=leakage_mode,
        sector_target=sector_target,
        spin_ordering=spin_ordering,
        exact_ground_energy=exact_ground_energy,
        target_error=target_error,
        tau_s=tau_s,
        rel_tau=rel_tau,
        max_condition=max_condition,
    )

    initial_gens = (
        as_generators(initial) if initial is not None else [identity_generator(bank.n)]
    )
    basis = tuple(bank.extend(initial_gens))
    pool: list[int] = []
    for index in bank.extend(as_generators(candidates)):
        if index not in basis and index not in pool:
            pool.append(index)
    if not pool:
        raise ValueError("no candidate generators outside the initial basis")

    bank.retain_basis(basis)
    solved = bank.solve(basis, tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition)
    value, root_energies = _acase_objective(solved, config)
    state = ACASEState(
        basis=basis,
        pool=tuple(pool),
        result=solved,
        energy=value,
        energy_history=(value,),
        records=(),
        root_energies=root_energies,
    )

    step_breakdowns: list[dict[str, Any]] = []
    tracked = tuple(range(roots))

    if exact_ground_energy is not None and target_error is not None:
        if abs(value - exact_ground_energy) <= target_error:
            state = acase_step(
                state,
                ACASEStepOutcome(
                    stopped_reason=(
                        f"target error reached ({abs(value - exact_ground_energy) * 1e3:.4f} mHa "
                        f"<= {target_error * 1e3:.4f} mHa)"
                    )
                ),
            )

    if state.stopped_reason == "basis budget reached":
        for step in range(1, max_size + 1):
            step_started = time.perf_counter()
            remaining = [idx for idx in state.pool if idx not in state.basis]
            if not remaining:
                state = acase_step(
                    state, ACASEStepOutcome(stopped_reason="candidate pool exhausted")
                )
                break

            basis_words = bank.word_set(state.basis)
            need_qwc = (
                cost_model.setting_weight > 0.0
                or cost_model.time_weight_us > 0.0
            )
            qwc_bases = (
                qwc_bases_for_words(bank.n, basis_words) if need_qwc else ()
            )
            scores = [
                score_information_candidate(
                    bank,
                    state.basis,
                    state.result,
                    idx,
                    cost_model=cost_model,
                    roots=tracked,
                    aggregation=aggregation,
                    basis_words=basis_words,
                    qwc_bases=qwc_bases,
                    min_orthogonality=min_orthogonality,
                    leakage_tol=leakage_tol,
                    leakage_mode=leakage_mode,
                    sector_target=sector_target,
                    spin_ordering=spin_ordering,
                )
                for idx in remaining
            ]
            best = select_information_candidate(scores, min_lowering=min_lowering)
            if best is None:
                reason = ("predicted lowering below threshold"
                          if any(s.accepted for s in scores)
                          else "every candidate rejected (conditioning or sector)")
                state = acase_step(state, ACASEStepOutcome(stopped_reason=reason))
                break

            new_basis = state.basis + (best.index,)
            bank.retain_basis(new_basis)
            solved = bank.solve(
                new_basis, tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition
            )
            value, root_energies = _acase_objective(solved, config)
            rejected_conditioning = sum(
                1 for s in scores if s.rejected and s.rejected.startswith("orthogonal")
            )
            rejected_sector = sum(
                1 for s in scores if s.rejected and s.rejected.startswith("sector")
            )
            record = GrowthRecord(
                step=step,
                selected_label=best.label,
                energy=value,
                predicted_lowering=best.predicted_lowering,
                actual_lowering=state.energy - value,
                residual_coupling=best.residual_coupling,
                orthogonal_fraction=best.orthogonal_fraction,
                basis_size=len(new_basis),
                effective_rank=solved.effective_rank,
                condition_number=solved.condition_number,
                candidates_scored=len(remaining),
                rejected_conditioning=rejected_conditioning,
                rejected_sector=rejected_sector,
                new_words=best.new_words,
                word_universe=solved.resources["word_universe"],
                selection_word_universe=len(bank.word_set()),
                step_seconds=time.perf_counter() - step_started,
                leakage=best.leakage,
                root_energies=root_energies,
                per_root_lowering=best.per_root,
                selection_criterion="information_ratio",
                selection_score=best.score,
            )
            step_breakdowns.append(
                {
                    "step": step,
                    "label": best.label,
                    "new_words": best.new_words,
                    "incremental_groups": best.incremental_groups,
                    "incremental_time_us": best.incremental_time_us,
                    "classical_products": best.classical_products,
                    "classical_terms": best.classical_terms,
                    "incremental_cost": best.incremental_cost,
                    "score": best.score,
                }
            )
            stopped_reason = None
            if exact_ground_energy is not None and target_error is not None:
                err = abs(value - exact_ground_energy)
                if err <= target_error:
                    stopped_reason = (
                        f"target error reached ({err * 1e3:.4f} mHa <= "
                        f"{target_error * 1e3:.4f} mHa)"
                    )
            state = acase_step(
                state,
                ACASEStepOutcome(
                    selected_index=best.index,
                    result=solved,
                    energy=value,
                    record=record,
                    root_energies=root_energies,
                    stopped_reason=stopped_reason,
                ),
            )
            if stopped_reason is not None:
                break

    resources = dict(state.result.resources)
    resources.update(bank.resources(state.basis))
    certificate = retained_sector_certificate(
        bank, state.basis, leakage_tol=leakage_tol, leakage_mode=leakage_mode,
        sector_target=sector_target, spin_ordering=spin_ordering)
    if certificate is not None:
        resources["sector_certificate"] = certificate
    final_words = bank.word_set(state.basis)
    resources.update(
        {
            "candidate_pool_size": len(state.pool),
            "growth_seconds": time.perf_counter() - started,
            "registered_generators": len(bank),
            "selection_criterion": "information_ratio",
            "final_qwc_groups": len(qwc_bases_for_words(bank.n, final_words)),
            "cost_breakdowns": step_breakdowns,
        }
    )
    relative = None
    if exact_ground_energy is not None:
        relative = abs(state.result.ground_energy - exact_ground_energy) / max(
            abs(exact_ground_energy), 1e-12
        )
    return AdaptiveResult(
        labels=tuple(bank.generator(i).label for i in state.basis),
        energy=state.energy,
        energy_history=state.energy_history,
        records=state.records,
        result=state.result,
        bank=bank,
        indices=state.basis,
        stopped_reason=state.stopped_reason,
        exact_ground_energy=exact_ground_energy,
        relative_error=relative,
        resources=resources,
        root_energies=state.root_energies,
    )


# ------------------------------------------------------------ closed-loop acquisition


@dataclass(frozen=True)
class AcquisitionStep:
    """One closed-loop decision (expand subspace or refine shots)."""

    step: int
    action: str
    selected_label: str | None
    basis_size: int
    effective_rank: int
    energy: float
    stat_std_error: float
    ci_halfwidth: float
    estimated_remaining_bias: float
    total_error_estimate: float
    shots_added: int
    cumulative_shots: int
    circuits_added: int
    cumulative_circuits: int
    qwc_groups: int
    time_us_added: float
    cumulative_time_us: float
    expected_info_rate: float
    step_seconds: float
    note: str = ""


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of closed-loop accuracy-targeted information acquisition."""

    labels: tuple[str, ...]
    energy: float
    std_error: float
    ci_interval: Interval
    estimated_remaining_bias: float
    total_error_estimate: float
    target_accuracy: float
    converged: bool
    stopped_reason: str
    history: tuple[AcquisitionStep, ...]
    result: SubspaceResult
    bank: MatrixElementBank
    cache: GroupedWordCache
    indices: tuple[int, ...]
    total_shots: int
    total_circuits: int
    qwc_groups: int
    total_time_us: float
    evidence: str = EvidenceLevel.ASYMPTOTIC.value
    resources: dict[str, Any] = field(default_factory=dict)

    @property
    def basis_size(self) -> int:
        return len(self.labels)


class _AcquisitionBudgetExhausted(RuntimeError):
    """The next measurement cannot fit the remaining physical resources."""


class _IncrementalGroupRegistry:
    """Monotone QWC circuit ledger compatible with cumulative ``GroupedWordCache``.

    Existing groups retain fixed ``word_codes`` and readout assignments so
    ``GroupedWordCache.add_batch`` never observes conflicting assignments across
    growing subspaces. Newly encountered words that already commute inside an
    existing group's measurement basis are covered with zero new circuits and
    immediately readable via ``cache``; only genuinely uncovered words generate
    fresh QWC groups.
    """

    def __init__(self, n: int, pooling: str = "shots", card: DeviceCard | None = None):
        self.n = int(n)
        self.pooling = pooling
        self.card = card
        self.groups: list[list[PauliWord]] = []
        self.group_keys: list[tuple[tuple[int, str], ...]] = []
        self.group_basis_dicts: list[dict[int, str]] = []
        self.group_basis_codes: list[int] = []
        self.group_durations_us: list[float] = []
        self.group_fidelities: list[float] = []
        # Every word each group reads for this run: its own words plus words
        # registered later that its basis already measures. Each word belongs
        # to exactly one group, so the allocator can attribute it.
        self._cover_words: list[set[int]] = []
        self._owner: dict[int, int] = {}
        self.key_to_pos: dict[tuple[tuple[int, str], ...], int] = {}

    def register_words(self, codes: Iterable[int]) -> list[list[PauliWord]]:
        """Add uncovered non-identity codes as new immutable QWC groups; return new ones."""
        unique_nonzero = sorted({int(c) for c in codes if int(c) != 0})
        uncovered: list[PauliWord] = []
        for code in unique_nonzero:
            if code in self._owner:
                continue
            # A setting reads a word iff the word's every non-identity letter
            # equals the setting's letter on that qubit.
            matched_pos = next(
                (g_pos for g_pos, basis_code in enumerate(self.group_basis_codes)
                 if _qwc_reads_word(self.n, code, basis_code)), None)
            if matched_pos is not None:
                self._cover_words[matched_pos].add(code)
                self._owner[code] = matched_pos
            else:
                uncovered.append(PauliWord(self.n, code))

        if not uncovered:
            return []

        fresh_groups = qwc_groups(uncovered)
        for group in fresh_groups:
            bdict = shared_basis(group)
            bkey = tuple(sorted(bdict.items()))
            if bkey in self.key_to_pos:  # pragma: no cover - disjoint by construction
                raise AssertionError("duplicate QWC group basis encountered")
            bcode = basis_dict_to_code(bdict)
            res = qwc_setting_resources(self.n, bcode)
            dur = (
                setting_duration_us(self.card, res)
                if self.card is not None
                else 1.0
            )
            fid = (
                setting_fidelity(self.card, res, self.n)
                if self.card is not None
                else 1.0
            )
            pos = len(self.groups)
            self.groups.append(list(group))
            self.group_keys.append(bkey)
            self.group_basis_dicts.append(bdict)
            self.group_basis_codes.append(bcode)
            self.group_durations_us.append(dur)
            self.group_fidelities.append(fid)
            self._cover_words.append({w.code for w in group})
            self._owner.update({w.code: pos for w in group})
            self.key_to_pos[bkey] = pos
        return fresh_groups

    def reader_groups(self) -> list[list[PauliWord]]:
        """Each group listing every word it reads, in registration order.

        ``groups`` keeps the words a group was opened with, which fixes its
        readout. This view adds the words registered later that it also reads,
        so a shot allocator sees them; it changes nothing.
        """
        return [[PauliWord(self.n, code) for code in sorted(cover)]
                for cover in self._cover_words]

    def session_groups(self, required_codes: Iterable[int]) -> list[list[PauliWord]]:
        """Groups listing every word in ``required_codes``, which must be registered.

        Registering here would let an unsampled setting look readable, so an
        unregistered word is an error; register and fund it first.
        """
        needed = {int(c) for c in required_codes if int(c) != 0}
        if not needed:
            return []
        missing = needed.difference(self._owner)
        if missing:
            raise ValueError(f"{len(missing)} word(s) are not registered with a measured "
                             "setting; register and sample them before building a session")
        return self.reader_groups()

    def effective_costs(self, *, fidelity_inflate: bool = True) -> dict[tuple, float]:
        costs: dict[tuple, float] = {}
        for key, dur, fid in zip(
            self.group_keys, self.group_durations_us, self.group_fidelities
        ):
            if fidelity_inflate and self.card is not None:
                costs[key] = dur / max(fid * fid, 1e-12)
            else:
                costs[key] = dur
        return costs

    def batch_time_us(self, groups: Sequence[Sequence[PauliWord]], shot_map_or_int) -> float:
        terms = []
        for group in groups:
            if not group:
                continue
            key = tuple(sorted(shared_basis(group).items()))
            pos = self.key_to_pos[key]
            dur = self.group_durations_us[pos]
            if isinstance(shot_map_or_int, int):
                shots = shot_map_or_int
            else:
                shots = max((int(shot_map_or_int.get(w.code, 0)) for w in group), default=0)
            terms.append(shots * dur)
        return math.fsum(terms)


def _qwc_reads_word(n: int, word_code: int, basis_code: int) -> bool:
    """Whether a setting with merged code ``basis_code`` records ``word_code``.

    The packed test ``GroupedWordCache.reader_keys`` applies: the word's support
    lies inside the setting's, and their letters agree there.
    """
    word = _nonident_lanes(n, word_code)
    return (word & ~_nonident_lanes(n, basis_code)) == 0 and (
        word & _nonident_lanes(n, word_code ^ basis_code)) == 0


def _words_needed_for_pairs(
    bank: MatrixElementBank,
    left: Sequence[int],
    right: Sequence[int],
) -> set[int]:
    codes: set[int] = set()
    for i in left:
        for j in right:
            bank.update_pair_support(codes, i, j)
    return codes


def _candidate_scoring_words(
    bank: MatrixElementBank,
    basis: Sequence[int],
    candidates: Sequence[int],
) -> set[int]:
    """Words strictly needed to score ``candidates`` against ``basis`` plus ``(c, c)``."""
    codes: set[int] = set()
    for c in candidates:
        bank.update_pair_support(codes, c, c)
        for b in basis:
            bank.update_pair_support(codes, b, c)
    return codes


def acquire_to_accuracy(
    rho: MV,
    hamiltonian,
    candidates: Sequence,
    backend,
    *,
    target_accuracy: float,
    cost_model: IncrementalCostModel | None = None,
    initial: Sequence | None = None,
    bank: MatrixElementBank | None = None,
    max_basis_size: int = 8,
    max_steps: int = 30,
    max_total_shots: int = 200_000,
    max_total_time_us: float | None = None,
    pilot_shots: int = 200,
    refine_batch_shots: int = 400,
    min_lowering: float = 1e-6,
    min_orthogonality: float = 1e-6,
    delta: float = 0.05,
    stop_metric: str = "std_error",
    pooling: str = "shots",
    leakage_tol: float | None = None,
    leakage_mode: str = "operator",
    sector_target: tuple[int, float] | None = None,
    spin_ordering: str | tuple = "interleaved",
    tau_s: float = DEFAULT_TAU_S,
    rel_tau: float = 0.0,
    max_condition: float = DEFAULT_MAX_CONDITION,
    calibrate_overlap: bool = False,
) -> AcquisitionResult:
    """Closed-loop finite-shot controller balancing expansion vs shot refinement.

    Never reads an exact ground-state oracle. Instead it tracks:
      1. Asymptotic statistical error ``sigma_stat`` of the current Ritz energy
         from cumulative ``GroupedWordCache`` histograms via ``ritz_uncertainty``
         and ``ritz_functional``;
      2. Non-oracle remaining bias estimate ``b_hat`` over still-active candidates
         (risk-adjusted 2x2 expected lowering + uncertainty guard);
      3. Resource-normalized information gain per unit cost:
         - **Expand**: adding ``A*`` cuts remaining bias while incurring
           ``C_incremental(A*)`` (new QWC settings, ``DeviceCard`` duration, and/or
           new observables).
         - **Refine Shots**: routing ``refine_batch_shots`` via covariance-aware
           ``variance_optimal_group_plan`` weighted by ``DeviceCard`` costs reduces
           ``sigma_stat`` smoothly according to exact per-group empirical variances
           ``sum_g V_{f,g} / (N_g + dN_g)``.

    Stops when the combined statistical and guarded remaining-lowering proxy
    reaches ``target_accuracy``, or a resource limit is reached. This proxy
    concerns the supplied candidate family; it does not bound true ground-state
    error. The reported interval is an asymptotic single-look diagnostic, not
    an anytime or finite-sample certificate for this adaptive schedule.

    Shot and execution-time limits apply before every batch, including pilots.
    A pilot is reduced uniformly to fit; if even one shot per required setting
    cannot fit initially, raises ``ValueError`` without sampling. Later limits
    return the last measured solve with unresolved remaining bias preserved.
    """
    if target_accuracy <= 0.0 or not math.isfinite(target_accuracy):
        raise ValueError("target_accuracy must be positive and finite")
    if stop_metric not in _STOP_METRICS:
        raise ValueError(f"stop_metric must be one of {_STOP_METRICS}")
    for name, value, minimum in (
        ("pilot_shots", pilot_shots, 1), ("refine_batch_shots", refine_batch_shots, 1),
        ("max_total_shots", max_total_shots, 1), ("max_basis_size", max_basis_size, 1),
        ("max_steps", max_steps, 0),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    pilot_shots, refine_batch_shots = int(pilot_shots), int(refine_batch_shots)
    max_total_shots, max_steps = int(max_total_shots), int(max_steps)
    if max_total_time_us is not None and (
        not math.isfinite(max_total_time_us) or max_total_time_us <= 0.0
    ):
        raise ValueError("max_total_time_us must be positive and finite")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if pooling not in ("assigned", "shots"):
        raise ValueError("pooling must be 'assigned' or 'shots'")

    started = time.perf_counter()
    if cost_model is None:
        cost_model = IncrementalCostModel(
            word_weight=0.0,
            setting_weight=1.0,
            time_weight_us=0.0,
            epsilon=1.0,
            risk_aversion=1.0,
            require_stat_orthogonality=True,
        )

    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    else:
        _require_matching_bank(bank, rho, hamiltonian)

    initial_gens = (
        as_generators(initial) if initial is not None else [identity_generator(bank.n)]
    )
    basis = list(bank.extend(initial_gens))
    if len(basis) > max_basis_size:
        raise ValueError("the initial basis exceeds max_basis_size")
    pool: list[int] = []
    for index in bank.extend(as_generators(candidates)):
        if index not in basis and index not in pool:
            pool.append(index)
    if not pool:
        raise ValueError("no candidate generators outside the initial basis")

    # Apply upfront operator/reference sector filtering when requested.
    if leakage_tol is not None:
        filtered_pool: list[int] = []
        for idx in pool:
            _, worst = _candidate_sector_leakage(
                bank,
                idx,
                leakage_tol=leakage_tol,
                leakage_mode=leakage_mode,
                sector_target=sector_target,
                spin_ordering=spin_ordering,
            )
            if worst <= leakage_tol:
                filtered_pool.append(idx)
        pool = filtered_pool

    cache = GroupedWordCache(bank.n, pooling=pooling)
    registry = _IncrementalGroupRegistry(
        bank.n, pooling=pooling, card=cost_model.device_card
    )
    total_time_us = 0.0
    solver_kwargs = dict(
        tau_s=tau_s,
        rel_tau=rel_tau,
        max_condition=max_condition,
        calibrate_overlap=calibrate_overlap,
    )

    def _group_duration(group) -> float:
        if cost_model.device_card is None:
            return 1.0
        basis_code = basis_dict_to_code(shared_basis(group))
        return setting_duration_us(cost_model.device_card, qwc_setting_resources(bank.n, basis_code))

    def _bounded_plan(groups, shots_spec) -> dict[int, int]:
        """Clip a proposed batch before the backend can spend either budget."""
        groups = [group for group in groups if group]
        counts = [shots_spec if isinstance(shots_spec, int) else
                  max((int(shots_spec.get(word.code, 0)) for word in group), default=0)
                  for group in groups]
        durations = [_group_duration(group) for group in groups]
        shots_left = max_total_shots - cache.total_shots
        time_left = (float("inf") if max_total_time_us is None else
                     max_total_time_us - total_time_us)
        requested = sum(counts)
        requested_time = math.fsum(count * dur for count, dur in zip(counts, durations))
        if requested <= 0 or shots_left <= 0:
            raise _AcquisitionBudgetExhausted("shot budget reached")
        scale = min(1.0, shots_left / requested,
                    time_left / requested_time if requested_time > 0.0 else 1.0)
        bounded = [max(0, math.floor(count * scale)) for count in counts]
        if not any(bounded):
            eligible = [i for i, (count, dur) in enumerate(zip(counts, durations))
                        if count > 0 and dur <= time_left]
            if not eligible:
                raise _AcquisitionBudgetExhausted("hardware time budget reached")
            best_group = max(eligible, key=lambda i: counts[i] / max(durations[i], 1e-30))
            bounded[best_group] = 1
        # Guard an integer boundary against floating-point rounding as well.
        while (sum(bounded) > shots_left or
               math.fsum(count * dur for count, dur in zip(bounded, durations)) > time_left):
            last = next((i for i in reversed(range(len(bounded))) if bounded[i]), None)
            if last is None:
                raise _AcquisitionBudgetExhausted("hardware time budget reached")
            bounded[last] -= 1
        if not any(bounded):
            raise _AcquisitionBudgetExhausted("hardware time budget reached")
        return {word.code: count for group, count in zip(groups, bounded) if count
                for word in group}

    def _sample_groups(groups_to_shoot, shots_spec) -> tuple[int, int, float]:
        nonlocal total_time_us
        plan = _bounded_plan(groups_to_shoot, shots_spec)
        non_empty = [g for g in groups_to_shoot if g and plan.get(g[0].code, 0) > 0]
        if not non_empty:
            return 0, 0, 0.0
        batch = backend.sample_grouped_from_state(rho, non_empty, plan)
        cache.add_batch(batch)
        dt_us = registry.batch_time_us(non_empty, plan)
        total_time_us += dt_us
        return batch.hardware_shots, batch.circuits, dt_us

    def _ensure_covered(codes: Iterable[int]) -> tuple[int, int, float]:
        codes = set(codes)
        # Preview the fresh circuits before registration. An unfunded circuit
        # must not look registered/readable to later scoring passes.
        uncovered = [code for code in codes if code != 0 and not any(
            _qwc_reads_word(bank.n, code, bc) for bc in registry.group_basis_codes)]
        new_groups = qwc_groups(_non_identity_words(bank.n, uncovered)) if uncovered else []
        if not new_groups:
            registry.register_words(codes)
            return 0, 0, 0.0
        shots_left = max_total_shots - cache.total_shots
        shots_each = min(pilot_shots, shots_left // len(new_groups))
        if shots_each < 1:
            raise _AcquisitionBudgetExhausted("shot budget reached before candidate screening")
        round_time = math.fsum(_group_duration(group) for group in new_groups)
        if max_total_time_us is not None and round_time > 0.0:
            shots_each = min(shots_each, math.floor((max_total_time_us - total_time_us) / round_time))
            while shots_each > 0 and math.fsum(
                shots_each * _group_duration(group) for group in new_groups
            ) > max_total_time_us - total_time_us:
                shots_each -= 1
        if shots_each < 1:
            raise _AcquisitionBudgetExhausted("hardware time budget reached before candidate screening")
        new_groups = registry.register_words(codes)
        return _sample_groups(new_groups, shots_each)

    # Initial measurement of basis block + candidate rows against initial basis
    init_codes = _words_needed_for_pairs(bank, basis, basis)
    init_cand_codes = _candidate_scoring_words(bank, basis, pool)
    try:
        s0, c0, t0 = _ensure_covered(init_codes | init_cand_codes)
    except _AcquisitionBudgetExhausted as error:
        raise ValueError(f"initial pilot cannot fit the resource budget: {error}") from error

    def _solve_current():
        req = _words_needed_for_pairs(bank, basis, basis)
        sess = SharedMeasurement(
            bank, basis, groups=registry.session_groups(req), pooling=pooling
        )
        res = sess.solve(cache, **solver_kwargs)
        iv = ritz_uncertainty(sess, cache, res, root=0, delta=delta)
        return sess, res, iv

    session, result, interval = _solve_current()
    steps: list[AcquisitionStep] = []
    converged = False
    stopped_reason = "step budget reached"

    def _stat_err(iv: Interval) -> float:
        if stop_metric == "std_error":
            return float(iv.std_error or 0.0)
        return 0.5 * (iv.upper - iv.lower)

    def _predict_refined_stat(
        ritz_f: WordFunctional,
        target_funcs: Sequence[WordFunctional],
        budget: int,
    ) -> tuple[dict[int, int], float, float, str | None]:
        """``(plan, new_stat_error, batch_time_us, blocked)`` via the Neyman formula.

        ``blocked`` names the exhausted budget when one prevents any refinement.
        """
        costs = registry.effective_costs(fidelity_inflate=cost_model.fidelity_inflate)
        unchanged = float(interval.std_error or 0.0)
        if not target_funcs:
            return {}, unchanged, 0.0, None
        # The allocator matches coefficients against listed words, so it needs
        # every word a setting reads, including those registered after it.
        readers = registry.reader_groups()
        try:
            plan = variance_optimal_group_plan(
                readers,
                cache,
                target_funcs,
                budget,
                costs=costs,
                min_shots=0,
            )
            if plan:
                plan = _bounded_plan(readers, plan)
        except _AcquisitionBudgetExhausted as error:
            return {}, unchanged, 0.0, str(error)
        except ValueError:
            return {}, unchanged, 0.0, None
        if not plan:
            return {}, unchanged, 0.0, None

        # Predict new variance of ritz_func using exact group empirical variances
        stats = (
            None
            if ritz_f.is_deterministic
            else cache.candidate_group_statistics(dict(ritz_f.coefficients))
        )
        if stats is None:
            new_std = float(interval.std_error or 0.0)
        else:
            key_to_add: dict[tuple, int] = {}
            for key, group in zip(registry.group_keys, readers):
                add = max((int(plan.get(w.code, 0)) for w in group), default=0)
                if add > 0:
                    key_to_add[key] = add
            cur_stats_var = 0.0
            pred_var = 0.0
            for key, (N_g, var_g, _) in stats.items():
                add = key_to_add.get(key, 0)
                cur_stats_var += float(var_g) / max(1, int(N_g))
                pred_var += float(var_g) / max(1, int(N_g) + add)
            cur_se_val = float(interval.std_error or 0.0)
            if cur_stats_var > 0.0 and cur_se_val > 0.0:
                new_std = cur_se_val * math.sqrt(max(0.0, pred_var / cur_stats_var))
            else:
                new_std = math.sqrt(max(0.0, pred_var))
        if stop_metric == "ci_halfwidth":
            cur_se = max(float(interval.std_error or 0.0), 1e-30)
            new_stat = new_std * (0.5 * (interval.upper - interval.lower) / cur_se)
        else:
            new_stat = new_std
        batch_t = registry.batch_time_us(readers, plan)
        return plan, new_stat, batch_t, None

    def _budget_remaining() -> bool:
        if cache.total_shots >= max_total_shots:
            return False
        if max_total_time_us is not None and total_time_us >= max_total_time_us:
            return False
        return True

    assessment_stop_reason: str | None = None

    def _assess_state(*, allow_new_measurements: bool = True):
        nonlocal session, result, interval, assessment_stop_reason
        assessment_stop_reason = None
        remaining = [i for i in pool if i not in basis]
        add_s = add_c = 0
        add_t = 0.0
        if (
            allow_new_measurements
            and remaining
            and len(basis) < max_basis_size
            and _budget_remaining()
        ):
            # ``_budget_remaining`` guarantees shots remain to fund new settings.
            cand_codes = _candidate_scoring_words(bank, basis, remaining)
            try:
                add_s, add_c, add_t = _ensure_covered(cand_codes)
            except _AcquisitionBudgetExhausted as error:
                assessment_stop_reason = str(error)
            if add_s:
                # With pooled readouts these new circuits also change the
                # current basis means, energy and interval.
                session, result, interval = _solve_current()

        basis_words = bank.word_set(basis)
        qwc_bases = tuple(registry.group_basis_codes)
        ritz_f = ritz_functional(
            bank, basis, result.coefficients[:, 0], result.energies[0]
        )
        scores: list[InformationScore] = []
        cand_funcs: dict[int, tuple[WordFunctional | None, WordFunctional | None]] = {}
        required = _candidate_scoring_words(bank, basis, remaining)
        if any(code != 0 and not cache.group_weights(code) for code in required):
            if assessment_stop_reason is None:
                assessment_stop_reason = ("basis budget reached before remaining bias could be assessed"
                                          if len(basis) >= max_basis_size else "resource budget reached")
            return (remaining, scores, None, float("inf"), float("inf"), ritz_f,
                    cand_funcs, add_s, add_c, add_t)
        if remaining:
            response = _basis_response(bank, basis, result, cache)
            for idx in remaining:
                sc, of_0, lf_0 = _score_information_candidate_with_details(
                    bank,
                    basis,
                    result,
                    idx,
                    cost_model=cost_model,
                    cache=cache,
                    roots=(0,),
                    basis_words=basis_words,
                    qwc_bases=qwc_bases,
                    min_orthogonality=min_orthogonality,
                    basis_response=response,
                )
                scores.append(sc)
                cand_funcs[idx] = (of_0, lf_0)
        best = select_information_candidate(
            scores, min_lowering=min_lowering, lowering="risk_adjusted_lowering")
        live_scores = [s for s in scores if s.accepted]
        unresolved = [s for s in scores if s.rejected is not None and
                      s.rejected.startswith("orthogonal fraction unresolved")]
        raw_best_upper = max((s.predicted_lowering + s.lowering_std
                              for s in [*live_scores, *unresolved]), default=0.0)
        active_bias = max(
            (s.predicted_lowering for s in [*live_scores, *unresolved]),
            default=0.0,
        )
        return (
            remaining,
            scores,
            best,
            active_bias,
            raw_best_upper,
            ritz_f,
            cand_funcs,
            add_s,
            add_c,
            add_t,
        )

    remaining, scores, best, active_bias, bias_upper, ritz_f, cand_funcs, _, _, _ = _assess_state()
    stat_now = _stat_err(interval)
    rmse_now = math.hypot(stat_now, bias_upper)

    steps.append(
        AcquisitionStep(
            step=0,
            action="init",
            selected_label=None,
            basis_size=len(basis),
            effective_rank=result.effective_rank,
            energy=result.ground_energy,
            stat_std_error=float(interval.std_error or 0.0),
            ci_halfwidth=0.5 * (interval.upper - interval.lower),
            estimated_remaining_bias=bias_upper,
            total_error_estimate=rmse_now,
            shots_added=s0,
            cumulative_shots=cache.total_shots,
            circuits_added=c0,
            cumulative_circuits=cache.total_circuits,
            qwc_groups=cache.num_groups(),
            time_us_added=t0,
            cumulative_time_us=total_time_us,
            expected_info_rate=0.0,
            step_seconds=time.perf_counter() - started,
            note="pilot basis & candidate cache",
        )
    )

    for step in range(1, max_steps + 1):
        step_t0 = time.perf_counter()
        stat_now = _stat_err(interval)
        rmse_now = math.hypot(stat_now, active_bias)

        shots_left_now = max_total_shots - cache.total_shots
        can_expand = (
            best is not None
            and len(basis) < max_basis_size
            and best.risk_adjusted_lowering >= min_lowering
            and shots_left_now >= best.incremental_groups
        )
        harm_thr = max(min_lowering, 0.5 * target_accuracy)
        ambig_targets = [
            s
            for s in scores
            if s.accepted
            and (not math.isfinite(s.lowering_std) or (
                s.risk_adjusted_lowering < min_lowering
                and (s.predicted_lowering + s.lowering_std) >= harm_thr))
        ]
        unres_targets = [
            s
            for s in scores
            if s.rejected is not None
            and s.rejected.startswith("orthogonal fraction unresolved")
            and (
                (not math.isfinite(s.lowering_std))
                or (s.predicted_lowering + 1.0 * s.lowering_std) >= harm_thr
            )
        ]
        has_ambiguous = bool(ambig_targets)
        has_unresolved_orth = bool(unres_targets)

        if assessment_stop_reason is not None:
            stopped_reason = assessment_stop_reason
            break
        if math.hypot(stat_now, bias_upper) <= target_accuracy:
            converged = True
            stopped_reason = (
                f"target accuracy reached (stat={stat_now:.3e} <= {target_accuracy:.3e}, "
                f"rem_bias={bias_upper:.3e})"
            )
            break
        if remaining and len(basis) >= max_basis_size and bias_upper > target_accuracy:
            stopped_reason = "basis budget reached with unresolved remaining bias"
            break

        if shots_left_now <= 0:
            stopped_reason = "shot budget reached"
            break
        if max_total_time_us is not None and total_time_us >= max_total_time_us:
            stopped_reason = "hardware time budget reached"
            break

        # Build shot-refinement plan and score its expected RMSE reduction per unit cost
        refine_targets: list[WordFunctional] = [ritz_f] if not ritz_f.is_deterministic else []
        if has_ambiguous or has_unresolved_orth:
            blocker_cands = unres_targets if unres_targets else ambig_targets
            top_cands = sorted(
                blocker_cands,
                key=lambda s: (
                    -(s.predicted_lowering + (s.lowering_std if math.isfinite(s.lowering_std) else 0.0)),
                    s.index,
                ),
            )[:2]
            for sc in top_cands:
                of_c, lf_c = cand_funcs.get(sc.index, (None, None))
                if of_c is not None and not of_c.is_deterministic:
                    refine_targets.append(of_c)
                if lf_c is not None and not lf_c.is_deterministic:
                    refine_targets.append(lf_c)
            if any(not math.isfinite(s.lowering_std) for s in top_cands):
                refine_targets.extend(session.matrix_functionals())

        batch_budget = min(refine_batch_shots, shots_left_now)
        plan, stat_after_refine, refine_time_us, refine_blocked = _predict_refined_stat(
            ritz_f, refine_targets, batch_budget
        )
        rmse_after_refine = math.hypot(stat_after_refine, active_bias)
        delta_rmse_refine = max(0.0, rmse_now - rmse_after_refine)
        use_device_time = (
            cost_model.time_weight_us > 0.0 and cost_model.device_card is not None
        )
        refine_cost = (
            refine_time_us + cost_model.epsilon
            if use_device_time
            else float(sum(max((plan.get(w.code, 0) for w in group), default=0)
                           for group in registry.groups)) + cost_model.epsilon
        )
        rate_refine = delta_rmse_refine / refine_cost if plan else 0.0

        # Score candidate expansion on commensurate RMSE reduction per unit cost
        rate_expand = 0.0
        if can_expand and best is not None:
            bias_after_expand = max(0.0, active_bias - best.risk_adjusted_lowering)
            rmse_after_expand = math.hypot(stat_now, bias_after_expand)
            delta_rmse_expand = max(0.0, rmse_now - rmse_after_expand)
            eff_groups = max(1, best.incremental_groups)
            if use_device_time:
                if best.incremental_groups > 0:
                    unit_time_delta = best.incremental_time_us / max(
                        cost_model.shots_per_setting, 1e-12
                    )
                else:
                    eff_costs = registry.effective_costs(
                        fidelity_inflate=cost_model.fidelity_inflate
                    )
                    unit_time_delta = (
                        float(np.mean(list(eff_costs.values()))) if eff_costs else 1.0
                    )
                expand_time_us = unit_time_delta * float(pilot_shots)
                expand_cost = expand_time_us + cost_model.epsilon
            else:
                expand_cost = (
                    float(eff_groups * pilot_shots)
                    + cost_model.word_weight * float(best.new_words)
                    + cost_model.setting_weight * float(best.incremental_groups)
                    + cost_model.epsilon
                )
            rate_expand = delta_rmse_expand / expand_cost

        # Decide action: Expand vs Refine Shots
        if can_expand and best is not None and (not plan or rate_expand >= rate_refine):
            next_basis = [*basis, best.index]
            add_req = _words_needed_for_pairs(bank, next_basis, next_basis)
            try:
                s_a, c_a, t_a = _ensure_covered(add_req)
            except _AcquisitionBudgetExhausted as error:
                stopped_reason = str(error)
                break
            basis.append(best.index)
            session, result, interval = _solve_current()
            (
                remaining,
                scores,
                best_next,
                active_bias,
                bias_upper,
                ritz_f,
                cand_funcs,
                s_c,
                c_c,
                t_c,
            ) = _assess_state(allow_new_measurements=_budget_remaining())
            best = best_next
            steps.append(
                AcquisitionStep(
                    step=step,
                    action="expand",
                    selected_label=bank.generator(basis[-1]).label,
                    basis_size=len(basis),
                    effective_rank=result.effective_rank,
                    energy=result.ground_energy,
                    stat_std_error=float(interval.std_error or 0.0),
                    ci_halfwidth=0.5 * (interval.upper - interval.lower),
                    estimated_remaining_bias=bias_upper,
                    total_error_estimate=math.hypot(_stat_err(interval), bias_upper),
                    shots_added=s_a + s_c,
                    cumulative_shots=cache.total_shots,
                    circuits_added=c_a + c_c,
                    cumulative_circuits=cache.total_circuits,
                    qwc_groups=cache.num_groups(),
                    time_us_added=t_a + t_c,
                    cumulative_time_us=total_time_us,
                    expected_info_rate=rate_expand,
                    step_seconds=time.perf_counter() - step_t0,
                    note=f"expand rate={rate_expand:.3e} vs refine={rate_refine:.3e}",
                )
            )
        elif plan:
            active_groups_shot = [
                group
                for group in registry.groups
                if max((int(plan.get(w.code, 0)) for w in group), default=0) > 0
            ]
            try:
                s_r, c_r, t_r = _sample_groups(active_groups_shot, plan)
            except _AcquisitionBudgetExhausted as error:
                stopped_reason = str(error)
                break
            result = session.solve(cache, **solver_kwargs)
            interval = ritz_uncertainty(session, cache, result, root=0, delta=delta)
            (
                remaining,
                scores,
                best,
                active_bias,
                bias_upper,
                ritz_f,
                cand_funcs,
                s_c,
                c_c,
                t_c,
            ) = _assess_state(allow_new_measurements=_budget_remaining())
            steps.append(
                AcquisitionStep(
                    step=step,
                    action="refine_shots",
                    selected_label=None,
                    basis_size=len(basis),
                    effective_rank=result.effective_rank,
                    energy=result.ground_energy,
                    stat_std_error=float(interval.std_error or 0.0),
                    ci_halfwidth=0.5 * (interval.upper - interval.lower),
                    estimated_remaining_bias=bias_upper,
                    total_error_estimate=math.hypot(_stat_err(interval), bias_upper),
                    shots_added=s_r + s_c,
                    cumulative_shots=cache.total_shots,
                    circuits_added=c_r + c_c,
                    cumulative_circuits=cache.total_circuits,
                    qwc_groups=cache.num_groups(),
                    time_us_added=t_r + t_c,
                    cumulative_time_us=total_time_us,
                    expected_info_rate=rate_refine,
                    step_seconds=time.perf_counter() - step_t0,
                    note=f"refine rate={rate_refine:.3e} vs expand={rate_expand:.3e}",
                )
            )
        else:
            stopped_reason = (refine_blocked if refine_blocked is not None else
                              "no remaining beneficial action (deterministic basis exhausted)")
            break

    final_stat = _stat_err(interval)
    final_rmse = math.hypot(final_stat, bias_upper)
    if assessment_stop_reason is not None:
        stopped_reason = assessment_stop_reason
    elif final_rmse <= target_accuracy:
        converged = True
        stopped_reason = (f"target accuracy reached (stat={final_stat:.3e} <= {target_accuracy:.3e}, "
                          f"rem_bias={bias_upper:.3e})")
    resources = dict(result.resources)
    resources.update(bank.resources(basis))
    resources.update(
        {
            "evidence": EvidenceLevel.ASYMPTOTIC.value,
            "target_accuracy": target_accuracy,
            "stop_metric": stop_metric,
            "remaining_bias_point_estimate": active_bias,
            "pooling": pooling,
            "total_shots": cache.total_shots,
            "total_circuits": cache.total_circuits,
            "qwc_groups": cache.num_groups(),
            "total_time_us": total_time_us,
            "device_card": (
                cost_model.device_card.name if cost_model.device_card is not None else None
            ),
            "expand_actions": sum(1 for s in steps if s.action == "expand"),
            "refine_actions": sum(1 for s in steps if s.action == "refine_shots"),
            "acquisition_seconds": time.perf_counter() - started,
        }
    )
    return AcquisitionResult(
        labels=tuple(bank.generator(i).label for i in basis),
        energy=result.ground_energy,
        std_error=float(interval.std_error or 0.0),
        ci_interval=interval,
        estimated_remaining_bias=bias_upper,
        total_error_estimate=final_rmse,
        target_accuracy=target_accuracy,
        converged=converged,
        stopped_reason=stopped_reason,
        history=tuple(steps),
        result=result,
        bank=bank,
        cache=cache,
        indices=tuple(basis),
        total_shots=cache.total_shots,
        total_circuits=cache.total_circuits,
        qwc_groups=cache.num_groups(),
        total_time_us=total_time_us,
        evidence=EvidenceLevel.ASYMPTOTIC.value,
        resources=resources,
    )


__all__ = [
    "AcquisitionResult",
    "AcquisitionStep",
    "IncrementalCostBreakdown",
    "IncrementalCostModel",
    "InformationScore",
    "acquire_to_accuracy",
    "basis_dict_to_code",
    "estimate_incremental_cost",
    "expected_lowering_estimate",
    "qwc_bases_for_words",
    "qwc_setting_resources",
    "retained_sector_certificate",
    "run_resource_aware_acase",
    "score_information_candidate",
    "select_information_candidate",
]
