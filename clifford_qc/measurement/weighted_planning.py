"""Variance-first, Ritz-weighted, and noise-gated Pauli measurement planning.

Historical block-commuting records used unweighted largest-conflict-degree
colouring and Stim tableau elimination (:func:`compile_block_measurement_plan`
with ``method="elimination"``).  This module adds the variance-oriented layer
above the same verified block-commuting / :math:`k`-commuting compilation
contract (DalFavero et al., Phys. Rev. A 112, 2025, arXiv:2312.11840):

- **Weighted sorted insertion** (:func:`sorted_insertion_partition`), ordering
  words by descending target weight magnitude so heavy observables pack into
  fewer measurement settings (Crawford et al., Quantum 5, 385, 2021).
- **Ritz-sensitivity weights** (:func:`ritz_sensitivity_weights`), deriving
  per-word weights ``g_w`` directly from the first-order Ritz functional
  ``c†(H - E S)c`` (and optional overlap-mode sensitivity) without touching any
  exact reference state or density matrix.
- **GALIC-style per-group noise gating** (:class:`NoiseGatePolicy`), accepting
  a commuting merge only while the compiled setting's routed two-qubit gate
  count stays within ``floor(log(1 - eps) / log(1 - p))`` (Burns et al.,
  arXiv:2409.00576), with optional device-card fidelity-floor enforcement.
- **Variance-first reporting** (:func:`evaluate_plan_variance`,
  :func:`compare_block_rungs`), reporting Crawford's optimal shot ratio
  ``R_hat``, state-independent and optional covariance-aware state-dependent
  optimal variance, two-qubit gate counts/depths, and noise-gate admissibility.
- **Graph-state default compilation** (:func:`plan_measurement`), defaulting to
  the NumPy-only ``method="graph_state"`` compiler with no Stim requirement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

from ..ir import PauliSum
from ..multivector import MV
from ..pauli_kernel import pauli_lane_mask, validate_word_code, word_mul
from .block_commuting import (
    block_ranges,
    conflict_masks,
    first_fit_partition,
    packed_conflicts,
)
from .block_synthesis import _local_xz, independent_codes, local_code
from .cache import GroupedWordCache
from .cost import (
    DeviceCard,
    SettingResources,
    _declared_multiplier,
    _routed_ceil,
    setting_fidelity,
)
from .functionals import WordFunctional, ritz_functional
from .graph_state import graph_diagonalizer, native_stats
from .planning import CompiledMeasurementPlan, compile_block_measurement_plan

GROUPING_STRATEGIES = ("sorted_insertion", "largest_degree")
WEIGHT_AGGREGATIONS = ("l2", "l1", "max")
_LOCAL_GATES = ("H", "S", "SDG")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _validate_codes(n: int, codes: Sequence[int]) -> tuple[int, ...]:
    n = _positive_int(n, "n")
    out: list[int] = []
    for code in codes:
        if isinstance(code, (bool, np.bool_)) or not isinstance(code, Integral):
            raise TypeError("word codes must be integers")
        int_code = int(code)
        validate_word_code(n, int_code)
        out.append(int_code)
    if len(set(out)) != len(out):
        raise ValueError("word codes must be distinct")
    return tuple(out)


@dataclass(frozen=True)
class NoiseGatePolicy:
    """GALIC-style per-group two-qubit gate acceptance rule.

    Under independent two-qubit error rate ``p = two_qubit_error`` and target
    relative bias ``eps = max_relative_bias``, a setting with routed two-qubit
    gate count ``N_2q`` satisfies ``(1 - p)**N_2q >= 1 - eps`` iff
    ``N_2q <= floor(log(1 - eps) / log(1 - p))`` (Burns et al.,
    arXiv:2409.00576).  Qubit-wise commuting settings have ``N_2q == 0`` and
    always satisfy the two-qubit gate bound.  Optionally, supplying
    ``device_card`` with ``enforce_fidelity_floor=True`` also requires
    ``setting_fidelity(device_card, resources, n_qubits) >=
    device_card.fidelity_floor``.
    """

    two_qubit_error: float
    max_relative_bias: float = 0.05
    routing_2q_multiplier: float = 1.0
    device_card: DeviceCard | None = None
    enforce_fidelity_floor: bool = False

    def __post_init__(self) -> None:
        p = _finite_real(self.two_qubit_error, "two_qubit_error")
        if not 0.0 <= p < 1.0:
            raise ValueError("two_qubit_error must lie in [0, 1)")
        eps = _finite_real(self.max_relative_bias, "max_relative_bias")
        if not 0.0 < eps < 1.0:
            raise ValueError("max_relative_bias must lie in (0, 1)")
        mult = _finite_real(self.routing_2q_multiplier, "routing_2q_multiplier")
        if mult < 1.0:
            raise ValueError("routing_2q_multiplier must be at least 1")
        if type(self.enforce_fidelity_floor) is not bool:
            raise TypeError("enforce_fidelity_floor must be a boolean")
        if self.device_card is not None and not isinstance(self.device_card, DeviceCard):
            raise TypeError("device_card must be a DeviceCard")
        if self.enforce_fidelity_floor and self.device_card is None:
            raise ValueError("enforce_fidelity_floor requires a device_card")
        object.__setattr__(self, "two_qubit_error", p)
        object.__setattr__(self, "max_relative_bias", eps)
        object.__setattr__(self, "routing_2q_multiplier", mult)

    @classmethod
    def from_device_card(
        cls,
        card: DeviceCard,
        *,
        max_relative_bias: float = 0.05,
        enforce_fidelity_floor: bool = False,
    ) -> "NoiseGatePolicy":
        """Build a noise gate from a declared :class:`DeviceCard`."""
        if not isinstance(card, DeviceCard):
            raise TypeError("card must be a DeviceCard")
        return cls(
            two_qubit_error=card.eps_2q,
            max_relative_bias=max_relative_bias,
            routing_2q_multiplier=card.routing_2q_multiplier,
            device_card=card,
            enforce_fidelity_floor=enforce_fidelity_floor,
        )

    @property
    def max_routed_two_qubit_gates(self) -> int | None:
        """Largest routed 2Q gate count ``k`` with ``(1 - p)**k >= 1 - eps``."""
        if self.two_qubit_error == 0.0:
            return None
        log_survive = math.log1p(-self.two_qubit_error)
        log_target = math.log1p(-self.max_relative_bias)
        cap = max(0, int(math.floor(log_target / log_survive)))
        # Refine by at most one step against floating-point boundary roundoff.
        if math.exp((cap + 1) * log_survive) >= 1.0 - self.max_relative_bias:
            cap += 1
        elif cap > 0 and math.exp(cap * log_survive) < 1.0 - self.max_relative_bias:
            cap -= 1
        return cap

    @property
    def max_logical_two_qubit_gates(self) -> int | None:
        """Largest logical 2Q gate count whose routed count stays within budget."""
        routed_cap = self.max_routed_two_qubit_gates
        if routed_cap is None:
            return None
        if self.routing_2q_multiplier == 1.0:
            return routed_cap
        ratio = _declared_multiplier(self.routing_2q_multiplier)
        # ceil(g * num / den) <= routed_cap  iff  g * num <= routed_cap * den.
        return (routed_cap * ratio.denominator) // ratio.numerator

    def admits(self, setting: SettingResources, n_qubits: int) -> bool:
        """Whether one compiled setting satisfies this noise gate."""
        n_qubits = _positive_int(n_qubits, "n_qubits")
        routed_cap = self.max_routed_two_qubit_gates
        if routed_cap is not None:
            if _routed_ceil(setting.n_2q, self.routing_2q_multiplier) > routed_cap:
                return False
        if self.enforce_fidelity_floor:
            assert self.device_card is not None
            if setting_fidelity(self.device_card, setting, n_qubits) < self.device_card.fidelity_floor:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "two_qubit_error": self.two_qubit_error,
            "max_relative_bias": self.max_relative_bias,
            "routing_2q_multiplier": self.routing_2q_multiplier,
            "max_routed_two_qubit_gates": self.max_routed_two_qubit_gates,
            "max_logical_two_qubit_gates": self.max_logical_two_qubit_gates,
            "device_card": None if self.device_card is None else self.device_card.name,
            "device_card_sha256": None if self.device_card is None else self.device_card.sha256,
            "enforce_fidelity_floor": self.enforce_fidelity_floor,
        }


@dataclass(frozen=True)
class GroupingMetrics:
    """Variance-first and circuit-resource summary for one compiled plan."""

    block_size: int
    strategy: str
    method: str
    n_words: int
    n_settings: int
    weighted_words: int
    r_hat: float
    ungrouped_variance: float
    grouped_variance: float
    two_qubit_gates_total: int
    two_qubit_gates_mean: float
    two_qubit_gates_max: int
    two_qubit_depth_mean: float
    two_qubit_depth_max: int
    readable_pairs: int
    mean_readers_per_word: float
    state_dependent_variance: float | None = None
    state_dependent_ungrouped_variance: float | None = None
    state_dependent_r: float | None = None
    state_dependent_source: str | None = None
    noise_gate_admissible_settings: int | None = None
    noise_gate_max_routed_2q: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _single_signed_coefficients(target: Any) -> dict[int, float]:
    """Extract real non-identity Pauli coefficients from one linear target."""
    if isinstance(target, WordFunctional):
        return {
            int(code): float(val)
            for code, val in target.coefficients.items()
            if code != 0 and float(val) != 0.0
        }
    if isinstance(target, (MV, PauliSum)):
        out: dict[int, float] = {}
        for code, val in target.terms.items():
            if code == 0:
                continue
            if abs(val.imag) > 1e-12:
                raise ValueError("operator target must have real Pauli coefficients (Hermitian)")
            if val.real != 0.0:
                out[int(code)] = float(val.real)
        return out
    if isinstance(target, Mapping):
        out = {}
        for code, val in target.items():
            if isinstance(code, (bool, np.bool_)) or not isinstance(code, Integral) or int(code) < 0:
                raise ValueError("weight mapping keys must be nonnegative integer Pauli codes")
            if int(code) == 0:
                continue
            if isinstance(val, (bool, np.bool_)):
                raise ValueError("weight values must be numeric, not bool")
            if isinstance(val, Real):
                number = float(val)
                if not math.isfinite(number):
                    raise ValueError("weight values must be finite")
                if number != 0.0:
                    out[int(code)] = number
            elif isinstance(val, (complex, np.complexfloating)):
                cval = complex(val)
                if not (math.isfinite(cval.real) and math.isfinite(cval.imag)):
                    raise ValueError("weight values must be finite")
                if abs(cval.imag) > 1e-12:
                    raise ValueError("signed target mapping requires real coefficients")
                if cval.real != 0.0:
                    out[int(code)] = float(cval.real)
            else:
                raise TypeError(f"unsupported weight value type: {type(val).__name__}")
        return out
    raise TypeError(
        "target must be a WordFunctional, MV, PauliSum, mapping, or sequence of them"
    )


def _single_magnitudes(target: Any) -> dict[int, float]:
    """Extract nonnegative non-identity magnitudes from one target."""
    if isinstance(target, WordFunctional):
        return {
            int(code): abs(float(val))
            for code, val in target.coefficients.items()
            if code != 0 and val != 0.0
        }
    if isinstance(target, (MV, PauliSum)):
        return {
            int(code): abs(complex(val))
            for code, val in target.terms.items()
            if code != 0 and val != 0
        }
    if isinstance(target, Mapping):
        out: dict[int, float] = {}
        for code, val in target.items():
            if isinstance(code, (bool, np.bool_)) or not isinstance(code, Integral) or int(code) < 0:
                raise ValueError("weight mapping keys must be nonnegative integer Pauli codes")
            if int(code) == 0:
                continue
            if isinstance(val, (bool, np.bool_)):
                raise ValueError("weight values must be numeric, not bool")
            if isinstance(val, Real):
                mag = abs(float(val))
            elif isinstance(val, (complex, np.complexfloating)):
                cval = complex(val)
                if not (math.isfinite(cval.real) and math.isfinite(cval.imag)):
                    raise ValueError("weight values must be finite")
                mag = abs(cval)
            else:
                raise TypeError(f"unsupported weight value type: {type(val).__name__}")
            if not math.isfinite(mag):
                raise ValueError("weight values must be finite")
            if mag > 0.0:
                out[int(code)] = mag
        return out
    raise TypeError(
        "target must be a WordFunctional, MV, PauliSum, mapping, or sequence of them"
    )


def _is_target_sequence(target: Any) -> bool:
    return (
        isinstance(target, Sequence)
        and not isinstance(target, (str, bytes, bytearray, MV, PauliSum, WordFunctional))
    )


def word_weights(
    target: Any,
    *,
    aggregation: str = "l2",
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """Extract nonnegative per-word grouping weights from one or more targets.

    ``target`` may be a :class:`WordFunctional` (such as
    :func:`~clifford_qc.measurement.functionals.ritz_functional`), an
    :class:`~clifford_qc.multivector.MV`, a :class:`~clifford_qc.ir.PauliSum`,
    a ``{code: weight}`` mapping, or a sequence of targets combined with
    ``aggregation`` (``"l2"``, ``"l1"``, or ``"max"``).  The identity code ``0``
    is excluded because its expectation is deterministic.
    """
    if aggregation not in WEIGHT_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {WEIGHT_AGGREGATIONS}")
    if not _is_target_sequence(target):
        if weights is not None:
            raise ValueError("weights applies only when target is a sequence of targets")
        return _single_magnitudes(target)
    targets = list(target)
    if not targets:
        return {}
    if weights is None:
        scales = [1.0] * len(targets)
    else:
        scales = [_finite_real(w, "target weight") for w in weights]
        if len(scales) != len(targets):
            raise ValueError("weights length must match target sequence length")
        if any(s < 0.0 for s in scales):
            raise ValueError("target weights must be nonnegative")
    combined: dict[int, float] = {}
    for item, scale in zip(targets, scales):
        if scale == 0.0:
            continue
        mags = _single_magnitudes(item)
        for code, mag in mags.items():
            scaled = scale * mag
            if aggregation == "l2":
                combined[code] = combined.get(code, 0.0) + scaled * scaled
            elif aggregation == "l1":
                combined[code] = combined.get(code, 0.0) + scaled
            else:
                prev = combined.get(code, 0.0)
                if scaled > prev:
                    combined[code] = scaled
    if aggregation == "l2":
        return {code: math.sqrt(val) for code, val in combined.items() if val > 0.0}
    return {code: val for code, val in combined.items() if val > 0.0}


def ritz_sensitivity_weights(
    bank: Any,
    indices: Sequence[int] | None,
    ritz_vector: Sequence[complex] | np.ndarray,
    energy: float,
    *,
    overlap_functional: WordFunctional | None = None,
    overlap_weight: float = 0.0,
    aggregation: str = "l2",
) -> dict[int, float]:
    """First-order Ritz-value per-word sensitivity weights ``|g_w|``.

    Uses only the algebraic operator bank coefficients and the supplied
    ``(ritz_vector, energy)`` (for example, from an independent pilot solve);
    no exact reference expectation or density matrix is touched.  Optionally
    blends in an overlap-mode sensitivity functional (such as the weakest-mode
    derivative from :func:`~clifford_qc.subspace.rank_stopping.rank_margin`)
    with nonnegative ``overlap_weight``.
    """
    resolved = bank.resolve(indices)
    energy_val = _finite_real(energy, "energy")
    vec = np.asarray(ritz_vector, dtype=complex)
    functional = ritz_functional(bank, resolved, vec, energy_val)
    ov_weight = _finite_real(overlap_weight, "overlap_weight")
    if ov_weight < 0.0:
        raise ValueError("overlap_weight must be nonnegative")
    if overlap_functional is None or ov_weight == 0.0:
        return word_weights(functional)
    if not isinstance(overlap_functional, WordFunctional):
        raise TypeError("overlap_functional must be a WordFunctional")
    return word_weights(
        (functional, overlap_functional),
        aggregation=aggregation,
        weights=(1.0, ov_weight),
    )


def _qwc_compatible(a: int, b: int, lane_mask: int) -> bool:
    nz_a = (a & lane_mask) | ((a >> 1) & lane_mask)
    nz_b = (b & lane_mask) | ((b >> 1) & lane_mask)
    diff = a ^ b
    nz_diff = (diff & lane_mask) | ((diff >> 1) & lane_mask)
    return (nz_a & nz_b & nz_diff) == 0


def _qwc_block_stats(merged_basis: int, size: int) -> tuple[int, int, int, int]:
    """Exact (n_1q, n_2q, d_1q, d_2q) for a QWC local block basis."""
    n_1q = 0
    has_x = False
    has_y = False
    for q in range(size):
        letter = (merged_basis >> (2 * q)) & 3
        if letter == 1:
            n_1q += 1
            has_x = True
        elif letter == 2:
            n_1q += 2
            has_y = True
    d_1q = 2 if has_y else (1 if has_x else 0)
    return n_1q, 0, d_1q, 0


def _in_gf2_span(code: int, pivots: Mapping[int, int], size: int) -> bool:
    reduced = _local_xz(code, size)
    for bit in sorted(pivots, reverse=True):
        if (reduced >> bit) & 1:
            reduced ^= pivots[bit]
    return reduced == 0


def _gf2_pivots(basis: Sequence[int], size: int) -> dict[int, int]:
    pivots: dict[int, int] = {}
    for code in basis:
        reduced = _local_xz(code, size)
        for bit in sorted(pivots, reverse=True):
            if (reduced >> bit) & 1:
                reduced ^= pivots[bit]
        if reduced:
            pivots[reduced.bit_length() - 1] = reduced
    return pivots


def _noise_gated_first_fit(
    n: int,
    codes: tuple[int, ...],
    block_size: int,
    packed: np.ndarray,
    order: Sequence[int],
    noise_gate: NoiseGatePolicy,
) -> list[list[int]]:
    """First-fit k-commuting insertion admitting only noise-gated merges."""
    width, lanes = packed.shape
    if not width:
        return []
    ranges = tuple(block_ranges(n, block_size))
    n_blocks = len(ranges)
    lane_masks = tuple(pauli_lane_mask(size) for _, size in ranges)
    local_by_block = tuple(
        tuple(local_code(code, start, size) for code in codes)
        for start, size in ranges
    )
    diag_cache: dict[tuple[int, tuple[int, ...]], tuple[int, int, int, int]] = {}

    def block_stats(size: int, basis: tuple[int, ...]) -> tuple[int, int, int, int]:
        key = (size, basis)
        cached = diag_cache.get(key)
        if cached is not None:
            return cached
        diagonalizer = graph_diagonalizer(basis, size)
        counts, _critical, d1, d2 = native_stats(diagonalizer)
        stats = (
            sum(int(counts[g]) for g in _LOCAL_GATES),
            int(counts["CZ"]),
            int(d1),
            int(d2),
        )
        diag_cache[key] = stats
        return stats

    # Per-group state arrays (mirrors compile_graph_settings exactly).
    groups: list[list[int]] = []
    group_code_sets: list[list[set[int]]] = []
    group_qwc: list[list[bool]] = []
    group_qwc_basis: list[list[int]] = []
    group_basis: list[list[tuple[int, ...]]] = []
    group_pivots: list[list[dict[int, int]]] = []
    group_block_stats: list[list[tuple[int, int, int, int]]] = []

    capacity = 8
    masks = np.zeros((capacity, lanes), dtype=np.uint64)
    open_groups = 0

    for idx in order:
        conflict = packed[idx]
        word_locals = [local_by_block[b][idx] for b in range(n_blocks)]
        chosen = -1
        chosen_updates: dict[
            int, tuple[bool, int, tuple[int, ...], dict[int, int], tuple[int, int, int, int]]
        ] = {}

        if open_groups:
            blocked = np.bitwise_and(masks[:open_groups], conflict).any(axis=1)
            candidates = np.flatnonzero(~blocked)
            for cand_raw in candidates:
                g = int(cand_raw)
                g_sets = group_code_sets[g]
                g_qwc = group_qwc[g]
                g_qwc_basis = group_qwc_basis[g]
                g_basis = group_basis[g]
                g_pivots = group_pivots[g]
                g_stats = group_block_stats[g]

                updates: dict[
                    int,
                    tuple[bool, int, tuple[int, ...], dict[int, int], tuple[int, int, int, int]],
                ] = {}
                total_1q = 0
                total_2q = 0
                max_d1 = 0
                max_d2 = 0

                for b in range(n_blocks):
                    lc = word_locals[b]
                    old_stat = g_stats[b]
                    if lc == 0 or lc in g_sets[b]:
                        n1, n2, d1, d2 = old_stat
                    elif g_qwc[b] and _qwc_compatible(g_qwc_basis[b], lc, lane_masks[b]):
                        merged = g_qwc_basis[b] | lc
                        new_stat = _qwc_block_stats(merged, ranges[b][1])
                        updates[b] = (True, merged, g_basis[b], g_pivots[b], new_stat)
                        n1, n2, d1, d2 = new_stat
                    else:
                        size = ranges[b][1]
                        prev_basis = g_basis[b]
                        if (
                            not g_qwc[b]
                            and prev_basis
                            and lc > prev_basis[-1]
                            and _in_gf2_span(lc, g_pivots[b], size)
                        ):
                            new_basis = prev_basis
                            new_pivots = g_pivots[b]
                            new_stat = old_stat
                        else:
                            new_basis = tuple(
                                independent_codes(sorted(g_sets[b] | {lc}), size)
                            )
                            new_pivots = _gf2_pivots(new_basis, size)
                            new_stat = block_stats(size, new_basis)
                        updates[b] = (False, 0, new_basis, new_pivots, new_stat)
                        n1, n2, d1, d2 = new_stat
                    total_1q += n1
                    total_2q += n2
                    if d1 > max_d1:
                        max_d1 = d1
                    if d2 > max_d2:
                        max_d2 = d2

                candidate_res = SettingResources(
                    n_1q=total_1q, n_2q=total_2q, d_1q=max_d1, d_2q=max_d2
                )
                if noise_gate.admits(candidate_res, n):
                    chosen = g
                    chosen_updates = updates
                    break

        if chosen < 0:
            if open_groups == capacity:
                capacity *= 2
                grown = np.zeros((capacity, lanes), dtype=np.uint64)
                grown[:open_groups] = masks[:open_groups]
                masks = grown
            chosen = open_groups
            open_groups += 1
            groups.append([idx])
            new_sets: list[set[int]] = []
            new_qwc: list[bool] = []
            new_qwc_basis: list[int] = []
            new_bases: list[tuple[int, ...]] = []
            new_pivots_list: list[dict[int, int]] = []
            new_stats: list[tuple[int, int, int, int]] = []
            for b in range(n_blocks):
                lc = word_locals[b]
                size = ranges[b][1]
                if lc == 0:
                    new_sets.append(set())
                    new_qwc.append(True)
                    new_qwc_basis.append(0)
                    new_bases.append(())
                    new_pivots_list.append({})
                    new_stats.append((0, 0, 0, 0))
                else:
                    new_sets.append({lc})
                    new_qwc.append(True)
                    new_qwc_basis.append(lc)
                    new_bases.append((lc,))
                    new_pivots_list.append(_gf2_pivots((lc,), size))
                    new_stats.append(_qwc_block_stats(lc, size))
            group_code_sets.append(new_sets)
            group_qwc.append(new_qwc)
            group_qwc_basis.append(new_qwc_basis)
            group_basis.append(new_bases)
            group_pivots.append(new_pivots_list)
            group_block_stats.append(new_stats)
        else:
            groups[chosen].append(idx)
            for b in range(n_blocks):
                lc = word_locals[b]
                if lc != 0:
                    group_code_sets[chosen][b].add(lc)
            for b, (is_qwc, qbasis, basis, pivots, stat) in chosen_updates.items():
                group_qwc[chosen][b] = is_qwc
                group_qwc_basis[chosen][b] = qbasis
                group_basis[chosen][b] = basis
                group_pivots[chosen][b] = pivots
                group_block_stats[chosen][b] = stat

        masks[chosen, idx >> 6] |= np.uint64(1) << np.uint64(idx & 63)

    # Independent grouping invariant: no member conflicts with its group mask.
    for index, members in enumerate(groups):
        member_mask = masks[index]
        for member in members:
            if bool(np.bitwise_and(packed[member], member_mask).any()):
                raise AssertionError("noise-gated block-commuting partition invariant failed")
    return groups


def sorted_insertion_partition(
    n: int,
    codes: Sequence[int],
    block_size: int,
    weights: Any = None,
    *,
    noise_gate: NoiseGatePolicy | None = None,
) -> list[list[int]]:
    """Weighted ``k``-commuting Sorted Insertion partition of ``codes``.

    Words are inserted into the first compatible ``k``-commuting setting
    (DalFavero et al., Phys. Rev. A 112, 2025) in descending target weight
    magnitude (Crawford et al., Quantum 5, 385, 2021), breaking ties by
    descending conflict degree and then ascending Pauli code.  When ``weights``
    is ``None`` and ``noise_gate`` is ``None``, the ordering matches
    :func:`~clifford_qc.measurement.block_commuting.block_commuting_partition`.

    When ``noise_gate`` is supplied, a candidate merge into an existing group
    is admitted only if the merged group's compiled ``graph_state`` circuit
    satisfies :meth:`NoiseGatePolicy.admits`.
    """
    n = _positive_int(n, "n")
    block_size = _positive_int(block_size, "block_size")
    validated = _validate_codes(n, codes)
    if noise_gate is not None and not isinstance(noise_gate, NoiseGatePolicy):
        raise TypeError("noise_gate must be a NoiseGatePolicy")
    if not validated:
        return []
    weight_map = {} if weights is None else word_weights(weights)
    conflicts, degrees = conflict_masks(n, validated, block_size)
    packed = packed_conflicts(conflicts)
    order = sorted(
        range(len(validated)),
        key=lambda i: (-weight_map.get(validated[i], 0.0), -int(degrees[i]), validated[i]),
    )
    gate_active = noise_gate is not None and (
        noise_gate.max_routed_two_qubit_gates is not None
        or noise_gate.enforce_fidelity_floor
    )
    if not gate_active or (
        block_size == 1 and noise_gate is not None and not noise_gate.enforce_fidelity_floor
    ):
        return first_fit_partition(packed, order)
    assert noise_gate is not None
    return _noise_gated_first_fit(n, validated, block_size, packed, order, noise_gate)


def plan_measurement(
    n: int,
    codes: Sequence[int],
    block_size: int,
    *,
    weights: Any = None,
    strategy: str = "sorted_insertion",
    method: str = "graph_state",
    noise_gate: NoiseGatePolicy | None = None,
    groups: Sequence[Sequence[int]] | None = None,
    spin_conserving_jw_hamiltonian: MV | PauliSum | None = None,
) -> CompiledMeasurementPlan:
    """Preferred measurement planner: ``graph_state`` default, weighted & noise-gated.

    Unlike the historical :func:`compile_block_measurement_plan` (which keeps
    ``method="elimination"`` for frozen record replay), this entry point
    defaults to the NumPy-only ``method="graph_state"`` compiler (no Stim
    dependency) and supports coefficient- or Ritz-weighted ``k``-commuting
    sorted insertion with an optional GALIC-style :class:`NoiseGatePolicy`.
    """
    if strategy not in GROUPING_STRATEGIES:
        raise ValueError(f"strategy must be one of {GROUPING_STRATEGIES}")
    if noise_gate is not None and not isinstance(noise_gate, NoiseGatePolicy):
        raise TypeError("noise_gate must be a NoiseGatePolicy")
    if groups is not None and weights is not None:
        raise ValueError("supply either explicit groups or grouping weights, not both")
    n = _positive_int(n, "n")
    block_size = _positive_int(block_size, "block_size")
    validated = _validate_codes(n, codes)
    if noise_gate is not None and method != "graph_state" and groups is None:
        raise ValueError("noise-gated grouping uses the graph_state readout compiler")

    chosen_groups: Sequence[Sequence[int]] | None
    if groups is not None:
        chosen_groups = groups
    elif strategy == "sorted_insertion":
        chosen_groups = sorted_insertion_partition(
            n, validated, block_size, weights=weights, noise_gate=noise_gate
        )
    else:
        if noise_gate is not None:
            chosen_groups = sorted_insertion_partition(
                n, validated, block_size, weights=None, noise_gate=noise_gate
            )
        else:
            chosen_groups = None

    plan = compile_block_measurement_plan(
        n,
        validated,
        block_size,
        groups=chosen_groups,
        method=method,
        spin_conserving_jw_hamiltonian=spin_conserving_jw_hamiltonian,
    )
    if noise_gate is not None:
        # Every merged group must satisfy the gate (singletons are irreducible
        # when a device card's fidelity floor rejects even QWC single words).
        for group, resource in zip(plan.groups, plan.resources):
            if len(group) > 1 and not noise_gate.admits(resource, n):
                raise ValueError(
                    "compiled multi-word setting exceeds the declared NoiseGatePolicy"
                )
            if (
                not noise_gate.enforce_fidelity_floor
                and not noise_gate.admits(resource, n)
            ):
                raise ValueError(
                    "compiled setting exceeds the declared two-qubit NoiseGatePolicy"
                )
    return plan


def _exact_expectation(rho: MV, code: int) -> float:
    value = rho.terms.get(code)
    return 0.0 if value is None else float((2 ** rho.n) * value.real)


def _exact_group_variance(
    n: int, group_coeffs: Mapping[int, float], reference: MV
) -> float:
    """Exact one-shot variance of ``sum_{w in g} a_w P_w`` on ``reference``."""
    items = [(int(code), float(coeff)) for code, coeff in group_coeffs.items() if coeff != 0.0]
    if not items:
        return 0.0
    means = [_exact_expectation(reference, code) for code, _ in items]
    variance = 0.0
    for i, (code_i, ci) in enumerate(items):
        mi = means[i]
        variance += ci * ci * max(0.0, 1.0 - mi * mi)
        for j in range(i + 1, len(items)):
            code_j, cj = items[j]
            phase, prod = word_mul(n, code_i, code_j)
            if abs(phase.imag) > 1e-12:
                raise ValueError("group words must pairwise commute for exact covariance")
            cov = phase.real * _exact_expectation(reference, prod) - mi * means[j]
            variance += 2.0 * ci * cj * cov
    return max(0.0, float(variance))


def _state_dependent_from_exact(
    plan: CompiledMeasurementPlan,
    target_list: list[dict[int, float]],
    reference: MV,
) -> tuple[float, float, float]:
    if reference.n != plan.n:
        raise ValueError("reference and measurement plan have different qubit counts")
    group_sigmas = 0.0
    for setting in plan.settings:
        assigned = set(setting.assigned_word_codes)
        v_g = 0.0
        for coeffs in target_list:
            sub = {c: v for c, v in coeffs.items() if c in assigned}
            if sub:
                v_g += _exact_group_variance(plan.n, sub, reference)
        group_sigmas += math.sqrt(max(0.0, v_g))
    grouped_var = group_sigmas * group_sigmas

    all_codes = sorted({c for coeffs in target_list for c in coeffs})
    ungrouped_sigmas = 0.0
    for code in all_codes:
        mu = _exact_expectation(reference, code)
        word_var = max(0.0, 1.0 - mu * mu)
        coeff_sq = sum(coeffs.get(code, 0.0) ** 2 for coeffs in target_list)
        ungrouped_sigmas += math.sqrt(coeff_sq * word_var)
    ungrouped_var = ungrouped_sigmas * ungrouped_sigmas
    ratio = _variance_ratio(ungrouped_var, grouped_var)
    return grouped_var, ungrouped_var, ratio


def _state_dependent_from_cache(
    plan: CompiledMeasurementPlan,
    target_list: list[dict[int, float]],
    cache: GroupedWordCache,
) -> tuple[float, float, float]:
    if cache.n != plan.n:
        raise ValueError("cache and measurement plan have different qubit counts")
    setting_keys = [setting.key for setting in plan.settings]
    key_set = set(setting_keys)
    per_setting_var: dict[tuple, float] = {key: 0.0 for key in setting_keys}
    for coeffs in target_list:
        if not coeffs:
            continue
        stats = cache.candidate_group_statistics(coeffs)
        if stats is None:
            raise ValueError("cache is missing shots for words in target")
        if not set(stats).issubset(key_set):
            raise ValueError("cache settings do not match the compiled plan")
        for key, (_shots, var, _rng) in stats.items():
            per_setting_var[key] += max(0.0, float(var))
    grouped_sigmas = sum(math.sqrt(max(0.0, v)) for v in per_setting_var.values())
    grouped_var = grouped_sigmas * grouped_sigmas

    all_codes = sorted({c for coeffs in target_list for c in coeffs})
    ungrouped_sigmas = 0.0
    for code in all_codes:
        word_stats = cache.candidate_group_statistics({code: 1.0})
        if word_stats is None:
            raise ValueError("cache is missing shots for words in target")
        # For pooling='assigned' there is one term with exact single-word var;
        # for pooling='shots', average the per-shot word variance weighted by shots.
        total_n = sum(n_g for n_g, _, _ in word_stats.values())
        if cache.pooling == "assigned":
            word_var = sum(max(0.0, float(v)) for _, v, _ in word_stats.values())
        else:
            # Under shots pooling each reading group has the same marginal law;
            # compute marginal empirical variance over pooled shots with Jeffreys floor.
            mu = cache.candidate_estimate({code: 1.0})
            word_var = max(max(0.0, 1.0 - mu * mu), 1.0 / (total_n + 1.0))
        coeff_sq = sum(coeffs.get(code, 0.0) ** 2 for coeffs in target_list)
        ungrouped_sigmas += math.sqrt(coeff_sq * word_var)
    ungrouped_var = ungrouped_sigmas * ungrouped_sigmas
    ratio = _variance_ratio(ungrouped_var, grouped_var)
    return grouped_var, ungrouped_var, ratio


def _variance_ratio(ungrouped_var: float, grouped_var: float) -> float:
    if grouped_var <= 0.0:
        return 1.0 if ungrouped_var <= 0.0 else float("inf")
    return ungrouped_var / grouped_var


def evaluate_plan_variance(
    plan: CompiledMeasurementPlan,
    target: Any,
    *,
    cache: GroupedWordCache | None = None,
    reference: MV | None = None,
    strategy: str = "custom",
    noise_gate: NoiseGatePolicy | None = None,
    aggregation: str = "l2",
    weights: Sequence[float] | None = None,
) -> GroupingMetrics:
    """Evaluate Crawford's ``R_hat``, optimal variance, and 2Q gate metrics.

    ``R_hat`` is the state-independent optimal shot ratio
    ``(sum_w |a_w| / sum_g sqrt(sum_{w in g} a_w^2))^2`` under disjoint
    assignment (Crawford et al., Quantum 5, 385, 2021; DalFavero et al.,
    Phys. Rev. A 112, 2025).  Supplying ``cache`` (from measured joint shots)
    or ``reference`` (exact ``MV`` density for offline diagnostics) additionally
    computes the covariance-aware state-dependent optimal variance and shot
    ratio ``R``.  Pass at most one of ``cache`` and ``reference``.
    """
    if not isinstance(plan, CompiledMeasurementPlan):
        raise TypeError("plan must be a CompiledMeasurementPlan")
    if cache is not None and reference is not None:
        raise ValueError("pass at most one of cache and reference")
    if cache is not None and not isinstance(cache, GroupedWordCache):
        raise TypeError("cache must be a GroupedWordCache")
    if reference is not None and not isinstance(reference, MV):
        raise TypeError("reference must be an MV density")
    if noise_gate is not None and not isinstance(noise_gate, NoiseGatePolicy):
        raise TypeError("noise_gate must be a NoiseGatePolicy")

    wmap = word_weights(target, aggregation=aggregation, weights=weights)
    plan_codes = set(plan.codes)
    missing = set(wmap) - plan_codes
    if missing:
        raise ValueError("target contains Pauli words not covered by the measurement plan")

    ungrouped_sigma = sum(wmap.values())
    ungrouped_var = ungrouped_sigma * ungrouped_sigma
    grouped_sigma = 0.0
    for setting in plan.settings:
        sq_sum = sum(wmap[code] ** 2 for code in setting.assigned_word_codes if code in wmap)
        grouped_sigma += math.sqrt(sq_sum)
    grouped_var = grouped_sigma * grouped_sigma
    r_hat = _variance_ratio(ungrouped_var, grouped_var)

    n_settings = len(plan.settings)
    two_q = tuple(r.n_2q for r in plan.resources)
    depth_2q = tuple(r.d_2q for r in plan.resources)
    readable_pairs = int(plan.compatibility.sum()) if plan.compatibility.size else 0
    mean_readers = (readable_pairs / len(plan.codes)) if plan.codes else 0.0

    sd_var: float | None = None
    sd_ungrouped: float | None = None
    sd_r: float | None = None
    sd_source: str | None = None
    if cache is not None or reference is not None:
        if _is_target_sequence(target):
            target_items = list(target)
            if weights is None:
                scales = [1.0] * len(target_items)
            else:
                scales = [_finite_real(w, "target weight") for w in weights]
                if len(scales) != len(target_items):
                    raise ValueError("weights length must match target sequence length")
            signed_targets = [
                {code: scale * val for code, val in _single_signed_coefficients(item).items()}
                for item, scale in zip(target_items, scales)
                if scale > 0.0
            ]
        else:
            signed_targets = [_single_signed_coefficients(target)]
        if cache is not None:
            sd_var, sd_ungrouped, sd_r = _state_dependent_from_cache(
                plan, signed_targets, cache
            )
            sd_source = f"grouped_cache_{cache.pooling}"
        else:
            assert reference is not None
            sd_var, sd_ungrouped, sd_r = _state_dependent_from_exact(
                plan, signed_targets, reference
            )
            sd_source = "exact_reference_covariance"

    admissible: int | None = None
    max_routed: int | None = None
    if noise_gate is not None:
        admissible = sum(noise_gate.admits(res, plan.n) for res in plan.resources)
        max_routed = noise_gate.max_routed_two_qubit_gates

    return GroupingMetrics(
        block_size=plan.block_size,
        strategy=strategy,
        method=plan.method,
        n_words=len(plan.codes),
        n_settings=n_settings,
        weighted_words=len(wmap),
        r_hat=float(r_hat),
        ungrouped_variance=float(ungrouped_var),
        grouped_variance=float(grouped_var),
        two_qubit_gates_total=sum(two_q),
        two_qubit_gates_mean=(sum(two_q) / n_settings) if n_settings else 0.0,
        two_qubit_gates_max=max(two_q, default=0),
        two_qubit_depth_mean=(sum(depth_2q) / n_settings) if n_settings else 0.0,
        two_qubit_depth_max=max(depth_2q, default=0),
        readable_pairs=readable_pairs,
        mean_readers_per_word=float(mean_readers),
        state_dependent_variance=sd_var,
        state_dependent_ungrouped_variance=sd_ungrouped,
        state_dependent_r=sd_r,
        state_dependent_source=sd_source,
        noise_gate_admissible_settings=admissible,
        noise_gate_max_routed_2q=max_routed,
    )


def _default_rungs(n: int) -> tuple[int, ...]:
    rungs: list[int] = []
    k = 1
    while k < n:
        rungs.append(k)
        k *= 2
    if not rungs or rungs[-1] != n:
        rungs.append(n)
    return tuple(rungs)


def compare_block_rungs(
    n: int,
    codes: Sequence[int],
    target: Any,
    *,
    block_sizes: Sequence[int] | None = None,
    strategy: str = "sorted_insertion",
    method: str = "graph_state",
    noise_gate: NoiseGatePolicy | None = None,
    reference: MV | None = None,
    aggregation: str = "l2",
    weights: Sequence[float] | None = None,
) -> tuple[GroupingMetrics, ...]:
    """Evaluate variance (``R_hat`` / ``R``) and 2Q gate cost across ``k`` rungs.

    Default rungs are powers of two up to ``n`` (with ``n`` included), matching
    the dyadic ``k``-commutativity hierarchy while reporting optimal shot
    variance rather than setting counts alone.
    """
    n = _positive_int(n, "n")
    validated = _validate_codes(n, codes)
    rungs = (
        _default_rungs(n)
        if block_sizes is None
        else tuple(_positive_int(k, "block_size") for k in block_sizes)
    )
    if not rungs:
        raise ValueError("block_sizes must be non-empty")
    wmap = word_weights(target, aggregation=aggregation, weights=weights)
    results: list[GroupingMetrics] = []
    for k in rungs:
        plan = plan_measurement(
            n,
            validated,
            k,
            weights=wmap if strategy == "sorted_insertion" else None,
            strategy=strategy,
            method=method,
            noise_gate=noise_gate,
        )
        results.append(
            evaluate_plan_variance(
                plan,
                target,
                reference=reference,
                strategy=strategy,
                noise_gate=noise_gate,
                aggregation=aggregation,
                weights=weights,
            )
        )
    return tuple(results)


__all__ = [
    "GROUPING_STRATEGIES",
    "GroupingMetrics",
    "NoiseGatePolicy",
    "WEIGHT_AGGREGATIONS",
    "compare_block_rungs",
    "evaluate_plan_variance",
    "plan_measurement",
    "ritz_sensitivity_weights",
    "sorted_insertion_partition",
    "word_weights",
]
