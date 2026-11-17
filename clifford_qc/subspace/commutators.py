"""Hierarchical commutator operator generation and demand-driven expansion.

Starts from a seed family ``A_0 = {A_1, A_2, ...}`` and builds the nested
Lie-algebraic commutator hierarchy against ``H``:

    A_0 = {A_1, A_2, ...}
    A_1 = {[H, A_i]}
    A_2 = {[H, [H, A_i]]}
    ...

forming virtual A-CASE subspaces ``V_m = span{A_0|psi>, ..., A_m|psi>}``.

Why a dedicated module instead of reusing ``commutator_response``:

1. ``commutator_response`` operates only on single ``PauliWord`` inputs at
   depth 1 via ``CommutatorBank``. Hierarchical growth requires computing
   ``[H, A]`` on arbitrary multivector generators ``A`` (such as symmetry-
   conserving fermionic excitation multivectors, local spin operators, or
   prior commutator outputs) across arbitrary depths.
2. ``MV`` prunes coefficients with ``|c| <= 1e-12`` *absolutely* upon
   construction. Iterating nested commutators without normalizing between
   levels can either wipe out small physical directions or inflate large
   operators. Here sparse commutator accumulation happens in raw dictionaries
   first, checks zero commutators *relatively* to ``||H||_2 ||A||_2``, and
   canonically normalizes (when requested) before wrapping in ``MV``.
3. Operator growth is explicitly tracked and controllable: support (``S_A``),
   Pauli weight, dimensionless Lie-bracket norm ratio ``||[H, A]|| / (||H||||A||)``,
   collinear deduplication (Lie closure indicator), optional tail truncation,
   and demand-driven expansion (``CommutatorFrontier`` /
   ``run_dynamic_commutator_acase``) from accepted or high-value parents.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Any, Sequence

from ..measurement.bank import _pauli_anticommute
from ..multivector import MV
from ..pauli_kernel import _word_mul_unchecked
from .adaptive import (
    DEFAULT_MAX_CONDITION,
    DEFAULT_MIN_LOWERING,
    DEFAULT_MIN_ORTHOGONALITY,
    DEFAULT_TAU_S,
    ACASEConfig,
    ACASEState,
    ACASEStepOutcome,
    AdaptiveResult,
    GrowthRecord,
    _acase_objective,
    _require_matching_bank,
    acase_step,
    score_candidate,
    select_candidate,
)
from .contracts import as_multivector
from .elements import MatrixElementBank
from .generator_core import Generator, as_generators, scalar_free_key
from .generators import identity_generator
from .symmetry import sector_leakage

if TYPE_CHECKING:
    from .information import IncrementalCostModel

_PHASE_MODES = ("raw", "hermitian", "i_commutator")
_SUPPORT_POLICIES = ("drop", "truncate")
_EXPAND_MODES = ("on_select", "on_stall", "on_select_or_stall", "top_k")


def _max_pauli_weight(n: int, codes) -> tuple[int, float]:
    if not codes:
        return 0, 0.0
    weights = [sum(1 for j in range(n) if (code >> (2 * j)) & 3) for code in codes]
    return max(weights), float(sum(weights) / len(weights))


def _coeff_l2(terms: dict[int, complex]) -> float:
    return math.sqrt(sum(abs(v) ** 2 for v in terms.values()))


def _truncate_terms(
    terms: dict[int, complex],
    *,
    top_k: int | None = None,
    l2_keep: float | None = None,
    max_support: int | None = None,
) -> tuple[dict[int, complex], float]:
    """Deterministic tail truncation preserving dominant Pauli components."""
    total_sq = sum(abs(v) ** 2 for v in terms.values())
    if total_sq <= 0.0 or not terms:
        return {}, 0.0

    # Deterministic tie-breaking on equal magnitudes via word code.
    ranked = sorted(terms.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    limit = len(ranked)
    if top_k is not None:
        limit = min(limit, max(1, int(top_k)))
    if max_support is not None:
        limit = min(limit, max(1, int(max_support)))

    if l2_keep is not None:
        target_sq = float(l2_keep) * total_sq
        acc_sq = 0.0
        kept: list[tuple[int, complex]] = []
        for idx, item in enumerate(ranked):
            if idx >= limit:
                break
            kept.append(item)
            acc_sq += abs(item[1]) ** 2
            if acc_sq >= target_sq:
                break
    else:
        kept = ranked[:limit]

    kept_sq = sum(abs(v) ** 2 for _, v in kept)
    return dict(kept), float(kept_sq / total_sq)


def _apply_phase_mode(terms: dict[int, complex], phase_mode: str) -> dict[int, complex]:
    if phase_mode not in _PHASE_MODES:
        raise ValueError(f"phase_mode must be one of {_PHASE_MODES}, got {phase_mode!r}")
    if phase_mode == "raw":
        return terms
    if phase_mode == "i_commutator":
        return {code: -1j * coeff for code, coeff in terms.items()}
    # ``hermitian``: keep exact Hermiticity if the commutator is closer to
    # anti-Hermitian than Hermitian (which happens whenever A is Hermitian).
    sym_sq = sum(abs(c + c.conjugate()) ** 2 for c in terms.values())
    antisym_sq = sum(abs(c - c.conjugate()) ** 2 for c in terms.values())
    if sym_sq < antisym_sq:
        return {code: -1j * coeff for code, coeff in terms.items()}
    return terms


@dataclass(frozen=True)
class CommutatorNode:
    """One operator in a hierarchical commutator tree with algebraic metrics."""

    generator: Generator
    depth: int
    parent_label: str | None
    seed_label: str
    raw_coeff_norm: float
    raw_hs_norm: float
    growth_ratio: float
    raw_support: int
    retained_l2_fraction: float
    l1_norm: float
    max_pauli_weight: int
    mean_pauli_weight: float

    @property
    def label(self) -> str:
        return self.generator.label

    @property
    def mv(self) -> MV:
        return self.generator.mv

    @property
    def n(self) -> int:
        return self.generator.n

    def support(self) -> int:
        return self.generator.support()


@dataclass(frozen=True)
class DepthGrowthSummary:
    """Per-depth statistics tracking operator growth and collinear closure."""

    depth: int
    attempted: int
    kept: int
    vanished: int
    deduplicated: int
    dropped_by_filter: int
    closure_fraction: float
    mean_support: float
    max_support: int
    total_support: int
    mean_pauli_weight: float
    max_pauli_weight: int
    mean_growth_ratio: float
    max_growth_ratio: float


@dataclass(frozen=True)
class CommutatorHierarchy:
    """Complete hierarchical commutator tree ``A_0, A_1, ..., A_K``."""

    max_depth: int
    phase_mode: str
    levels: tuple[tuple[CommutatorNode, ...], ...]
    summaries: tuple[DepthGrowthSummary, ...]
    closure_reached: bool
    closure_depth: int | None

    @property
    def generators_by_depth(self) -> tuple[tuple[Generator, ...], ...]:
        return tuple(tuple(node.generator for node in level) for level in self.levels)

    @property
    def all_nodes(self) -> tuple[CommutatorNode, ...]:
        return tuple(node for level in self.levels for node in level)

    @property
    def all_generators(self) -> tuple[Generator, ...]:
        return tuple(node.generator for level in self.levels for node in level)

    def generators(
        self,
        *,
        max_depth: int | None = None,
        include_depth0: bool = True,
    ) -> list[Generator]:
        """Return flattened generators up to ``max_depth``."""
        start = 0 if include_depth0 else 1
        end = len(self.levels) if max_depth is None else min(len(self.levels), int(max_depth) + 1)
        out: list[Generator] = []
        for level in self.levels[start:end]:
            out.extend(node.generator for node in level)
        return out

    def summary_table(self) -> list[dict[str, Any]]:
        """JSON-serializable growth summary per depth."""
        return [
            {
                "depth": s.depth,
                "attempted": s.attempted,
                "kept": s.kept,
                "vanished": s.vanished,
                "deduplicated": s.deduplicated,
                "dropped_by_filter": s.dropped_by_filter,
                "closure_fraction": s.closure_fraction,
                "mean_support": s.mean_support,
                "max_support": s.max_support,
                "total_support": s.total_support,
                "mean_pauli_weight": s.mean_pauli_weight,
                "max_pauli_weight": s.max_pauli_weight,
                "mean_growth_ratio": s.mean_growth_ratio,
                "max_growth_ratio": s.max_growth_ratio,
            }
            for s in self.summaries
        ]


def commutator_mv(
    hamiltonian,
    operator,
    *,
    phase_mode: str = "raw",
    normalize: bool = True,
    relative_zero_tol: float = 1e-12,
    truncation_top_k: int | None = None,
    truncation_l2_keep: float | None = None,
    max_support: int | None = None,
    support_policy: str = "drop",
) -> tuple[MV | None, dict[str, Any]]:
    """Compute the sparse commutator ``[H, A]`` safely without MV pruning loss.

    Uses packed anticommutation checks (``[W_h, W_a] = 2 W_h W_a`` iff ``{W_h, W_a} = 0``)
    and accumulates in raw dictionaries *before* constructing ``MV``, avoiding
    spurious loss from ``MV``'s absolute ``TOL = 1e-12`` under small scales.

    Returns ``(mv_or_none, diagnostics)``. ``mv_or_none`` is ``None`` if the
    commutator vanishes identically (relative to ``relative_zero_tol * ||H||_2 * ||A||_2``)
    or is rejected by ``max_support`` when ``support_policy='drop'``.
    """
    if phase_mode not in _PHASE_MODES:
        raise ValueError(f"phase_mode must be one of {_PHASE_MODES}, got {phase_mode!r}")
    if support_policy not in _SUPPORT_POLICIES:
        raise ValueError(
            f"support_policy must be one of {_SUPPORT_POLICIES}, got {support_policy!r}"
        )
    if relative_zero_tol < 0.0:
        raise ValueError("relative_zero_tol must be nonnegative")
    if truncation_l2_keep is not None and not 0.0 < float(truncation_l2_keep) <= 1.0:
        raise ValueError("truncation_l2_keep must lie in (0, 1]")

    H = as_multivector(hamiltonian)
    A = as_multivector(getattr(operator, "mv", operator))
    if H.n != A.n:
        raise ValueError("Hamiltonian and generator act on different qubit counts")
    n = H.n

    h_norm = _coeff_l2(H.terms)
    a_norm = _coeff_l2(A.terms)
    if h_norm == 0.0 or a_norm == 0.0:
        return None, {
            "status": "vanished",
            "raw_coeff_norm": 0.0,
            "raw_hs_norm": 0.0,
            "growth_ratio": 0.0,
            "raw_support": 0,
            "retained_l2_fraction": 0.0,
        }

    h_items = tuple(H.terms.items())
    a_items = tuple(A.terms.items())
    raw: dict[int, complex] = {}
    for code_h, ch in h_items:
        for code_a, ca in a_items:
            if not _pauli_anticommute(n, code_h, code_a):
                continue
            phase, out = _word_mul_unchecked(n, code_h, code_a)
            raw[out] = raw.get(out, 0j) + 2.0 * ch * ca * phase

    scale_ref = h_norm * a_norm
    term_cutoff = relative_zero_tol * scale_ref
    raw = {code: coeff for code, coeff in raw.items() if abs(coeff) > term_cutoff}
    raw_coeff_norm = _coeff_l2(raw)
    if raw_coeff_norm <= term_cutoff or not raw:
        return None, {
            "status": "vanished",
            "raw_coeff_norm": 0.0,
            "raw_hs_norm": 0.0,
            "growth_ratio": 0.0,
            "raw_support": 0,
            "retained_l2_fraction": 0.0,
        }

    raw_hs_norm = (2.0 ** (0.5 * n)) * raw_coeff_norm
    growth_ratio = raw_coeff_norm / scale_ref
    raw_support = len(raw)

    phased = _apply_phase_mode(raw, phase_mode)

    if max_support is not None and support_policy == "drop" and raw_support > int(max_support):
        return None, {
            "status": "dropped_support",
            "raw_coeff_norm": raw_coeff_norm,
            "raw_hs_norm": raw_hs_norm,
            "growth_ratio": growth_ratio,
            "raw_support": raw_support,
            "retained_l2_fraction": 0.0,
        }

    retained_l2 = 1.0
    if (
        truncation_top_k is not None
        or truncation_l2_keep is not None
        or (max_support is not None and support_policy == "truncate")
    ):
        phased, retained_l2 = _truncate_terms(
            phased,
            top_k=truncation_top_k,
            l2_keep=truncation_l2_keep,
            max_support=max_support if support_policy == "truncate" else None,
        )
        if not phased:
            return None, {
                "status": "vanished",
                "raw_coeff_norm": raw_coeff_norm,
                "raw_hs_norm": raw_hs_norm,
                "growth_ratio": growth_ratio,
                "raw_support": raw_support,
                "retained_l2_fraction": 0.0,
            }

    if normalize:
        norm_now = _coeff_l2(phased)
        scaled = {code: coeff / norm_now for code, coeff in phased.items()}
    else:
        scaled = phased

    mv = MV(n, scaled)
    if mv.is_zero():
        return None, {
            "status": "vanished",
            "raw_coeff_norm": raw_coeff_norm,
            "raw_hs_norm": raw_hs_norm,
            "growth_ratio": growth_ratio,
            "raw_support": raw_support,
            "retained_l2_fraction": 0.0,
        }

    return mv, {
        "status": "ok",
        "raw_coeff_norm": raw_coeff_norm,
        "raw_hs_norm": raw_hs_norm,
        "growth_ratio": growth_ratio,
        "raw_support": raw_support,
        "retained_l2_fraction": retained_l2,
    }


def commutator_apply(
    hamiltonian,
    generator,
    *,
    label: str | None = None,
    phase_mode: str = "raw",
    normalize: bool = True,
    relative_zero_tol: float = 1e-12,
    truncation_top_k: int | None = None,
    truncation_l2_keep: float | None = None,
    max_support: int | None = None,
    support_policy: str = "drop",
) -> tuple[Generator | None, dict[str, Any]]:
    """Apply ``[H, .]`` to a ``Generator`` (or ``MV``) returning ``(Generator | None, diag)``."""
    gen = as_generators([generator])[0]
    mv, diag = commutator_mv(
        hamiltonian,
        gen.mv,
        phase_mode=phase_mode,
        normalize=normalize,
        relative_zero_tol=relative_zero_tol,
        truncation_top_k=truncation_top_k,
        truncation_l2_keep=truncation_l2_keep,
        max_support=max_support,
        support_policy=support_policy,
    )
    if mv is None:
        return None, diag
    out_label = label if label is not None else f"[H,{gen.label}]"
    return Generator(out_label, mv), diag


def _node_from_generator(
    gen: Generator,
    *,
    depth: int,
    parent_label: str | None,
    seed_label: str,
    raw_coeff_norm: float,
    raw_hs_norm: float,
    growth_ratio: float,
    raw_support: int,
    retained_l2_fraction: float,
) -> CommutatorNode:
    max_w, mean_w = _max_pauli_weight(gen.n, gen.mv.terms.keys())
    l1 = float(sum(abs(v) for v in gen.mv.terms.values()))
    return CommutatorNode(
        generator=gen,
        depth=depth,
        parent_label=parent_label,
        seed_label=seed_label,
        raw_coeff_norm=raw_coeff_norm,
        raw_hs_norm=raw_hs_norm,
        growth_ratio=growth_ratio,
        raw_support=raw_support,
        retained_l2_fraction=retained_l2_fraction,
        l1_norm=l1,
        max_pauli_weight=max_w,
        mean_pauli_weight=mean_w,
    )


def nested_commutator_hierarchy(
    hamiltonian,
    seeds: Sequence,
    *,
    max_depth: int = 2,
    phase_mode: str = "raw",
    normalize: bool = True,
    deduplicate: bool = True,
    max_per_depth: int | None = None,
    max_total_generators: int | None = None,
    max_support: int | None = None,
    support_policy: str = "drop",
    max_pauli_weight: int | None = None,
    truncation_top_k: int | None = None,
    truncation_l2_keep: float | None = None,
    leakage_tol: float | None = None,
    spin_ordering: str | tuple = "interleaved",
    relative_zero_tol: float = 1e-12,
    drop_zero_norm: bool = True,
    prefix: str = "adH",
) -> CommutatorHierarchy:
    """Build ``A_0, A_1 = {[H, A_i]}, A_2 = {[H, [H, A_i]]}, ...`` recursively.

    Every level stores canonically normalized generators (when ``normalize=True``)
    with full operator-growth statistics (support ``S_A``, Pauli weight,
    dimensionless commutator norm ratio, and collinear Lie-closure rate).
    """
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    if max_per_depth is not None and max_per_depth < 1:
        raise ValueError("max_per_depth must be positive when set")
    if max_total_generators is not None and max_total_generators < 1:
        raise ValueError("max_total_generators must be positive when set")
    if phase_mode not in _PHASE_MODES:
        raise ValueError(f"phase_mode must be one of {_PHASE_MODES}, got {phase_mode!r}")

    H = as_multivector(hamiltonian)
    seed_gens = as_generators(seeds)
    if seed_gens[0].n != H.n:
        raise ValueError("Hamiltonian and seed generators act on different qubit counts")

    seen: set[tuple] = set()
    used_labels: set[str] = set()
    levels: list[tuple[CommutatorNode, ...]] = []
    summaries: list[DepthGrowthSummary] = []
    total_kept = 0
    closure_reached = False
    closure_depth: int | None = None

    def _unique_label(candidate_label: str) -> str:
        if candidate_label not in used_labels:
            used_labels.add(candidate_label)
            return candidate_label
        suffix = 1
        while f"{candidate_label}#{suffix}" in used_labels:
            suffix += 1
        final = f"{candidate_label}#{suffix}"
        used_labels.add(final)
        return final

    # Depth 0: seed generators
    d0_nodes: list[CommutatorNode] = []
    d0_vanished = d0_dedup = d0_dropped = 0
    for gen in seed_gens:
        if max_total_generators is not None and total_kept >= max_total_generators:
            break
        if max_per_depth is not None and len(d0_nodes) >= max_per_depth:
            break
        coeff_norm = _coeff_l2(gen.mv.terms)
        if coeff_norm == 0.0:
            if drop_zero_norm:
                d0_vanished += 1
                continue
        supp = gen.support()
        if max_support is not None and support_policy == "drop" and supp > max_support:
            d0_dropped += 1
            continue
        mw, _ = _max_pauli_weight(gen.n, gen.mv.terms.keys())
        if max_pauli_weight is not None and mw > max_pauli_weight:
            d0_dropped += 1
            continue
        if leakage_tol is not None:
            worst = max(sector_leakage(gen, spin_ordering=spin_ordering).values())
            if worst > leakage_tol:
                d0_dropped += 1
                continue
        mv_store = (
            MV(gen.n, {c: v / coeff_norm for c, v in gen.mv.terms.items()})
            if (normalize and coeff_norm > 0.0)
            else gen.mv
        )
        key = scalar_free_key(mv_store)
        if deduplicate and (key is None or key in seen):
            d0_dedup += 1
            continue
        if key is not None:
            seen.add(key)
        clean_label = _unique_label(gen.label)
        stored_gen = Generator(clean_label, mv_store)
        hs_norm = (2.0 ** (0.5 * gen.n)) * coeff_norm
        d0_nodes.append(
            _node_from_generator(
                stored_gen,
                depth=0,
                parent_label=None,
                seed_label=gen.label,
                raw_coeff_norm=coeff_norm,
                raw_hs_norm=hs_norm,
                growth_ratio=1.0,
                raw_support=supp,
                retained_l2_fraction=1.0,
            )
        )
        total_kept += 1

    levels.append(tuple(d0_nodes))
    d0_attempted = len(seed_gens)
    summaries.append(
        DepthGrowthSummary(
            depth=0,
            attempted=d0_attempted,
            kept=len(d0_nodes),
            vanished=d0_vanished,
            deduplicated=d0_dedup,
            dropped_by_filter=d0_dropped,
            closure_fraction=(d0_vanished + d0_dedup) / max(1, d0_attempted),
            mean_support=float(sum(node.support() for node in d0_nodes) / max(1, len(d0_nodes))),
            max_support=max((node.support() for node in d0_nodes), default=0),
            total_support=sum(node.support() for node in d0_nodes),
            mean_pauli_weight=float(
                sum(node.mean_pauli_weight for node in d0_nodes) / max(1, len(d0_nodes))
            ),
            max_pauli_weight=max((node.max_pauli_weight for node in d0_nodes), default=0),
            mean_growth_ratio=1.0 if d0_nodes else 0.0,
            max_growth_ratio=1.0 if d0_nodes else 0.0,
        )
    )

    # Higher depths k = 1..max_depth
    for depth in range(1, max_depth + 1):
        parents = levels[depth - 1]
        if not parents:
            levels.append(())
            summaries.append(
                DepthGrowthSummary(
                    depth=depth,
                    attempted=0,
                    kept=0,
                    vanished=0,
                    deduplicated=0,
                    dropped_by_filter=0,
                    closure_fraction=1.0,
                    mean_support=0.0,
                    max_support=0,
                    total_support=0,
                    mean_pauli_weight=0.0,
                    max_pauli_weight=0,
                    mean_growth_ratio=0.0,
                    max_growth_ratio=0.0,
                )
            )
            if not closure_reached:
                closure_reached = True
                closure_depth = depth
            continue

        dk_nodes: list[CommutatorNode] = []
        attempted = vanished = deduplicated = dropped = 0
        for pnode in parents:
            if max_total_generators is not None and total_kept >= max_total_generators:
                break
            if max_per_depth is not None and len(dk_nodes) >= max_per_depth:
                break
            attempted += 1
            mv, diag = commutator_mv(
                H,
                pnode.mv,
                phase_mode=phase_mode,
                normalize=normalize,
                relative_zero_tol=relative_zero_tol,
                truncation_top_k=truncation_top_k,
                truncation_l2_keep=truncation_l2_keep,
                max_support=max_support,
                support_policy=support_policy,
            )
            if mv is None:
                if diag["status"] == "vanished":
                    vanished += 1
                else:
                    dropped += 1
                continue
            mw, _ = _max_pauli_weight(H.n, mv.terms.keys())
            if max_pauli_weight is not None and mw > max_pauli_weight:
                dropped += 1
                continue
            if leakage_tol is not None:
                worst = max(sector_leakage(mv, spin_ordering=spin_ordering).values())
                if worst > leakage_tol:
                    dropped += 1
                    continue
            key = scalar_free_key(mv)
            if deduplicate and (key is None or key in seen):
                deduplicated += 1
                continue
            if key is not None:
                seen.add(key)

            cand_label = f"{prefix}^{depth}[{pnode.seed_label}]"
            child_label = _unique_label(cand_label)
            child_gen = Generator(child_label, mv)
            dk_nodes.append(
                _node_from_generator(
                    child_gen,
                    depth=depth,
                    parent_label=pnode.label,
                    seed_label=pnode.seed_label,
                    raw_coeff_norm=diag["raw_coeff_norm"],
                    raw_hs_norm=diag["raw_hs_norm"],
                    growth_ratio=diag["growth_ratio"],
                    raw_support=diag["raw_support"],
                    retained_l2_fraction=diag["retained_l2_fraction"],
                )
            )
            total_kept += 1

        levels.append(tuple(dk_nodes))
        closure_frac = (vanished + deduplicated) / max(1, attempted)
        summaries.append(
            DepthGrowthSummary(
                depth=depth,
                attempted=attempted,
                kept=len(dk_nodes),
                vanished=vanished,
                deduplicated=deduplicated,
                dropped_by_filter=dropped,
                closure_fraction=closure_frac,
                mean_support=float(sum(node.support() for node in dk_nodes) / max(1, len(dk_nodes))),
                max_support=max((node.support() for node in dk_nodes), default=0),
                total_support=sum(node.support() for node in dk_nodes),
                mean_pauli_weight=float(
                    sum(node.mean_pauli_weight for node in dk_nodes) / max(1, len(dk_nodes))
                ),
                max_pauli_weight=max((node.max_pauli_weight for node in dk_nodes), default=0),
                mean_growth_ratio=float(
                    sum(node.growth_ratio for node in dk_nodes) / max(1, len(dk_nodes))
                ),
                max_growth_ratio=max((node.growth_ratio for node in dk_nodes), default=0.0),
            )
        )
        if attempted > 0 and len(dk_nodes) == 0 and dropped == 0 and not closure_reached:
            closure_reached = True
            closure_depth = depth

    return CommutatorHierarchy(
        max_depth=max_depth,
        phase_mode=phase_mode,
        levels=tuple(levels),
        summaries=tuple(summaries),
        closure_reached=closure_reached,
        closure_depth=closure_depth,
    )


def hierarchy_span_diagnostics(
    rho: MV,
    hamiltonian,
    hierarchy: CommutatorHierarchy,
    *,
    initial: Sequence | None = None,
    tau_s: float = DEFAULT_TAU_S,
    max_condition: float = DEFAULT_MAX_CONDITION,
) -> list[dict[str, Any]]:
    """Measure incremental rank, word universe, and Ritz energy across depths."""
    bank = MatrixElementBank(rho, hamiltonian)
    base_indices = bank.extend(
        as_generators(initial) if initial is not None else [identity_generator(bank.n)]
    )
    active = list(base_indices)
    rows: list[dict[str, Any]] = []
    prev_rank = bank.solve(active, tau_s=tau_s, max_condition=max_condition).effective_rank
    prev_energy = bank.solve(active, tau_s=tau_s, max_condition=max_condition).ground_energy

    for depth, level_nodes in enumerate(hierarchy.levels):
        added = bank.extend([node.generator for node in level_nodes])
        for idx in added:
            if idx not in active:
                active.append(idx)
        solved = bank.solve(active, tau_s=tau_s, max_condition=max_condition)
        res = bank.resources(active)
        rows.append(
            {
                "up_to_depth": depth,
                "level_generators": len(level_nodes),
                "cumulative_basis_size": len(active),
                "effective_rank": solved.effective_rank,
                "new_independent_directions": max(0, solved.effective_rank - prev_rank),
                "ground_energy": solved.ground_energy,
                "energy_gain_at_depth": max(0.0, prev_energy - solved.ground_energy),
                "condition_number": solved.condition_number,
                "word_universe": res["word_universe"],
                "max_generator_support": res["max_generator_support"],
            }
        )
        prev_rank = solved.effective_rank
        prev_energy = solved.ground_energy
    return rows


class CommutatorFrontier:
    """Demand-driven commutator pool that expands parents lazily inside A-CASE."""

    def __init__(
        self,
        bank: MatrixElementBank,
        seeds: Sequence,
        *,
        initial_indices: Sequence[int] = (),
        max_depth: int = 2,
        initial_depth: int = 0,
        phase_mode: str = "raw",
        normalize: bool = True,
        deduplicate: bool = True,
        max_support: int | None = None,
        support_policy: str = "drop",
        max_pauli_weight: int | None = None,
        truncation_top_k: int | None = None,
        truncation_l2_keep: float | None = None,
        max_children_total: int | None = None,
        relative_zero_tol: float = 1e-12,
        prefix: str = "adH",
    ):
        if max_depth < 0 or initial_depth < 0 or initial_depth > max_depth:
            raise ValueError("require 0 <= initial_depth <= max_depth")
        self.bank = bank
        self.max_depth = int(max_depth)
        self.phase_mode = phase_mode
        self.normalize = normalize
        self.deduplicate = deduplicate
        self.max_support = max_support
        self.support_policy = support_policy
        self.max_pauli_weight = max_pauli_weight
        self.truncation_top_k = truncation_top_k
        self.truncation_l2_keep = truncation_l2_keep
        self.max_children_total = max_children_total
        self.relative_zero_tol = relative_zero_tol
        self.prefix = prefix

        self._seen_keys: set[tuple] = set()
        self._used_labels: set[str] = set()
        self.depth_of: dict[int, int] = {}
        self.seed_label_of: dict[int, str] = {}
        self.parent_of: dict[int, int | None] = {}
        self.expanded_parents: set[int] = set()
        self.active_pool: list[int] = []
        self.commutators_generated = 0
        self.commutators_vanished = 0
        self.commutators_deduplicated = 0
        self.commutators_dropped_filter = 0

        for idx in initial_indices:
            gen = bank.generator(idx)
            self._used_labels.add(gen.label)
            key = scalar_free_key(gen.mv)
            if key is not None:
                self._seen_keys.add(key)

        seed_list = as_generators(seeds)
        frontier_by_depth: list[list[int]] = [[] for _ in range(initial_depth + 1)]
        for gen in seed_list:
            coeff_norm = _coeff_l2(gen.mv.terms)
            if coeff_norm == 0.0:
                continue
            mv_store = (
                MV(gen.n, {c: v / coeff_norm for c, v in gen.mv.terms.items()})
                if normalize
                else gen.mv
            )
            key = scalar_free_key(mv_store)
            if deduplicate and (key is None or key in self._seen_keys):
                continue
            if key is not None:
                self._seen_keys.add(key)
            lbl = self._unique_label(gen.label)
            idx = bank.add(Generator(lbl, mv_store))
            self.depth_of[idx] = 0
            self.seed_label_of[idx] = gen.label
            self.parent_of[idx] = None
            if idx not in initial_indices and idx not in self.active_pool:
                self.active_pool.append(idx)
            frontier_by_depth[0].append(idx)

        for d in range(initial_depth):
            for p_idx in frontier_by_depth[d]:
                child = self.expand_parent(p_idx, initial_indices=initial_indices)
                if child is not None:
                    frontier_by_depth[d + 1].append(child)

    def _unique_label(self, candidate: str) -> str:
        if candidate not in self._used_labels and candidate not in self.bank.labels:
            self._used_labels.add(candidate)
            return candidate
        suffix = 1
        while (
            f"{candidate}#{suffix}" in self._used_labels
            or f"{candidate}#{suffix}" in self.bank.labels
        ):
            suffix += 1
        out = f"{candidate}#{suffix}"
        self._used_labels.add(out)
        return out

    def expandable_indices(self, ordered_candidates: Sequence[int] | None = None) -> list[int]:
        """Parents at depth < max_depth that have not yet been expanded."""
        if self.max_children_total is not None and self.commutators_generated >= self.max_children_total:
            return []
        candidates = (
            ordered_candidates
            if ordered_candidates is not None
            else [idx for idx in self.depth_of]
        )
        return [
            idx
            for idx in candidates
            if idx not in self.expanded_parents
            and self.depth_of.get(idx, self.max_depth) < self.max_depth
        ]

    def expand_parent(
        self,
        parent_idx: int,
        *,
        initial_indices: Sequence[int] = (),
    ) -> int | None:
        """Compute ``[H, parent]`` lazily and add it to the bank & pool if novel."""
        if parent_idx in self.expanded_parents:
            return None
        depth = self.depth_of.get(parent_idx, self.max_depth)
        if depth >= self.max_depth:
            return None
        if self.max_children_total is not None and self.commutators_generated >= self.max_children_total:
            return None

        self.expanded_parents.add(parent_idx)
        parent_gen = self.bank.generator(parent_idx)
        mv, diag = commutator_mv(
            self.bank.hamiltonian,
            parent_gen.mv,
            phase_mode=self.phase_mode,
            normalize=self.normalize,
            relative_zero_tol=self.relative_zero_tol,
            truncation_top_k=self.truncation_top_k,
            truncation_l2_keep=self.truncation_l2_keep,
            max_support=self.max_support,
            support_policy=self.support_policy,
        )
        self.commutators_generated += 1
        if mv is None:
            if diag["status"] == "vanished":
                self.commutators_vanished += 1
            else:
                self.commutators_dropped_filter += 1
            return None
        if self.max_pauli_weight is not None:
            mw, _ = _max_pauli_weight(self.bank.n, mv.terms.keys())
            if mw > self.max_pauli_weight:
                self.commutators_dropped_filter += 1
                return None
        key = scalar_free_key(mv)
        if self.deduplicate and (key is None or key in self._seen_keys):
            self.commutators_deduplicated += 1
            return None
        if key is not None:
            self._seen_keys.add(key)

        next_depth = depth + 1
        seed_lbl = self.seed_label_of.get(parent_idx, parent_gen.label)
        child_label = self._unique_label(f"{self.prefix}^{next_depth}[{seed_lbl}]")
        child_idx = self.bank.add(Generator(child_label, mv))
        self.depth_of[child_idx] = next_depth
        self.seed_label_of[child_idx] = seed_lbl
        self.parent_of[child_idx] = parent_idx
        if child_idx not in initial_indices and child_idx not in self.active_pool:
            self.active_pool.append(child_idx)
        return child_idx


def run_dynamic_commutator_acase(
    rho: MV,
    hamiltonian,
    seeds: Sequence,
    *,
    initial: Sequence | None = None,
    bank: MatrixElementBank | None = None,
    max_depth: int = 2,
    initial_depth: int = 0,
    expand_mode: str = "on_select_or_stall",
    expand_top_k: int = 2,
    phase_mode: str = "raw",
    normalize: bool = True,
    deduplicate: bool = True,
    max_support: int | None = None,
    support_policy: str = "drop",
    max_pauli_weight: int | None = None,
    truncation_top_k: int | None = None,
    truncation_l2_keep: float | None = None,
    max_children_total: int | None = None,
    max_size: int = 10,
    roots: int = 1,
    aggregation: str = "mean",
    min_lowering: float = DEFAULT_MIN_LOWERING,
    min_orthogonality: float = DEFAULT_MIN_ORTHOGONALITY,
    gamma: float = 0.0,
    cost_model: IncrementalCostModel | None = None,
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
    """Demand-driven adaptive A-CASE with lazy hierarchical commutator expansion.

    Unlike static ``nested_commutator_hierarchy`` (which constructs all depth
    ``1..max_depth`` commutators upfront before scoring), this workflow maintains
    a live ``CommutatorFrontier`` starting at depth ``initial_depth`` (default
    ``0``) and generates higher-depth ``[H, A]`` children on demand:

    - ``'on_select'``: expand the accepted generator ``A*`` after each addition.
    - ``'on_stall'``: expand up to ``expand_top_k`` highest-ranked unexpanded
      parents only when current candidates fail ``min_lowering`` or conditioning.
    - ``'on_select_or_stall'``: expand the accepted generator *and* fall back to
      expanding unexpanded parents if the active frontier stalls.
    - ``'top_k'``: at every step before scoring, expand top ``expand_top_k``
      unexpanded basis/ranked parents.
    """
    if expand_mode not in _EXPAND_MODES:
        raise ValueError(f"expand_mode must be one of {_EXPAND_MODES}, got {expand_mode!r}")
    if expand_top_k < 1:
        raise ValueError("expand_top_k must be positive")

    started = time.perf_counter()
    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    else:
        _require_matching_bank(bank, rho, hamiltonian)

    if cost_model is not None:
        from .information import (
            qwc_bases_for_words,
            score_information_candidate,
            select_information_candidate,
        )

    config = ACASEConfig(
        max_size=max_size,
        roots=roots,
        aggregation=aggregation,
        min_lowering=min_lowering,
        min_orthogonality=min_orthogonality,
        gamma=gamma,
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
    frontier = CommutatorFrontier(
        bank,
        seeds,
        initial_indices=basis,
        max_depth=max_depth,
        initial_depth=initial_depth,
        phase_mode=phase_mode,
        normalize=normalize,
        deduplicate=deduplicate,
        max_support=max_support,
        support_policy=support_policy,
        max_pauli_weight=max_pauli_weight,
        truncation_top_k=truncation_top_k,
        truncation_l2_keep=truncation_l2_keep,
        max_children_total=max_children_total,
    )
    if not frontier.active_pool:
        raise ValueError("no seed generators outside the initial basis")

    bank.retain_basis(basis)
    solved = bank.solve(basis, tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition)
    value, root_energies = _acase_objective(solved, config)
    state = ACASEState(
        basis=basis,
        pool=tuple(frontier.active_pool),
        result=solved,
        energy=value,
        energy_history=(value,),
        records=(),
        root_energies=root_energies,
    )

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

    last_ranked_parents: list[int] = list(state.pool)
    tracked = tuple(range(roots))

    if state.stopped_reason == "basis budget reached":
        for step in range(1, max_size + 1):
            step_started = time.perf_counter()

            if expand_mode == "top_k" and step > 1:
                candidates_to_expand = frontier.expandable_indices(last_ranked_parents)[:expand_top_k]
                for p_idx in candidates_to_expand:
                    frontier.expand_parent(p_idx, initial_indices=state.basis)

            def _score_frontier():
                remaining = [i for i in frontier.active_pool if i not in state.basis]
                if not remaining:
                    return remaining, [], None
                basis_words = bank.word_set(state.basis)
                if cost_model is not None:
                    qwc_bases = (
                        qwc_bases_for_words(bank.n, basis_words)
                        if (cost_model.setting_weight > 0.0 or cost_model.time_weight_us > 0.0)
                        else ()
                    )
                    info_scores = [
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
                    best_info = select_information_candidate(info_scores)
                    return remaining, info_scores, best_info
                std_scores = [
                    score_candidate(
                        bank,
                        state.basis,
                        state.result,
                        idx,
                        roots=tracked,
                        aggregation=aggregation,
                        basis_words=basis_words,
                        min_orthogonality=min_orthogonality,
                        gamma=gamma,
                        leakage_tol=leakage_tol,
                        leakage_mode=leakage_mode,
                        sector_target=sector_target,
                        spin_ordering=spin_ordering,
                    )
                    for idx in remaining
                ]
                best_std = select_candidate(std_scores)
                return remaining, std_scores, best_std

            def _effective_lowering(item: Any) -> float:
                return float(getattr(item, "risk_adjusted_lowering", item.predicted_lowering))

            remaining, scores, best = _score_frontier()
            best_ok = best is not None and _effective_lowering(best) >= min_lowering

            # On-stall fallback expansion when frontier stalls or empties
            while not best_ok and expand_mode in ("on_stall", "on_select_or_stall"):
                accepted_scores = sorted(
                    (s for s in scores if s.accepted),
                    key=lambda x: (-_effective_lowering(x), x.index),
                )
                basis_parents = [
                    i for i in reversed(state.basis) if i in frontier.depth_of
                ]
                other_scores = sorted(
                    (
                        s
                        for s in scores
                        if not s.accepted
                        and not (s.rejected and s.rejected.startswith("sector"))
                    ),
                    key=lambda x: (-_effective_lowering(x), x.index),
                )
                priority = (
                    [s.index for s in accepted_scores]
                    + basis_parents
                    + [s.index for s in other_scores]
                )
                expandable = frontier.expandable_indices(priority)
                if not expandable:
                    break
                expanded_any = False
                for p_idx in expandable[:expand_top_k]:
                    if frontier.expand_parent(p_idx, initial_indices=state.basis) is not None:
                        expanded_any = True
                if not expanded_any:
                    # Keep draining expandable parents in batches if all chosen skipped
                    if not frontier.expandable_indices(priority):
                        break
                    continue
                remaining, scores, best = _score_frontier()
                best_ok = best is not None and _effective_lowering(best) >= min_lowering
                if not remaining:
                    break

            if not remaining:
                state = acase_step(
                    state, ACASEStepOutcome(stopped_reason="candidate pool exhausted")
                )
                break
            if best is None:
                state = acase_step(
                    state,
                    ACASEStepOutcome(
                        stopped_reason="every candidate rejected (conditioning or sector)"
                    ),
                )
                break
            if _effective_lowering(best) < min_lowering:
                state = acase_step(
                    state,
                    ACASEStepOutcome(stopped_reason="predicted lowering below threshold"),
                )
                break

            last_ranked_parents = [
                s.index
                for s in sorted(
                    scores,
                    key=lambda x: (-x.score, x.index),
                )
            ]

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
                selection_criterion=(
                    "information_ratio" if cost_model is not None else "lowering"
                ),
                selection_score=best.score,
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

            if stopped_reason is None and expand_mode in ("on_select", "on_select_or_stall"):
                frontier.expand_parent(best.index, initial_indices=state.basis)

            state = ACASEState(
                basis=state.basis,
                pool=tuple(frontier.active_pool),
                result=state.result,
                energy=state.energy,
                energy_history=state.energy_history,
                records=state.records,
                root_energies=state.root_energies,
                stopped_reason=state.stopped_reason,
            )

            if stopped_reason is not None:
                break

    depth_counts: dict[int, int] = {}
    for idx in state.basis:
        if idx in frontier.depth_of:
            d = frontier.depth_of[idx]
            depth_counts[d] = depth_counts.get(d, 0) + 1

    resources = dict(state.result.resources)
    resources.update(bank.resources(state.basis))
    resources.update(
        {
            "candidate_pool_size": len(frontier.active_pool),
            "growth_seconds": time.perf_counter() - started,
            "registered_generators": len(bank),
            "expand_mode": expand_mode,
            "max_depth": max_depth,
            "initial_depth": initial_depth,
            "commutators_generated": frontier.commutators_generated,
            "commutators_vanished": frontier.commutators_vanished,
            "commutators_deduplicated": frontier.commutators_deduplicated,
            "commutators_dropped_filter": frontier.commutators_dropped_filter,
            "depth_counts_in_basis": depth_counts,
            "selection_criterion": (
                "information_ratio" if cost_model is not None else "lowering"
            ),
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


__all__ = [
    "CommutatorFrontier",
    "CommutatorHierarchy",
    "CommutatorNode",
    "DepthGrowthSummary",
    "commutator_apply",
    "commutator_mv",
    "hierarchy_span_diagnostics",
    "nested_commutator_hierarchy",
    "run_dynamic_commutator_acase",
]
