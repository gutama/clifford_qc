"""Phase 11: declared ordering and coarse-to-fine A-CASE packet selection.

The Haar transform itself lives in :mod:`configuration`.  This module owns the
policy layer the transform intentionally did not infer: how sampled
configurations are ordered, which coarse intervals are exposed first, when an
interval is refined, and when the convergence-complete individual/dressed pool
takes over.

Nothing here is a shared ADAPT/A-CASE runner.  Candidate blocks are scored by
the A-CASE criteria in :mod:`adaptive`, and every accepted packet is handed
back through :func:`run_acase` for the generalized Ritz solve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .adaptive import (
    ACASEConfig, AdaptiveResult, OverlapTarget, _prepare_target_overlap_context,
    _score_target_overlap_precomputed, run_acase, score_candidate,
    select_candidate, select_target_candidate,
)
from .configuration import configuration_haar_packets
from .elements import MatrixElementBank
from .generator_core import Generator, as_generators
from .generators import identity_generator

__all__ = [
    "ConfigurationOrdering",
    "HaarPacketNode",
    "MultiresolutionResult",
    "configuration_ordering",
    "configuration_packet_hierarchy",
    "run_coarse_to_fine_acase",
]

_PACKET_INTERVAL = re.compile(r"H\[(\d+):(\d+)\)$")


@dataclass(frozen=True)
class ConfigurationOrdering:
    """One Phase-11C ordering ablation and the permutation that produced it."""

    method: str
    words: np.ndarray
    permutation: np.ndarray
    metadata: dict = field(default_factory=dict)


def _occupation_signature(word: int, n: int) -> tuple[int, ...]:
    """Two-spin-orbital blocks: 0 empty, 1 first, 2 second, 3 double."""
    signature = []
    for first in range(0, n, 2):
        state = 0
        if (word >> (n - 1 - first)) & 1:
            state |= 1
        second = first + 1
        if second < n and (word >> (n - 1 - second)) & 1:
            state |= 2
        signature.append(state)
    return tuple(signature)


def configuration_ordering(words, *, method: str,
                           probabilities=None, reference_word: int | None = None,
                           n: int | None = None, graph_matrix=None,
                           seed: int | None = 0) -> ConfigurationOrdering:
    """Order sampled configurations for the required §11C ablations.

    ``probability``
        Descending retained probability, with the occupation word as the
        deterministic tie break.
    ``physics``
        Excitation rank from a declared reference, then the adjacent-spin
        occupation signature (empty/up/down/double).  The signature is useful
        for both interleaved-spin chemistry and Hubbard site pairs and remains
        an explicit convention rather than inferred model metadata.
    ``graph``
        Best-connected traversal of the sampled determinant graph.  At each
        step the unvisited determinant with the largest Hamiltonian coupling to
        the visited set is exposed next.  ``graph_matrix`` is the restricted
        Hamiltonian in the *input* word order.
    ``random``
        Seeded random control.
    """
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    if words.size == 0:
        raise ValueError("cannot order an empty configuration set")
    if np.unique(words).size != words.size:
        raise ValueError("configuration ordering needs unique words")
    methods = ("probability", "physics", "graph", "random")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}")

    probability = None
    if probabilities is not None:
        probability = np.asarray(probabilities, dtype=float).reshape(-1)
        if probability.size != words.size:
            raise ValueError("probabilities and words have different lengths")
        if np.any(probability < 0.0) or not np.all(np.isfinite(probability)):
            raise ValueError("probabilities must be finite and nonnegative")

    if method == "probability":
        if probability is None:
            raise ValueError("probability ordering needs one probability per word")
        order = np.array(sorted(range(words.size),
                                key=lambda i: (-probability[i], int(words[i]))),
                         dtype=np.int64)
        metadata = {"probability_sum": float(probability.sum())}
    elif method == "physics":
        if reference_word is None or n is None:
            raise ValueError("physics ordering needs reference_word and n")
        if n < 1 or any(int(word).bit_length() > n for word in words):
            raise ValueError("n is too small for the supplied occupation words")
        reference_word = int(reference_word)
        keys = []
        for index, word in enumerate(words.tolist()):
            rank = (int(word) ^ reference_word).bit_count() // 2
            signature = _occupation_signature(int(word), int(n))
            keys.append((rank, signature, int(word), index))
        order = np.array([item[-1] for item in sorted(keys)], dtype=np.int64)
        metadata = {"reference_word": reference_word,
                    "metadata": "excitation_rank+paired_occupation_signature"}
    elif method == "graph":
        matrix = np.asarray(graph_matrix, dtype=complex)
        if matrix.shape != (words.size, words.size):
            raise ValueError("graph_matrix must be square in the supplied word order")
        if probability is not None:
            fallback = probability
        else:
            fallback = np.zeros(words.size, dtype=float)
        if reference_word is not None and int(reference_word) in set(words.tolist()):
            start = int(np.flatnonzero(words == int(reference_word))[0])
        else:
            start = min(range(words.size),
                        key=lambda i: (-fallback[i], int(words[i])))
        visited = [start]
        remaining = set(range(words.size)) - {start}
        weights = np.abs(matrix)
        while remaining:
            nxt = min(
                remaining,
                key=lambda j: (-float(np.max(weights[visited, j])),
                               -fallback[j], int(words[j])),
            )
            visited.append(nxt)
            remaining.remove(nxt)
        order = np.asarray(visited, dtype=np.int64)
        metadata = {"start_word": int(words[start]),
                    "traversal": "max_coupling_to_visited"}
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(words.size).astype(np.int64)
        metadata = {"seed": seed}

    return ConfigurationOrdering(method, words[order].copy(), order, metadata)


@dataclass(frozen=True)
class HaarPacketNode:
    """One Haar detail interval and its position in the full binary tree."""

    generator: Generator
    lo: int
    hi: int
    parent: tuple[int, int] | None
    depth: int
    eligible: bool

    @property
    def interval(self) -> tuple[int, int]:
        return self.lo, self.hi


def _interval(generator: Generator) -> tuple[int, int]:
    match = _PACKET_INTERVAL.search(generator.label)
    if match is None:
        raise ValueError(f"cannot read Haar interval from {generator.label!r}")
    return int(match.group(1)), int(match.group(2))


def configuration_packet_hierarchy(configurations: Sequence, *,
                                   max_support: int | None = 16,
                                   label_prefix: str = "cfgMR"
                                   ) -> tuple[HaarPacketNode, ...]:
    """Full Haar tree annotated with support eligibility and parent links."""
    generators = tuple(as_generators(configurations))
    if len(generators) < 2:
        raise ValueError("a packet hierarchy needs at least two configurations")
    if max_support is not None and max_support < 2:
        raise ValueError("max_support must be at least 2")
    packets = configuration_haar_packets(
        generators, min_support=2, max_support=None,
        label_prefix=label_prefix)
    intervals = {_interval(packet): packet for packet in packets}
    parent_by_interval = {}
    for interval in intervals:
        lo, hi = interval
        containers = [other for other in intervals
                      if other != interval and other[0] <= lo and hi <= other[1]]
        parent_by_interval[interval] = (
            min(containers, key=lambda item: item[1] - item[0])
            if containers else None)

    def depth(interval):
        count, current = 0, parent_by_interval[interval]
        while current is not None:
            count += 1
            current = parent_by_interval[current]
        return count

    return tuple(HaarPacketNode(
        generator=packet, lo=interval[0], hi=interval[1],
        parent=parent_by_interval[interval], depth=depth(interval),
        eligible=(max_support is None or packet.support() <= max_support),
    ) for interval, packet in intervals.items())


def _initial_frontier(nodes: Sequence[HaarPacketNode]) -> dict[tuple[int, int], HaarPacketNode]:
    eligible = [node for node in nodes if node.eligible]
    frontier = {}
    for node in eligible:
        has_eligible_ancestor = any(
            other.interval != node.interval
            and other.lo <= node.lo and node.hi <= other.hi
            for other in eligible)
        if not has_eligible_ancestor:
            frontier[node.interval] = node
    return frontier


def _children(node: HaarPacketNode, nodes: Sequence[HaarPacketNode]
              ) -> list[HaarPacketNode]:
    descendants = [other for other in nodes if other.eligible
                   and other.interval != node.interval
                   and node.lo <= other.lo and other.hi <= node.hi]
    out = []
    for candidate in descendants:
        has_between = any(
            middle.interval != candidate.interval
            and middle.interval != node.interval
            and node.lo <= middle.lo <= candidate.lo
            and candidate.hi <= middle.hi <= node.hi
            for middle in descendants)
        if not has_between:
            out.append(candidate)
    return sorted(out, key=lambda item: (item.lo, item.hi))


@dataclass(frozen=True)
class MultiresolutionResult:
    """Coarse packet trace plus the convergence-complete final A-CASE solve."""

    result: AdaptiveResult
    packet_labels: tuple[str, ...]
    packet_energy_history: tuple[float, ...]
    refined_intervals: tuple[tuple[int, int], ...]
    frontiers_scored: int
    packet_candidates: int
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "energy": float(self.result.energy),
            "basis_size": int(self.result.basis_size),
            "condition_number": float(self.result.result.condition_number),
            "packet_labels": list(self.packet_labels),
            "packet_directions": len(self.packet_labels),
            "packet_candidates": int(self.packet_candidates),
            "frontiers_scored": int(self.frontiers_scored),
            "refined_intervals": [list(interval) for interval in self.refined_intervals],
            "selection_criterion": self.result.resources.get("selection_criterion"),
            **self.metadata,
        }


def run_coarse_to_fine_acase(rho, hamiltonian, configurations: Sequence,
                             final_pool: Sequence, *,
                             initial: Sequence | None = None,
                             bank: MatrixElementBank | None = None,
                             config: ACASEConfig | None = None,
                             target: OverlapTarget | None = None,
                             packet_steps: int | None = None,
                             max_packet_support: int | None = 16,
                             competitive_ratio: float = 0.5,
                             max_competitive: int = 4,
                             label_prefix: str = "cfgMR") -> MultiresolutionResult:
    """§11B coarse-to-fine packet stage followed by a complete final pool.

    Only admissible coarse intervals are exposed initially.  After each A-CASE
    decision, the selected interval and up to ``max_competitive`` intervals
    whose score is within ``competitive_ratio`` of the best are refined to
    their nearest support-admissible children.  The packet stage consumes part
    of ``config.max_size``; every remaining growth step is then spent on
    ``final_pool``, which must contain the individual/dressed directions needed
    for convergence.  The hierarchy is therefore a staging policy, never a
    replacement basis.
    """
    if config is None:
        config = ACASEConfig()
    if not 0.0 <= competitive_ratio <= 1.0:
        raise ValueError("competitive_ratio must lie in [0, 1]")
    if max_competitive < 1:
        raise ValueError("max_competitive must be positive")
    if config.criterion == "target_overlap" and target is None:
        raise ValueError("target_overlap hierarchy needs an OverlapTarget")
    if target is not None and config.criterion != "target_overlap":
        raise ValueError("an OverlapTarget requires criterion='target_overlap'")

    configurations = tuple(as_generators(configurations))
    final_pool = tuple(as_generators(final_pool))
    nodes = configuration_packet_hierarchy(
        configurations, max_support=max_packet_support,
        label_prefix=label_prefix)
    eligible = tuple(node for node in nodes if node.eligible)
    if not eligible:
        raise ValueError("the packet support cap admits no coarse direction")
    frontier = _initial_frontier(nodes)
    if not frontier:
        raise RuntimeError("packet hierarchy has no admissible frontier")

    if bank is None:
        bank = MatrixElementBank(rho, hamiltonian)
    seeds = (tuple(as_generators(initial)) if initial is not None
             else (identity_generator(bank.n),))
    retained = tuple(bank.extend(seeds))
    budget = config.max_size
    coarse_budget = (max(1, budget // 3) if packet_steps is None
                     else int(packet_steps))
    coarse_budget = max(0, min(coarse_budget, budget))
    packet_labels = []
    packet_energy_history = []
    refined = []
    frontiers_scored = 0

    for _ in range(coarse_budget):
        live_nodes = [node for node in frontier.values()
                      if bank.add(node.generator) not in retained]
        if not live_nodes:
            break
        solved = bank.solve(
            retained, tau_s=config.tau_s, rel_tau=config.rel_tau,
            max_condition=config.max_condition)
        candidate_indices = [bank.add(node.generator) for node in live_nodes]
        if config.criterion == "lowering":
            basis_words = bank.word_set(retained)
            scores = [score_candidate(
                bank, retained, solved, index,
                roots=tuple(range(config.roots)), aggregation=config.aggregation,
                basis_words=basis_words,
                min_orthogonality=config.min_orthogonality,
                gamma=config.gamma, leakage_tol=config.leakage_tol,
                leakage_mode=config.leakage_mode,
                sector_target=config.sector_target)
                for index in candidate_indices]
            selected_score = select_candidate(scores)
            ranking_values = {
                score.index: score.score for score in scores if score.accepted}
            gate_value = (None if selected_score is None
                          else selected_score.predicted_lowering)
            threshold = config.min_lowering
        else:
            target_context = _prepare_target_overlap_context(
                bank, retained, solved, target)
            scores = [_score_target_overlap_precomputed(
                bank, retained, solved, index, target_context,
                min_orthogonality=config.min_orthogonality,
                leakage_tol=config.leakage_tol,
                leakage_mode=config.leakage_mode,
                sector_target=config.sector_target)
                for index in candidate_indices]
            selected_score = select_target_candidate(scores)
            ranking_values = {
                score.index: score.score for score in scores if score.accepted}
            gate_value = None if selected_score is None else selected_score.score
            threshold = config.min_target_overlap
        frontiers_scored += len(candidate_indices)
        if not ranking_values or gate_value is None:
            break
        best_value = max(ranking_values.values())
        # Selection may include a measurement-width penalty, but the stopping
        # threshold has the same unpenalized semantics as run_acase.
        if gate_value < threshold:
            break
        competitive = sorted(
            (node for node in live_nodes
             if ranking_values.get(bank.add(node.generator), -1.0)
             >= competitive_ratio * best_value),
            key=lambda node: (-ranking_values[bank.add(node.generator)],
                              node.lo, node.hi),
        )[:max_competitive]

        one_step = replace(config, max_size=1)
        staged = run_acase(
            rho, hamiltonian, [node.generator for node in live_nodes],
            initial=[bank.generator(index) for index in retained], bank=bank,
            config=one_step, target=target)
        if not staged.records:
            break
        # The local pass above and run_acase both score this frontier; report
        # the actual work rather than only the first pass.
        frontiers_scored += staged.records[-1].candidates_scored
        selected_index = staged.indices[-1]
        selected = next(node for node in live_nodes
                        if bank.add(node.generator) == selected_index)
        selected_label = selected.generator.label
        retained = staged.indices
        packet_labels.append(selected_label)
        packet_energy_history.append(float(staged.energy))

        refinement_set = {node.interval: node for node in competitive}
        refinement_set[selected.interval] = selected
        for interval, node in sorted(refinement_set.items()):
            children = _children(node, nodes)
            if interval == selected.interval or children:
                frontier.pop(interval, None)
            if children:
                refined.append(interval)
                for child in children:
                    frontier[child.interval] = child

    remaining = max(0, budget - len(packet_labels))
    final_config = replace(config, max_size=remaining)
    final = run_acase(
        rho, hamiltonian, final_pool,
        initial=[bank.generator(index) for index in retained], bank=bank,
        config=final_config, target=target)
    return MultiresolutionResult(
        result=final, packet_labels=tuple(packet_labels),
        packet_energy_history=tuple(packet_energy_history),
        refined_intervals=tuple(refined), frontiers_scored=frontiers_scored,
        packet_candidates=len(eligible),
        metadata={"competitive_ratio": float(competitive_ratio),
                  "max_competitive": int(max_competitive),
                  "max_packet_support": max_packet_support,
                  "packet_budget": int(coarse_budget),
                  "final_growth_budget": int(remaining)},
    )
