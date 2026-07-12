"""Commuting-layer construction for layered ADAPT (hypothesis H5).

After the top candidate is selected, a layer greedily adds further
candidates that (a) score at least ``alpha`` times the best score and
(b) mutually commute with everything already in the layer. Mutual
commutation makes the layer's rotor product order-independent, so the
whole layer can be appended in one ADAPT step: parameters start at zero
and joint re-optimization can only lower the energy.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..ir import PauliWord


def words_commute(a: PauliWord, b: PauliWord) -> bool:
    """True when the Pauli words commute (even number of anticommuting
    single-qubit positions: both non-identity with different letters)."""
    if a.n != b.n:
        raise ValueError("words act on different qubit counts")
    clashes = 0
    for j in range(a.n):
        la = (a.code >> (2 * j)) & 3
        lb = (b.code >> (2 * j)) & 3
        if la and lb and la != lb:
            clashes += 1
    return clashes % 2 == 0


def build_layer(pool, scores: Mapping[int, float], best: int,
                candidates: Sequence[int], alpha: float) -> list[int]:
    """Indices of the commuting layer around ``best``, strongest first.

    ``scores`` maps candidate index -> (possibly estimated) selection score;
    candidates without a score are skipped. ``alpha`` in (0, 1] is the
    strength floor relative to the best score.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    floor = alpha * abs(scores[best])
    layer = [best]
    ranked = sorted((j for j in candidates if j != best and j in scores),
                    key=lambda j: -abs(scores[j]))
    for j in ranked:
        if abs(scores[j]) < floor:
            break  # ranked descending: everything after is weaker
        if all(words_commute(pool[j].word, pool[k].word) for k in layer):
            layer.append(j)
    return layer
