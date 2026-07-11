"""ADAPT operator pools and the real-sector (odd-Y) restriction.

A Pauli word with an odd number of Y letters is purely imaginary in the
computational basis, so its rotor exp(-i theta P/2) = cos - i sin P is a
real matrix. For a Hamiltonian and reference state that are real in the
computational basis, the ADAPT commutator gradient of every even-Y
candidate vanishes identically: -i[H, P] is imaginary for even-Y P while
rho stays real, so Tr[rho (-i/2)[H,P]] = 0. Restricting the pool to the
odd-Y sector therefore cannot change the exact ADAPT trajectory — it only
removes candidates whose scores are exactly zero (hypothesis H1).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Sequence

from ..ir import PauliWord


@dataclass(frozen=True)
class PoolOperator:
    label: str
    word: PauliWord


def is_odd_y(word: PauliWord) -> bool:
    """True when the word contains an odd number of Y letters."""
    return sum(1 for j in word.support() if word.letter(j) == "Y") % 2 == 1


def odd_y_filter(pool: Iterable[PoolOperator]) -> list[PoolOperator]:
    return [op for op in pool if is_odd_y(op.word)]


def _word_from_factors(n: int, factors: dict[int, str]) -> PauliWord:
    letters = ["I"] * n
    for j, ch in factors.items():
        letters[j] = ch
    return PauliWord.from_label("".join(letters), n)


def local_pool(n: int, *, periodic_context: bool = True) -> list[PoolOperator]:
    """Y-centered local pool of the TFIM study: one Y per site, optionally
    dressed by Z on the neighbouring qubits. All members are odd-Y."""
    if n < 1:
        raise ValueError("n must be positive")
    ops: dict[int, PoolOperator] = {}
    for i in range(n):
        contexts: list[tuple[str, dict[int, str]]] = [(f"Y{i}", {i: "Y"})]
        left, right = (i - 1) % n, (i + 1) % n
        left_ok = (periodic_context or i > 0) and left != i
        right_ok = (periodic_context or i < n - 1) and right != i
        if left_ok:
            contexts.append((f"Z{left}Y{i}", {left: "Z", i: "Y"}))
        if right_ok:
            contexts.append((f"Y{i}Z{right}", {i: "Y", right: "Z"}))
        if left_ok and right_ok and left != right:
            contexts.append((f"Z{left}Y{i}Z{right}", {left: "Z", i: "Y", right: "Z"}))
        for label, factors in contexts:
            word = _word_from_factors(n, factors)
            # de-duplicate small-n periodic collisions, preserving order
            ops.setdefault(word.code, PoolOperator(label, word))
    return list(ops.values())


def all_words_pool(n: int, max_weight: int = 2,
                   letters: Sequence[str] = ("X", "Y", "Z")) -> list[PoolOperator]:
    """Every Pauli word up to ``max_weight`` non-identity letters.

    The unrestricted baseline pool for the H1 equivalence experiments;
    combine with ``odd_y_filter`` for the restricted variant.
    """
    if not (1 <= max_weight <= n):
        raise ValueError("max_weight must be in [1, n]")
    pool: list[PoolOperator] = []
    for weight in range(1, max_weight + 1):
        for sites in combinations(range(n), weight):
            for choice in product(letters, repeat=weight):
                word = _word_from_factors(n, dict(zip(sites, choice)))
                label = "".join(f"{ch}{j}" for j, ch in zip(sites, choice))
                pool.append(PoolOperator(label, word))
    return pool
