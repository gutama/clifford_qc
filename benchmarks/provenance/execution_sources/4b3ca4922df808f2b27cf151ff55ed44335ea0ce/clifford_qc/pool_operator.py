"""Shared named Pauli candidates at the program-IR boundary.

Model builders and algorithms can share these values without importing a
solver. This module depends on PauliWord, so it belongs to Program IR rather
than the representation-only Pauli kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import PauliWord


@dataclass(frozen=True)
class PoolOperator:
    label: str
    word: PauliWord


def is_odd_y(word: PauliWord) -> bool:
    """True when the word contains an odd number of Y letters."""
    return sum(1 for j in word.support() if word.letter(j) == "Y") % 2 == 1
