"""Qubit-wise-commuting (QWC) measurement grouping.

Two Pauli words qubit-wise commute when at every qubit their letters are
equal or at least one is identity. All words of a QWC group are diagonal
in one shared per-qubit measurement basis, so the whole group costs a
single measurement circuit: every shot yields one outcome for *every* word
in the group. Grouping therefore divides N_circuits and multiplies the
information per shot; the per-word marginal statistics are unchanged
(outcomes across words become correlated, which per-word variance
estimates are insensitive to).
"""

from __future__ import annotations

from typing import Sequence

from ..ir import PauliWord


def qubit_wise_commute(a: PauliWord, b: PauliWord) -> bool:
    if a.n != b.n:
        raise ValueError("words act on different qubit counts")
    for j in range(a.n):
        la, lb = a.letter(j), b.letter(j)
        if la != "I" and lb != "I" and la != lb:
            return False
    return True


def qwc_groups(words: Sequence[PauliWord]) -> list[list[PauliWord]]:
    """Greedy largest-degree-first partition of ``words`` into QWC groups.

    Graph coloring on the QWC-incompatibility graph: words are processed in
    descending conflict degree and placed into the first compatible group.
    Not optimal (that is NP-hard) but standard and effective.
    """
    words = list(words)
    if not words:
        return []
    conflicts = [sum(1 for other in words if other is not w
                     and not qubit_wise_commute(w, other)) for w in words]
    order = sorted(range(len(words)), key=lambda i: -conflicts[i])
    groups: list[list[PauliWord]] = []
    for i in order:
        w = words[i]
        for group in groups:
            if all(qubit_wise_commute(w, member) for member in group):
                group.append(w)
                break
        else:
            groups.append([w])
    return groups


def shared_basis(group: Sequence[PauliWord]) -> dict[int, str]:
    """The per-qubit measurement basis of a QWC group: qubit -> X/Y/Z.

    Qubits untouched by every word in the group are omitted.
    """
    basis: dict[int, str] = {}
    for w in group:
        for j in w.support():
            letter = w.letter(j)
            if basis.setdefault(j, letter) != letter:
                raise ValueError("group is not qubit-wise commuting")
    return basis
