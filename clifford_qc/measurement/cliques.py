"""Deterministic unitary partitioning of a fixed real Pauli functional.

A setting measures only its coefficient-weighted clique, not the individual
word means. Its generally non-Clifford rotations therefore cannot replace a
``CompiledSetting`` in the reusable matrix-element bank. No sampler, allocation
policy or finite-shot confidence contract is supplied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

from ..ir import PauliSum, PauliWord, Rotor
from ..qasm3 import lower_rotor
from .bank import _pauli_anticommute
from .cost import SettingResources


COVER_RULE = "commuting-degree-desc/code-asc/first-compatible-v1"


def _words(words: Sequence[PauliWord]) -> tuple[PauliWord, ...]:
    result = tuple(words)
    if any(not isinstance(word, PauliWord) for word in result):
        raise TypeError("every member must be a PauliWord")
    if result and any(word.n != result[0].n for word in result):
        raise ValueError("all words must have the same qubit count")
    if any(word.code == 0 or isinstance(word.code, bool) for word in result):
        raise ValueError("clique words must be non-identity Pauli words")
    if len({word.code for word in result}) != len(result):
        raise ValueError("clique words must be distinct")
    return result


def anticommuting_clique_cover(
    words: Sequence[PauliWord],
) -> tuple[tuple[PauliWord, ...], ...]:
    """Greedy disjoint cover, independent of the input permutation.

    Order by descending *commuting* degree among the other input words, then
    ascending packed Pauli code. Insert each word into the first previously
    created clique it anticommutes with in full, or create a new clique. Within
    each clique, ascending code fixes the pivot and Givens order. This is a
    constructive cover, not a minimum-clique-cover solver. Identity and
    duplicate words are rejected; an empty input returns an empty cover.
    """
    members = sorted(_words(words), key=lambda word: word.code)
    conflicts = {word.code: set() for word in members}
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if not _pauli_anticommute(a.n, a.code, b.code):
                conflicts[a.code].add(b.code)
                conflicts[b.code].add(a.code)
    order = sorted(members, key=lambda word: (-len(conflicts[word.code]), word.code))
    groups: list[list[PauliWord]] = []
    for word in order:
        for group in groups:
            if all(other.code not in conflicts[word.code] for other in group):
                group.append(word)
                break
        else:
            groups.append([word])
    return tuple(tuple(sorted(group, key=lambda word: word.code)) for group in groups)


def _real_coefficient(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("clique coefficients must be real numbers")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("clique coefficients must be finite")
    return value


def _setting_resources(ops: tuple[tuple, ...]) -> SettingResources:
    """Price the emitted circuit with a feasible type-homogeneous schedule.

    One shared per-qubit clock preserves dependencies through both gate kinds.
    A gate goes into the earliest available layer of its kind after its qubits'
    predecessors (the same scheduling convention as block synthesis). Separate
    1q/2q clocks would incorrectly erase dependencies through the other kind.
    Measurement/reset/preparation and device routing belong to the device card.
    """
    ready: dict[int, int] = {}
    layers: dict[int, int] = {}
    counts = {1: 0, 2: 0}
    for op in ops:
        if op[0] == "cx":
            qubits = op[1:]
        elif op[0] in {"h", "s", "sdg", "rz"}:
            qubits = (op[-1],)
        else:
            raise ValueError(f"unsupported clique circuit gate {op[0]!r}")
        kind = len(qubits)
        counts[kind] += 1
        layer = 1 + max(ready.get(q, 0) for q in qubits)
        while layers.get(layer, kind) != kind:
            layer += 1
        layers[layer] = kind
        for q in qubits:
            ready[q] = layer
    return SettingResources(
        n_1q=counts[1], n_2q=counts[2],
        d_1q=sum(kind == 1 for kind in layers.values()),
        d_2q=sum(kind == 2 for kind in layers.values()),
    )


@dataclass(frozen=True)
class CompiledClique:
    """One fixed observable and its executable rotation/readout contract.

    Apply ``ops`` in circuit order and measure Z on ``readout_qubits``. Multiply
    their +/-1 outcomes and then ``weight`` to estimate ``sum(h_i A_i)``.
    ``ops`` includes the final local pivot basis changes, without a readout CX
    ladder: the pivot is read as the product of its support's Z outcomes.

    For m >= 2, ``rotations`` implements V B V† = +A_1 with
    B = sum(normalized_coefficients[i] A_i); the sign of i A_1 A_k is folded
    into the unsigned IR rotor's angle. For a singleton V is empty and its
    coefficient sign stays in ``weight``. All values are immutable snapshots.
    """

    words: tuple[PauliWord, ...]
    coefficients: tuple[float, ...]
    normalized_coefficients: tuple[float, ...]
    weight: float
    rotations: tuple[Rotor, ...]
    ops: tuple[tuple, ...]
    readout_qubits: tuple[int, ...]
    resources: SettingResources

    @property
    def pivot(self) -> PauliWord:
        return self.words[0]


def compile_anticommuting_clique(
    words: Sequence[PauliWord], coefficients: Sequence[Real],
) -> CompiledClique:
    """Compile an ordered, nonempty clique; preserve the supplied pivot/order.

    Coefficients may be exactly zero, but their norm must be positive and finite.
    Normalization is scaled to avoid overflow/underflow in the sum of squares.
    No tolerance drops terms or rotations: only an exactly zero Givens angle is
    skipped. In particular (-1, 0, ...) fires its pi rotation for m >= 2.
    """
    members = _words(words)
    if not members:
        raise ValueError("a compiled clique must be nonempty")
    values = tuple(_real_coefficient(value) for value in coefficients)
    if len(values) != len(members):
        raise ValueError("one coefficient is required per word")
    for i, a in enumerate(members):
        if any(not _pauli_anticommute(a.n, a.code, b.code) for b in members[i + 1:]):
            raise ValueError("clique members must pairwise anticommute")
    scale = max(abs(value) for value in values)
    if scale == 0:
        raise ValueError("a compiled clique must have a nonzero coefficient")
    scaled = tuple(value / scale for value in values)
    scaled_norm = math.hypot(*scaled)
    norm = scale * scaled_norm
    if not math.isfinite(norm):
        raise ValueError("clique coefficient norm exceeds the finite float range")
    normalized = tuple(value / scaled_norm for value in scaled)
    pivot = members[0]
    rotations: list[Rotor] = []
    ops: list[tuple] = []
    accumulated = normalized[0]
    for word, component in zip(members[1:], normalized[1:]):
        angle = math.atan2(component, accumulated) if accumulated or component else 0.0
        accumulated = math.hypot(accumulated, component)
        if angle == 0.0:
            continue
        # The Hermitian involution i A_1 A_k is a *signed* Pauli word.
        generator = 1j * (pivot.to_mv() * word.to_mv())
        code, sign = next(iter(generator.terms.items()))
        assert len(generator.terms) == 1 and sign in (-1, 1)
        rotation = Rotor(PauliWord(pivot.n, code), float(sign.real) * angle)
        rotations.append(rotation)
        ops.extend(lower_rotor(rotation.word, rotation.angle))
    for q in pivot.support():
        if pivot.letter(q) == "Y":
            ops.append(("sdg", q))
        if pivot.letter(q) in {"X", "Y"}:
            ops.append(("h", q))
    emitted = tuple(ops)
    return CompiledClique(
        words=members, coefficients=values, normalized_coefficients=normalized,
        weight=values[0] if len(members) == 1 else norm,
        rotations=tuple(rotations), ops=emitted, readout_qubits=pivot.support(),
        resources=_setting_resources(emitted),
    )


@dataclass(frozen=True)
class CliqueMeasurementPlan:
    """Immutable fixed-functional cover; identity is an exact offset.

    The functional mean is ``identity_offset + sum(setting.weight * parity_mean)``.
    Changing coefficients requires recompilation. An identity-only or zero
    functional has no settings and must bypass shot/cost routines requiring a
    nonempty setting list.
    """

    n: int
    identity_offset: float
    settings: tuple[CompiledClique, ...]
    cover_rule: str = COVER_RULE

    @property
    def resources(self) -> tuple[SettingResources, ...]:
        return tuple(setting.resources for setting in self.settings)


def compile_clique_measurement_plan(functional: PauliSum) -> CliqueMeasurementPlan:
    """Freeze the real coefficients, deterministic cover and lowered resources.

    Reject any nonzero imaginary coefficient, even below a Hermiticity tolerance.
    Omit only exact-zero coefficients; the identity receives no measured setting.
    """
    if not isinstance(functional, PauliSum):
        raise TypeError("functional must be a PauliSum")
    values: dict[PauliWord, float] = {}
    offset = 0.0
    for word, coefficient in functional.items():
        if coefficient.imag != 0:
            raise ValueError("functional coefficients must be exactly real")
        value = _real_coefficient(coefficient.real)
        if word.code == 0:
            offset = value
        elif value != 0:
            values[word] = value
    cover = anticommuting_clique_cover(tuple(values))
    return CliqueMeasurementPlan(
        n=functional.n, identity_offset=offset,
        settings=tuple(compile_anticommuting_clique(group, [values[w] for w in group])
                       for group in cover),
    )
