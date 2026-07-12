"""Finite-shot sampling backend.

Simulator-backed for now: exact single-word expectations from the MV state
turned into seeded binomial outcome counts, exactly the sampling model of
the standalone noisy-ADAPT study. ``sample_grouped_from_state`` measures a
whole qubit-wise-commuting group per circuit by sampling bitstrings from
the shared rotated-basis distribution, so word outcomes carry the true
joint correlations. Swappable later for a hardware/PennyLane sampler
behind the same ``SamplingBackend`` protocol.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np

from ..multivector import MV
from .. import gates as _gates
from ..states import computational_probabilities, evolve, expectation, partial_trace
from ..ir import PauliSum, PauliWord, Program
from .exact_mv import ExactMVBackend
from .protocol import MeasurementBatch


class FiniteShotBackend:
    """Seeded finite-shot sampler over an exact inner backend."""

    def __init__(self, seed: int, inner: ExactMVBackend | None = None):
        self.rng = np.random.default_rng(seed)
        self.inner = inner if inner is not None else ExactMVBackend()

    def state(self, program: Program, values=None, initial_state: MV | None = None) -> MV:
        return self.inner.state(program, values, initial_state)

    def expectation(self, program: Program, observable: PauliSum, values=None,
                    initial_state: MV | None = None) -> float:
        return self.inner.expectation(program, observable, values, initial_state)

    def sample_paulis(self, program: Program, words: Sequence[PauliWord],
                      shots: Union[int, Mapping[int, int]], values=None,
                      initial_state: MV | None = None) -> MeasurementBatch:
        rho = self.state(program, values, initial_state)
        return self.sample_words_from_state(rho, words, shots)

    def sample_words_from_state(self, rho: MV, words: Sequence[PauliWord],
                                shots: Union[int, Mapping[int, int]]) -> MeasurementBatch:
        """Sample without re-preparing the state (one ADAPT step measures many
        rounds from the same state)."""
        shot_map = ({w.code: int(shots) for w in words} if isinstance(shots, int)
                    else {int(k): int(v) for k, v in shots.items()})
        out_shots: dict[int, int] = {}
        plus: dict[int, int] = {}
        circuits = 0
        for w in words:
            N = shot_map.get(w.code, 0)
            if N < 0:
                raise ValueError("shots must be non-negative")
            if N == 0:
                continue
            ev = expectation(rho, w.to_mv()).real
            p = 0.5 * (1.0 + min(1.0, max(-1.0, ev)))
            out_shots[w.code] = N
            plus[w.code] = int(self.rng.binomial(N, p))
            circuits += 1
        return MeasurementBatch(n=rho.n, shots=out_shots, plus_counts=plus, circuits=circuits)

    def sample_grouped_from_state(self, rho: MV, groups: Sequence[Sequence[PauliWord]],
                                  shots: Union[int, Mapping[int, int]]) -> MeasurementBatch:
        """Jointly sample each qubit-wise-commuting group with one circuit.

        Every group member is measured on every shot of its group's circuit,
        so a word requesting fewer shots than a groupmate still receives the
        group maximum — extra outcomes are free on hardware too.
        """
        from ..measurement.grouping import shared_basis

        all_words = [w for group in groups for w in group]
        shot_map = ({w.code: int(shots) for w in all_words} if isinstance(shots, int)
                    else {int(k): int(v) for k, v in shots.items()})
        out_shots: dict[int, int] = {}
        plus: dict[int, int] = {}
        circuits = 0
        for group in groups:
            N = max((shot_map.get(w.code, 0) for w in group), default=0)
            if N < 0:
                raise ValueError("shots must be non-negative")
            if N == 0:
                continue
            circuits += 1
            # rotate the shared basis onto Z: X -> H, Y -> H*SDG per qubit
            rho_rot = rho
            for j, letter in shared_basis(group).items():
                if letter == "X":
                    rho_rot = evolve(rho_rot, _gates.H(rho.n, j))
                elif letter == "Y":
                    rho_rot = evolve(rho_rot, _gates.H(rho.n, j) * _gates.S(rho.n, j).dagger())
            keep = tuple(sorted({j for w in group for j in w.support()}))
            traced = rho_rot if len(keep) == rho.n else \
                partial_trace(rho_rot, {j for j in range(rho.n) if j not in keep})
            outcomes = sorted(computational_probabilities(traced).items())
            probs = np.clip([p for _, p in outcomes], 0.0, None)
            probs = probs / probs.sum()
            counts = self.rng.multinomial(N, probs)
            position = {q: i for i, q in enumerate(keep)}
            for w in group:
                positions = [position[j] for j in w.support()]
                n_plus = sum(int(c) for (bits, _), c in zip(outcomes, counts)
                             if sum(bits[pos] == "1" for pos in positions) % 2 == 0)
                out_shots[w.code] = out_shots.get(w.code, 0) + N
                plus[w.code] = plus.get(w.code, 0) + n_plus
        return MeasurementBatch(n=rho.n, shots=out_shots, plus_counts=plus, circuits=circuits)
