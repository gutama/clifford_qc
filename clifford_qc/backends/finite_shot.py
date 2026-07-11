"""Finite-shot sampling backend.

Simulator-backed for now: exact single-word expectations from the MV state
turned into seeded binomial outcome counts, exactly the sampling model of
the standalone noisy-ADAPT study. Swappable later for a hardware/PennyLane
sampler behind the same ``SamplingBackend`` protocol.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np

from ..multivector import MV
from ..states import expectation
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
