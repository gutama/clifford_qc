"""Backend protocol: the algorithm layer talks to execution only through this.

``Backend`` covers exact evaluation (state preparation and expectations);
``SamplingBackend`` adds finite-shot single-word Pauli sampling, returning a
``MeasurementBatch`` whose counts a ``WordCache`` can accumulate across
allocation rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, Union, runtime_checkable

from ..multivector import MV
from ..ir import PauliSum, PauliWord, Program


@dataclass(frozen=True)
class GroupSample:
    """Joint outcome histogram of one qubit-wise-commuting measurement group.

    ``support`` is the sorted tuple of measured qubits; ``basis`` is the
    per-qubit measurement letter (``((qubit, 'X'|'Y'|'Z'), ...)``) that
    identifies the group's circuit and determines which words it can read;
    ``hist`` maps each observed length-``|support|`` bitstring (aligned to
    ``support``) to its shot count; ``shots`` is the group total. From this
    histogram the mean, variance, and covariance of every word read out of
    the group are exact, which is what makes the candidate variance
    covariance-aware.
    """

    support: tuple
    basis: tuple
    hist: Mapping[str, int]
    shots: int


@dataclass(frozen=True)
class MeasurementBatch:
    """Outcome counts of one allocation round of single-word measurements.

    For each measured word ``w``: ``shots[w.code]`` total shots and
    ``plus_counts[w.code]`` outcomes of +1, so the sample mean is
    ``(2*N+ - N)/N``. ``circuits`` counts the distinct measurement circuits
    this batch would need on hardware (one per word without grouping).
    ``groups`` optionally carries the per-group joint histograms (one
    ``GroupSample`` per QWC circuit) that a grouped cache uses for
    covariance-aware variance; it is empty for ungrouped single-word batches.
    """

    n: int
    shots: Mapping[int, int]
    plus_counts: Mapping[int, int]
    circuits: int
    groups: tuple = ()

    def mean(self, code: int) -> float:
        N = self.shots[code]
        return (2.0 * self.plus_counts[code] - N) / N


@runtime_checkable
class Backend(Protocol):
    def state(self, program: Program, values=None, initial_state: MV | None = None) -> MV:
        ...

    def expectation(self, program: Program, observable: PauliSum, values=None,
                    initial_state: MV | None = None) -> float:
        ...


@runtime_checkable
class SamplingBackend(Backend, Protocol):
    def sample_paulis(self, program: Program, words: Sequence[PauliWord],
                      shots: Union[int, Mapping[int, int]], values=None,
                      initial_state: MV | None = None) -> MeasurementBatch:
        ...
