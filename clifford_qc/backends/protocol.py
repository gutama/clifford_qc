"""Backend protocol: the algorithm layer talks to execution only through this.

``Backend`` covers exact evaluation (state preparation and expectations);
``SamplingBackend`` adds finite-shot single-word Pauli sampling, returning a
``MeasurementBatch`` whose counts a ``WordCache`` can accumulate across
allocation rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
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
    word_codes: tuple[int, ...] = ()

    def __post_init__(self):
        support = tuple(int(q) for q in self.support)
        basis = tuple((int(q), str(letter)) for q, letter in self.basis)
        hist = {str(bits): int(count) for bits, count in self.hist.items()}
        shots = int(self.shots)
        if shots < 0 or any(count < 0 for count in hist.values()):
            raise ValueError("group shots and counts must be non-negative")
        if sum(hist.values()) != shots:
            raise ValueError("group histogram counts must sum to shots")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "hist", MappingProxyType(hist))
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "word_codes",
                           tuple(dict.fromkeys(int(c) for c in self.word_codes)))


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
    state_key: object | None = None

    def __post_init__(self):
        shots = {int(code): int(count) for code, count in self.shots.items()}
        plus = {int(code): int(count) for code, count in self.plus_counts.items()}
        if shots.keys() != plus.keys():
            raise ValueError("shots and plus_counts must have identical word codes")
        for code, count in shots.items():
            if count < 0 or not 0 <= plus[code] <= count:
                raise ValueError("shot counts must satisfy 0 <= plus <= shots")
        if int(self.circuits) < 0:
            raise ValueError("circuits must be non-negative")
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "shots", MappingProxyType(shots))
        object.__setattr__(self, "plus_counts", MappingProxyType(plus))
        object.__setattr__(self, "circuits", int(self.circuits))
        object.__setattr__(self, "groups", tuple(self.groups))

    def mean(self, code: int) -> float:
        N = self.shots[code]
        if N <= 0:
            raise ValueError("cannot compute a mean from zero shots")
        return (2.0 * self.plus_counts[code] - N) / N

    @property
    def hardware_shots(self) -> int:
        """Physical circuit executions, without double-counting grouped words."""
        return (sum(group.shots for group in self.groups) if self.groups
                else sum(self.shots.values()))


def state_fingerprint(rho: MV) -> tuple:
    """Value identity used to prevent pooling shots from different states."""
    return rho.n, tuple(sorted(rho.terms.items()))


@runtime_checkable
class Backend(Protocol):
    def state(self, program: Program, values=None, initial_state: MV | None = None) -> MV:
        ...

    def expectation(self, program: Program, observable: PauliSum, values=None,
                    initial_state: MV | None = None) -> float:
        ...


@runtime_checkable
class SamplingBackend(Protocol):
    """Finite-shot capability, deliberately separate from exact expectation."""

    def state(self, program: Program, values=None, initial_state: MV | None = None) -> MV:
        ...

    def sample_paulis(self, program: Program, words: Sequence[PauliWord],
                      shots: Union[int, Mapping[int, int]], values=None,
                      initial_state: MV | None = None) -> MeasurementBatch:
        ...

    def sample_words_from_state(self, rho: MV, words: Sequence[PauliWord],
                                shots: Union[int, Mapping[int, int]]) -> MeasurementBatch:
        ...

    def sample_grouped_from_state(self, rho: MV,
                                  groups: Sequence[Sequence[PauliWord]],
                                  shots: Union[int, Mapping[int, int]]) -> MeasurementBatch:
        ...
