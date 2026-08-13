"""Backend protocol: the algorithm layer talks to execution only through this.

``Backend`` covers exact evaluation (state preparation and expectations);
``SamplingBackend`` adds finite-shot single-word Pauli sampling, returning a
``MeasurementBatch`` whose counts a ``WordCache`` can accumulate across
allocation rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, Union, runtime_checkable

from ..multivector import MV
from ..ir import PauliSum, PauliWord, Program


@dataclass(frozen=True)
class GroupSample:
    """Joint outcome histogram of one commuting measurement setting.

    ``support`` is the sorted tuple of measured qubits; ``basis`` is the
    per-qubit measurement letter (``((qubit, 'X'|'Y'|'Z'), ...)``) that
    identifies the group's circuit and determines which words it can read;
    ``hist`` maps each observed length-``|support|`` bitstring (aligned to
    ``support``) to its shot count; ``shots`` is the group total. From this
    histogram the mean, variance, and covariance of every word read out of
    the group are exact, which is what makes the candidate variance
    covariance-aware.

    The default representation is a QWC setting: ``basis`` determines the
    parity positions for every readable word.  A Clifford-diagonalized
    commuting setting instead supplies a stable ``setting_key`` and explicit
    ``readouts`` mapping ``word_code -> (sign, positions)``.  Its word outcome
    is ``sign * (-1)**parity(bits[positions])``.  Keeping both forms in one
    outcome contract lets the nonlinear estimator consume real joint samples
    from the dyadic hierarchy without pretending that it was measured QWC.
    """

    support: tuple
    basis: tuple
    hist: Mapping[str, int]
    shots: int
    word_codes: tuple[int, ...] = ()
    setting_key: tuple | None = None
    readouts: Mapping[int, tuple[int, tuple[int, ...]]] = field(default_factory=dict)

    def __post_init__(self):
        support = tuple(int(q) for q in self.support)
        basis = tuple((int(q), str(letter)) for q, letter in self.basis)
        hist = {str(bits): int(count) for bits, count in self.hist.items()}
        shots = int(self.shots)
        if shots < 0 or any(count < 0 for count in hist.values()):
            raise ValueError("group shots and counts must be non-negative")
        if sum(hist.values()) != shots:
            raise ValueError("group histogram counts must sum to shots")
        if len(set(support)) != len(support):
            raise ValueError("group support qubits must be distinct")
        if any(len(bits) != len(support) or set(bits) - {"0", "1"} for bits in hist):
            raise ValueError("group histogram keys must be binary strings aligned to support")
        setting_key = None if self.setting_key is None else tuple(self.setting_key)
        readouts = {}
        for code, raw in self.readouts.items():
            sign, positions = raw
            sign = int(sign)
            positions = tuple(int(position) for position in positions)
            if sign not in (-1, 1):
                raise ValueError("readout signs must be +1 or -1")
            if len(set(positions)) != len(positions):
                raise ValueError("readout positions must be distinct")
            if any(not 0 <= position < len(support) for position in positions):
                raise ValueError("readout position lies outside the histogram support")
            readouts[int(code)] = (sign, positions)
        word_codes = tuple(dict.fromkeys(int(c) for c in self.word_codes))
        if setting_key is not None and any(code not in readouts for code in word_codes):
            raise ValueError("every assigned word needs an explicit readout")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "hist", MappingProxyType(hist))
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "word_codes", word_codes)
        object.__setattr__(self, "setting_key", setting_key)
        object.__setattr__(self, "readouts", MappingProxyType(readouts))


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
