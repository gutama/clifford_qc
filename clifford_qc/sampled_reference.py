"""Preparation-bound observations without an exact-state capability.

The draw callback is the trusted preparation/device boundary. It must return
joint histograms for the requested compiled circuits, bound to this reference's
identity. No state, density, amplitudes or expectation method is exposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable

from .backends.protocol import MeasurementBatch
from .ir import PauliWord
from .reference_states import ReferenceMetadata, _digest


def nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def settings_identity(settings):
    """Bind circuit action as well as keys, assignment and signed readouts."""
    rows = []
    for setting in settings:
        images = []
        for q in range(setting.n):
            for letter in (1, 3):
                phase, image = setting.clifford.conjugate(
                    PauliWord(setting.n, letter << (2*q)))
                images.append([complex(phase).real, complex(phase).imag, image.code])
        rows.append([setting.n, setting.key, setting.assigned_word_codes,
                     sorted(setting.readouts.items()), images])
    return _digest(rows)


def shot_vector(settings, shots):
    settings = tuple(settings)
    if isinstance(shots, Integral) and not isinstance(shots, bool):
        counts = (nonnegative_integer(shots, "shots"),) * len(settings)
    else:
        counts = tuple(nonnegative_integer(s, "shots") for s in shots)
    if len(counts) != len(settings):
        raise ValueError("one shot count is required per setting")
    if len({s.key for s in settings}) != len(settings):
        raise ValueError("setting keys must be unique")
    codes = [c for s in settings for c in s.assigned_word_codes]
    if len(codes) != len(set(codes)):
        raise ValueError("each word must be assigned exactly once")
    return counts


def validate_batch(batch, reference_identity, n, settings, counts):
    """Reject wrong preparations, readouts, assignments or physical costs."""
    if not isinstance(batch, MeasurementBatch):
        raise TypeError("draw must return a MeasurementBatch")
    if batch.n != n or batch.state_key != reference_identity:
        raise ValueError("observations do not match the declared reference")
    expected = {s.key: (s, count) for s, count in zip(settings, counts) if count}
    actual = {g.setting_key: g for g in batch.groups}
    if len(actual) != len(batch.groups) or actual.keys() != expected.keys():
        raise ValueError("observed settings differ from the frozen schedule")
    word_shots, plus = {}, {}
    for key, (setting, count) in expected.items():
        group = actual[key]
        if (group.shots != count or group.support != tuple(range(n)) or group.basis
                or group.word_codes != setting.assigned_word_codes
                or dict(group.readouts) != dict(setting.readouts)):
            raise ValueError("observed shots/readouts differ from the frozen schedule")
        for code in group.word_codes:
            sign, positions = group.readouts[code]
            word_shots[code] = count
            plus[code] = sum(value for bits, value in group.hist.items()
                             if sign * (-1)**sum(bits[p] == "1" for p in positions) == 1)
    if (dict(batch.shots) != word_shots or dict(batch.plus_counts) != plus
            or batch.circuits != len(expected) or batch.hardware_shots != sum(counts)):
        raise ValueError("word counts or physical cost disagree with joint histograms")


@dataclass(frozen=True)
class SampledObservations:
    reference_identity: str
    seed: int
    settings_identity: str
    counts: tuple[int, ...]
    batch: MeasurementBatch

    @property
    def identity(self):
        return _digest([self.reference_identity, self.seed, self.settings_identity,
                        self.counts, [[g.setting_key, sorted(g.hist.items())]
                                      for g in self.batch.groups]])


@dataclass(frozen=True)
class PreparedSampledReference:
    """Concrete sampled-only reference; ``draw(settings, counts, seed, id)``.

    Distinct seeds identify independent IID random streams in the supported
    simulator protocol. Devices must provide fresh IID preparations; metadata
    validation cannot establish a device's physical independence.
    """
    n: int
    preparation_identity: str
    draw: Callable = field(repr=False, compare=False)
    spin_ordering: str = "qubits"
    metadata: ReferenceMetadata = field(init=False)

    def __post_init__(self):
        if not isinstance(self.preparation_identity, str) or not self.preparation_identity:
            raise ValueError("preparation identity must be a nonempty string")
        if not callable(self.draw):
            raise TypeError("draw must be callable")
        identity = _digest(["sampled_reference.v1", self.n,
                            self.preparation_identity, self.spin_ordering])
        object.__setattr__(self, "metadata", ReferenceMetadata(
            self.n, identity, frozenset({"sample_joint"}), spin_ordering=self.spin_ordering))

    def sample(self, settings, shots, *, seed):
        seed = nonnegative_integer(seed, "seed")
        settings = tuple(settings)
        if any(s.n != self.n for s in settings):
            raise ValueError("setting and reference register sizes differ")
        counts = shot_vector(settings, shots)
        circuit_identity = settings_identity(settings)
        batch = self.draw(settings, counts, seed, self.metadata.identity)
        if circuit_identity != settings_identity(settings):
            raise ValueError("sampling callback changed the circuits")
        validate_batch(batch, self.metadata.identity, self.n, settings, counts)
        return SampledObservations(self.metadata.identity, seed, circuit_identity, counts, batch)


__all__ = ["PreparedSampledReference", "SampledObservations"]
