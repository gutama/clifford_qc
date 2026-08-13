"""Joint sampling for explicitly Clifford-diagonalized commuting settings.

QWC sampling can infer every readout from a per-qubit basis.  A fully
commuting setting cannot: after its entangling Clifford, an original Pauli word
becomes a signed product of computational-basis Z outcomes.  ``CompiledSetting``
stores that signed parity map and ``CompiledMeasurementSampler`` draws the
actual shared bitstrings.  No Gaussian or independent-word surrogate enters
this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence, Union

import numpy as np

from ..backends.protocol import GroupSample, MeasurementBatch, state_fingerprint
from ..states import computational_probabilities
from ..subspace.restriction import Restriction


@dataclass(frozen=True)
class CompiledSetting:
    """One commuting setting after exact Clifford synthesis.

    ``readouts[code] = (sign, positions)`` means that measuring the compiled
    circuit gives the original word outcome as ``sign`` times the Z parity at
    those output-qubit positions.  ``assigned_word_codes`` is the partition
    assignment used by the single-assignment estimator; ``readouts`` may be
    wider and therefore supplies the pooled estimator from the same shots.
    """

    key: tuple
    clifford: object
    assigned_word_codes: tuple[int, ...]
    readouts: Mapping[int, tuple[int, tuple[int, ...]]]

    def __post_init__(self) -> None:
        key = tuple(self.key)
        if not key:
            raise ValueError("compiled setting key must be non-empty")
        n = getattr(self.clifford, "n", None)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("compiled setting Clifford must declare a positive n")
        assigned = tuple(dict.fromkeys(int(code) for code in self.assigned_word_codes))
        readouts = {}
        for code, raw in self.readouts.items():
            sign, positions = raw
            sign = int(sign)
            positions = tuple(int(position) for position in positions)
            if sign not in (-1, 1):
                raise ValueError("compiled readout sign must be +1 or -1")
            if len(set(positions)) != len(positions):
                raise ValueError("compiled readout positions must be distinct")
            if any(not 0 <= position < n for position in positions):
                raise ValueError("compiled readout position lies outside the register")
            readouts[int(code)] = (sign, positions)
        if any(code not in readouts for code in assigned):
            raise ValueError("every assigned word needs a compiled readout")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "assigned_word_codes", assigned)
        object.__setattr__(self, "readouts", MappingProxyType(readouts))

    @property
    def n(self) -> int:
        return self.clifford.n


@dataclass(frozen=True)
class _SamplingPlan:
    bits: tuple[str, ...]
    probabilities: np.ndarray


class CompiledMeasurementSampler:
    """Seeded exact-state sampler for :class:`CompiledSetting` objects."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self._plans: dict[tuple, _SamplingPlan] = {}

    def reseed(self, seed: int) -> None:
        """Reset sampled outcomes while retaining deterministic rotations."""
        self.rng = np.random.default_rng(seed)

    def _plan(self, rho, setting: CompiledSetting) -> _SamplingPlan:
        fingerprint = state_fingerprint(rho)
        cache_key = (fingerprint, setting.key)
        plan = self._plans.get(cache_key)
        if plan is not None:
            return plan
        rotated = Restriction.encoding(setting.clifford).state(rho)
        outcomes = sorted(computational_probabilities(rotated).items())
        bits = tuple(bitstring for bitstring, _ in outcomes)
        probabilities = np.asarray([probability for _, probability in outcomes], dtype=float)
        if (not np.all(np.isfinite(probabilities))
                or probabilities.min(initial=0.0) < -1e-10
                or abs(float(probabilities.sum()) - 1.0) > 1e-9):
            raise ValueError("compiled setting produced an invalid probability distribution")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        probabilities /= probabilities.sum()
        plan = _SamplingPlan(bits, probabilities)
        self._plans[cache_key] = plan
        return plan

    def prepare(self, rho, settings: Sequence[CompiledSetting]) -> int:
        """Compile all fixed state/setting probability tables without sampling."""
        for setting in settings:
            if setting.n != rho.n:
                raise ValueError("compiled setting and state act on different qubit counts")
            self._plan(rho, setting)
        return len(settings)

    def sample_from_state(
        self,
        rho,
        settings: Sequence[CompiledSetting],
        shots: Union[int, Mapping[tuple, int]],
    ) -> MeasurementBatch:
        """Draw one joint computational histogram per compiled setting."""
        settings = tuple(settings)
        if len({setting.key for setting in settings}) != len(settings):
            raise ValueError("compiled setting keys must be unique")
        assigned = [code for setting in settings for code in setting.assigned_word_codes]
        if len(set(assigned)) != len(assigned):
            raise ValueError("a word is assigned to more than one compiled setting")
        shot_map = None if isinstance(shots, int) else {
            tuple(key): int(value) for key, value in shots.items()
        }
        uniform = int(shots) if isinstance(shots, int) else None
        if uniform is not None and uniform < 0:
            raise ValueError("shots must be non-negative")
        if shot_map is not None and any(value < 0 for value in shot_map.values()):
            raise ValueError("shots must be non-negative")

        groups = []
        out_shots: dict[int, int] = {}
        plus_counts: dict[int, int] = {}
        circuits = 0
        support = tuple(range(rho.n))
        for setting in settings:
            if setting.n != rho.n:
                raise ValueError("compiled setting and state act on different qubit counts")
            count = uniform if uniform is not None else shot_map.get(setting.key, 0)
            if count <= 0:
                continue
            plan = self._plan(rho, setting)
            drawn = self.rng.multinomial(count, plan.probabilities)
            hist = {bits: int(value) for bits, value in zip(plan.bits, drawn) if value}
            groups.append(GroupSample(
                support=support,
                basis=(),
                hist=hist,
                shots=count,
                word_codes=setting.assigned_word_codes,
                setting_key=setting.key,
                readouts=setting.readouts,
            ))
            circuits += 1
            for code in setting.assigned_word_codes:
                sign, positions = setting.readouts[code]
                plus = 0
                for bits, value in hist.items():
                    parity_sign = -1 if sum(bits[p] == "1" for p in positions) % 2 else 1
                    if sign * parity_sign == 1:
                        plus += value
                out_shots[code] = count
                plus_counts[code] = plus
        return MeasurementBatch(
            n=rho.n,
            shots=out_shots,
            plus_counts=plus_counts,
            circuits=circuits,
            groups=tuple(groups),
            state_key=state_fingerprint(rho),
        )


__all__ = ["CompiledSetting", "CompiledMeasurementSampler"]
