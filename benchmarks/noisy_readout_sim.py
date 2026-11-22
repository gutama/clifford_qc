"""Gate-level noisy readout simulator for compiled graph-state settings.

This is a simulation device used strictly at the sampling boundary of
``PreparedSampledReference`` and for post-hoc infinite-shot bias anchors.
It is never passed to inference, grouping, or shot allocation as an oracle.

Channel conventions (matching ``clifford_qc/channels.py`` and standard Pauli
error-rate definitions):

- **1-qubit depolarizing** after each native local gate (``H``, ``S``, ``SDG``)
  with Pauli error probability ``p_1q`` in ``[0, 0.75]``:
  ``E_1q(rho) = (1 - l1) * rho + l1 * (I/2) x Tr_q(rho)``, ``l1 = (4/3) * p_1q``.
- **2-qubit depolarizing** after each native ``CZ`` gate with Pauli error
  probability ``p_2q`` in ``[0, 15/16]`` and routing multiplier ``m >= 1``:
  composing ``m`` identical 2Q depolarizing channels gives
  ``E_2q(rho) = (1 - l2) * rho + l2 * (I_4/4) x Tr_{q0,q1}(rho)`` with
  ``l2 = 1 - (1 - (16/15) * p_2q)**m``.
- **Symmetric bit-flip readout error** with probability ``p_readout`` in
  ``[0, 0.5]`` applied independently to every qubit on the final Z-basis
  outcome distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np

from clifford_qc.backends.protocol import GroupSample, MeasurementBatch
from clifford_qc.measurement.cost import DeviceCard

_H_MAT = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
_S_MAT = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
_SDG_MAT = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=complex)
_LOCAL_MATRICES = {"H": _H_MAT, "S": _S_MAT, "SDG": _SDG_MAT}

MAX_DENSITY_QUBITS = 10


@dataclass(frozen=True)
class ReadoutNoiseModel:
    """Declarative readout circuit noise parameters."""

    p_1q: float = 0.0
    p_2q: float = 0.0
    routing_2q_multiplier: float = 1.0
    p_readout: float = 0.0
    label: str = "custom"

    def __post_init__(self) -> None:
        for name, val, lo, hi in (
            ("p_1q", self.p_1q, 0.0, 0.75),
            ("p_2q", self.p_2q, 0.0, 15.0 / 16.0),
            ("p_readout", self.p_readout, 0.0, 0.5),
        ):
            if isinstance(val, bool) or not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite")
            fval = float(val)
            if not (lo <= fval <= hi):
                raise ValueError(f"{name} must lie in [{lo}, {hi}]")
            object.__setattr__(self, name, fval)
        if isinstance(self.routing_2q_multiplier, bool) or not math.isfinite(
            float(self.routing_2q_multiplier)
        ):
            raise ValueError("routing_2q_multiplier must be finite")
        mult = float(self.routing_2q_multiplier)
        if mult < 1.0:
            raise ValueError("routing_2q_multiplier must be at least 1.0")
        object.__setattr__(self, "routing_2q_multiplier", mult)

    @classmethod
    def ideal(cls) -> "ReadoutNoiseModel":
        return cls(p_1q=0.0, p_2q=0.0, routing_2q_multiplier=1.0, p_readout=0.0, label="ideal")

    @classmethod
    def from_device_card(
        cls,
        card: DeviceCard,
        *,
        include_1q: bool = True,
        include_readout: bool = True,
        label: str | None = None,
    ) -> "ReadoutNoiseModel":
        return cls(
            p_1q=card.eps_1q if include_1q else 0.0,
            p_2q=card.eps_2q,
            routing_2q_multiplier=card.routing_2q_multiplier,
            p_readout=card.eps_readout if include_readout else 0.0,
            label=card.name if label is None else label,
        )

    @property
    def is_gate_noiseless(self) -> bool:
        return self.p_1q == 0.0 and self.p_2q == 0.0

    @property
    def lambda_1q(self) -> float:
        return (4.0 / 3.0) * self.p_1q

    @property
    def lambda_2q_routed(self) -> float:
        if self.p_2q == 0.0:
            return 0.0
        return 1.0 - (1.0 - (16.0 / 15.0) * self.p_2q) ** self.routing_2q_multiplier

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "lambda_1q": self.lambda_1q,
            "lambda_2q_routed": self.lambda_2q_routed,
        }


def _apply_1q_unitary_density(rho: np.ndarray, mat: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply 1-qubit unitary ``mat`` to qubit ``q`` of rank-2n density tensor."""
    rho = np.tensordot(mat, rho, axes=([1], [q]))
    rho = np.moveaxis(rho, 0, q)
    rho = np.tensordot(mat.conj(), rho, axes=([1], [n + q]))
    return np.moveaxis(rho, 0, n + q)


def _apply_cz_density(rho: np.ndarray, q0: int, q1: int, n: int) -> None:
    """In-place CZ conjugation on rank-2n density tensor."""
    ket_idx: list[Any] = [slice(None)] * (2 * n)
    ket_idx[q0] = 1
    ket_idx[q1] = 1
    rho[tuple(ket_idx)] *= -1.0
    bra_idx: list[Any] = [slice(None)] * (2 * n)
    bra_idx[n + q0] = 1
    bra_idx[n + q1] = 1
    rho[tuple(bra_idx)] *= -1.0


def _apply_1q_depolarizing(rho: np.ndarray, q: int, n: int, lam: float) -> None:
    """In-place 1-qubit depolarizing channel with depolarization strength ``lam``."""
    if lam == 0.0:
        return
    idx00: list[Any] = [slice(None)] * (2 * n)
    idx00[q] = 0
    idx00[n + q] = 0
    idx11: list[Any] = [slice(None)] * (2 * n)
    idx11[q] = 1
    idx11[n + q] = 1
    tr_q = rho[tuple(idx00)] + rho[tuple(idx11)]
    rho *= 1.0 - lam
    half = (0.5 * lam) * tr_q
    rho[tuple(idx00)] += half
    rho[tuple(idx11)] += half


def _apply_2q_depolarizing(rho: np.ndarray, q0: int, q1: int, n: int, lam: float) -> None:
    """In-place 2-qubit depolarizing channel with depolarization strength ``lam``."""
    if lam == 0.0:
        return
    slices = []
    for a in (0, 1):
        for b in (0, 1):
            idx: list[Any] = [slice(None)] * (2 * n)
            idx[q0] = a
            idx[q1] = b
            idx[n + q0] = a
            idx[n + q1] = b
            slices.append(tuple(idx))
    tr_pair = rho[slices[0]] + rho[slices[1]] + rho[slices[2]] + rho[slices[3]]
    rho *= 1.0 - lam
    quarter = (0.25 * lam) * tr_pair
    for sl in slices:
        rho[sl] += quarter


def _apply_symmetric_readout_noise(probs: np.ndarray, n: int, p_readout: float) -> np.ndarray:
    """Apply independent symmetric bit-flip readout channel to Z-basis probabilities."""
    if p_readout == 0.0:
        return probs
    mat = np.array(
        [[1.0 - p_readout, p_readout], [p_readout, 1.0 - p_readout]], dtype=float
    )
    tensor = probs.reshape((2,) * n)
    for q in range(n):
        tensor = np.tensordot(mat, tensor, axes=([1], [q]))
        tensor = np.moveaxis(tensor, 0, q)
    out = tensor.reshape(1 << n)
    out = np.clip(out, 0.0, None)
    return out / float(out.sum())


def _statevector_probabilities(state: np.ndarray, n: int, gates: tuple[tuple[str, tuple[int, ...]], ...]) -> np.ndarray:
    vec = state.copy()
    index = np.arange(len(vec))
    for name, qubits in gates:
        mask = 1 << (n - 1 - qubits[0])
        if name == "CZ":
            second = 1 << (n - 1 - qubits[1])
            vec[(index & mask != 0) & (index & second != 0)] *= -1.0
        elif name == "H":
            low = index[index & mask == 0]
            high = low | mask
            a, b = vec[low].copy(), vec[high].copy()
            vec[low], vec[high] = (a + b) / math.sqrt(2.0), (a - b) / math.sqrt(2.0)
        elif name in ("S", "SDG"):
            vec[index & mask != 0] *= 1.0j if name == "S" else -1.0j
        else:
            raise ValueError(f"unsupported readout gate: {name!r}")
    probs = np.abs(vec) ** 2
    total = float(probs.sum())
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError("measurement state lost normalization")
    return probs / total


class NoisyReadoutSimulator:
    """Gate-level noisy readout device for compiled graph-state settings.

    Caches per-setting exact noisy Z-basis probability distributions keyed by
    ``setting.clifford.identity`` and draws fresh joint multinomial histograms.
    When gate noise is zero (``p_1q == p_2q == 0``), uses the ``O(2**n)``
    statevector path and matches ``JointShotSimulator`` identically.
    """

    def __init__(
        self,
        state: np.ndarray,
        n: int,
        noise: ReadoutNoiseModel | None = None,
    ) -> None:
        vec = np.asarray(state, dtype=complex).copy()
        if vec.ndim != 1 or vec.shape[0] != (1 << n):
            raise ValueError(f"state must be a length-{1 << n} complex vector")
        norm = float(np.vdot(vec, vec).real)
        if not math.isclose(norm, 1.0, abs_tol=1e-12):
            raise ValueError("input statevector must be normalized")
        self.state = vec
        self.n = int(n)
        self.noise = ReadoutNoiseModel.ideal() if noise is None else noise
        if not self.noise.is_gate_noiseless and self.n > MAX_DENSITY_QUBITS:
            raise ValueError(
                f"exact density gate-noise simulation requires n <= {MAX_DENSITY_QUBITS}, got {self.n}"
            )
        self._rho0: np.ndarray | None = (
            None
            if self.noise.is_gate_noiseless
            else np.outer(self.state, self.state.conj()).reshape((2,) * (2 * self.n))
        )
        self.tables: dict[str, np.ndarray] = {}
        self.draw_seconds = 0.0

    def probabilities(self, setting: Any) -> np.ndarray:
        """Exact Z-basis outcome probability distribution for one compiled setting."""
        key = setting.clifford.identity
        cached = self.tables.get(key)
        if cached is not None:
            return cached
        gates = tuple(setting.clifford.gates)
        if self.noise.is_gate_noiseless:
            probs = _statevector_probabilities(self.state, self.n, gates)
        else:
            assert self._rho0 is not None
            rho = self._rho0.copy()
            lam1 = self.noise.lambda_1q
            lam2 = self.noise.lambda_2q_routed
            for name, qubits in gates:
                if name == "CZ":
                    q0, q1 = int(qubits[0]), int(qubits[1])
                    if q0 == q1:
                        raise ValueError("CZ requires two distinct qubits")
                    _apply_cz_density(rho, q0, q1, self.n)
                    _apply_2q_depolarizing(rho, q0, q1, self.n, lam2)
                elif name in _LOCAL_MATRICES:
                    q = int(qubits[0])
                    rho = _apply_1q_unitary_density(rho, _LOCAL_MATRICES[name], q, self.n)
                    _apply_1q_depolarizing(rho, q, self.n, lam1)
                else:
                    raise ValueError(f"unsupported readout gate: {name!r}")
            mat = rho.reshape(1 << self.n, 1 << self.n)
            diag = np.diag(mat)
            if float(np.max(np.abs(diag.imag))) > 1e-10:
                raise ValueError("density diagonal has non-negligible imaginary part")
            probs = np.real(diag)
            if float(np.min(probs)) < -1e-10:
                raise ValueError("density diagonal became negative beyond tolerance")
            probs = np.clip(probs, 0.0, None)
            total = float(probs.sum())
            if not math.isclose(total, 1.0, abs_tol=1e-10):
                raise ValueError(f"density trace drifted from 1: {total}")
            probs = probs / total
        probs = _apply_symmetric_readout_noise(probs, self.n, self.noise.p_readout)
        self.tables[key] = probs
        return probs

    def exact_word_expectations(self, settings: tuple[Any, ...] | list[Any]) -> dict[int, float]:
        """Exact infinite-shot noisy Pauli expectations under ``settings`` (assigned words)."""
        dim = 1 << self.n
        indices = np.arange(dim, dtype=np.int64)
        expectations: dict[int, float] = {0: 1.0}
        for setting in settings:
            probs = self.probabilities(setting)
            for code in setting.assigned_word_codes:
                sign, positions = setting.readouts[code]
                parity = np.zeros(dim, dtype=np.int8)
                for pos in positions:
                    parity ^= (indices >> (self.n - 1 - int(pos))) & 1
                eigvals = np.where(parity == 0, float(sign), -float(sign))
                expectations[int(code)] = float(np.dot(probs, eigvals))
        return expectations

    def draw(
        self,
        settings: tuple[Any, ...] | list[Any],
        counts: tuple[int, ...] | list[int],
        seed: int,
        identity: str,
    ) -> MeasurementBatch:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        groups: list[GroupSample] = []
        shots: dict[int, int] = {}
        plus: dict[int, int] = {}
        for setting, count in zip(settings, counts):
            if not count:
                continue
            drawn = rng.multinomial(int(count), self.probabilities(setting))
            hist = {
                format(i, f"0{self.n}b"): int(value)
                for i, value in enumerate(drawn)
                if value
            }
            groups.append(
                GroupSample(
                    tuple(range(self.n)),
                    (),
                    hist,
                    int(count),
                    setting.assigned_word_codes,
                    setting.key,
                    setting.readouts,
                )
            )
            for code in setting.assigned_word_codes:
                sign, positions = setting.readouts[code]
                shots[code] = int(count)
                plus[code] = sum(
                    value
                    for bits, value in hist.items()
                    if sign * (-1) ** sum(bits[p] == "1" for p in positions) == 1
                )
        result = MeasurementBatch(self.n, shots, plus, len(groups), tuple(groups), identity)
        self.draw_seconds += time.perf_counter() - started
        return result


__all__ = ["MAX_DENSITY_QUBITS", "NoisyReadoutSimulator", "ReadoutNoiseModel"]
