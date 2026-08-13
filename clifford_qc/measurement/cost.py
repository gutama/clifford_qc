"""Declared device-card accounting for grouped Pauli measurements.

This module prices already-synthesized measurement settings.  It does not
choose a subspace, allocate shots to an accuracy target, or turn the simple
independent-error fidelity surrogate into a hardware calibration claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DEVICE_CARD_SCHEMA = "clifford_qc.device_card.v1"
COST_LEDGER_SCHEMA = "clifford_qc.measurement_cost.v1"
BREAK_EVEN_SCHEMA = "clifford_qc.measurement_break_even.v1"
EVIDENCE_TIERS = frozenset({"exact", "asymptotic", "finite_sample"})
_TIME_REL_TOL = 1e-12
_TIME_ABS_TOL_US = 1e-9


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class DeviceCard:
    """Versioned timing and independent-error scenario for one device class."""

    name: str
    t_prep_us: float
    t_1q_us: float
    t_2q_us: float
    t_readout_us: float
    t_reset_us: float
    eps_1q: float
    eps_2q: float
    eps_readout: float
    connectivity: str
    routing: bool
    fidelity_floor: float = 0.5
    calibration_status: str = "illustrative_scenario"
    description: str = ""
    source: str = ""
    routing_2q_multiplier: float = 1.0
    routing_depth_multiplier: float = 1.0
    schema: str = DEVICE_CARD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DEVICE_CARD_SCHEMA:
            raise ValueError(
                f"unsupported device-card schema {self.schema!r}; "
                f"expected {DEVICE_CARD_SCHEMA!r}"
            )
        for field in ("name", "connectivity", "calibration_status"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if type(self.routing) is not bool:
            raise TypeError("routing must be a boolean")
        for field in (
            "t_prep_us", "t_1q_us", "t_2q_us", "t_readout_us", "t_reset_us"
        ):
            value = _finite_number(getattr(self, field), field)
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative")
        for field in ("eps_1q", "eps_2q", "eps_readout"):
            value = _finite_number(getattr(self, field), field)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{field} must lie in [0, 1)")
        floor = _finite_number(self.fidelity_floor, "fidelity_floor")
        if not 0.0 < floor <= 1.0:
            raise ValueError("fidelity_floor must lie in (0, 1]")
        for field in ("routing_2q_multiplier", "routing_depth_multiplier"):
            value = _finite_number(getattr(self, field), field)
            if value < 1.0:
                raise ValueError(f"{field} must be at least one")
            if not self.routing and value != 1.0:
                raise ValueError(f"{field} must be one when routing is false")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DeviceCard":
        allowed = {
            "schema", "name", "t_prep_us", "t_1q_us", "t_2q_us",
            "t_readout_us", "t_reset_us", "eps_1q", "eps_2q",
            "eps_readout", "connectivity", "routing", "fidelity_floor",
            "calibration_status", "description", "source",
            "routing_2q_multiplier", "routing_depth_multiplier",
        }
        extra = sorted(set(data) - allowed)
        if extra:
            raise ValueError(f"unknown device-card fields: {extra}")
        required = allowed - {
            "fidelity_floor", "calibration_status", "description", "source",
            "routing_2q_multiplier", "routing_depth_multiplier",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"missing device-card fields: {missing}")
        return cls(**dict(data))

    @classmethod
    def load(cls, path: str | Path) -> "DeviceCard":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("a device card must be a JSON object")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "name": self.name,
            "t_prep_us": self.t_prep_us,
            "t_1q_us": self.t_1q_us,
            "t_2q_us": self.t_2q_us,
            "t_readout_us": self.t_readout_us,
            "t_reset_us": self.t_reset_us,
            "eps_1q": self.eps_1q,
            "eps_2q": self.eps_2q,
            "eps_readout": self.eps_readout,
            "connectivity": self.connectivity,
            "routing": self.routing,
            "fidelity_floor": self.fidelity_floor,
            "calibration_status": self.calibration_status,
            "description": self.description,
            "source": self.source,
            "routing_2q_multiplier": self.routing_2q_multiplier,
            "routing_depth_multiplier": self.routing_depth_multiplier,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SettingResources:
    """Logical resources of one compiled measurement setting."""

    n_1q: int
    n_2q: int
    d_1q: int
    d_2q: int

    def __post_init__(self) -> None:
        for field in ("n_1q", "n_2q", "d_1q", "d_2q"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        if (self.n_1q == 0) != (self.d_1q == 0):
            raise ValueError("n_1q and d_1q must be zero together")
        if (self.n_2q == 0) != (self.d_2q == 0):
            raise ValueError("n_2q and d_2q must be zero together")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SettingResources":
        return cls(**{field: data[field] for field in ("n_1q", "n_2q", "d_1q", "d_2q")})


def _resources(values: Sequence[SettingResources | Mapping[str, object]]) -> list[SettingResources]:
    result = [
        value if isinstance(value, SettingResources) else SettingResources.from_mapping(value)
        for value in values
    ]
    if not result:
        raise ValueError("at least one measurement setting is required")
    return result


def _shot_vector(shots: int | Sequence[int], settings: int) -> np.ndarray:
    if isinstance(shots, bool):
        raise TypeError("shots must be an integer or integer sequence")
    if isinstance(shots, int):
        values = np.full(settings, shots, dtype=np.int64)
    else:
        raw = list(shots)
        if len(raw) != settings:
            raise ValueError("shot vector length must match the number of settings")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
            raise TypeError("every shot count must be an integer")
        values = np.asarray(raw, dtype=np.int64)
    if bool((values < 0).any()):
        raise ValueError("shot counts must be non-negative")
    return values


def setting_duration_us(card: DeviceCard, setting: SettingResources) -> float:
    """Wall-clock surrogate from logical depths, readout, reset, and preparation."""
    routed_d_2q = math.ceil(setting.d_2q * card.routing_depth_multiplier)
    return (
        card.t_prep_us
        + card.t_1q_us * setting.d_1q
        + card.t_2q_us * routed_d_2q
        + card.t_readout_us
        + card.t_reset_us
    )


def setting_fidelity(card: DeviceCard, setting: SettingResources, n_qubits: int) -> float:
    """Independent depolarizing-style fidelity surrogate declared by the card."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    routed_n_2q = math.ceil(setting.n_2q * card.routing_2q_multiplier)
    return (
        (1.0 - card.eps_1q) ** setting.n_1q
        * (1.0 - card.eps_2q) ** routed_n_2q
        * (1.0 - card.eps_readout) ** n_qubits
    )


def inflate_shots_for_fidelity(
    card: DeviceCard,
    settings: Sequence[SettingResources | Mapping[str, object]],
    effective_shots: int | Sequence[int],
    n_qubits: int,
) -> list[int]:
    """Raw shots needed for a requested effective schedule under ``F^-2``."""
    compiled = _resources(settings)
    target = _shot_vector(effective_shots, len(compiled))
    fidelities = np.asarray(
        [setting_fidelity(card, item, n_qubits) for item in compiled], dtype=float
    )
    scaled = target / np.square(fidelities)
    if not bool(np.isfinite(scaled).all()) or bool(
        (scaled > np.iinfo(np.int64).max).any()
    ):
        raise OverflowError("fidelity inflation exceeds the int64 shot ledger")
    raw = np.ceil(scaled).astype(np.int64)
    return [int(value) for value in raw]


def cost_schedule(
    card: DeviceCard,
    settings: Sequence[SettingResources | Mapping[str, object]],
    shots: int | Sequence[int],
    *,
    n_qubits: int,
    evidence_tier: str,
    epsilon: float | None = None,
) -> dict[str, object]:
    """Price an explicit raw-shot schedule under one named device card.

    Passing ``epsilon`` asserts only that the caller supplied the schedule from
    an external shot-to-target search.  This function never infers accuracy
    from group count or fidelity.
    """
    if evidence_tier not in EVIDENCE_TIERS:
        raise ValueError(f"evidence_tier must be one of {sorted(EVIDENCE_TIERS)}")
    if epsilon is not None:
        epsilon = _finite_number(epsilon, "epsilon")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
    compiled = _resources(settings)
    shot_vector = _shot_vector(shots, len(compiled))
    durations = np.asarray([setting_duration_us(card, item) for item in compiled])
    fidelities = np.asarray(
        [setting_fidelity(card, item, n_qubits) for item in compiled], dtype=float
    )
    admissible = fidelities >= card.fidelity_floor
    raw_total = int(shot_vector.sum())
    effective = shot_vector * np.square(fidelities)
    fixed_time = float(np.dot(shot_vector, durations))
    weighted_n_1q = sum(int(shots_g) * item.n_1q for shots_g, item in zip(shot_vector, compiled))
    weighted_n_2q = sum(int(shots_g) * item.n_2q for shots_g, item in zip(shot_vector, compiled))
    modeled_n_2q = sum(
        int(shots_g) * math.ceil(item.n_2q * card.routing_2q_multiplier)
        for shots_g, item in zip(shot_vector, compiled)
    )
    accuracy_status = "fixed_shot_only"
    c_time_epsilon = None
    if epsilon is not None:
        accuracy_status = "priced" if bool(admissible.all()) else "inadmissible"
        if bool(admissible.all()):
            c_time_epsilon = fixed_time
    return {
        "schema": COST_LEDGER_SCHEMA,
        "device_card": {
            "name": card.name,
            "sha256": card.sha256,
            "calibration_status": card.calibration_status,
        },
        "n_qubits": n_qubits,
        "settings": len(compiled),
        "raw_shots": raw_total,
        "effective_shots": float(effective.sum()),
        "fixed_shot_time_us": fixed_time,
        "logical_1q_applications": weighted_n_1q,
        "logical_2q_applications": weighted_n_2q,
        "modeled_2q_applications": modeled_n_2q,
        "routing": {
            "required": card.routing,
            "two_qubit_count_multiplier": card.routing_2q_multiplier,
            "two_qubit_depth_multiplier": card.routing_depth_multiplier,
            "model": "declared scalar sensitivity surrogate",
        },
        "fidelity": {
            "min": float(fidelities.min()),
            "mean": float(fidelities.mean()),
            "max": float(fidelities.max()),
            "floor": card.fidelity_floor,
        },
        "admissible": bool(admissible.all()),
        "inadmissible_settings": int((~admissible).sum()),
        "accuracy": {
            "epsilon": epsilon,
            "evidence_tier": evidence_tier,
            "status": accuracy_status,
            "C_time_epsilon_us": c_time_epsilon,
        },
    }


def estimator_information(
    compatibility: Sequence[Sequence[bool]],
    assignment: Sequence[int],
    shots: int | Sequence[int],
    *,
    estimator: str,
    fidelities: Sequence[float] | None = None,
) -> dict[str, object]:
    """Summarize raw and fidelity-adjusted word observations.

    ``compatibility[g][w]`` says setting ``g`` reads word ``w``.  The assigned
    estimator uses exactly ``assignment[w]``; the pooled estimator uses every
    compatible setting.  No independence assumption is made across words.
    """
    matrix = np.asarray(compatibility, dtype=bool)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("compatibility must be a non-empty settings-by-words matrix")
    n_settings, n_words = matrix.shape
    assigned = np.asarray(assignment, dtype=np.int64)
    if assigned.shape != (n_words,):
        raise ValueError("assignment must contain one setting index per word")
    if bool(((assigned < 0) | (assigned >= n_settings)).any()):
        raise ValueError("assignment contains an out-of-range setting index")
    if not bool(matrix[assigned, np.arange(n_words)].all()):
        raise ValueError("every assigned setting must be compatible with its word")
    shot_vector = _shot_vector(shots, n_settings).astype(float)
    if fidelities is None:
        fidelity_vector = np.ones(n_settings, dtype=float)
    else:
        fidelity_vector = np.asarray(fidelities, dtype=float)
        if fidelity_vector.shape != (n_settings,):
            raise ValueError("fidelities must contain one value per setting")
        if not bool(np.isfinite(fidelity_vector).all()) or bool(
            ((fidelity_vector < 0.0) | (fidelity_vector > 1.0)).any()
        ):
            raise ValueError("fidelities must be finite values in [0, 1]")
    if estimator == "single_assignment":
        coverage = np.ones(n_words, dtype=np.int64)
        raw_word_shots = shot_vector[assigned]
        effective_word_shots = raw_word_shots * np.square(fidelity_vector[assigned])
    elif estimator == "pooled":
        coverage = matrix.sum(axis=0)
        raw_word_shots = matrix.T @ shot_vector
        effective_word_shots = matrix.T @ (shot_vector * np.square(fidelity_vector))
    else:
        raise ValueError("estimator must be 'single_assignment' or 'pooled'")
    fractions = coverage / n_settings
    return {
        "estimator": estimator,
        "word_samples_per_preparation": (
            float(raw_word_shots.sum() / shot_vector.sum())
            if float(shot_vector.sum()) > 0.0 else None
        ),
        "coverage_fraction": {
            "min": float(fractions.min()),
            "mean": float(fractions.mean()),
            "max": float(fractions.max()),
        },
        "compatible_settings_per_word": {
            "min": int(coverage.min()),
            "mean": float(coverage.mean()),
            "max": int(coverage.max()),
        },
        "raw_word_shots": {
            "min": float(raw_word_shots.min()),
            "mean": float(raw_word_shots.mean()),
            "max": float(raw_word_shots.max()),
            "total": float(raw_word_shots.sum()),
        },
        "effective_word_shots": {
            "min": float(effective_word_shots.min()),
            "mean": float(effective_word_shots.mean()),
            "max": float(effective_word_shots.max()),
            "total": float(effective_word_shots.sum()),
        },
    }


def break_even_surface(
    card: DeviceCard,
    reference: Sequence[SettingResources | Mapping[str, object]],
    candidate: Sequence[SettingResources | Mapping[str, object]],
    effective_shots: int,
    *,
    n_qubits: int,
    t_2q_ratios: Sequence[float],
    eps_2q_values: Sequence[float],
    evidence_tier: str,
    epsilon: float | None = None,
) -> dict[str, object]:
    """Evaluate the winner over ``t_2q/(t_readout+t_reset)`` and ``eps_2q``."""
    reference = _resources(reference)
    candidate = _resources(candidate)
    cycle = card.t_readout_us + card.t_reset_us
    if cycle <= 0.0:
        raise ValueError("t_readout_us + t_reset_us must be positive for a ratio surface")
    points: list[dict[str, object]] = []
    for ratio_value in t_2q_ratios:
        ratio = _finite_number(ratio_value, "t_2q_ratio")
        if ratio < 0.0:
            raise ValueError("t_2q ratios must be non-negative")
        for error_value in eps_2q_values:
            error = _finite_number(error_value, "eps_2q")
            if not 0.0 <= error < 1.0:
                raise ValueError("eps_2q values must lie in [0, 1)")
            scenario = replace(card, t_2q_us=ratio * cycle, eps_2q=error)
            ref_raw = inflate_shots_for_fidelity(
                scenario, reference, effective_shots, n_qubits
            )
            candidate_raw = inflate_shots_for_fidelity(
                scenario, candidate, effective_shots, n_qubits
            )
            ref_cost = cost_schedule(
                scenario, reference, ref_raw, n_qubits=n_qubits,
                evidence_tier=evidence_tier, epsilon=epsilon,
            )
            candidate_cost = cost_schedule(
                scenario, candidate, candidate_raw, n_qubits=n_qubits,
                evidence_tier=evidence_tier, epsilon=epsilon,
            )
            if not ref_cost["admissible"] and not candidate_cost["admissible"]:
                winner = "neither"
            elif not ref_cost["admissible"]:
                winner = "candidate"
            elif not candidate_cost["admissible"]:
                winner = "reference"
            else:
                ref_time = ref_cost["fixed_shot_time_us"]
                candidate_time = candidate_cost["fixed_shot_time_us"]
                if math.isclose(
                    candidate_time,
                    ref_time,
                    rel_tol=_TIME_REL_TOL,
                    abs_tol=_TIME_ABS_TOL_US,
                ):
                    winner = "tie"
                else:
                    winner = (
                        "candidate" if candidate_time < ref_time else "reference"
                    )
            points.append({
                "t_2q_over_readout_reset": ratio,
                "eps_2q": error,
                "winner": winner,
                "reference_time_us": ref_cost["fixed_shot_time_us"],
                "candidate_time_us": candidate_cost["fixed_shot_time_us"],
                "reference_admissible": ref_cost["admissible"],
                "candidate_admissible": candidate_cost["admissible"],
            })
    return {
        "schema": BREAK_EVEN_SCHEMA,
        "base_device_card": {"name": card.name, "sha256": card.sha256},
        "effective_shots_per_setting": effective_shots,
        "evidence_tier": evidence_tier,
        "epsilon": epsilon,
        "points": points,
    }


__all__ = [
    "BREAK_EVEN_SCHEMA", "COST_LEDGER_SCHEMA", "DEVICE_CARD_SCHEMA",
    "DeviceCard", "EVIDENCE_TIERS", "SettingResources", "break_even_surface",
    "cost_schedule", "estimator_information", "inflate_shots_for_fidelity",
    "setting_duration_us", "setting_fidelity",
]
