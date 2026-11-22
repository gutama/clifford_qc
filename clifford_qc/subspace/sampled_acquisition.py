"""Run the common precision controller on compiled, sampled-only observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Real
from time import perf_counter

import numpy as np

from ..ir import PauliWord
from ..sampled_reference import (SampledObservations, nonnegative_integer,
                                 shot_vector, validate_batch)
from .acquisition import AcquisitionError, RitzAcquisition, acquire_ritz_precision
from .rank_stopping import RankStopPolicy


@dataclass(frozen=True)
class PrecisionAcquisitionConfig:
    """Predeclared physical-shot schedule and precision policy.

    Pilot counts are per compiled setting; subsequent budgets are total
    physical shots. The common controller validates the schedule before drawing.
    Only the raw Ritz estimator is supported: exact-state certification of a
    control-variate family is not a sampled-reference capability.
    """

    pilot_shots_per_group: int
    round_budgets: tuple[int, ...]
    target_std_error: float
    allocation: str = "ritz_targeted"
    rank_stop: RankStopPolicy = RankStopPolicy()
    max_bootstrap_half_width: float | None = None
    bootstrap_replicates: int | None = None
    bootstrap_seed: int = 0
    root: int = 0

    def __post_init__(self):
        object.__setattr__(self, "round_budgets", tuple(self.round_budgets))

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SampledPrecisionRun:
    acquisition: RitzAcquisition | None
    looks: tuple[dict, ...]
    observations: tuple[SampledObservations, ...]
    requests: tuple[dict, ...]
    error: dict | None
    sampling_seconds: float
    inference_seconds: float
    measured_solve_attempts: int
    bootstrap_solve_attempts: int

    @property
    def status(self):
        return self.acquisition.stopped if self.acquisition is not None else "failed"

    @property
    def confirmed_shots(self):
        return sum(o.batch.hardware_shots for o in self.observations)

    @property
    def confirmed_circuits(self):
        return sum(o.batch.circuits for o in self.observations)


class _CompiledSession:
    """Adapter for the common measured solver/allocator, never an exact backend."""

    def __init__(self, provider, seed):
        self.provider = provider
        self.seed = seed
        self.settings = tuple(provider.plan.settings)
        self.acquisition_setting_keys = tuple(s.key for s in self.settings)
        self.groups = tuple(tuple(PauliWord(provider.bank.n, code)
                                  for code in s.assigned_word_codes) for s in self.settings)
        session = provider.session
        self.bank, self.n, self.indices = session.bank, session.n, session.indices
        self.matrices = session.matrices
        self.diagonal_functional = session.diagonal_functional
        self.cache = session.new_cache()
        self.observations = []
        self.requests = []
        self.sampling_seconds = 0.0
        self.measured_solve_attempts = self.bootstrap_solve_attempts = 0

    def _draw(self, reference, counts, cache):
        if reference is not self.provider.reference or cache is not self.cache:
            raise ValueError("acquisition must use the bound reference and cache")
        self.provider._check_identity()
        counts = shot_vector(self.settings, counts)
        if not sum(counts):
            raise ValueError("empty acquisition round")
        # Separate stream per look, reproducible from the run seed, independent
        # of the bootstrap stream. Domain tag avoids reusing a terminal seed.
        seed = int(np.random.SeedSequence([self.seed, 0x53414D50, len(self.requests)])
                   .generate_state(1, np.uint64)[0])
        if any(r["seed"] == seed for r in self.requests):
            raise ValueError("derived sampling seeds collided")
        request = {"look": len(self.requests), "seed": seed, "counts": counts,
                   "requested_shots": sum(counts), "status": "unconfirmed"}
        self.requests.append(request)
        start = perf_counter()
        try:
            observation = reference.sample(self.settings, counts, seed=seed)
            self.provider._check_identity()
            if (not isinstance(observation, SampledObservations)
                    or observation.reference_identity != self.provider.reference_identity
                    or observation.settings_identity != self.provider.circuit_identity
                    or nonnegative_integer(observation.seed, "observation seed") != seed
                    or not isinstance(observation.counts, tuple)
                    or shot_vector(self.settings, observation.counts) != counts):
                raise ValueError("observations do not match the requested reference, circuits, seed and counts")
            validate_batch(observation.batch, self.provider.reference_identity,
                           self.n, self.settings, counts)
        except Exception as exc:
            # The device may have consumed shots before returning an error or
            # invalid batch. Do not assert zero expenditure or cache bad data.
            request["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise ValueError(f"unconfirmed sampling request: {exc}") from exc
        finally:
            self.sampling_seconds += perf_counter()-start
        # Stable histogram insertion order makes resampling reproducible from
        # strict JSON exports regardless of the callback's mapping order.
        observation = replace(observation, batch=replace(observation.batch, groups=tuple(
            replace(g, hist=dict(sorted(g.hist.items()))) for g in observation.batch.groups)))
        self.observations.append(observation)
        request.update(status="confirmed", observation_identity=observation.identity)
        cache.add_batch(observation.batch)
        return cache

    def measure(self, reference, shots_per_group):
        return self._draw(reference, shots_per_group, self.cache)

    def measure_plan(self, reference, plan, cache):
        counts = []
        covered = set()
        for group in self.groups:
            codes = [w.code for w in group]
            covered.update(codes)
            values = {nonnegative_integer(plan.get(code, 0), "shots") for code in codes}
            if len(values) != 1:
                raise ValueError("all words in a compiled setting need the same shots")
            counts.append(values.pop())
        if set(plan)-covered:
            raise ValueError("allocation names words outside the compiled plan")
        return self._draw(reference, counts, cache)

    def solve(self, cache, **kwargs):
        if cache is self.cache:
            self.measured_solve_attempts += 1
        else:
            self.bootstrap_solve_attempts += 1
        point = self.provider.session.solve(cache, **kwargs)
        # The coefficient session has a legacy QWC partition, not the circuits
        # used here. Never describe compiled FC measurements as QWC groups.
        point.resources.pop("qwc_groups", None)
        point.resources.update(provider="sampled_reference",
                               provider_cache_identity=self.provider.cache_identity,
                               reference_identity=self.provider.reference_identity,
                               hamiltonian_identity=self.provider.hamiltonian_identity,
                               compiled_settings=len(self.settings), spectral_certificate=False)
        return point


def run_sampled_precision(provider, config, *, seed, solve_options=None, delta=0.05):
    """Acquire a raw Ritz estimate with the primary rank-aware controller.

    Before the first request, invalid configuration raises. Once a request is
    issued, failures return a receipt with confirmed observations, completed
    looks and any unconfirmed request; physical spending on the latter is
    unknown. The provider and its compiled plan are reused unchanged.
    """
    if not isinstance(config, PrecisionAcquisitionConfig):
        raise TypeError("config must be a PrecisionAcquisitionConfig")
    seed = nonnegative_integer(seed, "seed")
    root = nonnegative_integer(config.root, "root")
    if root >= len(provider.bank):
        raise ValueError("root must be within the declared bank")
    if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, Real) or not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    options = dict(solve_options or {})
    if set(options)-{"tau_s", "rel_tau", "max_condition", "norm_floor"}:
        raise ValueError("precision acquisition requires fixed numerical overlap cutoffs")
    for name, value in options.items():
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or (name == "max_condition" and (math.isnan(value) or value <= 0))
                or (name != "max_condition" and (not math.isfinite(value) or value < 0))):
            raise ValueError(f"invalid fixed cutoff: {name}")
    provider._check_identity()
    if not provider.plan.settings:
        raise ValueError("precision acquisition requires at least one compiled setting")
    shared = _CompiledSession(provider, seed)
    start = perf_counter()
    acquisition, error, looks = None, None, ()
    kwargs = config.as_dict()
    kwargs["rank_stop"] = config.rank_stop
    try:
        acquisition = acquire_ritz_precision(shared, provider.reference, **kwargs,
                                             solver_kwargs=options, delta=delta)
        looks = acquisition.looks
    except (ValueError, np.linalg.LinAlgError) as exc:
        if not shared.requests:
            raise
        if isinstance(exc, AcquisitionError):
            looks = exc.looks
        error = {"type": type(exc).__name__, "message": str(exc)}
    return SampledPrecisionRun(
        acquisition, looks, tuple(shared.observations), tuple(shared.requests), error,
        shared.sampling_seconds, max(0.0, perf_counter()-start-shared.sampling_seconds),
        shared.measured_solve_attempts, shared.bootstrap_solve_attempts)


__all__ = ["PrecisionAcquisitionConfig", "SampledPrecisionRun", "run_sampled_precision"]
