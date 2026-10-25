"""Coefficient-only projected provider and two-stage sampled consumer."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from ..measurement.cache import GroupedWordCache
from ..measurement.functionals import quadratic_functional, ritz_functional, WordFunctional
from ..measurement.planning import compile_block_measurement_plan
from ..measurement.ratio import FrozenRatioPlan, hoeffding_radius, pilot_setting_plan
from ..measurement.session import SharedMeasurement
from ..reference_states import (SampledReference, immutable_array, operator_identity,
                                UnsupportedReferenceOperation, _digest)
from ..sampled_reference import SampledObservations, settings_identity, validate_batch
from .projection import MatrixElementBank, ProjectedProblem


@dataclass(frozen=True)
class MeasuredProjection:
    problem: ProjectedProblem
    cache: GroupedWordCache
    entry_radii: tuple[float, ...]
    entry_variances: tuple[float, ...]
    observation_identity: str
    evidence: str = "simultaneous_entrywise_hoeffding_iid_bounded_shots"


class SampledProjectedProvider:
    """Compile S/H rows, reconstruct observations, and freeze a ratio plan.

    ``bank`` is coefficient-only: exact solves/expectations fail explicitly.
    The numerical consumer uses the same stable small-matrix solver as the
    exact provider. Rank, retained frame and c are learned from the pilot.
    """
    capabilities = frozenset({"sampled_projected_blocks", "frozen_ratio"})

    def __init__(self, reference, hamiltonian, generators, *, storage="packed",
                 block_size=None, method="graph_state"):
        if (not isinstance(reference, SampledReference)
                or "sample_joint" not in reference.metadata.capabilities):
            raise UnsupportedReferenceOperation("reference lacks joint sampling capability")
        self.reference = reference
        self.reference_identity = reference.metadata.identity
        self.bank = MatrixElementBank(reference, hamiltonian, generators,
                                      storage=storage, coefficient_only=True)
        self.hamiltonian_identity = operator_identity(self.bank.hamiltonian)
        self.session = SharedMeasurement(self.bank, coefficient_storage=storage)
        self.plan = compile_block_measurement_plan(
            self.bank.n, [w.code for w in self.session.words],
            self.bank.n if block_size is None else block_size, method=method)
        self.circuit_identity = settings_identity(self.plan.settings)
        self.cache_identity = _digest([
            "sampled_projection.v1", self.reference_identity, self.hamiltonian_identity,
            [operator_identity(self.bank.generator(i).mv) for i in range(len(self.bank))],
            storage, self.circuit_identity])

    def compile_measurement(self, directions=None):
        if directions is not None and self.bank.resolve(directions) != self.session.indices:
            raise ValueError("construct a provider for the requested directions")
        self._check_identity()
        return self.plan

    def _check_identity(self):
        if (self.reference.metadata.identity != self.reference_identity
                or settings_identity(self.plan.settings) != self.circuit_identity):
            raise ValueError("provider inputs changed; construct a new provider")

    def reconstruct(self, observations, *, delta=0.05):
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0,1)")
        self._check_identity()
        if (not isinstance(observations, SampledObservations)
                or observations.reference_identity != self.reference_identity
                or observations.settings_identity != self.circuit_identity):
            raise ValueError("observations do not belong to this provider")
        validate_batch(observations.batch, self.reference_identity, self.bank.n,
                       self.plan.settings, observations.counts)
        if any(s < 1 for s in observations.counts):
            raise ValueError("complete-bank reconstruction needs every setting")
        cache = self.session.new_cache()
        if observations.batch.groups:
            cache.add_batch(observations.batch)
        s, h = self.session.matrices(cache)
        functionals = self.session.matrix_functionals()
        radii = tuple(hoeffding_radius(f, cache, delta/max(1, len(functionals)))
                      for f in functionals)
        resources = {**self.bank.resources(), "provider": "sampled_reference",
                     "provider_cache_identity": self.cache_identity,
                     "reference_identity": self.reference_identity,
                     "hamiltonian_identity": self.hamiltonian_identity,
                     "shots": cache.total_shots, "circuits": cache.total_circuits,
                     "confidence_delta": delta, "spectral_certificate": False}
        return MeasuredProjection(
            ProjectedProblem(immutable_array(s), immutable_array(h), self.bank.labels, resources),
            cache, radii, tuple(f.variance(cache) for f in functionals), observations.identity)

    def freeze_ratio(self, pilot, evaluation_shots, *, tau_s=0.05, max_condition=1e4,
                     delta=0.05, denominator_floor=0.05, minimum=2, allocation="ratio"):
        if allocation not in ("ratio", "bank"):
            raise ValueError("allocation must be ratio or bank")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0,1)")
        if not math.isfinite(denominator_floor) or denominator_floor <= 0:
            raise ValueError("denominator_floor must be finite and positive")
        reconstruction = self.reconstruct(pilot, delta=delta)
        solved = reconstruction.problem.solve(tau_s=tau_s, max_condition=max_condition)
        coefficients = immutable_array(solved.ritz_vector())
        frame = immutable_array(solved.coefficients)
        order = self.session.indices

        def freeze(kind):
            f = quadratic_functional(self.bank, order, coefficients, kind)
            return WordFunctional(MappingProxyType(dict(f.coefficients)), f.constant)

        numerator, denominator = freeze("element"), freeze("overlap")
        needed = set(numerator.coefficients) | set(denominator.coefficients)
        settings = tuple(s for s in self.plan.settings
                         if allocation == "bank" or needed.intersection(s.assigned_word_codes))
        targets = ([ritz_functional(self.bank, order, coefficients, solved.ground_energy)]
                   if allocation == "ratio" else self.session.matrix_functionals())
        counts = pilot_setting_plan(settings, reconstruction.cache, targets,
                                    evaluation_shots, minimum=minimum)
        return FrozenRatioPlan(
            self.bank.n, self.reference_identity, self.hamiltonian_identity, pilot.identity,
            pilot.seed, pilot.batch.hardware_shots, solved.ground_energy, coefficients,
            frame, numerator, denominator, settings, counts, float(delta),
            float(denominator_floor), settings_identity(settings))


__all__ = ["SampledProjectedProvider", "MeasuredProjection"]
