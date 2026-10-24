"""Numerical projected-provider boundary and the first exact reference consumer.

The bank bridge preserves the existing A-CASE selection workflow. Consumers of
this provider need numerical blocks, not coefficient-row or density accessors.
Action and observation consumers remain separate future implementations.
"""
from __future__ import annotations

import hashlib
import json
from typing import Protocol, runtime_checkable

from ..reference_states import require_exact_reference, operator_identity
from .projection import MatrixElementBank, ProjectedProblem
from .streaming import StreamingMatrixElementBank


@runtime_checkable
class ProjectedProblemProvider(Protocol):
    @property
    def reference_identity(self) -> str: ...

    @property
    def hamiltonian_identity(self) -> str: ...

    def projected_problem(self, indices=None) -> ProjectedProblem: ...


@runtime_checkable
class ActionProvider(Protocol):
    """Step 3: block H actions in a declared sector, without measurement rows.

    The consumer retains vectors/actions, owns its memory admission plan, and
    computes action residuals. Neither a density nor H-squared is required.
    """
    @property
    def dimension(self) -> int: ...

    @property
    def sector_identity(self) -> str: ...

    def apply_hamiltonian(self, vectors): ...


@runtime_checkable
class MeasurementProvider(Protocol):
    """Step 5: coefficient compilation and reconstruction from observations.

    Reconstruction must return uncertainty and actual sampling cost alongside
    its numerical projected problem. It cannot invoke an exact reference oracle.
    """
    @property
    def reference_identity(self) -> str: ...

    @property
    def hamiltonian_identity(self) -> str: ...

    def compile_measurement(self, directions): ...

    def reconstruct(self, observations): ...


class ExactProjectedProvider:
    """Exact S/H evaluation on an immutable exact-reference capability.

    Input ownership and reference evaluation are independent of the existing
    coefficient-storage/lifetime policies. No action or sampling capability is
    claimed. ``bank`` is the explicit compatibility bridge to ``run_acase``.
    """
    capabilities = frozenset({"exact_projected_blocks"})

    def __init__(self, reference, hamiltonian, generators=(), *, storage="object",
                 policy="retain_all", frontier_pairs=32, layout="interleaved"):
        reference = require_exact_reference(reference)
        if policy not in ("retain_all", "stream_recompute"):
            raise ValueError("policy must be retain_all or stream_recompute")
        cls = MatrixElementBank if policy == "retain_all" else StreamingMatrixElementBank
        kwargs = {} if policy == "retain_all" else {"frontier_pairs": frontier_pairs}
        self._bank = cls(reference, hamiltonian, generators, storage=storage,
                         layout=layout, **kwargs)
        self._reference_identity = reference.metadata.identity
        self._hamiltonian_identity = operator_identity(self._bank.hamiltonian)
        self._policy, self._storage, self._layout = policy, storage, layout

    @property
    def reference_identity(self): return self._reference_identity

    @property
    def hamiltonian_identity(self): return self._hamiltonian_identity

    @property
    def cache_identity(self):
        inputs = ["exact_projection.v1", self.reference_identity,
                        self.hamiltonian_identity, self._policy, self._storage, self._layout,
                        [operator_identity(self._bank.generator(i).mv) for i in range(len(self._bank))]]
        return hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest()

    @property
    def bank(self): return self._bank

    def matrices(self, indices=None):
        return self._bank.matrices(indices)

    def projected_problem(self, indices=None):
        order = self._bank.resolve(indices)
        overlap, hamiltonian = self.matrices(order)
        return ProjectedProblem(overlap, hamiltonian,
                                tuple(self._bank.generator(i).label for i in order),
                                self.resources(order))

    def solve(self, indices=None, **kwargs):
        result = self._bank.solve(indices, **kwargs)
        result.resources.update(self.execution_metadata())
        return result

    def resources(self, indices=None):
        return {**self._bank.resources(indices), **self.execution_metadata()}

    def execution_metadata(self):
        return {"provider": "exact_reference", "provider_cache_identity": self.cache_identity,
                "reference_identity": self.reference_identity,
                "hamiltonian_identity": self.hamiltonian_identity,
                "reference_evaluation": self._bank.reference_resources(),
                "input_snapshot_coefficient_payload_bytes": 24 * (
                    self._bank.hamiltonian.nnz() + sum(self._bank.generator(i).mv.nnz()
                                                     for i in range(len(self._bank)))),
                "input_snapshot_scope": "24-byte coefficient model for owned H/generator inputs; excludes Python overhead and bank-derived operators"}
