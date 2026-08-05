"""Finite-shot sampling backend.

Simulator-backed for now: exact single-word expectations from the MV state
turned into seeded binomial outcome counts, exactly the sampling model of
the standalone noisy-ADAPT study. ``sample_grouped_from_state`` measures a
whole qubit-wise-commuting group per circuit by sampling bitstrings from
the shared rotated-basis distribution, so word outcomes carry the true
joint correlations. Swappable later for a hardware/PennyLane sampler
behind the same ``SamplingBackend`` protocol.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence, Union

import numpy as np

from ..multivector import MV
from .. import gates as _gates
from ..states import computational_probabilities, evolve, expectation, partial_trace
from ..ir import PauliSum, PauliWord, Program
from .exact_mv import ExactMVBackend
from .protocol import MeasurementBatch, state_fingerprint


class _GroupPlan:
    """Everything about sampling one QWC group that does not depend on the shots.

    Rotating the state onto the group's shared basis, tracing out the qubits the
    group does not touch, and reading off the computational distribution are
    fixed by ``(state, group)`` alone -- and so is *which* outcomes are
    even-parity for each word in the group. Only the multinomial draw depends on
    the shot count.

    Recomputing all of it per batch is what confined certified A-CASE growth to
    four qubits. The certified loop measures the same universe twice per step
    (construction and certification) over the same never-changing reference, so
    at ``M = 8`` on eight qubits the same 1689 group rotations were being redone
    sixteen times at ~60 ms each. Held here instead, a batch is one multinomial
    draw and two array reductions per group.
    """

    __slots__ = ("keep", "basis", "bits", "probs", "parity")

    def __init__(self, keep, basis, bits, probs, parity):
        self.keep = keep
        self.basis = basis
        self.bits = bits
        self.probs = probs
        self.parity = parity


class FiniteShotBackend:
    """Seeded finite-shot sampler over an exact inner backend."""

    # How many distinct states keep their group plans. A-CASE measures one
    # never-changing reference, so one would do; ADAPT moves the state every
    # step, and holding a couple avoids thrashing without unbounded growth.
    _PLAN_STATES = 2

    def __init__(self, seed: int, inner: ExactMVBackend | None = None):
        self.rng = np.random.default_rng(seed)
        self.inner = inner if inner is not None else ExactMVBackend()
        self._plans: OrderedDict[object, dict[tuple[int, ...], _GroupPlan]] = OrderedDict()

    def _group_plans(self, rho: MV) -> dict[tuple[int, ...], _GroupPlan]:
        """The plan table for ``rho``, keyed by the state's own terms.

        Keyed by value, not by ``id``: an equal state rebuilt from the same
        program must hit, and a mutated one must miss.
        """
        key = state_fingerprint(rho)
        table = self._plans.get(key)
        if table is None:
            if len(self._plans) >= self._PLAN_STATES:
                self._plans.popitem(last=False)
            table = self._plans[key] = {}
        else:
            self._plans.move_to_end(key)
        return table

    def _plan(self, rho: MV, group: Sequence[PauliWord],
              table: dict[tuple[int, ...], _GroupPlan]) -> _GroupPlan:
        from ..measurement.grouping import shared_basis

        signature = tuple(sorted(w.code for w in group))
        plan = table.get(signature)
        if plan is not None:
            return plan

        # rotate the shared basis onto Z: X -> H, Y -> H*SDG per qubit
        basis = shared_basis(group)
        rho_rot = rho
        for j, letter in basis.items():
            if letter == "X":
                rho_rot = evolve(rho_rot, _gates.H(rho.n, j))
            elif letter == "Y":
                rho_rot = evolve(rho_rot, _gates.H(rho.n, j) * _gates.S(rho.n, j).dagger())
        keep = tuple(sorted({j for w in group for j in w.support()}))
        traced = rho_rot if len(keep) == rho.n else \
            partial_trace(rho_rot, {j for j in range(rho.n) if j not in keep})
        outcomes = sorted(computational_probabilities(traced).items())
        probs = np.asarray([p for _, p in outcomes], dtype=float)
        if (not np.all(np.isfinite(probs)) or probs.min(initial=0.0) < -1e-10
                or abs(float(probs.sum()) - 1.0) > 1e-9):
            raise ValueError("rotated state produced an invalid probability distribution")
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / probs.sum()
        position = {q: i for i, q in enumerate(keep)}
        bits = tuple("".join(row[position[q]] for q in keep) for row, _ in outcomes)
        parity = {}
        for w in group:
            positions = [position[j] for j in w.support()]
            parity[w.code] = np.fromiter(
                (sum(row[pos] == "1" for pos in positions) % 2 == 0
                 for row, _ in outcomes), dtype=bool, count=len(outcomes))
        plan = _GroupPlan(keep, tuple(sorted(basis.items())), bits, probs, parity)
        table[signature] = plan
        return plan

    def state(self, program: Program, values=None, initial_state: MV | None = None) -> MV:
        return self.inner.state(program, values, initial_state)

    def sample_paulis(self, program: Program, words: Sequence[PauliWord],
                      shots: Union[int, Mapping[int, int]], values=None,
                      initial_state: MV | None = None) -> MeasurementBatch:
        rho = self.state(program, values, initial_state)
        return self.sample_words_from_state(rho, words, shots)

    def sample_words_from_state(self, rho: MV, words: Sequence[PauliWord],
                                shots: Union[int, Mapping[int, int]]) -> MeasurementBatch:
        """Sample without re-preparing the state (one ADAPT step measures many
        rounds from the same state)."""
        shot_map = ({w.code: int(shots) for w in words} if isinstance(shots, int)
                    else {int(k): int(v) for k, v in shots.items()})
        if any(count < 0 for count in shot_map.values()):
            raise ValueError("shots must be non-negative")
        out_shots: dict[int, int] = {}
        plus: dict[int, int] = {}
        circuits = 0
        for w in words:
            N = shot_map.get(w.code, 0)
            if N < 0:
                raise ValueError("shots must be non-negative")
            if N == 0:
                continue
            ev = expectation(rho, w.to_mv()).real
            if not np.isfinite(ev) or ev < -1.0 - 1e-10 or ev > 1.0 + 1e-10:
                raise ValueError("state produced an invalid Pauli expectation")
            p = 0.5 * (1.0 + float(np.clip(ev, -1.0, 1.0)))
            out_shots[w.code] = N
            plus[w.code] = int(self.rng.binomial(N, p))
            circuits += 1
        return MeasurementBatch(n=rho.n, shots=out_shots, plus_counts=plus,
                                circuits=circuits,
                                state_key=state_fingerprint(rho))

    def sample_grouped_from_state(self, rho: MV, groups: Sequence[Sequence[PauliWord]],
                                  shots: Union[int, Mapping[int, int]]) -> MeasurementBatch:
        """Jointly sample each qubit-wise-commuting group with one circuit.

        Every group member is measured on every shot of its group's circuit,
        so a word requesting fewer shots than a groupmate still receives the
        group maximum — extra outcomes are free on hardware too.
        """
        all_words = [w for group in groups for w in group]
        shot_map = ({w.code: int(shots) for w in all_words} if isinstance(shots, int)
                    else {int(k): int(v) for k, v in shots.items()})
        if any(count < 0 for count in shot_map.values()):
            raise ValueError("shots must be non-negative")
        from .protocol import GroupSample

        table = self._group_plans(rho)
        out_shots: dict[int, int] = {}
        plus: dict[int, int] = {}
        group_samples: list = []
        circuits = 0
        for group in groups:
            N = max((shot_map.get(w.code, 0) for w in group), default=0)
            if N < 0:
                raise ValueError("shots must be non-negative")
            if N == 0:
                continue
            circuits += 1
            plan = self._plan(rho, group, table)
            # One draw per group in group order, as before: the RNG stream a
            # committed record was produced under is part of the record.
            counts = self.rng.multinomial(N, plan.probs)
            hist: dict[str, int] = {}
            for key, c in zip(plan.bits, counts):
                if c:
                    hist[key] = hist.get(key, 0) + int(c)
            group_samples.append(GroupSample(
                support=plan.keep, basis=plan.basis, hist=hist, shots=N,
                word_codes=tuple(w.code for w in group)))
            for w in group:
                n_plus = int(counts[plan.parity[w.code]].sum())
                out_shots[w.code] = out_shots.get(w.code, 0) + N
                plus[w.code] = plus.get(w.code, 0) + n_plus
        return MeasurementBatch(n=rho.n, shots=out_shots, plus_counts=plus,
                                circuits=circuits, groups=tuple(group_samples),
                                state_key=state_fingerprint(rho))
