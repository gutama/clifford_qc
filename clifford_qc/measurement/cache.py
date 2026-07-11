"""Cumulative per-word measurement cache.

One cache holds every single-word measurement taken while the state is
fixed (one ADAPT selection step). All candidates reconstruct their
estimates from the same cache, so a Pauli word shared by many commutator
observables is paid for once — hypothesis H2 of the research plan.
"""

from __future__ import annotations

from .confidence import jeffreys_mean_var


class WordCache:
    """Accumulates (shots, +1 counts) per Pauli word code across rounds."""

    def __init__(self, n: int):
        self.n = n
        self._shots: dict[int, int] = {}
        self._plus: dict[int, int] = {}
        self.total_shots = 0
        self.total_circuits = 0
        self.rounds = 0

    def add_batch(self, batch) -> None:
        if batch.n != self.n:
            raise ValueError("batch and cache act on different qubit counts")
        for code, N in batch.shots.items():
            self._shots[code] = self._shots.get(code, 0) + N
            self._plus[code] = self._plus.get(code, 0) + batch.plus_counts[code]
            self.total_shots += N
        self.total_circuits += batch.circuits
        self.rounds += 1

    def shots(self, code: int) -> int:
        return self._shots.get(code, 0)

    def unique_words(self) -> int:
        return len(self._shots)

    def mean(self, code: int) -> float:
        """Plain sample mean of the +/-1 outcomes."""
        N = self._shots.get(code, 0)
        if N == 0:
            raise KeyError(f"word code {code} has no measurements")
        return (2.0 * self._plus[code] - N) / N

    def mean_var(self, code: int) -> tuple[float, float]:
        """Jeffreys-pseudocount (mean, variance) of the word mean.

        Nonzero variance even when all outcomes agree, matching the
        conservative empirical estimator of the standalone study. Words
        never measured return (0, inf) so untouched candidates stay
        maximally uncertain rather than falsely resolved.
        """
        N = self._shots.get(code, 0)
        if N == 0:
            return 0.0, float("inf")
        return jeffreys_mean_var(self._plus[code], N)
