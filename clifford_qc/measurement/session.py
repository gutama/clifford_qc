"""Shared grouped-measurement sessions for projected subspaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..ir import PauliWord
from ..selection import EvidenceLevel
from .cache import GroupedWordCache
from .functionals import WordFunctional, entry_functionals
from .grouping import qwc_groups

if TYPE_CHECKING:
    from ..subspace.linalg import SubspaceResult
    from ..subspace.projection import MatrixElementBank

# Kept equal to the canonical linalg defaults without importing the higher
# layer while this lower-level measurement module is initialized.
DEFAULT_TAU_S = 1e-10
DEFAULT_MAX_CONDITION = 1e12
DEFAULT_NORM_FLOOR = 1e-14

HEURISTIC = EvidenceLevel.HEURISTIC.value

class SharedMeasurement:
    """One QWC-grouped measurement of a bank subspace's whole word universe (4A).

    Every entry of ``(S, H)`` is reconstructed from the same shots, which is the
    measurement-sharing proposition made concrete: a word appearing in many
    element operators is paid for once. The grouping is fixed at construction so
    the same circuits recur every batch, which is what makes each group's
    cumulative histogram a sufficient statistic -- and what the
    finite-sample bounds require.
    """

    def __init__(self, bank: MatrixElementBank, indices: Sequence[int] | None = None,
                 *, groups: Sequence[Sequence[PauliWord]] | None = None):
        self.bank = bank
        self.indices = bank.resolve(indices)
        bank.matrices(self.indices)  # force every pair, so the universe is complete
        self._pairs = {(i, j): entry_functionals(bank, i, j)
                       for a, i in enumerate(self.indices)
                       for j in self.indices[a:]}
        codes = sorted({code
                        for pair in self._pairs.values()
                        for part in pair for f in part
                        for code in f.coefficients})
        self.words = tuple(PauliWord(bank.n, code) for code in codes)
        # A caller may supply a wider grouping -- one covering a superset of
        # these words -- so that a sub-block and the full universe are read out
        # of the *same* circuits and the same cache.
        self.groups = qwc_groups(list(self.words)) if groups is None else [
            list(group) for group in groups]
        covered = {w.code for group in self.groups for w in group}
        if not covered.issuperset(codes):
            raise ValueError("supplied groups do not cover this subspace's words")

    def diagonal_functional(self, index: int) -> WordFunctional:
        """``S_ii`` as a functional -- the candidate norm, when it must be measured."""
        return self._pairs[(index, index)][0][0]

    @property
    def n(self) -> int:
        return self.bank.n

    def new_cache(self) -> GroupedWordCache:
        return GroupedWordCache(self.n)

    def measure(self, backend, shots_per_group: int,
                cache: GroupedWordCache | None = None) -> GroupedWordCache:
        """Sample every group ``shots_per_group`` times into (a fresh) cache.

        A predeclared uniform allocation: the schedule's endpoints are fixed
        before any outcome is seen, which is the condition the
        empirical-Bernstein bounds are valid under.
        """
        if shots_per_group <= 0:
            raise ValueError("shots_per_group must be positive")
        if not self.groups:
            raise ValueError("nothing to measure: the universe is empty")
        cache = self.new_cache() if cache is None else cache
        batch = backend.sample_grouped_from_state(self.bank.reference, self.groups,
                                                 shots_per_group)
        cache.add_batch(batch)
        return cache

    def _assemble(self, evaluate) -> tuple[np.ndarray, np.ndarray]:
        m = len(self.indices)
        S = np.zeros((m, m), dtype=complex)
        Hm = np.zeros((m, m), dtype=complex)
        for a, i in enumerate(self.indices):
            for b, j in enumerate(self.indices[a:], start=a):
                (s_re, s_im), (h_re, h_im) = self._pairs[(i, j)]
                S[a, b] = complex(evaluate(s_re), evaluate(s_im))
                Hm[a, b] = complex(evaluate(h_re), evaluate(h_im))
        # Same structural Hermiticity as the exact path: the lower triangle is
        # the conjugate by definition, never an independent estimate that would
        # have to be symmetrized (and would double the noise on the diagonal).
        for b in range(m):
            S[b, b] = S[b, b].real
            Hm[b, b] = Hm[b, b].real
            for a in range(b):
                S[b, a] = S[a, b].conjugate()
                Hm[b, a] = Hm[a, b].conjugate()
        return S, Hm

    def matrices(self, cache: GroupedWordCache) -> tuple[np.ndarray, np.ndarray]:
        """``(S_hat, H_hat)`` from measured word means."""
        return self._assemble(lambda f: f.estimate(cache))

    def exact_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """The infinite-shot limit of the same reconstruction (4A acceptance)."""
        rho = self.bank.reference
        return self._assemble(lambda f: f.exact(rho))

    def solve(self, cache: GroupedWordCache, *, tau_s: float = DEFAULT_TAU_S,
              rel_tau: float = 0.0, max_condition: float = DEFAULT_MAX_CONDITION,
              norm_floor: float = DEFAULT_NORM_FLOOR) -> SubspaceResult:
        """Thresholded GEP on the measured matrices.

        The result reports the same conditioning metrics as the exact path plus
        ``overlap_negative_modes``: with estimated entries ``S_hat`` is no longer
        positive semidefinite, and thresholding those modes away is a repair
        whose effect on the variational bound is not established (Q3).
        """
        from ..subspace.linalg import solve_projected

        S, Hm = self.matrices(cache)
        resources = dict(self.bank.resources(self.indices))
        resources.update({
            "evidence": HEURISTIC,
            "shots": cache.total_shots,
            "circuits": cache.total_circuits,
            "qwc_groups": len(self.groups),
            "measured_words": len(self.words),
            "shots_per_measured_word": (cache.total_shots / len(self.words)
                                        if self.words else 0.0),
        })
        return solve_projected(S, Hm,
                               [self.bank.generator(i).label for i in self.indices],
                               tau_s=tau_s, rel_tau=rel_tau,
                               max_condition=max_condition, norm_floor=norm_floor,
                               resources=resources).with_bank(self.bank, self.indices)


# --------------------------------------------------------------- 4B: uncertainty

__all__ = ["SharedMeasurement"]
