"""Global commutator bank: every selection observable in one sparse basis.

For an ADAPT pool {P_j} and Hamiltonian H, the selection score of candidate
j is g_j = Tr[rho * G_j] with G_j = -i/2 [H, P_j]. The bank stores all G_j
as one sparse candidate-by-word coefficient map over the shared word set
W = union_j supp(G_j), so estimates and variances of every candidate are
linear reconstructions from a single ``WordCache`` — no per-candidate
accumulators owning duplicate measurements.
"""

from __future__ import annotations

from typing import Sequence

from ..multivector import MV
from ..pauli import comm
from ..states import expectation
from ..ir import PauliSum, PauliWord


class CommutatorBank:
    """Sparse coefficient matrix C = (c_jw) with G_j = sum_w c_jw w.

    Coefficients are real: H has real coefficients and P_j is a Pauli word,
    so G_j = -i/2 [H, P_j] is Hermitian with real word coefficients.
    """

    def __init__(self, hamiltonian: PauliSum, pool_words: Sequence[PauliWord],
                 labels: Sequence[str] | None = None, tol: float = 1e-12):
        if not hamiltonian.is_hermitian():
            raise ValueError("Hamiltonian must have real coefficients")
        self.n = hamiltonian.n
        self.labels = tuple(labels) if labels is not None else tuple(
            w.label for w in pool_words)
        if len(self.labels) != len(pool_words):
            raise ValueError("labels and pool_words length mismatch")
        H = hamiltonian.to_mv()
        self.coeffs: list[dict[int, float]] = []
        for P in pool_words:
            if P.n != self.n:
                raise ValueError("pool word and Hamiltonian qubit counts differ")
            G = -0.5j * comm(H, P.to_mv())
            row: dict[int, float] = {}
            for code, c in G.terms.items():
                if abs(c) <= tol:
                    continue
                if abs(c.imag) > 1e-9 * max(1.0, abs(c.real)):
                    raise ValueError(f"selection observable has non-real coefficient {c}")
                row[code] = c.real
            self.coeffs.append(row)
        self.words: tuple[PauliWord, ...] = tuple(
            PauliWord(self.n, code)
            for code in sorted(set().union(*self.coeffs) if self.coeffs else ()))

    def __len__(self) -> int:
        return len(self.coeffs)

    def words_for(self, candidates: Sequence[int]) -> list[PauliWord]:
        """Shared word set of a candidate subset (measurement plan input)."""
        codes = sorted(set().union(*(self.coeffs[j].keys() for j in candidates))
                       if candidates else set())
        return [PauliWord(self.n, c) for c in codes]

    def estimate(self, j: int, cache) -> tuple[float, float]:
        """(g_hat_j, Var_hat(g_hat_j)) reconstructed from the shared cache:
        g_hat = sum_w c_jw mu_hat_w, var = sum_w c_jw^2 Var(mu_hat_w)."""
        est = 0.0
        var = 0.0
        for code, c in self.coeffs[j].items():
            mean, v = cache.mean_var(code)
            est += c * mean
            var += c * c * v
        return est, var

    def exact_score(self, j: int, rho: MV) -> float:
        """Exact g_j = Tr[rho G_j] (validation and exact-selection mode)."""
        return sum(c * expectation(rho, MV(self.n, {code: 1.0})).real
                   for code, c in self.coeffs[j].items())
