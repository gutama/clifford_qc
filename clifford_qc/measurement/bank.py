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

from ..multivector import MV, _lane_mask, word_mul
from ..states import expectation
from ..ir import PauliSum, PauliWord


def _pauli_anticommute(n: int, a: int, b: int) -> int:
    """1 if the encoded Pauli words anticommute, 0 if they commute.

    Parity of the symplectic inner product ``x_a . z_b + z_a . x_b`` over the
    packed x/z bit planes (same lane-mask convention as ``word_mul``): each
    dot product a popcount of an AND, so the test is constant-time in the
    register width rather than a per-qubit scan.
    """
    lo = _lane_mask(n)
    za = (a >> 1) & lo
    xa = (a & lo) ^ za
    zb = (b >> 1) & lo
    xb = (b & lo) ^ zb
    return ((xa & zb).bit_count() + (za & xb).bit_count()) & 1


class CommutatorBank:
    """Sparse coefficient matrix C = (c_jw) with G_j = sum_w c_jw w.

    Coefficients are real: H has real coefficients and P_j is a Pauli word,
    so G_j = -i/2 [H, P_j] is Hermitian with real word coefficients.

    Each row is built directly from the anticommuting terms of H. Since
    ``[W_k, P] = 0`` when the Hamiltonian word W_k commutes with P and
    ``2 W_k P`` when it anticommutes, ``G_j = -i sum_{k: {W_k,P}=0} h_k W_k P``
    — one Pauli-word product per surviving term, rather than forming the two
    general operator products ``H P`` and ``P H`` and cancelling them.
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
        # H has real coefficients (checked above); keep them as floats.
        h_terms = [(code, coeff.real) for code, coeff in hamiltonian.terms.items()]
        self.coeffs: list[dict[int, float]] = []
        for P in pool_words:
            if P.n != self.n:
                raise ValueError("pool word and Hamiltonian qubit counts differ")
            pc = P.code
            # accumulate G_j = -i sum_{anticommuting k} h_k (W_k . P)
            acc: dict[int, complex] = {}
            for wk, h in h_terms:
                if not _pauli_anticommute(self.n, wk, pc):
                    continue
                phase, out = word_mul(self.n, wk, pc)
                acc[out] = acc.get(out, 0j) + (-1j) * h * phase
            row: dict[int, float] = {}
            for code, c in acc.items():
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
