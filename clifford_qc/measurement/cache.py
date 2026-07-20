"""Cumulative measurement caches.

One cache holds every measurement taken while the state is fixed (one ADAPT
selection step). All candidates reconstruct their estimates from the same
cache, so a Pauli word shared by many commutator observables is paid for
once. ``WordCache`` accumulates independent per-word counts (ungrouped
path). ``GroupedWordCache`` accumulates the joint outcome histogram of each
qubit-wise-commuting group, from which a candidate's variance is computed
covariance-aware -- carrying the within-group correlations that the
per-word diagonal ignores.
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


class GroupedWordCache:
    """Cumulative joint-histogram cache for QWC-grouped measurement.

    The grouping is fixed for the selection step (the same measurement
    circuits recur every round), so each group's samples are i.i.d. and its
    cumulative histogram is the sufficient statistic. From it, a candidate's
    per-shot combined value within a group -- ``v_{j,g} = sum_{w in g} c_jw
    o_w`` -- has an exact sample mean and variance, so the candidate
    estimate and its variance carry within-group covariance without any
    diagonal approximation.
    """

    def __init__(self, n: int):
        self.n = n
        # basis key (sorted (qubit,letter) tuple) -> group dict
        self._groups: dict[tuple, dict] = {}
        self.total_shots = 0
        self.total_circuits = 0
        self.rounds = 0

    def add_batch(self, batch) -> None:
        if batch.n != self.n:
            raise ValueError("batch and cache act on different qubit counts")
        if not batch.groups:
            raise ValueError("GroupedWordCache requires a grouped batch")
        for gs in batch.groups:
            g = self._groups.setdefault(
                gs.basis, {"support": gs.support, "basis": dict(gs.basis),
                           "hist": {}, "N": 0})
            for bits, c in gs.hist.items():
                g["hist"][bits] = g["hist"].get(bits, 0) + c
            g["N"] += gs.shots
            self.total_shots += gs.shots
        self.total_circuits += batch.circuits
        self.rounds += 1

    # -- word/group bookkeeping ---------------------------------------------

    @staticmethod
    def _word_support(n: int, code: int) -> tuple:
        return tuple(j for j in range(n) if (code >> (2 * j)) & 3)

    @staticmethod
    def _letter(code: int, j: int) -> str:
        return "IXYZ"[(code >> (2 * j)) & 3]

    def _group_of(self, code: int):
        """The group whose measurement basis reads this word: the group whose
        per-qubit basis letter equals the word's letter at every qubit in the
        word's support. QWC grouping partitions the word set, so it is unique."""
        supp = self._word_support(self.n, code)
        for g in self._groups.values():
            basis = g["basis"]
            if all(j in basis and basis[j] == self._letter(code, j) for j in supp):
                return g
        return None

    def shots(self, code: int) -> int:
        g = self._group_of(code)
        return g["N"] if g else 0

    def num_groups(self) -> int:
        """Distinct QWC measurement bases (measurement circuits) accumulated."""
        return len(self._groups)

    # -- per-word marginal (diagonal; used for the threshold gate) ----------

    def mean_var(self, code: int) -> tuple[float, float]:
        g = self._group_of(code)
        if g is None or g["N"] == 0:
            return 0.0, float("inf")
        N = g["N"]
        pos = [g["support"].index(q) for q in self._word_support(self.n, code)]
        plus = 0
        for bits, c in g["hist"].items():
            if sum(bits[p] == "1" for p in pos) % 2 == 0:
                plus += c
        from .confidence import jeffreys_mean_var
        return jeffreys_mean_var(plus, N)

    # -- covariance-aware candidate statistics ------------------------------

    def candidate_estimate(self, coeffs: dict) -> float:
        """g_hat_j = sum_w c_jw <W_w>, reconstructed from group histograms."""
        est = 0.0
        for code, c in coeffs.items():
            g = self._group_of(code)
            if g is None or g["N"] == 0:
                continue
            pos = [g["support"].index(q) for q in self._word_support(self.n, code)]
            mean = 0.0
            for bits, cnt in g["hist"].items():
                o = -1.0 if sum(bits[p] == "1" for p in pos) % 2 else 1.0
                mean += o * cnt
            est += c * mean / g["N"]
        return est

    def candidate_group_terms(self, coeffs: dict):
        """Yield ``(N_g, sample_var_g, range_g)`` of the candidate's per-shot
        combined value within each group it touches. Feeds ``candidate_radius``.
        Returns ``None`` if any touched word is unmeasured (unresolved)."""
        # bucket the candidate's coefficients by the group that reads each word
        buckets: dict[tuple, tuple] = {}
        for code, c in coeffs.items():
            g = self._group_of(code)
            if g is None or g["N"] == 0:
                return None
            key = tuple(sorted(g["basis"].items()))
            buckets.setdefault(key, (g, {}))[1][code] = c
        terms = []
        for key, (g, rowc) in buckets.items():
            support = g["support"]
            N = g["N"]
            # per-outcome combined value v and its histogram moments
            code_pos = {code: [support.index(q) for q in self._word_support(self.n, code)]
                        for code in rowc}
            s1 = s2 = 0.0
            for bits, cnt in g["hist"].items():
                v = 0.0
                for code, c in rowc.items():
                    o = -1.0 if sum(bits[p] == "1" for p in code_pos[code]) % 2 else 1.0
                    v += c * o
                s1 += v * cnt
                s2 += v * v * cnt
            mean = s1 / N
            var = max(0.0, s2 / N - mean * mean)
            # a-priori range of v = sum_w c_w o_w, o_w in {-1,+1}: width 2*sum|c|.
            # The theoretical range (not the observed one) is required for the
            # empirical-Bernstein bound to remain a valid finite-sample bound.
            rng = 2.0 * sum(abs(c) for c in rowc.values())
            terms.append((N, var, rng))
        return terms
