"""Cumulative measurement caches.

One cache holds every measurement taken while the state is fixed (one ADAPT
selection step). All candidates reconstruct their estimates from the same
cache, so a Pauli word shared by many commutator observables is paid for
once. ``WordCache`` accumulates independent per-word counts (ungrouped
path). ``GroupedWordCache`` accumulates the joint outcome histogram of each
qubit-wise-commuting group, from which a candidate's variance is computed
covariance-aware -- carrying the within-group correlations that the
per-word diagonal ignores.  Grouped outcomes may come from either a QWC basis
or an explicitly compiled Clifford readout map.
"""

from __future__ import annotations

from .confidence import jeffreys_mean_var

_UNBOUND = object()


def _bind_state(cache, batch) -> None:
    key = getattr(batch, "state_key", None)
    if cache._state_key is _UNBOUND:
        cache._state_key = key
    elif cache._state_key != key:
        raise ValueError("cannot combine measurement batches from different states")


class WordCache:
    """Accumulates (shots, +1 counts) per Pauli word code across rounds."""

    def __init__(self, n: int):
        self.n = n
        self._shots: dict[int, int] = {}
        self._plus: dict[int, int] = {}
        self.total_shots = 0
        self.total_circuits = 0
        self.rounds = 0
        self._state_key = _UNBOUND

    @property
    def state_key(self):
        return None if self._state_key is _UNBOUND else self._state_key

    def add_batch(self, batch) -> None:
        if batch.n != self.n:
            raise ValueError("batch and cache act on different qubit counts")
        if batch.groups:
            raise ValueError("WordCache requires an ungrouped batch")
        _bind_state(self, batch)
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
    """Cumulative joint-histogram cache for commuting grouped measurement.

    The grouping is fixed for the selection step (the same measurement
    circuits recur every round), so each group's samples are i.i.d. and its
    cumulative histogram is the sufficient statistic. From it, a candidate's
    per-shot combined value within a group -- ``v_{j,g} = sum_{w in g} c_jw
    o_w`` -- has an exact sample mean and variance, so the candidate
    estimate and its variance carry within-group covariance without any
    diagonal approximation.

    ``pooling`` selects which shots a word's mean is read from.

    - ``'assigned'`` (default): the one group the QWC partition assigned the
      word to. This is what every committed record was produced under.
    - ``'shots'``: *every* group whose measurement basis can read the word,
      combined with weights proportional to that group's shot count.

    Pooling is a post-processing change, not a measurement change: the extra
    outcomes are already in the histograms.  A group's basis fixes a letter on
    each qubit of its support, and any word supported inside that set with
    matching letters is a function of the *same* bitstrings, so its outcome is
    recorded in that group's histogram whether or not the partition assigned it
    there.  Each such group therefore supplies an unbiased estimate of the same
    word mean, and -- because the marginal law of a word's +/-1 outcome does not
    depend on which compatible basis was used to read it -- every reading group
    has the same per-shot variance ``1 - mu_w^2``.  Shot-count weights are then
    exactly the inverse-variance (Neyman) weights, so pooling is the
    minimum-variance combination of the readings and reduces a word's variance
    by its reader count.  The weights depend only on the predeclared shot
    schedule, never on outcomes, so a fixed-endpoint finite-sample bound stays
    valid under pooling.
    """

    def __init__(self, n: int, *, pooling: str = "assigned"):
        if pooling not in ("assigned", "shots"):
            raise ValueError("pooling must be 'assigned' or 'shots'")
        self.n = n
        self.pooling = pooling
        # QWC basis key or explicit compiled-setting key -> group dict.
        self._groups: dict[tuple, dict] = {}
        self._word_group: dict[int, tuple] = {}
        self.total_shots = 0
        self.total_circuits = 0
        self.rounds = 0
        self._state_key = _UNBOUND
        # Readers depend only on the set of group bases; per-group word sums
        # depend on the histograms.  Both are rebuilt when those change.
        self._readers: dict[int, tuple] = {}
        self._reader_arrays = None
        self._word_sums: dict[tuple, tuple[int, int]] = {}

    @property
    def state_key(self):
        return None if self._state_key is _UNBOUND else self._state_key

    def add_batch(self, batch) -> None:
        if batch.n != self.n:
            raise ValueError("batch and cache act on different qubit counts")
        if not batch.groups:
            raise ValueError("GroupedWordCache requires a grouped batch")
        _bind_state(self, batch)
        # New shots change every per-group word sum, and a new basis changes
        # which groups read which word.
        self._word_sums.clear()
        keys = [gs.setting_key if gs.setting_key is not None else gs.basis
                for gs in batch.groups]
        compiled_keys = [
            gs.setting_key for gs in batch.groups if gs.setting_key is not None
        ]
        if len(compiled_keys) != len(set(compiled_keys)):
            raise ValueError("a grouped batch contains duplicate compiled settings")
        if any(key not in self._groups for key in keys):
            self._readers.clear()
            self._reader_arrays = None
        for gs, key in zip(batch.groups, keys):
            readouts = dict(gs.readouts)
            g = self._groups.setdefault(
                key, {"support": gs.support, "basis": dict(gs.basis),
                      "basis_tuple": gs.basis, "setting_key": gs.setting_key,
                      "readouts": readouts, "hist": {}, "N": 0,
                      "word_codes": set()})
            if g["support"] != gs.support:
                raise ValueError("same measurement setting has inconsistent support")
            if (g["basis_tuple"] != gs.basis
                    or g["setting_key"] != gs.setting_key
                    or g["readouts"] != readouts):
                raise ValueError("same measurement setting has inconsistent readout metadata")
            for code in gs.word_codes:
                previous = self._word_group.get(code)
                if previous is not None and previous != key:
                    raise ValueError(f"word code {code} is assigned to multiple groups")
                self._word_group[code] = key
                g["word_codes"].add(code)
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
        assigned = self._word_group.get(code)
        if assigned is not None:
            return self._groups.get(assigned)
        # Compatibility fallback for legacy hand-built GroupSample objects
        # that predate explicit word assignments.
        supp = self._word_support(self.n, code)
        for g in self._groups.values():
            if self._group_reads(g, code, supp):
                return g
        return None

    def group_key(self, code: int) -> tuple | None:
        """Basis key of the group this cache reads a word from, or ``None``.

        Several groups can be *able* to read the same word -- any group whose
        basis agrees with the word's letters on its support does -- so a
        consumer that buckets coefficients group by group must use this single
        assignment rather than "every group that matches". Counting a word once
        per capable group inflates a variance by the number of such groups.
        """
        assigned = self._word_group.get(code)
        if assigned is not None:
            return assigned
        supp = self._word_support(self.n, code)
        for key, g in self._groups.items():
            if self._group_reads(g, code, supp):
                return key
        return None

    def reader_keys(self, code: int) -> tuple:
        """Basis keys of *every* group whose shots record this word's outcome.

        A group reads a word when the word's support lies inside the group's
        support and their letters agree there.  Written on packed codes: with
        ``nz(c)`` the non-identity lane mask of ``c`` and ``b`` the group's
        merged basis code, that is ``nz(c) & ~keep == 0`` (support inside) and
        ``nz(c) & nz(c ^ b) == 0`` (letters agree), the same two machine-word
        tests the QWC partition itself uses.
        """
        cached = self._readers.get(code)
        if cached is not None:
            return cached
        keys = tuple(self._scan_readers(code))
        self._readers[code] = keys
        return keys

    def _scan_readers(self, code: int):
        supp = self._word_support(self.n, code)
        if (any(g["setting_key"] is not None for g in self._groups.values())
                or 2 * self.n > 63 or len(self._groups) < 32):
            return [key for key, g in self._groups.items()
                    if self._group_reads(g, code, supp)]
        import numpy as np

        from ..pauli_kernel import pauli_lane_mask

        if self._reader_arrays is None:
            lo = np.int64(pauli_lane_mask(self.n))
            keys = tuple(self._groups)
            merged = np.fromiter(
                (self._basis_code(g["basis"]) for g in self._groups.values()),
                dtype=np.int64, count=len(keys))
            keep = (merged & lo) | ((merged >> np.int64(1)) & lo)
            self._reader_arrays = (keys, merged, keep, lo)
        keys, merged, keep, lo = self._reader_arrays
        packed = np.int64(code)
        nz = (packed & lo) | ((packed >> np.int64(1)) & lo)
        diff = merged ^ packed
        nz_diff = (diff & lo) | ((diff >> np.int64(1)) & lo)
        readable = ((nz & ~keep) == 0) & ((nz & nz_diff) == 0)
        return [keys[i] for i in np.flatnonzero(readable)]

    @staticmethod
    def _basis_code(basis: dict) -> int:
        return sum("IXYZ".index(letter) << (2 * qubit)
                   for qubit, letter in basis.items())

    def _group_reads(self, group: dict, code: int, support: tuple | None = None) -> bool:
        if group["setting_key"] is not None:
            return code in group["readouts"]
        supp = self._word_support(self.n, code) if support is None else support
        basis = group["basis"]
        return all(j in basis and basis[j] == self._letter(code, j) for j in supp)

    def _readout(self, group: dict, code: int) -> tuple[int, tuple[int, ...]]:
        explicit = group["readouts"].get(code)
        if explicit is not None:
            return explicit
        if group["setting_key"] is not None:
            raise KeyError(
                f"compiled setting {group['setting_key']!r} has no readout "
                f"for word code {code}"
            )
        positions = tuple(
            group["support"].index(q) for q in self._word_support(self.n, code)
        )
        return 1, positions

    def group_weights(self, code: int) -> dict[tuple, float]:
        """How this cache's estimate of a word mean is split over groups.

        Under ``pooling='assigned'`` this is the single assigned group at weight
        one -- the historical behavior.  Under ``pooling='shots'`` it is every
        reading group at weight ``N_g / sum_g' N_g'``.  A consumer that buckets
        a functional's coefficients group by group must scale each coefficient
        by these weights: they sum to one over the reading groups, so the
        combination stays unbiased, and no group is counted twice.
        """
        if self.pooling == "assigned":
            key = self.group_key(code)
            if key is None or self._groups[key]["N"] <= 0:
                return {}
            return {key: 1.0}
        keys = [key for key in self.reader_keys(code) if self._groups[key]["N"] > 0]
        total = float(sum(self._groups[key]["N"] for key in keys))
        if total <= 0.0:
            return {}
        return {key: self._groups[key]["N"] / total for key in keys}

    def _group_word_sums(self, key: tuple, code: int) -> tuple[int, int]:
        """``(+1 count, shots)`` for one word inside one group's histogram."""
        cached = self._word_sums.get((key, code))
        if cached is not None:
            return cached
        g = self._groups[key]
        sign, pos = self._readout(g, code)
        plus = 0
        for bits, count in g["hist"].items():
            parity_sign = -1 if sum(bits[p] == "1" for p in pos) % 2 else 1
            if sign * parity_sign == 1:
                plus += count
        out = (plus, g["N"])
        self._word_sums[(key, code)] = out
        return out

    def shots(self, code: int) -> int:
        """Shots this cache reads the word from -- pooled over readers when
        ``pooling='shots'``, so it is the resource the word's variance reflects."""
        if self.pooling == "shots":
            return sum(self._groups[key]["N"] for key in self.reader_keys(code))
        g = self._group_of(code)
        return g["N"] if g else 0

    def reader_count(self, code: int) -> int:
        """How many measurement settings record this word (pooling diagnostic)."""
        return len(self.reader_keys(code))

    def num_groups(self) -> int:
        """Distinct QWC measurement bases (measurement circuits) accumulated."""
        return len(self._groups)

    def group_states(self) -> tuple[dict, ...]:
        """Read-only view of the sufficient statistic: one dict per group with
        ``key``, ``basis``, ``support``, ``hist`` (copied), ``shots``, and any
        explicit compiled-setting ``readouts``.

        This *is* the cache's information content -- everything downstream
        (estimates, covariance-aware variances, bootstrap resampling) is a
        function of these histograms, so exposing them avoids each consumer
        reaching into the private store.
        """
        return tuple({"key": key, "basis": g["basis_tuple"],
                      "support": g["support"],
                      "setting_key": g["setting_key"],
                      "hist": dict(g["hist"]), "shots": g["N"],
                      "readouts": dict(g["readouts"]),
                      "word_codes": tuple(sorted(g["word_codes"]))}
                     for key, g in self._groups.items())

    # -- per-word marginal (diagonal; used for the threshold gate) ----------

    def mean_var(self, code: int) -> tuple[float, float]:
        """Jeffreys ``(mean, variance)`` of one word mean, over the shots this
        cache reads it from -- one group, or every reading group when pooling."""
        keys = self.group_weights(code)
        if not keys:
            return 0.0, float("inf")
        plus = shots = 0
        for key in keys:
            group_plus, group_shots = self._group_word_sums(key, code)
            plus += group_plus
            shots += group_shots
        if shots == 0:
            return 0.0, float("inf")
        from .confidence import jeffreys_mean_var
        return jeffreys_mean_var(plus, shots)

    # -- covariance-aware candidate statistics ------------------------------

    def candidate_estimate(self, coeffs: dict) -> float:
        """g_hat_j = sum_w c_jw <W_w>, reconstructed from group histograms."""
        est = 0.0
        for code, c in coeffs.items():
            for key, weight in self.group_weights(code).items():
                plus, shots = self._group_word_sums(key, code)
                est += c * weight * (2.0 * plus - shots) / shots
        return est

    def candidate_group_statistics(self, coeffs: dict):
        """Per-group statistics of a linear functional of word outcomes.

        Returns ``{basis_key: (N_g, sample_var_g, range_g)}``, or ``None`` if
        any touched word is unmeasured.  Keeping the basis key is what a
        physical-shot allocator needs: every member of one QWC group is read by
        the same circuit shot, so allocating independently to the words would
        count a resource the hardware cannot spend independently.
        """
        # Bucket the candidate's coefficients by the groups that read each word,
        # scaled by this cache's pooling weights.  Every group then still holds
        # one per-shot combined value, so the covariance stays exact.
        buckets: dict[tuple, tuple] = {}
        for code, c in coeffs.items():
            weights = self.group_weights(code)
            if not weights:
                return None
            for key, weight in weights.items():
                row = buckets.setdefault(key, (self._groups[key], {}))[1]
                row[code] = row.get(code, 0.0) + c * weight
        terms = {}
        for key, (g, rowc) in buckets.items():
            support = g["support"]
            N = g["N"]
            # per-outcome combined value v and its histogram moments
            code_readouts = {code: self._readout(g, code) for code in rowc}
            s1 = s2 = 0.0
            for bits, cnt in g["hist"].items():
                v = 0.0
                for code, c in rowc.items():
                    sign, positions = code_readouts[code]
                    parity_sign = (-1.0 if sum(bits[p] == "1" for p in positions) % 2
                                   else 1.0)
                    o = sign * parity_sign
                    v += c * o
                s1 += v * cnt
                s2 += v * v * cnt
            mean = s1 / N
            raw_var = max(0.0, s2 / N - mean * mean)
            # a-priori range of v = sum_w c_w o_w, o_w in {-1,+1}: width 2*sum|c|.
            # The theoretical range (not the observed one) is required for the
            # empirical-Bernstein bound to remain a valid finite-sample bound.
            rng = 2.0 * sum(abs(c) for c in rowc.values())
            # A zero plug-in variance after unanimous outcomes makes a normal
            # interval collapse at finite N.  This Jeffreys-scale floor retains
            # the covariance estimate while refusing false certainty.
            var = max(raw_var, rng * rng / (4.0 * (N + 1.0)))
            terms[key] = (N, var, rng)
        return terms

    def candidate_group_terms(self, coeffs: dict):
        """``(N_g, sample_var_g, range_g)`` per touched measurement group.

        Compatibility view for confidence routines; allocators should use
        :meth:`candidate_group_statistics` so the group identity is retained.
        """
        statistics = self.candidate_group_statistics(coeffs)
        return None if statistics is None else list(statistics.values())
