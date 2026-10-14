"""Q18-S3: certify on the ladder's Krylov basis first, then allocate practically.

Q18-S2 found no prefix of the two committed Hubbard 2x2 A-CASE trajectories
with a resolved variance that certifies the ground state. The ladder's fixed
Krylov sequence ``{H^k|psi>, k <= 8}`` lies outside that screen. This module
supplies what a declaration over its prefixes needs, in the order the
declaration reads it:

1. **Certify.** :class:`KrylovBank` gives the prefixes the bank's own
   ``(S, H)`` rows and :class:`~clifford_qc.subspace.SecondMomentBank` its
   ``K`` rows, through the Hankel identity ``A_i' Q A_j = H^(i+j+q)``. The
   identity is exact operator algebra, and on these integer-and-dyadic
   Hamiltonians the coefficients are exact too, so the rows are the bank
   route's to the bit while costing one product per power instead of one per
   pair. Each prefix's ``sigma^2`` is then established twice more, by
   :class:`DecimalPencil` (the same rows and means at eighty digits, no
   truncation) and by a dense statevector Ritz state. Only a prefix whose
   variance clears ``SecondMomentBank``'s floor, whose double-precision value
   both independent routes reproduce, and whose interval puts ``E + sigma``
   below ``E1``, certifies.
2. **Rank margin.** Krylov overlaps are ill-conditioned by construction. A
   prefix enters the decision only when :func:`rank_margin` finds the whole
   block retained with two decades to spare on the package's thresholds, so
   that a small perturbation of the means cannot change which subspace
   ``solve_projected`` keeps. Q18's linearization assumes exactly that.
3. **Validate the linearization at eighty digits.** Q18-S1's fixed-step
   Richardson ladder perturbs word means by ``2e-3`` to ``2.5e-4``; on a pencil
   whose smallest normalized overlap eigenvalue is ``1e-7`` that is far outside
   the linear regime and inside double-precision rounding at once. The
   declared check instead differentiates :class:`DecimalPencil` at a step of
   ``1e-30``, where truncation and rounding are both far below the tolerance,
   and compares the energy's and the residual's analytic functionals with it.
   The fixed ladder is still evaluated and reported, read by no clause.
4. **Oracle bound.** Q18's estimator, groupings and Neyman ratio, imported
   from Q18 and Q18-S2 rather than restated.
5. **Practical allocation.** Only where the oracle bound leaves room:
   :func:`practical_stage` samples pilots and production shots from the exact
   grouped outcome distributions, estimates functional weights and setting
   variances from the pilot, allocates, and validates the resulting
   intervals over seeded replicates.
"""

from __future__ import annotations

import math
import time
from decimal import Decimal, localcontext

from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import phase15_certifying_screen as screen
from benchmarks import run_phase15_measured_residual_preflight as q18_producer
from benchmarks.phase15_residual_successor import integer_oracle_allocation
from clifford_qc.subspace.projection import MatrixElementBank

# One decade on each of solve_projected's two retention rules.
RANK_MARGIN = 10.0
# Digits of the independent pencil, and its central-difference step.
DECIMAL_DIGITS = 80
DECIMAL_STEP = "1e-30"
# Both independent routes must reproduce the double-precision sigma^2 to six
# digits, and the energy to this absolute tolerance per unit |E|.
VARIANCE_AGREEMENT = 1e-6
ENERGY_AGREEMENT = 1e-9
# Inverse iteration in the Decimal pencil: the shift sits this fraction of the
# double-precision gap below the double-precision ground root.
SHIFT_FRACTION = 1e-4
MAX_INVERSE_ITERATIONS = 400
STATUSES = screen.STATUSES
UNPRICED = ("RANK_UNSTABLE", "UNRESOLVED")
ORACLE_VERDICTS = screen.VERDICTS
PRACTICAL_STATUSES = ("PRACTICAL_AFFORDABLE", "PRACTICAL_PROHIBITIVE", "VALIDATION_FAILED")
FINAL_VERDICTS = ("LICENSED", "PRACTICAL_PROHIBITIVE", "VALIDATION_FAILED", "CLOSED",
                  "GROUPING_SENSITIVE", "UNREACHED", "INVALID")
# Two-sided and one-sided 95% normal quantiles.
Z_TWO_SIDED = 1.959963984540054
Z_ONE_SIDED = 1.6448536269514722


# ------------------------------------------------------------------ the bank

class KrylovBank(MatrixElementBank):
    """``{I, H, ..., H^order}`` with every row read off one power table.

    ``A_i = H^i`` is Hermitian, so ``A_i' A_j = H^(i+j)``,
    ``A_i' (H A_j) = H^(i+j+1)`` and ``A_i' (H^2 A_j) = H^(i+j+2)``: the same
    operators the base bank multiplies out pair by pair, at one product per
    power. Only object storage is supported, and only ``H * H`` among the
    projected observables takes the Hankel route; any other observable falls
    back to the base bank's products.
    """

    def __init__(self, rho, hamiltonian, order: int):
        from clifford_qc.multivector import MV
        from clifford_qc.subspace import identity_generator, krylov_response
        from clifford_qc.subspace.contracts import as_multivector

        if order < 1:
            raise ValueError("a Krylov bank needs at least one power")
        H = as_multivector(hamiltonian)
        powers = [MV.scalar(H.n, 1.0)]
        for _ in range(2 * order + 2):
            powers.append(powers[-1] * H)
        self._powers = tuple(powers)
        generators = [identity_generator(H.n), *krylov_response(H, order)]
        for k, generator in enumerate(generators):
            if generator.mv.terms != powers[k].terms:
                raise AssertionError(f"generator {generator.label} is not H^{k}")
        super().__init__(rho, H, generators, storage="object")

    @property
    def powers(self) -> tuple:
        return self._powers

    def _build_pair(self, i: int, j: int, *, evaluate: bool = False) -> None:
        # The base method's bookkeeping, with the two products replaced.
        started = time.perf_counter()
        overlap_op = self._powers[i + j].copy()
        element_op = self._powers[i + j + 1].copy()
        self._seconds += time.perf_counter() - started
        self._account(j, overlap_op)
        self._account(j, element_op)
        if evaluate:
            self._entries[(i, j)] = (self._scale * overlap_op.trace_pairing(self._rho),
                                     self._scale * element_op.trace_pairing(self._rho))
        self._overlap_ops[(i, j)] = overlap_op
        self._element_ops[(i, j)] = element_op
        self._pairs_built += 1
        self._peak_resident_rows = max(self._peak_resident_rows, self._resident_row_count())

    def observable_operator(self, observable, i: int, j: int, *, label: str | None = None):
        from clifford_qc.subspace.contracts import as_multivector
        from clifford_qc.subspace.projection import _identity_key

        Q = as_multivector(observable)
        if Q.terms != self._powers[2].terms:
            return super().observable_operator(Q, i, j, label=label)
        for index in (i, j):
            if not 0 <= index < len(self._generators):
                raise IndexError(f"generator index {index} out of range")
        record = self._observables.setdefault(
            _identity_key(Q), {"label": label or f"Q{len(self._observables)}",
                               "universe": set(), "acted": {}, "operators": {},
                               "hermitian": True})
        if i > j:
            return self.observable_operator(Q, j, i, label=label).dagger()
        cache = record["operators"]
        if (i, j) not in cache:
            cache[(i, j)] = self._powers[i + j + 2].copy()
            record["universe"].update(cache[(i, j)].terms)
            self._account_observable(cache[(i, j)])
            self._peak_resident_rows = max(self._peak_resident_rows,
                                           self._resident_row_count())
        return cache[(i, j)]


def krylov_bank(model, order: int) -> KrylovBank:
    """The ladder's Krylov basis on ``model``'s reference, at ``order``."""
    from clifford_qc.backends import ExactMVBackend

    rho = ExactMVBackend().state(model.reference, ())
    return KrylovBank(rho, model.hamiltonian, order)


# ------------------------------------------------------------------ rank margin

def rank_margin(result, size: int) -> dict:
    """Whether ``solve_projected`` keeps the whole block with room to spare.

    ``overlap_eigenvalues`` are the normalized overlap's, before truncation.
    The margin asks for ``RANK_MARGIN`` times the absolute floor and a
    condition number ``RANK_MARGIN`` below the cap: a perturbation that moves
    the normalized overlap by less than ninety-nine floors cannot change the
    retained block.
    """
    from clifford_qc.subspace.linalg import DEFAULT_MAX_CONDITION, DEFAULT_TAU_S

    smallest = float(min(result.overlap_eigenvalues))
    largest = float(max(result.overlap_eigenvalues))
    full = int(result.effective_rank) == size
    condition = largest / smallest if smallest > 0.0 else math.inf
    return {
        "effective_rank": int(result.effective_rank),
        "smallest_overlap_eigenvalue": smallest,
        "largest_overlap_eigenvalue": largest,
        "overlap_condition_number": condition,
        "retained_condition_number": float(result.condition_number),
        "full_rank": full,
        "rank_stable": bool(full and smallest >= RANK_MARGIN * DEFAULT_TAU_S
                            and condition <= DEFAULT_MAX_CONDITION / RANK_MARGIN),
    }


# ------------------------------------------------------------------ eighty digits

def _real(value) -> Decimal:
    value = complex(value)
    if value.imag != 0.0:
        raise ValueError("the Decimal pencil needs real rows; this one is complex")
    return Decimal(value.real)


class DecimalPencil:
    """``(S, H, K)`` along one direction, ``entry(t) = base + t * slope``.

    Every coefficient and mean is converted exactly, and sums carry
    ``DECIMAL_DIGITS`` digits. Rows must be real, which Krylov rows of a real
    Hamiltonian are; a complex row is refused rather than truncated.
    """

    def __init__(self, rows, size: int, means, direction=None):
        direction = direction or {}
        self.size = size
        self.base, self.slope = {}, {}
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            converted = {}
            for (a, b), parts in rows.items():
                for k, terms in enumerate(parts):
                    base = slope = Decimal(0)
                    for word, coefficient in terms.items():
                        c = converted.get((k, word, coefficient))
                        if c is None:
                            c = converted[(k, word, coefficient)] = _real(coefficient)
                        base += c * Decimal(float(means.get(word, 0.0)))
                        if word in direction:
                            slope += c * Decimal(float(direction[word]))
                    self.base[(k, a, b)] = base
                    self.slope[(k, a, b)] = slope

    def matrices(self, t: Decimal):
        out = []
        with localcontext() as context:
            context.prec = DECIMAL_DIGITS
            for k in range(3):
                matrix = [[Decimal(0)] * self.size for _ in range(self.size)]
                for a in range(self.size):
                    for b in range(a, self.size):
                        value = self.base[(k, a, b)] + t * self.slope[(k, a, b)]
                        matrix[a][b] = matrix[b][a] = value
                out.append(matrix)
        return tuple(out)


def _cholesky(S):
    m = len(S)
    L = [[Decimal(0)] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1):
            total = S[i][j] - sum((L[i][k] * L[j][k] for k in range(j)), Decimal(0))
            if i == j:
                if total <= 0:
                    raise ValueError("the overlap is not positive definite at eighty digits")
                L[i][i] = total.sqrt()
            else:
                L[i][j] = total / L[j][j]
    return L


def _forward(L, column):
    out = []
    for i in range(len(L)):
        out.append((column[i] - sum((L[i][k] * out[k] for k in range(i)), Decimal(0)))
                   / L[i][i])
    return out


def _backward_transpose(L, column):
    m = len(L)
    out = [Decimal(0)] * m
    for i in reversed(range(m)):
        out[i] = (column[i] - sum((L[k][i] * out[k] for k in range(i + 1, m)), Decimal(0))
                  ) / L[i][i]
    return out


def _solve(A, column):
    """Gaussian elimination with partial pivoting."""
    m = len(A)
    M = [list(row) + [value] for row, value in zip(A, column)]
    for k in range(m):
        pivot = max(range(k, m), key=lambda r: abs(M[r][k]))
        if M[pivot][k] == 0:
            raise ValueError("singular shifted pencil")
        M[k], M[pivot] = M[pivot], M[k]
        for r in range(k + 1, m):
            factor = M[r][k] / M[k][k]
            if factor:
                for c in range(k, m + 1):
                    M[r][c] -= factor * M[k][c]
    out = [Decimal(0)] * m
    for i in reversed(range(m)):
        out[i] = (M[i][m] - sum((M[i][c] * out[c] for c in range(i + 1, m)), Decimal(0))
                  ) / M[i][i]
    return out


def decimal_ritz(S, H, K, shift: Decimal) -> tuple[Decimal, Decimal]:
    """The ground Ritz energy and ``sigma^2 = c'Kc - E^2`` at eighty digits.

    No truncation: the whole block is kept, as the rank margin requires.
    ``C = L^-1 H L^-T`` is the pencil in an ``S``-orthonormal basis; shifted
    inverse iteration from just below the double-precision ground root
    converges to that root, and ``c = L^-T x`` with ``|x| = 1`` has ``c'Sc = 1``.
    """
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        m = len(S)
        L = _cholesky(S)
        Y = [_forward(L, [H[i][j] for i in range(m)]) for j in range(m)]  # columns of L^-1 H
        C = [[Decimal(0)] * m for _ in range(m)]
        for i in range(m):
            row = _forward(L, [Y[j][i] for j in range(m)])  # (L^-1 (L^-1 H)^T)
            for j in range(m):
                C[i][j] = row[j]
        shifted = [[C[i][j] - (shift if i == j else 0) for j in range(m)] for i in range(m)]
        x = [Decimal(1)] * m
        energy = None
        tolerance = Decimal(10) ** (-(DECIMAL_DIGITS - 12))
        for iteration in range(MAX_INVERSE_ITERATIONS):
            y = _solve(shifted, x)
            norm = sum((value * value for value in y), Decimal(0)).sqrt()
            x = [value / norm for value in y]
            Cx = [sum((C[i][j] * x[j] for j in range(m)), Decimal(0)) for i in range(m)]
            new = sum((x[i] * Cx[i] for i in range(m)), Decimal(0))
            if energy is not None and iteration >= 4 and abs(new - energy) <= tolerance * max(
                    Decimal(1), abs(new)):
                energy = new
                break
            energy = new
        else:
            raise ValueError("inverse iteration did not converge at eighty digits")
        c = _backward_transpose(L, x)
        second = sum((c[i] * K[i][j] * c[j] for i in range(m) for j in range(m)), Decimal(0))
        return energy, second - energy * energy


def precise_values(rows, size: int, means, *, shift: float) -> dict:
    """The ground energy and ``sigma^2`` of the exact-mean pencil, at eighty digits."""
    pencil = DecimalPencil(rows, size, means)
    energy, variance = decimal_ritz(*pencil.matrices(Decimal(0)), Decimal(shift))
    return {"energy": float(energy), "variance": float(variance)}


def precise_derivatives(rows, size: int, means, direction, *, shift: float) -> dict:
    """Central differences of the eighty-digit pencil at ``t = DECIMAL_STEP``.

    The truncation error is ``O(t^2)`` and the rounding ``1e-80 / t``, both
    tens of orders below any tolerance read here.
    """
    pencil = DecimalPencil(rows, size, means, direction)
    t = Decimal(DECIMAL_STEP)
    with localcontext() as context:
        context.prec = DECIMAL_DIGITS
        plus = decimal_ritz(*pencil.matrices(t), Decimal(shift))
        minus = decimal_ritz(*pencil.matrices(-t), Decimal(shift))
        return {"energy": float((plus[0] - minus[0]) / (2 * t)),
                "variance": float((plus[1] - minus[1]) / (2 * t))}


def ground_shift(result) -> float:
    energies = [float(value) for value in result.energies]
    gap = energies[1] - energies[0] if len(energies) > 1 else 1.0
    return energies[0] - SHIFT_FRACTION * gap


# ------------------------------------------------------------------ dense route

def dense_ritz(model, size: int) -> dict:
    """``sigma^2 = ||(H - E)Psi||^2`` of the normalized Ritz state, by statevector.

    Independent of the word rows: the Krylov vectors are ``H^k|psi>`` applied
    matrix-free, their pencil is solved by ``solve_projected`` at the default
    thresholds, and the residual is read off the state itself.
    """
    import numpy as np

    from benchmarks.run_phase15_h2_preflight import reference_vector
    from clifford_qc.pauli_action import PauliLinearOperator
    from clifford_qc.subspace.linalg import solve_projected

    hamiltonian = PauliLinearOperator(model.hamiltonian.to_mv())
    vectors = [reference_vector(model)]
    for _ in range(size - 1):
        vectors.append(hamiltonian.matvec(vectors[-1]))
    basis = np.column_stack(vectors)
    images = np.column_stack([hamiltonian.matvec(v) for v in vectors])
    overlap = basis.conj().T @ basis
    projected = basis.conj().T @ images
    result = solve_projected(0.5 * (overlap + overlap.conj().T),
                             0.5 * (projected + projected.conj().T))
    state = basis @ result.coefficients[:, 0]
    state = state / np.linalg.norm(state)
    acted = hamiltonian.matvec(state)
    energy = float(np.vdot(state, acted).real)
    return {"energy": energy, "variance": float(np.linalg.norm(acted - energy * state) ** 2),
            "effective_rank": int(result.effective_rank)}


# ------------------------------------------------------------------ functionals

def energy_functional(rows, size: int, means) -> tuple[dict[int, float], float]:
    """``ritz_functional``'s weights from the rows and a set of means.

    The word expansion of ``B'(H - E)B`` for the ground root of the pencil the
    means give, identity included under code ``0``. At the exact means this is
    the existing ``ritz_functional``; a pilot's means give the plug-in weights.
    """
    import numpy as np

    from clifford_qc.subspace.linalg import solve_projected

    S, H, _ = q18.projected_matrices(rows, size, means)
    result = solve_projected(S, H)
    if result.effective_rank != size:
        raise ValueError("the linearization assumes the whole block is retained")
    c, energy = result.coefficients[:, 0], float(result.energies[0])
    weights: dict[int, float] = {}
    for (a, b), (s, h, _) in rows.items():
        outer = np.conj(c[a]) * c[b]
        for word in set(s) | set(h):
            value = outer * (complex(h.get(word, 0.0)) - energy * complex(s.get(word, 0.0)))
            weights[word] = weights.get(word, 0.0) + (value.real if a == b else
                                                      2.0 * value.real)
    return weights, energy


def _dot(weights, direction) -> float:
    return float(sum(weights.get(word, 0.0) * value for word, value in direction.items()))


def _norm(weights) -> float:
    return math.sqrt(sum(value * value for word, value in weights.items() if word != 0))


def linearization_check(rows, size, means, energy_weights, residual_weights, direction, *,
                        shift: float, relative_tolerance: float) -> dict:
    """Both analytic functionals against the eighty-digit derivative along ``direction``."""
    numeric = precise_derivatives(rows, size, means, direction, shift=shift)
    out = {}
    for side, weights in (("energy", energy_weights), ("variance", residual_weights)):
        analytic = _dot(weights, direction)
        tolerance = relative_tolerance * _norm(weights)
        out[side] = {"analytic": analytic, "numeric": numeric[side], "tolerance": tolerance,
                     "passes": bool(math.isfinite(numeric[side])
                                    and abs(numeric[side] - analytic) <= tolerance)}
    return out


# ------------------------------------------------------------------ one prefix

def certification(bank, second, size, spectrum, model) -> dict:
    """Stage 1 for one prefix: the interval, its resolution, the rank margin."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    order = list(range(size))
    result = bank.solve(order)
    residual = second.residual(result, 0)
    energy = float(result.ground_energy)
    variance = float(residual.variance)
    sigma = math.sqrt(max(variance, 0.0))
    scale = float(residual.cancellation_scale)
    ground, first = spectrum["ground_energy"], spectrum["first_excited_energy"]
    energies = [float(value) for value in result.energies]
    margin = rank_margin(result, size)
    entry = {
        "basis_size": size,
        "ground_energy": energy,
        "error": energy - ground,
        "variance": variance,
        "residual_norm": sigma,
        "variance_resolved": bool(residual.resolved),
        "cancellation_scale": scale,
        "variance_resolution": RESOLUTION * scale,
        "weinstein_upper": energy + sigma,
        "ground_weight_lower_bound": screen.ground_weight_lower_bound(energy, ground, first),
        "temple_lower_bound": screen.temple_lower_bound(energy, variance, first),
        "subspace_gap": energies[1] - energies[0] if len(energies) > 1 else None,
        "rank": margin,
    }
    dense = dense_ritz(model, size)
    entry["dense"] = dense
    entry["certifying"] = bool(residual.resolved) and screen.certifying(
        energy, sigma, ground, first)
    if margin["rank_stable"]:
        rows = q18.block_rows(bank, second, order)
        words = q18.row_words(rows)
        means = q18.reference_means(bank.reference, words)
        entry["precise"] = precise_values(rows, size, means, shift=ground_shift(result))
    return entry


def independent_checks(entry) -> dict:
    """Both independent routes reproduce the double-precision pair."""
    precise, dense = entry["precise"], entry["dense"]
    reference = precise["variance"]
    bound = VARIANCE_AGREEMENT * abs(reference)
    return {
        "precise_energy_agrees": abs(entry["ground_energy"] - precise["energy"])
        <= ENERGY_AGREEMENT * max(1.0, abs(precise["energy"])),
        "precise_variance_agrees": abs(entry["variance"] - reference) <= bound,
        "dense_energy_agrees": abs(dense["energy"] - precise["energy"])
        <= ENERGY_AGREEMENT * max(1.0, abs(precise["energy"])),
        "dense_variance_agrees": abs(dense["variance"] - reference) <= bound,
    }


def nondegenerate(gap, energy) -> bool:
    """Q18's gap floor; a one-root pencil has no other root to meet."""
    return gap is None or gap > q18.GAP_FLOOR * max(1.0, abs(energy))


def price_prefix(config, bank, second, entry, spectrum, *, protocols, say=None) -> dict:
    """Stages 2 and 3 for a rank-stable, resolved prefix: validate and price."""
    from clifford_qc.measurement.functionals import ritz_functional

    size = entry["basis_size"]
    order = list(range(size))
    result = bank.solve(order)
    rows = q18.block_rows(bank, second, order)
    sh_words = set(bank.word_set(order))
    combined = q18.row_words(rows)
    means = q18.reference_means(bank.reference, combined)
    weights, functional_variance = q18.residual_functional(rows, size, means)
    residual_weights = {int(word): float(value) for word, value in weights.items()}
    measured_residual = {w: v for w, v in residual_weights.items() if w != 0}
    ritz = ritz_functional(bank, result.indices, result.ritz_vector(), result.ground_energy)
    energy_weights = dict(ritz.coefficients)
    scale = entry["cancellation_scale"]
    reference_mean = float(sum(value * means.get(word, 0.0)
                               for word, value in residual_weights.items()))
    entry["universes"] = {"raw_sh_word_universe": len(sh_words),
                          "raw_combined_word_universe": len(combined),
                          "measured_sh_words": len(sh_words - {0}),
                          "measured_combined_words": len(combined - {0})}
    entry["estimator"] = {
        "functional_variance": float(functional_variance),
        "matched_precision_half": entry["residual_norm"] / 4.0,
        "residual_functional_words": len(measured_residual),
        "residual_functional_norm": _norm(residual_weights),
        "residual_identity_weight": float(residual_weights.get(0, 0.0)),
        "residual_reference_mean": reference_mean,
        "energy_functional_words": len(energy_weights),
        "energy_functional_norm": _norm(energy_weights),
    }

    rule = config["derivative_validation"]
    directions = q18.check_directions(residual_weights, sh_words, seed=int(rule["seed"]))
    shift = ground_shift(result)
    entry["precise_derivatives"] = {
        key: linearization_check(rows, size, means, energy_weights, residual_weights,
                                 directions[key], shift=shift,
                                 relative_tolerance=float(rule["relative_tolerance"]))
        for key in rule["directions"]}
    fixed = config["fixed_step_validation"]
    compiled = screen.CompiledRows(rows, size, combined)
    entry["fixed_step"] = {
        key: screen.derivative_check(compiled, means, residual_weights, directions[key],
                                     rule=fixed, cancellation_scale=scale)
        for key in rule["directions"]}
    if say:
        say(f"M={size}: sigma={entry['residual_norm']:.6g} certifying={entry['certifying']} "
            "linearization " + ", ".join(
                f"{k}={'pass' if all(s['passes'] for s in v.values()) else 'FAIL'}"
                for k, v in entry["precise_derivatives"].items()))

    priced = {}
    declared, alternative = protocols
    for role, protocol in (("declared", declared), ("alternative", alternative)):
        if protocol is None:
            continue
        price = q18_producer.price_protocol(bank.reference, protocol, bank.reference.n,
                                            sh_words, combined, energy_weights,
                                            measured_residual, functional_variance,
                                            include_setting_variances=True)
        price["allocation_diagnostic"] = (
            screen.allocation_summary(price, functional_variance,
                                      config["allocation_diagnostic"])
            if price["variances_nonnegative"] else None)
        priced[protocol] = {"role": role, **price}
    entry["protocols"] = priced

    energy = entry["ground_energy"]
    checks = {
        "full_rank": entry["rank"]["full_rank"],
        "ground_root_nondegenerate": nondegenerate(entry["subspace_gap"], energy),
        "energy_consistent_with_spectrum": energy >= spectrum["ground_energy"]
        - screen.spectral_rounding(energy, spectrum["ground_energy"]),
        **independent_checks(entry),
        "residual_functional_reproduces_variance": (
            abs(functional_variance - entry["variance"])
            <= screen.VARIANCE_TOLERANCE * (1.0 + scale)),
        "residual_functional_mean_zero": abs(reference_mean)
        <= screen.MEAN_TOLERANCE * (1.0 + scale),
        "linearization_validated": all(side["passes"]
                                       for row in entry["precise_derivatives"].values()
                                       for side in row.values()),
        "fixed_step_validated": all(row["passes"] for row in entry["fixed_step"].values()),
        "partitions_valid": all(not p["partition_problems"] for p in priced.values()),
        "group_variances_nonnegative": all(p["variances_nonnegative"] for p in priced.values()),
    }
    entry["deterministic_checks"] = checks
    ratios = {protocol: price["neyman_ratio"] for protocol, price in priced.items()}
    entry["status"] = screen.status_of(ratios, declared, alternative,
                                       float(config["statistic"]["max_ratio"]), checks)
    return entry


def evaluate_prefixes(config, model, spectrum, *, prefixes, protocols, progress=None) -> dict:
    """Stage 1 on every prefix, then validation and pricing where it applies."""
    from clifford_qc.subspace import SecondMomentBank

    order = max(prefixes) - 1
    bank = krylov_bank(model, order)
    second = SecondMomentBank(bank)
    out = {}
    for size in prefixes:
        started = time.perf_counter()
        entry = certification(bank, second, size, spectrum, model)
        if not entry["rank"]["rank_stable"]:
            entry.update(status="RANK_UNSTABLE", role="unpriced", in_domain=False)
        elif not entry["variance_resolved"]:
            entry.update(status="UNRESOLVED", role="unpriced", in_domain=False)
        else:
            entry = price_prefix(config, bank, second, entry, spectrum, protocols=protocols,
                                 say=progress)
            entry["in_domain"] = entry["certifying"]
            entry["role"] = "decision" if entry["in_domain"] else "diagnostic"
        entry["seconds"] = round(time.perf_counter() - started, 2)
        if progress is not None:
            progress(f"M={size}: {entry['status']} ({entry['role']}) in {entry['seconds']} s")
        out[str(size)] = entry
    return out


def oracle_verdict(entries) -> str:
    """Q18-S2's verdict rule on this domain: certifying and rank-stable prefixes."""
    return screen.verdict_of(entries)


def practical_prefixes(prefixes: dict) -> list[str]:
    """The prefixes the practical stage runs on: AFFORDABLE and in the domain."""
    return [size for size, entry in prefixes.items()
            if entry["in_domain"] and entry["status"] == "AFFORDABLE"]


def final_verdict(oracle: str, practical: dict) -> str:
    """The oracle verdict, unless it is OPEN; then the practical stage decides."""
    if oracle != "OPEN":
        return oracle
    statuses = list(practical.values())
    if not statuses:
        raise ValueError("an OPEN oracle verdict runs the practical stage on some prefix")
    if "PRACTICAL_AFFORDABLE" in statuses:
        return "LICENSED"
    if "VALIDATION_FAILED" in statuses:
        return "VALIDATION_FAILED"
    return "PRACTICAL_PROHIBITIVE"


# ------------------------------------------------------------------ practical stage

class SettingSampler:
    """The exact grouped outcome distribution of every setting of one partition.

    Built by ``FiniteShotBackend``'s own grouped-sampling plans: the state is
    rotated onto each setting's shared basis, the untouched qubits are traced
    out, and each word's readout is the parity of its support. A shot count
    then costs one multinomial draw per setting, whatever its size.
    """

    def __init__(self, rho, groups):
        import numpy as np

        from clifford_qc.backends import FiniteShotBackend

        backend = FiniteShotBackend(0)
        table = backend._group_plans(rho)
        self.settings = []
        for group in groups:
            plan = backend._plan(rho, group, table)
            codes = tuple(word.code for word in group)
            signs = np.column_stack([np.where(plan.parity[code], 1.0, -1.0)
                                     for code in codes])
            self.settings.append((codes, plan.probs, signs))

    def __len__(self) -> int:
        return len(self.settings)

    def draw(self, rng, shots):
        return [rng.multinomial(int(n), probs) for (_, probs, _), n in zip(self.settings, shots)]

    def means(self, counts, shots) -> dict[int, float]:
        out = {0: 1.0}
        for (codes, _, signs), tally, n in zip(self.settings, counts, shots):
            values = (tally @ signs) / float(n)
            out.update(zip(codes, values.tolist()))
        return out

    def variances(self, counts, shots, weights) -> list[float]:
        """Unbiased per-setting sample variances of the functional's shot values."""
        import numpy as np

        out = []
        for (codes, _, signs), tally, n in zip(self.settings, counts, shots):
            f = np.array([weights.get(code, 0.0) for code in codes])
            values = signs @ f
            mean = float(tally @ values) / n
            out.append(float(tally @ (values - mean) ** 2) / (n - 1) if n > 1 else 0.0)
        return out


def _functional(side, rows, size, means):
    if side == "energy":
        return energy_functional(rows, size, means)
    weights, variance = q18.residual_functional(rows, size, means)
    return {int(w): float(v) for w, v in weights.items()}, float(variance)


def _estimate(side, compiled, size, means):
    """The declared estimator on sampled means; ``None`` outside its domain."""
    from clifford_qc.subspace.linalg import solve_projected

    S, H, K = compiled.matrices(compiled.vector(means))
    result = solve_projected(S, H)
    energies = [float(value) for value in result.energies]
    gap = energies[1] - energies[0] if len(energies) > 1 else math.inf
    if result.effective_rank != size or gap <= q18.GAP_FLOOR * max(1.0, abs(energies[0])):
        return None
    if side == "energy":
        return energies[0]
    c = result.coefficients[:, 0]
    norm = float((c.conj() @ S @ c).real)
    return float((c.conj() @ K @ c).real) / norm - energies[0] ** 2


def campaign(side, sampler, rows, compiled, size, rule, budget, rng) -> dict:
    """One pilot-allocated campaign on one side, from the shots alone.

    The pilot's means give plug-in functional weights, and its shot values
    give each setting's variance under them. Those set an integer Neyman
    allocation for the target standard error, with Q18-S1's floor. Production
    shots alone give the estimate and its delta-method standard error.
    """
    import numpy as np

    target = budget["target_standard_error"]
    pilot = [budget["pilot_shots_per_setting"]] * len(sampler)
    counts = sampler.draw(rng, pilot)
    means = sampler.means(counts, pilot)
    try:
        weights, value = _functional(side, rows, size, means)
    except ValueError:
        return {"failure": "pilot_pencil_rank_deficient"}
    variances = [max(v, 0.0) for v in sampler.variances(counts, pilot, weights)]
    if side == "residual":
        sigma = math.sqrt(max(value, target * target))
        variances = [v / (4.0 * sigma * sigma) for v in variances]
    allocation = integer_oracle_allocation(
        variances, target, pilot_shots=budget["pilot_shots_per_setting"],
        minimum_shots=int(rule["minimum_shots_per_setting"]))
    shots = allocation["shots_per_setting"]
    counts = sampler.draw(rng, shots)
    means = sampler.means(counts, shots)
    estimate = _estimate(side, compiled, size, means)
    if estimate is None:
        return {"failure": "production_pencil_outside_domain",
                "production_shots": allocation["production_shots"],
                "pilot_shots": allocation["pilot_shots"]}
    try:
        weights, _ = _functional(side, rows, size, means)
    except ValueError:
        return {"failure": "production_pencil_rank_deficient"}
    se2 = float(np.sum(np.array(sampler.variances(counts, shots, weights))
                       / np.array(shots, dtype=float)))
    out = {"failure": None, "production_shots": allocation["production_shots"],
           "pilot_shots": allocation["pilot_shots"], "total_shots": allocation["total_shots"]}
    if side == "energy":
        out.update(estimate=estimate, standard_error=math.sqrt(max(se2, 0.0)))
    else:
        sigma = math.sqrt(max(estimate, 0.0))
        out.update(variance_estimate=estimate, estimate=sigma,
                   standard_error=(math.sqrt(max(se2, 0.0)) / (2.0 * sigma)
                                   if sigma > 0.0 else math.inf))
    return out


def practical_budget(price, variance, rule) -> dict:
    """Pilot and target per side, from the declared rule and the stage-2 record.

    The target is the matched standard error ``sigma / 100``. Each side's
    pilot spreads ``pilot_fraction`` of its own oracle Neyman production
    budget uniformly over its settings, never fewer than Q18-S1's 64 shots.
    """
    sigma = math.sqrt(variance)
    target = sigma * float(rule["standard_error_fraction_of_sigma"])
    out = {}
    for side, settings, neyman, scale in (
            ("energy", price["energy_settings"], price["energy_neyman_sum"], 1.0),
            ("residual", price["residual_settings"], price["residual_neyman_sum"],
             1.0 / (4.0 * variance))):
        oracle = neyman * neyman * scale / (target * target)
        pilot = max(int(rule["minimum_pilot_shots_per_setting"]),
                    math.ceil(float(rule["pilot_fraction"]) * oracle / settings))
        out[side] = {"target_standard_error": target, "oracle_production_shots": oracle,
                     "pilot_shots_per_setting": pilot}
    return out


def practical_stage(config, model, entry, spectrum, *, progress=None) -> dict:
    """The sampled pilot-allocation experiment on one priced, certifying prefix."""
    import numpy as np

    from clifford_qc.subspace import SecondMomentBank

    rule = config["practical_allocation"]
    size = entry["basis_size"]
    protocol = config["grouping"]["declared_protocol"]
    price = entry["protocols"][protocol]
    exact_energy, exact_sigma = entry["ground_energy"], entry["residual_norm"]
    bank = krylov_bank(model, max(size - 1, 1))
    second = SecondMomentBank(bank)
    order = list(range(size))
    rows = q18.block_rows(bank, second, order)
    sh_words = set(bank.word_set(order))
    combined = q18.row_words(rows)
    budget = practical_budget(price, entry["variance"], rule)
    # The pencil is compiled over every row word; the energy campaign's means
    # leave the K-only words at zero, which only K reads.
    compiled = screen.CompiledRows(rows, size, combined)
    sides = {}
    for side, universe in (("energy", sh_words), ("residual", combined)):
        groups = q18.group_words(universe, protocol, model.n)
        sides[side] = (SettingSampler(bank.reference, groups), compiled)
    replicates = []
    seeds = np.random.SeedSequence(int(rule["seed"])).spawn(int(rule["replicate_count"]))
    for index, seed in enumerate(seeds):
        streams = [np.random.default_rng(child) for child in seed.spawn(2)]
        row = {"replicate": index}
        for (side, (sampler, compiled)), rng in zip(sides.items(), streams):
            row[side] = campaign(side, sampler, rows, compiled, size, rule, budget[side], rng)
        replicates.append(row)
        if progress is not None and (index + 1) % 10 == 0:
            progress(f"  M={size}: {index + 1} replicates")
    return summarize_practical(replicates, budget, rule, exact_energy=exact_energy,
                               exact_sigma=exact_sigma,
                               first_excited=spectrum["first_excited_energy"])


def effective_shots(row: dict, target: float) -> float:
    """Shots this allocation needs for the target: pilot plus production rescaled.

    Production cost scales as ``1/SE^2`` at fixed allocation proportions, and
    the pilot is a fixed charge. A pilot that misjudges the setting variances
    misallocates production, and the rescaling charges that inefficiency;
    the coverage checks are what make ``standard_error`` safe to rescale by.
    """
    ratio = row["standard_error"] / target
    return row["pilot_shots"] + row["production_shots"] * ratio * ratio


def summarize_practical(replicates, budget, rule, *, exact_energy, exact_sigma,
                        first_excited) -> dict:
    """Coverage, calibration, effective cost ratio and the declared status."""
    import numpy as np

    failures = sum(1 for row in replicates
                   if row["energy"]["failure"] or row["residual"]["failure"])
    good = [row for row in replicates
            if not row["energy"]["failure"] and not row["residual"]["failure"]]
    target = budget["energy"]["target_standard_error"]
    out = {"budget": budget, "replicates": replicates, "failures": failures}
    validation = rule["validation"]
    if len(good) < 2:
        out.update(summary=None, status="VALIDATION_FAILED")
        return out
    E = np.array([row["energy"]["estimate"] for row in good])
    s = np.array([row["residual"]["estimate"] for row in good])
    seE = np.array([row["energy"]["standard_error"] for row in good])
    seS = np.array([row["residual"]["standard_error"] for row in good])
    zE, zS = (E - exact_energy) / seE, (s - exact_sigma) / seS
    residual_effective = np.array([effective_shots(row["residual"], target) for row in good])
    energy_effective = np.array([effective_shots(row["energy"], target) for row in good])
    effective = residual_effective / energy_effective
    totals = np.array([row["residual"]["total_shots"] / row["energy"]["total_shots"]
                       for row in good])
    upper = E + s + Z_ONE_SIDED * np.sqrt(seE ** 2 + seS ** 2)
    summary = {
        "valid_replicates": len(good),
        "energy_coverage": float(np.mean(np.abs(zE) <= Z_TWO_SIDED)),
        "sigma_coverage": float(np.mean(np.abs(zS) <= Z_TWO_SIDED)),
        "energy_z_spread": float(np.std(zE, ddof=1)),
        "sigma_z_spread": float(np.std(zS, ddof=1)),
        "energy_spread_over_target": float(np.std(E, ddof=1) / target),
        "sigma_spread_over_target": float(np.std(s, ddof=1) / target),
        "energy_bias_over_target": float((np.mean(E) - exact_energy) / target),
        "sigma_bias_over_target": float((np.mean(s) - exact_sigma) / target),
        "expected_effective_cost_ratio": float(np.mean(residual_effective)
                                               / np.mean(energy_effective)),
        "median_effective_cost_ratio": float(np.median(effective)),
        "upper_decile_effective_cost_ratio": float(np.quantile(effective, 0.9)),
        "median_total_cost_ratio": float(np.median(totals)),
        "certification_rate": float(np.mean(upper < first_excited)),
    }
    checks = {
        "failures_within_limit": failures <= int(validation["max_failures"]),
        "energy_coverage": summary["energy_coverage"] >= float(validation["min_coverage"]),
        "sigma_coverage": summary["sigma_coverage"] >= float(validation["min_coverage"]),
        "energy_calibrated": summary["energy_z_spread"] <= float(validation["max_z_spread"]),
        "sigma_calibrated": summary["sigma_z_spread"] <= float(validation["max_z_spread"]),
    }
    if not all(checks.values()):
        status = "VALIDATION_FAILED"
    elif summary["expected_effective_cost_ratio"] <= float(rule["max_ratio"]):
        status = "PRACTICAL_AFFORDABLE"
    else:
        status = "PRACTICAL_PROHIBITIVE"
    out.update(summary=summary, validation_checks=checks, status=status)
    return out
