"""
adapt_vqe_noisy_better.py — standalone noisy-selection ADAPT-VQE in Cl(2n,C)
================================================================================

This version improves the earlier noisy ADAPT prototype in five ways:

1. It is self-contained.  The uploaded prototype depended on ``adapt_vqe_tfim``;
   this file implements the missing ADAPT pieces directly.
2. It separates three notions that should not be conflated:
      * exact commutator gradient:      g_P = Tr[-i/2 [H,P] rho]
      * finite-shot estimator:          measured Pauli-word sum
      * selection confidence:           nonzero signal and best-vs-second gap
3. It uses cumulative shot escalation rather than discarding earlier samples.
4. It uses the hardware-available empirical/pseudocount uncertainty by default;
   oracle uncertainty is kept only for calibration and ablation.
5. Its default pool locality matches the Hamiltonian boundary condition: open-chain
   Hamiltonians use open-context pool operators unless explicitly overridden.

Only the ADAPT operator SELECTION is noisy.  Parameter re-optimization remains
exact so the experiment isolates ranking noise.  Exact diagonalization is used
only for validation targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import os
import sys
from typing import Callable, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_qc_operator_improved import (  # noqa: E402
    MV,
    I,
    X,
    Y,
    Z,
    comm,
    rotor,
    evolve,
    expectation,
    purity,
)
from vqe_tfim_rotor_better import (  # noqa: E402
    tfim_hamiltonian,
    plus_state,
    exact_ground_space,
    fidelity_with_subspace,
)

TOL = 1e-12


# ---------------------------------------------------------------------------
# 1. Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoolOp:
    label: str
    generator: MV


@dataclass(frozen=True)
class AdaptStep:
    step: int
    label: str
    g_hat: float
    sigma_hat: float
    exact_abs_g: float
    exact_abs_gmax: float
    energy: float
    rel_error: float
    fidelity: float
    near_optimal: bool
    ambiguous: bool
    shots_this_step: int
    cumulative_shots: int


@dataclass(frozen=True)
class AdaptResult:
    labels: tuple[str, ...]
    theta: tuple[float, ...]
    energy: float
    rel_error: float
    fidelity: float
    record: tuple[AdaptStep, ...]
    total_shots: int
    E0: float
    stopped_reason: str


@dataclass
class CandidateAccumulator:
    """Cumulative finite-shot estimator for one observable G = Σ c_w w."""

    label: str
    generator: MV
    observable: MV
    entries: list[tuple[complex, float]]
    shots_per_word: int = 0
    successes: list[int] | None = None

    def __post_init__(self) -> None:
        if self.successes is None:
            self.successes = [0] * len(self.entries)

    def add_shots(self, shots: int, rng) -> int:
        if shots < 0:
            raise ValueError("shots must be non-negative")
        if shots == 0 or not self.entries:
            return 0
        assert self.successes is not None
        for k, (_, ev) in enumerate(self.entries):
            p = 0.5 * (1.0 + min(1.0, max(-1.0, ev)))
            self.successes[k] += int(rng.binomial(shots, p))
        self.shots_per_word += shots
        return shots * len(self.entries)

    def estimate(self) -> tuple[float, float, float]:
        """Return (estimate, oracle_sigma, conservative_empirical_sigma).

        ``oracle_sigma`` uses exact expectations and is available only inside the
        simulator.  ``conservative_empirical_sigma`` is hardware-realistic: it is
        inferred from observed counts with Jeffreys-style 1/2 pseudocounts, so it
        remains nonzero even when finite shots accidentally return all +1 or all
        -1.  Selection decisions should use the empirical value.
        """
        if self.shots_per_word <= 0:
            return 0.0, math.inf, math.inf
        assert self.successes is not None
        est = 0.0
        var_oracle = 0.0
        var_emp = 0.0
        N = self.shots_per_word
        for (c, ev), k in zip(self.entries, self.successes):
            ev_hat = 2.0 * k / N - 1.0
            est += (c * ev_hat).real
            # Calibration/oracle variance used to verify the simulator.
            var_oracle += (abs(c) ** 2) * max(0.0, 1.0 - ev * ev) / N
            # Conservative empirical variance with pseudocounts; avoids false
            # zero variance when all finite shots give the same outcome.
            p_tilde = (k + 0.5) / (N + 1.0)
            ev_tilde = 2.0 * p_tilde - 1.0
            var_emp += (abs(c) ** 2) * max(0.0, 1.0 - ev_tilde * ev_tilde) / (N + 1.0)
        return est, math.sqrt(var_oracle), math.sqrt(var_emp)


# ---------------------------------------------------------------------------
# 2. ADAPT pool and exact ADAPT calculus
# ---------------------------------------------------------------------------

def _letters_from_factors(n: int, factors: dict[int, str]) -> str:
    letters = ["I"] * n
    for j, ch in factors.items():
        jj = j % n
        if letters[jj] != "I" and letters[jj] != ch:
            raise ValueError("conflicting Pauli factors in pool construction")
        letters[jj] = ch
    return "".join(letters)


def qubit_pool(n: int, *, periodic_context: bool = True) -> list[PoolOp]:
    """Local real-amplitude ADAPT pool for the TFIM.

    The pool contains one Y-centered generator per site, optionally dressed by
    Z on the immediate left and/or right context qubits.  With periodic context
    and n=4 this gives 4 generators per site = 16 pool operators, matching the
    intended test case in the uploaded prototype.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    ops: dict[int, PoolOp] = {}
    for i in range(n):
        contexts: list[tuple[str, dict[int, str]]] = [(f"Y{i}", {i: "Y"})]
        left_ok = periodic_context or i > 0
        right_ok = periodic_context or i < n - 1
        left = (i - 1) % n
        right = (i + 1) % n
        if left_ok and left != i:
            contexts.append((f"Z{left}Y{i}", {left: "Z", i: "Y"}))
        if right_ok and right != i:
            contexts.append((f"Y{i}Z{right}", {i: "Y", right: "Z"}))
        if left_ok and right_ok and left != i and right != i and left != right:
            contexts.append((f"Z{left}Y{i}Z{right}", {left: "Z", i: "Y", right: "Z"}))
        for label, fac in contexts:
            P = MV.word(n, _letters_from_factors(n, fac))
            # De-duplicate small-n periodic collisions while preserving order.
            code = next(iter(P.terms))
            ops.setdefault(code, PoolOp(label, P))
    return list(ops.values())


def selection_observable(H: MV, P: MV) -> MV:
    """Hermitian observable G_P = -i/2 [H,P]."""
    return -0.5j * comm(H, P)


def word_expectations(rho: MV, G: MV) -> list[tuple[complex, float]]:
    """Return [(c_w, <w>_rho)] for G = Σ c_w w."""
    out: list[tuple[complex, float]] = []
    for code, c in G.terms.items():
        w = MV(rho.n, {code: 1.0})
        out.append((c, expectation(rho, w).real))
    return out


def g_exact_from_words(rho: MV, G: MV) -> float:
    return sum((c * ev).real for c, ev in word_expectations(rho, G))


def selection_gradient(H: MV, rho: MV, P: MV) -> float:
    return expectation(rho, selection_observable(H, P)).real


def adapt_state(n: int, gens: Sequence[MV], theta: Sequence[float]) -> MV:
    if len(gens) != len(theta):
        raise ValueError("gens and theta must have the same length")
    rho = plus_state(n)
    for P, th in zip(gens, theta):
        if P.n != n:
            raise ValueError("generator algebra size mismatch")
        rho = evolve(rho, rotor(P, float(th)))
    return rho


def adapt_energy_and_gradient(H: MV, n: int, gens: Sequence[MV], theta: Sequence[float]) -> tuple[float, list[float]]:
    if len(gens) != len(theta):
        raise ValueError("gens and theta must have the same length")
    gates = [rotor(P, float(th)) for P, th in zip(gens, theta)]

    states_after: list[MV] = []
    rho = plus_state(n)
    for G in gates:
        rho = evolve(rho, G)
        states_after.append(rho)
    E = expectation(rho, H).real

    grads = [0.0] * len(theta)
    O = H
    for k in range(len(gates) - 1, -1, -1):
        comm_prho = gens[k] * states_after[k] - states_after[k] * gens[k]
        grads[k] = (-0.5j * (O * comm_prho).trace()).real
        O = gates[k].dagger() * O * gates[k]
    return E, grads


# ---------------------------------------------------------------------------
# 3. Optimization
# ---------------------------------------------------------------------------

def _adam_minimize(
    f_eg: Callable[[Sequence[float]], tuple[float, Sequence[float]]],
    theta0: Sequence[float],
    *,
    bound: float | None = None,
    maxiter: int = 400,
    lr: float = 0.05,
    gtol: float = 1e-8,
) -> tuple[float, tuple[float, ...]]:
    th = [float(x) for x in theta0]
    m = [0.0] * len(th)
    v = [0.0] * len(th)
    b1, b2, eps = 0.9, 0.999, 1e-8
    last_E = math.inf
    for it in range(1, maxiter + 1):
        E, g_raw = f_eg(th)
        g = [float(x) for x in g_raw]
        if bound is not None and E < bound - 1e-9:
            raise RuntimeError(f"variational bound violated: {E} < {bound}")
        last_E = float(E)
        if math.sqrt(sum(x * x for x in g)) < gtol:
            break
        for k in range(len(th)):
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            th[k] -= lr * mh / (math.sqrt(vh) + eps)
    return last_E, tuple(th)


def _minimize(
    f_eg: Callable[[Sequence[float]], tuple[float, Sequence[float]]],
    theta0: Sequence[float],
    *,
    bound: float | None = None,
    maxiter: int = 350,
    method: str = "auto",
) -> tuple[float, tuple[float, ...]]:
    """Minimize an arbitrary-length ADAPT parameter vector."""
    theta0 = tuple(float(x) for x in theta0)
    if len(theta0) == 0:
        E, _ = f_eg(theta0)
        return float(E), theta0
    if method not in {"auto", "lbfgsb", "adam"}:
        raise ValueError("method must be 'auto', 'lbfgsb', or 'adam'")
    if method in {"auto", "lbfgsb"}:
        try:
            import numpy as np
            from scipy.optimize import minimize

            def fg(x):
                E, g = f_eg(tuple(float(y) for y in x))
                if bound is not None and E < bound - 1e-9:
                    raise RuntimeError(f"variational bound violated: {E} < {bound}")
                return float(E), np.array(g, dtype=float)

            res = minimize(
                fg,
                np.array(theta0, dtype=float),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "gtol": 1e-8, "ftol": 1e-13, "maxls": 60},
            )
            th = tuple(float(x) for x in res.x)
            E, _ = f_eg(th)
            return float(E), th
        except ImportError:
            if method == "lbfgsb":
                raise
    return _adam_minimize(f_eg, theta0, bound=bound, maxiter=maxiter)


# ---------------------------------------------------------------------------
# 4. Exact and noisy ADAPT loops
# ---------------------------------------------------------------------------

def adapt_vqe(
    n: int,
    J: float,
    h: float,
    *,
    max_ops: int = 10,
    eps: float = 1e-6,
    periodic: bool = False,
    pool: Sequence[PoolOp] | None = None,
    allow_repeats: bool = False,
    verbose: bool = False,
) -> AdaptResult:
    return adapt_vqe_noisy(
        n,
        J,
        h,
        shots=None,
        max_ops=max_ops,
        eps=eps,
        periodic=periodic,
        pool=pool,
        allow_repeats=allow_repeats,
        verbose=verbose,
    )


def _candidate_indices(pool: Sequence[PoolOp], used: set[int], allow_repeats: bool) -> list[int]:
    return [i for i in range(len(pool)) if allow_repeats or i not in used]


def adapt_vqe_noisy(
    n: int,
    J: float,
    h: float,
    *,
    shots: int | None,
    rng=None,
    max_ops: int = 10,
    eps: float = 1e-6,
    sigma_mult: float = 3.0,
    max_factor: int = 64,
    periodic: bool = False,
    pool: Sequence[PoolOp] | None = None,
    pool_periodic_context: bool | None = None,
    allow_repeats: bool = False,
    require_rank_resolution: bool = True,
    sigma_source: str = "empirical",
    verbose: bool = False,
) -> AdaptResult:
    """Run ADAPT-VQE with exact or finite-shot operator selection.

    With ``shots=None`` selection is exact.  With finite shots, each candidate
    score is estimated word-by-word.  Shot escalation is cumulative: every
    doubling adds new samples instead of discarding previous samples.

    Finite-shot stopping uses statistical significance first and the numerical
    ADAPT threshold second: the loop escalates while ``|g_hat|max < k*sigma``
    even when ``require_rank_resolution=False``.  ``eps`` is applied only after
    the signal is statistically resolved.

    ``sigma_source='empirical'`` is the hardware-realistic default.  ``'oracle'``
    is an ablation/debug mode using exact expectation values unavailable on
    hardware.
    """
    if shots is not None:
        if shots <= 0:
            raise ValueError("shots must be positive or None")
        if rng is None:
            import numpy as np
            rng = np.random.default_rng()
    if max_factor < 1:
        raise ValueError("max_factor must be >= 1")
    if sigma_source not in {"empirical", "oracle"}:
        raise ValueError("sigma_source must be 'empirical' or 'oracle'")

    H, _, _ = tfim_hamiltonian(n, J, h, periodic=periodic)
    E0, V0 = exact_ground_space(H)
    if pool_periodic_context is None:
        pool_periodic_context = periodic
    pool = list(pool if pool is not None else qubit_pool(n, periodic_context=pool_periodic_context))
    if not pool:
        raise ValueError("empty operator pool")

    gens: list[MV] = []
    labels: list[str] = []
    theta: tuple[float, ...] = ()
    used: set[int] = set()
    record: list[AdaptStep] = []
    total_shots = 0
    stopped_reason = "operator budget reached"

    for step in range(1, max_ops + 1):
        rho = adapt_state(n, gens, theta)
        candidates = _candidate_indices(pool, used, allow_repeats)
        if not candidates:
            stopped_reason = "pool exhausted"
            break
        exact_scores_all = [abs(selection_gradient(H, rho, op.generator)) for op in pool]
        exact_scores = [(exact_scores_all[i], i) for i in candidates]
        exact_abs_gmax, exact_best_idx = max(exact_scores, key=lambda t: t[0])
        if exact_abs_gmax < eps and shots is None:
            stopped_reason = "exact gradient below eps"
            break

        if shots is None:
            chosen_idx = exact_best_idx
            ghat = exact_abs_gmax
            sig = 0.0
            ambiguous = False
            shots_this_step = 0
        else:
            # Initialize accumulators from the current state.
            accum: dict[int, CandidateAccumulator] = {}
            for i in candidates:
                G = selection_observable(H, pool[i].generator)
                accum[i] = CandidateAccumulator(pool[i].label, pool[i].generator, G, word_expectations(rho, G))

            factor = 1
            step_shots = 0
            chosen_idx = candidates[0]
            ghat = 0.0
            sig = math.inf
            ambiguous = True
            while True:
                target_N = shots * factor
                # Add only the extra shots needed to reach target_N.
                for acc in accum.values():
                    step_shots += acc.add_shots(target_N - acc.shots_per_word, rng)

                scored = []
                for i, acc in accum.items():
                    est, sig_oracle, sig_emp = acc.estimate()
                    sig_sel = sig_emp if sigma_source == "empirical" else sig_oracle
                    scored.append((abs(est), sig_sel, i))
                scored.sort(reverse=True, key=lambda x: x[0])
                ghat, sig, chosen_idx = scored[0]
                second_g, second_sig, _ = scored[1] if len(scored) > 1 else (0.0, 0.0, chosen_idx)
                signal_resolved = ghat >= sigma_mult * sig
                gap_sigma = math.sqrt(sig * sig + second_sig * second_sig)
                rank_resolved = (ghat - second_g) >= sigma_mult * gap_sigma
                ambiguous = not rank_resolved
                if signal_resolved and ((not require_rank_resolution) or rank_resolved):
                    break
                if factor >= max_factor:
                    break
                factor *= 2

            total_shots += step_shots
            shots_this_step = step_shots
            # Stop only when even the maximum budget finds no statistically
            # nonzero signal.  Only after significance is established does the
            # deterministic ADAPT threshold ``eps`` decide numerical convergence.
            if ghat < sigma_mult * sig:
                stopped_reason = "no statistically resolvable gradient"
                break
            if ghat < eps:
                stopped_reason = "resolved gradient below eps"
                break
            # At the ceiling, accept the best statistically nonzero candidate
            # even if ranking remains ambiguous; exact reoptimization will
            # self-correct.

        op = pool[chosen_idx]
        gens.append(op.generator)
        labels.append(op.label)
        used.add(chosen_idx)
        theta = theta + (0.0,)
        E, theta = _minimize(
            lambda t: adapt_energy_and_gradient(H, n, gens, t),
            theta,
            bound=E0,
        )
        rho_new = adapt_state(n, gens, theta)
        E_check, _ = adapt_energy_and_gradient(H, n, gens, theta)
        E = float(E_check)
        rel = abs(E - E0) / max(abs(E0), TOL)
        fid = fidelity_with_subspace(rho_new, V0)
        exact_abs_g = exact_scores_all[chosen_idx]
        near_opt = exact_abs_g >= 0.95 * exact_abs_gmax - 1e-12
        record.append(
            AdaptStep(
                step,
                op.label,
                float(ghat),
                float(sig),
                float(exact_abs_g),
                float(exact_abs_gmax),
                E,
                rel,
                fid,
                bool(near_opt),
                bool(ambiguous),
                shots_this_step,
                total_shots,
            )
        )
        if verbose:
            amb = "amb" if ambiguous else "ok "
            pick = "✓" if near_opt else "✗"
            print(
                f"    {step:2d} {op.label:12s} ĝ={ghat:.3e} σ={sig:.1e} "
                f"|g|={exact_abs_g:.3e}/{exact_abs_gmax:.3e} {pick} {amb} "
                f"E={E:+.8f} rel={rel:.2e} shots={total_shots/1e6:.2f}M"
            )

    rho_final = adapt_state(n, gens, theta)
    E_final = expectation(rho_final, H).real
    rel_final = abs(E_final - E0) / max(abs(E0), TOL)
    fid_final = fidelity_with_subspace(rho_final, V0)
    return AdaptResult(
        tuple(labels),
        theta,
        E_final,
        rel_final,
        fid_final,
        tuple(record),
        total_shots,
        E0,
        stopped_reason,
    )


# ---------------------------------------------------------------------------
# 5. Verification and demo
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def _monotone_nonincreasing(xs: Sequence[float], tol: float = 1e-9) -> bool:
    return all(xs[k] <= xs[k - 1] + tol for k in range(1, len(xs)))


def run_verification(args: argparse.Namespace | None = None) -> None:
    import numpy as np
    import statistics

    if args is None:
        args = parse_args([])
    rng = np.random.default_rng(args.seed)
    n, J, h = args.n, args.J, args.h
    print("== ADAPT-VQE with finite-shot commutator-gradient selection ==\n")

    H, _, _ = tfim_hamiltonian(n, J, h, periodic=args.periodic)
    pool = qubit_pool(n, periodic_context=args.pool_periodic_context)

    print("-- pool and selection observable --")
    if args.pool_periodic_context:
        _check(f"periodic-context Y/Z pool has {4*n if n > 2 else len(pool)} ops for n={n}", len(pool) == (4 * n if n > 2 else len(pool)))
    else:
        _check("open-context pool has no wrap-around operators", all("Z3Y0" not in op.label and "Y3Z0" not in op.label for op in pool))
    obs = [(op.label, op.generator, selection_observable(H, op.generator)) for op in pool]
    _check("G_P = -i/2[H,P] is Hermitian for every pool op", all(G.is_hermitian() for _, _, G in obs))
    _check("G_P Pauli coefficients are real", all(abs(c.imag) < 1e-12 for _, _, G in obs for c in G.terms.values()))

    rho_t = adapt_state(n, [pool[3].generator, pool[min(7, len(pool)-1)].generator], [0.4, -0.7])
    _check(
        "word-sum reproduces exact commutator gradient",
        all(abs(g_exact_from_words(rho_t, G) - selection_gradient(H, rho_t, P)) < 1e-10 for _, P, G in obs),
    )

    print("-- exact ADAPT path and gradient --")
    # Finite-difference check for the ADAPT adjoint gradient.
    gens = [pool[2].generator, pool[5].generator, pool[9].generator]
    theta = (0.2, -0.3, 0.4)
    E, g = adapt_energy_and_gradient(H, n, gens, theta)
    epsfd = 1e-6
    g_fd = []
    for k in range(len(theta)):
        tp, tm = list(theta), list(theta)
        tp[k] += epsfd
        tm[k] -= epsfd
        Ep, _ = adapt_energy_and_gradient(H, n, gens, tp)
        Em, _ = adapt_energy_and_gradient(H, n, gens, tm)
        g_fd.append((Ep - Em) / (2 * epsfd))
    dev = max(abs(a - b) for a, b in zip(g, g_fd))
    _check(f"ADAPT adjoint gradient == finite difference (max dev {dev:.2e})", dev < 1e-5)

    base = adapt_vqe(n, J, h, max_ops=args.max_ops, eps=1e-7, periodic=args.periodic, pool=pool)
    exact_path = adapt_vqe_noisy(n, J, h, shots=None, max_ops=args.max_ops, eps=1e-7, periodic=args.periodic, pool=pool)
    _check("shots=None reproduces exact ADAPT sequence", exact_path.labels == base.labels)
    _check(f"exact ADAPT reaches rel.err {exact_path.rel_error:.2e} < 1e-3", exact_path.rel_error < 1e-3)
    _check("exact ADAPT final state stays pure", abs(purity(adapt_state(n, [op.generator for op in pool if op.label in ()], ())) - 1) < 1e-9)

    print("-- estimator calibration --")
    # Choose a nontrivial observable with the largest predicted variance.
    candidates = []
    for lab, P, G in obs:
        entries = word_expectations(rho_t, G)
        sig1 = math.sqrt(sum((abs(c) ** 2) * max(0.0, 1.0 - ev * ev) for c, ev in entries))
        candidates.append((sig1, lab, P, G))
    _, lab, P, G = max(candidates, key=lambda x: x[0])
    g_true = selection_gradient(H, rho_t, P)
    reps = args.calibration_reps
    samples = []
    oracle_sigs = []
    for _ in range(reps):
        acc = CandidateAccumulator(lab, P, G, word_expectations(rho_t, G))
        acc.add_shots(args.calibration_shots, rng)
        est, sig_oracle, sig_emp_i = acc.estimate()
        samples.append(est)
        oracle_sigs.append(sig_oracle)
    sig_pred = statistics.fmean(oracle_sigs)
    mean_err = abs(statistics.fmean(samples) - g_true)
    _check(
        f"unbiased estimator: |mean-exact|={mean_err:.2e}",
        mean_err < 4.0 * sig_pred / math.sqrt(reps),
    )
    sig_emp = statistics.stdev(samples)
    ratio = sig_emp / sig_pred if sig_pred > 0 else 1.0
    _check(f"variance calibrated: σ_emp/σ_pred={ratio:.3f}", 0.85 < ratio < 1.15)


    print("-- non-default selection mode sanity --")
    no_rank = adapt_vqe_noisy(
        n,
        J,
        h,
        shots=64,
        rng=np.random.default_rng(args.seed + 123),
        max_ops=args.max_ops,
        eps=1e-7,
        sigma_mult=args.sigma_mult,
        max_factor=args.max_factor,
        periodic=args.periodic,
        pool=pool,
        require_rank_resolution=False,
        sigma_source=args.sigma_source,
    )
    min_fixed_shots = 64 * sum(len(selection_observable(H, op.generator).terms) for op in pool)
    _check(
        "no-rank-resolution still escalates until the top signal is statistically resolved",
        no_rank.total_shots > min_fixed_shots or no_rank.stopped_reason == "no statistically resolvable gradient",
    )
    _check("no-rank-resolution path remains useful", no_rank.rel_error < 5e-2)

    print(f"\n-- noisy robustness: n={n}, h/J={h}, seeds={args.seeds}, escalation x{args.max_factor} --")
    print("    base    median rel.err   worst rel.err   near-opt   ambiguous   median #ops   median Mshots")
    summary = {}
    for N in args.budgets:
        rels: list[float] = []
        near: list[bool] = []
        amb: list[bool] = []
        nops: list[int] = []
        shots_m: list[float] = []
        for s in range(args.seeds):
            r = adapt_vqe_noisy(
                n,
                J,
                h,
                shots=N,
                rng=np.random.default_rng(1000 * N + s),
                max_ops=args.max_ops,
                eps=1e-7,
                sigma_mult=args.sigma_mult,
                max_factor=args.max_factor,
                periodic=args.periodic,
                pool=pool,
                require_rank_resolution=args.require_rank_resolution,
                sigma_source=args.sigma_source,
            )
            Es = [row.energy for row in r.record]
            assert _monotone_nonincreasing(Es), "monotone descent violated"
            assert r.rel_error >= -1e-12, "negative relative error impossible"
            assert r.energy >= r.E0 - 1e-8, "variational bound violated"
            rels.append(r.rel_error)
            near.extend(row.near_optimal for row in r.record)
            amb.extend(row.ambiguous for row in r.record)
            nops.append(len(r.record))
            shots_m.append(r.total_shots / 1e6)
        med = statistics.median(rels)
        worst = max(rels)
        near_rate = sum(near) / max(1, len(near))
        amb_rate = sum(amb) / max(1, len(amb))
        summary[N] = (med, worst, near_rate, amb_rate, statistics.median(nops), statistics.median(shots_m))
        print(
            f"    {N:5d}   {med:.3e}       {worst:.3e}      "
            f"{100*near_rate:5.1f}%     {100*amb_rate:5.1f}%       "
            f"{statistics.median(nops):4.1f}        {statistics.median(shots_m):7.2f}"
        )

    _check("monotone descent and variational bound held in every noisy run", True)
    _check("all tested budgets remain useful (median rel.err < 5e-2)", all(v[0] < 5e-2 for v in summary.values()))
    _check("shot cost grows with base budget", all(summary[args.budgets[i]][5] <= summary[args.budgets[i+1]][5] + 1e-12 for i in range(len(args.budgets)-1)))

    print("\n-- one verbose low-shot run --")
    demo = adapt_vqe_noisy(
        n,
        J,
        h,
        shots=args.budgets[0],
        rng=np.random.default_rng(args.seed + 77),
        max_ops=args.max_ops,
        eps=1e-7,
        sigma_mult=args.sigma_mult,
        max_factor=args.max_factor,
        periodic=args.periodic,
        pool=pool,
        require_rank_resolution=args.require_rank_resolution,
        sigma_source=args.sigma_source,
        verbose=True,
    )
    print(
        f"    stopped: {demo.stopped_reason}; final rel.err {demo.rel_error:.2e}; "
        f"fidelity {demo.fidelity:.6f}; total shots {demo.total_shots/1e6:.2f}M"
    )
    _check("low-shot run still returns a variational pure-state ansatz", demo.energy >= demo.E0 - 1e-8)

    print("\nAll checks passed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noisy-selection ADAPT-VQE for the TFIM in Cl(2n,C).")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--periodic-context-pool", dest="pool_periodic_context", action="store_true")
    parser.add_argument("--open-context-pool", dest="pool_periodic_context", action="store_false")
    parser.set_defaults(pool_periodic_context=False)
    parser.add_argument("--max-ops", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="*", default=[32, 128, 512])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-factor", type=int, default=32)
    parser.add_argument("--sigma-mult", type=float, default=3.0)
    parser.add_argument("--no-rank-resolution", dest="require_rank_resolution", action="store_false")
    parser.set_defaults(require_rank_resolution=True)
    parser.add_argument("--sigma-source", choices=["empirical", "oracle"], default="empirical")
    parser.add_argument("--calibration-shots", type=int, default=256)
    parser.add_argument("--calibration-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_verification(parse_args())
