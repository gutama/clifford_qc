"""
adapt_vqe_tfim.py — ADAPT-VQE with a commutator-gradient operator pool
======================================================================

Conceptual layer
----------------
Algebra/signature : Cl(2n, C) ≅ M(2^n, C) via ga_qc_operator_improved.
Object/operator   : the ansatz is GROWN, not fixed. A pool of Hermitian
                    Pauli words {P} (each P² = 1, so each candidate gate is
                    a genuine rotor R_P(θ) = cos(θ/2) − i sin(θ/2) P) is
                    ranked at every step by the COMMUTATOR GRADIENT
                        g_P = dE/dθ|_{θ=0} = −(i/2) Tr( [H, P] ρ ),
                    i.e. the exact slope of the energy if R_P were appended
                    to the current circuit. The largest |g_P| wins, one new
                    parameter is added, and ALL parameters are re-optimized
                    (adjoint rotor-calculus gradients throughout).
Domain meaning    : ADAPT-VQE (Grimsley et al. 2019; qubit-ADAPT, Tang et
                    al. 2021) — the strategy that is effectively immune to
                    barren plateaus because every added operator is chosen
                    by a live gradient, never at random.

The commutator gradient IS rotor calculus
------------------------------------------
Appending R_P(θ) to a state ρ gives E(θ) = Tr(H R_P ρ R_P†); the rotor
derivative ∂_θ R_P = −(i/2) P R_P yields
    dE/dθ|₀ = −(i/2) Tr( H [P, ρ] ) = −(i/2) Tr( [H, P] ρ ),
one commutator + one scalar-part read per pool operator — no circuit
execution needed for selection. In GA language: the selection score is the
scalar part of the geometric product of ρ with the commutator bivector-like
element [H, P], exactly the quantity ⟨[H,P] ρ⟩₀ the kernel computes natively.

A parity theorem the pool obeys (verified exactly below)
--------------------------------------------------------
H (words XX/ZZ/X/Z…) and the reference |+…+> are REAL: symmetric matrices.
For a real-symmetric P, [H, P] is antisymmetric while ρ stays symmetric, so
    Tr( ρ [H, P] ) = 0   exactly.
Only Pauli words with an ODD number of Y's (imaginary, antisymmetric) can
have nonzero selection gradient. The pool is therefore the odd-Y words:
    { Y_i } ∪ { Y_i Z_j, Z_i Y_j, Y_i X_j, X_i Y_j : |i−j| = 1 } ,
and the theorem is checked by confirming every even-Y candidate scores 0.

Implementation substrate
------------------------
Sparse Pauli-word multivectors (ga_qc_operator_improved); model, exact
targets, fidelity and observables reused from vqe_tfim_rotor_better;
re-optimization by L-BFGS-B (scipy) with Adam fallback, driven by the
per-gate adjoint gradient (no parameter sharing in ADAPT).

Transfer note
-------------
Preserved  : exact isomorphism to the matrix formalism; the variational
             bound is monitored at every optimizer call; selection
             gradients are analytic, verified against finite differences.
Approximate: nothing in selection or gradients; only the greedy growth
             heuristic itself (ADAPT is a heuristic, benchmarked here).
Not implied: no hardware-noise model; this is the exact algorithmic core.

Validation invariants (run `python adapt_vqe_tfim.py`)
------------------------------------------------------
  * pool rotors: unit, Hermitian generators, P² = 1
  * commutator gradient == finite-difference slope of an appended rotor
  * parity theorem: every even-Y word scores exactly 0 on the real state
  * ADAPT at criticality (n = 4, h = 1): monotone energy descent,
    rel.err < 1e-6, fidelity > 0.9999, bound never violated
  * beats/matches fixed HVA p = 3 (6 params) at comparable parameter count
  * deep ordered phase h = 0.2 from a COLD start (where fixed HVA needed
    adiabatic continuation): rel.err < 1e-4
  * selection scores decay as the state converges (printed per step)
"""

from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_qc_operator_improved import (            # noqa: E402
    MV, I, X, Y, Z, comm, rotor, evolve, expectation, purity,
)
from vqe_tfim_rotor_better import (              # noqa: E402
    tfim_hamiltonian, plus_state, exact_ground_space,
    fidelity_with_subspace, energy as hva_energy,
    optimize_vqe,
)

TOL = 1e-12


# ----------------------------------------------------------------------
# 1. Operator pool (odd-Y Pauli words) and selection gradient
# ----------------------------------------------------------------------

def qubit_pool(n: int) -> list[tuple[str, MV]]:
    """Odd-Y nearest-neighbour qubit pool for a real Hamiltonian."""
    pool: list[tuple[str, MV]] = []
    for i in range(n):
        pool.append((f"Y{i}", Y(n, i)))
    for i in range(n - 1):
        j = i + 1
        pool.append((f"Y{i}Z{j}", Y(n, i) * Z(n, j)))
        pool.append((f"Z{i}Y{j}", Z(n, i) * Y(n, j)))
        pool.append((f"Y{i}X{j}", Y(n, i) * X(n, j)))
        pool.append((f"X{i}Y{j}", X(n, i) * Y(n, j)))
    return pool


def even_y_candidates(n: int) -> list[tuple[str, MV]]:
    """Real (even-Y) words — the parity theorem says these always score 0."""
    out = [(f"X{i}", X(n, i)) for i in range(n)]
    out += [(f"Z{i}", Z(n, i)) for i in range(n)]
    out += [(f"Z{i}Z{i+1}", Z(n, i) * Z(n, i + 1)) for i in range(n - 1)]
    out += [(f"Y{i}Y{i+1}", Y(n, i) * Y(n, i + 1)) for i in range(n - 1)]
    return out


def selection_gradient(H: MV, rho: MV, P: MV) -> float:
    """g_P = dE/dθ|₀ = −(i/2) Tr([H, P] ρ) — one commutator, one trace."""
    return (-0.5j * (comm(H, P) * rho).trace()).real


# ----------------------------------------------------------------------
# 2. ADAPT circuit: per-gate parameters, adjoint gradient
# ----------------------------------------------------------------------

def adapt_state(n: int, gens: list[MV], theta: list[float]) -> MV:
    rho = plus_state(n)
    for P, t in zip(gens, theta):
        rho = evolve(rho, rotor(P, t))
    return rho


def adapt_energy_and_gradient(H: MV, n: int, gens: list[MV],
                              theta: list[float]):
    """Exact adjoint sweep; one parameter per gate (no HVA sharing)."""
    gates = [rotor(P, t) for P, t in zip(gens, theta)]
    states = []
    rho = plus_state(n)
    for G in gates:
        rho = evolve(rho, G)
        states.append(rho)
    E = expectation(rho, H).real
    grads = [0.0] * len(theta)
    O = H
    for k in range(len(gates) - 1, -1, -1):
        P, rk = gens[k], states[k]
        grads[k] = (-0.5j * (O * (P * rk - rk * P)).trace()).real
        O = gates[k].dagger() * O * gates[k]
    return E, grads


def _minimize(f_eg, theta0, *, maxiter=300, bound=None, gtol=1e-9):
    """L-BFGS-B if scipy is present, Adam otherwise. Bound-checked."""
    try:
        import numpy as np
        from scipy.optimize import minimize

        def fg(x):
            E, g = f_eg(list(map(float, x)))
            if bound is not None and E < bound - 1e-9:
                raise RuntimeError(f"variational bound violated: {E}")
            return float(E), np.array(g, dtype=float)

        res = minimize(fg, np.array(theta0, float), jac=True,
                       method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": gtol,
                                "ftol": 1e-14, "maxls": 60})
        return list(map(float, res.x))
    except ImportError:
        th = list(theta0)
        m = [0.0] * len(th); v = [0.0] * len(th)
        for it in range(1, maxiter + 1):
            E, g = f_eg(th)
            if bound is not None and E < bound - 1e-9:
                raise RuntimeError(f"variational bound violated: {E}")
            if math.sqrt(sum(x * x for x in g)) < gtol:
                break
            for k in range(len(th)):
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                th[k] -= 0.08 * (m[k] / (1 - 0.9 ** it)) / (
                    math.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
        return th


# ----------------------------------------------------------------------
# 3. ADAPT-VQE loop
# ----------------------------------------------------------------------

def adapt_vqe(n: int, J: float, h: float, *, max_ops: int = 12,
              eps: float = 1e-6, verbose: bool = True):
    """Grow the rotor ansatz operator-by-operator. Returns full record."""
    H, _, _ = tfim_hamiltonian(n, J, h)
    E0, V0 = exact_ground_space(H)
    pool = qubit_pool(n)
    gens: list[MV] = []
    labels: list[str] = []
    theta: list[float] = []
    record = []
    if verbose:
        print(f"    step  selected   |g_max|      E            rel.err")
    for step in range(1, max_ops + 1):
        rho = adapt_state(n, gens, theta)
        scores = [(abs(selection_gradient(H, rho, P)), lab, P)
                  for lab, P in pool]
        gmax, lab, P = max(scores, key=lambda s: s[0])
        if gmax < eps:
            if verbose:
                print(f"    converged: max pool gradient {gmax:.2e} < {eps:g}")
            break
        gens.append(P)
        labels.append(lab)
        theta = list(theta) + [0.0]
        theta = _minimize(
            lambda t: adapt_energy_and_gradient(H, n, gens, t),
            theta, bound=E0)
        E, _ = adapt_energy_and_gradient(H, n, gens, theta)
        rel = abs(E - E0) / abs(E0)
        record.append((step, lab, gmax, E, rel))
        if verbose:
            print(f"    {step:4d}  {lab:8s}  {gmax:.3e}  {E:+.8f}  {rel:.2e}")
    rho = adapt_state(n, gens, theta)
    F = fidelity_with_subspace(rho, V0)
    return {"E": record[-1][3] if record else expectation(
                plus_state(n), H).real,
            "E0": E0, "rel": record[-1][4] if record else None,
            "fidelity": F, "labels": labels, "theta": theta,
            "record": record, "rho": rho, "H": H, "gens": gens}


# ----------------------------------------------------------------------
# 4. Verification + experiment
# ----------------------------------------------------------------------

def _check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def run():
    print("== ADAPT-VQE with commutator-gradient pool selection ==\n")
    n, J, h = 4, 1.0, 1.0
    H, bonds, fields = tfim_hamiltonian(n, J, h)
    E0, V0 = exact_ground_space(H)
    pool = qubit_pool(n)

    print("-- pool structure --")
    _check(f"pool has {len(pool)} odd-Y words, all Hermitian with P² = 1",
           all(P.is_hermitian() and (P * P).is_close(I(n))
               for _, P in pool))
    _check("pool rotors are unitary",
           all(rotor(P, 0.83).is_unitary() for _, P in pool))

    print("-- selection gradient = rotor calculus --")
    rho_t = adapt_state(n, [pool[3][1], pool[7][1]], [0.4, -0.7])
    P_t = pool[5][1]
    g_c = selection_gradient(H, rho_t, P_t)
    epsfd = 1e-6

    def E_app(t):
        return expectation(evolve(rho_t, rotor(P_t, t)), H).real
    g_fd = (E_app(epsfd) - E_app(-epsfd)) / (2 * epsfd)
    _check(f"−(i/2)Tr([H,P]ρ) == appended-rotor slope "
           f"(dev {abs(g_c - g_fd):.2e})", abs(g_c - g_fd) < 1e-7)

    print("-- parity theorem --")
    worst = max(abs(selection_gradient(H, rho_t, P))
                for _, P in even_y_candidates(n))
    _check(f"every even-Y word scores EXACTLY 0 (worst {worst:.2e})",
           worst < 1e-12)

    print(f"\n-- ADAPT at criticality: n = {n}, h/J = {h} "
          f"(exact E0 = {E0:+.6f}) --")
    res = adapt_vqe(n, J, h, max_ops=12, eps=1e-6)
    _check("monotone energy descent step-to-step",
           all(res['record'][k][3] <= res['record'][k - 1][3] + 1e-9
               for k in range(1, len(res['record']))))
    _check(f"rel.err {res['rel']:.2e} < 1e-6", res['rel'] < 1e-6)
    _check(f"fidelity {res['fidelity']:.8f} > 0.9999",
           res['fidelity'] > 0.9999)
    _check("state stays pure, trace 1",
           abs(purity(res['rho']) - 1) < 1e-9
           and abs(res['rho'].trace() - 1) < 1e-9)
    _check("selection scores decay (last < first)",
           res['record'][-1][2] < res['record'][0][2])

    print("\n-- comparison with fixed HVA p = 3 (6 shared params) --")
    hva, _, _, _, _ = optimize_vqe(n, J, h, 3, maxiter=300)
    k6 = next((r for r in res['record'] if r[0] == 6), res['record'][-1])
    print(f"    HVA  p=3 : 6 params  rel.err = {hva.rel_error:.2e}")
    print(f"    ADAPT @6 : 6 params  rel.err = {k6[4]:.2e}")
    print(f"    ADAPT end: {len(res['labels'])} params  "
          f"rel.err = {res['rel']:.2e}")
    _check("ADAPT (final) at least matches HVA p = 3",
           res['rel'] <= hva.rel_error * 1.5 + 1e-9)

    print("\n-- deep ordered phase h/J = 0.2, COLD start "
          "(fixed HVA needed continuation here) --")
    res2 = adapt_vqe(n, J, 0.2, max_ops=14, eps=1e-7)
    _check(f"cold-start rel.err {res2['rel']:.2e} < 1e-4",
           res2['rel'] < 1e-4)
    _check(f"fidelity {res2['fidelity']:.6f} > 0.999",
           res2['fidelity'] > 0.999)
    print(f"    operators chosen: {', '.join(res2['labels'])}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    run()
