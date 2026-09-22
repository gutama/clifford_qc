"""Pilot probe for Phase 16B: is a real-time Krylov pencil shot-survivable?

This is a *probe*, not a record producer: it writes no committed artifact and
its numbers gate nothing.  It exists to size the Phase 16B feasibility
experiment before that experiment is built, and to make the two structural
facts in `PHASE16B_FEASIBILITY.md` reproducible.

Fact 1 -- the real-time pencil is Toeplitz.  For ``A_k = e^{-iH k dt}`` and
Hermitian ``H``,

    S_ij = <psi|e^{+iH i dt} e^{-iH j dt}|psi>   = c(j - i),
    H_ij = <psi|e^{+iH i dt} H e^{-iH j dt}|psi> = d(j - i),

because ``H`` commutes with its own propagator.  The whole ``m x m`` pencil is
therefore ``2m - 1`` values of each of two scalar functions of one time
argument -- not ``m^2`` operator matrix elements, and not a word universe that
grows with ``|H|``.  This is the one structural reason Phase 16B is worth
pricing at all, so the probe checks it numerically rather than asserting it.

Fact 2 -- conditioning and accuracy grow together.  The same basis that reaches
chemical accuracy is the one whose overlap spectrum spans the decades that shot
noise erases.  The probe sweeps a per-scalar noise amplitude ``eps`` and reports
the resolvable rank and the resulting ground-energy error, which is the
quantity the feasibility experiment has to decide.

Fact 3 -- the Hamiltonian need not be measured at all.  With ``S0_ij = c(j-i)``
and ``S1_ij = c(j-i+1)``, the generalized eigenvalues of ``S1 v = lam S0 v`` are
``lam = e^{-iE dt}``, so ``d`` disappears and the energies are phases.  This
halves the estimand count and is better in the median, but it is non-Hermitian
and unbounded below, so the probe reports its p90 beside the median: a noisy
replica can return the wrong root rather than an inaccurate one.

The noise model here is deliberately crude -- i.i.d. complex Gaussian of width
``eps`` on each Toeplitz scalar, Hermiticity restored, hard truncation of the
normalized overlap spectrum at ``eps``.  It is a stand-in for a shot model, not
a shot model: no estimator variance, no Hadamard-test overhead, no ridge
regularization.  Those are arms of the real experiment (`PHASE16B_FEASIBILITY.md`
Section 6), and a better regularizer can only improve these rows.

    python benchmarks/probe_realtime_krylov.py
    python benchmarks/probe_realtime_krylov.py --replicas 200 --seed 7
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import expm

from clifford_qc.dense_reference import to_matrix
from clifford_qc.models import tfim

# Predeclared so a rerun is the same experiment; see the spec's Section 6.
NOISE_LEVELS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
BASIS_SIZES = (4, 6, 8, 10)
CHEMICAL_ACCURACY_HA = 1.6e-3


def dense_model(n: int = 4):
    """TFIM at criticality: the hardest small instance for a product reference."""
    model = tfim(n, J=1.0, h=1.0)
    H = to_matrix(model.hamiltonian.to_mv())
    return 0.5 * (H + H.conj().T)


def toeplitz_scalars(H: np.ndarray, psi: np.ndarray, m: int, dt: float):
    """``c(k)`` and ``d(k)`` for ``k = -(m-1) .. (m-1)`` -- the entire pencil."""
    c: dict[int, complex] = {}
    d: dict[int, complex] = {}
    for k in range(-(m - 1), m):
        evolved = expm(-1j * H * (k * dt)) @ psi
        c[k] = complex(psi.conj() @ evolved)
        d[k] = complex(psi.conj() @ (H @ evolved))
    return c, d


def assemble(c, d, m: int):
    S = np.array([[c[j - i] for j in range(m)] for i in range(m)])
    Hm = np.array([[d[j - i] for j in range(m)] for i in range(m)])
    return 0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T)


def solve(S: np.ndarray, Hm: np.ndarray, floor: float):
    """Hard-truncated Rayleigh-Ritz in the whitened metric; returns (E, rank)."""
    values, vectors = np.linalg.eigh(S)
    keep = values > max(floor, 1e-14) * max(values[-1], 1e-300)
    if not keep.any():
        return float("nan"), 0
    X = vectors[:, keep] / np.sqrt(values[keep])
    return float(np.linalg.eigvalsh(X.conj().T @ Hm @ X)[0]), int(keep.sum())


def solve_unitary(c, m: int, dt: float, floor: float):
    """Prony pencil ``S1 v = lam S0 v`` with ``lam = e^{-i E dt}`` -- needs only ``c``.

    No Hamiltonian is measured: the energies are the phases of the generalized
    eigenvalues.  The price is that this is a non-Hermitian problem with no
    variational floor, so a noisy replica can return an arbitrarily wrong root
    rather than a merely inaccurate one.  That is why the caller reports p90
    beside the median.
    """
    S0 = np.array([[c[j - i] for j in range(m)] for i in range(m)])
    S1 = np.array([[c[j - i + 1] for j in range(m)] for i in range(m)])
    S0 = 0.5 * (S0 + S0.conj().T)
    values, vectors = np.linalg.eigh(S0)
    keep = values > max(floor, 1e-13) * max(values[-1], 1e-300)
    if not keep.any():
        return float("nan"), 0
    X = vectors[:, keep] / np.sqrt(values[keep])
    lam = np.linalg.eigvals(X.conj().T @ S1 @ X)
    lam = lam[np.abs(lam) > 1e-12]
    if lam.size == 0:
        return float("nan"), 0
    return float(np.min(np.angle(lam) / (-dt))), int(keep.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--qubits", type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    H = dense_model(args.qubits)
    dim = H.shape[0]
    values, vectors = np.linalg.eigh(H)
    exact = float(values[0])
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0

    weights = np.abs(vectors.conj().T @ psi) ** 2
    dt = float(np.pi / (values[-1] - values[0]))
    print(f"n={args.qubits}  spectrum [{values[0]:.6f}, {values[-1]:.6f}]  "
          f"gap {values[1] - values[0]:.6f}  dt {dt:.6f}")
    print(f"reference spectral support: {int((weights > 1e-6).sum())} of {dim} "
          f"eigenstates carry weight > 1e-6")

    print("\nFact 1 -- Toeplitz residual of the assembled pencil (exact arithmetic):")
    for m in BASIS_SIZES:
        c, d = toeplitz_scalars(H, psi, m, dt)
        S, Hm = assemble(c, d, m)
        res_s = max(abs(S[i, j] - S[i + 1, j + 1])
                    for i in range(m - 1) for j in range(m - 1))
        res_h = max(abs(Hm[i, j] - Hm[i + 1, j + 1])
                    for i in range(m - 1) for j in range(m - 1))
        cond = float(np.linalg.eigvalsh(S)[-1] / max(np.linalg.eigvalsh(S)[0], 1e-300))
        energy, rank = solve(S, Hm, floor=1e-13)
        print(f"  m={m:2d}  |S_ij - S_i+1,j+1| <= {res_s:.1e}   "
              f"|H_ij - H_i+1,j+1| <= {res_h:.1e}   "
              f"cond(S) {cond:.2e}  rank {rank}  err {abs(energy - exact):.3e}")

    print(f"\nFact 2 -- ground-energy error under per-scalar noise "
          f"({args.replicas} replicas; chemical accuracy = {CHEMICAL_ACCURACY_HA:.1e} Ha):")
    print(f"  {'eps':>8} {'m':>3} {'median err':>12} {'p90 err':>12} {'rank':>5}  verdict")
    for eps in NOISE_LEVELS:
        for m in BASIS_SIZES:
            c, d = toeplitz_scalars(H, psi, m, dt)
            errors: list[float] = []
            ranks: list[int] = []
            replicas = 1 if eps == 0.0 else args.replicas
            for _ in range(replicas):
                noisy_c = {k: v + (rng.normal(0, eps) + 1j * rng.normal(0, eps)
                                   if eps else 0.0) for k, v in c.items()}
                noisy_d = {k: v + (rng.normal(0, eps) + 1j * rng.normal(0, eps)
                                   if eps else 0.0) for k, v in d.items()}
                noisy_c[0] = complex(noisy_c[0].real)
                noisy_d[0] = complex(noisy_d[0].real)
                S, Hm = assemble(noisy_c, noisy_d, m)
                energy, rank = solve(S, Hm, floor=max(eps, 1e-13))
                errors.append(abs(energy - exact) if np.isfinite(energy) else np.nan)
                ranks.append(rank)
            median = float(np.nanmedian(errors))
            verdict = "chemical" if median <= CHEMICAL_ACCURACY_HA else ""
            print(f"  {eps:>8.0e} {m:>3} {median:>12.3e} "
                  f"{float(np.nanpercentile(errors, 90)):>12.3e} "
                  f"{float(np.median(ranks)):>5.0f}  {verdict}")
        print()

    print(f"Fact 3 -- unitary (Prony) pencil, `c` only, no Hamiltonian measured "
          f"({args.replicas} replicas):")
    print(f"  {'eps':>8} {'m':>3} {'median err':>12} {'p90 err':>12} {'rank':>5}")
    for eps in NOISE_LEVELS:
        for m in BASIS_SIZES:
            base = {k: complex(psi.conj() @ (expm(-1j * H * (k * dt)) @ psi))
                    for k in range(-m, m + 1)}
            errors: list[float] = []
            ranks: list[int] = []
            replicas = 1 if eps == 0.0 else args.replicas
            for _ in range(replicas):
                noisy = {k: v + (rng.normal(0, eps) + 1j * rng.normal(0, eps)
                                 if eps else 0.0) for k, v in base.items()}
                noisy[0] = complex(noisy[0].real)
                for k in range(1, m + 1):
                    noisy[-k] = np.conj(noisy[k])
                energy, rank = solve_unitary(noisy, m, dt, floor=max(eps, 1e-13))
                errors.append(abs(energy - exact) if np.isfinite(energy) else np.nan)
                ranks.append(rank)
            print(f"  {eps:>8.0e} {m:>3} {float(np.nanmedian(errors)):>12.3e} "
                  f"{float(np.nanpercentile(errors, 90)):>12.3e} "
                  f"{float(np.median(ranks)):>5.0f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
