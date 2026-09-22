"""Pilot for Phase 16B; this does not produce a feasibility verdict or record.

Check the exact real-time Toeplitz pencil against independently propagated
statevectors, then compare Hermitian and unitary pencils under heuristic noise.
The unitary pencil gives exact eigenphases only on an invariant subspace; with
a truncated basis its roots are approximations. Phase decoding requires a
declared energy branch, and neither noisy pencil has a variational guarantee.

Noise has standard deviation ``eps`` in each real/imaginary component of each
independent positive-lag scalar. Negative lags reuse the conjugate, ``c(0)`` is
known exactly, and both arms use the same noisy ``c`` samples. ``d`` noise is in
the model's energy units, not a normalized Hadamard-test observable. No shot
claim follows from this pilot. Additional regularization may help or hurt.

    python benchmarks/probe_realtime_krylov.py
    python benchmarks/probe_realtime_krylov.py --replicas 200 --seed 7
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import expm

from clifford_qc.dense_reference import to_matrix
from clifford_qc.models import tfim

NOISE_LEVELS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
BASIS_SIZES = (4, 6, 8, 10)
TARGET_ERROR = 1.6e-3  # TFIM energy units (J=1); not Hartree.


def dense_model(n: int = 4):
    """Open-chain TFIM at J=h=1; the pilot sets its own |0...0> reference."""
    model = tfim(n, J=1.0, h=1.0)
    H = to_matrix(model.hamiltonian.to_mv())
    return 0.5 * (H + H.conj().T)


def toeplitz_scalars(H: np.ndarray, psi: np.ndarray, m: int, dt: float):
    """Exact c(k), d(k), |k| < m; negative lags share conjugates."""
    c: dict[int, complex] = {}
    d: dict[int, complex] = {}
    for k in range(m):
        evolved = expm(-1j * H * (k * dt)) @ psi
        c[k] = complex(np.vdot(psi, evolved))
        d[k] = complex(np.vdot(psi, H @ evolved))
        if k:
            c[-k], d[-k] = c[k].conjugate(), d[k].conjugate()
    c[0], d[0] = complex(c[0].real), complex(d[0].real)
    return c, d


def assemble(c, d, m: int):
    S = np.array([[c[j - i] for j in range(m)] for i in range(m)])
    Hm = np.array([[d[j - i] for j in range(m)] for i in range(m)])
    return 0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T)


def direct_pencil(H: np.ndarray, psi: np.ndarray, m: int, dt: float):
    """Independent statevector construction, with no Toeplitz assumption."""
    states = np.column_stack([expm(-1j * H * (k * dt)) @ psi for k in range(m)])
    return states.conj().T @ states, states.conj().T @ H @ states


def sample_scalars(base, max_lag: int, eps: float, rng, *, exact_zero: bool):
    """Draw each independent component once, retaining its declared variance."""
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    result = {0: complex(base[0].real + (0.0 if exact_zero else rng.normal(0, eps)))}
    for k in range(1, max_lag + 1):
        result[k] = base[k] + rng.normal(0, eps) + 1j * rng.normal(0, eps)
        result[-k] = result[k].conjugate()
    return result


def _whitener(S: np.ndarray, floor: float):
    """Pilot relative cutoff; not a calibrated per-mode statistical floor."""
    values, vectors = np.linalg.eigh(S)
    keep = values > max(floor, 1e-13) * max(values[-1], 1e-300)
    return vectors[:, keep] / np.sqrt(values[keep]), int(keep.sum())


def solve(S: np.ndarray, Hm: np.ndarray, floor: float):
    """Hard-truncated Hermitian pencil; returns (energy, retained rank)."""
    X, rank = _whitener(S, floor)
    if not rank:
        return float("nan"), 0
    reduced = X.conj().T @ Hm @ X
    reduced = 0.5 * (reduced + reduced.conj().T)
    return float(np.linalg.eigvalsh(reduced)[0]), rank


def solve_unitary(c, m: int, dt: float, floor: float, *, energy_shift: float = 0.0):
    """Decode roots on the branch centred at energy_shift with width 2*pi/dt.

    Rephase c(k) to evolve H-energy_shift*I, then restore the shift. The caller
    must choose dt and the shift so the spectral enclosure lies strictly inside
    this branch. A compressed unitary is not generally unitary or variational.
    """
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(energy_shift):
        raise ValueError("dt must be positive and finite, and energy_shift finite")
    shifted = {k: value * np.exp(1j * energy_shift * k * dt) for k, value in c.items()}
    S0 = np.array([[shifted[j - i] for j in range(m)] for i in range(m)])
    S1 = np.array([[shifted[j - i + 1] for j in range(m)] for i in range(m)])
    S0 = 0.5 * (S0 + S0.conj().T)
    X, rank = _whitener(S0, floor)
    if not rank:
        return float("nan"), 0
    lam = np.linalg.eigvals(X.conj().T @ S1 @ X)
    # A vanished or nonfinite root is a failed solve, not a removable replica.
    if not np.all(np.isfinite(lam)) or np.any(np.abs(lam) <= 1e-12):
        return float("nan"), rank
    return float(energy_shift + np.min(-np.angle(lam) / dt)), rank


def spectral_support(values, vectors, psi, *, weight_floor: float = 1e-6):
    """Count supported distinct energies, invariant under degenerate rotations.

    This thresholded diagnostic need not bound the exact rank: arbitrarily
    small nonzero weights also contribute in exact arithmetic.
    """
    weights = np.abs(vectors.conj().T @ psi) ** 2
    grouped: list[float] = []
    anchor = None
    for value, weight in zip(values, weights):
        if anchor is None or not np.isclose(value, anchor, atol=1e-10, rtol=1e-12):
            grouped.append(float(weight))
            anchor = value
        else:
            grouped[-1] += float(weight)
    return sum(weight > weight_floor for weight in grouped)


def error_summary(energies, exact: float):
    """All replicas count: failed solves have infinite error, including in p90."""
    energies = np.asarray(energies, dtype=float)
    if energies.size == 0:
        raise ValueError("at least one replica is required")
    errors = np.where(np.isfinite(energies), np.abs(energies - exact), np.inf)
    return {
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.9, method="higher")),
        "failures": int((~np.isfinite(energies)).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--qubits", type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    if args.replicas <= 0 or args.seed < 0:
        parser.error("replicas must be positive and seed must be nonnegative")

    H = dense_model(args.qubits)
    values, vectors = np.linalg.eigh(H)
    exact = float(values[0])
    psi = np.zeros(H.shape[0], dtype=complex)
    psi[0] = 1.0
    dt = float(np.pi / (values[-1] - values[0]))
    energy_shift = float((values[-1] + values[0]) / 2)
    # Include c(m), needed by the shifted unitary pencil, and reuse propagation.
    c, d = toeplitz_scalars(H, psi, max(BASIS_SIZES) + 1, dt)
    print(f"n={args.qubits}  spectrum [{values[0]:.6f}, {values[-1]:.6f}]  "
          f"gap {values[1] - values[0]:.6f}  dt {dt:.6f}  shift {energy_shift:.6f}")
    print(f"reference spectral support: {spectral_support(values, vectors, psi)} "
          "distinct energies carry total weight > 1e-6 (degeneracy tolerance 1e-10)")

    print("\nExact pencil vs independent statevector construction:")
    for m in BASIS_SIZES:
        S, Hm = assemble(c, d, m)
        direct_s, direct_h = direct_pencil(H, psi, m, dt)
        res_s, res_h = np.max(np.abs(S - direct_s)), np.max(np.abs(Hm - direct_h))
        if max(res_s, res_h / max(float(np.linalg.norm(H, 2)), 1.0)) > 1e-10:
            raise RuntimeError("Toeplitz pencil disagrees with direct propagation")
        spectrum = np.linalg.eigvalsh(S)
        cond = float(spectrum[-1] / spectrum[0]) if spectrum[0] > 0 else float("inf")
        energy, rank = solve(S, Hm, floor=1e-13)
        print(f"  m={m:2d}  residual(S) {res_s:.1e}  residual(H) {res_h:.1e}  "
              f"cond(S) {cond:.2e}  rank {rank}  err {abs(energy - exact):.3e}  "
              f"real components H/U={4*m-3}/{2*m}")

    print(f"\nPaired heuristic noise ({args.replicas} replicas; seed={args.seed}; "
          f"target={TARGET_ERROR:.1e} in TFIM J=1 energy units):")
    print(f"  {'eps':>8} {'m':>3} {'arm':>9} {'median err':>12} {'p90 err':>12} "
          f"{'rank':>5} {'failed':>6}  target")
    for noise_index, eps in enumerate(NOISE_LEVELS):
        for m in BASIS_SIZES:
            streams = np.random.SeedSequence([args.seed, noise_index, m]).spawn(2)
            c_rng, d_rng = (np.random.default_rng(stream) for stream in streams)
            energies = {"hermitian": [], "unitary": []}
            ranks = {"hermitian": [], "unitary": []}
            for _ in range(1 if eps == 0 else args.replicas):
                noisy_c = sample_scalars(c, m, eps, c_rng, exact_zero=True)
                noisy_d = sample_scalars(d, m - 1, eps, d_rng, exact_zero=False)
                S, Hm = assemble(noisy_c, noisy_d, m)
                for arm in energies:
                    try:
                        energy, rank = (solve(S, Hm, eps) if arm == "hermitian" else
                                        solve_unitary(noisy_c, m, dt, eps,
                                                      energy_shift=energy_shift))
                    except np.linalg.LinAlgError:
                        energy, rank = float("nan"), 0
                    energies[arm].append(energy)
                    ranks[arm].append(rank)
            for arm in energies:
                stats = error_summary(energies[arm], exact)
                verdict = "met" if stats["median"] <= TARGET_ERROR else ""
                print(f"  {eps:>8.0e} {m:>3} {arm:>9} {stats['median']:>12.3e} "
                      f"{stats['p90']:>12.3e} {float(np.median(ranks[arm])):>5.0f} "
                      f"{stats['failures']:>6}  {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
