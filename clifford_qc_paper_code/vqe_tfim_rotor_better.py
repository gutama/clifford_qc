"""
vqe_tfim_rotor_better.py — rotor/HVA VQE for the 1D TFIM in Cl(2n,C)
================================================================================

What this file improves over the first VQE prototype
----------------------------------------------------
* keeps the operator-centric Clifford/Pauli-word formulation from
  ``ga_qc_operator_improved``;
* fixes an important subtlety: the two-point parameter-shift rule is exact for
  each individual rotor gate, while a shared HVA parameter controls many rotor
  gates.  Therefore the exact shared-parameter gradient is the SUM of the
  gate-wise shifts, or equivalently the adjoint sweep implemented here;
* replaces slow default Adam runs with L-BFGS-B when SciPy is available, with a
  pure-Python Adam fallback;
* adds typed experiment results, optional periodic boundary conditions,
  observables, half-chain entropy, gate-wise parameter-shift validation,
  continuation sweeps, and a small CLI;
* keeps exact diagonalization as a validation target, not as the variational
  simulation engine.

Model
-----
    H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i

The ansatz is a Hamiltonian-variational / QAOA-style product of Pauli-word
rotors, starting from |+...+>, the exact J=0 ground state:

    Π_l [ Π_bonds exp(-i γ_l ZZ/2) · Π_sites exp(-i β_l X/2) ] |+...+>

The code evolves density multivectors gate-by-gate:

    ρ(θ) = U(θ) ρ0 U(θ)†,     E(θ)=Tr(Hρ)=2^n <Hρ>_0.

Run
---
    python vqe_tfim_rotor_better.py
    python vqe_tfim_rotor_better.py --n 5 --pmax 3 --sweep
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    Z,
    rotor,
    ket_density,
    H_gate,
    evolve,
    expectation,
    purity,
    partial_trace,
    vn_entropy,
    to_matrix,
)

TOL = 1e-10


# ---------------------------------------------------------------------------
# 1. Data containers and validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateSpec:
    """One physical rotor gate in the shared-parameter HVA circuit."""

    generator: MV
    param_index: int
    label: str


@dataclass(frozen=True)
class VQEResult:
    p: int
    energy: float
    theta: tuple[float, ...]
    rel_error: float | None
    fidelity: float | None
    grad_norm: float
    nfev: int
    nit: int
    success: bool
    method: str


def _require_int(name: str, value: int, lo: int | None = None) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if lo is not None and value < lo:
        raise ValueError(f"{name} must be >= {lo}, got {value}")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_theta(theta: Sequence[float]) -> tuple[float, ...]:
    if len(theta) % 2 != 0:
        raise ValueError("theta must have even length: [gamma1, beta1, ...]")
    out = tuple(float(x) for x in theta)
    for k, x in enumerate(out):
        _require_finite(f"theta[{k}]", x)
    return out


# ---------------------------------------------------------------------------
# 2. TFIM model, states, ansatz
# ---------------------------------------------------------------------------

def tfim_hamiltonian(
    n: int,
    J: float,
    h: float,
    *,
    periodic: bool = False,
) -> tuple[MV, list[MV], list[MV]]:
    """Return ``(H, bonds, fields)`` for the 1D transverse-field Ising model.

    Open chain by default.  With ``periodic=True`` the final bond ``Z_{n-1}Z_0``
    is included.  For ``n=2`` the periodic chain would duplicate the same bond,
    so this function keeps only one bond unless ``n > 2``.
    """
    _require_int("n", n, 1)
    _require_finite("J", J)
    _require_finite("h", h)

    pairs = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        pairs.append((n - 1, 0))

    bonds = [Z(n, i) * Z(n, j) for i, j in pairs]
    fields = [X(n, i) for i in range(n)]

    H = MV(n)
    for P in bonds:
        H = H + (-float(J)) * P
    for P in fields:
        H = H + (-float(h)) * P
    return H, bonds, fields


@lru_cache(maxsize=32)
def plus_state(n: int) -> MV:
    """Return |+...+><+...+|, cached because every objective call uses it."""
    _require_int("n", n, 1)
    rho = ket_density(n, "0" * n)
    for j in range(n):
        rho = evolve(rho, H_gate(n, j))
    return rho


def gate_specs(bonds: Sequence[MV], fields: Sequence[MV], p: int) -> list[GateSpec]:
    _require_int("p", p, 0)
    specs: list[GateSpec] = []
    for layer in range(p):
        gamma_idx, beta_idx = 2 * layer, 2 * layer + 1
        for i, P in enumerate(bonds):
            specs.append(GateSpec(P, gamma_idx, f"L{layer+1}:ZZ[{i}]") )
        for i, P in enumerate(fields):
            specs.append(GateSpec(P, beta_idx, f"L{layer+1}:X[{i}]") )
    return specs


def ansatz_gates(
    bonds: Sequence[MV],
    fields: Sequence[MV],
    theta: Sequence[float],
    *,
    gate_angle_shifts: dict[int, float] | None = None,
) -> list[MV]:
    """Build all physical rotor gates.

    ``theta = [γ1, β1, γ2, β2, ...]``.  ``gate_angle_shifts`` is used for
    gate-wise parameter-shift validation: it shifts one physical gate without
    shifting all other gates that share the same HVA parameter.
    """
    theta = validate_theta(theta)
    p = len(theta) // 2
    shifts = gate_angle_shifts or {}
    gates: list[MV] = []
    for k, spec in enumerate(gate_specs(bonds, fields, p)):
        angle = theta[spec.param_index] + shifts.get(k, 0.0)
        gates.append(rotor(spec.generator, angle))
    return gates


def full_unitary(n: int, bonds: Sequence[MV], fields: Sequence[MV], theta: Sequence[float]) -> MV:
    """Return U = G_m ... G_2 G_1 for the ansatz gate order."""
    U = I(n)
    for G in ansatz_gates(bonds, fields, theta):
        U = G * U
    return U


def prepared_state(
    n: int,
    bonds: Sequence[MV],
    fields: Sequence[MV],
    theta: Sequence[float],
    *,
    gate_angle_shifts: dict[int, float] | None = None,
) -> MV:
    rho = plus_state(n)
    for G in ansatz_gates(bonds, fields, theta, gate_angle_shifts=gate_angle_shifts):
        rho = evolve(rho, G)  # gate-by-gate keeps word sparsity
    return rho


def energy(H: MV, n: int, bonds: Sequence[MV], fields: Sequence[MV], theta: Sequence[float]) -> float:
    return expectation(prepared_state(n, bonds, fields, theta), H).real


# ---------------------------------------------------------------------------
# 3. Exact gradients: adjoint sweep and gate-wise parameter shift
# ---------------------------------------------------------------------------

def energy_and_gradient(
    H: MV,
    n: int,
    bonds: Sequence[MV],
    fields: Sequence[MV],
    theta: Sequence[float],
) -> tuple[float, list[float]]:
    """Return exact ``(E, dE/dtheta)`` by one adjoint sweep.

    For one physical rotor ``G_k = exp(-i θ_a P_k/2)``:

        dρ_k/dθ_a = -(i/2) [P_k, ρ_k]
        dE_k/dθ_a = -(i/2) Tr(O_k [P_k, ρ_k])

    where ``ρ_k`` is the state after gate ``k`` and ``O_k`` is the Hamiltonian
    pulled back through gates after ``k``.  If a parameter is shared by many
    gates, their exact contributions are summed.
    """
    theta = validate_theta(theta)
    p = len(theta) // 2
    specs = gate_specs(bonds, fields, p)
    gates = [rotor(spec.generator, theta[spec.param_index]) for spec in specs]

    states_after: list[MV] = []
    rho = plus_state(n)
    for G in gates:
        rho = evolve(rho, G)
        states_after.append(rho)
    E = expectation(rho, H).real

    grads = [0.0] * len(theta)
    O = H
    for k in range(len(gates) - 1, -1, -1):
        P = specs[k].generator
        rho_k = states_after[k]
        comm = P * rho_k - rho_k * P
        contrib = (-0.5j * (O * comm).trace()).real
        grads[specs[k].param_index] += contrib
        O = gates[k].dagger() * O * gates[k]
    return E, grads


def gradient(H: MV, n: int, bonds: Sequence[MV], fields: Sequence[MV], theta: Sequence[float]) -> list[float]:
    return energy_and_gradient(H, n, bonds, fields, theta)[1]


def gatewise_parameter_shift_gradient(
    H: MV,
    n: int,
    bonds: Sequence[MV],
    fields: Sequence[MV],
    theta: Sequence[float],
) -> list[float]:
    """Exact hardware-style gradient for shared parameters.

    The two-point rule is exact for each individual rotor gate.  Since HVA uses
    parameter sharing, the gradient of one shared parameter is the sum over the
    physical gates controlled by that parameter.
    """
    theta = validate_theta(theta)
    p = len(theta) // 2
    grads = [0.0] * len(theta)
    specs = gate_specs(bonds, fields, p)
    for gate_index, spec in enumerate(specs):
        plus = {gate_index: math.pi / 2}
        minus = {gate_index: -math.pi / 2}
        e_plus = expectation(
            prepared_state(n, bonds, fields, theta, gate_angle_shifts=plus), H
        ).real
        e_minus = expectation(
            prepared_state(n, bonds, fields, theta, gate_angle_shifts=minus), H
        ).real
        grads[spec.param_index] += 0.5 * (e_plus - e_minus)
    return grads


def naive_shared_parameter_shift_gradient(
    H: MV,
    n: int,
    bonds: Sequence[MV],
    fields: Sequence[MV],
    theta: Sequence[float],
) -> list[float]:
    """Naive two-point shift of all gates sharing one θ.

    This is intentionally provided as a diagnostic.  It is generally NOT exact
    for shared HVA parameters because the shared generator has more than two
    eigenvalues, even though each individual Pauli-word rotor has P²=1.
    """
    theta = list(validate_theta(theta))
    out = []
    for k in range(len(theta)):
        tp, tm = list(theta), list(theta)
        tp[k] += math.pi / 2
        tm[k] -= math.pi / 2
        out.append(0.5 * (energy(H, n, bonds, fields, tp) - energy(H, n, bonds, fields, tm)))
    return out


# ---------------------------------------------------------------------------
# 4. Optimizers
# ---------------------------------------------------------------------------

def adam_minimize(
    f_eg: Callable[[Sequence[float]], tuple[float, Sequence[float]]],
    theta0: Sequence[float],
    *,
    iters: int = 300,
    lr: float = 0.05,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    bound: float | None = None,
    gtol: float = 1e-8,
) -> tuple[tuple[float, ...], list[float], int, int, bool]:
    """Pure-Python Adam fallback.  Returns θ, history, nit, nfev, success."""
    th = [float(x) for x in theta0]
    m = [0.0] * len(th)
    v = [0.0] * len(th)
    history: list[float] = []
    success = False
    for it in range(1, iters + 1):
        E, g_raw = f_eg(th)
        g = [float(x) for x in g_raw]
        history.append(float(E))
        if bound is not None and E < bound - 1e-9:
            raise RuntimeError(f"variational bound violated: {E} < {bound}")
        if math.sqrt(sum(x * x for x in g)) < gtol:
            success = True
            break
        for k in range(len(th)):
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            th[k] -= lr * mh / (math.sqrt(vh) + eps)
    return tuple(th), history, len(history), len(history), success


def minimize_energy(
    f_eg: Callable[[Sequence[float]], tuple[float, Sequence[float]]],
    theta0: Sequence[float],
    *,
    maxiter: int = 200,
    bound: float | None = None,
    gtol: float = 1e-8,
    method: str = "auto",
) -> tuple[tuple[float, ...], list[float], int, int, bool, str, float]:
    """Minimize energy with L-BFGS-B if available; otherwise Adam."""
    theta0 = validate_theta(theta0)
    if method not in {"auto", "lbfgsb", "adam"}:
        raise ValueError("method must be 'auto', 'lbfgsb', or 'adam'")

    if method in {"auto", "lbfgsb"}:
        try:
            import numpy as np
            from scipy.optimize import minimize

            history: list[float] = []

            def fg(x):
                E, g = f_eg(tuple(float(y) for y in x))
                if bound is not None and E < bound - 1e-9:
                    raise RuntimeError(f"variational bound violated: {E} < {bound}")
                history.append(float(E))
                return float(E), np.array(g, dtype=float)

            res = minimize(
                fg,
                np.array(theta0, dtype=float),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-13, "maxls": 50},
            )
            final_theta = tuple(float(x) for x in res.x)
            E, g = f_eg(final_theta)
            grad_norm = math.sqrt(sum(float(x) * float(x) for x in g))
            # ``success`` can be False due to precision-loss warnings even when
            # the variational result is excellent; expose it but do not discard.
            return final_theta, history or [float(E)], int(res.nit), int(res.nfev), bool(res.success), "L-BFGS-B", grad_norm
        except ImportError:
            if method == "lbfgsb":
                raise

    th, hist, nit, nfev, success = adam_minimize(
        f_eg, theta0, iters=maxiter, bound=bound, gtol=gtol
    )
    E, g = f_eg(th)
    grad_norm = math.sqrt(sum(float(x) * float(x) for x in g))
    return th, hist or [float(E)], nit, nfev, success, "Adam", grad_norm


# ---------------------------------------------------------------------------
# 5. Exact targets, fidelity, observables
# ---------------------------------------------------------------------------

def exact_ground(H: MV):
    import numpy as np

    w, V = np.linalg.eigh(to_matrix(H))
    return float(w[0]), V[:, 0]


def exact_ground_space(H: MV, *, tol: float = 1e-9):
    """Return ``(E0, V0)`` where columns of V0 span the ground subspace."""
    import numpy as np

    w, V = np.linalg.eigh(to_matrix(H))
    mask = np.abs(w - w[0]) <= tol
    return float(w[0]), V[:, mask]


def fidelity_with_vector(rho: MV, psi) -> float:
    import numpy as np

    return float(np.real(psi.conj() @ to_matrix(rho) @ psi))


def fidelity_with_subspace(rho: MV, vectors) -> float:
    """Fidelity with a possibly degenerate exact ground subspace."""
    import numpy as np

    R = to_matrix(rho)
    P = vectors @ vectors.conj().T
    return float(np.real(np.trace(P @ R)))


def magnetization_x(rho: MV) -> float:
    return sum(expectation(rho, X(rho.n, j)).real for j in range(rho.n)) / rho.n


def magnetization_z_abs(rho: MV) -> float:
    return abs(sum(expectation(rho, Z(rho.n, j)).real for j in range(rho.n)) / rho.n)


def zz_correlation(rho: MV, *, periodic: bool = False) -> float:
    n = rho.n
    pairs = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        pairs.append((n - 1, 0))
    if not pairs:
        return 0.0
    return sum(expectation(rho, Z(n, i) * Z(n, j)).real for i, j in pairs) / len(pairs)


def half_chain_entropy(rho: MV) -> float:
    """Von Neumann entropy of the left half, in bits."""
    n = rho.n
    traced = set(range(n // 2, n))
    return vn_entropy(partial_trace(rho, traced), base=2.0)


def optimize_vqe(
    n: int,
    J: float,
    h: float,
    p: int,
    *,
    theta0: Sequence[float] | None = None,
    periodic: bool = False,
    maxiter: int = 200,
    method: str = "auto",
) -> tuple[VQEResult, MV, MV, list[MV], list[MV]]:
    H, bonds, fields = tfim_hamiltonian(n, J, h, periodic=periodic)
    E0, V0 = exact_ground_space(H)
    if theta0 is None:
        theta0 = (0.1,) * (2 * p)
    theta0 = validate_theta(theta0)
    if len(theta0) != 2 * p:
        raise ValueError(f"theta0 length must be {2*p} for p={p}")

    th, hist, nit, nfev, success, used_method, grad_norm = minimize_energy(
        lambda t: energy_and_gradient(H, n, bonds, fields, t),
        theta0,
        maxiter=maxiter,
        bound=E0,
        method=method,
    )
    rho = prepared_state(n, bonds, fields, th)
    E = energy(H, n, bonds, fields, th)
    rel = abs(E - E0) / max(abs(E0), TOL)
    F = fidelity_with_subspace(rho, V0)
    return (
        VQEResult(p, E, tuple(th), rel, F, grad_norm, nfev, nit, success, used_method),
        rho,
        H,
        bonds,
        fields,
    )


# ---------------------------------------------------------------------------
# 6. Verification and demos
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def run_verification(n: int = 4, J: float = 1.0, h: float = 1.0) -> None:
    import numpy as np

    rng = np.random.default_rng(11)
    print("== VQE on the TFIM with the rotor/HVA ansatz ==\n")
    H, bonds, fields = tfim_hamiltonian(n, J, h)
    E0, psi0 = exact_ground(H)

    print("-- ansatz structure --")
    th_rand = tuple(float(x) for x in rng.uniform(-1, 1, size=6))
    U = full_unitary(n, bonds, fields, th_rand)
    _check("U(θ)†U(θ) = 1 at random θ", (U.dagger() * U).is_close(I(n), 1e-9))
    rho = prepared_state(n, bonds, fields, th_rand)
    _check(
        "ρ(θ) remains a pure density multivector",
        abs(purity(rho) - 1) < 1e-9 and abs(rho.trace() - 1) < 1e-9,
    )
    _check(
        "rotor grades match JW blade decode",
        rotor(bonds[0], 0.7).grades() == {0, 4}
        and rotor(fields[0], 0.7).grades() == {0, 1}
        and rotor(fields[1], 0.7).grades() == {0, 3},
    )

    print("-- gradient exactness --")
    _, g_adj = energy_and_gradient(H, n, bonds, fields, th_rand)
    g_gate_shift = gatewise_parameter_shift_gradient(H, n, bonds, fields, th_rand)
    err_gate = max(abs(a - b) for a, b in zip(g_adj, g_gate_shift))
    _check(f"adjoint gradient == gate-wise parameter shift (max dev {err_gate:.2e})", err_gate < 1e-9)

    epsfd = 1e-6
    g_fd = []
    for k in range(len(th_rand)):
        tp, tm = list(th_rand), list(th_rand)
        tp[k] += epsfd
        tm[k] -= epsfd
        g_fd.append((energy(H, n, bonds, fields, tp) - energy(H, n, bonds, fields, tm)) / (2 * epsfd))
    err_fd = max(abs(a - b) for a, b in zip(g_adj, g_fd))
    _check(f"adjoint gradient == finite difference (max dev {err_fd:.2e})", err_fd < 1e-5)

    g_naive = naive_shared_parameter_shift_gradient(H, n, bonds, fields, th_rand)
    naive_err = max(abs(a - b) for a, b in zip(g_adj, g_naive))
    _check("naive shared-θ shift is detected as non-exact", naive_err > 1e-3)

    print(f"\n-- optimization: n={n}, J={J}, h={h}; exact E0={E0:+.6f} --")
    targets = {1: 5e-2, 2: 5e-3, 3: 1e-3}
    results: dict[int, VQEResult] = {}
    prev_theta: tuple[float, ...] = ()
    for p in (1, 2, 3):
        theta0 = prev_theta + (0.1, 0.1)
        result, rho_p, _, _, _ = optimize_vqe(
            n, J, h, p, theta0=theta0, maxiter=220 if p < 3 else 320
        )
        prev_theta = result.theta
        results[p] = result
        print(
            f"    p={p}: E={result.energy:+.6f}  rel.err={result.rel_error:.2e}  "
            f"F={result.fidelity:.6f}  |g|={result.grad_norm:.1e}  "
            f"{result.method}/{result.nfev} evals"
        )
        _check(f"p={p}: variational target met", result.rel_error is not None and result.rel_error < targets[p])
        if p == 3:
            _check("p=3 half-chain entropy is finite", half_chain_entropy(rho_p) > 0)
    _check("deeper ansatz improves energy monotonically", results[1].rel_error > results[2].rel_error > results[3].rel_error)
    _check("p=3 fidelity > 0.999", results[3].fidelity is not None and results[3].fidelity > 0.999)


def continuation_sweep(
    n: int = 4,
    J: float = 1.0,
    h_values: Iterable[float] = (2.0, 1.4, 1.0, 0.6, 0.2),
    *,
    p: int = 3,
    periodic: bool = False,
) -> list[tuple[float, VQEResult, float, float, float]]:
    """Warm-start a sweep from the paramagnet toward the ordered phase.

    Returns rows ``(h, result, mx, zz, S_half)`` in the optimization order.
    """
    rows = []
    theta = (0.1,) * (2 * p)
    for hh in h_values:
        result, rho, _, _, _ = optimize_vqe(
            n, J, float(hh), p, theta0=theta, periodic=periodic, maxiter=260
        )
        theta = result.theta
        rows.append((float(hh), result, magnetization_x(rho), zz_correlation(rho, periodic=periodic), half_chain_entropy(rho)))
    return rows


def print_sweep(rows: Sequence[tuple[float, VQEResult, float, float, float]]) -> None:
    print("\n-- h/J continuation sweep --")
    print("     h/J      E_vqe      rel.err    fidelity     <X>       <ZZ>     S_half")
    for hh, r, mx, zz, ent in rows:
        print(
            f"    {hh:4.1f}   {r.energy:+.6f}   {r.rel_error:.2e}   "
            f"{r.fidelity:.6f}   {mx:+.4f}   {zz:+.4f}   {ent:.4f}"
        )


def run_demo(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args([])
    run_verification(n=args.n, J=args.J, h=args.h)
    if args.sweep:
        rows = continuation_sweep(
            n=args.n,
            J=args.J,
            h_values=args.h_values,
            p=args.pmax,
            periodic=args.periodic,
        )
        print_sweep(rows)
        _check("sweep rel.err < 2e-2 at every h", all(r.rel_error < 2e-2 for _, r, _, _, _ in rows))
    print("\nAll checks passed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotor/HVA VQE for the 1D TFIM in Cl(2n,C).")
    parser.add_argument("--n", type=int, default=4, help="number of qubits/sites")
    parser.add_argument("--J", type=float, default=1.0, help="Ising coupling")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field for verification run")
    parser.add_argument("--pmax", type=int, default=3, help="layers for continuation sweep")
    parser.add_argument("--periodic", action="store_true", help="use periodic boundary conditions")
    parser.add_argument("--sweep", action="store_true", default=True, help="run the h/J continuation sweep")
    parser.add_argument(
        "--h-values",
        type=float,
        nargs="*",
        default=[2.0, 1.4, 1.0, 0.6, 0.2],
        help="h/J values for continuation sweep, optimized in listed order",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_demo(parse_args())
