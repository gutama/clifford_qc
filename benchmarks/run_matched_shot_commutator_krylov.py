#!/usr/bin/env python3
"""Finite-shot commutator subspaces against Krylov at matched shot budgets.

An exploratory study of strongly correlated molecules, where Hartree-Fock is
hundreds of millihartree from the exact energy, at 6, 8, 12 and 16 qubits:
stretched water in growing active spaces, stretched hydrogen chains and
stretched N2. ``benchmarks/MATCHED_SHOT_COMMUTATOR_KRYLOV.md`` states the
question, the shot model and what is and is not claimed.

Everything here is exact linear algebra in the (N, S_z) sector of the active
space, built with PySCF's FCI machinery. At 16 qubits that sector has at most
4,900 determinants, so every basis vector, matrix element and target energy
is exact. Finite shots enter only through the declared element-level model:
each real quantity of the normalized pencil is estimated by a two-outcome
(Hadamard-test style) measurement, ``+-1`` for overlaps and ``+-lambda`` for
Hamiltonian elements, where ``lambda`` is the Pauli 1-norm of the
Jordan-Wigner Hamiltonian without its identity term. Each arm divides the same
total budget equally among the real quantities it must measure.

    python benchmarks/run_matched_shot_commutator_krylov.py
    python benchmarks/run_matched_shot_commutator_krylov.py --systems h4_r2.0 --replicas 20
    python benchmarks/run_matched_shot_commutator_krylov.py --report

``--report`` prints the Markdown tables the study document embeds, read from
the record. Running the study requires PySCF (the ``molecular`` extra).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Threaded BLAS and OpenMP reductions reorder floating-point sums, so the same
# run differs from itself in the last bits, and near-tied selections can flip.
# One thread makes a run reproducible bit for bit, as the molecular runner does.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifford_qc.fermion import c_op, cdag_op  # noqa: E402
from clifford_qc.ir import PauliWord  # noqa: E402
from clifford_qc.measurement.grouping import qwc_groups  # noqa: E402
from clifford_qc.multivector import MV  # noqa: E402
from clifford_qc.reproducibility import execution_provenance  # noqa: E402
from clifford_qc.selection import canonical_argmax  # noqa: E402
from clifford_qc.subspace.adaptive import (  # noqa: E402
    DEFAULT_MIN_LOWERING, DEFAULT_MIN_ORTHOGONALITY, _two_by_two_lowering,
)
from clifford_qc.subspace.linalg import solve_projected  # noqa: E402

DEFAULT_OUT = ROOT / "benchmarks" / "matched_shot_commutator_krylov.json"
SCHEMA = "clifford_qc.matched_shot_commutator_krylov.v1"
SEED_ROOT = 20261118
BASIS_SIZES = (4, 6, 8, 12)
BUDGETS = tuple(10 ** k for k in range(4, 13))
REPLICAS = 200
CHEMICAL_ACCURACY = 1.6e-3
ARMS = ("rt_krylov", "power_krylov", "commutator_acase", "pure_commutator_acase",
        "excitation_acase")
STRETCHED_WATER = "O 0 0 0.2346; H 0 1.5144 -0.9384; H 0 -1.5144 -0.9384"


def chain(atoms: int, spacing: float) -> str:
    return "; ".join(f"H 0 0 {index * spacing:.6f}" for index in range(atoms))


# Fixed before any finite-shot draw. Active spaces are (electrons, spatial
# orbitals); ``None`` means every orbital and electron of the basis.
SYSTEMS: tuple[dict[str, Any], ...] = (
    {"key": "h2o_2x_cas4e3o", "family": "stretched H2O (2x)", "atom": STRETCHED_WATER,
     "basis": "sto-3g", "active": (4, 3)},
    {"key": "h4_r2.0", "family": "stretched H chain (2.0 A)", "atom": chain(4, 2.0),
     "basis": "sto-3g", "active": None},
    {"key": "h2o_2x_cas8e6o", "family": "stretched H2O (2x)", "atom": STRETCHED_WATER,
     "basis": "sto-3g", "active": (8, 6)},
    {"key": "h6_r2.0", "family": "stretched H chain (2.0 A)", "atom": chain(6, 2.0),
     "basis": "sto-3g", "active": None},
    {"key": "n2_r2.0_cas6e6o", "family": "stretched N2 (2.0 A)", "atom": "N 0 0 0; N 0 0 2.0",
     "basis": "sto-3g", "active": (6, 6)},
    {"key": "h2o_2x_631g_cas8e8o", "family": "stretched H2O (2x)", "atom": STRETCHED_WATER,
     "basis": "6-31g", "active": (8, 8)},
    {"key": "h8_r2.0", "family": "stretched H chain (2.0 A)", "atom": chain(8, 2.0),
     "basis": "sto-3g", "active": None},
    {"key": "n2_r2.0_cas10e8o", "family": "stretched N2 (2.0 A)", "atom": "N 0 0 0; N 0 0 2.0",
     "basis": "sto-3g", "active": (10, 8)},
)


# --------------------------------------------------------------- chemistry

@dataclass
class Sector:
    """An active space and its exact (N, S_z = 0) sector."""

    key: str
    norb: int
    nelec: tuple[int, int]
    h1: np.ndarray
    eri: np.ndarray
    ecore: float
    e_rhf: float
    hamiltonian: np.ndarray  # sector matrix including the core energy
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spins: np.ndarray  # <S^2> of each eigenvector
    reference: np.ndarray  # the RHF determinant

    @property
    def dimension(self) -> int:
        return self.hamiltonian.shape[0]

    def shape(self, nelec: tuple[int, int]) -> tuple[int, int]:
        from pyscf.fci import cistring

        return (cistring.num_strings(self.norb, nelec[0]),
                cistring.num_strings(self.norb, nelec[1]))


def build_sector(spec: dict[str, Any]) -> Sector:
    from pyscf import ao2mo, gto, mcscf, scf
    from pyscf.fci import direct_spin1, spin_op

    mol = gto.M(atom=spec["atom"], basis=spec["basis"], unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 500
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {spec['key']}")
    electrons, orbitals = spec["active"] or (mol.nelectron, mol.nao)
    casci = mcscf.CASCI(mf, orbitals, electrons)
    casci.verbose = 0
    h1, ecore = casci.get_h1eff()
    eri = ao2mo.restore(1, casci.get_h2eff(), orbitals)
    nelec = (electrons // 2, electrons // 2)
    absorbed = direct_spin1.absorb_h1e(h1, eri, orbitals, nelec, 0.5)
    from pyscf.fci import cistring

    na, nb = (cistring.num_strings(orbitals, nelec[0]),
              cistring.num_strings(orbitals, nelec[1]))
    dimension = na * nb
    columns = np.empty((dimension, dimension))
    unit = np.zeros(dimension)
    for index in range(dimension):
        unit[index] = 1.0
        columns[:, index] = direct_spin1.contract_2e(
            absorbed, unit.reshape(na, nb), orbitals, nelec).ravel()
        unit[index] = 0.0
    hamiltonian = 0.5 * (columns + columns.T) + float(ecore) * np.eye(dimension)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    spins = np.array([spin_op.spin_square0(eigenvectors[:, k].reshape(na, nb),
                                           orbitals, nelec)[0]
                      for k in range(min(dimension, 12))])
    reference = np.zeros(dimension)
    reference[0] = 1.0  # the lowest alpha and beta strings: the RHF determinant
    return Sector(spec["key"], orbitals, nelec, np.asarray(h1), eri, float(ecore),
                  float(mf.e_tot), hamiltonian, eigenvalues, eigenvectors, spins, reference)


def qubit_hamiltonian(sector: Sector) -> MV:
    """The interleaved Jordan-Wigner Hamiltonian of the active space.

    The same assembly as ``models.fcidump.model_from_fcidump``, with a
    coefficient-level Hermiticity check in place of its Hilbert-Schmidt one,
    which scales as 2^(n/2) and rejects 16-qubit N2 on rounding alone.
    """
    norb = sector.norb
    n = 2 * norb
    total = MV(n, {0: complex(sector.ecore)})
    for p in range(norb):
        for q in range(norb):
            value = float(sector.h1[p, q])
            if abs(value) <= 1e-12:
                continue
            for spin in (0, 1):
                total = total + value * (cdag_op(n, 2 * p + spin) * c_op(n, 2 * q + spin))
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    value = 0.5 * float(sector.eri[p, q, r, s])
                    if abs(value) <= 1e-12:
                        continue
                    for spin in (0, 1):
                        for other in (0, 1):
                            total = total + value * (
                                cdag_op(n, 2 * p + spin) * cdag_op(n, 2 * r + other)
                                * c_op(n, 2 * s + other) * c_op(n, 2 * q + spin))
    worst = max((abs(c.imag) for c in total.terms.values()), default=0.0)
    scale = max((abs(c) for c in total.terms.values()), default=1.0)
    if worst > 1e-9 * max(1.0, scale):
        raise ValueError(f"non-Hermitian qubit Hamiltonian: imaginary part {worst:.3g}")
    return MV(n, {code: complex(c.real) for code, c in total.terms.items()
                  if abs(c.real) > 1e-12})


# --------------------------------------------------------------- generators

def apply_string(sector: Sector, vector: np.ndarray,
                 operators: tuple[tuple[str, int, int], ...]) -> np.ndarray:
    """Apply ``operators`` (rightmost first) to a sector vector.

    Each operator is ``(kind, spin, orbital)`` with kind ``cre`` or ``des``
    and spin 0 for alpha, 1 for beta. PySCF's string ordering fixes the
    fermionic signs, consistently for every generator here.
    """
    from pyscf.fci import addons

    nelec = sector.nelec
    current = vector.reshape(sector.shape(nelec))
    for kind, spin, orbital in reversed(operators):
        function = {("cre", 0): addons.cre_a, ("cre", 1): addons.cre_b,
                    ("des", 0): addons.des_a, ("des", 1): addons.des_b}[(kind, spin)]
        current = function(current, sector.norb, nelec, orbital)
        delta = 1 if kind == "cre" else -1
        nelec = (nelec[0] + delta, nelec[1]) if spin == 0 else (nelec[0], nelec[1] + delta)
    if nelec != sector.nelec:
        raise ValueError("operator string leaves the sector")
    return np.asarray(current).ravel()


def excitations(sector: Sector) -> list[tuple[str, tuple, tuple]]:
    """S_z-conserving singles and doubles of the RHF determinant.

    Each is ``(label, excitation, de-excitation)``; the generator is the
    anti-Hermitian difference of the two strings.
    """
    occupied = range(sector.nelec[0])
    virtual = range(sector.nelec[0], sector.norb)
    out = []
    for spin in (0, 1):
        for i in occupied:
            for a in virtual:
                out.append((f"s{spin}:{i}->{a}", (("cre", spin, a), ("des", spin, i)),
                            (("cre", spin, i), ("des", spin, a))))
    for spin in (0, 1):
        for i in occupied:
            for j in occupied:
                if j <= i:
                    continue
                for a in virtual:
                    for b in virtual:
                        if b <= a:
                            continue
                        out.append((f"d{spin}{spin}:{i}{j}->{a}{b}",
                                    (("cre", spin, a), ("cre", spin, b),
                                     ("des", spin, j), ("des", spin, i)),
                                    (("cre", spin, i), ("cre", spin, j),
                                     ("des", spin, b), ("des", spin, a))))
    for i in occupied:
        for j in occupied:
            for a in virtual:
                for b in virtual:
                    out.append((f"d01:{i}{j}->{a}{b}",
                                (("cre", 0, a), ("cre", 1, b), ("des", 1, j), ("des", 0, i)),
                                (("cre", 0, i), ("cre", 1, j), ("des", 1, b), ("des", 0, a))))
    return out


def apply_excitation(sector: Sector, excitation, vector: np.ndarray) -> np.ndarray:
    _, up, down = excitation
    return apply_string(sector, vector, up) - apply_string(sector, vector, down)


def candidate_pool(sector: Sector, *, depth: int
                   ) -> list[tuple[str, np.ndarray, float]]:
    """Normalized A_0 (depth 0) up to depth-``depth`` nested commutators on |HF>.

    ``[H, t]|psi> = H t|psi> - t H|psi>`` and
    ``[H, [H, t]]|psi> = H^2 t|psi> - 2 H t H|psi> + t H^2|psi>``. Vectors
    that vanish are dropped. Each entry is ``(label, unit vector, <v|H|v>)``.
    """
    H = sector.hamiltonian
    psi = sector.reference
    h_psi = H @ psi
    hh_psi = H @ h_psi
    pool = []
    for excitation in excitations(sector):
        label = excitation[0]
        t_psi = apply_excitation(sector, excitation, psi)
        vectors = [(label, t_psi)]
        if depth >= 1:
            t_h_psi = apply_excitation(sector, excitation, h_psi)
            vectors.append((f"[H,{label}]", H @ t_psi - t_h_psi))
            if depth >= 2:
                t_hh_psi = apply_excitation(sector, excitation, hh_psi)
                vectors.append((f"[H,[H,{label}]]",
                                H @ (H @ t_psi) - 2.0 * (H @ t_h_psi) + t_hh_psi))
        for name, vector in vectors:
            norm = float(np.linalg.norm(vector))
            if norm > 1e-10:
                unit = vector / norm
                pool.append((name, unit, float(np.vdot(unit, H @ unit).real)))
    return pool


def solve_exact(S: np.ndarray, H: np.ndarray):
    return solve_projected(S, H)


def pencil(vectors: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vectors.conj().T @ vectors, vectors.conj().T @ (H @ vectors)


def greedy_acase(sector: Sector, pool: list[tuple[str, np.ndarray, float]], size: int
                 ) -> tuple[list[str], np.ndarray]:
    """A-CASE growth with its two-by-two lowering criterion, in sector vectors.

    Starts from the reference and adds, one at a time, the candidate with the
    largest exact lowering, as ``run_acase`` does: candidates almost inside the
    current span are rejected by the same orthogonality floor, and growth stops
    when no candidate lowers the energy by ``run_acase``'s ``min_lowering``.
    """
    H = sector.hamiltonian
    basis = [sector.reference.astype(complex)]
    labels = ["I"]
    while len(basis) < size:
        B = np.column_stack(basis)
        S, Hm = pencil(B, H)
        solved = solve_exact(S, Hm)
        energy = float(solved.energies[0])
        ritz = B @ solved.coefficients[:, 0]
        ritz = ritz / np.linalg.norm(ritz)
        h_ritz = H @ ritz
        Q, _ = np.linalg.qr(B)
        lowerings: dict[int, float] = {}
        for index, (name, vector, h_cc) in enumerate(pool):
            if name in labels:
                continue
            projected = Q.conj().T @ vector
            if 1.0 - float(np.vdot(projected, projected).real) < DEFAULT_MIN_ORTHOGONALITY:
                continue
            lowerings[index] = _two_by_two_lowering(
                energy, complex(np.vdot(ritz, vector)), complex(np.vdot(h_ritz, vector)),
                1.0, h_cc)
        # The same stopping rule as run_acase: nothing lowers by min_lowering.
        if not lowerings or max(lowerings.values()) < DEFAULT_MIN_LOWERING:
            break
        # Symmetry makes exact ties common; resolve them by pool order, as the
        # library does, rather than by the last bits of the arithmetic.
        best = canonical_argmax(sorted(lowerings), lambda index: lowerings[index])
        labels.append(pool[best][0])
        basis.append(pool[best][1].astype(complex))
    return labels, np.column_stack(basis)


def krylov_bases(sector: Sector, size: int, identity: float
                 ) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    w, V = sector.eigenvalues, sector.eigenvectors
    coefficients = V.T @ sector.reference
    spread = float(w[-1] - w[0])
    step = math.pi / spread
    real_time = np.column_stack([V @ (np.exp(-1j * w * step * k) * coefficients)
                                 for k in range(size)])
    shifted = w - identity
    power = []
    for k in range(size):
        vector = V @ (shifted ** k * coefficients)
        power.append(vector / np.linalg.norm(vector))
    return {"rt_krylov": (real_time, {"time_step": step, "spectral_range": spread}),
            "power_krylov": (np.column_stack(power).astype(complex), {})}


# --------------------------------------------------------------- shot model

def quantities(arm: str, size: int) -> int:
    """Real quantities an arm measures for a ``size`` x ``size`` pencil."""
    if arm == "rt_krylov":
        # Toeplitz: s_d for d >= 1 (complex), h_0 (real), h_d for d >= 1 (complex).
        return 4 * size - 3
    return 2 * size * (size - 1) + size


def sample_pencil(arm: str, S: np.ndarray, Hp: np.ndarray, shots: int, scale: float,
                  rng: np.random.Generator, draws: int) -> tuple[np.ndarray, np.ndarray]:
    """``draws`` noisy copies of a normalized pencil under the element model.

    A real quantity ``x`` with per-shot outcomes ``+-b`` is estimated by
    ``b (2k/n - 1)`` with ``k ~ Binomial(n, (1 + x/b)/2)``: ``b = 1`` for
    overlaps and ``b = scale`` for Hamiltonian elements. The diagonal of the
    normalized overlap is exactly one and is not measured.
    """
    size = S.shape[0]

    def estimate(values: np.ndarray, bound: float) -> np.ndarray:
        p = np.clip((1.0 + values / bound) / 2.0, 0.0, 1.0)
        k = rng.binomial(shots, np.broadcast_to(p, (draws,) + p.shape))
        return bound * (2.0 * k / shots - 1.0)

    S_out = np.broadcast_to(np.eye(size, dtype=complex), (draws, size, size)).copy()
    H_out = np.zeros((draws, size, size), dtype=complex)
    if arm == "rt_krylov":
        s = S[0, 1:]
        h = Hp[0, :]
        s_hat = estimate(s.real, 1.0) + 1j * estimate(s.imag, 1.0)
        h_hat = estimate(h.real, scale) + 1j * estimate(h.imag, scale)
        h_hat[:, 0] = h_hat[:, 0].real
        for j in range(size):
            for k in range(j + 1, size):
                S_out[:, j, k] = s_hat[:, k - j - 1]
                S_out[:, k, j] = np.conj(s_hat[:, k - j - 1])
            for k in range(j, size):
                H_out[:, j, k] = h_hat[:, k - j]
                H_out[:, k, j] = np.conj(h_hat[:, k - j])
        return S_out, H_out
    upper = np.triu_indices(size, 1)
    s = S[upper]
    S_hat = estimate(s.real, 1.0) + 1j * estimate(s.imag, 1.0)
    S_out[:, upper[0], upper[1]] = S_hat
    S_out[:, upper[1], upper[0]] = np.conj(S_hat)
    diagonal = np.arange(size)
    H_out[:, diagonal, diagonal] = estimate(Hp[diagonal, diagonal].real, scale)
    h = Hp[upper]
    H_hat = estimate(h.real, scale) + 1j * estimate(h.imag, scale)
    H_out[:, upper[0], upper[1]] = H_hat
    H_out[:, upper[1], upper[0]] = np.conj(H_hat)
    return S_out, H_out


def overlap_floor(size: int, shots: int) -> float:
    """Declared truncation floor: about twice the spectral norm of S noise.

    Off-diagonal overlap elements have complex variance at most ``2/n``; a
    Hermitian noise matrix with that entry variance has spectral norm near
    ``2 sqrt(2 size / n)``.
    """
    return 2.0 * math.sqrt(2.0 * size / shots)


def summarize(errors: list[float], failures: int, exact_error: float) -> dict[str, Any]:
    """Errors against the target, and shot deviations from the arm's exact energy.

    The first is what a user gets. The second separates the shot noise from
    the subspace's own error, which matters because noise can move an
    estimate from a poor subspace closer to the target by chance.
    """
    values = np.asarray([e for e in errors if math.isfinite(e)])
    total = len(errors)
    out: dict[str, Any] = {"replicas": total, "failures": failures}
    if not len(values):
        out.update({"median_abs_error": None, "p90_abs_error": None, "within_chemical": 0.0,
                    "below_target_beyond_chemical": 0.0, "median_error": None,
                    "median_abs_shot_deviation": None})
        return out
    absolute = np.abs(values)
    out.update({
        "median_abs_error": float(np.median(absolute)),
        "p90_abs_error": float(np.quantile(absolute, 0.9)),
        "median_error": float(np.median(values)),
        "within_chemical": float(np.sum(absolute <= CHEMICAL_ACCURACY) / total),
        "below_target_beyond_chemical": float(np.sum(values < -CHEMICAL_ACCURACY) / total),
        "median_abs_shot_deviation": float(np.median(np.abs(values - exact_error))),
        "error_quartiles": [float(q) for q in np.quantile(values, [0.25, 0.75])],
    })
    return out


# --------------------------------------------------------------- the study

def target_state(sector: Sector) -> tuple[float, int]:
    """Lowest singlet of the sector: the energy every arm is judged against."""
    for index, spin in enumerate(sector.spins):
        if abs(spin) < 1e-6:
            return float(sector.eigenvalues[index]), index
    raise RuntimeError(f"no singlet among the lowest states of {sector.key}")


def run_system(index: int, spec: dict[str, Any], *, replicas: int,
               sizes=BASIS_SIZES, budgets=BUDGETS) -> dict[str, Any]:
    started = time.perf_counter()
    sector = build_sector(spec)
    qubit = qubit_hamiltonian(sector)
    identity = float(qubit.terms.get(0, 0.0).real)
    scale = float(sum(abs(c) for code, c in qubit.terms.items() if code != 0))
    words = [PauliWord(qubit.n, code) for code in qubit.terms if code != 0]
    target, target_index = target_state(sector)
    target_vector = sector.eigenvectors[:, target_index]
    system: dict[str, Any] = {
        "key": spec["key"], "family": spec["family"], "atom": spec["atom"],
        "basis": spec["basis"], "active": spec["active"], "qubits": qubit.n,
        "sector_dimension": sector.dimension, "e_rhf": sector.e_rhf,
        "e_target": target, "target_index": target_index,
        "e_sector_ground": float(sector.eigenvalues[0]),
        "sector_ground_spin": float(sector.spins[0]),
        "hf_error": sector.e_rhf - target,
        "hf_target_overlap": float(abs(np.vdot(target_vector, sector.reference)) ** 2),
        "pauli_terms": len(words), "pauli_one_norm": scale,
        "identity_coefficient": identity, "hamiltonian_qwc_settings": len(qwc_groups(words)),
        "lowest_spins": [float(s) for s in sector.spins],
        "lowest_energies": [float(e) for e in sector.eigenvalues[:len(sector.spins)]],
    }
    pool_full = candidate_pool(sector, depth=2)
    pool_excitations = [entry for entry in pool_full if not entry[0].startswith("[")]
    pool_commutators = [entry for entry in pool_full if entry[0].startswith("[")]
    system["pool"] = {"excitations": len(pool_excitations),
                      "commutators": len(pool_commutators), "all": len(pool_full)}
    H = sector.hamiltonian
    arms: dict[str, Any] = {}
    largest = max(sizes)
    grown = {"commutator_acase": greedy_acase(sector, pool_full, largest),
             "pure_commutator_acase": greedy_acase(sector, pool_commutators, largest),
             "excitation_acase": greedy_acase(sector, pool_excitations, largest)}
    krylov = krylov_bases(sector, largest, identity)
    for arm_index, arm in enumerate(ARMS):
        if arm in grown:
            labels, full_basis = grown[arm]
            details: dict[str, Any] = {"selected": labels}
        else:
            full_basis, details = krylov[arm]
            labels = [f"{arm}[{k}]" for k in range(largest)]
        per_size = {}
        for size in sizes:
            if full_basis.shape[1] < size:
                # Greedy growth stopped: no candidate lowers the energy further.
                per_size[str(size)] = {"available": False,
                                       "exhausted_at": int(full_basis.shape[1])}
                continue
            B = full_basis[:, :size]
            S, Hm = pencil(B, H)
            Hp = Hm - identity * S
            exact = solve_exact(S, Hm)
            eig_s = np.linalg.eigvalsh(S)
            row: dict[str, Any] = {
                "available": True,
                "exact_error": float(exact.energies[0]) - target,
                "exact_rank": int(exact.effective_rank),
                "overlap_condition": float(eig_s[-1] / max(eig_s[0], 1e-300)),
                "measured_quantities": quantities(arm, size),
                "budgets": {},
            }
            for budget_index, budget in enumerate(budgets):
                shots = budget // quantities(arm, size)
                if shots < 1:
                    row["budgets"][str(budget)] = {"shots_per_quantity": 0, "skipped": True}
                    continue
                rng = np.random.default_rng(np.random.SeedSequence(
                    [SEED_ROOT, index, arm_index, size, budget_index]))
                S_draws, H_draws = sample_pencil(arm, S, Hp, shots, scale, rng, replicas)
                floor = overlap_floor(size, shots)
                errors, failures, ranks = [], 0, []
                for S_hat, H_hat in zip(S_draws, H_draws):
                    try:
                        solved = solve_projected(S_hat, H_hat, overlap_noise_floor=floor)
                    except (ValueError, np.linalg.LinAlgError):
                        errors.append(float("nan"))
                        failures += 1
                        continue
                    errors.append(float(solved.energies[0]) + identity - target)
                    ranks.append(int(solved.effective_rank))
                summary = summarize(errors, failures, row["exact_error"])
                summary.update({"shots_per_quantity": shots, "overlap_floor": floor,
                                "median_rank": float(np.median(ranks)) if ranks else None})
                row["budgets"][str(budget)] = summary
            per_size[str(size)] = row
        arms[arm] = {"labels": labels, **details, "sizes": per_size}
    system["arms"] = arms
    system["wall_seconds"] = time.perf_counter() - started
    return system


ARM_NAMES = {"rt_krylov": "Real-time Krylov", "power_krylov": "Power Krylov",
             "commutator_acase": "A-CASE, excitations + commutators",
             "pure_commutator_acase": "A-CASE, commutators only",
             "excitation_acase": "A-CASE, excitations only"}
REPORT_BUDGETS = ("1000000", "100000000", "10000000000", "1000000000000")


def _mha(value) -> str:
    return "n/a" if value is None else f"{1e3 * value:,.1f}"


def report(record: dict[str, Any]) -> str:
    """The study document's tables, generated from the record."""
    lines = ["| System | Qubits | Sector dim. | HF error (mHa) | Overlap of HF with target "
             "| Pauli terms | lambda (Ha) | QWC settings of H |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for system in record["systems"]:
        lines.append(
            f"| `{system['key']}` | {system['qubits']} | {system['sector_dimension']:,} "
            f"| {_mha(system['hf_error'])} | {system['hf_target_overlap']:.3f} "
            f"| {system['pauli_terms']:,} | {system['pauli_one_norm']:.1f} "
            f"| {system['hamiltonian_qwc_settings']:,} |")
    sizes = record["design"]["basis_sizes"]
    lines += ["", "Exact subspace error (mHa) with no shot noise, and the overlap condition "
                  "number, by basis size M:", "",
              "| System | Arm | " + " | ".join(f"M = {m}" for m in sizes) + " |",
              "| --- | --- | " + " | ".join("---:" for _ in sizes) + " |"]
    for system in record["systems"]:
        for arm in record["design"]["arms"]:
            cells = []
            for m in sizes:
                row = system["arms"][arm]["sizes"][str(m)]
                cells.append(f"exhausted at {row['exhausted_at']}" if not row["available"]
                             else f"{_mha(row['exact_error'])} (κ {row['overlap_condition']:.0e})")
            lines.append(f"| `{system['key']}` | {ARM_NAMES[arm]} | " + " | ".join(cells) + " |")
    for m in sizes:
        lines += ["", f"Median absolute error against the target (mHa) at M = {m}, by total "
                      "shots. Brackets give the median deviation from the arm's own exact "
                      "energy:", "",
                  "| System | Arm | " + " | ".join(f"{int(b):.0e} shots" for b in REPORT_BUDGETS)
                  + " |",
                  "| --- | --- | " + " | ".join("---:" for _ in REPORT_BUDGETS) + " |"]
        for system in record["systems"]:
            for arm in record["design"]["arms"]:
                row = system["arms"][arm]["sizes"][str(m)]
                if not row["available"]:
                    continue
                cells = []
                for budget in REPORT_BUDGETS:
                    found = row["budgets"][budget]
                    if found.get("skipped"):
                        cells.append("under 1 shot each")
                    else:
                        cells.append(f"{_mha(found['median_abs_error'])} "
                                     f"[{_mha(found['median_abs_shot_deviation'])}]")
                lines.append(f"| `{system['key']}` | {ARM_NAMES[arm]} | "
                             + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", action="store_true",
                        help="print the record's tables as Markdown and exit")
    parser.add_argument("--systems", nargs="*", default=None,
                        help="keys of the systems to run (default: all)")
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    args = parser.parse_args(argv)
    if args.report:
        print(report(json.loads(args.out.read_text(encoding="utf-8"))), end="")
        return 0
    if args.replicas < 1:
        parser.error("--replicas must be positive")
    chosen = [s for s in SYSTEMS if args.systems is None or s["key"] in args.systems]
    if not chosen:
        parser.error("no known system selected")
    provenance = execution_provenance()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "metadata": {"git_sha": provenance["git_sha"], "git_dirty": provenance["git_dirty"],
                     "python": provenance["python"],
                     "numpy": provenance["dependencies"].get("numpy"),
                     "utc": provenance["utc"]},
        "design": {"seed_root": SEED_ROOT, "basis_sizes": list(BASIS_SIZES),
                   "budgets": list(BUDGETS), "replicas": args.replicas, "arms": list(ARMS),
                   "chemical_accuracy": CHEMICAL_ACCURACY},
        "systems": [],
    }
    for spec in chosen:
        index = SYSTEMS.index(spec)
        print(f"{spec['key']} ...", flush=True)
        system = run_system(index, spec, replicas=args.replicas)
        record["systems"].append(system)
        print(f"  {system['qubits']} qubits, dim {system['sector_dimension']}, "
              f"HF error {1e3 * system['hf_error']:.1f} mHa, "
              f"{system['wall_seconds']:.1f} s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
