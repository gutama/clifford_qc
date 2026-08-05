"""What the single-particle basis costs: words, determinants, rotations.

Every fermionic number this package reports is quoted in the site basis by
default and without saying so.  The default is a choice, and it is not free:

    b_p = sum_i W_pi a_i,   W W^T = 1

leaves the spectrum exactly invariant while changing the Pauli-word count, the
weight of the leading determinant, and the subspace size needed for chemical
accuracy.  This script measures all three across the bases in
``models/orbital.py``, for a clean half-filled Hubbard ring and for the same
ring with diagonal disorder, and writes one record.

Three things are worth reading out of the record rather than assumed:

1. **The word count is not a property of the model.** On the 8-site chain it
   spans roughly a factor of 19 between the site basis and db4, at identical
   physics.  A word count quoted without its basis is not reproducible.
2. **No basis wins.** The momentum basis is far more compact in determinants on
   a clean ring and far worse under strong disorder, where non-interacting
   natural orbitals take the lead.  The compactness ranking is a property of
   the *disorder strength*, not of the basis.
3. **Wavelets do not pay off here.** They are the interesting candidate --
   local *and* multiscale, and the free-fermion MERA correspondence suggests
   they should help -- but they lose on both axes to whichever of site or
   momentum suits the model, and their ``O(N)`` rotation count assumes free
   mode reordering that a Jordan-Wigner register does not have.

Invariance is asserted, not reported: every basis must reproduce the site-basis
ground energy to ``1e-8`` or the run fails.  A rotation that is subtly wrong
would otherwise produce an attractive-looking cost table for the wrong
Hamiltonian.

    python benchmarks/run_orbital_basis.py
    python benchmarks/run_orbital_basis.py --sites 6 --disorder-seeds 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from clifford_qc.models.orbital import (as_basis, givens_network,
                                        natural_orbital_basis, rotate_model,
                                        rotate_one_body,
                                        rotate_onsite_interaction)
from clifford_qc.reproducibility import stamp_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmarks" / "reference_results" / "orbital_basis.json"

# Predeclared parameters. Chemical accuracy is 1.6 mHa; the Hubbard model is
# quoted in units of t, so the subspace target is stated in the same units and
# fixed before the sweep rather than tuned to make a basis look good.
CHEMICAL_ACCURACY = 1.6e-3
INVARIANCE_TOL = 1e-8
DEFAULT_SITES = 8
DEFAULT_U = 4.0
DEFAULT_DISORDER = (0.0, 2.0, 6.0, 12.0)
DEFAULT_DISORDER_SEEDS = 3
BASES = ("site", "momentum", "db1", "db2", "db3", "db4")


def ring_hopping(sites: int, t: float = 1.0) -> np.ndarray:
    """Periodic nearest-neighbour hopping, as in ``hubbard(..., periodic=True)``."""
    matrix = np.zeros((sites, sites))
    bonds = [(i, i + 1) for i in range(sites - 1)]
    if sites > 2:
        bonds.append((sites - 1, 0))
    for i, j in bonds:
        matrix[i, j] -= t
        matrix[j, i] -= t
    return matrix


# ------------------------------------------------- determinant-space solver
def _spin_determinants(orbitals: int, electrons: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(orbitals), electrons))


def _excite(occupied: tuple[int, ...], p: int, q: int):
    """``c†_p c_q`` on one occupation string: ``(sign, new_string)`` or ``None``."""
    if q not in occupied:
        return None
    position = occupied.index(q)
    rest = occupied[:position] + occupied[position + 1:]
    sign = (-1) ** position
    if p in rest:
        return None
    insert = 0
    while insert < len(rest) and rest[insert] < p:
        insert += 1
    return sign * (-1) ** insert, rest[:insert] + (p,) + rest[insert:]


def _excitation_matrices(orbitals: int, electrons: int):
    determinants = _spin_determinants(orbitals, electrons)
    index = {det: i for i, det in enumerate(determinants)}
    size = len(determinants)
    matrices = [[None] * orbitals for _ in range(orbitals)]
    for p in range(orbitals):
        for q in range(orbitals):
            rows, cols, values = [], [], []
            for column, occupied in enumerate(determinants):
                moved = _excite(occupied, p, q)
                if moved:
                    rows.append(index[moved[1]])
                    cols.append(column)
                    values.append(float(moved[0]))
            matrices[p][q] = sp.csr_matrix((values, (rows, cols)),
                                           shape=(size, size))
    return matrices, determinants


def sector_hamiltonian(one_body, onsite_u, basis, n_up: int, n_down: int, *,
                       chemical_potential=None):
    """Sparse ``H`` on the ``(N_up, N_dn)`` determinant space of a rotated model.

    The same spatial rotation acts on both spin channels, so the sector
    Hamiltonian is a sum of Kronecker products of one-spin excitation matrices
    and nothing ever loops over the full determinant space.  That factorisation
    is what makes an 8-site sweep over six bases cheap enough to be a benchmark
    rather than a cluster job.
    """
    orbitals = one_body.shape[0]
    interaction_mean = float(np.atleast_1d(np.asarray(onsite_u, float)).mean())
    mu = 0.5 * interaction_mean if chemical_potential is None \
        else float(chemical_potential)
    rotated = rotate_one_body(one_body, basis)
    tensor = rotate_onsite_interaction(onsite_u, basis, n_orbitals=orbitals)
    up_matrices, up_determinants = _excitation_matrices(orbitals, n_up)
    down_matrices, down_determinants = _excitation_matrices(orbitals, n_down)
    n_up_dets, n_down_dets = len(up_determinants), len(down_determinants)
    identity_up = sp.identity(n_up_dets, format="csr")
    identity_down = sp.identity(n_down_dets, format="csr")
    hamiltonian = sp.csr_matrix((n_up_dets * n_down_dets,) * 2)
    for p in range(orbitals):
        for q in range(orbitals):
            coefficient = rotated[p, q]
            if abs(coefficient) > 1e-14:
                hamiltonian = hamiltonian + coefficient * (
                    sp.kron(up_matrices[p][q], identity_down, format="csr")
                    + sp.kron(identity_up, down_matrices[p][q], format="csr"))
    for p in range(orbitals):
        for q in range(orbitals):
            block = None
            for r in range(orbitals):
                for s in range(orbitals):
                    value = tensor[p, q, r, s]
                    if abs(value) > 1e-14:
                        term = value * down_matrices[r][s]
                        block = term if block is None else block + term
            if block is not None and block.nnz:
                hamiltonian = hamiltonian + sp.kron(up_matrices[p][q], block,
                                                    format="csr")
    if mu:
        hamiltonian = hamiltonian - mu * (n_up + n_down) * sp.identity(
            n_up_dets * n_down_dets, format="csr")
    return hamiltonian.tocsr()


def _ground(hamiltonian):
    if hamiltonian.shape[0] <= 400:
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
        return float(values[0]), vectors[:, 0]
    values, vectors = spla.eigsh(hamiltonian, k=1, which="SA", tol=1e-12,
                                 maxiter=20000)
    return float(values[0]), vectors[:, 0]


def _truncated_energy(hamiltonian, order, size: int) -> float:
    indices = np.sort(order[:size])
    block = hamiltonian[indices][:, indices]
    if size <= 400:
        return float(np.linalg.eigvalsh(block.toarray())[0])
    return float(spla.eigsh(block, k=1, which="SA", tol=1e-10,
                            maxiter=20000)[0][0])


def determinants_for_accuracy(hamiltonian, vector, energy, target: float):
    """Smallest amplitude-ranked subspace reaching ``target`` above ``energy``.

    Exponential bracket then bisection.  The bisection assumes the truncation
    error is monotone in the subspace size, which it need not be term by term;
    the returned size is therefore re-verified against the bracket, and the
    reported number is an upper bound on the true minimum either way.
    """
    order = np.argsort(-np.abs(vector))
    dimension = len(vector)
    low, high = 1, 1
    while high < dimension and _truncated_energy(hamiltonian, order, high) - energy > target:
        low = high
        high = min(dimension, high * 2)
    if _truncated_energy(hamiltonian, order, high) - energy > target:
        return dimension, None
    while low + 1 < high:
        middle = (low + high) // 2
        if _truncated_energy(hamiltonian, order, middle) - energy <= target:
            high = middle
        else:
            low = middle
    return high, _truncated_energy(hamiltonian, order, high)


# ------------------------------------------------------------- measurement
def measure(one_body, onsite_u, basis_name, basis, *, target: float):
    """Cost currencies of one (model, basis) pair."""
    orbitals = one_body.shape[0]
    electrons = orbitals // 2
    model = rotate_model(one_body, onsite_u, basis)
    hamiltonian = sector_hamiltonian(one_body, onsite_u, basis,
                                     electrons, electrons)
    energy, vector = _ground(hamiltonian)
    dimension = hamiltonian.shape[0]
    subspace, truncated = determinants_for_accuracy(hamiltonian, vector,
                                                    energy, target)
    rotations, _ = givens_network(as_basis(basis, orbitals).matrix)
    return {
        "basis": basis_name,
        "energy": energy,
        "pauli_words": int(model.metadata["pauli_words"]),
        "determinant_space": int(dimension),
        "leading_determinant_weight": float(np.max(vector ** 2)),
        "determinants_for_accuracy": int(subspace),
        "subspace_fraction": subspace / dimension,
        "truncated_energy": truncated,
        "givens_rotations": len(rotations),
    }


def clean_sweep(sites: int, interaction: float, target: float) -> list[dict]:
    one_body = ring_hopping(sites)
    rows = []
    names = [name for name in BASES
             if not (name.startswith("db") and 2 * int(name[2:]) > sites)]
    for name in names:
        rows.append(measure(one_body, interaction, name, name, target=target))
    rows.append(measure(one_body, interaction, "natural",
                        natural_orbital_basis(one_body), target=target))
    return rows


def disorder_sweep(sites: int, interaction: float, strengths, seeds: int,
                   target: float) -> list[dict]:
    rows = []
    for strength in strengths:
        for seed in range(seeds):
            generator = np.random.default_rng(1000 * seed + int(10 * strength))
            one_body = ring_hopping(sites)
            if strength:
                one_body = one_body + np.diag(
                    generator.uniform(-strength / 2, strength / 2, sites))
            for name in ("site", "momentum", "db2"):
                row = measure(one_body, interaction, name, name, target=target)
                row.update(disorder=float(strength), seed=int(seed))
                rows.append(row)
            row = measure(one_body, interaction, "natural",
                          natural_orbital_basis(one_body), target=target)
            row.update(disorder=float(strength), seed=int(seed))
            rows.append(row)
    return rows


def assert_invariance(rows: list[dict], tol: float, label: str) -> None:
    """Every basis must reproduce the site-basis energy, or the record is void."""
    reference = next(row["energy"] for row in rows if row["basis"] == "site")
    for row in rows:
        drift = abs(row["energy"] - reference)
        if drift > tol:
            raise AssertionError(
                f"{label}: basis {row['basis']} moved the ground energy by "
                f"{drift:.2e} (tolerance {tol:.0e}) -- the rotation is wrong")


def build_record(sites: int, interaction: float, strengths, seeds: int,
                 target: float) -> dict:
    clean = clean_sweep(sites, interaction, target)
    assert_invariance(clean, INVARIANCE_TOL, "clean ring")
    disordered = disorder_sweep(sites, interaction, strengths, seeds, target)
    for strength in strengths:
        for seed in range(seeds):
            group = [row for row in disordered
                     if row["disorder"] == strength and row["seed"] == seed]
            assert_invariance(group, INVARIANCE_TOL,
                              f"disorder W={strength} seed={seed}")
    words = {row["basis"]: row["pauli_words"] for row in clean}
    subspaces = {row["basis"]: row["determinants_for_accuracy"] for row in clean}
    return {
        "experiment": "orbital_basis_cost",
        "parameters": {
            "sites": sites,
            "U": interaction,
            "t": 1.0,
            "periodic": True,
            "filling": "half",
            "chemical_potential": 0.5 * interaction,
            "accuracy_target": target,
            "invariance_tolerance": INVARIANCE_TOL,
            "disorder_strengths": [float(w) for w in strengths],
            "disorder_seeds": seeds,
            "disorder_distribution": "uniform(-W/2, W/2) on the site energies",
        },
        "claim": (
            "The single-particle basis is a free parameter. It leaves the "
            "spectrum invariant (asserted to "
            f"{INVARIANCE_TOL:.0e}) and changes every reported cost: on the "
            "clean ring the Pauli-word count spans "
            f"{min(words.values())}-{max(words.values())} and the subspace "
            f"reaching {target} spans "
            f"{min(subspaces.values())}-{max(subspaces.values())} "
            "determinants. Which basis is most compact depends on the "
            "disorder strength, so there is no basis to standardise on -- "
            "only a parameter to report."),
        "clean": clean,
        "disorder": disordered,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=DEFAULT_SITES)
    parser.add_argument("--interaction", type=float, default=DEFAULT_U)
    parser.add_argument("--disorder", type=float, nargs="*",
                        default=list(DEFAULT_DISORDER))
    parser.add_argument("--disorder-seeds", type=int,
                        default=DEFAULT_DISORDER_SEEDS)
    parser.add_argument("--accuracy", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.sites % 2:
        parser.error("half filling needs an even site count")
    record = stamp_record(build_record(
        args.sites, args.interaction, args.disorder,
        args.disorder_seeds, args.accuracy))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
