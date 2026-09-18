"""Stabilizer-aware initialization (Paper B machinery).

Two classically-computed scaffolds for seeding ADAPT/VQE:

- **Variant A — discrete Clifford-point search**: coordinate descent over
  HVA angles restricted to {0, pi/2, pi, 3pi/2}. Every such point is a
  stabilizer state, so a stim backend evaluates candidates at large n.
- **Variant B — stabilizer Hamiltonian approximation**: a greedy,
  sign-consistent, mutually commuting subset S of the Hamiltonian's Pauli
  words maximizing sum |h_w|; the joint eigenstate with <w> = -sign(h_w)
  is a stabilizer ground state of H_stab = sum_S h_w w, prepared as a
  Clifford Program (via stim's tableau synthesis, ``stim`` extra).

Both produce a Clifford preparation Program; ``seed_model`` swaps it in as
a model's reference so residual ADAPT starts from the scaffold and every
appended rotor is non-Clifford correction, which is what the go/no-go
metric counts.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..multivector import word_mul
from ..ir import PauliSum, PauliWord, Program, Rotor
from .vqe import hva_program

CLIFFORD_ANGLES = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)


# ---------------------------------------------------------------------------
# Variant A: discrete Clifford-point search


@dataclass(frozen=True)
class CliffordPointResult:
    values: tuple[float, ...]
    energy: float
    evaluations: int
    restarts: int


def clifford_point_search(program: Program, hamiltonian: PauliSum, backend, *,
                          seed: int, restarts: int = 8,
                          max_sweeps: int = 20) -> CliffordPointResult:
    """Seeded coordinate descent over Clifford angles of ``program``.

    Restart 0 starts from all-zero angles (the bare reference state); later
    restarts start from seeded random Clifford points. Each sweep tries all
    four angles per parameter, keeping the best; sweeps repeat until no
    parameter changes. Deterministic for a fixed seed.
    """
    if restarts < 1:
        raise ValueError("restarts must be positive")
    rng = np.random.default_rng(seed)
    k = len(program.parameters)
    evaluations = 0

    def energy(values):
        nonlocal evaluations
        evaluations += 1
        return backend.expectation(program, hamiltonian, values)

    best_values: tuple[float, ...] | None = None
    best_energy = math.inf
    for restart in range(restarts):
        if restart == 0:
            values = [0.0] * k
        else:
            values = [CLIFFORD_ANGLES[i] for i in rng.integers(0, 4, size=k)]
        current = energy(values)
        for _ in range(max_sweeps):
            improved = False
            for j in range(k):
                held = values[j]
                candidate_angle, candidate_energy = held, current
                for angle in CLIFFORD_ANGLES:
                    if angle == held:
                        continue
                    values[j] = angle
                    e = energy(values)
                    if e < candidate_energy - 1e-12:
                        candidate_angle, candidate_energy = angle, e
                values[j] = candidate_angle
                if candidate_energy < current - 1e-12:
                    current = candidate_energy
                    improved = True
            if not improved:
                break
        if current < best_energy - 1e-12:
            best_energy = current
            best_values = tuple(values)
    return CliffordPointResult(best_values, best_energy, evaluations, restarts)


def bound_hva_program(model, depth: int, values: Sequence[float]) -> Program:
    """The model's HVA with angles bound to numbers: a concrete preparation
    Program (Clifford when every angle is a multiple of pi/2)."""
    template = hva_program(model, depth)
    bindings = template.parameters.bind(list(values))
    prog = Program(model.n)
    for op in template.ops:
        if isinstance(op, Rotor):
            prog.append(Rotor(op.word, op.resolved_angle(bindings)))
        else:
            prog.append(op)
    return prog


# ---------------------------------------------------------------------------
# Variant B: stabilizer Hamiltonian approximation


@dataclass(frozen=True)
class StabilizerApprox:
    """Sign-consistent commuting approximation of a Hamiltonian.

    ``generators``: independent (word, sign) pairs defining the stabilizer
    group; ``selected``: every Hamiltonian word absorbed into H_stab (its
    stabilizer expectation is consistent), with its sign; ``excluded``:
    words dropped for anticommuting or carrying an inconsistent sign.
    ``scaffold_energy`` = sum of h_w * sign_w over ``selected`` — the exact
    <H_stab> of the scaffold state, and a lower-bound estimate of <H>.
    """

    n: int
    generators: tuple[tuple[PauliWord, int], ...]
    selected: tuple[tuple[PauliWord, int], ...]
    excluded: tuple[PauliWord, ...]
    scaffold_energy: float


def _xz(n: int, code: int) -> int:
    """Symplectic bit vector of a word code: x bits in the low n, z in the high."""
    x = z = 0
    for j in range(n):
        letter = (code >> (2 * j)) & 3
        if letter in (1, 2):
            x |= 1 << j
        if letter in (2, 3):
            z |= 1 << j
    return x | (z << n)


def _commutes(n: int, vec_a: int, vec_b: int) -> bool:
    mask = (1 << n) - 1
    ax, az = vec_a & mask, vec_a >> n
    bx, bz = vec_b & mask, vec_b >> n
    return ((ax & bz).bit_count() + (az & bx).bit_count()) % 2 == 0


def stabilizer_hamiltonian_approximation(hamiltonian: PauliSum,
                                         tol: float = 1e-12) -> StabilizerApprox:
    """Greedy sign-consistent commuting subset maximizing sum |h_w|.

    Words are visited in descending |h_w| with target sign -sign(h_w).
    A word joins H_stab when it commutes with all accepted generators and,
    if it is a product of them, its implied sign matches the target
    (checked by symplectic Gaussian elimination that tracks the exact Pauli
    product, so phases are never guessed).
    """
    n = hamiltonian.n
    ranked = sorted(((code, coeff.real) for code, coeff in hamiltonian.terms.items()
                     if code != 0 and abs(coeff.real) > tol),
                    key=lambda item: -abs(item[1]))
    # basis rows: (vector, word code, word phase, sigma product)
    basis: list[tuple[int, int, complex, int]] = []
    generators: list[tuple[PauliWord, int]] = []
    selected: list[tuple[PauliWord, int]] = []
    excluded: list[PauliWord] = []
    scaffold_energy = 0.0

    for code, h in ranked:
        sigma = -1 if h > 0 else 1
        vec = _xz(n, code)
        if not all(_commutes(n, vec, row[0]) for row in basis):
            excluded.append(PauliWord(n, code))
            continue
        # reduce against the basis, accumulating the exact Pauli product
        r = vec
        acc_code, acc_phase, acc_sigma = 0, 1.0 + 0j, 1
        for row_vec, row_code, row_phase, row_sigma in basis:
            pivot = row_vec & -row_vec
            if r & pivot:
                r ^= row_vec
                phase, acc_code = word_mul(n, acc_code, row_code)
                acc_phase *= phase * row_phase
                acc_sigma *= row_sigma
        if r == 0:
            # dependent: implied operator is acc_phase * word(acc_code)
            assert acc_code == code
            implied = acc_sigma * int(round(acc_phase.real))
            if implied == sigma:
                selected.append((PauliWord(n, code), sigma))
                scaffold_energy += h * sigma
            else:
                excluded.append(PauliWord(n, code))
        else:
            # store the operator that actually corresponds to the reduced
            # vector r: the accumulated row product times the new word
            phase, r_code = word_mul(n, acc_code, code)
            basis.append((r, r_code, acc_phase * phase, acc_sigma * sigma))
            basis.sort(key=lambda row: row[0] & -row[0])
            generators.append((PauliWord(n, code), sigma))
            selected.append((PauliWord(n, code), sigma))
            scaffold_energy += h * sigma
    return StabilizerApprox(n, tuple(generators), tuple(selected),
                            tuple(excluded), scaffold_energy)


def stabilizer_ground_program(approx: StabilizerApprox) -> Program:
    """Clifford Program preparing a joint eigenstate with <w> = sign_w for
    every generator (stim tableau synthesis; requires the ``stim`` extra).
    Underconstrained groups are completed arbitrarily by stim."""
    from ..capabilities import require

    require("contextual_restriction", feature="stabilizer_ground_program")

    import stim

    stabilizers = [stim.PauliString(("+" if sign > 0 else "-") + word.label)
                   for word, sign in approx.generators]
    tableau = stim.Tableau.from_stabilizers(
        stabilizers, allow_underconstrained=True, allow_redundant=False)
    circuit = tableau.to_circuit("elimination")
    prog = Program(approx.n)
    for inst in circuit:
        name = {"H": "H", "S": "S", "S_DAG": "SDG", "X": "X", "Y": "Y",
                "Z": "Z", "CX": "CX", "CZ": "CZ", "SWAP": "SWAP"}.get(inst.name)
        if name is None:
            raise ValueError(f"unexpected gate {inst.name!r} in synthesized circuit")
        targets = [t.value for t in inst.targets_copy()]
        arity = 2 if name in ("CX", "CZ", "SWAP") else 1
        for i in range(0, len(targets), arity):
            prog.clifford(name, *targets[i:i + arity])
    return prog


def seed_model(model, preparation: Program, label: str = "seeded"):
    """The same model with its reference state replaced by ``preparation``,
    so run_adapt/run_vqe start from the scaffold."""
    if preparation.n != model.n:
        raise ValueError("preparation program and model qubit counts differ")
    return dataclasses.replace(model, reference=preparation,
                               name=f"{model.name}+{label}")
