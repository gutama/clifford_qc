"""Spin-free excitation pools and real Abelian point-group selection.

Operators use the native complex Cl(2n, C)/Jordan--Wigner representation with
interleaved alpha/beta spin orbitals. Each spatial-orbital irrep is an explicit
nonnegative bit character of (Z2)^k: tensor products are XOR and zero is the
identity. Arbitrary program-specific irrep numbers are not that convention.

Spin-free transfers E_ai = sum_sigma c†_(a,sigma) c_(i,sigma) commute with N,
Sz and both spin ladders. Their products and anti-Hermitian parts therefore
preserve total spin on any reference; they do not purify a spin-mixed state.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from numbers import Integral
from typing import Iterable, Sequence

from ..fermion import c_op, cdag_op
from ..multivector import MV
from .generator_core import Generator, as_generators, to_mv


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _irreps(orbital_irreps: Sequence[int]) -> tuple[int, ...]:
    labels = tuple(_integer(value, "orbital irrep") for value in orbital_irreps)
    if not labels:
        raise ValueError("one irrep character per spatial orbital is required")
    return labels


def _word_character(code: int, irreps: tuple[int, ...]) -> int:
    character = 0
    for p, label in enumerate(irreps):
        letters = code >> (4 * p)
        # X and Y change occupation; Z and I do not. Two flips on one
        # spatial orbital contribute the same character twice and cancel.
        if ((letters & 3) in (1, 2)) != (((letters >> 2) & 3) in (1, 2)):
            character ^= label
    return character


def _preserves_point_group(operator: MV, irreps: tuple[int, ...]) -> bool:
    if operator.n != 2 * len(irreps):
        raise ValueError("operators require two interleaved spin orbitals per spatial irrep")
    return all(_word_character(code, irreps) == 0 for code in operator.terms)


def _check_hamiltonian(hamiltonian, irreps: tuple[int, ...]) -> None:
    if hamiltonian is not None and not _preserves_point_group(to_mv(hamiltonian), irreps):
        raise ValueError("Hamiltonian does not preserve the declared point-group characters")


def point_group_filter(generators: Iterable, orbital_irreps: Sequence[int], *,
                       hamiltonian=None) -> list[Generator]:
    """Keep whole operators that commute with all declared irrep parities.

    The symmetry for character bit b is the product of Z_(2p) Z_(2p+1) over
    orbitals whose label has bit b. This tests global operator commutation,
    rather than whether a direction happens to vanish on one determinant.
    Input order and Generator objects are preserved; mixed-character operators
    are rejected whole, never projected or split into individual Pauli words.

    Supply ``hamiltonian`` to check its symmetry before removing directions.
    Without it, Hamiltonian invariance under these supplied characters is a
    caller assumption. No spin conservation is added by this filter alone.
    """
    irreps = _irreps(orbital_irreps)
    _check_hamiltonian(hamiltonian, irreps)
    items = list(generators)
    if not items:
        return []
    pool = as_generators(items)
    return [g for g in pool if _preserves_point_group(g.mv, irreps)]


def spin_adapted_excitations(n_spatial: int, occupied_spatial: Sequence[int], *,
                            max_rank: int = 2, orbital_irreps: Sequence[int] | None = None,
                            hamiltonian=None) -> list[Generator]:
    """Spin-scalar singles/doubles over a closed-shell orbital partition.

    ``occupied_spatial`` lists doubly occupied spatial orbitals; every other
    spatial orbital is virtual. The output acts on ``2*n_spatial`` interleaved
    spin orbitals and contains E_ai - E_ai† plus E_ai E_bj - (E_ai E_bj)† for
    unordered transfer pairs, including repeated indices. Both matchings of
    four distinct spatial indices are retained. These span the singlet CISD
    directions of the closed-shell determinant; they need not be orthogonal.
    Open-shell-specific spin couplings and generalized occupied/virtual
    rotations are outside this pool.

    Optional ``orbital_irreps`` are real Abelian XOR characters, one per
    spatial orbital. Singles require irrep(a) XOR irrep(i) = 0, while doubles
    require the XOR of all four labels to vanish. A forbidden single transfer
    can participate in an allowed double. All spin terms stay in one MV.
    ``hamiltonian``, if supplied, must preserve the declared point group.
    No chemistry extra is needed and no measurement/solver default is changed.
    """
    count = _integer(n_spatial, "n_spatial", minimum=1)
    rank = _integer(max_rank, "max_rank", minimum=1)
    if rank not in (1, 2):
        raise ValueError("max_rank must be 1 or 2")
    occupied = tuple(_integer(p, "occupied spatial orbital") for p in occupied_spatial)
    if len(set(occupied)) != len(occupied) or any(p >= count for p in occupied):
        raise ValueError("occupied spatial orbitals must be distinct and in range")
    occupied = tuple(sorted(occupied))
    virtual = tuple(p for p in range(count) if p not in occupied)
    irreps = (0,) * count if orbital_irreps is None else _irreps(orbital_irreps)
    if len(irreps) != count:
        raise ValueError("one irrep character per spatial orbital is required")
    _check_hamiltonian(hamiltonian, irreps)

    n = 2 * count
    transfers = [(a, i) for i in occupied for a in virtual]
    operators = {}
    out = []
    for a, i in transfers:
        # Build even forbidden singles: two of them may form an allowed double.
        term = sum((cdag_op(n, 2*a+s) * c_op(n, 2*i+s) for s in (0, 1)), MV(n))
        operators[a, i] = term
        if irreps[a] ^ irreps[i] == 0:
            out.append(Generator(f"SF({a}<-{i})", term - term.dagger()))
    if rank == 2:
        for (a, i), (b, j) in combinations_with_replacement(transfers, 2):
            if irreps[a] ^ irreps[i] ^ irreps[b] ^ irreps[j]:
                continue
            term = operators[a, i] * operators[b, j]
            out.append(Generator(f"SF({a}<-{i};{b}<-{j})", term - term.dagger()))
    return out


__all__ = ["point_group_filter", "spin_adapted_excitations"]
